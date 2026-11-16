#!/usr/bin/env python3
"""Download Pareto Lab's collector, run it through Kubernetes or SSH, and archive results."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
import urllib.parse
import uuid
from datetime import datetime, timezone

COLLECTOR_COMMIT = "82f8b7eeef90c76bfc55c20027aa4e4c7d725c98"
COLLECTOR_SHA256 = "0c6481dd20299db7f352bcfad68a84c32bac5946f917f122af65a956e02d9743"
COLLECTOR_URL = ("https://raw.githubusercontent.com/shiyu7/pareto-lab/"
                 + COLLECTOR_COMMIT + "/tools/collect_pareto.py")
MAX_COLLECTOR_BYTES = 1024 * 1024


def positive(value):
    try:
        result = int(value)
        if result <= 0:
            raise ValueError
        return result
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("expected a positive integer") from None


def number_list(value):
    parts = value.split(",")
    try:
        values = [positive(part.strip()) for part in parts]
        if len(set(values)) != len(values):
            raise ValueError
        return values
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("expected distinct comma-separated positive integers") from None


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", choices=("kubernetes", "ssh"), default="kubernetes")
    p.add_argument("--context", help="Kubernetes context; default is the current context")
    p.add_argument("--namespace", default="default")
    p.add_argument("--pod", help="existing pod containing the benchmark client")
    p.add_argument("--container", help="container name when the pod has multiple containers")
    p.add_argument("--ssh-target", help="SSH target as user@host; uses existing SSH configuration and credentials")
    p.add_argument("--ssh-port", type=positive)
    p.add_argument("--ssh-proxyjump", help="OpenSSH ProxyJump host/alias")
    p.add_argument("--python", default="python3", help="Python in the remote SGLang environment")
    p.add_argument("--server-url", required=True, help="native SGLang /generate server root reachable from the client")
    p.add_argument("--model", required=True, help="model ID or local tokenizer/model path on the client")
    p.add_argument("--tokenizer", help="local tokenizer path if different from model")
    p.add_argument("--num-gpus", type=positive, required=True, help="all GPUs used by the measured endpoint")
    p.add_argument("--scheme", required=True, help="measured deployment label")
    p.add_argument("--workload-prefix", help="shared model/hardware/precision/cache conditions, excluding lengths")
    p.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    p.add_argument("--input-lengths", type=number_list, default=number_list("8192,32768,131072"))
    p.add_argument("--output-length", type=positive, default=1536)
    p.add_argument("--concurrency", type=number_list, default=number_list("1,2,4,8,16,32"))
    p.add_argument("--waves", type=positive, default=8)
    p.add_argument("--min-requests", type=positive, default=16)
    p.add_argument("--timeout", type=positive, default=1800, help="timeout in seconds per point")
    p.add_argument("--flush-cache", action="store_true", help="explicitly request server prefix-cache flush after warmup")
    for name in ("tp-size", "dp-size", "ep-size", "pp-size", "chunk-length"):
        p.add_argument("--" + name, type=positive)
    p.add_argument("--chunk-scope", choices=("request", "dp-rank"))
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--archive-root", type=Path, default=Path("pareto-archives"))
    p.add_argument("--dry-run", action="store_true", help="display the collection plan without download or remote actions")
    return p


def validate(args):
    if args.transport == "kubernetes" and not args.pod:
        raise ValueError("--pod is required for Kubernetes")
    if args.transport == "ssh" and (not args.ssh_target or args.ssh_target.startswith("-")):
        raise ValueError("--ssh-target is required for SSH and must not start with '-'")
    if args.ssh_port and args.ssh_port > 65535:
        raise ValueError("--ssh-port must not exceed 65535")
    if args.transport == "kubernetes" and (args.ssh_target or args.ssh_port or args.ssh_proxyjump):
        raise ValueError("SSH options require --transport ssh")
    if args.transport == "ssh" and (args.context or args.pod or args.container or args.namespace != "default"):
        raise ValueError("Kubernetes options require --transport kubernetes")
    if args.output_length < 2 and args.phase != "prefill":
        raise ValueError("Decode output length must be at least 2")
    if bool(args.chunk_length) != bool(args.chunk_scope):
        raise ValueError("--chunk-length and --chunk-scope must be specified together")
    if args.no_chunked_prefill and args.chunk_length:
        raise ValueError("disabled chunking cannot specify a chunk size")
    if args.phase == "decode" and (args.chunk_length or args.no_chunked_prefill):
        raise ValueError("chunk metadata is only for prefill or both")
    for name in ("server_url", "model", "scheme", "python"):
        if not getattr(args, name).strip():
            raise ValueError(f"--{name.replace('_', '-')} must not be blank")
    address = urllib.parse.urlsplit(args.server_url)
    if (address.scheme not in ("http", "https") or not address.hostname or address.query or address.fragment
            or address.username is not None or address.password is not None):
        raise ValueError("--server-url must be an HTTP(S) URL without credentials, query or fragment")
    if address.path.rstrip("/").endswith(("/generate", "/v1", "/completions")):
        raise ValueError("--server-url must be the server root, not an API endpoint")


def current_context():
    result = subprocess.run(["kubectl", "config", "current-context"], check=True,
                            capture_output=True, text=True, timeout=20)
    return result.stdout.strip()


def transport_command(args, remote_argv, stdin=False):
    if args.transport == "kubernetes":
        cmd = ["kubectl", "--context", args.context, "-n", args.namespace, "exec"]
        if stdin:
            cmd.append("-i")
        cmd.append(args.pod)
        if args.container:
            cmd += ["-c", args.container]
        return cmd + ["--"] + remote_argv
    cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15"]
    if args.ssh_port:
        cmd += ["-p", str(args.ssh_port)]
    if args.ssh_proxyjump:
        cmd += ["-J", args.ssh_proxyjump]
    return cmd + ["--", args.ssh_target, shlex.join(remote_argv)]


def download_collector():
    request = urllib.request.Request(COLLECTOR_URL, headers={"User-Agent": "pareto-serving-benchmark-skill"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_COLLECTOR_BYTES + 1)
    if len(data) > MAX_COLLECTOR_BYTES or hashlib.sha256(data).hexdigest() != COLLECTOR_SHA256:
        raise ValueError("downloaded GitHub collector failed size/SHA-256 verification")
    return data


def init_remote(args, remote_dir, data):
    code = ("import hashlib,os,pathlib,sys; "
            "data=sys.stdin.buffer.read(1048577); "
            "assert len(data)<=1048576 and hashlib.sha256(data).hexdigest()==sys.argv[2], 'collector hash mismatch'; "
            "os.umask(0o077); root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False); "
            "(root/'collect_pareto.py').write_bytes(data)")
    cmd = transport_command(args, [args.python, "-c", code, remote_dir, COLLECTOR_SHA256], stdin=True)
    subprocess.run(cmd, input=data, check=True, timeout=60)


def workload(args, input_length):
    prefix = args.workload_prefix or args.model
    return f"{prefix} | input={input_length} output={args.output_length} | serving-closed-loop-v1"


def collector_command(args, remote_dir, input_length):
    root = remote_dir.rstrip("/")
    output = f"{root}/input-{input_length}.points.json"
    cmd = [args.python, f"{root}/collect_pareto.py", "--base-url", args.server_url,
           "--model", args.model, "--num-gpus", str(args.num_gpus), "--scheme", args.scheme,
           "--workload", workload(args, input_length), "--phase", args.phase,
           "--input-len", str(input_length), "--output-len", str(args.output_length),
           "--concurrency", ",".join(map(str, args.concurrency)), "--waves", str(args.waves),
           "--min-requests", str(args.min_requests), "--timeout", str(args.timeout), "--output", output]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.flush_cache:
        cmd.append("--flush-cache")
    for name in ("tp_size", "dp_size", "ep_size", "pp_size"):
        if (value := getattr(args, name)) is not None:
            cmd += ["--" + name.replace("_", "-"), str(value)]
    if args.chunk_length:
        cmd += ["--chunk-length", str(args.chunk_length), "--chunk-scope", args.chunk_scope]
    elif args.no_chunked_prefill:
        cmd.append("--no-chunked-prefill")
    return cmd


def fetch_remote(args, remote_path, destination, tar=False):
    # Stream files to a temporary local file and publish only complete transfers.
    if tar:
        code = ("import pathlib,sys,tarfile; p=pathlib.Path(sys.argv[1]); "
                "assert p.is_dir(), 'run directory not found'; "
                "t=tarfile.open(mode='w|gz',fileobj=sys.stdout.buffer); "
                "t.add(p,arcname=p.name,recursive=True); t.close()")
    else:
        code = "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
    cmd = transport_command(args, [args.python, "-c", code, remote_path])
    tmp = destination.with_name(destination.name + ".part")
    try:
        with tmp.open("wb") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.PIPE, timeout=180)
        if result.returncode:
            return False, result.stderr.decode("utf-8", errors="replace")[-1000:]
        os.replace(tmp, destination)
        return True, None
    finally:
        tmp.unlink(missing_ok=True)


def save_manifest(path, manifest):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_point_file(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "hpc-pareto-points" or data.get("version") not in (1, 2):
            return 0, False
        points = data.get("points")
        if not isinstance(points, list):
            return 0, False
        return len(points), data.get("collection", {}).get("complete") is True
    except (OSError, ValueError, TypeError, AttributeError):
        return 0, False


def split_phase_files(path, phases, expected_per_phase):
    """Preserve the original and write importable phase views with honest completion status."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "hpc-pareto-points" or not isinstance(data.get("points"), list):
        raise ValueError(f"invalid point document: {path}")
    written = {}
    for phase in phases:
        points = [point for point in data["points"] if point.get("phase") == phase]
        if not points:
            continue
        split = copy.deepcopy(data)
        split["points"] = points
        collection = split.setdefault("collection", {})
        source_complete = collection.get("complete") is True
        collection["complete"] = source_complete and len(points) == expected_per_phase
        collection["planned_points"] = expected_per_phase
        collection["phase"] = phase
        collection["split_from"] = path.name
        dest = path.with_name(f"{path.stem}.{phase}{path.suffix}")
        if dest.exists():
            raise ValueError(f"refusing to overwrite phase file: {dest}")
        save_manifest(dest, split)
        written[phase] = dest.name
    return written


def execute(args):
    validate(args)
    if args.transport == "kubernetes" and not args.context:
        args.context = current_context()
    if args.dry_run:
        for length in args.input_lengths:
            print(shlex.join(transport_command(args, collector_command(args, "/tmp/pareto-skill-EXAMPLE", length))))
        return None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = args.archive_root.expanduser().resolve() / run_id
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(root, 0o700)
    remote_dir = "/tmp/pareto-skill-" + run_id
    manifest = dict(version=1, run_id=run_id, status="preparing", transport=args.transport,
                    kubernetes=dict(context=args.context, namespace=args.namespace, pod=args.pod, container=args.container)
                    if args.transport == "kubernetes" else None,
                    ssh_target=args.ssh_target if args.transport == "ssh" else None,
                    ssh_port=args.ssh_port if args.transport == "ssh" else None,
                    ssh_proxyjump=args.ssh_proxyjump if args.transport == "ssh" else None,
                    endpoint=args.server_url, model=args.model, scheme=args.scheme, num_gpus=args.num_gpus,
                    collector=dict(url=COLLECTOR_URL, commit=COLLECTOR_COMMIT, sha256=COLLECTOR_SHA256),
                    settings=dict(input_lengths=args.input_lengths, output_length=args.output_length, phase=args.phase,
                                  concurrency=args.concurrency, waves=args.waves, min_requests=args.min_requests,
                                  timeout_per_point=args.timeout, tokenizer=args.tokenizer, client_python=args.python,
                                  workload_prefix=args.workload_prefix, flush_cache=args.flush_cache,
                                  tp_size=args.tp_size, dp_size=args.dp_size, ep_size=args.ep_size, pp_size=args.pp_size,
                                  chunk_length=args.chunk_length, chunk_scope=args.chunk_scope,
                                  no_chunked_prefill=args.no_chunked_prefill),
                    remote_directory=remote_dir, results=[])
    manifest_path = root / "manifest.json"
    save_manifest(manifest_path, manifest)
    print(f"Local archive: {root}", flush=True)
    try:
        data = download_collector()
        (root / "collect_pareto.py").write_bytes(data)
        init_remote(args, remote_dir, data)
        manifest["status"] = "running"
        save_manifest(manifest_path, manifest)
        for length in args.input_lengths:
            name = f"input-{length}.points"
            print(f"\nCollecting input={length}, requested output={args.output_length}", flush=True)
            cmd = transport_command(args, collector_command(args, remote_dir, length))
            log_path = root / f"{name}.driver.log"
            interrupted = None
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except (KeyboardInterrupt, Exception) as exc:
                    interrupted = exc
                    try:
                        process.terminate()
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait()
            local_json = root / f"{name}.json"
            ok_json, json_error = fetch_remote(args, f"{remote_dir}/{name}.json", local_json)
            ok_logs, log_error = fetch_remote(args, f"{remote_dir}/{name}.runs", root / f"{name}.runs.tar.gz", tar=True)
            count, complete = valid_point_file(local_json) if ok_json else (0, False)
            phase_files = split_phase_files(local_json,
                ("prefill", "decode") if args.phase == "both" else (args.phase,),
                len(args.concurrency)) if count else {}
            result = dict(input_tokens=length, requested_output_tokens=args.output_length, exit_code=returncode,
                          points=count, complete=complete, json=local_json.name if ok_json else None,
                          phase_json=phase_files,
                          logs=f"{name}.runs.tar.gz" if ok_logs else None,
                          transfer_error=json_error, log_transfer_error=log_error)
            manifest["results"].append(result)
            save_manifest(manifest_path, manifest)
            print(f"Archived input={length}: points={count}, complete={complete}, exit={returncode}", flush=True)
            if interrupted is not None:
                raise interrupted
            if returncode or not complete or not ok_json or not ok_logs:
                print("Stopping after an incomplete result or transfer; inspect the archive before retrying.", flush=True)
                break
        manifest["status"] = "complete" if (len(manifest["results"]) == len(args.input_lengths)
            and all(x["complete"] and x["exit_code"] == 0 and x["logs"] for x in manifest["results"])) else "partial"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest(manifest_path, manifest)
        print(f"Archive status: {manifest['status']} · {root}", flush=True)
    return root, manifest


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    try:
        result = execute(args)
        return 0 if result is None or result[1]["status"] == "complete" else 1
    except KeyboardInterrupt:
        print("Interrupted; see the local archive manifest.", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        p.exit(1, f"collection failed: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
