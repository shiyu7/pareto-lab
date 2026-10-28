#!/usr/bin/env python3
r"""Run SGLang serving sweeps and save Pareto Lab JSON directly.

Run in the Python environment that already contains SGLang and its benchmark
client dependencies. No GPU is required on the client. The server must already
be running; this script records, but does not change, its parallel/chunk config.

Example:
  python3 collect_pareto.py --base-url http://127.0.0.1:30000 \
    --model /path/to/model --num-gpus 8 --concurrency 1,2,4,8 \
    --input-len 4096 --output-len 256 --output points.json

This is an HTTP serving benchmark, not an isolated GPU-step benchmark. Prefill
uses one output token and reports TTFT (including queueing). Decode throughput
includes the full request lifetime. Neither is labelled as pure phase capacity.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlsplit

MODULES = ("sglang.benchmark.serving", "sglang.bench_serving")
MODE = "serving-closed-loop-v1"


def positive_int(value):
    try:
        result = int(value)
        if result < 1 or result > 2**31 - 1:
            raise ValueError
        return result
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("must be a positive integer <= 2147483647") from None


def concurrency_list(value):
    try:
        values = [positive_int(part.strip()) for part in value.split(",")]
        if len(values) != len(set(values)):
            raise ValueError
        return values
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("use distinct positive integers, e.g. 1,2,4,8,16") from None


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base-url", default="http://127.0.0.1:30000", help="SGLang server root URL, without /generate")
    cli.add_argument("--model", required=True, help="served model ID or local model/tokenizer path")
    cli.add_argument("--tokenizer", help="local tokenizer path if different from --model")
    cli.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    cli.add_argument("--scheme", help="label this deployed configuration")
    cli.add_argument("--workload", help="shared model/hardware/precision/cache/length label for comparable runs")
    cli.add_argument("--num-gpus", type=positive_int, required=True, help="actual GPUs used by the measured instance")
    for flag in ("tp-size", "dp-size", "ep-size", "pp-size"):
        cli.add_argument("--" + flag, type=positive_int, help="record actual server configuration; missing stays unknown")
    cli.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    cli.add_argument("--chunk-length", type=positive_int, help="actual configured Prefill chunk token limit")
    cli.add_argument("--chunk-scope", choices=("request", "dp-rank"))
    cli.add_argument("--concurrency", type=concurrency_list, default=concurrency_list("1,2,4,8,16,32"))
    cli.add_argument("--input-len", type=positive_int, default=4096)
    cli.add_argument("--output-len", type=positive_int, default=256, help="Decode output length; Prefill always generates 1 token")
    cli.add_argument("--waves", type=positive_int, default=8, help="requests per concurrency slot at each point (default 8)")
    cli.add_argument("--min-requests", type=positive_int, default=16)
    cli.add_argument("--warmup-requests", type=positive_int, default=2)
    cli.add_argument("--seed", type=int, default=42, help="base seed; each phase/load uses a different seed")
    cli.add_argument("--flush-cache", action="store_true", help="explicitly flush server prefix cache after each warmup")
    cli.add_argument("--timeout", type=positive_int, default=1800, help="maximum seconds per measured point, including warmup")
    cli.add_argument("--bench-module", choices=("auto",) + MODULES, default="auto")
    cli.add_argument("--output", type=Path, default=Path("pareto_points.json"), help="directly importable point JSON; existing files are never overwritten")
    cli.add_argument("--dry-run", action="store_true", help="print commands without running requests or creating files")
    return cli


def validate(args):
    for name in ("model", "tokenizer", "scheme", "workload"):
        value = getattr(args, name)
        if value is not None and not value.strip():
            raise ValueError(f"--{name} must not be blank")
    address = urlsplit(args.base_url)
    if address.scheme not in ("http", "https") or not address.hostname or address.query or address.fragment:
        raise ValueError("--base-url must be an HTTP(S) server root URL without a query/fragment")
    if address.username is not None or address.password is not None:
        raise ValueError("use OPENAI_API_KEY for authentication, not credentials in --base-url")
    if address.path.rstrip("/").endswith(("/generate", "/v1", "/completions")):
        raise ValueError("--base-url must be the server root, not an API endpoint")
    if args.phase != "prefill" and args.output_len < 2:
        raise ValueError("Decode --output-len must be >= 2 to measure TPOT")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if args.chunk_length is not None and args.chunked_prefill is None:
        args.chunked_prefill = True
    if args.chunked_prefill is True and (args.chunk_length is None or args.chunk_scope is None):
        raise ValueError("enabled chunking requires --chunk-length and --chunk-scope")
    if args.chunked_prefill is not True and (args.chunk_length is not None or args.chunk_scope is not None):
        raise ValueError("chunk length/scope require enabled chunking")
    if args.phase == "decode" and args.chunked_prefill is not None:
        raise ValueError("chunk metadata is for --phase prefill or both")
    if any(max(c * args.waves, args.min_requests) > 2**31 - 1 for c in args.concurrency):
        raise ValueError("request count is too large")
    args.base_url = args.base_url.rstrip("/")


def benchmark_environment():
    env = os.environ.copy()
    # SGLang can implicitly flush when this CI variable is true. Only the user's
    # explicit --flush-cache should control this shared-server mutation.
    env["SGLANG_IS_IN_CI"] = "false"
    return env


def find_benchmark(args):
    candidates = MODULES if args.bench_module == "auto" else (args.bench_module,)
    for module in candidates:
        result = subprocess.run([sys.executable, "-m", module, "--help"],
                                env=benchmark_environment(), capture_output=True,
                                text=True, timeout=60)
        if result.returncode == 0:
            return module, set(re.findall(r"--[a-z][a-z0-9-]*", result.stdout))
    raise ValueError("SGLang benchmark client is unavailable. Run this script in your SGLang environment; "
                     "check `python3 -m sglang.bench_serving --help` there. No benchmark was started.")


def jobs(args):
    phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
    for phase in phases:
        for index, concurrency in enumerate(args.concurrency):
            yield dict(phase=phase, concurrency=concurrency,
                       requests=max(args.min_requests, concurrency * args.waves),
                       output_len=1 if phase == "prefill" else args.output_len,
                       seed=(args.seed + index + (100000 if phase == "decode" else 0)) % 2**32,
                       name=f"{phase}-c{concurrency}")


def command(args, job, module, result_path):
    cmd = [sys.executable, "-m", module, "--backend", "sglang", "--base-url", args.base_url,
           "--model", args.model, "--dataset-name", "random-ids", "--tokenize-prompt",
           "--random-input-len", str(args.input_len), "--random-output-len", str(job["output_len"]),
           "--random-range-ratio", "1", "--num-prompts", str(job["requests"]),
           "--request-rate", "inf", "--max-concurrency", str(job["concurrency"]),
           "--warmup-requests", str(args.warmup_requests), "--seed", str(job["seed"]),
           "--output-file", str(result_path)]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.flush_cache:
        cmd += ["--flush-cache"]
    return cmd


def finite(row, key, positive=True, integer=False):
    value = row.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0) or (integer and int(value) != value)):
        raise ValueError(f"missing/invalid benchmark metric: {key}")
    return value


def make_point(args, job, row):
    completed = finite(row, "completed", integer=True)
    if completed != job["requests"]:
        raise ValueError(f"only {completed:g}/{job['requests']} requests completed; point rejected")
    for key in ("failed", "failed_requests"):
        if key in row and finite(row, key, positive=False) != 0:
            raise ValueError(f"{key} is nonzero; point rejected")
    for key, expected in (("max_concurrency", job["concurrency"]), ("random_output_len", job["output_len"]),
                          ("random_input_len", args.input_len)):
        if row.get(key) != expected:
            raise ValueError(f"benchmark {key} differs from the requested value")
    finite(row, "duration")
    total_input = finite(row, "total_input_tokens", integer=True)
    total_output = finite(row, "total_output_tokens", integer=True)
    if total_input != completed * args.input_len or total_output != completed * job["output_len"]:
        raise ValueError("actual token counts differ from requested fixed lengths; check tokenizer, EOS and server limits")
    phase = job["phase"]
    latency = finite(row, "mean_ttft_ms" if phase == "prefill" else "mean_tpot_ms")
    metric = "input_throughput" if phase == "prefill" else "output_throughput"
    throughput = finite(row, metric)
    expected_throughput = (total_input if phase == "prefill" else total_output) / row["duration"]
    if not math.isclose(throughput, expected_throughput, rel_tol=0.005):
        raise ValueError("throughput is inconsistent with token counts / duration")
    x_value = latency if phase == "prefill" else 1000 / latency
    if not math.isfinite(x_value) or x_value <= 0:
        raise ValueError("derived x coordinate is not finite and positive")
    parallel = {name: getattr(args, name) for name in ("tp_size", "dp_size", "ep_size", "pp_size")}
    default_scheme = " ".join(f"{label}{parallel[key] if parallel[key] is not None else '?'}" for label, key in
                              (("TP", "tp_size"), ("DP", "dp_size"), ("EP", "ep_size"), ("PP", "pp_size")))
    metrics = {key: value for key, value in row.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               and (key.startswith(("mean_", "median_", "p90_", "p95_", "p99_")) or key in
                    ("completed", "duration", "total_input_tokens", "total_output_tokens", "request_throughput",
                     "input_throughput", "output_throughput", "concurrency", "max_concurrent_requests"))}
    return dict(phase=phase, scheme=args.scheme or default_scheme,
                workload=args.workload or f"{args.model} | input={args.input_len} output={args.output_len} | {MODE}",
                source=f"{job['name']}.jsonl:1 · HTTP serving (includes queueing)",
                x_metric="ttft_ms" if phase == "prefill" else "per_user_tps",
                x_value=x_value,
                throughput=throughput, throughput_metric=metric, throughput_unit="tokens/s",
                num_gpus=args.num_gpus, concurrency=job["concurrency"], prefill_batch_size=None,
                prefill_service_ms=None, **parallel,
                chunked_prefill=args.chunked_prefill if phase == "prefill" else None,
                chunk_size_tokens=args.chunk_length if phase == "prefill" else None,
                chunk_scope=args.chunk_scope if phase == "prefill" else None,
                measurement=dict(mode=MODE, latency_statistic="mean", concurrency_kind="client-limit",
                                 requests=job["requests"], input_tokens=args.input_len, output_tokens=job["output_len"],
                                 seed=job["seed"], warmup_requests=args.warmup_requests,
                                 flush_cache=args.flush_cache, decode_token_scope="all-output", metrics=metrics))


def save_document(path, data):
    # Keep already-finished points importable after a failed run or interruption.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def collect(args):
    validate(args)
    output = args.output.resolve()
    run_dir = output.with_name(output.stem + ".runs")
    schedule = list(jobs(args))
    if args.dry_run:
        module = MODULES[0] if args.bench_module == "auto" else args.bench_module
        print("DRY RUN: no requests sent; installed benchmark capabilities have not been checked.")
        for job in schedule:
            print(shlex.join(command(args, job, module, run_dir / (job["name"] + ".jsonl"))))
        return None
    if output.exists() or run_dir.exists():
        raise ValueError(f"output or run directory already exists; choose a new --output: {output}")
    module, supported_flags = find_benchmark(args)
    for job in schedule:
        cmd = command(args, job, module, run_dir / (job["name"] + ".jsonl"))
        missing = {arg for arg in cmd if arg.startswith("--")} - supported_flags
        if missing:
            raise ValueError(f"{module} lacks required options: {', '.join(sorted(missing))}. Update the benchmark client.")
    run_dir.mkdir(parents=True)
    report = dict(schema="hpc-pareto-points", version=2, points=[],
                  collection=dict(mode=MODE, complete=False, planned_points=len(schedule), benchmark_module=module,
                                  started_at=datetime.now(timezone.utc).isoformat(),
                                  notes=["Prefill x is mean client TTFT, including queueing and transport, not pure service time.",
                                         "Decode throughput uses the full benchmark wall time, including Prefill and drain.",
                                         "Concurrency is a client request cap, not measured active Decode requests.",
                                         "Input throughput counts all prompt tokens; prefix cache state follows the server.",
                                         "Random token IDs are synthetic and may not reproduce real MoE routing/speculation."]))
    save_document(output, report)
    current = None
    try:
        for job in schedule:
            current = job["name"]
            raw = run_dir / (current + ".jsonl")
            log = run_dir / (current + ".log")
            print(f"[{len(report['points']) + 1}/{len(schedule)}] {current}: {job['requests']} requests, "
                  f"output={job['output_len']} tokens; log: {log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command(args, job, module, raw), env=benchmark_environment(),
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
            if result.returncode:
                raise ValueError(f"benchmark exited {result.returncode}; inspect {log}")
            rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError(f"expected exactly one JSON result in {raw}")
            point = make_point(args, job, rows[0])
            report["points"].append(point)
            save_document(output, report)
            print(f"  saved: x={point['x_value']:.3f}, throughput={point['throughput']:.3f} tokens/s", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["collection"]["failed_point"] = current
        report["collection"]["error"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
        save_document(output, report)
        raise
    report["collection"]["complete"] = True
    report["collection"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_document(output, report)
    return report


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        report = collect(args)
    except KeyboardInterrupt:
        print(f"Interrupted; completed points are saved in {args.output}", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        cli.exit(1, f"collect_pareto.py: {exc}\n")
    if report is not None:
        print(f"Saved {len(report['points'])} points to {args.output}. Import this JSON directly into Pareto Lab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
