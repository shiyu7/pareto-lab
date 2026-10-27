#!/usr/bin/env python3
"""Export prefill/decode measurements for the Pareto Lab SVG viewer. No image dependencies."""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Point:
    phase: str
    workload: str
    scheme: str
    x: float
    throughput: float
    concurrency: float = None
    batch_tokens: float = None
    dp_size: float = None
    num_gpus: float = None
    source: str = ""
    x_metric: str = ""
    throughput_metric: str = ""
    throughput_unit: str = "tokens/s"


def number(row, *keys, required=False, integer=False):
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a positive number")
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be a positive number") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        if integer and not value.is_integer():
            raise ValueError(f"{key} must be an integer")
        return value
    if required:
        raise ValueError(f"missing {' / '.join(keys)}")
    return None


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = [rows]
        else:
            raise ValueError("expected .csv, .json or .jsonl")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must contain at least one measurement")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every measurement must be an object")
    return rows


def make_point(row, scheme, source, args):
    phase = row.get("phase") or row.get("mode")
    phase = {"decode-only": "decode", "prefill-only": "prefill"}.get(phase, phase)
    if not phase and args.phase != "both":
        phase = args.phase
    if phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode (or specify --phase)")
    if args.phase != "both" and phase != args.phase:
        return None

    # Explicit workload labels can include model, context, precision and hardware.
    workload = str(row.get("workload") or "").strip()
    if not workload:
        dimensions = []
        for key, aliases in (
            ("input", ("input_len", "random_input_len")),
            ("output", ("output_len", "random_output_len")),
            ("context", ("context_len",)),
        ):
            value = number(row, *aliases, integer=True)
            if value is not None:
                dimensions.append(f"{key}={value:g}")
        workload = ", ".join(dimensions) or "default"

    # A supplied completion count must describe a successful measurement.
    for key in ("failed", "failed_requests"):
        if key in row and row[key] not in (None, ""):
            failures = float(row[key])
            if not math.isfinite(failures) or failures != 0:
                raise ValueError(f"{key} must be zero")
    completed = number(row, "completed", "successful_requests", integer=True)
    expected = number(row, "num_prompts", integer=True)
    if completed is not None and expected is not None and completed != expected:
        raise ValueError("completed requests differ from num_prompts")

    gpu_row = dict(row)
    if gpu_row.get("num_gpus") in (None, ""):
        gpu_row["num_gpus"] = args.num_gpus
    num_gpus = number(gpu_row, "num_gpus", integer=True,
                      required=args.throughput_unit == "per-gpu")
    if phase == "decode":
        x = number(row, "per_user_tps")
        if x is None:
            x = 1000 / number(row, "tpot_ms", "mean_tpot_ms", required=True)
        throughput = number(row, "output_throughput", required=True)
    else:
        if args.prefill_latency == "step":
            x = number(row, "prefill_latency_ms", required=True)
        else:
            x = number(row, "ttft_ms", "mean_ttft_ms", required=True)
        field = ("non_cached_input_throughput"
                 if args.prefill_throughput == "non-cached" else "input_throughput")
        throughput = number(row, field, required=True)
    if args.throughput_unit == "per-gpu":
        throughput /= num_gpus
    if not math.isfinite(x) or not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("derived speed/throughput must be finite and positive")
    return Point(
        phase=phase, workload=workload,
        scheme=str(scheme or row.get("scheme") or "").strip(),
        x=x, throughput=throughput,
        concurrency=number(row, "max_concurrency", "concurrency", integer=True),
        batch_tokens=number(row, "prefill_batch_size", integer=True),
        dp_size=number(row, "dp_size", integer=True), num_gpus=num_gpus,
        source=source,
        x_metric="per_user_tps" if phase == "decode" else (
            "prefill_latency_ms" if args.prefill_latency == "step" else "ttft_ms"),
        throughput_metric="output_throughput" if phase == "decode" else field,
        throughput_unit="tokens/s/GPU" if args.throughput_unit == "per-gpu" else "tokens/s",
    )


def load_points(args):
    inputs = [(None, Path(path)) for path in args.inputs]
    for spec in args.series:
        label, separator, path = spec.partition("=")
        if not separator or not label.strip() or not path:
            raise ValueError("--series must be NAME=PATH")
        inputs.append((label, Path(path)))
    if not inputs:
        raise ValueError("provide input files or --series NAME=PATH")
    points = []
    for label, path in inputs:
        try:
            rows = read_rows(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for index, row in enumerate(rows, 1):
            source = f"{path}:{index}"
            try:
                point = make_point(row, label, source, args)
                if point is not None:
                    if not point.scheme:
                        point.scheme = path.stem
                    points.append(point)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{source}: {exc}") from exc
    if not points:
        raise ValueError("no measurements match --phase")
    layouts = {}
    for point in points:
        key = (point.phase, point.workload, point.scheme)
        layout = (point.num_gpus, point.dp_size)
        if key in layouts and layouts[key] != layout:
            raise ValueError(f"{key}: GPU/DP counts change within a curve; use distinct scheme names")
        layouts[key] = layout
    return points


SCHEMA = "hpc-pareto-points"
VERSION = 1


def document(points):
    """Versioned, self-describing points; frontier and SLO depend on the comparison."""
    rows = []
    for point in points:
        row = asdict(point)
        row["x_value"] = row.pop("x")
        row["prefill_batch_size"] = row.pop("batch_tokens")
        rows.append(row)
    return {"schema": SCHEMA, "version": VERSION, "points": rows}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("inputs", nargs="*", help="CSV, JSON object/array, or JSONL measurements")
    result.add_argument("--series", action="append", default=[], metavar="NAME=PATH",
                        help="override scheme name for a file; repeat for multiple schemes")
    result.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    result.add_argument("--num-gpus", type=int, help="fallback GPU count for per-GPU normalization")
    result.add_argument("--throughput-unit", choices=("per-gpu", "total"), default="total")
    result.add_argument("--prefill-latency", choices=("step", "ttft"), default="step")
    result.add_argument("--prefill-throughput", choices=("non-cached", "input"), default="non-cached")
    result.add_argument("--output", type=Path, default=Path("pareto_points.json"),
                        help="versioned sampling-point JSON output (default: pareto_points.json)")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.num_gpus is not None and args.num_gpus <= 0:
        cli.error("--num-gpus must be positive")
    try:
        points = load_points(args)
        # Refuse an input/output collision instead of replacing benchmark data.
        inputs = [Path(p) for p in args.inputs]
        inputs += [Path(spec.partition("=")[2]) for spec in args.series]
        if any(path.resolve() == args.output.resolve() for path in inputs):
            raise ValueError("--output must differ from all input paths")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document(points), ensure_ascii=False,
                                         allow_nan=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(points)} sampling points to {args.output}")
    except (OSError, ValueError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
