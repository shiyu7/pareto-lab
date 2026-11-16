#!/usr/bin/env python3
"""Summarize measured Pareto points without mixing workloads or metric definitions."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def pos(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def escape(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def load(paths, unit):
    groups = defaultdict(list)
    warnings = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != "hpc-pareto-points" or data.get("version") not in (1, 2):
            raise ValueError(f"{path}: not a supported point document")
        points = data.get("points")
        if not isinstance(points, list):
            raise ValueError(f"{path}: points must be an array")
        collection = data.get("collection")
        if isinstance(collection, dict) and collection.get("complete") is False:
            warnings.append(f"{path.name}: incomplete collection; showing {len(points)} saved point(s)")
        if not points:
            warnings.append(f"{path.name}: no saved points")
        for index, p in enumerate(points, 1):
            if not isinstance(p, dict):
                raise ValueError(f"{path}:{index}: point must be an object")
            phase, xmetric, ymetric = p.get("phase"), p.get("x_metric"), p.get("throughput_metric")
            if (phase == "decode" and (xmetric, ymetric) != ("per_user_tps", "output_throughput")) or (
                phase == "prefill" and (xmetric not in ("ttft_ms", "prefill_latency_ms") or
                                        ymetric not in ("input_throughput", "non_cached_input_throughput"))) or phase not in ("decode", "prefill"):
                raise ValueError(f"{path}:{index}: unsupported phase/metrics")
            if not all(isinstance(p.get(k), str) and p[k].strip() for k in ("workload", "scheme")):
                raise ValueError(f"{path}:{index}: missing workload or scheme")
            if not pos(p.get("x_value")) or not pos(p.get("throughput")):
                raise ValueError(f"{path}:{index}: coordinates must be finite and positive")
            measured_unit = p.get("throughput_unit")
            if measured_unit not in ("tokens/s", "tokens/s/GPU"):
                raise ValueError(f"{path}:{index}: unknown throughput unit")
            gpus = p.get("num_gpus")
            if (unit == "per-gpu" or measured_unit == "tokens/s/GPU") and (
                    not pos(gpus) or int(gpus) != gpus):
                raise ValueError(f"{path}:{index}: GPU count required to normalize throughput")
            total = p["throughput"] * gpus if measured_unit == "tokens/s/GPU" else p["throughput"]
            y = total / gpus if unit == "per-gpu" else total
            if not pos(y):
                raise ValueError(f"{path}:{index}: normalized throughput is invalid")
            group = (phase, p["workload"], xmetric, ymetric)
            groups[group].append(dict(scheme=p["scheme"], concurrency=p.get("concurrency"),
                                      x=p["x_value"], y=y, source=path.name))
    return groups, warnings


def is_pareto(point, points, phase):
    def dominates(other):
        x_ok = other["x"] >= point["x"] if phase == "decode" else other["x"] <= point["x"]
        x_better = other["x"] > point["x"] if phase == "decode" else other["x"] < point["x"]
        return x_ok and other["y"] >= point["y"] and (x_better or other["y"] > point["y"])
    return not any(dominates(other) for other in points)


def report(groups, warnings, unit):
    label = "tokens/s/GPU" if unit == "per-gpu" else "tokens/s/endpoint"
    lines = ["# Pareto sampling analysis", "",
             f"Throughput: **{label}**. GPU normalization uses all GPUs assigned to the measured endpoint.",
             "Serving TTFT includes queueing; client concurrency is a limit, and finite sampling does not establish steady state.",
             "Decode TPOT is shown as 1000/speed; for collector points, it equals the reported mean TPOT.",
             "The frontier covers measured points only; it does not establish a global maximum beyond the sampled load range.", ""]
    if warnings:
        lines += ["## Incomplete or empty inputs", ""] + [f"- {escape(item)}" for item in warnings] + [""]
    for (phase, workload, xmetric, ymetric), points in sorted(groups.items()):
        header = ("| Scheme | Client concurrency cap | Speed (tokens/s/user) | 1000/speed (ms/token) | Throughput | Pareto | File |"
                  if phase == "decode" else "| Scheme | Client concurrency cap | x | Throughput | Pareto | File |")
        separator = ("| --- | ---: | ---: | ---: | ---: | :---: | --- |" if phase == "decode"
                     else "| --- | ---: | ---: | ---: | :---: | --- |")
        lines += [f"## {phase.upper()} · {escape(workload)}", "",
                  f"x: `{xmetric}`; throughput: `{ymetric}` · {label}.", "",
                  header, separator]
        for p in sorted(points, key=lambda item: (item["scheme"], item["x"], item["y"])):
            concurrency = str(p["concurrency"]) if p["concurrency"] is not None else "—"
            tpot = f"{1000 / p['x']:,.3f} | " if phase == "decode" else ""
            lines.append(f"| {escape(p['scheme'])} | {escape(concurrency)} | {p['x']:,.3f} | {tpot}{p['y']:,.3f} | "
                         f"{'yes' if is_pareto(p, points, phase) else 'no'} | {escape(p['source'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", nargs="+", type=Path)
    cli.add_argument("--unit", choices=("per-gpu", "total"), default="per-gpu")
    cli.add_argument("--output", type=Path, help="new report path (existing files are not overwritten)")
    args = cli.parse_args(argv)
    try:
        groups, warnings = load(args.inputs, args.unit)
        if not groups:
            raise ValueError("no nonempty point documents")
        text = report(groups, warnings, args.unit)
        if args.output:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"Saved analysis: {args.output}")
        else:
            print(text)
    except (OSError, ValueError) as exc:
        cli.exit(1, f"analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
