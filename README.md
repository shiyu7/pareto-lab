# Pareto Lab

Browser-based Prefill / Decode performance comparison with native SVG export.

**Open the app: https://shiyu7.github.io/pareto-lab/**

## Use the viewer

1. Open the website, then import one or more point JSON files (or click **载入合成示例**).
2. Choose **阶段与 workload** to switch between **Decode** and **Prefill**.
3. Select schemes, choose total or per-GPU throughput, and optionally set an SLO.
4. Download the current SVG or export merged point data for later comparisons.

All imported files are processed in your browser. They are not uploaded, stored on a server,
or committed to this repository. Refreshing the page clears the comparison; export merged
points if you want to save it. The bundled example is entirely synthetic.

- Decode: per-user generation speed vs output throughput (higher is better on both axes).
- Prefill: step latency or TTFT vs input throughput (lower latency, higher throughput).
- Measurements with different workloads, latency definitions or input throughput definitions
  are kept in separate comparison groups.
- The frontier is recomputed among the currently selected schemes. No interpolation is used
  to infer unmeasured SLO capacity.

## Convert benchmark measurements

Python 3.9+ is only needed to convert raw results, not to run the hosted viewer.

```bash
python3 tools/plot_pareto.py examples/pareto_example.csv --output /tmp/demo.points.json
python3 tools/plot_pareto.py --series EP64=ep64.jsonl \
  --phase decode --num-gpus 64 --output /tmp/ep64.points.json
python3 tools/plot_pareto.py prefill.jsonl --phase prefill --num-gpus 8 \
  --prefill-latency ttft --prefill-throughput input --output /tmp/prefill.points.json
```

Raw inputs may be CSV, a JSON object/array, or JSONL. Throughput inputs are **instance totals**.
Use `scheme` to label each configuration and `workload` to identify model/context/precision/
hardware/cache conditions. If absent, input/output/context lengths are used as the workload.
Set `num_gpus` (or `--num-gpus`) to enable per-GPU comparison.

| Input field | Meaning |
| --- | --- |
| `phase` | `decode` / `prefill`, or select it with `--phase` |
| `scheme`, `workload` | Configuration name and comparison conditions |
| `num_gpus`, `dp_size` | Total GPU count and DP rank count |
| `concurrency` / `max_concurrency` | Total concurrent requests, not total benchmark requests |
| `prefill_batch_size` | Prefill tokens per DP rank |
| `per_user_tps` | Decode speed; otherwise derived as `1000 / tpot_ms` or `mean_tpot_ms` |
| `output_throughput` | Total output tokens/s for Decode |
| `prefill_latency_ms` | Prefill step latency; default Prefill x axis |
| `non_cached_input_throughput` | Non-cached input tokens/s; default Prefill y axis |
| `ttft_ms` / `mean_ttft_ms` | TTFT selected with `--prefill-latency ttft` |
| `input_throughput` | Input tokens/s selected with `--prefill-throughput input` |

The exported format is `{"schema":"hpc-pareto-points","version":1,"points":[...]}`.
Every point retains its metric names, throughput units, resource/load metadata and source.
Pareto and SLO flags are deliberately calculated in the viewer, not stored in measurements.

## Local preview and tests

```bash
python3 -m http.server 8000 --directory docs
# Open http://127.0.0.1:8000
node --test tests/app.test.mjs
```

No build step, npm installation, external JS libraries, CDN, API server, or analytics.
GitHub Pages publishes `docs/` from the `main` branch. Push changes to update the website.
