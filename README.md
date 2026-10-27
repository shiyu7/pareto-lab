# Pareto Lab

Browser-based Prefill / Decode performance comparison with native SVG export.

**Open the app: https://shiyu7.github.io/pareto-lab/**

## Use the viewer

1. Open the website, then import one or more point JSON files (or click **载入合成示例**).
2. Choose **阶段与 workload** to switch between **Decode** and **Prefill**.
3. Select schemes, choose total or per-GPU throughput, and optionally set an SLO.
4. Download the current SVG or export merged point data for later comparisons.

The scheme list, SVG legends, point tooltips and SLO table show structured parallel and
chunk configuration. A `?` means unknown; it is never treated as `1` or disabled.
Different TP/DP/EP/PP/GPU or chunk configurations form separate curves within the same
workload comparison, even when they share a scheme name. The example includes two
chunk sizes for the same Prefill parallel configuration.

All imported files are processed in your browser. They are not uploaded, stored on a server,
or committed to this repository. Refreshing the page clears the comparison; export merged
points if you want to save it. The bundled example is entirely synthetic.

- Decode: per-user generation speed vs output throughput (higher is better on both axes).
- Prefill: step latency or TTFT vs input throughput (lower latency, higher throughput).
- Measurements with different workloads, latency definitions or input throughput definitions
  are kept in separate comparison groups.
- The frontier is recomputed among the currently selected schemes. No interpolation is used
  to infer unmeasured SLO capacity.

## Plan a P/D deployment from TTFT and TPOT

The planner uses all imported points for one workload, independently of chart selection.
Enter mean TTFT/TPOT targets, a maximum GPU search size, mean full/non-cached input length,
mean output length, a capacity utilization ceiling (default 80%) and extra router/KV
transfer overhead. Defaults are for the **synthetic** example, not a production profile.
The supplied lengths must match the measured workload; changing them does not produce
valid extrapolation of GPU performance to a different workload.

One P or D is one complete instance, including its declared GPUs. The planner searches
measured P/D operating points and integer replica allocations within the GPU search size,
maximizing estimated output tokens/s **per total P+D GPU**. It does not try to fill
the GPU search size. All coprime P:D integer ratios in the search range are considered;
proportional scaling preserves efficiency in this model, so only primitive ratios
are displayed. This is a finite-search optimum, not a proof of a global optimum. All TP/DP/EP/PP dimensions and GPU counts must be
explicit. P chunk enablement and D measured concurrency must also be known.

Each Prefill point needs optional **`prefill_service_ms`**: mean elapsed service time for
an entire request's input, including all chunks but excluding queue/admission waits.
It is preserved by the converter. Neither step latency nor measured TTFT is substituted.
For legacy files, expand the calibration panel to supply a per-configuration fallback
for missing points; recorded per-point values take precedence. Calibrations are included
in the exported planning JSON and reset when all imported data is cleared.

The capacity/queue surrogate assumes balanced Poisson arrivals, independent instances,
and linear replica scaling. With request rates in requests/s:

- P capacity per instance = measured input tokens/s divided by mean input tokens,
  using non-cached tokens only for the non-cached throughput metric.
- D capacity = min(measured output tokens/s / output tokens,
  measured active concurrency / ((output tokens − 1) / per-user tokens/s)).
- Per-replica queue wait = ρ / [μ (1 − ρ)], with ρ = replica QPS / replica capacity μ.
- TTFT ≈ complete Prefill service + P queue + D admission queue + router/KV overhead
  + one TPOT as a first-token approximation.
- End-to-end mean in-flight requests = QPS × [TTFT + (output tokens − 1) × TPOT].

The [queueing and Little's law formulas](https://web.mit.edu/dimitrib/www/Queueing_Data_Nets.pdf)
are borrowed from basic queueing theory. Capacity and full-request latency are calibrated
separately: this is a **heuristic queue model**, not a faithful M/M/1 representation of
continuous batching. No tail-latency guarantees, burst simulation, distribution effects,
network contention, scheduler simulation or measured production optimum are implied.
Imported TPOT values must use mean statistics for this planner. P95/P99 SLOs require
matching end-to-end benchmark measurements; do not enter them as mean targets.

Results retain selected source points, parallel/chunk configurations, P/D counts,
resource use, candidate QPS, throughput, queue components and load recommendations.
The scan covers 50%, 70%, 85%, 100%, 110% of the candidate QPS (capped below raw capacity).
Use open-loop arrivals and record offered versus achieved QPS. The client protection cap
is a starting heuristic (1.3 × estimated mean in-flight, rounded up), distinct from the
measured per-D active concurrency. Hitting this cap changes the offered arrival process.
Refine near the highest measured SLO-compliant point; stop increasing load if queues grow.
Closed-loop concurrency scans are a separate experiment, with achieved QPS as an outcome.

## Convert benchmark measurements

Python 3.9+ is only needed to convert raw results, not to run the hosted viewer.

```bash
python3 tools/plot_pareto.py examples/pareto_example.csv --output /tmp/demo.points.json
python3 tools/plot_pareto.py --series EP64=ep64.jsonl \
  --phase decode --num-gpus 64 --output /tmp/ep64.points.json
python3 tools/plot_pareto.py prefill.jsonl --phase prefill --num-gpus 8 \
  --prefill-latency ttft --prefill-throughput input --output /tmp/prefill.points.json
python3 tools/plot_pareto.py prefill.jsonl --phase prefill --num-gpus 8 \
  --tp-size 1 --dp-size 8 --ep-size 8 --pp-size 1 \
  --chunked-prefill --chunk-length 8192 --chunk-scope request \
  --output /tmp/prefill-chunk8k.points.json
```

Raw inputs may be CSV, a JSON object/array, or JSONL. Throughput inputs are **instance totals**.
Use `scheme` to label each configuration and `workload` to identify model/context/precision/
hardware/cache conditions. If absent, input/output/context lengths are used as the workload.
Set `num_gpus` (or `--num-gpus`) to enable per-GPU comparison.

Parallel and chunk fields come from **recorded server configuration**, not inference from
scheme names or performance numbers. This converter does not query a live server.
Record these fields with benchmark results, or supply CLI fallbacks; explicit row values
take precedence over CLI fallbacks. If `scheme` is absent, declared parallel fields generate
a name (for example `TP1 DP8 EP8 PP1`); without any parallel fields the input filename is used.

| Input field | Meaning |
| --- | --- |
| `phase` | `decode` / `prefill`, or select it with `--phase` |
| `scheme`, `workload` | Configuration name and comparison conditions |
| `num_gpus`, `dp_size` | Total GPU count and DP rank count |
| `tp_size`, `ep_size`, `pp_size` | Declared TP, EP and PP sizes; absent dimensions remain unknown |
| `chunked_prefill` | Prefill only: `true` enabled, `false` disabled, empty/null unknown |
| `chunk_size_tokens` / `chunk_length` | Configured chunk size in tokens (`chunk_length` is a raw-input alias) |
| `chunk_scope` | `request` for a per-request chunk; `dp-rank` for a per-DP-rank scheduler token budget |
| `concurrency` / `max_concurrency` | Total concurrent requests, not total benchmark requests |
| `prefill_batch_size` | Prefill tokens per DP rank |
| `per_user_tps` | Decode speed; otherwise derived as `1000 / tpot_ms` or `mean_tpot_ms` |
| `output_throughput` | Total output tokens/s for Decode |
| `prefill_service_ms` | Complete request Prefill service time in ms, excluding queueing; optional planner calibration |
| `prefill_latency_ms` | Prefill step latency; default Prefill x axis |
| `non_cached_input_throughput` | Non-cached input tokens/s; default Prefill y axis |
| `ttft_ms` / `mean_ttft_ms` | TTFT selected with `--prefill-latency ttft` |
| `input_throughput` | Input tokens/s selected with `--prefill-throughput input` |

The exported format is `{"schema":"hpc-pareto-points","version":2,"points":[...]}`.
The viewer also accepts legacy v1 records and retains their unknown configuration fields.
Every point retains its metric names, throughput units, resource/load metadata and source.
Pareto and SLO flags are deliberately calculated in the viewer, not stored in measurements.

Chunk size is a **configuration**, `prefill_batch_size` is the **actual step input tokens per
DP rank**, and full input/context length identifies the **workload**. They are independent
fields. A request-level chunk may be smaller than the aggregate batch across many requests.
For a `dp-rank` token budget, the actual per-rank batch must not exceed that budget.
Enabled chunking requires both a positive size and an explicit scope; size alone implies
enabled, but never implies scope. `--no-chunked-prefill` records disabled chunking without
a size/scope. Missing metadata means unknown, distinct from disabled.

Example v2 point configuration (these fields coexist with all the usual metrics):

```json
{
  "tp_size": 1, "dp_size": 8, "ep_size": 8, "pp_size": 1, "num_gpus": 8,
  "chunked_prefill": true, "chunk_size_tokens": 8192, "chunk_scope": "request",
  "prefill_batch_size": 16384
}
```

## Local preview and tests

```bash
python3 -m http.server 8000 --directory docs
# Open http://127.0.0.1:8000
node --test tests/*.test.mjs
python3 -m unittest discover -s tests -v
```

No build step, npm installation, external JS libraries, CDN, API server, or analytics.
GitHub Pages publishes `docs/` from the `main` branch. Push changes to update the website.
