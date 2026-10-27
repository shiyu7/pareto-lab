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
node --test tests/app.test.mjs
python3 -m unittest discover -s tests -v
```

No build step, npm installation, external JS libraries, CDN, API server, or analytics.
GitHub Pages publishes `docs/` from the `main` branch. Push changes to update the website.
