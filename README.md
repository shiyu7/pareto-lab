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

## Select phase operating points, then balance P/D capacity

The planner follows four explicit steps:

1. Measure complete Prefill/Decode curves without preselecting an SLO. Keep workload,
   model, hardware, precision, lengths and cache conditions consistent.
2. Independently select the highest **per-GPU** throughput among feasible measured
   points: complete Prefill service time <= TTFT minus a manual latency reserve;
   Decode TPOT <= its limit. Throughput ties prefer lower latency, then fewer GPUs.
   Maximum feasible latency is not assumed to imply maximum throughput.
3. Convert each selected throughput to requests/s/GPU, then match capacities using
   actual input/output lengths and the GPU count of a complete instance.
4. Report phase configurations, GPU ratio, integer instance counts, bottleneck,
   theoretical output throughput and a joint benchmark scan.

All imported points in the selected workload participate, independently of chart
checkboxes. TP/DP/EP/PP and GPU counts must be recorded; Prefill chunk enablement
and Decode sampled concurrency must also be known. These are not inferred from labels.
The current model is `independent-stage-balance-v1`. It contains **no queue model**,
no assumed Poisson arrivals and no end-to-end TTFT prediction. The earlier
`capacity-queue-mean-v1` optimizer has been replaced.

### Complete request latency and token accounting

`prefill_service_ms` is the complete request's Prefill service time, including all
chunks but excluding queue waits. Neither a chunk step nor an existing TTFT is
silently substituted. The calibration panel supplies per-configuration fallbacks
for missing points; recorded values take precedence. A separate explicit checkbox
allows unchunked (`chunked_prefill=false`) step measurements to be used only when
the user confirms that each step covers the complete request. It never applies to
chunked points or an axis labelled TTFT. Calibrations are exported in planning JSON
and cleared by the clear-data button.

Latency reserve is manually allocated to whatever queue, batching, routing or transfer
actually lies on the first-token path. It is subtracted from TTFT before choosing P.
No additional TPOT is automatically added. Mean points do not prove tail SLOs.

Let tP, tD be selected input/output tokens/s/GPU and LP, LD be tokens counted per request:

- qP = tP / LP; qD = tD / LD, in requests/s/GPU.
- LP uses full or non-cached input length to match the imported throughput metric.
- For output throughput counting all generated tokens, LD = total output length.
  For D-only throughput excluding the Prefill-generated first token, LD = output length − 1.
  The UI requires choosing this counting scope. Total output always uses the full length.
- The supplied lengths must match the measured workload. They are conversion factors,
  not a way to extrapolate hardware performance to a new sequence-length distribution.

### GPU ratio, instance ratio and rounding

The raw GPU ratio is GP/GD = qD/qP. One instance uses gP or gD GPUs, so the continuous
instance ratio is NP/ND = gD*qD / (gP*qP). These two ratios are displayed separately.

Optional Prefill capacity reserve r (default 0) uses only (1−r) of measured P capacity
for allocation. It changes the P supply requirement, not the selected D capability.
With per-instance usable capacities cP = gP*qP*(1−r), cD = gD*qD:

- Without a target QPS, enumerate D counts and set NP = ceil(ND*cD/cP). Among allocations
  within the P surplus tolerance (default 5%), choose the one with fewest total GPUs.
  The default search cap is 1024 instances **per phase**, configurable up to 4096.
  If no candidate meets tolerance, show the smallest P surplus with an explicit warning.
  If no supply-covering ratio exists within the cap, report that while retaining selected points.
- With a target QPS Q, use NP = ceil(Q/cP), ND = ceil(Q/cD), independently. The base-ratio
  search cap does not limit this target deployment. Invalid/overflowing counts are rejected.
- Planned capacity = min(NP*cP, ND*cD). Raw theoretical capacity is also shown before
  subtracting P reserve. Full-output throughput = planned capacity × total output length.
- Scaling both instance counts by k scales theoretical throughput by k, assuming independent
  linear replicas. It leaves each selected phase's per-GPU capability unchanged.

Total P+D GPU count is used only to choose a compact integer allocation after selecting
phase points. It is not a denominator for ranking output capability. The output keeps
selected D per-GPU throughput separate from realized D throughput under a P bottleneck.

### Benchmarking the result

Record each D instance's sampled concurrency and its normalized requests/GPU. This
normalization is not a TP-rank admission limit. Confirm whether the sample records actual
active requests or a configured concurrency ceiling. The planner reports a consistency
warning if throughput/TPOT imply materially more D active requests than recorded; it does
not overwrite measured token throughput with a concurrency-derived capacity.

The scan covers 50%, 70%, 85%, 100%, 105% of target QPS, or of planned capacity if no target
was supplied. It labels the theoretical boundary and overload exploration. D active requests
are approximated by QPS × (output length−1) × TPOT. The end-to-end in-flight reference adds
complete Prefill time and the manual latency reserve. Neither is a client concurrency cap
or a guarantee of no queueing. Measure offered and completed QPS, queue growth, throughput
and TTFT/TPOT percentiles to locate the actual SLO boundary. Run closed-loop concurrency
scans separately. Synthetic examples demonstrate calculations only.

## Convert benchmark measurements

Python 3.9+ is only needed to convert raw results, not to run the hosted viewer.

Download **点位导出脚本** and the synthetic example CSV from the website's
**准备采样点位** panel; no repository clone or third-party packages are needed.
Save both files in one directory, run the command below there, then import `points.json`:

```bash
python3 plot_pareto.py pareto_example.csv --output points.json
```

Replace the synthetic CSV with your own benchmark results for real comparisons.
The script converts existing measurements; it does not launch a GPU benchmark.

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

Keep `docs/downloads/plot_pareto.py` and `docs/downloads/pareto_example.csv` in sync with
`tools/plot_pareto.py` and `examples/pareto_example.csv` when updating them; the Python tests
check that downloadable files match their maintained sources.
