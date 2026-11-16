# SPEED-Bench workload and collection contract

Use this for real-text P/D steady Pareto collection. The paper's dataset and the local phase-window protocol are separate: the new mode is `speed-bench-real-e2e-steady-v1`, not a claim of exact paper reproduction.

## Dataset and defaults

Official sources:
- Dataset and revision notes: https://huggingface.co/datasets/nvidia/SPEED-Bench
- Paper: https://arxiv.org/abs/2604.09557
- Official resolver/framework: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench
- AIPerf preparation/use: https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-speed-bench-dataset

The Throughput configs 1K/2K/8K/16K/32K contain 512 prompts per entropy tier, 1536 per config. Dataset v1.1 adds 100K with 128 per tier, 384 total. Check actual revision/counts at collection time. Default **1K/8K/32K x low_entropy/mixed/high_entropy** yields nine independent cells. The tiers are task-domain predictability labels, not measured per-prompt entropy values. Never pool them silently. Optional mixed-traffic results need a separately named, recorded mixture and a compatible driver; the bundled sweep does not implement such a mixture.

`qualitative` has 880 prompts across 11 domains and is for AR/AL analysis. Use the official framework for that optional analysis. The bundled corpus builder accepts single-turn Throughput records only; it rejects multi-turn input rather than flattening it. Server acceptance counters/lengths are retained where observable; the local collector does not infer conditional AR from arbitrary SSE chunks.

## Resolve, tokenize, freeze

Resolve placeholders through an existing official preparation environment. Inspect its `--help`, pin/archive its exact source and actual dataset revision. The inspected Model-Optimizer interface is:

```bash
python3 /path/to/Model-Optimizer/examples/specdec_bench/prepare_data.py \
  --dataset speed --config throughput_8k --output_dir /path/to/resolved
```

It resolves external source references and prints the output parquet path. Prepare 1K and 32K similarly; no need to fetch every optional config. Dependency installation or data preparation may require network and additional time; use an existing compatible environment or report the specific missing capability. Do not send unresolved placeholder text to the model. Do not download/execute an uninspected shell pipeline as a shortcut.

Convert resolved JSONL/parquet with the bundled helper (transformers plus pyarrow for parquet):

```bash
python3 /path/to/skill/scripts/prepare_speed_bench.py \
  --input /path/to/resolved/throughput_8k.parquet --config throughput_8k \
  --dataset-revision ACTUAL_HF_REVISION --tokenizer /path/to/local/tokenizer \
  --prompt-format chat --context-limit 65536 --max-output-tokens 1536 \
  --output /path/to/corpus/throughput_8k.jsonl
```

Use the target model's actual context limit, not the illustrative 65536. Select `completion` only for the intended raw-completion format. The builder applies the selected target tokenizer/template once and saves input IDs. It makes no padding/truncation changes. Nominal SPEED-Bench ISLs were built with another tokenizer; preserve the bucket label and archive actual target-tokenizer min/mean/max lengths. Context overflow fails with the sample ID; choose a supported workload/output cap rather than silently truncating instructions.

Each tokenized JSONL has an adjacent `.manifest.json`: dataset/config and declared revision, resolved source SHA-256, tokenizer file/template hashes, formatting choice, reserved output/context limit, per-tier counts, actual input-length statistics and tokenized SHA-256. The revision is recorded provenance, not independently verified by the tokenizer helper. Retain the official resolver source/hash, dataset fetch metadata and source files alongside it. Freeze those files across schemes/points.

## Run from the selected client

The client needs Python and aiohttp plus reachable native endpoints/counters. It does not need transformers if input IDs were prepared elsewhere. Copy all four helper files together; the sweep also archives their exact copies/hashes. `--dry-run` validates files and prints point/time coverage without requests or archive creation:

```bash
python3 /path/to/skill/scripts/collect_speed_bench.py \
  --dataset-jsonl /path/to/corpus/throughput_1k.jsonl \
  --dataset-jsonl /path/to/corpus/throughput_8k.jsonl \
  --dataset-jsonl /path/to/corpus/throughput_32k.jsonl \
  --phase decode --server-url http://router:port \
  --prefill-url http://prefill:port --decode-url http://decode:port \
  --model ACTUAL_MODEL --scheme 'ACTUAL_P_AND_D_LAYOUT' \
  --endpoint-gpus 16 --cache-profile 'ACTUAL_CACHE_AND_REUSE_CONDITION' \
  --concurrency 1,2,4,8,16,32,64,128,192,256 \
  --output-len 1536 --eos-policy natural --archive-root /path/to/archives \
  --dry-run
```

Use actual verified resource values and an authorized client-load range, then omit `--dry-run` to collect. This example plans 90 Decode points and at least 5400 seconds of measured windows, plus warmup/settling/drain. Do not launch that range as a connectivity smoke test.

For Prefill use `--phase prefill`, one explicitly chosen adequate `--concurrency`, and the **effective** `--chunk-length VALUE --chunk-scope dp-rank` (or verified request scope). Run all cells at that chunk before the next authorized runtime update. If the server lacks a setter, [prefill_runtime_patch.md](prefill_runtime_patch.md) permits a compatible SGLang patch and one rollout within the authorized benchmark scope, followed by verified runtime chunk changes. The driver itself never patches code, changes chunks, flushes caches, restarts or allocates replicas. A full six-chunk plan with one load per chunk is 54 points before repetitions. A single-chunk invocation must not be reported as the full chunk sweep.

Decode concurrency is the **global client cap**. Native active batch is observed across attention-DP ranks, which are archived separately. The helper accepts stable positive observed batch/throughput by default and flags batch below 85% of client cap as supply-limited. That operating point is useful E2E evidence but not isolated D capacity. For an explicit occupancy target use `--target-batch-fraction`, or call the lower-level helper with `--target-decode-batch`. No plateau under the bound remains a failed/unavailable point.

## EOS, warmup, replay and caching

- Default Decode cap is 1536, carried over from the earlier workflow, with natural EOS. Actual outputs are recorded. Use model-appropriate `--temperature` and `--top-p`; do not assume the temperature=0 default is appropriate for every reasoning model. The seed fixes corpus selection/order, not stochastic token sampling in the server.
- A requested fixed-length variant uses `--eos-policy fixed` (`ignore_eos=true`). Its conditions are part of the workload key; it must not be merged with natural-EOS results or called the paper's natural-output protocol.
- Default warmup holds out 16 shuffled question IDs per tier. The remaining measured corpus is deterministic and cycles only as needed for sustained refill. Thus a full 512-row tier has 496 distinct measured IDs by default; never claim all 512 contributed to its measured window. Archive actual selected IDs, epochs and request counts. Repeated replay can strongly alter cache hits, output determinism and MoE load; this is a declared replay/as-configured cache profile, not cold independent input.
- `--replay-policy stop` fails on corpus exhaustion. Use a sufficiently long distinct corpus or a separately labeled finite-cohort protocol if no replay is allowed; do not invent extra random samples or lengthen outputs to fill the window.
- No artificial common prefix/suffix is added. Natural source overlap is preserved. Warmup IDs are disjoint, but underlying source contexts may overlap. Record actual cached tokens and native non-cached work; user-authorized cache resets belong in orchestration and change the workload condition.

## Results and limits

The unique archive contains `manifest.json`, `sources/`, copied `corpora/` and one directory per config/entropy/phase. Each cell has incrementally saved `points.json`, per-load observations, `.events.jsonl` and driver logs. The manifest distinguishes planned/completed points and retains the first error. A failed cell has no invented coordinate; successful earlier points remain exportable and marked partial. Transport disconnects require checking the owned process/files before retrying.

Viewer fields use `hpc-pareto-points` v2. Coordinates use **the same accepted windows**:
- P: summed uncached input tokens / native wall time, against busy-time / representative-forward scheduler cadence in ms. It is not CUDA kernel timing or complete request Prefill service time.
- D: delivered D-generated tokens (excluding the first P-produced token) / window wall time, against eligible token transitions / corresponding summed inter-token exposure. Timing is burst-granular. This token-weighted speed is not the arithmetic mean of individual request rates.
- Total tokens/s includes all observed ranks. Default GPU scope is the whole endpoint. `--gpu-scope phase --phase-gpus N --supply-evidence FILE` is available only after the operator verifies isolation/sufficient supply and the actual phase count; attaching a file does not automate proof. An underfilled/supply-limited point is not isolated phase capacity.

The workload key includes config, entropy, corpus hash, model, cache/replay, sampling/EOS and output cap. The scheme identifies actual hardware/precision/P/D parallelism. Never merge changed conditions because curves otherwise look sparse. Keep source-resolution/observer/server provenance with the archive. SLO selection and whole-service tail verification occur after frontier collection.

Analyze the exported files with `analyze_points.py --unit total`, then import each cell's `points.json` once into the existing Pareto Lab. Verify separate groups and actual counts. Existing synthetic archives remain separate. Qualitative AR/AL and official whole-run SPEED-Bench/AIPerf measurements are complementary, separately named protocols.
