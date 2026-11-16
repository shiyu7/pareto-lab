# Real E2E phase Pareto protocol

This is the measurement contract for a steady-state collector. The pinned GitHub HTTP baseline does not implement it. The separate bundled `scripts/collect_pareto_steady.py` implements bounded real-stream/native-counter window collection; service changes and transport remain orchestration tasks; collect_speed_bench.py performs the SPEED-Bench sweep and viewer export. Inspect actual capabilities before choosing a tool; changing metadata or splitting JSON does not supply a missing measurement.

## Objectives and fixed conditions

- Collect phase frontiers without SLO pruning. Apply TTFT/TPOT choices after collecting points. Prefill optimizes actual non-cached input throughput per P GPU versus measured Prefill step latency; also retain full request TTFT and, if observable, complete Prefill service time. Decode optimizes actual committed output throughput per D GPU versus per-user generation speed. Do not use total input+output throughput as either phase's objective.
- Default SPEED-Bench inputs are throughput_1k/8k/32k, with low_entropy/mixed/high_entropy measured separately. Use target-tokenizer actual lengths, not a nominal bucket as a fabricated fixed length. Decode defaults to natural EOS with a 1536 output cap; Prefill cap is 1. Fixed output is an explicit, separately labeled variant. Keep real weights, KV production/transfer, sampling, DSPARK, attention and MoE paths. No fake KV, force-accepted speculation or longer outputs to manufacture a plateau.
- Hold hardware, topology, precision, cache policy, scheduler policy and other phase configuration fixed within a comparison. Record warmup, cache hits, actual context-length distribution and changes caused by output generation. A shared prefix is a declared workload characteristic, not an approximation to independent prompts.

## Chunk changes, authorization and restart cost

Plan the target Prefill workload/controller, original effective configuration,
planned chunk values, any code patch, restart impact and restoration. Reuse
existing authorization for those service changes; ask only for missing scope.
This skill permits a small, reversible SGLang patch to accelerate an authorized
Prefill sweep, without a separate patch-only approval step. Inference-only or
Kubernetes/SSH access to a shared service is not service-change authorization.
Do not restart Decode merely to change Prefill chunk.

1. Inspect the installed version's runtime update implementation. A route named
   `/set_internal_state` is insufficient: chunk must be supported, update the
   cached scheduler budget and configuration, and reach DP/TP/PP participants.
   Read back effective values and check allocation/workspace/capture limits.
2. If the setter or required observations are missing, follow
   [prefill_runtime_patch.md](prefill_runtime_patch.md). Prepare a compatible,
   bounded patch, validate it, and deploy it once within the authorized scope.
   If the planned maximum fits, provision its startup ceiling in that rollout;
   a setter capped at the original 8192 cannot subsequently admit 32768.
   Record the patch/base revision, loaded worker code and revert procedure.
3. With a verified setter, stop new benchmark arrivals, drain this run's requests,
   update chunk at a safe boundary, verify replies/readbacks and participant
   coverage, warm the new shape, and run all planned length/entropy cells. Keep
   cache conditions and complete measurement windows unchanged. Archive setup,
   restart/readiness/warmup and measurement time separately.
4. If a compatible setter cannot be used, group all planned cells and repetitions
   under each chunk to minimize restarts. Start at the current setting when
   appropriate and account for restoration. Do not silently allocate extra
   replicas or compare concurrently competing replicas as unchanged hardware.

After success, failure or interruption, restore this run's authorized code and
configuration changes and verify both the loaded code/image and effective chunk,
including any required rollback restart. Retain a patch only when requested.
Check for intervening changes before restoring; report any unresolved restoration.
Record unsupported/OOM chunks as unavailable rather than substituting values.
The linked PR is a version-specific example, not evidence the running server
already supports the setter.

## Prefill workload and point selection

Use the prepared real-text SPEED-Bench corpus and disjoint warmup/measured question IDs. Preserve corpus content across chunks; do not generate random IDs or alter original text to defeat caching. Record actual cached tokens; do not call total prompt tokens non-cached work. Record deterministic order, replay epochs and actual cross-point cache reuse; without an authorized cache reset this is a declared warm/replayed condition, not cold non-cached input. Flushing cache is a separate explicit operation; a chunk-change authorization does not imply it.

Chunk is the six-point sweep variable. Client rate/concurrency is used to provide sufficient work at each chunk, within an authorized bound. Pilot for adequate input supply, actual per-rank token fill, and a non-growing queue; hold a reproducible load policy across points. Saturating an engine can require queued work, so client TTFT at saturation must not be presented as queue-free service time. Record configured budget, actual non-cached batch tokens, request mix/context positions and fill distribution. A 32768-token budget may remain underfilled on a nominal 8K workload; report actual fill instead of assuming a full chunk.

Warm the actual scheduled shape: a single 8K request does not warm a 16K token batch formed from multiple requests. The helper's `--warmup-concurrency` supports bounded real-request shape warmup, with `--request-timeout` separating a cold-shape allowance from the steady-window duration. Record these settings. If a request stalls, inspect timestamp freshness, P progress counters and D transfer queues; do not label a socket timeout as a proven GPU/kernel failure or silently extend timeouts indefinitely.

Use actual completed P-work counters in the accepted window divided by elapsed wall time and P GPUs for phase throughput, with correct DP summation and no duplicate TP/PP counting. Verify that Decode/transfer/metadata slots are not backpressuring Prefill. If only client request-completion counters are available, export the value as E2E input throughput with that scope and account for window boundary work; do not rename it engine-only throughput.

Keep these latencies distinct:

- Prefill step latency: engine-measured steps; retain actual batch tokens and PP timing definition. This matches a DeepInfer-style step chart.
- Complete Prefill service time: actual full request execution across all chunks, excluding queue/transfer where observable; do not infer it from one chunk or blindly multiply mean step time by a chunk count.
- TTFT: real client observation, including any P/D handoff, waiting and network. This is the user-facing SLO metric.

Without step observations, a chunk-swept **TTFT** chart is still useful, but label its axis accordingly; do not claim it is a DeepInfer step-latency chart. The phase allocation planner still needs measured complete request service time/calibration.

## Decode workload and plateau

Use the same SPEED-Bench entropy-specific real prompts on the actual Router/P/D path, retaining real KV production and transfer. Do not create an artificial common prefix or append synthetic token suffixes. Shared context naturally present in the source corpus remains intact and is recorded through actual cache hits. Warmup uses held-out question IDs. Decode uses natural EOS under the declared output cap unless the user requests fixed output; retain actual output-length statistics. Verify that P and transfer replenish D fast enough, otherwise label the point supply-limited. Finite-corpus replay is explicit in provenance and changes cache conditions; it must not be described as independent cold input.

The legacy synthetic whole-run collector's eight request waves reduce boundary dilution but its whole-run summary still includes ramp-up/drain. Reuse dataset and mode semantics, not the assumption that a long run is already a measured plateau. Its Prefill search currently uses total token throughput; use non-cached input throughput for the new phase objective instead.

Preferred steady serving procedure:

1. Choose a target **actual active Decode batch**, observed per DP rank, and a bounded arrival/concurrency policy. Maintain a real request reservoir and replace completions while measuring. Client MC includes Prefill, transfer and waiting, so MC=64 does not imply active Decode batch=64. A server running-request limit can help only if its scope is understood and changing it is authorized.
2. Warm up and wait for occupancy to reach the target. Predeclare stability criteria and a maximum settle/run duration. Example starting settings are three consecutive 5-second bins with at least 95% of observed time near the target batch, no sustained occupancy/queue growth and no large throughput drift, followed by a fixed 30-second window. These are configurable acceptance heuristics, not proof of steady state; require sufficient decode steps and completed request replacements for the chosen workload. Monitor both throughput and occupancy; a flat but underfilled batch is a different operating point.
3. Start the first eligible measurement window after settling, before knowing its score. Continue replenishing arrivals for the **entire** fixed window. Finish measurement before stopping arrivals, then drain this run's requests outside the window. Exclude the initial fill and final drain by lifecycle/state, not a fixed first/last percentage of total requests.
4. Retain the complete contiguous window, including natural refill holes, ordinary scheduler stalls and legitimate slow steps. Do not cherry-pick peak seconds, drop slow intervals to meet stability, or sum only full-batch GPU busy time as E2E throughput. Record actual batch mean/distribution, undersupply fraction, queue/transfer state, accepted tokens and DSPARK accept length. If occupancy cannot be maintained, mark the point supply-limited/no plateau, or report its actual batch with uncertainty; do not label the client cap as achieved D capacity.
5. Repeat accepted windows under the same policy and report dispersion. Use bounded retries; a missing plateau at fixed output length/load limits is a result, not permission to increase output length, force acceptance, restart or increase load indefinitely.

For a short fixed-output cohort where continuous replenishment is unsuitable, an alternative is repeated real-request cohorts and measurement only on the common full-batch overlap: after all target requests enter Decode and before the first leaves. Verify adequate steps/duration in each cohort. Report this as a cohort plateau, separately from sustained serving throughput. If no overlap exists, especially with 128K Prefill and short fast Decode, do not fabricate a plateau or silently lengthen outputs. This alternative conditions on the cohort's context/age distribution and does not establish replenishment capacity.

## Time-resolved observations and aggregation

Archive time-indexed request/stream events, actual Decode batch/DP-rank observations, P/D/transfer queue state and relevant engine work/token counters. Preserve interval token counts and committed/accepted output tokens, not speculative proposals. An SSE event can contain multiple tokens, especially with DSPARK; event count is not token count. Separate the first P-produced output token from D-generated output. Record observer overhead and cadence; coarse gauge polling can miss batch holes and must be labeled approximate. If step-level timing is unavailable, say so rather than calling polling samples exact engine steps.

For remote runs, prefer a run-owned detached client with file-backed stdout and an atomic per-point exit-status file, queried using short Kubernetes/SSH calls. A lost control connection is not evidence that inference stopped or measurements failed. Check the owned process, complete raw/observation files and window/request validity before retrying, changing chunk or restoring a service; retain transport failures in provenance even when complete measurements are recovered. A bounded timeout should stop only the verified run-owned process group before service changes. Retry missing points in a new linked archive instead of rerunning successful phases.

Use one monotonic client clock for request/stream observations and window boundaries; when joining server events, establish clock alignment and uncertainty. Server counter deltas can use paired timestamps without pretending separate machine clocks are identical. Handle resets/restarts, lag and TP/PP duplicates explicitly. Preserve ordinary idle time in the elapsed window denominator.

For a window W of duration T:

- D throughput/GPU = committed **D-generated** output tokens in W / T / D GPU count.
- Window token-weighted TPOT = sum of eligible per-request inter-token durations wholly in W / sum of corresponding token transitions. Count actual token multiplicity. A multi-token streaming burst does not supply exact timestamps for every accepted token; describe that timing granularity. Also retain the distribution of per-user rates and client-observed stream gaps; `1 / mean(TPOT)` is not generally the mean of individual users' rates.
- With a truly constant active batch B and all per-user Decode exposure accounted for, token-weighted per-user speed equals output TPS/B. This is a consistency check, not a way to invent B from client MC. With changing B, retain request exposure and compute sum(tokens)/sum(active request seconds), plus actual batch distribution.

Use the same accepted time window for the graph's throughput and latency/speed. Do not pair trimmed throughput with the whole-run benchmark's TPOT/TTFT. A client's full-request TPOT includes portions outside W; either recompute eligible intervals or keep it explicitly as a separate whole-request metric. Similarly retain original TTFT percentiles for request cohorts, with boundary/censoring rules, rather than inferring TTFT from token timestamps after first output. Preserve tail-SLO evidence separately from mean frontier metrics.

Use separate P, D and endpoint GPU counts. Phase-normalized output is only evidence of D capacity when observed batch is achieved and P/transfer supply is sufficient; otherwise describe an E2E rate per allocated D GPU. Keep cache-reuse conditions and actual context/age distributions in comparison groups, and avoid mixing steady windows, cohort plateaus, fake-KV runs and whole-run baselines.

## Export and implementation status

Store Prefill and Decode separately, keeping whole-run baselines and raw time series for audit. Include window bounds/duration, acceptance criteria/results, actual batch statistics, selected latency definition/statistic, GPU scope, configured/actual chunk, cache state and point completeness. Reject or explicitly label points with no accepted window. Apply SLO selection to exported measured points later.

For Prefill chunk sweeps, declare `sweep_parameter="chunk_size_tokens"` so the viewer treats chunk as the point label/sweep variable within a fixed parallel scheme. Without that declaration it deliberately treats chunk as fixed configuration and separates curves. Do not discard chunk metadata to make a line appear.

Pareto Lab now supports `sweep_parameter="chunk_size_tokens"` to connect the chunks of a fixed Prefill scheme, and `measurement.actual_decode_batch_mean` for a measured Decode batch label. Keep actual chunk metadata on each point. Its whole-run HTTP download remains distinct from the new steady observation helper. Native P busy-time/representative-forward-count deltas provide a scheduler completion-cadence estimate under PP; label it accordingly, retain counter endpoints, and do not describe it as full-request or exact CUDA-event latency. Native Decode moments skip certain idle boundaries, so use real-stream delivered token counts and full wall time for E2E throughput; retain native moments to validate actual batch/exposure. Report implemented capabilities and do not claim a sweep completed until its archived data exists.


## SPEED-Bench helper semantics

The default helper now consumes validated tokenized corpus files; legacy random input requires `--synthetic`. See [speed_bench.md](speed_bench.md). `--target-decode-batch` is an optional **global observed batch** target, independent of client maximum concurrency. Without it, stable throughput and stable positive observed batch qualify an observed E2E operating point; batch below 85% of client concurrency is labeled supply-limited. That label is a diagnostic heuristic, not causal proof that P is the bottleneck. Other rank imbalance, queue, transfer and cache evidence must still be inspected. With a target, the helper also requires at least 85% of that target, preserving the earlier bounded check. The current 20% bin variability and 85% target thresholds are explicit helper heuristics; archive them and do not substitute the reference's stricter illustrative criteria as though implemented. Native P tokens/busy-time sum per-DP-rank deltas; representative forward counts must have matching DP/TP/PP scope.

The new wrapper exports accepted observations with exact whole-window denominators and preserves failed raw data. Default total throughput covers all serving ranks. Its endpoint GPU normalization is not pure phase capacity. No native counters means the steady helper is unavailable: an official whole-run SPEED-Bench framework/AIPerf run can still be useful if explicitly labeled with that protocol; do not silently switch to the old synthetic helper or claim steady results.
