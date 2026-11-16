---
name: pareto-serving-benchmark
description: Collect, archive, analyze, and display SPEED-Bench-based Prefill/Decode steady-state Pareto points for an existing SGLang deployment via Kubernetes or SSH. Use for real-text length/entropy/concurrency or Prefill-chunk sweeps and serving configuration comparisons. Not for isolated GPU kernel profiling.
---

# SPEED-Bench Prefill/Decode Pareto sampling

Use real SPEED-Bench text on the actual P/D request path, continuously replenish requests, and measure time-resolved steady windows. Preserve the existing skill name and Pareto Lab import format. This is a **SPEED-Bench workload adaptation of phase steady sampling**, not a claim to reproduce the paper's entire protocol or a queue-free engine microbenchmark.

Read [references/speed_bench.md](references/speed_bench.md) for corpus preparation, executable examples, defaults, provenance and export. Read [references/steady_state_e2e.md](references/steady_state_e2e.md) for phase measurement, chunk changes, native counters and window validity.

## Default sampling plan

- Throughput configs: **throughput_1k, throughput_8k, throughput_32k**, each split into **low_entropy, mixed, high_entropy**: nine distinct workload cells. Offer 2K/16K or the current dataset's 100K config when requested/relevant; never silently replace 100K with 128K. These labels are nominal ISL buckets, not target-model token counts.
- Record actual post-template target-tokenizer lengths. Preserve real prompt content. No random token replacement, arbitrary shared prefix, 16-token synthetic suffix, fake KV or forced speculative acceptance in this mode.
- Prefill: output cap 1, scan effective chunk budgets **1024,2048,4096,8192,16384,32768** when service changes are authorized. Use one recorded, sufficient client-load policy per comparison. The helper measures one configured chunk per invocation; chunk metadata does not configure the server.
- Decode: output cap **1536** retains the earlier workflow's scale; default EOS is **natural**, so actual outputs can be shorter. This cap is a local convention, not SPEED-Bench's prescribed output length. Honor requested/model-specific generation settings. A requested exact output length uses `--eos-policy fixed`; keep it in a separate comparison group.
- Decode starting sweep: global client caps **1,2,4,8,16,32,64,128,192,256**, including the high-concurrency points requested in the linked chat. First establish the target's requested load bound; truncate to it, and report unmeasured points. Do not stop at 32 and claim an EP deployment's maximum. Client concurrency is not per-rank concurrency or actual active Decode batch.
- After bounded settling, sample **two consecutive 30-second windows**, preserving ordinary refill gaps/stalls. Use the first eligible windows, not the fastest windows; collect the frontier before applying SLO selection. Report observed batch, per-rank data and supply-limited points. A fixed output cap may fail to sustain a plateau; record that result without changing output length or load to manufacture one.

## Identify the environment and scope

Honor the user's endpoint, model, client and Kubernetes/SSH route. Otherwise inspect only enough current state to identify an unambiguous target; ask if ambiguous. Do not infer an SSH target from a Kubernetes node address. Reuse existing configured credentials without collecting secrets.

Find a client with Python, aiohttp and access to the Router `/generate`, native P/D `/v1/loads?include=all` and P `/metrics`. Verify the native schema and counter freshness; SGLang versions may not expose the required counters. Tokenization/source preparation can happen locally and be transferred, so remote GitHub/Hugging Face access is unnecessary. Declare service-code patches/restarts in the benchmark plan and use the applicable authorization below. Do not silently install packages, enable telemetry or flush caches. A small connectivity smoke run is appropriate before an unverified full run; keep smoke data separate.

Record effective model/precision, hardware/topology, P and D parallel layouts, chunk scope, cache policy, speculative method/draft settings, observer cadence and competing traffic. Confirm allocated endpoint GPUs and retain separate P/D GPU counts. Default export uses endpoint GPU accounting and total tokens/s, matching the linked chat's total-throughput convention. Phase-only GPU normalization requires a verified count plus archived evidence that the other phase/transfer is not limiting; the helper's stability test alone cannot establish this. Never divide by attention-DP ranks and call it total or per-GPU throughput.

For collection, reuse authorization already given for the specific run/mutation scope. Chunk-change authorization must cover target, values, necessary restarts and restoration. Follow the reference's runtime-setter verification and restoration procedure; group all length/entropy cells under each chunk to minimize restarts.

## Accelerate Prefill setup with an SGLang patch

This skill allows small, reversible SGLang source patches for runtime chunk changes or required native observations when they reduce total Prefill test time. Prefer an existing verified setter; otherwise prepare a compatible bounded setter, deploy once, and sweep chunks without reloading weights at every point. Read [references/prefill_runtime_patch.md](references/prefill_runtime_patch.md) before this path; it includes the `poc_dsv4.1` PR #895 example, startup allocation ceiling, per-attention-DP-rank values, verification and rollback.

Apply service edits within the already-authorized benchmark target, restart and restoration scope. Do not require another confirmation solely because the implementation uses a patch; inference-only access to a shared service is insufficient. Keep real Prefill work and the two 30-second windows intact. The collectors remain read-only: patching, rollout, runtime updates and restoration are orchestration work, with exact code/configuration and costs archived.

## Prepare and run

1. Resolve the official `nvidia/SPEED-Bench` sources with the official preparer at an inspected revision. Reject unresolved placeholder text. Record dataset revision and exact preparer/hash; prepare only requested configs. Do not substitute similarly named datasets.
2. Run bundled `scripts/prepare_speed_bench.py` on the resolved single-turn throughput JSONL/parquet, using an existing local target tokenizer and declared chat/completion format. It writes tokenized JSONL plus a SHA-256 manifest, all three categories and actual length statistics. It rejects overflow instead of truncating. Archive the corpus and source-resolution provenance. Qualitative multi-turn AR/AL evaluation uses the official framework separately; do not flatten it into this throughput sampler.
3. Run `scripts/collect_speed_bench.py --help` in the selected client. Use `--dry-run` first to check cells, points and minimum measured time. Pass explicit URLs, verified GPU counts, scheme/cache conditions and prepared files. It validates hashes, runs `collect_pareto_steady.py` once per cell/load and exports viewer-compatible points. For Prefill run one authorized/effectively verified chunk at a time, covering all cells before the next chunk.
4. The finite real corpus is replayed deterministically by default to replenish load. Reserve disjoint question IDs for warmup; archive selected IDs, epochs, replay counts and actual cache hits. Replay is **not a cold-cache workload**. `--replay-policy stop` forbids replay and reports exhaustion if the corpus cannot cover settling/windows. Cross-point cache reuse remains part of the measured condition; no implicit flush or cache-policy mutation.
5. Retain exact input IDs across schemes/chunks. Keep nominal length, entropy, sampling/EOS, cache/replay conditions, corpus fingerprint and metric definitions in `workload`; this separates incompatible curves automatically. Keep all failed/partial observations, with no fabricated point for a missing window.

## Transport, archiving and display

The new sweep helper executes in a chosen client; it does not implement Kubernetes/SSH transport itself. Copy the four bundled scripts (`collect_speed_bench.py`, `collect_pareto_steady.py`, `speed_bench_data.py`, `prepare_speed_bench.py`) and prepared corpora/manifests to a unique run-owned client directory using the existing Kubernetes or configured SSH route. For remote runs, launch a run-owned detached process with file-backed logs and an atomic exit-status file; poll via short control calls. Retrieve the complete archive to a unique local directory and verify its hashes. Do not execute a pinned legacy random-token collector for a SPEED-Bench run.

A dropped control connection does not stop remote inference. Before retry/restoration, inspect the verified run-owned process and saved points; do not start a duplicate. Stop only that run's process group when required. Retry missing points in a new linked archive. The helper stops on the first failed point and preserves successful partial exports, corpus copies, exact scripts/hashes, logs and raw events in a mode-0700 archive. It does not overwrite previous runs. Restore only this run's authorized service changes, checking for intervening changes.

Analyze nonempty `*/points.json` with `scripts/analyze_points.py --unit total`; per-GPU analysis uses each point's declared GPU scope. Import those files into the user's existing Pareto Lab file-import control (not raw observations). Verify separate length/entropy/phase groups and point counts; preserve existing browser data and avoid duplicate imports. Do not publish measured data, corpora or cluster details to the public viewer repository. If the viewer is unavailable, deliver the archive/analysis and state the display blocker.

Return the local archive, observed coverage (including 64/128/192/256 status), per-cell completion/plateau status and key Pareto results with explicit metric and GPU scope. No missing or supply-limited point proves a global maximum. Phase allocation needs matched workloads and complete Prefill service-time calibration; step cadence/TTFT alone cannot supply it. Final deployment tail-SLO verification remains an E2E task.

## Capacity matching

Use compatible measured points that satisfy the declared latency criteria. Convert phase throughput to per-instance request capacity: `q_P = X_P / U`, `q_D = X_D / K`, with mean non-cached input `U` and mean Decode output `K` matching the measured token scopes. For fixed `N` output tokens, `K = N - 1` only when D throughput excludes the first P-produced token.

For a chosen D instance count, compute `n_P = ceil(n_D * q_D / q_P)` and `capacity = min(n_P * q_P, n_D * q_D)`. For target QPS `lambda`, independently compute `n_P = ceil(lambda / q_P)` and `n_D = ceil(lambda / q_D)`. Total GPUs are `n_P * g_P + n_D * g_D`; the continuous P:D GPU ratio is `(q_D / g_D) / (q_P / g_P)`. Use these measured capacities directly when calculating or reviewing a deployment candidate; integer rounding and ratio-approximation tolerance are separate from capacity measurement.

These calculations propose an allocation, not a tail-SLO guarantee. Joint E2E load sweeps must verify queue growth, actual D batch/supply, KV transfer and the declared TTFT/TPOT statistics; adjust the allocation from that evidence.

## Explicit legacy diagnostics only

`scripts/collect_and_archive.py` and `snapshot_running.py` retain the old pinned GitHub whole-run synthetic baseline for explicitly requested legacy diagnostics/recovery. Their 8K/32K/128K, forced-length/random-token semantics are not SPEED-Bench. `collect_pareto_steady.py --synthetic` also requires an explicit choice. Keep these results and original definitions separate from new SPEED-Bench archives; never relabel old points.
