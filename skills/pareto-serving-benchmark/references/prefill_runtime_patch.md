# Accelerate Prefill sweeps with a bounded SGLang patch

Use this when repeated chunk changes would otherwise reload weights, or when a
small observation patch is needed to collect the required native counters. The
goal is to reduce setup, restart and recovery time while preserving the real
Prefill workload and accepted measurement windows.

## Scope and choice of path

Preparing a small, reversible SGLang source patch is an allowed part of this
skill. Apply it to the identified benchmark deployment when the user's existing
authorization covers that deployment's code/configuration changes, any required
restart and restoration. Reuse that authorization; do not ask for a separate
confirmation solely because the implementation uses a patch. Permission only to
send inference requests to a shared service does not authorize changing its code.
An unspecified target, extra replica, broader outage or unrelated optimization
still needs the missing scope resolved.

Prefer, in order:

1. An already-supported and verified runtime chunk setter.
2. A compatible bounded setter patch, deployed once, followed by runtime changes.
3. Grouped restarts if a setter cannot be made compatible within the run's bounds.

Do not require a separate replica or an upstream merge before using an inspected
local patch on the authorized benchmark deployment. Do not apply a patch or
restart a service merely to update this skill.

## Inspect and prepare the patch

Record the running SGLang revision, actual import/source path in each serving
container, launch command, effective per-rank chunk, attention-DP partitioning,
page/truncation alignment and relevant buffer/CUDA-graph limits. Save the original
source or image reference and a revert patch before editing. A file changed on
disk is not evidence that existing Python worker processes loaded it.

An inspected example is [bytedance-iaas/sglang PR #895](https://github.com/bytedance-iaas/sglang/pull/895),
commit `ac287f8f872f27a6aab65cf631a64e4c60ad2f64`, targeting `poc_dsv4.1`.
It adds `chunked_prefill_size` to the existing runtime controller. Check the live
version and PR status before adapting it; this link does not mean every SGLang
release contains the feature. Do not blindly cherry-pick a patch from a different
branch or overwrite other local modifications.

A runtime setter must:

- Validate the whole update before mutating state; reject unsupported values
  without partial changes.
- Update both the scheduler's cached chunk budget and effective configuration
  readback. Changing only CLI metadata or an allowlist does not change scheduling.
- Reach the actual DP/TP/PP participants using the deployment's control path.
  Understand which ranks each acknowledgement and readback represents.
- Preserve unfinished prompt progress, KV state, and already-built batches.
- Enforce positive integral values, page alignment, truncation constraints and a
  fixed allocation ceiling. Reject incompatible modes instead of silently
  accepting an ineffective or unsafe update.

PR #895 limits updates to the effective startup per-attention-DP-rank budget and
rejects dynamic chunking, diffusion LLMs and KV staging. Its API takes the
**per-rank** budget directly. With attention-DP size A, a requested maximum
per-rank chunk C normally requires startup CLI chunk `C * A`; verify the installed
version's resolution rather than dividing twice. Starting at 8192 does not permit
runtime 16384/32768 with this patch. If the planned maximum fits the model,
memory and backend constraints, provision it in the one-time patched rollout,
then sweep downward and back within that ceiling. Otherwise record larger chunks
as unavailable or plan a separately authorized reconfiguration.

Run focused validation for accepted/rejected updates, readback, restoration,
alignment, allocation limits and unfinished requests. After the authorized
rollout, verify the loaded revision on every relevant worker, readiness and a
small real-request smoke check. This usually costs one initial P rollout instead
of one restart per chunk; restoration can require another rollout. Record actual
costs rather than promising a fixed speedup. Do not restart D solely to change P's
chunk; if a shared image/topology requires wider impact, resolve that scope first.

## Sweep and verify

Stop this run's arrivals and drain its requests before switching chunks, even if
the setter can preserve in-flight progress. Use the native P administrative
endpoint, existing authentication and bounded request timeouts. For the inspected
setter, the payload is:

```json
{"server_args": {"chunked_prefill_size": 2048}}
```

Check every expected response for success and read each reported effective value
from `/server_info`'s `internal_states`. A successful HTTP status alone is not
enough. Verify control fan-out and rank coverage from implementation/topology;
one frontend response must not be described as direct observation of every rank.
Warm the new scheduled shape, reset window baselines after the change, and run
all planned length/entropy cells at that chunk before switching again. Keep patch
revision and other serving settings fixed across the comparison.

Keep the two 30-second accepted windows, real text/weights/KV transfer, cache
conditions and ordinary stalls. A patch must not skip Prefill work, fabricate KV
or counters, force speculative acceptance, drop slow intervals, or shorten the
declared windows to make testing faster. A patch that changes actual inference
compute is a separate serving scheme, not an invisible test-speed shortcut.

## Archive and restore

Archive the exact patch/hash and base revision, original/loaded deployment
configuration, worker verification, per-chunk update replies/readbacks, smoke and
validation results, setup/restart/readiness/warmup times and revert procedure.
Reference these artifacts from the run manifest; the collectors do not generate
them or apply patches automatically. Compare actual wall time separately from
measurement duration. Retry only missing/failed points in a linked archive.

On success, failure or interruption, stop only this run's owned load and restore
its chunk and service edits as agreed. Verify both effective chunk and loaded
code/image; reverting files alone does not revert already-running workers.
Retain a patch only when the user requested it. Check for intervening changes
before restoration, and report any unresolved restoration explicitly.
