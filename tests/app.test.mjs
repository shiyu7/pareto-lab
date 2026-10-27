import assert from "node:assert/strict";
import {test} from "node:test";
import {readFileSync} from "node:fs";
import {parseDocument, groupKey, curveKey, parallelLabel, chunkLabel, mergePoints, displayPoints, paretoMask, meetsSlo, summarize, ticks} from "../docs/app.mjs";

const point = overrides => ({phase: "decode", workload: "128k", scheme: "EP64", source: "run:1", x_metric: "per_user_tps", x_value: 300,
  throughput_metric: "output_throughput", throughput_unit: "tokens/s", throughput: 614400, num_gpus: 64, dp_size: 64, concurrency: 2048, ...overrides});
const doc = points => ({schema: "hpc-pareto-points", version: 1, points});
const imported = points => ({points: parseDocument(doc(points))});

test("normalizes imports exactly once and rejects unlabelled data", () => {
  const points = parseDocument(doc([point({throughput: 9600, throughput_unit: "tokens/s/GPU"})]));
  assert.equal(points[0].throughput, 614400);
  assert.equal(displayPoints(points, "per-gpu")[0].y, 9600);
  assert.equal(parseDocument(doc(points))[0].throughput, 614400);
  assert.throws(() => parseDocument([point()]), /v1/);
});

test("does not mix phase, workload, latency or input token definitions", () => {
  const points = [point(), point({workload: "32k"}), point({phase: "prefill", x_metric: "ttft_ms", throughput_metric: "input_throughput"}),
    point({phase: "prefill", x_metric: "prefill_latency_ms", throughput_metric: "input_throughput"}),
    point({phase: "prefill", x_metric: "prefill_latency_ms", throughput_metric: "non_cached_input_throughput"})];
  assert.equal(new Set(parseDocument(doc(points)).map(groupKey)).size, 5);
});

test("merged files deduplicate measurements while preserving distinct layouts", () => {
  const a = imported([point()]), b = imported([point({source: "other:1"})]);
  assert.equal(mergePoints([a, b]).length, 1);
  assert.equal(mergePoints([a, imported([point({x_value: 240, concurrency: 4096})])]).length, 2);
  const differentLayout = mergePoints([a, imported([point({num_gpus: 8, tp_size: 8})])]);
  assert.equal(differentLayout.length, 2);
  assert.equal(summarize(displayPoints(differentLayout, "total"), null).length, 2);
});

test("decode and prefill use opposite dominance directions; ties retained", () => {
  const pairs = [[300, 100], [200, 200], [200, 100], [300, 100], [100, 200]];
  const points = pairs.map(([x_value, y]) => ({...point(), x_value, y}));
  assert.deepEqual(paretoMask(points), [true, true, false, true, false]);
  const prefill = [[100, 100], [200, 200], [200, 100], [300, 150]].map(([x_value, y]) => ({phase: "prefill", x_value, y}));
  assert.deepEqual(paretoMask(prefill), [true, true, false, false]);
  assert.deepEqual(paretoMask([points[2]]), [true]);
});

test("SLO boundary and best throughput versus maximum feasible concurrency", () => {
  const points = [point({x_value: 340, throughput: 174080, concurrency: 512}), point(),
    point({x_value: 301, throughput: 500000, concurrency: 2500}), point({x_value: 240, concurrency: 4096})];
  const result = summarize(displayPoints(points, "per-gpu"), 300)[0];
  assert.equal(result.feasible, 3);
  assert.equal(result.best.concurrency, 2048);
  assert.equal(result.maxConcurrency, 2500);
  assert.equal(summarize(displayPoints(points, "total"), 400)[0].best, null);
  assert.ok(meetsSlo({phase: "prefill", x_value: 150}, 150));
  assert.ok(!meetsSlo({phase: "prefill", x_value: 151}, 150));
});

test("rejects invalid numbers, mismatched metrics and missing GPU count", () => {
  for (const invalid of [point({x_value: 0}), point({throughput: NaN}), point({num_gpus: 1.5}),
    point({x_metric: "ttft_ms"}), point({throughput_unit: "tokens/s/GPU", num_gpus: null})]) {
    assert.throws(() => parseDocument(doc([invalid])));
  }
  const points = parseDocument(doc([point({num_gpus: null})]));
  assert.equal(displayPoints(points, "total")[0].y, 614400);
  assert.throws(() => displayPoints(points, "per-gpu"), /缺少 GPU/);
});

test("tick ranges cover data and include zero", () => {
  for (const maximum of [0.004, 1, 340, 23040, 1000000]) {
    const t = ticks(maximum);
    assert.equal(t.values[0], 0);
    assert.ok(t.top >= maximum);
    assert.ok(t.values.length < 15);
  }
});

if (process.env.PARETO_EXAMPLE) {
  test("Python export satisfies the browser contract and 300 tokens/s scenario", () => {
    const points = parseDocument(JSON.parse(readFileSync(process.env.PARETO_EXAMPLE, "utf8")));
    assert.equal(points.length, 17);
    const decode = displayPoints(points.filter(p => p.phase === "decode"), "per-gpu");
    assert.deepEqual(paretoMask(decode), [true, true, true, true, false, false, false, false]);
    assert.equal(summarize(decode, 300).find(s => s.scheme === "EP64DP64").maxConcurrency, 2048);
  });
}

const prefill = overrides => point({phase: "prefill", x_metric: "prefill_latency_ms", throughput_metric: "non_cached_input_throughput",
  tp_size: 1, dp_size: 8, ep_size: 8, pp_size: 1, chunked_prefill: true, chunk_size_tokens: 8192, chunk_scope: "dp-rank", ...overrides});

test("v1 records retain unknown metadata, v2 can generate names from parallel fields", () => {
  const old = parseDocument(doc([point()]))[0];
  assert.equal(old.tp_size, null);
  assert.equal(old.ep_size, null); // EP64 in the scheme name is never parsed.
  assert.equal(parallelLabel(old), "TP? DP64 EP? PP?");
  const data = {...doc([prefill({scheme: undefined})]), version: 2};
  const p = parseDocument(data)[0];
  assert.equal(p.scheme, "TP1 DP8 EP8 PP1");
  assert.equal(chunkLabel(p), "Chunk=8,192 tok/DP");
  assert.throws(() => parseDocument({...data, version: 3}));
});

test("chunk configuration splits curves without splitting workload groups or discarding coincident points", () => {
  const rows = [prefill(), prefill({chunk_size_tokens: 16384}), prefill({chunk_scope: "request"}),
    prefill({chunked_prefill: false, chunk_size_tokens: null, chunk_scope: null}),
    prefill({chunked_prefill: null, chunk_size_tokens: null, chunk_scope: null})];
  const points = parseDocument({...doc(rows), version: 2});
  assert.equal(new Set(points.map(groupKey)).size, 1);
  assert.equal(new Set(points.map(curveKey)).size, 5);
  assert.equal(mergePoints([{points}]).length, 5);
  assert.equal(summarize(displayPoints(points, "per-gpu"), null).length, 5);
  assert.equal(chunkLabel(points[3]), "Chunk=off");
  assert.equal(chunkLabel(points[4]), "Chunk=?");
  assert.notEqual(curveKey(points[3]), curveKey(points[4]));
  assert.deepEqual(parseDocument({...doc(points), version: 2}), points);
});

test("rejects inconsistent chunk policies and invalid parallel dimensions", () => {
  for (const row of [prefill({chunked_prefill: false}), prefill({chunk_scope: null}),
    prefill({chunk_size_tokens: null}), prefill({chunked_prefill: "true"}), prefill({chunk_scope: "instance"}),
    prefill({tp_size: 0}), prefill({ep_size: 1.5}), prefill({chunk_size_tokens: 0}),
    prefill({prefill_batch_size: 16384}),
    point({chunked_prefill: false})]) {
    assert.throws(() => parseDocument({...doc([row]), version: 2}));
  }
});
