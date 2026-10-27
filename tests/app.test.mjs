import assert from "node:assert/strict";
import {test} from "node:test";
import {readFileSync} from "node:fs";
import {parseDocument, groupKey, mergePoints, displayPoints, paretoMask, meetsSlo, summarize, ticks} from "../docs/app.mjs";

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

test("merged files deduplicate measurements and detect layout conflicts", () => {
  const a = imported([point()]), b = imported([point({source: "other:1"})]);
  assert.equal(mergePoints([a, b]).length, 1);
  assert.equal(mergePoints([a, imported([point({x_value: 240, concurrency: 4096})])]).length, 2);
  assert.throws(() => mergePoints([a, imported([point({num_gpus: 8})])]), /GPU\/DP/);
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
    assert.equal(points.length, 14);
    const decode = displayPoints(points.filter(p => p.phase === "decode"), "per-gpu");
    assert.deepEqual(paretoMask(decode), [true, true, true, true, false, false, false, false]);
    assert.equal(summarize(decode, 300).find(s => s.scheme === "EP64DP64").maxConcurrency, 2048);
  });
}
