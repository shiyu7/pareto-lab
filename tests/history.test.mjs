import assert from "node:assert/strict";
import {test} from "node:test";
import {parseDocument} from "../docs/app.mjs";
import {RecentFiles, HISTORY_KEY} from "../docs/history.mjs";

const point = {phase: "decode", workload: "test", scheme: "EP8", source: "synthetic:1",
  x_metric: "per_user_tps", x_value: 300, throughput_metric: "output_throughput",
  throughput_unit: "tokens/s/GPU", throughput: 1000, num_gpus: 8};
const file = (name, version = 2, changes = {}) => ({name, data: {schema: "hpc-pareto-points", version, points: [{...point, ...changes}]}});
function setup(options = {}) {
  const values = new Map();
  const storage = {getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)};
  const store = new RecentFiles({storage, validate: parseDocument, now: () => "2026-11-02T13:00:00.000Z", ...options});
  return {store, storage, values};
}

test("v1/v2 snapshots survive a new session and repeated opens never renormalize stored throughput", () => {
  const {store, storage} = setup();
  store.remember([file("旧版.json", 1), file("新版.json")]);
  const reopened = new RecentFiles({storage, validate: parseDocument});
  assert.deepEqual(reopened.list().map(entry => [entry.name, entry.pointCount, entry.error]),
    [["旧版.json", 1, null], ["新版.json", 1, null]]);
  const id = reopened.list()[0].id;
  for (let i = 0; i < 3; i++) {
    const entry = reopened.get(id);
    assert.equal(entry.data.points[0].throughput, 1000);
    assert.equal(entry.points[0].throughput, 8000);
    reopened.remember([entry]);
  }
  assert.equal(reopened.list().length, 2);
});

test("reopening moves a snapshot to the front without duplicates; same-name changed files stay distinct", () => {
  const {store} = setup();
  const a = file("points.json"), b = file("other.json");
  store.remember([a]);
  const originalId = store.list()[0].id;
  store.remember([b]);
  store.remember([a, a]);
  assert.deepEqual(store.list().map(entry => entry.name), [a.name, b.name]);
  assert.equal(store.list()[0].id, originalId);
  store.remember([file(a.name, 2, {throughput: 2000})]);
  assert.equal(store.list().length, 3);
  assert.notEqual(store.list()[0].id, originalId);
  assert.equal(store.get(originalId).data.points[0].throughput, 1000);
});

test("invalid batches never modify history", () => {
  const {store, storage} = setup();
  store.remember([file("saved.json")]);
  const before = storage.getItem(HISTORY_KEY);
  for (const invalid of [file("future.json", 3), file("broken.json", 2, {x_value: -1})]) {
    assert.throws(() => store.remember([file("valid.json"), invalid]));
    assert.equal(storage.getItem(HISTORY_KEY), before);
  }
});

test("cached documents are revalidated; incompatible and corrupt points stay visible but cannot open", () => {
  const {store, storage} = setup();
  store.remember([file("ok.json"), file("future.json"), file("invalid.json")]);
  const record = JSON.parse(storage.getItem(HISTORY_KEY));
  record.entries[1].data.version = 3;
  record.entries[2].data.points[0].throughput = 0;
  storage.setItem(HISTORY_KEY, JSON.stringify(record));
  const entries = store.list();
  assert.equal(entries[0].error, null);
  assert.match(entries[1].error, /v1\/v2/);
  assert.match(entries[2].error, /有限正数/);
  assert.throws(() => store.get(entries[1].id));
  assert.throws(() => store.get(entries[2].id));
  store.remove(entries[1].id);
  assert.equal(store.list().length, 2);
});

test("count and UTF-16 byte budgets evict the least recent snapshots", () => {
  const {store} = setup({maxEntries: 2});
  store.remember([file("a.json")]);
  store.remember([file("b.json")]);
  store.remember([file("c.json")]);
  assert.deepEqual(store.list().map(entry => entry.name), ["c.json", "b.json"]);
  const bytes = store.encode([store.read()[0]]).length * 2 + 10;
  store.maxBytes = bytes;
  store.remember([file("d.json")]);
  assert.deepEqual(store.list().map(entry => entry.name), ["d.json"]);
  assert.ok(store.encode(store.read()).length * 2 <= bytes);
});

test("oversized files do not erase usable history and omitted batch entries are reported", () => {
  const {store, storage} = setup();
  store.remember([file("saved.json")]);
  const before = storage.getItem(HISTORY_KEY);
  store.maxBytes = before.length * 2 + 10;
  assert.deepEqual(store.remember([file("large.json", 2, {source: "x".repeat(2000)})]), {saved: 0, omitted: 1});
  assert.equal(storage.getItem(HISTORY_KEY), before);
  store.maxBytes = 4 * 1024 * 1024;
  store.maxEntries = 1;
  assert.deepEqual(store.remember([file("a.json"), file("b.json")]), {saved: 1, omitted: 1});
});

test("quota failures preserve previously saved snapshots", () => {
  const {store, storage} = setup();
  store.remember([file("saved.json")]);
  const before = storage.getItem(HISTORY_KEY);
  storage.setItem = () => { throw new Error("QuotaExceededError"); };
  assert.throws(() => store.remember([file("new.json")]), /QuotaExceededError/);
  assert.equal(storage.getItem(HISTORY_KEY), before);
});

test("deletion is persistent and clearing touches only the recent-files key", () => {
  const {store, storage} = setup();
  storage.setItem("unrelated-setting", "keep");
  store.remember([file("a.json"), file("b.json")]);
  const id = store.list()[0].id;
  store.remove(id);
  assert.throws(() => store.get(id), /已被删除/);
  assert.equal(store.list().length, 1);
  store.clear();
  assert.deepEqual(store.list(), []);
  assert.equal(storage.getItem("unrelated-setting"), "keep");
});

test("unknown or damaged history formats are preserved until explicitly cleared", () => {
  const {store, storage} = setup();
  for (const raw of ["not json", JSON.stringify({version: 99, entries: []}),
    JSON.stringify({version: 1, entries: [{id: "bad", name: "x", openedAt: "invalid"}]})]) {
    storage.setItem(HISTORY_KEY, raw);
    assert.throws(() => store.list());
    assert.throws(() => store.remember([file("new.json")]));
    assert.equal(storage.getItem(HISTORY_KEY), raw);
    store.clear();
    assert.deepEqual(store.list(), []);
  }
});

test("storage access is lazy so browser restrictions do not prevent app initialization", () => {
  const store = new RecentFiles({validate: parseDocument, storage: () => { throw new Error("SecurityError"); }});
  assert.throws(() => store.list(), /SecurityError/);
  assert.throws(() => store.remember([file("new.json")]), /SecurityError/);
  assert.throws(() => store.clear(), /SecurityError/);
});
