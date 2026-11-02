export const HISTORY_KEY = "pareto-lab.recent-files";
const HISTORY_VERSION = 1;

// Store original documents so reopening always uses the current point parser.
// In particular, per-GPU throughput must never be normalized twice.
export class RecentFiles {
  constructor({validate, storage = () => globalThis.localStorage, maxEntries = 20,
    maxBytes = 4 * 1024 * 1024, now = () => new Date().toISOString()} = {}) {
    this.validate = validate;
    this.storage = typeof storage === "function" ? storage : () => storage;
    this.maxEntries = maxEntries;
    this.maxBytes = maxBytes;
    this.now = now;
  }

  read() {
    const text = this.storage().getItem(HISTORY_KEY);
    if (text === null) return [];
    const record = JSON.parse(text);
    if (record?.version !== HISTORY_VERSION || !Array.isArray(record.entries)) {
      throw new Error("历史记录格式不兼容，可清空历史后重新导入文件。");
    }
    const ids = new Set();
    for (const entry of record.entries) {
      if (!entry || typeof entry.id !== "string" || ids.has(entry.id)
          || typeof entry.name !== "string" || !entry.name.trim()
          || typeof entry.openedAt !== "string" || !Number.isFinite(Date.parse(entry.openedAt))) {
        throw new Error("历史记录已损坏，可清空历史后重新导入文件。");
      }
      ids.add(entry.id);
    }
    return record.entries;
  }

  list() {
    return this.read().map(entry => {
      try {
        return {...entry, pointCount: this.validate(entry.data, entry.name).length, error: null};
      } catch (error) {
        return {...entry, pointCount: null, error: error.message};
      }
    });
  }

  get(id) {
    const entry = this.read().find(entry => entry.id === id);
    if (!entry) throw new Error("该历史记录已被删除，请重新导入文件。");
    return {...entry, points: this.validate(entry.data, entry.name)};
  }

  encode(entries) {
    return JSON.stringify({version: HISTORY_VERSION, entries});
  }

  remember(entries) {
    // Validate the whole batch before touching storage, just like a normal import.
    for (const entry of entries) this.validate(entry.data, entry.name);
    const current = this.read(), openedAt = this.now();
    const signature = entry => JSON.stringify([entry.name, entry.data]);
    const previous = new Map(current.map(entry => [signature(entry), entry]));
    const seen = new Set(), incoming = [];
    for (const entry of entries) {
      const key = signature(entry);
      if (seen.has(key)) continue;
      seen.add(key);
      const saved = {id: previous.get(key)?.id ?? `${openedAt}-${Math.random().toString(36).slice(2)}`,
        name: entry.name, data: entry.data, openedAt};
      // An oversized file must not evict the entire usable history.
      if (this.encode([saved]).length * 2 <= this.maxBytes) incoming.push(saved);
    }
    const next = [...incoming, ...current.filter(entry => !seen.has(signature(entry)))].slice(0, this.maxEntries);
    while (next.length && this.encode(next).length * 2 > this.maxBytes) next.pop();
    // setItem is atomic: quota/security failures preserve the previous history.
    if (incoming.length) this.storage().setItem(HISTORY_KEY, this.encode(next));
    const saved = incoming.filter(entry => next.includes(entry)).length;
    return {saved, omitted: seen.size - saved};
  }

  remove(id) {
    const next = this.read().filter(entry => entry.id !== id);
    if (next.length) this.storage().setItem(HISTORY_KEY, this.encode(next));
    else this.clear();
  }

  clear() {
    this.storage().removeItem(HISTORY_KEY);
  }
}
