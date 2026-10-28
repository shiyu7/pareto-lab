import {planDeployment, calibrationKey} from "./planner.mjs";
const SCHEMA = "hpc-pareto-points";
const VERSION = 2;
const PARALLEL = [["TP", "tp_size"], ["DP", "dp_size"], ["EP", "ep_size"], ["PP", "pp_size"]];
const COLORS = ["#197b70", "#df8242", "#497cc0", "#9671b8", "#be5775", "#8c8139", "#477f92", "#69755a"];

export function parallelLabel(p) {
  return PARALLEL.map(([label, key]) => `${label}${p[key] ?? "?"}`).join(" ");
}

export function chunkLabel(p) {
  if (p.phase !== "prefill") return "—";
  if (p.chunked_prefill === false) return "Chunk=off";
  if (p.chunked_prefill !== true) return "Chunk=?";
  return `Chunk=${p.chunk_size_tokens.toLocaleString("en-US")} tok/${p.chunk_scope === "request" ? "request" : "DP"}`;
}

export const configurationLabel = p => `${parallelLabel(p)} · GPU=${p.num_gpus ?? "?"}${p.phase === "prefill" ? ` · ${chunkLabel(p)}` : ""}`;
export const curveKey = p => JSON.stringify([p.phase, p.scheme, ...PARALLEL.map(([, key]) => p[key] ?? null), p.num_gpus ?? null,
  p.chunked_prefill ?? null, p.chunk_size_tokens ?? null, p.chunk_scope ?? null]);
export const curveLabel = p => `${p.scheme} · ${configurationLabel(p)}`;

function normalizeConfiguration(p, fail) {
  for (const key of ["tp_size", "dp_size", "ep_size", "pp_size", "chunk_size_tokens"]) p[key] ??= null;
  p.chunked_prefill ??= null;
  p.chunk_scope ??= null;
  if (p.chunked_prefill !== null && typeof p.chunked_prefill !== "boolean") fail("chunked_prefill 必须为 true / false / null");
  if (p.phase === "decode" && [p.chunked_prefill, p.chunk_size_tokens, p.chunk_scope].some(v => v !== null)) fail("chunk 配置仅适用于 Prefill");
  if (p.chunked_prefill === false && (p.chunk_size_tokens !== null || p.chunk_scope !== null)) fail("禁用 chunking 时不能提供大小或作用范围");
  if (p.chunk_size_tokens !== null && p.chunked_prefill === null) p.chunked_prefill = true;
  if (p.chunked_prefill === true && (p.chunk_size_tokens === null || !["request", "dp-rank"].includes(p.chunk_scope))) fail("启用 chunking 需要 chunk_size_tokens 和 chunk_scope=request 或 dp-rank");
  if (p.chunked_prefill === null && p.chunk_scope !== null) fail("chunk_scope 需要启用 chunking 并提供大小");
  if (p.chunked_prefill === true && p.chunk_scope === "dp-rank" && p.prefill_batch_size > p.chunk_size_tokens) fail("实际 Prefill batch tokens 超出每 DP rank 的 chunk 预算");
  if (p.scheme == null || p.scheme === "") {
    if (!PARALLEL.some(([, key]) => p[key] !== null)) fail("缺少 scheme 或结构化并行配置");
    p.scheme = parallelLabel(p);
  }
  if (typeof p.scheme !== "string" || !p.scheme.trim()) fail("scheme 必须为非空字符串");
}

export function parseDocument(data, name = "points.json") {
  if (data?.schema !== SCHEMA || ![1, VERSION].includes(data.version) || !Array.isArray(data.points) || !data.points.length) {
    throw new Error(`${name}：需要包含有效点位的 hpc-pareto-points v1/v2 JSON，请先运行采集或转换脚本。`);
  }
  return data.points.map((raw, index) => {
    const p = {...raw};
    const fail = message => { throw new Error(`${name} 第 ${index + 1} 点：${message}`); };
    const positive = value => typeof value === "number" && Number.isFinite(value) && value > 0;
    for (const field of ["workload", "source"]) {
      if (typeof p[field] !== "string" || !p[field].trim()) fail(`缺少 ${field}`);
    }
    if (!["decode", "prefill"].includes(p.phase)) fail("phase 无效");
    if (!positive(p.x_value) || !positive(p.throughput)) fail("坐标必须是有限正数");
    if (p.phase === "decode" && (p.x_metric !== "per_user_tps" || p.throughput_metric !== "output_throughput")) fail("Decode 指标不匹配");
    if (p.phase === "prefill" && (!["prefill_latency_ms", "ttft_ms"].includes(p.x_metric)
        || !["non_cached_input_throughput", "input_throughput"].includes(p.throughput_metric))) fail("Prefill 指标不匹配");
    for (const field of ["num_gpus", "tp_size", "dp_size", "ep_size", "pp_size", "concurrency", "prefill_batch_size", "chunk_size_tokens"]) {
      if (p[field] != null && (!positive(p[field]) || !Number.isSafeInteger(p[field]))) fail(`${field} 必须是正整数`);
    }
    if (p.prefill_service_ms != null && (p.phase !== "prefill" || !positive(p.prefill_service_ms))) fail("prefill_service_ms 仅限 Prefill，须为有限正数");
    normalizeConfiguration(p, fail);
    if (!["tokens/s", "tokens/s/GPU"].includes(p.throughput_unit)) fail("吞吐单位无效");
    if (p.throughput_unit === "tokens/s/GPU") {
      if (!positive(p.num_gpus)) fail("每 GPU 吞吐必须包含 num_gpus");
      p.throughput *= p.num_gpus;
      if (!positive(p.throughput)) fail("还原实例吞吐溢出");
    }
    p.throughput_unit = "tokens/s";
    return p;
  });
}

export const groupKey = p => JSON.stringify([p.phase, p.workload, p.x_metric, p.throughput_metric]);

export function mergePoints(documents) {
  const points = [], seen = new Set();
  for (const document of documents) {
    for (const p of document.points) {
      const curve = JSON.stringify([groupKey(p), curveKey(p)]);
      // Distinct configurations remain distinct even when names/coordinates match.
      const identity = JSON.stringify([curve, p.x_value, p.throughput, p.concurrency ?? null, p.prefill_batch_size ?? null, p.prefill_service_ms ?? null]);
      if (!seen.has(identity)) { points.push(p); seen.add(identity); }
    }
  }
  return points;
}

export function displayPoints(points, unit) {
  return points.map(p => {
    if (unit === "per-gpu" && !p.num_gpus) throw new Error(`方案「${p.scheme}」缺少 GPU 数，请选择单实例吞吐，或重新导出时提供 --num-gpus。`);
    return {...p, y: p.throughput / (unit === "per-gpu" ? p.num_gpus : 1)};
  });
}

export function paretoMask(points) {
  return points.map(p => !points.some(other => {
    const direction = p.phase === "decode" ? 1 : -1;
    return direction * other.x_value >= direction * p.x_value && other.y >= p.y
      && (direction * other.x_value > direction * p.x_value || other.y > p.y);
  }));
}

export const meetsSlo = (p, slo) => slo == null || (p.phase === "decode" ? p.x_value >= slo : p.x_value <= slo);
export const pointLabel = p => p.phase === "prefill" && p.prefill_batch_size != null
  ? `B/DP=${p.prefill_batch_size.toLocaleString("en-US")}`
  : p.concurrency != null ? `C=${p.concurrency.toLocaleString("en-US")}` : "";

export function summarize(points, slo) {
  return [...new Set(points.map(curveKey))].map(key => {
    const series = points.filter(p => curveKey(p) === key), feasible = series.filter(p => meetsSlo(p, slo));
    const best = feasible.reduce((best, p) => !best || p.y > best.y ? p : best, null);
    const loads = feasible.map(p => p.concurrency).filter(n => n != null);
    return {scheme: series[0].scheme, key, config: series[0], total: series.length, feasible: feasible.length, best, maxConcurrency: loads.length ? Math.max(...loads) : null};
  });
}

export function ticks(maximum, count = 6) {
  const rough = maximum / count, power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map(n => n * power).find(n => n >= rough);
  const top = Math.ceil(maximum / step) * step;
  return {top, values: Array.from({length: Math.round(top / step) + 1}, (_, i) => i * step)};
}

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  let documents = [], hidden = new Set(), nextId = 1;
  let activeSvg = null;
  const format = n => n == null ? "—" : n.toLocaleString("en-US", {maximumFractionDigits: 2});
  const colors = new Map();
  const color = scheme => {
    if (!colors.has(scheme)) {
      const index = colors.size;
      colors.set(scheme, COLORS[index] ?? `hsl(${(index * 137.508) % 360} 48% 42%)`);
    }
    return colors.get(scheme);
  };
  const element = (tag, text, parent) => {
    const el = document.createElement(tag);
    if (text != null) el.textContent = text;
    if (parent) parent.append(el);
    return el;
  };
  const svgElement = (tag, attributes, text, parent) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, String(value));
    if (text != null) el.textContent = text;
    parent.append(el);
    return el;
  };
  function notice(message, error = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", error);
  }
  function download(text, filename, type) {
    const url = URL.createObjectURL(new Blob([text], {type}));
    const link = element("a");
    link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function refreshOptions() {
    const points = mergePoints(documents), old = $("group").value;
    $("group").replaceChildren();
    for (const key of new Set(points.map(groupKey))) {
      const p = points.find(p => groupKey(p) === key);
      const metric = p.x_metric === "ttft_ms" ? " · TTFT" : p.phase === "prefill" ? " · 单步延迟" : "";
      const input = p.phase === "prefill" ? (p.throughput_metric === "input_throughput" ? " · 全部输入" : " · 非缓存输入") : "";
      const option = element("option", `${p.phase.toUpperCase()} · ${p.workload}${metric}${input}`, $("group"));
      option.value = key;
    }
    if ([...$("group").options].some(o => o.value === old)) $("group").value = old;
    if ($("group").value !== old) $("slo").value = "";
    $("documents").replaceChildren();
    for (const entry of documents) {
      const li = element("li", null, $("documents"));
      element("span", `${entry.name} · ${entry.points.length} 点`, li);
      const remove = element("button", "移除", li); remove.className = "text-button";
      remove.setAttribute("aria-label", `移除 ${entry.name}`);
      remove.onclick = () => { documents = documents.filter(d => d.id !== entry.id); refreshOptions(); };
    }
    $("save-points").disabled = !points.length;
    refreshPlanner();
    render();
  }
  function importDocuments(entries) {
    const added = entries.map(entry => ({id: nextId++, name: entry.name, points: parseDocument(entry.data, entry.name)}));
    // Validate the entire import before changing the current comparison.
    mergePoints([...documents, ...added]);
    documents = [...documents, ...added];
    refreshOptions();
  }
  async function importFiles(files) {
    try {
      const entries = await Promise.all([...files].map(async file => ({name: file.name, data: JSON.parse((await file.text()).replace(/^\uFEFF/, ""))})));
      if (entries.length) importDocuments(entries);
    } catch (error) { notice(`导入失败：${error.message}`, true); }
  }
  function draw(points, frontier, slo) {
    const phase = points[0].phase, metric = points[0].x_metric, workload = points[0].workload;
    const curves = [...new Map(points.map(p => [curveKey(p), p])).values()];
    const legendRows = Math.ceil(curves.length / 2), width = 1100, height = 680 + legendRows * 54;
    const left = 110, top = 110 + legendRows * 54, right = 45, bottom = 85;
    const xt = ticks(Math.max(...points.map(p => p.x_value), slo || 0) * 1.12);
    const yt = ticks(Math.max(...points.map(p => p.y)) * 1.15);
    const x = n => left + n / xt.top * (width - left - right);
    const y = n => height - bottom - n / yt.top * (height - top - bottom);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", width); svg.setAttribute("height", height);
    svg.setAttribute("font-family", "Arial, sans-serif");
    svg.setAttribute("font-size", "12"); svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${phase} performance curve: ${workload}`);
    const add = (tag, attrs, text) => svgElement(tag, attrs, text, svg);
    add("title", {}, `${phase.toUpperCase()} · ${workload}`);
    add("desc", {}, "Measured points only. Outlined points are nondominated among the visible schemes. Lines do not imply measured intermediate values.");
    add("rect", {width, height, fill: "white"});
    add("text", {x: left, y: 33, "font-size": 20, "font-weight": 700, fill: "#1b2b38"}, `${phase.toUpperCase()} · ${workload}`);
    curves.forEach((p, index) => {
      const lx = left + index % 2 * 465, ly = 60 + Math.floor(index / 2) * 54;
      add("line", {x1: lx, x2: lx + 20, y1: ly, y2: ly, stroke: color(curveKey(p)), "stroke-width": 3});
      const label = add("text", {x: lx + 28, y: ly + 4, fill: "#354c5e"}, p.scheme.length > 48 ? p.scheme.slice(0, 46) + "…" : p.scheme);
      svgElement("title", {}, curveLabel(p), label);
      add("text", {x: lx + 28, y: ly + 18, fill: "#607585", "font-size": 10}, `${parallelLabel(p)} · GPU=${p.num_gpus ?? "?"}`);
      if (p.phase === "prefill") add("text", {x: lx + 28, y: ly + 32, fill: "#607585", "font-size": 10}, chunkLabel(p));
    });
    if (slo != null) {
      const start = phase === "decode" ? x(slo) : left;
      const end = phase === "decode" ? width - right : x(slo);
      add("rect", {x: start, y: top, width: end - start, height: height - top - bottom, fill: "#ecf6f1"});
    }
    for (const value of xt.values) {
      add("line", {x1: x(value), x2: x(value), y1: top, y2: height - bottom, stroke: "#e4eaee"});
      add("text", {x: x(value), y: height - bottom + 25, "text-anchor": "middle", fill: "#607585"}, format(value));
    }
    for (const value of yt.values) {
      add("line", {x1: left, x2: width - right, y1: y(value), y2: y(value), stroke: "#e4eaee"});
      add("text", {x: left - 13, y: y(value) + 4, "text-anchor": "end", fill: "#607585"}, format(value));
    }
    add("path", {d: `M${left},${top}V${height - bottom}H${width - right}`, fill: "none", stroke: "#8b9ba7"});
    const unit = $("unit").value === "per-gpu" ? "tokens/s/GPU" : "tokens/s/instance";
    const ylabel = phase === "decode" ? "Output throughput" : points[0].throughput_metric === "input_throughput" ? "Input throughput" : "Non-cached input throughput";
    const xlabel = phase === "decode" ? "Per-user generation speed (tokens/s/user)" : metric === "ttft_ms" ? "Time to first token (ms, includes queueing)" : "Prefill step latency (ms)";
    add("text", {x: (left + width - right) / 2, y: height - 24, "text-anchor": "middle", fill: "#354c5e", "font-size": 14}, xlabel);
    add("text", {transform: `translate(26 ${(top + height - bottom) / 2}) rotate(-90)`, "text-anchor": "middle", fill: "#354c5e", "font-size": 14}, `${ylabel} (${unit})`);
    if (slo != null) {
      add("line", {x1: x(slo), x2: x(slo), y1: top, y2: height - bottom, stroke: "#648578", "stroke-dasharray": "6 5"});
      add("text", {x: x(slo), y: top - 12, "text-anchor": "middle", fill: "#3f7260"}, `SLO ${phase === "decode" ? "≥" : "≤"} ${format(slo)}`);
    }
    for (const config of curves) {
      const key = curveKey(config);
      const series = points.filter(p => curveKey(p) === key).sort((a, b) => a.x_value - b.x_value);
      add("polyline", {points: series.map(p => `${x(p.x_value)},${y(p.y)}`).join(" "), fill: "none", stroke: color(key), "stroke-width": 2.5});
    }
    points.forEach((p, index) => {
      const circle = add("circle", {cx: x(p.x_value), cy: y(p.y), r: 5.5, fill: color(curveKey(p)), tabindex: 0});
      const detail = `${curveLabel(p)} · x=${format(p.x_value)} · ${format(p.y)} ${unit} · ${pointLabel(p) || "负载未提供"} · ${meetsSlo(p, slo) ? "SLO 可行" : "SLO 未达标"} · ${p.source}`;
      svgElement("title", {}, detail, circle);
      circle.setAttribute("aria-label", detail);
      circle.onmouseenter = circle.onfocus = () => { $("detail").textContent = detail; };
      if ($("frontier").checked && frontier[index]) add("circle", {cx: x(p.x_value), cy: y(p.y), r: 9, fill: "none", stroke: "#263f4e", "stroke-width": 1.3, "pointer-events": "none"});
      if ($("labels").checked && pointLabel(p)) add("text", {x: x(p.x_value), y: y(p.y) - 15, "text-anchor": "middle", fill: color(curveKey(p)), "font-size": 10}, pointLabel(p));
    });
    return svg;
  }
  function render() {
    const points = mergePoints(documents).filter(p => groupKey(p) === $("group").value);
    $("schemes").replaceChildren();
    for (const [key, config] of new Map(points.map(p => [curveKey(p), p]))) {
      const label = element("label", null, $("schemes")); label.className = "scheme";
      const checkbox = element("input", null, label); checkbox.type = "checkbox";
      checkbox.checked = !hidden.has(key);
      checkbox.setAttribute("aria-label", curveLabel(config));
      checkbox.onchange = () => { checkbox.checked ? hidden.delete(key) : hidden.add(key); render(); };
      const swatch = element("span", null, label); swatch.className = "swatch"; swatch.style.background = color(key);
      const copy = element("span", null, label); copy.className = "scheme-copy";
      element("span", config.scheme, copy);
      element("small", configurationLabel(config), copy);
    }
    const phase = points[0]?.phase;
    $("slo-label").textContent = phase === "decode" ? "最低单用户速度 · tokens/s" : "最大延迟 · ms";
    $("chart-title").textContent = phase ? `${phase.toUpperCase()} / 性能曲线` : "等待导入采样点";
    $("summary").replaceChildren();
    $("save-svg").disabled = true; activeSvg = null;
    $("detail").textContent = "悬停或聚焦测量点，查看并发、吞吐和来源。";
    const visible = points.filter(p => !hidden.has(curveKey(p)));
    if (!visible.length) {
      $("chart").replaceChildren(element("div", points.length ? "请至少勾选一个方案。" : "导入点位文件或载入合成示例，即可开始比较。"));
      $("chart").firstChild.className = "empty";
      notice(points.length ? "没有选中的方案。" : "等待导入采样点。", false);
      return;
    }
    try {
      const slo = $("slo").value === "" ? null : Number($("slo").value);
      if ($("slo").validity.badInput || (slo != null && (!Number.isFinite(slo) || slo <= 0))) throw new Error("SLO 必须是有限正数，或留空。 ");
      const display = displayPoints(visible, $("unit").value), frontier = paretoMask(display);
      activeSvg = draw(display, frontier, slo);
      $("chart").replaceChildren(activeSvg);
      $("save-svg").disabled = false;
      for (const row of summarize(display, slo)) {
        const tr = element("tr", null, $("summary"));
        for (const text of [row.scheme, parallelLabel(row.config), format(row.config.num_gpus), chunkLabel(row.config), `${row.feasible} / ${row.total}`, format(row.best?.y), row.best ? pointLabel(row.best) || "—" : "—", format(row.maxConcurrency)]) element("td", text, tr);
      }
      notice(`${display.length} 个实测点 · ${new Set(display.map(curveKey)).size} 条配置曲线 · ${frontier.filter(Boolean).length} 个非支配点 · ${slo == null ? "未设置 SLO" : "浅绿色为 SLO 可行区域"}`);
    } catch (error) {
      $("chart").replaceChildren(element("div", "请调整比较条件后继续。")); $("chart").firstChild.className = "empty";
      notice(error.message, true);
    }
  }
  let activePlan = null;
  const serviceOverrides = {};
  function invalidatePlan() {
    activePlan = null; $("save-plan").disabled = true;
    $("plan-result").replaceChildren(element("p", "参数或数据已更新，点击推荐按钮重新计算。"));
  }
  function refreshCalibration() {
    $("plan-calibration").replaceChildren();
    const prefills = mergePoints(documents).filter(p => p.phase === "prefill" && p.workload === $("plan-workload").value && p.throughput_metric === $("plan-metric").value);
    for (const [key, p] of new Map(prefills.filter(p => p.prefill_service_ms == null).map(p => [calibrationKey(p), p]))) {
      const label = element("label", null, $("plan-calibration")); label.className = "calibration-row";
      element("span", `${curveLabel(p)} · 缺少完整 Prefill 服务时间`, label);
      const input = element("input", null, label); input.type = "number"; input.min = "0.001"; input.step = "any";
      input.placeholder = "完整请求服务时间 · ms"; input.value = serviceOverrides[key] ?? "";
      input.oninput = () => { if (input.value === "") delete serviceOverrides[key]; else serviceOverrides[key] = Number(input.value); invalidatePlan(); };
    }
    if (!$("plan-calibration").children.length) element("p", prefills.length ? "所选 Prefill 点均已记录完整请求服务时间。" : "当前没有对应的 Prefill 点位。", $("plan-calibration"));
  }
  function refreshPlanner() {
    const points = mergePoints(documents), old = $("plan-workload").value;
    $("plan-workload").replaceChildren();
    for (const workload of new Set(points.map(p => p.workload))) {
      const phases = new Set(points.filter(p => p.workload === workload).map(p => p.phase));
      const option = element("option", `${workload}${phases.size < 2 ? "（缺少 P 或 D）" : ""}`, $("plan-workload")); option.value = workload;
    }
    if ([...$("plan-workload").options].some(o => o.value === old)) $("plan-workload").value = old;
    refreshCalibration(); invalidatePlan();
  }
  function planTable(parent, headers, rows) {
    const wrap = element("div", null, parent); wrap.className = "table-scroll";
    const table = element("table", null, wrap), tr = element("tr", null, element("thead", null, table));
    headers.forEach(h => element("th", h, tr));
    const body = element("tbody", null, table);
    rows.forEach(row => { const tr = element("tr", null, body); row.forEach(cell => element("td", cell, tr)); });
  }
  function renderPlan(result) {
    const root = $("plan-result"); root.replaceChildren();
    const {p,d,base,deployment:a,options:o} = result;
    element("h3", "01 / 延迟筛选 → 独立选择两阶段实测点", root);
    element("p", `Prefill 完整请求预算 ${format(result.prefillBudgetMs)} ms（TTFT ${format(o.ttftMs)} − 预留 ${format(o.latencyReserveMs)}）；Decode TPOT ≤ ${format(o.tpotMs)} ms。每个阶段选择可行点中 tokens/s/GPU 最高者，吞吐相同时优先更低延迟。`, root);
    for (const [name,item] of [["Prefill",p],["Decode",d]]) {
      const line = element("p", null, root); line.className = "plan-copy";
      element("strong", `${name}：`, line);
      if (!item) { element("span", "没有可选实测点，请检查延迟门槛及数据完整性。", line); continue; }
      element("span", curveLabel(item.point), line);
      element("br", null, line);
      element("span", `${format(item.tpsPerGpu)} tokens/s/GPU · ${format(item.qpsPerGpu)} requests/s/GPU · ${name === "Prefill" ? `完整请求 ${format(item.serviceMs)} ms（${item.serviceSource}），原横轴${item.point.x_metric === "ttft_ms" ? " TTFT" : "单步延迟"} ${format(item.point.x_value)} ms` : `TPOT ${format(item.tpotMs)} ms · 采样并发/GPU ${format(item.point.concurrency / item.point.num_gpus)}`}`, line);
      element("br", null, line); element("small", `采样点：${pointLabel(item.point)} · 实例总吞吐 ${format(item.point.throughput)} tokens/s · ${item.point.source}`, line);
    }
    if (p && d) {
      element("h3", "02 / 请求处理能力 → GPU 比例 → 整数实例", root);
      planTable(root, ["阶段", "每请求计数 tokens", "吞吐换算 requests/s/实例", "配比可用 requests/s/实例", "每实例 GPU"], [
        ["Prefill",format(p.tokensPerRequest),format(p.qpsPerInstance),format(p.qpsPerInstance*(1-o.prefillReserve)),String(p.point.num_gpus)],
        ["Decode",format(d.tokensPerRequest),format(d.qpsPerInstance),format(d.qpsPerInstance),String(d.point.num_gpus)]
      ]);
      element("p", `无余量 GPU 比例 P:D = ${format(result.ratio.gpuRaw)}:1；加入 ${format(o.prefillReserve*100)}% P 容量余量后为 ${format(result.ratio.gpuWithReserve)}:1。对应实例连续比例 P:D ≈ ${format(result.ratio.instancesWithReserve)}:1；实例数依据未取整容量向上取整。`, root);
      if (base) element("p", `基础部署：${base.nP}P + ${base.nD}D（${base.gP} 张 P GPU + ${base.gD} 张 D GPU），P 可用容量比 D 多 ${format(base.pExcess*100)}%。${base.withinTolerance ? "在配平容差内选择 GPU 数最少的一组。" : "未达到配平容差，需调整搜索条件。"} 等比例扩容 k 倍时，实例数和理论吞吐同步乘 k。`, root);
    }
    if (a) {
      element("h3", `03 / ${a.mode === "target" ? "目标 QPS 部署" : "基础部署"} → 理论吞吐与负载`, root);
      const stats = element("div", null, root); stats.className = "plan-stats";
      for (const [value,label] of [[`${a.nP}P + ${a.nD}D`,`${a.gP} 张 P GPU + ${a.gD} 张 D GPU`], [format(a.qps),"配比可承载 QPS · 含 P 余量"], [format(a.outputTps),"全输出理论 tokens/s"], [format(d.tpsPerGpu),"D 选定点 tokens/s/GPU"]]) {
        const card=element("div",null,stats);card.className="plan-stat";element("strong",value,card);element("span",label,card);
      }
      if (a.mode === "target") element("p", `目标 ${format(o.targetQps)} QPS：P = ceil(目标 / P 可用容量)，D = ceil(目标 / D 容量)，两侧独立向上取整。目标负载对应总输出 ${format(o.targetQps*o.outputTokens)} tokens/s；上方显示这组实例的容量上限，不代表需要按满载发压。`, root);
      element("p", `QPS = min(P 可用 ${format(a.capacityP)}, D ${format(a.capacityD)}) = ${format(a.qps)}。瓶颈：${a.bottleneck}；忽略 P 容量余量时，原始理论上限 ${format(a.rawQps)} QPS / ${format(a.rawOutputTps)} 全输出 tokens/s。P、D 按各自测得容量计算，未使用 P+D 总卡数重新排名。`, root);
      element("p", `Decode 吞吐范围：${o.outputTokenScope === "decode-only" ? "排除首 token，除以总输出长度 − 1" : "包含全部输出 token，除以总输出长度"}。按导入 D 吞吐口径换算的理论吞吐 ${format(a.decodeTps)} tokens/s，归一化到 D 卡为 ${format(a.decodeTpsPerGpu)} tokens/s/GPU。减少 P 若导致供给不足，只会降低可持续 QPS，不会提高 D 的选点能力。`, root);
      element("p", `每个 D 采样并发 C=${format(a.measuredDConcurrencyPerInstance)}，共 ${a.nD}D 对应 ${format(a.measuredDConcurrencyTotal)}；归一化为 ${format(a.normalizedDConcurrency)} 请求/GPU，用于比较，不能直接当作 TP rank 的请求上限。理论容量下 D 平均活动请求参考 ${format(a.estimatedDActive)}（QPS × 后续生成时间），须核对采样并发是活动请求还是配置上限。`, root);
      const note=element("p", "理论配比假设稳定供给、无排队、实例可线性扩展。延迟预留是手动预算，并非测出的等待时间；未预测线上 TTFT，也不保证均值或 P95/P99 达标。样例是合成数据。",root);note.className="plan-warning";
      element("h3","04 / 联合压测：验证理论工作点",root);
      planTable(root,["开放环路 QPS","D 活动请求参考","无排队在途参考*","位置"],result.scans.map(s=>[format(s.qps),format(s.estimatedDActive),format(s.noQueueInFlight),!s.withinCapacity?"超过配比容量，探索边界":s.atCapacity?"理论容量边界，SLO 待验证":"逐步升压"]));
      element("p", "* 在途参考 = QPS ×（完整 Prefill 时间 + 手动延迟预留 + 后续生成时间），仅用于负载核对，不是客户端并发上限。按固定长度/缓存条件逐步提高到达 QPS，记录实际到达与完成 QPS、输出吞吐、队列变化和 TTFT/TPOT 分位数；接近边界时加密测点。闭环固定并发测试单独扫描，QPS 为测量结果。",root);
    }
    for(const warning of result.warnings) {const line=element("p",warning,root);line.className="plan-warning";}
    element("p", `Prefill 可行点 ${result.prefillCandidates} 个；Decode 可行点 ${result.decodeCandidates} 个。`, root);
    if(Object.keys(result.skipped).length) element("p", `跳过点位：${Object.entries(result.skipped).map(([reason,n])=>`${reason} × ${n}`).join("；")}。缺失完整请求时间可展开上方配置补充。`,root);
  }
  $("planner-form").oninput = invalidatePlan;
  $("plan-workload").onchange = $("plan-metric").onchange = () => { refreshCalibration(); invalidatePlan(); };
  $("planner-form").onsubmit = event => {
    event.preventDefault();
    try {
      const num = id => Number($(id).value);
      const options = {workload: $("plan-workload").value, inputMetric: $("plan-metric").value,
        ttftMs:num("plan-ttft"),tpotMs:num("plan-tpot"),inputTokens:num("plan-input"),nonCachedTokens:num("plan-noncached"),outputTokens:num("plan-output"),
        outputTokenScope:$("plan-output-scope").value,latencyReserveMs:num("plan-latency-reserve"),prefillReserve:num("plan-p-reserve")/100,
        ratioTolerance:num("plan-ratio-tolerance")/100,maxInstances:num("plan-max-instances"),targetQps:$("plan-target").value===""?null:num("plan-target"),
        useCompleteStep:$("plan-complete-step").checked,serviceOverrides:{...serviceOverrides}};
      activePlan = planDeployment(mergePoints(documents), options);
      renderPlan(activePlan); $("save-plan").disabled = !activePlan.deployment;
    } catch (error) { invalidatePlan(); $("plan-result").textContent = error.message; }
  };
  $("save-plan").onclick = () => {
    if (activePlan) download(JSON.stringify(activePlan, null, 2), "pareto_deployment_plan.json", "application/json;charset=utf-8");
  };
  $("files").onchange = async event => { await importFiles(event.target.files); event.target.value = ""; };
  $("dropzone").ondragover = event => { event.preventDefault(); $("dropzone").classList.add("dragging"); };
  $("dropzone").ondragleave = () => $("dropzone").classList.remove("dragging");
  $("dropzone").ondrop = event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); importFiles(event.dataTransfer.files); };
  $("demo").onclick = async () => {
    try {
      const response = await fetch(new URL("./example.json", import.meta.url));
      if (!response.ok) throw new Error("无法加载示例");
      importDocuments([{name: "合成示例.json", data: await response.json()}]);
    } catch (error) { notice(error.message, true); }
  };
  $("clear").onclick = () => { documents = []; hidden.clear(); for (const key of Object.keys(serviceOverrides)) delete serviceOverrides[key]; refreshOptions(); };
  $("all").onclick = () => { hidden.clear(); render(); };
  $("group").onchange = () => { $("slo").value = ""; render(); };
  for (const id of ["unit", "frontier", "labels"]) $(id).onchange = render;
  $("slo").oninput = render;
  $("save-svg").onclick = () => {
    if (activeSvg) download(new XMLSerializer().serializeToString(activeSvg), `${JSON.parse($("group").value)[0]}_pareto.svg`, "image/svg+xml;charset=utf-8");
  };
  $("save-points").onclick = () => download(JSON.stringify({schema: SCHEMA, version: VERSION, points: mergePoints(documents)}, null, 2), "pareto_merged.json", "application/json;charset=utf-8");
  try {
    const entries = JSON.parse($("preloaded-points")?.textContent || "[]");
    if (entries.length) importDocuments(entries);
  } catch (error) { notice(error.message, true); }
}
