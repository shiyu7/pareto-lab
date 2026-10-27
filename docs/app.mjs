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
    throw new Error(`${name}：需要 hpc-pareto-points v1/v2 点位 JSON，请先运行导出脚本。`);
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
    const best = result.candidates[0];
    if (!best) {
      const p = element("p", "未找到可推荐的组合。检查搜索规模上限、TPOT 门槛、完整 Prefill 服务时间及两阶段是否属于同一 workload。单步延迟或已有 TTFT 不会被自动代替为服务时间。", root); p.className = "plan-warning";
    } else {
      const stats = element("div", null, root); stats.className = "plan-stats";
      for (const [value, label] of [[`${best.nP}P + ${best.nD}D`, `合计 ${best.totalGpus} GPU · 最小整数配比`], [format(best.qps), "候选全局 QPS · 均值模型"], [format(best.outputTpsPerGpu), "估算输出 tokens/s/GPU · 优化目标"], [format(best.ttftMs), `估算 TTFT ms · TPOT ${format(best.d.tpotMs)} ms`]]) {
        const card = element("div", null, stats); card.className = "plan-stat"; element("strong", value, card); element("span", label, card);
      }
      for (const [name, count, item] of [["Prefill", best.nP, best.p], ["Decode", best.nD, best.d]]) {
        const p = element("p", null, root); p.className = "plan-copy";
        element("strong", `${name} × ${count}：`, p); element("span", curveLabel(item.point), p);
        element("br", null, p); element("small", `采样锚点：${pointLabel(item.point)} · ${format(item.point.throughput)} tokens/s/实例 · ${item.point.source}`, p);
      }
      element("p", `该组合总输出 ${format(best.outputTps)} tokens/s，除以 P/D 合计 ${best.totalGpus} GPU，得到 ${format(best.outputTpsPerGpu)} tokens/s/GPU。搜索规模上限 ${result.options.gpuBudget} GPU，不要求用满；结果为该有限范围内的单卡最优候选。完整 Prefill ${format(best.p.serviceMs)} ms（${best.p.serviceSource}）+ P 排队 ${format(best.queuePMs)} ms + D 准入排队 ${format(best.queueDMs)} ms + 额外开销 ${format(result.options.overheadMs)} ms + 首 token 近似 ${format(best.d.tpotMs)} ms。P / D 容量使用率 ${format(best.utilizationP * 100)}% / ${format(best.utilizationD * 100)}%。`, root);
      element("p", `Decode 每实例以实测 C=${format(best.d.point.concurrency)} 作为活动请求上限候选（全局 ${format(best.decodeMeasuredCap)}）；候选 QPS 下平均 D 活动请求约 ${format(best.decodeActiveMean)}。端到端平均在途请求约 ${format(best.meanInFlight)}，由 QPS × 平均完成时间计算；压测客户端并发保护阈值先设 ${best.clientCap}（均值 × 1.3 向上取整），它不是服务端活动并发，也不保证没有排队。`, root);
      const warning = element("p", "这是所选搜索范围、导入采样与均值模型下的单卡最优候选，不是已测最大吞吐或 P95/P99 承诺。请先确认点位长度和均值口径；额外副本扩展、路由、KV 传输及 P/D 相互干扰需要联合压测。", root); warning.className = "plan-warning";
      element("h3", "压测扫描：先控 QPS，再定位 SLO 边界", root);
      planTable(root, ["开放环路 QPS", "平均在途估算", "客户端保护阈值", "TTFT 均值估算 · ms", "位置"], result.scans.map(s => [format(s.qps), format(s.meanInFlight), String(s.clientCap), format(s.ttftMs), s.withinModel ? "模型约束内" : "探索边界，可能超 SLO"]));
      element("p", "按上表递增，用泊松到达的开放环路发压，固定输入/输出分布、缓存条件和 D 活动请求上限。预热后持续采样，记录实际到达/完成 QPS、输出吞吐、均值及 P95/P99 TTFT/TPOT、队列长度、错误与限流。保护阈值触发时，实际到达不再符合目标 QPS，应记录后调整。达标最高点附近继续二分加密；队列持续增长即停止升压。闭环固定并发测试应单独扫描，不能同时把实际 QPS 当独立控制量。", root);
      element("h3", "其他候选配比（按单卡输出吞吐排序）", root);
      planTable(root, ["配比 / GPU", "P 配置", "D 配置", "候选 QPS", "输出 tokens/s/GPU"], result.candidates.map(c => [`${c.nP}P + ${c.nD}D / ${c.totalGpus}`, curveLabel(c.p.point), curveLabel(c.d.point), format(c.qps), format(c.outputTpsPerGpu)]));
    }
    element("p", `可计算的 Prefill 点 ${result.prefillCandidates} 个，TPOT 达标的 Decode 点 ${result.decodeCandidates} 个。`, root);
    if (Object.keys(result.skipped).length) element("p", `跳过点位：${Object.entries(result.skipped).map(([reason, n]) => `${reason} × ${n}`).join("；")}。展开“模型假设与缺失数据校准”可补充服务时间。`, root);
  }
  $("planner-form").oninput = invalidatePlan;
  $("plan-workload").onchange = $("plan-metric").onchange = () => { refreshCalibration(); invalidatePlan(); };
  $("planner-form").onsubmit = event => {
    event.preventDefault();
    try {
      const num = id => Number($(id).value);
      const options = {workload: $("plan-workload").value, inputMetric: $("plan-metric").value,
        ttftMs: num("plan-ttft"), tpotMs: num("plan-tpot"), gpuBudget: num("plan-gpus"), inputTokens: num("plan-input"),
        nonCachedTokens: num("plan-noncached"), outputTokens: num("plan-output"), utilization: num("plan-util") / 100,
        overheadMs: num("plan-overhead"), serviceOverrides: {...serviceOverrides}};
      activePlan = planDeployment(mergePoints(documents), options);
      renderPlan(activePlan); $("save-plan").disabled = !activePlan.candidates.length;
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
