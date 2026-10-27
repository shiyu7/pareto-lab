const SCHEMA = "hpc-pareto-points";
const COLORS = ["#197b70", "#df8242", "#497cc0", "#9671b8", "#be5775", "#8c8139", "#477f92", "#69755a"];

export function parseDocument(data, name = "points.json") {
  if (data?.schema !== SCHEMA || data.version !== 1 || !Array.isArray(data.points) || !data.points.length) {
    throw new Error(`${name}：需要 hpc-pareto-points v1 点位 JSON，请先运行导出脚本。`);
  }
  return data.points.map((raw, index) => {
    const p = {...raw};
    const fail = message => { throw new Error(`${name} 第 ${index + 1} 点：${message}`); };
    const positive = value => typeof value === "number" && Number.isFinite(value) && value > 0;
    for (const field of ["scheme", "workload", "source"]) {
      if (typeof p[field] !== "string" || !p[field].trim()) fail(`缺少 ${field}`);
    }
    if (!["decode", "prefill"].includes(p.phase)) fail("phase 无效");
    if (!positive(p.x_value) || !positive(p.throughput)) fail("坐标必须是有限正数");
    if (p.phase === "decode" && (p.x_metric !== "per_user_tps" || p.throughput_metric !== "output_throughput")) fail("Decode 指标不匹配");
    if (p.phase === "prefill" && (!["prefill_latency_ms", "ttft_ms"].includes(p.x_metric)
        || !["non_cached_input_throughput", "input_throughput"].includes(p.throughput_metric))) fail("Prefill 指标不匹配");
    for (const field of ["num_gpus", "dp_size", "concurrency", "prefill_batch_size"]) {
      if (p[field] != null && (!positive(p[field]) || !Number.isSafeInteger(p[field]))) fail(`${field} 必须是正整数`);
    }
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
  const points = [], seen = new Set(), layouts = new Map();
  for (const document of documents) {
    for (const p of document.points) {
      const curve = JSON.stringify([groupKey(p), p.scheme]);
      const layout = JSON.stringify([p.num_gpus ?? null, p.dp_size ?? null]);
      if (layouts.has(curve) && layouts.get(curve) !== layout) {
        throw new Error(`方案「${p.scheme}」的 GPU/DP 数不同，请用 --series 为不同布局指定不同方案名。`);
      }
      layouts.set(curve, layout);
      // Ignore only exact duplicates; source paths do not identify a measurement.
      const identity = JSON.stringify([curve, layout, p.x_value, p.throughput, p.concurrency ?? null, p.prefill_batch_size ?? null]);
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
  return [...new Set(points.map(p => p.scheme))].map(scheme => {
    const series = points.filter(p => p.scheme === scheme), feasible = series.filter(p => meetsSlo(p, slo));
    const best = feasible.reduce((best, p) => !best || p.y > best.y ? p : best, null);
    const loads = feasible.map(p => p.concurrency).filter(n => n != null);
    return {scheme, total: series.length, feasible: feasible.length, best, maxConcurrency: loads.length ? Math.max(...loads) : null};
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
    const schemes = [...new Set(points.map(p => p.scheme))];
    const legendRows = Math.ceil(schemes.length / 3), width = 1100, height = 680 + legendRows * 22;
    const left = 110, top = 110 + legendRows * 22, right = 45, bottom = 85;
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
    schemes.forEach((scheme, index) => {
      const lx = left + index % 3 * 310, ly = 60 + Math.floor(index / 3) * 22;
      add("line", {x1: lx, x2: lx + 20, y1: ly, y2: ly, stroke: color(scheme), "stroke-width": 3});
      const label = add("text", {x: lx + 28, y: ly + 4, fill: "#354c5e"}, scheme.length > 33 ? scheme.slice(0, 31) + "…" : scheme);
      svgElement("title", {}, scheme, label);
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
    for (const scheme of schemes) {
      const series = points.filter(p => p.scheme === scheme).sort((a, b) => a.x_value - b.x_value);
      add("polyline", {points: series.map(p => `${x(p.x_value)},${y(p.y)}`).join(" "), fill: "none", stroke: color(scheme), "stroke-width": 2.5});
    }
    points.forEach((p, index) => {
      const circle = add("circle", {cx: x(p.x_value), cy: y(p.y), r: 5.5, fill: color(p.scheme), tabindex: 0});
      const detail = `${p.scheme} · x=${format(p.x_value)} · ${format(p.y)} ${unit} · ${pointLabel(p) || "负载未提供"} · GPU=${p.num_gpus ?? "—"}, DP=${p.dp_size ?? "—"} · ${meetsSlo(p, slo) ? "SLO 可行" : "SLO 未达标"} · ${p.source}`;
      svgElement("title", {}, detail, circle);
      circle.setAttribute("aria-label", detail);
      circle.onmouseenter = circle.onfocus = () => { $("detail").textContent = detail; };
      if ($("frontier").checked && frontier[index]) add("circle", {cx: x(p.x_value), cy: y(p.y), r: 9, fill: "none", stroke: "#263f4e", "stroke-width": 1.3, "pointer-events": "none"});
      if ($("labels").checked && pointLabel(p)) add("text", {x: x(p.x_value), y: y(p.y) - 15, "text-anchor": "middle", fill: color(p.scheme), "font-size": 10}, pointLabel(p));
    });
    return svg;
  }
  function render() {
    const points = mergePoints(documents).filter(p => groupKey(p) === $("group").value);
    $("schemes").replaceChildren();
    for (const scheme of new Set(points.map(p => p.scheme))) {
      const label = element("label", null, $("schemes")); label.className = "scheme";
      const checkbox = element("input", null, label); checkbox.type = "checkbox";
      checkbox.checked = !hidden.has(scheme);
      checkbox.onchange = () => { checkbox.checked ? hidden.delete(scheme) : hidden.add(scheme); render(); };
      const swatch = element("span", null, label); swatch.className = "swatch"; swatch.style.background = color(scheme);
      element("span", scheme, label);
    }
    const phase = points[0]?.phase;
    $("slo-label").textContent = phase === "decode" ? "最低单用户速度 · tokens/s" : "最大延迟 · ms";
    $("chart-title").textContent = phase ? `${phase.toUpperCase()} / 性能曲线` : "等待导入采样点";
    $("summary").replaceChildren();
    $("save-svg").disabled = true; activeSvg = null;
    $("detail").textContent = "悬停或聚焦测量点，查看并发、吞吐和来源。";
    const visible = points.filter(p => !hidden.has(p.scheme));
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
        for (const text of [row.scheme, `${row.feasible} / ${row.total}`, format(row.best?.y), row.best ? pointLabel(row.best) || "—" : "—", format(row.maxConcurrency)]) element("td", text, tr);
      }
      notice(`${display.length} 个实测点 · ${new Set(display.map(p => p.scheme)).size} 个方案 · ${frontier.filter(Boolean).length} 个非支配点 · ${slo == null ? "未设置 SLO" : "浅绿色为 SLO 可行区域"}`);
    } catch (error) {
      $("chart").replaceChildren(element("div", "请调整比较条件后继续。")); $("chart").firstChild.className = "empty";
      notice(error.message, true);
    }
  }
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
  $("clear").onclick = () => { documents = []; hidden.clear(); refreshOptions(); };
  $("all").onclick = () => { hidden.clear(); render(); };
  $("group").onchange = () => { $("slo").value = ""; render(); };
  for (const id of ["unit", "frontier", "labels"]) $(id).onchange = render;
  $("slo").oninput = render;
  $("save-svg").onclick = () => {
    if (activeSvg) download(new XMLSerializer().serializeToString(activeSvg), `${JSON.parse($("group").value)[0]}_pareto.svg`, "image/svg+xml;charset=utf-8");
  };
  $("save-points").onclick = () => download(JSON.stringify({schema: SCHEMA, version: 1, points: mergePoints(documents)}, null, 2), "pareto_merged.json", "application/json;charset=utf-8");
  try {
    const entries = JSON.parse($("preloaded-points")?.textContent || "[]");
    if (entries.length) importDocuments(entries);
  } catch (error) { notice(error.message, true); }
}
