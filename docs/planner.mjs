// Capacity-based queue approximation, not an inference scheduler simulator.
// Per-instance queue: Wq = rho / (mu * (1-rho)); replicas receive balanced traffic.
export const calibrationKey = p => JSON.stringify([p.workload, p.scheme, p.tp_size, p.dp_size, p.ep_size, p.pp_size,
  p.num_gpus, p.chunked_prefill, p.chunk_size_tokens, p.chunk_scope]);
export function queueWait(rate, capacity) {
  if (rate === 0) return 0;
  if (!(rate >= 0 && capacity > rate)) return Infinity;
  return rate / (capacity * (capacity - rate));
}
const gcd = (a, b) => b ? gcd(b, a % b) : a;
const positive = n => typeof n === 'number' && Number.isFinite(n) && n > 0;
export function planDeployment(points, options) {
  const o = {...options};
  for (const key of ['ttftMs', 'tpotMs', 'inputTokens', 'outputTokens', 'gpuBudget', 'utilization']) {
    if (!positive(o[key])) throw new Error(`${key} 必须为有限正数。`);
  }
  if (!Number.isInteger(o.gpuBudget) || o.gpuBudget > 4096) throw new Error('搜索规模上限须为 1–4096 GPU 的整数。');
  if (o.outputTokens < 2) throw new Error('输出长度须至少为 2 tokens，才能定义 TPOT。');
  if (!(o.utilization < 1)) throw new Error('容量使用上限必须低于 100%。');
  if (!Number.isFinite(o.overheadMs) || o.overheadMs < 0) throw new Error('额外开销须为非负数。');
  if (o.inputMetric === 'non_cached_input_throughput' && (!positive(o.nonCachedTokens) || o.nonCachedTokens > o.inputTokens)) {
    throw new Error('非缓存输入长度须大于 0 且不超过完整输入长度。');
  }
  const skipped = {}, skip = reason => { skipped[reason] = (skipped[reason] || 0) + 1; };
  const pCandidates = [], dCandidates = [];
  for (const p of points.filter(p => p.workload === o.workload)) {
    if (p.phase === 'prefill' && p.throughput_metric !== o.inputMetric) continue;
    if (![p.num_gpus, p.tp_size, p.dp_size, p.ep_size, p.pp_size].every(n => positive(n) && Number.isSafeInteger(n))) { skip('缺少 GPU 或 TP/DP/EP/PP 配置'); continue; }
    if (p.phase === 'prefill') {
      const serviceMs = p.prefill_service_ms ?? o.serviceOverrides?.[calibrationKey(p)];
      if (!positive(serviceMs)) { skip('缺少完整请求 Prefill 服务时间（排除排队）'); continue; }
      if (p.chunked_prefill == null) { skip('Prefill chunk 开关未知'); continue; }
      const tokens = o.inputMetric === 'input_throughput' ? o.inputTokens : o.nonCachedTokens;
      const mu = p.throughput / tokens;
      if (!positive(mu)) { skip('容量换算超出数值范围'); continue; }
      pCandidates.push({point: p, serviceMs, serviceSource: p.prefill_service_ms != null ? '点位记录' : '手动校准', mu});
    } else if (p.phase === 'decode') {
      if (1000 / p.x_value > o.tpotMs) { skip('Decode 点位不满足 TPOT'); continue; }
      if (!positive(p.concurrency)) { skip('Decode 缺少实测总并发'); continue; }
      // Bound by both measured token throughput and measured active-request slots.
      const seconds = (o.outputTokens - 1) / p.x_value;
      const mu = Math.min(p.throughput / o.outputTokens, p.concurrency / seconds);
      if (!positive(mu) || !positive(seconds)) { skip('容量换算超出数值范围'); continue; }
      dCandidates.push({point: p, mu, seconds, tpotMs: 1000 / p.x_value});
    }
  }
  const estimatedWork = pCandidates.reduce((sum, p) => sum + dCandidates.reduce((n, d) => n + Math.floor(o.gpuBudget / p.point.num_gpus) * Math.floor(o.gpuBudget / d.point.num_gpus) / 2, 0), 0);
  if (estimatedWork > 2000000) throw new Error('候选组合过多，请减少导入点位或缩小 GPU 搜索范围。');
  const bestByLayout = new Map();
  for (const p of pCandidates) for (const d of dCandidates) {
    const baseMs = p.serviceMs + o.overheadMs + d.tpotMs;
    if (baseMs >= o.ttftMs) continue;
    // Efficiency is invariant under proportional scaling in this model. Search all
    // primitive integer ratios; maximum affordable D alone is not sufficient.
    for (let nP = 1; nP * p.point.num_gpus + d.point.num_gpus <= o.gpuBudget; nP++) {
      for (let nD = 1; nP * p.point.num_gpus + nD * d.point.num_gpus <= o.gpuBudget; nD++) {
        if (gcd(nP, nD) !== 1) continue;
        const capacityP = nP * p.mu, capacityD = nD * d.mu;
        const physicalCapacity = Math.min(capacityP, capacityD);
        if (!positive(physicalCapacity)) continue;
        const ttftAt = q => baseMs + 1000 * (queueWait(q / nP, p.mu) + queueWait(q / nD, d.mu));
        let low = 0, high = physicalCapacity * o.utilization;
        if (ttftAt(high) <= o.ttftMs) low = high;
        else for (let i = 0; i < 48; i++) {
          const mid = (low + high) / 2;
          if (ttftAt(mid) <= o.ttftMs) low = mid; else high = mid;
        }
        const qps = low;
        if (!positive(qps)) continue;
        const ttftMs = ttftAt(qps), meanInFlight = qps * (ttftMs / 1000 + d.seconds);
        const totalGpus = nP * p.point.num_gpus + nD * d.point.num_gpus;
        const row = {p, d, nP, nD, qps, totalGpus, ttftMs, meanInFlight,
          outputTps: qps * o.outputTokens, outputTpsPerGpu: qps * o.outputTokens / totalGpus, inputTps: qps * o.inputTokens,
          queuePMs: 1000 * queueWait(qps / nP, p.mu), queueDMs: 1000 * queueWait(qps / nD, d.mu),
          utilizationP: qps / capacityP, utilizationD: qps / capacityD,
          physicalCapacity, capacityP, capacityD, clientCap: Math.ceil(meanInFlight * 1.3),
          decodeActiveMean: qps * d.seconds, decodeMeasuredCap: nD * d.point.concurrency};
        const key = JSON.stringify([calibrationKey(p.point), calibrationKey(d.point), nP, nD]);
        if (!bestByLayout.has(key) || bestByLayout.get(key).qps < qps) bestByLayout.set(key, row);
      }
    }
  }
  const candidates = [...bestByLayout.values()].sort((a,b) => Math.abs(b.outputTpsPerGpu - a.outputTpsPerGpu) <= 1e-9 * Math.max(a.outputTpsPerGpu, b.outputTpsPerGpu) ? a.totalGpus - b.totalGpus : b.outputTpsPerGpu - a.outputTpsPerGpu).slice(0, 5);
  const best = candidates[0];
  const scans = best ? [...new Set([.5, .7, .85, 1, 1.1].map(f => Math.min(best.qps * f, best.physicalCapacity * Math.max(.98, (1 + o.utilization) / 2))))].map(qps => {
    const ttftMs = best.p.serviceMs + o.overheadMs + best.d.tpotMs + 1000 * (queueWait(qps / best.nP, best.p.mu) + queueWait(qps / best.nD, best.d.mu));
    const inFlight = qps * (ttftMs / 1000 + best.d.seconds);
    return {qps, ttftMs, meanInFlight: inFlight, clientCap: Math.ceil(inFlight * 1.3), withinModel: ttftMs <= o.ttftMs + 1e-7 && qps <= best.physicalCapacity * o.utilization * (1 + 1e-12)};
  }) : [];
  return {model: 'capacity-queue-mean-v1', objective: 'output_tokens_per_second_per_total_gpu', options: o, candidates, scans, skipped,
    prefillCandidates: pCandidates.length, decodeCandidates: dCandidates.length};
}
