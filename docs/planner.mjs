// Select measured operating points independently, then match request capacities.
// No arrival distribution or queue model is assumed by this theoretical planner.
export const calibrationKey = p => JSON.stringify([p.workload, p.scheme, p.tp_size, p.dp_size, p.ep_size, p.pp_size,
  p.num_gpus, p.chunked_prefill, p.chunk_size_tokens, p.chunk_scope]);
const positive = n => typeof n === 'number' && Number.isFinite(n) && n > 0;
const ceilCapacity = n => Math.max(1, Math.ceil(n - 1e-12 * Math.max(1, n)));

export function evaluateAllocation(p, d, nP, nD, o) {
  if (![nP,nD].every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('实例数超出正整数范围。');
  const gP = nP * p.point.num_gpus, gD = nD * d.point.num_gpus;
  const rawCapacityP = nP * p.qpsPerInstance, capacityP = rawCapacityP * (1 - o.prefillReserve);
  const capacityD = nD * d.qpsPerInstance;
  const qps = Math.min(capacityP, capacityD), rawQps = Math.min(rawCapacityP, capacityD);
  const dSeconds = (o.outputTokens - 1) / d.point.x_value;
  const row = {nP,nD,gP,gD,totalGpus:gP+gD,rawCapacityP,capacityP,capacityD,qps,rawQps,
    outputTps:qps*o.outputTokens, decodeTps:qps*d.tokensPerRequest,
    rawOutputTps:rawQps*o.outputTokens, decodeTpsPerGpu:qps*d.tokensPerRequest/gD,
    normalizedDConcurrency:d.point.concurrency/d.point.num_gpus,
    measuredDConcurrencyPerInstance:d.point.concurrency,
    measuredDConcurrencyTotal:nD*d.point.concurrency,
    estimatedDActive:qps*dSeconds,
    noQueueInFlight:qps*((p.serviceMs+o.latencyReserveMs)/1000+dSeconds),
    pHeadroom:1-qps/rawCapacityP, dHeadroom:1-qps/capacityD,
    imbalance:Math.abs(capacityP-capacityD)/Math.max(capacityP,capacityD),
    bottleneck:Math.abs(capacityP-capacityD)<1e-9*Math.max(capacityP,capacityD)?'平衡':capacityP<capacityD?'Prefill':'Decode'};
  if (!Object.values(row).filter(v=>typeof v==='number').every(Number.isFinite)
    || !Number.isSafeInteger(row.totalGpus) || !Number.isSafeInteger(row.measuredDConcurrencyTotal)) throw new Error('部署规模或换算结果超出可表示范围。');
  return row;
}

export function planDeployment(points, options) {
  const o = {latencyReserveMs:0,prefillReserve:0,ratioTolerance:.05,maxInstances:1024,
    outputTokenScope:'all-output',targetQps:null,useCompleteStep:false,...options};
  for (const key of ['ttftMs','tpotMs','inputTokens','outputTokens']) if (!positive(o[key])) throw new Error(`${key} 必须为有限正数。`);
  if (o.outputTokens <= 1) throw new Error('输出长度须大于 1 token，才能定义 TPOT。');
  if (!['input_throughput','non_cached_input_throughput'].includes(o.inputMetric)) throw new Error('Prefill 吞吐口径无效。');
  if (!['all-output','decode-only'].includes(o.outputTokenScope)) throw new Error('Decode token 统计范围无效。');
  if (o.inputMetric==='non_cached_input_throughput' && (!positive(o.nonCachedTokens)||o.nonCachedTokens>o.inputTokens)) throw new Error('非缓存输入长度须大于 0 且不超过完整输入长度。');
  if (!Number.isFinite(o.latencyReserveMs)||o.latencyReserveMs<0||o.latencyReserveMs>=o.ttftMs) throw new Error('延迟预留须非负且小于 TTFT 上限。');
  if (!Number.isFinite(o.prefillReserve)||o.prefillReserve<0||o.prefillReserve>=1) throw new Error('P 容量余量须在 0%–100% 之间（不含 100%）。');
  if (!Number.isFinite(o.ratioTolerance)||o.ratioTolerance<0||o.ratioTolerance>1) throw new Error('配平容差须在 0%–100% 之间。');
  if (!Number.isInteger(o.maxInstances)||o.maxInstances<1||o.maxInstances>4096) throw new Error('单侧实例搜索上限须为 1–4096 的整数。');
  if (o.targetQps!=null&&!positive(o.targetQps)) throw new Error('目标 QPS 须为有限正数或留空。');
  const skipped = {}, skip = reason => { skipped[reason]=(skipped[reason]||0)+1; };
  const pCandidates=[],dCandidates=[];
  const prefillBudgetMs=o.ttftMs-o.latencyReserveMs;
  for (const p of points.filter(p=>p.workload===o.workload)) {
    if (p.phase==='prefill'&&p.throughput_metric!==o.inputMetric) continue;
    if (![p.num_gpus,p.tp_size,p.dp_size,p.ep_size,p.pp_size].every(n=>positive(n)&&Number.isSafeInteger(n))) {skip('缺少 GPU 或 TP/DP/EP/PP 配置');continue;}
    if (p.throughput_unit!=='tokens/s'||!positive(p.throughput)) throw new Error('规划器需要归一化后的实例总吞吐。');
    const tpsPerGpu=p.throughput/p.num_gpus;
    if (p.phase==='prefill') {
      let serviceMs=p.prefill_service_ms??o.serviceOverrides?.[calibrationKey(p)];
      let serviceSource=p.prefill_service_ms!=null?'完整请求点位记录':'手动校准';
      if (serviceMs==null&&o.useCompleteStep&&p.chunked_prefill===false&&p.x_metric==='prefill_latency_ms') {serviceMs=p.x_value;serviceSource='用户确认单步覆盖完整请求';}
      if (!positive(serviceMs)) {skip('缺少完整请求 Prefill 服务时间');continue;}
      if (p.chunked_prefill==null) {skip('Prefill chunk 开关未知');continue;}
      if (serviceMs>prefillBudgetMs) {skip('Prefill 完整请求延迟超预算');continue;}
      const tokensPerRequest=o.inputMetric==='input_throughput'?o.inputTokens:o.nonCachedTokens;
      const qpsPerInstance=p.throughput/tokensPerRequest;
      if (!positive(qpsPerInstance)) {skip('容量换算超出数值范围');continue;}
      pCandidates.push({point:p,serviceMs,serviceSource,tpsPerGpu,tokensPerRequest,qpsPerInstance,qpsPerGpu:qpsPerInstance/p.num_gpus});
    } else if(p.phase==='decode') {
      const tpotMs=1000/p.x_value;
      if (!positive(tpotMs)||tpotMs>o.tpotMs) {skip('Decode 点位不满足 TPOT');continue;}
      if (!positive(p.concurrency)) {skip('Decode 缺少采样并发');continue;}
      const tokensPerRequest=o.outputTokenScope==='decode-only'?o.outputTokens-1:o.outputTokens;
      const qpsPerInstance=p.throughput/tokensPerRequest;
      if (!positive(qpsPerInstance)) {skip('容量换算超出数值范围');continue;}
      dCandidates.push({point:p,tpotMs,tpsPerGpu,tokensPerRequest,qpsPerInstance,qpsPerGpu:qpsPerInstance/p.num_gpus});
    }
  }
  // Monotonic throughput along a curve is not assumed; latency only constrains selection.
  const order=(a,b)=>b.tpsPerGpu-a.tpsPerGpu || (a.serviceMs??a.tpotMs)-(b.serviceMs??b.tpotMs) || a.point.num_gpus-b.point.num_gpus;
  pCandidates.sort(order); dCandidates.sort(order);
  const p=pCandidates[0]??null,d=dCandidates[0]??null;
  const result={model:'independent-stage-balance-v1',objective:'independent_prefill_and_decode_tokens_per_second_per_gpu',
    options:o,prefillBudgetMs,p,d,stageCandidates:{prefill:pCandidates.slice(0,5),decode:dCandidates.slice(0,5)},
    prefillCandidates:pCandidates.length,decodeCandidates:dCandidates.length,skipped,ratio:null,base:null,deployment:null,scans:[],warnings:[]};
  if(!p||!d) return result;
  const usableP=p.qpsPerInstance*(1-o.prefillReserve);
  if(!positive(usableP)) throw new Error('P 可用容量超出数值范围。');
  result.ratio={gpuRaw:d.qpsPerGpu/p.qpsPerGpu,gpuWithReserve:d.qpsPerGpu/(p.qpsPerGpu*(1-o.prefillReserve)),
    instancesRaw:d.qpsPerInstance/p.qpsPerInstance,instancesWithReserve:d.qpsPerInstance/usableP};
  const allocations=[];
  // For each D count, ceil() supplies the minimum P count that feeds its full capacity.
  for(let nD=1;nD<=o.maxInstances;nD++) {
    const nP=ceilCapacity(nD*d.qpsPerInstance/usableP);
    if(nP>o.maxInstances) break;
    const a=evaluateAllocation(p,d,nP,nD,o);
    a.pExcess=a.capacityP/a.capacityD-1;
    allocations.push(a);
  }
  const matched=allocations.filter(a=>a.pExcess<=o.ratioTolerance+1e-12);
  const base=(matched.length?matched.sort((a,b)=>a.totalGpus-b.totalGpus||a.pExcess-b.pExcess):allocations.sort((a,b)=>a.pExcess-b.pExcess||a.totalGpus-b.totalGpus))[0];
  if(base) result.base={...base,withinTolerance:matched.length>0};
  if(!base) result.warnings.push('当前单侧实例上限内无法让 P 供给覆盖 D。可增大配比搜索上限；已选阶段点不受此上限影响。');
  else if(!matched.length) result.warnings.push('基础配比未达到所设配平容差，展示搜索范围内 P 富余最少的一组；可放宽容差或增加实例搜索上限。');
  if(o.targetQps!=null) {
    const nP=ceilCapacity(o.targetQps/usableP),nD=ceilCapacity(o.targetQps/d.qpsPerInstance);
    result.deployment={...evaluateAllocation(p,d,nP,nD,o),mode:'target',targetQps:o.targetQps};
  } else if(result.base) result.deployment={...result.base,mode:'base',targetQps:null};
  const a=result.deployment;
  if(!a) return result;
  const dTime=(o.outputTokens-1)/d.point.x_value;
  if(a.estimatedDActive>a.measuredDConcurrencyTotal*1.1) result.warnings.push('吞吐与 TPOT 推导的 D 活动并发明显高于采样并发，请核对吞吐范围、并发含义和统计口径；未用并发反推值覆盖实测吞吐。');
  const anchor=o.targetQps??a.qps;
  result.scans=[.5,.7,.85,1,1.05].map(f=>{
    const qps=anchor*f;
    return {qps,fraction:f,estimatedDActive:qps*dTime,noQueueInFlight:qps*((p.serviceMs+o.latencyReserveMs)/1000+dTime),
      withinCapacity:qps<=a.qps*(1+1e-12),atCapacity:Math.abs(qps-a.qps)<=a.qps*1e-9};
  });
  return result;
}
