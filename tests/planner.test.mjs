import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {planDeployment,evaluateAllocation,calibrationKey} from '../docs/planner.mjs';
import {parseDocument,mergePoints} from '../docs/app.mjs';
const p={phase:'prefill',workload:'test',scheme:'P',num_gpus:4,tp_size:1,dp_size:4,ep_size:4,pp_size:1,
  chunked_prefill:false,prefill_service_ms:50,throughput:10000,throughput_metric:'input_throughput',x_metric:'prefill_latency_ms',x_value:10,source:'p:1',throughput_unit:'tokens/s'};
const d={...p,phase:'decode',scheme:'D',chunked_prefill:null,prefill_service_ms:null,throughput:2000,
  throughput_metric:'output_throughput',x_metric:'per_user_tps',x_value:100,concurrency:20};
const o={workload:'test',inputMetric:'input_throughput',inputTokens:1000,outputTokens:100,ttftMs:100,tpotMs:10,
  latencyReserveMs:10,prefillReserve:0,ratioTolerance:.05,maxInstances:1024,outputTokenScope:'all-output'};
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);

test('selects maximum feasible throughput/GPU independently, not maximum latency or total throughput',()=>{
  const r=planDeployment([p,{...p,scheme:'slower',prefill_service_ms:80,throughput:9000},
    {...p,scheme:'too-late',prefill_service_ms:91,throughput:20000},
    d,{...d,scheme:'more-GPUs',throughput:10000,num_gpus:32},
    {...d,scheme:'too-slow',x_value:99,throughput:3000}],o);
  assert.equal(r.p.point.scheme,'P');assert.equal(r.d.point.scheme,'D');
  close(r.p.tpsPerGpu,2500);close(r.d.tpsPerGpu,500);
  close(r.p.qpsPerInstance,10); // Not 1 / 50ms.
  close(r.d.qpsPerInstance,20);close(r.ratio.gpuRaw,2);
  assert.equal(r.base.nP,2);assert.equal(r.base.nD,1);close(r.deployment.qps,20);
  close(r.deployment.outputTps,2000);
});
test('latency boundaries are inclusive and no extra TPOT is added to TTFT',()=>{
  const r=planDeployment([{...p,prefill_service_ms:90},d],o);
  assert.ok(r.deployment);assert.equal(r.prefillBudgetMs,90);
  assert.equal(planDeployment([{...p,prefill_service_ms:90.01},d],o).p,null);
  assert.equal(planDeployment([p,{...d,x_value:99.99}],o).d,null);
});
test('full-request service is required; explicit unchunked confirmation is narrowly scoped',()=>{
  const missing={...p,prefill_service_ms:null};
  assert.equal(planDeployment([missing,d],o).p,null);
  const confirmed=planDeployment([missing,d],{...o,useCompleteStep:true});
  close(confirmed.p.serviceMs,10);
  for(const bad of [{...missing,x_metric:'ttft_ms'}, {...missing,chunked_prefill:true}, {...missing,chunked_prefill:null}]) {
    assert.equal(planDeployment([bad,d],{...o,useCompleteStep:true}).p,null);
  }
  assert.ok(planDeployment([missing,d],{...o,serviceOverrides:{[calibrationKey(missing)]:50}}).p);
  close(planDeployment([p,d],{...o,serviceOverrides:{[calibrationKey(p)]:900}}).p.serviceMs,50);
});
test('P capacity reserve changes allocation without changing selected P/D capability',()=>{
  const r=planDeployment([p,d],{...o,prefillReserve:.2});
  close(r.ratio.gpuRaw,2);close(r.ratio.gpuWithReserve,2.5);
  assert.equal(r.base.nP,5);assert.equal(r.base.nD,2);
  close(r.deployment.capacityP,40);close(r.deployment.capacityD,40);
  close(r.p.tpsPerGpu,2500);close(r.d.tpsPerGpu,500);
});
test('removing excess P cannot improve Decode capability; P starvation reduces realized output',()=>{
  const r=planDeployment([p,d],o);
  const more=evaluateAllocation(r.p,r.d,3,1,r.options),balanced=evaluateAllocation(r.p,r.d,2,1,r.options),less=evaluateAllocation(r.p,r.d,1,1,r.options);
  close(more.decodeTpsPerGpu,balanced.decodeTpsPerGpu);close(balanced.decodeTpsPerGpu,500);
  close(less.decodeTpsPerGpu,250);close(less.qps,10);
  assert.equal(less.bottleneck,'Prefill');
});
test('target QPS rounds each stage separately and does not alter selected points',()=>{
  const r=planDeployment([p,d],{...o,targetQps:25,maxInstances:1});
  assert.equal(r.base,null);assert.equal(r.deployment.nP,3);assert.equal(r.deployment.nD,2);
  assert.equal(r.deployment.mode,'target');close(r.deployment.qps,30);
  assert.equal(r.scans[3].qps,25);assert.equal(r.p.point,p);assert.equal(r.d.point,d);
});
test('GPU ratios and instance ratios are distinct; proportional replication preserves per-D capability',()=>{
  const r=planDeployment([{...p,num_gpus:8},d],o);
  close(r.ratio.gpuRaw,4);close(r.ratio.instancesRaw,2);
  const a=r.deployment,b=evaluateAllocation(r.p,r.d,a.nP*3,a.nD*3,r.options);
  close(b.qps,a.qps*3);close(b.outputTps,a.outputTps*3);close(b.decodeTpsPerGpu,a.decodeTpsPerGpu);
});
test('first-token scope affects requests/s while full-output and D-only throughput stay distinct',()=>{
  const r=planDeployment([p,d],{...o,outputTokenScope:'decode-only',targetQps:10});
  close(r.d.tokensPerRequest,99);close(r.d.qpsPerInstance,2000/99);
  close(r.deployment.qps,10);close(r.deployment.outputTps,1000);close(r.deployment.decodeTps,990);
  close(r.deployment.estimatedDActive,9.9);
});
test('cached input denominator and incomplete metadata are handled explicitly',()=>{
  const cached={...p,throughput:5000,throughput_metric:'non_cached_input_throughput'};
  close(planDeployment([cached,d],{...o,inputMetric:cached.throughput_metric,nonCachedTokens:500}).p.qpsPerInstance,10);
  for(const bad of [{...p,tp_size:null},{...p,chunked_prefill:null},{...p,workload:'other'}]) assert.equal(planDeployment([bad,d],o).p,null);
  assert.equal(planDeployment([p,{...d,concurrency:null}],o).d,null);
});
test('unattainable integer ratio tolerance is reported, not used to change the chosen stage',()=>{
  const r=planDeployment([{...p,throughput:12000},d],{...o,maxInstances:2,ratioTolerance:0});
  assert.equal(r.base.withinTolerance,false);assert.ok(r.warnings.length);close(r.p.tpsPerGpu,3000);
  assert.ok(r.base.capacityP>=r.base.capacityD);
});
test('rejects invalid or unnormalized inputs and reports concurrency mismatch without replacing throughput',()=>{
  for(const bad of [{prefillReserve:1},{prefillReserve:-1},{ratioTolerance:-1},{maxInstances:1.5},{maxInstances:5000},
    {inputTokens:0},{outputTokens:1},{latencyReserveMs:-1},{latencyReserveMs:100},{ttftMs:Infinity},{targetQps:0},{outputTokenScope:'unknown'}]) assert.throws(()=>planDeployment([p,d],{...o,...bad}));
  assert.throws(()=>planDeployment([p,d],{...o,inputMetric:'non_cached_input_throughput',nonCachedTokens:2000}));
  assert.throws(()=>planDeployment([{...p,throughput_unit:'tokens/s/GPU'},d],o));
  const r=planDeployment([p,{...d,concurrency:1}],o);close(r.d.qpsPerInstance,20);assert.ok(r.warnings.some(s=>s.includes('并发')));
});
test('service measurements survive parse and deduplication',()=>{
  const doc={schema:'hpc-pareto-points',version:2,points:[p,{...p,prefill_service_ms:60}]};
  assert.equal(mergePoints([{points:parseDocument(doc)}]).length,2);
  for(const invalid of [-1,Infinity,'50']) assert.throws(()=>parseDocument({...doc,points:[{...p,prefill_service_ms:invalid}]}));
});
test('synthetic example exports source points, ratio, allocations and scans without queue predictions',()=>{
  const pts=parseDocument(JSON.parse(readFileSync(new URL('../docs/example.json',import.meta.url))));
  const r=planDeployment(pts,{...o,workload:'synthetic-context-128k',inputMetric:'non_cached_input_throughput',inputTokens:131072,nonCachedTokens:131072,outputTokens:1024,ttftMs:3000,tpotMs:3.34});
  assert.equal(r.d.point.scheme,'EP64DP64');assert.equal(r.d.point.concurrency,2048);
  assert.equal(r.p.point.chunk_size_tokens,16384);assert.ok(r.deployment);
  assert.equal(r.model,'independent-stage-balance-v1');assert.ok(r.deployment.capacityP>=r.deployment.capacityD*(1-1e-12));
  assert.ok(!('ttftMs' in r.scans[0]));assert.equal(r.scans.length,5);
  const data=JSON.parse(JSON.stringify(r));assert.ok(data.p.point.source);assert.ok(data.d.point.source);
});
