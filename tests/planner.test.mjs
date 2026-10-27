import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {planDeployment, queueWait, calibrationKey} from '../docs/planner.mjs';
import {parseDocument, mergePoints} from '../docs/app.mjs';
const p = {phase:'prefill',workload:'test',scheme:'P',num_gpus:4,tp_size:1,dp_size:4,ep_size:4,pp_size:1,
  chunked_prefill:false,prefill_service_ms:50,throughput:10000,throughput_metric:'input_throughput',x_metric:'prefill_latency_ms',x_value:10,source:'p:1',throughput_unit:'tokens/s'};
const d = {...p,phase:'decode',scheme:'D',chunked_prefill:null,prefill_service_ms:null,throughput:2000,
  throughput_metric:'output_throughput',x_metric:'per_user_tps',x_value:100,concurrency:20};
const o = {workload:'test',inputMetric:'input_throughput',inputTokens:1000,outputTokens:100,gpuBudget:16,ttftMs:500,tpotMs:10,overheadMs:10,utilization:.8};
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);
test('queue wait and stability boundaries',()=>{close(queueWait(5,10),.1); assert.equal(queueWait(0,10),0); assert.equal(queueWait(10,10),Infinity);assert.equal(queueWait(11,10),Infinity);});
test('per-GPU optimizer prefers efficient ratio over higher total throughput',()=>{
  const r=planDeployment([p,d],o), b=r.candidates[0];
  assert.ok(b);assert.equal(b.nP,2);assert.equal(b.nD,1);close(b.qps, 14.827586206896552);
  assert.equal(b.totalGpus,12);close(b.outputTpsPerGpu,b.outputTps/12);
  assert.ok(r.candidates[1].qps>b.qps);
  assert.ok(r.candidates[1].outputTpsPerGpu<b.outputTpsPerGpu);
  assert.ok(b.ttftMs<=o.ttftMs);assert.ok(b.utilizationP<=.8);assert.ok(b.utilizationD<=.8);
  close(b.meanInFlight,b.qps*(b.ttftMs/1000+99/100));
  assert.ok(b.decodeActiveMean<=b.decodeMeasuredCap);
  assert.ok(r.scans.every(s=>s.qps<b.physicalCapacity));
  assert.equal(r.scans.at(-1).withinModel,false);
});
test('tight TTFT limits rate continuously instead of using raw step or mean TTFT',()=>{
  const b=planDeployment([p,d],{...o,ttftMs:100}).candidates[0];
  assert.ok(b.qps<16);close(b.ttftMs,100);
  assert.equal(planDeployment([p,d],{...o,ttftMs:70}).candidates.length,0);
  const missing={...p,prefill_service_ms:null,x_metric:'ttft_ms',x_value:1};
  assert.equal(planDeployment([missing,d],o).candidates.length,0);
  assert.ok(planDeployment([missing,d],{...o,serviceOverrides:{[calibrationKey(missing)]:50}}).candidates.length);
  assert.equal(planDeployment([missing,d],{...o,serviceOverrides:{[calibrationKey(missing)]:-10}}).candidates.length,0);
});
test('excludes unmeasured configurations, absent decode concurrency and mismatched workloads',()=>{
  for(const bad of [{...p,tp_size:null},{...p,chunked_prefill:null},{...p,prefill_service_ms:null},{...p,workload:'other'}]) assert.equal(planDeployment([bad,d],o).candidates.length,0);
  for(const bad of [{...d,concurrency:null},{...d,x_value:99},{...d,num_gpus:17}]) assert.equal(planDeployment([p,bad],o).candidates.length,0);
});
test('cached input normalization matches capacity and rejects bad budgets',()=>{
  const cached={...p,throughput:5000,throughput_metric:'non_cached_input_throughput'};
  const b=planDeployment([cached,d],{...o,inputMetric:cached.throughput_metric,nonCachedTokens:500}).candidates[0];
  close(b.qps,planDeployment([p,d],o).candidates[0].qps);
  for(const bad of [{utilization:1},{gpuBudget:1.5},{gpuBudget:5000},{inputTokens:0},{outputTokens:1},{overheadMs:-1},{ttftMs:Infinity}]) assert.throws(()=>planDeployment([p,d],{...o,...bad}));
  assert.throws(()=>planDeployment([cached,d],{...o,inputMetric:cached.throughput_metric,nonCachedTokens:2000}));
});
test('service measurements survive export parsing and deduplication',()=>{
  const doc={schema:'hpc-pareto-points',version:2,points:[p,{...p,prefill_service_ms:60}]};
  const pts=parseDocument(doc);assert.equal(mergePoints([{points:pts}]).length,2);
  for(const invalid of [-1,Infinity,'50']) assert.throws(()=>parseDocument({...doc,points:[{...p,prefill_service_ms:invalid}]}));
  assert.throws(()=>parseDocument({...doc,points:[{...d,prefill_service_ms:10}]}));
});
test('decode capacity is bounded by measured active slots',()=>{
  const b=planDeployment([p,{...d,throughput:100000,concurrency:1}],o).candidates[0];
  close(b.d.mu,1/.99);assert.ok(b.decodeActiveMean<=b.decodeMeasuredCap);
});
test('synthetic example provides a complete reproducible plan',()=>{
  const pts=parseDocument(JSON.parse(readFileSync(new URL('../docs/example.json',import.meta.url))));
  const r=planDeployment(pts,{...o,workload:'synthetic-context-128k',inputMetric:'non_cached_input_throughput',inputTokens:131072,nonCachedTokens:131072,outputTokens:1024,gpuBudget:128,ttftMs:3000,tpotMs:3.34,overheadMs:20});
  const b=r.candidates[0];assert.ok(b);assert.ok(b.totalGpus<=128);assert.ok(b.ttftMs<=3000);assert.ok(b.d.tpotMs<=3.34);
  assert.ok(JSON.parse(JSON.stringify(r)).candidates[0].p.point.source);
});

test('scaling the search does not require consuming GPUs and duplicate scaled ratios are omitted',()=>{
  const a=planDeployment([p,d],{...o,ttftMs:1000,gpuBudget:12}).candidates[0];
  const r=planDeployment([p,d],{...o,ttftMs:1000,gpuBudget:48});
  const b=r.candidates[0];
  assert.equal(b.nP,2);assert.equal(b.nD,1);assert.equal(b.totalGpus,12);
  close(a.outputTpsPerGpu,b.outputTpsPerGpu);
  assert.ok(!r.candidates.some(c=>c.nP===4&&c.nD===2));
  assert.equal(r.objective,'output_tokens_per_second_per_total_gpu');
});
