#!/usr/bin/env python3
"""Real SGLang E2E collection using bounded, time-resolved steady windows.

Run in a client environment containing aiohttp. Does not modify the server.
Native /v1/loads counters are required; no fake KV or speculative acceptance.
SPEED-Bench tokenized JSONL is the default input; synthetic input is explicit-only.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

from speed_bench_data import ENTROPIES, PromptPool, load_corpus


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def prom_values(raw):
    result = {}
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.match(r'([^ {]+)(?:\{(.*)\})?\s+([^ ]+)', line)
        if not match:
            continue
        name, labels, value = match.groups()
        tags = dict(re.findall(r'(\w+)="([^"]*)"', labels or ''))
        if tags.get('tp_rank') != '0' or tags.get('pp_rank') != '0':
            continue
        if name == 'sglang:cuda_graph_passes_total' and tags.get('mode', '').startswith('prefill'):
            result['steps'] = result.get('steps', 0) + float(value)
        elif name in ('sglang:per_stage_req_latency_seconds_sum', 'sglang:per_stage_req_latency_seconds_count'):
            result[tags.get('stage', '') + '_' + name.rsplit('_', 1)[-1]] = float(value)
    return result


def window_metrics(events, requests, snapshots, start, end, phase):
    """Join exact client event windows with paired native counter snapshots."""
    duration = end - start
    if duration <= 0:
        raise ValueError('nonpositive window')
    selected = [e for e in events if start <= e['t'] < end]
    generated = sum(e['d_tokens'] for e in selected)
    intervals = [e for e in selected if e['previous_t'] is not None and e['previous_t'] >= start]
    interval_tokens = sum(e['delta_tokens'] for e in intervals)
    interval_seconds = sum(e['t'] - e['previous_t'] for e in intervals)
    exposure = sum(max(0, min(end, r.get('end', end)) - max(start, r['first']))
                   for r in requests if 'first' in r)
    complete = [r for r in requests if r['start'] >= start and r.get('end', math.inf) <= end]
    # Native counters use the actual paired sample timestamps, without mixing clocks.
    a, b = min(snapshots, key=lambda s: abs(s['t'] - start)), min(snapshots, key=lambda s: abs(s['t'] - end))
    native_t = b['t'] - a['t']
    if native_t <= 0:
        raise ValueError('missing native window endpoints')
    before = {x.get('dp_rank', i): x for i, x in enumerate(a['p']['loads'])}
    after = {x.get('dp_rank', i): x for i, x in enumerate(b['p']['loads'])}
    if before.keys() != after.keys():
        raise ValueError('Prefill rank membership changed')
    p_tokens = p_busy = 0
    for rank, last in after.items():
        old = before[rank]
        dt = last['total_prefill_uncached_tokens'] - old['total_prefill_uncached_tokens']
        db = last['total_prefill_busy_us'] - old['total_prefill_busy_us']
        if min(dt, db) < 0:
            raise ValueError('Prefill counter reset inside window')
        p_tokens += dt
        p_busy += db
    p_steps = b['pm'].get('steps', 0) - a['pm'].get('steps', 0)
    if min(p_tokens, p_busy, p_steps) < 0:
        raise ValueError('Prefill counter reset inside window')
    rank_before = {x['dp_rank']: x for x in a['d']['loads']}
    if set(rank_before) != {x['dp_rank'] for x in b['d']['loads']}:
        raise ValueError('Decode rank membership changed')
    native_exposure, native_steps, native_generated = 0., 0., 0.
    rank_stats = []
    for x in b['d']['loads']:
        old = rank_before.get(x['dp_rank'])
        if old is None:
            raise ValueError('Decode rank membership changed')
        first, last = old.get('decode_moments'), x.get('decode_moments')
        if not isinstance(first, list) or not isinstance(last, list) or len(first) != 6 or len(last) != 6:
            raise ValueError('missing native decode moments')
        delta = [v - u for u, v in zip(first, last)]
        if any(v < 0 for v in delta):
            raise ValueError('Decode counter reset inside window')
        native_exposure += delta[4] / 1e6
        native_steps += delta[0]
        native_generated += delta[5]
        rank_stats.append(dict(dp_rank=x['dp_rank'], steps=delta[0],
                               active_batch_step_mean=delta[1] / delta[0] if delta[0] else 0,
                               active_batch_second_moment=delta[3] / delta[0] if delta[0] else 0,
                               request_seconds=delta[4] / 1e6, generated_tokens=delta[5]))
    result = dict(start=start, end=end, seconds=duration, completed_requests=len(complete),
                  output_tokens=generated, output_tps=generated / duration,
                  stream_active_batch_mean=exposure / duration,
                  native_decode_batch_time_mean=native_exposure / native_t,
                  native_decode_steps=native_steps, native_generated_tokens=native_generated,
                  native_rank_stats=rank_stats,
                  tpot_ms=1000 * interval_seconds / interval_tokens if interval_tokens else None,
                  eligible_interval_tokens=interval_tokens, interval_seconds=interval_seconds,
                  ttft_ms=statistics.mean(r['first'] - r['start'] for r in complete) * 1000 if complete else None,
                  non_cached_input_tps=p_tokens / native_t, prefill_uncached_tokens=p_tokens,
                  prefill_steps=p_steps,
                  prefill_step_ms=p_busy / p_steps / 1000 if p_steps else None,
                  prefill_actual_tokens_per_step=p_tokens / p_steps if p_steps else None,
                  native_window_seconds=native_t,
                  sampled_decode_batch_mean=statistics.mean(sum(x['num_running_reqs'] for x in s['d']['loads'])
                      for s in snapshots if start <= s['t'] <= end),
                  timing_note='P step is scheduler completion busy-time / representative forward-count; PP pipeline, not GPU kernel time. D stream timing is burst-granular, counts actual token multiplicity.')
    for stage in ('prefill_forward', 'chunked_prefill'):
        count = b['pm'].get(stage + '_count', 0) - a['pm'].get(stage + '_count', 0)
        seconds = b['pm'].get(stage + '_sum', 0) - a['pm'].get(stage + '_sum', 0)
        result[stage + '_ms'] = 1000 * seconds / count if count > 0 else None
    return result


class Sampler:
    def __init__(self, args):
        self.args = args
        self.events, self.requests, self.snapshots = [], [], []
        self.windows = []
        self.stop = False
        self.error = None
        self.sequence = 0
        self.origin = time.monotonic()
        self.rng = random.Random(args.seed)
        self.pool, self.corpus_manifest = None, None
        if args.dataset_jsonl:
            rows, self.corpus_manifest = load_corpus(args.dataset_jsonl, args.entropy)
            if args.output_len > self.corpus_manifest['output_reserve']:
                raise ValueError('output length exceeds prepared corpus context reserve')
            self.pool = PromptPool(rows, args.seed, args.warmup_prompts, args.replay_policy)
        self.prefix = ([self.rng.randrange(1000, 100000) for _ in range(args.input_len - 16)]
                       if args.synthetic else None)
        self.raw = args.output.with_suffix('.events.jsonl').open('x')

    def now(self):
        return time.monotonic() - self.origin

    def log(self, kind, record):
        self.raw.write(json.dumps(dict(kind=kind, **record), ensure_ascii=False) + '\n')

    async def get_json(self, url):
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def snapshot(self):
        p, d = await asyncio.gather(self.get_json(self.args.prefill_url + '/v1/loads?include=all'),
                                    self.get_json(self.args.decode_url + '/v1/loads?include=all'))
        pm = {}
        if self.args.phase == 'prefill':
            async with self.session.get(self.args.prefill_url + '/metrics') as response:
                response.raise_for_status()
                pm = prom_values(await response.text())
        s = dict(t=self.now(), p=p, d=d, pm=pm)
        self.snapshots.append(s)
        self.log('snapshot', s)
        return s

    async def request(self, worker, warmup=False):
        self.sequence += 1
        rid = f'{self.args.run_id}-{self.sequence}'
        if self.pool:
            row, epoch = self.pool.next(warmup)
            ids = row['input_ids']
            provenance = dict(question_id=row['question_id'], category=row['category'], replay_epoch=epoch)
        else:
            ids = ([self.rng.randrange(1000, 100000) for _ in range(self.args.input_len)]
                   if self.args.phase == 'prefill' else self.prefix + [self.rng.randrange(1000, 100000) for _ in range(16)])
            provenance = dict(dataset='legacy-synthetic')
        record = dict(id=rid, worker=worker, start=self.now(), warmup=warmup,
                      input_tokens=len(ids), **provenance)
        self.requests.append(record)
        output_len = 1 if self.args.phase == 'prefill' or warmup else self.args.output_len
        payload = dict(input_ids=ids, rid=rid, sampling_params=dict(temperature=self.args.temperature,
                       top_p=self.args.top_p, max_new_tokens=output_len,
                       ignore_eos=self.args.eos_policy == 'fixed'), stream=True, return_output_ids=True)
        previous_t, previous_count = None, 0
        async with self.session.post(self.args.server_url + '/generate', json=payload) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    break
                obj = json.loads(data)
                if 'error' in obj:
                    raise ValueError(str(obj['error']))
                meta = obj.get('meta_info', {})
                current = meta.get('completion_tokens')
                if not isinstance(current, int) or current < previous_count:
                    raise ValueError('invalid cumulative output token count')
                if current == previous_count:
                    continue
                now = self.now()
                record.setdefault('first', now)
                event = dict(id=rid, t=now, previous_t=previous_t, delta_tokens=current - previous_count,
                             d_tokens=max(0, current - 1) - max(0, previous_count - 1),
                             completion_tokens=current, dp_rank=meta.get('dp_rank'))
                if not warmup:
                    self.events.append(event)
                self.log('token', event)
                previous_t, previous_count = now, current
                record['meta'] = {k: meta.get(k) for k in ('prompt_tokens', 'cached_tokens', 'num_retractions',
                   'dp_rank', 'spec_accept_length', 'spec_verify_ct', 'queue_time', 'e2e_latency')}
        record['end'], record['output_tokens'] = self.now(), previous_count
        self.log('request', record)
        if not 1 <= previous_count <= output_len or (self.args.eos_policy == 'fixed' and previous_count != output_len):
            raise ValueError(f'{rid}: invalid output count {previous_count} for cap {output_len}')
        if record['meta'].get('prompt_tokens') != len(ids):
            raise ValueError('wrong prompt length')
        if record['meta'].get('num_retractions', 0):
            raise ValueError('request retraction observed')

    def workload_metadata(self):
        return dict(dataset='nvidia/SPEED-Bench' if self.pool else 'legacy-synthetic',
                    corpus_manifest=self.corpus_manifest,
                    corpus_selection=self.pool.summary() if self.pool else None,
                    entropy=self.args.entropy if self.pool else None,
                    input_length_note='actual target-tokenizer lengths; bucket is nominal',
                    eos_policy=self.args.eos_policy, output_cap=self.args.output_len,
                    temperature=self.args.temperature, top_p=self.args.top_p,
                    cache_profile=self.args.cache_profile)

    async def worker(self, index):
        try:
            while not self.stop:
                await self.request(index)
        except Exception as exc:
            self.error = str(exc)
            self.stop = True

    async def run(self):
        try:
            await self._run()
        except BaseException as exc:
            self.error = self.error or str(exc) or type(exc).__name__
            if not self.raw.closed:
                self.raw.flush()
                self.raw.close()
            if not self.args.output.exists():
                save(self.args.output, dict(schema='pareto-steady-observations', version=1,
                    phase=self.args.phase, concurrency=self.args.concurrency,
                    input_len=None if self.pool else self.args.input_len,
                    output_len=1 if self.args.phase == 'prefill' else self.args.output_len,
                    windows=self.windows, complete=False, error=self.error, requests=self.requests,
                    workload_metadata=self.workload_metadata(),
                    raw_events=self.args.output.with_suffix('.events.jsonl').name))
            raise

    async def _run(self):
        import aiohttp
        connector = aiohttp.TCPConnector(limit=self.args.concurrency + 20)
        timeout = aiohttp.ClientTimeout(total=self.args.request_timeout + 60, sock_read=self.args.request_timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as self.session:
            warmups = [asyncio.create_task(self.request(-1-i, warmup=True)) for i in range(self.args.warmup_concurrency)]
            try:
                if len(warmups) > 1:
                    while not all(task.done() for task in warmups):
                        await asyncio.sleep(self.args.sample_interval)
                        await self.snapshot()
                await asyncio.gather(*warmups)
            except Exception as exc:
                await asyncio.gather(*warmups, return_exceptions=True)
                self.raw.flush()
                self.raw.close()
                save(self.args.output, dict(schema='pareto-steady-observations',version=1,
                    phase=self.args.phase,concurrency=self.args.concurrency,input_len=None if self.pool else self.args.input_len,
                    output_len=1 if self.args.phase=='prefill' else self.args.output_len,
                    chunk_length=self.args.chunk_length,seed=self.args.seed,windows=[],complete=False,
                    error='shape warmup: '+str(exc),requests=self.requests,
                    request_timeout=self.args.request_timeout,warmup_concurrency=self.args.warmup_concurrency,
                    workload_metadata=self.workload_metadata(),
                    raw_events=self.args.output.with_suffix('.events.jsonl').name))
                raise
            print(f'WARMUP complete requests={len(warmups)} seconds={self.now():.3f}',flush=True)
            self.requests.clear()
            self.snapshots.clear()
            await self.snapshot()
            workers = [asyncio.create_task(self.worker(i)) for i in range(self.args.concurrency)]
            windows, start = self.windows, None
            initial = self.now()
            try:
                while self.now() - initial < self.args.max_seconds:
                    await asyncio.sleep(self.args.sample_interval)
                    snapshot = await self.snapshot()
                    now = snapshot['t']
                    if self.error:
                        raise ValueError(self.error)
                    if start is None and now - initial >= 18:
                        bins = [window_metrics(self.events, self.requests, self.snapshots,
                                              now - (3 - i) * 5, now - (2 - i) * 5, self.args.phase)
                                for i in range(3)]
                        rates = [b['non_cached_input_tps'] if self.args.phase == 'prefill' else b['output_tps'] for b in bins]
                        mean = statistics.mean(rates)
                        drift = statistics.pstdev(rates) / mean if mean > 0 else math.inf
                        target = self.args.target_decode_batch
                        occupancy_ok = self.args.phase == 'prefill' or all(
                            b['native_decode_batch_time_mean'] >= (target * .85 if target else 0.01) for b in bins)
                        batches = [b['native_decode_batch_time_mean'] for b in bins]
                        batch_mean = statistics.mean(batches)
                        batch_drift = statistics.pstdev(batches) / batch_mean if batch_mean > 0 else math.inf
                        if occupancy_ok and drift <= .20 and min(rates) > 0 and (
                                self.args.phase == 'prefill' or batch_drift <= .20):
                            start = now
                            print(f'MEASURE start={start:.3f} phase={self.args.phase} C={self.args.concurrency} drift={drift:.3f}', flush=True)
                    if start is not None and now - start >= self.args.window_seconds:
                        window = window_metrics(self.events, self.requests, self.snapshots, start, now, self.args.phase)
                        window['accepted'] = (self.args.phase == 'prefill' and window['prefill_steps'] >= 10
                            and window['non_cached_input_tps'] > 0 and window['prefill_step_ms'] is not None
                            or self.args.phase == 'decode' and window['native_decode_batch_time_mean'] >= (self.args.target_decode_batch * .85 if self.args.target_decode_batch else 0.01)
                            and window['native_decode_steps'] >= 20 and window['tpot_ms'] is not None)
                        window['target_decode_batch'] = self.args.target_decode_batch
                        window['supply_limited'] = (self.args.phase == 'decode' and
                            window['native_decode_batch_time_mean'] < self.args.concurrency * .85)
                        windows.append(window)
                        print(f'WINDOW {len(windows)} accepted={window["accepted"]} P_TPS={window["non_cached_input_tps"]:.2f} D_TPS={window["output_tps"]:.2f} B={window["native_decode_batch_time_mean"]:.2f}', flush=True)
                        if len(windows) == self.args.windows:
                            break
                        start = now
                if len(windows) != self.args.windows:
                    raise ValueError('no adequate steady plateau within bounded duration')
            except BaseException as exc:
                self.error = self.error or str(exc) or type(exc).__name__
                raise
            finally:
                self.stop = True
                await asyncio.gather(*workers, return_exceptions=True)
                try:
                    await self.snapshot()
                except Exception as exc:
                    self.error = self.error or ('final native snapshot: ' + str(exc))
                self.raw.flush()
                self.raw.close()
                save(self.args.output, dict(schema='pareto-steady-observations', version=1,
                    phase=self.args.phase, concurrency=self.args.concurrency, input_len=None if self.pool else self.args.input_len,
                    output_len=1 if self.args.phase == 'prefill' else self.args.output_len,
                    chunk_length=self.args.chunk_length, seed=self.args.seed, windows=windows,
                    request_timeout=self.args.request_timeout,warmup_concurrency=self.args.warmup_concurrency,
                    complete=len(windows) == self.args.windows and all(w['accepted'] for w in windows) and not self.error,
                    error=self.error, requests=self.requests, workload_metadata=self.workload_metadata(),
                    raw_events=self.args.output.with_suffix('.events.jsonl').name))
            if self.error or not all(w['accepted'] for w in windows):
                raise ValueError(self.error or 'measurement window did not satisfy plateau criteria')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('server-url', 'prefill-url', 'decode-url', 'phase', 'run-id'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--concurrency', type=int, required=True, help='global client cap, not per-DP-rank batch')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset-jsonl', type=Path, help='prepared SPEED-Bench input and adjacent manifest')
    inputs.add_argument('--synthetic', action='store_true', help='explicit legacy random-token diagnostic only')
    parser.add_argument('--entropy', choices=ENTROPIES)
    parser.add_argument('--warmup-prompts', type=int, default=16)
    parser.add_argument('--replay-policy', choices=('cycle', 'stop'), default='cycle')
    parser.add_argument('--cache-profile', required=True, help='actual fixed cache configuration/condition')
    parser.add_argument('--eos-policy', choices=('natural', 'fixed'), default='natural')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top-p', type=float, default=1)
    parser.add_argument('--target-decode-batch', type=float, help='optional observed global active-batch target')
    parser.add_argument('--input-len', type=int, default=8192)
    parser.add_argument('--output-len', type=int, default=1536)
    parser.add_argument('--chunk-length', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=82611)
    parser.add_argument('--sample-interval', type=float, default=.5)
    parser.add_argument('--window-seconds', type=float, default=30)
    parser.add_argument('--windows', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=180)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--warmup-concurrency', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.dataset_jsonl) != bool(args.entropy):
        parser.error('--dataset-jsonl requires --entropy; synthetic mode has no entropy category')
    if args.synthetic and args.target_decode_batch is None:
        args.target_decode_batch = args.concurrency
    if (not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1
            or args.window_seconds <= 0 or args.windows <= 0 or args.sample_interval <= 0
            or args.max_seconds <= 18 + args.window_seconds * args.windows
            or (args.target_decode_batch is not None and (not math.isfinite(args.target_decode_batch)
                or not 0 < args.target_decode_batch <= args.concurrency))):
        parser.error('invalid sampling, timing or active-batch settings')
    if args.phase not in ('prefill', 'decode') or args.concurrency <= 0 or args.input_len < 16 or args.output_len < 2:
        parser.error('invalid phase or workload')
    if args.request_timeout <= 0 or not 1 <= args.warmup_concurrency <= args.concurrency:
        parser.error('invalid request timeout or warmup concurrency')
    if args.output.exists() or args.output.with_suffix('.events.jsonl').exists():
        parser.error('output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(Sampler(args).run())


if __name__ == '__main__':
    main()
