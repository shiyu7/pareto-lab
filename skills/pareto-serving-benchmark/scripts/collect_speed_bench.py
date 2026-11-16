#!/usr/bin/env python3
"""Run a SPEED-Bench length x entropy x global-concurrency steady sweep.

Run in the chosen client (locally, a Kubernetes pod, or an SSH host). Transport
and service/chunk changes are orchestration tasks. --dry-run sends no requests.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from collect_pareto_steady import save
from speed_bench_data import ENTROPIES, load_corpus, manifest_path, sha256
from urllib.parse import urlsplit


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return n


def numbers(text):
    values = [positive_int(x) for x in text.split(',')]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('expected distinct positive integers')
    return values


def export_point(observation, args, corpus, entropy, source):
    observed = observation.get('workload_metadata', {})
    if (observation.get('phase') != args.phase or observed.get('dataset') != 'nvidia/SPEED-Bench'
            or observed.get('entropy') != entropy
            or observed.get('corpus_manifest', {}).get('sha256') != corpus['sha256']):
        raise ValueError('observation phase/corpus/entropy provenance mismatch')
    windows = observation.get('windows', [])
    if not observation.get('complete') or not windows or not all(w.get('accepted') for w in windows):
        raise ValueError('no complete, accepted steady observations; no viewer point exported')
    seconds = sum(w['seconds'] for w in windows)
    native_seconds = sum(w['native_window_seconds'] for w in windows)
    if args.phase == 'decode':
        transitions = sum(w['eligible_interval_tokens'] for w in windows)
        exposure = sum(w['interval_seconds'] for w in windows)
        if transitions <= 0 or exposure <= 0:
            raise ValueError('missing eligible decode token transitions')
        x = transitions / exposure
        y = sum(w['output_tokens'] for w in windows) / seconds
        x_metric, y_metric = 'per_user_tps', 'output_throughput'
    else:
        steps = sum(w['prefill_steps'] for w in windows)
        if steps <= 0:
            raise ValueError('missing Prefill step observations')
        x = sum(w['prefill_step_ms'] * w['prefill_steps'] for w in windows) / steps
        y = sum(w['prefill_uncached_tokens'] for w in windows) / native_seconds
        x_metric, y_metric = 'prefill_latency_ms', 'non_cached_input_throughput'
    if not all(math.isfinite(v) and v > 0 for v in (x, y)):
        raise ValueError('invalid point coordinates')
    counts = corpus['counts']
    condition = dict(model=args.model, dataset=corpus['dataset'], config=corpus['config'],
                     revision=corpus['declared_dataset_revision'], corpus_sha256=corpus['sha256'],
                     entropy=entropy, sampling=dict(temperature=args.temperature, top_p=args.top_p),
                     eos_policy=args.eos_policy, output_cap=1 if args.phase == 'prefill' else args.output_len,
                     cache_profile=args.cache_profile, replay_policy=args.replay_policy, seed=args.seed,
                     protocol='speed-bench-real-e2e-steady-v1')
    workload = (f"{args.model} | SPEED-Bench {corpus['config']} {entropy} | "
                f"corpus={corpus['sha256'][:12]} | {args.cache_profile} | "
                f"{args.eos_policy} max-out={condition['output_cap']} | "
                f"T={args.temperature} top-p={args.top_p} seed={args.seed} | replay={args.replay_policy} | steady-v1")
    measured_gpus = args.phase_gpus if args.gpu_scope == 'phase' else args.endpoint_gpus
    point = dict(phase=args.phase, scheme=args.scheme, workload=workload, source=source,
                 x_metric=x_metric, x_value=x, throughput=y, throughput_metric=y_metric,
                 throughput_unit='tokens/s', num_gpus=measured_gpus,
                 concurrency=observation['concurrency'],
                 measurement=dict(mode='speed-bench-real-e2e-steady-v1',
                     gpu_scope=args.gpu_scope, endpoint_num_gpus=args.endpoint_gpus,
                     phase_num_gpus=args.phase_gpus,
                     actual_decode_batch_mean=sum(w['native_decode_batch_time_mean'] * w['native_window_seconds']
                                                  for w in windows) / native_seconds,
                     supply_limited=any(w.get('supply_limited') for w in windows),
                     decode_token_scope='decode-only-excludes-first-token',
                     latency_statistic='window-token-weighted-burst-granular' if args.phase == 'decode'
                                        else 'scheduler-completion-busy-time-per-representative-forward',
                     corpus_samples=counts[entropy], actual_input_tokens=corpus['actual_input_tokens'][entropy],
                     workload_conditions=condition, windows=windows,
                     corpus_selection=observation['workload_metadata']['corpus_selection']))
    if args.phase == 'prefill':
        point.update(chunked_prefill=True, chunk_size_tokens=args.chunk_length,
                     chunk_scope=args.chunk_scope, sweep_parameter='chunk_size_tokens')
    return point


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-jsonl', type=Path, action='append', required=True,
                   help='repeat for each prepared nominal input-length config')
    p.add_argument('--entropies', default=','.join(ENTROPIES))
    p.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    for name in ('server-url', 'prefill-url', 'decode-url', 'model', 'scheme', 'cache-profile'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--endpoint-gpus', type=positive_int, required=True)
    p.add_argument('--gpu-scope', choices=('endpoint', 'phase'), default='endpoint')
    p.add_argument('--phase-gpus', type=positive_int)
    p.add_argument('--supply-evidence', type=Path, help='recorded evidence of phase isolation/sufficient other-phase supply')
    p.add_argument('--concurrency', type=numbers, default=None)
    p.add_argument('--chunk-length', type=positive_int, help='effective server value; does not configure it')
    p.add_argument('--chunk-scope', choices=('request', 'dp-rank'))
    p.add_argument('--output-len', type=positive_int, default=1536)
    p.add_argument('--eos-policy', choices=('natural', 'fixed'), default='natural')
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--top-p', type=float, default=1)
    p.add_argument('--seed', type=int, default=82611)
    p.add_argument('--replay-policy', choices=('cycle', 'stop'), default='cycle')
    p.add_argument('--warmup-prompts', type=positive_int, default=16)
    p.add_argument('--warmup-concurrency', type=positive_int, default=1)
    p.add_argument('--windows', type=positive_int, default=2)
    p.add_argument('--window-seconds', type=float, default=30)
    p.add_argument('--max-seconds', type=float, default=180)
    p.add_argument('--request-timeout', type=float, default=120)
    p.add_argument('--target-batch-fraction', type=float, help='optional active global D-batch target / client concurrency')
    p.add_argument('--archive-root', type=Path, default=Path('pareto-archives'))
    p.add_argument('--dry-run', action='store_true')
    return p


def run(args):
    for key in ('server_url', 'prefill_url', 'decode_url'):
        url = urlsplit(getattr(args, key))
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment
                or url.path.rstrip('/').endswith(('/generate', '/v1', '/completions'))):
            raise ValueError('expected HTTP(S) server root without credentials/query: ' + key)
        setattr(args, key, getattr(args, key).rstrip('/'))
    if any(not getattr(args, key).strip() for key in ('model', 'scheme', 'cache_profile')):
        raise ValueError('model, scheme and cache profile must not be blank')
    if args.concurrency is None:
        if args.phase == 'prefill':
            raise ValueError('Prefill requires an explicit bounded client --concurrency policy')
        args.concurrency = numbers('1,2,4,8,16,32,64,128,192,256')
    entropies = args.entropies.split(',')
    if not entropies or len(set(entropies)) != len(entropies) or any(e not in ENTROPIES for e in entropies):
        raise ValueError('invalid or duplicate entropy selection')
    if args.phase == 'prefill' and (not args.chunk_length or not args.chunk_scope):
        raise ValueError('Prefill requires verified --chunk-length and --chunk-scope metadata')
    if args.phase == 'decode' and (args.chunk_length or args.chunk_scope):
        raise ValueError('chunk metadata belongs to Prefill observations only')
    if args.gpu_scope == 'phase' and (not args.phase_gpus or not args.supply_evidence or not args.supply_evidence.is_file()):
        raise ValueError('phase GPU normalization requires phase GPU count and archived supply evidence')
    if args.phase_gpus and args.phase_gpus > args.endpoint_gpus:
        raise ValueError('phase GPUs cannot exceed endpoint GPUs')
    if (not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1
            or not all(math.isfinite(v) and v > 0 for v in
                       (args.window_seconds, args.max_seconds, args.request_timeout))
            or args.max_seconds <= 18 + args.windows * args.window_seconds
            or args.request_timeout <= 0 or args.output_len < 2
            or args.warmup_concurrency > min(args.concurrency)
            or (args.target_batch_fraction is not None and not 0 < args.target_batch_fraction <= 1)):
        raise ValueError('invalid timing, sampling, output, warmup or batch-target settings')
    corpora, configs = [], set()
    for path in args.dataset_jsonl:
        # Hash/validate each corpus once; retain no more than one entropy's token lists in memory.
        rows, meta = load_corpus(path, entropies[0])
        if meta['config'] in configs or args.warmup_prompts >= min(meta['counts'][e] for e in entropies):
            raise ValueError('duplicate config or insufficient prompts after warmup holdout')
        if args.output_len > meta['output_reserve']:
            raise ValueError('output length exceeds corpus context reserve')
        configs.add(meta['config'])
        corpora.append((path, meta))
        del rows
    plan = dict(protocol='speed-bench-real-e2e-steady-v1', phase=args.phase,
                configs=sorted(configs), entropies=entropies, global_concurrency=args.concurrency,
                point_count=len(corpora) * len(entropies) * len(args.concurrency),
                minimum_measured_seconds=len(corpora) * len(entropies) * len(args.concurrency)
                                         * args.windows * args.window_seconds,
                cache_profile=args.cache_profile, replay_policy=args.replay_policy,
                note='settling, warmup and drain add time; this helper never changes service configuration')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return None
    os.umask(0o077)
    root = args.archive_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-speed-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, mode=0o700)
    (root / 'sources').mkdir()
    (root / 'corpora').mkdir()
    for name in ('collect_speed_bench.py', 'collect_pareto_steady.py', 'speed_bench_data.py', 'prepare_speed_bench.py'):
        shutil.copy2(Path(__file__).with_name(name), root / 'sources' / name)
    manifest = dict(plan=plan, complete=False, results=[], source_hashes={p.name: sha256(p) for p in (root / 'sources').iterdir()},
                    settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                              if k != 'dataset_jsonl'})
    if args.supply_evidence:
        shutil.copy2(args.supply_evidence, root / 'supply-evidence.txt')
    save(root / 'manifest.json', manifest)
    print(f'Archive: {root}', flush=True)
    try:
        for path, meta in corpora:
            local = root / 'corpora' / (meta['config'] + '.jsonl')
            shutil.copy2(path, local)
            shutil.copy2(manifest_path(path), manifest_path(local))
            for entropy in entropies:
                cell = root / (meta['config'] + '-' + entropy + '-' + args.phase)
                cell.mkdir()
                doc = dict(schema='hpc-pareto-points', version=2, points=[], collection=dict(complete=False))
                save(cell / 'points.json', doc)
                for concurrency in args.concurrency:
                    output = cell / f'c{concurrency}.observations.json'
                    cmd = [sys.executable, str(root / 'sources' / 'collect_pareto_steady.py'),
                           '--dataset-jsonl', str(local.resolve()), '--entropy', entropy,
                           '--phase', args.phase, '--run-id', root.name + '-' + cell.name + f'-c{concurrency}',
                           '--concurrency', str(concurrency), '--output', str(output.resolve())]
                    for key in ('server_url', 'prefill_url', 'decode_url', 'output_len', 'eos_policy', 'cache_profile',
                                'temperature', 'top_p', 'seed', 'replay_policy', 'warmup_prompts',
                                'warmup_concurrency', 'windows', 'window_seconds', 'max_seconds', 'request_timeout'):
                        cmd += ['--' + key.replace('_', '-'), str(getattr(args, key))]
                    if args.chunk_length:
                        cmd += ['--chunk-length', str(args.chunk_length)]
                    if args.target_batch_fraction is not None:
                        cmd += ['--target-decode-batch', str(concurrency * args.target_batch_fraction)]
                    result = dict(config=meta['config'], entropy=entropy, concurrency=concurrency,
                                  command=cmd, complete=False, observation=str(output.relative_to(root)))
                    manifest['results'].append(result)
                    save(root / 'manifest.json', manifest)
                    with output.with_suffix('.driver.log').open('x') as log:
                        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                 timeout=args.max_seconds + 3 * (args.request_timeout + 60))
                    result['exit_code'] = process.returncode
                    if process.returncode:
                        raise ValueError(f'{cell.name} c{concurrency} failed; raw/partial data preserved')
                    observation = json.loads(output.read_text())
                    doc['points'].append(export_point(observation, args, meta, entropy, output.name))
                    result['complete'] = True
                    save(cell / 'points.json', doc)
                    save(root / 'manifest.json', manifest)
                doc['collection']['complete'] = True
                save(cell / 'points.json', doc)
        manifest['complete'] = True
        return root
    except BaseException as exc:
        manifest['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        save(root / 'manifest.json', manifest)


def main():
    p = make_parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        p.exit(1, f'collection failed: {exc}\n')


if __name__ == '__main__':
    main()
