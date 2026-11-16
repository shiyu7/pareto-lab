"""Offline behavioral checks; all data and server responses here are test fixtures."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from collect_pareto_steady import Sampler
from collect_speed_bench import export_point, make_parser, run
from prepare_speed_bench import encode_row, prepare
from speed_bench_data import ENTROPIES, PromptPool, load_corpus, manifest_path


class Tokenizer:
    chat_template = 'test-only-template'

    def apply_chat_template(self, turns, **kwargs):
        return [7] + [ord(c) for c in turns[0]['content']] + [8]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class SpeedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'resolved.jsonl'
        rows = [dict(question_id=f'{cat}-{i}', category=cat, turns=[f'test fixture {cat} {i}'],
                     source='test-only', src_id=str(i)) for cat in ENTROPIES for i in range(4)]
        source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.path = self.root / 'throughput_1k.jsonl'
        self.prep_args = SimpleNamespace(input=source, output=self.path, config='throughput_1k',
            dataset_revision='test-only-revision', tokenizer=self.root, prompt_format='chat',
            context_limit=4096, max_output_tokens=1536)
        self.meta = prepare(self.prep_args, Tokenizer())

    def args(self, phase='decode'):
        argv = ['--dataset-jsonl', str(self.path), '--phase', phase,
                '--server-url', 'http://test-router', '--prefill-url', 'http://test-p',
                '--decode-url', 'http://test-d', '--model', 'test-only-model', '--scheme', 'test-only-scheme',
                '--cache-profile', 'test-only-cache', '--endpoint-gpus', '4',
                '--concurrency', '1', '--warmup-prompts', '1', '--archive-root', str(self.root / 'archive')]
        if phase == 'prefill':
            argv += ['--chunk-length', '1024', '--chunk-scope', 'dp-rank']
        return make_parser().parse_args(argv)

    def observation(self, args, concurrency=1, entropy='mixed'):
        windows = [dict(accepted=True, seconds=10, native_window_seconds=10, output_tokens=100,
                       eligible_interval_tokens=20, interval_seconds=2,
                       prefill_steps=10, prefill_step_ms=20, prefill_uncached_tokens=1000,
                       native_decode_batch_time_mean=.9, supply_limited=False),
                   dict(accepted=True, seconds=20, native_window_seconds=20, output_tokens=300,
                       eligible_interval_tokens=60, interval_seconds=3,
                       prefill_steps=30, prefill_step_ms=40, prefill_uncached_tokens=3000,
                       native_decode_batch_time_mean=.9, supply_limited=False)]
        return dict(complete=True, phase=args.phase, concurrency=concurrency, windows=windows,
                    workload_metadata=dict(dataset='nvidia/SPEED-Bench', entropy=entropy,
                        corpus_manifest=self.meta, corpus_selection=dict(repeated_requests=0)))

    def test_real_text_tokenization_and_unresolved_or_multiturn_rejected(self):
        rows, meta = load_corpus(self.path, 'low_entropy')
        self.assertEqual(rows[0]['input_ids'], [7] + list(map(ord, 'test fixture low_entropy 0')) + [8])
        self.assertEqual(meta['counts'], dict.fromkeys(ENTROPIES, 4))
        for turns in [['FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH'], ['one', 'two']]:
            with self.assertRaises(ValueError):
                encode_row(dict(question_id='bad', category='mixed', turns=turns), Tokenizer(), 'chat')

    def test_context_overflow_is_not_silently_truncated(self):
        self.prep_args.output = self.root / 'overflow.jsonl'
        self.prep_args.context_limit = 10
        with self.assertRaisesRegex(ValueError, 'context limit'):
            prepare(self.prep_args, Tokenizer())
        self.assertFalse(self.prep_args.output.exists())

    def test_hash_tampering_rejected(self):
        with self.path.open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_corpus(self.path, 'mixed')

    def test_holdout_replay_and_exhaustion(self):
        rows, _ = load_corpus(self.path, 'mixed')
        pool = PromptPool(rows, 7, warmup_prompts=1, replay_policy='cycle')
        warmup = pool.next(True)[0]['question_id']
        first = [pool.next() for _ in range(3)]
        self.assertNotIn(warmup, [r['question_id'] for r, epoch in first])
        row, epoch = pool.next()
        self.assertEqual((row['question_id'], epoch), (first[0][0]['question_id'], 1))
        self.assertEqual(pool.summary()['repeated_requests'], 1)
        pool = PromptPool(rows, 7, warmup_prompts=1, replay_policy='stop')
        for _ in range(3):
            pool.next()
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            pool.next()

    def test_weighted_metrics_and_entropy_grouping(self):
        args = self.args()
        d = export_point(self.observation(args), args, self.meta, 'mixed', 'test.json')
        self.assertEqual(d['x_value'], 16)  # 80 transitions / 5 request-seconds
        self.assertAlmostEqual(d['throughput'], 400 / 30)
        other = export_point(self.observation(args, entropy='high_entropy'), args, self.meta, 'high_entropy', 'test.json')
        self.assertNotEqual(d['workload'], other['workload'])
        args = self.args('prefill')
        p = export_point(self.observation(args), args, self.meta, 'mixed', 'test.json')
        self.assertEqual(p['x_value'], 35)
        self.assertEqual(p['sweep_parameter'], 'chunk_size_tokens')
        bad = self.observation(args)
        bad['complete'] = False
        with self.assertRaises(ValueError):
            export_point(bad, args, self.meta, 'mixed', 'test.json')
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            export_point(self.observation(args), args, self.meta, 'low_entropy', 'test.json')

    def test_dry_run_has_no_server_calls_or_archive(self):
        args = self.args()
        args.dry_run = True
        with patch('collect_speed_bench.subprocess.run', side_effect=AssertionError('must not send requests')):
            run(args)
        self.assertFalse(args.archive_root.exists())

    def test_archive_and_partial_failure_preserved(self):
        args = self.args()
        args.concurrency = [1, 2]
        def child(cmd, **kwargs):
            concurrency = int(cmd[cmd.index('--concurrency') + 1])
            path = Path(cmd[cmd.index('--output') + 1])
            if concurrency == 1:
                entropy = cmd[cmd.index('--entropy') + 1]
                path.write_text(json.dumps(self.observation(args, concurrency, entropy)))
            return SimpleNamespace(returncode=0 if concurrency == 1 else 1)
        with patch('collect_speed_bench.subprocess.run', side_effect=child):
            with self.assertRaisesRegex(ValueError, 'partial'):
                run(args)
        archive = next(args.archive_root.iterdir())
        manifest = json.loads((archive / 'manifest.json').read_text())
        self.assertFalse(manifest['complete'])
        self.assertTrue((archive / 'corpora' / self.path.name).exists())
        points = json.loads(next(archive.glob('*/points.json')).read_text())
        self.assertEqual(len(points['points']), 1)
        self.assertFalse(points['collection']['complete'])

    def test_sampler_sends_exact_corpus_ids_and_accepts_natural_eos(self):
        args = SimpleNamespace(dataset_jsonl=self.path, entropy='mixed', output_len=1536,
            warmup_prompts=1, replay_policy='cycle', seed=7, input_len=8192, synthetic=False,
            output=self.root / 'observations.json', phase='decode', run_id='test', server_url='http://test',
            temperature=0, top_p=1, eos_policy='natural', cache_profile='test')
        sampler = Sampler(args)
        sent = {}
        class Content:
            def __aiter__(self):
                async def lines():
                    for n in (1, 3):
                        obj = dict(meta_info=dict(completion_tokens=n, prompt_tokens=len(sent['input_ids']),
                                                  cached_tokens=0, num_retractions=0))
                        yield b'data: ' + json.dumps(obj).encode() + b'\n'
                return lines()
        class Response:
            content = Content()
            def raise_for_status(self):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *unused):
                pass
        class Session:
            def post(self, url, json):
                sent.update(json)
                return Response()
        sampler.session = Session()
        try:
            asyncio.run(sampler.request(0))
        finally:
            sampler.raw.close()
        expected = sampler.pool.measured[0]['input_ids']
        self.assertEqual(sent['input_ids'], expected)
        self.assertFalse(sent['sampling_params']['ignore_eos'])
        self.assertEqual(sampler.requests[0]['output_tokens'], 3)
        self.assertEqual(sum(e['d_tokens'] for e in sampler.events), 2)


if __name__ == '__main__':
    unittest.main()
