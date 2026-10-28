import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import collect_pareto as collector


# A subprocess-level fake of the documented benchmark CLI and JSON output.
# It launches no server and generates no GPU performance claims.
FAKE_BENCH = r'''
import argparse, json, os
p = argparse.ArgumentParser()
for name in ['backend', 'base-url', 'model', 'dataset-name', 'random-input-len',
             'random-output-len', 'random-range-ratio', 'num-prompts', 'request-rate',
             'max-concurrency', 'warmup-requests', 'seed', 'output-file', 'tokenizer']:
    p.add_argument('--' + name)
p.add_argument('--tokenize-prompt', action='store_true')
p.add_argument('--flush-cache', action='store_true')
a = p.parse_args()
n, inp, out, c = map(int, [a.num_prompts, a.random_input_len, a.random_output_len, a.max_concurrency])
if os.environ.get('FAKE_FAIL') == '1' and out > 1 and c == 4:
    n -= 1
row = dict(completed=n, duration=2, total_input_tokens=n*inp, total_output_tokens=n*out,
           input_throughput=n*inp/2, output_throughput=n*out/2, mean_ttft_ms=20,
           mean_tpot_ms=5 if out > 1 else 0, random_input_len=inp, random_output_len=out,
           max_concurrency=c, concurrency=c/2, p99_ttft_ms=40,
           server_info={'must_not_export': 'private-server-config'})
with open(a.output_file, 'w') as f:
    f.write(json.dumps(row)+'\n')
'''


class CollectorTest(unittest.TestCase):
    def args(self, *extras):
        return collector.parser().parse_args(['--model', 'test-model', '--num-gpus', '8',
                                              '--concurrency', '1,4', '--input-len', '128',
                                              '--output-len', '16', *extras])

    def row(self, job, **changes):
        n = job['requests']
        return dict(completed=n, duration=2, total_input_tokens=n*128,
                    total_output_tokens=n*job['output_len'], input_throughput=n*64,
                    output_throughput=n*job['output_len']/2, mean_ttft_ms=20, mean_tpot_ms=5,
                    max_concurrency=job['concurrency'], random_input_len=128,
                    random_output_len=job['output_len'], **changes)

    def fake_environment(self, root):
        package = root / 'sglang/benchmark'
        package.mkdir(parents=True)
        (root / 'sglang/__init__.py').touch()
        (package / '__init__.py').touch()
        (package / 'serving.py').write_text(FAKE_BENCH)
        return {'PYTHONPATH': str(root), 'SGLANG_IS_IN_CI': 'true', 'OPENAI_API_KEY': 'test-only-secret'}

    def test_subprocess_sweep_creates_directly_importable_points_and_audit_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args('--output', str(root / 'points.json'), '--tp-size', '1', '--dp-size', '8',
                             '--ep-size', '8', '--pp-size', '1', '--chunk-length', '8192', '--chunk-scope', 'dp-rank')
            with patch.dict(os.environ, self.fake_environment(root)), contextlib.redirect_stdout(io.StringIO()):
                report = collector.collect(args)
            stored = json.loads(args.output.read_text())
            self.assertEqual(report, stored)
            self.assertTrue(stored['collection']['complete'])
            self.assertEqual(len(stored['points']), 4)
            p, d = stored['points'][0], stored['points'][2]
            self.assertEqual((p['x_metric'], p['throughput_metric']), ('ttft_ms', 'input_throughput'))
            self.assertIsNone(p['prefill_service_ms'])
            self.assertIsNone(p['prefill_batch_size'])
            self.assertEqual(p['measurement']['output_tokens'], 1)
            self.assertEqual(p['chunk_size_tokens'], 8192)
            self.assertEqual((d['x_value'], d['measurement']['output_tokens']), (200, 16))
            self.assertIsNone(d['chunked_prefill'])
            self.assertEqual(d['concurrency'], 1)
            self.assertEqual(d['measurement']['metrics']['concurrency'], 0.5)
            self.assertNotIn('private-server-config', args.output.read_text())
            self.assertNotIn('test-only-secret', args.output.read_text())
            self.assertEqual(len(list((root / 'points.runs').glob('*.log'))), 4)

    def test_rejected_point_preserves_completed_points_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'partial.json'
            env = self.fake_environment(root)
            with patch.dict(os.environ, dict(env, FAKE_FAIL='1')), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'requests completed'):
                    collector.collect(self.args('--output', str(output)))
            report = json.loads(output.read_text())
            self.assertFalse(report['collection']['complete'])
            self.assertEqual(report['collection']['failed_point'], 'decode-c4')
            self.assertEqual(len(report['points']), 3)

    def test_rejects_failed_malformed_mismatched_and_nonfinite_metrics(self):
        args = self.args()
        job = list(collector.jobs(args))[2]
        bad_rows = [dict(completed=15), dict(failed=1), dict(total_output_tokens=1),
                    dict(total_input_tokens=1), dict(random_input_len=129), dict(max_concurrency=99),
                    dict(mean_tpot_ms=0), dict(mean_tpot_ms=float('nan')), dict(mean_tpot_ms=1e-320),
                    dict(completed=True), dict(output_throughput=float('inf')), dict(duration=0),
                    dict(output_throughput=1)]
        for change in bad_rows:
            with self.subTest(change=change), self.assertRaises(ValueError):
                collector.make_point(args, job, dict(self.row(job), **change))

    def test_unknown_config_stays_unknown(self):
        args = self.args()
        job = list(collector.jobs(args))[0]
        point = collector.make_point(args, job, self.row(job))
        self.assertIsNone(point['tp_size'])
        self.assertIsNone(point['chunked_prefill'])

    def test_invalid_options_fail_before_benchmark(self):
        cases = [('--output-len', '1'), ('--chunk-length', '8192'),
                 ('--no-chunked-prefill', '--chunk-length', '8192'),
                 ('--base-url', 'https://user:secret@host'), ('--base-url', 'https://host/v1'),
                 ('--seed', '-1'), ('--workload', ' '),
                 ('--phase', 'decode', '--no-chunked-prefill')]
        for flags in cases:
            with self.subTest(flags=flags), patch.object(collector, 'find_benchmark') as probe:
                with self.assertRaises(ValueError):
                    collector.collect(self.args(*flags))
                probe.assert_not_called()

    def test_no_overwrite_or_dry_run_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing.json'
            output.write_text('keep me')
            with patch.object(collector, 'find_benchmark') as probe:
                with self.assertRaises(ValueError):
                    collector.collect(self.args('--output', str(output)))
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    collector.collect(self.args('--output', str(output), '--dry-run'))
                probe.assert_not_called()
            self.assertEqual(output.read_text(), 'keep me')
            self.assertFalse(output.with_name('existing.runs').exists())
            self.assertIn('--random-output-len 1 ', captured.getvalue())
            self.assertIn('--random-output-len 16 ', captured.getvalue())

    def test_module_fallback_and_required_flag_preflight(self):
        args = self.args()
        with patch.object(collector.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 1, '', 'missing'),
            subprocess.CompletedProcess([], 0, '--base-url --num-prompts', '')]):
            module, flags = collector.find_benchmark(args)
        self.assertEqual(module, 'sglang.bench_serving')
        self.assertIn('--num-prompts', flags)
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / 'points.json'
            with patch.object(collector, 'find_benchmark', return_value=(module, flags)):
                with self.assertRaisesRegex(ValueError, 'lacks required options'):
                    collector.collect(args)
            self.assertFalse(args.output.exists())

    def test_environment_only_flushes_when_explicit(self):
        with patch.dict(os.environ, {'SGLANG_IS_IN_CI': 'true'}):
            self.assertEqual(collector.benchmark_environment()['SGLANG_IS_IN_CI'], 'false')
        args = self.args()
        job = list(collector.jobs(args))[0]
        self.assertNotIn('--flush-cache', collector.command(args, job, 'bench', Path('raw.jsonl')))
        args.flush_cache = True
        self.assertIn('--flush-cache', collector.command(args, job, 'bench', Path('raw.jsonl')))

    def test_standalone_cli_and_failed_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, **self.fake_environment(root))
            output = root / 'cli.json'
            cmd = [sys.executable, str(Path(collector.__file__).resolve()), '--model', 'test-model',
                   '--num-gpus', '8', '--concurrency', '1,4', '--input-len', '128', '--output-len', '16',
                   '--output', str(output)]
            result = subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(output.read_text())['points']), 4)
            cmd[-1] = str(root / 'failed.json')
            result = subprocess.run(cmd, env=dict(env, FAKE_FAIL='1'), cwd=root,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertIn('requests completed', result.stderr)
            self.assertEqual(len(json.loads(Path(cmd[-1]).read_text())['points']), 3)

    def test_timeout_preserves_a_well_formed_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args('--output', str(Path(directory) / 'timeout.json'))
            job = list(collector.jobs(args))[0]
            flags = {a for a in collector.command(args, job, 'bench', Path('raw')) if a.startswith('--')}
            with patch.object(collector, 'find_benchmark', return_value=('bench', flags)), \
                    patch.object(collector.subprocess, 'run', side_effect=subprocess.TimeoutExpired('bench', 1)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.TimeoutExpired):
                    collector.collect(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report['points'], [])
            self.assertFalse(report['collection']['complete'])
            self.assertEqual(report['collection']['failed_point'], 'prefill-c1')


if __name__ == '__main__':
    unittest.main()
