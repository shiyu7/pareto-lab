import contextlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import collect_and_archive as runner


class RunnerTest(unittest.TestCase):
    def test_split_preserves_points_and_partial_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input-8192.points.json'
            data = {'schema': 'hpc-pareto-points', 'version': 2,
                    'points': [{'phase': 'prefill', 'x_value': 1}, {'phase': 'decode', 'x_value': 2}],
                    'collection': {'complete': False, 'failed_point': 'decode-c4'}}
            source.write_text(json.dumps(data))
            files = runner.split_phase_files(source, ('prefill', 'decode'), 2)
            for phase in ('prefill', 'decode'):
                split = json.loads((Path(tmp) / files[phase]).read_text())
                self.assertEqual(split['points'], [p for p in data['points'] if p['phase'] == phase])
                self.assertFalse(split['collection']['complete'])
                self.assertEqual(split['collection']['planned_points'], 2)
                self.assertEqual(split['collection']['split_from'], source.name)
            self.assertEqual(json.loads(source.read_text()), data)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                runner.split_phase_files(source, ('prefill',), 2)

    def args(self, *more):
        return runner.make_parser().parse_args([
            '--pod', 'test-pod', '--container', 'test-container', '--context', 'test-context',
            '--server-url', 'http://127.0.0.1:30000', '--model', '/models/example',
            '--num-gpus', '8', '--scheme', 'example', *more])

    def test_requested_default_workloads_and_one_full_endpoint_gpu_count(self):
        args = self.args()
        runner.validate(args)
        self.assertEqual(args.input_lengths, [8192, 32768, 131072])
        self.assertEqual(args.output_length, 1536)
        self.assertEqual(args.phase, 'both')
        self.assertEqual(args.concurrency, [1, 2, 4, 8, 16, 32])
        commands = [runner.collector_command(args, '/tmp/example', n) for n in args.input_lengths]
        self.assertEqual([c[c.index('--input-len') + 1] for c in commands], ['8192', '32768', '131072'])
        self.assertTrue(all(c[c.index('--num-gpus') + 1] == '8' for c in commands))
        self.assertEqual(len({c[c.index('--workload') + 1] for c in commands}), 3)
        self.assertTrue(all('--flush-cache' not in c for c in commands))
        explicit = self.args('--flush-cache')
        self.assertIn('--flush-cache', runner.collector_command(explicit, '/tmp/example', 8192))

    def test_ssh_quoting_preserves_arbitrary_arguments_as_data(self):
        args = runner.make_parser().parse_args(['--transport', 'ssh', '--ssh-target', 'user@example',
                    '--server-url', 'http://127.0.0.1:30000', '--model', '/data/a b', '--num-gpus', '8',
                    '--scheme', "x'; echo should-not-run; '"])
        runner.validate(args)
        remote = runner.collector_command(args, '/tmp/work', 8192)
        command = runner.transport_command(args, remote)
        self.assertEqual(shlex.split(command[-1]), remote)
        self.assertIn('StrictHostKeyChecking=yes', command)
        self.assertIn('BatchMode=yes', command)

    def test_invalid_connection_and_secret_urls_are_rejected_before_work(self):
        cases = [('--server-url', 'https://u:password@server'),
                 ('--server-url', 'https://server?token=secret'),
                 ('--server-url', 'http://server/v1'),
                 ('--chunk-length', '8192'),
                 ('--chunk-length', '8192', '--chunk-scope', 'request', '--no-chunked-prefill'),
                 ('--phase', 'decode', '--no-chunked-prefill')]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.validate(self.args(*extra))

    def test_download_checks_content_hash_and_size(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, _size): return b'untrusted bytes'
        with patch.object(runner.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                runner.download_collector()

    def test_kubernetes_arguments_are_not_shell_interpreted(self):
        args = self.args('--scheme', 'test; echo SHOULD_NOT_EXECUTE')
        remote = runner.collector_command(args, '/tmp/work', 8192)
        command = runner.transport_command(args, remote)
        self.assertEqual(command[command.index('--') + 1:], remote)
        self.assertIn('test; echo SHOULD_NOT_EXECUTE', command)

    def test_archive_and_stop_after_first_failure(self):
        class Process:
            def __init__(self, *args, **kwargs): self.stdout = iter(['mock benchmark\n'])
            def wait(self, **kwargs): return 1
        def fake_fetch(_args, remote, dest, tar=False):
            if tar:
                dest.write_bytes(b'archive')
            else:
                dest.write_text(json.dumps({'schema': 'hpc-pareto-points', 'version': 2,
                                            'points': [{'phase':'prefill'}],
                                            'collection': {'complete':False}}))
            return True, None
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args('--archive-root', tmp)
            with patch.object(runner, 'download_collector', return_value=b'collector'), \
                    patch.object(runner, 'init_remote'), patch.object(runner, 'fetch_remote', side_effect=fake_fetch), \
                    patch.object(runner.subprocess, 'Popen', Process), contextlib.redirect_stdout(io.StringIO()):
                root, report = runner.execute(args)
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(len(report['results']), 1)
            self.assertEqual(report['results'][0]['points'], 1)
            self.assertFalse(report['results'][0]['complete'])
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'partial')
            self.assertTrue((root / 'input-8192.points.runs.tar.gz').exists())
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)

    def test_dry_run_has_no_download_or_remote_mutation(self):
        args = self.args('--dry-run')
        with patch.object(runner, 'download_collector') as download, \
                patch.object(runner, 'init_remote') as init, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(runner.execute(args))
            download.assert_not_called()
            init.assert_not_called()
        self.assertEqual(out.getvalue().count('--input-len '), 3)

    def test_interrupt_attempts_to_archive_the_current_partial_result(self):
        class Process:
            def __init__(self, *args, **kwargs):
                def lines():
                    yield 'one point finished\n'
                    raise KeyboardInterrupt()
                self.stdout = lines()
            def terminate(self): pass
            def wait(self, **kwargs): return -15
        def fake_fetch(_args, remote, dest, tar=False):
            if tar:
                dest.write_bytes(b'archive')
            else:
                dest.write_text(json.dumps({'schema': 'hpc-pareto-points', 'version': 2,
                                            'points': [{'phase':'prefill'}],
                                            'collection': {'complete':False}}))
            return True, None
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args('--archive-root', tmp)
            with patch.object(runner, 'download_collector', return_value=b'collector'), \
                    patch.object(runner, 'init_remote'), patch.object(runner, 'fetch_remote', side_effect=fake_fetch), \
                    patch.object(runner.subprocess, 'Popen', Process), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute(args)
            root = next(Path(tmp).iterdir())
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertEqual(manifest['results'][0]['points'], 1)
            self.assertFalse(manifest['results'][0]['complete'])


if __name__ == '__main__':
    unittest.main()
