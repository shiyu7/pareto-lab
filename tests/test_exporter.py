import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.plot_pareto import document, load_points, main, make_point, parser


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.args = parser().parse_args([])
        self.row = dict(phase="prefill", workload="128k", prefill_latency_ms=120,
                        non_cached_input_throughput=273066.67, num_gpus=4,
                        tp_size=1, dp_size=4, ep_size=4, pp_size=1,
                        chunked_prefill=True, chunk_size_tokens=8192, chunk_scope="dp-rank",
                        prefill_batch_size=4096)

    def point(self, **changes):
        return make_point(dict(self.row, **changes), None, "run:1", self.args)

    def test_parallel_and_chunk_are_preserved_separately_from_load(self):
        p = self.point()
        self.assertEqual(p.scheme, "TP1 DP4 EP4 PP1")
        self.assertEqual((p.tp_size, p.dp_size, p.ep_size, p.pp_size), (1, 4, 4, 1))
        self.assertEqual((p.chunk_size_tokens, p.batch_tokens), (8192, 4096))
        self.assertEqual(document([p])["version"], 2)
        self.assertEqual(document([p])["points"][0]["chunk_scope"], "dp-rank")

    def test_unknown_differs_from_disabled_and_scheme_names_are_not_parsed(self):
        unknown = self.point(scheme="EP64DP64", tp_size=None, ep_size=None,
                             chunked_prefill=None, chunk_size_tokens=None, chunk_scope=None)
        disabled = self.point(chunked_prefill=False, chunk_size_tokens=None, chunk_scope=None)
        self.assertIsNone(unknown.tp_size)
        self.assertIsNone(unknown.ep_size)
        self.assertIsNone(unknown.chunked_prefill)
        self.assertIs(disabled.chunked_prefill, False)

    def test_cli_configuration_fallback_and_row_precedence(self):
        args = parser().parse_args(["--phase", "prefill", "--tp-size", "2", "--dp-size", "4",
                                    "--ep-size", "4", "--pp-size", "1", "--chunk-length", "8192",
                                    "--chunk-scope", "request"])
        row = dict(phase="prefill", tp_size=1, prefill_latency_ms=100, non_cached_input_throughput=10000)
        p = make_point(row, "custom", "test:1", args)
        self.assertEqual((p.tp_size, p.dp_size, p.ep_size, p.pp_size), (1, 4, 4, 1))
        self.assertTrue(p.chunked_prefill)
        self.assertEqual(p.chunk_scope, "request")
        self.assertEqual(p.scheme, "custom")

    def test_csv_booleans_and_chunk_length_alias(self):
        p = self.point(chunked_prefill="true", chunk_size_tokens=None, chunk_length="16384")
        self.assertEqual(p.chunk_size_tokens, 16384)
        self.assertTrue(p.chunked_prefill)
        self.assertFalse(self.point(chunked_prefill="false", chunk_size_tokens=None, chunk_scope=None).chunked_prefill)

    def test_request_chunk_and_rank_budget_have_different_batch_constraints(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.point(prefill_batch_size=16384)
        self.assertEqual(self.point(prefill_batch_size=16384, chunk_scope="request").batch_tokens, 16384)

    def test_invalid_configuration_is_rejected(self):
        for changes in (dict(chunked_prefill=False), dict(chunk_size_tokens=None), dict(chunk_scope=None),
                        dict(chunk_scope="instance"), dict(chunk_size_tokens=0), dict(tp_size=0),
                        dict(ep_size=1.5), dict(pp_size=True), dict(chunked_prefill="maybe")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.point(**changes)

    def test_decode_rejects_accidental_prefill_configuration(self):
        with self.assertRaisesRegex(ValueError, "only to prefill"):
            self.point(phase="decode", per_user_tps=300, output_throughput=10000)

    def test_same_scheme_accepts_distinct_configurations_and_v2_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            output = Path(directory) / "points.json"
            source.write_text(json.dumps([dict(self.row, scheme="same"),
                                          dict(self.row, scheme="same", chunk_size_tokens=16384, tp_size=2)]))
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(source), "--output", str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(data["version"], 2)
            self.assertEqual(len(data["points"]), 2)
            self.assertEqual(data["points"][1]["chunk_size_tokens"], 16384)
            self.assertEqual(data["points"][1]["tp_size"], 2)

    def test_full_prefill_service_time_is_explicit_and_validated(self):
        self.assertIsNone(self.point().prefill_service_ms)
        self.assertEqual(document([self.point(prefill_service_ms=420)])["points"][0]["prefill_service_ms"], 420)
        for value in (0, -1, "nan", True):
            with self.assertRaises(ValueError):
                self.point(prefill_service_ms=value)
        with self.assertRaisesRegex(ValueError, "only to prefill"):
            self.point(phase="decode", chunked_prefill=None, chunk_size_tokens=None, chunk_scope=None,
                       per_user_tps=300, output_throughput=10000, prefill_service_ms=50)

    def test_synthetic_example_has_three_prefill_configurations(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "pareto_example.csv"
        points = load_points(parser().parse_args([str(path)]))
        prefill = [p for p in points if p.phase == "prefill"]
        self.assertEqual(len(points), 17)
        self.assertEqual(len({(p.scheme, p.chunk_size_tokens) for p in prefill}), 3)


if __name__ == "__main__":
    unittest.main()
