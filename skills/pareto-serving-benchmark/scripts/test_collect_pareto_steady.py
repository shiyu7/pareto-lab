import unittest

from collect_pareto_steady import prom_values, window_metrics


class SteadyTest(unittest.TestCase):
    def snapshots(self):
        def snapshot(t, value):
            return dict(t=t, p={'loads': [{'total_prefill_uncached_tokens': value * 8192,
                'total_prefill_busy_us': value * 100000}]}, pm={'steps': value},
                d={'loads': [dict(dp_rank=0, num_running_reqs=1,
                    decode_moments=[value, value, value * 100000, value, value * 100000, value * 4])]})
        return [snapshot(0, 0), snapshot(10, 10)]

    def test_tokens_not_sse_events_and_same_window_latency(self):
        events = [dict(t=1, previous_t=None, d_tokens=0, delta_tokens=1),
                  dict(t=2, previous_t=1, d_tokens=4, delta_tokens=4),
                  dict(t=9, previous_t=2, d_tokens=2, delta_tokens=2),
                  dict(t=11, previous_t=9, d_tokens=3, delta_tokens=3)]
        result = window_metrics(events, [dict(start=0, first=1, end=11)], self.snapshots(), 0, 10, 'decode')
        self.assertEqual(result['output_tokens'], 6)
        self.assertEqual(result['output_tps'], .6)
        self.assertAlmostEqual(result['tpot_ms'], 8000 / 6)
        self.assertEqual(result['native_decode_batch_time_mean'], .1)  # idle wall time stays in denominator
        self.assertEqual(result['prefill_step_ms'], 100)

    def test_native_counter_reset_is_rejected(self):
        snapshots = self.snapshots()
        snapshots[1]['d']['loads'][0]['decode_moments'] = [-1] * 6
        with self.assertRaisesRegex(ValueError, 'counter reset'):
            window_metrics([], [], snapshots, 0, 10, 'decode')

    def test_prefill_dp_counts_sum_and_rank_disappearance_rejected(self):
        snapshots = self.snapshots()
        for snapshot in snapshots:
            snapshot['p']['loads'][0]['dp_rank'] = 0
            snapshot['p']['loads'].append(dict(snapshot['p']['loads'][0], dp_rank=1))
        result = window_metrics([], [], snapshots, 0, 10, 'prefill')
        self.assertEqual(result['prefill_uncached_tokens'], 2 * 10 * 8192)
        snapshots[1]['d']['loads'] = []
        with self.assertRaisesRegex(ValueError, 'rank membership changed'):
            window_metrics([], [], snapshots, 0, 10, 'prefill')

    def test_rank_duplicates_are_not_summed(self):
        metrics = '\n'.join([
            'sglang:cuda_graph_passes_total{tp_rank="0",pp_rank="0",mode="prefill_none"} 12',
            'sglang:cuda_graph_passes_total{tp_rank="1",pp_rank="0",mode="prefill_none"} 12',
            'sglang:cuda_graph_passes_total{tp_rank="0",pp_rank="1",mode="prefill_none"} 12'])
        self.assertEqual(prom_values(metrics)['steps'], 12)


if __name__ == '__main__':
    unittest.main()
