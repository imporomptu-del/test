from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from streaming_schedule_v14 import ScheduleConfig, Scheduler, required_halo, queue_model
from check_streaming_feasibility_v14 import transient_opportunities
from benchmark_selective_v14 import tile_set, compare_core


class ScheduleTests(unittest.TestCase):
    def test_gpu_scope_and_core_comparison_guards(self):
        cfg = ScheduleConfig()
        ids = tile_set(32, cfg)
        self.assertEqual(len(set(ids)), 32)
        self.assertTrue({0, 18, 123, 246} <= set(ids))
        expected = [np.arange(120, dtype=np.float32).reshape(10, 12) for _ in range(4)]
        observed = [a[1:9, 1:11].copy() for a in expected]
        comparison = compare_core(expected, observed, (3, 3, 8, 7), (1, 1, 11, 9))
        self.assertTrue(comparison['exact'])
        observed[0][3, 3] += 1
        self.assertFalse(compare_core(expected, observed, (3, 3, 8, 7), (1, 1, 11, 9))['exact'])

    def test_bounds(self):
        for kwargs in ({'cap': 0}, {'blind': 33}, {'halo': -1}, {'tile': True},
                       {'cap': 248}, {'height': 0}, {'stride': 17}):
            with self.assertRaises(ValueError):
                ScheduleConfig(**kwargs)

    def test_native_partition_includes_bottom_and_right(self):
        cfg = ScheduleConfig()
        self.assertEqual(cfg.tiles, 247)
        areas = [(cfg.rect(k)[2]-cfg.rect(k)[0])*(cfg.rect(k)[3]-cfg.rect(k)[1])
                 for k in range(cfg.tiles)]
        self.assertEqual(sum(areas), cfg.height*cfg.width)
        self.assertEqual(cfg.rect(246), (4608, 3072, 4784, 3190))
        self.assertIn(246, cfg.requested(4783, 3189))
        with self.assertRaises(ValueError):
            cfg.requested(4784, 3189)

    def test_warmup_stride_capacity_and_no_event_blind_coverage(self):
        scheduler = Scheduler()
        decisions = [r for i in range(84) if (r := scheduler.update(i, i*100000000, 0, []))]
        self.assertEqual([r['frame'] for r in decisions], list(range(19, 84, 8)))
        self.assertTrue(all(len(set(r['tiles'])) == 32 for r in decisions))
        self.assertEqual(set(k for r in decisions[:8] for k in r['tiles']), set(range(247)))

    def test_labels_and_future_data_not_inputs(self):
        a, b = Scheduler(), Scheduler()
        for i in range(40):
            seeds = [dict(x=250+i, y=3180, score=4)] if i > 12 else []
            self.assertEqual(a.update(i, i*100000000, 0, seeds), b.update(i, i*100000000, 0, seeds))
        # A turn near a tile seam does not leave out the adjacent requested core.
        cfg = ScheduleConfig()
        self.assertEqual(set(cfg.requested(255, 256)), {0, 1, 19, 20})

    def test_segment_reset_and_ttl(self):
        scheduler = Scheduler()
        for i in range(40):
            r = scheduler.update(i, i*100000000, int(i >= 20),
                                 [dict(x=100, y=100, score=5)] if i == 0 else [])
            if i == 19:
                self.assertEqual(r['requested'], 0)
            if i == 20:
                self.assertIsNone(r)
                self.assertTrue(all(v == -1 for v in scheduler.visits))
        self.assertEqual(r['frame'], 39)

    def test_bad_input_leaves_state_unchanged(self):
        scheduler = Scheduler()
        scheduler.update(0, 0, 0, [])
        for index, stamp, seeds in ((2, 100, []), (1, 0, []),
                                    (1, 100, [dict(x=-1, y=2, score=3)]),
                                    (1, 100, [dict(x=1, y=2, score=math.nan)])):
            with self.assertRaises(ValueError):
                scheduler.update(index, stamp, 0, seeds)
            self.assertEqual(scheduler.last_index, 0)

    def test_halo_checks_real_timestamps_not_nominal_fps(self):
        self.assertEqual(required_halo([i*100000000 for i in range(16)]), 4)
        self.assertEqual(required_halo([i*333333333 for i in range(16)]), 9)
        s = Scheduler()
        for i in range(19):
            s.update(i, i*333333333, 0, [])
        with self.assertRaises(ValueError):
            s.update(19, 19*333333333, 0, [])
        self.assertEqual(s.last_index, 18)

    def test_clutter_cannot_starve_reserved_blind_work(self):
        s = Scheduler()
        seen = set()
        for i in range(19+32*8):
            seeds = [dict(x=128+j*256, y=128, score=1000-j) for j in range(18)]
            r = s.update(i, i*100000000, 0, seeds)
            if r:
                self.assertEqual(len(r['blind_reserved']), 8)
                self.assertLessEqual(len(r['tiles']), 32)
                seen.update(r['tiles'])
                self.assertLessEqual(len(s.seeds), 247)
        self.assertEqual(len(seen), 247)

    def test_queue_exposes_backlog_without_dropping_frames(self):
        r = queue_model([200.]*10)
        self.assertEqual(r['final_wait_ms'], 900)
        self.assertEqual(r['final_arrival_to_completion_ms'], 1100)
        self.assertTrue(r['mean_exceeds_budget'])
        self.assertEqual(queue_model([50.]*10)['maximum_wait_ms'], 0)
        for values in ([], [-1], [math.inf]):
            with self.assertRaises(ValueError):
                queue_model(values)

    def test_transient_enumerator_matches_brute_force(self):
        cfg = ScheduleConfig(height=32, width=64, tile=16, cap=2, blind=1, seed_radius=0)
        result = transient_opportunities(cfg)
        period = cfg.tiles*cfg.stride
        first = cfg.warmup+cfg.window+period
        s = Scheduler(cfg)
        windows = []
        for i in range(first+period+64+cfg.window):
            row = s.update(i, i*100000000, 0, [])
            if row:
                windows.append(row)
        for case in result['cases']:
            counts = []
            for tile in range(cfg.tiles):
                for onset in range(first, first+period):
                    count = sum(tile in r['tiles'] and len(
                        set(range(onset, onset+case['visibility_frames'])) &
                        set(range(r['frame']-15, r['frame']+1))) >= 12 for r in windows)
                    counts.append(count)
            self.assertEqual(case['selective_at_least_one'], sum(c >= 1 for c in counts))
            self.assertEqual(case['selective_three_without_feedback'], sum(c >= 3 for c in counts))
            ideal = 0
            for tile in range(cfg.tiles):
                for onset in range(first, first+period):
                    qualifying = [r['frame'] for r in windows if tile in r['tiles'] and len(
                        set(range(onset, onset+case['visibility_frames'])) &
                        set(range(r['frame']-15, r['frame']+1))) >= 12]
                    if qualifying:
                        ends = [qualifying[0]+j*cfg.stride for j in range(3)]
                        ideal += all(len(set(range(onset, onset+case['visibility_frames'])) &
                                         set(range(end-15, end+1))) >= 12 for end in ends)
            self.assertEqual(case['selective_three_with_ideal_first_hit_feedback'], ideal)
        full = transient_opportunities(replace(cfg, cap=cfg.tiles, blind=cfg.tiles))
        self.assertTrue(all(c['lost_all_opportunities_vs_full'] == 0 and c['lost_three_even_with_ideal_feedback'] == 0
                            for c in full['cases']))


if __name__ == '__main__':
    unittest.main()
