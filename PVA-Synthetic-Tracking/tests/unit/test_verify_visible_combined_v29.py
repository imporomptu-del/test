import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import test_combined_v29 as fixtures
from combined_v29_protocol import schedule, full_schedule, samples, speed_gate
from verify_visible_combined_v29 import schedules, performance, transformed_hash


def parsed(receipts):
    return ({n: dict(r, count=r['processed_frames']) for n, r in receipts.items()},
            {n: samples(r) for n, r in receipts.items()})


class CombinedAuditTests(unittest.TestCase):
    def test_independent_schedule_and_source_transform(self):
        self.assertEqual(schedules(), (schedule(), full_schedule()))
        self.assertEqual(len(transformed_hash()), 64)

    def test_independent_gate_matches_fixed_examples(self):
        for case in ('nominal', 'threshold', 'one_slow_pair', 'latency', 'single_faster'):
            rows = fixtures.receipts()
            if case == 'threshold':
                for r in rows.values():
                    if r['arm'] == 'combined': r.update(fps=1.19, wall_s=128/1.19)
            elif case == 'one_slow_pair':
                rows['0126_repeat0_combined'].update(fps=.99, wall_s=128/.99)
            elif case == 'latency':
                for i in range(2): rows[f'0126_repeat{i}_combined']['consumer_frame_ms'] = [300.]*128
            elif case == 'single_faster':
                for i in range(3): rows[f'0126_repeat{i}_v26'].update(fps=1.6, wall_s=80.)
            self.assertEqual(performance(*parsed(rows)), speed_gate(rows))

    def test_reject_missing_or_instrumented_sample(self):
        trials, values = parsed(fixtures.receipts())
        trials.pop('0126_repeat0_v20')
        with self.assertRaises(ValueError): performance(trials, values)
        trials, values = parsed(fixtures.receipts())
        trials['0126_repeat0_combined']['state_audit'] = True
        with self.assertRaises(ValueError): performance(trials, values)

    def test_bad_durations_and_missing_latency(self):
        for duration in (0, -1, float('nan'), float('inf'), True):
            trials, values = parsed(fixtures.receipts())
            values['0126_repeat0_combined']['consumer_cadence'][0] = duration
            with self.assertRaises(ValueError): performance(trials, values)
        trials, values = parsed(fixtures.receipts())
        values['0126_repeat0_combined']['queue_aware'].pop()
        with self.assertRaises(ValueError): performance(trials, values)

    def test_pooled_fps_is_not_average_pair_ratio(self):
        rows = fixtures.receipts()
        rows['0126_repeat0_combined'].update(fps=2., wall_s=64.)
        rows['0126_repeat1_combined'].update(fps=1., wall_s=128.)
        rows['0126_repeat2_combined'].update(fps=4., wall_s=32.)
        gate = performance(*parsed(rows))['clips']['0126']
        self.assertAlmostEqual(gate['pooled_fps']['combined'], 384/224)
        self.assertNotAlmostEqual(gate['pooled_fps']['combined'], 7/3)


if __name__ == '__main__':
    unittest.main()
