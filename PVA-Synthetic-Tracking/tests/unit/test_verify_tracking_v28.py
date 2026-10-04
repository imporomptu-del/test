import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from verify_tracking_v28 import summarize


def samples():
    return [dict(clip='chunk_' + c, repeat=i, mode=m, frames=128, profiled=False,
                 tracking_ms=[{'v20': 90., 'v27': 75., 'v28': 50.}[m]]*128)
            for c in ('0126', '0082') for i in range(2)
            for m in (('v20', 'v27', 'v28') if i == 0 else ('v28', 'v27', 'v20'))]


class VerifyTrackingStageTests(unittest.TestCase):
    def test_time_and_throughput_are_distinct(self):
        result = summarize(samples())['0126']['v28_vs']
        self.assertAlmostEqual(result['v20']['time_reduction_percent'], 100*40/90)
        self.assertEqual(result['v20']['throughput_ratio'], 1.8)
        self.assertEqual(result['v27']['saved_ms'], 25.)

    def test_schedule_and_scope(self):
        for change in ('missing', 'reorder', 'profiled', 'frames', 'length'):
            rows = samples()
            if change == 'missing': rows.pop()
            elif change == 'reorder': rows.reverse()
            elif change == 'profiled': rows[0]['profiled'] = True
            elif change == 'frames': rows[0]['frames'] = 127
            else: rows[0]['tracking_ms'].pop()
            with self.assertRaises(ValueError): summarize(rows)

    def test_bad_durations(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            rows = samples()
            rows[0]['tracking_ms'][0] = value
            with self.assertRaises(ValueError): summarize(rows)

    def test_pool_durations_not_ratios(self):
        rows = samples()
        rows[0]['tracking_ms'] = [100.]*128
        rows[5]['tracking_ms'] = [50.]*128
        rows[2]['tracking_ms'] = [25.]*128
        rows[3]['tracking_ms'] = [50.]*128
        result = summarize(rows)['0126']['v28_vs']['v20']
        self.assertEqual(result['paired_ratios'], [4., 1.])
        self.assertEqual(result['throughput_ratio'], 2.)


if __name__ == '__main__':
    unittest.main()
