import copy
import sys
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from diagnose_accuracy_v41_transport import evaluate_lag, count_state, preference


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.window = dict(frame_start=10, crop_xywh=[0, 0, 80, 80])
        y, x = np.mgrid[:80, :80]
        image = (50+60*np.exp(-((x-40)**2+(y-40)**2)/4)).astype(np.uint8)
        self.pixels = np.stack([image]*20)
        self.track = dict(track_id='bright:1', segment=0, measured=True,
                          measurement_source_xy=[40., 40.], qualified_moving=True)
        self.rows = {i:dict(frame_index=i, timestamp_ns=i*100_000_000, segment=0,
                          motion=dict(reset=False), source_to_reference=np.eye(3).tolist(),
                          tracks=[copy.deepcopy(self.track)]) for i in range(10, 30)}

    def run_lag(self, frame=18, lag=8):
        return evaluate_lag(self.window, self.rows, self.pixels, frame, self.track, lag)

    def test_all_nine_zero_transport_ties(self):
        result = self.run_lag()
        self.assertTrue(result['pair_available'])
        self.assertTrue(result['all_nine_available'])
        self.assertEqual(len(result['variants']), 9)
        self.assertEqual(result['offset_sign_agreement'], 'numerical_tie')

    def test_no_history_or_prediction_substitution(self):
        self.assertEqual(self.run_lag(frame=10)['reasons'], ['insufficient_retained_window_history'])
        self.rows[10]['tracks'][0]['measured'] = False
        self.rows[10]['tracks'][0]['measurement_source_xy'] = None
        self.assertEqual(self.run_lag()['reasons'], ['prior_identity_not_measured'])

    def test_no_identity_interpolation(self):
        self.rows[10]['tracks'][0]['track_id'] = 'bright:2'
        self.assertEqual(self.run_lag()['reasons'], ['prior_identity_absent'])

    def test_reset_and_time_gap_unavailable(self):
        self.rows[14]['motion']['reset'] = True
        self.assertEqual(self.run_lag()['reasons'], ['segment_or_reset_boundary'])
        self.rows[14]['motion']['reset'] = False
        self.rows[18]['timestamp_ns'] += 1
        self.assertEqual(self.run_lag()['reasons'], ['non_nominal_timestamp_spacing'])

    def test_duplicate_identity_or_corrupt_measurement_fails(self):
        self.rows[10]['tracks'].append(copy.deepcopy(self.track))
        with self.assertRaises(ValueError): self.run_lag()
        self.rows[10]['tracks'].pop()
        self.rows[10]['tracks'][0]['measurement_source_xy'] = None
        with self.assertRaises(ValueError): self.run_lag()

    def test_unavailable_stays_in_denominator(self):
        results = [self.run_lag(frame=10, lag=lag) for lag in (1, 2, 4, 8)]
        counter = Counter()
        count_state(counter, dict(qualified_moving=True, lags=results))
        self.assertEqual(counter['actual_states'], 1)
        self.assertEqual(counter['8:planned_variants'], 9)
        self.assertEqual(counter['8:available_variants'], 0)
        self.assertEqual(counter['8:zero:unavailable'], 1)
        self.assertEqual(counter['all_36_available'], 0)

    def test_incomplete_lag_bank_cannot_claim_all_36(self):
        with self.assertRaises(ValueError):
            count_state(Counter(), dict(qualified_moving=True, lags=[self.run_lag()]))

    def test_numerical_ties_not_confidence(self):
        self.assertEqual(preference(1e-10), 'numerical_tie')
        self.assertEqual(preference(-1e-10), 'numerical_tie')
        self.assertEqual(preference(.1), 'transported')
        self.assertEqual(preference(-.1), 'stationary')


if __name__ == '__main__':
    unittest.main()
