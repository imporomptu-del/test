"""Appearance is soft, causal and optional; it cannot manufacture detections."""
from dataclasses import replace
import unittest
from test_kalman_tracking import batch, candidate, config
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig


class ResponseConsistencyTests(unittest.TestCase):
    def manager(self, mode='log_response', response=20):
        m = KalmanTrackManager(config(measurement_model='position_only',
            association_cost='gaussian_nll', association_appearance=mode,
            maximum_position_residual_px=30, mahalanobis_gate_squared=100))
        m.update(batch(0, (0,), (replace(candidate(0,20,30),raw_sum_score=response),)))
        t = m._tracks[0]
        t.mean[2:] = 0
        t.independent_confirmation_hits = m.config.confirmation_independent_hits
        return m

    def test_similar_response_wins_close_position_tie(self):
        m=self.manager()
        candidates=(replace(candidate(0,20,30),raw_sum_score=2),
                    replace(candidate(1,21,30),raw_sum_score=18))
        r=m.update(batch(100_000_000,(1,),candidates))
        self.assertEqual(r.associations[0]['candidate_index'],1)
        r=self.manager('none').update(batch(100_000_000,(1,),candidates))
        self.assertEqual(r.associations[0]['candidate_index'],0)

    def test_not_brightest_preference(self):
        r=self.manager(response=2).update(batch(100_000_000,(1,),(
            replace(candidate(0,20,30),raw_sum_score=20),
            replace(candidate(1,21,30),raw_sum_score=2))))
        self.assertEqual(r.associations[0]['candidate_index'],1)

    def test_fade_without_alternative_is_still_measured(self):
        r=self.manager().update(batch(100_000_000,(1,),(
            replace(candidate(0,20,30),raw_sum_score=0.1),)))
        self.assertEqual(len(r.associations),1)

    def test_missing_measurement_never_created_and_gate_unchanged(self):
        for cs in [(),(replace(candidate(0,300,30),raw_sum_score=20),)]:
            r=self.manager().update(batch(100_000_000,(1,),cs))
            self.assertFalse(r.associations)

    def test_stale_history_not_used(self):
        m=self.manager();m._tracks[0].missed_windows=1
        r=m.update(batch(100_000_000,(1,),(
            replace(candidate(0,20,30),raw_sum_score=2),
            replace(candidate(1,21,30),raw_sum_score=20))))
        self.assertEqual(r.associations[0]['candidate_index'],0)

    def test_default_off_invalid_rejected(self):
        self.assertEqual(VisibleConfig().tracking_association_appearance,'none')
        with self.assertRaises(ValueError): VisibleConfig(tracking_association_appearance='bad')
        with self.assertRaises(ValueError): config(association_appearance='log_response')
