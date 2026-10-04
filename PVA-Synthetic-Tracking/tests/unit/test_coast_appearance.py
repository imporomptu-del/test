"""Coasting retains discounted observed appearance, never invented measurements."""
from dataclasses import replace
import unittest
import numpy as np
from test_kalman_tracking import batch, candidate, config
from test_visible_learning import cfg as learning_config
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector, VisibleTracks


class CoastAppearanceTests(unittest.TestCase):
    def manager(self, misses=0, response=20, mode="log_response_coast"):
        manager = KalmanTrackManager(config(measurement_model="position_only",
            association_cost="gaussian_nll", association_appearance=mode,
            maximum_position_residual_px=30, mahalanobis_gate_squared=100,
            max_missed_windows=7))
        manager.update(batch(0, (0,), (replace(candidate(0, 20, 30), raw_sum_score=response),)))
        track = manager._tracks[0]
        track.mean[2:] = 0
        track.independent_confirmation_hits = manager.config.confirmation_independent_hits
        track.missed_windows = misses
        return manager

    def alternatives(self, similar=20, different=2):
        return (replace(candidate(0, 20, 30), raw_sum_score=different),
                replace(candidate(1, 21, 30), raw_sum_score=similar))

    def test_coasting_preserves_consistency_with_decreasing_weight(self):
        weights = []
        for misses in (0, 1, 4, 7):
            result = self.manager(misses).update(batch(100_000_000, (1,), self.alternatives()))
            self.assertEqual(result.associations[0]["candidate_index"], 1)
            weights.append(result.metrics["association_audit"][0]["appearance_coast_weight"])
        self.assertEqual(weights[0], 1)
        self.assertTrue(all(a > b > 0 for a, b in zip(weights, weights[1:])))

    def test_old_mode_still_forgets_and_expired_history_is_unused(self):
        for manager in (self.manager(1, mode="log_response"), self.manager(8)):
            result = manager.update(batch(100_000_000, (1,), self.alternatives()))
            self.assertEqual(result.associations[0]["candidate_index"], 0)

    def test_not_a_brightest_candidate_preference(self):
        result = self.manager(4, response=2).update(
            batch(100_000_000, (1,), self.alternatives(similar=2, different=20)))
        self.assertEqual(result.associations[0]["candidate_index"], 1)

    def test_unopposed_fade_keeps_measurement_and_no_gate_is_relaxed(self):
        faded = (replace(candidate(0, 20, 30), raw_sum_score=.1),)
        self.assertEqual(len(self.manager(4).update(batch(100_000_000, (1,), faded)).associations), 1)
        for observations in ((), (candidate(0, 200, 200),)):
            self.assertFalse(self.manager(4).update(batch(100_000_000, (1,), observations)).associations)

    def test_missing_nonfinite_and_unconfirmed_appearance_do_not_bias(self):
        for last in (None, 0., float("nan"), float("inf")):
            manager = self.manager(2)
            manager._tracks[0].last_raw_response = last
            result = manager.update(batch(100_000_000, (1,), self.alternatives()))
            self.assertEqual(result.associations[0]["candidate_index"], 0)
        manager = self.manager(2)
        manager._tracks[0].independent_confirmation_hits = 1
        result = manager.update(batch(100_000_000, (1,), self.alternatives()))
        self.assertEqual(result.associations[0]["candidate_index"], 0)

    def test_separate_near_crossing_movers_are_not_merged(self):
        cfg = replace(learning_config(), tracking_association_appearance="log_response_coast")
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        identities = None
        for index in range(40):
            image = np.full((90, 140), 20, np.float32)
            if index >= 6:
                image[35, 10 + index * 2] = 150
            if index >= 12:
                image[41, 125 - index * 2] = 100
            proposals, _ = detector.update(image, np.ones_like(image, bool), 0,
                tracker.learning_centers(index * 100_000_000, 0))
            rows, _ = tracker.update(proposals, index, index * 100_000_000, 0, np.eye(3), image.shape)
            if index >= 22:
                observed = {round(t["measurement_source_xy"][1]): t["track_id"] for t in rows
                    if t["measured"] and t["qualified_moving"] and t["track_id"].startswith("bright:")}
                self.assertEqual(set(observed), {35, 41})
                if identities is None:
                    identities = observed
                self.assertEqual(identities, observed)

    def test_new_policy_is_explicit_and_optional(self):
        self.assertEqual(VisibleConfig().tracking_association_appearance, "none")
        with self.assertRaises(ValueError):
            VisibleConfig(tracking_association_appearance="log_response_coast")
        self.assertEqual(replace(learning_config(), tracking_association_appearance="log_response_coast")
                         .tracker(10).association_appearance, "log_response_coast")
