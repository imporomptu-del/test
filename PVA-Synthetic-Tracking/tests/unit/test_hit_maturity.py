"""Causal soft association preference; never a hard established-track cascade."""
from dataclasses import replace
import unittest
import numpy as np
from test_kalman_tracking import batch, candidate, config
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector, VisibleTracks


class HitMaturityTests(unittest.TestCase):
    def manager(self, mode="hit_maturity", old_x=20, old_misses=0):
        manager = KalmanTrackManager(config(
            measurement_model="position_only", association_cost="gaussian_nll",
            association_prior=mode, confirmation_independent_hits=4,
            maximum_position_residual_px=30, mahalanobis_gate_squared=100,
        ))
        manager.update(batch(0, (0,), (candidate(0, old_x, 30), candidate(1, 20, 30))))
        # Isolate association from velocity estimation: identical finite covariance.
        for t in manager._tracks.values():
            t.mean[2:] = 0
        manager._tracks[0].independent_confirmation_hits = 40
        manager._tracks[0].missed_windows = old_misses
        manager._tracks[1].independent_confirmation_hits = 4
        return manager

    def test_small_likelihood_difference_prefers_observation_history(self):
        m = self.manager(old_x=21)
        result = m.update(batch(100_000_000, (1,), (candidate(0, 20, 30),)))
        self.assertEqual(result.associations[0]["track_id"], 0)
        plain = self.manager(mode="none", old_x=21)
        result = plain.update(batch(100_000_000, (1,), (candidate(0, 20, 30),)))
        self.assertEqual(result.associations[0]["track_id"], 1)

    def test_better_new_track_can_win_and_stale_track_loses_preference(self):
        for kwargs in ({"old_x": 27}, {"old_x": 20, "old_misses": 3}):
            with self.subTest(**kwargs):
                result = self.manager(**kwargs).update(
                    batch(100_000_000, (1,), (candidate(0, 20, 30),)))
                self.assertEqual(result.associations[0]["track_id"], 1)

    def test_gates_and_one_to_one_evidence_still_apply(self):
        m = self.manager()
        result = m.update(batch(100_000_000, (1,), (candidate(0, 200, 200),)))
        self.assertFalse(result.associations)
        self.assertEqual(len(result.born_track_ids), 1)
        self.assertEqual(len({a["candidate_index"] for a in result.associations}),
                         len(result.associations))

    def test_near_crossing_and_unequal_age_movers_keep_ids(self):
        cfg = VisibleConfig(spatial_background="median5", warmup_frames=4,
            shape_measurement_mode="mutual_half_height_r8", motion_quality_enabled=True,
            tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity")
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        identities = None
        for i in range(40):
            im = np.full((90, 140), 20, np.float32)
            if i >= 6:
                im[35, 10 + i * 2] = 150
            if i >= 12:
                im[41, 125 - i * 2] = 100
            ps, _ = detector.update(im, np.ones_like(im, bool), 0)
            rows, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            if i >= 22:
                observed = {round(t["measurement_source_xy"][1]): t["track_id"]
                    for t in rows if t["measured"] and t["qualified_moving"]
                    and t["track_id"].startswith("bright:")}
                self.assertEqual(set(observed), {35, 41})
                if identities is None:
                    identities = observed
                self.assertEqual(observed, identities)

    def test_disabled_and_invalid_combinations(self):
        self.assertEqual(VisibleConfig().tracking_association_prior, "none")
        with self.assertRaises(ValueError):
            config(association_prior="hit_maturity")
        with self.assertRaises(ValueError):
            VisibleConfig(tracking_association_prior="invalid")
        with self.assertRaises(ValueError):
            replace(config(association_cost="gaussian_nll"),
                    association_prior="hit_maturity", association_cascade="confirmed_first")

    def test_combination_does_not_qualify_stationary_flickering_lights(self):
        cfg = VisibleConfig(spatial_background="median5", warmup_frames=4,
            pixel_noise_enabled=True, pixel_noise_model="background_residual",
            shape_measurement_mode="mutual_half_height_r8", motion_quality_enabled=True,
            learning_exclusion_radius_px=6, learning_protection_mode="variance_only",
            tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity")
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        for i in range(70):
            im = np.full((90, 140), 20, np.float32)
            im[30, 35] = 100 + 40 * np.sin(i * 0.5)
            im[55, 90] = 140 if i % 4 < 2 else 50
            centers = tracker.learning_centers(i * 100_000_000, 0)
            ps, _ = detector.update(im, np.ones_like(im, bool), 0, centers)
            rows, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            self.assertFalse(any(t["qualified_moving"] for t in rows))

    def test_combination_crossing_uses_separate_measurements(self):
        cfg = VisibleConfig(spatial_background="median5", warmup_frames=4,
            pixel_noise_enabled=True, pixel_noise_model="background_residual",
            shape_measurement_mode="mutual_half_height_r8", motion_quality_enabled=True,
            learning_exclusion_radius_px=6, learning_protection_mode="variance_only",
            tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity")
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        identities = None
        for i in range(40):
            im = np.full((90, 140), 20, np.float32)
            if i >= 6:
                im[35, 10 + i * 2] = 150
            if i >= 12:
                im[41, 125 - i * 2] = 100
            ps, _ = detector.update(im, np.ones_like(im, bool), 0,
                                   tracker.learning_centers(i * 100_000_000, 0))
            rows, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            if i >= 22:
                observed = {round(t["measurement_source_xy"][1]): t["track_id"]
                    for t in rows if t["measured"] and t["qualified_moving"]
                    and t["track_id"].startswith("bright:")}
                self.assertEqual(set(observed), {35, 41})
                if identities is None:
                    identities = observed
                self.assertEqual(observed, identities)


if __name__ == "__main__":
    unittest.main()
