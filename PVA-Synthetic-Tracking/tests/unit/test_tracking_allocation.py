"""Adversarial capacity/identity checks independent of the reviewed videos."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from test_kalman_tracking import batch, candidate, config
from tiny_target.tracking import KalmanTrackManager


class TrackingAllocationTests(unittest.TestCase):
    def test_default_birth_policy_preserves_input_order(self):
        result = KalmanTrackManager(config(max_active_tracks=2)).update(
            batch(
                0,
                (0,),
                (
                    candidate(0, 1, 1, score=4),
                    candidate(1, 20, 1, score=5),
                    candidate(2, 1000, 1, score=100),
                ),
            )
        )
        self.assertEqual(
            [t.state_xy_vx_vy[:2] for t in result.tracks], [(1, 1), (20, 1)]
        )

    def test_likelihood_cost_does_not_relax_absolute_position_gate(self):
        tracker = KalmanTrackManager(config(association_cost="gaussian_nll"))
        tracker.update(batch(0, (0,), (candidate(0, 10, 10),)))
        tracker._tracks[0].covariance *= 1e6
        result = tracker.update(batch(100_000_000, (1,), (candidate(0, 100, 10),)))
        self.assertEqual(result.associations, ())
        self.assertEqual(result.metrics["rejected_pair_counts"]["position_gate"], 1)

    def test_capacity_and_admission_audit_remain_consistent_under_churn(self):
        tracker = KalmanTrackManager(
            config(max_active_tracks=8, birth_policy="spatial_fair")
        )
        rng = np.random.default_rng(841)
        for frame in range(80):
            points = tuple(
                candidate(i, int(x), int(y), score=float(score))
                for i, (x, y, score) in enumerate(
                    rng.uniform([0, 0, 4], [1800, 900, 20], (30, 3))
                )
            )
            result = tracker.update(batch(frame * 100_000_000, (frame,), points))
            self.assertLessEqual(len(result.tracks), 8)
            m = result.metrics
            self.assertEqual(
                m["unmatched_candidate_count"],
                m["birth_count"] + m["dropped_birth_count_at_active_track_cap"],
            )
            self.assertEqual(
                len(m["birth_admission"]["rejected_candidate_indices"]),
                m["dropped_birth_count_at_active_track_cap"],
            )
            self.assertEqual(m["birth_admission"]["confirmed_tracks_evicted"], 0)

    def test_ambiguous_competitors_are_logged_not_claimed_as_confidence(self):
        tracker = KalmanTrackManager(config(association_cost="gaussian_nll"))
        tracker.update(batch(0, (0,), (candidate(0, 10, 10), candidate(1, 12, 10))))
        result = tracker.update(batch(100_000_000, (1,), (candidate(0, 11, 10),)))
        audit = result.metrics["association_audit"][0]
        self.assertTrue(audit["competing_alternative_within_likelihood_factor_three"])
        self.assertAlmostEqual(audit["alternate_minus_selected_cost"], 0)
        self.assertFalse(result.metrics["detector_score_used_as_track_confidence"])

    def test_fair_births_ignore_raster_order_and_global_brightness(self):
        points = [candidate(i, i * 10, 20, score=100 - i) for i in range(8)]
        points += [candidate(8, 1200, 20, score=4)]
        cfg = config(max_active_tracks=3, birth_policy="spatial_fair")
        outputs = []
        for ordering in (points, list(reversed(points))):
            result = KalmanTrackManager(cfg).update(batch(0, (0,), tuple(ordering)))
            outputs.append({t.state_xy_vx_vy[:2] for t in result.tracks})
            self.assertEqual(len(result.tracks), 3)
            self.assertEqual(
                result.metrics["dropped_birth_count_at_active_track_cap"], 6
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn((1200, 20), outputs[0])

    def test_sparse_new_target_can_replace_only_unobserved_tentative_clutter(self):
        tracker = KalmanTrackManager(
            config(max_active_tracks=3, birth_policy="spatial_fair")
        )
        first = tracker.update(
            batch(0, (0,), tuple(candidate(i, i * 20, 0) for i in range(3)))
        )
        result = tracker.update(
            batch(100_000_000, (1,), (candidate(0, 1200, 10, score=4),))
        )
        self.assertEqual(len(result.tracks), 3)
        self.assertEqual(len(result.born_track_ids), 1)
        self.assertEqual(len(result.deleted_tracks), 1)
        self.assertEqual(
            result.deleted_tracks[0]["reason"], "spatial_capacity_tentative_replacement"
        )
        self.assertIn((1200, 10), {t.state_xy_vx_vy[:2] for t in result.tracks})
        self.assertEqual(len(first.tracks), 3)

    def test_confirmed_tracks_are_not_evicted_even_when_they_coast(self):
        tracker = KalmanTrackManager(
            config(max_active_tracks=2, birth_policy="spatial_fair")
        )
        points = (candidate(0, 10, 10), candidate(1, 40, 10))
        tracker.update(batch(0, (0,), points))
        tracker.update(batch(100_000_000, (1,), points))
        result = tracker.update(
            batch(200_000_000, (2,), (candidate(0, 1200, 10, score=1000),))
        )
        self.assertEqual(result.born_track_ids, ())
        self.assertEqual(result.deleted_tracks, ())
        self.assertEqual(
            result.metrics["birth_admission"]["rejected_candidate_indices"], [0]
        )

    def test_observed_tentative_tracks_are_protected_from_same_frame_eviction(self):
        tracker = KalmanTrackManager(
            config(
                max_active_tracks=2,
                confirmation_independent_hits=4,
                birth_policy="spatial_fair",
            )
        )
        points = (candidate(0, 10, 10), candidate(1, 40, 10))
        tracker.update(batch(0, (0,), points))
        result = tracker.update(
            batch(100_000_000, (1,), points + (candidate(2, 1200, 10),))
        )
        self.assertEqual(result.deleted_tracks, ())
        self.assertEqual(len(result.associations), 2)

    def test_gaussian_likelihood_does_not_reward_diffuse_competitor(self):
        winners = {}
        for policy in ("mahalanobis", "gaussian_nll"):
            tracker = KalmanTrackManager(
                config(association_cost=policy, measurement_model="position_only")
            )
            tracker.update(batch(0, (0,), (candidate(0, 10, 10), candidate(1, 12, 10))))
            # Both gates accept the observation; a diffuse track's normalized
            # distance is smaller despite worse absolute localization.
            tracker._tracks[0].covariance = np.diag([1.0, 1.0, 1.0, 1.0])
            tracker._tracks[1].covariance = np.diag([400.0, 400.0, 1.0, 1.0])
            result = tracker.update(batch(100_000_000, (1,), (candidate(0, 10.5, 10),)))
            winners[policy] = result.associations[0]["track_id"]
        self.assertEqual(winners, {"mahalanobis": 1, "gaussian_nll": 0})

    def test_close_crossing_tracks_keep_ids_despite_fluctuating_strength(self):
        tracker = KalmanTrackManager(
            config(
                measurement_model="position_only",
                association_cost="gaussian_nll",
                birth_policy="spatial_fair",
                initial_velocity_sigma_px_s=30,
            )
        )
        for frame in range(30):
            points = (
                candidate(0, 20 + frame, 10, score=5 if frame % 2 else 50),
                candidate(1, 49 - frame, 13, score=50 if frame % 2 else 5),
            )
            result = tracker.update(batch(frame * 100_000_000, (frame,), points))
            self.assertEqual(len(result.tracks), 2)
            if frame == 0:
                ids = {round(t.state_xy_vx_vy[1]): t.track_id for t in result.tracks}
            else:
                self.assertEqual(
                    {a["track_id"]: a["candidate_index"] for a in result.associations},
                    {ids[10]: 0, ids[13]: 1},
                )

    def test_invalid_policies_fail_closed(self):
        for kwargs in (
            {"birth_policy": "magic"},
            {"association_cost": "magic"},
            {"birth_cell_size_px": 0},
            {"birth_cell_size_px": float("nan")},
        ):
            with self.assertRaises(ValueError):
                config(**kwargs)

    def test_prefix_requires_explicit_opt_in_and_exact_completion(self):
        path = (
            Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        )
        spec = importlib.util.spec_from_file_location("replay_capacity", path)
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        launch = dict(expected_frames=100, max_frames=20)
        report = dict(completed=True, full_clip=False, frames=20)
        with self.assertRaises(ValueError):
            replay.validate_parent(launch, report)
        replay.validate_parent(launch, report, True)
        for patch in (dict(frames=19), dict(completed=False), dict(full_clip=True)):
            with self.assertRaises(ValueError):
                replay.validate_parent(launch, {**report, **patch}, True)


if __name__ == "__main__":
    unittest.main()
