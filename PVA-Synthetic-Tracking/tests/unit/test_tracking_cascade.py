"""Confirmation-aware association experiments, not physical identity claims."""
import unittest
import numpy as np
from test_kalman_tracking import config, batch, candidate
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig


class CascadeTests(unittest.TestCase):
    def make_tracker(self, policy):
        tracker = KalmanTrackManager(
            config(
                measurement_model="position_only",
                association_cost="gaussian_nll",
                association_cascade=policy,
            )
        )
        tracker.update(batch(0, (0,), (candidate(0, 10, 10), candidate(1, 12, 10))))
        for t in tracker._tracks.values():
            t.covariance = np.eye(4)
        tracker._tracks[0].confirmation_timestamp_ns = 0
        tracker._tracks[0].lifecycle_state = "confirmed"
        return tracker

    def test_confirmed_priority_is_opt_in(self):
        winners = {}
        for policy in ("none", "confirmed_first"):
            t = self.make_tracker(policy)
            r = t.update(batch(100_000_000, (1,), (candidate(0, 12, 10),)))
            winners[policy] = r.associations[0]["track_id"]
        self.assertEqual(winners, {"none": 1, "confirmed_first": 0})

    def test_unavailable_gate_never_becomes_prediction_measurement(self):
        t = self.make_tracker("confirmed_first")
        t._tracks[0].mean[0] = -100
        r = t.update(batch(100_000_000, (1,), (candidate(0, 12, 10),)))
        self.assertEqual(r.associations[0]["track_id"], 1)
        self.assertNotEqual(r.tracks[0].last_measurement_timestamp_ns, 100_000_000)

    def test_two_separable_points_remain_two_measurements(self):
        t = self.make_tracker("confirmed_first")
        r = t.update(
            batch(100_000_000, (1,), (candidate(0, 10, 10), candidate(1, 12, 10)))
        )
        self.assertEqual(len(r.associations), 2)
        self.assertEqual(len({a["candidate_index"] for a in r.associations}), 2)

    def test_invalid_policy_rejected_and_default_unchanged(self):
        self.assertEqual(VisibleConfig().tracking_association_cascade, "none")
        with self.assertRaises(ValueError):
            config(association_cascade="oldest")
        with self.assertRaises(ValueError):
            VisibleConfig(tracking_association_cascade=True)


if __name__ == "__main__":
    unittest.main()
