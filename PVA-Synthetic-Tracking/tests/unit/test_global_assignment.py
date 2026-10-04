"""Gated global matching: exhaustive tiny cases and neighboring trajectories."""
import itertools
import unittest
import numpy as np
from tiny_target.tracking.kalman import minimum_cost_pairs
from tiny_target.tracking import KalmanTrackManager
from tiny_target.visible_baseline import VisibleConfig
from test_kalman_tracking import config, batch, candidate


class GlobalAssignmentTests(unittest.TestCase):
    def test_greedy_order_trap(self):
        self.assertEqual(
            minimum_cost_pairs([(0, 0, 1), (0, 1, 2), (1, 0, 2)]), {(0, 1), (1, 0)}
        )

    def test_total_cost_not_first_pair(self):
        self.assertEqual(
            minimum_cost_pairs([(0, 0, 1), (0, 1, 2), (1, 0, 2), (1, 1, 9)]),
            {(0, 1), (1, 0)},
        )

    def test_exhaustive_random_small_graphs(self):
        rng = np.random.default_rng(219)
        for _ in range(100):
            edges = [
                (i, j, float(rng.integers(-5, 15)))
                for i in range(3)
                for j in range(3)
                if rng.random() < 0.6
            ]
            costs = {(i, j): c for i, j, c in edges}
            best = (0, 0)
            for choice in itertools.product(range(-1, 3), repeat=3):
                pairs = {(i, j) for i, j in enumerate(choice) if j >= 0}
                if (
                    len({j for _, j in pairs}) != len(pairs)
                    or not pairs <= costs.keys()
                ):
                    continue
                best = min(best, (-len(pairs), sum(costs[p] for p in pairs)))
            actual = minimum_cost_pairs(edges)
            self.assertEqual((-len(actual), sum(costs[p] for p in actual)), best)
            self.assertEqual(actual, minimum_cost_pairs(list(reversed(edges))))

    def test_empty_and_single_observation_are_not_fabricated(self):
        self.assertEqual(minimum_cost_pairs([]), set())
        self.assertEqual(len(minimum_cost_pairs([(0, 0, 1), (1, 0, 2)])), 1)
        with self.assertRaises(ValueError):
            minimum_cost_pairs([(0, 0, float("nan"))])

    def test_neighboring_movers_stay_separate_and_gap_is_prediction(self):
        tracker = KalmanTrackManager(
            config(
                measurement_model="position_only",
                association_assignment="global_min_cost",
            )
        )
        for f in range(6):
            r = tracker.update(
                batch(
                    f * 100_000_000,
                    (f,),
                    (candidate(0, 10 + f, 10), candidate(1, 10 + f, 16)),
                )
            )
            if f:
                self.assertEqual(
                    {a["track_id"]: a["candidate_index"] for a in r.associations},
                    {0: 0, 1: 1},
                )
        r = tracker.update(batch(600_000_000, (6,), (candidate(0, 16, 10),)))
        self.assertEqual(len(r.associations), 1)
        self.assertEqual(r.associations[0]["track_id"], 0)
        self.assertNotEqual(r.tracks[1].last_measurement_timestamp_ns, 600_000_000)

    def test_invalid_combinations_and_default(self):
        self.assertEqual(VisibleConfig().tracking_association_assignment, "greedy")
        with self.assertRaises(ValueError):
            config(association_assignment="unknown")
        with self.assertRaises(ValueError):
            config(
                association_assignment="global_min_cost",
                association_cascade="confirmed_first",
            )
        with self.assertRaises(ValueError):
            VisibleConfig(tracking_association_assignment=True)


if __name__ == "__main__":
    unittest.main()
