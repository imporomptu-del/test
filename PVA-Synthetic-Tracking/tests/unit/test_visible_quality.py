import importlib.util
from pathlib import Path
import unittest

import numpy as np

from tiny_target.visible_baseline import VisibleConfig, VisibleTracks
from tiny_target.visible_quality import CausalMotionQuality, suppress_nearby


class VisibleQualityTests(unittest.TestCase):
    def test_strict_runner_rejects_anchor_loss_and_empty_evidence(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "scripts/run_phase20_visible_regression.py"
        )
        spec = importlib.util.spec_from_file_location("runner", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertFalse(runner.all_confident_anchors_pass([]))
        for hits, expected in ((12, True), (11, False), (9, False)):
            self.assertEqual(
                runner.all_confident_anchors_pass(
                    [
                        {
                            "events": [
                                {
                                    "dominant_track_anchor_hits": hits,
                                    "required_anchor_count": 12,
                                }
                            ]
                        }
                    ]
                ),
                expected,
            )

    def test_accelerating_turn_with_irregular_gaps_passes(self):
        quality = CausalMotionQuality()
        for i in [0, 1, 3, 5, 8, 9, 12, 15, 16]:
            t = i / 10
            result = quality.observe(
                i * 100_000_000, 20 + 10 * t - 8 * t * t, 80 - 2 * t * t
            )
        self.assertTrue(result["passed"])
        self.assertLess(result["quadratic_fit_rmse_px"], 1e-8)
        self.assertEqual(result["measured_history_count"], 8)

    def test_random_hops_fail_and_four_hits_are_insufficient(self):
        quality = CausalMotionQuality()
        for i, y in enumerate([0, 20, -20, 20, -20, 20, -20, 20]):
            result = quality.observe(i * 100_000_000, i * 2, y)
            if i < 4:
                self.assertFalse(result["ready"])
        self.assertFalse(result["passed"])

    def test_bad_observations_can_revoke_previously_good_quality(self):
        quality = CausalMotionQuality()
        for i in range(8):
            result = quality.observe(i * 100_000_000, i * 2, 30)
        self.assertTrue(result["passed"])
        result = quality.observe(800_000_000, 16, 60)
        self.assertFalse(result["passed"])

    def test_predictions_do_not_add_consistency_evidence(self):
        config = VisibleConfig(motion_quality_enabled=True)
        tracker = VisibleTracks(config, 10)

        def p(x):
            return dict(x=x, y=50, polarity="bright", score=10, response_dn=5)

        for i in range(4):
            records, _ = tracker.update(
                [p(20 + 8 * i)], i, i * 100_000_000, 0, np.eye(3), (100, 200)
            )
        self.assertFalse(records[0]["qualified_moving"])
        records, _ = tracker.update([], 4, 400_000_000, 0, np.eye(3), (100, 200))
        self.assertEqual(records[0]["motion_quality"]["measured_history_count"], 4)
        self.assertFalse(records[0]["qualified_moving"])

    def test_resolution_nms_preserves_provenance_polarity_and_separated_points(self):
        def p(x, score, polarity="bright"):
            return dict(x=x, y=10, score=score, polarity=polarity)

        proposals = [p(10, 5), p(12, 9), p(12, 5, "dark"), p(20, 5)]
        kept, dropped = suppress_nearby(proposals, 4)
        self.assertEqual(kept, proposals[1:])
        self.assertEqual(dropped[0]["candidate_index"], 0)
        self.assertEqual(dropped[0]["retained_candidate_index"], 1)
        self.assertEqual(suppress_nearby(proposals, 0), (proposals, []))

    def test_replay_disallows_changed_detection_and_motion(self):
        path = (
            Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        )
        spec = importlib.util.spec_from_file_location("replay", path)
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        self.assertEqual(
            replay.validate_configuration(
                {}, VisibleConfig(motion_quality_enabled=True)
            ),
            ["motion_quality_enabled"],
        )
        for kwargs in (
            {"temporal_threshold_sigma": 5},
            {"motion_backend": "pva"},
            {"spatial_background": "median5"},
        ):
            with self.assertRaises(ValueError):
                replay.validate_configuration({}, VisibleConfig(**kwargs))


if __name__ == "__main__":
    unittest.main()
