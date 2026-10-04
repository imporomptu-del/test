from dataclasses import replace
import contextlib
import io
import json
import unittest

import numpy as np

from tiny_target.motion import (
    GlobalMotionConfig,
    GlobalMotionTracker,
    fit_global_motion,
)
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector
from tiny_target.visible_coverage import DetectionAvailability, finite_json
from test_global_motion import correspondences, grid_points


def strip(vertical=False, offset=0.0):
    rng = np.random.default_rng(801)
    centers = [[65 + 125 * i, 725 + offset] for i in range(8)]
    points = np.concatenate([rng.normal(c, [12, 8], (12, 2)) for c in centers])
    if vertical:
        points = points[:, ::-1] * [1.25, 0.8]
    return points


def pairs(previous, current, index=1, reasons=None):
    result = correspondences(
        previous,
        current,
        previous_index=index - 1,
        current_index=index,
        phase2_usable=False,
    )
    result.metrics["quality_rejection_reasons"] = (
        ["low_grid_coverage"] if reasons is None else reasons
    )
    return result


class SparseTranslationTests(unittest.TestCase):
    def config(self, **kwargs):
        return GlobalMotionConfig(
            coverage_policy="translation_consensus",
            minimum_inlier_grid_coverage=0.2,
            **kwargs
        )

    def test_sparse_horizontal_and_vertical_translations_and_shake(self):
        for vertical in (False, True):
            for offset in (0, -600):
                previous = strip(vertical, offset)
                for delta in ([0, 0], [11, -9], [-5, 3]):
                    current = previous + delta
                    match = pairs(previous, current)
                    self.assertFalse(
                        fit_global_motion(match, GlobalMotionConfig()).accepted
                    )
                    result = fit_global_motion(match, self.config())
                    self.assertTrue(result.accepted, result.rejection_reasons)
                    self.assertEqual(
                        result.metrics["coverage_acceptance_path"],
                        "sparse_translation_consensus",
                    )
                    np.testing.assert_allclose(
                        result.previous_to_current_matrix[:2, 2], delta, atol=1e-4
                    )

    def test_full_grid_default_path_unchanged(self):
        previous = grid_points()
        match = correspondences(previous, previous + [3, 1])
        old = fit_global_motion(match, GlobalMotionConfig())
        new = fit_global_motion(match, self.config())
        self.assertTrue(new.accepted)
        self.assertEqual(new.metrics["coverage_acceptance_path"], "full_grid")
        np.testing.assert_array_equal(
            old.previous_to_current_matrix, new.previous_to_current_matrix
        )
        np.testing.assert_array_equal(old.inlier_mask, new.inlier_mask)

    def test_one_region_cannot_dominate_despite_many_features(self):
        previous = strip()
        cluster = np.random.default_rng(4).normal([65, 725], [10, 5], (500, 2))
        previous = np.concatenate([previous, cluster])
        current = previous.copy()
        current[96:] += [4, 0]
        result = fit_global_motion(pairs(previous, current), self.config())
        self.assertFalse(result.accepted)
        self.assertIn(
            "point_weighted_fit_disagrees_with_cells",
            result.metrics["sparse_translation_support"]["rejection_reasons"],
        )

    def test_local_foreground_straddling_four_cells_is_rejected(self):
        rng = np.random.default_rng(5)
        previous = rng.normal([500, 400], [4, 4], (96, 2))
        result = fit_global_motion(pairs(previous, previous + [3, 1]), self.config())
        self.assertFalse(result.accepted)
        self.assertIn(
            "compact_spatial_support",
            result.metrics["sparse_translation_support"]["rejection_reasons"],
        )

    def test_local_foreground_region_does_not_veto_background_consensus(self):
        previous = strip()
        current = previous + [2, 1]
        current[:12] += [4, -2]
        result = fit_global_motion(pairs(previous, current), self.config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        support = result.metrics["sparse_translation_support"]
        self.assertEqual(support["consensus_cells"], 7)
        self.assertEqual(len(support["excluded_cell_ids"]), 1)
        np.testing.assert_allclose(
            result.previous_to_current_matrix[:2, 2], [2, 1], atol=1e-4
        )

    def test_split_spatial_motion_without_consensus_still_rejects(self):
        previous = strip()
        current = previous + [2, 1]
        current[:48] += [4, -2]
        result = fit_global_motion(pairs(previous, current), self.config())
        self.assertFalse(result.accepted)
        self.assertIn(
            "insufficient_held_out_consensus",
            result.metrics["sparse_translation_support"]["rejection_reasons"],
        )

    def test_agreeing_cells_must_span_scene_without_help_from_bad_cells(self):
        rng = np.random.default_rng(23)
        # Four nearby cells straddle one grid intersection; remote moving cells
        # must not lend their spatial extent to the accepted background subset.
        near = rng.normal([500, 400], [4, 4], (120, 2))
        far = np.concatenate(
            [
                rng.normal([80, 725], [5, 5], (6, 2)),
                rng.normal([920, 725], [5, 5], (6, 2)),
            ]
        )
        previous = np.concatenate([near, far])
        current = previous + [1, 1]
        current[120:] += [5, 0]
        result = fit_global_motion(pairs(previous, current), self.config())
        self.assertFalse(result.accepted)
        self.assertIn(
            "compact_spatial_support",
            result.metrics["sparse_translation_support"]["rejection_reasons"],
        )

    def test_isolated_bad_flows_do_not_veto_spatial_consensus(self):
        previous = strip()
        current = previous + [2, -1]
        current[::12] += [4, 3]  # One bad flow in each region, not a bad region.
        result = fit_global_motion(pairs(previous, current), self.config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        support = result.metrics["sparse_translation_support"]
        self.assertTrue(all(c["held_out_inlier_count"] == 11 for c in support["cells"]))
        np.testing.assert_allclose(
            result.previous_to_current_matrix[:2, 2], [2, -1], atol=1e-4
        )

    def test_noise_rotation_and_large_shift_fail(self):
        previous = strip()
        rng = np.random.default_rng(8)
        angle = 0.04
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        for current in (
            previous + rng.normal(0, 3, previous.shape),
            previous @ rotation.T,
            previous + [121, 0],
        ):
            result = fit_global_motion(pairs(previous, current), self.config())
            self.assertFalse(result.accepted)

    def test_unknown_quality_failure_and_too_few_points_stay_closed(self):
        previous = strip()
        for reasons in (
            [],
            ["low_grid_coverage", "insufficient_accepted_features"],
            ["unknown"],
        ):
            result = fit_global_motion(
                pairs(previous, previous + [2, 1], reasons=reasons), self.config()
            )
            self.assertFalse(result.accepted)
            self.assertIn("correspondence_quality_gate", result.rejection_reasons)
        result = fit_global_motion(
            pairs(previous[:20], previous[:20] + [2, 1]), self.config()
        )
        self.assertFalse(result.accepted)
        self.assertIn("insufficient_correspondences", result.rejection_reasons)

    def test_policy_is_translation_only_and_configuration_validated(self):
        for kwargs in (
            dict(model="similarity"),
            dict(sparse_minimum_cells=1),
            dict(sparse_minimum_span_fraction=float("nan")),
        ):
            with self.assertRaises(ValueError):
                self.config(**kwargs)

    def test_recovery_requires_fresh_background_after_genuinely_bad_pair(self):
        previous = strip()
        tracker = GlobalMotionTracker(self.config(), 0)
        detector = VisiblePointDetector(VisibleConfig(warmup_frames=2))
        base = np.full((96, 96), 30, np.float32)
        mask = np.ones_like(base, bool)
        for index in range(1, 10):
            match = pairs(previous, previous + [2, 1], index)
            if index == 5:
                match = pairs(previous, previous + [130, 0], index)
            state = tracker.update(fit_global_motion(match, self.config()))
            _, coverage = detector.update(base, mask, state.segment_index)
            self.assertEqual(state.window_reset, index == 5)
            if index in (3, 4, 7, 8, 9):
                self.assertFalse(coverage["warmup"])
            if index in (5, 6):
                self.assertTrue(coverage["warmup"])


class AvailabilityTests(unittest.TestCase):
    def test_completed_unavailable_cli_exits_nonzero_after_report(self):
        from unittest.mock import patch
        from tiny_target.visible_baseline import main

        report = dict(
            frames=20,
            processed_fps=1.0,
            qualified_track_count=0,
            detection_status="unavailable",
            availability={},
            full_clip=True,
        )
        with patch(
            "sys.argv",
            [
                "visible",
                "--source",
                "unused.avi",
                "--config",
                "unused.json",
                "--output",
                "unused",
            ],
        ), patch(
            "tiny_target.visible_baseline.run", return_value=report
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)

    def test_warmup_only_prefix_is_reported_without_full_run_success_claim(self):
        from unittest.mock import patch
        from tiny_target.visible_baseline import main

        report = dict(
            frames=4,
            processed_fps=1.0,
            qualified_track_count=0,
            detection_status="unavailable",
            availability={},
            full_clip=False,
        )
        with patch(
            "sys.argv",
            [
                "visible",
                "--source",
                "unused.avi",
                "--config",
                "unused.json",
                "--output",
                "unused",
                "--max-frames",
                "4",
            ],
        ), patch(
            "tiny_target.visible_baseline.run", return_value=report
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            main()

    def test_perpetual_reset_is_not_empty_success(self):
        monitor = DetectionAvailability()
        for _ in range(20):
            coverage = dict(warmup=True, searchable_pixels=0)
            monitor.update(
                coverage,
                dict(
                    reset=True,
                    accepted=False,
                    rejection_reasons=["low_inlier_grid_coverage"],
                ),
            )
            self.assertFalse(coverage["detection_ready"])
        report = monitor.report()
        self.assertEqual(report["detection_status"], "unavailable")
        self.assertEqual(report["longest_unavailable_streak_frames"], 20)
        self.assertEqual(
            report["motion_rejection_reason_counts"]["low_inlier_grid_coverage"], 20
        )

    def test_no_valid_pixels_not_usable_even_after_warmup(self):
        monitor = DetectionAvailability()
        monitor.update(dict(warmup=False, searchable_pixels=0), dict(reset=False))
        self.assertFalse(monitor.report()["usable_detection_coverage"])
        monitor.update(dict(warmup=False, searchable_pixels=100), dict(reset=False))
        self.assertEqual(monitor.report()["detection_status"], "available_unlabeled")

    def test_nonfinite_rejected_fit_metrics_are_serializable(self):
        cleaned = finite_json(dict(metrics=[float("inf"), float("nan"), 1.0]))
        self.assertEqual(
            json.loads(json.dumps(cleaned, allow_nan=False)),
            dict(metrics=[None, None, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
