from __future__ import annotations

import math
import unittest

import numpy as np

from tiny_target.motion import (
    GlobalMotionConfig,
    GlobalMotionEstimate,
    GlobalMotionTracker,
    MotionCorrespondences,
    fit_global_motion,
)


def correspondences(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    previous_index: int = 0,
    current_index: int = 1,
    image_size: tuple[int, int] = (1000, 800),
    phase2_usable: bool | None = None,
) -> MotionCorrespondences:
    metrics = {}
    if phase2_usable is not None:
        metrics["usable_for_transform"] = phase2_usable
    return MotionCorrespondences(
        previous_points=previous,
        current_points=current,
        harris_scores=np.ones(len(previous), np.float32),
        forward_backward_error_px=np.zeros(len(previous), np.float32),
        previous_frame_index=previous_index,
        current_frame_index=current_index,
        previous_timestamp_ns=previous_index * 100_000_000,
        current_timestamp_ns=current_index * 100_000_000,
        full_image_size=image_size,
        motion_image_size=image_size,
        metrics=metrics,
        timings_ms={},
        backends={},
    )


def grid_points() -> np.ndarray:
    x, y = np.meshgrid(np.linspace(50, 950, 10), np.linspace(50, 750, 8))
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ matrix.T)[:, :2].astype(np.float32)


def accepted_estimate(
    previous_index: int,
    current_index: int,
    matrix: np.ndarray,
) -> GlobalMotionEstimate:
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    return GlobalMotionEstimate(
        model="translation",
        previous_frame_index=previous_index,
        current_frame_index=current_index,
        previous_to_current_matrix=matrix,
        inlier_mask=np.ones(1, bool),
        residuals_px=np.zeros(1),
        parameters={
            "translation_x_px": tx,
            "translation_y_px": ty,
            "translation_magnitude_px": math.hypot(tx, ty),
            "rotation_deg": 0.0,
            "scale": 1.0,
            "shear": 0.0,
            "perspective": 0.0,
        },
        metrics={},
        quality_status="accepted",
        rejection_reasons=(),
        timing_ms=0.0,
    )


def rejected_estimate(previous_index: int, current_index: int) -> GlobalMotionEstimate:
    return GlobalMotionEstimate(
        model="translation",
        previous_frame_index=previous_index,
        current_frame_index=current_index,
        previous_to_current_matrix=None,
        inlier_mask=np.zeros(0, bool),
        residuals_px=np.zeros(0),
        parameters=None,
        metrics={},
        quality_status="rejected",
        rejection_reasons=("fixture_failure",),
        timing_ms=0.0,
    )


class GlobalMotionRansacTests(unittest.TestCase):
    def test_translation_recovers_with_outliers_and_is_deterministic(self) -> None:
        previous = grid_points()
        rng = np.random.default_rng(75)
        for outlier_fraction in (0.0, 0.2, 0.4, 0.6):
            current = previous + np.array([4.25, -2.5], np.float32)
            count = round(len(previous) * outlier_fraction)
            if count:
                indices = rng.choice(len(previous), count, replace=False)
                current[indices] = rng.uniform([0, 0], [1000, 800], (count, 2))
            config = GlobalMotionConfig(
                model="translation",
                ransac_reprojection_px=0.1,
                minimum_correspondences=10,
                minimum_inliers=10,
                minimum_inlier_ratio=0.35,
                minimum_inlier_grid_coverage=0.20,
            )
            first = fit_global_motion(correspondences(previous, current), config)
            second = fit_global_motion(correspondences(previous, current), config)
            self.assertTrue(first.accepted, (outlier_fraction, first.rejection_reasons))
            np.testing.assert_allclose(
                first.previous_to_current_matrix[:2, 2], [4.25, -2.5], atol=1e-5
            )
            np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)
            np.testing.assert_array_equal(
                first.previous_to_current_matrix, second.previous_to_current_matrix
            )

    def test_similarity_recovers_rotation_scale_and_translation(self) -> None:
        previous = grid_points().astype(np.float64)
        angle = math.radians(1.2)
        scale = 1.005
        a = scale * math.cos(angle)
        b = scale * math.sin(angle)
        expected = np.array(
            [[a, -b, 3.0], [b, a, -4.0], [0.0, 0.0, 1.0]], np.float64
        )
        current = transform_points(previous, expected)
        rng = np.random.default_rng(76)
        current += rng.normal(0, 0.02, current.shape).astype(np.float32)
        outliers = rng.choice(len(previous), 20, replace=False)
        current[outliers] = rng.uniform([0, 0], [1000, 800], (20, 2))
        result = fit_global_motion(
            correspondences(previous, current),
            GlobalMotionConfig(
                model="similarity",
                ransac_reprojection_px=0.2,
                minimum_correspondences=10,
                minimum_inliers=30,
                minimum_inlier_ratio=0.6,
                minimum_inlier_grid_coverage=0.25,
                maximum_rotation_deg=2.0,
            ),
        )
        self.assertTrue(result.accepted, result.rejection_reasons)
        np.testing.assert_allclose(result.previous_to_current_matrix, expected, atol=0.01)
        self.assertAlmostEqual(result.parameters["rotation_deg"], 1.2, places=2)
        self.assertAlmostEqual(result.parameters["scale"], 1.005, places=3)

    def test_collinear_similarity_and_clustered_translation_fail_closed(self) -> None:
        previous = np.column_stack([np.linspace(100, 900, 40), np.full(40, 100)])
        similarity = fit_global_motion(
            correspondences(previous, previous + [2, 1]),
            GlobalMotionConfig(
                model="similarity",
                minimum_correspondences=10,
                minimum_inliers=10,
                minimum_inlier_grid_coverage=0.0,
                minimum_noncollinearity_ratio=0.05,
            ),
        )
        self.assertFalse(similarity.accepted)
        self.assertIn("degenerate_inlier_geometry", similarity.rejection_reasons)

        clustered = np.column_stack(
            [np.linspace(10, 80, 40), np.linspace(10, 80, 40)]
        )
        translation = fit_global_motion(
            correspondences(clustered, clustered + [2, 1]),
            GlobalMotionConfig(
                model="translation",
                minimum_correspondences=10,
                minimum_inliers=10,
                minimum_inlier_grid_coverage=0.25,
            ),
        )
        self.assertFalse(translation.accepted)
        self.assertIn("low_inlier_grid_coverage", translation.rejection_reasons)

    def test_phase2_gate_and_parameter_limits_are_enforced(self) -> None:
        previous = grid_points()
        result = fit_global_motion(
            correspondences(previous, previous + [200, 0], phase2_usable=False),
            GlobalMotionConfig(
                minimum_correspondences=10,
                minimum_inliers=10,
                minimum_inlier_grid_coverage=0.2,
                maximum_translation_px=100,
            ),
        )
        self.assertFalse(result.accepted)
        self.assertIn("correspondence_quality_gate", result.rejection_reasons)
        self.assertIn("translation_limit", result.rejection_reasons)


class TransformCompositionTests(unittest.TestCase):
    def test_long_translation_chain_and_inverse_round_trip(self) -> None:
        config = GlobalMotionConfig()
        tracker = GlobalMotionTracker(config, initial_frame_index=0)
        pair = np.array([[1, 0, 1], [0, 1, 0.5], [0, 0, 1]], np.float64)
        for index in range(100):
            state = tracker.update(accepted_estimate(index, index + 1, pair))
        np.testing.assert_allclose(
            state.reference_from_current_matrix,
            [[1, 0, -100], [0, 1, -50], [0, 0, 1]],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            state.reference_from_current_matrix
            @ np.linalg.inv(state.reference_from_current_matrix),
            np.eye(3),
            atol=1e-12,
        )

    def test_rejected_pair_resets_reference_by_default(self) -> None:
        tracker = GlobalMotionTracker(GlobalMotionConfig(), initial_frame_index=0)
        pair = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], np.float64)
        tracker.update(accepted_estimate(0, 1, pair))
        reset = tracker.update(rejected_estimate(1, 2))
        self.assertEqual(reset.status, "reset_reference")
        self.assertTrue(reset.window_reset)
        self.assertEqual(reset.reference_frame_index, 2)
        np.testing.assert_array_equal(reset.reference_from_current_matrix, np.eye(3))
        recovered = tracker.update(accepted_estimate(2, 3, pair))
        self.assertEqual(recovered.status, "accepted")
        self.assertEqual(recovered.reference_frame_index, 2)
        self.assertEqual(recovered.segment_index, 1)
        np.testing.assert_allclose(
            recovered.reference_from_current_matrix,
            [[1, 0, -2], [0, 1, -1], [0, 0, 1]],
        )

    def test_previous_transform_reuse_is_strictly_bounded(self) -> None:
        tracker = GlobalMotionTracker(
            GlobalMotionConfig(failure_policy="reuse_previous", maximum_reuse_pairs=1),
            initial_frame_index=0,
        )
        held = tracker.update(rejected_estimate(0, 1))
        self.assertEqual(held.status, "reused_previous")
        reset = tracker.update(rejected_estimate(1, 2))
        self.assertEqual(reset.status, "reset_reference")
        self.assertTrue(reset.window_reset)


if __name__ == "__main__":
    unittest.main()
