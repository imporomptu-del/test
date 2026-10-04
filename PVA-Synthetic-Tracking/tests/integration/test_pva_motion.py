from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tiny_target.motion import (
    GlobalMotionConfig,
    GlobalMotionTracker,
    PvaMotionConfig,
    PvaPyrLkMotionEstimator,
    fit_global_motion,
)
from tiny_target.types import Frame, TimestampSource


@unittest.skipIf(importlib.util.find_spec("vpi") is None, "NVIDIA VPI is not installed")
class PvaMotionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _frame(image: np.ndarray, index: int) -> Frame:
        return Frame(
            image=image,
            timestamp_ns=index * 100_000_000,
            frame_index=index,
            source_id="synthetic_motion",
            bit_depth=8,
            timestamp_source=TimestampSource.MANIFEST,
        )

    def test_known_translation_uses_pva_and_recovers_motion(self) -> None:
        rng = np.random.default_rng(75)
        previous_image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        current_image = np.roll(previous_image, shift=(2, 3), axis=(0, 1))
        previous = Frame(
            image=previous_image,
            timestamp_ns=0,
            frame_index=0,
            source_id="synthetic_translation",
            bit_depth=8,
            timestamp_source=TimestampSource.MANIFEST,
        )
        current = Frame(
            image=current_image,
            timestamp_ns=100_000_000,
            frame_index=1,
            source_id="synthetic_translation",
            bit_depth=8,
            timestamp_source=TimestampSource.MANIFEST,
        )
        estimator = PvaPyrLkMotionEstimator(
            PvaMotionConfig(
                feature_image_scale=1.0,
                max_features=200,
                grid_rows=4,
                grid_cols=6,
                harris_strength=20.0,
                minimum_accepted_features=20,
                minimum_grid_coverage=0.25,
            )
        )
        result = estimator.estimate(previous, current)
        median_motion = np.median(result.current_points - result.previous_points, axis=0)
        self.assertGreaterEqual(result.count, 20)
        np.testing.assert_allclose(median_motion, [3, 2], atol=0.35)
        self.assertEqual(result.backends["harris"], "PVA")
        self.assertEqual(result.backends["optical_flow_pyrlk"], "PVA")
        self.assertFalse(result.backends["cpu_fallback"])
        global_motion = fit_global_motion(
            result,
            GlobalMotionConfig(
                model="translation",
                minimum_correspondences=20,
                minimum_inliers=20,
                grid_rows=4,
                grid_cols=6,
                minimum_inlier_grid_coverage=0.25,
            ),
        )
        self.assertTrue(global_motion.accepted, global_motion.rejection_reasons)
        np.testing.assert_allclose(
            global_motion.previous_to_current_matrix[:2, 2], [3, 2], atol=0.15
        )
        chain = GlobalMotionTracker(
            GlobalMotionConfig(), initial_frame_index=0
        ).update(global_motion)
        np.testing.assert_allclose(
            chain.reference_from_current_matrix[:2, 2], [-3, -2], atol=0.15
        )

    def test_small_rotation_and_scale_produce_consistent_correspondences(self) -> None:
        import cv2  # type: ignore[import-not-found]

        rng = np.random.default_rng(76)
        previous_image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        expected = cv2.getRotationMatrix2D((319.5, 239.5), 0.5, 1.002)
        current_image = cv2.warpAffine(
            previous_image,
            expected,
            (640, 480),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        estimator = PvaPyrLkMotionEstimator(
            PvaMotionConfig(
                feature_image_scale=1.0,
                max_features=300,
                grid_rows=4,
                grid_cols=6,
                harris_strength=20.0,
                minimum_accepted_features=20,
                minimum_grid_coverage=0.25,
            )
        )
        result = estimator.estimate(
            self._frame(previous_image, 0), self._frame(current_image, 1)
        )
        homogeneous = np.column_stack(
            [result.previous_points, np.ones(result.count, np.float32)]
        )
        expected_current = homogeneous @ expected.T
        error = np.linalg.norm(result.current_points - expected_current, axis=1)
        self.assertGreaterEqual(result.count, 20)
        self.assertLess(float(np.median(error)), 0.4)
        self.assertLess(float(np.percentile(error, 95)), 1.0)

    def test_low_texture_fails_closed(self) -> None:
        from tiny_target.motion import PvaMotionError

        image = np.zeros((480, 640), np.uint8)
        estimator = PvaPyrLkMotionEstimator(
            PvaMotionConfig(feature_image_scale=1.0, harris_strength=20.0)
        )
        with self.assertRaisesRegex(PvaMotionError, "zero features"):
            estimator.estimate(self._frame(image, 0), self._frame(image, 1))


if __name__ == "__main__":
    unittest.main()
