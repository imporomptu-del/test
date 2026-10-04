from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tiny_target.motion import ComposedMotionState
from tiny_target.stabilization import (
    FullResolutionStabilizer,
    StabilizationConfig,
    ValidMaskWindow,
    alignment_improvement_metrics,
)
from tiny_target.types import Frame, TimestampSource


def frame(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(
        image=image,
        timestamp_ns=index * 100_000_000,
        frame_index=index,
        source_id="stabilization_fixture",
        bit_depth=16 if image.dtype == np.uint16 else 8,
        timestamp_source=TimestampSource.MANIFEST,
    )


def state(
    matrix: np.ndarray,
    index: int = 0,
    *,
    reference: int = 0,
    segment: int = 0,
    reset: bool = False,
) -> ComposedMotionState:
    return ComposedMotionState(
        reference_frame_index=reference,
        current_frame_index=index,
        segment_index=segment,
        reference_from_current_matrix=matrix,
        status="reset_reference" if reset else "accepted",
        window_reset=reset,
        reused_pairs=0,
        pair_parameter_delta=None,
    )


@unittest.skipIf(importlib.util.find_spec("cv2") is None, "OpenCV is not installed")
class StabilizationIntegrationTests(unittest.TestCase):
    def test_identity_preserves_values_without_resampling_and_erodes_mask(self) -> None:
        image = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
        result = FullResolutionStabilizer(
            StabilizationConfig(
                backend="opencv_cpu", valid_mask_erosion_px=2
            )
        ).stabilize(frame(image), state(np.eye(3)))
        self.assertEqual(result.frame.image.dtype, np.float32)
        np.testing.assert_array_equal(result.frame.image, image.astype(np.float32))
        self.assertEqual(result.resampling_count, 0)
        self.assertFalse(result.frame.valid_mask[0, 0])
        self.assertTrue(result.frame.valid_mask[2, 2])

    def test_integer_translation_moves_impulse_once_and_masks_border(self) -> None:
        image = np.zeros((32, 40), np.uint16)
        image[14, 15] = 4096
        matrix = np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], np.float64)
        result = FullResolutionStabilizer(
            StabilizationConfig(
                backend="opencv_cpu",
                interpolation="linear",
                valid_mask_erosion_px=1,
            )
        ).stabilize(frame(image), state(matrix))
        self.assertEqual(float(result.frame.image[12, 18]), 4096.0)
        self.assertEqual(float(np.sum(result.frame.image)), 4096.0)
        self.assertEqual(result.resampling_count, 1)
        self.assertFalse(np.any(result.frame.valid_mask[:, :4]))

    def test_subpixel_interpolations_preserve_flux_and_float_range(self) -> None:
        import cv2  # type: ignore[import-not-found]

        yy, xx = np.mgrid[:41, :41]
        psf = np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 0.8**2))
        psf = (psf / psf.sum() * 10000).astype(np.float32)
        matrix = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float64)
        peaks: dict[str, float] = {}
        for interpolation in ("linear", "cubic", "lanczos4"):
            result = FullResolutionStabilizer(
                StabilizationConfig(
                    backend="opencv_cpu",
                    interpolation=interpolation,
                    valid_mask_erosion_px=0,
                )
            ).stabilize(frame(psf), state(matrix))
            valid = result.frame.valid_mask
            peaks[interpolation] = float(np.max(result.frame.image[valid]))
            self.assertAlmostEqual(
                float(np.sum(result.frame.image[valid])),
                float(np.sum(psf)),
                delta=float(np.sum(psf)) * 0.005,
            )
            self.assertEqual(result.frame.image.dtype, np.float32)
        self.assertGreater(peaks["cubic"], peaks["linear"])
        self.assertGreater(peaks["lanczos4"], peaks["linear"])
        self.assertTrue(hasattr(cv2, "INTER_LANCZOS4"))

    def test_cpu_and_cuda_linear_warps_agree_when_cuda_is_available(self) -> None:
        import cv2  # type: ignore[import-not-found]

        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            self.skipTest("OpenCV CUDA device is unavailable")
        rng = np.random.default_rng(75)
        image = rng.integers(0, 4096, (96, 128), dtype=np.uint16)
        matrix = np.array(
            [[0.9999, -0.003, 1.25], [0.003, 0.9999, -0.75], [0, 0, 1]],
            np.float64,
        )
        cpu = FullResolutionStabilizer(
            StabilizationConfig(
                backend="opencv_cpu", interpolation="linear", valid_mask_erosion_px=2
            )
        ).stabilize(frame(image), state(matrix))
        cuda = FullResolutionStabilizer(
            StabilizationConfig(
                backend="opencv_cuda", interpolation="linear", valid_mask_erosion_px=2
            )
        ).stabilize(frame(image), state(matrix))
        common = cpu.frame.valid_mask & cuda.frame.valid_mask
        difference = np.abs(cpu.frame.image[common] - cuda.frame.image[common])
        relative_rmse = float(np.sqrt(np.mean(difference**2)) / 4095)
        self.assertLess(relative_rmse, 0.005)
        self.assertLess(float(np.percentile(difference, 99)), 64.0)
        self.assertLess(float(np.max(difference)), 128.0)
        np.testing.assert_array_equal(cpu.frame.valid_mask, cuda.frame.valid_mask)

    def test_inverse_round_trip_bound_and_actual_alignment_improvement(self) -> None:
        import cv2  # type: ignore[import-not-found]

        rng = np.random.default_rng(76)
        reference = cv2.GaussianBlur(
            rng.uniform(0, 4096, (160, 192)).astype(np.float32), (0, 0), 2
        )
        forward = np.array(
            [[1, 0, 3.25], [0, 1, -2.5], [0, 0, 1]], np.float64
        )
        current = cv2.warpPerspective(
            reference,
            forward,
            (192, 160),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        stabilizer = FullResolutionStabilizer(
            StabilizationConfig(
                backend="opencv_cpu", interpolation="linear", valid_mask_erosion_px=3
            )
        )
        reference_frame = frame(reference, 0)
        current_frame = frame(current, 1)
        stabilized_reference = stabilizer.stabilize(reference_frame, state(np.eye(3)))
        stabilized_current = stabilizer.stabilize(
            current_frame, state(np.linalg.inv(forward), 1)
        )
        metrics = alignment_improvement_metrics(
            reference_frame,
            current_frame,
            stabilized_reference,
            stabilized_current,
            sample_stride=1,
        )
        self.assertLess(
            metrics["median_absolute_difference_after"],
            metrics["median_absolute_difference_before"] * 0.15,
        )

        round_trip = stabilizer.stabilize(
            stabilized_current.frame,
            state(forward, 1),
        )
        valid = round_trip.frame.valid_mask
        rmse = np.sqrt(np.mean((round_trip.frame.image[valid] - current[valid]) ** 2))
        self.assertLess(float(rmse), 50.0)

    def test_valid_support_window_resets_between_segments(self) -> None:
        stabilizer = FullResolutionStabilizer(
            StabilizationConfig(backend="opencv_cpu", valid_mask_erosion_px=0)
        )
        window = ValidMaskWindow(maximum_frames=3)
        first = stabilizer.stabilize(
            frame(np.ones((8, 8), np.uint16), 0), state(np.eye(3), 0)
        )
        shifted = stabilizer.stabilize(
            frame(np.ones((8, 8), np.uint16), 1),
            state(np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]]), 1),
        )
        window.update(first)
        support = window.update(shifted)
        self.assertEqual(support.frame_count, 2)
        self.assertLess(float(np.mean(support.common_valid_mask)), 1.0)
        reset = stabilizer.stabilize(
            frame(np.ones((8, 8), np.uint16), 2),
            state(np.eye(3), 2, reference=2, segment=1, reset=True),
        )
        support = window.update(reset)
        self.assertEqual(support.frame_count, 1)
        self.assertEqual(support.segment_index, 1)


if __name__ == "__main__":
    unittest.main()
