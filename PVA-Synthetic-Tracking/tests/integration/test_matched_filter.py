from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tiny_target.detection import (
    MatchedFilterConfig,
    PsfMatchedFilter,
    build_kernel_bank,
    correlate_reference,
)
from tiny_target.preprocessing import ResidualFrame


def residual(
    whitened: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    ready: bool = True,
) -> ResidualFrame:
    if valid is None:
        valid = np.ones(whitened.shape, bool) if ready else np.zeros(whitened.shape, bool)
    return ResidualFrame(
        value=np.asarray(whitened, np.float32),
        sigma=np.ones(whitened.shape, np.float32),
        whitened=np.asarray(whitened, np.float32),
        valid_mask=np.asarray(valid, bool),
        timestamp_ns=100,
        frame_index=1,
        reference_frame_index=0,
        segment_index=0,
        detection_ready=ready,
        history_frames=8,
        background_method="fixture",
        noise_method="fixture",
        metrics={},
        timings_ms={},
    )


def place_kernel(shape: tuple[int, int], kernel: np.ndarray, y: int, x: int, flux: float) -> np.ndarray:
    image = np.zeros(shape, np.float32)
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    image[y - ry : y + ry + 1, x - rx : x + rx + 1] = flux * kernel
    return image


class MatchedFilterIntegrationTests(unittest.TestCase):
    def test_reference_correlation_orientation(self) -> None:
        image = np.zeros((7, 8), np.float32)
        image[3, 4] = 2
        kernel = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]], np.float32)
        output = correlate_reference(image, kernel)
        self.assertEqual(float(output[3, 3]), 4.0)
        self.assertEqual(float(output[3, 4]), 2.0)

    def test_known_psf_has_expected_snr_gain_and_flux_linearity(self) -> None:
        config = MatchedFilterConfig(
            backend="numpy_reference", phases_per_axis=1, polarity="bright"
        )
        detector = PsfMatchedFilter(config)
        kernel = detector.bank.kernels[0]
        l2 = float(np.sqrt(np.sum(kernel.astype(np.float64) ** 2)))
        outputs = []
        for flux in (10.0, 25.0):
            image = place_kernel((31, 35), kernel, 15, 17, flux)
            result = detector.process(residual(image))
            outputs.append(float(result.response[15, 17]))
            self.assertAlmostEqual(outputs[-1], flux * l2, places=5)
            single_pixel_snr = flux * float(np.max(kernel))
            self.assertGreater(outputs[-1], single_pixel_snr)
        self.assertAlmostEqual(outputs[1] / outputs[0], 2.5, places=5)

    def test_fractional_phase_bank_selects_matching_template(self) -> None:
        detector = PsfMatchedFilter(
            MatchedFilterConfig(
                backend="numpy_reference", phases_per_axis=2, polarity="bright"
            )
        )
        phase = 3
        image = place_kernel((33, 37), detector.bank.kernels[phase], 16, 18, 30)
        result = detector.process(residual(image))
        self.assertEqual(int(result.phase_index[16, 18]), phase)
        self.assertEqual(result.kernel_metadata["phase_count"], 4)

    def test_signed_dark_response_and_polarity_score(self) -> None:
        detector = PsfMatchedFilter(
            MatchedFilterConfig(backend="numpy_reference", polarity="dark")
        )
        kernel = detector.bank.kernels[0]
        image = place_kernel((25, 25), kernel, 12, 12, -20)
        result = detector.process(residual(image))
        self.assertLess(float(result.response[12, 12]), 0)
        self.assertGreater(float(result.score()[12, 12]), 0)

    def test_incomplete_kernel_support_is_invalid(self) -> None:
        detector = PsfMatchedFilter(
            MatchedFilterConfig(backend="numpy_reference", radius_px=2)
        )
        valid = np.ones((15, 17), bool)
        valid[7, 8] = False
        result = detector.process(residual(np.zeros((15, 17), np.float32), valid=valid))
        self.assertFalse(result.valid_mask[7, 8])
        self.assertFalse(result.valid_mask[7, 6])
        self.assertTrue(result.valid_mask[7, 11])
        self.assertEqual(result.metrics["support"]["minimum_required"], 25)

    def test_warmup_frame_skips_filtering(self) -> None:
        detector = PsfMatchedFilter(MatchedFilterConfig())
        result = detector.process(residual(np.ones((12, 13), np.float32), ready=False))
        self.assertFalse(result.detection_ready)
        self.assertFalse(np.any(result.valid_mask))
        self.assertEqual(result.timings_ms["suppressed_no_filter"], 0.0)

    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "OpenCV is not installed")
    def test_opencv_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(75)
        image = rng.normal(0, 1, (48, 64)).astype(np.float32)
        valid = rng.random((48, 64)) > 0.05
        config = dict(
            gaussian_sigma_px=0.8,
            radius_px=3,
            phases_per_axis=2,
            polarity="both",
        )
        reference = PsfMatchedFilter(
            MatchedFilterConfig(backend="numpy_reference", **config)
        ).process(residual(image, valid=valid))
        optimized = PsfMatchedFilter(
            MatchedFilterConfig(backend="opencv_cpu", **config)
        ).process(residual(image, valid=valid))
        np.testing.assert_array_equal(reference.valid_mask, optimized.valid_mask)
        np.testing.assert_array_equal(reference.phase_index, optimized.phase_index)
        common = reference.valid_mask
        np.testing.assert_allclose(
            reference.response[common], optimized.response[common], rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
