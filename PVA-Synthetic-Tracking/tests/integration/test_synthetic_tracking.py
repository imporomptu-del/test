from __future__ import annotations

import unittest

import numpy as np

from tiny_target.detection import (
    MatchedFilterFrame,
    ReferenceShiftAndStack,
    ReferenceSyntheticWindow,
    SyntheticTrackingConfig,
    SyntheticTrackingError,
)


def matched(
    response: np.ndarray,
    index: int,
    timestamp_ns: int,
    *,
    valid: np.ndarray | None = None,
    ready: bool = True,
    segment: int = 0,
    polarity: str = "bright",
) -> MatchedFilterFrame:
    if valid is None:
        valid = np.ones(response.shape, bool) if ready else np.zeros(response.shape, bool)
    return MatchedFilterFrame(
        response=np.asarray(response, np.float32),
        phase_index=np.zeros(response.shape, np.uint16),
        valid_mask=np.asarray(valid, bool),
        valid_support_count=np.asarray(valid, np.uint16),
        timestamp_ns=timestamp_ns,
        frame_index=index,
        reference_frame_index=0,
        segment_index=segment,
        detection_ready=ready,
        polarity=polarity,
        backend="fixture",
        kernel_metadata={},
        metrics={},
        timings_ms={},
    )


def config(
    frames: int,
    *,
    vx: tuple[float, float] = (0, 0),
    vy: tuple[float, float] = (0, 0),
    step: float = 1,
    support: float = 1,
    stride: int = 1,
) -> SyntheticTrackingConfig:
    return SyntheticTrackingConfig(
        window_frames=frames,
        window_stride_frames=stride,
        vx_min_px_s=vx[0],
        vx_max_px_s=vx[1],
        vy_min_px_s=vy[0],
        vy_max_px_s=vy[1],
        velocity_step_px_s=step,
        min_valid_fraction=support,
        tile_rows=5,
    )


def gaussian_response(
    shape: tuple[int, int], x: float, y: float, amplitude: float = 8
) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (
        amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.7**2))
    ).astype(np.float32)


class SyntheticTrackingIntegrationTests(unittest.TestCase):
    def test_zero_velocity_golden_score_and_sqrt_n_scaling(self) -> None:
        shape = (11, 13)
        for count in (1, 2, 4):
            frames = []
            for index in range(count):
                response = np.zeros(shape, np.float32)
                response[5, 6] = 3
                frames.append(matched(response, index, index * 100_000_000))
            result = ReferenceShiftAndStack(config(count)).integrate(frames)
            self.assertAlmostEqual(
                float(result.score[5, 6]), 3 * np.sqrt(count), delta=1e-6
            )
            self.assertEqual(int(result.valid_support_count[5, 6]), count)
            self.assertEqual(result.metrics["velocity_trial_count"], 1)
            self.assertEqual(
                result.metrics["maximum_velocity_quantization_endpoint_error_px"],
                0.0,
            )

    def test_irregular_timestamps_recover_fractional_diagonal_velocity(self) -> None:
        timestamps = [0, 400_000_000, 1_000_000_000, 1_700_000_000, 2_500_000_000]
        reference_ns = (timestamps[0] + timestamps[-1]) // 2
        true_x, true_y = 20.0, 19.0
        true_vx, true_vy = 2.0, -1.0
        frames = []
        for index, timestamp in enumerate(timestamps):
            dt = (timestamp - reference_ns) / 1e9
            response = gaussian_response(
                (39, 41), true_x + true_vx * dt, true_y + true_vy * dt
            )
            frames.append(matched(response, index, timestamp))
        tracker = ReferenceShiftAndStack(
            config(5, vx=(-3, 3), vy=(-3, 3), step=1)
        )
        result = tracker.integrate(frames)
        peak_y, peak_x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
        velocity = result.velocity_grid_xy_px_s[result.velocity_index[peak_y, peak_x]]
        self.assertLessEqual(abs(peak_x - true_x), 1)
        self.assertLessEqual(abs(peak_y - true_y), 1)
        np.testing.assert_array_equal(velocity, [true_vx, true_vy])
        self.assertEqual(result.reference_timestamp_ns, reference_ns)

    def test_axis_aligned_velocity_is_recovered(self) -> None:
        timestamps = [0, 500_000_000, 1_000_000_000, 1_500_000_000]
        reference_ns = (timestamps[0] + timestamps[-1]) // 2
        frames = []
        for index, timestamp in enumerate(timestamps):
            dt = (timestamp - reference_ns) / 1e9
            frames.append(
                matched(
                    gaussian_response((31, 33), 16 + 2 * dt, 15),
                    index,
                    timestamp,
                )
            )
        result = ReferenceShiftAndStack(
            config(4, vx=(-2, 2), vy=(0, 0), step=1)
        ).integrate(frames)
        y, x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
        selected = result.velocity_grid_xy_px_s[result.velocity_index[y, x]]
        np.testing.assert_array_equal(selected, [2, 0])
        self.assertEqual(
            result.metrics["maximum_velocity_quantization_endpoint_error_xy_px"],
            (0.75, 0.0),
        )

    def test_dark_polarity_uses_detection_sign_and_both_is_rejected(self) -> None:
        response = np.zeros((9, 11), np.float32)
        response[4, 5] = -3
        dark_frames = [
            matched(response, index, index * 100, polarity="dark")
            for index in range(4)
        ]
        dark = ReferenceShiftAndStack(config(4)).integrate(dark_frames)
        self.assertAlmostEqual(float(dark.score[4, 5]), 6.0, delta=1e-6)

        both_frames = [
            matched(response, index, index * 100, polarity="both")
            for index in range(4)
        ]
        with self.assertRaisesRegex(SyntheticTrackingError, "cannot be coherently"):
            ReferenceShiftAndStack(config(4)).integrate(both_frames)

    def test_velocity_halfway_between_trials_selects_an_adjacent_trial(self) -> None:
        timestamps = [0, 250_000_000, 500_000_000, 750_000_000]
        reference_ns = (timestamps[0] + timestamps[-1]) // 2
        frames = []
        for index, timestamp in enumerate(timestamps):
            dt = (timestamp - reference_ns) / 1e9
            frames.append(
                matched(
                    gaussian_response((25, 27), 13 + 0.5 * dt, 12),
                    index,
                    timestamp,
                )
            )
        result = ReferenceShiftAndStack(
            config(4, vx=(0, 1), step=1)
        ).integrate(frames)
        y, x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
        selected = float(result.velocity_grid_xy_px_s[result.velocity_index[y, x], 0])
        self.assertIn(selected, (0.0, 1.0))
        self.assertAlmostEqual(
            result.metrics["maximum_velocity_quantization_endpoint_error_px"],
            0.375,
        )

    def test_boundary_uses_only_configured_temporal_support(self) -> None:
        frames = []
        for index in range(4):
            response = np.ones((8, 9), np.float32)
            valid = np.ones((8, 9), bool)
            if index == 0:
                valid[:, 0] = False
            frames.append(matched(response, index, index * 100_000_000, valid=valid))
        strict = ReferenceShiftAndStack(config(4, support=1)).integrate(frames)
        relaxed = ReferenceShiftAndStack(config(4, support=0.75)).integrate(frames)
        self.assertFalse(strict.valid_mask[4, 0])
        self.assertTrue(relaxed.valid_mask[4, 0])
        self.assertEqual(int(relaxed.valid_support_count[4, 0]), 3)
        self.assertAlmostEqual(float(relaxed.score[4, 0]), np.sqrt(3), delta=1e-6)

    def test_streaming_windows_respect_stride_segment_and_suppression(self) -> None:
        window = ReferenceSyntheticWindow(
            ReferenceShiftAndStack(config(3, stride=2))
        )
        image = np.ones((7, 8), np.float32)
        outputs = []
        for index in range(5):
            output = window.update(matched(image, index, index * 100))
            if output is not None:
                outputs.append(output.frame_indices)
        self.assertEqual(outputs, [(0, 1, 2), (2, 3, 4)])
        self.assertIsNone(
            window.update(matched(image, 5, 500, ready=False))
        )
        self.assertIsNone(window.update(matched(image, 6, 600, segment=1)))
        self.assertIsNone(window.update(matched(image, 7, 700, segment=1)))
        output = window.update(matched(image, 8, 800, segment=1))
        self.assertEqual(output.frame_indices, (6, 7, 8))


if __name__ == "__main__":
    unittest.main()
