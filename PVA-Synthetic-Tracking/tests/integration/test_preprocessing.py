from __future__ import annotations

import unittest

import numpy as np

from tiny_target.preprocessing import (
    BackgroundConfig,
    NoiseConfig,
    RobustPreprocessor,
)
from tiny_target.stabilization import StabilizedFrame
from tiny_target.types import Frame, TimestampSource


def stabilized(
    image: np.ndarray,
    index: int,
    *,
    valid: np.ndarray | None = None,
    segment: int = 0,
) -> StabilizedFrame:
    if valid is None:
        valid = np.ones(image.shape, bool)
    frame = Frame(
        image=np.asarray(image, np.float32),
        timestamp_ns=index * 100_000_000,
        frame_index=index,
        source_id="preprocessing_fixture",
        bit_depth=16,
        timestamp_source=TimestampSource.MANIFEST,
        valid_mask=np.asarray(valid, bool),
    )
    return StabilizedFrame(
        frame=frame,
        reference_frame_index=0 if segment == 0 else index,
        segment_index=segment,
        source_to_reference_matrix=np.eye(3),
        interpolation="cubic",
        backend="opencv_cpu",
        resampling_count=0,
        metrics={},
        timings_ms={},
    )


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_warmup_is_explicit_and_current_frame_does_not_model_itself(self) -> None:
        model = RobustPreprocessor(
            BackgroundConfig(warmup_frames=3, history_frames=5),
            NoiseConfig(sigma_floor=1),
        )
        for index in range(3):
            result = model.process(stabilized(np.full((8, 9), 100), index))
            self.assertFalse(result.detection_ready)
            self.assertFalse(np.any(result.valid_mask))
        image = np.full((8, 9), 100, np.float32)
        image[4, 5] += 25
        result = model.process(stabilized(image, 3))
        self.assertTrue(result.detection_ready)
        self.assertEqual(float(result.value[4, 5]), 25.0)
        self.assertEqual(float(result.whitened[4, 5]), 25.0)

    def test_temporal_mad_makes_bright_and_dark_region_scales_comparable(self) -> None:
        rng = np.random.default_rng(75)
        model = RobustPreprocessor(
            BackgroundConfig(
                warmup_frames=15,
                history_frames=20,
                minimum_history_samples=12,
            ),
            NoiseConfig(sigma_floor=0.25),
        )
        shape = (96, 128)
        for index in range(16):
            image = np.empty(shape, np.float32)
            image[:, :64] = 100 + rng.normal(0, 2, (96, 64))
            image[:, 64:] = 1000 + rng.normal(0, 8, (96, 64))
            result = model.process(stabilized(image, index))
        self.assertTrue(result.detection_ready)
        left = result.whitened[:, :64][result.valid_mask[:, :64]]
        right = result.whitened[:, 64:][result.valid_mask[:, 64:]]
        left_scale = 1.4826 * np.median(np.abs(left - np.median(left)))
        right_scale = 1.4826 * np.median(np.abs(right - np.median(right)))
        self.assertLess(abs(float(left_scale - right_scale)), 0.2)
        self.assertGreater(float(left_scale), 0.8)
        self.assertLess(float(left_scale), 1.4)

    def test_invalid_saturated_dead_and_bad_pixels_never_detect(self) -> None:
        bad = np.zeros((6, 7), bool)
        bad[1, 1] = True
        model = RobustPreprocessor(
            BackgroundConfig(
                warmup_frames=2,
                history_frames=4,
                minimum_history_samples=2,
            ),
            NoiseConfig(
                sigma_floor=1,
                saturation_value=4095,
                dead_level_max=0,
            ),
            bad_pixel_mask=bad,
        )
        for index in range(2):
            model.process(stabilized(np.full((6, 7), 100), index))
        image = np.full((6, 7), 100, np.float32)
        image[0, 0] = 0
        image[2, 2] = 4095
        valid = np.ones((6, 7), bool)
        valid[3, 3] = False
        result = model.process(stabilized(image, 2, valid=valid))
        for y, x in ((0, 0), (1, 1), (2, 2), (3, 3)):
            self.assertFalse(result.valid_mask[y, x])

        source = Frame(
            image=image.astype(np.uint16),
            timestamp_ns=0,
            frame_index=0,
            source_id="raw_mask_fixture",
            bit_depth=16,
            timestamp_source=TimestampSource.MANIFEST,
        )
        prepared = model.prepare_source(source)
        self.assertFalse(prepared.valid_mask[0, 0])
        self.assertFalse(prepared.valid_mask[1, 1])
        self.assertFalse(prepared.valid_mask[2, 2])

    def test_missing_history_does_not_become_valid_when_pixel_returns(self) -> None:
        model = RobustPreprocessor(
            BackgroundConfig(
                warmup_frames=3,
                history_frames=5,
                minimum_history_samples=3,
            ),
            NoiseConfig(sigma_floor=1),
        )
        history_valid = np.ones((5, 5), bool)
        history_valid[2, 2] = False
        for index in range(3):
            model.process(
                stabilized(np.full((5, 5), 100), index, valid=history_valid)
            )
        result = model.process(stabilized(np.full((5, 5), 110), 3))
        self.assertFalse(result.valid_mask[2, 2])
        self.assertEqual(float(result.value[2, 2]), 0.0)

    def test_moving_target_flux_is_retained_by_both_background_models(self) -> None:
        for method in ("temporal_median", "robust_running"):
            with self.subTest(method=method):
                model = RobustPreprocessor(
                    BackgroundConfig(
                        method=method,
                        warmup_frames=4,
                        history_frames=8,
                        minimum_history_samples=3,
                        update_rate=0.2,
                        outlier_clip_sigma=3,
                        update_exclusion_sigma=5,
                    ),
                    NoiseConfig(
                        method="temporal_mad" if method == "temporal_median" else "robust_ewma",
                        sigma_floor=1,
                    ),
                )
                for index in range(4):
                    model.process(stabilized(np.full((12, 16), 100), index))
                retained = []
                for offset, x in enumerate((3, 5, 7, 9), start=4):
                    image = np.full((12, 16), 100, np.float32)
                    image[6, x] += 40
                    result = model.process(stabilized(image, offset))
                    retained.append(float(result.value[6, x]) / 40)
                self.assertGreater(min(retained), 0.95)

    def test_running_model_follows_slow_drift_and_resets_segments(self) -> None:
        model = RobustPreprocessor(
            BackgroundConfig(
                method="robust_running",
                warmup_frames=4,
                history_frames=8,
                minimum_history_samples=3,
                update_rate=0.25,
            ),
            NoiseConfig(method="robust_ewma", sigma_floor=1),
        )
        for index in range(30):
            result = model.process(
                stabilized(np.full((8, 8), 100 + index, np.float32), index)
            )
        self.assertLess(float(np.median(np.abs(result.value))), 6.0)
        reset = model.process(
            stabilized(np.full((8, 8), 500, np.float32), 30, segment=1)
        )
        self.assertFalse(reset.detection_ready)
        self.assertEqual(reset.history_frames, 0)
        self.assertFalse(np.any(reset.valid_mask))

    def test_global_illumination_event_suppresses_detection_but_keeps_residual(self) -> None:
        model = RobustPreprocessor(
            BackgroundConfig(
                method="robust_running",
                warmup_frames=3,
                history_frames=6,
                minimum_history_samples=3,
                global_change_median_sigma=2,
            ),
            NoiseConfig(method="robust_ewma", sigma_floor=2),
        )
        for index in range(3):
            model.process(stabilized(np.full((10, 12), 100), index))
        changed = model.process(stabilized(np.full((10, 12), 120), 3))
        self.assertFalse(changed.detection_ready)
        self.assertFalse(np.any(changed.valid_mask))
        self.assertTrue(changed.metrics["warmup_complete"])
        self.assertTrue(changed.metrics["global_change_suppressed"])
        self.assertEqual(float(np.median(changed.value)), 20.0)
        for index in range(4, 16):
            recovered = model.process(stabilized(np.full((10, 12), 120), index))
        self.assertTrue(recovered.detection_ready)
        self.assertFalse(recovered.metrics["global_change_suppressed"])


if __name__ == "__main__":
    unittest.main()
