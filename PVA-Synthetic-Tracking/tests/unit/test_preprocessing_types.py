from __future__ import annotations

import unittest

import numpy as np

from tiny_target.preprocessing import BackgroundConfig, NoiseConfig, ResidualFrame


class PreprocessingTypeTests(unittest.TestCase):
    def test_residual_contract_is_float_read_only_and_signed(self) -> None:
        value = np.array([[-2, 3]], np.float32)
        residual = ResidualFrame(
            value=value,
            sigma=np.ones((1, 2), np.float32),
            whitened=value,
            valid_mask=np.ones((1, 2), bool),
            timestamp_ns=10,
            frame_index=1,
            reference_frame_index=0,
            segment_index=0,
            detection_ready=True,
            history_frames=4,
            background_method="temporal_median",
            noise_method="temporal_mad",
            metrics={},
            timings_ms={},
        )
        self.assertEqual(float(residual.value[0, 0]), -2.0)
        self.assertFalse(residual.value.flags.writeable)
        self.assertFalse(residual.valid_mask.flags.writeable)

    def test_warmup_cannot_expose_valid_detection_pixels(self) -> None:
        with self.assertRaisesRegex(ValueError, "warm-up"):
            ResidualFrame(
                value=np.zeros((2, 2), np.float32),
                sigma=np.ones((2, 2), np.float32),
                whitened=np.zeros((2, 2), np.float32),
                valid_mask=np.ones((2, 2), bool),
                timestamp_ns=0,
                frame_index=0,
                reference_frame_index=0,
                segment_index=0,
                detection_ready=False,
                history_frames=0,
                background_method="temporal_median",
                noise_method="temporal_mad",
                metrics={},
                timings_ms={},
            )

    def test_configuration_rejects_unknown_and_inconsistent_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            BackgroundConfig(warmup_frames=5, history_frames=4)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            BackgroundConfig.from_mapping({"mystery": 1})
        with self.assertRaisesRegex(ValueError, "positive"):
            NoiseConfig(sigma_floor=0)


if __name__ == "__main__":
    unittest.main()
