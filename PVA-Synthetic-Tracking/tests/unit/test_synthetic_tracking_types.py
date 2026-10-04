from __future__ import annotations

import unittest

import numpy as np

from tiny_target.detection import SyntheticTrackWindow, SyntheticTrackingConfig


class SyntheticTrackingTypeTests(unittest.TestCase):
    def test_configuration_requires_calibrated_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            SyntheticTrackingConfig.from_mapping({})
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            SyntheticTrackingConfig(
                window_frames=4,
                window_stride_frames=1,
                vx_min_px_s=-1,
                vx_max_px_s=1,
                vy_min_px_s=0,
                vy_max_px_s=0,
                velocity_step_px_s=0.3,
                min_valid_fraction=1,
            )

    def test_velocity_grid_order_is_deterministic(self) -> None:
        config = SyntheticTrackingConfig(
            window_frames=4,
            window_stride_frames=2,
            vx_min_px_s=-1,
            vx_max_px_s=1,
            vy_min_px_s=-1,
            vy_max_px_s=1,
            velocity_step_px_s=1,
            min_valid_fraction=0.75,
        )
        np.testing.assert_array_equal(
            config.velocity_grid()[:4],
            np.array([[-1, -1], [0, -1], [1, -1], [-1, 0]], np.float32),
        )

    def test_cuda_execution_configuration_is_validated(self) -> None:
        values = {
            "backend": "cuda",
            "window_frames": 4,
            "window_stride_frames": 1,
            "vx_min_px_s": 0,
            "vx_max_px_s": 0,
            "vy_min_px_s": 0,
            "vy_max_px_s": 0,
            "velocity_step_px_s": 1,
            "min_valid_fraction": 1,
        }
        configured = SyntheticTrackingConfig.from_mapping(values)
        self.assertEqual(configured.backend, "cuda")
        self.assertEqual(configured.velocity_batch_size, 32)
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            SyntheticTrackingConfig.from_mapping(
                {**values, "cuda_threads_per_block": 100}
            )
        with self.assertRaisesRegex(ValueError, "reference or cuda"):
            SyntheticTrackingConfig.from_mapping({**values, "backend": "auto"})

    def test_output_materializes_selected_velocity_on_demand(self) -> None:
        window = SyntheticTrackWindow(
            score=np.ones((2, 2), np.float32),
            velocity_index=np.array([[0, 1], [1, 0]], np.uint16),
            valid_support_count=np.full((2, 2), 4, np.uint16),
            valid_mask=np.ones((2, 2), bool),
            velocity_grid_xy_px_s=np.array([[0, 0], [2, -1]], np.float32),
            frame_indices=(0, 1, 2, 3),
            window_start_timestamp_ns=0,
            window_end_timestamp_ns=300,
            reference_timestamp_ns=150,
            segment_index=0,
            metrics={},
            timings_ms={},
        )
        selected = window.selected_velocity_xy_px_s()
        np.testing.assert_array_equal(selected[0, 1], [2, -1])
        self.assertFalse(window.score.flags.writeable)


if __name__ == "__main__":
    unittest.main()
