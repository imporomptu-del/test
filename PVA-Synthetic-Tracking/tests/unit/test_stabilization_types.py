from __future__ import annotations

import unittest

import numpy as np

from tiny_target.stabilization import (
    StabilizationConfig,
    ValidSupport,
    signal_preservation_metrics,
)


class StabilizationTypeTests(unittest.TestCase):
    def test_valid_support_contract_requires_exact_common_mask(self) -> None:
        support = np.array([[2, 1], [2, 0]], np.uint16)
        value = ValidSupport(
            common_valid_mask=support == 2,
            support_count=support,
            frame_count=2,
            segment_index=0,
            first_frame_index=0,
            last_frame_index=1,
        )
        self.assertEqual(value.metrics()["common_valid_fraction"], 0.5)
        self.assertFalse(value.support_count.flags.writeable)
        with self.assertRaisesRegex(ValueError, "full-window support"):
            ValidSupport(
                common_valid_mask=np.ones((2, 2), bool),
                support_count=support,
                frame_count=2,
                segment_index=0,
                first_frame_index=0,
                last_frame_index=1,
            )

    def test_configuration_rejects_cuda_lanczos_and_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support"):
            StabilizationConfig(backend="opencv_cuda", interpolation="lanczos4")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            StabilizationConfig.from_mapping({"mystery": 1})

    def test_signal_metrics_distinguish_peak_flux_and_energy(self) -> None:
        source = np.array([[0, 4], [0, 0]], np.float32)
        warped = np.array([[1, 1], [1, 1]], np.float32)
        metrics = signal_preservation_metrics(
            source, warped, np.ones((2, 2), bool)
        )
        self.assertEqual(metrics["peak_retention"], 0.25)
        self.assertEqual(metrics["flux_retention"], 1.0)
        self.assertEqual(metrics["l2_energy_retention"], 0.25)


if __name__ == "__main__":
    unittest.main()
