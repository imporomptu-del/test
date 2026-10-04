from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from tiny_target.detection import (
    MatchedFilterConfig,
    MatchedFilterFrame,
    PsfKernelBank,
    build_kernel_bank,
)


class MatchedFilterTypeTests(unittest.TestCase):
    def test_integrated_gaussian_bank_is_flux_normalized_and_symmetric(self) -> None:
        bank = build_kernel_bank(
            MatchedFilterConfig(
                gaussian_sigma_px=0.8, radius_px=3, phases_per_axis=1
            )
        )
        self.assertTrue(bank.provisional)
        self.assertEqual(bank.kernels.shape, (1, 7, 7))
        self.assertAlmostEqual(float(np.sum(bank.kernels[0])), 1.0, places=6)
        np.testing.assert_allclose(bank.kernels[0], bank.kernels[0, ::-1, ::-1])
        self.assertFalse(bank.kernels.flags.writeable)

    def test_phase_grid_uses_pixel_cell_centers(self) -> None:
        bank = build_kernel_bank(MatchedFilterConfig(phases_per_axis=2))
        expected = np.array(
            [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]],
            np.float32,
        )
        np.testing.assert_array_equal(bank.phase_offsets_xy, expected)

    def test_contract_rejects_non_normalized_kernel_and_warm_validity(self) -> None:
        with self.assertRaisesRegex(ValueError, "unit flux"):
            PsfKernelBank(
                kernels=np.ones((1, 3, 3), np.float32),
                phase_offsets_xy=np.zeros((1, 2), np.float32),
                source="fixture",
                source_identity={},
                provisional=True,
            )
        with self.assertRaisesRegex(ValueError, "suppressed"):
            MatchedFilterFrame(
                response=np.zeros((2, 2), np.float32),
                phase_index=np.zeros((2, 2), np.uint16),
                valid_mask=np.ones((2, 2), bool),
                valid_support_count=np.ones((2, 2), np.uint16),
                timestamp_ns=0,
                frame_index=0,
                reference_frame_index=0,
                segment_index=0,
                detection_ready=False,
                polarity="bright",
                backend="numpy_reference",
                kernel_metadata={},
                metrics={},
                timings_ms={},
            )

    def test_configuration_rejects_unknown_and_uncalibrated_npy(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            MatchedFilterConfig(source="npy")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            MatchedFilterConfig.from_mapping({"mystery": 1})

    def test_measured_npy_is_normalized_and_content_identified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured_psf.npy"
            np.save(path, np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32))
            bank = build_kernel_bank(
                MatchedFilterConfig(source="npy", kernel_path=path.name),
                base_path=directory,
            )
            self.assertFalse(bank.provisional)
            self.assertAlmostEqual(float(np.sum(bank.kernels)), 1.0, places=6)
            self.assertEqual(len(bank.source_identity["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
