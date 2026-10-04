"""Frame-accounting checks for frozen evaluation reporting, not detector changes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from summarize_phase20_v8_full_transfer import audit_rows


def row(index, warmup=False, pixels=100, reset=False):
    if warmup:
        pixels = 0
    ready = not warmup and pixels > 0
    motion = dict(reset=reset, pva_failure=False)
    if index:
        motion.update(
            accepted=True,
            motion_backends=dict(
                optical_flow_pyrlk="PVA",
                harris="PVA",
                gaussian_pyramid="PVA",
                cpu_fallback=False,
            ),
        )
    return dict(
        frame_index=index,
        motion=motion,
        coverage=dict(
            full_shape_hw=[10, 10],
            total_pixels=100,
            configured_crop=None,
            native_pixel_sampling=True,
            warmup=warmup,
            searchable_pixels=pixels,
            detection_ready=ready,
            unavailable_reason="warmup"
            if warmup
            else "no_valid_search_support"
            if not ready
            else None,
        ),
    )


class FullTransferAuditTests(unittest.TestCase):
    def test_initial_warmup_is_not_a_later_gap(self):
        result = audit_rows([row(i, warmup=i < 2) for i in range(6)], 6)
        self.assertEqual(result["initial_warmup_frames"], 2)
        self.assertEqual(result["post_startup_unavailable_frames"], 0)
        self.assertEqual(result["longest_continuous_ready_frames"], 4)
        self.assertEqual(result["actual_pva_pairs"], 5)
        self.assertTrue(result["actual_pva_without_fallback"])
        self.assertEqual(result["unavailable_intervals"][0]["end_frame"], 1)

    def test_middle_reset_and_final_no_support_gaps(self):
        rows = [row(i, warmup=i in (0, 3, 4), reset=i == 3) for i in range(8)]
        rows[7] = row(7, pixels=0)
        result = audit_rows(rows, 8)
        self.assertEqual(result["post_startup_unavailable_frames"], 3)
        self.assertEqual(result["motion_reset_frame_indices"], [3])
        self.assertEqual(
            [
                (g["start_frame"], g["end_frame"])
                for g in result["unavailable_intervals"]
            ],
            [(0, 0), (3, 4), (7, 7)],
        )
        self.assertEqual(
            result["unavailable_intervals"][-1]["reasons"],
            {"no_valid_search_support": 1},
        )

    def test_zero_availability_cannot_appear_healthy(self):
        result = audit_rows([row(i, warmup=True) for i in range(4)], 4)
        self.assertFalse(result["detection_ever_ready"])
        self.assertEqual(result["longest_continuous_ready_frames"], 0)
        self.assertEqual(result["unavailable_intervals"][0]["frames"], 4)

    def test_cpu_fallback_is_not_hardware_validation(self):
        rows = [row(0, warmup=True), row(1)]
        for key, bad in (
            ("cpu_fallback", True),
            ("harris", "CPU"),
            ("gaussian_pyramid", "CUDA"),
            ("optical_flow_pyrlk", "CPU"),
        ):
            changed = deepcopy(rows)
            changed[1]["motion"]["motion_backends"][key] = bad
            result = audit_rows(changed, 2)
            self.assertFalse(result["actual_pva_without_fallback"])
            self.assertEqual(result["backend_mismatch_frame_indices"], [1])

    def test_runtime_failure_is_preserved_as_incomplete_pva_coverage(self):
        rows = [row(0, warmup=True), row(1, warmup=True, reset=True)]
        rows[1]["motion"] = dict(reset=True, pva_failure=True)
        result = audit_rows(rows, 2)
        self.assertFalse(result["actual_pva_without_fallback"])
        self.assertEqual(result["motion_reset_frame_indices"], [1])

    def test_missing_frames_or_outcomes_fail(self):
        for rows, expected in (
            ([row(0), row(2)], 2),
            ([row(0)], 2),
            ([row(0), row(1)], 1),
        ):
            with self.assertRaises(ValueError):
                audit_rows(rows, expected)
        broken = row(1)
        del broken["motion"]["accepted"]
        with self.assertRaises(ValueError):
            audit_rows([row(0), broken], 2)

    def test_stored_availability_must_match_pixels_and_warmup(self):
        for key, bad in (("detection_ready", True), ("unavailable_reason", None)):
            broken = row(0, warmup=True)
            broken["coverage"][key] = bad
            with self.assertRaises(ValueError):
                audit_rows([broken], 1)

    def test_full_native_coverage_and_area_accounting(self):
        result = audit_rows(
            [row(0, warmup=True), row(1, pixels=80), row(2, pixels=100)], 3, [10, 10]
        )
        self.assertEqual(
            result["ready_frame_searchable_area_fraction"],
            dict(minimum=0.8, mean=0.9, maximum=1.0),
        )
        for key, bad in (
            ("configured_crop", [0, 0, 5, 5]),
            ("native_pixel_sampling", False),
            ("total_pixels", 101),
            ("searchable_pixels", 101),
        ):
            broken = row(0)
            broken["coverage"][key] = bad
            with self.assertRaises(ValueError):
                audit_rows([broken], 1)
        with self.assertRaises(ValueError):
            audit_rows([row(0)], 1, [20, 10])


if __name__ == "__main__":
    unittest.main()
