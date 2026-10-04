from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tiny_target.detection import CandidateRecord
from tiny_target.evaluation import (
    EvaluationError,
    SyntheticInjectionSpec,
    SyntheticInjector,
    SyntheticTarget,
    ThresholdAccumulator,
    latency_summary,
    load_dataset_inventory,
    load_injection_spec,
    match_candidates,
)
from tiny_target.types import Frame, TimestampSource


def candidate(index: int, x: int, y: int, vx: float = 0, vy: float = 0) -> CandidateRecord:
    return CandidateRecord(
        candidate_index=index,
        x_px=x,
        y_px=y,
        velocity_index=0,
        velocity_xy_px_s=(vx, vy),
        normalized_score_snr=8 + index,
        raw_sum_score=16 + 2 * index,
        supporting_frame_count=4,
        support_weight=4,
        peak_neighbor_max_score_snr=5,
        peak_contrast_snr=3,
        peak_to_neighbor_ratio=1.6,
        distance_to_border_px=10,
        distance_to_invalid_chebyshev_px=None,
        distance_to_invalid_is_lower_bound=True,
    )


class DatasetInventoryTests(unittest.TestCase):
    def test_manifest_requires_every_category_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for category in (
                "noise_only",
                "real_targets",
                "raw_injected",
                "stress_scene",
                "controlled_geometry",
            ):
                missing = category in {"real_targets", "stress_scene"}
                entries.append(
                    {
                        "dataset_id": category,
                        "category": category,
                        "availability": "missing" if missing else "available",
                        "label_status": "unavailable" if missing else "synthetic_ground_truth",
                        "source_path": None if missing else f"{category}.json",
                        "scene_tags": [category],
                        "notes": "fixture",
                    }
                )
            path = root / "inventory.json"
            path.write_text(
                json.dumps({"schema_version": 1, "manifest_id": "test", "entries": entries})
            )
            inventory = load_dataset_inventory(path)
            self.assertEqual(len(inventory.entries), 5)
            self.assertEqual(
                inventory.entries[0].source_path,
                str((root / "noise_only.json").resolve()),
            )
            entries.pop()
            path.write_text(
                json.dumps({"schema_version": 1, "manifest_id": "test", "entries": entries})
            )
            with self.assertRaisesRegex(EvaluationError, "omits required"):
                load_dataset_inventory(path)


class SyntheticInjectionTests(unittest.TestCase):
    def test_versioned_injection_spec_loads_with_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "injection.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "random_seed": 75,
                        "psf_sigma_px": 0.8,
                        "psf_radius_px": 3,
                        "targets": [
                            {
                                "target_id": "t",
                                "flux_dn": 10,
                                "reference_timestamp_ns": 0,
                                "reference_position_xy_px": [1, 2],
                                "velocity_xy_px_s": [0, 0],
                            }
                        ],
                    }
                )
            )
            spec, identity = load_injection_spec(path)
            self.assertEqual(spec.targets[0].target_id, "t")
            self.assertEqual(identity["path"], str(path.resolve()))
            self.assertEqual(len(identity["sha256"]), 64)

    def test_injection_is_early_subpixel_flux_recorded_and_dtype_preserved(self) -> None:
        target = SyntheticTarget(
            target_id="t0",
            flux_dn=1000,
            reference_timestamp_ns=0,
            reference_position_xy_px=(10.25, 8.75),
            velocity_xy_px_s=(2, -1),
        )
        injector = SyntheticInjector(
            SyntheticInjectionSpec(75, 0.8, 3, (target,))
        )
        frame = Frame(
            image=np.full((20, 24), 100, np.uint16),
            timestamp_ns=500_000_000,
            frame_index=0,
            source_id="fixture",
            bit_depth=16,
            timestamp_source=TimestampSource.MANIFEST,
        )
        injected = injector.inject(frame)
        self.assertEqual(injected.image.dtype, np.uint16)
        self.assertFalse(injected.image.flags.writeable)
        self.assertAlmostEqual(
            float(np.sum(injected.image.astype(int) - frame.image.astype(int))),
            1000,
            delta=4,
        )
        record = injector.records[0]
        self.assertEqual(
            record["injection_stage"],
            "decoded_source_before_motion_stabilization_preprocessing",
        )
        np.testing.assert_allclose(record["events"][0]["position_xy_px"], [11.25, 8.25])

    def test_frame_activation_and_acceleration_are_explicit(self) -> None:
        target = SyntheticTarget(
            target_id="accelerating",
            flux_dn=10,
            reference_timestamp_ns=0,
            reference_position_xy_px=(1, 2),
            velocity_xy_px_s=(2, 3),
            acceleration_xy_px_s2=(4, -2),
            first_frame_index=2,
            last_frame_index=3,
        )
        self.assertFalse(target.active(1))
        self.assertTrue(target.active(2))
        self.assertFalse(target.active(4))
        np.testing.assert_allclose(target.position_at(1_000_000_000), [5, 4])

    def test_injection_coverage_rejects_inactive_and_out_of_bounds_targets(self) -> None:
        inactive = SyntheticTarget(
            target_id="inactive",
            flux_dn=10,
            reference_timestamp_ns=0,
            reference_position_xy_px=(2, 2),
            velocity_xy_px_s=(0, 0),
            first_frame_index=2,
        )
        outside = SyntheticTarget(
            target_id="outside",
            flux_dn=10,
            reference_timestamp_ns=0,
            reference_position_xy_px=(1_000_000, 1_000_000),
            velocity_xy_px_s=(0, 0),
        )
        injector = SyntheticInjector(
            SyntheticInjectionSpec(75, 0.8, 3, (inactive, outside))
        )
        injector.inject(
            Frame(
                image=np.zeros((8, 8), np.uint16),
                timestamp_ns=0,
                frame_index=0,
                source_id="fixture",
                bit_depth=16,
                timestamp_source=TimestampSource.MANIFEST,
            )
        )
        with self.assertRaisesRegex(
            EvaluationError,
            "not active.*inactive.*never entered image bounds.*outside",
        ):
            injector.validate_coverage()

    def test_injection_coverage_is_reported_for_observable_target(self) -> None:
        target = SyntheticTarget(
            target_id="visible",
            flux_dn=100,
            reference_timestamp_ns=0,
            reference_position_xy_px=(4, 4),
            velocity_xy_px_s=(0, 0),
        )
        injector = SyntheticInjector(
            SyntheticInjectionSpec(75, 0.8, 2, (target,))
        )
        injector.inject(
            Frame(
                image=np.zeros((8, 8), np.uint16),
                timestamp_ns=0,
                frame_index=0,
                source_id="fixture",
                bit_depth=16,
                timestamp_source=TimestampSource.MANIFEST,
            )
        )
        coverage = injector.validate_coverage()
        self.assertEqual(coverage["processed_frame_count"], 1)
        self.assertEqual(coverage["targets"][0]["active_processed_frame_count"], 1)
        self.assertGreater(
            coverage["targets"][0]["in_bounds_requested_flux_dn"], 0
        )


class EvaluationMetricTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_deterministic(self) -> None:
        truths = [
            {"target_id": "a", "position_xy_px": [10, 10], "velocity_xy_px_s": [1, 0], "flux_dn": 100},
            {"target_id": "b", "position_xy_px": [20, 10], "velocity_xy_px_s": [-1, 0], "flux_dn": 200},
        ]
        result = match_candidates(
            (candidate(0, 10, 10, 1, 0), candidate(1, 20, 10, -1, 0), candidate(2, 30, 30)),
            truths,
            maximum_position_error_px=2,
            maximum_velocity_error_px_s=1,
        )
        self.assertEqual(result["true_positive_count"], 2)
        self.assertEqual(result["unmatched_candidate_indices"], [2])
        self.assertEqual([item["target_id"] for item in result["matches"]], ["a", "b"])

    def test_false_alarm_metrics_require_verified_empty_labels(self) -> None:
        truth = [{"target_id": "a", "position_xy_px": [5, 5], "velocity_xy_px_s": [0, 0], "flux_dn": 100}]
        matched = match_candidates(
            (candidate(0, 5, 5), candidate(1, 20, 20)),
            truth,
            maximum_position_error_px=2,
            maximum_velocity_error_px_s=1,
        )
        verified = ThresholdAccumulator(7, unmatched_are_false=True)
        verified.add(truth, matched)
        summary = verified.summary(
            evaluated_frame_count=4,
            duration_s=0.4,
            frame_shape=(10, 10),
        )
        self.assertEqual(summary["false_alarms"]["count"], 1)
        self.assertEqual(summary["precision"], 0.5)
        unknown = ThresholdAccumulator(7, unmatched_are_false=False)
        unknown.add(truth, matched)
        self.assertIsNone(
            unknown.summary(
                evaluated_frame_count=4,
                duration_s=0.4,
                frame_shape=(10, 10),
            )["false_alarms"]
        )

    def test_latency_summary_has_required_percentiles(self) -> None:
        summary = latency_summary([1, 2, 3, 4, 10])
        self.assertEqual(summary["median_ms"], 3)
        self.assertEqual(summary["maximum_ms"], 10)
        self.assertIn("p99_ms", summary)


if __name__ == "__main__":
    unittest.main()
