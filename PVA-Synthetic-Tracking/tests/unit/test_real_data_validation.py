from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tiny_target.real_data_validation import (
    RealDataValidationError,
    audit_manifest,
    main,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RealDataReadinessTests(unittest.TestCase):
    def _write_complete_fixture(self, root: Path) -> Path:
        kernel = root / "measured_psf.npy"
        np.save(kernel, np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32))
        condition = "camera-condition-1"
        metadata = root / "measured_psf.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "calibration_id": "measured-psf",
                    "source": "measured_camera_point_sources",
                    "kernel_sha256": _sha(kernel),
                    "operating_condition_id": condition,
                    "capture_method": "isolated unsaturated point-source stack",
                    "reviewer": "fixture-reviewer",
                    "reviewed_at_utc": "2026-09-06T12:00:00Z",
                    "camera": {"model": "fixture", "camera_id": "fixture-1"},
                    "conditions": {
                        "focus_setting": "locked",
                        "aperture": "fixed",
                        "wavelength_band": "visible",
                        "exposure_us": 1000,
                        "gain": 1,
                        "sensor_temperature_c": 30,
                        "field_region": "center",
                    },
                }
            )
        )

        recordings = []
        for name in ("targets", "empty"):
            recording = root / f"{name}.mkv"
            recording.write_bytes(f"fixture-{name}".encode())
            timestamps = root / f"{name}_timestamps.csv"
            timestamps.write_text("frame_index,timestamp_ns\n0,0\n")
            recordings.append((recording, timestamps))
        target_recording, target_timestamps = recordings[0]
        empty_recording, empty_timestamps = recordings[1]

        target_evidence = root / "targets.json"
        target_evidence.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_id": "targets-v1",
                    "evidence_type": "real_target_annotations",
                    "source_sha256": _sha(target_recording),
                    "operating_condition_id": condition,
                    "reviewer": "fixture-reviewer",
                    "reviewed_at_utc": "2026-09-06T12:00:00Z",
                    "method": "independent frame review",
                    "assertion": "authoritative_real_targets",
                    "targets": [
                        {
                            "target_id": "target-1",
                            "object_type": "point_source",
                            "frames": [
                                {
                                    "frame_index": 0,
                                    "timestamp_ns": 0,
                                    "x_px": 4.5,
                                    "y_px": 5.5,
                                    "visibility": "visible",
                                    "uncertainty_px": 0.5,
                                }
                            ],
                        }
                    ],
                }
            )
        )
        empty_evidence = root / "empty.json"
        empty_evidence.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_id": "empty-v1",
                    "evidence_type": "verified_empty_review",
                    "source_sha256": _sha(empty_recording),
                    "operating_condition_id": condition,
                    "reviewer": "fixture-reviewer",
                    "reviewed_at_utc": "2026-09-06T12:00:00Z",
                    "method": "independent full-frame review",
                    "assertion": "no_point_targets",
                    "reviewed_intervals": [
                        {"first_frame_index": 0, "last_frame_index": 0}
                    ],
                }
            )
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_id": "fixture",
                    "psf_calibrations": [
                        {
                            "calibration_id": "measured-psf",
                            "availability": "available",
                            "kernel_path": kernel.name,
                            "metadata_path": metadata.name,
                            "operating_condition_id": condition,
                            "notes": "fixture",
                        }
                    ],
                    "cohorts": [
                        {
                            "cohort_id": "targets",
                            "purpose": "real_target_sensitivity",
                            "availability": "available",
                            "source_path": target_recording.name,
                            "source_sha256": _sha(target_recording),
                            "timestamps_path": target_timestamps.name,
                            "timestamps_sha256": _sha(target_timestamps),
                            "evidence_path": target_evidence.name,
                            "psf_calibration_id": "measured-psf",
                            "operating_condition_id": condition,
                            "notes": "fixture",
                        },
                        {
                            "cohort_id": "empty",
                            "purpose": "verified_empty_false_alarm",
                            "availability": "available",
                            "source_path": empty_recording.name,
                            "source_sha256": _sha(empty_recording),
                            "timestamps_path": empty_timestamps.name,
                            "timestamps_sha256": _sha(empty_timestamps),
                            "evidence_path": empty_evidence.name,
                            "psf_calibration_id": "measured-psf",
                            "operating_condition_id": condition,
                            "notes": "fixture",
                        },
                    ],
                }
            )
        )
        return manifest

    def test_complete_measured_and_authoritative_cohorts_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = audit_manifest(self._write_complete_fixture(Path(directory)))
        self.assertTrue(report["overall_ready"])
        self.assertEqual(
            report["claim_readiness"]["real_target_sensitivity"]["eligible_cohort_ids"],
            ["targets"],
        )
        self.assertEqual(report["cohorts"][0]["artifacts"]["evidence"]["target_count"], 1)
        self.assertEqual(report["cohorts"][1]["artifacts"]["evidence"]["reviewed_frame_count"], 1)

    def test_inventory_mode_cannot_make_claims_without_source_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = audit_manifest(
                self._write_complete_fixture(Path(directory)),
                verify_source_hashes=False,
            )
        self.assertFalse(report["overall_ready"])
        self.assertIn("recording SHA-256 was not verified", report["cohorts"][0]["blockers"])

    def test_recording_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_complete_fixture(root)
            (root / "targets.mkv").write_bytes(b"changed after annotation")
            report = audit_manifest(manifest)
        self.assertFalse(report["claim_readiness"]["real_target_sensitivity"]["ready"])
        self.assertIn("recording SHA-256 does not match manifest", report["cohorts"][0]["blockers"])

    def test_wrong_empty_assertion_is_rejected_as_claim_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_complete_fixture(root)
            evidence_path = root / "empty.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["assertion"] = "probably_empty"
            evidence_path.write_text(json.dumps(evidence))
            report = audit_manifest(manifest)
        self.assertFalse(report["claim_readiness"]["verified_empty_false_alarm"]["ready"])
        self.assertTrue(
            any("no_point_targets" in blocker for blocker in report["cohorts"][1]["blockers"])
        )

    def test_unknown_manifest_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_complete_fixture(root)
            value = json.loads(manifest.read_text())
            value["assume_empty"] = True
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(RealDataValidationError, "Unknown manifest"):
                audit_manifest(manifest)

    def test_require_ready_returns_two_for_missing_default_contract(self) -> None:
        result = main(
            [
                "--manifest",
                str(
                    Path(__file__).resolve().parents[2]
                    / "configs"
                    / "evaluation"
                    / "phase16_real_data_manifest.json"
                ),
                "--require-ready",
            ]
        )
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
