from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tiny_target.field_review import FieldReviewPolicy, analyze, track_passes_policy


def _track(track_id: int, *, observations: int, fraction: float, speed: float) -> dict:
    return {
        "track_id": track_id,
        "lifecycle_state": "confirmed",
        "birth_timestamp_ns": 100,
        "last_measurement_timestamp_ns": 500,
        "age_windows": observations,
        "associated_update_count": observations,
        "missed_windows": 0,
        "confirmation": {"timestamp_ns": 200},
        "state": {
            "timestamp_ns": 500,
            "mean": [10.0, 20.0, speed, 0.0],
        },
        "quality_evidence": {
            "observation_count": observations,
            "observation_fraction_of_age_windows": fraction,
            "measurement_speed_px_s": {"mean": speed},
            "detector_score_snr": {"mean": 12.0},
            "selection_score": {"mean": 9.0},
            "peak_to_neighbor_ratio": {"mean": 1.2},
        },
    }


def _batch(frame_index: int, tracks: list[dict], reservations: int = 0) -> dict:
    return {
        "candidate_batch": {
            "frame_indices": [frame_index - 3, frame_index - 2, frame_index - 1, frame_index],
            "reference_timestamp_ns": frame_index * 100,
            "segment_index": 0,
            "candidates": [{"candidate_index": 0}, {"candidate_index": 1}],
            "metrics": {
                "output_truncated": False,
                "track_guided_reservation": {
                    "retained_reservation_count": reservations
                },
            },
        },
        "confirmed_or_coasted_tracks": tracks,
    }


class FieldReviewTests(unittest.TestCase):
    def test_frozen_policy_is_explicit_and_non_truth_claiming(self) -> None:
        policy = FieldReviewPolicy()
        self.assertTrue(track_passes_policy(_track(1, observations=5, fraction=0.96, speed=0.25), policy))
        self.assertFalse(track_passes_policy(_track(2, observations=5, fraction=0.96, speed=0.24), policy))

    def test_unlabeled_report_counts_workload_without_false_alarm_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "seaqr.tiny-target.motion.v10",
                        "source": {"identity": {"path": "fixture.mkv"}},
                        "summary": {
                            "frames_read": 8,
                            "pairs_attempted": 7,
                            "accepted_global_transforms": 7,
                            "rejected_global_transforms": 0,
                            "matched_filter": {"detection_ready_frames": 6},
                        },
                        "evaluation": {
                            "candidate_threshold_sweep_parameter": "cfar_threshold_sigma",
                            "candidate_threshold_sweep": {
                                "8": [
                                    _batch(
                                        3,
                                        [
                                            _track(1, observations=5, fraction=0.96, speed=0.25),
                                            _track(2, observations=5, fraction=1.0, speed=0.0),
                                        ],
                                        reservations=1,
                                    ),
                                    _batch(
                                        4,
                                        [
                                            _track(1, observations=6, fraction=1.0, speed=0.5),
                                            _track(2, observations=6, fraction=1.0, speed=0.0),
                                        ],
                                    ),
                                ]
                            },
                        },
                    }
                )
            )
            report = analyze(path)
        self.assertEqual(report["screening_counts"]["total_candidates_across_windows"], 4)
        self.assertEqual(report["screening_counts"]["unique_confirmed_or_coasted_track_count"], 2)
        self.assertEqual(report["screening_counts"]["ever_diagnostic_qualified_track_count"], 1)
        self.assertEqual(report["screening_counts"]["track_guided_reservation_count"], 1)
        self.assertIsNone(report["interpretation"]["real_object_count"])
        self.assertIsNone(report["interpretation"]["false_alarm_count"])
        self.assertEqual(report["tracks"][0]["review_label"], "unreviewed")

    def test_missing_threshold_fails_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "seaqr.tiny-target.motion.v10",
                        "evaluation": {
                            "candidate_threshold_sweep_parameter": "cfar_threshold_sigma",
                            "candidate_threshold_sweep": {"7": []},
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "CFAR threshold 8"):
                analyze(path)


if __name__ == "__main__":
    unittest.main()
