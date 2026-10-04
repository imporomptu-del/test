from __future__ import annotations

import unittest

from tiny_target.detection import CandidateBatch, CandidateRecord
from tiny_target.evaluation import EvaluationError
from tiny_target.track_quality_evaluation import (
    DiagnosticMotionPersistencePolicy,
    DiagnosticQualityPolicy,
    summarize_track_quality,
)
from tiny_target.tracking import KalmanTrackManager, KalmanTrackingConfig


def _candidate(
    index: int,
    x: int,
    score: float,
    peak_ratio: float = 1.1,
    velocity_xy_px_s: tuple[int, int] = (0, 0),
) -> CandidateRecord:
    return CandidateRecord(
        candidate_index=index,
        x_px=x,
        y_px=10,
        velocity_index=0,
        velocity_xy_px_s=velocity_xy_px_s,
        normalized_score_snr=score,
        raw_sum_score=score * 2,
        supporting_frame_count=4,
        support_weight=4,
        peak_neighbor_max_score_snr=score - 2,
        peak_contrast_snr=2,
        peak_to_neighbor_ratio=peak_ratio,
        distance_to_border_px=10,
        distance_to_invalid_chebyshev_px=None,
        distance_to_invalid_is_lower_bound=True,
        ranking_score=score + 1,
        ranking_score_units="local_robust_sigma",
    )


def _manager() -> KalmanTrackManager:
    return KalmanTrackManager(
        KalmanTrackingConfig(
            position_measurement_sigma_px=1,
            velocity_measurement_sigma_px_s=1,
            acceleration_process_sigma_px_s2=1,
            initial_position_sigma_px=2,
            initial_velocity_sigma_px_s=2,
            mahalanobis_gate_squared=16,
            maximum_position_residual_px=6,
            maximum_velocity_residual_px_s=3,
            confirmation_independent_hits=2,
            max_missed_windows=1,
            maximum_timestamp_gap_s=2,
            measurement_noise_source="synthetic_characterization",
            max_active_tracks=16,
        )
    )


def _batch(index: int, candidates: tuple[CandidateRecord, ...]) -> CandidateBatch:
    return CandidateBatch(
        candidates=candidates,
        frame_indices=(index,),
        reference_timestamp_ns=index * 100_000_000,
        segment_index=0,
        metrics={},
        timings_ms={},
    )


class TrackQualityEvaluationTests(unittest.TestCase):
    def test_summarizes_truth_and_unmatched_tracks_without_applying_gates(self) -> None:
        manager = _manager()
        outputs = [
            manager.update(
                _batch(
                    0,
                    (
                        _candidate(0, 10, 12, velocity_xy_px_s=(1, 0)),
                        _candidate(1, 30, 9, 1.01),
                    ),
                )
            ),
            manager.update(
                _batch(
                    1,
                    (
                        _candidate(0, 10, 13, velocity_xy_px_s=(1, 0)),
                        _candidate(1, 30, 9, 1.01),
                    ),
                )
            ),
            manager.update(
                _batch(2, (_candidate(0, 10, 14, velocity_xy_px_s=(1, 0)),))
            ),
        ]
        batches = [
            {
                "confirmed_or_coasted_tracks": [
                    track.to_dict()
                    for track in output.tracks
                    if track.lifecycle_state in {"confirmed", "coasted"}
                ]
            }
            for output in outputs
        ]
        truth_windows = [
            {"confirmed_track_matching": {"matches": []}},
            {
                "confirmed_track_matching": {
                    "matches": [{"track_id": 0, "target_id": "injected"}]
                }
            },
            {
                "confirmed_track_matching": {
                    "matches": [{"track_id": 0, "target_id": "injected"}]
                }
            },
        ]
        summary = summarize_track_quality(
            batches,
            truth_windows,
            diagnostic_policy=DiagnosticQualityPolicy(
                minimum_observations=2,
                minimum_observation_fraction=1,
                maximum_detector_score_standard_deviation=8,
                minimum_peak_to_neighbor_ratio_mean=1.05,
            ),
            motion_persistence_policy=DiagnosticMotionPersistencePolicy(
                minimum_observations=2,
                minimum_observation_fraction=1,
                minimum_measurement_speed_px_s=0.25,
            ),
        )
        self.assertEqual(summary["ever_confirmed_or_coasted_track_count"], 2)
        self.assertEqual(summary["injected_truth_matched_track_count"], 1)
        self.assertEqual(summary["unmatched_to_injected_truth_track_count"], 1)
        by_id = {item["track_id"]: item for item in summary["tracks"]}
        self.assertEqual(by_id[0]["classification"], "injected_truth_matched")
        self.assertEqual(by_id[1]["classification"], "unmatched_to_injected_truth")
        observation_fraction = next(
            item
            for item in summary["feature_analysis"]
            if item["feature"] == "observation_fraction_of_age_windows"
        )
        gate = observation_fraction["discovery_gate_retaining_all_injected_tracks"]
        self.assertEqual(gate["threshold"], 1)
        self.assertEqual(gate["retained_injected_track_count"], 1)
        self.assertEqual(gate["retained_unmatched_track_count"], 0)
        causal = summary["diagnostic_causal_policy_evaluation"]
        self.assertFalse(causal["used_by_live_tracker"])
        self.assertEqual(causal["ever_qualified_injected_track_count"], 1)
        self.assertEqual(causal["ever_qualified_unmatched_track_count"], 0)
        self.assertEqual(
            causal["injected_track_first_qualification"][0][
                "qualification_latency_after_confirmation_s"
            ],
            0,
        )
        motion_policy = summary[
            "diagnostic_motion_persistence_policy_evaluation"
        ]
        self.assertFalse(motion_policy["used_by_live_tracker"])
        self.assertEqual(motion_policy["ever_qualified_injected_track_count"], 1)
        self.assertEqual(motion_policy["ever_qualified_unmatched_track_count"], 0)

    def test_diagnostic_policy_requires_bounded_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            DiagnosticQualityPolicy(1, 1, 8, 1.05)
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\]"):
            DiagnosticQualityPolicy(2, 0, 8, 1.05)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DiagnosticMotionPersistencePolicy(2, 1, -0.1)

    def test_empty_tracking_windows_do_not_create_vacuous_discovery_gates(self) -> None:
        summary = summarize_track_quality(
            [],
            [],
            diagnostic_policy=DiagnosticQualityPolicy(
                minimum_observations=2,
                minimum_observation_fraction=1,
                maximum_detector_score_standard_deviation=8,
                minimum_peak_to_neighbor_ratio_mean=1.05,
            ),
        )

        self.assertEqual(summary["ever_confirmed_or_coasted_track_count"], 0)
        self.assertEqual(summary["injected_truth_matched_track_count"], 0)
        self.assertEqual(summary["unmatched_to_injected_truth_track_count"], 0)
        self.assertEqual(summary["best_discovery_only_two_feature_envelopes"], [])
        self.assertTrue(
            all(
                item["discovery_gate_retaining_all_injected_tracks"] is None
                for item in summary["feature_analysis"]
            )
        )
        causal = summary["diagnostic_causal_policy_evaluation"]
        self.assertEqual(causal["ever_qualified_track_count"], 0)
        self.assertEqual(causal["ever_qualified_injected_track_count"], 0)
        self.assertEqual(causal["ever_qualified_unmatched_track_count"], 0)

    def test_rejects_legacy_track_records_without_quality_evidence(self) -> None:
        batches = [
            {
                "confirmed_or_coasted_tracks": [
                    {
                        "track_id": 1,
                        "state": {"timestamp_ns": 0},
                    }
                ]
            }
        ]
        truth_windows = [{"confirmed_track_matching": {"matches": []}}]
        with self.assertRaisesRegex(EvaluationError, "lacks quality_evidence"):
            summarize_track_quality(batches, truth_windows)


if __name__ == "__main__":
    unittest.main()
