from __future__ import annotations

import unittest
from dataclasses import asdict

import numpy as np

from tiny_target.detection import CandidateBatch, CandidateRecord
from tiny_target.tracking import KalmanTrackManager, KalmanTrackingConfig


def candidate(
    index: int,
    x: int,
    y: int,
    vx: float = 0,
    vy: float = 0,
    score: float = 9,
    selection_score: float | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_index=index,
        x_px=x,
        y_px=y,
        velocity_index=0,
        velocity_xy_px_s=(vx, vy),
        normalized_score_snr=score,
        raw_sum_score=score * 2,
        supporting_frame_count=4,
        support_weight=4,
        peak_neighbor_max_score_snr=score - 2,
        peak_contrast_snr=2,
        peak_to_neighbor_ratio=score / (score - 2),
        distance_to_border_px=10,
        distance_to_invalid_chebyshev_px=None,
        distance_to_invalid_is_lower_bound=True,
        ranking_score=selection_score,
        ranking_score_units=(
            "local_robust_sigma"
            if selection_score is not None
            else "normalized_shift_and_stack_snr"
        ),
    )


def batch(
    timestamp_ns: int,
    frames: tuple[int, ...],
    candidates: tuple[CandidateRecord, ...] = (),
    *,
    segment: int = 0,
) -> CandidateBatch:
    return CandidateBatch(
        candidates=candidates,
        frame_indices=frames,
        reference_timestamp_ns=timestamp_ns,
        segment_index=segment,
        metrics={},
        timings_ms={},
    )


def config(**overrides: object) -> KalmanTrackingConfig:
    values: dict[str, object] = {
        "position_measurement_sigma_px": 1.0,
        "velocity_measurement_sigma_px_s": 1.0,
        "acceleration_process_sigma_px_s2": 1.0,
        "initial_position_sigma_px": 2.0,
        "initial_velocity_sigma_px_s": 2.0,
        "mahalanobis_gate_squared": 16.0,
        "maximum_position_residual_px": 6.0,
        "maximum_velocity_residual_px_s": 3.0,
        "confirmation_independent_hits": 2,
        "max_missed_windows": 1,
        "maximum_timestamp_gap_s": 2.0,
        "measurement_noise_source": "synthetic_characterization",
        "max_active_tracks": 16,
    }
    values.update(overrides)
    return KalmanTrackingConfig(**values)


class KalmanTrackingConfigTests(unittest.TestCase):
    def test_configuration_is_explicit_and_unknown_values_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            KalmanTrackingConfig.from_mapping({})
        with self.assertRaisesRegex(ValueError, "at least 2"):
            config(confirmation_independent_hits=1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            KalmanTrackingConfig.from_mapping(
                {**asdict(config()), "mystery": 1}
            )


class KalmanTrackManagerTests(unittest.TestCase):
    def test_reservation_hints_are_causal_predictions_without_state_mutation(self) -> None:
        manager = KalmanTrackManager(config())
        manager.update(batch(0, (0,), (candidate(0, 10, 8, 2, 0),)))
        first = manager.reservation_hints(
            reference_timestamp_ns=500_000_000, segment_index=0
        )
        second = manager.reservation_hints(
            reference_timestamp_ns=500_000_000, segment_index=0
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertAlmostEqual(first[0].position_xy_px[0], 11)
        self.assertEqual(first[0].velocity_xy_px_s, (2.0, 0.0))
        self.assertEqual(first[0].age_windows, 1)
        self.assertEqual(first[0].associated_update_count, 1)
        self.assertEqual(first[0].missed_windows, 0)
        self.assertEqual(first[0].mean_measurement_speed_px_s, 2)
        self.assertEqual(
            manager.reservation_hints(
                reference_timestamp_ns=500_000_000, segment_index=1
            ),
            (),
        )
        self.assertEqual(
            manager.reservation_hints(
                reference_timestamp_ns=3_000_000_000, segment_index=0
            ),
            (),
        )
        updated = manager.update(
            batch(500_000_000, (1,), (candidate(0, 11, 8, 2, 0),))
        )
        self.assertEqual(updated.tracks[0].age_windows, 2)

    def test_quality_evidence_uses_bounded_running_moments_without_gating(self) -> None:
        manager = KalmanTrackManager(config())
        first = manager.update(
            batch(
                0,
                (0,),
                (candidate(0, 10, 10, 0, 0, 9, selection_score=12),),
            )
        )
        initial = first.tracks[0].to_dict()["quality_evidence"]
        self.assertFalse(initial["used_for_association_or_confirmation"])
        self.assertEqual(initial["observation_count"], 1)
        self.assertEqual(
            initial["kalman_innovation"]["association_count_excluding_birth"],
            0,
        )

        associated = manager.update(
            batch(
                100_000_000,
                (1,),
                (candidate(0, 10, 10, 1, 0, 13, selection_score=18),),
            )
        )
        quality = associated.tracks[0].to_dict()["quality_evidence"]
        self.assertEqual(quality["observation_count"], 2)
        self.assertEqual(quality["observation_fraction_of_age_windows"], 1.0)
        self.assertEqual(quality["detector_score_snr"]["minimum"], 9)
        self.assertEqual(quality["detector_score_snr"]["mean"], 11)
        self.assertEqual(quality["detector_score_snr"]["maximum"], 13)
        self.assertEqual(quality["selection_score"]["units"], "local_robust_sigma")
        self.assertEqual(quality["selection_score"]["mean"], 15)
        self.assertEqual(quality["measurement_velocity_step_px_s"]["count"], 1)
        self.assertEqual(quality["measurement_velocity_step_px_s"]["rms"], 1)
        self.assertEqual(
            quality["kalman_innovation"]["association_count_excluding_birth"],
            1,
        )

        coasted = manager.update(batch(200_000_000, (2,)))
        coasted_quality = coasted.tracks[0].to_dict()["quality_evidence"]
        self.assertEqual(coasted_quality["observation_count"], 2)
        self.assertAlmostEqual(
            coasted_quality["observation_fraction_of_age_windows"], 2 / 3
        )

    def test_overlapping_windows_update_but_do_not_confirm(self) -> None:
        manager = KalmanTrackManager(config())
        born = manager.update(batch(150_000_000, (0, 1, 2, 3), (candidate(0, 10, 10),)))
        self.assertEqual(born.tracks[0].lifecycle_state, "tentative")
        overlap = manager.update(
            batch(250_000_000, (1, 2, 3, 4), (candidate(0, 10, 10),))
        )
        self.assertEqual(overlap.tracks[0].associated_update_count, 2)
        self.assertEqual(overlap.tracks[0].independent_confirmation_hits, 1)
        self.assertFalse(
            overlap.associations[0]["independent_confirmation_evidence_credited"]
        )
        independent = manager.update(
            batch(550_000_000, (4, 5, 6, 7), (candidate(0, 10, 10),))
        )
        self.assertEqual(independent.tracks[0].lifecycle_state, "confirmed")
        self.assertEqual(independent.tracks[0].confirmation_latency_s, 0.4)

    def test_birth_miss_reacquisition_coast_and_deletion(self) -> None:
        manager = KalmanTrackManager(config(max_missed_windows=1))
        manager.update(batch(0, (0,), (candidate(0, 10, 10),)))
        missed = manager.update(batch(100_000_000, (1,)))
        self.assertEqual(missed.tracks[0].lifecycle_state, "tentative")
        reacquired = manager.update(
            batch(200_000_000, (2,), (candidate(0, 10, 10),))
        )
        self.assertEqual(reacquired.tracks[0].lifecycle_state, "confirmed")
        coasted = manager.update(batch(300_000_000, (3,)))
        self.assertEqual(coasted.tracks[0].lifecycle_state, "coasted")
        resumed = manager.update(
            batch(400_000_000, (4,), (candidate(0, 10, 10),))
        )
        self.assertEqual(resumed.tracks[0].lifecycle_state, "confirmed")
        manager.update(batch(500_000_000, (5,)))
        deleted = manager.update(batch(600_000_000, (6,)))
        self.assertEqual(deleted.tracks, ())
        self.assertEqual(deleted.deleted_tracks[0]["reason"], "maximum_missed_windows_exceeded")

    def test_crossing_tracks_keep_identity_using_velocity(self) -> None:
        manager = KalmanTrackManager(config(maximum_position_residual_px=5))
        outputs = []
        for step in range(4):
            outputs.append(
                manager.update(
                    batch(
                        step * 1_000_000_000,
                        (step,),
                        (
                            candidate(0, 10 + 2 * step, 12, 2, 0),
                            candidate(1, 20 - 2 * step, 12, -2, 0),
                        ),
                    )
                )
            )
        final = outputs[-1]
        self.assertEqual([track.track_id for track in final.tracks], [0, 1])
        self.assertGreater(final.tracks[0].state_xy_vx_vy[2], 0)
        self.assertLess(final.tracks[1].state_xy_vx_vy[2], 0)
        self.assertTrue(all(track.lifecycle_state == "confirmed" for track in final.tracks))

    def test_irregular_timestamps_drive_prediction(self) -> None:
        manager = KalmanTrackManager(config(maximum_position_residual_px=2))
        manager.update(batch(0, (0,), (candidate(0, 10, 8, 2, 0),)))
        manager.update(batch(500_000_000, (1,), (candidate(0, 11, 8, 2, 0),)))
        result = manager.update(
            batch(2_000_000_000, (2,), (candidate(0, 14, 8, 2, 0),))
        )
        self.assertEqual(result.tracks[0].track_id, 0)
        self.assertAlmostEqual(result.tracks[0].state_xy_vx_vy[0], 14, delta=0.2)
        self.assertAlmostEqual(result.tracks[0].state_xy_vx_vy[2], 2, delta=0.2)

    def test_segment_and_timestamp_gap_reset_coordinate_state(self) -> None:
        manager = KalmanTrackManager(config(maximum_timestamp_gap_s=0.5))
        first = manager.update(batch(0, (0,), (candidate(0, 10, 10),)))
        self.assertEqual(first.born_track_ids, (0,))
        segment = manager.update(
            batch(100_000_000, (1,), (candidate(0, 10, 10),), segment=1)
        )
        self.assertEqual(segment.reset_reason, "coordinate_segment_changed")
        self.assertEqual(segment.born_track_ids, (1,))
        gap = manager.update(
            batch(1_000_000_000, (2,), (candidate(0, 10, 10),), segment=1)
        )
        self.assertEqual(gap.reset_reason, "reference_timestamp_gap")
        self.assertEqual(gap.born_track_ids, (2,))

    def test_active_track_cap_drops_births_deterministically(self) -> None:
        manager = KalmanTrackManager(config(max_active_tracks=2))
        result = manager.update(
            batch(
                0,
                (0,),
                (candidate(0, 1, 1), candidate(1, 10, 10), candidate(2, 20, 20)),
            )
        )
        self.assertEqual(result.born_track_ids, (0, 1))
        self.assertEqual(result.metrics["dropped_birth_count_at_active_track_cap"], 1)
        self.assertEqual([track.last_candidate_index for track in result.tracks], [0, 1])

    def test_covariance_stays_finite_symmetric_and_positive(self) -> None:
        manager = KalmanTrackManager(config())
        for step in range(5):
            result = manager.update(
                batch(step * 100_000_000, (step,), (candidate(0, 10, 10),))
            )
        covariance = np.asarray(result.tracks[0].covariance)
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))


if __name__ == "__main__":
    unittest.main()
