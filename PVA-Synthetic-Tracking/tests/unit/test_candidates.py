from __future__ import annotations

import unittest

import numpy as np

from tiny_target.detection import (
    CandidateExtractionConfig,
    CandidateExtractionError,
    CandidateExtractor,
    SyntheticTrackWindow,
    TrackPredictionHint,
)


def window(
    score: np.ndarray,
    *,
    velocity_index: np.ndarray | None = None,
    support: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> SyntheticTrackWindow:
    score = np.asarray(score, np.float32)
    if velocity_index is None:
        velocity_index = np.zeros(score.shape, np.uint16)
    if support is None:
        support = np.full(score.shape, 4, np.uint16)
    if valid is None:
        valid = np.ones(score.shape, bool)
    return SyntheticTrackWindow(
        score=score,
        velocity_index=velocity_index,
        valid_support_count=support,
        valid_mask=valid,
        velocity_grid_xy_px_s=np.array([[0, 0], [1, 0], [4, 0]], np.float32),
        frame_indices=(10, 11, 12, 13),
        window_start_timestamp_ns=0,
        window_end_timestamp_ns=300,
        reference_timestamp_ns=150,
        segment_index=2,
        metrics={},
        timings_ms={},
    )


def config(**overrides: object) -> CandidateExtractionConfig:
    values: dict[str, object] = {
        "score_threshold_snr": 5.0,
        "minimum_support_frames": 4,
        "border_margin_px": 0,
        "invalid_margin_px": 0,
        "spatial_nms_radius_px": 2.1,
        "velocity_nms_radius_px_s": 1.1,
        "pre_nms_candidate_limit": 16,
        "max_candidates_per_window": 8,
    }
    values.update(overrides)
    return CandidateExtractionConfig(**values)


class CandidateConfigTests(unittest.TestCase):
    def test_operating_point_is_explicit_and_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            CandidateExtractionConfig.from_mapping({})
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            config(pre_nms_candidate_limit=2, max_candidates_per_window=3)
        with self.assertRaisesRegex(ValueError, "all be zero or all positive"):
            config(quota_grid_rows=2)
        with self.assertRaisesRegex(ValueError, "cannot satisfy"):
            config(
                quota_grid_rows=1,
                quota_grid_cols=1,
                pre_nms_candidates_per_cell=2,
                max_candidates_per_cell=2,
            )
        with self.assertRaisesRegex(ValueError, "all be zero or all positive"):
            config(track_guided_reservation_position_radius_px=8)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            config(
                track_guided_reservation_position_radius_px=8,
                track_guided_reservation_velocity_radius_px_s=3,
                max_track_guided_reservations_per_window=9,
            )


class CandidateExtractionTests(unittest.TestCase):
    @staticmethod
    def hint(
        track_id: int,
        x: float,
        y: float,
        *,
        lifecycle_state: str = "tentative",
        age_windows: int = 4,
        associated_update_count: int = 4,
        missed_windows: int = 0,
        mean_measurement_speed_px_s: float = 0.25,
    ) -> TrackPredictionHint:
        return TrackPredictionHint(
            track_id=track_id,
            position_xy_px=(x, y),
            velocity_xy_px_s=(0.0, 0.0),
            lifecycle_state=lifecycle_state,
            age_windows=age_windows,
            associated_update_count=associated_update_count,
            independent_confirmation_hits=1,
            missed_windows=missed_windows,
            mean_measurement_speed_px_s=mean_measurement_speed_px_s,
        )

    def test_record_has_score_support_velocity_and_peak_evidence(self) -> None:
        score = np.zeros((9, 11), np.float32)
        score[4, 5] = 10
        score[4, 4] = 4
        velocity = np.zeros(score.shape, np.uint16)
        velocity[4, 5] = 1
        batch = CandidateExtractor(config()).extract(
            window(score, velocity_index=velocity)
        )
        self.assertEqual(len(batch.candidates), 1)
        candidate = batch.candidates[0]
        self.assertEqual((candidate.x_px, candidate.y_px), (5, 4))
        self.assertEqual(candidate.velocity_xy_px_s, (1.0, 0.0))
        self.assertEqual(candidate.raw_sum_score, 20.0)
        self.assertEqual(candidate.peak_neighbor_max_score_snr, 4.0)
        self.assertEqual(candidate.peak_contrast_snr, 6.0)
        self.assertEqual(candidate.peak_to_neighbor_ratio, 2.5)
        serialized = batch.to_dict()
        self.assertIsNone(serialized["candidates"][0]["refined_position_xy_px"])
        self.assertFalse(serialized["metrics"]["per_velocity_local_maxima_available"])

    def test_joint_position_velocity_nms_collapses_adjacent_bins(self) -> None:
        score = np.zeros((9, 12), np.float32)
        score[4, 4] = 10
        score[4, 6] = 9
        velocity = np.zeros(score.shape, np.uint16)
        velocity[4, 6] = 1
        batch = CandidateExtractor(config()).extract(
            window(score, velocity_index=velocity)
        )
        self.assertEqual([(c.x_px, c.y_px) for c in batch.candidates], [(4, 4)])
        self.assertEqual(batch.metrics["joint_nms_suppressed_count"], 1)

    def test_close_peaks_with_distinct_velocities_are_retained(self) -> None:
        score = np.zeros((9, 12), np.float32)
        score[4, 4] = 10
        score[4, 6] = 9
        velocity = np.zeros(score.shape, np.uint16)
        velocity[4, 6] = 2
        batch = CandidateExtractor(config()).extract(
            window(score, velocity_index=velocity)
        )
        self.assertEqual(len(batch.candidates), 2)

    def test_border_invalid_margin_and_low_support_are_rejected(self) -> None:
        score = np.zeros((10, 12), np.float32)
        score[0, 5] = 12
        score[5, 5] = 11
        score[5, 9] = 10
        score[8, 5] = 9
        support = np.full(score.shape, 4, np.uint16)
        support[8, 5] = 3
        valid = np.ones(score.shape, bool)
        valid[5, 10] = False
        batch = CandidateExtractor(
            config(border_margin_px=1, invalid_margin_px=1)
        ).extract(window(score, support=support, valid=valid))
        self.assertEqual([(c.x_px, c.y_px) for c in batch.candidates], [(5, 5)])

    def test_ineligible_higher_neighbor_does_not_hide_supported_peak(self) -> None:
        score = np.zeros((7, 8), np.float32)
        score[3, 3] = 9
        score[3, 4] = 20
        support = np.full(score.shape, 4, np.uint16)
        support[3, 4] = 3
        batch = CandidateExtractor(config()).extract(window(score, support=support))
        self.assertEqual([(c.x_px, c.y_px) for c in batch.candidates], [(3, 3)])

    def test_no_target_is_empty(self) -> None:
        batch = CandidateExtractor(config()).extract(
            window(np.zeros((8, 9), np.float32))
        )
        self.assertEqual(batch.candidates, ())
        self.assertEqual(batch.metrics["spatial_local_maximum_count"], 0)

    def test_cap_and_equal_score_order_are_deterministic_and_reported(self) -> None:
        score = np.zeros((9, 17), np.float32)
        for x in (1, 4, 7, 10, 13, 16):
            score[4, x] = 8
        extractor = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                pre_nms_candidate_limit=4,
                max_candidates_per_window=2,
            )
        )
        first = extractor.extract(window(score))
        second = extractor.extract(window(score))
        self.assertEqual(
            [(c.x_px, c.y_px) for c in first.candidates],
            [(1, 4), (4, 4)],
        )
        self.assertEqual(first.to_dict()["candidates"], second.to_dict()["candidates"])
        self.assertTrue(first.metrics["pre_nms_truncated"])
        self.assertTrue(first.metrics["output_truncated"])
        self.assertTrue(first.metrics["counts_after_pre_nms_limit_are_lower_bounds"])

    def test_support_requirement_cannot_exceed_window(self) -> None:
        with self.assertRaisesRegex(CandidateExtractionError, "exceeds"):
            CandidateExtractor(config(minimum_support_frames=5)).extract(
                window(np.zeros((5, 5), np.float32))
            )

    def test_tile_robust_cfar_recovers_target_below_global_raw_top_k(self) -> None:
        rng = np.random.default_rng(75)
        score = rng.normal(0, 1, (64, 96)).astype(np.float32)
        score[:, :48] = rng.normal(50, 10, (64, 48))
        score[32, 72] = 10
        raw = CandidateExtractor(
            config(
                score_threshold_snr=5,
                spatial_nms_radius_px=0,
                pre_nms_candidate_limit=8,
                max_candidates_per_window=4,
            )
        ).extract(window(score))
        self.assertNotIn((72, 32), [(item.x_px, item.y_px) for item in raw.candidates])
        cfar = CandidateExtractor(
            config(
                score_threshold_snr=5,
                ranking_mode="tile_robust_cfar",
                cfar_threshold_sigma=5,
                cfar_tile_height_px=32,
                cfar_tile_width_px=24,
                cfar_minimum_samples=512,
                cfar_scale_floor_snr=0.5,
                spatial_nms_radius_px=0,
                pre_nms_candidate_limit=8,
                max_candidates_per_window=4,
            )
        ).extract(window(score))
        self.assertIn((72, 32), [(item.x_px, item.y_px) for item in cfar.candidates])
        target = next(item for item in cfar.candidates if (item.x_px, item.y_px) == (72, 32))
        self.assertGreater(target.ranking_score, 5)
        self.assertEqual(target.ranking_score_units, "local_robust_sigma")
        self.assertIsNotNone(target.local_clutter_center_snr)
        self.assertEqual(cfar.metrics["ranking"]["mode"], "tile_robust_cfar")

    def test_spatial_quota_prevents_one_cell_from_monopolizing_output(self) -> None:
        score = np.zeros((40, 40), np.float32)
        for index, (x, y) in enumerate(
            [(2, 2), (5, 2), (8, 2), (11, 2), (25, 5), (5, 25), (25, 25)]
        ):
            score[y, x] = 20 - index
        batch = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                pre_nms_candidate_limit=16,
                max_candidates_per_window=4,
                quota_grid_rows=2,
                quota_grid_cols=2,
                pre_nms_candidates_per_cell=4,
                max_candidates_per_cell=1,
            )
        ).extract(window(score))
        cells = [item.quota_cell_row_col for item in batch.candidates]
        self.assertEqual(set(cells), {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertEqual(batch.metrics["spatial_quota"]["occupied_output_cells"], 4)
        self.assertEqual(
            batch.metrics["spatial_quota"]["maximum_output_candidates_in_one_cell"],
            1,
        )

    def test_track_guided_reservation_recovers_quota_suppressed_peak(self) -> None:
        score = np.zeros((40, 40), np.float32)
        score[2, 2] = 20
        score[12, 12] = 10
        extractor = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                max_candidates_per_window=4,
                quota_grid_rows=2,
                quota_grid_cols=2,
                pre_nms_candidates_per_cell=4,
                max_candidates_per_cell=1,
                track_guided_reservation_position_radius_px=3,
                track_guided_reservation_velocity_radius_px_s=1,
                max_track_guided_reservations_per_window=1,
            )
        )
        batch = extractor.extract(
            window(score),
            track_prediction_hints=(self.hint(7, 12, 12),),
        )
        self.assertEqual(
            [(item.x_px, item.y_px) for item in batch.candidates],
            [(2, 2), (12, 12)],
        )
        rescued = batch.candidates[1]
        self.assertEqual(rescued.track_guided_reservation_track_id, 7)
        self.assertEqual(
            rescued.to_dict()["selection"]["reservation_track_id"], 7
        )
        metrics = batch.metrics["track_guided_reservation"]
        self.assertEqual(metrics["retained_reservation_count"], 1)
        self.assertEqual(metrics["reserved_track_ids"], [7])

    def test_disabled_reservation_is_serialization_equivalent(self) -> None:
        score = np.zeros((9, 11), np.float32)
        score[4, 5] = 10
        extractor = CandidateExtractor(config())
        without_hints = extractor.extract(window(score)).to_dict()
        with_hints = extractor.extract(
            window(score), track_prediction_hints=(self.hint(3, 5, 4),)
        ).to_dict()
        self.assertEqual(with_hints["candidates"], without_hints["candidates"])
        self.assertEqual(with_hints["metrics"], without_hints["metrics"])
        self.assertNotIn("track_guided_reservation", with_hints["metrics"])

    def test_reservation_cap_displaces_baseline_without_growing_output(self) -> None:
        score = np.zeros((40, 40), np.float32)
        for value, (x, y) in (
            (20, (2, 2)),
            (19, (22, 2)),
            (18, (2, 22)),
            (17, (22, 22)),
            (16, (12, 12)),
            (15, (32, 12)),
        ):
            score[y, x] = value
        batch = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                max_candidates_per_window=4,
                quota_grid_rows=2,
                quota_grid_cols=2,
                pre_nms_candidates_per_cell=4,
                max_candidates_per_cell=1,
                track_guided_reservation_position_radius_px=2,
                track_guided_reservation_velocity_radius_px_s=1,
                max_track_guided_reservations_per_window=1,
            )
        ).extract(
            window(score),
            track_prediction_hints=(
                self.hint(1, 12, 12),
                self.hint(2, 32, 12),
            ),
        )
        self.assertEqual(len(batch.candidates), 4)
        metrics = batch.metrics["track_guided_reservation"]
        self.assertEqual(metrics["retained_reservation_count"], 1)
        self.assertEqual(metrics["baseline_candidates_displaced_at_global_cap"], 1)

    def test_reservation_prioritizes_mature_tentative_and_excludes_confirmed(self) -> None:
        score = np.zeros((40, 40), np.float32)
        for value, (x, y) in (
            (30, (2, 2)),
            (29, (22, 2)),
            (28, (2, 22)),
            (27, (22, 22)),
            (26, (12, 12)),
            (25, (32, 12)),
            (24, (12, 32)),
        ):
            score[y, x] = value
        batch = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                max_candidates_per_window=4,
                quota_grid_rows=2,
                quota_grid_cols=2,
                pre_nms_candidates_per_cell=4,
                max_candidates_per_cell=1,
                track_guided_reservation_position_radius_px=2,
                track_guided_reservation_velocity_radius_px_s=1,
                max_track_guided_reservations_per_window=1,
            )
        ).extract(
            window(score),
            track_prediction_hints=(
                self.hint(1, 12, 12, lifecycle_state="confirmed"),
                self.hint(
                    2,
                    32,
                    12,
                    age_windows=1,
                    associated_update_count=1,
                ),
                self.hint(3, 12, 32),
            ),
        )
        reserved = [
            item
            for item in batch.candidates
            if item.track_guided_reservation_track_id is not None
        ]
        self.assertEqual(len(reserved), 1)
        self.assertEqual((reserved[0].x_px, reserved[0].y_px), (12, 32))
        self.assertEqual(reserved[0].track_guided_reservation_track_id, 3)
        metrics = batch.metrics["track_guided_reservation"]
        self.assertEqual(metrics["configured_hint_count"], 3)
        self.assertEqual(metrics["eligible_continuous_tentative_hint_count"], 1)
        self.assertEqual(
            metrics["hint_eligibility_policy"],
            "tentative_zero_miss_full_window_continuous_observation_"
            "and_minimum_mean_measurement_speed",
        )

    def test_reservation_speed_admission_rejects_stationary_track(self) -> None:
        score = np.zeros((40, 40), np.float32)
        for value, (x, y) in (
            (30, (2, 2)),
            (29, (22, 2)),
            (28, (2, 22)),
            (27, (22, 22)),
            (26, (12, 12)),
            (25, (32, 12)),
        ):
            score[y, x] = value
        batch = CandidateExtractor(
            config(
                spatial_nms_radius_px=0,
                max_candidates_per_window=4,
                quota_grid_rows=2,
                quota_grid_cols=2,
                pre_nms_candidates_per_cell=4,
                max_candidates_per_cell=1,
                track_guided_reservation_position_radius_px=3,
                track_guided_reservation_velocity_radius_px_s=1,
                track_guided_reservation_minimum_mean_speed_px_s=0.25,
                max_track_guided_reservations_per_window=1,
            )
        ).extract(
            window(score),
            track_prediction_hints=(
                self.hint(1, 12, 12, mean_measurement_speed_px_s=0),
                self.hint(2, 32, 12, mean_measurement_speed_px_s=0.25),
            ),
        )
        reserved = [
            item
            for item in batch.candidates
            if item.track_guided_reservation_track_id is not None
        ]
        self.assertEqual(len(reserved), 1)
        self.assertEqual(reserved[0].track_guided_reservation_track_id, 2)
        metrics = batch.metrics["track_guided_reservation"]
        self.assertEqual(metrics["eligible_continuous_tentative_hint_count"], 1)
        self.assertEqual(metrics["minimum_mean_measurement_speed_px_s"], 0.25)


if __name__ == "__main__":
    unittest.main()
