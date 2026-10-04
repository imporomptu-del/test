from __future__ import annotations

import unittest

import numpy as np

from tiny_target.dense_screen import DensePointScreener, DenseScreenConfig
from tiny_target.detection import integrated_gaussian_kernel
from tiny_target.types import Frame, TimestampSource


def _frame(image: np.ndarray, index: int) -> Frame:
    return Frame(
        image=image,
        timestamp_ns=index * 333_333_333,
        frame_index=index,
        source_id="dense-screen-test",
        bit_depth=16,
        timestamp_source=TimestampSource.CONTAINER_RATE,
    )


def _inject(
    image: np.ndarray,
    x: float,
    y: float,
    flux_dn: float,
) -> np.ndarray:
    output = image.astype(np.float64, copy=True)
    center_x = int(round(x))
    center_y = int(round(y))
    radius = 3
    kernel = integrated_gaussian_kernel(
        0.8,
        radius,
        x - center_x,
        y - center_y,
    )
    output[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ] += flux_dn * kernel
    return np.clip(np.rint(output), 0, 65535).astype(np.uint16)


def _config(**overrides: object) -> DenseScreenConfig:
    values = {
        "crop_width": 96,
        "crop_height": 96,
        "background_warmup_frames": 3,
        "spatial_background_radius_px": 3,
        "tile_size_px": 48,
        "cfar_sample_stride_px": 2,
        "threshold_sigma": 4.0,
        "sigma_floor_dn": 1.0,
        "max_events_per_frame": 32,
        "association_radius_px_per_frame": 3.0,
        "minimum_track_hits": 4,
        "minimum_track_span_px": 1.0,
        "maximum_track_fit_rmse_px": 2.0,
        "retained_track_pool_size": 16,
        "max_shortlist_tracks_per_clip": 4,
        "opencv_threads": 1,
    }
    values.update(overrides)
    return DenseScreenConfig(**values)


class DenseScreenTests(unittest.TestCase):
    def test_unknown_configuration_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown dense-screen"):
            DenseScreenConfig.from_mapping({"mystery": 1})

    def test_moving_point_source_reaches_bounded_shortlist(self) -> None:
        rng = np.random.default_rng(75)
        screener = DensePointScreener(_config())
        truth = []
        for index in range(30):
            background = np.full((96, 96), 20000, np.float64)
            background += np.linspace(-100, 100, 96, dtype=np.float64)[None, :]
            background += rng.normal(0, 2, background.shape)
            x = 20.25 + 0.7 * index
            y = 35.25 + 0.3 * index
            truth.append((x, y))
            image = _inject(background, x, y, 1200.0)
            screener.process(_frame(image, index))
        result = screener.finalize()
        self.assertLessEqual(result["shortlist_count"], 4)
        self.assertGreater(result["shortlist_count"], 0)
        distances = []
        for track in result["shortlist"]:
            for event in track["events"]:
                x, y = event["position_xy_px"]
                tx, ty = truth[event["frame_index"]]
                distances.append(float(np.hypot(x - tx, y - ty)))
        self.assertLess(min(distances), 2.0)

    def test_static_hot_pixel_does_not_qualify_as_motion(self) -> None:
        screener = DensePointScreener(_config())
        for index in range(20):
            image = np.full((96, 96), 20000, np.uint16)
            image[30, 40] = 24000
            screener.process(_frame(image, index))
        result = screener.finalize()
        self.assertEqual(result["shortlist"], [])

    def test_event_and_shortlist_caps_are_hard_bounds(self) -> None:
        config = _config(max_events_per_frame=3, max_shortlist_tracks_per_clip=2)
        rng = np.random.default_rng(5)
        screener = DensePointScreener(config)
        for index in range(20):
            image = rng.integers(1000, 60000, size=(96, 96), dtype=np.uint16)
            screener.process(_frame(image, index))
        result = screener.finalize()
        self.assertLessEqual(result["point_events_per_frame"]["maximum"], 3)
        self.assertLessEqual(result["shortlist_count"], 2)

    def test_synthetic_candidate_persistence_links_a_linear_trajectory(self) -> None:
        screener = DensePointScreener(
            _config(
                synthetic_minimum_track_hits=3,
                synthetic_track_position_gate_px=4.0,
                synthetic_track_velocity_gate_px_s=1.0,
            )
        )
        for window_index in range(3):
            timestamp_ns = window_index * 2_000_000_000
            x = 20.0 + 2.0 * window_index
            entry = {
                "window_index": window_index,
                "frame_range": [8 * window_index, 8 * window_index + 15],
                "reference_frame_index": 8 * window_index + 8,
                "reference_timestamp_ns": timestamp_ns,
                "segment_index": 0,
                "ranking_score": 5.0,
                "candidate": {
                    "candidate_index": 0,
                    "discrete_position_xy_px": [x, 30.0],
                    "discrete_velocity_xy_px_s": [1.0, 0.0],
                    "normalized_score_snr": 6.0,
                },
                "followup_frame_range": [0, 64],
                "local_x_px": x,
                "local_y_px": 30.0,
                "velocity_x_px_s": 1.0,
                "velocity_y_px_s": 0.0,
            }
            screener._associate_synthetic_candidates(
                [entry],
                window_index=window_index,
                reference_timestamp_ns=timestamp_ns,
                segment_index=0,
            )
        self.assertEqual(len(screener._synthetic_active_tracks), 1)
        track = screener._synthetic_active_tracks.pop()
        screener._close_synthetic_track(track)
        self.assertEqual(len(screener._synthetic_qualified_tracks), 1)
        summary = screener._synthetic_qualified_tracks[0]
        self.assertEqual(summary["hit_count"], 3)
        self.assertAlmostEqual(summary["fitted_velocity_xy_px_s"][0], 1.0)
        # The result contract is emitted by the actual synthetic branch,
        # not only by a stand-alone report exporter. No CUDA work is needed
        # when serializing the already-associated synthetic hypotheses.
        screener._synthetic_window = object()
        result = screener._synthetic_tracking_result()
        self.assertEqual(result['output_contract']['retained_tracks']['count'], 1)
        self.assertEqual(result['track_pool'], result['shortlist'])
        self.assertTrue(result['output_contract']['review_preview']['is_complete_retained_set'])


if __name__ == "__main__":
    unittest.main()
