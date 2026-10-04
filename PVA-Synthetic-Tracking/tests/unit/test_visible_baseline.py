import unittest
from dataclasses import replace
import cv2
import numpy as np
from tiny_target.visible_baseline import (
    VisibleConfig,
    VisiblePointDetector,
    VisibleTracks,
    CpuTranslation,
    map_point,
)
from tiny_target.tracking import KalmanTrackManager
from tiny_target.detection import CandidateBatch, CandidateRecord


def point(x, y, polarity="bright"):
    return dict(x=x, y=y, polarity=polarity, score=10.0, response_dn=10.0)


class VisibleTests(unittest.TestCase):
    def test_noise_must_include_persistent_background_model_error(self):
        models = [
            VisiblePointDetector(
                VisibleConfig(
                    warmup_frames=8,
                    spatial_background="median5",
                    pixel_noise_enabled=True,
                    pixel_noise_model=m,
                )
            )
            for m in ("background_residual", "frame_difference")
        ]
        valid = np.ones((80, 80), bool)
        hits = [0, 0]
        for i in range(100):
            frame = np.full((80, 80), 20.0, np.float32)
            frame[40, 40] = 30 + i
            for k, d in enumerate(models):
                proposals, _ = d.update(frame, valid, 0)
                if i >= 80:
                    hits[k] += any(
                        abs(p["x"] - 40) <= 2 and abs(p["y"] - 40) <= 2
                        for p in proposals
                    )
        self.assertEqual(hits[0], 0)
        self.assertGreater(hits[1], 0)

    def test_pixel_noise_learns_flickering_location_without_hiding_new_mover(self):
        enabled = [
            VisiblePointDetector(
                VisibleConfig(
                    warmup_frames=8,
                    spatial_background="median5",
                    pixel_noise_enabled=True,
                    pixel_noise_model=model,
                )
            )
            for model in ("frame_difference", "background_residual")
        ]
        control = VisiblePointDetector(
            VisibleConfig(
                warmup_frames=8, spatial_background="median5", pixel_noise_enabled=False
            )
        )
        valid = np.ones((128, 128), bool)
        noisy_hits = [0, 0, 0]
        moving_hits = [0, 0]
        for i in range(60):
            frame = np.full((128, 128), 20.0, np.float32)
            frame[90, 90] = 150 if i % 2 else 50
            if i >= 40:
                frame[30, 20 + 3 * (i - 40)] = 130
            for k, detector in enumerate((*enabled, control)):
                proposals, _ = detector.update(frame, valid, 0)
                if i >= 40:
                    noisy_hits[k] += any(
                        abs(p["x"] - 90) <= 2 and abs(p["y"] - 90) <= 2
                        for p in proposals
                    )
                    if k < 2:
                        moving_hits[k] += any(
                            abs(p["x"] - (20 + 3 * (i - 40))) <= 1
                            and abs(p["y"] - 30) <= 1
                            and p["polarity"] == "bright"
                            for p in proposals
                        )
        self.assertLess(noisy_hits[0], noisy_hits[2])
        self.assertLess(noisy_hits[1], noisy_hits[2])
        self.assertEqual(moving_hits, [20, 20])

    def test_configuration_rejects_implicit_raw16_and_bad_controls(self):
        for kwargs in (
            {"input_bit_depth": 16},
            {"background_alpha": 2},
            {"motion_backend": "identity"},
            {"noise_floor_dn": float("nan")},
            {"confirmation_hits": 1},
        ):
            with self.assertRaises(ValueError):
                VisibleConfig(**kwargs)

    def test_full_height_tile_seams_and_both_polarities(self):
        cfg = VisibleConfig(warmup_frames=2, spatial_background="median5")
        for x in (254, 255, 256, 257, 500):
            for polarity, amplitude in [("bright", 80), ("dark", -18)]:
                detector = VisiblePointDetector(cfg)
                base = np.full((2200, 520), 20.0, np.float32)
                valid = np.ones_like(base, dtype=bool)
                detector.update(base, valid, 0)
                detector.update(base, valid, 0)
                changed = base.copy()
                changed[2100, x] += amplitude
                proposals, metrics = detector.update(changed, valid, 0)
                self.assertTrue(
                    any(
                        p["polarity"] == polarity
                        and abs(p["x"] - x) <= 1
                        and abs(p["y"] - 2100) <= 1
                        for p in proposals
                    ),
                    (x, polarity),
                )
                self.assertEqual(metrics["full_shape_hw"], [2200, 520])
                self.assertIsNone(metrics["configured_crop"])

    def test_static_hot_pixel_not_a_temporal_candidate_and_invalid_support_not_searched(
        self,
    ):
        detector = VisiblePointDetector(VisibleConfig(warmup_frames=2))
        base = np.full((80, 80), 20.0, np.float32)
        base[40, 40] = 255
        valid = np.ones_like(base, dtype=bool)
        for _ in range(10):
            proposals, metrics = detector.update(base, valid, 0)
        self.assertEqual(proposals, [])
        valid[20:30, 20:30] = False
        base[25, 25] = 255
        proposals, metrics = detector.update(base, valid, 0)
        self.assertFalse(
            any(14 <= p["x"] <= 35 and 14 <= p["y"] <= 35 for p in proposals)
        )
        self.assertLess(metrics["searchable_pixels"], metrics["total_pixels"])

    def test_isolated_bright_point_does_not_create_dark_filter_ring_candidates(self):
        detector = VisiblePointDetector(
            VisibleConfig(warmup_frames=1, spatial_background="median5")
        )
        base = np.full((80, 80), 20.0, np.float32)
        valid = np.ones_like(base, dtype=bool)
        detector.update(base, valid, 0)
        changed = base.copy()
        changed[39:42, 39:42] = 250
        proposals, _ = detector.update(changed, valid, 0)
        self.assertTrue(any(p["polarity"] == "bright" for p in proposals))
        self.assertFalse(any(p["polarity"] == "dark" for p in proposals))

    def test_clipping_is_reported(self):
        cfg = VisibleConfig(
            warmup_frames=1,
            max_candidates_per_tile_polarity=1,
            max_candidates_per_frame=1,
            tile_size=32,
        )
        detector = VisiblePointDetector(cfg)
        base = np.full((96, 96), 20.0, np.float32)
        valid = np.ones_like(base, dtype=bool)
        detector.update(base, valid, 0)
        changed = base.copy()
        for y, x in [(10, 10), (20, 20), (50, 50)]:
            changed[y, x] = 200
        proposals, metrics = detector.update(changed, valid, 0)
        self.assertEqual(len(proposals), 1)
        self.assertGreater(metrics["dropped_at_tile_cap"], 0)
        self.assertGreater(metrics["dropped_at_frame_cap"], 0)

    def test_position_only_tracks_fast_intermittent_point_without_fake_velocity(self):
        tracker = VisibleTracks(VisibleConfig(), 10)
        ids = set()
        for i in range(30):
            records, _ = tracker.update(
                [point(30 + 8 * i, 100)] if i % 3 == 0 else [],
                i,
                i * 100_000_000,
                0,
                np.eye(3),
                (240, 400),
            )
            for r in records:
                if r["qualified_moving"]:
                    ids.add(r["track_id"])
                    if i % 3:
                        self.assertFalse(r["measured"])
                        self.assertIsNone(r["measurement_source_xy"])
        self.assertEqual(len(ids), 1)
        self.assertAlmostEqual(records[0]["velocity_reference_xy_px_s"][0], 80, delta=2)
        quality = (
            tracker.managers["bright"]
            ._record(next(iter(tracker.managers["bright"]._tracks.values())))
            .quality_evidence
        )
        self.assertEqual(quality["measurement_speed_px_s"]["count"], 0)

    def test_turn_and_polarity_separation(self):
        tracker = VisibleTracks(VisibleConfig(), 10)
        ids = set()
        for i in range(60):
            x = 120 + 0.06 * (i - 25) ** 2
            y = 180 - i
            records, _ = tracker.update(
                [point(round(x), y), point(round(x), y, "dark")],
                i,
                i * 100_000_000,
                0,
                np.eye(3),
                (300, 400),
            )
            ids.update(r["track_id"] for r in records if r["qualified_moving"])
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(r["velocity_reference_xy_px_s"][0] > 0 for r in records))

    def test_motion_inverse_and_rejected_motion_reset(self):
        rng = np.random.default_rng(42)
        base = rng.normal(100, 20, (128, 192)).astype(np.float32)
        motion = CpuTranslation(VisibleConfig(motion_proxy_max_dimension=256))
        motion.update(base, 0, 0)
        moved = cv2.warpAffine(base, np.array([[1.0, 0, 4], [0, 1.0, -3]]), (192, 128))
        _, _, matrix, segment, meta = motion.update(moved, 1, 100_000_000)
        self.assertFalse(meta["reset"])
        np.testing.assert_allclose(map_point(matrix, 64, 57), [60, 60], atol=0.4)
        _, _, _, new_segment, meta = motion.update(np.zeros_like(base), 2, 200_000_000)
        self.assertTrue(meta["reset"])
        self.assertGreater(new_segment, segment)

    def test_segment_change_cannot_keep_old_track_identity(self):
        tracker = VisibleTracks(VisibleConfig(), 10)
        for i in range(5):
            records, _ = tracker.update(
                [point(20 + 8 * i, 100)], i, i * 100_000_000, 0, np.eye(3), (240, 400)
            )
        old = records[0]["track_id"]
        records, _ = tracker.update(
            [point(60, 100)], 5, 500_000_000, 1, np.eye(3), (240, 400)
        )
        self.assertNotEqual(old, records[0]["track_id"])
        self.assertFalse(records[0]["qualified_moving"])

    def test_pva_adapter_failure_is_an_explicit_full_frame_segment_reset(self):
        # Exercise adapter plumbing, not the unavailable PVA hardware backend.
        from unittest.mock import Mock
        from tiny_target.visible_baseline import PvaMotion
        from tiny_target.motion import GlobalMotionConfig, PvaMotionError
        from tiny_target.stabilization import (
            FullResolutionStabilizer,
            StabilizationConfig,
        )

        motion = PvaMotion.__new__(PvaMotion)
        motion.estimator = Mock()
        motion.estimator.estimate.side_effect = PvaMotionError("injected test failure")
        motion.global_config = GlobalMotionConfig()
        motion.stabilizer = FullResolutionStabilizer(
            StabilizationConfig(backend="opencv_cpu")
        )
        motion.previous = None
        motion.tracker = None
        motion.cuda_warp = None
        frame = np.full((80, 96), 50, np.uint8)
        image, valid, matrix, segment, meta = motion.update(frame, 0, 0)
        self.assertEqual(image.shape, frame.shape)
        self.assertEqual(valid.shape, frame.shape)
        self.assertFalse(meta["pva_failure"])
        image, valid, matrix, new_segment, meta = motion.update(
            frame.copy(), 1, 100_000_000
        )
        self.assertTrue(meta["pva_failure"])
        self.assertTrue(meta["reset"])
        self.assertGreater(new_segment, segment)
        np.testing.assert_array_equal(matrix, np.eye(3))
        self.assertEqual(image.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
