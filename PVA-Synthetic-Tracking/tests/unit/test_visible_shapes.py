"""Observed shape measurements, separate objects, valleys and invalid support."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from tiny_target.visible_shapes import consolidate_half_height
from tiny_target.visible_baseline import (
    VisibleConfig,
    VisiblePointDetector,
    VisibleTracks,
)


def peak(x, y, polarity="bright", score=10):
    return dict(
        x=x, y=y, polarity=polarity, score=score, response_dn=score, noise_sigma_dn=1
    )


class ShapeTests(unittest.TestCase):
    def test_unequal_neighbors_do_not_pull_weak_centroid_onto_strong_peak(self):
        im = np.zeros((60, 60), np.float32)
        im[30, 26:31] = 2
        im[30, 26] = 20
        im[30, 30] = 3
        ps, m = consolidate_half_height(
            [peak(26, 30, score=20), peak(30, 30, score=3)], im, np.ones_like(im, bool)
        )
        self.assertEqual([(p["x"], p["y"]) for p in ps], [(26, 30), (30, 30)])
        self.assertEqual(m["merged_peak_count"], 0)

    def test_streak_two_peaks_becomes_one_observed_center(self):
        image = np.zeros((60, 60), np.float32)
        image[30, 25:33] = 8
        image[30, 25] = 10
        image[30, 32] = 11
        ps, m = consolidate_half_height(
            [peak(25, 30), peak(32, 30, score=11)], image, np.ones_like(image, bool)
        )
        self.assertEqual(len(ps), 1)
        self.assertEqual(m["merged_peak_count"], 1)
        self.assertEqual(ps[0]["x"], 29)
        self.assertEqual(ps[0]["shape"]["peak_reference_xy"], [32, 30])
        self.assertEqual(ps[0]["score"], 11)

    def test_clear_valley_keeps_neighboring_objects_separate(self):
        y, x = np.mgrid[:60, :60]
        image = (
            10 * np.exp(-((x - 26) ** 2 + (y - 30) ** 2) / (2 * 0.7 ** 2))
            + 10 * np.exp(-((x - 30) ** 2 + (y - 30) ** 2) / (2 * 0.7 ** 2))
        ).astype(np.float32)
        ps, _ = consolidate_half_height(
            [peak(26, 30), peak(30, 30)], image, np.ones_like(image, bool)
        )
        self.assertEqual([(p["x"], p["y"]) for p in ps], [(26, 30), (30, 30)])

    def test_different_polarities_never_merge(self):
        image = np.zeros((60, 60), np.float32)
        image[30, 26], image[30, 30] = 10, -10
        ps, _ = consolidate_half_height(
            [peak(26, 30), peak(30, 30, "dark")], image, np.ones_like(image, bool)
        )
        self.assertEqual(len(ps), 2)

    def test_unsupported_or_unbounded_shapes_stay_original(self):
        image = np.zeros((60, 60), np.float32)
        image[30, 20:41] = 10
        p = peak(30, 30)
        mask = np.ones_like(image, bool)
        ps, m = consolidate_half_height([p], image, mask)
        self.assertEqual(ps, [p])
        self.assertEqual(m["merged_peak_count"], 0)
        image[:] = 0
        image[30, 29:32] = 10
        mask[30, 29] = False
        self.assertEqual(consolidate_half_height([p], image, mask)[0], [p])

    def test_hollow_region_does_not_create_center_measurement(self):
        image = np.zeros((60, 60), np.float32)
        image[27:34, 27:34] = 10
        image[28:33, 28:33] = 0
        p = peak(30, 27)
        self.assertEqual(
            consolidate_half_height([p], image, np.ones_like(image, bool))[0], [p]
        )

    def test_no_seed_no_candidate(self):
        im = np.ones((30, 30), np.float32)
        self.assertEqual(consolidate_half_height([], im, np.ones_like(im, bool))[0], [])

    def test_stationary_brightness_changes_do_not_become_moving_tracks(self):
        cfg = VisibleConfig(
            spatial_background="median5",
            shape_measurement_mode="mutual_half_height_r8",
            pixel_noise_enabled=True,
            pixel_noise_model="background_residual",
            motion_quality_enabled=True,
            warmup_frames=4,
        )
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        for i in range(50):
            im = np.full((90, 120), 20, np.float32)
            im[30, 30] = 100 + 40 * np.sin(i * 0.5)
            im[50, 70] = 140 if i % 4 < 2 else 50
            ps, _ = detector.update(im, np.ones_like(im, bool), 0)
            records, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            self.assertFalse(any(r["qualified_moving"] for r in records))

    def test_input_order_does_not_change_feature_positions(self):
        im = np.zeros((60, 60), np.float32)
        im[30, 25:33] = 10
        p = [peak(25, 30), peak(32, 30)]
        a, _ = consolidate_half_height(p, im, np.ones_like(im, bool))
        b, _ = consolidate_half_height(p[::-1], im, np.ones_like(im, bool))
        self.assertEqual([(r["x"], r["y"]) for r in a], [(r["x"], r["y"]) for r in b])

    def test_full_detector_and_tracker_two_neighbors_remain_separate(self):
        cfg = VisibleConfig(
            spatial_background="median5",
            shape_measurement_mode="mutual_half_height_r8",
            warmup_frames=4,
        )
        detector = VisiblePointDetector(cfg)
        tracker = VisibleTracks(cfg, 10)
        for i in range(25):
            im = np.full((90, 120), 20, np.float32)
            if i >= 6:
                im[35, 20 + i * 2] = 150
                im[41, 20 + i * 2] = 150
            ps, _ = detector.update(im, np.ones_like(im, bool), 0)
            tracks, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            if i >= 16:
                points = [
                    t
                    for t in tracks
                    if t["measured"]
                    and t["qualified_moving"]
                    and t["track_id"].startswith("bright:")
                ]
                self.assertEqual(len(points), 2)
                self.assertEqual(
                    sorted(round(t["measurement_source_xy"][1]) for t in points),
                    [35, 41],
                )

    def test_near_crossing_movers_keep_distinct_ids(self):
        cfg = VisibleConfig(
            spatial_background="median5",
            shape_measurement_mode="mutual_half_height_r8",
            warmup_frames=4,
            motion_quality_enabled=True,
        )
        detector, tracker = VisiblePointDetector(cfg), VisibleTracks(cfg, 10)
        identities = None
        for i in range(36):
            im = np.full((90, 120), 20, np.float32)
            if i >= 6:
                im[35, 10 + i * 2] = 150
                im[41, 100 - i * 2] = 100
            ps, _ = detector.update(im, np.ones_like(im, bool), 0)
            tracks, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
            if i >= 16:
                observed = {
                    round(t["measurement_source_xy"][1]): t["track_id"]
                    for t in tracks
                    if t["measured"]
                    and t["qualified_moving"]
                    and t["track_id"].startswith("bright:")
                }
                self.assertEqual(set(observed), {35, 41})
                if identities is None:
                    identities = observed
                self.assertEqual(observed, identities)

    def test_learning_and_disabled_behavior_unchanged_and_replay_forbidden(self):
        cfg = VisibleConfig(
            pixel_noise_enabled=True, pixel_noise_model="background_residual"
        )
        a = VisiblePointDetector(cfg)
        b = VisiblePointDetector(
            replace(cfg, shape_measurement_mode="mutual_half_height_r8")
        )
        rng = np.random.default_rng(27)
        for i in range(20):
            im = rng.normal(30, 2, (67, 99)).astype(np.float32)
            mask = np.ones_like(im, bool)
            a.update(im, mask, i // 10)
            b.update(im, mask, i // 10)
            np.testing.assert_array_equal(a.background, b.background)
            np.testing.assert_array_equal(a.variance, b.variance)
        self.assertEqual(VisibleConfig().shape_measurement_mode, "none")
        with self.assertRaises(ValueError):
            VisibleConfig(shape_measurement_mode="unknown")
        path = (
            Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        )
        spec = importlib.util.spec_from_file_location("shape_replay", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with self.assertRaises(ValueError):
            mod.validate_configuration(
                {}, replace(cfg, shape_measurement_mode="mutual_half_height_r8")
            )


if __name__ == "__main__":
    unittest.main()
