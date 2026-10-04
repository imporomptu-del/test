"""Causal learning protection and exact residual-noise memory optimization."""
from dataclasses import replace
import unittest
import numpy as np
from tiny_target.visible_baseline import (
    VisibleConfig,
    VisiblePointDetector,
    VisibleTracks,
)


class LearningProtectionTests(unittest.TestCase):
    def test_variance_only_keeps_background_identical(self):
        base = VisibleConfig(
            pixel_noise_enabled=True, pixel_noise_model="background_residual"
        )
        plain = VisiblePointDetector(base)
        protected = VisiblePointDetector(
            replace(
                base,
                learning_exclusion_radius_px=6,
                learning_protection_mode="variance_only",
            )
        )
        frame = np.full((80, 80), 20, np.float32)
        valid = np.ones(frame.shape, bool)
        for i in range(15):
            if i:
                frame[40, 40] = 200
            plain.update(frame, valid, 0)
            protected.update(frame, valid, 0, [(40, 40)])
            np.testing.assert_array_equal(plain.background, protected.background)
        self.assertGreater(plain.variance[40, 40], protected.variance[40, 40])
        self.assertGreater(protected.background[40, 40], 0)
        before = float(protected.variance[40, 40])
        protected.update(frame, valid, 0)
        self.assertGreater(protected.variance[40, 40], before)

    def test_invalid_protection_mode_and_replay_mode_change_rejected(self):
        with self.assertRaises(ValueError):
            VisibleConfig(learning_protection_mode="unknown")
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        )
        spec = importlib.util.spec_from_file_location("variance_replay_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.validate_configuration(
                {}, VisibleConfig(learning_protection_mode="variance_only")
            )

    def test_moving_point_that_stops_is_not_learned_away_but_nuisance_still_is(self):
        base = VisibleConfig(
            warmup_frames=4,
            spatial_background="median5",
            pixel_noise_enabled=True,
            pixel_noise_model="background_residual",
            motion_quality_enabled=True,
        )
        results = {}
        for radius in (0, 6):
            cfg = replace(base, learning_exclusion_radius_px=radius)
            d = VisiblePointDetector(cfg)
            t = VisibleTracks(cfg, 10)
            hits = 0
            nuisance = 0
            for i in range(70):
                im = np.full((160, 180), 20, np.float32)
                valid = np.ones(im.shape, bool)
                x = 20 + min(max(i - 10, 0), 25) * 2
                if i >= 10:
                    im[39:42, x - 1 : x + 2] = 160
                im[120, 150] = 40 + i * 2
                ps, _ = d.update(im, valid, 0, t.learning_centers(i * 100_000_000, 0))
                rs, _ = t.update(ps, i, i * 100_000_000, 0, np.eye(3), im.shape)
                if i >= 40:
                    hits += any(
                        r["measured"]
                        and r["qualified_moving"]
                        and np.linalg.norm(
                            np.array(r["measurement_source_xy"]) - [x, 40]
                        )
                        <= 3
                        for r in rs
                    )
                    nuisance += any(
                        abs(p["x"] - 150) <= 2 and abs(p["y"] - 120) <= 2 for p in ps
                    )
            results[radius] = (hits, nuisance)
        self.assertLess(results[0][0], 30)
        self.assertEqual(results[6], (30, 0))

    def test_invalid_radius_rejected(self):
        for v in (-1, 17, True, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                VisibleConfig(learning_exclusion_radius_px=v)

    def test_frame_difference_retains_required_previous_frame_state(self):
        d = VisiblePointDetector(
            VisibleConfig(
                pixel_noise_enabled=True, pixel_noise_model="frame_difference"
            )
        )
        im = np.full((80, 80), 20, np.float32)
        d.update(im, np.ones(im.shape, bool), 0)
        self.assertEqual(d.previous_spatial.shape, im.shape)

    def test_replay_cannot_smuggle_a_detection_learning_change(self):
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        )
        spec = importlib.util.spec_from_file_location("learning_replay_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.validate_configuration(
                {}, VisibleConfig(learning_exclusion_radius_px=6)
            )

    def test_protection_does_not_create_candidates_or_freeze_other_pixels(self):
        cfg = VisibleConfig(
            spatial_background="median5",
            pixel_noise_enabled=True,
            pixel_noise_model="background_residual",
            learning_exclusion_radius_px=6,
            warmup_frames=2,
        )
        detector = VisiblePointDetector(cfg)
        frame = np.full((96, 96), 20, np.float32)
        valid = np.ones(frame.shape, bool)
        detector.update(frame, valid, 0)
        before = detector.variance.copy()
        frame[40, 40] = 200
        frame[70, 70] = 200
        proposals, _ = detector.update(frame, valid, 0, [(40, 40)])
        self.assertEqual(proposals, [])  # warmup is never bypassed
        self.assertEqual(detector.variance[40, 40], before[40, 40])
        self.assertGreater(detector.variance[70, 70], before[70, 70])
        self.assertEqual(detector.background[40, 40], 0)
        self.assertGreater(detector.background[70, 70], 0)
        self.assertIsNone(detector.previous_spatial)

    def test_only_previous_measured_qualified_same_segment_can_protect(self):
        t = VisibleTracks(VisibleConfig(learning_exclusion_radius_px=6), 10)
        t.previous_timestamp_ns = 1_000_000_000
        template = dict(
            reference_xy=[20, 30],
            velocity_reference_xy_px_s=[10, -10],
            segment=0,
            measured=True,
            qualified_moving=True,
        )
        t.previous_records = [
            template,
            dict(template, measured=False),
            dict(template, qualified_moving=False),
            dict(template, segment=1),
        ]
        self.assertEqual(t.learning_centers(1_100_000_000, 0), [[21, 29]])
        self.assertEqual(t.learning_centers(1_000_000_000, 0), [])
        self.assertEqual(t.learning_centers(2_000_000_000, 0), [])
        self.assertEqual(t.learning_centers(1_100_000_000, 2), [])

    def test_disabled_protection_ignores_centers(self):
        cfg = VisibleConfig(
            pixel_noise_enabled=True, pixel_noise_model="background_residual"
        )
        a, b = VisiblePointDetector(cfg), VisiblePointDetector(cfg)
        rng = np.random.default_rng(56)
        for _ in range(12):
            im = rng.normal(30, 2, (80, 80)).astype(np.float32)
            valid = np.ones(im.shape, bool)
            x, _ = a.update(im, valid, 0)
            y, _ = b.update(im, valid, 0, [(40, 40)])
            self.assertEqual(x, y)
            np.testing.assert_array_equal(a.background, b.background)
            np.testing.assert_array_equal(a.variance, b.variance)

    def test_learning_resumes_without_fresh_protection(self):
        cfg = VisibleConfig(
            pixel_noise_enabled=True,
            pixel_noise_model="background_residual",
            learning_exclusion_radius_px=6,
        )
        d = VisiblePointDetector(cfg)
        frame = np.full((80, 80), 20, np.float32)
        valid = np.ones(frame.shape, bool)
        d.update(frame, valid, 0)
        frame[40, 40] = 200
        d.update(frame, valid, 0, [(40, 40)])
        before = float(d.variance[40, 40])
        d.update(frame, valid, 0)
        self.assertGreater(d.variance[40, 40], before)


if __name__ == "__main__":
    unittest.main()
