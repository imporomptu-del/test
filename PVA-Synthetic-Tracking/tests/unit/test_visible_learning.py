"""Observed shape, prior-only transport and no-detection-side-effect checks."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from tiny_target.visible_baseline import VisibleConfig, VisiblePointDetector, VisibleTracks
from tiny_target.visible_learning import shape_learning_mask
from tiny_target.visible_shapes import consolidate_half_height


def cfg():
    return VisibleConfig(spatial_background="median5", warmup_frames=4,
        pixel_noise_enabled=True, pixel_noise_model="background_residual",
        shape_measurement_mode="mutual_half_height_r8", motion_quality_enabled=True,
        learning_protection_mode="variance_only", learning_protection_geometry="observed_shape",
        tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity")


class ShapeLearningTests(unittest.TestCase):
    def test_observed_pixels_and_margin_not_six_pixel_blanket(self):
        support = np.ones((31, 41), bool)
        support[:, :2] = False
        learn = shape_learning_mask(support, [{"support_reference_xy": [[20, 15], [21, 15]]}], 2)
        self.assertFalse(learn[15, 18])
        self.assertFalse(learn[15, 23])
        self.assertTrue(learn[15, 15])
        self.assertTrue(learn[18, 20])
        self.assertFalse(learn[:, :2].any())

    def test_regions_must_be_bounded_finite_and_outside_points_safe(self):
        support = np.ones((21, 21), bool)
        for region in ([], [[float("nan"), 2]], [[2, 3]] * 1025):
            with self.assertRaises(ValueError):
                shape_learning_mask(support, [{"support_reference_xy": region}], 2)
        np.testing.assert_array_equal(shape_learning_mask(support,
            [{"support_reference_xy": [[1e100, 2], [-20, -20]]}], 2), support)

    def test_support_export_changes_no_shape_measurements(self):
        image = np.zeros((40, 40), np.float32)
        image[20, 18:22] = 10
        peak = dict(x=20, y=20, polarity="bright", score=10)
        a, am = consolidate_half_height([peak], image, np.ones_like(image, bool))
        b, bm = consolidate_half_height([peak], image, np.ones_like(image, bool), include_support=True)
        self.assertEqual(b[0]["shape"].pop("support_reference_xy"), [[18, 20], [19, 20], [20, 20], [21, 20]])
        self.assertEqual((a, am), (b, bm))

    def test_only_prior_qualified_measured_shapes_transport(self):
        tracker = VisibleTracks(cfg(), 10)
        self.assertEqual(tracker.learning_centers(100_000_000, 0), [])
        tracker.previous_timestamp_ns = 100_000_000
        observed = dict(segment=0, measured=True, qualified_moving=True,
            learning_shape_reference_xy=[[10, 20]], velocity_reference_xy_px_s=[20, -10])
        tracker.previous_records = [observed]
        self.assertEqual(tracker.learning_centers(200_000_000, 0),
                         [{"support_reference_xy": [[12, 19]]}])
        self.assertEqual(tracker.learning_centers(200_000_000, 1), [])
        self.assertEqual(tracker.learning_centers(1_000_000_000, 0), [])
        for change in ({"measured": False}, {"qualified_moving": False},
                       {"learning_shape_reference_xy": None}):
            tracker.previous_records = [{**observed, **change}]
            self.assertEqual(tracker.learning_centers(200_000_000, 0), [])

    def test_current_proposals_unchanged_background_unchanged_and_mask_only_variance(self):
        a, b = VisiblePointDetector(cfg()), VisiblePointDetector(cfg())
        image = np.full((61, 81), 20, np.float32)
        mask = np.ones(image.shape, bool)
        for i in range(8):
            a.update(image, mask, 0); b.update(image, mask, 0)
        image[30, 40] = 150
        x, _ = a.update(image, mask, 0)
        y, _ = b.update(image, mask, 0, [{"support_reference_xy": [[40, 30]]}])
        self.assertEqual(x, y)
        np.testing.assert_array_equal(a.background, b.background)
        self.assertGreater(a.variance[30, 40], b.variance[30, 40])

    def test_crossing_movers_and_flickering_lights(self):
        for moving in (False, True):
            detector, tracker = VisiblePointDetector(cfg()), VisibleTracks(cfg(), 10)
            identities = None
            for i in range(40):
                image = np.full((90, 140), 20, np.float32)
                if moving:
                    if i >= 6: image[35, 10 + i * 2] = 150
                    if i >= 12: image[41, 125 - i * 2] = 100
                else:
                    image[35, 40] = 100 + 40 * np.sin(i * 0.5)
                    image[41, 90] = 140 if i % 4 < 2 else 50
                ps, _ = detector.update(image, np.ones_like(image, bool), 0,
                                        tracker.learning_centers(i * 100_000_000, 0))
                rows, _ = tracker.update(ps, i, i * 100_000_000, 0, np.eye(3), image.shape)
                if not moving:
                    self.assertFalse(any(t["qualified_moving"] for t in rows))
                elif i >= 22:
                    observed = {round(t["measurement_source_xy"][1]): t["track_id"]
                        for t in rows if t["measured"] and t["qualified_moving"]
                        and t["track_id"].startswith("bright:")}
                    self.assertEqual(set(observed), {35, 41})
                    if identities is None: identities = observed
                    self.assertEqual(observed, identities)

    def test_invalid_policy_combinations_fail(self):
        self.assertEqual(VisibleConfig().learning_protection_geometry, "circle")
        for changes in ({"learning_exclusion_radius_px": 6},
                        {"shape_measurement_mode": "none"},
                        {"learning_protection_mode": "background_and_variance"}):
            with self.assertRaises(ValueError):
                replace(cfg(), **changes)

    def test_tracking_replay_cannot_change_closed_loop_detector_feedback(self):
        script = Path(__file__).resolve().parents[2] / "scripts/replay_phase20_tracking.py"
        spec = importlib.util.spec_from_file_location("shape_feedback_replay", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from dataclasses import asdict
        parent = cfg()
        with self.assertRaisesRegex(ValueError, "closed-loop"):
            module.validate_configuration(asdict(parent), replace(parent, tracking_association_prior="none"))
        self.assertEqual(module.validate_configuration(asdict(parent), parent), [])


if __name__ == "__main__":
    unittest.main()
