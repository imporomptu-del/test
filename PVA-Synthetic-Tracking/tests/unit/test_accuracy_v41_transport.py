import hashlib
import json
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v41_transport import compare_templates


Y, X = np.mgrid[-12:13, -12:13]


def point(x=0.0, y=0.0, sigma=1.4, amplitude=70.0):
    return amplitude * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))


def patch(texture, offset=40.0):
    return offset + 0.1 * X + 0.2 * Y + texture


class TransportDiagnosticTest(unittest.TestCase):
    def test_static_blinking_point_is_explained_by_amplitude_and_plane(self):
        result = compare_templates(patch(1.7 * point(), 65), patch(point()), patch(point(-4)))
        self.assertTrue(result["available"])
        self.assertLess(result["mse_stationary"], 1e-20)
        self.assertLess(result["advantage_stationary_minus_transported"], -1.0)
        for fold in result["folds"]:
            self.assertAlmostEqual(fold["models"]["stationary"]["amplitude"], 1.7, places=12)

    def test_translated_point_favors_correctly_transported_template(self):
        result = compare_templates(patch(point()), patch(point(-4)), patch(point()))
        self.assertTrue(result["available"])
        self.assertLess(result["mse_transported"], 1e-20)
        self.assertGreater(result["advantage_stationary_minus_transported"], 1.0)

    def test_identical_templates_are_equal_without_false_classification(self):
        result = compare_templates(patch(1.2 * point()), patch(point()), patch(point()))
        self.assertTrue(result["available"])
        self.assertEqual(result["advantage_stationary_minus_transported"], 0.0)
        self.assertNotIn("accept", result)
        self.assertNotIn("class", result)
        self.assertNotIn("reject", result)

    def test_evolving_broad_texture_only_produces_continuous_scores(self):
        broad = point(-2, 1, sigma=6, amplitude=45)
        evolved = point(0, 1, sigma=5, amplitude=40) + point(5, -2, sigma=4, amplitude=15)
        result = compare_templates(patch(evolved), patch(broad), patch(point(0, 1, 6, 45)))
        self.assertTrue(result["available"])
        self.assertTrue(all(np.isfinite(result[key]) for key in
                            ("mse_stationary", "mse_transported", "mse_plane")))
        # Broad changing structure can favor transport without a compact object.
        self.assertGreater(result["advantage_stationary_minus_transported"], 0.0)

    def test_wrong_camera_registration_can_falsely_favor_transport(self):
        # A fixed light shifts on the sensor solely because the camera jittered.
        result = compare_templates(patch(point(2)), patch(point()), patch(point(2)))
        self.assertTrue(result["available"])
        self.assertGreater(result["advantage_stationary_minus_transported"], 1.0)

    def test_point_on_edge_is_not_assumed_to_be_a_negative(self):
        edge = 35.0 * (X > 2)
        current = patch(edge + point())
        result = compare_templates(current, patch(edge + point(-3)), patch(edge + point()))
        self.assertTrue(result["available"])
        self.assertLess(result["mse_transported"], 1e-20)
        self.assertGreater(result["advantage_stationary_minus_transported"], 0.0)

    def test_alternating_identical_fixed_lights_can_falsely_favor_transport(self):
        # Two fixed locations, one illuminated per frame: the transport prior
        # exactly mimics motion. These patches cannot establish physical motion.
        earlier_left_light = patch(point(-3))
        current_right_light = patch(point(3))
        transported_left_light = patch(point(3))
        result = compare_templates(current_right_light, earlier_left_light, transported_left_light)
        self.assertTrue(result["available"])
        self.assertGreater(result["advantage_stationary_minus_transported"], 1.0)
        self.assertLess(result["mse_transported"], 1e-20)

    def test_intermittent_prior_absence_is_unavailable_not_rejection(self):
        result = compare_templates(patch(point()), patch(np.zeros((25, 25))), patch(point()))
        self.assertFalse(result["available"])
        self.assertIn("insufficient_prior_texture:stationary:0", result["reasons"])
        self.assertIsNone(result["mse_stationary"])
        self.assertIsNone(result["advantage_stationary_minus_transported"])

    def test_flat_and_low_texture_priors_are_unavailable(self):
        for texture in (np.zeros((25, 25)), 0.01 * point()):
            result = compare_templates(patch(point()), patch(texture), patch(texture))
            self.assertFalse(result["available"])
            self.assertTrue(any("insufficient_prior_texture" in reason for reason in result["reasons"]))
            self.assertIsNone(result["mse_transported"])

    def test_rank_deficient_common_support_is_unavailable(self):
        # 25 x 25 cannot supply 32 collinear pixels. Three rows can, but a
        # constant/planar prior makes the four-column template design deficient.
        prior = patch(np.zeros((25, 25)))
        result = compare_templates(patch(point()), prior, prior)
        self.assertFalse(result["available"])
        self.assertIn("template_rank_deficient:stationary:0", result["reasons"])

    def test_border_nan_uses_exact_common_support_for_all_models(self):
        current, stationary, transported = patch(point()), patch(point(-2)), patch(point())
        current[0, :] = np.nan
        current[1, 1] = np.nan
        stationary[:, 0] = np.nan
        transported[-1, :] = np.nan
        result = compare_templates(current, stationary, transported)
        self.assertTrue(result["available"])
        mask = np.ones((25, 25), dtype=np.uint8)
        mask[0, :] = 0
        mask[1, 1] = 0
        mask[:, 0] = 0
        mask[-1, :] = 0
        self.assertEqual(result["common_support_count"], 551)
        self.assertEqual(result["common_support_sha256"], hashlib.sha256(mask.tobytes()).hexdigest())
        self.assertEqual(sum(result["fold_support_counts"]), 551)
        self.assertNotEqual(*result["fold_support_counts"])
        # Aggregation is weighted by held-out support, not mean-of-fold-means.
        for name in ("stationary", "transported", "plane"):
            weighted = sum(fold["heldout_count"] * (fold["mse_plane"] if name == "plane"
                           else fold["models"][name]["mse"]) for fold in result["folds"]) / 551
            self.assertAlmostEqual(result[f"mse_{name}"], weighted, places=12)

    def test_training_fold_does_not_fit_heldout_current_pixels(self):
        prior = patch(point())
        current = patch(1.2 * point())
        original = compare_templates(current, prior, prior)
        current[(X + Y) % 2 == 1] += 0.4 * point()[(X + Y) % 2 == 1]
        altered = compare_templates(current, prior, prior)
        self.assertTrue(altered["available"])
        self.assertEqual(original["common_support_sha256"], altered["common_support_sha256"])
        for name in ("stationary", "transported"):
            original_train0 = original["folds"][0]["models"][name]
            altered_train0 = altered["folds"][0]["models"][name]
            self.assertEqual(original_train0["amplitude"], altered_train0["amplitude"])
            self.assertGreater(altered_train0["mse"], original_train0["mse"])
            self.assertAlmostEqual(altered["folds"][1]["models"][name]["amplitude"], 1.6, places=12)

    def test_saturated_pixels_excluded_jointly_including_plane_baseline(self):
        a, b, c = patch(point()), patch(point(-1)), patch(point())
        a[0, 0], b[1, 0], c[2, 0] = 0, 255, 300
        a[3, 0], b[4, 0] = -1, np.nan
        result = compare_templates(a, b, c)
        self.assertTrue(result["available"])
        self.assertEqual(result["common_support_count"], 620)
        self.assertEqual(sum(fold["heldout_count"] for fold in result["folds"]), 620)

    def test_inadequate_total_or_single_parity_support_is_unavailable(self):
        current = patch(point())
        current[Y != 0] = np.nan
        result = compare_templates(current, patch(point()), patch(point()))
        self.assertFalse(result["available"])
        self.assertIn("insufficient_common_support", result["reasons"])
        current = patch(point())
        current[(X + Y) % 2 != 0] = np.nan
        result = compare_templates(current, patch(point()), patch(point()))
        self.assertFalse(result["available"])
        self.assertGreaterEqual(result["common_support_count"], 64)
        self.assertIn("insufficient_fold_support:1", result["reasons"])

    def test_exact_minimum_support_is_usable_but_one_missing_pixel_is_not(self):
        current = np.full((25, 25), np.nan)
        current[8:16, 8:16] = patch(point())[8:16, 8:16]
        result = compare_templates(current, patch(point(-2)), patch(point()))
        self.assertTrue(result["available"])
        self.assertEqual(result["fold_support_counts"], [32, 32])
        current[8, 8] = np.nan
        result = compare_templates(current, patch(point(-2)), patch(point()))
        self.assertFalse(result["available"])
        self.assertIn("insufficient_common_support", result["reasons"])
        self.assertIn("insufficient_fold_support:0", result["reasons"])

    def test_all_missing_prior_is_explicitly_unavailable_and_json_safe(self):
        result = compare_templates(patch(point()), np.full((25, 25), np.nan), patch(point()))
        self.assertFalse(result["available"])
        self.assertEqual(result["common_support_count"], 0)
        self.assertIsNone(result["mse_plane"])
        json.dumps(result, allow_nan=False)

    def test_infinity_complex_boolean_and_malformed_inputs_raise(self):
        invalid = [np.zeros((24, 25)), np.ones((25, 25), dtype=complex),
                   np.ones((25, 25), dtype=bool), np.full((25, 25), "12"),
                   np.full((25, 25), np.inf), np.full((25, 25), -np.inf),
                   None, [[1], [1, 2]]]
        for value in invalid:
            for index in range(3):
                args = [patch(point()), patch(point()), patch(point())]
                args[index] = value
                with self.subTest(index=index, dtype=getattr(value, "dtype", None)):
                    with self.assertRaises(ValueError):
                        compare_templates(*args)

    def test_negative_unconstrained_amplitude_refits_plane(self):
        result = compare_templates(patch(-point(), 130), patch(point()), patch(point()))
        self.assertTrue(result["available"])
        self.assertEqual(result["mse_stationary"], result["mse_plane"])
        self.assertEqual(result["mse_transported"], result["mse_plane"])
        for fold in result["folds"]:
            for model in fold["models"].values():
                self.assertEqual(model["amplitude"], 0.0)
                self.assertTrue(model["nonnegative_constraint_active"])

    def test_reproducible_json_safe_and_does_not_mutate_inputs(self):
        args = [patch(point()), patch(point(-2)), patch(point())]
        args[0][0, 0] = np.nan
        originals = [array.copy() for array in args]
        first = compare_templates(*args)
        second = compare_templates(*args)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False, sort_keys=True)
        for before, after in zip(originals, args):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
