import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[2] / "scripts/accuracy_v37_registration.py"
SPEC = importlib.util.spec_from_file_location("registration_v37_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
AnnulusRegistration = MODULE.AnnulusRegistration


def texture():
    y, x = np.mgrid[-26:27, -26:27]
    return (105 + 13*np.sin(.31*x + .12*y) + 11*np.cos(.17*x - .23*y)
            + 7*np.sin(.08*x*y + .04*x)).astype(np.float64)


def independent_sample(prior, dx, dy):
    """Scalar bilinear oracle, separate indexing from vectorized implementation."""
    output = np.empty((49, 49), dtype=np.float64)
    for row in range(49):
        for col in range(49):
            x, y = col+2+dx, row+2+dy
            left, top = int(np.floor(x)), int(np.floor(y))
            fx, fy = x-left, y-top
            right, bottom = min(left+1, 52), min(top+1, 52)
            output[row, col] = ((1-fx)*(1-fy)*prior[top, left]
                + fx*(1-fy)*prior[top, right]
                + (1-fx)*fy*prior[bottom, left]
                + fx*fy*prior[bottom, right])
    return output


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.registration = AnnulusRegistration()

    def test_integer_shift_and_affine_photometry(self):
        prior = texture()
        current = 1.2*independent_sample(prior, 1, -1) - 8
        result = self.registration.measure(current, prior)
        self.assertTrue(result["available"], result["reasons"])
        self.assertEqual(result["shift_xy"], [1.0, -1.0])
        self.assertAlmostEqual(result["gain"], 1.2, places=12)
        self.assertAlmostEqual(result["offset"], -8, places=10)
        self.assertLess(result["mse"], 1e-24)
        self.assertFalse(result["registered_prior_patch_photometrically_corrected"])
        np.testing.assert_allclose(result["current_patch"],
            result["gain"]*result["registered_prior_patch"]+result["offset"], atol=1e-12)

    def test_half_pixel_shift_sign_and_scalar_sampling(self):
        prior = texture()
        current = independent_sample(prior, -.5, 1.5)
        result = self.registration.measure(current, prior)
        self.assertTrue(result["available"], result["reasons"])
        self.assertEqual(result["shift_xy"], [-.5, 1.5])
        np.testing.assert_allclose(result["registered_prior_patch"], current[12:37, 12:37])
        for dx, dy in ((-2, -2), (2, 2), (-1.5, .5), (0, 0)):
            sample, _ = self.registration._sample(prior, dx, dy)
            np.testing.assert_allclose(sample, independent_sample(prior, dx, dy), atol=1e-12)

    def test_current_target_center_cannot_drive_annulus(self):
        prior = texture()
        current = independent_sample(prior, .5, -.5)
        first = self.registration.measure(current, prior)
        changed = current.copy()
        changed[12:37, 12:37] = np.arange(625).reshape(25, 25) % 255
        second = self.registration.measure(changed, prior)
        for key in ("shift_xy", "gain", "offset", "mse", "mask_sha256", "hypotheses"):
            self.assertEqual(first[key], second[key])

    def test_previous_point_exclusion_is_search_box_dilation(self):
        prior = texture()
        current = independent_sample(prior, 0, 0)
        point = [15.0, 12.0]
        clean = self.registration.measure(current, prior, point)
        altered = prior.copy()
        yy, xx = np.mgrid[-26:27, -26:27]
        altered[(xx-point[0])**2 + (yy-point[1])**2 <= 9**2] += 20
        checked = self.registration.measure(current, altered, point)
        self.assertTrue(checked["available"], checked["reasons"])
        self.assertEqual(clean["mask_sha256"], checked["mask_sha256"])
        for key in ("shift_xy", "gain", "offset", "mse"):
            self.assertEqual(clean[key], checked[key])

    def test_all_hypotheses_share_common_mask_with_nan_and_saturation(self):
        prior = texture()
        current = independent_sample(prior, 0, 0)
        prior[4, 25] = np.nan
        prior[45, 26] = 255
        current[24, 44] = 0
        result = self.registration.measure(current, prior)
        self.assertTrue(result["common_mask_for_all_hypotheses"])
        self.assertGreater(len(set(result["individual_hypothesis_support_counts"])), 1)
        self.assertEqual({h["mask_count"] for h in result["hypotheses"]}, {result["mask_count"]})
        self.assertGreaterEqual(result["mask_count"], 128)
        self.assertTrue(result["available"], result["reasons"])

    def test_zero_weight_nan_corner_does_not_contaminate_integer_sample(self):
        prior = texture()
        expected = prior[26, 26]
        prior[26, 27] = np.nan
        sample, valid = self.registration._sample(prior, 0, 0)
        self.assertEqual(sample[24, 24], expected)
        self.assertTrue(valid[24, 24])
        shifted, valid_shifted = self.registration._sample(prior, .5, 0)
        self.assertTrue(np.isnan(shifted[24, 24]))
        self.assertFalse(valid_shifted[24, 24])

    def test_flat_and_saturated_are_unavailable(self):
        for value in (0., 100., 255.):
            result = self.registration.measure(np.full((49, 49), value), np.full((53, 53), value))
            self.assertFalse(result["available"])
            self.assertIsNone(result["registered_prior_patch"])

    def test_aperture_ambiguity_is_unknown(self):
        _, x = np.mgrid[-26:27, -26:27]
        prior = (105 + 20*np.sin(.31*x)).astype(float)
        result = self.registration.measure(independent_sample(prior, .5, 0), prior)
        self.assertFalse(result["available"])
        self.assertIn("ambiguous_nonlocal_registration", result["reasons"])
        self.assertLess(result["alternate_nonlocal_mse_gap"], 1/12)

    def test_exact_mse_tie_prefers_zero_but_still_flags_ambiguity(self):
        y, x = np.mgrid[0:53, 0:53]
        prior = (100 + 10*((x+y) % 2)).astype(float)
        current = prior[2:51, 2:51].copy()
        result = self.registration.measure(current, prior)
        self.assertEqual(result["shift_xy"], [0., 0.])
        self.assertEqual(result["mse"], 0)
        self.assertFalse(result["available"])
        self.assertIn("ambiguous_nonlocal_registration", result["reasons"])

    def test_search_boundary_is_unknown_even_for_exact_match(self):
        prior = texture()
        result = self.registration.measure(independent_sample(prior, 2, -.5), prior)
        self.assertFalse(result["available"])
        self.assertEqual(result["shift_xy"], [2., -.5])
        self.assertTrue(result["boundarywinner"])
        self.assertIn("search_boundary_winner", result["reasons"])

    def test_gain_outside_declared_range_is_unavailable(self):
        prior = texture()
        current = 3*independent_sample(prior, 0, 0)-210
        result = self.registration.measure(current, prior)
        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "no_admissible_fit")

    def test_large_annulus_residual_is_unavailable(self):
        prior = texture()
        rng = np.random.default_rng(314159)
        current = independent_sample(prior, 0, 0) + rng.normal(0, 18, (49, 49))
        result = self.registration.measure(current, prior)
        self.assertFalse(result["available"])
        self.assertIn("annulus_residual_too_large", result["reasons"])

    def test_insufficient_common_support_is_unknown(self):
        prior = texture()
        current = independent_sample(prior, 0, 0)
        current[:22, :] = np.nan
        current[27:, :] = np.nan
        current[:, :22] = np.nan
        current[:, 27:] = np.nan
        result = self.registration.measure(current, prior)
        self.assertEqual(result["status"], "insufficient_common_support")
        self.assertFalse(result["available"])

    def test_invalid_central_patch_does_not_pass_annulus_alone(self):
        prior = texture()
        current = independent_sample(prior, 0, 0)
        current[24, 24] = np.nan
        result = self.registration.measure(current, prior)
        self.assertFalse(result["available"])
        self.assertIn("invalid_evaluation_patch", result["reasons"])

    def test_bad_inputs_rejected(self):
        good_current = np.full((49, 49), 100.)
        good_prior = np.full((53, 53), 100.)
        for wrong in (np.zeros((48, 49)), np.full((49, 49), np.inf),
                      np.ones((49, 49), dtype=complex), "not pixels"):
            with self.subTest(wrong=type(wrong)):
                with self.assertRaises(ValueError):
                    self.registration.measure(wrong, good_prior)
        for point in ([np.inf, 0], [0], [True, 0], "point", [0, "x"],
                      np.array(1.), [1+2j, 0]):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    self.registration.measure(good_current, good_prior, point)

    def test_repeatability_no_mutation_and_owned_output_arrays(self):
        prior = texture()
        current = independent_sample(prior, -.5, .5)
        old_current, old_prior = current.copy(), prior.copy()
        first = self.registration.measure(current, prior)
        second = self.registration.measure(current, prior)
        for key in ("current_patch", "registered_prior_patch"):
            np.testing.assert_array_equal(first[key], second[key])
            first[key][:] = -999
            self.assertFalse(np.any(second[key] == -999))
        np.testing.assert_array_equal(current, old_current)
        np.testing.assert_array_equal(prior, old_prior)
        for result in (first, second):
            metadata = {k: v for k, v in result.items() if k not in ("current_patch", "registered_prior_patch")}
            json.dumps(metadata, allow_nan=False)
        first_metadata = {k: v for k, v in first.items() if k not in ("current_patch", "registered_prior_patch")}
        second_metadata = {k: v for k, v in second.items() if k not in ("current_patch", "registered_prior_patch")}
        self.assertEqual(first_metadata, second_metadata)


if __name__ == "__main__":
    unittest.main()
