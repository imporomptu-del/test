import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("nuisance_context_features_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/nuisance_context_features_v1.py"
spec = importlib.util.spec_from_file_location("context_features", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ContextFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.point = 6 * np.exp(-(m.XX ** 2 + m.YY ** 2) / 8)

    def test_frozen_gaussian_support_normalization(self):
        self.assertEqual(m.SIGMAS, (1.5, 4.0, 8.0))
        self.assertEqual((m.XX.min(), m.XX.max(), m.YY.min(), m.YY.max()), (-32, 32, -32, 32))
        for kernel in m.GAUSSIANS:
            self.assertEqual(kernel.shape, (65, 65))
            self.assertAlmostEqual(float(kernel.sum()), 1, places=15)
            self.assertFalse(kernel.flags.writeable)

    def test_dc_invariance_and_polarity(self):
        base = m.measure_patch(80 + self.point, "bright")
        offset = m.measure_patch(130 + self.point, "bright")
        dark = m.measure_patch(80 - self.point, "dark")
        self.assertAlmostEqual(base["A_dn"], offset["A_dn"], places=12)
        self.assertAlmostEqual(base["B_dn"], offset["B_dn"], places=12)
        self.assertAlmostEqual(base["R"], offset["R"], places=12)
        self.assertAlmostEqual(base["signed_A_dn"], dark["signed_A_dn"], places=12)
        self.assertAlmostEqual(base["A_dn"], -dark["A_dn"], places=12)
        self.assertGreaterEqual(base["R"], 0)
        self.assertLessEqual(base["R"], 1)
        self.assertFalse(base["classifier_applied"])

    def test_constant_guard_and_zero_constant(self):
        for value in (0, 80, 255):
            result = m.measure_patch(np.full((65, 65), value, np.uint8), "bright")
            self.assertEqual(result["A_dn"], 0)
            self.assertEqual(result["B_dn"], 0)
            self.assertIsNone(result["R"])
            self.assertTrue(result["denominator_numerically_zero"])
            self.assertFalse(result["interpretation_available"])

    def test_saturation_preserves_observed_values(self):
        p = 80 + self.point
        p[0, 0] = 255
        result = m.measure_patch(p, "bright")
        self.assertTrue(result["saturation"]["any_255"])
        self.assertFalse(result["interpretation_available"])
        self.assertIsNotNone(result["R"])
        pair = m.measure_pair(80 + self.point, p, 80 + self.point, "bright")
        self.assertTrue(pair["temporal"]["any_patch_saturated"])
        self.assertFalse(pair["temporal"]["interpretation_available"])

    def test_temporal_null_and_equal_errors(self):
        patch = 80 + self.point
        stopped = m.measure_pair(patch, patch, patch, "dark")["temporal"]
        self.assertEqual(stopped["error_sum_denominator_dn"], 0)
        self.assertIsNone(stopped["D"])
        equal = m.measure_pair(patch, np.full((65, 65), 80.), np.full((65, 65), 80.), "bright")["temporal"]
        self.assertEqual(equal["D"], 0)
        self.assertGreater(equal["error_sum_denominator_dn"], 0)

    def test_pair_availability_mirrors_temporal_separate_from_spatial(self):
        constant = np.full((65, 65), 80.)
        result = m.measure_pair(constant, 80 + self.point, constant, "bright")
        self.assertIsNone(result["spatial"]["current"]["R"])
        self.assertFalse(result["spatial"]["current"]["interpretation_available"])
        self.assertTrue(result["temporal"]["interpretation_available"])
        self.assertEqual(result["interpretation_available"], result["temporal"]["interpretation_available"])

    def test_temporal_polarity_invariance(self):
        cur, bg, actual = 80 + self.point, 80 + np.roll(self.point, -2, axis=1), 80 + self.point
        bright = m.measure_pair(cur, bg, actual, "bright")["temporal"]
        dark = m.measure_pair(160 - cur, 160 - bg, 160 - actual, "dark")["temporal"]
        for key in ("Acur_dn", "ApriorBG_dn", "ApriorActual_dn", "D"):
            self.assertAlmostEqual(bright[key], dark[key], places=12)
        self.assertEqual(bright["D"], 1)

    def test_invalid_patch_and_polarity(self):
        for p in (np.zeros((64, 65)), np.zeros((65, 65), np.int16), np.full((65, 65), np.nan)):
            with self.assertRaises(ValueError):
                m.measure_patch(p, "bright")
        with self.assertRaises(ValueError):
            m.measure_patch(np.zeros((65, 65)), "mixed")

    def test_bilinear_exact_borders_positive_corners(self):
        image = np.array([[0, 10], [20, 30]], np.uint8)
        result = m.bilinear_sample(image, np.array([0., 1., .5, 1., 1.01, -.001, np.nan]),
            np.array([0., 1., .5, .5, 1., 0., 0.]))
        np.testing.assert_array_equal(result[:4], [0, 30, 15, 20])
        self.assertTrue(np.isnan(result[4:]).all())
        singleton = m.bilinear_sample(np.array([[7.5]]), np.array([0, .1]), np.array([0, 0]))
        self.assertEqual(singleton[0], 7.5)
        self.assertTrue(np.isnan(singleton[1]))

    def test_bilinear_finite_float_values_not_clipped(self):
        image = np.array([[-100., 300.], [500., 900.]])
        result = m.bilinear_sample(image, [.5, 1], [.5, 1])
        np.testing.assert_array_equal(result, [400, 900])
        self.assertEqual(result.dtype, np.float64)

    def test_native_u8_sampling_does_not_copy_whole_image(self):
        image = np.zeros((100, 200), np.uint8)
        self.assertIs(m._image(image, as_float64=False), image)
        generated = np.ones((65, 65), np.float32)
        self.assertIs(m._image(generated, as_float64=False), generated)
        self.assertEqual(m._image(image).dtype, np.float64)

    def test_bilinear_invalid_input_or_map_shape(self):
        with self.assertRaises(ValueError):
            m.bilinear_sample(np.array([[np.nan]]), [0], [0])
        with self.assertRaises(ValueError):
            m.bilinear_sample(np.zeros((2, 2), np.uint8), [0, 1], [0])

    def test_fractional_native_saturation_not_hidden_by_interpolation(self):
        image = np.array([[255, 100], [100, 100]], np.uint8)
        x, y = np.array([.5, 1., .5]), np.array([.5, 1., 0.])
        samples = m.bilinear_sample(image, x, y)
        self.assertTrue(np.all((samples > 0) & (samples < 255)))
        np.testing.assert_array_equal(m.bilinear_saturation_mask(image, x, y), [True, False, True])
        image[0, 0] = 0
        self.assertTrue(m.bilinear_saturation_mask(image, [.5], [.5])[0])

    def test_saturation_zero_weight_neighbors_and_exact_borders(self):
        image = np.array([[100, 255], [0, 80]], np.uint8)
        np.testing.assert_array_equal(m.bilinear_saturation_mask(image, [0, 1, 1, -1, np.nan], [0, 1, 0, 0, 0]),
                                      [False, False, True, False, False])
        extended = np.array([[-1., 260.], [10., 20.]])
        self.assertTrue(m.bilinear_saturation_mask(extended, [.5], [.5])[0])

    def test_identity_transport_and_residual_displacement(self):
        result = m.transported_maps([100, 200], [98, 201], np.eye(3), np.eye(3))
        np.testing.assert_array_equal(result["current_x"], m.XX + 100)
        np.testing.assert_array_equal(result["prior_background_y"], m.YY + 200)
        np.testing.assert_array_equal(result["residual_displacement_xy"], [-2, 1])
        self.assertEqual(result["prior_track_x"][32, 32], 98)
        self.assertEqual(result["prior_track_y"][32, 32], 201)
        self.assertFalse(result["coordinate_reset_checked"])

    def test_camera_translation_transport_formula(self):
        hcur = np.array([[1, 0, -3], [0, 1, 2], [0, 0, 1.]])
        hprev = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1.]])
        result = m.transported_maps([100, 100], [98, 103], hcur, hprev)
        np.testing.assert_array_equal(result["F"], np.linalg.inv(hprev) @ hcur)
        np.testing.assert_array_equal(result["prior_background_x"], m.XX + 98)
        np.testing.assert_array_equal(result["prior_background_y"], m.YY + 103)
        np.testing.assert_array_equal(result["residual_displacement_xy"], [0, 0])

    def test_projective_grid_matches_explicit_mapping(self):
        hcur = np.array([[1, .01, 1], [.02, 1, 3], [.0001, .0002, 1]])
        result = m.transported_maps([100, 200], [101, 202], hcur, np.eye(3))
        for iy, ix in ((0, 0), (32, 32), (64, 64), (12, 50)):
            p = hcur @ np.array([100 + ix - 32, 200 + iy - 32, 1])
            self.assertAlmostEqual(result["prior_background_x"][iy, ix], p[0] / p[2], places=12)
            self.assertAlmostEqual(result["prior_background_y"][iy, ix], p[1] / p[2], places=12)

    def test_horizon_illconditioning_and_nonfinite_rejected(self):
        horizon = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -100]])
        for matrix in (horizon, np.diag([1., 1, 1e-14]), np.full((3, 3), np.nan), np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                m.transported_maps([100, 100], [100, 100], matrix, np.eye(3))

    def test_outside_predictions_remain_unclipped(self):
        result = m.transported_maps([0, 0], [-20, -30], np.eye(3), np.eye(3))
        self.assertEqual(result["prior_track_x"][32, 32], -20)
        values = m.bilinear_sample(np.ones((80, 80)), result["prior_track_x"], result["prior_track_y"])
        self.assertTrue(np.isnan(values[32, 32]))

    def test_generated_cases_scope_and_expected_degeneracy(self):
        cases = m.synthetic_cases()
        self.assertEqual(len(cases), 16)
        self.assertEqual(len({c["name"] for c in cases}), 16)
        measured = {}
        for case in cases:
            result = m.measure_pair(case["current"], case["prior_background"], case["prior_actual"], case["polarity"])
            measured[case["name"]] = result
            self.assertFalse(result["classifier_applied"])
            if case["name"].startswith(("stopped", "static_point")):
                self.assertIsNone(result["temporal"]["D"])
            elif case["name"].startswith("alternating_fixed"):
                self.assertEqual(result["temporal"]["D"], 0)
            elif not case["name"].startswith("point_on"):
                self.assertEqual(result["temporal"]["D"], 1)
        self.assertGreater(measured["dim_point_bright"]["spatial"]["current"]["R"],
                           measured["broad_moving_blob_bright"]["spatial"]["current"]["R"])
        self.assertLess(measured["slow_point_bright"]["temporal"]["error_sum_denominator_dn"],
                        measured["dim_point_bright"]["temporal"]["error_sum_denominator_dn"])

    def test_false_association_demonstrates_temporal_ambiguity(self):
        cases = {c["name"]: c for c in m.synthetic_cases()}
        for polarity in ("bright", "dark"):
            false = cases["alternating_sites_false_association_" + polarity]
            result = m.measure_pair(false["current"], false["prior_background"], false["prior_actual"], polarity)
            self.assertEqual(result["temporal"]["D"], 1)
            self.assertEqual(false["expected_scope"]["physical_entities"], 2)
            self.assertFalse(false["expected_scope"]["single_moving_object"])
            dim = cases["dim_point_" + polarity]
            self.assertEqual(abs(float(dim["current"][32, 32]) - 80), 2)


if __name__ == "__main__":
    unittest.main()
