"""Analytical source-pixel diagnostics, not tuned to any recording or class.

Synthetic patches exercise competing point/edge fits and nuisance-background
invariance. No test treats weak, cropped, or broad structure as a class label.
"""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/accuracy_v36_context.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v36_context_under_test", MODULE_PATH)
context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = context
SPEC.loader.exec_module(context)

Y, X = np.mgrid[-12:13, -12:13].astype(np.float64)
DESIGN = np.column_stack([v.ravel() for v in (np.ones_like(X), X, Y, X * X, X * Y, Y * Y)])
REQUIRED = {
    "background_residual_energy", "point_gain_fraction", "edge_gain_fraction",
    "point_minus_edge_fraction", "point_amplitude_dn", "point_sigma_px",
    "point_offset_xy", "edge_width_px", "edge_orientation_rad",
    "edge_offset_px", "edge_amplitude_dn", "residual_rms_dn", "informative",
    "point_absolute_gain", "edge_absolute_gain", "edge_residual_energy",
    "conditional_informative", "point_gain_after_edge_fraction",
    "point_after_edge_absolute_gain", "point_after_edge_amplitude_dn",
    "point_after_edge_sigma_px", "point_after_edge_offset_xy",
}


def point(sigma=2.0, dx=0.0, dy=0.0):
    return np.exp(-((X - dx) ** 2 + (Y - dy) ** 2) / (2.0 * sigma ** 2))


def edge(width=2.0, angle=0.0, offset=0.0):
    return np.tanh((X * np.cos(angle) + Y * np.sin(angle) - offset) / width)


def background():
    return 93.0 + 0.7 * X - 1.3 * Y + 0.12 * X * X - 0.05 * X * Y + 0.09 * Y * Y


class AccuracyV36ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = context.PointEdgeDiagnostic()

    def assert_diagnostic(self, value):
        self.assertTrue(REQUIRED <= set(value))
        self.assertIsInstance(value["informative"], bool)
        self.assertIsInstance(value["conditional_informative"], bool)
        for key in ("background_residual_energy", "point_gain_fraction", "edge_gain_fraction",
                    "point_minus_edge_fraction", "point_amplitude_dn", "edge_amplitude_dn",
                    "residual_rms_dn", "point_absolute_gain", "edge_absolute_gain",
                    "edge_residual_energy", "point_gain_after_edge_fraction",
                    "point_after_edge_absolute_gain", "point_after_edge_amplitude_dn"):
            self.assertTrue(math.isfinite(value[key]), key)
        self.assertGreaterEqual(value["background_residual_energy"], 0.0)
        self.assertGreaterEqual(value["point_amplitude_dn"], 0.0)
        self.assertGreaterEqual(value["residual_rms_dn"], 0.0)
        self.assertGreaterEqual(value["point_after_edge_amplitude_dn"], 0.0)
        self.assertGreaterEqual(value["edge_residual_energy"], 0.0)
        for key in ("point_gain_fraction", "edge_gain_fraction", "point_gain_after_edge_fraction"):
            self.assertGreaterEqual(value[key], -1e-12)
            self.assertLessEqual(value[key], 1.0 + 1e-12)
        self.assertAlmostEqual(value["point_minus_edge_fraction"],
                               value["point_gain_fraction"] - value["edge_gain_fraction"], places=11)
        if value["informative"]:
            self.assertIn(value["point_sigma_px"], (1, 2, 3))
            self.assertEqual(len(value["point_offset_xy"]), 2)
            self.assertTrue(all(v in (-1, 0, 1) for v in value["point_offset_xy"]))
            self.assertIn(value["edge_width_px"], (1, 2, 4))
            self.assertIn(value["edge_offset_px"], (-2, 0, 2))
            angle_index = value["edge_orientation_rad"] / (math.pi / 8)
            self.assertAlmostEqual(angle_index, round(angle_index), places=10)
            self.assertIn(round(angle_index), range(8))

    def assert_same_gains(self, a, b, places=9):
        for key in ("point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction"):
            self.assertAlmostEqual(a[key], b[key], places=places, msg=key)
        self.assertAlmostEqual(a["point_gain_after_edge_fraction"],
                               b["point_gain_after_edge_fraction"], places=places)

    def test_centered_supported_gaussian_recovers_width_and_amplitude(self):
        for sigma in (1, 2, 3):
            with self.subTest(sigma=sigma):
                result = self.diagnostic.measure(background() + 18.0 * point(sigma), "bright")
                self.assert_diagnostic(result)
                self.assertTrue(result["informative"])
                self.assertAlmostEqual(result["point_gain_fraction"], 1.0, places=10)
                self.assertGreater(result["point_minus_edge_fraction"], 0.0)
                self.assertAlmostEqual(result["point_amplitude_dn"], 18.0, places=8)
                self.assertEqual(result["point_sigma_px"], sigma)
                self.assertEqual(tuple(result["point_offset_xy"]), (0, 0))

    def test_all_supported_center_offsets_are_searchable(self):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                with self.subTest(dx=dx, dy=dy):
                    result = self.diagnostic.measure(background() + 11.0 * point(2, dx, dy), "bright")
                    self.assertAlmostEqual(result["point_gain_fraction"], 1.0, places=10)
                    self.assertAlmostEqual(result["point_amplitude_dn"], 11.0, places=8)
                    self.assertEqual(tuple(result["point_offset_xy"]), (dx, dy))
                    self.assertEqual(result["point_sigma_px"], 2)

    def test_subpixel_center_is_diagnostic_not_claimed_exact_grid_recovery(self):
        result = self.diagnostic.measure(100.0 + 13.0 * point(2, 0.35, -0.45), "bright")
        self.assert_diagnostic(result)
        self.assertTrue(result["informative"])
        self.assertGreater(result["point_gain_fraction"], result["edge_gain_fraction"])
        self.assertLess(result["point_gain_fraction"], 1.0 - 1e-8)

    def test_bright_dark_intensity_inversion_symmetry(self):
        patch = background() + 17.0 * point(2, 1, -1) + 2.5 * edge(4, math.pi / 8, 2)
        bright = self.diagnostic.measure(patch, "bright")
        dark = self.diagnostic.measure(255.0 - patch, "dark")
        self.assert_diagnostic(bright)
        self.assert_diagnostic(dark)
        self.assert_same_gains(bright, dark)
        self.assertAlmostEqual(bright["background_residual_energy"], dark["background_residual_energy"], places=7)
        self.assertAlmostEqual(bright["point_amplitude_dn"], dark["point_amplitude_dn"], places=8)
        self.assertEqual(bright["point_sigma_px"], dark["point_sigma_px"])
        self.assertEqual(tuple(bright["point_offset_xy"]), tuple(dark["point_offset_xy"]))

    def test_matching_dark_point_has_nonnegative_reported_amplitude(self):
        result = self.diagnostic.measure(background() - 23.0 * point(3, -1, 1), "dark")
        self.assert_diagnostic(result)
        self.assertAlmostEqual(result["point_gain_fraction"], 1.0, places=10)
        self.assertAlmostEqual(result["point_amplitude_dn"], 23.0, places=8)
        self.assertEqual(tuple(result["point_offset_xy"]), (-1, 1))

    def test_opposite_polarity_cannot_fit_point_with_negative_amplitude(self):
        patch = background() - 20.0 * point(2)
        result = self.diagnostic.measure(patch, "bright")
        self.assert_diagnostic(result)
        self.assertAlmostEqual(result["point_amplitude_dn"], 0.0, places=10)
        self.assertAlmostEqual(result["point_gain_fraction"], 0.0, places=10)

    def test_supported_straight_edges_fit_better_than_points(self):
        for width in (1, 2, 4):
            for angle in (0.0, math.pi / 4, 7 * math.pi / 8):
                for offset in (-2, 0, 2):
                    with self.subTest(width=width, angle=angle, offset=offset):
                        result = self.diagnostic.measure(background() + 12.0 * edge(width, angle, offset), "bright")
                        self.assert_diagnostic(result)
                        self.assertTrue(result["informative"])
                        self.assertAlmostEqual(result["edge_gain_fraction"], 1.0, places=10)
                        self.assertLess(result["point_minus_edge_fraction"], 0.0)
                        self.assertAlmostEqual(result["edge_amplitude_dn"], 12.0, places=8)
                        self.assertEqual(result["edge_width_px"], width)
                        self.assertEqual(result["edge_offset_px"], offset)
                        self.assertAlmostEqual(result["edge_orientation_rad"], angle, places=10)
                        self.assertFalse(result["conditional_informative"])
                        self.assertEqual(result["point_gain_after_edge_fraction"], 0.0)
                        self.assertEqual(result["point_after_edge_amplitude_dn"], 0.0)

    def test_edge_amplitude_is_unrestricted_and_independent_of_requested_polarity(self):
        patch = background() - 9.0 * edge(2, math.pi / 4, 2)
        bright = self.diagnostic.measure(patch, "bright")
        dark = self.diagnostic.measure(patch, "dark")
        for value in (bright, dark):
            self.assertAlmostEqual(value["edge_gain_fraction"], 1.0, places=10)
            self.assertAlmostEqual(value["edge_amplitude_dn"], -9.0, places=8)
        self.assertEqual(bright["edge_width_px"], dark["edge_width_px"])
        self.assertEqual(bright["edge_offset_px"], dark["edge_offset_px"])

    def test_flat_and_exact_quadratic_are_uninformative(self):
        for patch in (np.zeros((25, 25)), np.full((25, 25), 128.0),
                      X, Y, X * X, X * Y, Y * Y, background()):
            with self.subTest(energy=float(np.sum(patch * patch))):
                for polarity in ("bright", "dark"):
                    result = self.diagnostic.measure(patch, polarity)
                    self.assert_diagnostic(result)
                    self.assertFalse(result["informative"])
                    self.assertEqual(result["point_gain_fraction"], 0.0)
                    self.assertEqual(result["edge_gain_fraction"], 0.0)
                    self.assertEqual(result["point_minus_edge_fraction"], 0.0)
                    self.assertFalse(result["conditional_informative"])
                    self.assertEqual(result["point_gain_after_edge_fraction"], 0.0)

    def test_point_on_strong_edge_documents_exclusive_margin_limitation(self):
        patch = background() + 6.0 * point(2) + 20.0 * edge(2)
        result = self.diagnostic.measure(patch, "bright")
        self.assert_diagnostic(result)
        # The true point remains present even though the exclusive comparison
        # favors the much stronger edge. Negative margin is not an absence label.
        self.assertLess(result["point_minus_edge_fraction"], 0.0)
        self.assertTrue(result["conditional_informative"])
        self.assertAlmostEqual(result["point_gain_after_edge_fraction"], 1.0, places=10)
        self.assertAlmostEqual(result["point_after_edge_amplitude_dn"], 6.0, places=8)
        self.assertEqual(result["point_after_edge_sigma_px"], 2)
        self.assertEqual(tuple(result["point_after_edge_offset_xy"]), (0, 0))

    def test_orthogonalized_fits_equal_independent_seven_column_least_squares(self):
        rng = np.random.default_rng(185)
        for polarity, sign in (("bright", 1), ("dark", -1)):
            with self.subTest(polarity=polarity):
                patch = (background() + sign * 12.0 * point(2, 1, -1)
                         + 5.0 * edge(4, math.pi / 4, 2)
                         + rng.normal(0, 0.2, (25, 25)))
                result = self.diagnostic.measure(patch, polarity)
                vector = patch.ravel()
                quad = DESIGN @ np.linalg.lstsq(DESIGN, vector, rcond=None)[0]
                energy = float(np.sum((vector - quad) ** 2))
                dx, dy = result["point_offset_xy"]
                column = sign * point(result["point_sigma_px"], dx, dy).ravel()
                matrix = np.column_stack((DESIGN, column))
                coefficients = np.linalg.lstsq(matrix, vector, rcond=None)[0]
                self.assertGreater(coefficients[-1], 0.0)
                sse = float(np.sum((vector - matrix @ coefficients) ** 2))
                self.assertAlmostEqual(result["point_gain_fraction"], (energy - sse) / energy, places=10)
                self.assertAlmostEqual(result["point_amplitude_dn"], coefficients[-1], places=8)
                edge_column = edge(result["edge_width_px"], result["edge_orientation_rad"],
                                   result["edge_offset_px"]).ravel()
                matrix = np.column_stack((DESIGN, edge_column))
                coefficients = np.linalg.lstsq(matrix, vector, rcond=None)[0]
                edge_sse = float(np.sum((vector - matrix @ coefficients) ** 2))
                self.assertAlmostEqual(result["edge_gain_fraction"], (energy - edge_sse) / energy, places=10)
                self.assertAlmostEqual(result["edge_amplitude_dn"], coefficients[-1], places=8)
                self.assertAlmostEqual(result["edge_residual_energy"], edge_sse, places=7)
                self.assertAlmostEqual(result["point_absolute_gain"], energy - sse, places=7)
                self.assertAlmostEqual(result["edge_absolute_gain"], energy - edge_sse, places=7)

    def test_conditional_point_equals_independent_eight_column_joint_fit(self):
        rng = np.random.default_rng(8642)
        for polarity, sign in (("bright", 1), ("dark", -1)):
            with self.subTest(polarity=polarity):
                patch = (background() + sign * 9.0 * point(2, -1, 1)
                         - 16.0 * edge(2, math.pi / 4, 0)
                         + rng.normal(0, 0.3, (25, 25)))
                result = self.diagnostic.measure(patch, polarity)
                vector = patch.ravel()
                edge_column = edge(result["edge_width_px"], result["edge_orientation_rad"],
                                   result["edge_offset_px"]).ravel()
                baseline = np.column_stack((DESIGN, edge_column))
                baseline_fit = baseline @ np.linalg.lstsq(baseline, vector, rcond=None)[0]
                baseline_sse = float(np.sum((vector - baseline_fit) ** 2))
                dx, dy = result["point_after_edge_offset_xy"]
                point_column = sign * point(result["point_after_edge_sigma_px"], dx, dy).ravel()
                joint = np.column_stack((baseline, point_column))
                coefficients = np.linalg.lstsq(joint, vector, rcond=None)[0]
                self.assertGreater(coefficients[-1], 0.0)
                joint_sse = float(np.sum((vector - joint @ coefficients) ** 2))
                self.assertAlmostEqual(result["edge_residual_energy"], baseline_sse, places=7)
                self.assertAlmostEqual(result["point_gain_after_edge_fraction"],
                                       (baseline_sse - joint_sse) / baseline_sse, places=10)
                self.assertAlmostEqual(result["point_after_edge_absolute_gain"],
                                       baseline_sse - joint_sse, places=7)
                self.assertAlmostEqual(result["point_after_edge_amplitude_dn"], coefficients[-1], places=8)

    def test_quadratic_background_does_not_change_shape_evidence(self):
        signal = 14.0 * point(2, -1, 0) + 4.0 * edge(4, 3 * math.pi / 8, -2)
        plain = self.diagnostic.measure(signal, "bright")
        nuisance = self.diagnostic.measure(signal + background(), "bright")
        self.assert_same_gains(plain, nuisance)
        self.assertAlmostEqual(plain["background_residual_energy"], nuisance["background_residual_energy"], places=7)
        self.assertAlmostEqual(plain["point_amplitude_dn"], nuisance["point_amplitude_dn"], places=8)
        self.assertAlmostEqual(plain["edge_amplitude_dn"], nuisance["edge_amplitude_dn"], places=8)

    def test_residual_energy_matches_independent_quadratic_projection(self):
        rng = np.random.default_rng(2468)
        patch = background() + 14.0 * point(2) + rng.normal(0, 0.7, (25, 25))
        vector = patch.ravel()
        fitted = DESIGN @ np.linalg.lstsq(DESIGN, vector, rcond=None)[0]
        expected = float(np.sum((vector - fitted) ** 2))
        result = self.diagnostic.measure(patch, "bright")
        self.assertAlmostEqual(result["background_residual_energy"], expected, places=7)

    def test_positive_scale_and_intensity_offset_preserve_fractional_gains(self):
        patch = 5.0 * point(1, 1, 0) + 2.0 * edge(2, math.pi / 2, -2)
        original = self.diagnostic.measure(patch, "bright")
        for factor, offset in ((0.1, 0), (2.0, 73.0), (7.5, -17.0)):
            with self.subTest(factor=factor, offset=offset):
                changed = self.diagnostic.measure(factor * patch + offset, "bright")
                self.assert_same_gains(original, changed)
                self.assertAlmostEqual(changed["background_residual_energy"],
                                       factor ** 2 * original["background_residual_energy"], places=7)
                self.assertAlmostEqual(changed["point_amplitude_dn"],
                                       factor * original["point_amplitude_dn"], places=8)

    def test_ninety_degree_rotations_and_reflections_preserve_evidence(self):
        patch = background() + 11.0 * point(2, 1, -1) + 3.0 * edge(2, math.pi / 8, 2)
        original = self.diagnostic.measure(patch, "bright")
        variants = (np.rot90(patch), np.rot90(patch, 2), np.rot90(patch, 3),
                    np.fliplr(patch), np.flipud(patch), patch.T)
        for index, transformed in enumerate(variants):
            with self.subTest(transform=index):
                output = self.diagnostic.measure(transformed, "bright")
                self.assert_same_gains(original, output)
                self.assertAlmostEqual(original["background_residual_energy"], output["background_residual_energy"], places=7)
                self.assertAlmostEqual(original["point_amplitude_dn"], output["point_amplitude_dn"], places=8)
                self.assertEqual(original["point_sigma_px"], output["point_sigma_px"])

    def test_selected_point_offset_follows_reflection(self):
        patch = 20.0 * point(2, 1, -1)
        original = self.diagnostic.measure(patch, "bright")
        horizontal = self.diagnostic.measure(np.fliplr(patch), "bright")
        vertical = self.diagnostic.measure(np.flipud(patch), "bright")
        self.assertEqual(tuple(original["point_offset_xy"]), (1, -1))
        self.assertEqual(tuple(horizontal["point_offset_xy"]), (-1, -1))
        self.assertEqual(tuple(vertical["point_offset_xy"]), (1, 1))

    def test_boundary_low_contrast_and_extended_blobs_return_diagnostics_not_labels(self):
        patches = {
            "point_at_patch_boundary": 100.0 + 8.0 * point(2, 12, 0),
            "point_partly_outside_patch": 100.0 + 8.0 * point(2, 14, -3),
            "low_contrast": 100.0 + 0.01 * point(2, 0.25, -0.25),
            "broad_blob": background() + 9.0 * point(6, 0, 0),
            "elongated_blob": 100.0 + 9.0 * np.exp(-(X * X / 72.0 + Y * Y / 8.0)),
        }
        for name, patch in patches.items():
            with self.subTest(name=name):
                output = self.diagnostic.measure(patch, "bright")
                self.assert_diagnostic(output)
                self.assertFalse({"is_target", "accepted", "is_airborne", "rejected"} & set(output))

    def test_repeated_calls_are_deterministic_without_mutating_input_or_old_results(self):
        patch = background() + 8.0 * point(2, 1, 0)
        before = patch.copy()
        first = self.diagnostic.measure(patch, "bright")
        snapshot = copy.deepcopy(first)
        second = self.diagnostic.measure(patch, "bright")
        self.diagnostic.measure(background() + edge(2), "dark")
        np.testing.assert_array_equal(patch, before)
        self.assertEqual(first, second)
        self.assertEqual(first, snapshot)

    def test_read_only_and_non_contiguous_arrays_supported(self):
        patch = 30.0 + 10.0 * point(2)
        patch.flags.writeable = False
        expected = self.diagnostic.measure(patch, "bright")
        reflected_view = self.diagnostic.measure(patch[:, ::-1], "bright")
        self.assert_same_gains(expected, reflected_view)

    def test_finite_integer_and_float_inputs_have_equivalent_values(self):
        integer = np.rint(110.0 + 17.0 * point(2)).astype(np.uint8)
        a = self.diagnostic.measure(integer, "bright")
        b = self.diagnostic.measure(integer.astype(np.float64), "bright")
        self.assertEqual(a, b)

    def test_wrong_shapes_are_rejected(self):
        for shape in ((24, 25), (25, 24), (26, 26), (625,), (25, 25, 1), (0, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.diagnostic.measure(np.zeros(shape), "bright")

    def test_nonfinite_values_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            patch = np.zeros((25, 25))
            patch[2, 3] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.diagnostic.measure(patch, "bright")

    def test_invalid_polarity_and_nonnumeric_inputs_are_rejected(self):
        for polarity in (None, "", "BRIGHT", "positive", 1):
            with self.subTest(polarity=polarity), self.assertRaises(ValueError):
                self.diagnostic.measure(np.zeros((25, 25)), polarity)
        with self.assertRaises(ValueError):
            self.diagnostic.measure(np.full((25, 25), "not numeric"), "bright")


if __name__ == "__main__":
    unittest.main()
