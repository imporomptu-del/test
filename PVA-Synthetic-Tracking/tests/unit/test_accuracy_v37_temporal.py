"""Synthetic temporal source-pixel diagnostics, never recording-tuned gates.

No media, labels, locations, or clip-specific parameters are used. Fits are
checked against independently assembled least-squares designs. A positive
nested-model gain is deliberately not interpreted as an object classification.
"""

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/accuracy_v37_temporal.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v37_temporal_under_test", MODULE_PATH)
temporal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = temporal
SPEC.loader.exec_module(temporal)
REGISTRATION_PATH = MODULE_PATH.with_name("accuracy_v37_registration.py")
REGISTRATION_SPEC = importlib.util.spec_from_file_location("accuracy_v37_registration_for_temporal_tests", REGISTRATION_PATH)
registration = importlib.util.module_from_spec(REGISTRATION_SPEC)
sys.modules[REGISTRATION_SPEC.name] = registration
REGISTRATION_SPEC.loader.exec_module(registration)

Y, X = np.mgrid[-12:13, -12:13].astype(np.float64)
DESIGN = np.column_stack([v.ravel() for v in (np.ones_like(X), X, Y, X * X, X * Y, Y * Y)])
REQUIRED = {
    "available", "status", "reasons", "displacement_px",
    "background_residual_energy", "null_sse", "point_sse", "edge_sse",
    "point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
    "point_current_amplitude_dn", "point_previous_amplitude_dn", "point_sigma_px",
    "point_offset_xy", "point_previous_sigma_px", "edge_amplitude_dn",
    "edge_previous_amplitude_dn", "edge_width_px", "edge_orientation_rad",
    "edge_offset_px", "edge_previous_sigma_px", "point_pair_condition",
}


def point(sigma=2.0, xy=(0.0, 0.0)):
    return np.exp(-((X - xy[0]) ** 2 + (Y - xy[1]) ** 2) / (2.0 * sigma ** 2))


def edge(width=2.0, angle=0.0, offset=0.0):
    return np.tanh((X * np.cos(angle) + Y * np.sin(angle) - offset) / width)


def background():
    return 73.0 + 0.8 * X - 0.3 * Y + 0.04 * X * X - 0.07 * X * Y + 0.11 * Y * Y


def projected(vector):
    vector = np.asarray(vector, dtype=np.float64).ravel()
    return vector - DESIGN @ np.linalg.lstsq(DESIGN, vector, rcond=None)[0]


def fit_sse(matrix, vector):
    coefficients = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    residual = vector - matrix @ coefficients
    return float(residual @ residual), coefficients


def independent_null_sse(vector, previous_xy, sign):
    """Exhaust the three one-sided prior fits, retaining the quadratic-only fit."""
    energy = float(projected(vector) @ projected(vector))
    candidates = [energy]
    for sigma in (1, 2, 3):
        matrix = np.column_stack((DESIGN, -sign * point(sigma, previous_xy).ravel()))
        sse, coefficients = fit_sse(matrix, vector)
        if coefficients[-1] >= 0:
            candidates.append(sse)
    return min(candidates)


def constrained_pair_sse(vector, previous_column, current_column, current_free=False):
    """Independent active-set enumeration, with six unrestricted backgrounds."""
    candidates = [fit_sse(DESIGN, vector)[0]]
    for column, unrestricted in ((previous_column, False), (current_column, current_free)):
        sse, coefficients = fit_sse(np.column_stack((DESIGN, column)), vector)
        if unrestricted or coefficients[-1] >= 0:
            candidates.append(sse)
    matrix = np.column_stack((DESIGN, previous_column, current_column))
    sse, coefficients = fit_sse(matrix, vector)
    if coefficients[-2] >= 0 and (current_free or coefficients[-1] >= 0):
        candidates.append(sse)
    return min(candidates)


class AccuracyV37TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = temporal.TemporalPointDiagnostic()

    def assert_diagnostic(self, value):
        self.assertTrue({"available", "status", "reasons", "displacement_px"} <= set(value))
        self.assertIsInstance(value["available"], bool)
        self.assertIsInstance(value["status"], str)
        self.assertTrue(value["status"])
        self.assertIsInstance(value["reasons"], list)
        self.assertTrue(all(isinstance(reason, str) and reason for reason in value["reasons"]))
        # No NaN, Infinity, numpy arrays, or numpy booleans in saved artifacts.
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.assertFalse({"accepted", "rejected", "is_object", "is_noise", "classification"} & set(value))
        self.assertGreaterEqual(value["displacement_px"], 0)
        if not value["available"]:
            self.assertTrue(value["reasons"])
            return
        self.assertTrue(REQUIRED <= set(value))
        for key in ("background_residual_energy", "null_sse", "point_sse", "edge_sse",
                    "point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
                    "point_current_amplitude_dn", "point_previous_amplitude_dn",
                    "edge_amplitude_dn", "edge_previous_amplitude_dn", "point_pair_condition"):
            self.assertTrue(math.isfinite(value[key]), key)
        self.assertGreater(value["background_residual_energy"], 0)
        for key in ("null_sse", "point_sse", "edge_sse", "point_current_amplitude_dn",
                    "point_previous_amplitude_dn", "edge_previous_amplitude_dn"):
            self.assertGreaterEqual(value[key], -1e-9, key)
        energy = value["background_residual_energy"]
        self.assertLessEqual(value["null_sse"], energy + 1e-8 * max(1, energy))
        for model in ("point", "edge"):
            self.assertLessEqual(value[model + "_sse"], value["null_sse"] + 1e-8 * max(1, energy))
            self.assertGreaterEqual(value[model + "_gain_fraction"], -1e-10)
            self.assertLessEqual(value[model + "_gain_fraction"], 1.0 + 1e-10)
            self.assertAlmostEqual(value[model + "_gain_fraction"],
                                   (value["null_sse"] - value[model + "_sse"]) / energy, places=9)
        self.assertAlmostEqual(value["point_minus_edge_fraction"],
                               value["point_gain_fraction"] - value["edge_gain_fraction"], places=10)
        self.assertGreaterEqual(value["point_pair_condition"], 1.0 - 1e-10)
        self.assertLessEqual(value["point_pair_condition"], 1e4)
        self.assertIn(value["point_sigma_px"], (1, 2, 3))
        self.assertIn(value["point_previous_sigma_px"], (1, 2, 3))
        self.assertIn(value["edge_previous_sigma_px"], (1, 2, 3))
        self.assertEqual(len(value["point_offset_xy"]), 2)
        self.assertTrue(all(v in (-1, 0, 1) for v in value["point_offset_xy"]))
        self.assertIn(value["edge_width_px"], (1, 2, 4))
        self.assertIn(value["edge_offset_px"], (-2, 0, 2))

    def measure_pair(self, previous_xy=(-4.25, 2.5), current_xy=(0.3, -0.2),
                     current_sigma=2, previous_sigma=3, current_amplitude=11,
                     previous_amplitude=27, offset=(0, 0), polarity="bright"):
        sign = 1 if polarity == "bright" else -1
        center = np.asarray(current_xy) + np.asarray(offset)
        current = background() + sign * current_amplitude * point(current_sigma, center)
        prior = background() + sign * previous_amplitude * point(previous_sigma, previous_xy)
        result = self.diagnostic.measure(current, prior, previous_xy=previous_xy,
                                         polarity=polarity, current_xy=current_xy)
        return result, current, prior

    def test_exact_unequal_brightness_and_widths_recovered_independently(self):
        for current_sigma, previous_sigma, a, b in ((1, 3, 11, 27), (2, 1, 29, 7), (3, 2, 8, 31)):
            with self.subTest(current_sigma=current_sigma, previous_sigma=previous_sigma):
                result, _, _ = self.measure_pair(current_sigma=current_sigma,
                                                  previous_sigma=previous_sigma,
                                                  current_amplitude=a, previous_amplitude=b)
                self.assert_diagnostic(result)
                self.assertTrue(result["available"])
                self.assertLess(result["point_sse"], 1e-8)
                self.assertAlmostEqual(result["point_current_amplitude_dn"], a, places=7)
                self.assertAlmostEqual(result["point_previous_amplitude_dn"], b, places=7)
                self.assertEqual(result["point_sigma_px"], current_sigma)
                self.assertEqual(result["point_previous_sigma_px"], previous_sigma)
                self.assertEqual(result["point_offset_xy"], [0, 0])

    def test_supported_offsets_around_fractional_current_center(self):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                with self.subTest(offset=(dx, dy)):
                    result, _, _ = self.measure_pair(offset=(dx, dy))
                    self.assertTrue(result["available"])
                    self.assertLess(result["point_sse"], 1e-8)
                    self.assertEqual(result["point_offset_xy"], [dx, dy])
                    self.assertAlmostEqual(result["point_current_amplitude_dn"], 11, places=7)
                    self.assertAlmostEqual(result["point_previous_amplitude_dn"], 27, places=7)

    def test_previous_fractional_coordinate_is_not_rounded_or_inferred(self):
        previous_xy = (-3.47, 2.38)
        result, current, prior = self.measure_pair(previous_xy=previous_xy)
        self.assertLess(result["point_sse"], 1e-8)
        rounded = self.diagnostic.measure(current, prior, previous_xy=(-3, 2),
                                          current_xy=(0.3, -0.2), polarity="bright")
        self.assertGreater(rounded["point_sse"], result["point_sse"] + 1e-3)
        self.assertAlmostEqual(result["displacement_px"], math.dist(previous_xy, (0.3, -0.2)), places=10)

    def test_selected_point_fit_matches_independent_eight_column_least_squares(self):
        rng = np.random.default_rng(731)
        previous_xy, current_xy = (-4.25, 2.5), (0.3, -0.2)
        for polarity, sign in (("bright", 1), ("dark", -1)):
            with self.subTest(polarity=polarity):
                _, current, prior = self.measure_pair(polarity=polarity, offset=(1, -1))
                current = current + rng.normal(0, 0.08, (25, 25))
                result = self.diagnostic.measure(current, prior, previous_xy=previous_xy,
                                                 current_xy=current_xy, polarity=polarity)
                self.assert_diagnostic(result)
                vector = (current - prior).ravel()
                residual = projected(vector)
                self.assertAlmostEqual(result["background_residual_energy"], float(residual @ residual), places=7)
                self.assertAlmostEqual(result["null_sse"], independent_null_sse(vector, previous_xy, sign), places=7)
                previous_column = -sign * point(result["point_previous_sigma_px"], previous_xy).ravel()
                center = np.asarray(current_xy) + np.asarray(result["point_offset_xy"])
                current_column = sign * point(result["point_sigma_px"], center).ravel()
                matrix = np.column_stack((DESIGN, previous_column, current_column))
                sse, coefficients = fit_sse(matrix, vector)
                self.assertGreater(coefficients[-2], 0)
                self.assertGreater(coefficients[-1], 0)
                self.assertAlmostEqual(result["point_sse"], sse, places=7)
                self.assertAlmostEqual(result["point_previous_amplitude_dn"], coefficients[-2], places=7)
                self.assertAlmostEqual(result["point_current_amplitude_dn"], coefficients[-1], places=7)
                self.assertAlmostEqual(result["point_sse"],
                                       constrained_pair_sse(vector, previous_column, current_column), places=7)

    def test_prior_plus_free_signed_edge_matches_independent_joint_fit(self):
        previous_xy = (-5.2, 3.4)
        for width, angle, offset, amplitude in ((1, 0, -2, 12), (2, math.pi / 4, 0, -9),
                                                (4, 7 * math.pi / 8, 2, 17)):
            with self.subTest(width=width, amplitude=amplitude):
                current = background() + amplitude * edge(width, angle, offset)
                prior = background() + 21 * point(2, previous_xy)
                result = self.diagnostic.measure(current, prior, previous_xy=previous_xy, polarity="bright")
                self.assert_diagnostic(result)
                self.assertTrue(result["available"])
                self.assertLess(result["edge_sse"], 1e-8)
                self.assertAlmostEqual(result["edge_amplitude_dn"], amplitude, places=7)
                self.assertAlmostEqual(result["edge_previous_amplitude_dn"], 21, places=7)
                self.assertEqual(result["edge_width_px"], width)
                self.assertEqual(result["edge_previous_sigma_px"], 2)
                self.assertAlmostEqual(result["edge_orientation_rad"], angle, places=9)
                self.assertEqual(result["edge_offset_px"], offset)
                vector = (current - prior).ravel()
                prior_column = -point(result["edge_previous_sigma_px"], previous_xy).ravel()
                edge_column = edge(result["edge_width_px"], result["edge_orientation_rad"],
                                   result["edge_offset_px"]).ravel()
                sse, coefficients = fit_sse(np.column_stack((DESIGN, prior_column, edge_column)), vector)
                self.assertAlmostEqual(result["edge_sse"], sse, places=7)
                self.assertAlmostEqual(result["edge_previous_amplitude_dn"], coefficients[-2], places=7)
                self.assertAlmostEqual(result["edge_amplitude_dn"], coefficients[-1], places=7)

    def test_constrained_fits_match_active_sets_on_noise_not_signed_unrestricted_points(self):
        rng = np.random.default_rng(107)
        previous_xy = (-4, 3)
        for sign, polarity in ((1, "bright"), (-1, "dark")):
            current = background() - sign * 9 * point(2) + rng.normal(0, 0.5, (25, 25))
            prior = background() - sign * 6 * point(1, previous_xy)
            result = self.diagnostic.measure(current, prior, previous_xy=previous_xy, polarity=polarity)
            self.assert_diagnostic(result)
            vector = (current - prior).ravel()
            current_column = sign * point(result["point_sigma_px"], result["point_offset_xy"]).ravel()
            previous_column = -sign * point(result["point_previous_sigma_px"], previous_xy).ravel()
            expected = constrained_pair_sse(vector, previous_column, current_column)
            self.assertAlmostEqual(result["point_sse"], expected, places=7)
            edge_column = edge(result["edge_width_px"], result["edge_orientation_rad"],
                               result["edge_offset_px"]).ravel()
            previous_column = -sign * point(result["edge_previous_sigma_px"], previous_xy).ravel()
            expected = constrained_pair_sse(vector, previous_column, edge_column, current_free=True)
            self.assertAlmostEqual(result["edge_sse"], expected, places=7)

    def test_dark_sign_symmetry(self):
        bright, current, prior = self.measure_pair(offset=(1, -1))
        dark = self.diagnostic.measure(255 - current, 255 - prior, previous_xy=(-4.25, 2.5),
                                       current_xy=(0.3, -0.2), polarity="dark")
        self.assert_diagnostic(dark)
        for key in ("background_residual_energy", "null_sse", "point_sse", "edge_sse",
                    "point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
                    "point_current_amplitude_dn", "point_previous_amplitude_dn"):
            self.assertAlmostEqual(bright[key], dark[key], places=7, msg=key)
        self.assertEqual(bright["point_sigma_px"], dark["point_sigma_px"])
        self.assertEqual(bright["point_offset_xy"], dark["point_offset_xy"])
        self.assertAlmostEqual(bright["edge_amplitude_dn"], -dark["edge_amplitude_dn"], places=7)

    def test_different_quadratic_backgrounds_and_offsets_add_no_shape_evidence(self):
        original, current, prior = self.measure_pair()
        altered = self.diagnostic.measure(current + 2 * background() + 14,
                                          prior - 0.7 * background() - 37,
                                          previous_xy=(-4.25, 2.5), current_xy=(0.3, -0.2),
                                          polarity="bright")
        self.assert_diagnostic(altered)
        for key in ("background_residual_energy", "null_sse", "point_sse", "edge_sse",
                    "point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction",
                    "point_current_amplitude_dn", "point_previous_amplitude_dn"):
            self.assertAlmostEqual(original[key], altered[key], places=7, msg=key)

    def test_positive_intensity_scaling_preserves_gains_scales_energy_and_amplitudes(self):
        original, current, prior = self.measure_pair()
        scale = 2.75
        altered = self.diagnostic.measure(scale * current + 23, scale * prior - 45,
                                          previous_xy=(-4.25, 2.5), current_xy=(0.3, -0.2),
                                          polarity="bright")
        for key in ("point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction"):
            self.assertAlmostEqual(original[key], altered[key], places=9, msg=key)
        for key in ("background_residual_energy", "null_sse", "point_sse", "edge_sse"):
            self.assertAlmostEqual(original[key] * scale ** 2, altered[key], places=6, msg=key)
        for key in ("point_current_amplitude_dn", "point_previous_amplitude_dn"):
            self.assertAlmostEqual(original[key] * scale, altered[key], places=7, msg=key)

    def test_static_registered_edge_cancels_without_removing_moving_point(self):
        original, current, prior = self.measure_pair()
        clutter = 350 * edge(2, math.pi / 8, 2) + 75 * edge(4, 5 * math.pi / 8, -2)
        altered = self.diagnostic.measure(current + clutter, prior + clutter,
                                          previous_xy=(-4.25, 2.5), current_xy=(0.3, -0.2),
                                          polarity="bright")
        self.assert_diagnostic(altered)
        self.assertTrue(altered["available"])
        self.assertLess(altered["point_sse"], 1e-8)
        self.assertAlmostEqual(altered["point_current_amplitude_dn"], 11, places=7)
        for key in ("point_gain_fraction", "edge_gain_fraction", "point_minus_edge_fraction"):
            self.assertAlmostEqual(original[key], altered[key], places=9, msg=key)

    def test_registration_and_temporal_fit_recover_faint_point_on_static_textured_edge(self):
        cy, cx = np.mgrid[-24:25, -24:25].astype(np.float64)
        py, px = np.mgrid[-26:27, -26:27].astype(np.float64)

        def static_background(x, y):
            return (110 + 37 * np.tanh((x + 0.3 * y) / 2)
                    + 13 * np.sin(0.59 * x + 0.13 * y)
                    + 11 * np.cos(0.17 * x - 0.67 * y))

        def gaussian(x, y, center, sigma):
            return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))

        shift = np.array([1.0, -1.0])
        previous_registered_xy = np.array([-4.0, 3.0])
        previous_source_xy = previous_registered_xy + shift
        current_xy = np.array([0.25, -0.15])
        current = static_background(cx, cy) + 4 * gaussian(cx, cy, current_xy, 2)
        # prior(q + shift) agrees with the current grid after the annulus-only
        # gain/offset correction. Integer shift makes interpolation exact here.
        prior = ((static_background(px - shift[0], py - shift[1]) - 7) / 1.2
                 + (9 / 1.2) * gaussian(px, py, previous_source_xy, 1))
        aligned = registration.AnnulusRegistration().measure(
            current, prior, prior_point_xy=previous_source_xy.tolist())
        self.assertTrue(aligned["available"], aligned["reasons"])
        np.testing.assert_allclose(aligned["shift_xy"], shift, atol=0, rtol=0)
        self.assertAlmostEqual(aligned["gain"], 1.2, places=7)
        self.assertAlmostEqual(aligned["offset"], 7, places=7)
        corrected = aligned["gain"] * aligned["registered_prior_patch"] + aligned["offset"]
        mapped_previous = previous_source_xy - np.asarray(aligned["shift_xy"])
        np.testing.assert_allclose(mapped_previous, previous_registered_xy, atol=0, rtol=0)
        result = self.diagnostic.measure(aligned["current_patch"], corrected,
                                         previous_xy=mapped_previous, current_xy=current_xy,
                                         polarity="bright")
        self.assert_diagnostic(result)
        self.assertTrue(result["available"])
        self.assertLess(result["point_sse"], 1e-7)
        self.assertAlmostEqual(result["point_current_amplitude_dn"], 4, places=6)
        self.assertAlmostEqual(result["point_previous_amplitude_dn"], 9, places=6)

    def test_straight_edge_registration_abstains_despite_conditional_temporal_success(self):
        cy, cx = np.mgrid[-24:25, -24:25].astype(np.float64)
        py, px = np.mgrid[-26:27, -26:27].astype(np.float64)
        previous_xy = (-4, 3)
        current = 110 + 37 * np.tanh(cx / 2) + 4 * np.exp(-(cx * cx + cy * cy) / 2)
        prior = (110 + 37 * np.tanh(px / 2)
                 + 9 * np.exp(-((px - previous_xy[0]) ** 2 + (py - previous_xy[1]) ** 2) / 2))
        aligned = registration.AnnulusRegistration().measure(current, prior, prior_point_xy=previous_xy)
        self.assertFalse(aligned["available"])
        self.assertIn("ambiguous_nonlocal_registration", aligned["reasons"])
        # Exact supplied alignment can fit the pair, but the actual annulus
        # cannot determine motion along a straight edge (aperture ambiguity).
        # The caller must therefore preserve "unknown", not use this result.
        conditional = self.diagnostic.measure(current[12:37, 12:37], prior[14:39, 14:39],
                                              previous_xy=previous_xy, polarity="bright")
        self.assert_diagnostic(conditional)
        self.assertTrue(conditional["available"])
        self.assertLess(conditional["point_sse"], 1e-8)

    def test_motion_direction_can_change_between_independent_pairs(self):
        for previous_xy in ((-4, 0), (0, -4), (4, 0), (0, 4), (-3, 4), (3, -4)):
            with self.subTest(previous_xy=previous_xy):
                result, _, _ = self.measure_pair(previous_xy=previous_xy, current_xy=(0, 0))
                self.assert_diagnostic(result)
                self.assertTrue(result["available"])
                self.assertLess(result["point_sse"], 1e-8)
                self.assertAlmostEqual(result["point_current_amplitude_dn"], 11, places=7)

    def test_zero_and_exact_quadratic_difference_are_unavailable_not_negative_objects(self):
        for current, prior in ((np.zeros((25, 25)), np.zeros((25, 25))),
                               (background(), np.zeros((25, 25))),
                               (background() + 50 * edge(), background() + 50 * edge())):
            with self.subTest(energy=float(np.sum(current ** 2))):
                result = self.diagnostic.measure(current, prior, previous_xy=(-4, 0), polarity="bright")
                self.assert_diagnostic(result)
                self.assertFalse(result["available"])

    def test_stationary_blink_and_subpixel_relative_motion_abstain(self):
        current_xy = (0.2, -0.3)
        for previous_xy in (current_xy, (-0.2, -0.3), (current_xy[0] + 0.999, current_xy[1])):
            with self.subTest(previous_xy=previous_xy):
                result, _, _ = self.measure_pair(previous_xy=previous_xy, current_xy=current_xy,
                                                  current_amplitude=31, previous_amplitude=8)
                self.assert_diagnostic(result)
                self.assertFalse(result["available"])
                self.assertLess(result["displacement_px"], 1.0)

    def test_one_pixel_relative_motion_is_not_the_subpixel_abstention(self):
        result, _, _ = self.measure_pair(previous_xy=(-1, 0), current_xy=(0, 0),
                                          current_sigma=1, previous_sigma=1)
        self.assert_diagnostic(result)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["displacement_px"], 1.0, places=10)
        self.assertLess(result["point_sse"], 1e-8)

    def test_previous_center_outside_patch_abstains_instead_of_clipping(self):
        for previous_xy in ((12.00001, 0), (-12.00001, 0), (0, 12.00001), (0, -12.00001), (20, 20)):
            with self.subTest(previous_xy=previous_xy):
                result, _, _ = self.measure_pair(previous_xy=previous_xy)
                self.assert_diagnostic(result)
                self.assertFalse(result["available"])

    def test_finite_far_outside_coordinate_keeps_unavailable_result_strict_json_safe(self):
        patch = np.zeros((25, 25), dtype=np.float64)
        result = self.diagnostic.measure(patch, patch, previous_xy=(1e200, 0), polarity="bright")
        self.assert_diagnostic(result)
        self.assertFalse(result["available"])
        self.assertEqual(result["displacement_px"], 1e200)

    def test_nearly_singular_point_pair_is_not_fit_with_exploding_coefficients(self):
        class OnePointTemplate(temporal.TemporalPointDiagnostic):
            # A deliberately restricted synthetic bank makes every point pair
            # singular/nearly singular; it exercises the numerical safeguard,
            # not an alternative production configuration or tuning choice.
            SIGMAS = (2.0,)
            OFFSETS = (0.0,)

        diagnostic = OnePointTemplate()
        for previous_xy in ((0, 0), (1e-6, 0)):
            with self.subTest(previous_xy=previous_xy):
                current = background() + 17 * point(2)
                prior = background() + 9 * point(2, previous_xy)
                result = diagnostic.measure(current, prior, previous_xy=previous_xy, polarity="bright")
                self.assert_diagnostic(result)
                self.assertFalse(result["available"])
                self.assertIn("ill_conditioned_model_bank", result["reasons"])

    def test_previous_center_on_patch_boundary_is_representable(self):
        for previous_xy in ((12, 0), (-12, 0), (0, 12), (0, -12)):
            with self.subTest(previous_xy=previous_xy):
                result, _, _ = self.measure_pair(previous_xy=previous_xy)
                self.assert_diagnostic(result)
                self.assertTrue(result["available"])
                self.assertLess(result["point_sse"], 1e-8)

    def test_current_only_and_prior_only_are_fits_not_presence_classifications(self):
        for a, b in ((13, 0), (0, 19)):
            with self.subTest(current_amplitude=a, previous_amplitude=b):
                result, _, _ = self.measure_pair(current_amplitude=a, previous_amplitude=b)
                self.assert_diagnostic(result)
                self.assertTrue(result["available"])
                self.assertLess(result["point_sse"], 1e-8)
                self.assertAlmostEqual(result["point_current_amplitude_dn"], a, places=7)
                self.assertAlmostEqual(result["point_previous_amplitude_dn"], b, places=7)
                if a == 0:
                    self.assertLess(result["null_sse"], 1e-8)
                    self.assertAlmostEqual(result["point_gain_fraction"], 0, places=9)

    def test_faint_point_and_noise_report_diagnostics_without_a_classification_gate(self):
        result, _, _ = self.measure_pair(current_amplitude=0.01, previous_amplitude=0.02)
        self.assert_diagnostic(result)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["point_current_amplitude_dn"], 0.01, places=7)
        rng = np.random.default_rng(319)
        for _ in range(3):
            result = self.diagnostic.measure(rng.normal(50, 0.4, (25, 25)),
                                             rng.normal(50, 0.4, (25, 25)),
                                             previous_xy=(-3, 2), polarity="bright")
            self.assert_diagnostic(result)
            # Noise can earn a positive nested fit gain. No desired sign or
            # "rejected" outcome is asserted from such a diagnostic statistic.

    def test_default_current_center_equals_explicit_origin(self):
        previous_xy = (-3.5, 2.25)
        current = background() + 10 * point(2)
        prior = background() + 15 * point(1, previous_xy)
        default = self.diagnostic.measure(current, prior, previous_xy=previous_xy, polarity="bright")
        explicit = self.diagnostic.measure(current, prior, previous_xy=previous_xy,
                                           current_xy=(0, 0), polarity="bright")
        self.assertEqual(default, explicit)

    def test_invalid_patch_shape_or_nonfinite_values_raise_value_error(self):
        valid = np.zeros((25, 25), dtype=np.float64)
        for invalid in (np.zeros((24, 25)), np.zeros((25, 24)), np.zeros((625,)),
                        np.zeros((25, 25, 1)), np.full((25, 25), np.nan),
                        np.full((25, 25), np.inf), np.full((25, 25), -np.inf),
                        np.full((25, 25), "not-numeric"), np.full((25, 25), 1 + 2j)):
            for swap in (False, True):
                with self.subTest(shape=invalid.shape, swap=swap):
                    with self.assertRaises(ValueError):
                        self.diagnostic.measure(invalid if swap else valid, valid if swap else invalid,
                                                previous_xy=(-3, 2), polarity="bright")

    def test_invalid_polarity_and_coordinates_raise_value_error(self):
        valid = np.zeros((25, 25), dtype=np.float64)
        for polarity in ("", "Bright", "unknown", 1, None):
            with self.subTest(polarity=polarity):
                with self.assertRaises(ValueError):
                    self.diagnostic.measure(valid, valid, previous_xy=(-3, 2), polarity=polarity)
        for coordinate in ((), (1,), (1, 2, 3), (np.nan, 0), (0, np.inf), (-np.inf, 0),
                           ("bad", 0), [[1, 2]], (1 + 2j, 0), None):
            for field in ("previous_xy", "current_xy"):
                with self.subTest(coordinate=coordinate, field=field):
                    arguments = {"previous_xy": (-3, 2), "current_xy": (0, 0), "polarity": "bright"}
                    arguments[field] = coordinate
                    with self.assertRaises(ValueError):
                        self.diagnostic.measure(valid, valid, **arguments)
        for current_xy in ((0.50001, 0), (-0.50001, 0), (0, 0.50001), (0, -0.50001)):
            with self.subTest(current_xy=current_xy):
                with self.assertRaises(ValueError):
                    self.diagnostic.measure(valid, valid, previous_xy=(-3, 2), current_xy=current_xy,
                                            polarity="bright")

    def test_inputs_immutable_results_deterministic_and_no_cross_pair_history(self):
        _, current, prior = self.measure_pair()
        current_before, prior_before = current.copy(), prior.copy()
        previous_xy, current_xy = [-4.25, 2.5], [0.3, -0.2]
        coords_before = copy.deepcopy([previous_xy, current_xy])
        current.setflags(write=False)
        prior.setflags(write=False)
        first = self.diagnostic.measure(current, prior, previous_xy=previous_xy,
                                         current_xy=current_xy, polarity="bright")
        self.measure_pair(previous_xy=(6, -4), current_amplitude=37, previous_amplitude=3,
                          polarity="dark")
        second = self.diagnostic.measure(current, prior, previous_xy=previous_xy,
                                          current_xy=current_xy, polarity="bright")
        fresh = temporal.TemporalPointDiagnostic().measure(current, prior, previous_xy=previous_xy,
                                                            current_xy=current_xy, polarity="bright")
        np.testing.assert_array_equal(current, current_before)
        np.testing.assert_array_equal(prior, prior_before)
        self.assertEqual([previous_xy, current_xy], coords_before)
        self.assertEqual(first, second)
        self.assertEqual(first, fresh)
        self.assert_diagnostic(first)


if __name__ == "__main__":
    unittest.main()
