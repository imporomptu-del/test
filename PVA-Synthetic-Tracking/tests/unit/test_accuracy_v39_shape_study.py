"""Synthetic-only shape study checks; no file/media access or operational labels."""

import copy
import inspect
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import accuracy_v39_shape_study as study


Y, X = np.mgrid[-12:13, -12:13].astype(float)
BG = 110+.1*X-.15*Y+.004*X*Y
DESIGN = np.column_stack([a.ravel() for a in (np.ones_like(X), X, Y, X*X, X*Y, Y*Y)])


def independent_gaussian(xy, a, b, theta):
    dx, dy = X-xy[0], Y-xy[1]
    u, v = math.cos(theta)*dx+math.sin(theta)*dy, -math.sin(theta)*dx+math.cos(theta)*dy
    return np.exp(-.5*((u/a)**2+(v/b)**2))


def independent_compact(name, parameters):
    xy = np.asarray(parameters["center_xy"])
    if name.endswith("isotropic"):
        return independent_gaussian(xy, parameters["sigma_px"], parameters["sigma_px"], 0.)
    angle = parameters["orientation_rad"]
    if name == "localized_elongated":
        return independent_gaussian(xy, parameters["sigma_u_px"], parameters["sigma_v_px"], angle)
    delta = .5*parameters["separation_px"]*np.array([math.cos(angle), math.sin(angle)])
    sigma = parameters["sigma_px"]
    return independent_gaussian(xy-delta, sigma, sigma, 0)+independent_gaussian(xy+delta, sigma, sigma, 0)


def independent_fit(patch, point, e, sign):
    design = np.column_stack((DESIGN, e.ravel(), sign*point.ravel()))
    values = patch.ravel()
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    if coefficients[-1] < 0:
        null = design[:, :-1]
        coefficients = np.r_[np.linalg.lstsq(null, values, rcond=None)[0], 0.]
    residual = values-design@coefficients
    return float(residual@residual), coefficients


class SourceShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = study.SourceShapeDiagnostic()
        cls.cases = {case["name"]: case for case in study.synthetic_cases()}
        cls.results = {name: cls.model.measure(case["patch"], case["polarity"])
                       for name, case in cls.cases.items()}

    def test_declared_bank_counts_and_families(self):
        result = self.results["equal_pair_on_edge"]
        self.assertEqual(tuple(result["families"]), study.FAMILIES)
        for name, count in zip(study.FAMILIES, (27, 75, 200, 400)):
            family = result["families"][name]
            self.assertEqual(family["compact_template_count"], count)
            self.assertEqual(family["joint_template_pair_count"], count*72)
            self.assertEqual(family["nominal_linear_coefficient_count"], 8)
            self.assertFalse(family["complexity_calibrated"])
            self.assertTrue(family["search_complexity_excluded_from_linear_coefficient_count"])

    def test_api_has_no_truth_lag_clip_time_or_history_inputs(self):
        self.assertEqual(list(inspect.signature(study.SourceShapeDiagnostic).parameters), ["current_xy"])
        self.assertEqual(list(inspect.signature(self.model.measure).parameters), ["patch", "polarity"])

    def test_joint_source_localization_recovers_point_and_exposes_exclusive_failure(self):
        for offset in (0, 2, 4):
            result = self.results[f"point_offset_{offset}_on_edge"]
            self.assertLess(result["baseline"]["point_minus_edge_fraction"], 0)
            best = result["families"]["localized_isotropic"]["best"]
            self.assertAlmostEqual(best["gain_over_best_edge_fraction"], 1., places=11)
            self.assertAlmostEqual(best["compact_amplitude_dn"], 12., places=9)
            self.assertEqual(best["compact"]["center_offset_xy"], [offset, 0.])
            self.assertEqual(best["search_boundary_winner"], offset == 4)

    def test_finite_search_does_not_silently_fix_far_targets(self):
        for offset in (6, 8):
            result = self.results[f"point_offset_{offset}_on_edge"]
            best = result["families"]["localized_isotropic"]["best"]
            self.assertEqual(best["compact"]["center_offset_xy"], [4., 0.])
            self.assertTrue(best["search_boundary_winner"])
            self.assertLess(best["gain_over_best_edge_fraction"], .5)
            self.assertGreater(best["residual_energy"], 1.)

    def test_pairs_and_elongated_families_fit_in_bank_shapes_without_object_count(self):
        for case, family in (("equal_pair_on_edge", "localized_equal_pair"),
                             ("elongated_on_edge", "localized_elongated")):
            result = self.results[case]
            best = result["families"][family]["best"]
            self.assertAlmostEqual(best["gain_over_best_edge_fraction"], 1., places=11)
            self.assertAlmostEqual(best["compact_amplitude_dn"], 12., places=9)
            self.assertEqual(best["compact"]["center_offset_xy"], [2., -2.])
            self.assertIsNone(result["physical_object_count"])
            self.assertFalse(result["families"][family]["location_alternatives_are_object_count"])

    def test_close_separate_components_can_look_like_one_paired_template(self):
        result = self.results["nearby_unequal_separate_components"]
        pair = result["families"]["localized_equal_pair"]["best"]
        self.assertGreater(pair["gain_over_best_edge_fraction"], .9)
        self.assertIsNone(result["physical_object_count"])
        self.assertFalse(result["source_localization_replaces_measurement"])

    def test_off_bank_pair_retains_mismatch_and_boundary_disclosure(self):
        best = self.results["off_grid_pair"]["families"]["localized_equal_pair"]["best"]
        self.assertLess(best["gain_over_best_edge_fraction"], .9)
        self.assertGreater(best["residual_energy"], 1.)
        self.assertTrue(best["search_boundary_winner"])

    def test_curved_structure_and_noise_have_positive_compact_gain(self):
        for case in ("curved_cloud_boundary", "curved_cloud_knot", "noise_only_0", "noise_only_1", "noise_only_2"):
            result = self.results[case]
            self.assertGreater(result["families"]["localized_isotropic"]["best"]["gain_over_best_edge"], 0)
            self.assertIsNone(result["presence_classification"])
            self.assertEqual(result["physical_class"], "unknown")

    def test_multiplicative_exposure_preserves_shape_not_motion_proof(self):
        one = self.results["stationary_structure_exposure_1"]["families"]["localized_isotropic"]["best"]
        other = self.results["stationary_structure_exposure_1.15"]["families"]["localized_isotropic"]["best"]
        self.assertAlmostEqual(one["gain_over_best_edge_fraction"], other["gain_over_best_edge_fraction"], places=12)
        self.assertAlmostEqual(other["compact_amplitude_dn"], 1.15*one["compact_amplitude_dn"], places=9)
        a = self.cases["stationary_structure_exposure_1"]["patch"]
        b = self.cases["stationary_structure_exposure_1.15"]["patch"]
        # Additive quadratic correction cannot remove nonquadratic scaled structure.
        difference = (b-a).ravel()
        residual = difference-DESIGN@np.linalg.lstsq(DESIGN, difference, rcond=None)[0]
        self.assertGreater(float(residual@residual), 1.)

    def test_turn_and_intermittent_sequence_has_no_future_or_velocity_dependency(self):
        for n in range(5):
            name = f"slow_turn_intermittent_{n}"
            fresh = study.SourceShapeDiagnostic().measure(self.cases[name]["patch"], "bright")
            self.assertEqual(fresh, self.results[name])
        absent = self.results["slow_turn_intermittent_2"]
        self.assertFalse(absent["edge_residual_informative"])
        self.assertIsNone(absent["families"]["localized_isotropic"]["best"]["gain_over_best_edge_fraction"])
        self.assertIsNone(absent["presence_classification"])

    def test_dark_and_bright_symmetric_and_coefficient_sign_is_explicit(self):
        bright = self.results["point_offset_4_on_edge"]
        dark = self.results["dark_off_center_point_on_edge"]
        for name in study.FAMILIES:
            a, b = bright["families"][name]["best"], dark["families"][name]["best"]
            self.assertEqual(a["compact"], b["compact"])
            self.assertAlmostEqual(a["compact_amplitude_dn"], b["compact_amplitude_dn"], places=9)
            self.assertAlmostEqual(a["compact_signed_amplitude_dn"], -b["compact_signed_amplitude_dn"], places=9)
            self.assertAlmostEqual(a["edge"]["signed_amplitude_dn"], -b["edge"]["signed_amplitude_dn"], places=9)

    def test_all_selected_models_match_independent_full_eight_column_fit(self):
        for case_name in ("equal_pair_on_edge", "elongated_on_edge", "noise_only_0", "dark_off_center_point_on_edge"):
            case = self.cases[case_name]
            sign = 1 if case["polarity"] == "bright" else -1
            for family, data in self.results[case_name]["families"].items():
                for best in data["location_alternatives"]:
                    compact = independent_compact(family, best["compact"])
                    p = best["edge"]
                    e = np.tanh((X*math.cos(p["orientation_rad"])+Y*math.sin(p["orientation_rad"])-p["offset_px"])/p["width_px"])
                    sse, coefficients = independent_fit(case["patch"], compact, e, sign)
                    self.assertAlmostEqual(best["residual_energy"], sse, places=7)
                    self.assertAlmostEqual(best["compact_amplitude_dn"], coefficients[-1], places=8)
                    self.assertAlmostEqual(p["signed_amplitude_dn"], coefficients[-2], places=8)

    def test_centered_joint_global_winner_matches_exhaustive_independent_bank(self):
        patch = self.cases["noise_only_1"]["patch"]
        errors = []
        for sigma in (1., 2., 3.):
            for dx in (-1., 0., 1.):
                for dy in (-1., 0., 1.):
                    point = independent_gaussian((dx, dy), sigma, sigma, 0.)
                    for width in (1., 2., 4.):
                        for angle in (k*math.pi/8 for k in range(8)):
                            for offset in (-2., 0., 2.):
                                e = np.tanh((X*math.cos(angle)+Y*math.sin(angle)-offset)/width)
                                errors.append(independent_fit(patch, point, e, 1)[0])
        actual = self.results["noise_only_1"]["families"]["centered_isotropic"]["best"]["residual_energy"]
        self.assertAlmostEqual(actual, min(errors), places=8)

    def test_fractional_detector_center_is_preserved_not_truth_recentered(self):
        xy = (.3, -.2)
        model = study.SourceShapeDiagnostic(xy)
        target = (4.3, -.2)
        patch = BG+22*np.tanh((X+Y)/(2*math.sqrt(2)))+12*independent_gaussian(target, 1., 1., 0.)
        result = model.measure(patch, "bright")
        self.assertEqual(result["current_xy"], list(xy))
        best = result["families"]["localized_isotropic"]["best"]
        self.assertEqual(best["compact"]["center_xy"], list(target))
        self.assertAlmostEqual(best["compact_amplitude_dn"], 12., places=9)

    def test_family_location_alternatives_have_distinct_coordinates(self):
        for result in self.results.values():
            for family in result["families"].values():
                candidates = family["location_alternatives"]
                self.assertEqual(len(candidates), 3)
                self.assertTrue(family["location_alternatives_include_best"])
                self.assertEqual(candidates[0], family["best"])
                self.assertEqual(len({tuple(c["compact"]["center_xy"]) for c in candidates}), 3)
                energies = [c["residual_energy"] for c in candidates[1:]]
                self.assertEqual(energies, sorted(energies))

    def test_unsupported_border_and_uninformative_patch_are_explicit_unknown(self):
        patch = BG.copy()
        patch[0, 0] = np.nan
        unsupported = self.model.measure(patch, "bright")
        self.assertFalse(unsupported["available"])
        self.assertEqual(unsupported["reasons"], ["unsupported_source_patch"])
        for patch in (np.full((25, 25), 110.), BG):
            result = self.model.measure(patch, "bright")
            self.assertFalse(result["available"])
            self.assertEqual(result["reasons"], ["uninformative_source_patch"])
            self.assertEqual(result["families"], {})

    def test_invalid_inputs_fail_closed(self):
        for patch in (np.zeros((24, 25)), np.full((25, 25), np.inf),
                      np.full((25, 25), 256.), np.full((25, 25), -1.),
                      np.ones((25, 25), dtype=complex)):
            with self.assertRaises(ValueError):
                self.model.measure(patch, "bright")
        with self.assertRaises(ValueError):
            self.model.measure(BG, "airborne")
        for xy in ((0., .6), (np.nan, 0.), (1j, 0.), (0.,)):
            with self.assertRaises(ValueError):
                study.SourceShapeDiagnostic(xy)

    def test_no_mutation_and_strict_json_determinism(self):
        patch = self.cases["equal_pair_on_edge"]["patch"].copy()
        before = patch.copy()
        patch.setflags(write=False)
        first = self.model.measure(patch, "bright")
        snapshot = copy.deepcopy(first)
        second = self.model.measure(patch, "bright")
        np.testing.assert_array_equal(patch, before)
        self.assertEqual(first, snapshot)
        self.assertEqual(first, second)
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))

    def test_study_has_required_counterexamples_without_classification_claims(self):
        report = study.run_study()
        self.assertEqual(len(report["cases"]), 22)
        self.assertFalse(report["real_media_read"])
        self.assertFalse(report["inference_uses_truth_coordinates"])
        self.assertFalse(report["detection_accuracy_claimed"])
        self.assertEqual(report, json.loads(json.dumps(report, allow_nan=False)))
        for record in report["cases"]:
            result = record["diagnostic"]
            self.assertFalse(result["classifier_promoted"])
            self.assertFalse(result["family_selection_applied"])
            self.assertFalse(result["lag_selection_applied"])
            self.assertIsNone(result["presence_classification"])


if __name__ == "__main__":
    unittest.main()
