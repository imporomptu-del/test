"""Synthetic perturbation contracts, not physical model validation."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v43_stable_fit import supported_fit


def fit(a, y, b, sigma=0.5, constrained=False, ua=0.0, ub=0.0):
    return supported_fit(a, y, b, sigma, constrained,
                         train_design_bound=ua, test_design_bound=ub)


class StableFitTest(unittest.TestCase):
    def assertUnavailable(self, result, reason=None):
        self.assertFalse(result["available"])
        for key in ("coefficients", "prediction", "prediction_bound"):
            self.assertIsNone(result[key])
        if reason:
            self.assertTrue(any(reason in item for item in result["reasons"]), result["reasons"])
        json.dumps(result, allow_nan=False)

    def test_mean_exact_design_has_half_dn_prediction_bound(self):
        a = np.ones((12, 1)); b = np.ones((3, 1)); y = np.arange(12.0)
        result = fit(a, y, b)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["prediction"], 5.5)
        np.testing.assert_allclose(result["prediction_bound"], 0.5)
        np.testing.assert_allclose(result["diagnostics"]["free_fit"]["train_column_energy"], [12])
        json.dumps(result["diagnostics"], allow_nan=False)

    def test_plane_support_and_annulus_to_central_extrapolation(self):
        y, x = np.indices((9, 9)); xx, yy = (x-4).ravel()/4, (y-4).ravel()/4
        annulus = np.maximum(np.abs(xx), np.abs(yy)) >= 0.5
        a = np.column_stack((np.ones(len(xx)), xx, yy))[annulus]
        b = np.array([[1., 0., 0.], [1., 0.25, -0.25], [1., -0.25, 0.25]])
        beta = np.array([24., 3., -7.])
        result = fit(a, a@beta, b)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["coefficients"], beta, atol=1e-12)
        np.testing.assert_allclose(result["prediction"], b@beta, atol=1e-12)
        self.assertLessEqual(max(result["prediction_bound"]), 1.0)

    def test_missing_design_uncertainty_is_not_silently_exact(self):
        result = supported_fit(np.ones((4, 1)), np.ones(4), np.ones((2, 1)), 0.5)
        self.assertUnavailable(result, "design_uncertainty_not_supplied")

    def test_empty_dictionary_is_exact_zero_with_no_uncertain_columns(self):
        result = supported_fit(np.empty((4, 0)), np.ones(4), np.empty((3, 0)), 0.5)
        self.assertTrue(result["available"])
        np.testing.assert_array_equal(result["prediction"], np.zeros(3))
        np.testing.assert_array_equal(result["prediction_bound"], np.zeros(3))

    def test_zero_training_column_with_test_support_is_unavailable(self):
        self.assertUnavailable(fit(np.zeros((4, 1)), np.ones(4), np.ones((3, 1))), "unsupported")

    def test_zero_common_column_is_unavailable(self):
        self.assertUnavailable(fit(np.zeros((4, 1)), np.ones(4), np.zeros((3, 1))), "unsupported")

    def test_exact_rank_deficiency_is_unavailable(self):
        a = np.ones((8, 2)); b = np.ones((4, 2))
        self.assertUnavailable(fit(a, np.ones(8), b), "machine_rank_deficient")

    def test_tiny_nonzero_training_tail_cannot_explode_on_test(self):
        v = np.array([1., -1., 1., -1.]); m = np.array([1., 1., -1., -1.])
        a = np.column_stack((1e-12*v, m)); b = np.column_stack((1e-4*v, m))
        result = fit(a, v+m, b)
        self.assertUnavailable(result, "prediction_perturbation")
        self.assertEqual(result["diagnostics"]["free_fit"]["machine_rank"], 2)
        self.assertGreater(result["diagnostics"]["free_fit"]["maximum_prediction_bound_dn"], 1e6)

    def test_nearly_cancelling_fixed_columns_full_rank_but_unsafe(self):
        v = np.array([1., -1., 1., -1., 1., -1.])
        a = np.column_stack((np.ones(6), np.ones(6)+1e-6*v))
        b = np.column_stack((np.ones(6), np.ones(6)+1e-3*v))
        result = fit(a, v, b)
        self.assertUnavailable(result, "prediction_perturbation")
        self.assertEqual(result["diagnostics"]["free_fit"]["machine_rank"], 2)

    def test_design_error_catches_unstable_columns_even_same_train_test(self):
        v = np.array([1., -1., 1., -1., 1., -1.])
        a = np.column_stack((np.ones(6), np.ones(6)+1e-6*v))
        result = fit(a, v, a.copy(), ua=1e-4, ub=1e-4)
        self.assertUnavailable(result, "design_perturbation_can_destroy_rank")

    def test_test_design_error_is_not_ignored(self):
        a = np.ones((10, 1)); b = np.ones((2, 1))
        result = fit(a, np.full(10, 100.), b, ua=0., ub=0.01)
        self.assertUnavailable(result, "prediction_perturbation")
        self.assertGreater(result["diagnostics"]["free_fit"]["prediction_design_bound_dn"][0], 0.99)

    def test_scaled_and_permuted_designs_preserve_fit_and_bounds(self):
        x = np.linspace(-1, 1, 20)
        a = np.column_stack((np.ones(20), x, x*x))
        b = np.array([[1., 0., 0.], [1., .2, .04]])
        target = a@np.array([20., -3., 2.])
        ua = np.array([0., 1e-5, 1e-5]); ub = np.array([0., 1e-5, 1e-5])
        first = fit(a, target, b, ua=ua, ub=ub)
        order = [2, 0, 1]; scale = np.array([-1e7, 1e-6, 3.])
        second = fit(a[:, order]*scale, target, b[:, order]*scale,
                     ua=ua[order]*np.abs(scale), ub=ub[order]*np.abs(scale))
        self.assertTrue(first["available"] and second["available"])
        np.testing.assert_allclose(first["prediction"], second["prediction"], atol=1e-10)
        np.testing.assert_allclose(first["prediction_bound"], second["prediction_bound"], atol=1e-10)
        np.testing.assert_allclose(first["coefficients"][order]/scale, second["coefficients"], rtol=1e-8, atol=1e-10)

    def test_positive_moving_coefficient_and_signed_nuisance(self):
        x = np.tile(np.array([-1., 1.]), 10)
        a = np.column_stack((np.ones(20), x)); b = a[:4]
        result = fit(a, -7+13*x, b, constrained=True)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["coefficients"], [-7, 13], atol=1e-10)
        self.assertFalse(result["diagnostics"]["constraint_active_at_observed_inputs"])

    def test_negative_unconstrained_last_uses_certified_boundary(self):
        x = np.tile(np.array([-1., 1.]), 10)
        a = np.column_stack((np.ones(20), x)); b = a[:4]
        result = fit(a, 7-13*x, b, constrained=True)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["coefficients"], [7, 0], atol=1e-10)
        self.assertTrue(result["diagnostics"]["constraint_active_at_observed_inputs"])
        self.assertFalse(result["diagnostics"]["constraint_boundary_may_change_under_perturbation"])

    def test_constraint_exact_boundary_includes_both_prediction_intervals(self):
        x = np.tile(np.array([-1., 1.]), 10)
        a = np.column_stack((np.ones(20), x)); b = a[:4]
        result = fit(a, np.full(20, 7.), b, constrained=True)
        self.assertTrue(result["available"])
        self.assertTrue(result["diagnostics"]["constraint_boundary_may_change_under_perturbation"])
        self.assertGreaterEqual(result["coefficients"][-1], 0)

    def test_unstable_free_branch_is_not_hidden_by_constraint_boundary(self):
        x = np.tile(np.array([-1., 1.]), 10)
        a = np.column_stack((np.ones(20), 1e-12*x)); b = np.column_stack((np.ones(4), 1e-4*x[:4]))
        self.assertUnavailable(fit(a, 7-13*x, b, constrained=True), "prediction_perturbation")

    def test_admitted_prediction_bound_covers_simultaneous_bounded_perturbations(self):
        rng = np.random.default_rng(82)
        x = np.linspace(-1, 1, 20)
        a = np.column_stack((np.ones(20), x)); b = np.array([[1., 0.], [1., .2], [1., -.2]])
        y = a@np.array([20., 3.]) + 0.1*np.sin(x*4)
        ua, ub = 1e-4, 1e-4
        result = fit(a, y, b, ua=ua, ub=ub)
        self.assertTrue(result["available"])
        for _ in range(100):
            ap = a+rng.uniform(-ua, ua, a.shape)
            bp = b+rng.uniform(-ub, ub, b.shape)
            yp = y+rng.uniform(-.5, .5, y.shape)
            pred = bp@np.linalg.lstsq(ap, yp, rcond=None)[0]
            self.assertTrue(np.all(np.abs(pred-result["prediction"]) <= result["prediction_bound"]+1e-12))

    def test_inputs_are_not_mutated_and_repeat_is_deterministic(self):
        a = np.ones((10, 1)); y = np.arange(10.); b = np.ones((2, 1)); ua = np.zeros_like(a)
        copies = [v.copy() for v in (a, y, b, ua)]
        first = fit(a, y, b, ua=ua); second = fit(a, y, b, ua=ua)
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        for actual, expected in zip((a, y, b, ua), copies):
            np.testing.assert_array_equal(actual, expected)

    def test_unrepresentable_finite_column_scale_is_unknown_not_infinite_diagnostics(self):
        result = fit(np.full((10, 1), 1e308), np.ones(10), np.full((2, 1), 1e308))
        self.assertUnavailable(result, "unrepresentable_column_scaling")

    def test_extreme_finite_design_uncertainty_remains_json_safe_on_failure(self):
        result = fit(np.full((9, 1), 1/3), np.ones(9), np.zeros((1, 1)), ua=1e308, ub=0.)
        self.assertUnavailable(result, "unrepresentable_design_perturbation_norm")
        self.assertIsNone(result["diagnostics"]["free_fit"]["train_design_perturbation_spectral_bound"])
        self.assertIsNone(result["diagnostics"]["free_fit"]["robust_minimum_singular_value"])

    def test_constraint_positive_scale_and_nuisance_permutation_invariance(self):
        x = np.tile([-1., 1.], 20); z = np.repeat([-1., 1.], 20)
        a = np.column_stack((np.ones(40), z, x)); b = a[:8]
        y = a@np.array([2., -3., 11.])
        first = fit(a, y, b, constrained=True)
        order = [1, 0, 2]; scale = np.array([-2., 1e4, 1e-7])
        second = fit(a[:, order]*scale, y, b[:, order]*scale, constrained=True)
        self.assertTrue(first["available"] and second["available"])
        np.testing.assert_allclose(first["prediction"], second["prediction"], atol=1e-10)
        np.testing.assert_allclose(first["prediction_bound"], second["prediction_bound"], atol=1e-10)

    def test_narrow_operator_norm_matches_explicit_full_prediction_operator(self):
        x = np.linspace(-1, 1, 30)
        a = np.column_stack((np.ones(30), x, x*x))
        b = np.column_stack((np.ones(11), x[9:20], x[9:20]**2))
        result = fit(a, a@np.array([1., -2., 3.]), b)
        self.assertTrue(result["available"])
        expected = float(np.linalg.norm(b@np.linalg.pinv(a), ord=2))
        self.assertAlmostEqual(result["diagnostics"]["free_fit"]["response_prediction_operator_l2_norm"], expected, places=12)

    def test_malformed_inputs(self):
        cases = [
            (np.array([[np.nan]]), np.ones(1), np.ones((1, 1)), .5, 0., 0.),
            (np.ones((3, 1)), np.ones(2), np.ones((1, 1)), .5, 0., 0.),
            (np.ones((3, 1)), np.ones(3), np.ones((1, 2)), .5, 0., 0.),
            (np.ones((3, 1)), np.ones(3), np.ones((1, 1)), -.5, 0., 0.),
            (np.ones((3, 1)), np.ones(3), np.ones((1, 1)), True, 0., 0.),
            (np.ones((3, 1)), np.ones(3), np.ones((1, 1)), .5, -1., 0.),
            (np.ones((3, 1)), np.ones(3), np.ones((1, 1)), .5, 0., np.inf),
        ]
        for a, y, b, sigma, ua, ub in cases:
            with self.subTest(sigma=sigma, ua=ua, ub=ub), self.assertRaises(ValueError):
                fit(a, y, b, sigma=sigma, ua=ua, ub=ub)


if __name__ == "__main__":
    unittest.main()
