import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v44_contrast import source_contrast


def fixture():
    x = np.linspace(-1, 1, 81)
    P = np.column_stack((np.ones(len(x)), x))
    Z = np.column_stack((np.sin(3*x), np.cos(4*x)))
    m = np.exp(-((x-.15)/.08)**2)
    m /= np.linalg.norm(m)
    y = P @ [100, 3] + Z @ [2, -4] + 20*m
    return dict(y=y, Z=Z, m=m, P=P, response_bound=.5, nuisance_bound=.001, source_bound=.0001)


class SourceContrastTests(unittest.TestCase):
    def test_ordinary_source_has_available_positive_conditional_interval(self):
        result = source_contrast(**fixture())
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["estimate"], 20., places=10)
        self.assertGreater(result["interval"][0], 0)
        self.assertEqual(result["motion_status"], "unknown")
        self.assertEqual(result["physical_class"], "unknown")
        json.dumps(result, allow_nan=False)

    def test_exact_affine_brightness_does_not_change_contrast_or_bound(self):
        data = fixture()
        expected = source_contrast(**data)
        data["y"] = data["y"] + data["P"] @ [700, -30]
        actual = source_contrast(**data)
        self.assertAlmostEqual(actual["estimate"], expected["estimate"], places=10)
        self.assertAlmostEqual(actual["error_bound"], expected["error_bound"], places=10)

    def test_affine_basis_scaling_and_order_do_not_change_result(self):
        data = fixture()
        expected = source_contrast(**data)
        data["P"] = data["P"][:, ::-1] * [1e-6, -1e4]
        actual = source_contrast(**data)
        np.testing.assert_allclose(actual["interval"], expected["interval"], atol=1e-10)

    def test_nuisance_rescaling_and_permutation_preserve_interval(self):
        data = fixture()
        expected = source_contrast(**data)
        data["Z"] = data["Z"][:, ::-1] * [-50, .002]
        data["nuisance_bound"] = np.asarray([.05, .000002])
        actual = source_contrast(**data)
        np.testing.assert_allclose(actual["interval"], expected["interval"], atol=1e-10)

    def test_source_units_scale_signed_interval(self):
        data = fixture()
        expected = source_contrast(**data)
        data["m"] = data["m"] * -2
        data["source_bound"] *= 2
        actual = source_contrast(**data)
        np.testing.assert_allclose(actual["interval"], -np.asarray(expected["interval"])[::-1]/2, atol=1e-10)
        self.assertEqual(actual["coefficient_sign"], "negative")

    def test_background_only_interval_contains_zero(self):
        data = fixture()
        data["y"] -= 20*data["m"]
        result = source_contrast(**data)
        self.assertTrue(result["available"])
        self.assertFalse(result["interval_excludes_zero"])
        self.assertEqual(result["coefficient_sign"], "unresolved")

    def test_fixed_light_overlapping_source_is_not_identifiable(self):
        data = fixture()
        data["Z"] = np.column_stack((data["Z"], data["m"]))
        result = source_contrast(**data)
        self.assertFalse(result["available"])
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["interval"])

    def test_uncertainty_that_can_erase_template_is_unknown(self):
        data = fixture()
        data["source_bound"] = np.abs(data["m"])
        result = source_contrast(**data)
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"], ["projected_design_robust_rank_not_certified"])

    def test_no_nuisance_columns_supported(self):
        data = fixture()
        data["Z"] = np.empty((81, 0))
        data["nuisance_bound"] = None
        data["y"] = data["P"] @ [15., 2.] + data["m"] * 20
        self.assertAlmostEqual(source_contrast(**data)["estimate"], 20., places=10)

    def test_exact_affine_nuisance_removed_but_uncertain_one_not_silently_dropped(self):
        data = fixture()
        data["Z"] = np.ones((81, 1))
        data["nuisance_bound"] = 0
        data["y"] = data["P"] @ [15., 2.] + data["m"] * 20
        result = source_contrast(**data)
        self.assertTrue(result["available"])
        self.assertEqual(result["diagnostics"]["exact_nuisance_columns_removed_in_affine_span"], [0])
        data["nuisance_bound"] = .01
        self.assertEqual(source_contrast(**data)["reasons"], ["near_affine_nuisance_not_certified_redundant"])

    def test_tiny_source_bearing_nuisance_is_never_discarded_as_affine(self):
        m = np.array([.5, -.5, .5, -.5])
        P = np.ones((4, 1))
        Z = P + 2.**-50*m[:, None]
        np.testing.assert_array_equal((Z-P)[:, 0], 2.**-50*m)
        result = source_contrast(100+30*m, Z, m, P, response_bound=0,
                                 nuisance_bound=0, source_bound=0)
        self.assertFalse(result["available"])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["diagnostics"]["exact_nuisance_columns_removed_in_affine_span"], [])

    def test_source_in_affine_span_and_bad_affine_basis_decline(self):
        data = fixture()
        data["m"] = np.ones(81)
        self.assertEqual(source_contrast(**data)["reasons"], ["source_in_numerical_affine_span"])
        data = fixture()
        data["P"] = np.ones((81, 2))
        self.assertEqual(source_contrast(**data)["reasons"], ["rank_deficient_exact_affine_basis"])

    def test_empty_affine_span_is_allowed(self):
        data = fixture()
        data["y"] -= data["P"] @ [100, 3]
        data["P"] = np.empty((81, 0))
        self.assertAlmostEqual(source_contrast(**data)["estimate"], 20., places=10)

    def test_missing_uncertainty_unknown(self):
        for name in ("response_bound", "nuisance_bound", "source_bound"):
            data = fixture()
            data[name] = None
            self.assertEqual(source_contrast(**data)["reasons"], ["missing_declared_uncertainty"])

    def test_invalid_support_bounds_and_shapes_rejected(self):
        for name, value in (("response_bound", -1), ("source_bound", np.nan),
                            ("nuisance_bound", np.ones((81, 9))), ("y", np.ones(2))):
            data = fixture()
            data[name] = value
            with self.assertRaises(ValueError):
                source_contrast(**data)
        for name in ("y", "m", "Z", "P"):
            data = fixture()
            data[name].flat[0] = np.nan
            with self.assertRaises(ValueError):
                source_contrast(**data)

    def test_finite_extreme_bound_declines_without_nonfinite_json(self):
        data = fixture()
        data["source_bound"] = 1e308
        result = source_contrast(**data)
        self.assertFalse(result["available"])
        json.dumps(result, allow_nan=False)

    def test_inputs_not_mutated(self):
        data = fixture()
        before = copy.deepcopy(data)
        source_contrast(**data)
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])

    def test_signed_response_vertex_with_exact_design_attains_bound(self):
        data = fixture()
        data.update(nuisance_bound=0, source_bound=0)
        result = source_contrast(**data)
        full = np.column_stack((data["P"], data["Z"], data["m"]))
        dual = np.linalg.pinv(full)[-1]
        for sign in (-1, 1):
            changed = data["y"] + sign*.5*np.sign(dual)
            coefficient = np.linalg.lstsq(full, changed, rcond=None)[0][-1]
            self.assertAlmostEqual(coefficient, result["estimate"] + sign*result["error_bound"], places=10)


if __name__ == "__main__":
    unittest.main()
