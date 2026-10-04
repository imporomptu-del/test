import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v45_presence import source_presence


class PresenceTests(unittest.TestCase):
    def fixture(self, coefficient=12., nuisance_count=2):
        rng = np.random.default_rng(4501)
        y, x = np.indices((6, 8))
        p = np.column_stack((np.ones(48), ((x-3.5)/3.5).ravel(), ((y-2.5)/2.5).ravel()))
        raw = rng.normal(size=(48, nuisance_count+1))
        z, m = raw[:, :-1], raw[:, -1]
        target = p@np.asarray([100., 30., -50.])+z@np.arange(1., nuisance_count+1)+coefficient*m
        return dict(y=target, Z=z, m=m, P=p, response_bound=.5, nuisance_bound=.001, source_bound=.001)

    def test_signed_source_and_background_only(self):
        for coefficient, expected in ((12., "positive"), (-12., "negative"), (0., "unresolved")):
            result = source_presence(**self.fixture(coefficient))
            self.assertTrue(result["available"], result["reasons"])
            self.assertEqual(result["coefficient_sign"], expected)
            self.assertEqual(result["motion_status"], "unknown")
            self.assertEqual(result["physical_class"], "unknown")
            self.assertNotIn("estimate", result)
            json.dumps(result, allow_nan=False)

    def test_empty_nuisance_has_zero_projector_gap(self):
        args = self.fixture(nuisance_count=0)
        args["nuisance_bound"] = None
        result = source_presence(**args)
        self.assertTrue(result["available"])
        self.assertEqual(result["diagnostics"]["nuisance_projector_gap_bound"], 0.)
        self.assertEqual(result["diagnostics"]["projector_product_bound_used"], 0.)
        self.assertEqual(result["coefficient_sign"], "positive")

    def test_exact_source_nuisance_overlap_is_unknown(self):
        args = self.fixture()
        args["m"] = args["Z"][:, 0].copy()
        result = source_presence(**args)
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"], ["source_in_nominal_nuisance_span"])
        self.assertIsNone(result["numerator"])
        self.assertIsNone(result["interval"])

    def test_source_allowed_to_vanish_yields_zero_crossing_interval(self):
        args = self.fixture(nuisance_count=0)
        args["source_bound"] = np.abs(args["m"])
        result = source_presence(**args)
        self.assertTrue(result["available"])
        self.assertLessEqual(result["interval"][0], 0.)
        self.assertGreaterEqual(result["interval"][1], 0.)

    def test_uncertainty_cannot_destroy_nuisance_rank_silently(self):
        args = self.fixture()
        args["nuisance_bound"] = 10.
        result = source_presence(**args)
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"], ["projected_nuisance_robust_rank_not_certified"])
        self.assertIsNone(result["interval"])

    def test_source_bearing_near_affine_nuisance_cannot_be_dropped(self):
        m = np.asarray([.5, -.5, .5, -.5])
        p = np.ones((4, 1))
        result = source_presence(100+30*m, p+np.ldexp(m, -50)[:, None], m, p,
                                 response_bound=0., nuisance_bound=0., source_bound=0.)
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"], ["near_affine_nuisance_not_certified_redundant"])

    def test_only_literal_exact_affine_or_zero_nuisances_are_removable(self):
        args = self.fixture(nuisance_count=0)
        args["Z"] = np.column_stack((args["P"][:, 0], -args["P"][:, 1], np.zeros(48)))
        args["nuisance_bound"] = 0.
        result = source_presence(**args)
        self.assertTrue(result["available"])
        self.assertEqual(result["diagnostics"]["exact_nuisance_columns_removed_in_affine_span"], [0, 1, 2])
        args["nuisance_bound"] = .001
        self.assertFalse(source_presence(**args)["available"])

    def test_affine_brightness_changes_and_nuisance_units_are_invariant(self):
        args = self.fixture()
        baseline = source_presence(**args)
        changed = dict(args)
        changed["y"] = args["y"]+args["P"]@np.asarray([1000., -700., 500.])
        altered = source_presence(**changed)
        np.testing.assert_allclose(altered["analytic_interval"], baseline["analytic_interval"], atol=1e-9)
        self.assertGreater(altered["numerical_resolution_margin"], baseline["numerical_resolution_margin"])
        changed = dict(args)
        units = np.asarray([-1e-3, 1e3])
        changed["Z"] = args["Z"][:, ::-1]*units
        changed["nuisance_bound"] = abs(units)*.001
        altered = source_presence(**changed)
        np.testing.assert_allclose(altered["interval"], baseline["interval"], atol=1e-9)

    def test_large_affine_offset_may_reduce_numerical_sign_resolution(self):
        args = self.fixture()
        baseline = source_presence(**args)
        changed = dict(args)
        changed["y"] = args["y"]+args["P"]@np.asarray([1e16, 0., 0.])
        result = source_presence(**changed)
        self.assertTrue(result["available"])
        self.assertEqual(baseline["coefficient_sign"], "positive")
        self.assertEqual(result["coefficient_sign"], "unresolved")
        self.assertGreater(result["numerical_resolution_margin"], baseline["numerical_resolution_margin"])
        # Finite-precision y changed at 1e16 brightness, so no claim of exact
        # analytic equality is possible here. The numerical decline is intended.

    def test_source_units_transform_numerator_without_amplitude_division(self):
        args = self.fixture()
        baseline = source_presence(**args)
        for factor in (-.125, 7.):
            changed = dict(args)
            changed["m"] = factor*args["m"]
            changed["source_bound"] = abs(factor)*args["source_bound"]
            result = source_presence(**changed)
            self.assertAlmostEqual(result["numerator"], factor*baseline["numerator"])
            np.testing.assert_allclose(result["interval"], sorted(factor*np.asarray(baseline["interval"])), atol=1e-9)

    def test_simultaneous_correlated_perturbations_and_projector_gap_contained(self):
        rng = np.random.default_rng(4531)
        for coefficient in (-12., 0., 12.):
            args = self.fixture(coefficient)
            args["y"] = args["y"]+rng.normal(size=48)*10
            result = source_presence(**args)
            self.assertTrue(result["available"])
            affine = np.linalg.qr(args["P"], mode="reduced")[0]
            project = lambda v:v-affine@(affine.T@v)
            nuisance = project(args["Z"])
            nominal_basis = np.linalg.qr(nuisance, mode="reduced")[0]
            nominal_projector = nominal_basis@nominal_basis.T
            for iteration in range(80):
                signs = rng.choice([-1., 1.], 48)
                dy = signs*.5
                dm = signs*.001
                dz = signs[:, None]*.001 if iteration%2 else rng.choice([-1., 1.], args["Z"].shape)*.001
                changed_y, changed_m, changed_z = project(args["y"]+dy), project(args["m"]+dm), project(args["Z"]+dz)
                basis = np.linalg.qr(changed_z, mode="reduced")[0]
                residual_y = changed_y-basis@(basis.T@changed_y)
                numerator = float(changed_m@residual_y)
                self.assertLessEqual(abs(numerator-result["numerator"]), result["error_bound"]+1e-9)
                gap = np.linalg.norm(basis@basis.T-nominal_projector, 2)
                self.assertLessEqual(gap, result["diagnostics"]["nuisance_projector_gap_bound"]+1e-12)
                if result["interval_excludes_zero"]:
                    direct = np.linalg.lstsq(np.column_stack((args["P"], args["Z"]+dz, args["m"]+dm)),
                                             args["y"]+dy, rcond=None)[0][-1]
                    self.assertEqual(bool(direct > 0), result["coefficient_sign"] == "positive")

    def test_blockwise_projector_term_is_a_bound_not_a_threshold(self):
        result = source_presence(**self.fixture())
        diag = result["diagnostics"]
        self.assertEqual(diag["projector_product_bound_used"],
                         min(diag["projector_global_product_bound"], diag["projector_block_product_bound"]))
        self.assertAlmostEqual(result["analytic_error_bound"], diag["nominal_weighted_response_term"]+
                               diag["nominal_weighted_source_term"]+diag["source_response_cross_term"]+
                               diag["projector_product_bound_used"])
        self.assertAlmostEqual(result["error_bound"], result["analytic_error_bound"]+
                               result["numerical_resolution_margin"])

    def test_zero_error_background_only_cannot_gain_sign_from_projection_roundoff(self):
        rng = np.random.default_rng(4591)
        p = np.column_stack((np.ones(48), np.linspace(-1, 1, 48)))
        z, m = rng.normal(size=(48, 2)), rng.normal(size=48)
        guarded_nonzero = 0
        for brightness in (0., 1., 1e6, 1e12):
            for separation in (1., 1e-6):
                zz = z.copy()
                zz[:, 1] = zz[:, 0]+separation*zz[:, 1]
                for nuisance in (False, True):
                    y = p@np.asarray([brightness, -.3*brightness])
                    if nuisance: y = y+zz@np.asarray([2., -3.])
                    result = source_presence(y, zz, m, p, response_bound=0., nuisance_bound=0., source_bound=0.)
                    self.assertTrue(result["available"], result["reasons"])
                    self.assertEqual(result["analytic_error_bound"], 0.)
                    self.assertEqual(result["coefficient_sign"], "unresolved")
                    self.assertFalse(result["interval_excludes_zero"])
                    self.assertLessEqual(result["interval"][0], 0.)
                    self.assertGreaterEqual(result["interval"][1], 0.)
                    guarded_nonzero += result["diagnostics"]["sign_withheld_by_numerical_resolution"]
        self.assertGreater(guarded_nonzero, 0)

    def test_numerical_margin_expands_only_and_is_not_a_camera_noise_claim(self):
        for coefficient in (-12., 0., 12.):
            result = source_presence(**self.fixture(coefficient))
            diag = result["diagnostics"]
            self.assertFalse(diag["numerical_resolution_is_ieee_certified_enclosure"])
            self.assertFalse(diag["numerical_roundoff_in_error_set"])
            self.assertTrue(diag["numerical_margin_can_only_withhold_sign"])
            self.assertGreaterEqual(result["numerical_resolution_margin"], 0.)
            self.assertLessEqual(result["interval"][0], result["analytic_interval"][0])
            self.assertGreaterEqual(result["interval"][1], result["analytic_interval"][1])
            self.assertEqual(result["interval_excludes_zero"], result["coefficient_sign"] != "unresolved")
            gamma = diag["numerical_resolution_gamma"]
            expected = (gamma*diag["raw_source_l2_norm"]*diag["raw_response_l2_norm"]*
                        (1+diag["normalized_affine_condition"])*
                        (1+diag["nuisance_numerical_condition_amplification"]))
            self.assertEqual(result["numerical_resolution_margin"], expected)

    def test_large_numerical_resolution_guard_is_json_safe_unknown(self):
        args = self.fixture()
        args["m"] = args["m"]*1e200
        args["y"] = args["y"]*1e200
        with np.errstate(over="ignore", invalid="ignore"):
            result = source_presence(**args)
        self.assertFalse(result["available"])
        json.dumps(result, allow_nan=False)

    def test_rank_deficient_affine_or_nuisance_and_source_affine_are_unknown(self):
        args = self.fixture()
        args["P"] = np.column_stack((args["P"], args["P"][:, 0]))
        self.assertFalse(source_presence(**args)["available"])
        args = self.fixture()
        args["Z"][:, 1] = args["Z"][:, 0]
        self.assertEqual(source_presence(**args)["reasons"], ["projected_nuisance_machine_rank_deficient"])
        args = self.fixture()
        args["m"] = args["P"][:, 0].copy()
        self.assertEqual(source_presence(**args)["reasons"], ["source_in_numerical_affine_span"])

    def test_nonfinite_malformed_or_missing_error_contract(self):
        for field, value in (("y", np.ones(2)), ("m", np.full(48, np.nan)),
                             ("source_bound", -.1), ("response_bound", np.inf)):
            args = self.fixture(); args[field] = value
            with self.assertRaises(ValueError): source_presence(**args)
        for field in ("response_bound", "nuisance_bound", "source_bound"):
            args = self.fixture(); args[field] = None
            self.assertEqual(source_presence(**args)["reasons"], ["missing_declared_uncertainty"])

    def test_inputs_not_mutated_and_results_json_safe(self):
        args = self.fixture()
        before = {key:value.copy() for key,value in args.items() if isinstance(value, np.ndarray)}
        result = source_presence(**args)
        json.dumps(result, allow_nan=False)
        for key, value in before.items(): np.testing.assert_array_equal(args[key], value)


if __name__ == "__main__": unittest.main()
