"""Synthetic-only independent audit unit controls; no real inputs or scores."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v48_synthetic import (independent_interval, audit_gain, audit_guard_images,
    validate_hull, validate_context_override, CURRENT_USE_DESCRIPTION, independently_round_boundary,
    validate_preserved_arrays, verify_bindings, BASE, ROOT, CORRECTED_ID, audit_rendering_contract,
    audit_previous_boundary_failure)
from accuracy_v47_guard_gain import estimate_guard_gain
from accuracy_v48_synthetic_cases import build_cases, scenario_manifest, pre_score_contract
from accuracy_v47_synthetic_cases import build_cases as old_cases
from audit_accuracy_v46_synthetic import expected_arrays


class V48SyntheticAuditTests(unittest.TestCase):
    def test_inward_exact_endpoint_is_inside_and_nearest_boundary(self):
        for background in np.linspace(42,78,101):
            latent = Fraction.from_float(1.2)*Fraction.from_float(float(background))
            for sign in (-1,1):
                observed = independently_round_boundary(latent, sign)
                error = Fraction.from_float(observed)-latent
                self.assertLessEqual(abs(error), Fraction(1,2))
                neighbor = np.nextafter(observed, np.inf if sign > 0 else -np.inf)
                self.assertGreater(sign*(Fraction.from_float(float(neighbor))-latent), Fraction(1,2))

    def test_exact_corrected_annulus_rejects_legacy_boundary_rounding(self):
        cases = build_cases()
        old = next(c for c in old_cases() if c["case_id"] == CORRECTED_ID)
        corrected = next(c for c in cases if c["case_id"] == CORRECTED_ID)
        corrected["adapter_inputs"]["current129"] = old["adapter_inputs"]["current129"].copy()
        with self.assertRaisesRegex(ValueError, "exact inward"):
            audit_guard_images(cases, scenario_manifest())

    def test_exact_preservation_except_corrected_current_full_annulus(self):
        new, old = build_cases(), old_cases()
        for a,b in zip(new,old):
            validate_preserved_arrays("guard_"+a["case_id"], expected_arrays(a["adapter_inputs"]), expected_arrays(b["adapter_inputs"]))
        actual, previous = [expected_arrays(c[-1]["adapter_inputs"]) for c in (new,old)]
        for key,index in (("current129", (64,64)), ("history129", (0,8,8)), ("predicted_offset_xy", (0,))):
            changed = {k:v.copy() for k,v in actual.items()}; changed[key][index] += 1
            with self.assertRaises(ValueError):
                validate_preserved_arrays("guard_"+CORRECTED_ID, changed, previous)
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_preserved_arrays("guard_"+CORRECTED_ID, previous, previous)

    def test_no_real_data_dependency_can_be_opened(self):
        for name in (ROOT/"results/tiny_target/accuracy_v43_20260925/stability_01/inputs/forbidden.npz",
                     ROOT/"scripts/journals/forbidden.json", ROOT/"raw16/forbidden.raw"):
            with patch("audit_accuracy_v48_synthetic.sha") as reader:
                with self.assertRaisesRegex(ValueError, "synthetic-only"):
                    verify_bindings({str(name):"not-read"}, BASE/"synthetic_01")
                reader.assert_not_called()

    def test_exact_full_annulus_witness_is_reconstructed_and_tamper_rejected(self):
        cases = build_cases(); witness = pre_score_contract(cases)
        result = audit_rendering_contract(cases, witness)
        self.assertEqual(result["independently_recomputed_witness"], witness)
        self.assertEqual(witness["full_annulus_pixel_count"], 6528)
        self.assertEqual(witness["prior_witness_observation_count"], 51224)
        self.assertEqual(witness["prior_error_not_exact_endpoint_count"], 128)
        self.assertEqual(witness["prior_witness_violation_count"], 0)
        changed = dict(witness, current_max_abs_error_exact="0")
        with self.assertRaisesRegex(ValueError, "Pre-score exact"):
            audit_rendering_contract(cases, changed)

    def test_frozen_bad_fixture_failure_still_reproduces(self):
        # Reads only explicitly pinned V47 saved synthetic inputs/result metadata.
        result = audit_previous_boundary_failure()
        self.assertEqual(result["retained_invalid_current_pixels"], 73)
        self.assertEqual(result["retained_invalid_constraints"], 59)
        self.assertEqual(result["retained_exact_guard_outcome"], "unknown")

    def test_halfspaces_signed_zero_inconsistent_and_unbounded(self):
        # |10-10g| <= 2+2g gives g in [2/3,3/2].
        self.assertEqual(independent_interval([(Fraction(10), Fraction(10), Fraction(2), Fraction(2))]),
                         (True, Fraction(2,3), Fraction(3,2)))
        self.assertEqual(independent_interval([(Fraction(-10), Fraction(-10), Fraction(2), Fraction(2))]),
                         (True, Fraction(2,3), Fraction(3,2)))
        self.assertEqual(independent_interval([(Fraction(0), Fraction(0), Fraction(2), Fraction(2))]),
                         (True, Fraction(0), None))
        self.assertEqual(independent_interval([(Fraction(0), Fraction(3), Fraction(0), Fraction(2))]),
                         (False, None, None))

    @staticmethod
    def generated_guard():
        y,x = np.indices((129,129), dtype=float)
        background = 50+20*(x >= 64)+8*np.sin(y/12)
        bound = np.full_like(background, .5)
        history = np.broadcast_to(background, (8,129,129)).copy()
        current = 1.2*background+3+.02*x-.01*y
        inputs = dict(current129=current, history129=history, prior_centers_xy=[None]*8,
                      predicted_offset_xy=[0.,0.], polarity="bright")
        value = estimate_guard_gain(current, history, background, bound, inputs["prior_centers_xy"])
        return inputs, background, bound, value

    def test_independent_guard_reconstruction_and_tamper_rejection(self):
        inputs, background, bound, value = self.generated_guard()
        checked = audit_gain(inputs, background, bound, value)
        self.assertEqual(checked["outcome"], "finite")
        for field, replacement in (("gain_interval", [1.2,1.2]), ("used_count", 0)):
            changed = deepcopy(value); changed[field] = replacement
            with self.assertRaises(ValueError):
                audit_gain(inputs, background, bound, changed)
        changed = deepcopy(value); changed["contrast_constraints"][0]["response"] = "10000"
        with self.assertRaisesRegex(ValueError, "contrast/error"):
            audit_gain(inputs, background, bound, changed)

    def test_guard_support_cannot_be_reselected_after_current_nan(self):
        inputs, background, bound, value = self.generated_guard()
        x,y = value["used_points_xy"][0]
        inputs["current129"][y,x] = np.nan
        changed = estimate_guard_gain(inputs["current129"], inputs["history129"], background, bound, inputs["prior_centers_xy"])
        self.assertEqual(audit_gain(inputs, background, bound, changed)["outcome"], "missing_current")
        with self.assertRaises(ValueError):
            audit_gain(inputs, background, bound, value)

    def test_independent_guard_case_rendering_and_metadata(self):
        cases = build_cases()
        result = audit_guard_images(cases, scenario_manifest())
        self.assertEqual(result["full_images_reconstructed"], 180)
        self.assertLessEqual(result["max_abs_render_difference"], 1e-12)
        cases[0]["adapter_inputs"]["current129"][0,0] += .01
        with self.assertRaisesRegex(ValueError, "rendering"):
            audit_guard_images(cases, scenario_manifest())

    def test_gain_truth_or_validity_cannot_change(self):
        cases = build_cases()
        cases[15]["generator_truth"]["gain_transfer_valid_in_declared_world"] = True
        with self.assertRaisesRegex(ValueError, "truth changed"):
            audit_guard_images(cases, scenario_manifest())

    @staticmethod
    def fake_hull():
        def endpoint(gain, center):
            score = dict(available=True, reasons=[], numerator=center, error_bound=1., interval=[center-1,center+1],
                analytic_error_bound=.5, analytic_interval=[center-.5,center+.5], numerical_resolution_margin=.5,
                interval_excludes_zero=True, coefficient_sign="positive", motion_status="unknown", physical_class="unknown",
                diagnostics={"numerical_resolution_is_ieee_certified_enclosure":False})
            return dict(gain=gain, available=True, source_presence=score)
        return dict(available=True,reasons=[],quantity="bounded_gain_nuisance_residualized_source_numerator",
            motion_status="unknown",physical_class="unknown",is_motion_or_classification_gate=False,
            gain_provenance_certified=False,numerator=None,error_bound=None,gain_interval=[1.,2.],
            interval=[1.,12.],interval_excludes_zero=True,coefficient_sign="positive",
            endpoint_evaluations=[endpoint(1.,2.),endpoint(2.,11.)],diagnostics=dict(
                whole_pipeline_is_ieee_certified_enclosure=False,
                physical_guard_cleanliness_and_core_transfer_not_certified=True,
                fixed_columns_not_dropped_by_wrapper=True,common_support_not_changed=True))

    def test_hull_requires_both_endpoints_and_never_claims_amplitude(self):
        original = self.fake_hull(); validate_hull(original)
        for field,replacement in (("interval",[1.,3.]),("numerator",2.),("physical_class","airborne")):
            changed=deepcopy(original);changed[field]=replacement
            with self.assertRaises(ValueError):validate_hull(changed)
        changed=deepcopy(original);changed["endpoint_evaluations"]=changed["endpoint_evaluations"][:1]
        with self.assertRaises(ValueError):validate_hull(changed)

    def test_only_disclosed_current_guard_use_metadata_may_change(self):
        baseline = dict(prior_context=dict(current_values_used_for="legacy response only", frames=8))
        wrapped = dict(raw_adapter_result=dict(prior_context=dict(current_values_used_for=CURRENT_USE_DESCRIPTION, frames=8)),
            current_use_metadata_override={"path":"raw_adapter_result.prior_context.current_values_used_for",
                                           "from":"legacy response only", "to":CURRENT_USE_DESCRIPTION})
        validate_context_override(wrapped, baseline)
        wrapped["raw_adapter_result"]["prior_context"]["frames"] = 7
        with self.assertRaisesRegex(ValueError, "beyond declared"):
            validate_context_override(wrapped, baseline)


if __name__ == "__main__":
    unittest.main()
