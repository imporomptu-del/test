from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v44_contrast import (SEED, call_core, cases, independent_nominal,
                                         invariance_audit, refit_audit)


class ContrastAuditTests(unittest.TestCase):
    def test_predetermined_grid_not_filtered_for_success(self):
        generated = cases()
        self.assertEqual(len(generated), 55)
        self.assertEqual(sum(case["specification"]["kind"] == "grid" for case in generated), 48)
        self.assertEqual(len({case["case_id"] for case in generated}), 55)
        for a, b in zip(generated, cases()):
            for key in ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound"):
                np.testing.assert_array_equal(a[key], b[key])

    def test_positive_negative_empty_nuisance_and_correlated_refits(self):
        generated = cases()
        for index in (0, 3, 10, 19, 22, 47):
            case = generated[index]
            nominal, actual = independent_nominal(case), call_core(case)
            result = refit_audit(case, nominal, actual, count=40, seed=SEED+index)
            self.assertEqual(result["issues"], [], case["case_id"])
            self.assertEqual(result["refits"], 40)

    def test_direct_response_extremum_attains_exact_design_bound(self):
        case = cases()[0]
        nominal, actual = independent_nominal(case), call_core(case)
        result = refit_audit(case, nominal, actual, count=4)
        self.assertAlmostEqual(result["maximum_contrast_error_fraction"], 1., places=9)

    def test_affine_nuisance_and_signed_source_units_invariance(self):
        for index in (1, 21, 47):
            case = cases()[index]
            result = invariance_audit(case, call_core(case))
            self.assertEqual(result["issues"], [])
            self.assertGreaterEqual(result["checks"], 3)

    def test_rank_and_affine_ambiguity_retained_unknown(self):
        for case in cases()[48:]:
            nominal, actual = independent_nominal(case), call_core(case)
            result = refit_audit(case, nominal, actual, count=4)
            self.assertEqual(result["issues"], [], case["case_id"])
            expected = case["specification"]["expected_unknown_reason"]
            self.assertEqual(actual["reasons"], [] if expected is None else [expected])

    def test_independent_refit_detects_incorrect_reported_interval(self):
        case = cases()[0]
        nominal, actual = independent_nominal(case), call_core(case)
        actual["error_bound"] = 0.
        result = refit_audit(case, nominal, actual, count=4)
        self.assertTrue(any(issue["reason"] == "coefficient_interval_violation" for issue in result["issues"]))

    def test_source_bearing_near_affine_nuisance_is_not_discarded(self):
        case = cases()[-1]
        self.assertEqual(case["case_id"], "near_affine_nuisance_carries_source")
        self.assertFalse(np.array_equal(case["Z"], case["P"]))
        for outcome in (independent_nominal(case), call_core(case)):
            self.assertFalse(outcome["available"])

    def test_exact_redundancy_scaling_loss_is_reported_not_called_invariance(self):
        case = next(case for case in cases() if case["case_id"] == "exact_redundant_affine_nuisance")
        result = invariance_audit(case, call_core(case))
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["checks"], 3)
        self.assertEqual(len(result["conservative_transformed_controls"]), 1)


if __name__ == "__main__": unittest.main()
