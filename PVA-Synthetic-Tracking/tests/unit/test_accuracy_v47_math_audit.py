from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_accuracy_v47_math as audit


class IndependentMathAuditTests(unittest.TestCase):
    def test_matrix_is_predeclared_deterministic_and_not_selected_by_result(self):
        first,second=audit.build_matrix(),audit.build_matrix()
        self.assertEqual(len(first),24)
        self.assertEqual(audit.matrix_manifest()["total_realizations"],768)
        self.assertEqual(audit._input_hashes(first),audit._input_hashes(second))
        self.assertEqual(len(set(c["case_id"] for c in first)),24)
        self.assertEqual({c["args"]["F"].shape[1] for c in first},{0,2})

    def test_full_seeded_768_realization_matrix_contains_independent_refits(self):
        result=audit.audit_matrix()
        self.assertTrue(result["passed"],result["issues"])
        self.assertEqual(result["realizations"],768)
        self.assertEqual(result["direct_partial_regressions"],2304)
        self.assertTrue(all(row["available"] for row in result["records"]))
        self.assertLess(max(r["max_endpoint_affine_identity_error"] for r in result["records"]),1e-9)

    def test_full_space_partial_regression_has_affine_gain_and_coefficient_identity(self):
        a=audit.build_matrix()[-1]["args"]
        first=audit.direct_partial_regression(a["y"],a["B"],a["F"],a["m"],a["P"],.7)
        last=audit.direct_partial_regression(a["y"],a["B"],a["F"],a["m"],a["P"],1.4)
        middle=audit.direct_partial_regression(a["y"],a["B"],a["F"],a["m"],a["P"],1.05)
        self.assertAlmostEqual(middle[0],.5*(first[0]+last[0]),places=11)
        self.assertAlmostEqual(middle[0],middle[1]*middle[2],places=11)

    def test_exact_centering_wide_scale_overflow_and_subnormal_audit(self):
        result=audit.audit_centering()
        self.assertTrue(result["passed"],result["issues"])
        self.assertEqual(result["cases"],38)
        self.assertGreaterEqual(result["unknown_cases"],2)
        self.assertGreater(result["exact_scalar_rows_checked"],50)

    def test_unsupported_overlap_rank_and_missing_gain_are_unknown(self):
        result=audit.audit_structural_unknowns()
        self.assertTrue(result["passed"],result["issues"])
        self.assertEqual(len(result["records"]),4)

    def test_audit_detects_corrupted_hull_instead_of_merely_rerunning_core(self):
        actual=audit.bounded_background_presence
        def corrupt(**kwargs):
            value=deepcopy(actual(**kwargs));value["interval"]=[1e6,1e6+1]
            return value
        with mock.patch.object(audit,"bounded_background_presence",side_effect=corrupt):
            result=audit.audit_matrix(audit.build_matrix()[:1],perturbations=2)
        self.assertFalse(result["passed"])
        self.assertTrue(any(i["reason"]=="independent_refit_discrepancy" for i in result["issues"]))

    def test_persisted_execution_requires_explicit_authorization(self):
        with mock.patch.object(audit,"audit_matrix") as numeric,mock.patch.object(audit,"_sha") as reads:
            with self.assertRaisesRegex(ValueError,"execute"):
                audit.run(audit.OUTPUT_ROOT/"not_to_be_created.json")
        numeric.assert_not_called();reads.assert_not_called()

    def test_audit_grid_does_not_mutate_inputs(self):
        cases=audit.build_matrix()[:2];before=audit._input_hashes(cases)
        result=audit.audit_matrix(cases,perturbations=2)
        self.assertTrue(result["passed"],result["issues"])
        self.assertEqual(audit._input_hashes(cases),before)


if __name__=="__main__":unittest.main()
