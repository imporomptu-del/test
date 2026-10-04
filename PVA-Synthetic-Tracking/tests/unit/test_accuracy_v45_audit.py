from decimal import Decimal
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_accuracy_v45_uncertainty as audit


class V45IndependentAuditTests(unittest.TestCase):
    def test_decimal_independent_exact_coordinate_formula(self):
        endpoints = audit.decimal_endpoints(np.array([3., 4.]), np.array([3., 4.]))
        self.assertEqual(endpoints, [(Decimal(".6"), Decimal(".6")), (Decimal(".8"), Decimal(".8"))])

    def test_prespecified_case_denominators_and_seeds_are_deterministic(self):
        self.assertEqual(len(audit.endpoint_boxes()), 20)
        self.assertEqual(sum(2*len(lo) for _, lo, _ in audit.endpoint_boxes()), 10680)
        cases, again = audit.numerator_cases(), audit.numerator_cases()
        self.assertEqual(len(cases), 60)
        self.assertEqual(sum(c["kind"] == "grid" for c in cases), 48)
        for left, right in zip(cases, again):
            self.assertEqual(left["case_id"], right["case_id"])
            for key in audit.ARRAY_KEYS:
                np.testing.assert_array_equal(left[key], right[key])

    def test_independent_numerator_agrees_with_plain_full_least_squares(self):
        case = audit.numerator_cases()[8]
        value, source_residual, _, _ = audit.direct_numerator(case["y"], case["Z"], case["m"], case["P"])
        coefficient = np.linalg.lstsq(np.column_stack((case["P"], case["Z"], case["m"])), case["y"], rcond=None)[0][-1]
        self.assertAlmostEqual(value/(source_residual@source_residual), coefficient, places=10)

    def test_raw_fixture_membership_reconstruction_uses_saved_indices_not_new_peaks(self):
        case = audit.raw_cases()[1]
        original = audit.components43.prepare_components(case["history"], case["centers"], case["offset"], case["polarity"])
        self.assertGreater(len(original["fixed_templates"]), 0)
        reconstructed = audit.frozen_membership_components(case["history"], case["centers"], case["offset"], case["polarity"], original)
        for key in ("background", "moving_template", "fixed_templates"):
            np.testing.assert_allclose(reconstructed[key], original[key], rtol=1e-13, atol=1e-14, equal_nan=True)

    def test_preserves_original_rounding_failure_record(self):
        finding = audit.INITIAL_ENDPOINT_FINDING
        self.assertEqual(finding["endpoint_comparisons"], 10680)
        self.assertEqual(finding["strict_inward_endpoints"], 3437)
        self.assertGreater(finding["maximum_inward_ulps"], 11)


if __name__ == "__main__":
    unittest.main()
