import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v45_accounting import compare_archives, counts, validate_evidence


def numerator():
    return dict(available=True, reasons=[], numerator=2., analytic_error_bound=.5, analytic_interval=[1.5, 2.5],
                numerical_resolution_margin=.5, error_bound=1., interval=[1., 3.], interval_excludes_zero=True,
                coefficient_sign="positive", motion_status="unknown", physical_class="unknown",
                diagnostics={"numerical_resolution_is_ieee_certified_enclosure": False})


class V45AccountingTests(unittest.TestCase):
    def test_quantity_and_resolution_semantics(self):
        value = numerator()
        before = copy.deepcopy(value)
        validate_evidence(value, "numerator")
        self.assertEqual(value, before)
        for key, replacement in (("analytic_interval", [1., 3.]), ("numerical_resolution_margin", 0.),
                                 ("coefficient_sign", "negative"), ("motion_status", "moving")):
            changed = copy.deepcopy(value)
            changed[key] = replacement
            with self.assertRaises(ValueError):
                validate_evidence(changed, "numerator")

    def test_unknown_cannot_retain_operative_values(self):
        value = numerator()
        value.update(available=False, reasons=["unsupported"])
        with self.assertRaisesRegex(ValueError, "operative"):
            validate_evidence(value, "numerator")
        for key in ("numerator", "analytic_error_bound", "analytic_interval", "numerical_resolution_margin",
                    "error_bound", "interval", "interval_excludes_zero", "coefficient_sign"):
            value[key] = None
        validate_evidence(value, "numerator")

    def test_accounting_keeps_unavailable_cases_in_denominator(self):
        value = numerator()
        unresolved = numerator()
        unresolved.update(interval=[-1., 5.], error_bound=3., interval_excludes_zero=False, coefficient_sign="unresolved")
        result = counts([None, value, unresolved])
        self.assertEqual(result, dict(states=3, available=2, unavailable=1, excludes_zero=1, includes_zero=1,
                                     positive=1, negative=0, unresolved=1))

    def test_archive_comparison_requires_exact_dtype_shape_values_and_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/"old.npz", Path(directory)/"new.npz"
            arrays = dict(x=np.array([[1., np.nan]]), y=np.arange(3, dtype=np.int64))
            np.savez_compressed(old, **arrays)
            np.savez_compressed(new, **arrays)
            hashes = compare_archives(new, old, "case")
            self.assertEqual(set(hashes), {"case:x", "case:y"})
            np.savez_compressed(new, **dict(arrays, y=np.arange(3, dtype=np.float64)))
            with self.assertRaisesRegex(ValueError, "input changed"):
                compare_archives(new, old, "case")


if __name__ == "__main__":
    unittest.main()
