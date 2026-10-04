import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v44_accounting import compare_archive, finite_json, validate_contrast


def contrast(available=True):
    return dict(available=available, reasons=[] if available else ["unsupported"],
                estimate=2. if available else None, error_bound=.5 if available else None,
                interval=[1.5, 2.5] if available else None,
                interval_excludes_zero=True if available else None,
                coefficient_sign="positive" if available else None,
                motion_status="unknown", physical_class="unknown",
                diagnostics=dict(no_production_gate=True,
                    caller_template_temporal_or_placement_provenance_certified=False,
                    calibrated_noise_or_confidence=False))


class V44AccountingAuditTests(unittest.TestCase):
    def test_valid_available_and_unavailable_preserved_without_mutation(self):
        for available in (False, True):
            value = contrast(available)
            before = copy.deepcopy(value)
            validate_contrast(value, "fixture")
            self.assertEqual(value, before)

    def test_unavailable_cannot_keep_operative_score(self):
        for field, value in (("estimate", 0), ("error_bound", 0), ("interval", [0, 0]),
                             ("interval_excludes_zero", False), ("coefficient_sign", "unresolved")):
            record = contrast(False)
            record[field] = value
            with self.assertRaisesRegex(ValueError, "operative"):
                validate_contrast(record, "fixture")

    def test_physical_promotion_and_bad_interval_semantics_rejected(self):
        for field, value in (("motion_status", "moving"), ("physical_class", "airborne"),
                             ("interval", [1, 3]), ("interval_excludes_zero", False),
                             ("coefficient_sign", "negative")):
            record = contrast()
            record[field] = value
            with self.assertRaises(ValueError):
                validate_contrast(record, "fixture")

    def test_zero_crossing_available_is_not_unavailable_or_negative(self):
        record = contrast()
        record.update(estimate=0., error_bound=1., interval=[-1., 1.],
                      interval_excludes_zero=False, coefficient_sign="unresolved")
        validate_contrast(record, "fixture")
        self.assertTrue(record["available"])

    def test_nested_nonfinite_rejected(self):
        for value in (float("inf"), float("nan"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                finite_json({"nested": [0., {"value": value}]})

    def test_archive_shape_dtype_membership_values_and_missing_center_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.npz"
            expected = {"data": np.arange(4, dtype=float), "centers": np.array([[1., 2.], [np.nan, np.nan]])}
            np.savez_compressed(path, **expected)
            compare_archive(path, expected)
            for changed in ({"data": expected["data"]},
                            dict(expected, data=np.arange(4, dtype=np.int64)),
                            dict(expected, data=np.arange(4, dtype=float).reshape((2, 2))),
                            dict(expected, data=np.arange(4, dtype=float)+1)):
                with self.assertRaises(ValueError):
                    compare_archive(path, changed)


if __name__ == "__main__":
    unittest.main()
