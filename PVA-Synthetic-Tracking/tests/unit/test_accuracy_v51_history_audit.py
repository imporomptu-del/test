import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_accuracy_v51_history as audit


class V51HistoryAuditTests(unittest.TestCase):
    def diagnostic(self, values):
        return audit.scalar_diagnostic(np.asarray(values, dtype=float).reshape(8, 1))

    def case(self, family, amplitude=8, level=0):
        spec = next(spec for spec in audit.specifications() if spec["family"] == family
                    and spec["amplitude"] == amplitude and spec["noise_level"] == level)
        return audit.generated_input(spec)

    def test_auditor_imports_no_runtime_implementations(self):
        tree = ast.parse(inspect.getsource(audit))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertFalse(any("accuracy_v51" in name for name in imports))
        self.assertNotIn("np.median", inspect.getsource(audit))

    def test_scalar_median_uses_even_midpoint_and_odd_order_statistic(self):
        self.assertEqual(audit.median([4, 1, 6, 2]), 3)
        self.assertEqual(audit.median([9, -3, 2]), 2)
        self.assertEqual(audit.median([1]), 1)
        with self.assertRaises(ValueError):
            audit.median([])

    def test_constant_history_keeps_optional_descriptors_undefined(self):
        result = self.diagnostic([100] * 8)
        self.assertEqual(result["slow8"][0], 100)
        self.assertEqual(result["fast3"][0], 100)
        self.assertEqual(result["scale"][0], 1)
        self.assertTrue(result["point_available"][0])
        self.assertFalse(result["return_fraction_defined"][0])
        self.assertFalse(result["recent_center_defined"][0])
        self.assertTrue(math.isnan(result["return_fraction"][0]))
        np.testing.assert_array_equal(result["recent_center_margins"], 0)

    def test_scalar_ramp_statistics_and_normalization(self):
        result = self.diagnostic(range(8))
        for key, expected in (("slow8", 3.5), ("fast3", 6), ("early5", 2),
                               ("scale", 2), ("fast_slow_delta", 2.5),
                               ("early_recent_delta", 4), ("departure_envelope", 5),
                               ("return_fraction", 0), ("recent_center_suffix_length", 3)):
            self.assertEqual(result[key][0], expected)
        np.testing.assert_array_equal(result["recent_departures"][:, 0], [3, 4, 5])
        np.testing.assert_array_equal(result["recent_center_margins"][:, 0], [2, 4, 4])
        np.testing.assert_array_equal(result["recent_departures_normalized"][:, 0], [1.5, 2, 2.5])

    def test_return_and_strict_suffix_tie(self):
        result = self.diagnostic([100] * 5 + [140, 140, 120])
        self.assertEqual(result["return_fraction"][0], .5)
        self.assertEqual(result["recent_center_suffix_length"][0], 0)
        self.assertEqual(result["recent_center_margins"][-1, 0], 0)
        result = self.diagnostic([100] * 5 + [140, 140, 100])
        self.assertEqual(result["return_fraction"][0], 1)
        self.assertEqual(result["recent_center_suffix_length"][0], 0)

    def test_sign_reversal_is_not_return(self):
        result = self.diagnostic([100] * 5 + [140, 140, 60])
        self.assertEqual(result["return_fraction"][0], 0)
        np.testing.assert_array_equal(result["recent_departures"][:, 0], [40, 40, -40])
        self.assertEqual(result["recent_center_suffix_length"][0], 0)

    def test_scalar_suffix_all_four_lengths(self):
        for recent, length in (([90, 120, 120], 2), ([120, 90, 120], 1),
                               ([120, 120, 90], 0), ([120, 120, 120], 3)):
            self.assertEqual(self.diagnostic([100] * 5 + recent)["recent_center_suffix_length"][0], length)

    def test_missing_retains_column_and_all_numeric_unknowns(self):
        history = np.column_stack((np.arange(8, dtype=float), np.full(8, np.nan)))
        result = audit.scalar_diagnostic(history)
        self.assertEqual(result["available_count"], 1)
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["point_unavailable_reasons"], (None, "nonfinite_input_after_float64_conversion"))
        for name in audit.VECTORS:
            self.assertTrue(np.isnan(result[name][1]))
        for name in audit.MATRICES:
            self.assertTrue(np.isnan(result[name][:, 1]).all())

    def test_overflow_is_unavailable_not_zero(self):
        result = self.diagnostic([np.finfo(np.float64).max] * 8)
        self.assertFalse(result["point_available"][0])
        self.assertEqual(result["point_unavailable_reasons"][0], "nonfinite_descriptor_arithmetic")

    def test_invalid_history_schema_rejected(self):
        for history in (np.zeros((7, 2)), np.zeros(8), np.zeros((8, 2), dtype=bool),
                        np.zeros((8, 2), dtype=complex)):
            with self.assertRaises(ValueError):
                audit.scalar_diagnostic(history)

    def test_fingerprint_binds_arrays_mask_reason_metadata(self):
        baseline = self.diagnostic([100] * 5 + [140] * 3)
        original = audit.fingerprint(baseline)
        self.assertEqual(original, baseline["diagnostic_sha256"])
        for field in ("slow8", "point_available", "metadata", "point_unavailable_reasons"):
            value = deepcopy(baseline)
            if field == "slow8":
                value[field][0] += 1
            elif field == "point_available":
                value[field][0] = False
            elif field == "metadata":
                value[field]["current_argument_accepted"] = True
            else:
                value[field] = ("wrong",)
            self.assertNotEqual(audit.fingerprint(value), original)

    def test_fingerprint_nulls_unknowns_and_ignores_only_own_digest(self):
        result = self.diagnostic([100] * 8)
        original = audit.fingerprint(result)
        result["diagnostic_sha256"] = "altered"
        self.assertEqual(audit.fingerprint(result), original)
        json.dumps(audit.plain(result), allow_nan=False)

    def test_case_schema_has_98_unique_cases_and_40_twins(self):
        specs = audit.specifications()
        self.assertEqual(len(specs), 98)
        self.assertEqual(len({spec["case_id"] for spec in specs}), 98)
        self.assertEqual(len(audit.expected_twins()), 40)
        self.assertEqual(len(audit.POINTS), 144)

    def test_generated_cases_analytic_formulas_and_phases(self):
        case = self.case("ramp", amplitude=32)
        self.assertEqual(case["signal_delta"][20, 0], 0)
        self.assertEqual(case["signal_delta"][35, 0], 30)
        self.assertEqual(case["signal_delta"][36, 0], 32)
        self.assertTrue(case["event_active"][20])
        self.assertEqual(case["response_phase"][20], "onset")
        case = self.case("pulse2")
        self.assertTrue((case["signal_delta"][20:22] == 8).all())
        self.assertTrue((case["signal_delta"][22:] == 0).all())
        self.assertEqual(case["response_phase"][22], "post_event")

    def test_generated_noise_reuse_and_twin_prefixes(self):
        specs = {spec["case_id"]: spec for spec in audit.specifications()}
        for pair in audit.expected_twins():
            first = audit.generated_input(specs[pair["step_case_id"]])["values"]
            second = audit.generated_input(specs[pair["pulse_case_id"]])["values"]
            frame = pair["response_frame"]
            np.testing.assert_array_equal(first[frame - 8:frame], second[frame - 8:frame])
            np.testing.assert_allclose(first[frame] - second[frame], pair["amplitude"], rtol=0, atol=6e-14)

    def test_unknown_generated_spec_rejected(self):
        spec = deepcopy(audit.specifications()[0])
        spec["seed"] = 2
        with self.assertRaises(ValueError):
            audit.generated_input(spec)

    def test_independent_measurement_errors_and_missing_overlap(self):
        history = np.column_stack((np.zeros(8), np.ones(8), np.full(8, np.nan)))
        diagnostic = audit.scalar_diagnostic(history)
        result = audit.measurement(np.asarray([3., np.nan, np.nan]), diagnostic)
        self.assertEqual(result["scorable_points"], 1)
        self.assertEqual(result["prior_unavailable_points"], 1)
        self.assertEqual(result["current_nonfinite_points"], 2)
        self.assertEqual(result["unscorable_points"], 2)
        self.assertFalse(result["complete_window"])
        self.assertEqual(result["arms"]["median8"]["absolute_error_sum_dn"], 3)
        self.assertEqual(result["arms"]["median3"]["squared_error_sum_dn2"], 9)

    def test_missing_case_denominators_count_union_not_double_count(self):
        case = self.case("missing_window")
        rows = []
        for frame in range(8, 64):
            d = audit.scalar_diagnostic(case["values"][frame - 8:frame])
            rows.append(dict(measurement=audit.measurement(case["values"][frame], d),
                             diagnostic=audit.diagnostic_summary(d)))
        result = audit.aggregate(rows)
        self.assertEqual(result["response_windows"], 56)
        self.assertEqual(result["total_point_opportunities"], 8064)
        self.assertEqual(result["prior_unavailable_points"], 80)
        self.assertEqual(result["current_nonfinite_points"], 24)
        self.assertEqual(result["unscorable_points"], 88)
        self.assertEqual(result["scorable_points"], 7976)
        self.assertEqual(result["complete_windows"], 45)

    def test_empty_measurement_is_not_complete(self):
        d = audit.scalar_diagnostic(np.empty((8, 0)))
        result = audit.measurement(np.empty(0), d)
        self.assertFalse(result["complete_window"])
        self.assertIsNone(result["arms"]["median8"]["mae_dn"])

    def test_exact_array_rejects_tampering_shape_dtype_and_nonfinite_swap(self):
        expected = np.asarray([1., np.nan])
        audit.exact_array(expected, expected.copy(), "good")
        for actual in (np.asarray([2., np.nan]), np.asarray([1., np.inf]),
                       np.asarray([1., np.nan], dtype=np.float32), expected.reshape(1, 2)):
            with self.assertRaises(ValueError):
                audit.exact_array(expected, actual, "bad")

    def test_comparison_rejects_missing_extra_bool_and_large_numeric_change(self):
        audit.compare({"count": 3, "error": 1.25}, {"count": 3, "error": 1.25 + 1e-12})
        for actual in ({"count": 3}, {"count": 3, "error": 1.25, "extra": 1},
                       {"count": True, "error": 1.25}, {"count": 3, "error": 1.3}):
            with self.assertRaises(ValueError):
                audit.compare({"count": 3, "error": 1.25}, actual)

    def test_allowlist_rejects_unrelated_source_and_case_paths(self):
        for path in (audit.ROOT / "scripts/run_accuracy_v50_prediction.py",
                     audit.ROOT / "results/tiny_target/accuracy_v50_20260926/README.md",
                     Path("/tmp/accuracy_v51_fake.py")):
            with self.assertRaises(ValueError):
                audit.allow_source(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "known_input.npz"
            path.write_bytes(b"generated fixture")
            self.assertEqual(audit.allow_case_file(root, path, path.name), path)
            with self.assertRaises(ValueError):
                audit.allow_case_file(root, path, "another_input.npz")
            link = root / "link.npz"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                audit.canonical_file(link)

    def test_chronology_rejects_reversal_missing_timezone_and_stage_claim(self):
        start = datetime(2026, 9, 26, tzinfo=timezone.utc)
        stages = [dict(created_at_utc=(start + timedelta(seconds=i)).isoformat(), completed=True)
                  for i in range(4)]
        stages[0]["before_generation_and_scoring"] = True
        stages[1]["response_scoring_started"] = False
        audit.chronology(*stages)
        wrong = deepcopy(stages)
        wrong[2]["created_at_utc"] = (start - timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):
            audit.chronology(*wrong)
        wrong = deepcopy(stages)
        wrong[1]["response_scoring_started"] = True
        with self.assertRaises(ValueError):
            audit.chronology(*wrong)
        wrong = deepcopy(stages)
        wrong[0]["created_at_utc"] = "2026-09-26T00:00:00"
        with self.assertRaises(ValueError):
            audit.chronology(*wrong)

    def test_empty_aggregates_keep_unknown_metrics_and_zero_counts(self):
        result = audit.aggregate([])
        self.assertEqual(result["response_windows"], 0)
        self.assertEqual(result["total_point_opportunities"], 0)
        self.assertIsNone(result["observed_return_fraction_mean"])
        self.assertIsNone(result["arms"]["median8"]["conditional_point_mae_dn"])


if __name__ == "__main__":
    unittest.main()
