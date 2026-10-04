"""Generated-only adversarial tests for the independent V53 offset auditor."""

import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_accuracy_v53_offset as audit
import audit_accuracy_v52_crossfit as inherited
import accuracy_v52_benchmark as benchmark
import accuracy_v53_offset as model
import run_accuracy_v53_offset as runner


def fixture(family="stable", background="textured", noise=1, split="left_right"):
    spec = next(row for row in benchmark.specifications() if row["family"] == family
                and row["background"] == background and row["noise_level"] == noise)
    data = benchmark.generate_case(spec)
    row = audit.plain(dict(input_id=spec["case_id"], kind="synthetic", metadata=spec,
        **{name: data[name] for name in ("points_xy", "median8", "median3", "current",
            "clean_current_background", "contamination_mask")}))
    forecast = audit.plain(model.crossfit(data["points_xy"], data["median8"], data["current"], split))
    return row, forecast


def resign(forecast):
    for fit in forecast["fits"]["median_offset"].values():
        fit["model_sha256"] = audit.digest(fit, ("model_sha256",))
    forecast["crossfit_sha256"] = audit.digest(forecast, ("crossfit_sha256",))


def simple_fit(current, slow=None):
    xy = np.asarray(audit.POINTS[:len(current)], dtype=float)
    current = np.asarray(current, dtype=float)
    slow = np.zeros(len(current)) if slow is None else np.asarray(slow, dtype=float)
    return xy, slow, current, audit.plain(model.fit(xy, slow, current))


class ExactMedianAuditTests(unittest.TestCase):
    def test_constants_match_frozen_contract(self):
        self.assertEqual(model.model_constants(), audit.constants())
        self.assertEqual(("median_offset",), audit.LOSSES)

    def test_odd_exact_median_interval_counts_and_l1_certificate(self):
        xy, slow, current, result = simple_fit([1., 1., 2., 3., 100.])
        audit.check_fit(result, xy, slow, current)
        self.assertEqual(2., result["offset_dn"])
        self.assertEqual([2., 2.], result["median_interval_dn"])
        self.assertEqual(dict(below=2, above=2, tied=1), result["residual_order_counts"])
        self.assertEqual([-.2, .2], result["subgradient_interval"])

    def test_even_interval_has_fixed_midpoint_not_arbitrary_minimizer(self):
        xy, slow, current, result = simple_fit([-1., 0., 1., 100.])
        audit.check_fit(result, xy, slow, current)
        self.assertEqual(.5, result["offset_dn"])
        self.assertEqual([0., 1.], result["median_interval_dn"])
        self.assertEqual([0., 0.], result["subgradient_interval"])
        self.assertEqual(25.5, result["objective_mae_dn"])
        result["offset_dn"] = 0.
        result["model_sha256"] = audit.digest(result, ("model_sha256",))
        with self.assertRaises(ValueError):
            audit.check_fit(result, xy, slow, current)

    def test_subnormal_midpoint_correctly_rounded_without_early_underflow(self):
        tiny = float(np.nextafter(0., 1.))
        for residuals in ([tiny] * 4, [tiny, tiny, 2 * tiny, 2 * tiny], [-tiny, -tiny, tiny, tiny]):
            xy, slow, current, result = simple_fit(residuals)
            audit.check_fit(result, xy, slow, current)
            ordered = sorted(residuals)
            expected = float((Fraction(ordered[1]) + Fraction(ordered[2])) / 2)
            self.assertEqual(expected, result["offset_dn"])

    def test_extreme_finite_midpoint_and_objective_avoid_sum_overflow(self):
        maximum = np.finfo(float).max
        for residuals in ([maximum] * 4, [-maximum, -maximum, maximum, maximum],
                          [-maximum / 2, -maximum / 2, maximum / 2, maximum / 2]):
            xy, slow, current, result = simple_fit(residuals)
            audit.check_fit(result, xy, slow, current)
            self.assertTrue(result["available"])
            self.assertTrue(np.isfinite(result["offset_dn"]))
            self.assertTrue(np.isfinite(result["objective_mae_dn"]))

    def test_subtraction_overflow_invalidates_whole_fit(self):
        maximum = np.finfo(float).max
        xy, slow, current, result = simple_fit([maximum, 1., 2., 3.], [-maximum, 0., 0., 0.])
        audit.check_fit(result, xy, slow, current)
        self.assertFalse(result["available"])
        self.assertEqual(4, result["training_used_count"])
        self.assertEqual("nonfinite_training_residual_arithmetic", result["unavailable_reason"])
        self.assertIsNone(result["candidate_offset_dn"])

    def test_objective_deviation_overflow_retains_candidate_without_prediction(self):
        maximum = np.finfo(float).max
        xy, slow, current, result = simple_fit([-maximum, maximum, maximum, maximum])
        audit.check_fit(result, xy, slow, current)
        self.assertFalse(result["available"])
        self.assertEqual("nonfinite_objective_arithmetic", result["unavailable_reason"])
        self.assertEqual(maximum, result["candidate_offset_dn"])
        self.assertIsNone(result["offset_dn"])
        changed = deepcopy(result)
        changed["candidate_offset_dn"] = 0.
        changed["model_sha256"] = audit.digest(changed, ("model_sha256",))
        with self.assertRaises(ValueError):
            audit.check_fit(changed, xy, slow, current)

    def test_fewer_than_four_finite_rows_remain_unavailable(self):
        xy, slow, current, result = simple_fit([1., 2., 3., np.nan, np.inf])
        audit.check_fit(result, xy, slow, current)
        self.assertEqual(3, result["training_used_count"])
        self.assertEqual("insufficient_finite_training_rows", result["unavailable_reason"])
        self.assertEqual([True, True, True, False, False], result["training_used_mask"])

    def test_empty_fold_is_preserved_unavailable(self):
        xy = np.empty((0, 2))
        vector = np.empty(0)
        result = audit.plain(model.fit(xy, vector, vector))
        audit.check_fit(result, xy, vector, vector)
        self.assertEqual(0, result["training_count"])

    def test_every_fit_certificate_field_is_checked_after_resigning(self):
        xy, slow, current, initial = simple_fit([1., 2., 3., 40., 50.])
        changes = {
            "offset_dn": 4., "candidate_offset_dn": 4., "median_interval_dn": [2., 3.],
            "subgradient_interval": [1., 1.], "residual_order_counts": dict(below=0, above=0, tied=5),
            "objective_mae_dn": 100., "training_used_count": 4,
            "training_input_sha256": "unbound", "training_used_mask": [False] * 5,
            "available": False, "unavailable_reason": "insufficient_finite_training_rows",
        }
        for field, value in changes.items():
            changed = deepcopy(initial)
            changed[field] = value
            changed["model_sha256"] = audit.digest(changed, ("model_sha256",))
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.check_fit(changed, xy, slow, current)


class ForecastAndInputAuditTests(unittest.TestCase):
    def test_every_generated_case_both_splits_all_fits_predictions_scores(self):
        for spec in benchmark.specifications():
            for split in audit.SPLITS:
                row, forecast = fixture(spec["family"], spec["background"], spec["noise_level"], split)
                with self.subTest(case=spec["case_id"], split=split):
                    audit.compare(audit.generated_input(spec), row, rtol=0, atol=1e-12)
                    audit.check_forecast(row, forecast)
                    audit.compare(runner.evaluate(row, forecast), audit.expected_score(row, forecast))

    def test_predictions_folds_masks_and_reasons_cannot_be_changed_after_resigning(self):
        row, original = fixture()
        changes = ("prediction", "fold", "mask", "reason", "constant", "metadata", "loss")
        for name in changes:
            changed = deepcopy(original)
            prediction = changed["predictions"]["median_offset"]
            if name == "prediction":
                prediction["values"][0] += 1.
            elif name == "fold":
                changed["fold_id"][0] = 1 - changed["fold_id"][0]
            elif name == "mask":
                prediction["available"][0] = False
            elif name == "reason":
                prediction["unavailable_reasons"][0] = "pretend_missing"
            elif name == "constant":
                changed["constants"]["fixed_gain"] = 2.
            elif name == "metadata":
                changed["metadata"]["core_pixels_accepted"] = True
            else:
                changed["fits"]["unexpected"] = {}
            resign(changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.check_forecast(row, changed)

    def test_unresigned_forecast_tamper_fails_fingerprint(self):
        row, forecast = fixture()
        forecast["total_count"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            audit.check_forecast(row, forecast)

    def test_extra_forecast_fields_rejected_even_after_resigning(self):
        row, forecast = fixture()
        forecast["unapproved_field"] = "unused but forbidden"
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "top-level schema"):
            audit.check_forecast(row, forecast)

    def test_boolean_forecast_count_or_schema_is_not_accepted_as_integer(self):
        row, forecast = fixture()
        forecast["schema_version"] = True
        resign(forecast)
        with self.assertRaises(ValueError):
            audit.check_forecast(row, forecast)

    def test_changed_training_response_breaks_bound_model(self):
        row, forecast = fixture()
        row["current"][0] += 10
        with self.assertRaises(ValueError):
            audit.check_forecast(row, forecast)

    def test_heldout_response_only_changes_reverse_direction(self):
        for split in audit.SPLITS:
            for fold in (0, 1):
                row, before = fixture(split=split)
                points = np.asarray(row["points_xy"])
                slow, current = (np.asarray(row[key], dtype=float) for key in ("median8", "current"))
                selected = np.asarray(before["fold_id"]) == fold
                current[selected] += 1000
                after = audit.plain(model.crossfit(points, slow, current, split))
                row["current"] = current.tolist()
                audit.check_forecast(row, after)
                self.assertEqual(before["fits"]["median_offset"][str(fold)], after["fits"]["median_offset"][str(fold)])
                for index in np.flatnonzero(selected):
                    self.assertEqual(before["predictions"]["median_offset"]["values"][index],
                                     after["predictions"]["median_offset"]["values"][index])

    def test_labels_truth_and_references_do_not_enter_fit_audit(self):
        row, forecast = fixture()
        row["metadata"] = {"family": "unrelated"}
        row["clean_current_background"] = [-1e9] * len(row["current"])
        row["contamination_mask"] = [True] * len(row["current"])
        row["reference_context"] = {"false_hint": "change offset"}
        audit.check_forecast(row, forecast)

    def test_prediction_addition_overflow_remains_pointwise_unknown(self):
        points = np.asarray(audit.POINTS)
        maximum = np.finfo(float).max
        right = points[:, 0] >= 64
        slow = np.zeros(len(points)); slow[~right] = maximum
        current = slow.copy(); current[right] = maximum
        forecast = audit.plain(model.crossfit(points, slow, current))
        row = dict(points_xy=points.tolist(), median8=slow.tolist(), current=current.tolist())
        audit.check_forecast(row, forecast)
        prediction = forecast["predictions"]["median_offset"]
        self.assertTrue(all(prediction["values"][index] is None for index in np.flatnonzero(~right)))
        self.assertTrue(all(prediction["unavailable_reasons"][index] == "nonfinite_prediction_arithmetic"
                            for index in np.flatnonzero(~right)))

    def test_core_offgrid_and_duplicate_points_rejected(self):
        for points in ([[64, 64]], [[9, 8]], [[8, 8], [8, 8]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                audit.geometry(points)


class MetricAndBindingAuditTests(unittest.TestCase):
    def test_all_grouped_summaries_and_complete_frame_unknowns(self):
        rows, states = [], []
        for index, family in enumerate(("stable", "missing_current_left", "all_current_missing")):
            for split in audit.SPLITS:
                row, forecast = fixture(family=family, split=split)
                rows.append(runner.evaluate(row, forecast))
                real_row = deepcopy(row)
                real_row.update(kind="real", input_id="real_" + str(index), metadata=dict(
                    state_key=["0029", index + 10, 0, "test"], partition="calibration", available=True, reasons=[]))
                rows.append(runner.evaluate(real_row, forecast))
            states.append(dict(state_key=["0029", index + 10, 0, "test"], partition="calibration", v50_status="background_measured"))
        states.append(dict(state_key=["0029", 50, 0, "unknown"], partition="evaluation", v50_status="history_unknown"))
        actual = runner.summarize(rows, states)
        expected = audit.expected_summary(rows, states, actual["created_at_utc"])
        audit.compare(actual, expected)
        self.assertEqual(2, expected["splits"]["left_right"]["real_groups"]["calibration_0029"]["arms"]["median_offset"]["incomplete_archived_frame_count"])
        self.assertEqual(1, expected["splits"]["left_right"]["real_groups"]["evaluation_0029"]["frames_without_scored_archives"])

    def test_removed_archive_does_not_disappear_from_frame_denominator(self):
        states = [dict(state_key=["0029", 10, 0, "test"], partition="calibration", v50_status="background_measured")]
        with self.assertRaisesRegex(ValueError, "archived state"):
            audit.expected_aggregate([], states)

    def test_changed_metric_and_scored_mask_are_detected(self):
        row, forecast = fixture()
        expected = audit.expected_score(row, forecast)
        changes = ("count", "sum", "mask", "fold", "complete")
        for name in changes:
            actual = deepcopy(expected)
            arm = actual["arms"]["median_offset"]
            if name == "count":
                arm["scored_count"] += 1
            elif name == "sum":
                arm["metrics"]["corrected"]["absolute_sum_dn"] += 1
            elif name == "mask":
                arm["scored_indices"].pop()
            elif name == "fold":
                arm["fold_fits"]["0"]["offset_dn"] += 1
            else:
                arm["complete"] = False
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.compare(expected, actual)

    def test_reference_context_exact_comparison_detects_tiny_mutation(self):
        old = {"references": [{"id": "a", "value": .1}], "alternatives": [3], "misses": [9]}
        changed = deepcopy(old)
        changed["references"][0]["value"] = np.nextafter(.1, 1.).item()
        with self.assertRaises(ValueError):
            audit.compare(old, changed, "Reference context", rtol=0, atol=0)

    def test_only_synthetic_input_values_allow_expression_rounding(self):
        synthetic = [{"kind": "synthetic", "current": [.1]}]
        real = [{"kind": "real", "current": [.1]}]
        saved = deepcopy(synthetic + real)
        saved[0]["current"][0] = np.nextafter(.1, 1.).item()
        audit.check_input_rows(saved, synthetic, real)
        saved[1]["current"][0] = np.nextafter(.1, 1.).item()
        with self.assertRaisesRegex(ValueError, "exact inherited real"):
            audit.check_input_rows(saved, synthetic, real)

    def test_input_row_omission_rejected(self):
        with self.assertRaisesRegex(ValueError, "Input row count"):
            audit.check_input_rows([], [{"kind": "synthetic"}], [])

    def test_hash_checked_before_decode_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "fixture.json"
            path.write_text("not JSON")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                audit.checked_read(path, "wrong", {})
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "Canonical"):
                audit.checked_read(link, hashlib.sha256(path.read_bytes()).hexdigest(), {})

    def test_inherited_sources_are_independently_hard_pinned(self):
        self.assertEqual(6, len(audit.INHERITED_SHA256))
        self.assertEqual(audit.INHERITED_SHA256, runner.INHERITED_SHA256)
        audit.verify_inherited_sources(audit.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "Pinned inherited"):
                audit.verify_inherited_sources(root)

    def test_v52_loss_globals_are_unchanged(self):
        self.assertEqual(("ols", "huber"), inherited.LOSSES)
        row, forecast = fixture()
        audit.check_forecast(row, forecast)
        audit.expected_score(row, forecast)
        self.assertEqual(("ols", "huber"), inherited.LOSSES)

    def test_imports_only_independent_helpers_not_producer_modules(self):
        modules = []
        for node in ast.walk(ast.parse(Path(audit.__file__).read_text())):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        self.assertEqual(["audit_accuracy_v52_crossfit"], [name for name in modules if "accuracy_v" in name])
        self.assertFalse(set(modules) & {"cv2", "PIL", "imageio", "subprocess"})

    def test_closed_allowlists_and_wrong_run_location(self):
        self.assertEqual(13, len(audit.SOURCE_NAMES))
        self.assertEqual(set(runner.SOURCE_NAMES), set(audit.SOURCE_NAMES))
        self.assertEqual(7, len(audit.OLD_NAMES))
        self.assertEqual(8, len(audit.RUN_NAMES))
        with patch.object(audit, "raw_file", side_effect=AssertionError("should not read artifacts")):
            with self.assertRaisesRegex(ValueError, "Canonical immediate"):
                audit.audit_run(Path("/tmp/unrelated"))

    def test_unknown_values_fingerprint_as_json_null(self):
        self.assertEqual(audit.digest(dict(x=np.array([np.nan, np.inf, -np.inf]))),
                         audit.digest(dict(x=[None, None, None])))


if __name__ == "__main__":
    unittest.main()
