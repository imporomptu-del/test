"""Generated fixtures only for the independent V52 auditor and tamper checks."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_accuracy_v52_crossfit as audit
import accuracy_v52_benchmark as benchmark
import accuracy_v52_crossfit as model
import run_accuracy_v52_crossfit as runner
from test_accuracy_v52_real_scope import fixture as old_fixture


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
    for loss in audit.LOSSES:
        for fit in forecast["fits"][loss].values():
            fit["model_sha256"] = audit.digest(fit, ("model_sha256",))
    forecast["crossfit_sha256"] = audit.digest(forecast, ("crossfit_sha256",))


class IndependentInputsTests(unittest.TestCase):
    def test_specifications_match_fixed_design(self):
        self.assertEqual(benchmark.specifications(), audit.specifications())
        self.assertEqual(120, len(audit.specifications()))

    def test_all_analytic_input_rows_match_generated_formulas(self):
        for spec in audit.specifications():
            data = benchmark.generate_case(spec)
            expected = audit.plain(dict(input_id=spec["case_id"], kind="synthetic", metadata=spec,
                **{name: data[name] for name in ("points_xy", "median8", "median3", "current",
                    "clean_current_background", "contamination_mask")}))
            audit.compare(expected, audit.generated_input(spec), spec["case_id"], rtol=0, atol=1e-12)

    def test_unknown_specification_rejected(self):
        spec = audit.specifications()[0]
        spec["noise_level"] = 10
        with self.assertRaises(ValueError):
            audit.generated_input(spec)

    def test_independent_real_join_preserves_missing_packet_scope(self):
        rows, counts = audit.real_inputs(old_fixture())
        self.assertEqual(3, len(rows))
        self.assertEqual(9, counts["guard_point_opportunities"])
        self.assertEqual(6, counts["available_current_points"])
        self.assertEqual([None] * 3, rows[1]["current"])
        self.assertEqual([4.] * 3, rows[0]["current"])
        self.assertEqual([2., 3., 4.], rows[0]["median3"])

    def test_real_join_rejects_residual_inconsistency(self):
        docs = old_fixture()
        docs["calibration_measurements.jsonl"][0]["measurement"]["arms"]["median3_temporal_scale"]["residuals"][0] += 1
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            audit.real_inputs(docs)

    def test_real_join_rejects_missing_or_duplicate_membership(self):
        for name in ("state_results.jsonl", "calibration_forecasts.jsonl", "evaluation_measurements.jsonl"):
            docs = old_fixture()
            docs[name].append(deepcopy(docs[name][0]))
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.real_inputs(docs)

    def test_real_join_rejects_changed_forecast_fingerprint(self):
        docs = old_fixture()
        docs["calibration_forecasts.jsonl"][0]["forecast"]["arms"]["median8_unit_scale"]["prediction"][0] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            audit.real_inputs(docs)

    def test_real_join_rejects_duplicate_assignment_and_changed_source_flag(self):
        docs = old_fixture()
        docs["freeze.json"]["scope"]["assignments"].append(deepcopy(docs["freeze.json"]["scope"]["assignments"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate original assignments"):
            audit.real_inputs(docs)
        docs = old_fixture()
        docs["state_results.jsonl"][0]["source_scores_and_original_detections_unchanged"] = False
        with self.assertRaisesRegex(ValueError, "source decisions"):
            audit.real_inputs(docs)


class IndependentNumericsTests(unittest.TestCase):
    def test_all_generated_fits_predictions_and_scores(self):
        # Full fixed analytic design, no real artifacts or output-based changes.
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            row = audit.plain(dict(input_id=spec["case_id"], kind="synthetic", metadata=spec,
                **{name: data[name] for name in ("points_xy", "median8", "median3", "current",
                    "clean_current_background", "contamination_mask")}))
            for split in audit.SPLITS:
                with self.subTest(case=spec["case_id"], split=split):
                    forecast = audit.plain(model.crossfit(data["points_xy"], data["median8"], data["current"], split))
                    audit.check_forecast(row, forecast)
                    audit.compare(runner.evaluate(row, forecast), audit.expected_score(row, forecast), "independent score")

    def test_changed_prediction_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["predictions"]["huber"]["values"][0] += 1
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "prediction"):
            audit.check_forecast(row, forecast)

    def test_heldout_current_tamper_cannot_be_hidden_in_training_binding(self):
        row, forecast = fixture()
        row["current"][0] += 100
        with self.assertRaisesRegex(ValueError, "Training input"):
            audit.check_forecast(row, forecast)

    def test_each_heldout_fold_has_unchanged_training_model_when_its_response_changes(self):
        row, first = fixture()
        points = np.asarray(row["points_xy"])
        for split in audit.SPLITS:
            for fold in (0, 1):
                data = deepcopy(row)
                slow = np.asarray(row["median8"], dtype=float)
                current = np.asarray(row["current"], dtype=float)
                before = audit.plain(model.crossfit(points, slow, current, split))
                selected = np.asarray(before["fold_id"]) == fold
                current[selected] += 1000
                after = audit.plain(model.crossfit(points, slow, current, split))
                data["current"] = current.tolist()
                audit.check_forecast(data, after)
                for loss in audit.LOSSES:
                    self.assertEqual(before["fits"][loss][str(fold)], after["fits"][loss][str(fold)])
                    for index in np.flatnonzero(selected):
                        self.assertEqual(before["predictions"][loss]["values"][index], after["predictions"][loss]["values"][index])

    def test_wrong_train_center_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["fits"]["huber"]["0"]["center"] += 1
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "center"):
            audit.check_forecast(row, forecast)

    def test_wrong_objective_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["fits"]["huber"]["0"]["objective"] += 1
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "objective"):
            audit.check_forecast(row, forecast)

    def test_wrong_kkt_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["fits"]["huber"]["0"]["kkt_residual_dn"] = 100.
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "KKT"):
            audit.check_forecast(row, forecast)

    def test_unavailable_coefficients_must_remain_unknown(self):
        row, forecast = fixture(background="constant", noise=0)
        forecast["fits"]["ols"]["0"]["coefficients_conditioned"] = [0., 96., 0., 0.]
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "Unavailable conditioned"):
            audit.check_forecast(row, forecast)

    def test_unavailable_reason_must_match_independent_calculation(self):
        row, forecast = fixture(background="constant", noise=0)
        forecast["fits"]["ols"]["0"]["unavailable_reason"] = "insufficient_finite_training_rows"
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "Rank failure"):
            audit.check_forecast(row, forecast)

    def test_successful_solve_cannot_be_suppressed_as_unavailable(self):
        row, forecast = fixture()
        fit = forecast["fits"]["huber"]["0"]
        self.assertTrue(fit["available"])
        fit.update(available=False, converged=False, gain_bound_active=None,
                   coefficients_conditioned=[None] * 4, coefficients_physical=[None] * 4)
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "Availability differs"):
            audit.check_forecast(row, forecast)

    def test_weighted_failure_retains_and_checks_last_successful_iterate(self):
        xy = np.asarray(audit.POINTS, dtype=float)
        slow = np.zeros(len(xy))
        current = np.full(len(xy), 100.)
        for point, sign in zip(((8, 8), (120, 8), (8, 120), (120, 120)), (1, -1, -1, 1)):
            index = np.flatnonzero(np.all(xy == point, axis=1))[0]
            slow[index] = 1
            current[index] += sign * 1e16
        fit = audit.plain(model.fit(xy, slow, current, "huber"))
        self.assertEqual("weighted_ill_conditioned_design", fit["unavailable_reason"])
        self.assertEqual(0, fit["iterations"])
        self.assertTrue(all(value is not None for value in fit["last_candidate_coefficients_conditioned"]))
        audit.check_fit(fit, xy, slow, current, "huber")
        for field in ("objective", "kkt_residual_dn", "last_candidate_coefficients_conditioned"):
            changed = deepcopy(fit)
            if field == "last_candidate_coefficients_conditioned":
                changed[field][0] += 100
            else:
                changed[field] += abs(changed[field]) + 100
            changed["model_sha256"] = audit.digest(changed, ("model_sha256",))
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.check_fit(changed, xy, slow, current, "huber")

    def test_wrong_availability_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["predictions"]["huber"]["available"][0] = False
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "predictions"):
            audit.check_forecast(row, forecast)

    def test_wrong_fold_detected_after_resigning(self):
        row, forecast = fixture()
        forecast["fold_id"][0] = 1 - forecast["fold_id"][0]
        resign(forecast)
        with self.assertRaisesRegex(ValueError, "fold assignment"):
            audit.check_forecast(row, forecast)

    def test_negative_gain_is_rejected_even_if_resigned(self):
        row, forecast = fixture()
        forecast["fits"]["ols"]["0"]["coefficients_conditioned"][0] = -1
        resign(forecast)
        with self.assertRaises(ValueError):
            audit.check_forecast(row, forecast)

    def test_source_core_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "core"):
            audit.geometry([[64, 64]])

    def test_duplicate_guard_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.geometry([[8, 8], [8, 8]])

    def test_convex_objective_and_gain_boundary_kkt_known_values(self):
        design = np.eye(4)
        response = np.array([-1., 2., 3., 4.])
        coefficient = np.array([0., 2., 3., 4.])
        objective, kkt, _ = audit.optimality(design, response, coefficient, "ols")
        self.assertEqual(.125, objective)
        self.assertEqual(0., kkt)


class IndependentMetricTests(unittest.TestCase):
    def test_error_metrics_hand_calculated(self):
        result = audit.error_metrics([-2, 0, 4])
        self.assertEqual(3, result["count"])
        self.assertEqual(6., result["absolute_sum_dn"])
        self.assertEqual(20., result["squared_sum_dn2"])
        self.assertEqual(2., result["mae_dn"])
        self.assertEqual(2., result["median_absolute_error_dn"])
        self.assertAlmostEqual(3.6, result["p90_absolute_error_dn"])

    def test_empty_metric_and_distribution_keep_unknowns(self):
        result = audit.error_metrics([])
        self.assertEqual(0, result["count"])
        self.assertIsNone(result["mae_dn"])
        self.assertIsNone(result["max_absolute_error_dn"])
        self.assertEqual(dict(count=0, mean=None, median=None, p90=None, max=None), audit.distribution([]))

    def test_grouped_summary_and_complete_frame_accounting(self):
        rows = []
        states = []
        for index, family in enumerate(("stable", "missing_current_left", "all_current_missing")):
            for split in audit.SPLITS:
                row, forecast = fixture(family=family, split=split)
                rows.append(runner.evaluate(row, forecast))
                real_row = deepcopy(row)
                real_row["kind"] = "real"
                real_row["input_id"] = "real_" + str(index)
                real_row["metadata"] = dict(state_key=["0029", index + 10, 0, "test"], partition="calibration", available=True, reasons=[])
                rows.append(runner.evaluate(real_row, forecast))
            states.append(dict(state_key=["0029", index + 10, 0, "test"], partition="calibration", v50_status="background_measured"))
        states.append(dict(state_key=["0029", 50, 0, "unknown"], partition="evaluation", v50_status="history_unknown"))
        actual = runner.summarize(rows, states)
        expected = audit.expected_summary(rows, states, actual["created_at_utc"])
        audit.compare(actual, expected)
        self.assertEqual(2, expected["splits"]["left_right"]["real_groups"]["calibration_0029"]["arms"]["huber"]["incomplete_archived_frame_count"])
        self.assertEqual(1, expected["splits"]["left_right"]["real_groups"]["evaluation_0029"]["frames_without_scored_archives"])

    def test_matched_controls_share_identical_scored_indices(self):
        row, forecast = fixture("missing_prior")
        score = audit.expected_score(row, forecast)
        for loss in audit.LOSSES:
            self.assertEqual(score["arms"][loss]["scored_count"], len(score["arms"][loss]["scored_indices"]))
            self.assertEqual({score["arms"][loss]["scored_count"]}, {entry["count"] for entry in score["arms"][loss]["metrics"].values()})


class IntegrityTests(unittest.TestCase):
    def test_fingerprints_canonicalize_nonfinite_to_null(self):
        self.assertEqual(audit.digest({"x": np.array([np.nan, np.inf, -np.inf])}),
                         audit.digest({"x": [None, None, None]}))

    def test_comparison_checks_structure_types_and_values(self):
        for expected, actual in (({"a": 1}, {"b": 1}), ([1], []), (True, 1), (1., 2.), (None, 0)):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                audit.compare(expected, actual)

    def test_checked_read_verifies_bytes_before_json_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "fixture.json"
            path.write_text("not JSON")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                audit.checked_read(path, "wrong", {})

    def test_checked_read_jsonl_and_reject_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "fixture.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n')
            bindings = {}
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual([{"a": 1}, {"a": 2}], audit.checked_read(path, expected_hash, bindings))
            link = root / "link.jsonl"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "Canonical"):
                audit.checked_read(link, expected_hash, {})

    def test_auditor_imports_no_implementation_modules(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module)
        self.assertFalse(any("accuracy_v52" in name for name in modules))
        self.assertFalse(any(name in ("cv2", "PIL", "imageio", "subprocess") for name in modules))

    def test_closed_file_allowlist(self):
        self.assertEqual(11, len(audit.SOURCE_NAMES))
        self.assertEqual(7, len(audit.OLD_NAMES))
        self.assertEqual(8, len(audit.RUN_NAMES))
        self.assertTrue(all(name.endswith((".json", ".jsonl")) for name in audit.OLD_NAMES + audit.RUN_NAMES))

    def test_wrong_run_location_rejected_before_read(self):
        with patch.object(audit, "raw_file", side_effect=AssertionError("should not read")):
            with self.assertRaisesRegex(ValueError, "Canonical immediate"):
                audit.audit_run(Path("/tmp/unrelated"))


if __name__ == "__main__":
    unittest.main()
