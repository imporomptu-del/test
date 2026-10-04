from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import run_accuracy_v51_history as runner
from accuracy_v51_history_diagnostic import diagnose, diagnostic_fingerprint


def case_fixture(history=None, current=0., phase="baseline"):
    """Tiny runner fixture, not a benchmark evaluation or a camera sample."""
    if history is None:
        history = list(range(8))
    values = np.zeros((64, 144), dtype=np.float64)
    values[:8] = np.asarray(history, dtype=float)[:, None]
    values[8] = current
    return dict(values=values, event_active=np.zeros(64, dtype=bool),
        response_phase=np.full(64, phase),
        history_has_prior_event=np.zeros(64, dtype=bool),
        signal_delta=np.zeros_like(values),
        affected_points_mask=np.zeros_like(values, dtype=bool),
        missing_mask=np.zeros_like(values, dtype=bool),
        points_xy=np.zeros((144, 2), dtype=np.int64))


def specification(name, family="fixture"):
    return dict(case_id=name, family=family, noise_level=0)


class HistoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def prepared(self, cases=None, frames=(8,)):
        if cases is None:
            cases = {"fixture": case_fixture()}
        output = self.root/"prepared"
        output.mkdir()
        specs = [specification(name) for name in cases]
        with patch.object(runner, "RESPONSE_FRAMES", frames), patch.object(
                runner, "generate_case", side_effect=lambda spec: cases[spec["case_id"]]):
            manifest = runner.prepare(specs, output)
        return output, manifest

    def score_prepared(self, output, manifest, twins=None, frames=(8,)):
        with patch.object(runner, "RESPONSE_FRAMES", frames):
            return runner.score(manifest, {"twin_pairs": twins or []}, output)

    def twin_fixture(self, step_current=8., pulse_current=0., different_history=False):
        history = [0.]*5+[8.]*3
        pulse_history = list(history)
        if different_history:
            pulse_history[0] = 1
        output, manifest = self.prepared({
            "step": case_fixture(history, step_current),
            "pulse": case_fixture(pulse_history, pulse_current),
        })
        pair = dict(step_case_id="step", pulse_case_id="pulse", response_frame=8,
            duration=3, amplitude=8, response_step_minus_pulse_analytic=8)
        return output, manifest, pair

    def test_prepare_never_scores_and_passes_only_eight_prior_values(self):
        output = self.root/"prepare"
        output.mkdir()
        fixture = case_fixture(current=12345.)
        fixture["signal_delta"][:] = 54321.
        calls = []
        def observe(*args, **kwargs):
            self.assertEqual(len(args), 1)
            self.assertEqual(kwargs, {})
            self.assertEqual(args[0].shape, (8, 144))
            calls.append(args[0].copy())
            return diagnose(args[0])
        with patch.object(runner, "RESPONSE_FRAMES", (8,)), \
             patch.object(runner, "generate_case", return_value=fixture), \
             patch.object(runner, "diagnose", side_effect=observe), \
             patch.object(runner, "measure", side_effect=AssertionError("Scored during preparation")):
            manifest = runner.prepare([specification("fixture")], output)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], fixture["values"][:8])
        self.assertFalse(manifest["response_scoring_started"])
        self.assertEqual(manifest["response_window_count"], 1)
        self.assertTrue((output/"forecasts_frozen.json").is_file())

    def test_changed_response_truth_and_phase_do_not_change_saved_diagnostics(self):
        first = case_fixture(current=100., phase="first")
        second = case_fixture(current=-900., phase="second")
        second["signal_delta"][:] = 123
        second["event_active"][:] = True
        output, manifest = self.prepared({"first": first, "second": second})
        a, b = manifest["cases"]
        contexts = [runner.read_json(item["context_path"]) for item in (a, b)]
        arrays = [runner.load_npz(item["diagnostic_path"]) for item in (a, b)]
        restored = [runner.restore_diagnostic(array, context, 0)
                    for array, context in zip(arrays, contexts)]
        self.assertEqual(restored[0]["diagnostic_sha256"], restored[1]["diagnostic_sha256"])

    def test_prepare_saves_all_cases_and_windows_before_score(self):
        output, manifest = self.prepared({"a": case_fixture(), "b": case_fixture()}, frames=(8, 9, 10))
        self.assertEqual(manifest["case_count"], 2)
        self.assertEqual(manifest["response_window_count"], 6)
        self.assertEqual(len(manifest["files_sha256"]), 6)
        original = runner.measure
        count = []
        def observe(current, d):
            frozen = runner.read_json(output/"forecasts_frozen.json")
            self.assertEqual(frozen["response_window_count"], 6)
            runner.check_bindings(frozen["files_sha256"])
            count.append(1)
            return original(current, d)
        with patch.object(runner, "measure", side_effect=observe):
            summary = self.score_prepared(output, manifest, frames=(8, 9, 10))
        self.assertEqual(len(count), 6)
        self.assertEqual(summary["overall"]["response_windows"], 6)

    def test_prepare_rejects_unsafe_case_path_and_bad_schema(self):
        output = self.root/"prepare"
        output.mkdir()
        with self.assertRaises(ValueError):
            runner.prepare([specification("../escape")], output)
        for values in (np.zeros((8, 144)), np.zeros((64, 144), dtype=np.float32)):
            fixture = case_fixture()
            fixture["values"] = values
            with patch.object(runner, "generate_case", return_value=fixture):
                with self.assertRaises(ValueError):
                    runner.prepare([specification("fixture")], output)

    def test_restore_handles_tuple_to_json_list_and_undefined_nan_arrays(self):
        output, manifest = self.prepared({"constant": case_fixture([0.]*8)})
        entry = manifest["cases"][0]
        context = runner.read_json(entry["context_path"])
        self.assertIsInstance(context[0]["diagnostic"]["point_unavailable_reasons"], list)
        restored = runner.restore_diagnostic(runner.load_npz(entry["diagnostic_path"]), context, 0)
        self.assertEqual(diagnostic_fingerprint(restored), restored["diagnostic_sha256"])
        self.assertTrue(np.isnan(restored["return_fraction"]).all())

    def test_restore_rejects_changed_numeric_array(self):
        output, manifest = self.prepared()
        entry = manifest["cases"][0]
        arrays = dict(runner.load_npz(entry["diagnostic_path"]))
        arrays["slow8"] = arrays["slow8"].copy()
        arrays["slow8"][0, 0] += 1
        with self.assertRaises(ValueError):
            runner.restore_diagnostic(arrays, runner.read_json(entry["context_path"]), 0)

    def test_restore_rejects_changed_availability_or_metadata(self):
        output, manifest = self.prepared()
        entry = manifest["cases"][0]
        original = runner.read_json(entry["context_path"])
        arrays = runner.load_npz(entry["diagnostic_path"])
        for key, value in (("available_count", 0), ("metadata", {"changed": True})):
            context = deepcopy(original)
            context[0]["diagnostic"][key] = value
            with self.assertRaises(ValueError):
                runner.restore_diagnostic(arrays, context, 0)

    def test_measure_uses_identical_shared_mask_and_preserves_unknown_counts(self):
        history = np.zeros((8, 5))
        history[0, 1] = np.nan
        history[0, 3] = np.nan
        d = diagnose(history)
        measurement = runner.measure(np.asarray([1., 2., np.nan, np.nan, 4.]), d)
        self.assertEqual(measurement["total_points"], 5)
        self.assertEqual(measurement["prior_unavailable_points"], 2)
        self.assertEqual(measurement["current_nonfinite_points"], 2)
        self.assertEqual(measurement["scorable_points"], 2)
        self.assertEqual(measurement["unscorable_points"], 3)
        self.assertFalse(measurement["complete_window"])
        for arm in runner.ARMS:
            self.assertEqual(measurement["arms"][arm]["point_count"], 2)
            self.assertEqual(measurement["arms"][arm]["absolute_error_sum_dn"], 5)
            self.assertEqual(measurement["arms"][arm]["mae_dn"], 2.5)

    def test_measure_all_missing_has_none_errors_not_zero_accuracy(self):
        d = diagnose(np.zeros((8, 2)))
        result = runner.measure(np.asarray([np.nan, np.inf]), d)
        self.assertEqual(result["scorable_points"], 0)
        self.assertEqual(result["unscorable_points"], 2)
        self.assertFalse(result["complete_window"])
        for arm in result["arms"].values():
            self.assertIsNone(arm["mae_dn"])
            self.assertIsNone(arm["max_absolute_error_dn"])

    def test_measure_empty_is_not_a_complete_window(self):
        d = diagnose(np.empty((8, 0)))
        result = runner.measure(np.empty(0), d)
        self.assertFalse(result["complete_window"])
        summary = runner.aggregate([dict(measurement=result, diagnostic=runner.diagnostic_summary(d))])
        self.assertEqual(summary["total_point_opportunities"], 0)
        self.assertEqual(summary["complete_windows"], 0)
        for arm in summary["arms"].values():
            self.assertIsNone(arm["conditional_point_mae_dn"])
            self.assertIsNone(arm["mean_complete_window_mae_dn"])

    def test_measure_rejects_wrong_current_shape(self):
        with self.assertRaises(ValueError):
            runner.measure(np.zeros(2), diagnose(np.zeros((8, 3))))

    def test_measure_rejects_already_mutated_diagnostic(self):
        d = diagnose(np.zeros((8, 3)))
        d["slow8"] = np.full(3, 10.)
        with self.assertRaises(ValueError):
            runner.measure(np.zeros(3), d)

    def test_measure_preserves_forecast_for_different_currents(self):
        d = diagnose(np.tile(np.arange(8)[:, None], (1, 2)))
        before = d["diagnostic_sha256"]
        first = runner.measure(np.full(2, 8.), d)
        second = runner.measure(np.full(2, -8.), d)
        self.assertNotEqual(first["arms"]["median8"]["mae_dn"], second["arms"]["median8"]["mae_dn"])
        self.assertEqual(diagnostic_fingerprint(d), before)

    def test_measure_rejects_nonfinite_scoring_reductions(self):
        d = diagnose(np.zeros((8, 3)))
        with np.errstate(all="ignore"):
            for value in (1e200, np.finfo(float).max):
                with self.subTest(current=value):
                    with self.assertRaises(ValueError):
                        runner.measure(np.full(3, value), d)

    def test_diagnostic_summary_retains_undefined_counts_and_tiny_margins(self):
        constant = runner.diagnostic_summary(diagnose(np.zeros((8, 2))))
        self.assertEqual(constant["recent_center_defined_points"], 0)
        self.assertEqual(constant["observed_return_defined_points"], 0)
        self.assertIsNone(constant["observed_return_fraction_mean"])
        self.assertEqual(sum(constant["recent_center_suffix_counts"].values()), 0)
        tiny = diagnose(np.asarray([0.]*5+[1e-9]*3)[:, None])
        summary = runner.diagnostic_summary(tiny)
        self.assertEqual(summary["recent_center_suffix_counts"]["3"], 1)
        self.assertAlmostEqual(summary["mean_absolute_recent_center_margin_dn"], 1e-9, places=20)

    def test_aggregate_empty_and_all_missing_are_explicit(self):
        empty = runner.aggregate([])
        self.assertEqual(empty["response_windows"], 0)
        self.assertEqual(empty["total_point_opportunities"], 0)
        self.assertIsNone(empty["observed_return_fraction_mean"])
        d = diagnose(np.full((8, 2), np.nan))
        row = dict(measurement=runner.measure(np.ones(2), d), diagnostic=runner.diagnostic_summary(d))
        missing = runner.aggregate([row])
        self.assertEqual(missing["unscorable_points"], 2)
        self.assertEqual(missing["complete_windows"], 0)
        for result in (empty, missing):
            for arm in result["arms"].values():
                self.assertIsNone(arm["conditional_point_mae_dn"])
                self.assertIsNone(arm["conditional_point_rmse_dn"])
                self.assertIsNone(arm["max_absolute_error_dn"])

    def test_aggregate_weights_scorable_points_and_counts_complete_separately(self):
        d = diagnose(np.zeros((8, 2)))
        rows = [dict(measurement=runner.measure(current, d), diagnostic=runner.diagnostic_summary(d))
                for current in (np.asarray([2., 4.]), np.asarray([10., np.nan]))]
        result = runner.aggregate(rows)
        self.assertEqual(result["response_windows"], 2)
        self.assertEqual(result["complete_windows"], 1)
        self.assertEqual(result["scorable_points"], 3)
        self.assertEqual(result["unscorable_points"], 1)
        for arm in result["arms"].values():
            self.assertAlmostEqual(arm["conditional_point_mae_dn"], 16/3)
            self.assertAlmostEqual(arm["conditional_point_rmse_dn"], np.sqrt(120/3))
            self.assertEqual(arm["mean_complete_window_mae_dn"], 3)

    def test_twin_proof_identical_history_and_different_current(self):
        output, manifest, pair = self.twin_fixture()
        summary = self.score_prepared(output, manifest, [pair])
        self.assertEqual(summary["twin_pair_count"], 1)
        proof = runner.read_json(output/"identical_prefix_pairs.json")["pairs"][0]
        self.assertTrue(proof["identical_priors"])
        self.assertTrue(proof["identical_diagnostics"])
        self.assertTrue(proof["identical_forecasts"])
        self.assertEqual(proof["possible_response_separation_dn"], dict(min=8., max=8.))
        self.assertEqual(proof["necessary_worst_world_error_dn"], dict(min=4., max=4.))
        self.assertEqual(proof["minimum_interval_width_to_cover_both_dn"], dict(min=8., max=8.))

    def test_twin_proof_rejects_different_histories(self):
        output, manifest, pair = self.twin_fixture(different_history=True)
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest, [pair])

    def test_twin_proof_rejects_collapsed_future_responses(self):
        output, manifest, pair = self.twin_fixture(step_current=0., pulse_current=0.)
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest, [pair])

    def test_twin_proof_rejects_wrong_declared_future_separation(self):
        output, manifest, pair = self.twin_fixture(step_current=4., pulse_current=0.)
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest, [pair])

    def test_twin_proof_checks_signed_separation(self):
        output, manifest, pair = self.twin_fixture(step_current=-8., pulse_current=0.)
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest, [pair])

    def test_score_rejects_bound_input_mutation_before_measure(self):
        output, manifest = self.prepared()
        Path(manifest["cases"][0]["input_path"]).write_bytes(b"changed fixture")
        with patch.object(runner, "measure", side_effect=AssertionError("Must reject first")):
            with self.assertRaises(ValueError):
                self.score_prepared(output, manifest)

    def test_score_rejects_bound_context_mutation_before_measure(self):
        output, manifest = self.prepared()
        Path(manifest["cases"][0]["context_path"]).write_text("[]")
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest)

    def test_score_rejects_memory_manifest_mutation(self):
        output, manifest = self.prepared()
        manifest["case_count"] += 1
        with self.assertRaises(ValueError):
            self.score_prepared(output, manifest)

    def test_score_rejects_bound_file_changed_during_scoring(self):
        output, manifest = self.prepared()
        original = runner.measure
        def mutate(current, d):
            Path(manifest["cases"][0]["input_path"]).write_bytes(b"changed during fixture scoring")
            return original(current, d)
        with patch.object(runner, "measure", side_effect=mutate):
            with self.assertRaises(ValueError):
                self.score_prepared(output, manifest)
        self.assertFalse((output/"summary.json").exists())

    def test_score_rejects_manifest_changed_during_scoring(self):
        output, manifest = self.prepared()
        original = runner.measure
        def mutate(current, d):
            (output/"forecasts_frozen.json").write_text("{}")
            return original(current, d)
        with patch.object(runner, "measure", side_effect=mutate):
            with self.assertRaises(ValueError):
                self.score_prepared(output, manifest)
        self.assertFalse((output/"summary.json").exists())

    def test_score_preserves_case_family_phase_and_noise_denominators(self):
        first = case_fixture(phase="baseline")
        second = case_fixture(phase="event")
        second["values"][8, 0] = np.nan
        output, manifest = self.prepared({"a": first, "b": second})
        summary = self.score_prepared(output, manifest)
        self.assertEqual(summary["overall"]["response_windows"], 2)
        self.assertEqual(summary["overall"]["total_point_opportunities"], 288)
        self.assertEqual(summary["overall"]["scorable_points"], 287)
        self.assertEqual(summary["overall"]["complete_windows"], 1)
        self.assertEqual(set(summary["by_case"]), {"a", "b"})
        self.assertEqual(set(summary["by_family_phase"]["fixture"]), {"baseline", "event"})
        self.assertEqual(summary["by_noise_level"]["0"]["total_point_opportunities"], 288)
        self.assertTrue(summary["generated_only_not_real_camera_or_airborne_validation"])

    def test_canonical_file_hash_bindings_and_symlink_rejection(self):
        path = self.root/"bound.txt"
        path.write_text("unchanged")
        bindings = {str(path): runner.sha(path)}
        runner.check_bindings(bindings)
        link = self.root/"linked.txt"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            runner.sha(link)
        path.write_text("changed")
        with self.assertRaises(ValueError):
            runner.check_bindings(bindings)

    def test_exclusive_json_npz_writes_preserve_existing_outputs(self):
        json_path = self.root/"output.json"
        runner.write_json(json_path, {"original": True})
        with self.assertRaises(FileExistsError):
            runner.write_json(json_path, {"changed": True})
        self.assertEqual(runner.read_json(json_path), {"original": True})
        npz_path = self.root/"output.npz"
        runner.write_npz(npz_path, {"value": np.asarray([1.])})
        with self.assertRaises(FileExistsError):
            runner.write_npz(npz_path, {"value": np.asarray([2.])})
        np.testing.assert_array_equal(runner.load_npz(npz_path)["value"], [1.])

    def test_npz_load_is_readonly_and_rejects_object_arrays(self):
        path = self.root/"array.npz"
        runner.write_npz(path, {"value": np.asarray([1.])})
        self.assertFalse(runner.load_npz(path)["value"].flags.writeable)
        path = self.root/"object.npz"
        runner.write_npz(path, {"value": np.asarray([object()], dtype=object)})
        with self.assertRaises(ValueError):
            runner.load_npz(path)

    def mocked_run_context(self, source, prepare=None, score=None):
        stack = ExitStack()
        stack.enter_context(patch.object(runner, "OUTPUT", self.root))
        stack.enter_context(patch.object(runner, "sources", return_value=[source]))
        stack.enter_context(patch.object(runner, "specifications", return_value=[
            specification(f"fixture_{i}") for i in range(98)]))
        stack.enter_context(patch.object(runner, "benchmark_metadata", return_value={"twin_pairs": [{}]*40}))
        stack.enter_context(patch.object(runner, "prepare", side_effect=prepare or (lambda specs, output: {})))
        stack.enter_context(patch.object(runner, "score", side_effect=score or (
            lambda manifest, metadata, output: {"overall": {"response_windows": 5488, "total_point_opportunities": 790272}})))
        stack.enter_context(patch("builtins.print"))
        return stack

    def test_run_requires_fresh_immediate_child_without_symlinks(self):
        source = self.root/"source.py"
        source.write_text("fixture source")
        existing = self.root/"existing"
        existing.mkdir()
        target = self.root/"target"
        target.mkdir()
        symlink = self.root/"linked"
        symlink.symlink_to(target, target_is_directory=True)
        with self.mocked_run_context(source):
            for output in (existing, self.root/"nested"/"child", self.root.parent/"outside", symlink):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    runner.run(output)

    def test_run_freezes_inputs_before_prepare_and_scores_after_prepare(self):
        source = self.root/"source.py"
        source.write_text("fixture source")
        events = []
        def prepared(specs, output):
            freeze = runner.read_json(output/"freeze.json")
            self.assertEqual(len(freeze["specifications"]), 98)
            self.assertTrue(freeze["before_generation_and_scoring"])
            runner.check_bindings(freeze["source_files_sha256"])
            events.append("all diagnostics prepared")
            return {"fixture": True}
        def scored(manifest, metadata, output):
            self.assertEqual(events, ["all diagnostics prepared"])
            self.assertEqual(manifest, {"fixture": True})
            events.append("responses scored")
            return {"overall": {"response_windows": 5488, "total_point_opportunities": 790272}}
        output = self.root/"fresh"
        with self.mocked_run_context(source, prepared, scored):
            runner.run(output)
        self.assertEqual(events, ["all diagnostics prepared", "responses scored"])
        self.assertTrue(runner.read_json(output/"completion_receipt.json")["completed"])

    def test_run_rejects_source_hash_mutation(self):
        source = self.root/"source.py"
        source.write_text("fixture source")
        def scored(manifest, metadata, output):
            source.write_text("changed during fixture scoring")
            return {"overall": {"response_windows": 5488, "total_point_opportunities": 790272}}
        output = self.root/"fresh"
        with self.mocked_run_context(source, score=scored), self.assertRaises(ValueError):
            runner.run(output)
        self.assertFalse((output/"completion_receipt.json").exists())

    def test_run_rejects_freeze_hash_mutation(self):
        source = self.root/"source.py"
        source.write_text("fixture source")
        def scored(manifest, metadata, output):
            (output/"freeze.json").write_text("{}")
            return {"overall": {"response_windows": 5488, "total_point_opportunities": 790272}}
        output = self.root/"fresh"
        with self.mocked_run_context(source, score=scored), self.assertRaises(ValueError):
            runner.run(output)
        self.assertFalse((output/"completion_receipt.json").exists())

    def test_run_rejects_wrong_scored_denominators(self):
        source = self.root/"source.py"
        source.write_text("fixture source")
        scored = lambda manifest, metadata, output: {
            "overall": {"response_windows": 5487, "total_point_opportunities": 790272}}
        output = self.root/"fresh"
        with self.mocked_run_context(source, score=scored), self.assertRaises(ValueError):
            runner.run(output)
        self.assertFalse((output/"completion_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
