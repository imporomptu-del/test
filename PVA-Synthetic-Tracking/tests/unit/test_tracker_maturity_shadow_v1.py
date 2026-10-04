"""Generated metadata tests only: no media, saved real replay, Jetson or PVA."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_tracker_capacity_shadow as base
import check_tracker_maturity_shadow_v1 as runner
import compare_discovery_feature_supply as scorer


class Manager:
    def update(self):
        raise AssertionError("not an actual tracker")


class Tracker:
    def __init__(self, delta=0, state_delta=0, learning_from=None):
        self.delta, self.state_delta, self.learning_from = delta, state_delta, learning_from
        self.managers, self.extents, self.qualified, self.summary = {}, {}, set(), {}
        self.ever_qualified, self.previous_records = set(), []
        self.previous_timestamp_ns, self.quality = None, {}

    def learning_centers(self, timestamp, segment):
        return [(1, 1)] if self.learning_from is not None and timestamp >= self.learning_from else []

    def update(self, candidates, index, timestamp, segment, matrix, shape):
        self.extents["frame"] = index + self.state_delta
        self.previous_records = [{"value": index + self.delta}]
        self.previous_timestamp_ns = timestamp
        return self.previous_records, {"n": len(candidates)}


class CountOnly:
    def __init__(self):
        self.n = 0

    def add(self, row, output, tracker):
        self.n += 1
        return {"frames": self.n}

    def finish(self):
        return {"frames": self.n}


def rows_and_old(count=3):
    rows, old, tracker = [], [], Tracker()
    for i in range(count):
        row = dict(frame_index=i, timestamp_ns=i*100000000, segment=0, candidates=[],
            source_to_reference=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            coverage=dict(full_shape_hw=[3190, 4784], detection_ready=True),
            tracks=[{"value": i}], tracking_metrics={"n": 0})
        learning = base.value_sha(tracker.learning_centers(row["timestamp_ns"], 0))
        tracks, metrics = tracker.update([], i, row["timestamp_ns"], 0, None, None)
        output = base.value_sha(dict(tracks=tracks, tracking_metrics=metrics))
        state = base.value_sha(base.snapshot(tracker))
        rows.append(row)
        old.append(dict(frame_index=i, differences={}, output_sha256=[output]*2,
            archive_output_sha256=output, internal_state_sha256=[state]*2,
            learning_centers_sha256=[learning]*2, archive_derived_learning_sha256=learning))
    return rows, old


class ReplayTests(unittest.TestCase):
    def replay(self, rows, old, trackers=None, count=3):
        audit, journal = io.StringIO(), io.StringIO()
        saved = Manager.update
        result, targets = runner.replay_rows(base, rows, old, trackers or [Tracker() for _ in range(3)],
            [Manager.update]*3, Manager, audit, journal, expected_frames=count, workload_factory=CountOnly)
        self.assertIs(Manager.update, saved)
        return result, [json.loads(s) for s in audit.getvalue().splitlines()], [json.loads(s) for s in journal.getvalue().splitlines()]

    def test_baseline_old_states_and_three_copy_identical(self):
        rows, old = rows_and_old()
        result, audit, journal = self.replay(rows, old)
        self.assertTrue(result["passed"])
        for key in ("attempted_frames", "exact_archive_frames", "exact_previous_state_frames",
                    "deterministic_candidate_frames", "baseline_learning_exact_frames"):
            self.assertEqual(result[key], 3)
        self.assertIsNone(result["first_learning_divergence"])
        self.assertEqual(len(journal), 3)
        self.assertTrue(all(r["shadow"]["saved_proposals"] for r in journal))
        self.assertTrue(all(r["shadow"]["detector_coverage_motion_and_timings_are_archived_not_rerun"] for r in journal))

    def test_candidate_difference_allowed_but_learning_is_marked_before_update(self):
        rows, old = rows_and_old()
        result, audit, journal = self.replay(rows, old,
            [Tracker(), Tracker(delta=1, learning_from=100000000), Tracker(delta=1, learning_from=100000000)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["first_output_divergence"]["frame_index"], 0)
        self.assertEqual(result["first_learning_divergence"]["frame_index"], 1)
        self.assertTrue(result["first_learning_divergence"]["measured_before_frame_update"])
        self.assertEqual([r["shadow"]["past_or_at_learning_divergence"] for r in journal], [False, True, True])
        self.assertEqual(journal[0]["tracks"], [{"value": 1}])
        self.assertEqual(rows[0]["tracks"], [{"value": 0}])

    def test_baseline_archive_mismatch_not_excused_by_candidate_agreement(self):
        rows, old = rows_and_old()
        rows[0]["tracks"][0]["value"] = 0.0
        with self.assertRaisesRegex(ValueError, "baseline archived observable mismatch"):
            self.replay(rows, old)

    def test_previous_state_or_learning_mismatch_aborts(self):
        for key in ("internal_state_sha256", "learning_centers_sha256"):
            rows, old = rows_and_old()
            old[0][key][1] = "0"*64
            with self.assertRaisesRegex(ValueError, "baseline previous"):
                self.replay(rows, old)

    def test_candidate_output_state_or_learning_nondeterminism_aborts(self):
        for last in (Tracker(delta=1), Tracker(state_delta=1), Tracker(learning_from=0)):
            rows, old = rows_and_old()
            with self.assertRaisesRegex(ValueError, "candidate nondeterminism"):
                self.replay(rows, old, [Tracker(), Tracker(), last])

    def test_failure_does_not_consume_later_rows_and_restores_manager(self):
        rows, old = rows_and_old()
        rows[0]["tracks"] = []
        def streamed():
            yield rows[0]
            raise AssertionError("should not read the next frame")
        original = Manager.update
        with self.assertRaisesRegex(ValueError, "baseline archived observable mismatch"):
            self.replay(streamed(), iter(old))
        self.assertIs(Manager.update, original)

    def test_missing_extra_and_discontinuous_frames_fail(self):
        rows, old = rows_and_old()
        for a, b in ((rows[:-1], old), (rows, old[:-1]), (rows[:-1], old[:-1]), (rows+rows[-1:], old+old[-1:])):
            with self.assertRaises(ValueError):
                self.replay(a, b)
        rows[1]["timestamp_ns"] += 1
        with self.assertRaisesRegex(ValueError, "frame/timestamp"):
            self.replay(rows, old)


def target_rows():
    references = [dict(frame_index=f, measurement_source_xy=[100, 200]) for f in range(430, 465)]
    rows = [dict(frame_index=f, segment=2, detection_ready=True, tracks=[dict(
        track_id="dark:7", measured=True, qualified_moving=True, measurement_source_xy=[100, 200])]) for f in range(430, 465)]
    return rows, references


class TargetAndWorkloadTests(unittest.TestCase):
    def test_candidate_identity_need_not_equal_original_literal_id(self):
        rows, references = target_rows()
        candidate = copy.deepcopy(rows)
        for row in candidate:
            row["tracks"][0]["track_id"] = "dark:900"
        result = runner.target_comparison(scorer, rows, candidate, references)
        self.assertTrue(result["scientific_guard_passed"])
        self.assertEqual(result["candidate"]["complete_coherent_identities"], ["2/dark:900"])

    def test_prediction_not_actual_match_and_all_lost_samples_retained(self):
        rows, references = target_rows()
        candidate = copy.deepcopy(rows)
        candidate[3]["tracks"][0]["measured"] = False
        candidate[9]["tracks"][0]["qualified_moving"] = False
        result = runner.target_comparison(scorer, rows, candidate, references)
        self.assertFalse(result["scientific_guard_passed"])
        self.assertEqual(result["newly_lost_frames"], [433, 439])
        self.assertEqual(result["recovered_frames"], [])
        self.assertEqual(len(result["per_sample"]), 35)

    def test_full_any_match_does_not_hide_fragmentation_or_ambiguity(self):
        rows, references = target_rows()
        for mode in ("fragmented", "ambiguous"):
            candidate = copy.deepcopy(rows)
            if mode == "fragmented":
                candidate[-1]["tracks"][0]["track_id"] = "dark:8"
            else:
                candidate[4]["tracks"].append(dict(candidate[4]["tracks"][0], track_id="dark:8"))
            result = runner.target_comparison(scorer, rows, candidate, references)
            self.assertEqual(result["candidate"]["any_identity_matched_frames"], 35)
            self.assertFalse(result["scientific_guard_passed"])

    def test_corrupt_baseline_target_cannot_become_a_candidate_success(self):
        rows, references = target_rows()
        broken = copy.deepcopy(rows)
        broken[2]["detection_ready"] = False
        with self.assertRaisesRegex(ValueError, "original target reference"):
            runner.target_comparison(scorer, broken, rows, references)

    def test_workload_reports_measured_predictions_births_and_censoring_separately(self):
        collector = runner.Workload()
        cfg = SimpleNamespace(max_active_tracks=256, confirmation_independent_hits=4, max_missed_windows=7)
        tracker = SimpleNamespace(managers={"dark": SimpleNamespace(config=cfg, _tracks={})})
        metrics = {"dark": dict(birth_count=1, deleted_track_count=0,
            dropped_birth_count_at_active_track_cap=2, lifecycle_counts={"tentative": 2},
            birth_admission=dict(confirmed_tracks_evicted=0, tentative_replacements=[]))}
        row = dict(frame_index=50, segment=0, coverage=dict(detection_ready=True), candidates=[{}, {}])
        tracks = [dict(track_id="dark:0", independent_hits=1, qualified_moving=True, measured=True),
                  dict(track_id="dark:1", independent_hits=3, qualified_moving=True, measured=False)]
        per_frame = collector.add(row, dict(tracks=tracks, tracking_metrics=metrics), tracker)
        result = collector.finish()["burst"]
        self.assertEqual(result["counts"]["qualified_measured"], 1)
        self.assertEqual(result["counts"]["qualified_predicted"], 1)
        self.assertEqual(result["counts"]["dropped_births"], 2)
        self.assertEqual(result["first_seen_identity_count"], 2)
        self.assertEqual(result["never_reached_four_hits_by_end_of_clip"], 2)
        self.assertEqual(result["still_active_without_four_hits_at_end_of_clip"], 2)
        self.assertEqual(per_frame["independent_hit_histogram"], {"1": 1, "3": 1})


class BoundaryTests(unittest.TestCase):
    def test_real_candidate_source_introspection_reproduces_old_loader_failure_and_is_fixed(self):
        path = ROOT / "scripts/tracker_maturity_candidate_v1.py"
        digest = runner.sha(path)
        unregistered = "maturity_test_unregistered_candidate"
        self.assertNotIn(unregistered, sys.modules)
        spec = importlib.util.spec_from_file_location(unregistered, path)
        broken = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(broken)  # The original loader omitted registration.
        with self.assertRaisesRegex(TypeError, "built-in class"):
            broken._module_binding(digest)
        registered = "maturity_test_registered_candidate"
        self.assertNotIn(registered, sys.modules)
        try:
            fixed = runner.module(registered, path, digest)
            self.assertIs(sys.modules[registered], fixed)
            binding = fixed._module_binding(digest)
            self.assertEqual(binding["class_source_sha256"], runner.CLASS_SHA)
            self.assertEqual(binding["sha256"], runner.CANDIDATE_SHA)
            self.assertIs(runner.module(registered, path, digest), fixed)
        finally:
            sys.modules.pop(registered, None)

    def test_module_loader_preserves_existing_different_binding(self):
        name = "maturity_test_loader_collision"
        sentinel = SimpleNamespace(__seaqr_source_binding__=("different", "0"*64))
        path = ROOT / "scripts/tracker_maturity_candidate_v1.py"
        with patch.dict(sys.modules, {name: sentinel}):
            with self.assertRaisesRegex(ValueError, "existing module binding"):
                runner.module(name, path, runner.sha(path))
            self.assertIs(sys.modules[name], sentinel)

    def test_module_loader_removes_failed_import_registration(self):
        name = "maturity_test_loader_failure"
        self.assertNotIn(name, sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "failing.py"
            path.write_text("raise RuntimeError('generated import failure')\n")
            with self.assertRaisesRegex(RuntimeError, "generated import failure"):
                runner.module(name, path, runner.sha(path))
            self.assertNotIn(name, sys.modules)

    def test_module_loader_cleans_up_on_post_import_hash_failure(self):
        name = "maturity_test_loader_postcheck"
        self.assertNotIn(name, sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "plain.py"
            path.write_text("value = 1\n")
            with patch.object(runner, "sha", side_effect=["a"*64, "b"*64]):
                with self.assertRaisesRegex(ValueError, "module changed during import"):
                    runner.module(name, path, "a"*64)
            self.assertNotIn(name, sys.modules)

    def test_caller_hash_is_checked_before_loading_any_old_module(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            (workspace / "freeze.json").write_text("{}\n")
            with patch.object(runner, "module", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(ValueError, "caller-bound freeze"):
                    runner.load(workspace, "0"*64)

    def test_protocol_fixed_sources_counts_and_no_promotion(self):
        protocol = runner.freeze_spec()
        self.assertEqual(protocol["clips"], ["0170", "0240"])
        self.assertEqual(protocol["frames_per_clip"], 673)
        self.assertEqual(protocol["candidate_copies"], 2)
        self.assertEqual(protocol["target"]["samples"], 35)
        self.assertTrue(protocol["scientific_failure_completes"])
        self.assertFalse(protocol["production_promotion"])
        self.assertEqual(len(runner.FILES), 9)

    def test_reject_mode_and_existing_output_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            with patch.object(runner, "workspace_guard", return_value=workspace), \
                 patch.object(runner, "load", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    runner.child(workspace, "0"*64)
                (workspace / "preflight.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "existing receipt"):
                    runner.child(workspace, "0"*64, preflight=True)

    def test_runtime_error_retains_exclusive_failure_receipt_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            with patch.object(runner, "workspace_guard", return_value=workspace), \
                 patch.object(runner, "load", side_effect=ValueError("pinned input changed")) as mocked:
                result = runner.child(workspace, "0"*64, preflight=True)
            self.assertEqual(mocked.call_count, 1)
            self.assertFalse(result["passed"])
            self.assertIn("pinned input changed", result["error"])
            self.assertEqual(json.loads((workspace / "preflight.json").read_text()), result)


class ReceiptTests(unittest.TestCase):
    def fixture(self, root, count=673):
        workspace, old = root / "workspace", root / "old"
        workspace.mkdir()
        runtime = dict(blas=[{"threads": 12}], affinity=[0], numpy="1.26.1", opencv="4.10.0",
                       opencv_threads=12, thread_environment={}, clock_ticks=100)
        after = dict(runtime, opencv_threads=2)
        for clip in ("0170", "0240"):
            (old / clip).mkdir(parents=True)
            (old / clip / "result.json").write_text(json.dumps(dict(inputs_sha256={}, runtime_before=runtime, runtime_after=after)))
        value = dict(schema=runner.SCHEMA+".preflight", workspace=str(workspace), freeze_sha256="f"*64,
            clip=None, passed=True, error=None, inputs_unchanged_after_check=True, clock_controls_unchanged=True,
            clock_policy_before={"cpu": "unchanged"}, clock_policy_after={"cpu": "unchanged"},
            inputs_sha256={}, source_media_opened=False, detector_replayed=False, raw16_or_holdouts_accessed=False,
            production_promotion=False, scientific_improvement_claimed=False, saved_proposals_shadow_only=True,
            runtime_before=runtime)
        pre = workspace / "preflight.json"
        pre.write_text(json.dumps(value))
        (workspace / "0170").mkdir()
        artifacts = {}
        for name in ("candidate_frames.jsonl", "shadow_audit.jsonl"):
            path = workspace / "0170" / name
            path.write_text("".join(json.dumps(dict(frame_index=i))+"\n" for i in range(count)))
            artifacts[str(path)] = base.sha(path)
        replay = {k: 673 for k in ("attempted_frames", "exact_archive_frames", "exact_previous_state_frames",
                                   "deterministic_candidate_frames", "baseline_learning_exact_frames")}
        binding = dict(sha256=runner.CANDIDATE_SHA, class_source_sha256=runner.CLASS_SHA,
                       unchanged_transformed_source_sha256=base.METHOD_SHA, eligibility_changed=False, configuration_changed=False)
        counters = dict(geometry_calls=0, geometry_fallbacks=0, batch_fallbacks=0, innovation_fallbacks=0,
                        batch_track_rows=3, innovation_tracks=3)
        replay.update(passed=True, scientific_guard_passed=False, separate_adapter_owners_verified=True,
                      artifacts_sha256=artifacts, candidate_bindings=[binding, dict(binding)], adapter_runs=[dict(counters) for _ in range(3)])
        value = dict(value, schema=runner.SCHEMA+".run", clip="0170", runtime_after=after, replay=replay,
                     preflight_sha256=base.sha(pre), inputs_sha256={str(pre): base.sha(pre)})
        path = workspace / "0170/result.json"
        path.write_text(json.dumps(value))
        return workspace, old, value, path

    def validate(self, workspace, old):
        with patch.object(runner, "OLD", old), patch.object(runner, "load", return_value=(base, {}, {})):
            return runner.validate_child_receipt(workspace, "f"*64, "run_0170")

    def test_complete_scientifically_rejected_receipt_is_valid_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace, old, value, path = self.fixture(Path(temp).resolve())
            artifacts = self.validate(workspace, old)
            self.assertEqual(len(artifacts), 4)
            self.assertIn(str(path), artifacts)
            self.assertFalse(value["replay"]["scientific_guard_passed"])

    def test_class_runtime_and_input_bindings_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace, old, value, path = self.fixture(Path(temp).resolve())
            for mutate in (
                lambda v: v["replay"]["candidate_bindings"][0].update(class_source_sha256="0"*64),
                lambda v: v["runtime_after"].update(opencv_threads=12),
                lambda v: v["inputs_sha256"].update(unapproved="0"*64),
                lambda v: v["replay"]["adapter_runs"][0].update(batch_fallbacks=1),
            ):
                wrong = copy.deepcopy(value)
                mutate(wrong)
                path.write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):
                    self.validate(workspace, old)

    def test_artifact_rehash_and_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace, old, value, path = self.fixture(Path(temp).resolve(), count=672)
            with self.assertRaisesRegex(ValueError, "frame count"):
                self.validate(workspace, old)
        with tempfile.TemporaryDirectory() as temp:
            workspace, old, value, path = self.fixture(Path(temp).resolve())
            with (workspace / "0170/candidate_frames.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.validate(workspace, old)

    def test_receipt_changed_during_metadata_read_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace, old, value, path = self.fixture(Path(temp).resolve())
            original = base.read
            def read_then_change(p):
                result = original(p)
                if p == path:
                    path.write_text(json.dumps(dict(result, extra="tampered after read")))
                return result
            with patch.object(base, "read", side_effect=read_then_change):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    self.validate(workspace, old)


if __name__ == "__main__":
    unittest.main()
