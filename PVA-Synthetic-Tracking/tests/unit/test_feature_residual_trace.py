"""Generated arrays and mocked lifecycle only: no media, VPI, remote or holdouts."""
import base64
from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tiny_target.motion import GlobalMotionConfig, MotionCorrespondences, fit_global_motion
from tiny_target.types import Frame, TimestampSource

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("trace_test_runner", "scripts/run_feature_residual_trace.py")
selection = load("trace_test_selection", "scripts/run_discovery_feature_selection.py")
DIGEST = "a" * 64


def frame(index):
    return Frame(np.arange(4096, dtype=np.uint16).reshape(64, 64).astype(np.uint8),
                 index * 100_000_000, index, "generated", 8, TimestampSource.CONTAINER_RATE)


def correspondence(index=1):
    p = np.array([[x, y] for y in (8, 16, 24, 32, 40, 48) for x in (8, 16, 24, 32, 40, 48)], np.float32)
    return MotionCorrespondences(p, p + np.array([1, 0], np.float32), np.arange(len(p), dtype=np.float32),
        np.zeros(len(p), np.float32), index - 1, index, (index - 1) * 100_000_000, index * 100_000_000,
        (64, 64), (32, 32), {"aggregate_rejected": 3}, {}, {"extra_readback": False})


def decode(desc):
    raw = base64.b64decode(desc["data_base64"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == desc["sha256"]
    return np.frombuffer(raw, dtype=np.dtype(desc["dtype"])).reshape(desc["shape"])


def pre_fixture(root, clip, hashes, identities, audit):
    return dict(schema=runner.SCHEMA + ".preflight", passed=True, workspace=str(root), clip=clip,
        source=runner.source_spec(clip), candidate=dict(runner.CANDIDATE), input_sha256=hashes,
        probe_passed=True, detector_run=False, conversion={"passed": True}, clock_policy={"fixed": True},
        controls=[dict(name=n, passed=True, closed=True) for n in selection.CONTROL_NAMES],
        cpu_parity=dict(passed=True, numpy_version=np.__version__, same_candidate_quota_in_both_paths=True,
            exact_reference_comparison=True, source_pixels_unchanged=True, points_and_scores_unchanged=True,
            score_precision_changed=False, max_features=384, max_features_per_cell=8, grid_rows=6, grid_cols=8,
            cases=[dict(name=n, passed=True, generated_only=True) for n in selection.PARITY_NAMES]),
        passive_capture_check=dict(passed=True, generated_only=True, original_fit_calls=1, same_return_object=True,
            input_arrays_unchanged=True, exact_array_byte_serialization=True, native_gray_hashes=True),
        feature_adapter=audit, **identities)


class ScopeTests(unittest.TestCase):
    def test_exact_pairs_plan_and_failed_frames(self):
        plan = runner.read(ROOT / "configs/evaluation/feature_residual_trace_plan.json")
        runner.validate_plan(plan)
        self.assertEqual([len(runner.fixed_pairs(c)) for c in runner.SOURCE_HASHES], [56, 44])
        self.assertEqual(runner.fixed_groups("0240")["positive"], list(range(430, 465)))
        self.assertEqual(runner.FAILED["0240"], [344])
        for mutate in (lambda p: p["capture_pairs"]["0170"].append(643),
                       lambda p: p["capture_groups"]["0240"].update(positive=[430]),
                       lambda p: p["parity"]["ignored_exact_paths"].append(["motion"]),
                       lambda p: p["execution"].update(automatic_retries=1),
                       lambda p: p["candidate"].update(max_features=500),
                       lambda p: p["sources"].pop("0240")):
            bad = copy.deepcopy(plan)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_plan(bad)

    def test_frozen_bundle_paths_and_identity(self):
        files = {"run_feature_residual_trace.py": DIGEST, runner.PLAN_NAME: "b" * 64}
        value = dict(schema="feature_residual_trace.v1", candidate=dict(runner.CANDIDATE), files=files,
            sources={c: runner.source_spec(c) for c in runner.SOURCE_HASHES},
            original_workspace=str(runner.ORIGINAL_WORKSPACE), original_freeze_sha256=runner.ORIGINAL_FREEZE_SHA)
        runner.validate_freeze(value, files)
        for mutate in (lambda v: v["files"].update({"../escape": DIGEST}),
                       lambda v: v.update(original_workspace="/tmp/elsewhere"),
                       lambda v: v.update(original_freeze_sha256="b" * 64),
                       lambda v: v["files"].pop(runner.PLAN_NAME)):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_freeze(bad, bad["files"])

    def test_scope_no_root_no_overwrite_or_dangling_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(runner, "WORKSPACE_PATTERN", re.escape(temp)), patch.object(runner.os, "geteuid", return_value=1000):
                runner.workspace_guard(root, "0170", "preflight")
                (root / "0170").mkdir()
                for name in ("run", "execution_receipt.json", "preflight.json", "trace.json", "parity.json"):
                    path = root / "0170" / name
                    path.symlink_to(root / "absent")
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        runner.workspace_guard(root, "0170", "preflight")
                    path.unlink()
                with self.assertRaises(ValueError):
                    runner.workspace_guard(root, "0126", "run")
                with patch.object(runner.os, "geteuid", return_value=0), self.assertRaises(ValueError):
                    runner.workspace_guard(root, "0170", "run")
        with self.assertRaises(ValueError):
            runner.workspace_guard(Path("/tmp/wrong"), "0170", "run")

    def test_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.json"
            runner.write(path, {"first": True})
            with self.assertRaises(FileExistsError):
                runner.write(path, {"second": True})
            self.assertEqual(runner.read(path), {"first": True})


class CaptureTests(unittest.TestCase):
    def test_byte_descriptors_preserve_nan_payload_signed_zero_and_precision(self):
        for array in (np.array([0x7ff8000000000123, 0x8000000000000000], np.uint64).view(np.float64),
                      np.array([1, 16777217, 4294967295], np.uint32),
                      np.array([True, False], bool), np.arange(6, dtype=np.float32).reshape(3, 2)[:, ::-1]):
            desc = runner.array_descriptor(array)
            self.assertEqual(decode(desc).tobytes(), array.tobytes())
            self.assertEqual(desc["dtype"], array.dtype.str)
        with self.assertRaises(ValueError):
            runner.array_descriptor(np.array([object()], object))

    def test_passive_fit_once_same_objects_full_causal_and_copy(self):
        cfg = GlobalMotionConfig()
        capture = runner.TraceCapture("generated", [2], asdict(cfg), full_size=(64, 64), total_frames=3)
        original = Mock(side_effect=fit_global_motion)
        fits = capture.wrap_fit(original)
        first, second = correspondence(1), correspondence(2)
        capture.capture_source(first, frame(0), frame(1))
        result1 = fits(first, cfg)
        self.assertEqual(capture.rows, {})
        capture.capture_source(second, frame(1), frame(2))
        result2 = fits(second, cfg)
        self.assertEqual(original.call_count, 2)
        self.assertIs(original.call_args_list[0].args[0], first)
        self.assertIs(original.call_args_list[1].args[0], second)
        self.assertIs(original.call_args_list[1].args[1], cfg)
        self.assertFalse(capture.rows[2]["fit"]["residuals_px"].flags.writeable)
        self.assertFalse(np.shares_memory(capture.rows[2]["correspondence"]["previous_points"], second.previous_points))
        rows = capture.finish()
        self.assertEqual(rows[0]["native_gray_pixel_sha256"]["previous"], frame(1).pixel_sha256())
        self.assertEqual(decode(rows[0]["fit"]["residuals_px"]).tobytes(), result2.residuals_px.tobytes())
        self.assertEqual(decode(rows[0]["fit"]["inlier_mask"]).tobytes(), result2.inlier_mask.tobytes())
        self.assertEqual(decode(rows[0]["correspondence"]["harris_scores"]).tobytes(), second.harris_scores.tobytes())
        self.assertEqual(result1.current_frame_index, 1)
        json.dumps(rows, allow_nan=False)

    def test_same_fit_return_object_and_estimator_return(self):
        cfg, corr = GlobalMotionConfig(), correspondence()
        expected = fit_global_motion(corr, cfg)
        capture = runner.TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=2)
        class Candidate:
            def estimate(self, previous, current):
                return corr
        estimator = capture.estimator_class(Candidate)()
        self.assertIs(estimator.estimate(frame(0), frame(1)), corr)
        original = Mock(return_value=expected)
        self.assertIs(capture.wrap_fit(original)(corr, cfg), expected)
        original.assert_called_once_with(corr, cfg)

    def test_no_hashing_uncaptured_pairs(self):
        cfg = GlobalMotionConfig()
        capture = runner.TraceCapture("generated", [], asdict(cfg), full_size=(64, 64), total_frames=2)
        prev, cur = Mock(), Mock()
        capture.capture_source(correspondence(), prev, cur)
        prev.pixel_sha256.assert_not_called()
        cur.pixel_sha256.assert_not_called()

    def test_mutation_wrong_sequence_and_missing_source_fail_closed(self):
        cfg, corr = GlobalMotionConfig(), correspondence()
        capture = runner.TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=2)
        original = Mock(side_effect=fit_global_motion)
        with self.assertRaisesRegex(ValueError, "missing/duplicate source"):
            capture.wrap_fit(original)(corr, cfg)
        original.assert_called_once()
        capture = runner.TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=3)
        never = Mock()
        with self.assertRaisesRegex(ValueError, "full-causal"):
            capture.wrap_fit(never)(correspondence(2), cfg)
        never.assert_not_called()
        capture.capture_source(corr, frame(0), frame(1))
        def mutates(c, config):
            result = fit_global_motion(c, config)
            c.previous_points.setflags(write=True)
            c.previous_points[0, 0] += 1
            return result
        with self.assertRaisesRegex(ValueError, "mutated"):
            capture.wrap_fit(mutates)(corr, cfg)

    def test_config_changed_no_original_call_and_duplicate_capture(self):
        cfg, corr = GlobalMotionConfig(), correspondence()
        capture = runner.TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=2)
        original = Mock()
        with self.assertRaisesRegex(ValueError, "configuration"):
            capture.wrap_fit(original)(corr, GlobalMotionConfig(minimum_inliers=31))
        original.assert_not_called()
        capture.capture_source(corr, frame(0), frame(1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture.capture_source(corr, frame(0), frame(1))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            capture.finish()

    def test_fit_facade_patch_restored(self):
        import tiny_target.motion as motion
        cfg, corr = GlobalMotionConfig(), correspondence()
        capture = runner.TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=2)
        capture.capture_source(corr, frame(0), frame(1))
        original = motion.fit_global_motion
        with runner.passive_fit_capture(capture):
            self.assertIsNot(motion.fit_global_motion, original)
            motion.fit_global_motion(corr, cfg)
        self.assertIs(motion.fit_global_motion, original)
        self.assertEqual(len(capture.finish()), 1)
        with patch.object(motion, "fit_global_motion", Mock()), self.assertRaises(ValueError):
            with runner.passive_fit_capture(capture):
                pass

    def test_generated_preflight(self):
        self.assertTrue(runner.generated_capture_check()["passed"])


class ParityTests(unittest.TestCase):
    @staticmethod
    def rows(count=3):
        return [dict(frame_index=i, timestamp_ns=i * 100_000_000, timings_ms={"total": 1},
            motion=dict(pva_timings_ms={"x": 1}, motion_fit=dict(timing_ms=1, quality_status="accepted"),
                        warp_timings_ms={"x": 1}, correspondence_metrics={"count": 48}),
            coverage=dict(detection_ms=1, temporal_coverage=1), tracks=[], candidates=[]) for i in range(count)]

    def compare(self, left, right, count=3):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / str(i) for i in range(2)]
            for path, rows in zip(paths, (left, right)):
                with path.open("x") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
            result = runner.journal_parity(*paths, expected_frames=count)
            self.assertEqual(result["original_journal_sha256"], runner.sha(paths[0]))
            self.assertEqual(result["diagnostic_journal_sha256"], runner.sha(paths[1]))
            return result

    def test_exact_five_timing_paths_only(self):
        left, right = self.rows(), self.rows()
        for row in right:
            row["timings_ms"] = {"anything": 900}
            row["motion"]["pva_timings_ms"] = {"anything": 900}
            row["motion"]["warp_timings_ms"] = {"anything": 900}
            row["motion"]["motion_fit"]["timing_ms"] = 900
            row["coverage"]["detection_ms"] = 900
        self.assertTrue(self.compare(left, right)["passed"])
        self.assertEqual(left, self.rows())

    def test_no_tolerance_or_drop_for_non_timing_values(self):
        for mutate in (lambda r: r.update(timestamp_ns=1),
                       lambda r: r["coverage"].update(temporal_coverage=1.00000000001),
                       lambda r: r["motion"]["motion_fit"].update(quality_status="rejected"),
                       lambda r: r["motion"]["correspondence_metrics"].update(count=47),
                       lambda r: r.update(tracks=[{"p": 1}]),
                       lambda r: r.update(extra_time_ns=0)):
            left, right = self.rows(), self.rows()
            mutate(right[0])
            self.assertFalse(self.compare(left, right)["passed"])
        self.assertFalse(self.compare(self.rows(), self.rows()[:-1])["passed"])
        self.assertFalse(self.compare(self.rows(), self.rows() + self.rows()[:1])["passed"])
        right = self.rows()
        right[1]["frame_index"] = 1.0
        self.assertFalse(self.compare(self.rows(), right)["passed"])


class LifecycleTests(unittest.TestCase):
    def exercise_run(self, *, parity_failure=False, missing_second_preflight=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            originals = root / "originals"
            clip = "0170"
            journal_rows = ParityTests.rows(673)
            original_path = originals / clip / "run/frames.jsonl"
            original_path.parent.mkdir(parents=True)
            with original_path.open("x") as stream:
                for row in journal_rows:
                    stream.write(json.dumps(row) + "\n")
            hashes = dict(files={"run_feature_residual_trace.py": DIGEST}, freeze_sha256=DIGEST)
            plan = dict(capture_pairs={c: runner.fixed_pairs(c) for c in runner.SOURCE_HASHES},
                        capture_groups={c: runner.fixed_groups(c) for c in runner.SOURCE_HASHES},
                        original_artifacts={clip: {"journal": runner.sha(original_path)}})
            identities = {"dependencies": {"code": DIGEST}}
            audit = dict(source_transformation={"sha256": DIGEST},
                effective_motion_configuration={"exclusion_regions_xyxy": (), "minimum_accepted_features": 30},
                change_classification={"algorithm": ["gain16", "quota"], "motion_quality_gates_changed": False},
                estimator_instances=1, successful_pair_backend_checks=672)
            for c in runner.SOURCE_HASHES:
                (root / c).mkdir()
                if c != "0240" or not missing_second_preflight:
                    runner.write(root / c / "preflight.json", pre_fixture(root, c, hashes, identities, audit))
            cfg = GlobalMotionConfig()
            fake_selection = Mock(SCHEMA=selection.SCHEMA)
            fake_selection.validate_preflight.side_effect = selection.validate_preflight
            fake_selection.validate_adapter_roundtrip.side_effect = selection.validate_adapter_roundtrip
            fake_selection.configurations.return_value = (SimpleNamespace(), cfg)
            @contextmanager
            def adapter(reuse, motion, recorded):
                recorded.update(copy.deepcopy(audit))  # tuple in memory vs list in preflight JSON.
                yield object
            fake_selection.candidate_adapter.side_effect = adapter
            baseline, helper = Mock(), Mock(TRACKING_METHOD_SHA=DIGEST)
            fake_selection.load_baseline.return_value = baseline
            baseline.load_helper.return_value = helper
            modules = {"motion_reuse_v12": Mock(), "profile_visible_interaction_v30": Mock(runtime_info=Mock(return_value={"ok": True}))}
            helper.dependencies.return_value = (modules, identities)
            helper.clock_policy_snapshot.return_value = {"fixed": True}
            capture = Mock(rows={i: {} for i in runner.fixed_pairs(clip)}, fit_indices=list(range(1, 673)))
            capture.finish.return_value = [{"current_frame_index": i} for i in runner.fixed_pairs(clip)]
            capture.estimator_class.return_value = object
            def execute(_helper, _modules, source, output, receipt):
                self.assertEqual(str(source), runner.source_spec(clip)["path"])
                self.assertTrue((root / "0240/preflight.json").is_file())
                output.mkdir()
                rows = copy.deepcopy(journal_rows)
                if parity_failure:
                    rows[451]["motion"]["correspondence_metrics"]["count"] = 12
                with (output / "frames.jsonl").open("x") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                runner.write(output / "launch.json", {"source": str(source)})
                report = dict(frames=673, availability={"ready": 554}, detection_status="complete")
                runner.write(output / "report.json", report)
                receipt.update(decoded_frames_verified=673, processed_frames=673,
                    motion_attempts=[dict(frame=i, error=None) for i in range(1, 673)])
                return report
            baseline.execute_baseline.side_effect = execute
            with patch.object(runner, "WORKSPACE_PATTERN", re.escape(temp)), \
                 patch.object(runner.os, "geteuid", return_value=1000), \
                 patch.object(runner, "ORIGINAL_WORKSPACE", originals), \
                 patch.object(runner, "load_selection", return_value=fake_selection), \
                 patch.object(runner, "inputs", return_value=({}, {}, hashes, plan)), \
                 patch.object(runner, "TraceCapture", return_value=capture), \
                 patch.object(runner, "passive_fit_capture", return_value=nullcontext()):
                if parity_failure or missing_second_preflight:
                    with self.assertRaises(ValueError):
                        runner.run(root, clip)
                else:
                    runner.run(root, clip)
                receipt = runner.read(root / clip / "execution_receipt.json")
                self.assertIs(receipt["passed"], not (parity_failure or missing_second_preflight))
                self.assertFalse(receipt["candidate_algorithm_changed_relative_to_original"])
                if missing_second_preflight:
                    baseline.execute_baseline.assert_not_called()
                    return
                self.assertEqual(receipt["processed_frames"], 673)
                self.assertEqual(receipt["captured_pairs"], 56)
                self.assertEqual(receipt["trace_sha256"], runner.sha(root / clip / "trace.json"))
                self.assertEqual(receipt["parity_sha256"], runner.sha(root / clip / "parity.json"))
                self.assertIs(receipt["non_timing_journal_parity_passed"], not parity_failure)
                trace = runner.read(root / clip / "trace.json")
                self.assertEqual(trace["capture_pairs"], runner.fixed_pairs(clip))
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    runner.run(root, clip)
                fake_selection.validate_adapter_roundtrip.assert_called_once()

    def test_successful_receipt_json_roundtrip(self):
        self.exercise_run()

    def test_non_timing_mismatch_preserved_failed_receipt(self):
        self.exercise_run(parity_failure=True)

    def test_both_preflights_required_before_any_full_run(self):
        self.exercise_run(missing_second_preflight=True)


if __name__ == "__main__":
    unittest.main()
