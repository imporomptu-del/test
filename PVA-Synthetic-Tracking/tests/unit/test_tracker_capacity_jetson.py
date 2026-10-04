import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jetson = load("test_jetson_checker", "check_tracker_capacity_jetson.py")
base = load("test_base_checker", "check_tracker_capacity_shadow.py")


def runtime(threads=12):
    return dict(numpy="1.26.1", opencv="4.10.0", opencv_threads=threads,
                blas=[dict(threads=12, path="pinned", sha256="pinned")], affinity=list(range(12)),
                clock_ticks=100, thread_environment=dict(OPENBLAS_NUM_THREADS=None))


class Process:
    pid = 424242

    def __init__(self, results):
        self.results, self.returncode = iter(results), None

    def poll(self):
        result = next(self.results, self.returncode)
        if result is not None:
            self.returncode = result
        return result


class JetsonTests(unittest.TestCase):
    def make_bundle(self, workspace):
        files = {}
        for name in ("check_tracker_capacity_shadow.py", "check_tracker_capacity_jetson.py", "batch_discovery_pair.py"):
            shutil.copyfile(ROOT / "scripts" / name, workspace/name)
            files[name] = jetson.sha(workspace/name)
        freeze = dict(schema=jetson.FREEZE_SCHEMA, baseline_only=True, files=files,
                      sources={c: dict(frames=673, source_sha256=base.SOURCE_SHA[c],
                          journal_sha256=base.FILES[c]["run/frames.jsonl"]) for c in base.CLIPS})
        (workspace/"freeze.json").write_text(json.dumps(freeze))
        return freeze

    def test_bundle_pins_base_and_only_metadata_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            self.make_bundle(workspace)
            freeze_sha = jetson.sha(workspace/"freeze.json")
            module, _, hashes = jetson.bundle(workspace, freeze_sha)
            self.assertEqual(module.EVIDENCE, jetson.TRACE)
            self.assertEqual(len(hashes), 4)
            with (workspace/"check_tracker_capacity_shadow.py").open("a") as stream:
                stream.write("\n# tamper\n")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                jetson.bundle(workspace, freeze_sha)

    def test_bundle_wrong_source_and_media_filename_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            frozen = self.make_bundle(workspace)
            frozen["sources"]["0170"]["frames"] = 672
            (workspace/"freeze.json").write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "source metadata"):
                jetson.bundle(workspace, jetson.sha(workspace/"freeze.json"))
            frozen = self.make_bundle(workspace)
            frozen["files"]["clip.avi"] = "0"*64
            (workspace/"freeze.json").write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "unsafe bundle"):
                jetson.bundle(workspace, jetson.sha(workspace/"freeze.json"))

    def test_caller_freeze_hash_checked_before_any_bundle_module_import(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            self.make_bundle(workspace)
            with patch.object(jetson, "load_path", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(ValueError, "caller-bound freeze"):
                    jetson.bundle(workspace, "0"*64)

    def test_runtime_rejects_threads_and_scalar_math_mismatch(self):
        actual = runtime()
        expected_log = float.fromhex("0x1.193ea7aad030ap+1")/2
        with patch.object(jetson.sys, "version_info", (3, 10)), \
             patch.object(jetson.sys, "platform", "linux"), \
             patch.object(jetson.platform, "machine", return_value="aarch64"), \
             patch.object(jetson.math, "log1p", return_value=expected_log):
            jetson.check_runtime(actual, actual)
            jetson.check_runtime(runtime(2), actual, after=True)
            wrong = runtime()
            wrong["blas"][0]["threads"] = 1
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                jetson.check_runtime(wrong, actual)
            with patch.object(jetson.math, "log1p", return_value=0.0):
                with self.assertRaisesRegex(ValueError, "scalar"):
                    jetson.check_runtime(actual, actual)

    def supervisor(self, workspace, processes, temperatures=None, clock=None, monotonic=None):
        commands, stopped = [], []
        processes = iter(processes)
        def popen(command, **kwargs):
            commands.append(command)
            self.assertTrue(kwargs["start_new_session"])
            return next(processes)
        safety = SimpleNamespace(temperatures=lambda: {"cpu": 40, "tj": 40}, stop_owned=lambda p: stopped.append(p.pid))
        if temperatures is not None:
            safety.temperatures = temperatures
        with patch.object(jetson, "workspace_guard", side_effect=lambda p: Path(p)), \
             patch.object(jetson, "bundle", return_value=(base, {}, {})), \
             patch.object(jetson, "load_path", return_value=safety), \
             patch.object(jetson, "verify_child_receipt", return_value={}), \
             patch.object(jetson, "clock_policy", side_effect=clock or (lambda: {"clock": 1})), \
             patch.object(jetson.subprocess, "Popen", side_effect=popen), \
             patch.object(jetson.time, "sleep"), \
             patch.object(jetson.time, "monotonic", side_effect=monotonic or (lambda: 1.0)):
            result = jetson.supervise(workspace, "f"*64)
        return result, commands, stopped

    def test_supervisor_orders_preflight_and_single_full_clips(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            result, commands, stopped = self.supervisor(workspace, [Process([0]), Process([0]), Process([0])])
            self.assertTrue(result["passed"])
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][-1], "--preflight")
            self.assertIn("--freeze-sha256", commands[0])
            self.assertEqual(commands[0][commands[0].index("--freeze-sha256")+1], "f"*64)
            self.assertEqual(result["freeze_sha256"], "f"*64)
            self.assertEqual(commands[1][-2:], ["--clip", "0170"])
            self.assertEqual(commands[2][-2:], ["--clip", "0240"])
            self.assertEqual(stopped, [])
            self.assertEqual(json.loads((workspace/"batch_status.json").read_text()), result)
            self.assertFalse(result["candidate_implemented"])

    def test_supervisor_first_failed_clip_stops_next(self):
        with tempfile.TemporaryDirectory() as temp:
            result, commands, _ = self.supervisor(Path(temp).resolve(), [Process([0]), Process([1])])
            self.assertFalse(result["passed"])
            self.assertEqual(len(commands), 2)
            self.assertIn("failed phase run_0170", result["error"])

    def test_supervisor_rejects_65_start_and_stops_at75(self):
        with tempfile.TemporaryDirectory() as temp:
            result, commands, _ = self.supervisor(Path(temp).resolve(), [], temperatures=lambda: {"cpu": 65})
            self.assertFalse(result["passed"])
            self.assertEqual(commands, [])
        with tempfile.TemporaryDirectory() as temp:
            values = iter([{"cpu": 40}, {"cpu": 75}])
            result, commands, stopped = self.supervisor(Path(temp).resolve(), [Process([None])], temperatures=lambda: next(values))
            self.assertFalse(result["passed"])
            self.assertEqual(stopped, [424242])
            self.assertEqual(len(commands), 1)
        with tempfile.TemporaryDirectory() as temp:
            values = iter([{"cpu": 40}, {"cpu": 75}])
            result, _, _ = self.supervisor(Path(temp).resolve(), [Process([0])], temperatures=lambda: next(values))
            self.assertFalse(result["passed"])
            self.assertIn("after child exit", result["error"])

    def test_total_deadline_and_clock_change_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = iter([0, 0, 1801, 1801])
            result, commands, _ = self.supervisor(Path(temp).resolve(), [], monotonic=lambda: next(clock))
            self.assertFalse(result["passed"])
            self.assertEqual(commands, [])
            self.assertIn("batch deadline", result["error"])
        with tempfile.TemporaryDirectory() as temp:
            result, _, _ = self.supervisor(Path(temp).resolve(), [Process([0]), Process([0]), Process([0])],
                                           clock=iter([{"clock": 1}, {"clock": 2}]).__next__)
            self.assertFalse(result["passed"])
            self.assertFalse(result["clock_controls_unchanged"])

    def test_preflight_receipt_then_first_failure_gates_second_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            launches = {c: {"configuration": {"opencv_threads": 2}} for c in base.CLIPS}
            receipts = {c: {"runtime_before": runtime(), "runtime_after": runtime(2)} for c in base.CLIPS}
            profiler = SimpleNamespace(runtime_info=lambda: runtime())
            with patch.object(jetson, "workspace_guard", side_effect=lambda p: Path(p)), \
                 patch.object(jetson, "bundle", side_effect=lambda p, freeze: (base, {}, {})), \
                 patch.object(jetson, "dependencies", return_value=(launches, receipts, {}, profiler)), \
                 patch.object(jetson, "check_runtime"), patch.object(jetson, "clock_policy", return_value={}), \
                 patch.object(base, "run_clip", return_value={"passed": False, "exact_archive_frames": 15}) as run:
                pre = jetson.child(workspace, freeze_sha="f"*64, preflight=True)
                self.assertTrue(pre["passed"])
                first = jetson.child(workspace, freeze_sha="f"*64, clip="0170")
                self.assertFalse(first["passed"])
                second = jetson.child(workspace, freeze_sha="f"*64, clip="0240")
                self.assertFalse(second["passed"])
                self.assertIn("0170 full baseline parity", second["error"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(json.loads((workspace/"0170/result.json").read_text()), first)
                with self.assertRaisesRegex(ValueError, "existing child receipt"):
                    jetson.child(workspace, freeze_sha="f"*64, preflight=True)

    def test_supervisor_verifies_receipt_full_counts_and_state_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            (workspace/"0170").mkdir()
            (workspace/"preflight.json").write_text('{"test":"preflight"}')
            state = workspace/"0170/0170_state_hashes.jsonl"
            state.write_text('{"test":"state"}\n')
            inputs = {str(jetson.TRACE/c/name): expected for c in base.CLIPS for name, expected in base.FILES[c].items()}
            inputs[str(workspace/"preflight.json")] = jetson.sha(workspace/"preflight.json")
            inputs[str(workspace/"freeze.json")] = "f"*64
            bundle_hashes = {str(workspace/"freeze.json"): "f"*64}
            stats = dict(geometry_calls=0, geometry_fallbacks=0, batch_fallbacks=0, innovation_fallbacks=0,
                         batch_calls=2, batch_track_rows=5, innovation_tracks=5)
            replay = dict(passed=True, first_difference=None, attempted_frames=673, exact_archive_frames=673,
                          dual_state_exact_frames=673, derived_learning_exact_frames=673,
                          state_hashes_file=str(state), state_hashes_sha256=jetson.sha(state), adapter_runs=[stats, stats])
            receipt = dict(schema=jetson.SCHEMA+".run", passed=True, workspace=str(workspace), clip="0170", freeze_sha256="f"*64,
                           inputs_unchanged_after_check=True, clock_controls_unchanged=True,
                           candidate_implemented=False, detector_replayed=False, source_media_opened=False,
                           native_build_performed=False, original_native_binaries=True,
                           tracking_transformed_sha256=base.METHOD_SHA, inputs_sha256=inputs,
                           preflight_sha256=inputs[str(workspace/"preflight.json")], replay=replay)
            path = workspace/"0170/result.json"
            path.write_text(json.dumps(receipt))
            artifacts = jetson.verify_child_receipt(base, workspace, "run_0170", bundle_hashes)
            self.assertEqual(set(artifacts), {str(path), str(state)})
            replay["exact_archive_frames"] = 672
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "complete exact"):
                jetson.verify_child_receipt(base, workspace, "run_0170", bundle_hashes)
            replay["exact_archive_frames"] = 673
            path.write_text(json.dumps(receipt))
            state.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                jetson.verify_child_receipt(base, workspace, "run_0170", bundle_hashes)


if __name__ == "__main__":
    unittest.main()
