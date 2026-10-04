"""Generated/mocked scope and process tests; never import/run a real decoder or PVA."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_nuisance_origin_v1.py"
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / "scripts/run_nuisance_origin_v1.py"
spec = importlib.util.spec_from_file_location("nuisance_runner_test_subject", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Process:
    def __init__(self, code=0, live=False, after_poll=None):
        self.pid, self.returncode = 5000, None
        self.code, self.live, self.after_poll = code, live, after_poll
        self.finished = False

    def poll(self):
        if self.after_poll:
            self.after_poll()
        if self.live:
            return None
        self.returncode, self.finished = self.code, True
        return self.returncode


class BundleTests(unittest.TestCase):
    def fixture(self, root):
        plan = dict(frame_points={str(i): [{}] for i in runner.FRAMES})
        for name in runner.MEMBERS:
            (root / name).write_text("{}\n" if name.endswith(".json") else "# generated fixture\n")
        (root / "packet_plan.json").write_text(json.dumps(plan))
        freeze = dict(schema=runner.SCHEMA+".freeze",
                      files={name: runner.sha(root/name) for name in runner.MEMBERS})
        (root / "freeze.json").write_text(json.dumps(freeze))
        return freeze, plan

    def context(self, root):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(runner, "PATTERN", re.escape(str(root))))
        stack.enter_context(patch.object(runner, "__file__", str(root / "run_nuisance_origin_v1.py")))
        stack.enter_context(patch.object(runner.os, "geteuid", return_value=1000))
        stack.enter_context(patch.object(runner, "SAFETY_SHA", runner.sha(root / "batch_discovery_pair.py")))
        stack.enter_context(patch.object(runner, "load", return_value=SimpleNamespace(validate_plan=lambda plan: None)))
        return stack

    def test_exact_bounded_member_and_frame_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            freeze, plan = self.fixture(root)
            with self.context(root):
                actual, selected = runner.bundle(root, runner.sha(root / "freeze.json"))
            self.assertEqual(actual, freeze)
            self.assertEqual(selected, plan)
            self.assertEqual(len(runner.FRAMES), 25)
            self.assertEqual(runner.FRAMES, sorted(set(runner.FRAMES)))

    def test_wrong_caller_digest_never_imports_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fixture(root)
            with self.context(root), patch.object(runner, "load", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    runner.bundle(root, "0"*64)

    def test_changed_source_hash_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.fixture(root)
            (root / "nuisance_origin_capture_v1.py").write_text("# modified\n")
            with self.context(root), patch.object(runner, "load", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    runner.bundle(root, runner.sha(root / "freeze.json"))

    def test_extra_file_and_wrong_scope_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            freeze, plan = self.fixture(root)
            freeze["files"]["extra.avi"] = "0"*64
            (root / "freeze.json").write_text(json.dumps(freeze))
            with self.context(root):
                with self.assertRaisesRegex(ValueError, "inventory"):
                    runner.bundle(root, runner.sha(root / "freeze.json"))
            freeze, plan = self.fixture(root)
            plan["frame_points"].pop(str(runner.FRAMES[0]))
            (root / "packet_plan.json").write_text(json.dumps(plan))
            freeze["files"]["packet_plan.json"] = runner.sha(root / "packet_plan.json")
            (root / "freeze.json").write_text(json.dumps(freeze))
            with self.context(root):
                with self.assertRaisesRegex(ValueError, "frame scope"):
                    runner.bundle(root, runner.sha(root / "freeze.json"))

    def test_linked_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            freeze, _ = self.fixture(root)
            path = root / "nuisance_origin_capture_v1.py"
            source = root / "alternate.py"
            source.write_text(path.read_text())
            path.unlink()
            path.symlink_to(source)
            with self.context(root):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    runner.bundle(root, runner.sha(root / "freeze.json"))


class ExecutionTests(unittest.TestCase):
    def test_existing_output_stops_before_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "preflight.json").write_text("{\"existing\":true}\n")
            with patch.object(runner, "bundle", return_value=({}, {})), \
                 patch.object(runner, "dependencies", side_effect=AssertionError("must not run")):
                with self.assertRaisesRegex(ValueError, "existing result"):
                    runner.run(root, "a"*64, preflight=True)
            self.assertEqual((root / "preflight.json").read_text(), "{\"existing\":true}\n")

    def test_existing_replay_evidence_stops_before_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "capture.json").write_text("{}\n")
            with patch.object(runner, "bundle", return_value=({}, {})), \
                 patch.object(runner, "dependencies", side_effect=AssertionError("must not run")):
                with self.assertRaisesRegex(ValueError, "existing replay"):
                    runner.run(root, "a"*64)

    def test_dependency_failure_saved_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(runner, "bundle", return_value=({}, {})), \
                 patch.object(runner, "dependencies", side_effect=ValueError("frozen source differs")) as dep:
                with self.assertRaisesRegex(ValueError, "frozen source differs"):
                    runner.run(root, "a"*64, preflight=True)
            self.assertEqual(dep.call_count, 1)
            receipt = json.loads((root / "preflight.json").read_text())
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["algorithm_changed"])
            self.assertIn("frozen source differs", receipt["error"])


class SupervisorTests(unittest.TestCase):
    def supervise(self, root, *, process_factory=None, temperature=None, clock=None, bundle=None, validator=None):
        from contextlib import ExitStack
        calls, stopped, active = [], [], []
        def popen(command, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], runner.subprocess.DEVNULL)
            calls.append(command)
            name = "tests" if "unittest" in command else command[-1][2:]
            process = process_factory(name) if process_factory else Process()
            process.name, process.pid = name, 5000+len(calls)
            active[:] = [process]
            return process
        safety = SimpleNamespace(temperatures=lambda: temperature(active) if temperature else {"cpu": 40},
                                 stop_owned=lambda p: stopped.append(p.pid))
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "bundle", side_effect=bundle or (lambda *args: ({}, {}))))
            stack.enter_context(patch.object(runner, "load", return_value=safety))
            stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(runner.time, "sleep"))
            stack.enter_context(patch.object(runner.signal, "signal", return_value=None))
            stack.enter_context(patch.object(runner, "read", return_value={"passed": True}))
            if clock:
                stack.enter_context(patch.object(runner.time, "monotonic", side_effect=clock))
            if validator:
                stack.enter_context(patch.object(runner, "validate_child", side_effect=validator, create=True))
            try:
                runner.batch(root, "a"*64)
            except BaseException as exc:
                return calls, stopped, exc, json.loads((root / "batch_status.json").read_text())
        return calls, stopped, None, json.loads((root / "batch_status.json").read_text())

    def test_three_sequential_processes_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(), validator=lambda *a: {})
        self.assertIsNone(error)
        self.assertTrue(status["complete"] and status["passed"])
        self.assertEqual(len(calls), 3)
        self.assertEqual([p["name"] for p in status["phases"]], ["tests", "preflight", "run"])
        self.assertEqual(stopped, [])
        self.assertEqual(status["automatic_retries"], 0)

    def test_failure_stops_only_owned_child_and_does_not_launch_next(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(), process_factory=lambda _: Process(code=1))
        self.assertIsNotNone(error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stopped, [5001])
        self.assertFalse(status["passed"])

    def test_start_temperature_prevents_any_child(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(), temperature=lambda _: {"cpu": 65})
        self.assertIsNotNone(error)
        self.assertEqual(calls, [])
        self.assertEqual(stopped, [])

    def test_live_temperature_stops_only_owned_process(self):
        def temperature(active):
            return {"cpu": 75 if active else 40}
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(),
                temperature=temperature, process_factory=lambda _: Process(live=True))
        self.assertIsNotNone(error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stopped, [5001])

    def test_hot_final_exit_is_not_hidden_by_finished_poll(self):
        def temperature(active):
            return {"cpu": 75 if active and active[0].name == "run" and active[0].finished else 40}
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(),
                temperature=temperature, validator=lambda *a: {})
        self.assertIsNotNone(error, "post-exit75C must reject even if child immediately reports complete")
        self.assertFalse(status["passed"])

    def test_immediately_finished_child_still_must_meet_phase_deadline(self):
        now = [0]
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(), clock=lambda: now[0],
                process_factory=lambda name: Process(after_poll=lambda: now.__setitem__(0, now[0]+1801)),
                validator=lambda *a: {})
        self.assertIsNotNone(error, "post-exit phase deadline must be enforced")
        self.assertEqual(len(calls), 1)

    def test_total_deadline_prevents_a_new_launch(self):
        now, count = [0], [0]
        def bundle(*args):
            count[0] += 1
            if count[0] == 2:
                now[0] = 2401
            return {}, {}
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(), clock=lambda: now[0],
                bundle=bundle, validator=lambda *a: {})
        self.assertIsNotNone(error)
        self.assertEqual(calls, [])

    def test_passed_flag_cannot_replace_bound_child_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, stopped, error, status = self.supervise(Path(directory).resolve(),
                validator=lambda *a: (_ for _ in ()).throw(ValueError("child provenance differs")))
        self.assertIsNotNone(error, "supervisor must validate child identities/artifacts, not just passed")
        self.assertEqual(len(calls), 2)
        self.assertFalse(status["passed"])


if __name__ == "__main__":
    unittest.main()
