"""Generated supervisor checks; no remote/runtime/media access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("maturity_batch", ROOT/"scripts/batch_tracker_maturity_shadow_v1.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Supervisor(unittest.TestCase):
    def test_three_fixed_isolated_fresh_phases(self):
        self.assertEqual([p[0] for p in b.PHASES], ["preflight", "run_0170", "run_0240"])
        self.assertEqual([p[2] for p in b.PHASES], [300, 1800, 1800])
        for phase, args, _ in b.PHASES:
            cmd = b.command(Path("/tmp/scoped"), "a"*64, phase)
            self.assertEqual(cmd[1:3], ["-I", "-u"])
            self.assertEqual(cmd[-len(args):], list(args))
        with self.assertRaises(ValueError): b.command(Path("/tmp/scoped"), "a"*64, "run_0001")

    def test_temperatures_and_deadlines(self):
        b.guard({"CPU":64.9}, 0, 0, 0, 300, starting=True)
        b.guard({"CPU":74.9}, 1799, 0, 0, 1800)
        for temps, now, starting in (({"CPU":65},0,True), ({"CPU":75},0,False),
                ({"CPU":float("nan")},0,False), ({},0,False), ({"CPU":50},1800,False)):
            with self.assertRaises(ValueError): b.guard(temps, now, 0, 0, 1800, starting=starting)
        with self.assertRaises(ValueError): b.guard({"CPU":50},3900,0,3899,1800)

    def fake(self, directory, fault=None):
        created, stopped, checked = [], [], []
        def write_new(path, value):
            with Path(path).open("x") as stream: json.dump(value, stream, allow_nan=False)
        def verify(hashes):
            for name, digest in hashes.items():
                b.require(b.sha(name) == digest, "artifact changed")
        base = types.SimpleNamespace(write_new=write_new, verify_unchanged=verify)
        def validated(workspace, digest, phase):
            if fault == "integrity": raise ValueError("invalid child")
            checked.append(phase)
            p = workspace/(phase+"_receipt.json")
            write_new(p, dict(passed=True, scientific_guard_passed=False))
            return {str(p): b.sha(p)}
        runner = types.SimpleNamespace(validate_child_receipt=validated)
        safety = types.SimpleNamespace(temperatures=lambda: {"CPU":50., "Tj":50.},
                                       stop_owned=lambda process: stopped.append(process.pid))
        def popen(cmd, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], b.subprocess.DEVNULL)
            process = types.SimpleNamespace(pid=1000 if fault == "duplicate_pid" else 1000+len(created),
                                            returncode=1 if fault == "exit" else 0)
            process.poll = lambda: None if fault == "hot_live" else process.returncode
            created.append(process)
            return process
        with patch.object(b, "load", return_value=(base,runner,safety,{})), \
             patch.object(b.subprocess, "Popen", side_effect=popen), \
             patch.object(b, "guard", side_effect=ValueError("hot") if fault == "hot_start" else None) as guard:
            if fault in ("hot_live", "hot_exit"): guard.side_effect = [None,ValueError("hot")]
            status = b.run(directory, "a"*64)
        return status, created, stopped, checked

    def test_scientific_rejection_still_completes_all_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, created, stopped, checked = self.fake(Path(tmp))
            self.assertTrue(s["complete"] and s["execution_passed"])
            self.assertFalse(s["scientific_improvement_claimed"])
            self.assertEqual((len(created), stopped, s["not_run"]), (3, [], []))
            self.assertEqual(checked, [p[0] for p in b.PHASES])
            self.assertTrue(s["bundle_and_child_artifacts_unchanged"])

    def test_backend_or_integrity_failure_no_retry(self):
        for fault in ("exit", "integrity"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                s, created, stopped, _ = self.fake(Path(tmp), fault)
                self.assertFalse(s["complete"] or s["execution_passed"])
                self.assertEqual((len(created), len(s["not_run"]), stopped), (1,2,[1000]))
                self.assertTrue(s["error"])

    def test_heat_stops_only_owned_child_live_or_after_exit(self):
        for fault in ("hot_live", "hot_exit"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                s, created, stopped, _ = self.fake(Path(tmp), fault)
                self.assertEqual((len(created), stopped), (1,[1000]))
                self.assertFalse(s["execution_passed"])
                self.assertTrue(s["owned_child_stopped"])

    def test_too_warm_start_launches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, created, stopped, _ = self.fake(Path(tmp), "hot_start")
            self.assertEqual((created, stopped, len(s["not_run"])), ([],[],3))

    def test_unique_process_identity_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, created, stopped, _ = self.fake(Path(tmp), "duplicate_pid")
            self.assertEqual(len(created),3)
            self.assertFalse(s["execution_passed"])

    def test_no_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/"preflight.json").write_text("retained")
            with patch.object(b, "load", return_value=(None,None,None,{})):
                with self.assertRaisesRegex(ValueError,"no overwrite"): b.run(d,"a"*64)
            self.assertEqual((d/"preflight.json").read_text(),"retained")

    def test_strict_json_duplicate_nonfinite_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"input.json"
            for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
                p.write_text(value)
                with self.assertRaises(ValueError): b.read(p)
            p.write_text('{}')
            link=Path(tmp)/"link.json"
            link.symlink_to(p)
            with self.assertRaises(ValueError): b.read(link)


if __name__ == "__main__": unittest.main()
