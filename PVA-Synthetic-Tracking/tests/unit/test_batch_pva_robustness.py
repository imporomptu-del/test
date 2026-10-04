"""Supervisor contract and non-destructive failure tests without Jetson access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("robustness_batch", ROOT / "scripts/batch_pva_robustness.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class SupervisorTests(unittest.TestCase):
    def cases(self):
        return [{"id": "case_"+str(i)} for i in range(26)]

    def row(self, mode="base", depth=2):
        canonical = {"effective_motion_configuration": {"pyramid_levels": depth}, "fit": {"accepted": False}}
        return dict(schema="seaqr.pva-robustness.v1", completed=True, passed_integrity=True,
                    case="case_0", depth=depth, mode=mode, input_sha256={"freeze_sha256": "a"*64},
                    canonical_nontiming=canonical, canonical_nontiming_sha256=batch.canonical_sha(canonical))

    def test_inventory_is_104_explicit_fresh_roles(self):
        phases = batch.phases(self.cases())
        self.assertEqual(len(phases), 104)
        self.assertEqual(len(set(phases)), 104)
        self.assertEqual(phases[:4], [("case_0", 4, "base"), ("case_0", 4, "trace"),
                                      ("case_0", 2, "base"), ("case_0", 2, "trace")])

    def test_case_inventory_rejects_duplicates_short_and_paths(self):
        for cases in (self.cases()[:-1], self.cases()[:-1]+[{"id": "case_0"}],
                      self.cases()[:-1]+[{"id": "../escape"}]):
            with self.assertRaises(ValueError):
                batch.phases(cases)

    def test_command_isolated_no_shell_one_case(self):
        cmd = batch.command(Path("/tmp/scoped"), "a"*64, "case_0", 2, "trace")
        self.assertEqual(cmd[1], "-I")
        self.assertEqual(cmd[-3:], ["--depth", "2", "--trace"])
        self.assertEqual(cmd.count("--case"), 1)
        self.assertNotIn("--trace", batch.command(Path("/tmp/scoped"), "a"*64, "case_0", 4, "base"))

    def test_bad_command_roles_rejected(self):
        for case, depth, mode in (("../escape", 2, "base"), ("ok", 3, "base"), ("ok", 2, "retry")):
            with self.assertRaises(ValueError):
                batch.command(Path("/tmp/scoped"), "a"*64, case, depth, mode)

    def test_scientific_rejection_is_successful_execution(self):
        batch.validate_result(self.row(), "case_0", 2, "base", "a"*64)

    def test_result_identity_and_canonical_tampering_rejected(self):
        for key, value in (("case", "other"), ("depth", 4), ("mode", "trace"),
                           ("completed", False), ("passed_integrity", False),
                           ("canonical_nontiming_sha256", "0"*64)):
            row = self.row()
            row[key] = value
            with self.assertRaises(ValueError):
                batch.validate_result(row, "case_0", 2, "base", "a"*64)

    def test_parity_exact_not_approximate(self):
        self.assertTrue(batch.compare_pair(self.row(), self.row("trace"))["passed"])
        trace = self.row("trace")
        trace["canonical_nontiming"]["fit"]["accepted"] = True
        trace["canonical_nontiming_sha256"] = batch.canonical_sha(trace["canonical_nontiming"])
        self.assertFalse(batch.compare_pair(self.row(), trace)["passed"])

    def test_parity_wrong_roles_or_hashes_fail(self):
        with self.assertRaises(ValueError):
            batch.compare_pair(self.row(), self.row("trace", 4))
        trace = self.row("trace")
        trace["canonical_nontiming_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            batch.compare_pair(self.row(), trace)

    def test_temperature_and_deadlines(self):
        batch.guard({"CPU": 64.9, "Tj": 64}, 1, 0, 0, True)
        batch.guard({"CPU": 74.9}, 899, 0, 0)
        for temps, now, began, phase, starting in (({"CPU": 65}, 1, 0, 0, True),
                ({"CPU": 75}, 1, 0, 0, False), ({"CPU": float("nan")}, 1, 0, 0, False),
                ({}, 1, 0, 0, False), ({"CPU": 50}, 900, 0, 0, False),
                ({"CPU": 50}, 3600, 0, 3599, False)):
            with self.assertRaises(ValueError):
                batch.guard(temps, now, began, phase, starting)

    def test_fresh_completed_inventory(self):
        phases = batch.phases(self.cases())
        entries = [dict(case=c, depth=d, mode=m, pid=i+1, returncode=0)
                   for i, (c,d,m) in enumerate(phases)]
        batch.completed_inventory(entries, phases)
        entries[-1]["pid"] = 1
        with self.assertRaises(ValueError):
            batch.completed_inventory(entries, phases)
        entries[-1]["pid"] = 104
        entries[-1]["returncode"] = 1
        with self.assertRaises(ValueError):
            batch.completed_inventory(entries, phases)

    def test_strict_json_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.json"
            for body in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(body)
                with self.assertRaises(ValueError):
                    batch.read(path)
            path.write_text('{"x":1}')
            self.assertEqual(batch.read(path), {"x":1})
            link = Path(tmp)/"linked.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                batch.read(link)

    def test_existing_evidence_not_overwritten(self):
        class Probe:
            inventory = lambda _: [{"id":"case_"+str(i)} for i in range(26)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"case_0_depth4_base.json"
            path.write_text("retained")
            with patch.object(batch, "load", return_value=({"source_sha256":{}}, Probe(), None)):
                with self.assertRaisesRegex(ValueError, "no retry or overwrite"):
                    batch.run(Path(tmp), Path(tmp)/"freeze.json", "a"*64)
            self.assertEqual(path.read_text(), "retained")
            self.assertFalse((Path(tmp)/"batch.lock").exists())

    def fake_run(self, directory, fault=None):
        frozen = {"source_sha256": {}}
        probe = types.SimpleNamespace(inventory=self.cases, bundle=lambda *args: frozen)
        stopped, created = [], []
        safety = types.SimpleNamespace(temperatures=lambda: {"CPU": 50., "Tj": 50.},
                                       stop_owned=lambda child: stopped.append(child.pid))
        def popen(cmd, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], batch.subprocess.DEVNULL)
            index = len(created)
            child = types.SimpleNamespace(pid=index+1000, returncode=1 if fault == "exit" else 0)
            child.poll = lambda: None if fault == "temperature" else child.returncode
            created.append(child)
            case, depth = cmd[cmd.index("--case")+1], int(cmd[cmd.index("--depth")+1])
            mode = "trace" if "--trace" in cmd else "base"
            row = self.row(mode, depth)
            row["case"] = case
            if fault == "integrity":
                row["passed_integrity"] = False
            if fault == "parity" and index == 1:
                row["canonical_nontiming"]["fit"]["accepted"] = True
                row["canonical_nontiming_sha256"] = batch.canonical_sha(row["canonical_nontiming"])
            (directory/(batch.phase_name(case, depth, mode)+".json")).write_text(json.dumps(row))
            return child
        with patch.object(batch, "load", return_value=(frozen, probe, safety)), \
             patch.object(batch.subprocess, "Popen", side_effect=popen), \
             patch.object(batch, "guard", side_effect=ValueError("thermal") if fault == "start" else None) as guarded:
            if fault in ("temperature", "after_exit_temperature"):
                guarded.side_effect = [None, ValueError("thermal")]
            try:
                result = batch.run(directory, directory/"freeze.json", "a"*64)
            except ValueError:
                result = batch.read(directory/"batch_status.json")
        return result, created, stopped

    def test_all_scientific_rejections_still_complete_104(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, created, stopped = self.fake_run(Path(tmp))
            self.assertTrue(status["complete"])
            self.assertTrue(status["parity_passed"])
            self.assertEqual(len(created), 104)
            self.assertEqual(stopped, [])
            self.assertEqual(status["not_run"], [])
            self.assertEqual(len(batch.read(Path(tmp)/"parity.json")["cases"]), 52)

    def test_no_retry_on_child_or_integrity_failure(self):
        for fault in ("exit", "integrity"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                status, created, stopped = self.fake_run(Path(tmp), fault)
                self.assertFalse(status["complete"])
                self.assertEqual(len(created), 1)
                self.assertEqual(len(status["not_run"]), 103)
                self.assertTrue(status["error"])
                self.assertTrue((Path(tmp)/"case_0_depth4_base.json").exists())

    def test_temperature_stop_only_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, created, stopped = self.fake_run(Path(tmp), "temperature")
            self.assertFalse(status["complete"])
            self.assertTrue(status["owned_child_stopped"])
            self.assertEqual(stopped, [created[0].pid])
            self.assertEqual(len(created), 1)

    def test_start_failure_launches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, created, stopped = self.fake_run(Path(tmp), "start")
            self.assertFalse(status["complete"])
            self.assertEqual((created, stopped), ([], []))
            self.assertEqual(len(status["not_run"]), 104)

    def test_parity_failure_retained_and_later_cases_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, created, stopped = self.fake_run(Path(tmp), "parity")
            self.assertTrue(status["complete"])
            self.assertTrue(status["execution_passed"])
            self.assertFalse(status["parity_passed"])
            self.assertEqual(len(created), 104)
            parity = batch.read(Path(tmp)/"parity.json")
            self.assertEqual(sum(not p["passed"] for p in parity["cases"]), 1)

    def test_after_exit_temperature_is_checked_before_next_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, created, stopped = self.fake_run(Path(tmp), "after_exit_temperature")
            self.assertFalse(status["complete"])
            self.assertEqual(len(created), 1)
            self.assertEqual(len(status["not_run"]), 103)
            self.assertEqual(stopped, [created[0].pid])


if __name__ == "__main__":
    unittest.main()
