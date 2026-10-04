"""Bounded supervisor tests; no VPI imports, devices or media."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reverse_batch", ROOT/"scripts/batch_pva_reverse_ownership.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Supervisor(unittest.TestCase):
    def cases(self):
        return [{"id": case} for case in b.CASES]

    def row(self, arm="shared", mode="base", case=None):
        canonical = {"effective_motion_configuration": {"pyramid_levels": 2}, "fit": {"accepted": False}}
        return dict(schema="seaqr.pva-reverse-ownership.v1", completed=True, passed_integrity=True,
                    case=case or b.CASES[0], arm=arm, depth=2, mode=mode,
                    input_sha256={"freeze_sha256": "a"*64}, canonical_nontiming=canonical,
                    canonical_nontiming_sha256=b.canonical_sha(canonical))

    def test_exact30_order_and_no_plain_trace(self):
        p = b.phases(self.cases())
        self.assertEqual(len(p), 30)
        self.assertEqual(len(set(p)), 30)
        self.assertEqual(p[:5], [(b.CASES[0], a, m) for a, m in b.ROLES])
        self.assertNotIn(("plain", "trace"), b.ROLES)

    def test_reordered_missing_unknown_duplicate_cases_fail(self):
        for cases in (self.cases()[::-1], self.cases()[:-1],
                      self.cases()[:-1]+[{"id": "unknown"}], self.cases()[:-1]+self.cases()[:1]):
            with self.assertRaises(ValueError): b.phases(cases)

    def test_commands_isolated_exact_role(self):
        for arm, mode in b.ROLES:
            cmd = b.command(Path("/tmp/scoped"), "a"*64, b.CASES[0], arm, mode)
            self.assertEqual(cmd[1], "-I")
            self.assertEqual(cmd[cmd.index("--arm")+1], arm)
            self.assertEqual("--trace" in cmd, mode == "trace")
            self.assertEqual(cmd.count("--case"), 1)
        for case, arm, mode in (("../bad", "shared", "base"), (b.CASES[0], "plain", "trace"),
                                (b.CASES[0], "zero_reset", "base")):
            with self.assertRaises(ValueError): b.command(Path("/tmp/scoped"), "a"*64, case, arm, mode)

    def test_scientific_rejection_valid_execution(self):
        b.validate_result(self.row(), b.CASES[0], "shared", "base", "a"*64)

    def test_result_identity_tamper_fails(self):
        for key, value in (("depth", 4), ("arm", "copied"), ("case", "flat"),
                           ("mode", "trace"), ("passed_integrity", False),
                           ("canonical_nontiming_sha256", "0"*64)):
            r = self.row(); r[key] = value
            with self.assertRaises(ValueError): b.validate_result(r, b.CASES[0], "shared", "base", "a"*64)

    def test_exact_base_trace_and_plain_shared_checks(self):
        self.assertTrue(b.compare_pair(self.row(), self.row(mode="trace"))["passed"])
        self.assertTrue(b.compare_control(self.row("plain"), self.row())["passed"])
        changed = self.row(mode="trace")
        changed["canonical_nontiming"]["fit"]["accepted"] = True
        changed["canonical_nontiming_sha256"] = b.canonical_sha(changed["canonical_nontiming"])
        self.assertFalse(b.compare_pair(self.row(), changed)["passed"])
        changed["mode"] = "base"
        self.assertFalse(b.compare_control(self.row("plain"), changed)["passed"])
        with self.assertRaises(ValueError): b.compare_pair(self.row(), self.row("copied", "trace"))
        with self.assertRaises(ValueError): b.compare_control(self.row(), self.row())

    def test_guard_limits(self):
        b.guard({"CPU": 64.9}, 0, 0, 0, True)
        b.guard({"CPU": 74.9}, 899, 0, 0)
        for temps, now, start in (({"CPU": 65.}, 0, True), ({"CPU": 75.}, 0, False),
                ({"CPU": float("nan")}, 0, False), ({}, 0, False), ({"CPU": 50}, 900, False)):
            with self.assertRaises(ValueError): b.guard(temps, now, 0, 0, start)
        with self.assertRaises(ValueError): b.guard({"CPU": 50}, 3600, 0, 3599)

    def test_complete_unique_pid_inventory(self):
        p = b.phases(self.cases())
        rows = [dict(case=c, arm=a, mode=m, pid=i+1, returncode=0) for i, (c,a,m) in enumerate(p)]
        b.completed_inventory(rows, p)
        rows[-1]["pid"] = 1
        with self.assertRaises(ValueError): b.completed_inventory(rows, p)

    def test_existing_evidence_preserved(self):
        probe = types.SimpleNamespace(inventory=self.cases)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); path = d/(b.phase_name(b.CASES[0], "plain", "base")+".json")
            path.write_text("retained")
            with patch.object(b, "load", return_value=({}, probe, None)):
                with self.assertRaisesRegex(ValueError, "no retry or overwrite"): b.run(d, d/"freeze.json", "a"*64)
            self.assertEqual(path.read_text(), "retained")

    def run_fake(self, directory, fault=None):
        frozen = {"source_sha256": {}}
        probe = types.SimpleNamespace(inventory=self.cases, bundle=lambda *args: frozen)
        created, stopped = [], []
        safety = types.SimpleNamespace(temperatures=lambda: {"CPU": 50., "Tj": 50.},
                                       stop_owned=lambda child: stopped.append(child.pid))
        def popen(cmd, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], b.subprocess.DEVNULL)
            index = len(created)
            child = types.SimpleNamespace(pid=index+1000, returncode=1 if fault == "exit" else 0)
            child.poll = lambda: None if fault == "hot_live" else child.returncode
            created.append(child)
            case, arm = cmd[cmd.index("--case")+1], cmd[cmd.index("--arm")+1]
            mode = "trace" if "--trace" in cmd else "base"
            row = self.row(arm, mode, case)
            if fault == "integrity": row["passed_integrity"] = False
            if (fault == "control" and index in (1,2)) or (fault == "parity" and index == 2):
                row["canonical_nontiming"]["fit"]["accepted"] = True
                row["canonical_nontiming_sha256"] = b.canonical_sha(row["canonical_nontiming"])
            (directory/(b.phase_name(case, arm, mode)+".json")).write_text(json.dumps(row))
            return child
        with patch.object(b, "load", return_value=(frozen, probe, safety)), \
             patch.object(b.subprocess, "Popen", side_effect=popen), \
             patch.object(b, "guard", side_effect=ValueError("hot") if fault == "hot_start" else None) as guard:
            if fault in ("hot_live", "hot_exit"): guard.side_effect = [None, ValueError("hot")]
            try: status = b.run(directory, directory/"freeze.json", "a"*64)
            except ValueError: status = b.read(directory/"batch_status.json")
        return status, created, stopped

    def test_all_scientific_rejections_complete30(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, children, stopped = self.run_fake(Path(tmp))
            self.assertTrue(s["complete"] and s["execution_passed"] and s["parity_passed"] and s["control_parity_passed"])
            self.assertEqual(len(children), 30)
            self.assertEqual((stopped, s["not_run"]), ([], []))
            p = b.read(Path(tmp)/"parity.json")
            self.assertEqual((len(p["cases"]), len(p["controls"])), (12, 6))

    def test_no_retry_backend_and_integrity_failure(self):
        for fault in ("exit", "integrity"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                s, children, stopped = self.run_fake(Path(tmp), fault)
                self.assertFalse(s["complete"])
                self.assertEqual((len(children), len(s["not_run"])), (1,29))
                self.assertTrue(s["error"])

    def test_thermal_live_and_after_exit_stop_only_owned_child(self):
        for fault in ("hot_live", "hot_exit"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                s, children, stopped = self.run_fake(Path(tmp), fault)
                self.assertFalse(s["complete"])
                self.assertEqual((len(children), stopped), (1, [1000]))
                self.assertTrue(s["owned_child_stopped"])

    def test_too_warm_start_no_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            s, children, stopped = self.run_fake(Path(tmp), "hot_start")
            self.assertEqual((children, stopped, len(s["not_run"])), ([], [], 30))

    def test_mismatched_trace_or_control_does_not_retune_or_retry(self):
        for fault in ("parity", "control"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                s, children, stopped = self.run_fake(Path(tmp), fault)
                self.assertTrue(s["complete"] and s["execution_passed"])
                self.assertEqual(len(children), 30)
                self.assertFalse(s["parity_passed"] if fault == "parity" else s["control_parity_passed"])

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"input.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                p.write_text(text)
                with self.assertRaises(ValueError): b.read(p)
            p.write_text('{}')
            link = Path(tmp)/"link.json"; link.symlink_to(p)
            with self.assertRaises(ValueError): b.read(link)


if __name__ == "__main__":
    unittest.main()
