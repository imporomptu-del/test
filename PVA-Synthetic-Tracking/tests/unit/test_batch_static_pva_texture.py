import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/batch_static_pva_texture.py"
spec = importlib.util.spec_from_file_location("static_batch_tested", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class StaticBatchTests(unittest.TestCase):
    def test_four_fresh_phase_commands_without_preflight_or_media(self):
        for case, mode in batch.PHASES:
            cmd = batch.command(Path("/tmp/workspace"), "a"*64, case, mode)
            self.assertIn("-I", cmd)
            self.assertEqual("--trace" in cmd, mode == "trace")
            self.assertNotIn("--video", cmd)
        self.assertEqual(len(batch.PHASES), 4)
        self.assertEqual(batch.EXECUTION["automatic_retries"], 0)

    def test_no_unknown_scope(self):
        with self.assertRaises(ValueError):
            batch.command(Path("/tmp/workspace"), "a"*64, "other", "base")

    def test_canonical_order_independent(self):
        self.assertEqual(batch.canonical_sha({"a": 1, "b": 2}), batch.canonical_sha({"b": 2, "a": 1}))

    def test_exact_parity_not_tolerant(self):
        def make(mode, value):
            payload = {"points": [value]}
            return dict(case="bridge", mode=mode, canonical_nontiming=payload,
                        canonical_nontiming_sha256=batch.canonical_sha(payload))
        self.assertTrue(batch.compare_pair(make("base", 1.), make("trace", 1.))["passed"])
        self.assertFalse(batch.compare_pair(make("base", 1.), make("trace", 1.+1e-12))["passed"])

    def test_child_result_gate(self):
        result = dict(schema="seaqr.static-pva-texture.v1", completed=True, passed_integrity=True,
                      case="texture", mode="base", input_sha256={"freeze_sha256": "a"*64},
                      canonical_nontiming={}, canonical_nontiming_sha256=batch.canonical_sha({}))
        batch.validate_result(result, "texture", "base", "a"*64)
        for field, value in (("completed", False), ("passed_integrity", False), ("case", "bridge"),
                             ("canonical_nontiming_sha256", "b"*64)):
            with self.assertRaises(ValueError):
                batch.validate_result(dict(result, **{field: value}), "texture", "base", "a"*64)

    def test_temperature_and_deadline_boundaries(self):
        batch.guard({"cpu-thermal": 64.999, "tj-thermal": 64.}, 0, 0, 0, starting=True)
        batch.guard({"cpu-thermal": 74.999}, 899.9, 0, 0)
        for temps, now, phase, starting in (({},0,0,False), ({"cpu":float("nan")},0,0,False),
                ({"cpu":65},0,0,True), ({"cpu":75},0,0,False), ({"cpu":40},900,0,False),
                ({"cpu":40},3600,3500,False)):
            with self.assertRaises(ValueError):
                batch.guard(temps, now, 0, phase, starting)

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    batch.read(path)

    def test_frozen_contract_rejects_camera_or_extra_phase(self):
        freeze = dict(schema="static_pva_texture.v1", files={n: "a"*64 for n in batch.FILES},
                      cases=["bridge", "texture"], modes=["base", "trace"], shape_hw=[512,640],
                      photometric_workspace=batch.PHOTO_WORKSPACE, photometric_runner_sha256=batch.PHOTO_RUNNER_SHA,
                      candidate=batch.CANDIDATE, no_preflight_pva_calls=True, execution=batch.EXECUTION)
        freeze["files"]["batch_discovery_pair.py"] = batch.SAFETY_SHA
        batch.validate_freeze(freeze)
        with self.assertRaises(ValueError):
            batch.validate_freeze(dict(freeze, cases=["bridge", "texture", "camera"]))
        with self.assertRaises(ValueError):
            batch.validate_freeze(dict(freeze, execution=dict(batch.EXECUTION, phases=5)))


if __name__ == "__main__":
    unittest.main()
