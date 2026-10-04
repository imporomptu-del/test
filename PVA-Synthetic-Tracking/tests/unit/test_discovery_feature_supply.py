"""Generated metadata/arrays/mocks only; no media, VPI or remote execution."""
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "scripts/run_discovery_feature_supply.py"
SPEC = importlib.util.spec_from_file_location("supply_test_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
DIGEST = "a" * 64


@dataclass(frozen=True)
class Config:
    feature_image_scale: float = 0.5
    pyramid_levels: int = 4
    pyramid_scale: float = 0.5
    harris_capacity_policy: str = "legacy_default"
    flow_status_policy: str = "legacy_default"
    minimum_accepted_features: int = 30
    minimum_grid_coverage: float = 0.2
    harris_strength: float = 0.5
    max_features: int = 1000
    arbitrary_other_gate: float = 3.0


def freeze_fixture():
    hashes = {"run_discovery_feature_supply.py": DIGEST, "tests.py": "b" * 64}
    return dict(schema="feature_supply.v1", candidate=dict(runner.CANDIDATE),
                baseline_workspace=str(runner.BASELINE_WORKSPACE), files=dict(hashes),
                sources={c: runner.source_spec(c) for c in runner.SOURCE_HASHES}), hashes


def pre_fixture(root):
    hashes, identities = {"fixed": DIGEST}, {"adapters": {"fixed": DIGEST}}
    return dict(schema=runner.SCHEMA + ".preflight", passed=True, workspace=str(root), clip="0170",
                source=runner.source_spec("0170"), candidate=dict(runner.CANDIDATE), input_sha256=hashes,
                probe_passed=True, detector_run=False, conversion={"passed": True},
                controls=[dict(name=n, passed=True, closed=True) for n in runner.CONTROL_NAMES], **identities), hashes, identities


def correspondence():
    x, y = np.meshgrid(np.linspace(180, 460, 6), np.linspace(180, 300, 5))
    p = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    return SimpleNamespace(previous_points=p, current_points=p + [4, -2], count=len(p),
        full_image_size=(640, 480), motion_image_size=(320, 240), backends=dict(runner.EXPECTED_BACKENDS),
        metrics=dict(harris_output=dict(capacity_policy="complete_grid", capacity=1271, capacity_exhausted=False),
                     detected_count=60, selected_count=40, minimum_accepted_features=30, minimum_grid_coverage=0.2))


class ScopeTests(unittest.TestCase):
    def test_adapter_preflight_json_roundtrip(self):
        actual = dict(source_transformation=dict(transformed_method_sha256="a" * 64),
            effective_motion_configuration=dict(exclusion_regions_xyxy=(),
                harris_capacity_policy="complete_grid", minimum_accepted_features=30))
        saved = json.loads(json.dumps(actual))
        self.assertNotEqual(actual, saved)
        runner.validate_adapter_roundtrip(actual, saved)
        for mutate in (lambda x: x["effective_motion_configuration"].update(minimum_accepted_features=29),
                       lambda x: x["source_transformation"].update(transformed_method_sha256="b" * 64),
                       lambda x: x["effective_motion_configuration"].update(exclusion_regions_xyxy=[[0, 0, 1, 1]])):
            bad = copy.deepcopy(saved)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_adapter_roundtrip(actual, bad)

    def test_exact_sources_workspace(self):
        self.assertEqual(set(runner.SOURCE_HASHES), {"0170", "0240"})
        good = "/tmp/seaqr_feature_supply_20260929_Ab12zQ"
        self.assertEqual(runner.scope_path(good), Path(good))
        for value in ("/tmp", good + "/0170", good + "x", good.replace("29_", "28_"), good + "/../x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.scope_path(value)
        with self.assertRaises(ValueError):
            runner.source_spec("0126")

    def test_no_root_and_no_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "WORKSPACE_PATTERN", re.escape(str(root))):
                with patch.object(runner.os, "geteuid", return_value=0), self.assertRaises(ValueError):
                    runner.workspace_guard(root, "0170", "run")
                with patch.object(runner.os, "geteuid", return_value=1000):
                    runner.workspace_guard(root, "0170", "preflight")
                    (root / "0170").mkdir()
                    for name in ("run", "execution_receipt.json", "preflight.json"):
                        path = root / "0170" / name
                        path.symlink_to(root / "missing")
                        with self.subTest(name=name), self.assertRaises(ValueError):
                            runner.workspace_guard(root, "0170", "preflight")
                        path.unlink()

    def test_freeze_exact_candidate_no_semantic_relabel(self):
        value, hashes = freeze_fixture()
        runner.validate_freeze(value, hashes)
        for mutate in (lambda x: x.update(schema="discovery_pair.v1"),
                       lambda x: x["candidate"].update(harris_gain=1),
                       lambda x: x["candidate"].update(feature_image_scale=1),
                       lambda x: x["sources"].pop("0240"),
                       lambda x: x["sources"]["0170"].update(frames=673.0),
                       lambda x: x.update(baseline_workspace="/tmp/other"),
                       lambda x: x["files"].update({"../escape": DIGEST})):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_freeze(bad, hashes)

    def test_manifest_paths_fail_before_hash_reads(self):
        value, _ = freeze_fixture()
        value["files"]["../escape"] = DIGEST
        with patch.object(runner, "read", return_value=value), patch.object(runner, "sha") as sha:
            with self.assertRaises(ValueError):
                runner.inputs(Path("/generated"), "0170", Mock())
        sha.assert_not_called()

    def test_exclusive_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            runner.write(path, {"passed": False})
            with self.assertRaises(FileExistsError):
                runner.write(path, {"passed": True})
            self.assertFalse(runner.read(path)["passed"])


class CandidateTests(unittest.TestCase):
    def source(self):
        return ('def _estimate_v12(self, previous, current):\n'
                '    if True:\n        uses_u16 = False\n' + runner.GAIN_ANCHOR +
                '        return conversion\n')

    def test_single_reversible_gain_only(self):
        source = self.source()
        with patch.object(runner, "METHOD_SHA", hashlib.sha256(source.encode()).hexdigest()):
            changed = runner.transform_source(source)
        self.assertEqual(changed.replace(runner.GAIN_REPLACEMENT, runner.GAIN_ANCHOR), source)
        self.assertEqual(changed.count(runner.GAIN_REPLACEMENT), 1)
        compile(changed, "generated", "exec")
        with self.assertRaises(ValueError):
            runner.transform_source(source)
        for bad in (source.replace(runner.GAIN_ANCHOR, ""), source + runner.GAIN_ANCHOR):
            with patch.object(runner, "METHOD_SHA", hashlib.sha256(bad.encode()).hexdigest()), self.assertRaises(ValueError):
                runner.transform_source(bad)

    def test_capacity_only_configuration_change(self):
        original = Config()
        result, changes = runner.candidate_config(original)
        self.assertEqual(set(changes), {"harris_capacity_policy"})
        self.assertEqual(result.harris_capacity_policy, "complete_grid")
        self.assertEqual(result.arbitrary_other_gate, original.arbitrary_other_gate)
        self.assertEqual(original.harris_capacity_policy, "legacy_default")
        for values in (dict(feature_image_scale=1), dict(minimum_accepted_features=29),
                       dict(minimum_grid_coverage=0.1), dict(flow_status_policy="changed")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.candidate_config(Config(**values))

    def test_capacity_dimensions(self):
        self.assertEqual(runner.capacity_for_shape((480, 640)), 1271)
        self.assertEqual(runner.capacity_for_shape((512, 640)), 1353)
        self.assertEqual(runner.capacity_for_shape((3190, 4784)), 60300)
        for shape in ((0, 640), (True, 640), (480.0, 640)):
            with self.assertRaises(ValueError):
                runner.capacity_for_shape(shape)

    def test_pyramid_dimensions_bound_to_frozen_configuration(self):
        dimensions = runner.validate_pyramid_dimensions((512, 640), Config())
        self.assertEqual(dimensions["level_size_lower_bounds_wh"],
                         [[320, 256], [160, 128], [80, 64], [40, 32]])
        with self.assertRaisesRegex(ValueError, ">=32x32"):
            runner.validate_pyramid_dimensions((480, 640), Config())
        dimensions = runner.validate_pyramid_dimensions((3190, 4784), Config())
        self.assertEqual(dimensions["level_size_lower_bounds_wh"][-1], [299, 199])
        for config in (Config(pyramid_levels=3), Config(pyramid_scale=0.25),
                       Config(feature_image_scale=1), Config(pyramid_levels=True)):
            with self.assertRaisesRegex(ValueError, "configuration"):
                runner.validate_pyramid_dimensions((512, 640), config)
        with self.assertRaisesRegex(ValueError, "maximum"):
            runner.validate_pyramid_dimensions((512, 7000), Config())

    def test_control_dimensions_against_hash_pinned_motion_config(self):
        path = PATH.parent.parent / "configs/evaluation/phase20_motion_v8.json"
        self.assertEqual(runner.sha(path),
                         "fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1")
        config = SimpleNamespace(**runner.read(path)["motion"])
        result = runner.validate_pyramid_dimensions((runner.CONTROL_HEIGHT, runner.CONTROL_WIDTH), config)
        self.assertEqual(result["level_size_lower_bounds_wh"][-1], [40, 32])
        with self.assertRaisesRegex(ValueError, ">=32x32"):
            runner.validate_pyramid_dimensions((480, 640), config)

    def test_correspondence_backend_capacity_and_gates(self):
        runner.verify_correspondence(correspondence(), (480, 640))
        for mutate in (lambda c: c.backends.update(cpu_fallback=True),
                       lambda c: c.metrics["harris_output"].update(capacity=8192),
                       lambda c: c.metrics["harris_output"].update(capacity_exhausted=True),
                       lambda c: c.metrics.update(detected_count=1271),
                       lambda c: c.metrics.update(minimum_accepted_features=29),
                       lambda c: c.metrics.update(selected_count=1001),
                       lambda c: setattr(c, "motion_image_size", (640, 480))):
            corr = correspondence()
            mutate(corr)
            with self.assertRaises(ValueError):
                runner.verify_correspondence(corr, (480, 640))

    def test_adapter_restores_method_on_failure(self):
        source = self.source()
        original_method = object()
        reuse = SimpleNamespace(pva=SimpleNamespace(), generated_method=lambda: source, _ESTIMATE=original_method)
        class Base:
            def __init__(self, cfg): self.config = cfg
            def estimate(self, previous, current): return reuse._ESTIMATE(self, previous, current)
        reuse.ReuseMotionV12 = Base
        frame = SimpleNamespace(bit_depth=8, image=SimpleNamespace(dtype="uint8"), shape=(480, 640))
        audit = {}
        with patch.object(runner, "METHOD_SHA", hashlib.sha256(source.encode()).hexdigest()), \
                patch.object(runner, "verify_correspondence") as verify:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with runner.candidate_adapter(reuse, Config(), audit) as candidate:
                    instance = candidate(Config())
                    self.assertEqual(instance.config.harris_capacity_policy, "complete_grid")
                    self.assertEqual(instance.estimate(frame, frame), {"scale": 16.0, "offset": 0.0})
                    self.assertEqual(audit["successful_pair_backend_checks"], 1)
                    raise RuntimeError("stop")
            verify.assert_called_once()
        self.assertIs(reuse._ESTIMATE, original_method)
        self.assertEqual(audit["estimator_instances"], 1)
        self.assertTrue(audit["source_transformation"]["exact_original_recovered"])


class GeneratedControlTests(unittest.TestCase):
    def test_controls_shared_pattern_and_no_wrap(self):
        first, second = runner.generated_controls(), runner.generated_controls()
        self.assertEqual([c[0] for c in first], list(runner.CONTROL_NAMES))
        for one, two in zip(first, second):
            for image1, image2 in zip(one[1:3], two[1:3]):
                np.testing.assert_array_equal(image1, image2)
                self.assertEqual(image1.shape, (512, 640))
                self.assertEqual(image1.dtype, np.uint8)
        np.testing.assert_array_equal(first[0][1], first[0][2])
        np.testing.assert_array_equal(first[0][1], first[1][1])
        np.testing.assert_array_equal(first[1][2][:-2, 4:], first[1][1][2:, :-4])
        self.assertTrue(np.all(first[1][2][-2:] == 128))
        self.assertTrue(np.all(first[2][1] == 128))
        self.assertEqual((int(first[0][1].min()), int(first[0][1].max())), (120, 135))

    def test_all_uint8_gain_values_representable_without_clipping(self):
        source = np.arange(256, dtype=np.uint8)
        expected = source.astype(np.int16) * 16
        self.assertEqual((expected.min(), expected.max()), (0, 4080))
        self.assertEqual(len(np.unique(expected)), 256)

    def test_truth_gates_all_conditions(self):
        corr = correspondence()
        fit = SimpleNamespace(accepted=True, parameters=dict(translation_x_px=4.0, translation_y_px=-2.0))
        result = runner.control_metrics(corr, fit, (4, -2))
        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_interior_points"], 30)
        for change in ("fit", "vector", "points", "median", "max"):
            c, f = copy.deepcopy(corr), copy.deepcopy(fit)
            if change == "fit": f.accepted = False
            elif change == "vector": f.parameters["translation_x_px"] += 0.11
            elif change == "points": c.previous_points = c.previous_points[:-1]; c.current_points = c.current_points[:-1]; c.count -= 1
            elif change == "median": c.current_points += [0.11, 0]
            elif change == "max": c.current_points[0] += [0.51, 0]
            with self.subTest(change=change):
                self.assertFalse(runner.control_metrics(c, f, (4, -2))["passed"])

    def test_unavailable_truth_not_zero_success(self):
        corr = correspondence()
        corr.previous_points += 1000
        corr.current_points += 1000
        fit = SimpleNamespace(accepted=False, parameters=None)
        result = runner.control_metrics(corr, fit, (4, -2))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["median_error_px"])
        self.assertIsNone(result["translation_vector_error_px"])


class LifecycleTests(unittest.TestCase):
    def test_pyramid_guard_fails_before_estimator_creation(self):
        modules = {"tiny_target.types": SimpleNamespace(Frame=Mock(), TimestampSource=Mock()),
                   "tiny_target.motion": SimpleNamespace(PvaMotionError=RuntimeError, fit_global_motion=Mock())}
        pixels, candidate, rows = np.zeros((480, 640), np.uint8), Mock(), []
        controls = [(runner.CONTROL_NAMES[0], pixels, pixels, (0, 0))]
        with patch.dict(sys.modules, modules), patch.object(runner, "generated_controls", return_value=controls), \
                self.assertRaisesRegex(ValueError, ">=32x32"):
            runner.run_controls(candidate, Config(), None, rows)
        candidate.assert_not_called()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["passed"])
        self.assertIn("[40, 30]", rows[0]["error"])

    def test_original_pva_control_failure_retained_and_estimator_closed(self):
        class PvaError(RuntimeError):
            pass
        message = "PVA_ERROR_INVALID_ARGUMENT: smallest pyramid40x30 must>=32x32"
        estimator = SimpleNamespace(estimate=Mock(side_effect=PvaError(message)), failed=True, closed=False)
        estimator.close = Mock(side_effect=lambda: setattr(estimator, "closed", True))
        modules = {"tiny_target.types": SimpleNamespace(Frame=Mock(),
                    TimestampSource=SimpleNamespace(CONTAINER_RATE="generated")),
                   "tiny_target.motion": SimpleNamespace(PvaMotionError=PvaError, fit_global_motion=Mock())}
        rows = []
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "unexpected generated"):
            runner.run_controls(lambda config: estimator, Config(), None, rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pva_error"], message)
        self.assertEqual(rows[0]["error"], message)
        self.assertIn("unexpected generated", rows[0]["control_failure"])
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(rows[0]["closed"])
        estimator.close.assert_called_once()

    def test_preflight_controls_and_config_bound(self):
        root = Path("/generated")
        pre, hashes, identities = pre_fixture(root)
        runner.validate_preflight(pre, hashes, identities, root, "0170")
        for mutate in (lambda p: p.update(passed=1), lambda p: p.update(clip="0240"),
                       lambda p: p.update(input_sha256={}), lambda p: p["candidate"].update(harris_gain=1),
                       lambda p: p["controls"].pop(), lambda p: p["controls"][0].update(passed=False),
                       lambda p: p["controls"][1].update(closed=False), lambda p: p["conversion"].update(passed=False)):
            bad = copy.deepcopy(pre)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_preflight(bad, hashes, identities, root, "0170")

    def test_failed_preflight_cannot_start_candidate_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0170").mkdir()
            pre, hashes, identities = pre_fixture(root)
            pre["passed"] = False
            runner.write(root / "0170/preflight.json", pre)
            helper = SimpleNamespace(dependencies=Mock(return_value=({}, identities)))
            baseline = SimpleNamespace(load_helper=Mock(return_value=helper), execute_baseline=Mock())
            with patch.object(runner, "workspace_guard", return_value=root), \
                    patch.object(runner, "load_baseline", return_value=baseline), \
                    patch.object(runner, "inputs", return_value=({}, {}, hashes)), \
                    patch.object(runner, "configurations") as configure:
                with self.assertRaisesRegex(ValueError, "preflight"):
                    runner.run(root, "0170")
            baseline.execute_baseline.assert_not_called()
            configure.assert_not_called()
            self.assertFalse(runner.read(root / "0170/execution_receipt.json")["passed"])
            self.assertFalse((root / "0170/run").exists())

    def test_failed_intake_retains_honest_candidate_receipt(self):
        for mode in ("preflight", "run"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(runner, "workspace_guard", return_value=root), \
                        patch.object(runner, "load_baseline", side_effect=ValueError("bad helper")):
                    with self.assertRaisesRegex(ValueError, "bad helper"):
                        getattr(runner, mode)(root, "0170")
                name = "preflight.json" if mode == "preflight" else "execution_receipt.json"
                record = json.loads((root / "0170" / name).read_text())
                self.assertFalse(record["passed"])
                self.assertIn("bad helper", record["error"])
                self.assertFalse((root / "0170/run").exists())
                if mode == "run":
                    self.assertTrue(record["algorithm_changed"])
                    self.assertFalse(record["global_motion_gates_changed"])
                    self.assertFalse(record["detector_configuration_changed"])


if __name__ == "__main__":
    unittest.main()
