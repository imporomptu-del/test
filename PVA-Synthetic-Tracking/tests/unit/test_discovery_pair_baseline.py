"""Generated-only guards for the bounded discovery runner; no media/native imports."""
import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PATH = Path(__file__).resolve().parents[2] / "scripts/run_discovery_pair_baseline.py"
SPEC = importlib.util.spec_from_file_location("discovery_pair_test_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
DIGEST = "a" * 64


def freeze_fixture():
    hashes = {"run_discovery_pair_baseline.py": DIGEST, "tests.py": "b" * 64}
    return dict(schema="discovery_pair.v1", files=hashes.copy(),
                sources={c: runner.source_spec(c) for c in runner.SOURCE_HASHES}), hashes


def probe_fixture():
    return SimpleNamespace(codec="mjpeg", pixel_format="yuvj420p", width=4784,
                           height=3190, frame_rate=Fraction(10), declared_frame_count=673)


def pre_fixture(workspace, clip="0170"):
    hashes = {"fixture": DIGEST}
    identities = dict(adapters={"fixture": DIGEST}, libraries={"lib": DIGEST})
    pre = dict(schema=runner.SCHEMA + ".preflight", passed=True,
               workspace=str(workspace), clip=clip, input_sha256=hashes,
               source=runner.source_spec(clip), probe_passed=True, detector_run=False,
               clock_policy={"unchanged": True}, **identities)
    return pre, hashes, identities


def output_fixture(clip="0170"):
    reference = dict(configuration={"threshold": 4}, config_sha256=DIGEST,
                     motion_config_sha256="b" * 64, package_sha256="c" * 64)
    launch = dict(reference, source=runner.source_spec(clip)["path"],
                  source_sha256=runner.SOURCE_HASHES[clip], expected_frames=673,
                  max_frames=None, fps=10, annotations_supplied_to_detector=False,
                  source_probe=dict(codec="mjpeg", pixel_format="yuvj420p", width=4784,
                                    height=3190, declared_frame_count=673, frame_rate="10"))
    report = dict(completed=True, full_clip=True, frames=673,
                  source_sha256=runner.SOURCE_HASHES[clip],
                  configuration=copy.deepcopy(reference["configuration"]),
                  availability={"detection_ready_frames": 0}, detection_status="unavailable",
                  frame_decode=dict(decoded_frames=673, consumed_frames=673,
                                    dropped_frames=0, worker_joined=True, capture_released=True,
                                    maximum_observed_frames_ahead=1,
                                    contract={"execution": "prefetch_one"}))
    return report, launch, reference


class ScopeTests(unittest.TestCase):
    def test_exact_two_sources(self):
        self.assertEqual(set(runner.SOURCE_HASHES), {"0170", "0240"})
        for clip in runner.SOURCE_HASHES:
            source = runner.source_spec(clip)
            self.assertTrue(source["path"].endswith(f"/chunk_{clip}.avi"))
            self.assertEqual((source["frames"], source["width"], source["height"]), (673, 4784, 3190))
            self.assertEqual((source["codec"], source["pixel_format"], source["fps"]),
                             ("mjpeg", "yuvj420p", 10))
        for clip in ("0126", "170", 170, "0170/../0240"):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                runner.source_spec(clip)

    def test_workspace_exact_pattern(self):
        good = "/tmp/seaqr_discovery_pair_20260928_Ab12zQ"
        self.assertEqual(runner.scope_path(good), Path(good))
        for value in ("/tmp", good + "/0170", good + "x", good.replace("20260928", "20260927"),
                      good + "/../escape", good.replace("/tmp/", "/private/tmp/")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.scope_path(value)

    def test_workspace_guard_existing_and_dangling_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "WORKSPACE_PATTERN", re.escape(str(root))), patch.object(runner.os, "geteuid", return_value=1000):
                self.assertEqual(runner.workspace_guard(root, "0170", "run"), root)
                (root / "0170").mkdir()
                pre = root / "0170/preflight.json"
                pre.write_text("{}")
                runner.workspace_guard(root, "0170", "run")
                with self.assertRaises(ValueError):
                    runner.workspace_guard(root, "0170", "preflight")
                for name in ("run", "execution_receipt.json"):
                    target = root / "0170" / name
                    target.symlink_to(root / "does-not-exist")
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        runner.workspace_guard(root, "0170", "run")
                    target.unlink()

    def test_root_invalid_mode_and_linked_clip_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "WORKSPACE_PATTERN", re.escape(str(root))):
                with patch.object(runner.os, "geteuid", return_value=0), self.assertRaises(ValueError):
                    runner.workspace_guard(root, "0170", "run")
                with patch.object(runner.os, "geteuid", return_value=1000):
                    with self.assertRaises(ValueError):
                        runner.workspace_guard(root, "0170", "other")
                    (root / "0170").symlink_to(root / "absent")
                    with self.assertRaises(ValueError):
                        runner.workspace_guard(root, "0170", "run")

    def test_metadata_read_and_write_never_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            runner.write(path, {"original": True})
            self.assertEqual(runner.read(path), {"original": True})
            with self.assertRaises(FileExistsError):
                runner.write(path, {"original": False})
            link = Path(directory) / "linked.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                runner.read(link)
            self.assertEqual(json.loads(path.read_text()), {"original": True})


class MetadataTests(unittest.TestCase):
    def test_freeze_exact_inventory(self):
        value, hashes = freeze_fixture()
        runner.validate_freeze(value, hashes)
        for key in ("0170", "0240"):
            bad = copy.deepcopy(value)
            del bad["sources"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_freeze(bad, hashes)

    def test_freeze_changed_hash_path_and_source(self):
        value, hashes = freeze_fixture()
        for name in ("../escape", "nested/file", "/absolute", ".", "..", "freeze.json"):
            bad = copy.deepcopy(value)
            bad["files"][name] = DIGEST
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.validate_freeze(bad, bad["files"])
        for mutate in (lambda x: x.update(schema="other"),
                       lambda x: x["files"].update({"tests.py": "g" * 64}),
                       lambda x: x["files"].pop("run_discovery_pair_baseline.py"),
                       lambda x: x["sources"]["0170"].update(sha256="b" * 64),
                       lambda x: x["sources"]["0170"].update(frames=673.0)):
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_freeze(bad, hashes)

    def test_manifest_path_rejected_before_any_controlled_file_open(self):
        value, _ = freeze_fixture()
        value["files"]["../escape"] = DIGEST
        with patch.object(runner, "read", return_value=value) as read, patch.object(runner, "sha") as sha:
            with self.assertRaises(ValueError):
                runner.inputs(Path("/fixture"), "0170")
        read.assert_called_once_with(Path("/fixture/freeze.json"))
        sha.assert_not_called()

    def test_probe_native_contract(self):
        runner.validate_probe(probe_fixture())
        for key, value in (("codec", "h264"), ("pixel_format", "gray16le"), ("width", 2448),
                           ("height", 3191), ("frame_rate", Fraction(9)), ("declared_frame_count", 672)):
            p = probe_fixture()
            setattr(p, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_probe(p)

    def test_native_frame_shape_dtype_order_and_extra(self):
        for index in (0, 672):
            runner.check_frame(SimpleNamespace(index=index, gray=SimpleNamespace(shape=(3190, 4784), dtype="uint8")), index)
        for idx, actual, shape, dtype in ((1, 0, (3190, 4784), "uint8"),
                                        (673, 673, (3190, 4784), "uint8"),
                                        (-1, -1, (3190, 4784), "uint8"),
                                        (0, 0, (4784, 3190), "uint8"),
                                        (0, 0, (3190, 4784), "uint16")):
            with self.subTest(idx=idx, shape=shape, dtype=dtype), self.assertRaises(ValueError):
                runner.check_frame(SimpleNamespace(index=actual, gray=SimpleNamespace(shape=shape, dtype=dtype)), idx)

    def test_preflight_identity_and_flags(self):
        root = Path("/fixture")
        pre, hashes, identities = pre_fixture(root)
        runner.validate_preflight(pre, hashes, identities, root, "0170")
        for key, value in (("passed", 1), ("probe_passed", False), ("detector_run", True),
                           ("workspace", "/another"), ("clip", "0240"), ("input_sha256", {}),
                           ("source", runner.source_spec("0240")), ("libraries", {})):
            bad = copy.deepcopy(pre)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_preflight(bad, hashes, identities, root, "0170")

    def test_unavailable_output_is_valid_execution_not_accuracy_success(self):
        report, launch, reference = output_fixture()
        runner.validate_output(report, launch, reference, "0170", 673)
        self.assertEqual(report["detection_status"], "unavailable")

    def test_output_rejects_incomplete_or_wrong_source(self):
        for key, value in (("completed", False), ("full_clip", False), ("frames", 672),
                           ("source_sha256", "b" * 64), ("configuration", {"threshold": 3})):
            report, launch, reference = output_fixture()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_output(report, launch, reference, "0170", 673)
        report, launch, reference = output_fixture()
        with self.assertRaises(ValueError):
            runner.validate_output(report, launch, reference, "0170", 672)

    def test_output_rejects_configuration_probe_and_annotations(self):
        for key, value in (("config_sha256", "b" * 64), ("motion_config_sha256", DIGEST),
                           ("package_sha256", DIGEST), ("expected_frames", 672), ("max_frames", 673),
                           ("fps", 9), ("annotations_supplied_to_detector", True), ("source_probe", {})):
            report, launch, reference = output_fixture()
            launch[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_output(report, launch, reference, "0170", 673)

    def test_output_decode_lifecycle_fails_closed(self):
        for key, value in (("decoded_frames", 672), ("consumed_frames", 672), ("dropped_frames", 1),
                           ("worker_joined", False), ("capture_released", False),
                           ("maximum_observed_frames_ahead", 2), ("contract", {})):
            report, launch, reference = output_fixture()
            report["frame_decode"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_output(report, launch, reference, "0170", 673)


class OuterLifecycleTests(unittest.TestCase):
    def test_failed_input_preflight_writes_failure_without_importing_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "workspace_guard", return_value=root), \
                    patch.object(runner, "inputs", side_effect=ValueError("changed input")), \
                    patch.object(runner, "load_helper") as load:
                with self.assertRaisesRegex(ValueError, "changed input"):
                    runner.preflight(root, "0170")
            record = runner.read(root / "0170/preflight.json")
            self.assertFalse(record["passed"])
            self.assertFalse(record["detector_run"])
            self.assertIn("changed input", record["error"])
            load.assert_not_called()
            self.assertFalse((root / "0170/run").exists())

    def outer_context(self, root, pre=None):
        report, launch, reference = output_fixture()
        valid_pre, hashes, identities = pre_fixture(root)
        helper = SimpleNamespace(dependencies=Mock(return_value=({}, identities)))
        return report, launch, reference, valid_pre if pre is None else pre, hashes, identities, helper

    def test_missing_preflight_cannot_start_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, reference, _, hashes, _, helper = self.outer_context(root)
            with patch.object(runner, "workspace_guard", return_value=root), \
                    patch.object(runner, "inputs", return_value=(reference, {}, hashes)), \
                    patch.object(runner, "load_helper", return_value=helper), \
                    patch.object(runner, "execute_baseline") as execute:
                with self.assertRaises(ValueError):
                    runner.run(root, "0170")
            execute.assert_not_called()
            receipt = runner.read(root / "0170/execution_receipt.json")
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["processed_frames"], 0)
            self.assertFalse((root / "0170/run").exists())

    def test_changed_preflight_dependencies_cannot_start_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, reference, pre, hashes, _, helper = self.outer_context(root)
            (root / "0170").mkdir()
            pre["libraries"] = {"modified": DIGEST}
            runner.write(root / "0170/preflight.json", pre)
            with patch.object(runner, "workspace_guard", return_value=root), \
                    patch.object(runner, "inputs", return_value=(reference, {}, hashes)), \
                    patch.object(runner, "load_helper", return_value=helper), \
                    patch.object(runner, "execute_baseline") as execute:
                with self.assertRaisesRegex(ValueError, "dependencies"):
                    runner.run(root, "0170")
            execute.assert_not_called()
            self.assertFalse(runner.read(root / "0170/execution_receipt.json")["passed"])

    def test_execution_exception_preserves_partial_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, reference, pre, hashes, identities, _ = self.outer_context(root)
            (root / "0170").mkdir()
            runner.write(root / "0170/preflight.json", pre)
            modules = {"profile_visible_interaction_v30": SimpleNamespace(runtime_info=lambda: {"fixture": True})}
            helper = SimpleNamespace(dependencies=Mock(return_value=(modules, identities)),
                                     runtime_check=Mock(), clock_policy_snapshot=lambda: pre["clock_policy"],
                                     TRACKING_METHOD_SHA=DIGEST)
            def fail(helper, modules, source, output, receipt):
                receipt["decoded_frames_verified"] = 7
                receipt["cleanup_errors"] = []
                raise RuntimeError("generated execution failure")
            with patch.object(runner, "workspace_guard", return_value=root), \
                    patch.object(runner, "inputs", return_value=(reference, {}, hashes)), \
                    patch.object(runner, "load_helper", return_value=helper), \
                    patch.object(runner, "execute_baseline", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "generated execution failure"):
                    runner.run(root, "0170")
            receipt = runner.read(root / "0170/execution_receipt.json")
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["decoded_frames_verified"], 7)
            self.assertIn("generated execution failure", receipt["error"])
            self.assertFalse(receipt["algorithm_changed"])
            self.assertFalse(receipt["airborne_class_verified"])


if __name__ == "__main__":
    unittest.main()
