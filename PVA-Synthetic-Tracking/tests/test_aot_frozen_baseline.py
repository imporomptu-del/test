"""Generated dictionaries/mocks only: no runtime import, media, or remote access."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_aot_frozen_baseline.py"
SPEC = importlib.util.spec_from_file_location("aot_frozen_scope", SCRIPT)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def fixture():
    rows = [dict(source_frame=i + 1, timestamp_ns=str(10**18 + i * 100_000_000),
                 img_name=f"generated_{i}.png") for i in range(300)]
    images = [dict(row, pixel_sha256="a" * 64) for row in rows]
    hashes = {"frozen_image_manifest.json": harness.MANIFEST_SHA,
              "pilot_gray8_ffv1_10fps.avi": harness.VIDEO_SHA,
              "reference_launch_v34.json": harness.REFERENCE_SHA,
              "download_validation.json": "b" * 64, "scoring_freeze.json": "c" * 64}
    download = dict(images=images, frames=300, manifest_sha256=harness.MANIFEST_SHA,
                    resolution=[2448, 2048], dtype="uint8", grayscale=True)
    packaging = dict(passed=True, frames=300, video_sha256=harness.VIDEO_SHA,
                     pixel_hashes_verified=300, detector_run=False,
                     inputs_sha256={k: hashes[k] for k in
                                    ("frozen_image_manifest.json", "download_validation.json")})
    return dict(frames=rows), download, packaging, hashes


class AotFrozenScopeTests(unittest.TestCase):
    def test_only_exact_workspace_pattern(self):
        expected = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
        self.assertEqual(harness.scope_path(expected), expected)
        for value in ("/tmp", "/", "/tmp/seaqr_aot_pilot_20260927_aeZ0yA/../other",
                      "/home/serg/project/camera_reader_sky/srcsky/chunks",
                      "/tmp/seaqr_aot_pilot_20260927_aeZ0yA/extra",
                      "/tmp/seaqr_aot_pilot_20260927_short"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                harness.scope_path(value)

    def test_root_rejected_before_filesystem_or_dependencies(self):
        with patch.object(harness.os, "geteuid", return_value=0), \
                patch.object(Path, "is_dir", side_effect=AssertionError("filesystem touched")):
            with self.assertRaisesRegex(ValueError, "as root"):
                harness.workspace_guard("/tmp/seaqr_aot_pilot_20260927_aeZ0yA", "run")

    def test_valid_generated_inventory(self):
        harness.validate_inventory(*fixture())

    def test_wrong_fixed_source_identity_rejected(self):
        for key in ("frozen_image_manifest.json", "pilot_gray8_ffv1_10fps.avi",
                    "reference_launch_v34.json"):
            values = fixture()
            values[3][key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                harness.validate_inventory(*values)

    def test_no_partial_or_gapped_causal_inventory(self):
        for mode in ("partial", "gap", "timestamp", "mapping"):
            values = fixture()
            if mode == "partial":
                values[0]["frames"].pop()
            elif mode == "gap":
                values[0]["frames"][15]["source_frame"] += 1
                values[1]["images"][15]["source_frame"] += 1
            elif mode == "timestamp":
                values[0]["frames"][15]["timestamp_ns"] = values[0]["frames"][14]["timestamp_ns"]
                values[1]["images"][15]["timestamp_ns"] = values[0]["frames"][14]["timestamp_ns"]
            else:
                values[1]["images"][15]["source_frame"] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                harness.validate_inventory(*values)

    def test_incomplete_packaging_rejected(self):
        for key, value in (("passed", False), ("pixel_hashes_verified", 299), ("detector_run", True)):
            values = fixture()
            values[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                harness.validate_inventory(*values)

    def test_preflight_binds_scoring_inputs_script_and_dependencies(self):
        workspace = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
        hashes = fixture()[3]
        identities = dict(adapters={"generated": "a"}, libraries={"generated": "b"})
        pre = dict(schema=harness.SCHEMA + ".preflight", passed=True, workspace=str(workspace),
                   input_sha256=hashes, script_sha256="d" * 64,
                   decode=dict(passed=True, pixel_hashes_verified=300), **identities)
        harness.validate_preflight(pre, hashes, identities, "d" * 64, workspace)
        for mode in ("scoring", "script", "decode", "adapter"):
            changed = copy.deepcopy(pre)
            if mode == "scoring":
                changed["input_sha256"]["scoring_freeze.json"] = "0" * 64
            elif mode == "script":
                changed["script_sha256"] = "0" * 64
            elif mode == "decode":
                changed["decode"]["pixel_hashes_verified"] = 299
            else:
                changed["adapters"]["generated"] = "changed"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                harness.validate_preflight(changed, hashes, identities, "d" * 64, workspace)

    def test_numerical_policy_rejects_cpu_runtime_or_threads(self):
        expected = dict(blas=[dict(threads=12)], affinity=list(range(12)), numpy="1.26.1",
                        opencv="4.10.0", thread_environment={"OPENBLAS_NUM_THREADS": None},
                        clock_ticks=100, opencv_threads=12)
        harness.runtime_check(expected, expected)
        for mode in ("blas", "opencv", "environment", "post_threads"):
            actual = copy.deepcopy(expected)
            if mode == "blas":
                actual["blas"][0]["threads"] = 1
            elif mode == "opencv":
                actual["opencv"] = "other"
            elif mode == "environment":
                actual["thread_environment"]["OPENBLAS_NUM_THREADS"] = "1"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                harness.runtime_check(actual, expected, after=mode == "post_threads")

    def test_clock_policy_snapshot_is_read_only_and_bounded(self):
        seen = []

        def fake_read(path):
            seen.append(str(path))
            return "schedutil" if "governor" in path.name else "1000"

        with patch.object(Path, "read_text", fake_read):
            policies = harness.clock_policy_snapshot()
        self.assertEqual(set(policies), {"cpu0", "cpu4", "cpu8", "gpu"})
        self.assertEqual(len(seen), 12)
        self.assertTrue(all(name.startswith("/sys/") for name in seen))
        self.assertTrue(all(row == dict(minimum=1000, maximum=1000, governor="schedutil")
                            for row in policies.values()))


if __name__ == "__main__":
    unittest.main()
