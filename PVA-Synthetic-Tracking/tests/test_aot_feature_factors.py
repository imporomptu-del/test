"""Generated/mocked factor-diagnostic tests: no media, hardware, or production run."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_aot_feature_factors.py"
SPEC = importlib.util.spec_from_file_location("aot_factor_diagnostic_scope", SCRIPT)
factors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(factors)
HELPER_SCRIPT = SCRIPT.with_name("diagnose_aot_features.py")
HELPER_SPEC = importlib.util.spec_from_file_location("aot_original_diagnostic_scope", HELPER_SCRIPT)
helper = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helper)


class AotFactorScopeTests(unittest.TestCase):
    def test_only_exact_new_factor_workspace_allowed(self):
        valid = Path("/tmp/seaqr_aot_factors_20260927_A1b2C3")
        self.assertEqual(factors.scope_path(valid), valid)
        for value in ("/", "/tmp", "/tmp/seaqr_aot_factors_20260927_short",
                      "/tmp/seaqr_aot_factors_20260927_A1b2C3/../other",
                      "/tmp/seaqr_aot_factors_20260927_A1b2C3/extra",
                      "/tmp/seaqr_aot_features_20260927_A1b2C3",
                      "/tmp/seaqr_aot_pilot_20260927_A1b2C3",
                      "/home/serg/project/camera_reader_sky/srcsky/chunks"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                factors.scope_path(value)

    def test_root_fails_before_filesystem_reads(self):
        with patch.object(factors.os, "geteuid", return_value=0), \
                patch.object(Path, "is_dir", side_effect=AssertionError("filesystem accessed")):
            with self.assertRaisesRegex(ValueError, "as root"):
                factors.workspace_guard("/tmp/seaqr_aot_factors_20260927_A1b2C3")

    def test_success_failure_and_broken_symlink_are_no_overwrite(self):
        for name in ("result.json", "failure.json"):
            for linked in (False, True):
                with self.subTest(name=name, linked=linked), \
                        patch.object(factors.os, "geteuid", return_value=501), \
                        patch.object(Path, "is_dir", return_value=True), \
                        patch.object(Path, "exists", lambda p: p.name == name and not linked), \
                        patch.object(Path, "is_symlink", lambda p: p.name == name and linked):
                    with self.assertRaisesRegex(ValueError, "existing"):
                        factors.workspace_guard("/tmp/seaqr_aot_factors_20260927_A1b2C3")


def manifest_fixture():
    hashes = dict(script_sha256="a" * 64, tests_sha256="b" * 64, plan_sha256="c" * 64)
    manifest = dict(schema=factors.PLAN_SCHEMA, input_workspace=str(factors.INPUT_WORKSPACE),
        baseline_harness_sha256=factors.HARNESS_SHA, helper_sha256=factors.HELPER_SHA,
        baseline_journal_sha256=factors.BASELINE_JOURNAL_SHA,
        reference_diagnostic_result_sha256=factors.REFERENCE_RESULT_SHA,
        current_indices=list(factors.CURRENT_INDICES), control_names=list(factors.CONTROL_NAMES),
        control_seed=20260927, include_stationary_counterfactuals=True,
        motion_pair_calls=105, arms=copy.deepcopy(list(factors.ARMS)), **hashes)
    return manifest, hashes


class AotFactorManifestTests(unittest.TestCase):
    def test_exact_five_by_twenty_one_preregistered_matrix(self):
        expected_cases = [f"aot_{i:03d}_{kind}" for i in (1, 43, 86, 128, 171, 213, 256, 299)
                          for kind in ("adjacent", "stationary")] + ["high_contrast_static",
            "high_contrast_translated", "low_contrast_static", "low_contrast_translated", "flat_static"]
        self.assertEqual(factors.case_ids(), expected_cases)
        expected = [(case, arm["id"]) for case in expected_cases for arm in factors.ARMS]
        self.assertEqual(factors.planned_calls(), expected)
        self.assertEqual(len(expected), 105)
        self.assertEqual(len(set(expected)), 105)
        self.assertEqual([(arm["feature_image_scale"], arm["harris_gain"], arm["harris_capacity"])
                          for arm in factors.ARMS],
                         [(0.5, 1, 8192), (0.5, 1, 19866), (0.5, 16, 19866),
                          (1.0, 1, 78899), (1.0, 16, 78899)])
        factors.validate_manifest(*manifest_fixture())

    def test_fixed_inputs_cases_controls_references_and_arms_cannot_change(self):
        replacements = dict(schema="other", input_workspace="/tmp/other", control_seed=99,
            motion_pair_calls=104, current_indices=[1, 43, 86, 128, 171, 213, 256, 298],
            include_stationary_counterfactuals=False, control_names=["other"],
            baseline_harness_sha256="f"*64, helper_sha256="f"*64,
            baseline_journal_sha256="f"*64, reference_diagnostic_result_sha256="f"*64)
        for key, value in replacements.items():
            manifest, hashes = manifest_fixture()
            manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                factors.validate_manifest(manifest, hashes)
        for key, value in (("harris_gain", 8), ("feature_image_scale", 0.25),
                           ("harris_capacity", 8192), ("harris_capacity_policy", "legacy_default")):
            manifest, hashes = manifest_fixture()
            manifest["arms"][2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                factors.validate_manifest(manifest, hashes)

    def test_numeric_boolean_ambiguity_is_rejected(self):
        for path in ("index", "gain", "capacity", "scale", "seed", "calls", "include"):
            manifest, hashes = manifest_fixture()
            if path == "index": manifest["current_indices"][0] = True
            elif path == "gain": manifest["arms"][0]["harris_gain"] = True
            elif path == "capacity": manifest["arms"][0]["harris_capacity"] = 8192.0
            elif path == "scale": manifest["arms"][3]["feature_image_scale"] = True
            elif path == "seed": manifest["control_seed"] = 20260927.0
            elif path == "calls": manifest["motion_pair_calls"] = 105.0
            else: manifest["include_stationary_counterfactuals"] = 1
            with self.subTest(path=path), self.assertRaises(ValueError):
                factors.validate_manifest(manifest, hashes)

    def test_script_tests_and_plan_hashes_bound(self):
        for key in ("script_sha256", "tests_sha256", "plan_sha256"):
            manifest, hashes = manifest_fixture()
            manifest[key] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                factors.validate_manifest(manifest, hashes)


@dataclass(frozen=True)
class GeneratedMotionConfig:
    feature_image_scale: float = 0.5
    harris_capacity_policy: str = "legacy_default"
    minimum_accepted_features: int = 30
    harris_strength: float = 0.5
    max_features: int = 1000
    max_forward_backward_error_px: float = 3.0
    max_displacement_px: float = 120.0
    optical_flow_backend: str = "PVA"


class AotFactorTransformationTests(unittest.TestCase):
    def test_only_scale_and_storage_configuration_fields_change(self):
        baseline = GeneratedMotionConfig()
        before = asdict(baseline)
        for arm in factors.ARMS:
            after, changes = factors.resolve_config(baseline, arm)
            self.assertLessEqual(set(changes), {"feature_image_scale", "harris_capacity_policy"})
            for key, value in asdict(after).items():
                if key not in {"feature_image_scale", "harris_capacity_policy"}:
                    self.assertEqual(value, before[key])
        self.assertEqual(asdict(baseline), before)
        changed = dict(factors.ARMS[2], harris_gain=32)
        with self.assertRaises(ValueError):
            factors.resolve_config(baseline, changed)

    @staticmethod
    def source_fixture():
        return "# generated estimator fixture\n" + factors.GAIN_ANCHOR + "".join(
            anchor for anchor, _, _ in helper.HOOKS)

    def test_gain1_is_old_instrumentation_and_gain16_has_exactly_one_edit(self):
        source = self.source_fixture()
        digest = hashlib.sha256(source.encode()).hexdigest()
        with patch.object(factors, "METHOD_SHA", digest), patch.object(helper, "METHOD_SHA", digest):
            gain1 = factors.instrument_source(source, 1, helper)
            self.assertEqual(gain1, helper.instrument_source(source))
            gain16 = factors.instrument_source(source, 16, helper)
        self.assertEqual(gain16.count(factors.GAIN_REPLACEMENT), 1)
        self.assertNotIn(factors.GAIN_ANCHOR, gain16)
        self.assertEqual(gain16.replace(factors.GAIN_REPLACEMENT, factors.GAIN_ANCHOR), gain1)
        recovered = gain1
        for _, stage, indent in helper.HOOKS:
            recovered = recovered.replace(" " * indent + f'self._aot_capture.record("{stage}", locals())\n', "")
        self.assertEqual(recovered, source)

    def test_undeclared_gain_bad_source_or_duplicate_conversion_rejected(self):
        source = self.source_fixture()
        for gain in (True, 1.0, 2, 8, 32):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                factors.instrument_source(source, gain, helper)
        with self.assertRaisesRegex(ValueError, "source"):
            factors.instrument_source(source, 1, helper)
        for malformed in (source.replace(factors.GAIN_ANCHOR, ""), source + factors.GAIN_ANCHOR):
            digest = hashlib.sha256(malformed.encode()).hexdigest()
            with patch.object(factors, "METHOD_SHA", digest), patch.object(helper, "METHOD_SHA", digest):
                with self.assertRaisesRegex(ValueError, "anchor"):
                    factors.instrument_source(malformed, 16, helper)


class AotFactorRepresentationTests(unittest.TestCase):
    def test_all_256_u8_codes_gain16_exact_and_representable(self):
        source = np.arange(256, dtype=np.uint8).reshape(16, 16)
        amplified = source.astype(np.int16) * 16
        result = factors.conversion_comparison(source, amplified, 16)
        self.assertTrue(result["exact_expected_equality"])
        self.assertEqual(result["expected_range"], [0, 4080])
        self.assertEqual(result["actual_range"], [0, 4080])
        self.assertEqual(result["maximum_absolute_error"], 0)
        self.assertTrue(result["representable_without_clipping"])
        self.assertEqual(result["s16_boundary_pixels"], 0)
        self.assertEqual(source.max(), 255)

    def test_gain1_retains_baseline_codes_and_wrong_gain_is_visible(self):
        source = np.array([[0, 120, 135, 255]], np.uint8)
        baseline = source.astype(np.int16)
        self.assertTrue(factors.conversion_comparison(source, baseline, 1)["exact_expected_equality"])
        result = factors.conversion_comparison(source, baseline, 16)
        self.assertFalse(result["exact_expected_equality"])
        self.assertEqual(result["maximum_absolute_error"], 3825)

    def test_wrapped_clipped_wrong_dtype_and_shape_cannot_pass(self):
        source = np.array([[0, 32, 128, 255]], np.uint8)
        wrapped = (source * np.uint8(16)).astype(np.int16)
        self.assertFalse(factors.conversion_comparison(source, wrapped, 16)["exact_expected_equality"])
        clipped = np.array([[0, 512, 2048, 32767]], np.int16)
        result = factors.conversion_comparison(source, clipped, 16)
        self.assertFalse(result["exact_expected_equality"])
        self.assertEqual(result["s16_boundary_pixels"], 1)
        for bad in (np.zeros((2, 2), np.int16), np.zeros((1, 4), np.uint16)):
            with self.assertRaises(ValueError):
                factors.conversion_comparison(source, bad, 16)

    def test_full_capacity_arrays_and_exact_u32_scores_supported(self):
        array = np.array([0, 2**24 + 1, 2**31 + 7, 2**32 - 1], np.uint32)
        result = factors.raw_array(array, helper)
        self.assertEqual(result["values"], [0, 16777217, 2147483655, 4294967295])
        self.assertEqual(result["sha256"], hashlib.sha256(array.tobytes()).hexdigest())
        full = np.zeros((78899, 2), np.float32)
        self.assertEqual(factors.raw_array(full, helper)["shape"], [78899, 2])
        with self.assertRaises(ValueError):
            factors.raw_array(np.zeros((78900, 2), np.float32), helper)

    def test_source_pixel_center_mapping_and_displacements(self):
        proxy = np.array([[0, 0], [1223, 1023], [100, 200]], np.float32)
        expected = np.array([[0.5, 0.5], [2446.5, 2046.5], [200.5, 400.5]])
        actual = factors.native_center_points(proxy, (1224, 1024))
        np.testing.assert_array_equal(actual, expected)
        moved = factors.native_center_points(proxy + [2, -1], (1224, 1024))
        np.testing.assert_array_equal(moved - actual, np.tile([4, -2], (3, 1)))
        np.testing.assert_array_equal(factors.native_center_points(proxy, (2448, 2048)), proxy)
        np.testing.assert_array_equal(proxy[0], [0, 0])

    def test_native_no_resize_is_not_reported_as_cpu_fallback(self):
        for arm in factors.ARMS:
            result = factors.expected_backends(arm)
            self.assertFalse(result["cpu_fallback"])
            self.assertEqual(result["motion_image_rescale"], "CUDA" if arm["feature_image_scale"] == 0.5 else "none")
            for key in ("gaussian_pyramid", "harris", "optical_flow_pyrlk"):
                self.assertEqual(result[key], "PVA")


class FakeReadOnlyVpiArray:
    def __init__(self, array):
        self.array = array
        self.read_locks = 0

    @contextmanager
    def rlock_cpu(self):
        self.read_locks += 1
        yield self.array


class AotFactorCaptureTests(unittest.TestCase):
    def test_empty_harris_never_locks_empty_backend_storage(self):
        for arm in factors.ARMS:
            capture = factors.Capture(helper, arm)
            capacity = None if arm["harris_capacity_policy"] == "legacy_default" else arm["harris_capacity"]
            with patch.object(capture, "copy_vpi", side_effect=AssertionError("empty storage locked")):
                capture.record("harris", dict(detected_count=0, harris_capacity=capacity, timings={}))
            self.assertEqual(capture.data["harris"]["raw_count"], 0)
            self.assertEqual(capture.data["harris"]["coordinates"]["shape"], [0, 2])
            self.assertEqual(capture.data["harris"]["scores"]["dtype"], "uint32")
            self.assertEqual(capture.stages, ["harris"])

    def test_native_proxy_and_gain16_capture_preserves_unamplified_input(self):
        arm = factors.ARMS[4]
        capture = factors.Capture(helper, arm)
        proxy = np.array([[0, 1, 15, 16, 127, 255], [8, 9, 64, 65, 128, 200],
                          [10, 20, 30, 40, 50, 60], [11, 21, 31, 41, 51, 61]], np.uint8)
        before = proxy.copy()
        with patch.object(factors, "WIDTH", 6), patch.object(factors, "HEIGHT", 4):
            capture.record("pixels", dict(previous=SimpleNamespace(image=proxy),
                current=SimpleNamespace(image=proxy), previous_pixels=proxy,
                current_pixels=proxy, uses_u16=False, timings={}))
            capture.record("proxy", dict(previous_motion=FakeReadOnlyVpiArray(proxy),
                current_motion=FakeReadOnlyVpiArray(proxy.copy()), rescale_backend="none",
                pyramid_backend_name="PVA", config=SimpleNamespace(optical_flow_backend="PVA"), timings={}))
            capture.record("s16", dict(previous_s16=FakeReadOnlyVpiArray(proxy.astype(np.int16) * 16), timings={}))
        self.assertEqual(capture.data["backends_requested"]["motion_image_rescale"], "none")
        self.assertEqual(capture.data["proxy"]["previous"]["sha256"], hashlib.sha256(proxy.tobytes()).hexdigest())
        self.assertEqual(capture.data["s16"]["conversion"]["actual_range"], [0, 4080])
        self.assertIsNone(capture.previous_proxy)
        np.testing.assert_array_equal(proxy, before)

    def test_wrong_gain_or_wrong_resize_backend_fails_capture(self):
        arm = factors.ARMS[4]
        proxy = np.zeros((4, 6), np.uint8)
        capture = factors.Capture(helper, arm)
        capture.record("pixels", dict(previous=SimpleNamespace(image=proxy),
            current=SimpleNamespace(image=proxy), previous_pixels=proxy,
            current_pixels=proxy, uses_u16=False, timings={}))
        with patch.object(factors, "WIDTH", 6), patch.object(factors, "HEIGHT", 4):
            with self.assertRaisesRegex(RuntimeError, "backend"):
                capture.record("proxy", dict(
                    previous_motion=FakeReadOnlyVpiArray(proxy), current_motion=FakeReadOnlyVpiArray(proxy),
                    rescale_backend="CUDA", pyramid_backend_name="PVA",
                    config=SimpleNamespace(optical_flow_backend="PVA"), timings={}))
        capture = factors.Capture(helper, arm)
        capture.previous_proxy = np.array([[255]], np.uint8)
        with self.assertRaisesRegex(RuntimeError, "gain mapping"):
            capture.record("s16", dict(previous_s16=FakeReadOnlyVpiArray(np.array([[255]], np.int16)), timings={}))
        self.assertIsNotNone(capture.error)

    def test_u32_endpoint_rounding_and_native_coordinates_observed_exactly(self):
        arm = factors.ARMS[2]
        capture = factors.Capture(helper, arm)
        points = np.array([[0, 0], [100, 200]], np.float32)
        scores = np.array([2**24 + 1, 2**32 - 1], np.uint32)
        capture.record("harris", dict(detected_count=2, harris_capacity=19866,
            features=FakeReadOnlyVpiArray(points), scores=FakeReadOnlyVpiArray(scores), timings={}))
        result = capture.data["harris"]
        self.assertEqual(result["scores"]["values"], [16777217, 4294967295])
        self.assertEqual(result["max_u32_score_count"], 1)
        self.assertEqual(result["float32_rounding_changed_scores"], 2)
        self.assertEqual(result["float32_rounding_maximum_absolute_error"], 1)
        self.assertEqual(result["native_center_coordinates"]["values"], [[0.5, 0.5], [200.5, 400.5]])

    def test_capacity_exhaustion_is_observed_not_hidden_or_resized(self):
        # Reduced generated capacity stands in for the same bounded storage gate.
        arm = dict(factors.ARMS[1], harris_capacity=2)
        capture = factors.Capture(helper, arm)
        capture.record("harris", dict(detected_count=2, harris_capacity=2,
            features=FakeReadOnlyVpiArray(np.array([[1, 2], [3, 4]], np.float32)),
            scores=FakeReadOnlyVpiArray(np.array([1, 2], np.uint32)), timings={}))
        self.assertTrue(capture.data["harris"]["capacity_saturation_observed"])
        self.assertEqual(capture.data["harris"]["raw_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            factors.Capture(helper, factors.ARMS[1]).record("harris",
                dict(detected_count=0, harris_capacity=None, timings={}))


def truthful_fixture():
    return dict(case_id="high_contrast_static", capture=dict(harris=dict(raw_count=1000)),
        global_fit=dict(quality_status="accepted", parameters=dict(translation_x_px=0.0, translation_y_px=0.0)),
        generated_control_error=dict(accepted_interior_points=30, median_error_px=0.1, maximum_error_px=0.5))


class AotFactorScientificGateTests(unittest.TestCase):
    def test_all_preregistered_truth_boundaries_inclusive(self):
        row = truthful_fixture()
        row["global_fit"]["parameters"]["translation_x_px"] = 0.1
        result = factors.scientific_gates(row, (0, 0))
        self.assertTrue(result["passed"])
        self.assertEqual(result["translation_vector_error_px"], 0.1)
        self.assertTrue(all(result["criteria"].values()))

    def test_each_gate_can_independently_fail_despite_more_corners(self):
        for mode in ("fit", "vector", "count", "median", "maximum"):
            row = truthful_fixture()
            if mode == "fit": row["global_fit"]["quality_status"] = "rejected"
            elif mode == "vector": row["global_fit"]["parameters"]["translation_x_px"] = float(np.nextafter(0.1, np.inf))
            elif mode == "count": row["generated_control_error"]["accepted_interior_points"] = 29
            elif mode == "median": row["generated_control_error"]["median_error_px"] = float(np.nextafter(0.1, np.inf))
            else: row["generated_control_error"]["maximum_error_px"] = float(np.nextafter(0.5, np.inf))
            with self.subTest(mode=mode):
                self.assertFalse(factors.scientific_gates(row, (0, 0))["passed"])


def matrix_fixture():
    rows = []
    for case in factors.case_ids():
        is_aot = case.startswith("aot_")
        index = int(case.split("_")[1]) if is_aot else 1
        kind = ("aot_stationary_counterfactual" if case.endswith("stationary") else "aot_adjacent") if is_aot else "generated_control"
        for arm in factors.ARMS:
            proxy = helper.image_summary(np.array([[1, 2], [3, 4]], np.uint8))
            s16 = helper.image_summary(np.array([[1, 2], [3, 4]], np.int16) * arm["harris_gain"])
            points = factors.raw_array(np.array([[11, 12]], np.float32), helper)
            scores = factors.raw_array(np.array([2**24 + 1], np.uint32), helper)
            rows.append(dict(case_id=case, kind=kind, current_index=index, arm=copy.deepcopy(arm),
                native_pixel_sha256=dict(previous="a"*64, current="b"*64),
                capture=dict(proxy=dict(previous=copy.deepcopy(proxy), current=copy.deepcopy(proxy)),
                    s16=dict(previous=s16), harris=dict(raw_count=1, capacity_saturation_observed=False,
                                                       coordinates=points, scores=scores))))
    return rows


class AotFactorComparisonTests(unittest.TestCase):
    def test_complete_matrix_compares_all_counterfactuals_gains_and_capacities(self):
        result = factors.comparisons(matrix_fixture(), helper)
        self.assertEqual(set(result["real_stationary"]), {arm["id"] for arm in factors.ARMS})
        self.assertTrue(all(len(rows) == 8 for rows in result["real_stationary"].values()))
        self.assertEqual(len(result["gain_input_invariance"]), 42)
        self.assertEqual(len(result["capacity_bridge"]), 21)
        self.assertTrue(all(all(row["equal"].values()) for row in result["gain_input_invariance"]))

    def test_missing_duplicate_or_reordered_call_matrix_rejected(self):
        for mode in ("missing", "extra", "order", "changed_arm"):
            rows = matrix_fixture()
            if mode == "missing": rows.pop()
            elif mode == "extra": rows.append(copy.deepcopy(rows[0]))
            elif mode == "order": rows[0], rows[1] = rows[1], rows[0]
            else: rows[2]["arm"]["id"] = "unplanned"
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "matrix"):
                factors.comparisons(rows, helper)

    def test_harris_gain_cannot_change_native_or_u8_inputs(self):
        for mode in ("native", "proxy"):
            rows = matrix_fixture()
            if mode == "native": rows[2]["native_pixel_sha256"]["current"] = "f"*64
            else: rows[2]["capture"]["proxy"]["current"]["sha256"] = "f"*64
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "gain arm"):
                factors.comparisons(rows, helper)

    def test_capacity_ceiling_and_prefix_mismatch_remain_observations(self):
        rows = matrix_fixture()
        rows[0]["capture"]["harris"]["capacity_saturation_observed"] = True
        rows[1]["capture"]["harris"]["scores"] = factors.raw_array(np.array([2**24], np.uint32), helper)
        result = factors.comparisons(rows, helper)
        self.assertTrue(result["capacity_bridge"][0]["legacy_capacity_reached"])
        self.assertFalse(result["capacity_bridge"][0]["exact_legacy_prefix_scores"])


class AotFactorLifecycleTests(unittest.TestCase):
    @staticmethod
    def modules():
        package = ModuleType("tiny_target")
        package.__path__ = []
        types = ModuleType("tiny_target.types")
        motion = ModuleType("tiny_target.motion")

        class Frame:
            def __init__(self, image, timestamp_ns, frame_index, *rest):
                self.image, self.timestamp_ns, self.frame_index = image, timestamp_ns, frame_index

            def metadata_dict(self):
                return dict(timestamp_ns=self.timestamp_ns, frame_index=self.frame_index)

        class PvaMotionError(RuntimeError):
            pass

        types.Frame = Frame
        types.TimestampSource = SimpleNamespace(CONTAINER_RATE="generated")
        motion.PvaMotionError = PvaMotionError
        motion.fit_global_motion = lambda *args: (_ for _ in ()).throw(AssertionError("no fit allowed"))
        return {"tiny_target": package, "tiny_target.types": types, "tiny_target.motion": motion}, PvaMotionError

    def evaluate(self, exhausted=False, close_failure=False):
        modules, error_type = self.modules()
        arm = dict(factors.ARMS[1], harris_capacity=2) if exhausted else dict(factors.ARMS[2])

        class Estimator:
            def __init__(self, config):
                self.config = config
                self.hits, self.misses, self.resets = 0, 1, 0
                self.closed, self.failed = False, False

            def estimate(self, previous, current):
                capture = self._aot_capture
                proxy = previous.image[::2, ::2].copy()
                capture.record("pixels", dict(previous=previous, current=current,
                    previous_pixels=previous.image, current_pixels=current.image, uses_u16=False, timings={}))
                capture.record("proxy", dict(previous_motion=FakeReadOnlyVpiArray(proxy),
                    current_motion=FakeReadOnlyVpiArray(proxy.copy()), rescale_backend="CUDA",
                    pyramid_backend_name="PVA", config=self.config, timings={}))
                capture.record("s16", dict(previous_s16=FakeReadOnlyVpiArray(proxy.astype(np.int16) * arm["harris_gain"]), timings={}))
                state = dict(detected_count=2 if exhausted else 0, harris_capacity=arm["harris_capacity"], timings={})
                if exhausted:
                    state.update(features=FakeReadOnlyVpiArray(np.array([[0, 0], [1, 1]], np.float32)),
                                 scores=FakeReadOnlyVpiArray(np.array([1, 2], np.uint32)))
                capture.record("harris", state)
                if exhausted:
                    self.failed = True
                    raise error_type("Harris output capacity exhausted; full-image feature coverage is unknown")
                raise error_type("PVA Harris returned zero features; motion is unavailable")

            def close(self):
                if close_failure:
                    raise RuntimeError("generated cleanup failure")
                self.closed = True

        pixels = np.arange(24, dtype=np.uint8).reshape(4, 6)
        row = dict(case_id="aot_001_stationary", previous_index=0, current_index=1, arm=arm)
        with patch.dict(sys.modules, modules), patch.object(factors, "WIDTH", 6), patch.object(factors, "HEIGHT", 4):
            try:
                result = factors.evaluate_pair(helper, SimpleNamespace(ReuseMotionV12=Estimator),
                    SimpleNamespace(optical_flow_backend="PVA"), object(), pixels, pixels.copy(), row, (0, 0))
            except BaseException as exc:
                return row, exc
        return result, None

    def test_expected_empty_case_retained_as_truth_failure_not_run_failure(self):
        row, error = self.evaluate()
        self.assertIsNone(error)
        self.assertTrue(row["completed"])
        self.assertTrue(row["expected_unavailable"])
        self.assertFalse(row["scientific_gates"]["passed"])
        self.assertTrue(row["lifecycle"]["closed"])
        self.assertEqual(row["capture"]["harris"]["raw_count"], 0)

    def test_complete_capacity_exhaustion_fails_closed_with_raw_evidence(self):
        row, error = self.evaluate(exhausted=True)
        self.assertIsNotNone(error)
        self.assertFalse(row["completed"])
        self.assertFalse(row["expected_unavailable"])
        self.assertTrue(row["lifecycle"]["closed"])
        self.assertTrue(row["capture"]["harris"]["capacity_saturation_observed"])
        self.assertEqual(row["capture"]["harris"]["scores"]["values"], [1, 2])
        self.assertIn("unexpected_error", row)

    def test_cleanup_failure_cannot_leave_case_completed(self):
        row, error = self.evaluate(close_failure=True)
        self.assertIsNotNone(error)
        self.assertFalse(row["completed"])
        self.assertFalse(row["lifecycle"]["closed"])
        self.assertIn("cleanup_error", row)


class AotFactorAdditionalScientificGateTests(unittest.TestCase):
    def test_vector_error_is_euclidean_not_per_axis(self):
        row = truthful_fixture()
        row["global_fit"]["parameters"] = dict(translation_x_px=0.08, translation_y_px=0.08)
        self.assertFalse(factors.scientific_gates(row, (0, 0))["passed"])

    def test_correct_known_translation_passes_but_repeated_pattern_alias_does_not(self):
        row = truthful_fixture()
        row["global_fit"]["parameters"] = dict(translation_x_px=4.0, translation_y_px=-2.0)
        self.assertTrue(factors.scientific_gates(row, (4, -2))["passed"])
        row["global_fit"]["parameters"]["translation_x_px"] += 32
        self.assertFalse(factors.scientific_gates(row, (4, -2))["passed"])

    def test_flat_zero_features_is_expected_not_successful_motion(self):
        row = dict(case_id="flat_static", capture=dict(harris=dict(raw_count=0)))
        result = factors.scientific_gates(row, (0, 0))
        self.assertTrue(result["passed"])
        self.assertFalse(result["motion_estimate_success"])
        row["capture"]["harris"]["raw_count"] = 1
        self.assertFalse(factors.scientific_gates(row, (0, 0))["passed"])

    def test_real_pair_fit_acceptance_does_not_claim_correct_camera_motion(self):
        row = truthful_fixture()
        row["case_id"] = "aot_001_adjacent"
        result = factors.scientific_gates(row)
        self.assertTrue(result["original_fit_accepted"])
        self.assertIsNone(result["passed"])
        self.assertFalse(result["motion_truth_verified"])

    def test_unavailable_positive_has_failed_truth_not_zero_error(self):
        row = dict(case_id="low_contrast_static", capture=dict(harris=dict(raw_count=0)), global_fit=None)
        result = factors.scientific_gates(row, (0, 0))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["translation_vector_error_px"])

    def test_malformed_numeric_error_cannot_pass(self):
        for key, bad in (("median_error_px", -0.01), ("maximum_error_px", -1),
                         ("median_error_px", float("nan")), ("maximum_error_px", float("inf")),
                         ("median_error_px", False), ("accepted_interior_points", 30.5)):
            row = truthful_fixture()
            row["generated_control_error"][key] = bad
            with self.subTest(key=key, bad=bad):
                self.assertFalse(factors.scientific_gates(row, (0, 0))["passed"])


if __name__ == "__main__":
    unittest.main()
