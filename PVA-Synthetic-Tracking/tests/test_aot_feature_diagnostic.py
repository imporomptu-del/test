"""Generated inputs/mocks only; never open media, import VPI, or run a detector."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_aot_features.py"
SPEC = importlib.util.spec_from_file_location("aot_feature_diagnostic_scope", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class AotFeatureScopeTests(unittest.TestCase):
    def test_exact_diagnostic_workspace_scope(self):
        expected = Path("/tmp/seaqr_aot_features_20260927_A1b2C3")
        self.assertEqual(diagnostic.scope_path(expected), expected)
        for value in ("/", "/tmp", "/tmp/seaqr_aot_features_20260927_short",
                      "/tmp/seaqr_aot_features_20260927_A1b2C3/extra",
                      "/tmp/seaqr_aot_features_20260927_A1b2C3/../other",
                      "/tmp/seaqr_aot_pilot_20260927_A1b2C3",
                      "/home/serg/project/camera_reader_sky/srcsky/chunks"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnostic.scope_path(value)

    def test_root_rejected_before_filesystem_access(self):
        with patch.object(diagnostic.os, "geteuid", return_value=0), \
                patch.object(Path, "is_dir", side_effect=AssertionError("filesystem touched")):
            with self.assertRaisesRegex(ValueError, "as root"):
                diagnostic.workspace_guard("/tmp/seaqr_aot_features_20260927_A1b2C3")

    def test_retained_success_or_failure_is_never_overwritten(self):
        for name in ("result.json", "failure.json"):
            for linked in (False, True):
                def exists(path):
                    return path.name == name and not linked

                def is_symlink(path):
                    return path.name == name and linked

                with self.subTest(name=name, linked=linked), \
                        patch.object(diagnostic.os, "geteuid", return_value=501), \
                        patch.object(Path, "is_dir", return_value=True), \
                        patch.object(Path, "exists", exists), \
                        patch.object(Path, "is_symlink", is_symlink):
                    with self.assertRaisesRegex(ValueError, "existing"):
                        diagnostic.workspace_guard("/tmp/seaqr_aot_features_20260927_A1b2C3")


def plan_fixture():
    hashes = dict(script_sha256="a" * 64, tests_sha256="b" * 64, plan_sha256="c" * 64)
    manifest = dict(schema=diagnostic.PLAN_SCHEMA,
                    input_workspace=str(diagnostic.INPUT_WORKSPACE),
                    current_indices=[1, 43, 86, 128, 171, 213, 256, 299],
                    control_seed=20260927, motion_pair_calls=21, ablation="none",
                    include_stationary_counterfactuals=True,
                    baseline_harness_sha256=diagnostic.HARNESS_SHA, **hashes)
    return manifest, hashes


class AotFeatureManifestTests(unittest.TestCase):
    def test_fixed_generated_plan(self):
        diagnostic.validate_manifest(*plan_fixture())

    def test_scope_selection_and_configuration_changes_rejected(self):
        replacements = dict(schema="other", input_workspace="/tmp/other",
                            current_indices=[1, 43, 86, 128, 171, 213, 256, 298],
                            control_seed=123, motion_pair_calls=22, ablation="gain",
                            include_stationary_counterfactuals=False,
                            baseline_harness_sha256="f" * 64)
        for key, replacement in replacements.items():
            manifest, hashes = plan_fixture()
            manifest[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic.validate_manifest(manifest, hashes)

    def test_script_tests_and_policy_identities_all_bound(self):
        for key in ("script_sha256", "tests_sha256", "plan_sha256"):
            for mode in ("changed", "missing", "malformed"):
                manifest, hashes = plan_fixture()
                if mode == "changed":
                    manifest[key] = "f" * 64
                elif mode == "missing":
                    manifest.pop(key)
                else:
                    manifest[key] = hashes[key] = "bad"
                with self.subTest(key=key, mode=mode), self.assertRaises(ValueError):
                    diagnostic.validate_manifest(manifest, hashes)

    def test_boolean_or_floating_point_indices_are_not_integer_frame_ids(self):
        for replacement in (True, 1.0):
            manifest, hashes = plan_fixture()
            manifest["current_indices"][0] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                diagnostic.validate_manifest(manifest, hashes)


class AotFeatureInstrumentationTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # Generated source only. No production import or estimator execution.
        return "# generated source fixture\n" + "".join(anchor for anchor, _, _ in diagnostic.HOOKS)

    def test_hooks_can_be_removed_to_recover_exact_frozen_source(self):
        source = self.fixture()
        digest = hashlib.sha256(source.encode()).hexdigest()
        with patch.object(diagnostic, "METHOD_SHA", digest):
            transformed = diagnostic.instrument_source(source)
        self.assertEqual(len(diagnostic.HOOKS), 6)
        self.assertNotEqual(transformed, source)
        for anchor, stage, indent in diagnostic.HOOKS:
            hook = " " * indent + f'self._aot_capture.record("{stage}", locals())\n'
            self.assertEqual(transformed.count(hook), 1)
            self.assertIn(anchor + hook, transformed)
            transformed = transformed.replace(hook, "")
        self.assertEqual(transformed, source)

    def test_source_hash_checked_before_instrumenting(self):
        with self.assertRaisesRegex(ValueError, "source identity"):
            diagnostic.instrument_source(self.fixture())

    def test_missing_or_duplicate_anchor_fails_closed(self):
        source = self.fixture()
        for anchor, stage, _ in diagnostic.HOOKS:
            for malformed in (source.replace(anchor, ""), source + anchor):
                digest = hashlib.sha256(malformed.encode()).hexdigest()
                with self.subTest(stage=stage), patch.object(diagnostic, "METHOD_SHA", digest):
                    with self.assertRaisesRegex(ValueError, "anchor"):
                        diagnostic.instrument_source(malformed)

    def test_harris_and_selection_hooks_precede_early_failures(self):
        source = self.fixture()
        harris_anchor = next(anchor for anchor, stage, _ in diagnostic.HOOKS if stage == "harris")
        selection_anchor = next(anchor for anchor, stage, _ in diagnostic.HOOKS if stage == "selection")
        source = source.replace(harris_anchor, harris_anchor + '        raise ValueError("zero features")\n')
        source = source.replace(selection_anchor, selection_anchor + '        raise ValueError("no selected features")\n')
        with patch.object(diagnostic, "METHOD_SHA", hashlib.sha256(source.encode()).hexdigest()):
            transformed = diagnostic.instrument_source(source)
        self.assertLess(transformed.index('record("harris",'), transformed.index('raise ValueError("zero'))
        self.assertLess(transformed.index('record("selection",'), transformed.index('raise ValueError("no selected'))


class AotFeatureArrayTests(unittest.TestCase):
    def test_exact_u32_harris_scores_are_not_rounded_through_float(self):
        scores = np.array([0, 2**24 + 1, 2**31 + 7, 2**32 - 1], dtype=np.uint32)
        before = scores.copy()
        result = diagnostic.raw_array(scores)
        self.assertEqual(result["dtype"], "uint32")
        self.assertEqual(result["shape"], [4])
        self.assertEqual(result["values"], [0, 16777217, 2147483655, 4294967295])
        self.assertEqual(result["sha256"], hashlib.sha256(scores.tobytes()).hexdigest())
        np.testing.assert_array_equal(scores, before)

    def test_empty_features_retained_with_shape_and_dtype(self):
        for array in (np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.uint32)):
            result = diagnostic.raw_array(array)
            self.assertEqual(result["shape"], list(array.shape))
            self.assertEqual(result["values"], [])
            self.assertEqual(result["sha256"], hashlib.sha256(b"").hexdigest())

    def test_nonfinite_payload_is_json_safe_but_hash_is_unmodified(self):
        array = np.array([[np.nan, np.inf], [-np.inf, 2.5]], dtype=np.float32)
        result = diagnostic.raw_array(array)
        self.assertEqual(result["values"], [[None, None], [None, 2.5]])
        self.assertEqual(result["sha256"], hashlib.sha256(array.tobytes()).hexdigest())

    def test_unbounded_or_nonnumeric_raw_arrays_rejected(self):
        for array in (np.zeros(diagnostic.MAX_HARRIS * 2 + 1, dtype=np.float32),
                      np.array(["not numerical"]), np.array([1 + 2j])):
            with self.subTest(dtype=array.dtype), self.assertRaises(ValueError):
                diagnostic.raw_array(array)

    def test_native_unsigned_gradient_does_not_wrap(self):
        image = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        before = image.copy()
        result = diagnostic.image_summary(image)
        for axis in ("x", "y"):
            self.assertEqual(result["gradients"][axis]["maximum"], 255)
            self.assertEqual(result["gradients"][axis]["mean_abs"], 255.0)
            self.assertEqual(result["gradients"][axis]["rms"], 255.0)
        self.assertEqual(result["minimum"], 0)
        self.assertEqual(result["maximum"], 255)
        self.assertEqual(result["sha256"], hashlib.sha256(image.tobytes()).hexdigest())
        np.testing.assert_array_equal(image, before)

    def test_signed_full_range_gradients_and_single_pixel_are_valid(self):
        result = diagnostic.image_summary(np.array([[-32768, 32767]], dtype=np.int16))
        self.assertEqual(result["gradients"]["x"]["maximum"], 65535)
        self.assertEqual(result["gradients"]["x"]["rms"], 65535.0)
        self.assertEqual(result["gradients"]["y"]["count"], 0)
        result = diagnostic.image_summary(np.array([[17]], dtype=np.uint8))
        self.assertEqual(result["standard_deviation"], 0)
        self.assertEqual(result["gradients"]["x"]["maximum"], 0)

    def test_image_summary_rejects_wrong_dtype_shape_or_size(self):
        for image in (np.empty((0, 2), np.uint8), np.zeros((2, 2, 1), np.uint8),
                      np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint16)):
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                diagnostic.image_summary(image)
        with patch.object(diagnostic, "WIDTH", 2), patch.object(diagnostic, "HEIGHT", 2):
            with self.assertRaises(ValueError):
                diagnostic.image_summary(np.zeros((3, 2), np.uint8))

    def test_conversion_equality_compares_codes_not_just_range(self):
        proxy = np.array([[0, 1, 127, 255]], dtype=np.uint8)
        converted = proxy.astype(np.int16)
        self.assertTrue(diagnostic.conversion_comparison(proxy, converted)["exact_value_equality"])
        converted[0, 1] += 7
        converted[0, 2] -= 2
        result = diagnostic.conversion_comparison(proxy, converted)
        self.assertFalse(result["exact_value_equality"])
        self.assertEqual(result["unequal_pixels"], 2)
        self.assertEqual(result["maximum_absolute_error"], 7)
        self.assertEqual(result["signed_difference_minimum"], -2)
        self.assertEqual(result["signed_difference_maximum"], 7)

    def test_conversion_rejects_shape_or_representation_change(self):
        proxy = np.zeros((2, 2), dtype=np.uint8)
        for converted in (np.zeros((1, 4), np.int16), np.zeros((2, 2), np.uint16)):
            with self.assertRaises(ValueError):
                diagnostic.conversion_comparison(proxy, converted)


class FakeVpiReadOnlyArray:
    """Only the read-lock interface exists; accidental write locking fails."""
    def __init__(self, data):
        self.data = data
        self.read_locks = 0

    @contextmanager
    def rlock_cpu(self):
        self.read_locks += 1
        yield self.data


class AotFeatureCaptureTests(unittest.TestCase):
    def test_copy_owns_storage_and_uses_read_lock_only(self):
        source = np.array([[1, 2]], dtype=np.float32)
        value = FakeVpiReadOnlyArray(source)
        copied = diagnostic.Capture.copy_vpi(value)
        self.assertEqual(value.read_locks, 1)
        self.assertFalse(np.shares_memory(source, copied))
        copied[0, 0] = 99
        self.assertEqual(source[0, 0], 1)

    def test_zero_harris_capture_never_locks_empty_backend_arrays(self):
        capture = diagnostic.Capture()
        state = dict(detected_count=0, harris_capacity=None, timings={})
        with patch.object(capture, "copy_vpi", side_effect=AssertionError("empty array locked")):
            capture.record("harris", state)
        self.assertEqual(capture.stages, ["harris"])
        self.assertEqual(capture.data["harris"]["raw_count"], 0)
        self.assertEqual(capture.data["harris"]["coordinates"]["shape"], [0, 2])
        self.assertEqual(capture.data["harris"]["scores"]["dtype"], "uint32")
        self.assertIsNone(capture.error)

    def test_raw_harris_captured_before_any_float_cast_or_selection(self):
        capture = diagnostic.Capture()
        points = FakeVpiReadOnlyArray(np.array([[12, 15], [40, 71]], dtype=np.float32))
        scores = FakeVpiReadOnlyArray(np.array([2**24 + 1, 2**32 - 1], dtype=np.uint32))
        state = dict(detected_count=2, harris_capacity=None, features=points, scores=scores,
                     timings={"harris_pva": 1.5})
        capture.record("harris", state)
        self.assertEqual(capture.data["harris"]["scores"]["values"], [16777217, 4294967295])
        self.assertEqual(capture.data["harris"]["coordinates"]["values"], [[12, 15], [40, 71]])
        self.assertEqual(capture.data["last_production_timings_ms"], {"harris_pva": 1.5})
        state["timings"]["harris_pva"] = 9.0
        self.assertEqual(capture.data["last_production_timings_ms"]["harris_pva"], 1.5)

    def test_harris_wrong_representation_is_capture_error_not_empty_scene(self):
        capture = diagnostic.Capture()
        state = dict(detected_count=1, harris_capacity=None,
                     features=FakeVpiReadOnlyArray(np.array([[1, 2]], dtype=np.float32)),
                     scores=FakeVpiReadOnlyArray(np.array([7], dtype=np.float32)), timings={})
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            capture.record("harris", state)
        self.assertIsNotNone(capture.error)
        self.assertEqual(capture.stages, [])

    def test_empty_selection_retains_exclusions_and_zero_arrays(self):
        capture = diagnostic.Capture()
        capture.record("selection", dict(selected_indices=np.empty(0, np.int64),
            eligible_mask=np.array([False]), detected_points=np.array([[1, 2]], np.float32),
            detected_scores=np.array([9], np.float32),
            feature_exclusions={"unreliable_border": 1}, timings={}))
        row = capture.data["selection"]
        self.assertEqual(row["selected_count"], 0)
        self.assertEqual(row["coordinates"]["shape"], [0, 2])
        self.assertEqual(row["exclusions"], {"unreliable_border": 1})

    def test_duplicate_and_unknown_capture_stages_fail_closed(self):
        capture = diagnostic.Capture()
        capture.record("harris", dict(detected_count=0, harris_capacity=None, timings={}))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            capture.record("harris", {})
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            diagnostic.Capture().record("unrecognized", {})


def counterfactual_fixture():
    proxy = diagnostic.image_summary(np.array([[0, 7], [30, 255]], dtype=np.uint8))
    s16 = diagnostic.image_summary(np.array([[0, 7], [30, 255]], dtype=np.int16))
    points = diagnostic.raw_array(np.array([[11, 21]], dtype=np.float32))
    scores = diagnostic.raw_array(np.array([2**24 + 1], dtype=np.uint32))
    capture = dict(proxy=dict(previous=proxy), s16=dict(previous=s16),
                   harris=dict(raw_count=1, coordinates=points, scores=scores))
    return [dict(kind=kind, current_index=index, capture=copy.deepcopy(capture))
            for index in diagnostic.CURRENT_INDICES
            for kind in ("aot_adjacent", "aot_stationary_counterfactual")]


class AotCounterfactualComparisonTests(unittest.TestCase):
    def test_all_eight_exact_feature_inputs_and_outputs_compared(self):
        result = diagnostic.compare_counterfactuals(counterfactual_fixture())
        self.assertEqual([row["current_index"] for row in result], list(diagnostic.CURRENT_INDICES))
        self.assertTrue(all(row["all_previous_feature_observations_equal"] for row in result))
        self.assertTrue(all(set(row["equal"]) == {"previous_proxy", "previous_s16",
            "harris_coordinates", "harris_scores", "harris_count"} for row in result))

    def test_same_count_different_exact_score_is_not_called_equal(self):
        rows = counterfactual_fixture()
        rows[1]["capture"]["harris"]["scores"] = diagnostic.raw_array(
            np.array([2**24], dtype=np.uint32))
        result = diagnostic.compare_counterfactuals(rows)
        self.assertTrue(result[0]["equal"]["harris_count"])
        self.assertFalse(result[0]["equal"]["harris_scores"])
        self.assertFalse(result[0]["all_previous_feature_observations_equal"])

    def test_dtype_shape_and_hash_all_part_of_exact_equality(self):
        for key, value in (("dtype", "float64"), ("shape", [2, 1]), ("sha256", "f" * 64)):
            rows = counterfactual_fixture()
            rows[1]["capture"]["harris"]["coordinates"][key] = value
            result = diagnostic.compare_counterfactuals(rows)
            with self.subTest(key=key):
                self.assertFalse(result[0]["equal"]["harris_coordinates"])

    def test_zero_features_are_valid_observations_not_missing_pairs(self):
        rows = counterfactual_fixture()
        for row in rows:
            row["capture"]["harris"] = dict(raw_count=0,
                coordinates=diagnostic.raw_array(np.empty((0, 2), np.float32)),
                scores=diagnostic.raw_array(np.empty(0, np.uint32)))
        self.assertTrue(all(row["all_previous_feature_observations_equal"]
                            for row in diagnostic.compare_counterfactuals(rows)))

    def test_missing_or_duplicate_pairs_fail_closed(self):
        for rows in (counterfactual_fixture()[:-1], counterfactual_fixture() + counterfactual_fixture()[:1]):
            with self.assertRaisesRegex(ValueError, "missing/duplicate"):
                diagnostic.compare_counterfactuals(rows)


class AotControlGeometryTests(unittest.TestCase):
    def test_known_shift_error_only_uses_interior_without_changing_correspondences(self):
        previous = np.array([[300, 300], [400, 400], [20, 20]], dtype=np.float32)
        current = np.array([[304, 298], [405, 398], [90, 80]], dtype=np.float32)
        correspondence = SimpleNamespace(previous_points=previous, current_points=current, count=3)
        previous_before, current_before = previous.copy(), current.copy()
        result = diagnostic.control_error(correspondence, (4, -2))
        self.assertEqual(result["all_accepted_points"], 3)
        self.assertEqual(result["accepted_interior_points"], 2)
        self.assertEqual(result["error_xy_px"]["values"], [[0, 0], [1, 0]])
        self.assertEqual(result["median_error_px"], 0.5)
        self.assertEqual(result["maximum_error_px"], 1.0)
        np.testing.assert_array_equal(previous, previous_before)
        np.testing.assert_array_equal(current, current_before)

    def test_no_interior_points_does_not_report_zero_error_success(self):
        correspondence = SimpleNamespace(previous_points=np.empty((0, 2), np.float32),
                                         current_points=np.empty((0, 2), np.float32), count=0)
        result = diagnostic.control_error(correspondence, (0, 0))
        self.assertEqual(result["accepted_interior_points"], 0)
        self.assertIsNone(result["median_error_px"])
        self.assertIsNone(result["maximum_error_px"])


class AotPairFailureRetentionTests(unittest.TestCase):
    """Exercise wrapper cleanup/error records with entirely generated module fakes."""
    @staticmethod
    def fake_modules():
        package = ModuleType("tiny_target")
        package.__path__ = []
        types = ModuleType("tiny_target.types")
        motion = ModuleType("tiny_target.motion")

        class Frame:
            def __init__(self, image, timestamp_ns, frame_index, source_id, bit_depth, timestamp_source):
                self.image, self.timestamp_ns, self.frame_index = image, timestamp_ns, frame_index

            def metadata_dict(self):
                return dict(frame_index=self.frame_index, timestamp_ns=self.timestamp_ns)

        class PvaMotionError(RuntimeError):
            pass

        types.Frame = Frame
        types.TimestampSource = SimpleNamespace(CONTAINER_RATE="generated")
        motion.PvaMotionError = PvaMotionError
        motion.fit_global_motion = lambda *args: (_ for _ in ()).throw(
            AssertionError("a zero-feature pair must not fit motion"))
        return {"tiny_target": package, "tiny_target.types": types, "tiny_target.motion": motion}, PvaMotionError

    @staticmethod
    def fake_estimator(error_type, fail_capture=False, fail_close=False, unexpected=False):
        class Estimator:
            def __init__(self, config):
                self.config = config
                self.hits, self.misses, self.resets = 0, 1, 0
                self.failed, self.closed = False, False

            def estimate(self, previous, current):
                capture = self._aot_capture
                capture.record("pixels", dict(previous=previous, current=current,
                    previous_pixels=previous.image, current_pixels=current.image, uses_u16=False, timings={}))
                proxy = np.array([[0, 10], [20, 30]], dtype=np.uint8)
                capture.record("proxy", dict(previous_motion=FakeVpiReadOnlyArray(proxy),
                    current_motion=FakeVpiReadOnlyArray(proxy.copy()), rescale_backend="CUDA",
                    pyramid_backend_name="PVA", config=self.config, timings={}))
                capture.record("s16", dict(previous_s16=FakeVpiReadOnlyArray(proxy.astype(np.int16)), timings={}))
                capture.record("harris", dict(detected_count=0, harris_capacity=None, timings={}))
                if fail_capture:
                    capture.error = "generated capture failure"
                if unexpected:
                    self.failed = True
                    raise error_type("generated backend fault")
                raise error_type("PVA Harris returned zero features; motion is unavailable")

            def close(self):
                if fail_close:
                    raise RuntimeError("generated cleanup failure")
                self.closed = True

        return SimpleNamespace(ReuseMotionV12=Estimator)

    def evaluate(self, **options):
        modules, error_type = self.fake_modules()
        pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
        row = dict(previous_index=0, current_index=1, kind="aot_adjacent")
        with patch.dict(sys.modules, modules), patch.object(diagnostic, "WIDTH", 4), \
                patch.object(diagnostic, "HEIGHT", 4):
            try:
                result = diagnostic.evaluate_pair(self.fake_estimator(error_type, **options),
                    SimpleNamespace(optical_flow_backend="PVA"), object(), pixels, pixels.copy(), row)
            except BaseException as exc:
                return row, exc
        return result, None

    def test_zero_harris_is_completed_diagnostic_with_retained_failure(self):
        row, error = self.evaluate()
        self.assertIsNone(error)
        self.assertTrue(row["completed"])
        self.assertTrue(row["expected_unavailable"])
        self.assertEqual(row["outcome"], "motion_unavailable")
        self.assertEqual(row["capture"]["harris"]["raw_count"], 0)
        self.assertEqual(row["captured_stages"], ["pixels", "proxy", "s16", "harris"])
        self.assertTrue(row["lifecycle"]["closed"])
        self.assertIsNone(row["global_fit"])

    def test_capture_failure_cannot_masquerade_as_expected_unavailable(self):
        row, error = self.evaluate(fail_capture=True)
        self.assertIsInstance(error, ValueError)
        self.assertFalse(row["completed"])
        self.assertFalse(row["expected_unavailable"])
        self.assertIn("unexpected_error", row)
        self.assertEqual(row["capture_error"], "generated capture failure")
        self.assertTrue(row["lifecycle"]["closed"])

    def test_unexpected_backend_failure_retains_capture_and_closes(self):
        row, error = self.evaluate(unexpected=True)
        self.assertIsInstance(error, ValueError)
        self.assertFalse(row["completed"])
        self.assertFalse(row["expected_unavailable"])
        self.assertTrue(row["lifecycle"]["failed"])
        self.assertTrue(row["lifecycle"]["closed"])
        self.assertEqual(row["capture"]["harris"]["raw_count"], 0)

    def test_cleanup_failure_cannot_leave_completed_true(self):
        row, error = self.evaluate(fail_close=True)
        self.assertIsNotNone(error)
        self.assertFalse(row["completed"])
        self.assertFalse(row["lifecycle"]["closed"])
        self.assertIn("cleanup_error", row)


class AotGeneratedTranslationTests(unittest.TestCase):
    def test_known_shift_moves_content_without_wrap_or_source_mutation(self):
        original = np.arange(30, dtype=np.uint8).reshape(5, 6)
        before = original.copy()
        actual = diagnostic.translate_no_wrap(original, 2, -1, 127)
        expected = np.full_like(original, 127)
        expected[:4, 2:] = original[1:, :4]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(original, before)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(actual.shape, original.shape)

    def test_opposite_shift_does_not_wrap(self):
        original = np.arange(42, dtype=np.uint8).reshape(6, 7)
        actual = diagnostic.translate_no_wrap(original, -2, 1, 200)
        expected = np.full_like(original, 200)
        expected[1:, :5] = original[:5, 2:]
        np.testing.assert_array_equal(actual, expected)

    def test_zero_shift_preserves_codes(self):
        original = np.array([[0, 255, 128], [3, 21, 180]], dtype=np.uint8)
        actual = diagnostic.translate_no_wrap(original, 0, 0, 77)
        np.testing.assert_array_equal(actual, original)

    def test_invalid_translation_or_representation_rejected(self):
        image = np.zeros((4, 6), dtype=np.uint8)
        for dx, dy, fill in ((6, 0, 128), (0, -4, 128), (0, 0, 256), (0, 0, -1)):
            with self.subTest(dx=dx, dy=dy, fill=fill), self.assertRaises(ValueError):
                diagnostic.translate_no_wrap(image, dx, dy, fill)
        with self.assertRaises(ValueError):
            diagnostic.translate_no_wrap(image.astype(np.int16), 0, 0, 128)

    def test_fixed_controls_share_texture_and_preserve_known_shifts(self):
        # Same algorithm and seed, reduced generated geometry for a fast pure test.
        with patch.object(diagnostic, "WIDTH", 96), patch.object(diagnostic, "HEIGHT", 64):
            first = diagnostic.generated_controls()
            second = diagnostic.generated_controls()
        self.assertEqual([row[0] for row in first], list(diagnostic.CONTROL_NAMES))
        self.assertEqual(len(first), 5)
        for row, repeated in zip(first, second):
            name, previous, current, shift = row
            self.assertEqual(previous.shape, (64, 96))
            self.assertEqual(previous.dtype, np.uint8)
            np.testing.assert_array_equal(previous, repeated[1])
            np.testing.assert_array_equal(current, repeated[2])
            if name.endswith("_static"):
                self.assertEqual(shift, (0, 0))
                np.testing.assert_array_equal(previous, current)
            else:
                self.assertEqual(shift, (4, -2))
                expected = diagnostic.translate_no_wrap(previous, 4, -2, 128)
                np.testing.assert_array_equal(current, expected)
        high, low = first[0][1], first[2][1]
        np.testing.assert_array_equal((high.astype(np.int16) - 16) // 14,
                                      low.astype(np.int16) - 120)
        self.assertGreater(int(high.max()) - int(high.min()), int(low.max()) - int(low.min()))
        self.assertTrue(np.all(first[4][1] == 128))
        self.assertGreaterEqual(int(high.min()), 16)
        self.assertLessEqual(int(high.max()), 226)
        self.assertGreaterEqual(int(low.min()), 120)
        self.assertLessEqual(int(low.max()), 135)


if __name__ == "__main__":
    unittest.main()
