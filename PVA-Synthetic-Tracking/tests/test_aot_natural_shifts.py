"""Generated/mocked natural-shift tests; no media, SSH, VPI, or detector runs."""
import base64
import copy
import gzip
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location("test_scope_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


natural = load_script("diagnose_aot_natural_shifts")
factor = load_script("diagnose_aot_feature_factors")
helper = load_script("diagnose_aot_features")


class AotNaturalFrozenDependenciesTests(unittest.TestCase):
    def test_existing_helpers_are_exact_frozen_sources(self):
        self.assertEqual(hashlib.sha256((SCRIPTS / "diagnose_aot_features.py").read_bytes()).hexdigest(),
                         "bef64132f13112435d006642413fa706bd0e70f3a19e4001ab0d382518e6c2e4")
        self.assertEqual(hashlib.sha256((SCRIPTS / "diagnose_aot_feature_factors.py").read_bytes()).hexdigest(),
                         "f0234ad5e1f4ff5686b979adea8086ab00783ef82a83c6fe6751166f42719c61")

    def test_bad_helper_hash_is_rejected_before_import(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "is_symlink", return_value=False), \
                patch.object(natural, "sha", return_value="f" * 64), \
                patch.object(natural.importlib.util, "spec_from_file_location",
                             side_effect=AssertionError("unverified source imported")):
            with self.assertRaisesRegex(ValueError, "helper differs"):
                natural.load_module(Path("/generated/helper.py"), "a" * 64, "generated")


class AotNaturalScopeTests(unittest.TestCase):
    def test_only_exact_new_natural_workspace_is_allowed(self):
        valid = Path("/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3")
        self.assertEqual(natural.scope_path(valid), valid)
        for bad in ("/", "/tmp", "/tmp/seaqr_aot_natural_shifts_20260927_short",
                    "/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3/../other",
                    "/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3/extra",
                    "/tmp/seaqr_aot_factors_20260927_A1b2C3",
                    "/tmp/seaqr_aot_features_20260927_A1b2C3",
                    "/tmp/seaqr_aot_pilot_20260927_A1b2C3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                natural.scope_path(bad)

    def test_root_rejected_before_any_filesystem_read(self):
        with patch.object(natural.os, "geteuid", return_value=0), \
                patch.object(Path, "is_dir", side_effect=AssertionError("filesystem touched")):
            with self.assertRaisesRegex(ValueError, "as root"):
                natural.workspace_guard("/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3")

    def test_success_failure_and_compressed_evidence_are_not_overwritten(self):
        for name in ("result.json", "result.json.gz", "compression_receipt.json", "failure.json"):
            for linked in (False, True):
                with self.subTest(name=name, linked=linked), \
                        patch.object(natural.os, "geteuid", return_value=501), \
                        patch.object(Path, "is_dir", return_value=True), \
                        patch.object(Path, "exists", lambda p: p.name == name and not linked), \
                        patch.object(Path, "is_symlink", lambda p: p.name == name and linked):
                    with self.assertRaisesRegex(ValueError, "existing"):
                        natural.workspace_guard("/tmp/seaqr_aot_natural_shifts_20260927_A1b2C3")


class AotNaturalFrozenCopyTests(unittest.TestCase):
    def test_all_seven_integer_shifts_copy_exact_codes_with_correct_sign(self):
        original = np.arange(120, dtype=np.uint8).reshape(10, 12)
        before = original.copy()
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2)):
            expected = np.full_like(original, 128)
            for y in range(10):
                for x in range(12):
                    if 0 <= x - dx < 12 and 0 <= y - dy < 10:
                        expected[y, x] = original[y - dy, x - dx]
            actual = helper.translate_no_wrap(original, dx, dy, 128)
            with self.subTest(dx=dx, dy=dy):
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(original, before)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, (10, 12))
                self.assertFalse(np.shares_memory(original, actual))

    def test_read_only_input_is_not_modified_by_copy(self):
        image = np.array([[0, 255, 19], [78, 121, 216], [1, 2, 3]], np.uint8)
        image.setflags(write=False)
        actual = helper.translate_no_wrap(image, 1, -1, 128)
        np.testing.assert_array_equal(actual, [[128, 78, 121], [128, 1, 2], [128, 128, 128]])
        self.assertFalse(image.flags.writeable)


def manifest_fixture():
    hashes = dict(script_sha256="a"*64, tests_sha256="b"*64, plan_sha256="c"*64)
    manifest = dict(schema=natural.PLAN_SCHEMA, input_workspace=str(natural.INPUT_WORKSPACE),
        helper_sha256=natural.HELPER_SHA, factor_helper_sha256=natural.FACTOR_SHA,
        baseline_harness_sha256=natural.HARNESS_SHA,
        baseline_journal_sha256=natural.BASELINE_JOURNAL_SHA,
        reference_factor_result_sha256=natural.REFERENCE_FACTOR_SHA,
        previous_indices=list(natural.PREVIOUS_INDICES), shifts=[list(s) for s in natural.SHIFTS],
        control_names=list(natural.CONTROL_NAMES), control_seed=20260927,
        arm=copy.deepcopy(natural.ARM), motion_pair_calls=61, **hashes)
    return manifest, hashes


class AotNaturalManifestTests(unittest.TestCase):
    def test_exact_sixty_one_cases_and_only_frozen_arm(self):
        expected = [f"aot_prev{i:03d}_dx{dx:+d}_dy{dy:+d}"
            for i in (0, 42, 85, 127, 170, 212, 255, 298)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))]
        expected += ["high_contrast_static", "high_contrast_translated", "low_contrast_static",
                     "low_contrast_translated", "flat_static"]
        self.assertEqual(natural.case_ids(), expected)
        self.assertEqual(len(expected), 61)
        self.assertEqual(len(set(expected)), 61)
        self.assertEqual(natural.ARM, factor.ARMS[2])
        natural.validate_manifest(*manifest_fixture())

    def test_source_shift_arm_count_and_reference_changes_rejected(self):
        changes = dict(schema="other", input_workspace="/tmp/other", control_seed=123,
            motion_pair_calls=60, previous_indices=[0, 42, 85, 127, 170, 212, 255, 299],
            shifts=[[0, 0]], control_names=["other"], arm=copy.deepcopy(factor.ARMS[4]),
            helper_sha256="f"*64, factor_helper_sha256="f"*64,
            baseline_harness_sha256="f"*64, baseline_journal_sha256="f"*64,
            reference_factor_result_sha256="f"*64)
        for key, value in changes.items():
            manifest, hashes = manifest_fixture()
            manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                natural.validate_manifest(manifest, hashes)

    def test_indices_and_shifts_are_strict_integers_not_bool_or_float(self):
        for mode in ("index", "shift_bool", "shift_float", "gain", "capacity", "calls"):
            manifest, hashes = manifest_fixture()
            if mode == "index": manifest["previous_indices"][0] = False
            elif mode == "shift_bool": manifest["shifts"][1][0] = True
            elif mode == "shift_float": manifest["shifts"][1][0] = 1.0
            elif mode == "gain": manifest["arm"]["harris_gain"] = 16.0
            elif mode == "capacity": manifest["arm"]["harris_capacity"] = 19866.0
            else: manifest["motion_pair_calls"] = 61.0
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                natural.validate_manifest(manifest, hashes)

    def test_all_new_artifacts_are_hash_bound(self):
        for key in ("script_sha256", "tests_sha256", "plan_sha256"):
            manifest, hashes = manifest_fixture()
            manifest[key] = "f"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                natural.validate_manifest(manifest, hashes)


class AotNaturalCaseGenerationTests(unittest.TestCase):
    def test_known_current_is_copied_previous_not_actual_next_image(self):
        frames = {i: np.arange(120, dtype=np.uint8).reshape(10, 12)
                  for i in natural.PREVIOUS_INDICES}
        for frame in frames.values():
            frame.setflags(write=False)
        source_rows = [dict(source_frame=i+3, timestamp_ns=str(10**18 + i*100_000_000),
                            img_name=f"generated_{i}.png", entities=["not used for selection"])
                       for i in range(300)]
        image = np.zeros((10, 12), np.uint8)
        bridge = [(name, image, image, (4, -2) if name.endswith("translated") else (0, 0))
                  for name in natural.CONTROL_NAMES]
        fake = SimpleNamespace(translate_no_wrap=helper.translate_no_wrap, generated_controls=lambda: bridge)
        results = list(natural.generate_cases(fake, frames, source_rows))
        self.assertEqual([r[0]["case_id"] for r in results], natural.case_ids())
        for row, previous, current, shift in results[:56]:
            index = row["source_previous_index"]
            self.assertIs(previous, frames[index])
            self.assertEqual(row["previous_index"], index)
            self.assertEqual(row["current_index"], index + 1)
            self.assertEqual(row["expected_shift_xy"], list(shift))
            self.assertEqual(set(row["source_previous"]), {"source_frame", "timestamp_ns", "img_name"})
            self.assertEqual(row["source_previous"]["timestamp_ns"], source_rows[index]["timestamp_ns"])
            np.testing.assert_array_equal(current, helper.translate_no_wrap(previous, *shift, 128))
            self.assertFalse(np.shares_memory(previous, current))
            self.assertFalse(previous.flags.writeable)


class AotNaturalPreflowSupportTests(unittest.TestCase):
    def test_positive_and_negative_overlap_edges_are_half_open(self):
        for shift, bounds in (((4, -2), (128, 130, 2316, 1920)),
                              ((-4, 2), (132, 128, 2320, 1918)),
                              ((0, 0), (128, 128, 2320, 1920))):
            x0, y0, x1, y1 = bounds
            points = np.array([[x0, y0], [x1-0.5, y1-0.5], [x0-0.5, y0],
                               [x0, y0-0.5], [x1, y0], [x0, y1], [np.nan, 300]])
            result = natural.preflow_support(points, shift)
            with self.subTest(shift=shift):
                self.assertEqual(result["selected_count"], 7)
                self.assertEqual(result["fixed_support_count"], 2)
                self.assertEqual(result["selected_indices"], [0, 1])
                self.assertEqual(result["mask"], [True, True, False, False, False, False, False])

    def test_one_pixel_support_is_native_not_proxy_pixels(self):
        points = np.array([[128, 300], [129, 300], [2318.5, 300], [2319.5, 300]])
        self.assertEqual(natural.preflow_support(points, (1, 0))["mask"], [True, True, True, False])
        self.assertEqual(natural.preflow_support(points, (-1, 0))["mask"], [False, True, True, True])

    def test_support_depends_on_previous_and_expected_only(self):
        previous = np.array([[200.5, 300.5], [400.5, 500.5]])
        before = previous.copy()
        result = natural.preflow_support(previous, (1, 0))
        self.assertEqual(result["fixed_support_count"], 2)
        self.assertEqual(result["selected_indices"], [0, 1])
        np.testing.assert_array_equal(previous, before)
        self.assertEqual(natural.preflow_support(previous, (1, 0)), result)

    def test_empty_support_and_invalid_input_are_explicit(self):
        result = natural.preflow_support(np.empty((0, 2)), (0, 0))
        self.assertEqual(result["selected_count"], result["fixed_support_count"])
        self.assertEqual(result["mask"], [])
        for points, shift in ((np.zeros((2, 3)), (0, 0)), (np.zeros((1001, 2)), (0, 0)),
                              (np.zeros((2, 2)), (True, 0)), (np.zeros((2, 2)), (1.0, 0))):
            with self.assertRaises(ValueError):
                natural.preflow_support(points, shift)


class AotNaturalRawBytesTests(unittest.TestCase):
    def test_nonfinite_payloads_signed_zero_and_flags_roundtrip_exactly(self):
        bits = np.array([0x7fc00001, 0x7fc12345, 0x7f800000, 0xff800000,
                         0x80000000, 0, 0x3f800000, 0xbf800000], dtype=np.uint32)
        arrays = [bits.view(np.float32).reshape(4, 2),
                  np.array([0, 1, 2, 255], dtype=np.uint8),
                  np.array([[1.25, -0.0], [np.inf, -np.inf]], dtype=">f4"),
                  np.arange(24, dtype=np.float32).reshape(12, 2)[::2],
                  np.empty((0, 2), np.float32), np.empty(0, np.uint8)]
        for array in arrays:
            with self.subTest(dtype=array.dtype, shape=array.shape):
                record = natural.encode_raw(array)
                restored = natural.decode_raw(record)
                self.assertEqual(restored.dtype.str, array.dtype.str)
                self.assertEqual(restored.shape, array.shape)
                self.assertEqual(restored.tobytes(), array.tobytes(order="C"))
                self.assertEqual(record["sha256"], hashlib.sha256(array.tobytes(order="C")).hexdigest())
                self.assertFalse(np.shares_memory(restored, array))

    def test_only_bounded_float32_xy_and_uint8_flags_are_encoded(self):
        for array in (np.zeros((2, 2), np.float64), np.zeros((2, 2), np.uint32),
                      np.zeros((2, 3), np.float32), np.zeros(2, np.float32),
                      np.zeros((1001, 2), np.float32), np.zeros(1001, np.uint8),
                      np.zeros(2, bool), np.zeros((2, 1), np.uint8)):
            with self.subTest(dtype=array.dtype, shape=array.shape), self.assertRaises(ValueError):
                natural.encode_raw(array)

    def test_malformed_metadata_bytes_and_hashes_are_rejected(self):
        source = natural.encode_raw(np.array([[0.0, -0.0]], np.float32))
        invalid = [dict(dtype="<f8"), dict(dtype="object"), dict(shape=[1, True]),
                   dict(shape=[1.0, 2]), dict(shape=[-1, 2]), dict(shape=[1001, 2]),
                   dict(shape=[2, 1]), dict(shape=[]), dict(shape=(1, 2)),
                   dict(order="F"), dict(byte_count=True), dict(byte_count=7),
                   dict(base64="!" * len(source["base64"])), dict(base64=""),
                   dict(sha256="not-a-sha"), dict(sha256="a" * 64),
                   dict(base64=base64.b64encode(b"changed!").decode())]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                natural.decode_raw(dict(source, **change))


def attrition_row():
    """Eight fixed-cohort points, one per disjoint outcome, plus an exterior point."""
    selected = np.array([[100 + 30*i, 200] for i in range(8)] + [[0, 0]], np.float32)
    p = selected * np.float32(2) + np.float32(0.5)
    q, b = p.copy(), p.copy()
    q[:, 0] += 1
    q[0, 0] = np.nan
    flags, back_flags = np.zeros(9, np.uint8), np.zeros(9, np.uint8)
    flags[1] = 2
    q[2, 0] = -1
    q[3, 0] = p[3, 0] + 121
    b[4, 0] = np.inf
    back_flags[5] = 255
    b[6, 0] += 4
    raw = dict(current_motion_points=(q - np.float32(.5)) / np.float32(2),
               forward_status_array=flags,
               backward_motion_points=(b - np.float32(.5)) / np.float32(2),
               backward_status_array=back_flags)
    row = dict(expected_shift_xy=[1, 0],
        capture=dict(selection=dict(coordinates=dict(values=selected.tolist())),
                     preflow_support=natural.preflow_support(p, (1, 0)),
                     raw_flow_bytes={k: natural.encode_raw(v) for k, v in raw.items()}),
        correspondence=dict(correspondences=[dict(previous_xy=p[i].tolist(), current_xy=q[i].tolist())
                                              for i in (7, 8)]),
        generated_control_error=dict(accepted_interior_points=1))
    return row


class AotNaturalAttritionTests(unittest.TestCase):
    def test_disjoint_rejections_preserve_fixed_denominator_and_do_not_make_zero_error(self):
        row = attrition_row()
        natural.attach_attrition(row)
        audit = row["fixed_support_attrition"]
        self.assertEqual(audit["selected_count"], 9)
        self.assertEqual(audit["preflow_support_count"], 8)
        self.assertEqual(audit["accepted_support_count"], 1)
        self.assertEqual(audit["lost_support_count"], 7)
        self.assertEqual(audit["support_survival_fraction"], 1/8)
        self.assertEqual(audit["all_accepted_points"], 2)
        self.assertEqual(set(audit["disjoint_stages"].values()), {1})
        self.assertEqual(sum(audit["disjoint_stages"].values()), 8)
        self.assertEqual(audit["direct_truth_error_px"], [None]*7 + [0.0])
        self.assertEqual(audit["missing_or_rejected_truth_observations"], 7)
        self.assertEqual(audit["accepted_truth_within_0_1_count"], 1)
        self.assertEqual(audit["accepted_truth_within_0_5_count"], 1)
        self.assertEqual(audit["forward_status_values"], {"0": 7, "2": 1})
        self.assertEqual(audit["backward_status_values"], {"0": 7, "255": 1})
        self.assertIsNone(audit["raw_finite_coordinate_error_px"][0])
        self.assertEqual(audit["raw_finite_coordinate_error_px"][1], 0.0)
        self.assertFalse(audit["additional_gate"])

    def test_observed_q_interior_summary_cannot_change_support(self):
        left, right = attrition_row(), attrition_row()
        right["generated_control_error"]["accepted_interior_points"] = 0
        natural.attach_attrition(left)
        natural.attach_attrition(right)
        for key in ("preflow_support_count", "accepted_support_count", "lost_support_count",
                    "support_survival_fraction", "direct_truth_error_px", "disjoint_stages"):
            self.assertEqual(left["fixed_support_attrition"][key], right["fixed_support_attrition"][key])

    def test_no_selection_no_flow_and_zero_support_are_explicit(self):
        row = dict(capture={})
        natural.attach_attrition(row)
        self.assertEqual(row["fixed_support_attrition"], dict(available=False, reason="selection_not_reached"))
        row = attrition_row()
        del row["capture"]["raw_flow_bytes"]
        row["correspondence"] = None
        natural.attach_attrition(row)
        self.assertEqual(row["fixed_support_attrition"]["disjoint_stages"], {"flow_not_observed": 8})
        self.assertEqual(row["fixed_support_attrition"]["direct_truth_error_px"], [None]*8)
        row["capture"]["selection"]["coordinates"]["values"] = []
        row["capture"]["preflow_support"] = natural.preflow_support(np.empty((0, 2)), (1, 0))
        natural.attach_attrition(row)
        self.assertIsNone(row["fixed_support_attrition"]["support_survival_fraction"])
        self.assertEqual(row["fixed_support_attrition"]["direct_truth_error_px"], [])

    def test_mismatch_between_saved_flow_and_estimator_acceptance_fails_closed(self):
        for mode in ("missing_accepted", "unknown_previous", "missing_raw", "wrong_raw_count"):
            row = attrition_row()
            if mode == "missing_accepted": row["correspondence"]["correspondences"] = []
            elif mode == "unknown_previous":
                row["correspondence"]["correspondences"][0]["previous_xy"] = [777, 888]
            elif mode == "missing_raw": del row["capture"]["raw_flow_bytes"]
            else: row["capture"]["raw_flow_bytes"]["forward_status_array"] = natural.encode_raw(np.zeros(8, np.uint8))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                natural.attach_attrition(row)


class AotNaturalCaptureTests(unittest.TestCase):
    def test_selection_uses_exact_native_centers_and_flow_keeps_original_bytes(self):
        cls = natural.make_capture_class(factor, helper)
        cls.active_shift = (1, 0)
        capture = cls(helper, natural.ARM)
        points = np.array([[63, 200], [64, 200], [1159, 200]], np.float32)
        capture.record("selection", dict(detected_points=points, selected_indices=np.arange(3),
            detected_scores=np.ones(3, np.float32), eligible_mask=np.ones(3, bool),
            feature_exclusions={}, motion_size=(1224, 1024), timings={}))
        self.assertEqual(capture.data["preflow_support"]["mask"], [False, True, True])
        raw = dict(current_motion_points=np.array([[np.nan, -0.0], [np.inf, -np.inf]], np.float32),
            backward_motion_points=np.array([[0, 1], [2, 3]], np.float32),
            forward_status_array=np.array([0, 255], np.uint8),
            backward_status_array=np.array([2, 1], np.uint8))
        capture.record("flow", dict(raw, timings={}))
        for key, original in raw.items():
            restored = natural.decode_raw(capture.data["raw_flow_bytes"][key])
            self.assertEqual(restored.tobytes(), original.tobytes())
            self.assertEqual(capture.data["flow"][key]["sha256"], capture.data["raw_flow_bytes"][key]["sha256"])


def invariant_rows():
    rows = []
    def identity(value):
        return dict(dtype="float32", shape=[1, 2], sha256=hashlib.sha256(value.encode()).hexdigest())
    for index in natural.PREVIOUS_INDICES:
        for shift in natural.SHIFTS:
            previous, current = identity(str(index)), identity(str((index, shift)))
            rows.append(dict(case_id=natural.case_id(index, shift), source_previous_index=index,
                expected_shift_xy=list(shift), native_pixel_sha256=dict(current=current["sha256"]),
                capture=dict(pixels=dict(previous=dict(native=copy.deepcopy(previous))),
                    proxy=dict(previous=copy.deepcopy(previous), current=current),
                    s16=dict(previous=copy.deepcopy(previous)),
                    harris=dict(coordinates=copy.deepcopy(previous), scores=copy.deepcopy(previous)),
                    selection=dict(selected_indices=copy.deepcopy(previous)))))
    return rows + [dict(case_id=name) for name in natural.CONTROL_NAMES]


class AotNaturalInvarianceTests(unittest.TestCase):
    def test_all_seven_currents_share_previous_features_with_fixed_complete_order(self):
        rows = invariant_rows()
        result = natural.invariance(rows)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(all(item["equal"].values()) for item in result))
        self.assertEqual(natural.invariance(rows[:7], complete=False), result[:1])

    def test_missing_reordered_or_changed_previous_features_fail(self):
        for mode in ("missing", "reordered", "native", "raw_u32", "selected_indices"):
            rows = invariant_rows()
            if mode == "missing": rows.pop()
            elif mode == "reordered": rows[0], rows[1] = rows[1], rows[0]
            elif mode == "native": rows[1]["capture"]["pixels"]["previous"]["native"]["sha256"] = "f"*64
            elif mode == "raw_u32": rows[1]["capture"]["harris"]["scores"]["sha256"] = "f"*64
            else: rows[1]["capture"]["selection"]["selected_indices"]["sha256"] = "f"*64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                natural.invariance(rows)


class AotNaturalCompressionTests(unittest.TestCase):
    def test_generated_evidence_is_lossless_preserved_and_never_replaced(self):
        with tempfile.TemporaryDirectory(prefix="aot-natural-test-") as directory:
            workspace = Path(directory)
            raw = b'{"generated_test": true, "values": [0, 1, null]}\n' * 100
            (workspace / "result.json").write_bytes(raw)
            receipt = natural.compress_result(workspace)
            self.assertEqual((workspace / "result.json").read_bytes(), raw)
            self.assertEqual(gzip.decompress((workspace / "result.json.gz").read_bytes()), raw)
            self.assertEqual(receipt["decompressed_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(receipt["raw_preserved"])
            before = {name: (workspace / name).read_bytes() for name in natural.OUTPUT_NAMES[:3]}
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                natural.compress_result(workspace)
            self.assertEqual(before, {name: (workspace / name).read_bytes() for name in before})


if __name__ == "__main__":
    unittest.main()
