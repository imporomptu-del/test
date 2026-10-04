import copy
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v54_adapter as adapter


def fixture():
    return dict(guard_xy=np.array(adapter.GUARD_POINTS), core_xy=np.array(adapter.CORE_POINTS),
        guard_history=np.full((8, 144), 96.), guard_current=np.full(144, 96.),
        core_history_on=np.full((8, 625), 96.), core_history_off=np.full((8, 625), 96.))


class FrozenCoreAdapterTests(unittest.TestCase):
    def test_inherited_hard_pins_and_constants(self):
        self.assertEqual(adapter.V53_MODEL_SHA256, "aa490ecf8305dd0c5f3facff83a5fa8fef43c67460eb55572f516b298a079d04")
        self.assertEqual(adapter.V53_TEST_SHA256, "d5c53052fca8f47bf1f380582702e62217a353587b906fe759d564b460e70f05")
        for path, sha in adapter.INHERITED_FILES:
            self.assertTrue(adapter._read_pinned(path, sha))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                adapter._read_pinned(path, "0"*64)
        constants = adapter.model_constants()
        self.assertEqual(constants["inherited_constants"], adapter._V53.model_constants())
        self.assertFalse(constants["clipping_selection_blending_or_fallback"])

    def test_both_pins_checked_before_any_inherited_code_execution(self):
        with patch.object(adapter, "_read_pinned", side_effect=[
                b"raise RuntimeError('should not execute')", ValueError("test pin differs")]):
            with self.assertRaisesRegex(ValueError, "test pin differs"):
                adapter._load_pinned_v53()

    def test_inherited_loader_executes_verified_bytes(self):
        with patch.object(adapter, "_read_pinned", side_effect=[b"sentinel = 53", b"tests"]):
            inherited = adapter._load_pinned_v53()
        self.assertEqual(inherited.sentinel, 53)

    def test_observed_only_interface(self):
        self.assertEqual(tuple(inspect.signature(adapter.predict).parameters), (
            "guard_xy", "core_xy", "guard_history", "guard_current", "core_history_on", "core_history_off"))
        for name in ("core_current_on", "core_current_off", "truth", "source_template", "labels", "amplitude"):
            with self.assertRaises(TypeError):
                adapter.predict(**fixture(), **{name: np.zeros(625)})

    def test_full_canonical_geometry_only(self):
        for name in ("guard_xy", "core_xy"):
            for operation in (lambda a: a[::-1], lambda a: a[:-1],
                              lambda a: np.concatenate([a[:1], a[:-1]]),
                              lambda a: a+1, lambda a: a.astype(complex),
                              lambda a: a.astype(bool)):
                args = fixture()
                args[name] = operation(args[name])
                with self.subTest(name=name, shape=args[name].shape):
                    with self.assertRaises(ValueError):
                        adapter.predict(**args)

    def test_geometry_is_disjoint_and_has_frozen_counts(self):
        self.assertEqual(len(adapter.GUARD_POINTS), 144)
        self.assertEqual(len(adapter.CORE_POINTS), 625)
        self.assertFalse(set(adapter.GUARD_POINTS) & set(adapter.CORE_POINTS))
        self.assertEqual(adapter.CORE_POINTS[0], (52, 52))
        self.assertEqual(adapter.CORE_POINTS[-1], (76, 76))

    def test_observation_shape_and_type_checks(self):
        for name in ("guard_history", "guard_current", "core_history_on", "core_history_off"):
            for operation in (lambda a: a[:-1], lambda a: a.reshape(-1, 1),
                              lambda a: a.astype(complex), lambda a: a.astype(bool),
                              lambda a: a.astype(str), lambda a: a.astype(object)):
                args = fixture()
                args[name] = operation(args[name])
                with self.subTest(name=name, shape=args[name].shape):
                    with self.assertRaises(ValueError):
                        adapter.predict(**args)

    def test_six_exact_branches_baseline_constant_predictions(self):
        result = adapter.predict(**fixture())
        self.assertEqual(tuple(result["predictions"]), adapter.METHODS)
        self.assertEqual(result["total_core_count"], 625)
        self.assertEqual(result["total_guard_count"], 144)
        for name, prediction in result["predictions"].items():
            for pair in ("on", "off"):
                np.testing.assert_array_equal(prediction[pair]["values"], 96)
                self.assertTrue(prediction[pair]["available"].all())
                self.assertEqual(prediction[pair]["unavailable_reasons"], (None,)*625)
            if name in ("median8", "median3"):
                self.assertIsNone(prediction["guard_fit_sha256"])
            else:
                self.assertEqual(len(prediction["guard_fit_sha256"]), 64)

    def test_all_four_offsets_preserved_without_blending(self):
        args = fixture()
        args["guard_current"][args["guard_xy"][:, 0] >= 64] += 16
        result = adapter.predict(**args)
        np.testing.assert_array_equal(result["predictions"]["offset_left_right_fold0"]["on"]["values"], 112)
        np.testing.assert_array_equal(result["predictions"]["offset_left_right_fold1"]["on"]["values"], 96)
        for split in ("left_right", "checkerboard"):
            for fold in (0, 1):
                model = result["guard_crossfits"][split]["fits"]["median_offset"][str(fold)]
                branch = result["predictions"][f"offset_{split}_fold{fold}"]
                self.assertEqual(branch["guard_fit_sha256"], model["model_sha256"])
                np.testing.assert_array_equal(branch["on"]["values"], 96+model["offset_dn"])

    def test_guard_shift_changes_offsets_not_temporal_baselines(self):
        args = fixture()
        args["guard_current"] += 8
        result = adapter.predict(**args)
        for name, branch in result["predictions"].items():
            expected = 96 if name in ("median8", "median3") else 104
            for pair in ("on", "off"):
                np.testing.assert_array_equal(branch[pair]["values"], expected)

    def test_only_guard_values_reach_inherited_fit_and_predict(self):
        args = fixture()
        args["core_history_on"][:] = 10000
        fit_calls, predict_calls = [], []
        original_fit, original_predict = adapter._V53.fit, adapter._V53.predict
        def checked_fit(xy, slow, current, *other, **kwargs):
            fit_calls.append(len(xy))
            self.assertTrue(set(map(tuple, xy)) <= set(adapter.GUARD_POINTS))
            np.testing.assert_array_equal(slow, 96)
            np.testing.assert_array_equal(current, 96)
            return original_fit(xy, slow, current, *other, **kwargs)
        def checked_predict(model, xy, slow):
            predict_calls.append(len(xy))
            self.assertTrue(set(map(tuple, xy)) <= set(adapter.GUARD_POINTS))
            return original_predict(model, xy, slow)
        with patch.object(adapter._V53, "fit", side_effect=checked_fit), \
                patch.object(adapter._V53, "predict", side_effect=checked_predict):
            result = adapter.predict(**args)
        self.assertEqual(sorted(fit_calls), [69, 72, 72, 75])
        self.assertEqual(sorted(predict_calls), [69, 72, 72, 75])
        np.testing.assert_array_equal(result["predictions"]["offset_left_right_fold0"]["on"]["values"], 10000)

    def test_core_history_changes_neither_guard_fits_nor_other_pair_predictions(self):
        args = fixture()
        first = adapter.predict(**args)
        args["core_history_on"][:, 312] += 4
        second = adapter.predict(**args)
        for split in ("left_right", "checkerboard"):
            self.assertEqual(first["guard_crossfits"][split]["crossfit_sha256"],
                             second["guard_crossfits"][split]["crossfit_sha256"])
        for name in adapter.METHODS:
            np.testing.assert_array_equal(first["predictions"][name]["off"]["values"],
                                          second["predictions"][name]["off"]["values"])
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_each_fit_ignores_its_heldout_guard_current(self):
        args = fixture()
        first = adapter.predict(**args)
        args["guard_current"][args["guard_xy"][:, 0] < 64] += 10
        second = adapter.predict(**args)
        self.assertEqual(first["predictions"]["offset_left_right_fold0"]["guard_fit_sha256"],
                         second["predictions"]["offset_left_right_fold0"]["guard_fit_sha256"])
        self.assertNotEqual(first["predictions"]["offset_left_right_fold1"]["guard_fit_sha256"],
                            second["predictions"]["offset_left_right_fold1"]["guard_fit_sha256"])

    def test_guard_crossfits_equal_frozen_model_on_identical_inputs(self):
        args = fixture()
        args["guard_current"] += np.arange(144) % 5
        result = adapter.predict(**args)
        for split in ("left_right", "checkerboard"):
            direct = adapter._V53.crossfit(args["guard_xy"], np.full(144, 96.), args["guard_current"], split)
            self.assertEqual(direct["crossfit_sha256"], result["guard_crossfits"][split]["crossfit_sha256"])

    def test_stationary_source_history_is_absorbed_by_all_six_methods(self):
        args = fixture()
        args["core_history_on"][:, 312] += 4
        result = adapter.predict(**args)
        for name in adapter.METHODS:
            predicted_on = result["predictions"][name]["on"]["values"][312]
            predicted_off = result["predictions"][name]["off"]["values"][312]
            self.assertEqual((100-predicted_on)-(96-predicted_off), 0)

    def test_appearing_source_retains_increment_with_same_observed_histories(self):
        result = adapter.predict(**fixture())
        for name in adapter.METHODS:
            predicted_on = result["predictions"][name]["on"]["values"][312]
            predicted_off = result["predictions"][name]["off"]["values"][312]
            self.assertEqual((100-predicted_on)-(96-predicted_off), 4)

    def test_median3_and_median8_history_self_subtraction_differ(self):
        args = fixture()
        args["core_history_on"][:5, 312] += 4
        result = adapter.predict(**args)
        self.assertEqual(result["predictions"]["median8"]["on"]["values"][312], 100)
        self.assertEqual(result["predictions"]["median3"]["on"]["values"][312], 96)
        for name in adapter.METHODS[2:]:
            self.assertEqual(result["predictions"][name]["on"]["values"][312], 100)

    def test_paired_increment_does_not_prove_correct_raw_residual(self):
        args = fixture()
        args["guard_current"] += 8
        result = adapter.predict(**args)
        branch = result["predictions"]["offset_left_right_fold0"]
        on_residual = 100-branch["on"]["values"][312]
        off_residual = 96-branch["off"]["values"][312]
        self.assertEqual(on_residual, -4)
        self.assertEqual(on_residual-off_residual, 4)

    def test_missing_first_prior_disables_median8_not_median3(self):
        args = fixture()
        for key in ("core_history_on", "core_history_off"):
            args[key][0, 312] = np.nan
        result = adapter.predict(**args)
        for name in adapter.METHODS:
            for pair in ("on", "off"):
                prediction = result["predictions"][name][pair]
                expected = [] if name == "median3" else [312]
                self.assertEqual(np.flatnonzero(~prediction["available"]).tolist(), expected)
        self.assertEqual(result["predictions"]["median8"]["on"]["unavailable_reasons"][312], "nonfinite_history")
        self.assertEqual(result["predictions"]["offset_left_right_fold0"]["on"]["unavailable_reasons"][312], "nonfinite_core_median8")

    def test_missing_recent_prior_disables_all_methods_without_imputation(self):
        args = fixture()
        args["core_history_on"][-1, 9] = np.inf
        result = adapter.predict(**args)
        for name in adapter.METHODS:
            self.assertEqual(np.flatnonzero(~result["predictions"][name]["on"]["available"]).tolist(), [9])
            self.assertTrue(result["predictions"][name]["off"]["available"].all())

    def test_missing_guard_history_remains_visible_in_certificates(self):
        args = fixture()
        args["guard_history"][0, 0] = np.nan
        result = adapter.predict(**args)
        self.assertEqual(np.flatnonzero(~result["guard_median8"]["available"]).tolist(), [0])
        self.assertEqual(result["guard_median8"]["unavailable_reasons"][0], "nonfinite_history")
        self.assertTrue(all(result["predictions"][name]["on"]["available"].all() for name in adapter.METHODS))
        for split in ("left_right", "checkerboard"):
            used = sum(f["training_used_count"] for f in result["guard_crossfits"][split]["fits"]["median_offset"].values())
            self.assertEqual(used, 143)

    def test_missing_guard_current_left_preserves_all_four_fold_outcomes(self):
        args = fixture()
        args["guard_current"][args["guard_xy"][:, 0] < 64] = np.nan
        result = adapter.predict(**args)
        for name in adapter.METHODS:
            prediction = result["predictions"][name]["on"]
            self.assertEqual(int(prediction["available"].sum()), 0 if name == "offset_left_right_fold1" else 625)
        self.assertEqual(result["predictions"]["offset_left_right_fold1"]["on"]["unavailable_reasons"],
                         ("fit_unavailable:insufficient_finite_training_rows",)*625)

    def test_all_guard_current_missing_no_offset_fallback(self):
        args = fixture()
        args["guard_current"][:] = np.nan
        result = adapter.predict(**args)
        for name in adapter.METHODS:
            prediction = result["predictions"][name]["on"]
            self.assertEqual(int(prediction["available"].sum()), 625 if name in ("median8", "median3") else 0)
            if name not in ("median8", "median3"):
                self.assertTrue(np.isnan(prediction["values"]).all())

    def test_nonfinite_median_arithmetic_is_explicit(self):
        args = fixture()
        args["core_history_on"][:, 3] = 1e308
        result = adapter.predict(**args)
        prediction = result["predictions"]["median8"]["on"]
        self.assertFalse(prediction["available"][3])
        self.assertEqual(prediction["unavailable_reasons"][3], "nonfinite_median_arithmetic")
        self.assertTrue(result["predictions"]["median3"]["on"]["available"][3])

    def test_nonfinite_guard_median_produces_unavailable_fits_not_imputation(self):
        args = fixture()
        args["guard_history"][:] = 1e308
        result = adapter.predict(**args)
        self.assertFalse(result["guard_median8"]["available"].any())
        for name in adapter.METHODS[2:]:
            self.assertFalse(result["predictions"][name]["on"]["available"].any())

    def test_nonfinite_core_offset_addition_is_per_index_unknown(self):
        args = fixture()
        args["guard_history"][:] = 0
        args["guard_current"][:] = 1.5e308
        args["core_history_on"][:, 3] = 5e307
        result = adapter.predict(**args)
        for name in adapter.METHODS[2:]:
            prediction = result["predictions"][name]["on"]
            self.assertEqual(np.flatnonzero(~prediction["available"]).tolist(), [3])
            self.assertEqual(prediction["unavailable_reasons"][3], "nonfinite_offset_addition")

    def test_fit_hash_tampering_rejected_before_application(self):
        args = fixture()
        result = adapter.predict(**args)
        fitted = copy.deepcopy(result["guard_crossfits"]["left_right"]["fits"]["median_offset"]["0"])
        fitted["offset_dn"] = 99
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            adapter._apply_offset(fitted, result["predictions"]["median8"]["on"])

    def test_inputs_unmodified_all_array_outputs_readonly(self):
        args = fixture()
        saved = {key: value.copy() for key, value in args.items()}
        result = adapter.predict(**args)
        for key in args:
            np.testing.assert_array_equal(args[key], saved[key])
        def check(value):
            if isinstance(value, np.ndarray):
                self.assertFalse(value.flags.writeable)
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
        check(result)

    def test_prediction_fingerprint_json_roundtrip_and_mutation(self):
        args = fixture()
        args["core_history_on"][0, 312] = np.nan
        result = adapter.predict(**args)
        restored = json.loads(json.dumps(adapter._plain(result), allow_nan=False))
        self.assertEqual(result["prediction_sha256"], adapter.prediction_fingerprint(restored))
        restored["metadata"]["scoring_performed"] = True
        self.assertNotEqual(result["prediction_sha256"], adapter.prediction_fingerprint(restored))

    def test_metadata_limits_claims(self):
        metadata = adapter.predict(**fixture())["metadata"]
        self.assertFalse(metadata["current_core_or_truth_argument_accepted"])
        self.assertFalse(metadata["core_values_enter_guard_fit"])
        self.assertTrue(metadata["new_guard_to_core_extrapolation_adapter"])
        self.assertTrue(metadata["inherited_guard_only_predict_api_unchanged"])
        self.assertFalse(metadata["source_core_safety_or_real_detection_accuracy_certified"])
        self.assertFalse(metadata["production_decisions_modified"])


if __name__ == "__main__":
    unittest.main()
