import copy
from fractions import Fraction
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v53_offset as module
from accuracy_v53_offset import (
    crossfit, crossfit_fingerprint, fit, model_constants, model_fingerprint,
    predict, split_ids,
)


def fixture():
    xy = np.array([(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                   if 40 <= max(abs(x-64), abs(y-64)) <= 56], dtype=float)
    slow = 96 + 7*np.sin(xy[:, 0]/13) + 5*np.cos(xy[:, 1]/17)
    return xy, slow, slow+8


def residual_fit(residuals):
    xy, _, _ = fixture()
    residuals = np.array(residuals, dtype=float)
    return fit(xy[:len(residuals)], np.zeros(len(residuals)), residuals)


class MedianOffsetTests(unittest.TestCase):
    def test_frozen_constants_no_gain_slopes_clipping_tuning_or_fallback(self):
        c = model_constants()
        self.assertEqual(c["minimum_finite_training_rows"], 4)
        self.assertEqual((c["fixed_gain"], c["fixed_x_slope"], c["fixed_y_slope"]), (1, 0, 0))
        self.assertFalse(c["clipping_or_tuning"])
        self.assertFalse(c["implicit_fallback"])
        self.assertTrue(c["arithmetic_failure_discards_entire_fit"])

    def test_interfaces_exclude_current_from_prediction(self):
        self.assertEqual(tuple(inspect.signature(fit).parameters), ("train_xy", "slow", "current", "loss"))
        self.assertEqual(tuple(inspect.signature(predict).parameters), ("model", "test_xy", "slow"))
        xy, s, c = fixture()
        model = fit(xy, s, c)
        for keyword in ("current", "response", "labels", "threshold"):
            with self.assertRaises(TypeError):
                predict(model, xy, s, **{keyword: c})

    def test_geometry_rejects_core_offgrid_duplicates_and_invalid_types(self):
        for xy in (np.array([[64, 64]]), np.array([[32, 32]]), np.array([[0, 8]]),
                   np.array([[128, 8]]), np.array([[9, 8]]), np.array([[np.nan, 8]]),
                   np.array([[8, np.inf]]), np.array([[8, 8], [8, 8]]),
                   np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=complex),
                   np.ones((2, 2), dtype=object), np.ones((2,)), np.ones((2, 3))):
            with self.subTest(xy=xy):
                with self.assertRaises(ValueError):
                    fit(xy, np.zeros(len(xy)), np.zeros(len(xy)))

    def test_vector_shape_and_type_validation(self):
        xy, slow, current = fixture()
        for value in (slow[:-1], slow[:, None], slow.astype(complex),
                      slow.astype(bool), slow.astype(str), slow.astype(object)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    fit(xy, value, current)
                with self.assertRaises(ValueError):
                    fit(xy, slow, value)

    def test_only_fixed_loss_and_split_names_accepted(self):
        xy, slow, current = fixture()
        for loss in ("ols", "huber", "automatic", None):
            with self.assertRaises(ValueError):
                fit(xy, slow, current, loss)
        with self.assertRaises(ValueError):
            crossfit(xy, slow, current, "adaptive")

    def test_both_frozen_split_formulas_and_counts(self):
        xy, _, _ = fixture()
        np.testing.assert_array_equal(split_ids(xy), xy[:, 0] >= 64)
        np.testing.assert_array_equal(split_ids(xy, "checkerboard"),
            ((xy[:, 0]/8+xy[:, 1]/8) % 2).astype(np.int8))
        self.assertEqual(np.bincount(split_ids(xy)).tolist(), [69, 75])
        self.assertEqual(np.bincount(split_ids(xy, "checkerboard")).tolist(), [72, 72])

    def test_constant_offset_exact_fit(self):
        xy, slow, current = fixture()
        result = fit(xy, slow, current)
        self.assertTrue(result["available"])
        self.assertEqual(result["offset_dn"], 8)
        self.assertEqual(result["objective_mae_dn"], 0)
        np.testing.assert_array_equal(result["median_interval_dn"], [8, 8])
        np.testing.assert_array_equal(result["subgradient_interval"], [-1, 1])
        self.assertEqual(result["residual_order_counts"], dict(below=0, above=0, tied=144))
        np.testing.assert_array_equal(predict(result, xy, slow)["values"], current)

    def test_negative_offset_is_not_clipped(self):
        xy, slow, _ = fixture()
        result = fit(xy, slow, slow-300)
        self.assertEqual(result["offset_dn"], -300)
        self.assertTrue((predict(result, xy, slow)["values"] < 0).all())

    def test_large_positive_offset_is_not_clipped_to_eight_bit_range(self):
        xy, slow, _ = fixture()
        result = fit(xy, slow, slow+300)
        self.assertEqual(result["offset_dn"], 300)
        self.assertTrue((predict(result, xy, slow)["values"] > 255).all())

    def test_even_median_interval_midpoint_and_l1_certificate(self):
        result = residual_fit([0, 2, 4, 6])
        self.assertTrue(result["available"])
        self.assertEqual(result["offset_dn"], 3)
        np.testing.assert_array_equal(result["median_interval_dn"], [2, 4])
        self.assertEqual(result["objective_mae_dn"], 2)
        self.assertEqual(result["residual_order_counts"], dict(below=2, above=2, tied=0))
        np.testing.assert_array_equal(result["subgradient_interval"], [0, 0])

    def test_odd_median_and_tied_counts(self):
        result = residual_fit([0, 0, 5, 8, 10])
        self.assertEqual(result["offset_dn"], 5)
        self.assertEqual(result["objective_mae_dn"], 3.6)
        self.assertEqual(result["residual_order_counts"], dict(below=2, above=2, tied=1))
        np.testing.assert_array_equal(result["median_interval_dn"], [5, 5])
        np.testing.assert_array_equal(result["subgradient_interval"], [-.2, .2])

    def test_tie_even_midpoint_can_round_to_endpoint_with_valid_certificate(self):
        adjacent = np.nextafter(1., np.inf)
        result = residual_fit([0, 1, adjacent, 2])
        self.assertEqual(result["offset_dn"], 1)
        self.assertEqual(result["residual_order_counts"], dict(below=1, above=2, tied=1))
        np.testing.assert_array_equal(result["subgradient_interval"], [-.5, 0])

    def test_both_extreme_same_sign_middle_values_avoid_sum_overflow(self):
        maximum = np.finfo(float).max
        for value in (maximum, -maximum):
            result = residual_fit([value]*4)
            self.assertTrue(result["available"])
            self.assertEqual(result["offset_dn"], value)
            self.assertEqual(result["objective_mae_dn"], 0)

    def test_opposite_sign_extremes_have_finite_correct_objective(self):
        maximum = np.finfo(float).max
        result = residual_fit([-maximum, -maximum, maximum, maximum])
        self.assertTrue(result["available"])
        self.assertEqual(result["offset_dn"], 0)
        self.assertEqual(result["objective_mae_dn"], maximum)
        np.testing.assert_array_equal(result["subgradient_interval"], [0, 0])

    def test_finite_objective_mean_does_not_overflow_its_sum(self):
        result = residual_fit([-1e308, -1e308, 1e308, 1e308])
        self.assertTrue(result["available"])
        self.assertEqual(result["objective_mae_dn"], 1e308)

    def test_subnormal_midpoint_is_correctly_rounded(self):
        tiny = np.nextafter(0., 1.)
        result = residual_fit([tiny, tiny, 2*tiny, 2*tiny])
        self.assertTrue(result["available"])
        # Midpoint1.5*tiny ties to the even representable2*tiny, not tiny.
        self.assertEqual(result["offset_dn"], 2*tiny)
        self.assertEqual(result["objective_mae_dn"], 0)
        self.assertEqual(result["residual_order_counts"], dict(below=2, above=0, tied=2))
        np.testing.assert_array_equal(result["subgradient_interval"], [0, 1])

    def test_subnormal_objective_does_not_lose_terms_through_early_division(self):
        tiny = np.nextafter(0., 1.)
        result = residual_fit([-tiny, -tiny, tiny, tiny])
        self.assertTrue(result["available"])
        self.assertEqual(result["offset_dn"], 0)
        self.assertEqual(result["objective_mae_dn"], tiny)

    def test_finite_input_subtraction_overflow_invalidates_entire_fit(self):
        xy, slow, current = fixture()
        slow[0], current[0] = -np.finfo(float).max, np.finfo(float).max
        result = fit(xy, slow, current)
        self.assertFalse(result["available"])
        self.assertEqual(result["unavailable_reason"], "nonfinite_training_residual_arithmetic")
        self.assertEqual(result["training_used_count"], 144)
        self.assertTrue(result["training_used_mask"].all())
        self.assertEqual(result["training_unavailable_reasons"], (None,)*144)
        self.assertIsNone(result["offset_dn"])
        self.assertIsNone(result["candidate_offset_dn"])
        self.assertTrue(np.isnan(result["median_interval_dn"]).all())

    def test_deviation_overflow_keeps_diagnostic_candidate_not_prediction(self):
        maximum = np.finfo(float).max
        result = residual_fit([-maximum, maximum, maximum, maximum])
        self.assertFalse(result["available"])
        self.assertEqual(result["unavailable_reason"], "nonfinite_objective_arithmetic")
        self.assertIsNone(result["offset_dn"])
        self.assertIsNone(result["objective_mae_dn"])
        self.assertEqual(result["candidate_offset_dn"], maximum)
        self.assertEqual(result["residual_order_counts"], dict(below=1, above=0, tied=3))
        xy, slow, _ = fixture()
        pred = predict(result, xy, slow)
        self.assertFalse(pred["available"].any())
        self.assertTrue(np.isnan(pred["values"]).all())

    def test_prediction_overflow_is_explicit_per_index_unknown(self):
        xy, slow, _ = fixture()
        maximum = np.finfo(float).max
        result = fit(xy, np.zeros(len(xy)), np.full(len(xy), maximum))
        slow[:] = 0
        slow[7] = maximum
        pred = predict(result, xy, slow)
        self.assertEqual(np.flatnonzero(~pred["available"]).tolist(), [7])
        self.assertEqual(pred["unavailable_reasons"][7], "nonfinite_prediction_arithmetic")
        self.assertTrue(np.isnan(pred["values"][7]))

    def test_missing_train_rows_explicit_not_imputed(self):
        xy, slow, current = fixture()
        slow[3] = np.nan
        current[8] = np.inf
        result = fit(xy, slow, current)
        self.assertTrue(result["available"])
        self.assertEqual(result["training_count"], 144)
        self.assertEqual(result["training_used_count"], 142)
        self.assertEqual(np.flatnonzero(~result["training_used_mask"]).tolist(), [3, 8])
        self.assertEqual(result["training_unavailable_reasons"][3], "nonfinite_training_slow_or_current")
        self.assertEqual(result["offset_dn"], 8)

    def test_minimum_four_finite_train_rows(self):
        xy, slow, current = fixture()
        for count in (0, 1, 2, 3):
            result = fit(xy[:count], slow[:count], current[:count])
            self.assertFalse(result["available"])
            self.assertEqual(result["unavailable_reason"], "insufficient_finite_training_rows")
            self.assertEqual(result["training_used_mask"].shape, (count,))
            self.assertIsNone(result["residual_order_counts"])
        self.assertTrue(fit(xy[:4], slow[:4], current[:4])["available"])

    def test_flat_and_planar_baselines_do_not_require_identifiable_gain(self):
        xy, _, _ = fixture()
        for slow in (np.full(len(xy), 100.), 100+.1*xy[:, 0]+.2*xy[:, 1]):
            result = fit(xy, slow, slow+5)
            self.assertTrue(result["available"])
            self.assertEqual(result["offset_dn"], 5)

    def test_few_outliers_do_not_move_median_or_get_silently_dropped(self):
        xy, slow, current = fixture()
        current[:3] += 1000
        result = fit(xy, slow, current)
        self.assertEqual(result["training_used_count"], 144)
        self.assertEqual(result["offset_dn"], 8)
        self.assertAlmostEqual(result["objective_mae_dn"], 3000/144)

    def test_majority_contamination_is_not_claimed_to_be_safe(self):
        xy, slow, current = fixture()
        current[:80] += 100
        result = fit(xy, slow, current)
        self.assertAlmostEqual(result["offset_dn"], 108, places=12)
        self.assertTrue(result["available"])

    def test_optimality_against_nearby_offsets_for_generated_residuals(self):
        rng = np.random.default_rng(4253)
        for count in (4, 5, 8, 17, 144):
            residuals = rng.integers(-20, 21, count).astype(float)
            result = residual_fit(residuals)
            self.assertTrue(result["available"])
            for candidate in np.arange(-22, 23, .5):
                objective = float(np.mean(np.abs(residuals-candidate)))
                self.assertGreaterEqual(objective+1e-13, result["objective_mae_dn"])

    def test_result_independent_of_training_row_order(self):
        xy, slow, current = fixture()
        current += np.arange(len(current)) % 7
        order = np.random.default_rng(42).permutation(len(xy))
        first, second = fit(xy, slow, current), fit(xy[order], slow[order], current[order])
        for name in ("offset_dn", "objective_mae_dn", "residual_order_counts"):
            self.assertEqual(first[name], second[name])
        np.testing.assert_array_equal(first["median_interval_dn"], second["median_interval_dn"])
        self.assertNotEqual(first["training_input_sha256"], second["training_input_sha256"])

    def test_missing_heldout_slow_retains_index(self):
        xy, slow, current = fixture()
        result = fit(xy, slow, current)
        slow[[2, 13]] = [np.nan, np.inf]
        pred = predict(result, xy, slow)
        self.assertEqual(np.flatnonzero(~pred["available"]).tolist(), [2, 13])
        self.assertEqual(pred["unavailable_reasons"][2], "nonfinite_prediction_slow")

    def test_both_crossfit_splits_exact_and_point_preserving(self):
        xy, slow, current = fixture()
        for split in module.SPLITS:
            result = crossfit(xy, slow, current, split)
            self.assertEqual(set(result["fits"]), {"median_offset"})
            self.assertEqual(set(result["fits"]["median_offset"]), {"0", "1"})
            np.testing.assert_array_equal(result["predictions"]["median_offset"]["values"], current)
            self.assertEqual(result["total_count"], 144)
            self.assertFalse(result["metadata"]["scoring_or_forecast_selection_performed"])
            self.assertFalse(result["metadata"]["guard_purity_or_guard_to_core_transfer_certified"])

    def test_heldout_response_change_cannot_change_its_own_forecast(self):
        xy, slow, current = fixture()
        for split in module.SPLITS:
            for fold in (0, 1):
                first = crossfit(xy, slow, current, split)
                mask = first["fold_id"] == fold
                changed = current.copy()
                changed[mask] += 300
                second = crossfit(xy, slow, changed, split)
                self.assertEqual(first["fits"]["median_offset"][str(fold)]["model_sha256"],
                                 second["fits"]["median_offset"][str(fold)]["model_sha256"])
                a, b = (r["predictions"]["median_offset"]["values"] for r in (first, second))
                np.testing.assert_array_equal(a[mask], b[mask])
                np.testing.assert_allclose(b[~mask]-a[~mask], 300, atol=1e-12)

    def test_missing_heldout_current_does_not_suppress_prediction(self):
        xy, slow, current = fixture()
        first = crossfit(xy, slow, current)
        current[0] = np.nan
        second = crossfit(xy, slow, current)
        a, b = (r["predictions"]["median_offset"] for r in (first, second))
        self.assertTrue(b["available"][0])
        self.assertEqual(a["values"][0], b["values"][0])

    def test_no_crossfit_fallback_when_one_fold_has_no_train_samples(self):
        xy, slow, current = fixture()
        keep = xy[:, 0] < 64
        result = crossfit(xy[keep], slow[keep], current[keep])
        pred = result["predictions"]["median_offset"]
        self.assertFalse(pred["available"].any())
        self.assertTrue(all(reason == "fit_unavailable:insufficient_finite_training_rows"
                            for reason in pred["unavailable_reasons"]))

    def test_empty_crossfit_shapes(self):
        result = crossfit(np.empty((0, 2)), np.empty(0), np.empty(0))
        self.assertEqual(result["total_count"], 0)
        self.assertEqual(result["fold_id"].shape, (0,))
        self.assertEqual(result["predictions"]["median_offset"]["values"].shape, (0,))

    def test_inputs_not_mutated_outputs_readonly(self):
        xy, slow, current = fixture()
        originals = [a.copy() for a in (xy, slow, current)]
        result = crossfit(xy, slow, current)
        for before, after in zip(originals, (xy, slow, current)):
            np.testing.assert_array_equal(before, after)
        arrays = [result["fold_id"], result["predictions"]["median_offset"]["values"],
                  result["predictions"]["median_offset"]["available"]]
        for fitted in result["fits"]["median_offset"].values():
            arrays.extend([fitted["training_used_mask"], fitted["median_interval_dn"], fitted["subgradient_interval"]])
        for array in arrays:
            self.assertFalse(array.flags.writeable)

    def test_model_fingerprint_and_tampering(self):
        xy, slow, current = fixture()
        result = fit(xy, slow, current)
        self.assertEqual(result["model_sha256"], model_fingerprint(result))
        changed = copy.deepcopy(result)
        changed["offset_dn"] = 100
        with self.assertRaises(ValueError):
            predict(changed, xy, slow)

    def test_crossfit_hash_survives_json_roundtrip_and_detects_mutation(self):
        xy, slow, current = fixture()
        current[0] = np.nan
        result = crossfit(xy, slow, current)
        restored = json.loads(json.dumps(module._plain(result), allow_nan=False))
        self.assertEqual(result["crossfit_sha256"], crossfit_fingerprint(restored))
        restored["split"] = "checkerboard"
        self.assertNotEqual(result["crossfit_sha256"], crossfit_fingerprint(restored))

    def test_midpoint_failure_is_explicit_not_promoted(self):
        with patch.object(module, "_midpoint", return_value=float("inf")):
            result = residual_fit([0, 2, 4, 6])
        self.assertFalse(result["available"])
        self.assertEqual(result["unavailable_reason"], "nonfinite_or_outside_median_interval")
        self.assertIsNone(result["offset_dn"])

    def test_midpoint_exact_rational_definition_for_boundary_examples(self):
        maximum = np.finfo(float).max
        tiny = np.nextafter(0., 1.)
        for lower, upper in ((maximum, maximum), (-maximum, maximum), (-maximum, -maximum),
                             (tiny, 2*tiny), (-2*tiny, -tiny), (1., np.nextafter(1., np.inf))):
            expected = float((Fraction.from_float(float(lower))+Fraction.from_float(float(upper)))/2)
            self.assertEqual(module._midpoint(lower, upper), expected)


if __name__ == "__main__":
    unittest.main()
