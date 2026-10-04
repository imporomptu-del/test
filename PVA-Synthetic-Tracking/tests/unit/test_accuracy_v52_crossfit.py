import copy
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v52_crossfit as module
from accuracy_v52_crossfit import (
    crossfit, crossfit_fingerprint, fit, model_constants, model_fingerprint,
    predict, split_ids,
)


def fixture():
    xy = np.array([(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                   if 40 <= max(abs(x-64), abs(y-64)) <= 56], dtype=float)
    slow = 96 + 7*np.sin(xy[:, 0]/13) + 5*np.cos(xy[:, 1]/17)
    current = 1.3*slow+12+3*(xy[:, 0]-64)/56-2*(xy[:, 1]-64)/56
    return xy, slow, current


class CrossfitModelTests(unittest.TestCase):
    def test_frozen_constants(self):
        constants = model_constants()
        self.assertEqual(constants["huber_delta_dn"], 2)
        self.assertEqual(constants["max_irls_updates"], 100)
        self.assertEqual(constants["kkt_tolerance_dn"], 1e-7)
        self.assertEqual(constants["rank_rtol"], 1e-12)
        self.assertEqual(constants["condition_limit"], 1e8)
        self.assertFalse(constants["implicit_ridge_or_fallback"])

    def test_predict_has_no_current_response_label_or_selection_argument(self):
        self.assertEqual(tuple(inspect.signature(predict).parameters), ("model", "test_xy", "slow"))
        xy, slow, current = fixture()
        fitted = fit(xy, slow, current)
        for key in ("current", "response", "labels", "threshold"):
            with self.assertRaises(TypeError):
                predict(fitted, xy, slow, **{key: current})

    def test_geometry_accepts_only_unique_inherited_guard_points(self):
        _, slow, current = fixture()
        for xy in (np.array([[64, 64]]), np.array([[32, 32]]),
                   np.array([[0, 8]]), np.array([[128, 8]]), np.array([[9, 8]]),
                   np.array([[np.nan, 8]]), np.array([[8, np.inf]]),
                   np.array([[8, 8], [8, 8]]), np.ones((2, 2), dtype=complex),
                   np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=object),
                   np.ones((2,)), np.ones((2, 3))):
            with self.subTest(xy=xy):
                with self.assertRaises(ValueError):
                    fit(xy, slow[:len(xy)], current[:len(xy)])

    def test_invalid_value_shapes_and_types(self):
        xy, slow, current = fixture()
        for value in (slow[:-1], slow[:, None], slow.astype(complex),
                      slow.astype(bool), slow.astype(str), slow.astype(object)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    fit(xy, value, current)
                with self.assertRaises(ValueError):
                    fit(xy, slow, value)

    def test_invalid_loss_and_split_rejected(self):
        xy, slow, current = fixture()
        with self.assertRaises(ValueError):
            fit(xy, slow, current, "adaptive")
        with self.assertRaises(ValueError):
            crossfit(xy, slow, current, "tuned")

    def test_folds_match_exact_formulas(self):
        xy, _, _ = fixture()
        self.assertEqual(len(xy), 144)
        np.testing.assert_array_equal(split_ids(xy), xy[:, 0] >= 64)
        np.testing.assert_array_equal(split_ids(xy, "checkerboard"),
            ((xy[:, 0]/8+xy[:, 1]/8) % 2).astype(np.int8))
        self.assertEqual(np.bincount(split_ids(xy)).tolist(), [69, 75])
        self.assertEqual(np.bincount(split_ids(xy, "checkerboard")).tolist(), [72, 72])

    def test_exact_gain_plus_plane_recovered_by_both_losses(self):
        xy, slow, current = fixture()
        for loss in ("ols", "huber"):
            with self.subTest(loss=loss):
                result = fit(xy, slow, current, loss)
                self.assertTrue(result["available"])
                self.assertTrue(result["converged"])
                self.assertEqual(result["rank"], 4)
                self.assertLessEqual(result["kkt_residual_dn"], 1e-7)
                np.testing.assert_allclose(result["coefficients_physical"], [1.3, 12, 3, -2], atol=1e-11)
                forecast = predict(result, xy, slow)
                self.assertTrue(forecast["available"].all())
                np.testing.assert_allclose(forecast["values"], current, atol=1e-11)
                self.assertEqual(forecast["unavailable_reasons"], (None,)*len(xy))

    def test_training_only_median_and_mad_scaling(self):
        xy, slow, current = fixture()
        mask = xy[:, 0] < 64
        result = fit(xy[mask], slow[mask], current[mask])
        center = np.median(slow[mask])
        self.assertEqual(result["center"], center)
        self.assertEqual(result["scale"], max(1, np.median(np.abs(slow[mask]-center))))
        self.assertNotEqual(result["center"], np.median(slow))

    def test_scale_floor_is_one_without_forcing_identity_gain(self):
        xy, slow, current = fixture()
        slow = 100 + .001*(slow-96)
        current = .75*slow+3
        result = fit(xy, slow, current)
        self.assertTrue(result["available"])
        self.assertEqual(result["scale"], 1)
        self.assertAlmostEqual(result["coefficients_physical"][0], .75, places=9)

    def test_exact_crossfit_all_indices_on_both_geometric_splits(self):
        xy, slow, current = fixture()
        for split in ("left_right", "checkerboard"):
            result = crossfit(xy, slow, current, split)
            self.assertEqual(result["total_count"], 144)
            for loss in ("ols", "huber"):
                self.assertEqual(result["predictions"][loss]["values"].shape, (144,))
                self.assertTrue(result["predictions"][loss]["available"].all())
                np.testing.assert_allclose(result["predictions"][loss]["values"], current, atol=1e-11)
            self.assertFalse(result["metadata"]["scoring_or_forecast_selection_performed"])
            self.assertFalse(result["metadata"]["guard_purity_or_guard_to_core_transfer_certified"])

    def test_heldout_response_changes_cannot_change_own_fold_forecast(self):
        xy, slow, current = fixture()
        for split in ("left_right", "checkerboard"):
            for heldout_id in (0, 1):
                with self.subTest(split=split, heldout_id=heldout_id):
                    before = crossfit(xy, slow, current, split)
                    heldout = before["fold_id"] == heldout_id
                    changed = current.copy()
                    changed[heldout] += 300
                    after = crossfit(xy, slow, changed, split)
                    for loss in ("ols", "huber"):
                        self.assertEqual(before["fits"][loss][str(heldout_id)]["model_sha256"],
                                         after["fits"][loss][str(heldout_id)]["model_sha256"])
                        np.testing.assert_array_equal(before["predictions"][loss]["values"][heldout],
                                                      after["predictions"][loss]["values"][heldout])
                        self.assertGreater(np.max(np.abs(before["predictions"][loss]["values"][~heldout]
                                                          -after["predictions"][loss]["values"][~heldout])), 250)

    def test_heldout_slow_does_not_change_own_fit_scaling(self):
        xy, slow, current = fixture()
        before = crossfit(xy, slow, current)
        heldout = before["fold_id"] == 1
        changed = slow.copy()
        changed[heldout] += 1000
        after = crossfit(xy, changed, current)
        for loss in ("ols", "huber"):
            self.assertEqual(before["fits"][loss]["1"]["model_sha256"], after["fits"][loss]["1"]["model_sha256"])
            self.assertGreater(np.max(after["predictions"][loss]["values"][heldout]), 1000)

    def test_missing_training_rows_explicit_and_not_imputed(self):
        xy, slow, current = fixture()
        slow[3] = np.nan
        current[8] = np.inf
        result = fit(xy, slow, current)
        self.assertTrue(result["available"])
        self.assertEqual(result["training_count"], 144)
        self.assertEqual(result["training_used_count"], 142)
        self.assertEqual(np.flatnonzero(~result["training_used_mask"]).tolist(), [3, 8])
        self.assertEqual(result["training_unavailable_reasons"][3], "nonfinite_training_slow_or_current")
        self.assertEqual(result["training_unavailable_reasons"][8], "nonfinite_training_slow_or_current")
        np.testing.assert_allclose(result["coefficients_physical"], [1.3, 12, 3, -2], atol=1e-10)

    def test_missing_heldout_current_does_not_suppress_prediction(self):
        xy, slow, current = fixture()
        before = crossfit(xy, slow, current)
        current[0] = np.nan
        after = crossfit(xy, slow, current)
        for loss in ("ols", "huber"):
            self.assertTrue(after["predictions"][loss]["available"][0])
            self.assertEqual(before["predictions"][loss]["values"][0], after["predictions"][loss]["values"][0])

    def test_missing_prediction_slow_retains_exact_index(self):
        xy, slow, current = fixture()
        fitted = fit(xy, slow, current)
        slow[[2, 13]] = [np.nan, np.inf]
        result = predict(fitted, xy, slow)
        self.assertEqual(np.flatnonzero(~result["available"]).tolist(), [2, 13])
        self.assertTrue(np.isnan(result["values"][[2, 13]]).all())
        self.assertEqual(result["unavailable_reasons"][2], "nonfinite_prediction_slow")

    def test_insufficient_rows_and_empty_preserved(self):
        xy, slow, current = fixture()
        for count in (0, 1, 2, 3):
            result = fit(xy[:count], slow[:count], current[:count])
            self.assertFalse(result["available"])
            self.assertEqual(result["unavailable_reason"], "insufficient_finite_training_rows")
            self.assertEqual(result["training_used_mask"].shape, (count,))
        empty = crossfit(np.empty((0, 2)), np.empty(0), np.empty(0))
        self.assertEqual(empty["total_count"], 0)
        for loss in ("ols", "huber"):
            self.assertEqual(empty["predictions"][loss]["values"].shape, (0,))

    def test_flat_and_planar_slow_fail_conservatively(self):
        xy, slow, current = fixture()
        for s in (np.full(len(xy), 100.), 100+.1*xy[:, 0]+.2*xy[:, 1]):
            for loss in ("ols", "huber"):
                result = fit(xy, s, current, loss)
                self.assertFalse(result["available"])
                self.assertEqual(result["unavailable_reason"], "rank_deficient_design")
                self.assertTrue(np.isnan(result["coefficients_physical"]).all())
                forecast = predict(result, xy, s)
                self.assertFalse(forecast["available"].any())
                self.assertTrue(np.isnan(forecast["values"]).all())
                self.assertTrue(all(reason == "fit_unavailable:rank_deficient_design"
                                    for reason in forecast["unavailable_reasons"]))

    def test_ill_conditioned_full_rank_slow_rejected(self):
        xy, slow, current = fixture()
        slow = 100+.1*xy[:, 0]+.2*xy[:, 1]+1e-8*np.sin(xy[:, 0]/13)
        result = fit(xy, slow, current)
        self.assertFalse(result["available"])
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["unavailable_reason"], "ill_conditioned_design")
        self.assertGreater(result["condition_number"], 1e8)

    def test_negative_unconstrained_gain_obeys_boundary_kkt(self):
        xy, slow, current = fixture()
        current = 150-.7*slow+.2*(xy[:, 0]-64)/56
        for loss in ("ols", "huber"):
            result = fit(xy, slow, current, loss)
            self.assertTrue(result["available"], result["unavailable_reason"])
            self.assertEqual(result["coefficients_physical"][0], 0)
            self.assertTrue(result["gain_bound_active"])
            self.assertLessEqual(result["kkt_residual_dn"], 1e-7)
            design = module._design(xy, slow, result["center"], result["scale"])
            residual = design @ result["coefficients_conditioned"]-current
            psi = residual if loss == "ols" else np.clip(residual, -2, 2)
            gradient = design.T @ psi/len(slow)
            self.assertGreaterEqual(gradient[0], -1e-7)
            np.testing.assert_allclose(gradient[1:], 0, atol=1e-7)

    def test_kkt_bound_sign_not_merely_small_parameter_step(self):
        xy, slow, current = fixture()
        center, scale = np.median(slow), max(1, np.median(np.abs(slow-np.median(slow))))
        design = module._design(xy, slow, center, scale)
        # Gain-zero stationary plane is not optimal for a positive true gain.
        plane = np.linalg.lstsq(design[:, 1:], current, rcond=1e-12)[0]
        _, kkt, _, _ = module._optimality(design, current, np.r_[0., plane], "ols")
        self.assertGreater(kkt, 1)

    def test_huber_limits_large_outlier_influence_without_dropping_rows(self):
        xy, slow, clean = fixture()
        corrupted = clean.copy()
        corrupted[0] += 1000
        ols, huber = fit(xy, slow, corrupted, "ols"), fit(xy, slow, corrupted, "huber")
        self.assertTrue(ols["available"])
        self.assertTrue(huber["available"], huber["unavailable_reason"])
        self.assertEqual(huber["training_used_count"], 144)
        huber_error = np.mean(np.abs(predict(huber, xy, slow)["values"]-clean))
        ols_error = np.mean(np.abs(predict(ols, xy, slow)["values"]-clean))
        self.assertLess(huber_error, .1)
        self.assertGreater(ols_error, 5)
        self.assertLess(huber_error, ols_error/50)

    def test_objective_and_kkt_match_direct_independent_formulas(self):
        xy, slow, current = fixture()
        current = current + 3*np.sin(np.arange(len(current)))
        for loss in ("ols", "huber"):
            fitted = fit(xy, slow, current, loss)
            self.assertTrue(fitted["available"])
            prediction = predict(fitted, xy, slow)["values"]
            residual = prediction-current
            objective = (.5*residual**2 if loss == "ols" else
                         np.where(np.abs(residual) <= 2, .5*residual**2, 2*np.abs(residual)-2))
            self.assertAlmostEqual(fitted["objective"], np.mean(objective), places=12)
            design = np.column_stack(((slow-fitted["center"])/fitted["scale"],
                                      np.ones(len(xy)), (xy[:, 0]-64)/56, (xy[:, 1]-64)/56))
            psi = residual if loss == "ols" else np.clip(residual, -2, 2)
            gradient = design.T @ psi/len(xy)
            scaled = gradient/np.maximum(1, np.sqrt(np.mean(design**2, axis=0)))
            self.assertAlmostEqual(fitted["kkt_residual_dn"], np.max(np.abs(scaled)), places=12)

    def test_nonconvergence_is_explicit_unknown_not_last_iterate(self):
        xy, slow, current = fixture()
        current[0] += 1000
        with patch.object(module, "MAX_IRLS_UPDATES", 0):
            fitted = fit(xy, slow, current, "huber")
        self.assertFalse(fitted["available"])
        self.assertFalse(fitted["converged"])
        self.assertEqual(fitted["unavailable_reason"], "irls_nonconvergence")
        self.assertGreater(fitted["kkt_residual_dn"], 1e-7)
        self.assertTrue(np.isnan(fitted["coefficients_conditioned"]).all())
        self.assertTrue(np.isfinite(fitted["last_candidate_coefficients_conditioned"]).all())
        design = module._design(xy, slow, fitted["center"], fitted["scale"])
        _, kkt, _, reason = module._optimality(design, current,
            fitted["last_candidate_coefficients_conditioned"], "huber")
        self.assertIsNone(reason)
        self.assertEqual(kkt, fitted["kkt_residual_dn"])
        self.assertFalse(predict(fitted, xy, slow)["available"].any())

    def test_weighted_singular_modes_not_silently_truncated(self):
        xy, slow, current = fixture()
        fitted = fit(xy, slow, current)
        design = module._design(xy, slow, fitted["center"], fitted["scale"])
        weights = np.zeros(len(xy))
        weights[:3] = 1
        coefficient, reason = module._weighted_solve(design, current, weights)
        self.assertIsNone(coefficient)
        self.assertEqual(reason, "weighted_rank_deficient_design")

    def test_unavailable_irls_weighted_solve_never_promoted(self):
        xy, slow, current = fixture()
        current[0] += 1000
        original = module._weighted_solve
        calls = []
        def fake(design, response, weights):
            calls.append(1)
            if len(calls) == 1:
                return original(design, response, weights)
            return None, "weighted_ill_conditioned_design"
        with patch.object(module, "_weighted_solve", side_effect=fake):
            fitted = fit(xy, slow, current)
        self.assertFalse(fitted["available"])
        self.assertEqual(fitted["unavailable_reason"], "weighted_ill_conditioned_design")

    def test_failed_optimality_does_not_report_stale_previous_candidate_metrics(self):
        xy, slow, current = fixture()
        current[0] += 1000
        original = module._optimality
        calls = []
        def fake(design, response, coefficients, loss):
            calls.append(1)
            if len(calls) == 1:
                return original(design, response, coefficients, loss)
            return None, None, None, "nonfinite_optimality_arithmetic"
        with patch.object(module, "_optimality", side_effect=fake):
            fitted = fit(xy, slow, current)
        self.assertFalse(fitted["available"])
        self.assertEqual(fitted["iterations"], 1)
        self.assertIsNone(fitted["objective"])
        self.assertIsNone(fitted["kkt_residual_dn"])
        self.assertTrue(np.isfinite(fitted["last_candidate_coefficients_conditioned"]).all())

    def test_nonfinite_train_scaling_arithmetic_is_unavailable(self):
        xy, slow, current = fixture()
        slow[::2], slow[1::2] = 1.7e308, -1.7e308
        fitted = fit(xy, slow, current)
        self.assertFalse(fitted["available"])
        self.assertIn(fitted["unavailable_reason"], ("nonfinite_training_scaling", "nonfinite_design_arithmetic",
                                                   "nonfinite_optimality_arithmetic"))

    def test_prediction_overflow_is_per_index_unknown(self):
        xy, slow, current = fixture()
        current = 10*slow
        fitted = fit(xy, slow, current)
        self.assertTrue(fitted["available"])
        slow[4] = 1e308
        predicted = predict(fitted, xy, slow)
        self.assertEqual(np.flatnonzero(~predicted["available"]).tolist(), [4])
        self.assertEqual(predicted["unavailable_reasons"][4], "nonfinite_prediction_arithmetic")

    def test_inputs_not_mutated_and_outputs_readonly(self):
        xy, slow, current = fixture()
        originals = [a.copy() for a in (xy, slow, current)]
        result = crossfit(xy, slow, current)
        for before, after in zip(originals, (xy, slow, current)):
            np.testing.assert_array_equal(before, after)
        arrays = [result["fold_id"]]
        for loss in ("ols", "huber"):
            arrays.extend((result["predictions"][loss]["values"], result["predictions"][loss]["available"]))
            for fitted in result["fits"][loss].values():
                arrays.extend((fitted["training_used_mask"], fitted["coefficients_conditioned"],
                               fitted["last_candidate_coefficients_conditioned"], fitted["coefficients_physical"]))
        for array in arrays:
            self.assertFalse(array.flags.writeable)

    def test_fingerprints_and_model_tampering(self):
        xy, slow, current = fixture()
        fitted = fit(xy, slow, current)
        self.assertEqual(model_fingerprint(fitted), fitted["model_sha256"])
        changed = copy.deepcopy(fitted)
        changed["scale"] = 999
        with self.assertRaises(ValueError):
            predict(changed, xy, slow)
        result = crossfit(xy, slow, current)
        self.assertEqual(crossfit_fingerprint(result), result["crossfit_sha256"])
        result["split"] = "checkerboard"
        self.assertNotEqual(crossfit_fingerprint(result), result["crossfit_sha256"])

    def test_no_support_fallback_when_one_fold_has_no_train_rows(self):
        xy, slow, current = fixture()
        keep = xy[:, 0] < 64
        result = crossfit(xy[keep], slow[keep], current[keep])
        for loss in ("ols", "huber"):
            self.assertFalse(result["predictions"][loss]["available"].any())
            self.assertTrue(all(reason == "fit_unavailable:insufficient_finite_training_rows"
                                for reason in result["predictions"][loss]["unavailable_reasons"]))


if __name__ == "__main__":
    unittest.main()
