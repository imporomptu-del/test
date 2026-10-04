from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v47_guard_gain import estimate_guard_gain
from accuracy_v50_predictive_background import (
    ARMS, forecast, forecast_fingerprint, measure_current,
)


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False,
                                default=lambda array: array.tolist()))


class PredictiveBackgroundTests(unittest.TestCase):
    def fixture(self, values=None):
        if values is None:
            values = [10.]*8
        history = np.stack([np.full((129, 129), value, dtype=float) for value in values])
        return history, [[64., 64.]]*8

    def oracle_support(self, history, centers):
        # Only prior support is compared. The V47 gain availability/result is
        # deliberately irrelevant to the new brightness prediction experiment.
        return estimate_guard_gain(np.zeros((129, 129)), history,
            np.zeros((129, 129)), np.ones((129, 129)), centers)

    def assert_support_equal(self, result, expected):
        for name in ("candidate_count", "eligible_count", "used_count", "stencil_count",
                     "candidate_support_sha256", "eligible_support_sha256",
                     "used_support_sha256", "stencil_sha256", "used_points_xy", "stencils"):
            self.assertEqual(result[name], expected[name], name)

    def test_forecast_has_no_current_parameter_or_inner_time_option(self):
        self.assertEqual(tuple(inspect.signature(forecast).parameters),
                         ("history129", "prior_centers_xy"))
        with self.assertRaises(TypeError):
            forecast(*self.fixture(), current129=np.zeros((129, 129)))

    def test_constant_scene_zero_error_all_three_arms(self):
        prediction = forecast(*self.fixture())
        self.assertTrue(prediction["available"])
        self.assertEqual(tuple(prediction["arms"]), ARMS)
        self.assertEqual(prediction["candidate_count"], 144)
        measurement = measure_current(np.full((129, 129), 10.), prediction)
        self.assertTrue(measurement["available"])
        for name in ARMS:
            np.testing.assert_array_equal(prediction["arms"][name]["prediction"], 10.)
            np.testing.assert_array_equal(prediction["arms"][name]["scale"], 1.)
            self.assertEqual(measurement["arms"][name]["max_score"], 0.)
            self.assertEqual(measurement["arms"][name]["mae_dn"], 0.)
            self.assertEqual(measurement["arms"][name]["rmse_dn"], 0.)
        self.assertEqual(measurement["forecast_sha256"], prediction["forecast_sha256"])

    def test_linear_trend_has_explicit_lag_and_shared_mad(self):
        prediction = forecast(*self.fixture(list(range(8))))
        current = measure_current(np.full((129, 129), 8.), prediction)
        for name, center, scale, score in (
            (ARMS[0], 3.5, 1., 4.5), (ARMS[1], 3.5, 2., 2.25),
            (ARMS[2], 6., 2., 1.),
        ):
            np.testing.assert_array_equal(prediction["arms"][name]["prediction"], center)
            np.testing.assert_array_equal(prediction["arms"][name]["scale"], scale)
            self.assertEqual(current["arms"][name]["max_score"], score)

    def test_last_three_step_changes_center_not_shared_scale(self):
        prediction = forecast(*self.fixture([10.]*5+[20.]*3))
        for name in ARMS[:2]:
            np.testing.assert_array_equal(prediction["arms"][name]["prediction"], 10.)
        np.testing.assert_array_equal(prediction["arms"][ARMS[2]]["prediction"], 20.)
        np.testing.assert_array_equal(prediction["arms"][ARMS[1]]["scale"], 1.)
        np.testing.assert_array_equal(prediction["arms"][ARMS[2]]["scale"], 1.)
        measurement = measure_current(np.full((129, 129), 20.), prediction)
        self.assertEqual(measurement["arms"][ARMS[0]]["max_score"], 10.)
        self.assertEqual(measurement["arms"][ARMS[2]]["max_score"], 0.)

    def test_one_outlier_is_not_learned_as_noise_coverage_guarantee(self):
        prediction = forecast(*self.fixture([10.]*7+[200.]))
        for arm in prediction["arms"].values():
            np.testing.assert_array_equal(arm["prediction"], 10.)
            np.testing.assert_array_equal(arm["scale"], 1.)
        measurement = measure_current(np.full((129, 129), 200.), prediction)
        self.assertEqual(measurement["arms"][ARMS[2]]["max_score"], 190.)
        self.assertFalse(prediction["metadata"]["interval_coverage_claimed"])

    def test_two_recent_outliers_show_declared_short_window_behavior(self):
        prediction = forecast(*self.fixture([10.]*6+[200.]*2))
        np.testing.assert_array_equal(prediction["arms"][ARMS[0]]["prediction"], 10.)
        np.testing.assert_array_equal(prediction["arms"][ARMS[2]]["prediction"], 200.)
        np.testing.assert_array_equal(prediction["arms"][ARMS[2]]["scale"], 1.)

    def test_normalization_floor_is_exactly_one_not_gaussian_mad_factor(self):
        prediction = forecast(*self.fixture([0., 0., 0., 0., 8., 8., 8., 8.]))
        np.testing.assert_array_equal(prediction["arms"][ARMS[1]]["scale"], 4.)
        small = forecast(*self.fixture([0.]*4+[.2]*4))
        np.testing.assert_array_equal(small["arms"][ARMS[1]]["scale"], 1.)
        self.assertTrue(small["metadata"]["scale_floor_is_normalization_not_physical_error_bound"])

    def test_both_adaptive_arms_have_identical_scales_on_random_values(self):
        history = np.random.default_rng(5001).normal(80, 20, (8, 129, 129))
        prediction = forecast(history, [None]*8)
        np.testing.assert_array_equal(prediction["arms"][ARMS[1]]["scale"],
                                      prediction["arms"][ARMS[2]]["scale"])
        np.testing.assert_array_equal(prediction["arms"][ARMS[0]]["prediction"],
                                      prediction["arms"][ARMS[1]]["prediction"])

    def test_canonical_support_matches_v47_without_exclusions(self):
        history, centers = self.fixture()
        prediction = forecast(history, centers)
        self.assert_support_equal(prediction, self.oracle_support(history, centers))
        self.assertEqual(prediction["used_points_xy"],
                         sorted(prediction["used_points_xy"], key=lambda p: (p[1], p[0])))

    def test_canonical_support_matches_v47_with_exclusions_and_nan(self):
        history, centers = self.fixture()
        centers = [[16., 64.], [112., 64.], [64., 16.], None, None, None, None, None]
        history[3, 8, 8] = np.nan
        history[1, 120, 120] = np.inf
        history[7, 8, 16] = -np.inf
        prediction = forecast(history, centers)
        self.assert_support_equal(prediction, self.oracle_support(history, centers))
        self.assertEqual(prediction["missing_prior_center_indices"], [3, 4, 5, 6, 7])
        for x, y in prediction["used_points_xy"]:
            self.assertTrue(all(point is None or max(abs(x-point[0]), abs(y-point[1])) > 12
                                for point in centers))

    def test_radius_twelve_boundary_is_excluded_not_rounded(self):
        history, _ = self.fixture()
        for offset, excluded in ((0., True), (.01, False)):
            centers = [[20.+offset, 8.]]+[None]*7
            prediction = forecast(history, centers)
            self.assertEqual([8, 8] not in prediction["used_points_xy"], excluded)
            self.assert_support_equal(prediction, self.oracle_support(history, centers))

    def test_prior_core_poison_never_changes_support_prediction_or_hash(self):
        history, centers = self.fixture(list(range(8)))
        original = plain(forecast(history, centers))
        for poison in (np.nan, np.inf, -np.inf, 1e300):
            changed = history.copy()
            changed[:, 32:97, 32:97] = poison
            self.assertEqual(plain(forecast(changed, centers)), original)

    def test_prior_non_candidate_pixel_poison_is_ignored(self):
        history, centers = self.fixture()
        original = plain(forecast(history, centers))
        history[:, 9, 9] = np.nan
        history[:, 0, :] = np.inf
        history[:, :, 128] = -np.inf
        self.assertEqual(plain(forecast(history, centers)), original)

    def test_finite_values_of_eligible_unused_prior_point_are_not_predicted(self):
        history = np.full((8, 129, 129), np.nan)
        for x, y in ((8, 8), (16, 8), (24, 8), (120, 120)):
            history[:, y, x] = 10.
        original = forecast(history, [None]*8)
        self.assertEqual(original["eligible_count"], 4)
        self.assertEqual(original["used_count"], 3)
        self.assertEqual(original["stencil_count"], 1)
        history[:, 120, 120] = np.arange(8)*100.
        self.assertEqual(plain(forecast(history, [None]*8)), plain(original))

    def test_any_nonfinite_prior_candidate_is_removed_before_complete_triplets(self):
        history, centers = self.fixture()
        history[0, 8, 8] = np.nan
        prediction = forecast(history, centers)
        self.assertNotIn([8, 8], prediction["used_points_xy"])
        self.assertEqual(prediction["eligible_count"], 143)
        self.assertEqual(prediction["prior_rejection_counts_nonexclusive"]["nonfinite_prior_history"], 1)
        self.assert_support_equal(prediction, self.oracle_support(history, centers))

    def test_border_missing_support_is_explicit_and_not_filled(self):
        history, centers = self.fixture()
        history[:, 110:, :] = np.nan
        prediction = forecast(history, centers)
        self.assertTrue(prediction["available"])
        self.assertTrue(all(y < 110 for x, y in prediction["used_points_xy"]))
        self.assert_support_equal(prediction, self.oracle_support(history, centers))

    def test_empty_history_support_is_unavailable_not_a_zero_score(self):
        prediction = forecast(np.full((8, 129, 129), np.nan), [None]*8)
        self.assertFalse(prediction["available"])
        self.assertEqual(prediction["reasons"], ["no_prior_selected_guard_contrasts"])
        self.assertEqual(prediction["arms"], {})
        measurement = measure_current(np.full((129, 129), np.inf), prediction)
        self.assertFalse(measurement["available"])
        self.assertEqual(measurement["reasons"], ["prior_forecast_unavailable"])
        self.assertIsNone(measurement["current_nonfinite_used_point_count"])
        self.assertEqual(measurement["arms"], {})

    def test_isolated_eligible_points_without_complete_triplet_are_unavailable(self):
        history = np.full((8, 129, 129), np.nan)
        history[:, 8, 8] = 20.
        prediction = forecast(history, [None]*8)
        self.assertEqual(prediction["eligible_count"], 1)
        self.assertEqual(prediction["used_count"], 0)
        self.assertFalse(prediction["available"])

    def test_current_core_poison_never_changes_measurements_or_forecast(self):
        prediction = forecast(*self.fixture())
        current = np.full((129, 129), 12.)
        baseline = plain(measure_current(current, prediction))
        before = prediction["forecast_sha256"]
        for poison in (np.nan, np.inf, -np.inf, 1e300):
            changed = current.copy()
            changed[32:97, 32:97] = poison
            self.assertEqual(plain(measure_current(changed, prediction)), baseline)
        self.assertEqual(forecast_fingerprint(prediction), before)

    def test_current_eligible_but_unused_point_is_not_gathered(self):
        history = np.full((8, 129, 129), np.nan)
        for x, y in ((8, 8), (16, 8), (24, 8), (120, 120)):
            history[:, y, x] = 10.
        prediction = forecast(history, [None]*8)
        current = np.full((129, 129), 10.)
        baseline = plain(measure_current(current, prediction))
        current[120, 120] = np.nan
        current[9, 9] = np.inf
        self.assertEqual(plain(measure_current(current, prediction)), baseline)

    def test_each_current_nonfinite_kind_invalidates_whole_packet_without_deletion(self):
        prediction = forecast(*self.fixture())
        before = plain(prediction)
        for poison in (np.nan, np.inf, -np.inf):
            current = np.full((129, 129), 10.)
            for x, y in prediction["used_points_xy"][:2]:
                current[y, x] = poison
            measurement = measure_current(current, prediction)
            self.assertFalse(measurement["available"])
            self.assertEqual(measurement["current_nonfinite_used_point_count"], 2)
            self.assertEqual(measurement["reasons"], ["nonfinite_current_on_fixed_used_guard_support"])
            self.assertEqual(measurement["used_count"], prediction["used_count"])
            self.assertEqual(measurement["arms"], {})
        self.assertEqual(plain(prediction), before)

    def test_current_changes_measurement_but_never_refits_forecast(self):
        prediction = forecast(*self.fixture())
        before = plain(prediction)
        one = measure_current(np.full((129, 129), 11.), prediction)
        two = measure_current(np.full((129, 129), 15.), prediction)
        self.assertEqual(one["arms"][ARMS[0]]["max_score"], 1.)
        self.assertEqual(two["arms"][ARMS[0]]["max_score"], 5.)
        self.assertEqual(one["forecast_sha256"], two["forecast_sha256"])
        self.assertEqual(plain(prediction), before)

    def test_current_residual_sign_and_direct_reduction_oracle(self):
        prediction = forecast(*self.fixture(list(range(8))))
        current = np.arange(129*129, dtype=float).reshape(129, 129)/1000
        measurement = measure_current(current, prediction)
        observed = np.array([current[y, x] for x, y in prediction["used_points_xy"]])
        for name in ARMS:
            residual = observed-prediction["arms"][name]["prediction"]
            normalized = np.abs(residual)/prediction["arms"][name]["scale"]
            value = measurement["arms"][name]
            np.testing.assert_array_equal(value["residuals"], residual)
            np.testing.assert_array_equal(value["normalized_absolute_errors"], normalized)
            self.assertEqual(value["max_score"], float(max(normalized)))
            self.assertAlmostEqual(value["mae_dn"], float(np.mean(np.abs(residual))))
            self.assertAlmostEqual(value["rmse_dn"], float(np.sqrt(np.mean(residual**2))))

    def test_zero_and_255_are_ordinary_finite_values_not_remasked(self):
        for value in (0., 255.):
            prediction = forecast(*self.fixture([value]*8))
            self.assertTrue(prediction["available"])
            self.assertEqual(prediction["eligible_count"], 144)
            measurement = measure_current(np.full((129, 129), value), prediction)
            self.assertTrue(measurement["available"])
            self.assertEqual(measurement["arms"][ARMS[0]]["max_score"], 0.)

    def test_all_inputs_remain_unchanged_and_arrays_do_not_alias_them(self):
        history, centers = self.fixture(list(range(8)))
        current = np.full((129, 129), 6.)
        h_before, c_before, y_before = history.copy(), deepcopy(centers), current.copy()
        prediction = forecast(history, centers)
        measurement = measure_current(current, prediction)
        np.testing.assert_array_equal(history, h_before)
        np.testing.assert_array_equal(current, y_before)
        self.assertEqual(centers, c_before)
        for arm in prediction["arms"].values():
            for array in arm.values():
                self.assertFalse(array.flags.writeable)
                self.assertFalse(np.shares_memory(array, history))
        for arm in measurement["arms"].values():
            for name in ("residuals", "normalized_absolute_errors"):
                self.assertFalse(arm[name].flags.writeable)
                self.assertFalse(np.shares_memory(arm[name], current))

    def test_identical_inputs_have_deterministic_fingerprint(self):
        first = forecast(*self.fixture(list(range(8))))
        second = forecast(*self.fixture(list(range(8))))
        self.assertEqual(first["forecast_sha256"], second["forecast_sha256"])
        self.assertEqual(first["forecast_sha256"], forecast_fingerprint(first))
        plain(first)
        plain(measure_current(np.zeros((129, 129)), first))

    def test_forecast_tampering_fails_before_current_measurement(self):
        original = forecast(*self.fixture())
        for field, value in (("used_count", 0), ("used_points_xy", [[64, 64]]),
                             ("forecast_sha256", "0"*64)):
            changed = deepcopy(original)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "frozen hash"):
                measure_current(np.zeros((129, 129)), changed)
        changed = deepcopy(original)
        changed["arms"][ARMS[0]]["prediction"] = np.zeros(original["used_count"])
        with self.assertRaisesRegex(ValueError, "frozen hash"):
            measure_current(np.zeros((129, 129)), changed)

    def test_bad_history_shapes_and_types_are_rejected(self):
        for history in (np.zeros((7, 129, 129)), np.zeros((8, 128, 129)),
                        np.zeros((8, 129, 129), dtype=complex),
                        np.zeros((8, 129, 129), dtype=bool),
                        np.full((8, 129, 129), "1")):
            with self.subTest(shape=history.shape, dtype=str(history.dtype)):
                with self.assertRaises(ValueError):
                    forecast(history, [None]*8)

    def test_bad_current_shapes_and_types_are_rejected(self):
        prediction = forecast(*self.fixture())
        for current in (np.zeros((128, 129)), np.zeros((129, 129), dtype=complex),
                        np.zeros((129, 129), dtype=bool), np.full((129, 129), "1")):
            with self.assertRaises(ValueError):
                measure_current(current, prediction)

    def test_bad_centers_are_rejected_and_missing_is_explicit(self):
        history, _ = self.fixture()
        for centers in (None, [], [None]*7, [None]*9, [[np.nan, np.nan]]*8,
                        [[np.inf, 1.]]*8, [[1.]]*8, [[1., 2., 3.]]*8,
                        [[1.+0j, 2.]]*8, [[True, False]]*8):
            with self.assertRaises(ValueError):
                forecast(history, centers)
        prediction = forecast(history, [None]*8)
        self.assertEqual(prediction["missing_prior_center_indices"], list(range(8)))

    def test_extreme_forecast_arithmetic_is_explicitly_unavailable(self):
        prediction = forecast(*self.fixture([np.finfo(float).max]*8))
        self.assertFalse(prediction["available"])
        self.assertEqual(prediction["reasons"], ["nonfinite_prior_forecast_arithmetic"])
        self.assertEqual(prediction["arms"], {})
        plain(prediction)

    def test_finite_large_current_errors_use_stable_reductions(self):
        prediction = forecast(*self.fixture([0.]*8))
        measurement = measure_current(np.full((129, 129), 1e300), prediction)
        self.assertTrue(measurement["available"])
        for arm in measurement["arms"].values():
            self.assertEqual(arm["mae_dn"], 1e300)
            self.assertEqual(arm["rmse_dn"], 1e300)


if __name__ == "__main__":
    unittest.main()
