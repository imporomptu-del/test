import inspect
import json
from pathlib import Path
import statistics
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v51_history_diagnostic import (
    MATRIX_FIELDS, VECTOR_FIELDS, diagnose, diagnostic_fingerprint,
)


def one(values):
    return diagnose(np.asarray(values, dtype=float).reshape(8, 1))


class HistoryDiagnosticTests(unittest.TestCase):
    def test_interface_accepts_only_history(self):
        self.assertEqual(tuple(inspect.signature(diagnose).parameters), ("history",))
        for keyword in ("current", "labels", "geometry", "threshold"):
            with self.assertRaises(TypeError):
                diagnose(np.ones((8, 1)), **{keyword: None})

    def test_invalid_shapes_and_types(self):
        for value in (np.ones(8), np.ones((7, 3)), np.ones((9, 3)),
                      np.ones((8, 3, 1)), np.ones((8, 3), dtype=bool),
                      np.ones((8, 3), dtype=complex), np.full((8, 3), "1"),
                      np.ones((8, 3), dtype=object)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    diagnose(value)

    def test_empty_keeps_all_declared_shapes(self):
        result = diagnose(np.empty((8, 0)))
        self.assertEqual((result["total_count"], result["available_count"]), (0, 0))
        for name in VECTOR_FIELDS:
            self.assertEqual(result[name].shape, (0,))
        for name in MATRIX_FIELDS:
            self.assertEqual(result[name].shape, (3, 0))
        self.assertEqual(result["point_available"].shape, (0,))
        self.assertEqual(result["point_unavailable_reasons"], ())

    def test_constant_is_available_but_return_and_center_are_undefined(self):
        result = one([100]*8)
        self.assertEqual(result["available_count"], 1)
        for name in ("slow8", "fast3", "early5"):
            self.assertEqual(result[name][0], 100)
        self.assertEqual(result["scale"][0], 1)
        self.assertEqual(result["departure_envelope"][0], 0)
        self.assertTrue(np.isnan(result["return_fraction"][0]))
        self.assertTrue(np.isnan(result["recent_center_suffix_length"][0]))
        self.assertFalse(result["return_fraction_defined"][0])
        self.assertFalse(result["recent_center_defined"][0])
        np.testing.assert_array_equal(result["recent_center_margins"], 0)

    def test_recent_positive_step(self):
        result = one([100]*5+[140]*3)
        self.assertEqual(result["fast_slow_delta"][0], 40)
        self.assertEqual(result["early_recent_delta"][0], 40)
        np.testing.assert_array_equal(result["recent_departures"], 40)
        np.testing.assert_array_equal(result["recent_center_margins"], 40)
        self.assertEqual(result["recent_center_suffix_length"][0], 3)
        self.assertEqual(result["return_fraction"][0], 0)

    def test_recent_negative_step_preserves_sign_and_positive_support(self):
        result = one([100]*5+[60]*3)
        self.assertEqual(result["fast_slow_delta"][0], -40)
        self.assertEqual(result["early_recent_delta"][0], -40)
        np.testing.assert_array_equal(result["recent_departures"], -40)
        np.testing.assert_array_equal(result["recent_center_margins"], 40)
        self.assertEqual(result["recent_center_suffix_length"][0], 3)

    def test_observed_return_and_suffix_reset(self):
        result = one([100]*5+[140, 140, 100])
        self.assertEqual(result["fast3"][0], 140)
        self.assertEqual(result["return_fraction"][0], 1)
        self.assertEqual(result["recent_center_suffix_length"][0], 0)
        np.testing.assert_array_equal(result["recent_center_margins"][:, 0], [40, 40, -40])

    def test_partial_return(self):
        result = one([100]*5+[140, 140, 120])
        self.assertEqual(result["return_fraction"][0], .5)
        self.assertEqual(result["recent_center_suffix_length"][0], 0)
        self.assertEqual(result["recent_center_margins"][-1, 0], 0)

    def test_sign_reversal_is_not_oversold_as_return(self):
        result = one([100]*5+[140, 140, 60])
        self.assertEqual(result["return_fraction"][0], 0)
        np.testing.assert_array_equal(result["recent_departures"][:, 0], [40, 40, -40])
        self.assertEqual(result["recent_center_suffix_length"][0], 0)

    def test_terminal_departure_visible_when_fast_equals_early(self):
        result = one([100]*7+[140])
        self.assertEqual(result["fast_slow_delta"][0], 0)
        self.assertEqual(result["early_recent_delta"][0], 0)
        self.assertEqual(result["recent_departures"][-1, 0], 40)
        self.assertEqual(result["departure_envelope"][0], 40)
        self.assertEqual(result["return_fraction"][0], 0)
        self.assertFalse(result["recent_center_defined"][0])

    def test_completed_early_recent_pulse_visible_after_fast_returns(self):
        result = one([100]*5+[140, 100, 100])
        self.assertEqual(result["fast3"][0], 100)
        self.assertEqual(result["return_fraction"][0], 1)
        self.assertFalse(result["recent_center_defined"][0])
        self.assertEqual(result["departure_envelope"][0], 40)

    def test_suffix_is_only_latest_three_and_stops_at_first_nonpositive(self):
        for recent, expected in (([90, 120, 120], 2), ([120, 90, 120], 1),
                                 ([120, 120, 90], 0), ([120]*3, 3)):
            with self.subTest(recent=recent):
                self.assertEqual(one([100]*5+recent)["recent_center_suffix_length"][0], expected)

    def test_tiny_margin_is_preserved_not_thresholded_as_confident_change(self):
        result = one([0]*5+[1e-9]*3)
        self.assertEqual(result["scale"][0], 1)
        self.assertEqual(result["recent_center_suffix_length"][0], 3)
        np.testing.assert_array_equal(result["recent_center_margins"], 1e-9)
        self.assertFalse(result["metadata"]["classification_threshold_or_model_selection_performed"])

    def test_scale_matches_v50_unscaled_mad_and_raw_values_retained(self):
        result = one(list(range(8)))
        self.assertEqual(result["slow8"][0], 3.5)
        self.assertEqual(result["early5"][0], 2)
        self.assertEqual(result["fast3"][0], 6)
        self.assertEqual(result["scale"][0], 2)
        for raw, normalized in (
            ("fast_slow_delta", "fast_slow_delta_normalized"),
            ("early_recent_delta", "early_recent_delta_normalized"),
            ("recent_departures", "recent_departures_normalized"),
            ("departure_envelope", "departure_envelope_normalized"),
            ("recent_center_margins", "recent_center_margins_normalized"),
        ):
            np.testing.assert_array_equal(result[normalized], result[raw]/2)

    def test_same_prefix_current_twins_have_identical_diagnostics(self):
        pairs = (([100]*5+[140]*3, 140, 100),
                 ([100]*7+[140], 140, 100),
                 ([100]*5+[140, 140, 100], 100, 140))
        for history, first_current, second_current in pairs:
            self.assertNotEqual(first_current, second_current)
            first = one(history)
            second = one(history)
            self.assertEqual(first["diagnostic_sha256"], second["diagnostic_sha256"])
            self.assertTrue(first["metadata"]["equal_prior_prefix_cannot_identify_different_future_responses"])

    def test_same_prefix_point_prediction_lower_bound(self):
        for amplitude in (1, 4, 16, 64):
            for forecast in (100, 100+amplitude/4, 100+amplitude/2, 100+amplitude, 300):
                self.assertGreaterEqual(max(abs(100-forecast), abs(100+amplitude-forecast)), amplitude/2)

    def test_missing_retains_indices_without_imputation(self):
        history = np.tile(np.arange(8, dtype=float)[:, None], (1, 5))
        history[1, 1] = np.nan
        history[4, 2] = np.inf
        history[7, 4] = -np.inf
        result = diagnose(history)
        np.testing.assert_array_equal(result["point_available"], [True, False, False, True, False])
        self.assertEqual(result["available_count"], 2)
        self.assertEqual(result["total_count"], 5)
        for name in VECTOR_FIELDS:
            self.assertTrue(np.isnan(result[name][[1, 2, 4]]).all(), name)
        for name in MATRIX_FIELDS:
            self.assertTrue(np.isnan(result[name][:, [1, 2, 4]]).all(), name)
        for index in (1, 2, 4):
            self.assertEqual(result["point_unavailable_reasons"][index], "nonfinite_input_after_float64_conversion")

    def test_all_missing_is_not_negative_or_zero_observation(self):
        result = diagnose(np.full((8, 4), np.nan))
        self.assertEqual(result["available_count"], 0)
        self.assertEqual(result["total_count"], 4)
        self.assertTrue(np.isnan(result["slow8"]).all())
        self.assertFalse(result["return_fraction_defined"].any())

    def test_overflow_marks_only_affected_points_unavailable(self):
        huge = np.finfo(np.float64).max
        history = np.column_stack((np.arange(8), [huge]*8,
                                   [-huge]*5+[huge]*3, [100]*8))
        with np.errstate(all="raise"):
            result = diagnose(history)
        np.testing.assert_array_equal(result["point_available"], [True, False, False, True])
        for index in (1, 2):
            self.assertEqual(result["point_unavailable_reasons"][index], "nonfinite_descriptor_arithmetic")
        for name in VECTOR_FIELDS:
            self.assertTrue(np.isnan(result[name][[1, 2]]).all())

    def test_input_mutation_cannot_change_output(self):
        history = np.arange(24, dtype=float).reshape(8, 3)
        result = diagnose(history)
        before = diagnostic_fingerprint(result)
        history[:] = -999
        self.assertEqual(diagnostic_fingerprint(result), before)

    def test_underflow_normalization_is_finite_not_an_exception(self):
        tiny = np.nextafter(0., 1.)
        with np.errstate(all="raise"):
            result = one([-1e100, -1e100, -1e100, 0, 0, tiny, 1e100, 1e100])
        self.assertTrue(result["point_available"][0])
        self.assertTrue(np.isfinite(result["early_recent_delta_normalized"][0]))

    def test_all_arrays_are_readonly_independent_and_input_is_unchanged(self):
        history = np.arange(24, dtype=float).reshape(8, 3)
        original = history.copy()
        result = diagnose(history)
        for name, array in result.items():
            if isinstance(array, np.ndarray):
                self.assertFalse(array.flags.writeable, name)
                self.assertFalse(np.shares_memory(array, history), name)
                with self.assertRaises(ValueError):
                    array.flat[0] = 0
        np.testing.assert_array_equal(history, original)

    def test_hash_covers_arrays_availability_metadata_and_reasons(self):
        result = one([100]*5+[140]*3)
        expected = result["diagnostic_sha256"]
        for name, value in (("available_count", 0),
                            ("point_unavailable_reasons", ["changed"]),
                            ("early5", np.asarray([101.])),
                            ("point_available", np.asarray([False])),
                            ("metadata", {"changed": True})):
            changed = dict(result)
            changed[name] = value
            self.assertNotEqual(diagnostic_fingerprint(changed), expected, name)

    def test_hash_ignores_own_digest_and_canonicalizes_nonfinite_to_null(self):
        result = one([100]*8)
        changed = dict(result, diagnostic_sha256="not part of digest")
        self.assertEqual(diagnostic_fingerprint(changed), result["diagnostic_sha256"])
        for value in (np.nan, np.inf, -np.inf, None):
            self.assertEqual(diagnostic_fingerprint({"x": [value]}),
                             diagnostic_fingerprint({"x": [None]}))
        with self.assertRaises(ValueError):
            diagnostic_fingerprint([])

    def test_hash_agrees_after_json_safe_serialization(self):
        def safe(value):
            if isinstance(value, np.ndarray):
                return safe(value.tolist())
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        result = one([100]*8)
        roundtrip = json.loads(json.dumps(safe(result), allow_nan=False))
        self.assertEqual(diagnostic_fingerprint(roundtrip), result["diagnostic_sha256"])

    def test_independent_columns_permutation_and_duplication(self):
        history = np.random.default_rng(5102).normal(size=(8, 9))
        result = diagnose(history)
        permutation = [4, 0, 8, 4, 1]
        other = diagnose(history[:, permutation])
        for name in VECTOR_FIELDS:
            np.testing.assert_array_equal(other[name], result[name][permutation])
        for name in MATRIX_FIELDS:
            np.testing.assert_array_equal(other[name], result[name][:, permutation])

    def test_integer_float32_and_noncontiguous_inputs(self):
        original = np.arange(80).reshape(8, 10)
        reference = diagnose(original[:, ::2].astype(float))
        for history in (original[:, ::2], original[:, ::2].astype(np.float32)):
            result = diagnose(history)
            for name in VECTOR_FIELDS+MATRIX_FIELDS:
                np.testing.assert_array_equal(result[name], reference[name])

    def test_sign_symmetry_and_translation(self):
        history = np.random.default_rng(5103).integers(-20, 20, (8, 17)).astype(float)
        base = diagnose(history)
        negative = diagnose(-history)
        translated = diagnose(history+100)
        for name in ("slow8", "fast3", "early5", "fast_slow_delta", "early_recent_delta", "recent_departures"):
            np.testing.assert_array_equal(negative[name], -base[name])
        for name in ("scale", "departure_envelope", "return_fraction", "recent_center_margins", "recent_center_suffix_length"):
            np.testing.assert_array_equal(negative[name], base[name])
        for name in ("slow8", "fast3", "early5"):
            np.testing.assert_array_equal(translated[name], base[name]+100)
        for name in ("scale", "fast_slow_delta", "early_recent_delta", "recent_departures", "departure_envelope", "return_fraction", "recent_center_margins", "recent_center_suffix_length"):
            np.testing.assert_array_equal(translated[name], base[name])

    def test_randomized_scalar_oracle(self):
        history = np.random.default_rng(5101).normal(100, 25, (8, 300))
        result = diagnose(history)
        self.assertEqual(result["available_count"], 300)
        for index, column in enumerate(history.T.tolist()):
            slow, fast, early = (statistics.median(column), statistics.median(column[-3:]), statistics.median(column[:5]))
            scale = max(1, statistics.median([abs(value-slow) for value in column]))
            departures = [value-early for value in column[-3:]]
            envelope = max(abs(value) for value in departures)
            margins = [abs(value-early)-abs(value-fast) for value in column[-3:]]
            expected = dict(slow8=slow, fast3=fast, early5=early, scale=scale,
                fast_slow_delta=fast-slow, early_recent_delta=fast-early,
                departure_envelope=envelope,
                return_fraction=1-abs(departures[-1])/envelope if envelope else np.nan)
            suffix = np.nan
            if fast != early:
                suffix = 0
                for margin in reversed(margins):
                    if margin <= 0:
                        break
                    suffix += 1
            expected["recent_center_suffix_length"] = suffix
            for name, value in expected.items():
                np.testing.assert_allclose(result[name][index], value, rtol=0, atol=0)
            np.testing.assert_array_equal(result["recent_departures"][:, index], departures)
            np.testing.assert_array_equal(result["recent_center_margins"][:, index], margins)

    def test_metadata_has_no_full_camera_or_detection_claim(self):
        metadata = one([0]*8)["metadata"]
        for name in ("geometry_causality_certified", "full_camera_online_causality_certified",
                     "support_purity_or_guard_to_core_transfer_certified",
                     "classification_threshold_or_model_selection_performed",
                     "intervals_or_coverage_guarantees_produced", "source_object_or_production_decision"):
            self.assertFalse(metadata[name], name)
        self.assertTrue(metadata["scale_is_descriptor_not_physical_noise_bound"])


if __name__ == "__main__":
    unittest.main()
