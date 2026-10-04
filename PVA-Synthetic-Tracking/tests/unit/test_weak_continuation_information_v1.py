import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("weak_continuation_information_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/weak_continuation_information_v1.py"
spec = importlib.util.spec_from_file_location("weak_information", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def forecast(identity="focal", center=(50, 50), polarity="bright", covariance=None):
    return dict(identity=identity, reference_xy=list(center), polarity=polarity,
                innovation_covariance_2x2=covariance or [[81, 0], [0, 81]])


def generated(points=(), *, support=True, capture=(0, 0, 101, 101), tile=(2, 2, 99, 99)):
    """Independent generated field/flag fixture, never loads captured evidence."""
    metadata = dict(float_fields=list(m.FLOAT_FIELDS), flag_fields=list(m.FLAG_FIELDS), ready=True,
                    temporal_threshold_sigma_float32=4.0, spatial_threshold_sigma_float32=3.0,
                    rectangle=dict(capture_bounds_exclusive_xyxy=list(capture), tile_bounds_exclusive_xyxy=list(tile), shape_hw=[200, 200]))
    values = np.zeros((capture[3]-capture[1], capture[2]-capture[0], 20), np.float32)
    flags = np.zeros((*values.shape[:2], 13), np.uint8)
    v = {name:values[..., i] for i, name in enumerate(m.FLOAT_FIELDS)}
    f = {name:flags[..., i] for i, name in enumerate(m.FLAG_FIELDS)}
    for name in ("variance", "tile_sigma_float", "noise"):
        v[name][:] = 1
    v["image"][:] = 100
    v["temporal_threshold_dn"][:] = 4
    v["spatial_threshold_dn"][:] = 3
    for x,y,temporal,spatial in points:
        v["temporal"][y-capture[1],x-capture[0]] = temporal
        v["spatial"][y-capture[1],x-capture[0]] = spatial
    v["centered_temporal"][:] = v["temporal"]
    for name in ("support", "previous_support", "native_eligible", "eligible"):
        f[name][:] = support
    f["ready"][:] = 1
    padded = np.pad(np.abs(v["temporal"]), 2, constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    maximum = windows.max(axis=(-2,-1))
    v["neighborhood_max_abs"][:] = maximum
    f["raw_absolute_peak"][:] = np.abs(v["temporal"]) >= maximum
    f["finite_neighborhood"][:] = np.isfinite(windows).all(axis=(-2,-1))
    for name, sign in (("positive", 1), ("negative", -1)):
        v[name+"_signed_temporal"][:] = sign*v["temporal"]
        v[name+"_signed_spatial"][:] = sign*v["spatial"]
        v[name+"_score"][:] = sign*v["temporal"]
        f[name+"_temporal_pass"][:] = v[name+"_signed_temporal"] >= 4
        f[name+"_spatial_pass"][:] = v[name+"_signed_spatial"] >= 3
        f[name+"_candidate"][:] = f["eligible"] & f[name+"_temporal_pass"] & f[name+"_spatial_pass"] & f["raw_absolute_peak"]
    return values, flags, metadata


def evaluate(data, focal=None, priors=None):
    focal = focal or forecast()
    return m.enumerate_peaks(*data, focal, priors if priors is not None else [focal])


class WeakInformationTests(unittest.TestCase):
    def test_weak_point_both_polarities_and_existing_spatial_equality(self):
        for polarity, sign in (("bright", 1), ("dark", -1)):
            out = evaluate(generated([(50,50,sign*3,sign*3)]), forecast(polarity=polarity))
            self.assertTrue(out["coverage_known"])
            self.assertEqual(out["weak_peak_count"], 1)
            self.assertEqual(out["original_threshold_peak_count"], 0)
            point = out["observed_peaks"][0]
            self.assertEqual(point["score"], 3)
            self.assertIsNone(point["identity_assignment"])
            self.assertIsNone(point["acceptance_decision"])

    def test_disappearing_blank_has_no_observed_candidates_not_identity_claim(self):
        out = evaluate(generated())
        self.assertEqual(out["observed_peaks"], [])
        self.assertTrue(out["coverage_known"])
        self.assertIsNone(out["identity_assignment"])

    def test_identical_observations_different_physical_truth_indistinguishable(self):
        for polarity, sign in (("bright",1), ("dark",-1)):
            true_weak_point = generated([(50,50,sign*3,sign*4)])
            disappeared_plus_unrelated_light = copy.deepcopy(true_weak_point)
            a = evaluate(true_weak_point, forecast(polarity=polarity))
            b = evaluate(disappeared_plus_unrelated_light, forecast(polarity=polarity))
            self.assertEqual(a, b)
            self.assertEqual(a["identity_observability"], "not_identified_from_these_observations")

    def test_two_equal_peaks_descriptive_y_x_ties_only(self):
        out = evaluate(generated([(55,50,3,4),(45,50,3,4)]))
        self.assertEqual([p["reference_xy"] for p in out["observed_peaks"]], [[45,50],[55,50]])
        self.assertEqual([p["descriptive_rank"] for p in out["observed_peaks"]], [1,2])
        self.assertEqual(out["weak_peak_count"], 2)

    def test_score_precedes_y_x(self):
        out = evaluate(generated([(45,50,2,4),(55,50,3,4),(50,40,3,4)]))
        self.assertEqual([p["reference_xy"] for p in out["observed_peaks"]], [[50,40],[55,50],[45,50]])

    def test_unsupported_gate_censors_coverage_not_zero_fill(self):
        out = evaluate(generated([(50,50,3,4)], support=False))
        self.assertFalse(out["coverage_known"])
        self.assertIn("gate_pixels_not_eligible", out["coverage_unknown_reasons"])
        self.assertEqual(out["observed_peak_count"], 0)

    def test_truncated_full_disk_even_if_peak_visible(self):
        out = evaluate(generated([(50,50,3,4)], tile=(20,20,90,90)))
        self.assertFalse(out["coverage_known"])
        self.assertIn("focal_45px_disk_outside_full_tile", out["coverage_unknown_reasons"])
        self.assertEqual(out["observed_peak_count"], 1)

    def test_missing_peak_neighborhood_explicit(self):
        out = evaluate(generated([(50,50,3,4)], capture=(5,5,96,96), tile=(2,2,99,99)))
        self.assertIn("peak_neighborhood_missing", out["coverage_unknown_reasons"])
        self.assertFalse(out["coverage_known"])

    def test_nonfinite_gate_pixel_censors_coverage(self):
        values, flags, metadata = generated([(50,50,3,4)])
        values[40,40,m.FLOAT_FIELDS.index("blur")] = np.nan
        out = evaluate((values, flags, metadata))
        self.assertIn("gate_pixels_nonfinite", out["coverage_unknown_reasons"])
        self.assertEqual(out["observed_peak_count"], 1)

    def test_opposite_polarity_neighbor_suppresses_raw_absolute_peak(self):
        out = evaluate(generated([(50,50,3,4),(51,50,-5,-4)]))
        self.assertEqual(out["observed_peak_count"], 0)
        dark = evaluate(generated([(50,50,3,4),(51,50,-5,-4)]), forecast(polarity="dark"))
        self.assertEqual(dark["original_threshold_peak_count"], 1)

    def test_unsupported_neighbor_still_participates_in_raw_peak(self):
        values, flags, metadata = generated([(50,50,3,4),(51,50,-5,-4)])
        for name in ("support", "eligible", "native_eligible", "negative_candidate"):
            flags[50,51,m.FLAG_FIELDS.index(name)] = 0
        out = evaluate((values, flags, metadata))
        self.assertEqual(out["observed_peak_count"], 0)
        self.assertIn("gate_pixels_not_eligible", out["coverage_unknown_reasons"])

    def test_nonfinite_neighborhood_is_explicitly_censored(self):
        values, flags, metadata = generated([(50,50,3,4),(51,50,np.nan,0)])
        out = evaluate((values, flags, metadata))
        self.assertEqual(out["observed_peak_count"], 0)
        self.assertIn("peak_neighborhood_nonfinite", out["coverage_unknown_reasons"])

    def test_external_forecast_center_still_competes(self):
        focal = forecast()
        external = forecast("external", (105,50))
        out = evaluate(generated([(94,50,3,4)]), focal, [external, focal])
        matches = out["observed_peaks"][0]["competing_prior_identity_gates"]
        self.assertEqual([p["identity"] for p in matches], ["external"])
        self.assertTrue(matches[0]["forecast_center_outside_capture"])

    def test_all_competing_gates_no_identity_or_visibility_filter(self):
        priors = [forecast("z", (54,50)), forecast("a", (52,50)), forecast("opposite", (50,50), "dark")]
        out = evaluate(generated([(50,50,3,4)]), priors=priors)
        self.assertEqual([p["identity"] for p in out["observed_peaks"][0]["competing_prior_identity_gates"]], ["a","z"])

    def test_competing_gates_evaluated_at_peaks_only_including_empty(self):
        for points, count in (([(50,50,3,4),(60,50,2,4)], 2), ([], 0)):
            with patch.object(m, "_gate", wraps=m._gate) as gate:
                evaluate(generated(points), priors=[forecast("other",(52,50))])
            self.assertEqual(gate.call_count, 2)
            self.assertEqual(gate.call_args_list[0].args[1].shape, (101,101))
            self.assertEqual(gate.call_args_list[1].args[1].shape, (count,))

    def test_changed_existing_sigma_settings_rejected(self):
        for name, changed in (("temporal_threshold_sigma_float32", 3.0), ("spatial_threshold_sigma_float32", 2.0)):
            data = generated([(50,50,3,4)])
            data[2][name] = changed
            with self.assertRaisesRegex(ValueError, "frozen temporal4/spatial3"):
                evaluate(data)

    def test_both_gates_and_exact_threshold_equality(self):
        out = evaluate(generated([(95,50,4,3)]))
        self.assertEqual(out["original_threshold_peak_count"], 1)
        point = out["observed_peaks"][0]
        self.assertEqual(point["distance_px"], 45)
        self.assertEqual(point["mahalanobis_squared"], 25)
        self.assertEqual(evaluate(generated([(96,50,4,3)]))["observed_peak_count"], 0)
        narrow = forecast(covariance=[[1,0],[0,1]])
        self.assertEqual(evaluate(generated([(56,50,4,3)]), narrow)["observed_peak_count"], 0)

    def test_zero_temporal_excluded_no_implicit_positive_threshold(self):
        out = evaluate(generated([(50,50,0,4),(60,50,np.finfo(np.float32).tiny,4)]))
        self.assertEqual(out["observed_peak_count"], 1)
        self.assertEqual(out["observed_peaks"][0]["reference_xy"], [60,50])

    def test_arrays_forecasts_metadata_read_only(self):
        data = generated([(50,50,3,4)])
        focal, priors = forecast(), [forecast("other",(52,50))]
        before = (data[0].tobytes(), data[1].tobytes(), copy.deepcopy(data[2]), copy.deepcopy(focal), copy.deepcopy(priors))
        data[0].setflags(write=False)
        data[1].setflags(write=False)
        evaluate(data, focal, priors)
        self.assertEqual(before, (data[0].tobytes(), data[1].tobytes(), data[2], focal, priors))

    def test_named_fields_permutation_preserves_results(self):
        data = generated([(50,50,3,4)])
        expected = evaluate(data)
        values, flags, metadata = copy.deepcopy(data)
        metadata["float_fields"].reverse()
        metadata["flag_fields"].reverse()
        actual = evaluate((values[...,::-1].copy(), flags[...,::-1].copy(), metadata))
        self.assertEqual(actual, expected)

    def test_bad_schema_flag_predicate_covariance_and_duplicate_identity(self):
        data = generated([(50,50,3,4)])
        bad = copy.deepcopy(data)
        bad[2]["float_fields"][0] = "unknown"
        with self.assertRaisesRegex(ValueError, "field schema"):
            evaluate(bad)
        bad = copy.deepcopy(data)
        bad[1][50,50,m.FLAG_FIELDS.index("raw_absolute_peak")] = 0
        with self.assertRaisesRegex(ValueError, "raw peak"):
            evaluate(bad)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            evaluate(data, forecast(covariance=[[0,0],[0,1]]))
        with self.assertRaisesRegex(ValueError, "duplicate prior"):
            evaluate(data, priors=[forecast("same"),forecast("same")])
        with self.assertRaisesRegex(ValueError, "focal forecast differs"):
            evaluate(data, priors=[forecast(center=(51,50))])


if __name__ == "__main__":
    unittest.main()
