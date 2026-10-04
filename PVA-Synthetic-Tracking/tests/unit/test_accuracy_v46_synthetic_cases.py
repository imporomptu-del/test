import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v46_synthetic_cases as cases


class SyntheticCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = cases.build_cases()
        cls.by_id = {case["case_id"]: case for case in cls.generated}

    def test_predeclared_manifest_order_and_count_are_exact(self):
        manifest = cases.scenario_manifest()
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(manifest["case_count"], 28)
        self.assertEqual(len(self.generated), 28)
        self.assertEqual(len(self.by_id), 28)
        self.assertEqual([s["case_id"] for s in manifest["scenarios"]], list(self.by_id))
        self.assertNotIn("baseline_v45_exact", self.by_id)
        self.assertNotIn("expected_sign", json.dumps(manifest))
        self.assertTrue(manifest["synthetic_only"])

    def test_adapter_has_only_observations_and_reported_geometry(self):
        for case in self.generated:
            self.assertEqual(set(case), {"case_id", "family", "adapter_inputs", "generator_truth", "provenance"})
            self.assertEqual(set(case["adapter_inputs"]), set(cases.ADAPTER_KEYS))
            self.assertEqual(case["adapter_inputs"]["history129"].shape, (8, 129, 129))
            self.assertEqual(case["adapter_inputs"]["current129"].shape, (129, 129))
            self.assertEqual(case["adapter_inputs"]["current129"].dtype, np.float64)
            for key in ("history129", "current129"):
                data = case["adapter_inputs"][key]
                self.assertTrue(np.isfinite(data).all())
                self.assertGreater(float(data.min()), 0.)
                self.assertLess(float(data.max()), 255.)
            json.dumps(case["generator_truth"], allow_nan=False)
            json.dumps(case["provenance"], allow_nan=False)

    def test_every_forecast_independently_reconstructs_from_prior_ols(self):
        for case in self.generated:
            p = case["provenance"]
            reported = p["reported_prior_world_centers_xy"]
            used = [i for i, point in enumerate(reported) if point is not None][-4:]
            times = np.asarray(p["prior_times"])[used]
            design = np.column_stack((np.ones(4), times))
            expected = np.linalg.lstsq(design, np.asarray([reported[i] for i in used]), rcond=None)[0][0]
            np.testing.assert_allclose(p["predicted_world_xy"], expected, atol=1e-12)
            self.assertEqual(p["ols_history_indices"], used)
            np.testing.assert_array_equal(p["ols_times"], times)
            self.assertAlmostEqual(sum(p["ols_weights"]), 1.)
            self.assertAlmostEqual(float(np.asarray(p["ols_weights"])@times), 0.)
            np.testing.assert_allclose(np.asarray(p["ols_weights"])@np.asarray([reported[i] for i in used]),
                                       p["predicted_world_xy"], atol=1e-12)

    def test_crop_origin_and_fractional_offset_follow_round_half_up(self):
        for case in self.generated:
            p, a = case["provenance"], case["adapter_inputs"]
            prediction = np.asarray(p["predicted_world_xy"])
            center = np.floor(prediction+.5).astype(int)
            origin = center-64
            np.testing.assert_array_equal(p["integer_crop_center_world_xy"], center)
            np.testing.assert_array_equal(p["crop_origin_world_xy"], origin)
            np.testing.assert_array_equal(a["predicted_offset_xy"], prediction-center)
            self.assertLessEqual(np.max(np.abs(a["predicted_offset_xy"])), .5)
            for world, local in zip(p["reported_prior_world_centers_xy"], a["prior_centers_xy"]):
                if world is None: self.assertIsNone(local)
                else: np.testing.assert_array_equal(local, np.asarray(world)-origin)
        self.assertEqual(self.by_id["prior_bias_halfpx"]["provenance"]["integer_crop_center_world_xy"], [65, 64])
        self.assertEqual(self.by_id["prior_bias_halfpx"]["adapter_inputs"]["predicted_offset_xy"], [-.5, 0.])

    def test_current_truth_variants_never_change_history_forecast_or_placement(self):
        base = self.by_id["ordinary_prior_forecast"]
        for case_id in cases.CURRENT_ONLY_GROUP:
            case = self.by_id[case_id]
            np.testing.assert_array_equal(case["adapter_inputs"]["history129"], base["adapter_inputs"]["history129"])
            for key in ("prior_centers_xy", "predicted_offset_xy", "polarity"):
                self.assertEqual(case["adapter_inputs"][key], base["adapter_inputs"][key])
            for key in ("reported_prior_world_centers_xy", "predicted_world_xy", "integer_crop_center_world_xy",
                        "crop_origin_world_xy", "ols_history_indices", "ols_weights"):
                self.assertEqual(case["provenance"][key], base["provenance"][key])
            for key in ("forecast_uses_current_image", "forecast_uses_current_truth", "crop_uses_current_image",
                        "crop_uses_current_truth"):
                self.assertFalse(case["provenance"][key])
            self.assertEqual(case["provenance"]["prior_equivalence_group"], "base_prior_world")
            self.assertTrue(case["provenance"]["forecast_is_synthetic_linear_test_not_real_quadratic_tracker"])
        self.assertEqual(list(inspect.signature(cases._forecast_geometry).parameters), ["reported_world"])

    def test_bias_changes_reported_geometry_and_crop_not_rendered_world_truth(self):
        base = self.by_id["ordinary_prior_forecast"]
        for case_id, integer_shift, measurement_bias in (("prior_bias_halfpx", 1, .5), ("prior_bias_twopx", 2, 2.)):
            case = self.by_id[case_id]
            self.assertEqual(case["generator_truth"]["apparent_point_prior_world_centers_xy"],
                             base["generator_truth"]["apparent_point_prior_world_centers_xy"])
            self.assertEqual(case["generator_truth"]["apparent_point_current_world_center_xy"], [64., 64.])
            delta = (np.asarray(case["provenance"]["reported_prior_world_centers_xy"])
                     - np.asarray(base["provenance"]["reported_prior_world_centers_xy"]))
            np.testing.assert_array_equal(delta, np.tile([measurement_bias, 0.], (8, 1)))
            np.testing.assert_array_equal(case["adapter_inputs"]["history129"][:, :, :-integer_shift],
                                          base["adapter_inputs"]["history129"][:, :, integer_shift:])
            np.testing.assert_array_equal(case["adapter_inputs"]["current129"][:, :-integer_shift],
                                          base["adapter_inputs"]["current129"][:, integer_shift:])

    def test_missing_measurements_preserve_original_images_and_time_indices(self):
        base = self.by_id["ordinary_prior_forecast"]
        for case_id, remaining in (("missing_three_measurements", 5), ("missing_four_measurements", 4)):
            case = self.by_id[case_id]
            self.assertEqual(sum(p is not None for p in case["adapter_inputs"]["prior_centers_xy"]), remaining)
            np.testing.assert_array_equal(case["adapter_inputs"]["history129"], base["adapter_inputs"]["history129"])
            self.assertEqual(case["provenance"]["prior_times"], list(range(-8, 0)))
            self.assertEqual(case["provenance"]["predicted_world_xy"], [64., 64.])

    def test_incorrect_measurements_do_not_change_generated_true_trajectory(self):
        base_truth = self.by_id["ordinary_prior_forecast"]["generator_truth"]
        for case_id in ("incorrect_last_two_measurements", "reversed_measurement_order", "prior_jitter_twopx"):
            case = self.by_id[case_id]
            self.assertEqual(case["generator_truth"]["apparent_point_prior_world_centers_xy"],
                             base_truth["apparent_point_prior_world_centers_xy"])
            self.assertNotEqual(case["provenance"]["predicted_world_xy"], [64., 64.])
        reversed_case = self.by_id["reversed_measurement_order"]
        self.assertEqual(reversed_case["provenance"]["reported_prior_world_centers_xy"],
                         list(reversed(base_truth["apparent_point_prior_world_centers_xy"])))

    def test_scalar_rendering_checks_independent_truth_not_measurement_positions(self):
        def point(x, y, center, amplitude, sigma, angle):
            dx, dy = x-center[0], y-center[1]
            u = np.cos(angle)*dx+np.sin(angle)*dy
            v = -np.sin(angle)*dx+np.cos(angle)*dy
            return amplitude*np.exp(-(u*u/(2*sigma[0]**2)+v*v/(2*sigma[1]**2)))
        for case in self.generated:
            t, p, a = case["generator_truth"], case["provenance"], case["adapter_inputs"]
            for local_x, local_y in ((64, 64), (63, 65), (40, 64)):
                x, y = local_x+p["crop_origin_world_xy"][0], local_y+p["crop_origin_world_xy"][1]
                background = 50+20*(x >= 64)+8*np.sin(y/12)
                for index in (0, 4, 7, 8):
                    is_current = index == 8
                    center = t["apparent_point_current_world_center_xy"] if is_current else t["apparent_point_prior_world_centers_xy"][index]
                    amplitude = t["current_peak_dn"] if is_current else t["prior_peak_dn"][index]
                    sigma = t["current_psf_sigma_xy"] if is_current else t["prior_psf_sigma_xy"][index]
                    angle = t["current_psf_angle_radians"] if is_current else t["prior_psf_angle_radians"][index]
                    expected = background+point(x, y, center, amplitude, sigma, angle)
                    fixed = t["fixed_emitter"]
                    if fixed is not None:
                        fa = fixed["current_peak_dn"] if is_current else fixed["prior_peak_dn"][index]
                        expected += point(x, y, fixed["center_world_xy"], fa, fixed["psf_sigma_xy"], fixed["psf_angle_radians"])
                    observed = a["current129"][local_y, local_x] if is_current else a["history129"][index, local_y, local_x]
                    self.assertAlmostEqual(observed, expected, places=12)

    def test_current_fixed_and_moving_presence_are_not_conflated(self):
        for case in self.generated:
            truth = case["generator_truth"]
            self.assertEqual(truth["any_current_emitter_present"],
                             truth["current_fixed_emitter_present"] or truth["current_moving_source_present"])
            self.assertFalse(truth["physical_identity_certified_from_images"])
            if case["family"] == "fixed_confuser":
                self.assertFalse(truth["current_moving_source_present"])
                self.assertTrue(truth["current_fixed_emitter_present"])
                self.assertTrue(truth["physical_identity_ambiguous"])
            if case["family"] == "mixed_source_fixed":
                self.assertTrue(truth["current_moving_source_present"])
                self.assertTrue(truth["current_fixed_emitter_present"])
        self.assertFalse(self.by_id["current_absent"]["generator_truth"]["any_current_emitter_present"])

    def test_observational_twins_are_bitwise_identical_but_latent_identities_differ(self):
        base = self.by_id["ordinary_prior_forecast"]
        for case_id in cases.TWIN_IDS:
            twin = self.by_id[case_id]
            for key in cases.ADAPTER_KEYS:
                if isinstance(base["adapter_inputs"][key], np.ndarray):
                    np.testing.assert_array_equal(twin["adapter_inputs"][key], base["adapter_inputs"][key])
                    self.assertEqual(twin["adapter_inputs"][key].tobytes(), base["adapter_inputs"][key].tobytes())
                else: self.assertEqual(twin["adapter_inputs"][key], base["adapter_inputs"][key])
        self.assertNotEqual(self.by_id[cases.TWIN_IDS[0]]["generator_truth"]["latent_world_interpretation"],
                            self.by_id[cases.TWIN_IDS[1]]["generator_truth"]["latent_world_interpretation"])
        self.assertTrue(self.by_id[cases.TWIN_IDS[0]]["generator_truth"]["current_moving_source_present"])
        self.assertFalse(self.by_id[cases.TWIN_IDS[1]]["generator_truth"]["current_moving_source_present"])

    def test_rendering_and_manifest_are_deterministic_and_unshared(self):
        repeat = cases.build_cases()
        for expected, observed in zip(self.generated, repeat):
            self.assertEqual(expected["provenance"], observed["provenance"])
            np.testing.assert_array_equal(expected["adapter_inputs"]["history129"], observed["adapter_inputs"]["history129"])
            self.assertFalse(np.shares_memory(expected["adapter_inputs"]["history129"], observed["adapter_inputs"]["history129"]))
        repeat[0]["adapter_inputs"]["current129"][0, 0] = -1.
        self.assertGreater(self.generated[0]["adapter_inputs"]["current129"][0, 0], 0.)
        manifest = cases.scenario_manifest()
        manifest["scenarios"][0]["generation_parameters"]["prior_peak_dn"][0] = -1.
        self.assertEqual(cases.scenario_manifest()["scenarios"][0]["generation_parameters"]["prior_peak_dn"][0], 30.)


if __name__ == "__main__":
    unittest.main()
