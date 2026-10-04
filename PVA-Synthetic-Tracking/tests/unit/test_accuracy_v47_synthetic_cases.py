"""Generator-only controls. No scoring, real media, cache, or journal access."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v47_synthetic_cases import (ADAPTER_KEYS, CASE_IDS, COUNTEREXAMPLE_IDS, TWIN_IDS,
                                         build_cases, scenario_manifest)


class V47SyntheticCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()
        cls.by_id = {c["case_id"]: c for c in cls.cases}
        cls.yy, cls.xx = np.indices((129, 129), dtype=float)
        cls.radius = np.maximum(abs(cls.xx-64), abs(cls.yy-64))
        cls.guard = (cls.radius >= 40) & (cls.radius <= 56)

    def test_exact_case_membership_shapes_and_metadata_separation(self):
        self.assertEqual(len(self.cases), 20)
        self.assertEqual(tuple(c["case_id"] for c in self.cases), CASE_IDS)
        self.assertEqual([c["case_id"] for c in scenario_manifest()["scenarios"]], list(CASE_IDS))
        for case in self.cases:
            a = case["adapter_inputs"]
            self.assertEqual(set(a), set(ADAPTER_KEYS))
            self.assertEqual(a["current129"].shape, (129, 129))
            self.assertEqual(a["history129"].shape, (8, 129, 129))
            self.assertEqual(a["current129"].dtype, np.float64)
            self.assertEqual(a["history129"].dtype, np.float64)
            self.assertFalse(np.isinf(a["current129"]).any())
            self.assertTrue(np.isfinite(a["history129"]).all())
            self.assertTrue(case["generator_truth"]["truth_is_not_an_adapter_argument"])
            self.assertFalse(case["generator_truth"]["physical_motion_and_class_certified"])

    def test_deterministic_images_truth_and_provenance(self):
        repeated = build_cases()
        for first, second in zip(self.cases, repeated):
            self.assertEqual(first["generator_truth"], second["generator_truth"])
            self.assertEqual(first["provenance"], second["provenance"])
            for name in ("history129", "current129"):
                np.testing.assert_array_equal(first["adapter_inputs"][name], second["adapter_inputs"][name])

    def test_forecast_independently_recomputed_from_last_four_priors(self):
        for case in self.cases:
            p, a = case["provenance"], case["adapter_inputs"]
            positions = np.asarray(p["reported_prior_world_centers_xy"])
            t = np.asarray(p["prior_times"][-4:])
            predicted = np.linalg.lstsq(np.column_stack((t, np.ones(4))), positions[-4:]-positions[-4], rcond=None)[0][1]+positions[-4]
            np.testing.assert_allclose(predicted, [64, 64], rtol=0, atol=1e-12)
            self.assertEqual(a["prior_centers_xy"], positions.tolist())
            self.assertEqual(a["predicted_offset_xy"], [0., 0.])
            self.assertFalse(p["forecast_uses_current_truth"])
            self.assertFalse(p["forecast_uses_current_image"])

    def test_true_current_gain_scales_background_not_source(self):
        background = 50+20*(self.xx >= 64)+8*np.sin(self.yy/12)
        source = 30*np.exp(-((self.xx-64)**2+(self.yy-64)**2)/2)
        for name, gain in (("ordinary_g1", 1), ("shared_g08", .8), ("shared_g12", 1.2), ("prior_only_counterexample_g2", 2)):
            current = self.by_id[name]["adapter_inputs"]["current129"]
            np.testing.assert_allclose(current, gain*background+source, rtol=0, atol=2e-14)
            self.assertEqual(self.by_id[name]["generator_truth"]["current_source_peak_dn"], 30)
        plane = 6+.08*(self.xx-64)-.04*(self.yy-64)
        np.testing.assert_allclose(self.by_id["shared_g12_plus_plane"]["adapter_inputs"]["current129"],
                                   1.2*background+source+plane, rtol=0, atol=5e-14)

    def test_identical_past_does_not_imply_current_gain(self):
        first, second = [self.by_id[key] for key in COUNTEREXAMPLE_IDS]
        np.testing.assert_array_equal(first["adapter_inputs"]["history129"], second["adapter_inputs"]["history129"])
        self.assertEqual(first["adapter_inputs"]["prior_centers_xy"], second["adapter_inputs"]["prior_centers_xy"])
        self.assertFalse(np.array_equal(first["adapter_inputs"]["current129"], second["adapter_inputs"]["current129"]))
        self.assertEqual([first["generator_truth"]["effective_background_gain_guard"], second["generator_truth"]["effective_background_gain_guard"]], [1, 2])

    def test_core_only_change_cannot_change_guard_observations(self):
        ordinary = self.by_id["ordinary_g1"]["adapter_inputs"]["current129"]
        changed = self.by_id["core_only_gain_change"]["adapter_inputs"]["current129"]
        np.testing.assert_array_equal(ordinary[self.guard], changed[self.guard])
        self.assertFalse(np.array_equal(ordinary[self.radius <= 12], changed[self.radius <= 12]))
        self.assertFalse(self.by_id["core_only_gain_change"]["generator_truth"]["gain_transfer_valid_in_declared_world"])

    def test_twins_are_identical_inputs_with_different_latent_explanations(self):
        first, second = [self.by_id[key] for key in TWIN_IDS]
        for key in ADAPTER_KEYS:
            np.testing.assert_array_equal(first["adapter_inputs"][key], second["adapter_inputs"][key])
        self.assertNotEqual(first["generator_truth"]["latent_global_photometric_gain"], second["generator_truth"]["latent_global_photometric_gain"])
        self.assertNotEqual(first["generator_truth"]["latent_additive_illumination"], second["generator_truth"]["latent_additive_illumination"])
        for case in (first, second):
            self.assertFalse(case["generator_truth"]["current_source_present"])
            self.assertFalse(case["generator_truth"]["guard_validity_or_transfer_certified_from_observations"])

    def test_absence_dark_and_dim_source_controls_are_independent_of_gain(self):
        background = self.by_id["background_only_g12"]["adapter_inputs"]["current129"]
        for name, peak in (("shared_g12", 30), ("dim_source_g12", 7.5), ("dark_source_g12", -30)):
            expected = peak*np.exp(-((self.xx-64)**2+(self.yy-64)**2)/2)
            np.testing.assert_allclose(self.by_id[name]["adapter_inputs"]["current129"]-background, expected, rtol=0, atol=2e-14)
        self.assertEqual(self.by_id["dark_source_g12"]["adapter_inputs"]["polarity"], "dark")
        for name in ("background_only_g12", "core_only_gain_change", *TWIN_IDS):
            self.assertEqual(self.by_id[name]["generator_truth"]["current_source_peak_dn"], 0)

    def test_flat_guard_has_nonaffine_core_but_affine_guard(self):
        a = self.by_id["flat_affine_guard"]["adapter_inputs"]
        affine = 50+.03*(self.xx-64)-.02*(self.yy-64)
        np.testing.assert_allclose(a["current129"][self.guard], 1.2*affine[self.guard], rtol=0, atol=1e-14)
        self.assertGreater(a["current129"][64, 64]-1.2*affine[64, 64], 30)

    def test_guard_light_current_scaling_and_blink_are_declared(self):
        shared = self.by_id["shared_g12"]["adapter_inputs"]
        persistent = self.by_id["persistent_guard_light_shared_gain"]["adapter_inputs"]
        blink = self.by_id["independently_blinking_guard_light"]["adapter_inputs"]
        self.assertAlmostEqual(persistent["current129"][64, 112]-shared["current129"][64, 112], 24)
        np.testing.assert_array_equal(persistent["history129"], blink["history129"])
        np.testing.assert_array_equal(blink["current129"], shared["current129"])
        self.assertTrue(self.by_id["independently_blinking_guard_light"]["generator_truth"]["known_guard_model_violation"])

    def test_new_and_broad_guard_contaminants_are_not_labeled_model_valid(self):
        ordinary = self.by_id["ordinary_g1"]["adapter_inputs"]["current129"]
        new = self.by_id["new_current_object_in_guard"]["adapter_inputs"]["current129"]
        self.assertAlmostEqual(new[64, 112]-ordinary[64, 112], 20)
        broad = self.by_id["broad_current_psf_reaches_guard"]["adapter_inputs"]["current129"]
        self.assertGreater(broad[64, 104]-ordinary[64, 104], 1)
        for name in ("new_current_object_in_guard", "broad_current_psf_reaches_guard"):
            self.assertTrue(self.by_id[name]["generator_truth"]["known_guard_model_violation"])

    def test_nan_is_one_predeclared_selected_guard_grid_pixel(self):
        counts = {c["case_id"]: int(np.isnan(c["adapter_inputs"]["current129"]).sum()) for c in self.cases}
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(counts["predeclared_guard_nan"], 1)
        self.assertTrue(np.isnan(self.by_id["predeclared_guard_nan"]["adapter_inputs"]["current129"][64, 112]))
        self.assertTrue(self.guard[64, 112])

    def test_correlated_extreme_errors_do_not_average_down(self):
        ordinary = self.by_id["shared_g12"]["adapter_inputs"]
        changed = self.by_id["correlated_guard_error_extremes"]["adapter_inputs"]
        dy = changed["current129"]-ordinary["current129"]
        dh = changed["history129"]-ordinary["history129"]
        # Separate floating additions can differ by one rounding bit at a
        # binade boundary. The realized absolute errors still obey .5DN.
        np.testing.assert_allclose(dh, np.broadcast_to(-dy, dh.shape), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(dh, np.broadcast_to(dh[0], dh.shape))
        self.assertLessEqual(float(abs(dh).max()), .5)
        self.assertEqual(float(abs(dy).max()), .5)
        self.assertTrue(np.all(dy[~self.guard] == 0))
        # The horizontal triplet (104,112,120),y64 reaches the full 2DN
        # weighted error, with the same historical error in all eight frames.
        weighted = dy[64, 104]-2*dy[64, 112]+dy[64, 120]
        self.assertEqual(abs(weighted), 2.)
        for i in range(8):
            self.assertEqual(dh[i, 64, 104]-2*dh[i, 64, 112]+dh[i, 64, 120], -weighted)


if __name__ == "__main__":
    unittest.main()
