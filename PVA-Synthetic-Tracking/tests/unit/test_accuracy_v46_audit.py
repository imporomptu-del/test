from pathlib import Path
from copy import deepcopy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v46_synthetic import (closed_form_forecast, expected_patch_geometry,
    array_fingerprint, rational_normal_equation_forecast, audit_geometry,
    audit_evidence, evidence_counts, ARMS)
from accuracy_v46_synthetic_cases import build_cases, scenario_manifest


class V46IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()
        cls.manifest = scenario_manifest()

    def test_centered_scalar_forecast_for_independent_true_trajectory(self):
        reported = [[32+4*i, 64] for i in range(8)]
        forecast, indices = closed_form_forecast(range(-8, 0), reported)
        np.testing.assert_array_equal(forecast, [64., 64.])
        self.assertEqual(indices, [4, 5, 6, 7])

    def test_missing_measurements_keep_original_timestamps(self):
        reported = [[32+4*i, 64] if i in (0, 2, 5, 7) else None for i in range(8)]
        forecast, indices = closed_form_forecast(range(-8, 0), reported)
        np.testing.assert_allclose(forecast, [64., 64.], rtol=0, atol=1e-14)
        self.assertEqual(indices, [0, 2, 5, 7])

    def test_less_than_four_has_no_truth_fallback(self):
        with self.assertRaisesRegex(ValueError, "Four reported"):
            closed_form_forecast(range(-8, 0), [[0, 0]]*3+[None]*5)

    def test_current_or_retimed_timestamps_rejected(self):
        for times in (range(-7, 1), [-1]*8, [np.nan]*8):
            with self.assertRaises(ValueError):
                closed_form_forecast(times, [[0, 0]]*8)

    def test_bias_affects_forecast_not_truth_and_crop_offset_is_fractional(self):
        positions = [[32+4*i+.6, 63.6] for i in range(8)]
        forecast, _ = closed_form_forecast(range(-8, 0), positions)
        patch = expected_patch_geometry(forecast, positions)
        np.testing.assert_allclose(forecast, [64.6, 63.6])
        self.assertEqual(patch["crop_center_world_xy"], [65., 64.])
        self.assertEqual(patch["crop_origin_world_xy"], [1., 0.])
        np.testing.assert_allclose(patch["predicted_offset_xy"], [-.4, -.4])
        np.testing.assert_allclose(patch["prior_centers_xy"][0], [31.6, 63.6])

    def test_fingerprint_binds_dtype_shape_and_values(self):
        array = np.arange(4, dtype=float)
        self.assertNotEqual(array_fingerprint(array), array_fingerprint(array.astype(int)))
        self.assertNotEqual(array_fingerprint(array), array_fingerprint(array.reshape(2, 2)))
        self.assertEqual(array_fingerprint(array), array_fingerprint(array.copy()))

    def test_rational_half_pixel_forecast_preserves_crop_boundary(self):
        reported = [[32+4*i+.5, 64] for i in range(8)]
        forecast, indices, weights = rational_normal_equation_forecast(range(-8, 0), reported)
        self.assertEqual(forecast, [64.5, 64.])
        self.assertEqual(indices, [4, 5, 6, 7])
        self.assertEqual(weights, [-.5, 0., .5, 1.])
        self.assertEqual(expected_patch_geometry(forecast, reported)["crop_center_world_xy"], [65., 64.])

    def test_all_unscored_cases_geometry_and_independent_world_render(self):
        result = audit_geometry(self.cases, self.manifest)
        self.assertTrue(result["passed"])
        self.assertEqual(result["images_reconstructed"], 252)
        self.assertEqual(result["current_only_group_cases"], 10)
        self.assertLessEqual(result["maximum_render_abs_error"], 2e-12)

    def test_rejects_current_truth_forecast_substitution(self):
        cases = deepcopy(self.cases)
        cases[8]["provenance"]["predicted_world_xy"] = [70., 64.]
        with self.assertRaisesRegex(ValueError, "beyond prior measurements"):
            audit_geometry(cases, self.manifest)

    def test_rejects_retimed_missing_observations(self):
        cases = deepcopy(self.cases)
        cases[16]["provenance"]["ols_times"] = [-4, -3, -2, -1]
        with self.assertRaisesRegex(ValueError, "retimed"):
            audit_geometry(cases, self.manifest)

    def test_rejects_measurement_bias_copied_into_truth(self):
        cases = deepcopy(self.cases)
        cases[2]["generator_truth"]["apparent_point_prior_world_centers_xy"][0][0] += .5
        with self.assertRaisesRegex(ValueError, "world truth"):
            audit_geometry(cases, self.manifest)

    def test_rejects_synthetic_image_corruption(self):
        for key in ("history129", "current129"):
            cases = deepcopy(self.cases)
            cases[0]["adapter_inputs"][key].flat[0] += .001
            with self.assertRaisesRegex(ValueError, "rendering mismatch"):
                audit_geometry(cases, self.manifest)

    def test_rejects_truth_input_or_case_omission(self):
        cases = deepcopy(self.cases)
        cases[0]["adapter_inputs"]["truth"] = {"moving": True}
        with self.assertRaisesRegex(ValueError, "truth or undeclared"):
            audit_geometry(cases, self.manifest)
        with self.assertRaisesRegex(ValueError, "All 28"):
            audit_geometry(self.cases[:-1], self.manifest)

    @staticmethod
    def unavailable_record():
        raw = dict(available=False, numerical_contrast=None, reasons=["insufficient_actual_prior_centers"],
            motion_status="unknown", physical_class="unknown", is_motion_or_classification_gate=False,
            synthetic_only=True, production_changed=False,
            learned_design_sha256=None, common_support_sha256=None, common_support_count=0,
            components=None, component_bounds=None, conditional_on=["fixed_geometry"], uncertainty_excludes=["physical_identity"],
            ambiguity_reasons=["physical_motion_and_class_not_certified_by_source_contrast"],
            prior_context=dict(templates_use_prior_images_only=True, minimum_actual_prior_centers=5,
                               supplied_forecast_offset_independently_verified=False))
        return dict(case_id="unknown_control", arms={arm: dict(arm=arm, quantity=q, bound_version=b,
            legacy_numerical_contrast_field_contains=q, motion_status="unknown", physical_class="unknown",
            is_motion_or_classification_gate=False, raw_adapter_result=deepcopy(raw)) for arm, (q, b) in ARMS.items()})

    def test_unknown_is_retained_in_denominator_not_negative(self):
        record = self.unavailable_record()
        self.assertEqual(len(audit_evidence([record])), 1)
        result = evidence_counts([record], "presence_box_bounds")["counts"]
        self.assertEqual(result["states"], 1)
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["negative"], 0)

    def test_evidence_rejects_missing_arm_and_physical_promotion(self):
        record = self.unavailable_record()
        del record["arms"]["presence_box_bounds"]
        with self.assertRaisesRegex(ValueError, "arm was omitted"):
            audit_evidence([record])
        record = self.unavailable_record()
        record["arms"]["presence_box_bounds"]["physical_class"] = "airborne"
        with self.assertRaisesRegex(ValueError, "motion/class"):
            audit_evidence([record])

    def test_evidence_rejects_changed_nominal_support(self):
        record = self.unavailable_record()
        record["arms"]["presence_box_bounds"]["raw_adapter_result"]["common_support_count"] = 625
        with self.assertRaisesRegex(ValueError, "nominal design/support"):
            audit_evidence([record])


if __name__ == "__main__":
    unittest.main()
