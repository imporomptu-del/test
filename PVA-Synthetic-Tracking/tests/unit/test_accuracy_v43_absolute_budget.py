"""Synthetic regression evidence for the V43 absolute-background budget."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from analyze_accuracy_v43_absolute_budget import diagnose, synthetic_fixture, write_diagnosis


class AbsoluteBudgetDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = diagnose()

    def test_fixture_exactly_matches_predeclared_ordinary_positive_control(self):
        spec = importlib.util.spec_from_file_location("v43_original_fixture", ROOT/"tests/unit/test_accuracy_v43_localized.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        actual, original = synthetic_fixture(), module.fixture()
        for index in (0, 1):
            np.testing.assert_array_equal(actual[index], original[index])
        self.assertEqual(actual[2:], original[2:])

    def test_native_input_construction_is_admissible_and_attains_one_point_five_dn(self):
        value = self.report["admissible_input_construction"]
        self.assertLessEqual(value["maximum_current_input_change_dn"], .5)
        self.assertLessEqual(value["maximum_prior_input_change_dn"], .5)
        self.assertLess(value["background_median_translation_error_dn"], 1e-12)
        self.assertTrue(value["background_support_unchanged"])
        self.assertTrue(value["gain_remains_nonnegative"])
        self.assertAlmostEqual(value["actual_prediction_shift_min_dn"], 1.5, places=11)
        self.assertAlmostEqual(value["actual_prediction_shift_max_dn"], 1.5, places=11)
        self.assertTrue(value["violates_one_dn_absolute_budget"])

    def test_response_only_vertex_exactly_attains_row_l1_bound(self):
        value = self.report["exact_response_only_extremum"]
        self.assertAlmostEqual(value["analytic_maximum_shift_dn"], value["attained_prediction_shift_dn"], places=11)
        self.assertGreater(value["attained_prediction_shift_dn"], 1.0)
        self.assertLessEqual(value["maximum_training_response_change_dn"], .5)
        self.assertTrue(value["gain_remains_nonnegative"])

    def test_conservative_bound_is_loose_without_claiming_input_unobservable(self):
        value = self.report["v43_conservative_screen"]
        self.assertFalse(value["available"])
        self.assertGreater(value["robust_minimum_singular_value"], 0)
        self.assertLess(value["scaled_train_condition_number"], 30)
        self.assertGreater(value["maximum_prediction_bound_dn"], 1.5)
        self.assertAlmostEqual(value["maximum_prediction_bound_dn"],
            value["response_term_at_worst_pixel_dn"]+value["design_term_at_worst_pixel_dn"], places=10)

    def test_fixed_design_contrast_cancels_common_offset_but_does_not_certify_motion(self):
        value = self.report["fixed_design_contrast_illustration"]
        self.assertGreater(value["nominal_unit_template_contrast_coefficient"], value["response_only_absolute_coefficient_bound"])
        self.assertLess(abs(value["coefficient_change_from_uniform_1p5_dn"]), 1e-9)
        self.assertLess(value["nuisance_orthogonality_norm"], 1e-8)
        self.assertFalse(value["design_uncertainty_certified"])
        self.assertFalse(value["moving_versus_fixed_light_discrimination_certified"])
        self.assertFalse(self.report["is_new_classifier_or_gate"])
        self.assertFalse(self.report["uses_real_media_journals_or_scores"])

    def test_report_is_finite_reproducible_and_output_never_overwrites(self):
        self.assertEqual(self.report, diagnose())
        json.dumps(self.report, allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"report.json"
            saved = write_diagnosis(path)
            self.assertEqual(json.loads(path.read_text()), saved)
            self.assertEqual(len(saved["inputs_sha256"]), 8)
            with self.assertRaises(FileExistsError):
                write_diagnosis(path)

    def test_frozen_output_directory_is_protected(self):
        with self.assertRaises(ValueError):
            write_diagnosis(ROOT/"results/tiny_target/accuracy_v43_20260925/stability_01/diagnosis.json")


if __name__ == "__main__":
    unittest.main()
