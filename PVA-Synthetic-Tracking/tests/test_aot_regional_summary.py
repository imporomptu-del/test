"""Generated accounting tests; no experiment inputs accessed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('regional_summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_aot_regional_motion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RegionalSummaryTests(unittest.TestCase):
    def predictions(self):
        return {
            'global_translation': dict(available=[True, True, True], predicted_displacement_xy=[[1, 0], [2, 0], [3, 0]]),
            'local_translation': dict(available=[True, False, True], predicted_displacement_xy=[[0, 0], None, [1, 0]]),
            'local_affine': dict(available=[True, True, False], predicted_displacement_xy=[[.5, 0], [0, 0], None]),
        }

    def test_pairwise_comparisons_use_identical_points_not_arm_survivor_means(self):
        result = module.cohort_summary([0, 1, 2], {'saved_lk': [[0, 0]]*3}, self.predictions())
        self.assertEqual(result['arms']['global_translation']['available_count'], 3)
        self.assertEqual(result['arms']['local_translation']['unavailable_count'], 1)
        paired = result['pairwise']['global_translation__local_translation']
        self.assertEqual(paired['common_query_indices'], [0, 2])
        self.assertEqual(paired['references']['saved_lk']['left']['median'], 2)
        self.assertEqual(paired['references']['saved_lk']['right']['median'], .5)
        self.assertEqual(result['all_three_common']['query_indices'], [0])

    def test_missing_predictions_stay_unscored_not_zero_error(self):
        result = module.cohort_summary([1], {'saved_lk': [[2, 0]]}, self.predictions())
        self.assertIsNone(result['arms']['local_translation']['conditional_errors']['saved_lk']['median'])
        self.assertEqual(result['all_three_common']['count'], 0)
        self.assertEqual(result['all_three_common']['missing_count'], 1)

    def test_references_stay_separate_and_paired_gains_can_be_negative(self):
        result = module.cohort_summary([0], {'ncc33': [[0, 0]], 'ncc65': [[1, 0]]}, self.predictions())
        pair = result['pairwise']['global_translation__local_translation']['references']
        self.assertEqual(pair['ncc33']['paired_error_reduction_left_minus_right']['median'], 1)
        self.assertEqual(pair['ncc65']['paired_error_reduction_left_minus_right']['median'], -1)

    def test_empty_cohort_has_explicit_zero_count_and_missing_statistics(self):
        result = module.cohort_summary([], {'ncc33': [], 'ncc65': []}, self.predictions())
        self.assertEqual(result['total_count'], 0)
        self.assertIsNone(result['all_three_common']['references']['ncc33']['local_affine']['maximum'])

    def test_duplicate_query_indices_or_false_availability_rejected(self):
        with self.assertRaises(AssertionError):
            module.cohort_summary([0, 0], {'saved_lk': [[0, 0]]*2}, self.predictions())
        predictions = self.predictions()
        predictions['local_translation']['available'][1] = True
        with self.assertRaises(AssertionError):
            module.cohort_summary([0], {'saved_lk': [[0, 0]]}, predictions)


if __name__ == '__main__':
    unittest.main()
