"""Generated-only checks of descriptive denominators and exact display crops."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('patch_summary', Path(__file__).resolve().parents[1] / 'scripts/summarize_aot_image_patches.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def row(self, supported=True):
        return dict(source_kind='actual_residual_extremum', previous_index=0,
            selection_roles=['minimum', 'maximum'], saved_lk_displacement_xy=[3, 4],
            saved_candidate_residual_xy=[1, 2],
            comparison=dict(two_sizes_qualified_and_consistent=supported,
                verdict='agrees_with_saved_lk' if supported else 'ambiguous', saved_lk_error_by_size_px=[0, 0]),
            scales=[dict(template_size=size, available=True, qualified=supported,
                qualification=dict(ncc=supported, gap=supported), best_offset_xy=[3, 4]) for size in (33, 65)])

    def test_all_rows_remain_denominator_but_only_supported_form_descriptor(self):
        summary = module.describe([self.row(True), self.row(False)])
        self.assertEqual(summary['count'], 2)
        self.assertEqual(summary['mutually_supported'], 1)
        self.assertEqual(summary['verdict_counts'], {'agrees_with_saved_lk': 1, 'ambiguous': 1})
        scale = summary['scales']['33']
        self.assertEqual(scale['failure_counts_nonexclusive'], {'gap': 1, 'ncc': 1})
        self.assertAlmostEqual(scale['supported_subset_residual_to_original_global_candidate_px']['median'], np.sqrt(5))

    def test_empty_support_is_missing_not_zero_error(self):
        scale = module.describe([self.row(False)])['scales']['65']
        self.assertIsNone(scale['supported_subset_offset_component_range_xy'])
        self.assertIsNone(scale['supported_subset_residual_to_original_global_candidate_px']['median'])

    def test_role_overlap_does_not_duplicate_overall_denominator(self):
        summary = module.make_summary([self.row()], 'generated')
        self.assertEqual(summary['actual']['count'], 1)
        self.assertEqual(summary['roles']['minimum']['count'], 1)
        self.assertEqual(summary['roles']['maximum']['count'], 1)
        self.assertEqual(len(summary['pairs']), 8)
        self.assertEqual(summary['pairs'][1]['count'], 0)

    def test_unavailable_does_not_become_qualified(self):
        row = self.row(False)
        row['scales'][0] = dict(available=False, qualified=False, unavailable_reason='previous_template_outside_image')
        summary = module.describe([row])
        self.assertEqual(summary['scales']['33']['qualified'], 0)
        self.assertEqual(summary['scales']['33']['failure_counts_nonexclusive'], {'previous_template_outside_image': 1})

    def test_native_crop_half_up_no_padding(self):
        values = np.arange(100, dtype=np.uint8).reshape(10, 10)
        crop = module.native_crop(values, [4.5, 4.5], 3)
        np.testing.assert_array_equal(np.asarray(crop), values[4:7, 4:7])
        self.assertIsNone(module.native_crop(values, [0, 0], 3))
        self.assertIsNone(module.native_crop(values, None, 3))

    def test_exact_shift_display_has_no_wrap_and_preserves_pixels(self):
        values = np.arange(100, dtype=np.uint8).reshape(10, 10)
        for dx, dy in [(2, -1), (-2, 1)]:
            shifted = module.exact_shift(values, [dx, dy])
            for y in range(10):
                for x in range(10):
                    expected = values[y-dy, x-dx] if 0 <= y-dy < 10 and 0 <= x-dx < 10 else 128
                    self.assertEqual(shifted[y, x], expected)


if __name__ == '__main__':
    unittest.main()
