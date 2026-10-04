"""Independent direct-LS auditor checks; no original video files are read."""
import ast
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[2] / 'scripts/audit_accuracy_v36_context.py'
SPEC = importlib.util.spec_from_file_location('independent_v36_context_audit_under_test', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = audit.DirectLeastSquares()
        cls.y, cls.x = np.indices((25, 25), dtype=np.float64)
        cls.x, cls.y = cls.x - 12, cls.y - 12

    def test_no_feature_or_runner_imports(self):
        tree = ast.parse(PATH.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn('accuracy_v36_context', imports)
        self.assertNotIn('diagnose_accuracy_v36_context', imports)

    def test_bank_and_design_dimensions_are_frozen(self):
        self.assertEqual(len(self.model.points), 27)
        self.assertEqual(len(self.model.edges), 72)
        self.assertEqual(self.model.background.shape, (625, 6))
        self.assertTrue(all(m.shape == (625, 7) for m in self.model.point_models + self.model.edge_models))

    def test_exact_gaussian_coefficient_is_native_dn_not_unit_template_scale(self):
        patch = 15 + 0.2 * self.x + 8 * np.exp(-(self.x**2 + self.y**2) / 8)
        feature = self.model.measure(patch, 'bright')
        self.assertTrue(feature['informative'])
        self.assertAlmostEqual(feature['point_amplitude_dn'], 8, places=10)
        self.assertAlmostEqual(feature['point_gain_fraction'], 1, places=12)
        self.assertEqual(feature['point_sigma_px'], 2.0)
        self.assertEqual(feature['point_offset_xy'], [0.0, 0.0])
        self.assertGreater(feature['point_minus_edge_fraction'], 0)

    def test_dark_gaussian_reports_nonnegative_contrast_magnitude(self):
        patch = 100 - 9 * np.exp(-(self.x**2 + self.y**2) / 2)
        feature = self.model.measure(patch, 'dark')
        self.assertAlmostEqual(feature['point_amplitude_dn'], 9, places=10)
        self.assertAlmostEqual(feature['point_gain_fraction'], 1, places=12)

    def test_edge_contrast_remains_signed(self):
        patch = 100 - 12 * np.tanh(self.x / 2)
        feature = self.model.measure(patch, 'bright')
        self.assertAlmostEqual(feature['edge_amplitude_dn'], -12, places=10)
        self.assertAlmostEqual(feature['edge_gain_fraction'], 1, places=12)
        self.assertEqual(feature['edge_width_px'], 2.0)
        self.assertEqual(feature['edge_orientation_rad'], 0.0)
        self.assertLess(feature['point_minus_edge_fraction'], 0)

    def test_joint_eight_column_fit_recovers_original_point_coefficient(self):
        patch = 100 + 80 * np.tanh(self.x / 2) + 8 * np.exp(-(self.x**2 + self.y**2) / 8)
        feature = self.model.measure(patch, 'bright')
        self.assertEqual(feature['edge_width_px'], 2.0)
        self.assertEqual(feature['edge_offset_px'], 0.0)
        self.assertEqual(feature['edge_orientation_rad'], 0.0)
        self.assertEqual(feature['point_after_edge_sigma_px'], 2.0)
        self.assertAlmostEqual(feature['point_after_edge_amplitude_dn'], 8, places=9)
        self.assertAlmostEqual(feature['point_gain_after_edge_fraction'], 1, places=10)
        self.assertLess(feature['point_gain_fraction'], feature['point_gain_after_edge_fraction'])

    def test_quadratic_patch_is_uninformative_not_a_negative(self):
        patch = 50 + 2 * self.x + self.y + 0.1 * self.x**2 + 0.05 * self.x * self.y
        feature = self.model.measure(patch, 'bright')
        self.assertFalse(feature['informative'])
        self.assertEqual(feature['point_gain_fraction'], 0.0)
        self.assertEqual(feature['edge_gain_fraction'], 0.0)

    def test_dc_addition_does_not_change_evidence(self):
        patch = 20 + 8 * np.exp(-(self.x**2 + self.y**2) / 8)
        first = self.model.measure(patch, 'bright')
        second = self.model.measure(patch + 100, 'bright')
        for key in ('point_gain_fraction', 'edge_gain_fraction', 'point_after_edge_amplitude_dn'):
            self.assertAlmostEqual(first[key], second[key], places=10)

    def test_polarity_constraint_cannot_use_wrong_signed_coefficient(self):
        design = np.column_stack((self.model.background, self.model.points[0]))
        values = -10 * self.model.points[0]
        energy, _ = self.model.fit(self.model.background, values)
        _, gain, amplitude, result_energy = self.model.best([design], values, energy, sign=1)
        self.assertEqual(gain, 0.0)
        self.assertEqual(amplitude, 0.0)
        self.assertEqual(result_energy, energy)

    def test_malformed_patches_and_polarity_fail_closed(self):
        for patch in (np.zeros((24, 25)), np.full((25, 25), math.nan)):
            with self.assertRaises(ValueError):
                self.model.measure(patch, 'bright')
        with self.assertRaises(ValueError):
            self.model.measure(np.zeros((25, 25)), 'unknown')

    def test_numeric_tree_only_tolerates_predeclared_roundoff(self):
        audit.close_tree({'gain': 0.5 + 1e-12}, {'gain': 0.5}, 'small roundoff')
        for actual in ({'gain': 0.6}, {'gain': math.nan}, {'gain': True}, {'gain': 0.5, 'extra': 0}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                audit.close_tree(actual, {'gain': 0.5}, 'bad feature')
        with self.assertRaises(ValueError):
            audit.close_tree({'passed': 1}, {'passed': True}, 'boolean not a count')

    def test_input_registry_refuses_media_even_under_project_root(self):
        with self.assertRaises(ValueError):
            audit.Integrity().bind(audit.ROOT / 'chunk_0126.avi')
        with self.assertRaises(ValueError):
            audit.Integrity().bind('/tmp/out-of-scope.json')

    def test_archive_quantiles_include_zero_values_and_empty_groups(self):
        self.assertEqual(audit.quantiles([]), {'count': 0})
        result = audit.quantiles([0, 2, 4])
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['minimum'], 0.0)
        self.assertEqual(result['median'], 2.0)
        self.assertEqual(result['maximum'], 4.0)

    def test_any_valid_alternative_retains_sample_and_all_alternatives_are_preserved(self):
        group = dict(kind='dense', window='test', keys=['first', 'second'])
        observations = {'first': {'zero_margin_ablation_passed': False},
                        'second': {'zero_margin_ablation_passed': True}}
        result = audit.annotate_group(group, observations)
        self.assertTrue(result['baseline_hit'])
        self.assertTrue(result['diagnostic_hit'])
        self.assertFalse(result['new_miss'])
        self.assertEqual(result['keys'], ['first', 'second'])
        self.assertEqual(result['passing_keys'], ['second'])

    def test_missing_baseline_measurement_is_not_recovered_by_another_observation(self):
        result = audit.annotate_group({'keys': []}, {'unrelated': {'zero_margin_ablation_passed': True}})
        self.assertFalse(result['baseline_hit'])
        self.assertFalse(result['diagnostic_hit'])
        self.assertFalse(result['new_miss'])
        result = audit.annotate_group({'keys': ['miss']}, {'miss': {'zero_margin_ablation_passed': False}})
        self.assertTrue(result['new_miss'])

    def test_unknown_or_duplicated_alternative_cannot_silently_change_denominator(self):
        for group in ({'keys': ['absent']}, {'keys': ['known', 'known']}):
            with self.subTest(group=group), self.assertRaises(ValueError):
                audit.annotate_group(group, {'known': {'zero_margin_ablation_passed': True}})


if __name__ == '__main__':
    unittest.main()
