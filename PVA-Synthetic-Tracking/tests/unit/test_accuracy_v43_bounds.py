"""Synthetic deterministic component-error bounds; no media or real scores."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import accuracy_v42_localized as legacy
import accuracy_v43_bounds as bounds
import accuracy_v43_components as protected_core


Y, X = np.indices((129, 129))


def fixture(polarity='bright', amplitude=80.0, fixed_amplitude=0.0, offset=(.3, -.2)):
    sign = 1.0 if polarity == 'bright' else -1.0
    centres = [[29 + 4*i, 64] for i in range(8)]
    background = np.full((129, 129), 100.0)
    background += sign * fixed_amplitude * np.exp(-((X-80)**2 + (Y-55)**2) / 2)
    amplitudes = [amplitude]*8 if np.isscalar(amplitude) else amplitude
    history = np.stack([background + sign * amp * np.exp(-((X-x)**2 + (Y-y)**2) / 2)
                        for amp, (x, y) in zip(amplitudes, centres)])
    return history, centres, list(offset), polarity


def components_and_bounds(args, protected=False):
    producer = protected_core.prepare_components if protected else legacy.prepare_components
    components = producer(*args)
    return components, bounds.component_bounds(*args, components, protected=protected)


class NormalizationTests(unittest.TestCase):
    def test_random_adversarial_perturbations_are_contained(self):
        rng = np.random.default_rng(901)
        vector = rng.uniform(20, 40, (17, 17))
        error = rng.uniform(.1, 1, (17, 17))
        bound, metadata = bounds.normalization_bound(vector, error)
        self.assertTrue(metadata['available'])
        nominal = vector / np.linalg.norm(vector)
        for _ in range(100):
            perturbed = vector + rng.choice([-1., 1.], vector.shape) * error
            difference = np.abs(perturbed / np.linalg.norm(perturbed) - nominal)
            self.assertTrue(np.all(difference <= bound + 1e-14))

    def test_delta_reaching_norm_is_unknown(self):
        bound, metadata = bounds.normalization_bound(np.ones((2, 2)), np.ones((2, 2)))
        self.assertIsNone(bound)
        self.assertFalse(metadata['available'])
        self.assertEqual(metadata['norm_lower_bound'], 0)

    def test_original_acceptance_threshold_cannot_be_crossed(self):
        bound, metadata = bounds.normalization_bound(np.array([1.1e-6]), np.array([.2e-6]))
        self.assertIsNone(bound)
        self.assertLessEqual(metadata['norm_lower_bound'], 1e-6)

    def test_zero_error_and_nan_support_preserved(self):
        vector = np.array([3., 4., np.nan])
        bound, metadata = bounds.normalization_bound(vector, np.array([0., 0., np.nan]))
        np.testing.assert_array_equal(bound[:2], [0., 0.])
        self.assertTrue(np.isnan(bound[2]))
        self.assertEqual(metadata['nominal_norm'], 5.)

    def test_invalid_or_missing_bound_support_fails_closed(self):
        for vector, error in (([1, 2], [1]), ([1, 2], [-1, 1]), ([1, 2], [np.nan, 1]),
                              ([np.inf, 1], [1, 1]), ([1+1j, 2], [1, 1])):
            with self.assertRaises(ValueError):
                bounds.normalization_bound(vector, error)

    def test_scaling_nominal_and_error_preserves_bound(self):
        a, _ = bounds.normalization_bound(np.array([30., 40.]), np.array([1., 2.]))
        b, _ = bounds.normalization_bound(np.array([300., 400.]), np.array([10., 20.]))
        np.testing.assert_allclose(a, b, atol=1e-15, rtol=1e-15)


class ComponentTests(unittest.TestCase):
    def test_legacy_moving_and_fixed_components_have_complete_bounds(self):
        args = fixture(fixed_amplitude=80.)
        components, result = components_and_bounds(args)
        self.assertTrue(result['available'], result['reasons'])
        self.assertGreater(len(components['fixed_templates']), 0)
        self.assertEqual(result['fixed_template_bounds'].shape, components['fixed_templates'].shape)
        for name, nominal in (('background_bound129', components['background']),
                              ('moving_template_bound129', components['moving_template'])):
            np.testing.assert_array_equal(np.isfinite(result[name]), np.isfinite(nominal))
        np.testing.assert_array_equal(result['background_bound129'], np.full((129, 129), .5))
        json.dumps(result['metadata'], allow_nan=False)

    def test_dark_polarity_has_same_error_envelope(self):
        _, bright = components_and_bounds(fixture('bright'))
        _, dark = components_and_bounds(fixture('dark'))
        self.assertTrue(bright['available'] and dark['available'])
        np.testing.assert_allclose(bright['moving_template_bound129'], dark['moving_template_bound129'], atol=1e-12)

    def test_protected_history_schema_and_legacy_moving_bound_agree(self):
        args = fixture(fixed_amplitude=80.)
        _, old = components_and_bounds(args)
        components, result = components_and_bounds(args, protected=True)
        self.assertTrue(result['available'], result['reasons'])
        np.testing.assert_allclose(result['moving_template_bound129'], old['moving_template_bound129'], equal_nan=True)
        self.assertEqual(len(result['metadata']['fixed']), len(components['fixed_templates']))
        self.assertTrue(result['metadata']['fixed_used_stamp_membership'])
        self.assertIn('Not bounded', result['metadata']['omitted_stamp_selection_uncertainty'])

    def test_correlated_history_perturbations_contained_without_averaging(self):
        args = fixture()
        original, result = components_and_bounds(args)
        self.assertTrue(result['available'])
        rng = np.random.default_rng(18)
        # Same perturbation across every history frame, then independent signs.
        for correlated in (True, False):
            for _ in range(10):
                shape = (1, 129, 129) if correlated else (8, 129, 129)
                perturbed_history = args[0] + rng.choice([-.5, .5], shape)
                changed = legacy.prepare_components(perturbed_history, *args[1:])
                self.assertEqual(original['metadata']['usable_moving_stamp_indices'],
                                 changed['metadata']['usable_moving_stamp_indices'])
                self.assertTrue(np.all(np.abs(changed['background']-original['background']) <= .5 + 1e-12))
                finite = np.isfinite(original['moving_template'])
                error = np.abs(changed['moving_template'] - original['moving_template'])
                self.assertTrue(np.all(error[finite] <= result['moving_template_bound129'][finite] + 1e-12))

    def test_one_weak_but_used_stamp_cannot_be_omitted(self):
        args = fixture(amplitude=[80., 80., .01, 80., 80., 80., 80., 80.])
        components, result = components_and_bounds(args)
        self.assertEqual(components['metadata']['usable_moving_stamp_indices'], list(range(8)))
        self.assertFalse(result['available'])
        self.assertIsNone(result['moving_template_bound129'])
        self.assertEqual(result['metadata']['moving']['used_history_indices'], list(range(8)))
        self.assertFalse(result['metadata']['moving']['uncertain_used_stamp_omitted'])
        self.assertEqual(result['metadata']['moving']['stamp_bounds'][-1]['history_index'], 2)

    def test_weak_fixed_component_stays_unknown_not_removed(self):
        args = fixture(fixed_amplitude=2.)
        components, result = components_and_bounds(args)
        self.assertGreater(len(components['fixed_templates']), 0)
        self.assertFalse(result['available'])
        self.assertEqual(len(result['fixed_template_bounds']), len(components['fixed_templates']))
        self.assertTrue(any(reason.startswith('fixed_component_uncertainty_unavailable') for reason in result['reasons']))

    def test_missing_nominal_moving_template_remains_unknown(self):
        args = list(fixture())
        args[1] = [[64, 64]]*8
        components, result = components_and_bounds(args)
        self.assertIsNone(components['moving_template'])
        self.assertFalse(result['available'])
        self.assertIsNone(result['moving_template_bound129'])

    def test_subpixel_placement_uses_positive_bilinear_weights(self):
        args = fixture(offset=(.3, -.2))
        components, result = components_and_bounds(args)
        provenance = bounds._legacy_history(args[0], args[1], components['background'], 1., components['metadata'])
        stamp_bound, _ = bounds._combined_bound(provenance['moving'], components['moving_stamp'], 1.)
        np.testing.assert_array_equal(result['moving_template_bound129'],
                                      legacy._place(stamp_bound, np.array([64.3, 63.8])))
        self.assertEqual(result['moving_template_bound129'][0, 0], 0.)

    def test_protected_missing_or_inconsistent_provenance_fails_closed(self):
        args = fixture(fixed_amplitude=80.)
        components = protected_core.prepare_components(*args)
        missing = copy.deepcopy(components)
        del missing['template_history']
        with self.assertRaises(ValueError):
            bounds.component_bounds(*args, missing, protected=True)
        changed = copy.deepcopy(components)
        changed['template_history']['moving']['raw_weighted_energy_dn2'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'energy'):
            bounds.component_bounds(*args, changed, protected=True)
        changed = copy.deepcopy(components)
        changed['template_history']['fixed'].pop()
        with self.assertRaisesRegex(ValueError, 'omits'):
            bounds.component_bounds(*args, changed, protected=True)

    def test_mutated_nominal_template_cannot_reuse_bounds(self):
        args = fixture()
        components = legacy.prepare_components(*args)
        components['moving_template'][64, 64] += .01
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            bounds.component_bounds(*args, components)

    def test_no_input_mutation(self):
        args = fixture(fixed_amplitude=80.)
        components = protected_core.prepare_components(*args)
        history = args[0].copy()
        moving = components['moving_template'].copy()
        fixed = components['fixed_templates'].copy()
        stamps = components['template_history']['moving']['normalized_stamps'].copy()
        bounds.component_bounds(*args, components, protected=True)
        np.testing.assert_array_equal(args[0], history)
        np.testing.assert_array_equal(components['moving_template'], moving)
        np.testing.assert_array_equal(components['fixed_templates'], fixed)
        np.testing.assert_array_equal(components['template_history']['moving']['normalized_stamps'], stamps)


if __name__ == '__main__':
    unittest.main()
