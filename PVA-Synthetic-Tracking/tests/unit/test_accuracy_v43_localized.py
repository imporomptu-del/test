import concurrent.futures
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import accuracy_v42_localized as v42
import accuracy_v43_localized as core


def fixture():
    y, x = np.indices((129, 129))
    background = 50+20*(x>=64)+8*np.sin(y/12)
    centers = [[29+4*i,64] for i in range(8)]
    def point(cx, cy): return 30*np.exp(-((x-cx)**2+(y-cy)**2)/2)
    return background+point(64,64), np.stack([background+point(*p) for p in centers]), centers, [0,0], 'bright'


class LocalizedTests(unittest.TestCase):
    def test_baseline_is_exact_frozen_algorithm(self):
        args = fixture()
        self.assertEqual(core.evaluate_arm(*args, 'baseline'), v42.evaluate_localized(*args))

    def test_isolated_component_injection_leaves_reference_globals_untouched(self):
        args = fixture(); components = v42.prepare_components(*args[1:])
        original = v42.evaluate_localized.__globals__['prepare_components']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda unused: core.legacy_with_components(*args, components), range(4)))
        self.assertTrue(all(value == v42.evaluate_localized(*args) for value in values))
        self.assertIs(v42.evaluate_localized.__globals__['prepare_components'], original)

    def test_protected_only_cannot_erase_unknown_alternatives(self):
        result = core.evaluate_arm(*fixture(), 'protected_only')
        self.assertTrue(result['ambiguous'])
        self.assertIn('fixed_alternative_prior_coverage_incomplete', result['ambiguity_reasons'])
        json.dumps(result, allow_nan=False)

    def test_ordinary_synthetic_point_documents_conservative_abstention_cost(self):
        # An intentionally stringent conditional sensitivity budget can abstain
        # even on a clear synthetic signal. Never hide this as a target negative.
        result = core.evaluate_arm(*fixture(), 'stable_only')
        self.assertFalse(result['available'])
        self.assertTrue(any(r.startswith('annulus:') for r in result['reasons']))
        self.assertIsNone(result['mse_stationary'])
        self.assertIsNone(result['mse_augmented'])
        self.assertNotIn('reject', result)

    def test_current_core_values_cannot_change_annulus_certificate(self):
        args = list(fixture())
        first = core.evaluate_arm(*args, 'stable_only')
        args[0] = args[0].copy(); args[0][52:77,52:77] += 100
        second = core.evaluate_arm(*args, 'stable_only')
        self.assertEqual(first['stability']['annulus'], second['stability']['annulus'])
        self.assertEqual(first['common_support_sha256'], second['common_support_sha256'])

    def test_unavailable_annulus_has_no_operative_comparison(self):
        failure = dict(available=False, reasons=['unsupported'], diagnostics={}, coefficients=None,
                       prediction=None, prediction_bound=None)
        with mock.patch.object(core, 'supported_fit', return_value=failure), \
             mock.patch.object(core, 'component_bounds') as bounds:
            result = core.evaluate_arm(*fixture(), 'combined')
        self.assertEqual(result['reasons'], ['annulus:unsupported'])
        self.assertIsNone(result['advantage_stationary_minus_augmented'])
        bounds.assert_not_called()

    def test_successful_integration_and_partial_core_failure_no_partial_mse(self):
        def fit(a, target, b, sigma_dn, nonnegative_last=False, **unused):
            coefficients = np.linalg.lstsq(a,target,rcond=None)[0]
            if nonnegative_last and coefficients[-1] < 0:
                coefficients = np.r_[np.linalg.lstsq(a[:,:-1],target,rcond=None)[0], 0]
            return dict(available=True,reasons=[],diagnostics={'test_stub':True},
                        coefficients=coefficients,prediction=b@coefficients,prediction_bound=np.zeros(len(b)))
        def bounds(*args, **kwargs):
            components = args[4]
            return dict(available=True,reasons=[],metadata={'test_stub':True},
                background_bound129=np.zeros((129,129)), moving_template_bound129=np.zeros((129,129)),
                fixed_template_bounds=np.zeros_like(components['fixed_templates']))
        with mock.patch.object(core,'supported_fit',side_effect=fit), \
             mock.patch.object(core,'component_bounds',side_effect=bounds):
            result = core.evaluate_arm(*fixture(),'stable_only')
        self.assertTrue(result['available'])
        self.assertEqual(len(result['folds']),2)
        self.assertLess(result['mse_augmented'],result['mse_stationary'])
        self.assertEqual(result['stability']['core_response_bound_dn'],.5)
        calls = 0
        def fail_second_fold(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                return dict(available=False,reasons=['bad'],diagnostics={},coefficients=None,
                            prediction=None,prediction_bound=None)
            return fit(*args, **kwargs)
        with mock.patch.object(core,'supported_fit',side_effect=fail_second_fold), \
             mock.patch.object(core,'component_bounds',side_effect=bounds):
            result = core.evaluate_arm(*fixture(),'stable_only')
        self.assertFalse(result['available']); self.assertEqual(calls,5)
        self.assertEqual(result['folds'],[])
        self.assertIsNone(result['mse_stationary'])

    def test_invalid_arm_fails_before_inference(self):
        with self.assertRaises(ValueError): core.evaluate_arm(*fixture(),'best_arm')


if __name__ == '__main__': unittest.main()
