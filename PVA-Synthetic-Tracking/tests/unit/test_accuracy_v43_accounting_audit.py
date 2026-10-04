"""Synthetic contracts for the post-score independent accounting auditor."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import audit_accuracy_v43_accounting as audit


def value(available=True, protected=False):
    reasons = ['incomplete_prior_fixed_support'] if protected else []
    return dict(available=available, ambiguous=bool(reasons),
        reasons=[] if available else ['unsupported'], ambiguity_reasons=list(reasons),
        components=dict(anchor_dictionary_truncated=False, persistent_anchor_overlaps_prediction=False,
                        fixed_learning_ambiguity_reasons=list(reasons)),
        mse_stationary=5. if available else None, mse_augmented=3. if available else None,
        advantage_stationary_minus_augmented=2. if available else None,
        folds=[{}, {}] if available else [], common_support_count=64,
        fold_support_counts=[32, 32], common_support_sha256='same')


class AccountingAuditTests(unittest.TestCase):
    def test_unavailable_never_has_operative_mse_or_partial_folds(self):
        audit.check_arm(value(False))
        for changed in ({'mse_stationary': 0.}, {'folds': [{}]}, {'advantage_stationary_minus_augmented': 0.}):
            current = value(False); current.update(changed)
            with self.assertRaises(AssertionError): audit.check_arm(current)

    def test_protected_ambiguity_cannot_disappear(self):
        current = value(protected=True)
        audit.check_arm(current)
        current['ambiguity_reasons'] = []; current['ambiguous'] = False
        with self.assertRaises(AssertionError): audit.check_arm(current)

    def test_hidden_and_json_nonfinite_values_fail(self):
        current = value(); current['diagnostic'] = {'nested': [float('nan')]}
        with self.assertRaises(AssertionError): audit.check_arm(current)
        with self.assertRaises(ValueError):
            json.loads('{"x":NaN}', parse_constant=audit.invalid_constant)

    def test_paired_support_accounting_does_not_cherry_pick_changed_support(self):
        a, b, c = value(), value(), value(False)
        b['common_support_sha256'] = 'different'
        rows = [dict(arms={'baseline': a, 'combined': copy.deepcopy(a)}),
                dict(arms={'baseline': a, 'combined': b}),
                dict(arms={'baseline': a, 'combined': c})]
        result = audit.independent_pair(rows, 'combined')
        self.assertEqual(result['counts']['both_available_same_support'], 1)
        self.assertEqual(result['counts']['both_available_different_support'], 1)
        self.assertEqual(result['counts']['available_1_0'], 1)
        self.assertEqual(result['same_support_advantage_change_quantiles'], [0., 0., 0.])


if __name__ == '__main__': unittest.main()
