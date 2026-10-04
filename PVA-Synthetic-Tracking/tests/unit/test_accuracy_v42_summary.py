"""Post-score descriptive summaries must not become classifier claims."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import summarize_accuracy_v42_localized as summary


def record(available=True, ambiguous=False, gain=1.0):
    return dict(clip='0029', frame_index=100, track_id='bright:1', qualified_moving=True,
        reference_samples=[0], grid_windows=['grid'], predicted_to_actual_distance_px=.5,
        geometry={'available': True, 'reasons': [], 'geometry': {'measured_prior_count': 8}},
        localized={'available': available, 'ambiguous': ambiguous,
            'reasons': [] if available else ['template'],
            'ambiguity_reasons': ['dictionary'] if ambiguous else [],
            'mse_stationary': 2 if available else None,
            'mse_augmented': 2-gain if available else None,
            'advantage_stationary_minus_augmented': gain if available else None,
            'components': {'persistent_anchor_count_before_cap': 3}})


class SummaryTests(unittest.TestCase):
    def test_score_availability_and_ambiguity_are_separate(self):
        records = [record(), record(ambiguous=True), record(available=False)]
        missing = record(); missing['geometry']['available'] = False
        missing['localized'] = None; missing['geometry']['geometry']['measured_prior_count'] = 3
        records.append(missing)
        result = summary.group(records)
        self.assertEqual(result['states'], 4)
        self.assertEqual(result['scores_available'], 2)
        self.assertEqual(result['scores_without_recorded_ambiguity'], 1)
        self.assertEqual(result['scores_with_recorded_ambiguity'], 1)
        self.assertEqual(result['geometry_but_core_unavailable'], 1)
        self.assertEqual(result['insufficient_prior_history'], 1)

    def test_empty_group_has_no_fabricated_distribution(self):
        result = summary.group([])
        self.assertIsNone(result['advantage_dn2'])
        self.assertEqual(result['states'], 0)

    def test_sign_reporting_preserves_negative_and_near_zero(self):
        result = summary.group([record(gain=1), record(gain=-1), record(gain=1e-12)])
        self.assertEqual(result['score_sign_at_reporting_epsilon_1e_9'],
                         {'positive': 1, 'negative': 1, 'near_zero': 1})

    def test_reference_denominators_misses_and_all_alternatives_survive(self):
        first, second = record(), record(ambiguous=True)
        references = {'panels': {'p': {'samples': 2}}, 'samples': [
            {'original': {'panel': 'p'}, 'measured_alternatives': [first, second]},
            {'original': {'panel': 'p'}, 'measured_alternatives': []}]}
        result = summary.summarize([first, second], references)
        coverage = result['panels']['p']['evidence_coverage']
        self.assertEqual(coverage['samples'], 2)
        self.assertEqual(coverage['all_actual_alternatives'], 2)
        self.assertEqual(coverage['actual_samples_no_original_alternative'], 1)
        self.assertEqual(coverage['actual_samples_with_any_available_score'], 1)
        self.assertEqual(coverage['actual_available_alternatives'], 2)
        self.assertEqual(coverage['actual_available_alternatives_without_recorded_ambiguity'], 1)

    def test_unflagged_negative_reference_is_reported_not_hidden(self):
        value = record(gain=-3)
        result = summary.summarize([value], {'panels': {}, 'samples': []})
        self.assertEqual(len(result['posthoc_reference_states_with_negative_gain_and_no_recorded_ambiguity']), 1)
        self.assertEqual(result['posthoc_largest_negative_gains'][0]['advantage'], -3)

    def test_input_nonmutation_and_no_classification_policy(self):
        records = [record()]; original = copy.deepcopy(records)
        result = summary.summarize(records, {'panels': {}, 'samples': []})
        self.assertEqual(records, original)
        self.assertNotIn('accepted', result)
        self.assertNotIn('false_positives_removed', result)
        self.assertTrue(any('No suppression' in text for text in result['caveats']))


if __name__ == '__main__':
    unittest.main()
