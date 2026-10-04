"""Generated data only: residual algebra, denominators and literal read scope."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import analyze_accuracy_v51_residuals as subject


POINTS = [[8, 8], [120, 8], [8, 120], [120, 120]]


def state(clip='0029', frame=20, track='a', partition='evaluation', status='background_measured'):
    return dict(state_key=[clip, frame, 0, track], partition=partition, v50_status=status)


def packet(s, residuals=None, available=True):
    return dict(state_key=s['state_key'], partition=s['partition'], available=available,
                reasons=[] if available else ['missing_current_support'],
                arms={arm: subject.residual_metrics(POINTS, residuals or [1, 1, 1, 1])
                      for arm in subject.ARMS} if available else {})


def documents():
    cal = state(frame=20, partition='calibration')
    eva = state(frame=40)
    unknown = state(frame=41, status='history_unknown')
    embargo = state(frame=29, partition='embargo', status='embargo_not_scored')
    rows = [cal, eva, unknown, embargo]
    docs = {'state_results.jsonl': rows, 'freeze.json': {'scope': {'assignments': [
        dict(state_key=s['state_key'], partition=s['partition'], archived=s != unknown) for s in rows]}}}
    for part, s in [('calibration', cal), ('evaluation', eva)]:
        forecast = dict(available=True, forecast_sha256='forecast', used_support_sha256='support',
                        used_points_xy=POINTS, arms={arm: {} for arm in subject.ARMS})
        measurement = dict(available=True, reasons=[], forecast_sha256='forecast', used_support_sha256='support',
                           arms={arm: dict(residuals=[1, -1, 2, -2]) for arm in subject.ARMS})
        docs[part + '_forecasts.jsonl'] = [dict(state_key=s['state_key'], forecast=forecast)]
        docs[part + '_measurements.jsonl'] = [dict(state_key=s['state_key'], measurement=measurement)]
    return docs


class ResidualAlgebraTests(unittest.TestCase):
    def test_uniform_signed_mass_is_coherent(self):
        result = subject.residual_metrics(POINTS, [2, 2, 2, 2])
        self.assertEqual(result['coherence'], 1)
        self.assertEqual(result['signed_median_residual_dn'], 2)
        self.assertEqual(result['top_tenth_absolute_mass']['share'], .25)

    def test_balanced_signs_cancel_coherence(self):
        result = subject.residual_metrics(POINTS, [2, -2, 2, -2])
        self.assertEqual(result['coherence'], 0)
        self.assertEqual(result['signed_median_residual_dn'], 0)
        self.assertEqual(result['positive_fraction'], .5)
        self.assertEqual(result['negative_fraction'], .5)

    def test_zero_mass_is_undefined_share(self):
        result = subject.residual_metrics(POINTS, [0, 0, 0, 0])
        self.assertIsNone(result['coherence'])
        self.assertIsNone(result['top_tenth_absolute_mass']['share'])
        self.assertEqual(result['zero_fraction'], 1)
        self.assertTrue(all(q['absolute_mass_share'] is None for q in result['quadrants'].values()))

    def test_quadrants_use_inclusive_right_lower_boundaries(self):
        result = subject.residual_metrics([[63, 63], [64, 63], [63, 64], [64, 64]], [1, 2, 3, 4])
        self.assertEqual([q['absolute_mass_share'] for q in result['quadrants'].values()], [.1, .2, .3, .4])

    def test_localized_mass(self):
        result = subject.residual_metrics(POINTS, [0, 0, 0, 10])
        self.assertEqual(result['top_tenth_absolute_mass']['share'], 1)
        self.assertEqual(result['quadrants']['lower_right']['absolute_mass_share'], 1)

    def test_top_count_uses_ceiling(self):
        result = subject.top_tenth_share([1]*11)
        self.assertEqual(result['selected_count'], 2)
        self.assertAlmostEqual(result['share'], 2/11)

    def test_empty_top_mass_retains_zero_count(self):
        self.assertEqual(subject.top_tenth_share([]), dict(count=0, selected_count=0, share=None))

    def test_invalid_residual_or_geometry_rejected(self):
        for points, residuals in [(POINTS, [1, 2, 3]), (POINTS, [1, 2, 3, np.nan]),
                                  (POINTS, [[1, 2, 3, 4]]), ([[8, 8], [8, 8]], [1, 2]),
                                  ([[8.5, 8]], [1]), ([[129, 8]], [1]), ([], [])]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                subject.residual_metrics(points, residuals)

    def test_negative_or_nonfinite_mass_rejected(self):
        for values in [[-1], [np.inf], [np.nan], [[1]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.top_tenth_share(values)

    def test_empty_distributions_are_not_zero_errors(self):
        result = subject.distribution([])
        self.assertEqual(result['count'], 0)
        self.assertIsNone(result['mean'])


class DenominatorTests(unittest.TestCase):
    def test_unknown_only_bin_is_retained(self):
        s = state(frame=41, status='history_unknown')
        result = subject.summarize({tuple(s['state_key']): s}, {})
        self.assertEqual(len(result['nine_response_frame_bins']), 1)
        metrics = result['nine_response_frame_bins'][0]['metrics']
        self.assertEqual(metrics['states'], 1)
        self.assertEqual(metrics['available_packets'], 0)
        self.assertIsNone(metrics['arms'][subject.ARMS[0]]['packet_mae_dn']['mean'])

    def test_unavailable_packet_invalidates_whole_archived_frame(self):
        a, b = state(track='a'), state(track='b')
        states = {tuple(s['state_key']): s for s in [a, b]}
        packets = {tuple(a['state_key']): packet(a), tuple(b['state_key']): packet(b, available=False)}
        result = subject.aggregate(states, packets)
        self.assertEqual(result['available_packets'], 1)
        self.assertEqual(result['unavailable_packets'], 1)
        self.assertEqual(result['complete_archived_response_frames'], 0)
        self.assertEqual(result['arms'][subject.ARMS[0]]['equal_frame_mean_packet_mae_dn']['count'], 0)

    def test_equal_frame_weighting_not_equal_packet_weighting(self):
        rows = [state(frame=20, track=str(i)) for i in range(3)] + [state(frame=30)]
        states = {tuple(s['state_key']): s for s in rows}
        packets = {tuple(s['state_key']): packet(s, [0, 0, 0, 0] if s['state_key'][1] == 20 else [10]*4) for s in rows}
        arm = subject.aggregate(states, packets)['arms'][subject.ARMS[0]]
        self.assertEqual(arm['packet_mae_dn']['mean'], 2.5)
        self.assertEqual(arm['equal_frame_mean_packet_mae_dn']['mean'], 5)
        self.assertEqual(arm['top_tenth_frame_mean_error_share']['share'], 1)

    def test_partition_clip_and_bin_boundaries_remain_separate(self):
        rows = [state(frame=26, partition='calibration'), state(frame=27, partition='embargo'),
                state(clip='0126', frame=27)]
        states = {tuple(s['state_key']): s for s in rows}
        result = subject.summarize(states, {})
        self.assertEqual(len(result['nine_response_frame_bins']), 3)
        self.assertEqual(sorted(b['frame_start'] for b in result['nine_response_frame_bins']), [18, 27, 27])
        self.assertEqual(result['groups']['embargo']['0029']['states'], 1)

    def test_unknown_state_does_not_become_zero_packet(self):
        a, b = state(track='a'), state(track='b', status='history_unknown')
        states = {tuple(s['state_key']): s for s in [a, b]}
        result = subject.aggregate(states, {tuple(a['state_key']): packet(a, [10]*4)})
        self.assertEqual(result['states_without_scored_archives'], 1)
        self.assertEqual(result['arms'][subject.ARMS[0]]['equal_frame_mean_packet_mae_dn']['mean'], 10)


class ScopeTests(unittest.TestCase):
    def test_generated_documents_validate(self):
        states, packets = subject.packet_diagnostics(documents())
        self.assertEqual(len(states), 4)
        self.assertEqual(len(packets), 2)

    def test_missing_packet_rejected(self):
        docs = documents()
        docs['evaluation_measurements.jsonl'] = []
        with self.assertRaises(ValueError): subject.packet_diagnostics(docs)

    def test_duplicate_state_rejected(self):
        docs = documents()
        docs['state_results.jsonl'].append(docs['state_results.jsonl'][0])
        with self.assertRaises(ValueError): subject.packet_diagnostics(docs)

    def test_changed_partition_rejected(self):
        docs = documents()
        docs['state_results.jsonl'][0]['partition'] = 'evaluation'
        with self.assertRaises(ValueError): subject.packet_diagnostics(docs)

    def test_changed_forecast_or_support_binding_rejected(self):
        for field in ['forecast_sha256', 'used_support_sha256']:
            docs = documents()
            docs['evaluation_measurements.jsonl'][0]['measurement'][field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.packet_diagnostics(docs)

    def test_unavailable_retained_without_invented_residuals(self):
        docs = documents()
        m = docs['evaluation_measurements.jsonl'][0]['measurement']
        m.update(available=False, reasons=['missing_current_support'], arms={})
        _, packets = subject.packet_diagnostics(docs)
        self.assertFalse(packets[('0029', 40, 0, 'a')]['available'])
        self.assertEqual(packets[('0029', 40, 0, 'a')]['arms'], {})

    def test_reader_does_not_follow_extra_receipt_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            bindings = {'/forbidden/never_open.npz': 'irrelevant'}
            for name in subject.ALLOWED_NAMES:
                payload = b'{}\n'
                (base / name).write_bytes(payload)
                bindings[str(base / name)] = hashlib.sha256(payload).hexdigest()
            payload = json.dumps(dict(completed=True, files_sha256=bindings)).encode()
            (base / 'completion_receipt.json').write_bytes(payload)
            with patch.object(subject, 'BASE', base), patch.object(subject, 'RECEIPT_SHA', hashlib.sha256(payload).hexdigest()):
                docs, actual = subject.read_authorized()
            self.assertEqual(set(docs), set(subject.ALLOWED_NAMES))
            self.assertEqual(len(actual), 7)

    def test_bad_receipt_hash_fails_before_artifact_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            (base / 'completion_receipt.json').write_text('{}')
            with patch.object(subject, 'BASE', base), self.assertRaisesRegex(ValueError, 'receipt hash'):
                subject.read_authorized()

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            (base / 'real').write_text('{}')
            (base / 'linked').symlink_to(base / 'real')
            with self.assertRaises(ValueError): subject._regular_bytes(base / 'linked')


if __name__ == '__main__':
    unittest.main()
