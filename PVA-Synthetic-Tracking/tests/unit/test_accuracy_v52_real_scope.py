"""Generated fixtures only: strict JSON allowlist and complete guard accounting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import accuracy_v52_real_scope as subject


POINTS = [[8, 8], [16, 8], [24, 8]]


def resign(forecast):
    forecast['forecast_sha256'] = subject.canonical_hash(
        {k: v for k, v in forecast.items() if k != 'forecast_sha256'})


def fixture():
    states = []
    for frame, partition, status in ((10, 'calibration', 'background_measured'),
            (11, 'calibration', 'response_unavailable'), (20, 'embargo', 'embargo_not_scored'),
            (40, 'evaluation', 'background_measured'), (41, 'evaluation', 'history_unknown')):
        states.append(dict(state_key=['0029', frame, 0, 'bright:1'], partition=partition,
            v50_status=status, source_scores_and_original_detections_unchanged=True,
            original_qualified_moving=False, original_reference_samples=[0], opaque_preserved_field='keep'))
    assignments = [dict(state_key=s['state_key'], partition=s['partition'],
                        archived=s['v50_status'] != 'history_unknown') for s in states]
    scope = dict(history_frames=8, embargo_frames=8,
        cutoffs=[dict(clip='0029', segment=0, cutoff_frame_index=15)],
        assignments=assignments, partitions={}, counts={})
    for part in subject.PARTITIONS:
        rows = [a for a in assignments if a['partition'] == part]
        archive = sum(a['archived'] for a in rows)
        scope['partitions'][part] = [a['state_key'] for a in rows]
        scope['counts'][part] = dict(states=len(rows), archived_states=archive,
            history_unknown_states=len(rows)-archive, unique_response_frames=len(rows),
            archived_response_frames=archive, frames_without_archives=len(rows)-archive)
    docs = {'freeze.json': {'scope': scope}, 'state_results.jsonl': states,
            'reference_context.json': {'samples': [{'not_fit_input': True}]}}
    for part in ('calibration', 'evaluation'):
        docs[part + '_forecasts.jsonl'] = []
        docs[part + '_measurements.jsonl'] = []
        for s in [s for s in states if s['partition'] == part and s['v50_status'] != 'history_unknown']:
            f = dict(available=True, reasons=[], used_points_xy=deepcopy(POINTS), used_count=3,
                used_support_sha256=subject.canonical_hash(POINTS),
                stencils=[dict(axis='x', center_xy=[16, 8], pixels_xy=deepcopy(POINTS), weights=[1, -2, 1])],
                stencil_count=1, arms={})
            f['stencil_sha256'] = subject.canonical_hash(f['stencils'])
            for arm in subject.ARMS:
                f['arms'][arm] = dict(prediction=[1., 2., 3.] if arm != subject.ARMS[2] else [2., 3., 4.],
                                     scale=[1., 1., 1.])
            resign(f)
            available = s['v50_status'] == 'background_measured'
            m = dict(available=available, reasons=[] if available else ['nonfinite_current_on_fixed_used_guard_support'],
                forecast_sha256=f['forecast_sha256'], used_support_sha256=f['used_support_sha256'], used_count=3,
                current_nonfinite_used_point_count=0 if available else 1, arms={})
            if available:
                for arm in subject.ARMS:
                    m['arms'][arm] = dict(residuals=[4.-v for v in f['arms'][arm]['prediction']])
            docs[part + '_forecasts.jsonl'].append(dict(state_key=s['state_key'], forecast=f))
            docs[part + '_measurements.jsonl'].append(dict(state_key=s['state_key'], clip='0029', segment=0,
                frame_index=s['state_key'][1], forecast_available=True, measurement=m))
    return deepcopy(docs)


def first(docs):
    return (docs['calibration_forecasts.jsonl'][0]['forecast'],
            docs['calibration_measurements.jsonl'][0]['measurement'])


def rebind(f, m):
    resign(f)
    m['forecast_sha256'] = f['forecast_sha256']
    m['used_support_sha256'] = f['used_support_sha256']


class AssembleTests(unittest.TestCase):
    def test_preserves_all_states_and_separates_references(self):
        docs = fixture()
        result = subject.assemble(docs)
        self.assertEqual(result['states'], docs['state_results.jsonl'])
        self.assertEqual(len(result['packets']), 3)
        self.assertEqual(result['references'], docs['reference_context.json'])
        self.assertTrue(all('references' not in p and 'original_reference_samples' not in p for p in result['packets']))
        self.assertEqual(result['counts']['states'], 5)
        self.assertEqual(result['counts']['available_current_packets'], 2)
        self.assertEqual(result['counts']['guard_point_opportunities'], 9)
        self.assertEqual(result['counts']['available_current_points'], 6)
        self.assertEqual(result['counts']['unavailable_current_packet_points'], 3)

    def test_unavailable_packet_retains_all_null_current(self):
        result = subject.assemble(fixture())
        row = next(p for p in result['packets'] if not p['available'])
        self.assertEqual(row['current'], [None, None, None])
        self.assertEqual(row['points_xy'], POINTS)
        self.assertEqual(row['metadata']['original_nonfinite_current_point_count'], 1)

    def test_signed_residual_reconstruction(self):
        result = subject.assemble(fixture())
        self.assertEqual(result['packets'][0]['current'], [4., 4., 4.])
        self.assertEqual(result['packets'][0]['median3'], [2., 3., 4.])

    def test_outputs_do_not_alias_inputs(self):
        docs = fixture()
        before = deepcopy(docs)
        result = subject.assemble(docs)
        result['states'][0]['original_reference_samples'].append(9)
        result['packets'][0]['points_xy'][0][0] = 999
        result['packets'][0]['median8'][0] = 999
        result['scope']['assignments'].clear()
        result['references']['samples'].clear()
        self.assertEqual(docs, before)

    def test_json_safe_without_nan(self):
        json.dumps(subject.assemble(fixture()), allow_nan=False)

    def test_rejects_extra_document(self):
        docs = fixture(); docs['unapproved.npz'] = {}
        with self.assertRaisesRegex(ValueError, 'seven literal'):
            subject.assemble(docs)

    def test_rejects_duplicate_state_assignment_forecast_measurement(self):
        for name in ('state_results.jsonl', 'calibration_forecasts.jsonl', 'evaluation_measurements.jsonl'):
            with self.subTest(name=name):
                docs = fixture(); docs[name].append(deepcopy(docs[name][0]))
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    subject.assemble(docs)
        docs = fixture(); docs['freeze.json']['scope']['assignments'].append(deepcopy(docs['freeze.json']['scope']['assignments'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            subject.assemble(docs)

    def test_rejects_changed_state_membership(self):
        docs = fixture(); docs['state_results.jsonl'].pop()
        with self.assertRaisesRegex(ValueError, 'membership'):
            subject.assemble(docs)

    def test_rejects_wrong_partition_and_source_mutation(self):
        for field, value in [('partition', 'evaluation'), ('source_scores_and_original_detections_unchanged', False)]:
            docs = fixture(); docs['state_results.jsonl'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.assemble(docs)

    def test_rejects_wrong_scope_counts_or_partition_membership(self):
        docs = fixture(); docs['freeze.json']['scope']['counts']['calibration']['states'] += 1
        with self.assertRaisesRegex(ValueError, 'denominator'):
            subject.assemble(docs)
        docs = fixture(); docs['freeze.json']['scope']['partitions']['calibration'].pop()
        with self.assertRaisesRegex(ValueError, 'membership'):
            subject.assemble(docs)

    def test_rejects_changed_history_and_cutoffs(self):
        for field in ('history_frames', 'embargo_frames'):
            docs = fixture(); docs['freeze.json']['scope'][field] = 7
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'eight-frame'):
                subject.assemble(docs)
        docs = fixture(); docs['freeze.json']['scope']['cutoffs'].append(deepcopy(docs['freeze.json']['scope']['cutoffs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            subject.assemble(docs)

    def test_rejects_missing_packet(self):
        docs = fixture(); docs['calibration_forecasts.jsonl'].pop()
        with self.assertRaisesRegex(ValueError, 'Packet membership'):
            subject.assemble(docs)

    def test_rejects_wrong_record_identity(self):
        docs = fixture(); docs['calibration_measurements.jsonl'][0]['clip'] = '0126'
        with self.assertRaisesRegex(ValueError, 'identity'):
            subject.assemble(docs)

    def test_rejects_forecast_tampering_before_reconstruction(self):
        docs = fixture(); first(docs)[0]['arms'][subject.ARMS[0]]['prediction'][0] += 1
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            subject.assemble(docs)

    def test_rejects_support_hash_and_current_binding_changes(self):
        for field in ('forecast_sha256', 'used_support_sha256', 'used_count'):
            docs = fixture(); first(docs)[1][field] = 'wrong'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'bindings'):
                subject.assemble(docs)

    def test_rejects_core_offgrid_duplicate_and_reordered_geometry(self):
        for points in ([[64, 64], [16, 8], [24, 8]], [[9, 8], [16, 8], [24, 8]],
                       [[8, 8], [8, 8], [24, 8]], [[24, 8], [16, 8], [8, 8]]):
            docs = fixture(); f, m = first(docs); f['used_points_xy'] = points
            f['used_support_sha256'] = subject.canonical_hash(points); rebind(f, m)
            with self.subTest(points=points), self.assertRaises(ValueError):
                subject.assemble(docs)

    def test_rejects_stencil_binding_geometry_and_union(self):
        for mutation in ('hash', 'weights', 'empty'):
            docs = fixture(); f, m = first(docs)
            if mutation == 'hash': f['stencil_sha256'] = 'wrong'
            elif mutation == 'weights':
                f['stencils'][0]['weights'] = [1, 1, 1]
                f['stencil_sha256'] = subject.canonical_hash(f['stencils'])
            else:
                f['stencils'] = []; f['stencil_count'] = 0
                f['stencil_sha256'] = subject.canonical_hash([])
            rebind(f, m)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.assemble(docs)

    def test_rejects_unexpected_forecast_or_measurement_arm(self):
        for target in ('forecast', 'measurement'):
            docs = fixture(); f, m = first(docs)
            (f if target == 'forecast' else m)['arms'].pop(subject.ARMS[1]); rebind(f, m)
            with self.subTest(target=target), self.assertRaises(ValueError):
                subject.assemble(docs)

    def test_rejects_nonfinite_wrong_length_and_boolean_vectors(self):
        for values in ([1, 2], [1, 2, float('inf')], [True, 2, 3]):
            docs = fixture(); f, m = first(docs); m['arms'][subject.ARMS[0]]['residuals'] = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'vector'):
                subject.assemble(docs)

    def test_rejects_changed_shared_predictions_and_scales(self):
        for field in ('prediction', 'scale'):
            docs = fixture(); f, m = first(docs); f['arms'][subject.ARMS[1]][field][0] += 1
            rebind(f, m)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'shared'):
                subject.assemble(docs)

    def test_rejects_crossarm_current_disagreement(self):
        docs = fixture(); first(docs)[1]['arms'][subject.ARMS[2]]['residuals'][0] += .25
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            subject.assemble(docs)

    def test_rejects_partial_recovery_of_unavailable_packet(self):
        docs = fixture(); m = docs['calibration_measurements.jsonl'][1]['measurement']
        m['arms'] = {subject.ARMS[0]: {'residuals': [None, 0, 0]}}
        with self.assertRaisesRegex(ValueError, 'Unavailable packet'):
            subject.assemble(docs)

    def test_rejects_invalid_missing_count_and_status(self):
        docs = fixture(); docs['calibration_measurements.jsonl'][1]['measurement']['current_nonfinite_used_point_count'] = 0
        with self.assertRaisesRegex(ValueError, 'Unavailable packet'):
            subject.assemble(docs)
        docs = fixture(); docs['state_results.jsonl'][1]['v50_status'] = 'background_measured'
        with self.assertRaisesRegex(ValueError, 'availability'):
            subject.assemble(docs)

    def test_preserves_embargo_and_history_unknown(self):
        for index in (2, 4):
            docs = fixture(); docs['state_results.jsonl'][index]['v50_status'] = 'background_measured'
            with self.subTest(index=index), self.assertRaises(ValueError):
                subject.assemble(docs)


class LiteralReadTests(unittest.TestCase):
    def payloads(self, complete=True):
        docs = fixture()
        payloads = {}
        for name, value in docs.items():
            raw = ('\n'.join(json.dumps(r) for r in value)+'\n').encode() if name.endswith('.jsonl') else json.dumps(value).encode()
            payloads[subject.BASE / name] = raw
        receipt = dict(completed=complete, files_sha256={str(p): hashlib.sha256(raw).hexdigest() for p, raw in payloads.items()})
        # Deliberately malicious/out-of-scope path must never be visited.
        receipt['files_sha256']['/unapproved/sealed_holdout.npz'] = 'unused'
        raw = json.dumps(receipt).encode()
        payloads[subject.BASE / 'completion_receipt.json'] = raw
        return docs, payloads, hashlib.sha256(raw).hexdigest()

    def test_reads_only_literal_names_and_rechecks_bytes(self):
        docs, payloads, digest = self.payloads()
        with patch.object(subject, 'RECEIPT_SHA', digest), patch.object(subject, '_regular_bytes', side_effect=payloads.__getitem__) as read:
            loaded, bindings = subject.read_authorized()
        self.assertEqual(loaded, docs)
        self.assertEqual(set(bindings), {str(p) for p in payloads})
        self.assertEqual({call.args[0] for call in read.call_args_list}, set(payloads))
        self.assertEqual(read.call_count, 2*len(payloads))

    def test_bad_receipt_hash_is_rejected_before_decoding(self):
        with patch.object(subject, '_regular_bytes', return_value=b'not json') as read:
            with self.assertRaisesRegex(ValueError, 'receipt hash'):
                subject.read_authorized()
        self.assertEqual(read.call_count, 1)

    def test_noncompleted_receipt_rejected(self):
        _, payloads, digest = self.payloads(False)
        with patch.object(subject, 'RECEIPT_SHA', digest), patch.object(subject, '_regular_bytes', side_effect=payloads.__getitem__):
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                subject.read_authorized()

    def test_bad_artifact_hash_rejected_before_decoding(self):
        _, payloads, digest = self.payloads()
        payloads[subject.BASE / 'freeze.json'] = b'not json'
        with patch.object(subject, 'RECEIPT_SHA', digest), patch.object(subject, '_regular_bytes', side_effect=payloads.__getitem__):
            with self.assertRaisesRegex(ValueError, 'artifact hash'):
                subject.read_authorized()

    def test_changed_input_during_read_rejected(self):
        _, payloads, digest = self.payloads()
        visits = {}
        def read(path):
            visits[path] = visits.get(path, 0)+1
            return payloads[path] if visits[path] == 1 else b'changed'
        with patch.object(subject, 'RECEIPT_SHA', digest), patch.object(subject, '_regular_bytes', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                subject.read_authorized()

    def test_regular_bytes_rejects_symlink_and_parent_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            path = base / 'safe.json'; path.write_bytes(b'{}')
            self.assertEqual(subject._regular_bytes(path), b'{}')
            alias = base / 'alias.json'; alias.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'literal regular'):
                subject._regular_bytes(alias)
            with self.assertRaisesRegex(ValueError, 'literal regular'):
                subject._regular_bytes(base / 'missing.json')


if __name__ == '__main__':
    unittest.main()
