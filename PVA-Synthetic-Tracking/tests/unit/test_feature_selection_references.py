import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('feature_reference_score',
    Path(__file__).resolve().parents[2] / 'scripts/score_feature_selection_references.py')
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def sample(frame=0, x=10, window='event', uncertainty=2, panel='dense'):
    saved = dict(hit=True, assigned_id='0/bright:old', all_gated_ids=['0/bright:old'])
    return dict(panel=panel, clip_id='0029', window_id=window, frame_index=frame,
        source_xy=[x, 10], position_uncertainty_px=uncertainty, polarity='bright',
        saved={stage: copy.deepcopy(saved) for stage in score.STAGES})


def row(frame=0, name='bright:new', x=10, **overrides):
    track = dict(track_id=name, segment=0, measured=True, qualified_moving=True,
                 measurement_source_xy=[x, 10], source_xy=[999, 999])
    track.update(overrides)
    return dict(frame_index=frame, timestamp_ns=frame * 100000000, segment=0,
        coverage=dict(full_shape_hw=[3190, 4784], configured_crop=None,
                      native_pixel_sampling=True, detection_ready=True, warmup=False,
                      searchable_pixels=100),
        motion=dict(reset=False, pva_failure=False), tracks=[track])


class ReferencePureTests(unittest.TestCase):
    def test_historical_metadata_exact_counts_gates_and_misses(self):
        bindings = {}
        refs = score.references(bindings)
        self.assertEqual(len(refs), 356)
        self.assertEqual(len(bindings), 2)
        self.assertEqual(score.Counter(s['panel'] for s in refs), score.PANELS)
        counts = {p: sum(s['saved']['qualified_measurement']['hit'] for s in refs if s['panel'] == p)
                  for p in score.PANELS}
        self.assertEqual(counts, score.BASELINE_QUALIFIED)
        self.assertEqual({s['position_uncertainty_px'] + 2 for s in refs if s['panel'] == 'pilot'
                          and s['window_id'] == '029_A_intermittent'}, {7})
        miss = next(s for s in refs if s['panel'] == 'dense' and s['clip_id'] == '0126'
                    and s['frame_index'] == 216)
        self.assertFalse(miss['saved']['actual_measurement']['hit'])
        self.assertFalse(miss['saved']['qualified_measurement']['hit'])
        grid = next(s for s in refs if s['clip_id'] == '0055')
        self.assertTrue(grid['saved']['actual_measurement']['hit'])
        self.assertFalse(grid['saved']['qualified_measurement']['hit'])

    def test_new_id_preserved_no_literal_baseline_id_requirement(self):
        s = sample()
        old = score.score_frame(row(name='bright:old'), [s])
        new = score.score_frame(row(), [s])
        result = score.compare(old, new)
        self.assertTrue(result['no_new_actual_losses'])
        self.assertTrue(result['no_new_qualified_losses'])
        self.assertEqual(new[0]['stages']['qualified_measurement']['assigned_id'], '0/bright:new')

    def test_actual_not_filtered_coordinate(self):
        records = score.score_frame(row(x=100, source_xy=[10, 10]), [sample()])
        self.assertFalse(records[0]['stages']['actual_measurement']['hit'])

    def test_coast_does_not_count(self):
        r = row(measured=False, measurement_source_xy=None, source_xy=[10, 10])
        result = score.score_frame(r, [sample()])[0]
        self.assertFalse(result['stages']['actual_measurement']['hit'])
        self.assertFalse(result['stages']['qualified_measurement']['hit'])

    def test_unqualified_is_actual_but_not_qualified(self):
        result = score.score_frame(row(qualified_moving=False), [sample()])[0]
        self.assertTrue(result['stages']['actual_measurement']['hit'])
        self.assertFalse(result['stages']['qualified_measurement']['hit'])

    def test_unavailable_denominator_retained_not_hit(self):
        r = row()
        r['coverage']['detection_ready'] = False
        result = score.score_frame(r, [sample()])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]['stages']['actual_measurement']['hit'])

    def test_radius_inclusive_and_per_sample(self):
        result = score.score_frame(row(x=14), [sample(uncertainty=2)])[0]
        self.assertTrue(result['stages']['qualified_measurement']['hit'])
        self.assertFalse(score.score_frame(row(x=14.001), [sample(uncertainty=2)])[0]
                         ['stages']['qualified_measurement']['hit'])

    def test_wrong_polarity_is_not_hit(self):
        result = score.score_frame(row(name='dark:new'), [sample()])[0]
        self.assertFalse(result['stages']['actual_measurement']['hit'])

    def test_one_observation_cannot_cover_two_same_window_samples(self):
        result = score.score_frame(row(), [sample(x=9), sample(x=11)])
        self.assertEqual(sum(r['stages']['actual_measurement']['hit'] for r in result), 1)
        self.assertTrue(all(r['stages']['actual_measurement']['shared_gated_observation'] for r in result))

    def test_overlapping_panels_scored_separately(self):
        result = score.score_frame(row(), [sample(), sample(panel='pilot')])
        self.assertEqual(sum(r['stages']['actual_measurement']['hit'] for r in result), 2)

    def test_multiple_alternatives_reported(self):
        r = row()
        r['tracks'].append(dict(r['tracks'][0], track_id='bright:other'))
        result = score.score_frame(r, [sample()])[0]['stages']['qualified_measurement']
        self.assertTrue(result['multiple_gated_alternatives'])
        self.assertEqual(result['all_gated_ids'], ['0/bright:new', '0/bright:other'])

    def test_maximum_cardinality_reassignment(self):
        samples = [sample(x=11), sample(x=8)]
        obs = [dict(id='a', xy=[10, 10], polarity='bright'),
               dict(id='b', xy=[14, 10], polarity='bright')]
        matches, _ = score.assign(samples, obs)
        self.assertEqual(matches, {0: 1, 1: 0})

    def test_lost_and_recovered_both_visible_despite_equal_total(self):
        old, new = [], []
        for f in (0, 1):
            old.extend(score.score_frame(row(f, x=10 if f == 0 else 100), [sample(f)]))
            new.extend(score.score_frame(row(f, x=100 if f == 0 else 10), [sample(f)]))
        result = score.compare(old, new)
        self.assertFalse(result['no_new_qualified_losses'])
        stage = result['panels']['dense']['stages']['qualified_measurement']
        self.assertEqual((stage['baseline_hits'], stage['candidate_hits']), (1, 1))
        self.assertEqual(len(stage['newly_lost']), 1)
        self.assertEqual(len(stage['newly_recovered']), 1)

    def test_coherence_does_not_merge_split_identities(self):
        records = score.score_frame(row(0), [sample(0)])
        records += score.score_frame(row(1, name='bright:other'), [sample(1)])
        group = score.coherence(records)['groups'][0]['stages']['qualified_measurement']
        self.assertEqual(group['any_identity_samples'], 2)
        self.assertEqual(group['best_coherent_identity_samples'], 1)
        self.assertEqual(group['complete_coherent_identities'], [])

    def test_segment_reset_breaks_coherence(self):
        a, b = row(0), row(1)
        b['segment'] = b['tracks'][0]['segment'] = 1
        records = score.score_frame(a, [sample(0)]) + score.score_frame(b, [sample(1)])
        self.assertEqual(score.coherence(records)['groups'][0]['stages']['qualified_measurement']
                         ['complete_coherent_identities'], [])

    def test_new_coherence_loss_exposed_without_sample_misses(self):
        old = score.score_frame(row(0), [sample(0)]) + score.score_frame(row(1), [sample(1)])
        new = score.score_frame(row(0), [sample(0)]) + score.score_frame(row(1, name='bright:other'), [sample(1)])
        self.assertTrue(score.compare(old, new)['no_new_qualified_losses'])
        changes = score.compare_coherence(score.coherence(old), score.coherence(new))
        self.assertFalse(changes['no_new_coherent_support_losses'])
        self.assertTrue(all(c['lost_complete_coherent_support'] for c in changes['changes']))

    def test_duplicate_identity_bad_point_and_coast_point_rejected(self):
        cases = []
        r = row(); r['tracks'].append(dict(r['tracks'][0])); cases.append(r)
        cases.extend([row(measurement_source_xy=None), row(measurement_source_xy=[float('nan'), 0]),
                      row(measured=False), row(segment=1), row(measured=1)])
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                score.score_frame(case, [sample()])

    def test_changed_or_duplicate_reference_denominator_rejected(self):
        records = score.score_frame(row(), [sample()])
        for candidate in ([], records + records):
            with self.assertRaises(ValueError):
                score.compare(records, candidate)

    def test_nonfinite_or_duplicate_json_rejected(self):
        for text in ('{"x":NaN}', '{"x":1e309}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                score.decode(text)

    def test_coverage_and_sequence_fail_closed(self):
        score.validate_row(row(), 0)
        cases = []
        for field, value in (('frame_index', 1), ('timestamp_ns', 1), ('segment', -1)):
            r = row(); r[field] = value; cases.append(r)
        for field, value in (('configured_crop', [0, 0, 10, 10]), ('warmup', True),
                             ('searchable_pixels', 0), ('native_pixel_sampling', False)):
            r = row(); r['coverage'][field] = value; cases.append(r)
        r = row(); r['motion']['reset'] = True; cases.append(r)
        for r in cases:
            with self.assertRaises(ValueError): score.validate_row(r, 0)


class ReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        p = self.root / name
        p.write_text(json.dumps(value))
        return p

    def candidate(self):
        launch = dict(source_sha256=score.SOURCES['0029'], fps=10,
            configuration=dict(input_bit_depth=8), config_sha256='cfg', motion_config_sha256='motion',
            package_sha256={'x.py': 'abc'})
        report = dict(source_sha256=score.SOURCES['0029'], frames=687, completed=True,
                      full_clip=True, configuration=launch['configuration'])
        journal = self.root / 'frames.jsonl'; journal.write_text('{}\n')
        paths = dict(journal=journal, launch=self.save('launch.json', launch),
                     report=self.save('report.json', report))
        receipt = dict(passed=True, error=None, clip='0029', source=dict(sha256=score.SOURCES['0029']),
            processed_frames=687, decoded_frames_verified=687, feature_algorithm_changed=True,
            **{role + '_sha256': score.sha(path) for role, path in paths.items()})
        for field in ('detector_configuration_changed', 'tracker_configuration_changed', 'global_motion_gates_changed',
                      'annotations_supplied_to_detector', 'raw16_accessed', 'sealed_holdouts_accessed', 'production_promotion'):
            receipt[field] = False
        paths['execution_receipt'] = self.save('execution_receipt.json', receipt)
        entry = dict(source_sha256=score.SOURCES['0029'], frames=687,
                     artifacts={role: dict(path=str(path), sha256=score.sha(path)) for role, path in paths.items()})
        return launch, receipt, entry

    def test_valid_hashbound_candidate_metadata(self):
        launch, _, entry = self.candidate()
        bindings = {}
        self.assertEqual(score.validate_candidate('0029', entry, launch, bindings), self.root / 'frames.jsonl')
        self.assertEqual(len(bindings), 4)

    def test_changed_artifact_bytes_rejected(self):
        launch, _, entry = self.candidate()
        (self.root / 'frames.jsonl').write_text('{"changed":true}\n')
        with self.assertRaises(ValueError): score.validate_candidate('0029', entry, launch, {})

    def test_receipt_hash_binding_rejected_even_rehashed_manifest(self):
        launch, receipt, entry = self.candidate()
        receipt['journal_sha256'] = '0' * 64
        path = self.save('execution_receipt.json', receipt)
        entry['artifacts']['execution_receipt']['sha256'] = score.sha(path)
        with self.assertRaises(ValueError): score.validate_candidate('0029', entry, launch, {})

    def test_full_receipt_safety_and_frozen_config_required(self):
        for field, value in (('passed', False), ('decoded_frames_verified', 686),
                             ('global_motion_gates_changed', True), ('feature_algorithm_changed', False),
                             ('annotations_supplied_to_detector', True)):
            launch, receipt, entry = self.candidate()
            receipt[field] = value
            path = self.save('execution_receipt.json', receipt)
            entry['artifacts']['execution_receipt']['sha256'] = score.sha(path)
            with self.subTest(field=field), self.assertRaises(ValueError):
                score.validate_candidate('0029', entry, launch, {})
        launch, _, entry = self.candidate()
        launch['configuration'] = dict(input_bit_depth=8, changed=True)
        with self.assertRaises(ValueError): score.validate_candidate('0029', entry, launch, {})

    def test_media_paths_and_symlinks_not_read(self):
        p = self.root / 'clip.avi'; p.write_bytes(b'not media')
        with self.assertRaises(ValueError): score.sha(p)
        metadata = self.save('metadata.json', {})
        link = self.root / 'link.json'; link.symlink_to(metadata)
        with self.assertRaises(ValueError): score.sha(link)

    def test_journal_full_history_and_saved_baseline_assignment(self):
        p = self.root / 'frames.jsonl'
        p.write_text('\n'.join(json.dumps(row(i, name='bright:old')) for i in range(2)) + '\n')
        with patch.dict(score.COUNTS, {'0029': 2}):
            result = score.score_journal(p, '0029', [sample(0), sample(1)], baseline=True)
            self.assertEqual(result['frames'], 2)
            self.assertEqual(result['samples'], 2)
            p.write_text(json.dumps(row(0)) + '\n')
            with self.assertRaises(ValueError): score.score_journal(p, '0029', [sample(0), sample(1)])
            p.write_text('\n'.join(json.dumps(row(i)) for i in range(2)) + '\n')
            with self.assertRaises(ValueError): score.score_journal(p, '0029', [sample(0), sample(1)], baseline=True)


if __name__ == '__main__':
    unittest.main()
