"""Generated compact-record checks; no historical evidence or media is opened."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import trace_accuracy_v55_archive as trace


def reference(sample=0, panel='dense', strict_name='baseline_qualified', assigned=True):
    identity = '0/bright:7'
    alternative = '0/bright:8'
    strict = [True, identity, [identity, alternative], True, 1.25] if assigned else [False, None, [], False, None]
    return dict(sample_index=sample, original=dict(panel=panel, clip_id='0029',
        window_id='generated', frame_index=20, source_xy=[100., 200.],
        position_uncertainty_px=4., polarity='bright', stages={
            'candidate': [True, 'candidate:1', ['candidate:1', 'candidate:2'], True, .75],
            'actual_measurement': [True, identity, [identity, alternative], True, 1.25],
            strict_name: strict}), original_strict_stage=strict_name,
        original_strict_assigned_identity=identity if assigned else None,
        measured_alternatives=[dict(identity=ident, state_key=['0029', 20, 0, track],
            original_qualified_moving=assigned) for ident, track in
            ((identity, 'bright:7'), (alternative, 'bright:8'))])


def state_fixture(qualified=True):
    detail = dict(state_key=['0029', 20, 0, 'bright:7'], previous_stage='available_positive',
        new_stage='guard_gain_unavailable', gain_reasons=['inconsistent_guard'],
        original_strict_qualified=qualified)
    ledger = dict(clip='0029', frame_index=20, segment=0, track_id='bright:7',
        actual_source_xy=[101.25, 199.5], qualified_moving=qualified)
    context = dict(segment=0, track_id='bright:7', measured=True,
        measurement_source_xy=[101.25, 199.5], source_xy=[999., 888.],
        accepted=True, reason='point_preferred', features=dict(point_minus_edge_fraction=.1))
    continuity = dict(segment=0, track_id='bright:7', measured=True,
        measurement_source_xy=[101.25, 199.5], source_xy=[777., 666.],
        baseline_qualified=True, status='baseline_qualified_measured', renderable=True)
    return detail, ledger, [reference(assigned=qualified)], context if qualified else None, continuity if qualified else None


def controls_fixture():
    before = [dict(frames_inclusive=[10, 12], crop_xywh=[3, 4, 16, 16], label='provisional_cloud',
                   measured_states=7, predicted_states=3),
              dict(frames_inclusive=[20, 22], crop_xywh=[30, 40, 16, 16], label='provisional_building',
                   measured_states=0, predicted_states=0)]
    after = deepcopy(before)
    after[0].update(measured_states=2, predicted_states=1)
    return dict(clips={'0126': dict(arms={'baseline': dict(provisional_controls=before),
                                        'point_context': dict(provisional_controls=after)})})


class ArchiveIndexTests(unittest.TestCase):
    def test_empty_index(self):
        self.assertEqual(trace.index_rows([], lambda r: r['id']), {})

    def test_index_keeps_complete_rows_and_order(self):
        rows = [dict(id='b', other=2), dict(id='a', other=3)]
        indexed = trace.index_rows(rows, lambda r: r['id'])
        self.assertEqual(list(indexed), ['b', 'a'])
        self.assertIs(indexed['b'], rows[0])

    def test_duplicate_identity_rejected_even_equal_records(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'):
            trace.index_rows([dict(id=1), dict(id=1)], lambda r: r['id'])

    def test_full_state_key_retains_segment(self):
        _, row, *_ = state_fixture()
        self.assertEqual(trace.state_key(row), ('0029', 20, 0, 'bright:7'))
        second = dict(row, segment=1)
        self.assertEqual(len(trace.index_rows([row, second], trace.state_key)), 2)


class ArchiveStateJoinTests(unittest.TestCase):
    def test_actual_coordinates_not_filtered_coordinates(self):
        result = trace.join_state(*state_fixture())
        self.assertEqual(result['actual_source_xy'], [101.25, 199.5])
        self.assertTrue(result['baseline_qualified_measured'])
        self.assertFalse(result['numerical_unknown_means_detector_miss'])
        self.assertEqual(result['numerical_later'], 'guard_gain_unavailable')

    def test_filtered_coordinates_need_not_match_across_shadows(self):
        args = list(state_fixture())
        args[3]['source_xy'] = [-1e9, 1e9]
        self.assertEqual(trace.join_state(*args)['actual_source_xy'], args[1]['actual_source_xy'])

    def test_actual_coordinate_perturbation_is_rejected_exactly(self):
        for position in (3, 4):
            with self.subTest(position=position):
                args = list(state_fixture())
                args[position]['measurement_source_xy'][0] += 1e-12
                with self.assertRaisesRegex(ValueError, 'Actual coordinate mismatch'):
                    trace.join_state(*args)

    def test_prediction_cannot_substitute_measurement(self):
        for position in (3, 4):
            with self.subTest(position=position):
                args = list(state_fixture())
                args[position]['measured'] = False
                with self.assertRaisesRegex(ValueError, 'Prediction cannot'):
                    trace.join_state(*args)

    def test_numeric_truthy_measured_flag_not_accepted(self):
        args = list(state_fixture())
        args[3]['measured'] = 1
        with self.assertRaisesRegex(ValueError, 'Prediction cannot'):
            trace.join_state(*args)

    def test_unqualified_absence_is_null_not_rejection(self):
        result = trace.join_state(*state_fixture(False))
        self.assertIsNone(result['v36_parallel_shadow'])
        self.assertIsNone(result['v39_parallel_shadow'])
        self.assertFalse(result['absent_decision_means_rejected'])

    def test_unqualified_unexpected_output_rejected(self):
        args = list(state_fixture(False))
        args[3] = state_fixture()[3]
        with self.assertRaisesRegex(ValueError, 'Unexpected output'):
            trace.join_state(*args)

    def test_qualified_missing_shadow_rejected(self):
        for position in (3, 4):
            with self.subTest(position=position):
                args = list(state_fixture())
                args[position] = None
                with self.assertRaisesRegex(ValueError, 'Missing qualified decision'):
                    trace.join_state(*args)

    def test_state_identity_mismatch_rejected(self):
        args = list(state_fixture())
        args[1]['frame_index'] += 1
        with self.assertRaisesRegex(ValueError, 'State identity mismatch'):
            trace.join_state(*args)

    def test_shadow_track_identity_mismatch_rejected(self):
        args = list(state_fixture())
        args[4]['segment'] = 1
        with self.assertRaisesRegex(ValueError, 'Joined track identity mismatch'):
            trace.join_state(*args)

    def test_qualification_mismatch_rejected(self):
        args = list(state_fixture())
        args[0]['original_strict_qualified'] = False
        with self.assertRaisesRegex(ValueError, 'Qualification mismatch'):
            trace.join_state(*args)

    def test_nonpositive_selection_rejected(self):
        args = list(state_fixture())
        args[0]['previous_stage'] = 'available_unresolved'
        with self.assertRaisesRegex(ValueError, 'Not an original numerical positive'):
            trace.join_state(*args)

    def test_nonbaseline_output_status_rejected(self):
        args = list(state_fixture())
        args[4]['status'] = 'degraded_measurement'
        with self.assertRaisesRegex(ValueError, 'Not unchanged baseline output'):
            trace.join_state(*args)

    def test_duplicate_reference_alternative_rejected(self):
        args = list(state_fixture())
        args[2][0]['measured_alternatives'].append(deepcopy(args[2][0]['measured_alternatives'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate reference alternative'):
            trace.join_state(*args)

    def test_positive_alternative_does_not_replace_original(self):
        args = list(state_fixture())
        args[0]['new_stage'] = 'available_positive'
        args[2][0]['original_strict_assigned_identity'] = '0/bright:8'
        args[2][0]['original']['stages']['baseline_qualified'][1] = '0/bright:8'
        result = trace.join_state(*args)
        self.assertEqual(result['all_reference_alternative_samples'], [0])
        self.assertEqual(result['original_assigned_reference_samples'], [])

    def test_overlapping_reference_panels_retained(self):
        args = list(state_fixture())
        args[2].append(reference(10, panel='pilot'))
        result = trace.join_state(*args)
        self.assertEqual(result['original_assigned_reference_samples'], [0, 10])

    def test_actual_context_rejection_is_retained_not_new_detector_miss(self):
        args = list(state_fixture())
        args[3].update(accepted=False, reason='edge_preferred_or_tie')
        result = trace.join_state(*args)
        self.assertFalse(result['v36_parallel_shadow']['accepted'])
        self.assertTrue(result['baseline_qualified_measured'])


class ArchiveReferenceSummaryTests(unittest.TestCase):
    def test_overlapping_panels_and_strict_aliases_kept_separate(self):
        refs = [reference(0), reference(1, 'grid', 'strict_qualified_measurement')]
        result = trace.summarize_references(refs)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['assigned_samples'], 2)
        self.assertEqual(result['unique_assigned_states'], 1)
        self.assertEqual(result['alternative_entries'], 4)
        self.assertEqual(result['unique_alternative_states'], 2)
        self.assertEqual(result['panels']['dense']['baseline_qualified'], 1)
        self.assertEqual(result['panels']['grid']['strict_qualified_measurement'], 1)
        self.assertTrue(result['overlapping_not_independent'])
        self.assertFalse(result['airborne_truth'])

    def test_unassigned_candidate_measurement_hits_stay_strict_miss(self):
        ref = reference(assigned=False)
        ref['original']['stages']['with_degraded'] = [True, '0/bright:7', ['0/bright:7'], False, 1.25]
        result = trace.summarize_references([ref])
        self.assertEqual(result['assigned_samples'], 0)
        self.assertEqual(result['original_unassigned'], [ref['original']])
        self.assertEqual(result['panels']['dense']['candidate'], 1)
        self.assertEqual(result['panels']['dense']['baseline_qualified'], 0)
        self.assertEqual(result['panels']['dense']['with_degraded'], 1)

    def test_ambiguity_and_all_original_alternatives_retained(self):
        ref = reference()
        result = trace.summarize_references([ref])
        self.assertEqual(result['records'][0]['original'], ref['original'])
        self.assertEqual([r['identity'] for r in result['records'][0]['measured_alternatives']],
                         ['0/bright:7', '0/bright:8'])

    def test_duplicate_sample_indices_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'):
            trace.summarize_references([reference(), reference()])

    def test_duplicate_alternative_identity_rejected(self):
        ref = reference()
        ref['measured_alternatives'].append(deepcopy(ref['measured_alternatives'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'):
            trace.summarize_references([ref])

    def test_original_assignment_cannot_be_substituted(self):
        ref = reference()
        ref['original_strict_assigned_identity'] = '0/bright:8'
        with self.assertRaisesRegex(ValueError, 'Original identity replaced'):
            trace.summarize_references([ref])

    def test_original_assignment_must_exist_in_alternatives(self):
        ref = reference()
        ref['measured_alternatives'].pop(0)
        with self.assertRaisesRegex(ValueError, 'Original assignment missing'):
            trace.summarize_references([ref])

    def test_no_assignment_cannot_claim_strict_hit(self):
        ref = reference()
        ref['original_strict_assigned_identity'] = None
        with self.assertRaisesRegex(ValueError, 'Unassigned strict hit'):
            trace.summarize_references([ref])

    def test_boolean_hit_required_not_truthy_integer(self):
        ref = reference()
        ref['original']['stages']['candidate'][0] = 1
        with self.assertRaisesRegex(ValueError, 'Stage hit must be boolean'):
            trace.summarize_references([ref])

    def test_unknown_strict_stage_rejected(self):
        ref = reference()
        del ref['original']['stages']['baseline_qualified']
        with self.assertRaisesRegex(ValueError, 'Unknown strict-stage schema'):
            trace.summarize_references([ref])

    def test_strict_assignment_cannot_use_unqualified_alternative(self):
        ref = reference()
        ref['measured_alternatives'][0]['original_qualified_moving'] = False
        with self.assertRaises(ValueError):
            trace.summarize_references([ref])

    def test_alternative_cannot_cross_reference_frame(self):
        ref = reference()
        ref['measured_alternatives'][0]['state_key'][1] = 21
        with self.assertRaises(ValueError):
            trace.summarize_references([ref])

    def test_alternative_identity_must_match_state_key(self):
        ref = reference()
        ref['measured_alternatives'][1]['state_key'][3] = 'bright:99'
        with self.assertRaises(ValueError):
            trace.summarize_references([ref])


class ArchiveControlTests(unittest.TestCase):
    def test_exact_scope_rows_and_zero_controls_preserved(self):
        result = trace.summarize_controls(controls_fixture())
        self.assertEqual(result['totals'], dict(baseline_measured=7, baseline_predicted=3,
                                             v36_measured=2, v36_predicted=1))
        self.assertEqual(len(result['windows']), 2)
        self.assertEqual(result['windows'][1]['baseline_measured'], 0)
        self.assertEqual(result['windows'][0]['frames_inclusive'], [10, 12])
        self.assertEqual(result['windows'][0]['crop_xywh'], [3, 4, 16, 16])
        self.assertIsNone(result['false_positive_rate'])
        self.assertTrue(all(not r['verified_airborne_negative'] for r in result['windows']))

    def test_right_order_is_not_used_as_scope_identity(self):
        fixture = controls_fixture()
        fixture['clips']['0126']['arms']['point_context']['provisional_controls'].reverse()
        self.assertEqual(trace.summarize_controls(fixture), trace.summarize_controls(controls_fixture()))

    def test_any_scope_change_rejected(self):
        for field, value in (('frames_inclusive', [10, 13]), ('crop_xywh', [4, 4, 16, 16]), ('label', 'changed')):
            with self.subTest(field=field):
                fixture = controls_fixture()
                fixture['clips']['0126']['arms']['point_context']['provisional_controls'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'Control scopes differ'):
                    trace.summarize_controls(fixture)

    def test_omitted_zero_control_rejected(self):
        fixture = controls_fixture()
        fixture['clips']['0126']['arms']['point_context']['provisional_controls'].pop()
        with self.assertRaisesRegex(ValueError, 'Control scopes differ'):
            trace.summarize_controls(fixture)

    def test_duplicate_control_scope_rejected(self):
        fixture = controls_fixture()
        rows = fixture['clips']['0126']['arms']['baseline']['provisional_controls']
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'):
            trace.summarize_controls(fixture)


class ArchiveFileSafetyTests(unittest.TestCase):
    def test_exclusive_write_and_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'record.json'
            trace.write(path, dict(finite=2, unknown=None))
            payload = path.read_bytes()
            self.assertEqual(json.loads(payload), dict(finite=2, unknown=None))
            self.assertEqual(trace.sha(path), hashlib.sha256(payload).hexdigest())
            with self.assertRaises(FileExistsError):
                trace.write(path, dict(replacement=True))
            self.assertEqual(path.read_bytes(), payload)

    def test_sha_rejects_relative_noncanonical_directory_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            path = directory / 'record.json'
            trace.write(path, {})
            link = directory / 'link.json'
            link.symlink_to(path)
            for invalid in (Path('relative.json'), directory, directory / 'sub' / '..' / 'record.json', link):
                with self.subTest(path=invalid):
                    with self.assertRaisesRegex(ValueError, 'Canonical regular file required'):
                        trace.sha(invalid)

    def test_nonfinite_json_not_silently_serialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                trace.write(Path(temporary).resolve() / 'record.json', {'invalid': float('nan')})


if __name__ == '__main__':
    unittest.main()
