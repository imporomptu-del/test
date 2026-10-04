"""Synthetic accounting and causal-boundary contracts; no media access."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_accuracy_v42_localized as runner


def fixture():
    tracks = [dict(segment=0, track_id=f'bright:{i}', source_xy=[i, 100],
                   qualified_moving=i < 381) for i in range(1367)]
    grid = [dict(clip_id='0029', window_id=f'w{i}', frames=[])
            for i in range(108)]
    grid[0]['frames'] = [dict(frame_index=100, actual_measurements=tracks)]
    columns = ['panel', 'clip_id', 'window_id', 'frame_index', 'source_xy',
               'position_uncertainty_px', 'polarity', 'saved_evidence_index', 'stages']
    inventory = dict(sample_columns=columns,
        panels={'dense': {'samples': 2}, 'pilot': {'samples': 1}},
        samples=[
            ['dense', '0029', 'ref', 100, [0, 100], 5, 'bright', 0,
             {'actual_measurement': [True, '0/bright:0', ['0/bright:0'], False, 0]}],
            ['dense', '0029', 'miss', 200, [1, 2], 5, 'bright', 1,
             {'actual_measurement': [False, None, [], False, None]}],
            ['pilot', '0126', 'ref', 200, [20, 30], 5, 'bright', 2,
             {'actual_measurement': [True, '0/bright:new', ['0/bright:new'], False, 0]}]],
        original_gated_measured_states=[
            ['0029', 100, 0, 'bright:0', [0, 100], True, [0], [0]],
            ['0126', 200, 0, 'bright:new', [20, 30], False, [2], []]])
    return inventory, grid


class RunnerTests(unittest.TestCase):
    def test_union_preserves_grid_and_all_reference_alternatives(self):
        inventory, grid = fixture()
        result = runner.make_selection(inventory, grid)
        self.assertEqual(result['unique_states'], 1368)
        self.assertEqual(result['grid_actual_states'], 1367)
        self.assertEqual(result['grid_strict_states'], 381)
        self.assertEqual(result['reference_panel_denominators'], {'dense': 2, 'pilot': 1})
        self.assertEqual(result['needed_source_frames']['0029'], list(range(92, 101)))
        self.assertEqual(result['needed_source_frames']['0126'], list(range(192, 201)))
        self.assertEqual(result['needed_source_frames']['0082'], [])
        shared = next(s for s in result['states'] if s['track_id'] == 'bright:0')
        self.assertEqual(shared['grid_windows'], ['w0'])
        self.assertEqual(shared['reference_samples'], [0])

    def test_reference_sample_overlap_deduplicates_only_state(self):
        inventory, grid = fixture()
        inventory['original_gated_measured_states'][0][6] = [0, 1]
        selected = runner.make_selection(inventory, grid)
        self.assertEqual(selected['unique_states'], 1368)
        shared = next(s for s in selected['states'] if s['track_id'] == 'bright:0')
        self.assertEqual(shared['reference_samples'], [0, 1])

    def test_conflicting_provenance_and_changed_grid_denominator_fail(self):
        inventory, grid = fixture()
        inventory['original_gated_measured_states'][0][4] = [-1, 100]
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.make_selection(inventory, grid)
        inventory, grid = fixture()
        grid.pop()
        with self.assertRaisesRegex(ValueError, 'denominator'):
            runner.make_selection(inventory, grid)

    def test_preselected_audit_keys_are_deterministic_and_scores_not_inputs(self):
        inventory, grid = fixture()
        original = copy.deepcopy((inventory, grid))
        first = runner.make_selection(inventory, grid)
        second = runner.make_selection(inventory, grid)
        self.assertEqual(first, second)
        self.assertEqual((inventory, grid), original)
        self.assertEqual(len(first['audit_state_keys']), 3)
        self.assertIn(['0126', 200, 0, 'bright:new'], first['audit_state_keys'])

    def test_original_misses_and_unqualified_alternatives_stay_in_reference_report(self):
        inventory, grid = fixture()
        records = runner.make_selection(inventory, grid)['states']
        for record in records:
            record.update(geometry={'available': False, 'reasons': ['short_history']},
                          localized=None, predicted_to_actual_distance_px=None)
        report = runner.reference_report(inventory, records)
        self.assertEqual(len(report['samples']), 3)
        self.assertEqual(report['samples'][1]['measured_alternatives'], [])
        self.assertFalse(report['samples'][2]['measured_alternatives'][0]['qualified_moving'])
        self.assertFalse(report['samples'][0]['measured_alternatives'][0]['geometry']['available'])
        self.assertEqual(report['panels'], inventory['panels'])
        self.assertTrue(report['no_filter_applied'])

    def test_inference_boundary_strips_current_track_and_truth_fields(self):
        current = dict(frame_index=20, timestamp_ns=2000000000, segment=0,
                       motion={'reset': False}, source_to_reference=np.eye(3).tolist(),
                       tracks=object(), truth_xy=[999, 999], velocity_xy=[999, 999])
        rows = [dict(frame_index=i, timestamp_ns=i*100000000, segment=0,
                     source_to_reference=np.eye(3).tolist(), motion={'reset': False},
                     tracks=[dict(track_id='bright:1', segment=0, measured=i != 2,
                                  measurement_source_xy=[i, 4] if i != 2 else None)])
                for i in range(8)] + [current]
        state = dict(clip='0029', frame_index=20, segment=0, track_id='bright:1',
                     actual_source_xy=[3, 4], qualified_moving=True, save_audit_inputs=False)
        geometry = dict(available=True, reasons=[], geometry={'predicted_source_xy': [0, 0]},
                        current129=np.ones((129, 129)), history129=np.ones((8, 129, 129)),
                        prior_centers_xy=np.full((8, 2), np.nan), predicted_offset_xy=np.zeros(2))
        localized = dict(available=False, ambiguous=False, reasons=['short_template'])
        with mock.patch.object(runner, 'prepare_history', return_value=geometry) as prepare, \
             mock.patch.object(runner, 'evaluate_localized', return_value=localized) as evaluate:
            result = runner.run_state(state, ['images'], rows, Path('/unused'))
        passed = prepare.call_args.args[1]
        self.assertEqual(set(passed[-1]), {'frame_index', 'timestamp_ns', 'segment',
                                         'motion', 'source_to_reference'})
        self.assertFalse(passed[0]['tracks'][0]['predicted'])
        self.assertTrue(passed[2]['tracks'][0]['predicted'])
        self.assertIsNone(passed[2]['tracks'][0]['measurement_source_xy'])
        self.assertEqual(evaluate.call_args.args[2], [None]*8)
        self.assertEqual(evaluate.call_args.args[4], 'bright')
        self.assertEqual(result['predicted_to_actual_distance_px'], 5)
        self.assertIn('tracks', rows[-1])  # no mutation
        self.assertNotIn('current129', result['geometry'])

    def test_unavailable_geometry_never_calls_core_or_removes_record(self):
        row = dict(frame_index=1, timestamp_ns=1, segment=0, motion={'reset': False},
                   source_to_reference=[], tracks=[])
        state = dict(clip='0029', frame_index=1, segment=0, track_id='bright:1',
                     actual_source_xy=[0, 0], qualified_moving=True, save_audit_inputs=False)
        with mock.patch.object(runner, 'prepare_history', return_value={
            'available': False, 'reasons': ['short_history'], 'geometry': {}}), \
             mock.patch.object(runner, 'evaluate_localized') as evaluate:
            result = runner.run_state(state, [], [row]*9, Path('/unused'))
        evaluate.assert_not_called()
        self.assertEqual(result['track_id'], state['track_id'])
        self.assertIsNone(result['localized'])

    def test_real_journal_schema_end_to_end_on_synthetic_native_images(self):
        y, x = np.indices((200, 220))
        background = 50 + 20*(x >= 92) + 8*np.sin(y/12)
        images = [(background + 30*np.exp(-((x-(60+4*i))**2+(y-100)**2)/2)).astype(np.uint8)
                  for i in range(9)]
        rows = [dict(frame_index=20+i, timestamp_ns=(20+i)*100000000, segment=0,
                     motion={'reset': False}, source_to_reference=np.eye(3).tolist(),
                     tracks=[dict(track_id='bright:1', segment=0, measured=True,
                                  measurement_source_xy=[60+4*i, 100])]) for i in range(9)]
        rows[-1]['tracks'] = object()  # adapter must not even iterate current tracks
        state = dict(clip='0029', frame_index=28, segment=0, track_id='bright:1',
                     actual_source_xy=[92, 100], qualified_moving=True, save_audit_inputs=False)
        result = runner.run_state(state, images, rows, Path('/unused'))
        self.assertTrue(result['geometry']['available'])
        self.assertTrue(result['localized']['available'])
        self.assertLess(result['predicted_to_actual_distance_px'], 1e-10)
        self.assertLess(result['localized']['mse_augmented'], result['localized']['mse_stationary']*.1)

    def test_malformed_journal_measurement_boolean_fails_closed(self):
        row = dict(frame_index=1, timestamp_ns=100000000, segment=0, motion={'reset': False},
                   source_to_reference=np.eye(3).tolist(), tracks=[dict(
                       track_id='bright:1', segment=0, measured=1, measurement_source_xy=[1, 1])])
        with self.assertRaisesRegex(ValueError, 'Malformed original journal'):
            runner.run_state({}, [], [row]*9, Path('/unused'))

    def test_summary_counts_unknowns_and_ambiguity_separately(self):
        records = [dict(qualified_moving=True, geometry={'available': True, 'reasons': []},
                        localized={'available': False, 'ambiguous': True,
                            'reasons': ['template'], 'ambiguity_reasons': ['overlap']}),
                   dict(qualified_moving=False, geometry={'available': False, 'reasons': ['history']},
                        localized=None)]
        result = runner.state_summary(records)
        self.assertEqual(result['states'], 2)
        self.assertEqual(result['geometry_available'], 1)
        self.assertEqual(result['localized_available'], 0)
        self.assertEqual(result['ambiguous'], 1)
        self.assertEqual(result['ambiguity_reason:overlap'], 1)

    def test_fresh_output_only_and_json_never_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): runner.run(Path(directory))
            path = Path(directory)/'evidence.json'
            runner.write(path, runner.clean({'values': np.array([1, np.nan, np.inf])}))
            self.assertEqual(json.loads(path.read_text())['values'], [1, None, None])
            with self.assertRaises(FileExistsError): runner.write(path, {})


if __name__ == '__main__':
    unittest.main()
