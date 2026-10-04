"""Independent auditor mechanics; no source footage or real journals required."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[2] / 'scripts/audit_accuracy_v36.py'
SPEC = importlib.util.spec_from_file_location('accuracy_v36_independent_audit_under_test', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def track(x=20, y=30, measured=True, qualified=True, shape=True, tid='bright:1', segment=0):
    return dict(track_id=tid, segment=segment, measured=measured, qualified_moving=qualified,
                measurement_source_xy=[x, y] if measured else None,
                learning_shape_reference_xy=[[20, 30]] if shape else None,
                reference_xy=[99999, -99999], source_xy=[-99999, 99999])


def row(frame, tracks=None, segment=0, matrix=None):
    return dict(frame_index=frame, timestamp_ns=frame * 100000000, segment=segment,
                source_to_reference=matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                tracks=[track()] if tracks is None else tracks)


def sample_score(samples):
    return dict(positive_windows=[dict(window_id='sample', evidence=[dict(
        frame_index=frame, qualified_measured_hit=hit, assigned_track_id=identity)
        for frame, hit, identity in samples])])


class AuditV36Tests(unittest.TestCase):
    def test_independent_module_never_imports_experiment_policy_or_evaluator(self):
        tree = ast.parse(PATH.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn('accuracy_v36_policy', imports)
        self.assertNotIn('evaluate_accuracy_v36', imports)

    def test_exact_comparison_does_not_accept_boolean_as_count(self):
        with self.assertRaises(ValueError):
            audit.exact({'count': True}, {'count': 1}, 'typed count')
        with self.assertRaises(ValueError):
            audit.exact({'count': 1, 'extra': 0}, {'count': 1}, 'extra field')
        audit.exact({'a': 1, 'b': 2}, {'b': 2, 'a': 1}, 'mapping order')

    def test_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.decode(text)

    def test_all_arms_match_expected_dense_motion(self):
        engine = audit.IndependentHistory()
        for frame in range(8):
            output = engine.step(row(frame, [track(20 + frame * 4)]))
        state = output[0, 'bright:1']
        self.assertEqual(state['recent_hits'], 8)
        self.assertEqual(state['recent_excursion_px'], 28.0)
        self.assertEqual(state['last_measurement_age_frames'], 0)
        self.assertTrue(all(state['decisions'].values()))
        self.assertEqual(engine.native_grid_decision_differences, 0)

    def test_sparse_measurements_never_gain_fresh_support(self):
        engine = audit.IndependentHistory()
        for frame in range(29):
            output = engine.step(row(frame, [track(20 + frame * 4, measured=frame % 7 == 0)]))
            self.assertFalse(output[0, 'bright:1']['decisions']['recent_support'])
        self.assertTrue(output[0, 'bright:1']['decisions']['recent_excursion'])
        self.assertFalse(output[0, 'bright:1']['decisions']['combined'])

    def test_predictions_expire_support_without_creating_measurements(self):
        engine = audit.IndependentHistory()
        for frame in range(9):
            output = engine.step(row(frame, [track(20 + frame * 3, measured=frame < 5)]))
            if frame == 7:
                self.assertTrue(output[0, 'bright:1']['decisions']['recent_support'])
        self.assertFalse(output[0, 'bright:1']['decisions']['recent_support'])
        self.assertEqual(engine.measurements, 5)
        self.assertEqual(output[0, 'bright:1']['last_measurement_age_frames'], 4)

    def test_stopped_target_loses_recent_excursion(self):
        engine = audit.IndependentHistory()
        for frame in range(30):
            output = engine.step(row(frame, [track(20 + min(frame, 7) * 4)]))
        self.assertTrue(output[0, 'bright:1']['decisions']['recent_support'])
        self.assertFalse(output[0, 'bright:1']['decisions']['recent_excursion'])

    def test_source_mapping_removes_camera_translation(self):
        engine = audit.IndependentHistory()
        for frame in range(8):
            matrix = [[1, 0, -4 * frame], [0, 1, 0], [0, 0, 1]]
            output = engine.step(row(frame, [track(20 + frame * 4)], matrix=matrix))
        self.assertEqual(output[0, 'bright:1']['recent_excursion_px'], 0.0)
        self.assertFalse(output[0, 'bright:1']['decisions']['recent_excursion'])

    def test_homogeneous_coordinates_and_inclusive_threshold(self):
        engine = audit.IndependentHistory()
        for frame in range(5):
            output = engine.step(row(frame, [track(20 + frame * 3)], matrix=[[2, 0, 10], [0, 2, 20], [0, 0, 2]]))
        self.assertEqual(output[0, 'bright:1']['recent_excursion_px'], 12.0)
        self.assertTrue(output[0, 'bright:1']['decisions']['recent_excursion'])

    def test_segment_and_disappearance_reset_histories(self):
        engine = audit.IndependentHistory()
        for frame in range(8):
            engine.step(row(frame, [track(20 + frame * 4)]))
        output = engine.step(row(8, [track(measured=False, segment=1)], segment=1))
        self.assertEqual(output[1, 'bright:1']['recent_hits'], 0)
        engine.step(row(9, [], segment=1))
        output = engine.step(row(10, [track(segment=1)], segment=1))
        self.assertEqual(output[1, 'bright:1']['recent_hits'], 1)

    def test_shape_memory_only_changes_on_measurement(self):
        engine = audit.IndependentHistory()
        engine.step(row(0, [track(shape=True)]))
        output = engine.step(row(1, [track(measured=False, shape=False)]))
        self.assertTrue(output[0, 'bright:1']['bounded_shape'])
        engine.step(row(2, [track(shape=False)]))
        output = engine.step(row(3, [track(measured=False, shape=True)]))
        self.assertFalse(output[0, 'bright:1']['bounded_shape'])

    def test_unqualified_baseline_never_promoted(self):
        engine = audit.IndependentHistory()
        for frame in range(8):
            output = engine.step(row(frame, [track(20 + frame * 4, qualified=False)]))
        self.assertFalse(any(output[0, 'bright:1']['decisions'].values()))

    def test_causal_prefix_is_identical_with_longer_future(self):
        supplied = [row(frame, [track(20 + frame * 4)]) for frame in range(12)]
        original = copy.deepcopy(supplied)
        short, long = audit.IndependentHistory(), audit.IndependentHistory()
        a = [short.step(r) for r in supplied[:5]]
        b = [long.step(r) for r in supplied]
        self.assertEqual(a, b[:5])
        self.assertEqual(supplied, original)

    def test_invalid_or_skipped_frame_timestamp_is_rejected(self):
        for bad in (row(1), {**row(0), 'timestamp_ns': 1}):
            with self.assertRaises(ValueError):
                audit.IndependentHistory().step(bad)

    def test_duplicate_identity_or_prediction_measurement_rejected(self):
        with self.assertRaises(ValueError):
            audit.IndependentHistory().step(row(0, [track(), track()]))
        bad = track(measured=False)
        bad['measurement_source_xy'] = [1, 2]
        with self.assertRaises(ValueError):
            audit.IndependentHistory().step(row(0, [bad]))

    def test_non_native_observation_grid_is_explicitly_out_of_scope(self):
        with self.assertRaises(ValueError):
            audit.IndependentHistory().step(row(0, [track(20.25)]))

    def test_native_roundtrip_failure_is_detected_without_rounding_policy(self):
        engine = audit.IndependentHistory()
        for frame in range(5):
            x = 20 + frame * 3 - (1e-10 if frame == 4 else 0)
            output = engine.step(row(frame, [track(x)]))
        self.assertFalse(output[0, 'bright:1']['decisions']['recent_excursion'])
        self.assertEqual(engine.native_grid_decision_differences, 2)
        self.assertEqual(engine.noninteger_roundtrips, 1)

    def test_retention_rejects_lost_hit_even_when_totals_equal(self):
        before = sample_score([(1, True, 'a'), (2, False, None)])
        after = sample_score([(1, False, None), (2, True, 'a')])
        result = audit.retention(before, after)
        self.assertFalse(result['no_new_misses'])
        self.assertEqual(result['lost_visible_samples'], [{'window': 'sample', 'frame': 1}])
        self.assertEqual(result['baseline_hits'], result['candidate_hits'])

    def test_retention_reports_changed_identity_and_denominator(self):
        before = sample_score([(1, True, 'a')])
        result = audit.retention(before, sample_score([(1, True, 'b')]))
        self.assertEqual(len(result['changed_assignments']), 1)
        with self.assertRaises(ValueError):
            audit.retention(before, sample_score([]))
        with self.assertRaises(ValueError):
            audit.retention(before, sample_score([(1, True, 'a'), (1, True, 'a')]))

    def test_integrity_rechecks_changed_input_and_rejects_wrong_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_text('{}')
            registry = audit.Integrity()
            expected = registry.bind(path)
            registry.recheck()
            with self.assertRaises(ValueError):
                registry.bind(path, '0' * 64)
            path.write_text('{"changed":true}')
            self.assertNotEqual(audit.digest(path), expected)
            with self.assertRaises(ValueError):
                registry.recheck()

    def test_integrity_rejects_symlink_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_text('{}')
            link = Path(directory) / 'alias.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                audit.digest(link)
            with self.assertRaises(ValueError):
                audit.digest(Path(directory) / 'absent.json')

    def test_unapproved_experiment_path_rejected_before_any_read(self):
        with self.assertRaises(ValueError):
            audit.check_scope(Path('/not-an-approved-experiment'), audit.Integrity())

    def test_provisional_crop_edges_have_explicit_half_open_bounds(self):
        self.assertTrue(audit.inside([10, 20], [10, 20, 5, 5]))
        self.assertFalse(audit.inside([15, 20], [10, 20, 5, 5]))
        self.assertFalse(audit.inside([10, 25], [10, 20, 5, 5]))

    def total_fixture(self):
        results = {}
        for clip in audit.COUNTS:
            dense = {'0029': (146, 146), '0126': (139, 138)}.get(clip, (0, 0))
            pilot = {'0029': 16, '0126': 12}.get(clip, 0)
            anchors = {'0029': 18, '0126': 6}.get(clip, 0)
            results[clip] = dict(arms={})
            for arm in audit.ARMS:
                results[clip]['arms'][arm] = dict(retention={
                    'dense': dict(samples=dense[0], candidate_hits=dense[1], lost_visible_samples=[], changed_assignments=[]),
                    'pilot': dict(samples=pilot, candidate_hits=pilot, lost_visible_samples=[], changed_assignments=[])},
                    required_anchor_evidence=[dict(ids=['0/bright:1']) for _ in range(anchors)],
                    provisional_control_measured_states=70 if clip == '0126' else 0)
        return results

    def test_global_reference_counts_and_baseline_scores_are_mandatory(self):
        fixture = self.total_fixture()
        result = audit.global_totals(fixture)
        self.assertEqual(result['baseline']['dense']['hits'], 284)
        self.assertEqual(result['baseline']['pilot']['hits'], 28)
        self.assertEqual(result['baseline']['anchors']['hits'], 24)
        for kind in ('dense', 'pilot'):
            bad = copy.deepcopy(fixture)
            bad['0126']['arms']['combined']['retention'][kind]['samples'] -= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.global_totals(bad)
        bad = copy.deepcopy(fixture)
        bad['0126']['arms']['recent_support']['required_anchor_evidence'].pop()
        with self.assertRaises(ValueError):
            audit.global_totals(bad)

    def test_baseline_miss_and_control_count_changes_cannot_pass(self):
        for kind in ('dense', 'pilot'):
            fixture = self.total_fixture()
            fixture['0126']['arms']['baseline']['retention'][kind]['candidate_hits'] -= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.global_totals(fixture)
        fixture = self.total_fixture()
        fixture['0126']['arms']['baseline']['required_anchor_evidence'][0]['ids'] = []
        with self.assertRaises(ValueError):
            audit.global_totals(fixture)
        fixture = self.total_fixture()
        fixture['0126']['arms']['baseline']['provisional_control_measured_states'] = 69
        with self.assertRaises(ValueError):
            audit.global_totals(fixture)


if __name__ == '__main__':
    unittest.main()
