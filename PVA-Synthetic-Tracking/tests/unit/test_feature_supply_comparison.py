import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('supply_comparison', Path(__file__).parents[2]/'scripts/compare_discovery_feature_supply.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def refs(self):
        return [dict(frame_index=f, measurement_source_xy=[f, 10]) for f in (430, 431, 432)]

    def rows(self, **changes):
        result = []
        for f in (430, 431, 432):
            track = dict(track_id='dark:new', measured=True, qualified_moving=True,
                         measurement_source_xy=[f, 10], source_xy=[0,0])
            track.update(changes)
            result.append(dict(frame_index=f, segment=0, tracks=[track], detection_ready=True))
        return result

    def test_new_identity_can_preserve_pass(self):
        r = comparison.retention(self.rows(), self.refs())
        self.assertTrue(r['preservation_guard_passed'])
        self.assertEqual(r['complete_coherent_identities'], ['0/dark:new'])

    def test_unavailable_frames_cannot_count(self):
        rows = self.rows()
        rows[1]['detection_ready'] = False
        r = comparison.retention(rows, self.refs())
        self.assertFalse(r['preservation_guard_passed'])
        self.assertEqual(r['any_identity_matched_frames'], 2)

    def test_baseline_receipt_cannot_be_candidate(self):
        receipt = dict(clip='0240', source=dict(sha256='abc'),
                       schema='seaqr.discovery-pair.baseline.v1', algorithm_changed=False)
        comparison.validate_receipt(receipt, 'baseline', '0240', 'abc')
        with self.assertRaises(ValueError):
            comparison.validate_receipt(receipt, 'candidate', '0240', 'abc')

    def test_exact_candidate_receipt_and_gate_binding(self):
        import copy
        original = dict(harris_capacity_policy='legacy_default', feature_image_scale=.5,
                        minimum_accepted_features=30)
        receipt = dict(clip='0240', source=dict(sha256='abc'),
            schema='seaqr.discovery-feature-supply.v1', algorithm_changed=True,
            feature_algorithm_changed=True,
            candidate=dict(harris_gain=16, harris_capacity_policy='complete_grid', feature_image_scale=.5),
            feature_adapter=dict(original_motion_configuration=original,
                                 effective_motion_configuration=dict(original, harris_capacity_policy='complete_grid')))
        for key in ('detector_configuration_changed', 'tracker_configuration_changed',
                    'global_motion_gates_changed', 'production_promotion', 'annotations_supplied_to_detector',
                    'raw16_accessed', 'sealed_holdouts_accessed'):
            receipt[key] = False
        comparison.validate_receipt(receipt, 'candidate', '0240', 'abc')
        for change in (lambda r: r['candidate'].update(harris_gain=1),
                       lambda r: r.update(detector_configuration_changed=True),
                       lambda r: r['feature_adapter']['effective_motion_configuration'].update(minimum_accepted_features=29)):
            bad = copy.deepcopy(receipt)
            change(bad)
            with self.assertRaises(ValueError):
                comparison.validate_receipt(bad, 'candidate', '0240', 'abc')

    def test_coasts_filtered_positions_and_unqualified_do_not_count(self):
        for change in (dict(measured=False), dict(qualified_moving=False),
                       dict(measurement_source_xy=[0,0]), dict(track_id='bright:new')):
            with self.subTest(change=change):
                r = comparison.retention(self.rows(**change), self.refs())
                self.assertFalse(r['preservation_guard_passed'])
                self.assertEqual(r['any_identity_matched_frames'], 0)

    def test_split_identity_not_silently_counted_as_coherent(self):
        rows = self.rows()
        rows[1]['tracks'][0]['track_id'] = 'dark:other'
        r = comparison.retention(rows, self.refs())
        self.assertEqual(r['any_identity_matched_frames'], 3)
        self.assertEqual(r['best_coherent_identity_frames'], 2)
        self.assertFalse(r['preservation_guard_passed'])

    def test_segment_resets_break_identity(self):
        rows = self.rows()
        rows[-1]['segment'] = 1
        self.assertFalse(comparison.retention(rows, self.refs())['preservation_guard_passed'])

    def test_ambiguity_reported(self):
        rows = self.rows()
        rows[1]['tracks'].append(dict(rows[1]['tracks'][0], track_id='dark:second'))
        r = comparison.retention(rows, self.refs())
        self.assertTrue(r['preservation_guard_passed'])
        self.assertEqual(r['ambiguous_frames'], 1)

    def test_radius_boundary_inclusive(self):
        rows = self.rows()
        for row in rows:
            row['tracks'][0]['measurement_source_xy'][1] += 8
        self.assertTrue(comparison.retention(rows, self.refs())['preservation_guard_passed'])
        rows[-1]['tracks'][0]['measurement_source_xy'][1] += .001
        self.assertFalse(comparison.retention(rows, self.refs())['preservation_guard_passed'])

    def test_missing_or_duplicate_frame_rejected(self):
        for rows in (self.rows()[:-1], self.rows()+self.rows()[:1]):
            with self.assertRaises(ValueError):
                comparison.retention(rows, self.refs())

    def test_duplicate_identity_rejected(self):
        rows = self.rows()
        rows[0]['tracks'] *= 2
        with self.assertRaises(ValueError):
            comparison.retention(rows, self.refs())

    def test_null_and_nonfinite_actual_rejected(self):
        for value in (None, [float('nan'),10], [float('inf'),0]):
            with self.assertRaises(ValueError):
                comparison.retention(self.rows(measurement_source_xy=value), self.refs())


if __name__ == '__main__':
    unittest.main()
