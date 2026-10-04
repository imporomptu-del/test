"""Generated-patch stress evidence only; no real recording or journal scoring."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import stress_accuracy_v57 as stress


class GenerationTests(unittest.TestCase):
    def test_exact_fixed_case_order(self):
        self.assertEqual([c['case_id'] for c in stress.generated_cases()], [
            'pure_point_bright','pure_point_dark','pure_edge_bright','pure_edge_dark',
            'point_on_strong_edge_bright','point_on_strong_edge_dark',
            'extended_gaussian_bright','two_lobes_bright','quadratic_only'])

    def test_images_finite_native_sized_detached_and_readonly(self):
        cases=stress.generated_cases()
        for c in cases:
            self.assertEqual(c['patch'].shape,(25,25))
            self.assertEqual(c['patch'].dtype,np.float64)
            self.assertTrue(np.isfinite(c['patch']).all())
            self.assertFalse(c['patch'].flags.writeable)
            self.assertIsNone(c['physical_airborne_class'])
        self.assertFalse(np.shares_memory(cases[0]['patch'],cases[1]['patch']))

    def test_known_counterexample_matches_original_expression_exactly(self):
        y,x=np.mgrid[-12:13,-12:13].astype(np.float64)
        background=93.0+.7*x-1.3*y+.12*x*x-.05*x*y+.09*y*y
        point=np.exp(-(x*x+y*y)/(2.0*2.0**2))
        edge=np.tanh(x/2.)
        expected=background+6.0*point+20.0*edge
        case=stress.generated_cases()[4]
        np.testing.assert_array_equal(case['patch'],expected)
        self.assertTrue(case['known_v36_counterexample'])
        self.assertTrue(case['analytic_localized_source_present'])

    def test_generated_truth_only_describes_components(self):
        by_id={c['case_id']:c for c in stress.generated_cases()}
        self.assertEqual(len(by_id['two_lobes_bright']['generation']['source_components']),2)
        self.assertEqual(by_id['pure_edge_bright']['generation']['source_components'],[])
        self.assertEqual(by_id['extended_gaussian_bright']['generation']['source_components'][0]['sigma_xy_px'],[6.,1.5])
        self.assertFalse(by_id['pure_edge_bright']['analytic_localized_source_present'])

    def test_no_noise_quantization_or_clipping_claim(self):
        for c in stress.generated_cases():
            self.assertEqual(c['generation']['noise'],'none')
            self.assertEqual(c['generation']['quantization'],'none')
            self.assertEqual(c['generation']['clipping'],'none')
            self.assertFalse(c['generation']['physical_sensor_calibration'])


class StressResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=stress.run_stress()
        cls.cases={c['case_id']:c for c in cls.result['cases']}

    def test_mechanics_pass_but_promotion_false(self):
        self.assertTrue(self.result['completed'])
        self.assertTrue(self.result['mechanics_checks_passed'])
        self.assertTrue(self.result['known_counterexample_rejected'])
        self.assertFalse(self.result['promotion_allowed'])
        self.assertFalse(self.result['universal_source_preservation_demonstrated'])

    def test_pure_matching_points_remain_accepted(self):
        for name in ('pure_point_bright','pure_point_dark'):
            c=self.cases[name]
            self.assertGreater(c['generated_features']['point_minus_edge_fraction'],0)
            self.assertEqual(c['rejected_frame_indices'],[])
            self.assertTrue(all(r['verdict']['reason']=='point_preferred' for r in c['sequence']))

    def test_pure_edges_rejected_after_second_observation(self):
        for name in ('pure_edge_bright','pure_edge_dark'):
            c=self.cases[name]
            self.assertLess(c['generated_features']['point_minus_edge_fraction'],0)
            self.assertEqual([r['verdict']['accepted'] for r in c['sequence']],[True,False,False,False])
            self.assertEqual([r['verdict']['edge_streak_count'] for r in c['sequence']],[1,2,3,4])
            self.assertFalse(c['generated_source_present_veto'])

    def test_known_point_is_still_present_when_margin_negative(self):
        c=self.cases['point_on_strong_edge_bright'];f=c['generated_features']
        self.assertTrue(c['analytic_localized_source_present'])
        self.assertLess(f['point_minus_edge_fraction'],0)
        self.assertTrue(f['conditional_informative'])
        self.assertAlmostEqual(f['point_gain_after_edge_fraction'],1.,places=10)
        self.assertAlmostEqual(f['point_after_edge_amplitude_dn'],6.,places=8)
        self.assertEqual(f['point_after_edge_sigma_px'],2.)
        self.assertEqual(f['point_after_edge_offset_xy'],[0.,0.])

    def test_known_point_on_edge_is_vetoed_after_two_informative_edges(self):
        c=self.cases['point_on_strong_edge_bright']
        self.assertTrue(c['generated_source_present_veto'])
        self.assertEqual(c['rejected_frame_indices'],[1,2,3])
        self.assertEqual(c['sequence'][1]['verdict']['reason'],'edge_consecutive_rejected')
        self.assertEqual(c['sequence'][1]['verdict']['edge_streak_count'],2)
        self.assertTrue(c['sequence'][1]['verdict']['evidence_informative'])

    def test_dark_point_on_edge_counterexample_also_reported(self):
        c=self.cases['point_on_strong_edge_dark']
        self.assertLess(c['generated_features']['point_minus_edge_fraction'],0)
        self.assertAlmostEqual(c['generated_features']['point_after_edge_amplitude_dn'],6.,places=8)
        self.assertTrue(c['generated_source_present_veto'])

    def test_quadratic_unknown_not_negative(self):
        c=self.cases['quadratic_only']
        self.assertFalse(c['generated_features']['informative'])
        self.assertEqual(c['rejected_frame_indices'],[])
        for r in c['sequence']:
            self.assertEqual(r['verdict']['tier'],'unknown_measured')
            self.assertEqual(r['verdict']['edge_streak_count'],0)

    def test_correlated_sequences_not_independent_trials(self):
        self.assertTrue(all(c['repeated_patches_are_independent_trials'] is False for c in self.result['cases']))
        self.assertTrue(any('correlated' in line for line in self.result['limitations']))

    def test_sequence_positions_and_qualification_are_stipulated(self):
        for c in self.result['cases']:
            self.assertEqual(len(c['sequence']),4)
            for i,r in enumerate(c['sequence']):
                self.assertEqual(r['frame_index'],i)
                self.assertEqual(r['timestamp_ns'],i*100_000_000)
                self.assertTrue(r['assumed_baseline_track']['qualified_moving'])
                self.assertEqual(r['assumed_baseline_track']['measurement_source_xy'],[128.+i,128.])
        self.assertTrue(any('stipulated' in line for line in self.result['limitations']))

    def test_physical_class_remains_unknown(self):
        for c in self.result['cases']:
            self.assertIsNone(c['physical_airborne_class'])
            for row in c['sequence']:
                self.assertEqual(row['verdict']['physical_class'],'unknown')
                self.assertFalse(row['verdict']['airborne_confirmed'])

    def test_source_and_case_hashes_present(self):
        self.assertEqual(self.result['source_sha256']['scripts/accuracy_v36_context.py'],stress.V36_SHA)
        self.assertEqual(self.result['source_sha256']['tests/unit/test_accuracy_v36_context.py'],stress.V36_TEST_SHA)
        for c in self.result['cases']:
            self.assertEqual(len(c['patch_sha256_float64_le_y_major']),64)

    def test_no_real_scoring_or_accuracy_claim(self):
        for key in ('source_media_accessed','real_journals_scored','raw16_accessed','sealed_holdout_accessed','production_changed'):
            self.assertFalse(self.result[key])
        self.assertTrue(any('No precision, recall' in line for line in self.result['limitations']))

    def test_deterministic_serializable_report(self):
        self.assertEqual(self.result,stress.run_stress())
        json.dumps(self.result,allow_nan=False)


class GuardTests(unittest.TestCase):
    def test_changed_v36_hash_refused(self):
        with mock.patch.object(stress,'sha',return_value='0'*64),self.assertRaises(ValueError):
            stress.run_stress()

    def test_changed_default_budget_refused(self):
        altered=stress.PersistenceConfig(required_consecutive_edges=3)
        with mock.patch.object(stress,'PersistenceConfig',return_value=altered),self.assertRaises(ValueError):
            stress.run_stress()

    def test_failure_to_exercise_counterexample_fails_closed(self):
        actual_sequence=stress.sequence
        def always_accept(features,polarity):
            rows=actual_sequence(features,polarity)
            for row in rows:row['verdict']['accepted']=True
            return rows
        with mock.patch.object(stress,'sequence',side_effect=always_accept),self.assertRaises(ValueError):
            stress.run_stress()

    def test_sequence_does_not_mutate_features(self):
        f=stress.PointEdgeDiagnostic().measure(stress.generated_cases()[4]['patch'],'bright')
        before=deepcopy(f);stress.sequence(f,'bright');self.assertEqual(f,before)

    def test_unknown_polarity_refused(self):
        with self.assertRaises(ValueError):stress.sequence({},'unknown')

    def test_cli_writes_new_json(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'stress.json'
            stress.main(['--output',str(output)])
            value=json.loads(output.read_text())
            self.assertTrue(value['known_counterexample_rejected'])
            self.assertFalse(value['promotion_allowed'])

    def test_cli_refuses_overwrite_before_work(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'stress.json';output.write_text('original')
            with mock.patch.object(stress,'run_stress') as run,self.assertRaises(ValueError):
                stress.main(['--output',str(output)])
            run.assert_not_called();self.assertEqual(output.read_text(),'original')

    def test_cli_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'stress.json';output.symlink_to(Path(d)/'missing')
            with self.assertRaises(ValueError):stress.main(['--output',str(output)])


if __name__=='__main__':unittest.main()
