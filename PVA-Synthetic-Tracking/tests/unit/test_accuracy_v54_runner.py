from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_accuracy_v54_sources as runner


def fixture(amplitude=4.,stationary=False):
    guard=np.array([(x,y) for y in range(8,121,8) for x in range(8,121,8)
                    if 40<=max(abs(x-64),abs(y-64))<=56])
    core=np.array([(x,y) for y in range(52,77) for x in range(52,77)])
    template=np.maximum(1-np.abs(core[:,0]-64)/2,0)*np.maximum(1-np.abs(core[:,1]-64)/2,0)
    history=np.full((8,625),96.);on=history+amplitude*template if stationary else history.copy()
    spec=dict(case_id='fixture',stratum='factorial',condition='stable',motion='stationary' if stationary else 'appearing',
        amplitude=amplitude,background='constant',noise_level=0,missingness=None)
    if amplitude==0:spec['motion']='absent'
    return runner.plain(dict(case_id='fixture',spec=spec,observed=dict(guard_xy=guard,core_xy=core,
        guard_history=np.full((8,144),96.),guard_current=np.full(144,96.),
        core_history_on=on,core_history_off=history,core_current_on=96+amplitude*template,
        core_current_off=np.full(625,96.)),truth=dict(clean_current_background=np.full(625,96.),
        current_source=amplitude*template,source_template=template,guard_contamination_current=np.zeros(144))))


def scored(row):
    return runner.evaluate(row,runner.predict_input(row)['forecast'])


class SourceRunnerTests(unittest.TestCase):
    def test_new_source_exact_raw_and_paired_projection(self):
        s=scored(fixture())
        for m in runner.METHODS:
            a=s['methods'][m]['metrics']['amplitudes']
            self.assertTrue(a['available']);self.assertEqual(a['template_support_count'],9)
            self.assertEqual(a['raw_on_dn'],4);self.assertEqual(a['raw_off_dn'],0)
            self.assertEqual(a['paired_increment_dn'],4);self.assertEqual(a['raw_retention'],1)
            self.assertEqual(a['paired_retention'],1);self.assertEqual(a['raw_sign'],'same')

    def test_dark_source_normalized_retention_positive(self):
        s=scored(fixture(-4))
        for m in runner.METHODS:
            a=s['methods'][m]['metrics']['amplitudes']
            self.assertEqual(a['raw_on_dn'],-4);self.assertEqual(a['raw_retention'],1)
            self.assertEqual(a['paired_retention'],1);self.assertEqual(a['raw_sign'],'same')

    def test_stationary_source_temporal_absorption_not_unknown(self):
        s=scored(fixture(stationary=True))
        for m in runner.METHODS:
            a=s['methods'][m]['metrics']['amplitudes']
            self.assertTrue(a['available']);self.assertEqual(a['paired_increment_dn'],0)
            self.assertEqual(a['paired_retention'],0);self.assertEqual(a['paired_sign'],'zero')

    def test_guard_bias_can_flip_raw_without_erasing_paired_increment(self):
        row=fixture();row['observed']['guard_current']=[104.]*144
        s=scored(row)
        for m in runner.CORRECTED:
            a=s['methods'][m]['metrics']['amplitudes']
            self.assertEqual(a['raw_sign'],'reversed')
            self.assertEqual(a['paired_sign'],'same');self.assertEqual(a['paired_retention'],1)
            self.assertAlmostEqual(a['raw_on_dn'],4-8*4/2.25)

    def test_absent_has_no_retention_sign_or_target_amplitude_error(self):
        s=scored(fixture(0))
        for m in runner.METHODS:
            a=s['methods'][m]['metrics']['amplitudes']
            self.assertTrue(a['available']);self.assertFalse(a['source_present'])
            for k in ('raw_retention','paired_retention','oracle_retention','raw_sign','paired_sign',
                      'raw_amplitude_error_dn','paired_amplitude_error_dn'):
                self.assertIsNone(a[k])
            self.assertEqual(a['raw_off_dn'],0)

    def test_missing_target_pixel_forces_entire_projection_unknown(self):
        row=fixture();center=row['observed']['core_xy'].index([64,64])
        row['observed']['core_current_on'][center]=None;row['observed']['core_current_off'][center]=None
        s=scored(row)
        for m in runner.METHODS:
            r=s['methods'][m];self.assertEqual(len(r['scored_indices']),624)
            self.assertFalse(r['complete']);self.assertFalse(r['metrics']['amplitudes']['available'])
            self.assertFalse(r['metrics']['amplitudes']['oracle_available'])
            self.assertIsNone(r['metrics']['amplitudes']['paired_increment_dn'])
            self.assertEqual(r['prediction_available_on'],625)

    def test_missing_pixel_outside_source_keeps_projection_available(self):
        row=fixture();row['observed']['core_current_on'][0]=None;row['observed']['core_current_off'][0]=None
        s=scored(row)
        for m in runner.METHODS:
            self.assertFalse(s['methods'][m]['complete'])
            self.assertTrue(s['methods'][m]['metrics']['amplitudes']['available'])

    def test_on_off_common_mask_no_one_sided_imputation(self):
        row=fixture();row['observed']['core_current_on'][0]=None
        s=scored(row)
        for m in runner.METHODS:
            r=s['methods'][m];self.assertEqual(r['current_available_on'],624)
            self.assertEqual(r['current_available_off'],625)
            self.assertEqual(r['metrics']['point_errors']['off_residual']['count'],624)

    def test_missing_early_prior_retains_m3_own_support_but_matched_unknown(self):
        row=fixture();center=row['observed']['core_xy'].index([64,64])
        row['observed']['core_history_on'][0][center]=None;row['observed']['core_history_off'][0][center]=None
        s=scored(row)
        self.assertEqual(len(s['methods']['median3']['scored_indices']),625)
        self.assertEqual(len(s['methods']['median8']['scored_indices']),624)
        self.assertTrue(s['methods']['median3']['metrics']['amplitudes']['available'])
        self.assertTrue(s['methods']['median8']['metrics']['amplitudes']['oracle_available'])
        self.assertEqual(s['methods']['median8']['metrics']['amplitudes']['oracle_on_dn'],4)
        for m in runner.CORRECTED:
            a=s['matched'][m];self.assertEqual(len(a['scored_indices']),624)
            self.assertFalse(a['methods']['median3']['amplitudes']['available'])

    def test_unavailable_guard_does_not_remove_baseline_or_invent_offset(self):
        row=fixture();row['observed']['guard_current']=[None]*144;s=scored(row)
        self.assertEqual(len(s['methods']['median8']['scored_indices']),625)
        for m in runner.CORRECTED:
            a=s['methods'][m];self.assertEqual(a['prediction_available_on'],0)
            self.assertEqual(a['scored_indices'],[]);self.assertFalse(a['metrics']['amplitudes']['available'])
            self.assertTrue(a['metrics']['amplitudes']['oracle_available'])
            self.assertEqual(a['metrics']['amplitudes']['oracle_retention'],1)
            self.assertEqual(s['matched'][m]['methods']['median8']['point_errors']['on_residual']['count'],0)

    def test_truth_current_and_metadata_never_reach_predictor(self):
        row=fixture();a=runner.predict_input(row)['forecast']
        row['observed']['core_current_on']=[999.]*625;row['truth']['current_source']=[100.]*625
        row['spec']['motion']='invented_label';b=runner.predict_input(row)['forecast']
        self.assertEqual(a['prediction_sha256'],b['prediction_sha256'])

    def test_evaluate_does_not_mutate_frozen_prediction(self):
        row=fixture();f=runner.predict_input(row)['forecast'];before=runner.content_sha(f)
        runner.evaluate(row,f);self.assertEqual(before,runner.content_sha(f))

    def test_bad_fingerprint_and_method_membership_rejected(self):
        row=fixture();f=runner.predict_input(row)['forecast'];f['metadata']['tampered']=True
        with self.assertRaises(ValueError):runner.evaluate(row,f)
        f=runner.predict_input(row)['forecast'];del f['predictions']['median3']
        f['prediction_sha256']=runner.adapter.prediction_fingerprint(f)
        with self.assertRaisesRegex(ValueError,'membership'):runner.evaluate(row,f)

    def test_wrong_source_truth_or_template_rejected(self):
        for invalid in ('source','template','shape'):
            row=fixture();f=runner.predict_input(row)['forecast']
            if invalid=='source':row['truth']['current_source'][0]=1
            elif invalid=='template':row['truth']['source_template']=[0.]*625
            else:row['truth']['clean_current_background'].pop()
            with self.assertRaises(ValueError):runner.evaluate(row,f)

    def test_empty_error_metrics_unknown_not_zero(self):
        m=runner.point_metrics([]);self.assertEqual(m['count'],0)
        self.assertIsNone(m['mae_dn']);self.assertIsNone(m['rmse_dn'])

    def test_metric_arithmetic_validation(self):
        self.assertEqual(runner.point_metrics([-1,3])['mae_dn'],2)
        for a in ([np.nan],[np.inf],[[1]],[1e308]):
            with self.assertRaises(ValueError):runner.point_metrics(a)

    def test_sign_tolerance_is_explicit_symmetric_and_not_detector_threshold(self):
        self.assertEqual(runner.sign_category(1e-11,4),'zero')
        self.assertEqual(runner.sign_category(-1e-11,-4),'zero')
        self.assertEqual(runner.sign_category(-1,4),'reversed')
        self.assertEqual(runner.sign_category(-1,-4),'same')
        self.assertFalse(runner.evaluation_constants()['sign_tolerance_is_detection_threshold'])

    def test_distribution_empty_finite_quantiles_and_overflow(self):
        self.assertIsNone(runner.distribution([])['mean'])
        d=runner.distribution([0,10]);self.assertEqual(d['mean'],5);self.assertEqual(d['p10'],1)
        for a in ([np.nan],[1e308,1e308]):
            with self.assertRaises(ValueError):runner.distribution(a)

    def test_summary_strata_separate_and_source_absence_not_perfect_retention(self):
        row=fixture(0);a=scored(row)
        row2=fixture();row2['case_id']='availability';row2['spec'].update(case_id='availability',stratum='availability',missingness='fixture_missing')
        row2['observed']['guard_current']=[None]*144;b=scored(row2)
        s=runner.summarize([a,b],[row['spec'],row2['spec']])
        self.assertEqual(s['strata']['factorial']['case_count'],1)
        self.assertEqual(s['strata']['availability']['case_count'],1)
        self.assertEqual(s['strata']['factorial']['methods']['median8']['amplitudes']['paired_retention']['count'],0)
        self.assertEqual(s['strata']['availability']['methods'][runner.CORRECTED[0]]['present_amplitude_available_cases'],0)
        self.assertEqual(s['strata']['availability']['methods'][runner.CORRECTED[0]]['present_amplitude_unknown_cases'],1)
        self.assertEqual(s['strata']['availability']['methods'][runner.CORRECTED[0]]['present_oracle_available_cases'],1)
        self.assertEqual(set(s['groups']['missingness']),{'fixture_missing'})
        self.assertEqual(s['groups']['condition']['stable']['case_count'],1)

    def test_duplicate_missing_reordered_or_changed_specs_rejected(self):
        row=fixture();s=scored(row)
        with self.assertRaises(ValueError):runner.summarize([s,s],[row['spec'],row['spec']])
        with self.assertRaises(ValueError):runner.summarize([],[row['spec']])
        different=deepcopy(row['spec']);different['condition']='x'
        with self.assertRaises(ValueError):runner.summarize([s],[different])

    def test_source_manifest_explicit_and_minimal(self):
        self.assertEqual(len(runner.SOURCE_NAMES),11);self.assertEqual(len(set(runner.SOURCE_NAMES)),11)
        self.assertEqual(len(runner.INHERITED_SHA256),2)

    def test_exclusive_output_and_hash_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()/'a.json';runner.write_json(p,{'x':1});bindings={str(p):runner.sha(p)}
            runner.check_bindings(bindings)
            with self.assertRaises(FileExistsError):runner.write_json(p,{'x':2})
            p.write_text('{}')
            with self.assertRaises(ValueError):runner.check_bindings(bindings)

    def test_runner_rejects_existing_or_outside_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):runner.run(Path(d)/'x')
            with patch.object(runner,'OUTPUT',Path(d).parent):
                with self.assertRaises(ValueError):runner.run(Path(d))


if __name__=='__main__':unittest.main()
