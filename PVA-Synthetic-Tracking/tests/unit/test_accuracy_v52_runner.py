from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import accuracy_v52_crossfit as model
import run_accuracy_v52_crossfit as runner


def fixture(kind='synthetic', frame=10, track='a'):
    xy=np.array([(x,y) for y in range(8,121,8) for x in range(8,121,8)
                 if 40<=max(abs(x-64),abs(y-64))<=56])
    slow=100+8*np.sin(xy[:,0]/13)+4*np.cos(xy[:,1]/11)
    current=slow+5
    metadata=(dict(family='fixture',background='textured',noise_level=0) if kind=='synthetic'
        else dict(state_key=['0029',frame,0,track],partition='evaluation',available=True,reasons=[]))
    return runner.plain(dict(input_id=f'{kind}_{frame}_{track}',kind=kind,metadata=metadata,
        points_xy=xy,median8=slow,median3=slow+2,current=current,
        clean_current_background=slow+5,contamination_mask=np.zeros(len(xy),dtype=bool)))


def forecast(row, split='left_right'):
    return model.crossfit(np.asarray(row['points_xy']),np.asarray(row['median8'],dtype=float),
        np.asarray(row['current'],dtype=float),split=split)


class RunnerTests(unittest.TestCase):
    def test_metrics_empty_is_undefined_not_zero_mae(self):
        a=runner.metrics([])
        self.assertEqual(a['count'],0)
        self.assertIsNone(a['mae_dn'])
        self.assertIsNone(a['max_absolute_error_dn'])

    def test_metrics_known_and_nonfinite_rejected(self):
        a=runner.metrics([-1,3])
        self.assertEqual(a['mae_dn'],2)
        self.assertEqual(a['absolute_sum_dn'],4)
        self.assertEqual(a['squared_sum_dn2'],10)
        for values in ([np.nan],[np.inf],[[1]], [1e308]):
            with self.assertRaises(ValueError): runner.metrics(values)

    def test_plain_roundtrip_preserves_forecast_fingerprint(self):
        f=forecast(fixture())
        restored=json.loads(json.dumps(runner.plain(f),allow_nan=False))
        self.assertEqual(model.crossfit_fingerprint(restored),f['crossfit_sha256'])

    def test_evaluate_preserves_forecast_and_matched_counts(self):
        row=fixture();f=forecast(row);before=runner.content_sha(f)
        result=runner.evaluate(row,f)
        self.assertEqual(before,runner.content_sha(f))
        for a in result['arms'].values():
            self.assertTrue(a['complete'])
            self.assertEqual({v['count'] for v in a['metrics'].values()},{144})
            self.assertLess(a['metrics']['corrected']['mae_dn'],1e-10)
            self.assertAlmostEqual(a['metrics']['median8']['mae_dn'],5)
            self.assertAlmostEqual(a['metrics']['median3']['mae_dn'],3)

    def test_scoring_missing_current_does_not_erase_prediction(self):
        row=fixture();f=forecast(row);row['current'][0]=None
        result=runner.evaluate(row,f)
        for a in result['arms'].values():
            self.assertFalse(a['complete']);self.assertEqual(a['prediction_available_count'],144)
            self.assertEqual(a['scored_count'],143)
            self.assertEqual(a['clean_truth_metrics']['corrected']['count'],144)
            self.assertNotIn(0,a['scored_indices'])

    def test_unavailable_fit_not_replaced_by_baseline(self):
        row=fixture();row['median8']=[100.]*144;f=forecast(row)
        result=runner.evaluate(row,f)
        self.assertEqual(result['baseline_all']['median8']['metrics']['count'],144)
        for a in result['arms'].values():
            self.assertFalse(a['complete']);self.assertEqual(a['scored_count'],0)
            self.assertIsNone(a['metrics']['corrected']['mae_dn'])

    def test_partially_available_fold_is_not_complete_packet(self):
        row=fixture();xy=np.asarray(row['points_xy'])
        for i in np.flatnonzero(xy[:,0]<64): row['current'][int(i)]=None
        f=forecast(row);result=runner.evaluate(row,f)
        for a in result['arms'].values():
            self.assertGreater(a['prediction_available_count'],0)
            self.assertLess(a['prediction_available_count'],144)
            self.assertEqual(a['scored_count'],0)
            self.assertFalse(a['complete'])

    def test_empty_window_not_complete(self):
        row=fixture()
        for k in ('points_xy','median8','median3','current','clean_current_background'): row[k]=[]
        f=model.crossfit(np.empty((0,2)),np.empty(0),np.empty(0))
        result=runner.evaluate(row,f)
        self.assertFalse(result['baseline_all']['median8']['complete'])
        self.assertFalse(result['arms']['huber']['complete'])

    def test_bad_fingerprint_and_shape_fail_closed(self):
        row=fixture();f=forecast(row)
        f['metadata']['core_pixels_accepted']=True
        with self.assertRaises(ValueError): runner.evaluate(row,f)
        f=forecast(row);row['current'].pop()
        with self.assertRaises(ValueError): runner.evaluate(row,f)

    def test_truth_and_metadata_changes_do_not_change_observed_scores(self):
        row=fixture();f=forecast(row);a=runner.evaluate(row,f)
        row['clean_current_background']=[999.]*144;row['metadata']['family']='different'
        b=runner.evaluate(row,f)
        for loss in runner.LOSSES:
            self.assertEqual(a['arms'][loss]['metrics'],b['arms'][loss]['metrics'])
            self.assertNotEqual(a['arms'][loss]['clean_truth_metrics'],b['arms'][loss]['clean_truth_metrics'])

    def test_combine_metrics_point_weighting_differs_from_packet_mean(self):
        out=runner.combine_metrics([runner.metrics([10]),runner.metrics([0,0,0])])
        self.assertEqual(out['conditional_point_mae_dn'],2.5)
        self.assertEqual(out['packet_mae']['mean'],5)

    def real_scores(self):
        rows=[fixture('real',10,'a'),fixture('real',10,'b'),fixture('real',11,'c')]
        scores=[runner.evaluate(r,forecast(r)) for r in rows]
        states=[dict(state_key=r['metadata']['state_key'],partition='evaluation',v50_status='background_measured') for r in rows]
        return rows,scores,states

    def test_missing_archive_invalidates_frame_but_keeps_denominator(self):
        rows,scores,states=self.real_scores()
        rows[0]['current']=[None]*144;scores[0]=runner.evaluate(rows[0],forecast(rows[0]))
        result=runner.aggregate(scores,states)
        self.assertEqual(result['frames_with_scored_archives'],2)
        self.assertEqual(result['arms']['huber']['complete_frame_count'],1)
        self.assertEqual(result['arms']['huber']['incomplete_archived_frame_count'],1)
        self.assertEqual(result['baseline_all']['median8']['complete_frame_count'],1)

    def test_history_unknown_no_archive_frame_remains_visible(self):
        rows,scores,states=self.real_scores()
        states.append(dict(state_key=['0029',99,0,'z'],partition='evaluation',v50_status='history_unknown'))
        result=runner.aggregate(scores,states)
        self.assertEqual(result['states'],4)
        self.assertEqual(result['frames_without_scored_archives'],1)
        self.assertEqual(result['original_status_counts']['history_unknown'],1)

    def test_complete_frame_macro_equal_weight_not_packet_pool(self):
        rows,scores,states=self.real_scores()
        for i,r in enumerate(scores):
            for loss in runner.LOSSES:
                r['arms'][loss]['metrics']['corrected']=runner.metrics([float([0,0,9][i])]*144)
        result=runner.aggregate(scores,states)
        self.assertEqual(result['arms']['huber']['matched_metrics']['corrected']['packet_mae']['mean'],3)
        self.assertEqual(result['arms']['huber']['complete_frame_mean_packet_mae']['corrected']['mean'],4.5)

    def test_unknown_only_and_embargo_bins_not_omitted(self):
        states=[dict(state_key=['0029',99,0,'z'],partition='evaluation',v50_status='history_unknown'),
                dict(state_key=['0029',90,0,'q'],partition='embargo',v50_status='embargo_not_scored')]
        summary=runner.summarize([],states)
        for split in runner.SPLITS:
            bins=summary['splits'][split]['nine_frame_bins']
            self.assertEqual(len(bins),2)
            self.assertEqual(bins['embargo_0029_10']['packet_records'],0)
            self.assertEqual(bins['evaluation_0029_11']['states'],1)

    def test_real_scores_do_not_claim_clean_background_truth(self):
        row=fixture('real');result=runner.evaluate(row,forecast(row))
        self.assertNotIn('clean_truth_metrics',result['arms']['huber'])

    def test_prepare_saves_all_forecasts_before_any_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve();rows=[fixture(),fixture(frame=11)]
            runner.write_lines(path/'inputs.jsonl',rows)
            with patch.object(runner,'evaluate',side_effect=AssertionError('scored too early')):
                runner.prepare_forecasts(rows,path)
            manifest=json.loads((path/'forecasts_frozen.json').read_text())
            self.assertFalse(manifest['evaluation_scoring_started'])
            self.assertEqual(manifest['forecast_count'],4)
            runner.check_bindings(manifest['files_sha256'])

    def test_file_writes_are_exclusive_and_hashes_detect_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'x.json'
            runner.write_json(path,{'a':1});bindings={str(path):runner.sha(path)}
            with self.assertRaises(FileExistsError):runner.write_json(path,{'a':2})
            runner.check_bindings(bindings)
            path.write_text('{}')
            with self.assertRaises(ValueError):runner.check_bindings(bindings)

    def test_output_outside_scope_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):runner.run(Path(directory)/'run')
            with patch.object(runner,'OUTPUT',Path(directory).parent):
                with self.assertRaises(ValueError):runner.run(Path(directory))

    def test_aggregate_overflow_rejected(self):
        with self.assertRaises(ValueError):
            runner.combine_metrics([runner.metrics([1e154]),runner.metrics([1e154])])
        with self.assertRaises(ValueError):runner.distribution([1e308,1e308])

    def test_truth_shape_explicitly_rejected(self):
        row=fixture();f=forecast(row);row['clean_current_background']=[0.]
        with self.assertRaisesRegex(ValueError,'truth shape'):runner.evaluate(row,f)

    def test_archive_membership_rejects_duplicate_missing_or_wrong_state(self):
        rows,scores,states=self.real_scores()
        for invalid in (scores[:-1],scores+[scores[0]]):
            with self.assertRaises(ValueError):runner.aggregate(invalid,states)
        wrong=deepcopy(scores);wrong[0]['metadata']['state_key'][3]='unknown'
        with self.assertRaises(ValueError):runner.aggregate(wrong,states)
        wrong=deepcopy(scores);wrong[0]['metadata']['partition']='calibration'
        with self.assertRaises(ValueError):runner.aggregate(wrong,states)

    def test_forecast_cartesian_membership_rejected_if_same_size_wrong_ids(self):
        inputs=[{'input_id':'a'},{'input_id':'b'}]
        records=[{'input_id':r['input_id'],'forecast':{'split':s}} for r in inputs for s in runner.SPLITS]
        runner.validate_forecast_membership(inputs,records)
        wrong=deepcopy(records);wrong[0]['input_id']='c'
        with self.assertRaises(ValueError):runner.validate_forecast_membership(inputs,wrong)
        with self.assertRaises(ValueError):runner.validate_forecast_membership(inputs,records[:-1])


if __name__=='__main__':unittest.main()
