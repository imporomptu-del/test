import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_accuracy_v43_verifier as runner


def record(available=True, ambiguous=False, support='abc'):
    value = dict(available=available, ambiguous=ambiguous, reasons=[], ambiguity_reasons=[],
                 common_support_sha256=support, advantage_stationary_minus_augmented=1 if available else None)
    return dict(clip='0029',frame_index=100,segment=0,track_id='bright:1',qualified_moving=True,
                grid_windows=['w'],geometry={'available':True},arms={a:dict(value) for a in runner.ARMS})


class RunnerTests(unittest.TestCase):
    def test_adapter_current_track_poison_and_prior_actual_coast_contract(self):
        row = dict(frame_index=1,timestamp_ns=100000000,segment=0,motion={'reset':False},
                   source_to_reference=np.eye(3).tolist(),tracks=[dict(track_id='bright:1',
                       segment=0,measured=True,measurement_source_xy=[1,2])])
        rows = [copy.deepcopy(row) for _ in range(9)]
        rows[1]['tracks'][0].update(measured=False,measurement_source_xy=None)
        rows[-1]['tracks'] = object()
        result = runner.causal_rows(rows)
        self.assertNotIn('tracks',result[-1])
        self.assertTrue(result[1]['tracks'][0]['predicted'])
        self.assertFalse(result[0]['tracks'][0]['predicted'])
        self.assertTrue(rows[0]['tracks'][0]['measured'])

    def test_adapter_rejects_malformed_original_bool_or_segment(self):
        row = dict(frame_index=1,timestamp_ns=1,segment=0,motion={'reset':False},
                   source_to_reference=np.eye(3).tolist(),tracks=[dict(track_id='x',
                       segment=0,measured=1,measurement_source_xy=[1,2])])
        with self.assertRaises(ValueError): runner.causal_rows([row]*9)
        row['tracks'][0].update(measured=True,segment=2)
        with self.assertRaises(ValueError): runner.causal_rows([row]*9)

    def test_pairing_does_not_compare_different_support_mses(self):
        a,b,c = record(),record(),record(False)
        b['arms']['combined']['common_support_sha256']='other'
        result=runner.paired([a,b,c],'baseline','combined')
        self.assertEqual(result['counts']['both_available_same_support'],1)
        self.assertEqual(result['counts']['both_available_different_support'],1)
        self.assertEqual(result['counts']['available_0_0'],1)
        self.assertEqual(result['same_support_advantage_change_quantiles'],[0,0,0])

    def test_unknown_and_ambiguous_counts_stay_separate(self):
        a,b,c=record(),record(ambiguous=True),record(False)
        d=record();d['geometry']['available']=False;d['arms']['baseline']=None
        result=runner.counts([a,b,c,d],'baseline')
        self.assertEqual(result['states'],4); self.assertEqual(result['score_available'],2)
        self.assertEqual(result['score_without_recorded_ambiguity'],1)
        self.assertEqual(result['score_with_ambiguity'],1)

    def test_original_strict_assignment_is_not_replaced_by_better_alternative(self):
        a,b=record(False),record();b['track_id']='bright:2'
        inventory=dict(sample_columns=['panel','clip_id','frame_index','stages'],
            panels={'p':{'samples':2,'original_strict_stage':'strict'}},
            samples=[['p','0029',100,{'strict':[True,'0/bright:1',['0/bright:1','0/bright:2']],
                'actual_measurement':[True,'0/bright:1',['0/bright:1','0/bright:2']]}],
                ['p','0029',101,{'strict':[False,None,[]],'actual_measurement':[False,None,[]]}]])
        result=runner.reference_report(inventory,[a,b])
        self.assertEqual(len(result['samples']),2)
        self.assertEqual(len(result['samples'][0]['measured_alternatives']),2)
        self.assertEqual(result['samples'][1]['measured_alternatives'],[])
        self.assertEqual(result['panels']['p']['arms']['combined']['same_assignment_score_available'],0)
        self.assertEqual(result['panels']['p']['arms']['combined']['samples'],2)

    def test_fresh_output_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):runner.run(Path(directory))


if __name__ == '__main__': unittest.main()
