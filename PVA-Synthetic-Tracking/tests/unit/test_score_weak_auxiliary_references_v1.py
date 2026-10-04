"""Generated fixtures only; no real experimental outcomes or media."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve()
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts' if (ROOT/'scripts/score_weak_auxiliary_references_v1.py').is_file() else HERE.parent))
import score_weak_auxiliary_references_v1 as m

WRAPPER=m.helper()
OLD=WRAPPER.historical()


def auxiliary(weak=True, frame=1):
    return dict(identity='0/bright:1',primary_track_id='bright:1',record_type='auxiliary_gap_support',
        ordinary_measurement=False,qualified_detection=False,physical_identity_verified=False,
        current_weak_observation=weak,prediction_from_weak=not weak,
        evidence_type='current_weak_observation' if weak else 'prediction_from_weak',
        source_xy=[100.,100.],origin_measurement_source_xy=[10.,20.],frame_index=frame,
        origin_frame_index=frame if weak else frame-1,current_weak_measurement_reference_xy=[10.,20.] if weak else None,
        strong_anchor_timestamp_ns=0)


def sample(panel='dense',assigned='0/bright:1'):
    value=dict(hit=assigned is not None,assigned_id=assigned,all_gated_ids=[assigned] if assigned else [])
    return dict(panel=panel,clip_id='0029',window_id='one',frame_index=0,source_xy=[10.,20.],
        position_uncertainty_px=1.,polarity='bright',saved={stage:deepcopy(value) for stage in OLD.STAGES})


def frames():
    track=dict(segment=0,track_id='bright:1',measured=True,qualified_moving=True,measurement_source_xy=[10.,20.],source_xy=[15.,25.])
    base=dict(frame_index=0,timestamp_ns=0,segment=0,tracks=[track],motion={},
        source_to_reference=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        coverage=dict(full_shape_hw=[3190,4784],configured_crop=None,native_pixel_sampling=True,
            detection_ready=True,warmup=False,searchable_pixels=123))
    main=dict(frame_index=0,timestamp_ns=0,segment=0,records=deepcopy(base['tracks']))
    aux=dict(frame_index=0,timestamp_ns=0,segment=0,capture_scheduled=True,
        auxiliary_records=[auxiliary(frame=0)],capture_calls=[dict(status='geometrically_complete_capture_supplied')],
        auxiliary_metrics=dict(prior_count=1,prior_strong_eligible_count=1,prepared_query_count=1,actual_provider_calls=1,
            decisions=[dict(status='weak_auxiliary_correction')],dropped_auxiliary=[]))
    return base,main,aux


class AuxiliaryScorerTests(unittest.TestCase):
    def test_raw_weak_point_not_posterior(self):
        result=m.separate_observations([auxiliary()])
        self.assertEqual(result['current_weak'][0]['xy'],[10.,20.])
        self.assertEqual(result['auxiliary_estimate'][0]['xy'],[100.,100.])
        self.assertEqual(result['prediction_from_weak'],[])

    def test_prediction_never_counts_old_measurement_again(self):
        result=m.separate_observations([auxiliary(False)])
        self.assertEqual(result['current_weak'],[])
        self.assertEqual(result['prediction_from_weak'][0]['xy'],[100.,100.])

    def test_ordinary_truth_duplicate_invalid_points_refused(self):
        for field in ('ordinary_measurement','qualified_detection','physical_identity_verified'):
            record=auxiliary();record[field]=True
            with self.assertRaises(ValueError):m.separate_observations([record])
        with self.assertRaises(ValueError):m.separate_observations([auxiliary(),auxiliary()])
        for value in (float('nan'),float('inf'),True):
            record=auxiliary();record['source_xy'][0]=value
            with self.assertRaises(ValueError):m.separate_observations([record])

    def test_invalid_temporal_classification_refused(self):
        for weak in (True,False):
            record=auxiliary(weak);record['origin_frame_index']=0 if weak else 1
            with self.assertRaises(ValueError):m.separate_observations([record])

    def run_rows(self, rows=None, samples=None):
        a,b,c=rows or frames()
        with patch.dict(OLD.COUNTS,{'0029':1}):
            return m.score_rows(OLD,WRAPPER,[a],[b],[c],'0029',samples or [sample()])

    def test_reference_overlap_and_separate_weak_evidence(self):
        before,after,records,work=self.run_rows(samples=[sample(),sample('pilot')])
        self.assertEqual(before,after)
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['auxiliary']['current_weak']['hit'] for r in records))
        self.assertFalse(any(r['auxiliary']['auxiliary_estimate']['hit'] for r in records))
        self.assertEqual(work['counts']['current_weak_records'],1)

    def test_primary_miss_stays_miss_even_with_weak_evidence(self):
        a,b,c=frames();a['tracks']=[];b['records']=[]
        _,_,records,_=self.run_rows((a,b,c),[sample(assigned=None)])
        self.assertTrue(records[0]['auxiliary']['current_weak']['hit'])
        self.assertFalse(records[0]['primary']['actual_measurement']['hit'])

    def test_primary_change_or_saved_assignment_change_fails(self):
        a,b,c=frames();b['records'][0]['measurement_source_xy'][0]+=1
        with self.assertRaisesRegex(ValueError,'Primary actual'):self.run_rows((a,b,c))
        with self.assertRaisesRegex(ValueError,'Historical assignment'):self.run_rows(samples=[sample(assigned=None)])

    def test_missing_capture_not_negative_and_no_fabricated_event(self):
        a,b,c=frames();c['auxiliary_records']=[]
        c['capture_calls']=[dict(status='no_geometrically_complete_cached_capture')]
        c['auxiliary_metrics']['decisions']=[dict(status='missing_capture')]
        _,_,records,work=self.run_rows((a,b,c))
        self.assertEqual(records[0]['unavailable_capture_queries'],1)
        self.assertEqual(work['weak_events'],[])

    def test_incomplete_or_unavailable_replay_fails(self):
        a,b,c=frames()
        with patch.dict(OLD.COUNTS,{'0029':1}):
            with self.assertRaisesRegex(ValueError,'Unequal'):m.score_rows(OLD,WRAPPER,[a],[],[c],'0029',[sample()])
        a['coverage']['detection_ready']=False
        with self.assertRaises(ValueError):self.run_rows((a,b,c))

    def test_pinned_reference_wrapper_refuses_change(self):
        with patch.object(m,'HELPER_SHA','0'*64):
            with self.assertRaises(ValueError):m.helper()


if __name__=='__main__':unittest.main()
