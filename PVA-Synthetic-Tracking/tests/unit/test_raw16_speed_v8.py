from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from raw16_speed_v8_common import FeaturePixelsCache, numeric_comparison, reference_ast_unchanged, frozen_motion_module, estimate_identity
from tiny_target.motion import global_motion as gm, MotionCorrespondences
from tiny_target.point_filter_cuda import PointFilterCuda
from tiny_target.types import Frame, TimestampSource


class V8Tests(unittest.TestCase):
    def test_cache_uses_object_not_frame_index(self):
        a = Frame(np.zeros((3,5),np.uint16),0,0,'generated',16,TimestampSource.MANIFEST)
        b = replace(a,image=np.ones((3,5),np.uint16))
        original = Mock(side_effect=lambda frame,mapping:frame.image.copy())
        cache = FeaturePixelsCache(original)
        first = cache(a,'raw')
        self.assertIs(first,cache(a,'raw'))
        self.assertTrue(np.all(cache(b,'raw')==1))
        cache(b,'different'); cache(a,'raw')
        self.assertEqual((cache.hits,cache.misses),(1,4))
        cache.clear(); self.assertIsNone(cache.frame)
        cache(a,'raw'); self.assertEqual(original.call_count,5)

    def test_numeric_gate_does_not_hide_threshold_flips(self):
        a = np.array([[np.nextafter(np.float32(4),np.float32(0))]],np.float32)
        b = np.array([[4]],np.float32)
        result = numeric_comparison(a,b,b)
        self.assertTrue(result['numerical_screen_passed'])
        self.assertFalse(result['bit_exact'])
        self.assertEqual(result['threshold4_changes'],1)

    def test_nan_is_not_accepted(self):
        a = np.ones((2,3),np.float32)
        b = a.copy();b[0,0]=np.nan
        self.assertFalse(numeric_comparison(a,b,a)['numerical_screen_passed'])

    def test_frozen_default_and_batched_results_match(self):
        oracle=frozen_motion_module(ROOT/'results/tiny_target/raw16_motion_v6_20260915/verified_runtime.tgz')
        rng=np.random.default_rng(765)
        p=rng.uniform(20,600,(150,2)).astype(np.float32)
        q=p+np.array([3.,-2.],np.float32);q[:50,0]+=5
        pairs=MotionCorrespondences(p,q,np.ones(150,np.float32),np.zeros(150,np.float32),
            0,1,0,100000000,(800,800),(400,400),{}, {}, {})
        cfg=gm.GlobalMotionConfig(minimum_inlier_grid_coverage=0.)
        old=oracle.fit_global_motion(pairs,oracle.GlobalMotionConfig(minimum_inlier_grid_coverage=0.))
        for policy in ('reference','translation_batched_exact_v1'):
            got=gm.fit_global_motion(pairs,cfg,execution=policy)
            self.assertEqual(estimate_identity(old),estimate_identity(got))

    def test_similarity_cannot_silently_use_translation(self):
        with self.assertRaises(ValueError):
            gm.fit_global_motion(None,gm.GlobalMotionConfig(model='similarity'),execution='translation_batched_exact_v1')
        with self.assertRaises(ValueError): gm.fit_global_motion(None,execution='fast')

    def test_filter_rejects_closed_failed_cross_thread_and_bad_inputs(self):
        filt=object.__new__(PointFilterCuda)
        filt.handle=ctypes.c_void_p(1);filt.failed=False;filt.thread=threading.get_ident()
        filt.shape=(2,3);filt.lib=Mock()
        for image in (np.ones((2,3),np.float64),np.ones((3,2),np.float32),np.full((2,3),np.inf,np.float32)):
            with self.assertRaises(ValueError): filt(image)
        filt.failed=True
        with self.assertRaises(RuntimeError): filt(np.ones((2,3),np.float32))
        filt.failed=False;filt.thread=-1
        with self.assertRaises(RuntimeError): filt(np.ones((2,3),np.float32))
        filt.thread=threading.get_ident();filt.close();filt.close()
        filt.lib.seaqr_point_v8_destroy.assert_called_once()
        with self.assertRaises(RuntimeError): filt(np.ones((2,3),np.float32))

    def test_filter_shape_and_kernel_guard_before_library_load(self):
        with self.assertRaises(ValueError): PointFilterCuda((1,32000001),np.ones((9,9),np.float32),1)
        with self.assertRaises(ValueError): PointFilterCuda((2,3),np.ones((7,7),np.float32),1)
        with self.assertRaises(ValueError): PointFilterCuda((2,3),np.ones((9,9),np.float32),0)

    def test_experimental_adapter_preserves_background_and_mask_contract(self):
        import run_raw16_speed_v8 as experiment
        from raw16_speed_v8_common import cpu_filter, dense
        cfg=replace(dense.DenseScreenConfig(),background_execution='cuda_temporal_exact_v1',
                    per_frame_event_screen_enabled=False)
        class Background:
            def __init__(self,*args): self.count=0
            def step(self,image,valid):
                ready=self.count>=4;initial=self.count==0;self.count+=1
                return None if initial else (np.where(valid,image/16,0).astype(np.float32),valid & ready,ready)
            def close(self): pass
        class Filter:
            def __init__(self,shape,kernel,norm):self.kernel=kernel;self.norm=norm
            def __call__(self,image):return cpu_filter(image,self.kernel,self.norm)
            def close(self):pass
        for dtype in (np.uint16,np.float32):
            left=dense.DensePointScreener(cfg);right=dense.DensePointScreener(cfg)
            adapter=experiment.FilterExperiment(True)
            with patch('tiny_target.raw_background_cuda.RawBackgroundCuda',Background), \
                 patch.object(experiment,'RawBackgroundCuda',Background), \
                 patch.object(experiment,'PointFilterCuda',Filter):
                for index in range(7):
                    image=np.full((17,23),1000+index,dtype);image.flat[:4]=[0,1,65207,65535]
                    mask=np.ones(image.shape,bool);mask[8,8]=False
                    frame=Frame(image,index*1000000,index,'generated',16,TimestampSource.MANIFEST,valid_mask=mask)
                    left._events_for_frame_cuda(frame);adapter.events(right,frame)
                    self.assertEqual(left._availability,right._availability)
                    self.assertEqual(left._background_frame_count,right._background_frame_count)
                    self.assertEqual(left._frames_screened,right._frames_screened)
                    a,b=left._last_synthetic_frame,right._last_synthetic_frame
                    if a is None:self.assertIsNone(b)
                    else:
                        np.testing.assert_array_equal(a.response,b.response)
                        np.testing.assert_array_equal(a.valid_mask,b.valid_mask)
                        self.assertEqual(a.detection_ready,b.detection_ready)
                self.assertEqual(len(adapter.comparisons),6)
                adapter.close();left.close();right.close()

    def test_schedule_has_two_balanced_three_arm_rounds_per_clip(self):
        from batch_raw16_speed_v8 import schedule
        rows=schedule('timing')
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r['name'] for r in rows}),12)
        self.assertEqual({r['clip'] for r in rows},{'0029','0040'})
        for clip in ('0029','0040'):
            self.assertEqual([r['mode'] for r in rows if r['clip']==clip],
                             ['reference','cpu','combined','combined','cpu','reference'])

    def test_source_allowlist_rejects_before_any_artifact_or_media_read(self):
        from argparse import Namespace
        import run_raw16_speed_v8 as experiment
        with patch.object(experiment,'sha') as hashed:
            with self.assertRaises(ValueError):experiment.verify(Namespace(clip='unapproved'))
        hashed.assert_not_called()

    def test_decision_comparison_excludes_scores_but_not_position_or_support(self):
        from summarize_raw16_speed_v8 import without_scores
        a={'hits':[{'candidate':{'normalized_score_snr':4.,'discrete_position_xy_px':[5,7],
                                'temporal_support':{'supporting_frame_count':12}}}]}
        b={'hits':[{'candidate':{'normalized_score_snr':4.00001,'discrete_position_xy_px':[5,7],
                                'temporal_support':{'supporting_frame_count':12}}}]}
        self.assertEqual(without_scores(a),without_scores(b))
        b['hits'][0]['candidate']['temporal_support']['supporting_frame_count']=11
        self.assertNotEqual(without_scores(a),without_scores(b))
        b['hits'][0]['candidate']['temporal_support']['supporting_frame_count']=12
        b['hits'][0]['candidate']['discrete_position_xy_px'][0]=6
        self.assertNotEqual(without_scores(a),without_scores(b))

    def test_partial_journal_cannot_be_summarized_as_complete(self):
        import json
        import tempfile
        from batch_raw16_speed_v8 import schedule
        from summarize_raw16_speed_v8 import validated_schedule
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'timing_plan.json').write_text(json.dumps({'schedule':schedule('timing')}))
            (path/'timing_journal.jsonl').write_text('')
            with self.assertRaises(ValueError):validated_schedule(path,'timing')

    def test_cross_workspace_audit_preserves_every_array_and_number(self):
        from copy import deepcopy
        from compare_raw16_v8_audits import COUNTS,compare_records,LIBRARY_PATHS
        paths=sorted(LIBRARY_PATHS)
        rows=[{'stage':name,'value':{'library_path':paths[0],'array':{'sha256':'a'*64},'value':1.}}
              for name,count in COUNTS.items() for _ in range(count)]
        other=deepcopy(rows)
        for row in other:row['value']['library_path']=paths[1]
        self.assertTrue(compare_records(rows,other)['passed'])
        other[3]['value']['array']['sha256']='b'*64
        self.assertFalse(compare_records(rows,other)['passed'])
        self.assertFalse(compare_records(rows[:-1],rows[:-1])['passed'])
        other[3]['value']['library_path']='/unverified/library.so'
        with self.assertRaises(ValueError):compare_records(rows,other)


if __name__=='__main__': unittest.main()
