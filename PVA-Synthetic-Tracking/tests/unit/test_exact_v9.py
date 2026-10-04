from __future__ import annotations
from argparse import Namespace
from copy import deepcopy
from contextlib import ExitStack
import ctypes
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tiny_target.point_filter_fft_exact import PointFilterFftExact
from tiny_target.warp_translation_cuda import WarpTranslationCuda,translation_inverse,cubic_table
from raw16_speed_v8_common import cpu_filter,filter_parameters
from exact_v9_common import normalize_audit,LIBRARY_PATHS,ExactExecution
from batch_exact_v9 import schedule

class ExactV9Tests(unittest.TestCase):
    def test_generated_fft_tiles_and_threads(self):
        if 'SSE3' in cv2.getCPUFeaturesLine():self.skipTest('Reference uses spatial dispatcher here')
        kernel,norm=filter_parameters()
        for shape in ((1,1),(3,7),(67,100),(248,248),(249,249),(257,499),(497,513)):
            image=np.random.default_rng(7551).normal(0,3,shape).astype(np.float32)
            expected=cpu_filter(image,kernel,norm);original=image.copy()
            for n in (1,2,4):
                dev=PointFilterFftExact(shape,kernel,norm,workers=n)
                first=dev(image);saved=first.copy();second=dev(image*0)
                self.assertEqual(first.tobytes(),expected.tobytes())
                self.assertEqual(first.tobytes(),saved.tobytes());self.assertFalse(np.shares_memory(first,second))
                raw=dev.correlate(image)
                self.assertEqual(raw.tobytes(),cv2.filter2D(image,cv2.CV_32F,kernel,borderType=cv2.BORDER_CONSTANT).tobytes())
                dev.close();dev.close()
                with self.assertRaises(RuntimeError):dev(image)
            self.assertEqual(image.tobytes(),original.tobytes())

    def test_fft_contract_guards(self):
        if 'SSE3' in cv2.getCPUFeaturesLine():self.skipTest('Reference uses spatial dispatcher here')
        k,n=filter_parameters()
        for workers in (True,0,3,5):
            with self.assertRaises(ValueError):PointFilterFftExact((3,7),k,n,workers=workers)
        with self.assertRaises(ValueError):PointFilterFftExact((3,7),k[:3],n)
        with self.assertRaises(ValueError):PointFilterFftExact((3,7),k,0)
        with patch.object(cv2,'getCPUFeaturesLine',return_value='SSE SSE2 SSE3'):
            with self.assertRaises(RuntimeError):PointFilterFftExact((3,7),k,n)
        dev=PointFilterFftExact((3,7),k,n)
        for a in (np.zeros((3,7),np.float64),np.zeros((7,3),np.float32),np.full((3,7),np.nan,np.float32)):
            with self.assertRaises(ValueError):dev(a)
        dev.owner=-1
        with self.assertRaises(RuntimeError):dev(np.zeros((3,7),np.float32))
        dev.owner=threading.get_ident();dev.close()

    def test_fft_failure_is_sticky_and_workers_join(self):
        if 'SSE3' in cv2.getCPUFeaturesLine():self.skipTest('Reference uses spatial dispatcher here')
        k,n=filter_parameters();dev=PointFilterFftExact((17,19),k,n)
        with patch.object(dev,'_tile',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):dev(np.ones((17,19),np.float32))
        self.assertTrue(dev.failed);self.assertIsNone(dev.pool)
        with self.assertRaises(RuntimeError):dev(np.ones((17,19),np.float32))
        dev.close()

    def test_translation_rejects_affine_and_perspective(self):
        for i,j,v in ((0,0,1.00001),(0,1,.00001),(2,0,.00001),(2,2,2.)):
            m=np.eye(3);m[i,j]=v
            with self.assertRaises(ValueError):translation_inverse(m)
        for m in (np.eye(3,dtype=np.float32),np.eye(4),np.full((3,3),np.nan)):
            with self.assertRaises(ValueError):translation_inverse(m)
        m=np.eye(3);m[:2,2]=[.015625,-120.75]
        np.testing.assert_array_equal(translation_inverse(m),cv2.invert(m)[1])

    def test_cubic_table_covers_all_phases(self):
        table=cubic_table();self.assertEqual(table.shape,(32,32,16))
        self.assertEqual(table.dtype,np.float32);self.assertTrue(np.isfinite(table).all())
        expected=np.zeros(16,np.float32);expected[5]=1
        np.testing.assert_array_equal(table[0,0],expected)

    def test_warp_closed_failed_and_thread_guards(self):
        dev=object.__new__(WarpTranslationCuda);dev.handle=ctypes.c_void_p(1)
        dev.failed=False;dev.thread=threading.get_ident();dev.shape=(3,7);dev.lib=Mock()
        image=np.ones((3,7),np.float32);mask=np.ones((3,7),np.uint8)
        for bad in (mask.astype(bool),mask*2,np.ones((7,3),np.uint8)):
            with self.assertRaises(ValueError):dev(image,bad,np.eye(3))
        dev.thread=-1
        with self.assertRaises(RuntimeError):dev(image,mask,np.eye(3))
        dev.thread=threading.get_ident();dev.failed=True
        with self.assertRaises(RuntimeError):dev(image,mask,np.eye(3))
        dev.close();dev.close();dev.lib.seaqr_warp_v9_destroy.assert_called_once()

    def test_audit_normalizes_execution_only(self):
        a=[dict(stage='full_resolution_warp',value={'backend':'opencv_cpu','image':{'sha256':'a'*64},'number':1.}),
           dict(stage='cuda_shift_stack',value={'library_path':sorted(LIBRARY_PATHS)[0]})]
        b=deepcopy(a);b[0]['value']['backend']='cuda_translation_reference_v9';b[1]['value']['library_path']=sorted(LIBRARY_PATHS)[1]
        self.assertEqual(normalize_audit(a),normalize_audit(b))
        b[0]['value']['number']=2.;self.assertNotEqual(normalize_audit(a),normalize_audit(b))
        b[0]['value']['backend']='unknown'
        with self.assertRaises(ValueError):normalize_audit(b)
        self.assertEqual(a[0]['value']['backend'],'opencv_cpu')

    def test_guard_before_any_media_or_archive_access(self):
        import run_exact_v9 as runner
        with patch.object(runner,'sha') as hashed:
            with self.assertRaises(ValueError):runner.verify(Namespace(clip='unapproved'))
            hashed.assert_not_called()

    def test_scoped_warp_dispatch_preserves_config_and_restores(self):
        import exact_v9_common as common
        from tiny_target.stabilization.warp import FullResolutionStabilizer,StabilizationConfig
        original=FullResolutionStabilizer._warp_cpu
        class FakeWarp:
            def __init__(self,shape):self.shape=shape
            def __call__(self,a,m,t):
                size=(self.shape[1],self.shape[0])
                return (cv2.warpPerspective(a,t,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT),
                        cv2.warpPerspective(m,t,size,flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT))
            def close(self):pass
        execution=ExactExecution(shadow=True)
        with patch.object(common,'WarpTranslationCuda',FakeWarp),ExitStack() as stack:
            execution.install(stack)
            cfg=StabilizationConfig(backend='opencv_cpu',interpolation='cubic')
            stabilizer=FullResolutionStabilizer(cfg)
            self.assertEqual(cfg.backend,'opencv_cpu')
            self.assertEqual(stabilizer.backend,'cuda_translation_reference_v9')
            a=np.arange(17*19,dtype=np.float32).reshape(17,19);m=np.ones(a.shape,np.uint8)
            matrix=np.eye(3);matrix[0,2]=.21875
            got,mask,_=stabilizer._warp_cpu(a,m,matrix)
            ref,rm,_=original(stabilizer,a,m,matrix)
            self.assertEqual(got.tobytes(),ref.tobytes());self.assertEqual(mask.tobytes(),rm.tobytes())
            with self.assertRaises(ValueError):FullResolutionStabilizer(StabilizationConfig(backend='opencv_cpu',interpolation='linear'))
        execution.close();self.assertIs(FullResolutionStabilizer._warp_cpu,original)
        self.assertEqual(execution.warp_checks,[True])

    def test_two_reversed_rounds_and_injected_check(self):
        rows=schedule('timing');self.assertEqual(len(rows),8)
        for clip in ('0040','0029'):
            self.assertEqual([r['mode'] for r in rows if r['clip']==clip],['reference','exact','exact','reference'])
        self.assertEqual(sum(r.get('injected',False) for r in schedule('checks')),1)
        with self.assertRaises(ValueError):schedule('bad')

    def test_summary_refuses_partial_or_failed_timing(self):
        import json,tempfile
        from summarize_exact_v9 import validated_schedule
        rows=schedule('timing')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'timing_plan.json').write_text(json.dumps({'schedule':rows}))
            journal=root/'timing_journal.jsonl';journal.write_text('')
            with self.assertRaises(ValueError):validated_schedule(root,'timing')
            events=[dict(**r,returncode=0) for r in rows];events[3]['returncode']=2
            journal.write_text('\n'.join(json.dumps(r) for r in events))
            with self.assertRaises(ValueError):validated_schedule(root,'timing')
            events[3]['returncode']=0;journal.write_text('\n'.join(json.dumps(r) for r in events))
            self.assertEqual(validated_schedule(root,'timing')[0],rows)

if __name__=='__main__':unittest.main()
