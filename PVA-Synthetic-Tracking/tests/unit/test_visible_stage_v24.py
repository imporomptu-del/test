"""Generated admission, GPU release, real decode-thread and adapter contracts."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from stage_control_v24 import Admission, GpuRelease, StageCancelled
from visible_stage_v24 import StageBinding, install
from test_visible_overlap_v23 import mock_runtime
from tiny_target.visible_decode import VisibleFrameReader as NativeReader
from run_visible_stage_v24 import validate_snapshot
import cv2


@contextmanager
def fixture(count=6, **kw):
    # Exercise the real decoder producer/consumer, without opening any media.
    with mock_runtime(count=count,**kw) as state, ExitStack() as stack:
        module=sys.modules['tiny_target.visible_warp_exact']
        base_warp=module.CudaCubicTranslation
        old_init=state.original_motion.__init__
        old_update=state.original_motion.update
        def init(motion,*a,**opts):
            old_init(motion,*a,**opts)
            # The frozen real constructor imports this class dynamically. The
            # old mock instead closes over its base; promote its Python object
            # to match the real factory lookup, with no allocation/close change.
            motion.cuda_warp.__class__=module.CudaCubicTranslation
        def warp_call(warp,gray):
            warp.image[:]=gray+1
            return warp.image,warp.valid
        def update(motion,gray,index,timestamp):
            motion.cuda_warp(gray)
            return old_update(motion,gray,index,timestamp)
        stack.enter_context(patch.object(base_warp,'__call__',warp_call,create=True))
        stack.enter_context(patch.object(state.original_motion,'__init__',init))
        stack.enter_context(patch.object(state.original_motion,'update',update))
        class Capture:
            def __init__(self,*a):self.index=0;self.released=False
            def isOpened(self):return True
            def get(self,key):return 29.97 if key==cv2.CAP_PROP_FPS else count
            def read(self):
                if self.index>=count:return False,None
                bgr=np.full((4,4,3),self.index,np.uint8);self.index+=1
                return True,bgr
            def release(self):self.released=True
        state.capture=Capture()
        stack.enter_context(patch.object(cv2,'VideoCapture',lambda *a:state.capture))
        stack.enter_context(patch.object(cv2,'cvtColor',lambda bgr,*a:bgr[:,:,0].copy()))
        state.visible.VisibleFrameReader=state.decode.VisibleFrameReader=NativeReader
        yield state


def consume(mode, count=6, maximum=True):
    with fixture(count=count) as state, ExitStack() as stack:
        binding=StageBinding(mode)
        install(stack,binding)
        reader=state.decode.VisibleFrameReader('synthetic',(4,4),execution='prefetch_one',
                                              max_frames=count if maximum else None)
        outputs=[]
        with reader:
            motion=state.visible.PvaMotion()
            detector=state.visible.VisiblePointDetector();tracks=state.visible.VisibleTracks()
            reader.start()
            for _ in range(count):
                frame,_=reader.read()
                value=motion.update(frame.gray,frame.index,round(frame.index/reader.fps*1e9))
                image,valid,matrix,segment,meta=value
                result=tracks.update(detector.update(image))
                outputs.append((image.tobytes(),valid.tobytes(),matrix.tobytes(),segment,deepcopy(meta),result))
            if not maximum:
                assert reader.read()[0] is None
            reader.close();motion.close()
        assert state.capture.released
        for resource in state.warps+state.estimators:
            assert resource.close_count==1
            assert resource.close_threads==[resource.owner]
        validate_snapshot(binding.snapshot(),count)
        return outputs,binding.snapshot(),reader.completed_stats()


class StageControlTests(unittest.TestCase):
    def test_capacity_reserves_before_decode_and_includes_wait(self):
        budget=Admission();budget.acquire(0);budget.acquire(1)
        entered=threading.Event();done=threading.Event();value=[]
        def third():entered.set();value.append(budget.acquire(2));done.set()
        worker=threading.Thread(target=third);worker.start();self.assertTrue(entered.wait(1))
        self.assertFalse(done.wait(.02));release=time.perf_counter_ns();budget.release(0)
        self.assertTrue(done.wait(1));worker.join(1)
        self.assertLess(value[0]['request_ns'],release)
        self.assertGreaterEqual(value[0]['admitted_ns'],release)
        self.assertEqual(budget.snapshot()['maximum'],2)
        budget.release(1);budget.release(2);budget.stop()
        self.assertEqual(budget.snapshot()['held'],[])

    def test_admission_cancel_wakes_waiter(self):
        budget=Admission();budget.acquire(0);budget.acquire(1);errors=[]
        def third():
            try:budget.acquire(2)
            except BaseException as e:errors.append(e)
        worker=threading.Thread(target=third);worker.start();budget.stop();worker.join(1)
        self.assertFalse(worker.is_alive());self.assertIsInstance(errors[0],StageCancelled)

    def test_invalid_order_double_release_and_capacity(self):
        for c in (1,3,True,2.0):
            with self.assertRaises(ValueError):Admission(c)
        budget=Admission()
        with self.assertRaises(ValueError):budget.acquire(1)
        budget.acquire(0);budget.release(0)
        with self.assertRaises(ValueError):budget.release(0)

    def test_gpu_first_frame_and_strict_release(self):
        gate=GpuRelease();gate.wait(0);done=threading.Event()
        worker=threading.Thread(target=lambda:(gate.wait(1),done.set()));worker.start()
        self.assertFalse(done.wait(.02));gate.detector_complete(0)
        self.assertTrue(done.wait(1));worker.join(1)
        with self.assertRaises(ValueError):gate.detector_complete(0)
        with self.assertRaises(ValueError):gate.detector_complete(2)

    def test_gpu_cancel_wakes_pending_frame(self):
        gate=GpuRelease();errors=[]
        def wait():
            try:gate.wait(1)
            except BaseException as e:errors.append(e)
        worker=threading.Thread(target=wait);worker.start();gate.stop();worker.join(1)
        self.assertFalse(worker.is_alive());self.assertIsInstance(errors[0],StageCancelled)


class StageAdapterTests(unittest.TestCase):
    def test_all_modes_exact_outputs_and_clean_shared_lifecycle(self):
        outputs=[]
        for mode in ('reference','bounded','staged'):
            output,snap,decode=consume(mode)
            outputs.append(output)
            self.assertEqual(snap['policy'],mode)
            self.assertEqual(snap['admission']['held'],[])
            self.assertLessEqual(snap['admission']['maximum'],2)
            self.assertEqual([e['disposition'] for e in snap['admission']['events']],['consumed']*6)
            self.assertTrue(snap['admission']['stopped']);self.assertTrue(snap['gpu_release']['stopped'])
            self.assertEqual(snap['gpu_release']['completed'],5)
            self.assertEqual(decode['decoded_frames'],6);self.assertEqual(decode['dropped_frames'],0)
            for i,r in enumerate(snap['frames']):
                keys=('request_ns','admitted_ns','ready_ns','cpu_prepare_start_ns','cpu_prepare_end_ns',
                      'warp_wait_start_ns','warp_gpu_start_ns','warp_gpu_end_ns','detector_start_ns',
                      'detector_end_ns','gpu_release_ns','tracking_start_ns','tracking_end_ns','consumer_complete_ns')
                self.assertEqual([r[k] for k in keys],sorted(r[k] for k in keys))
                if mode=='staged' and i:
                    self.assertGreaterEqual(r['warp_gpu_start_ns'],snap['frames'][i-1]['gpu_release_ns'])
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[1],outputs[2])

    def test_natural_eof_releases_reservation_without_extra_frame(self):
        _,snap,decode=consume('staged',maximum=False)
        self.assertEqual(len(snap['frames']),6)
        self.assertEqual(snap['admission']['events'][-1]['disposition'],'eof')
        self.assertEqual(snap['admission']['held'],[])
        self.assertEqual(decode['read_calls'],7)

    def test_one_frame_bootstrap_never_requires_a_prior_detector(self):
        _,snap,_=consume('staged',count=1)
        self.assertEqual(snap['gpu_release']['completed'],0)

    def test_failed_detector_never_releases_future_warp_and_close_unblocks(self):
        with fixture(count=3) as state,ExitStack() as stack:
            def fail(*a,**kw):raise RuntimeError('synthetic detector failure')
            stack.enter_context(patch.object(state.visible.VisiblePointDetector,'update',fail))
            binding=StageBinding('staged');install(stack,binding)
            reader=state.decode.VisibleFrameReader('synthetic',(4,4),execution='prefetch_one',max_frames=3)
            reader.__enter__();motion=state.visible.PvaMotion();reader.start()
            frame,_=reader.read();image=motion.update(frame.gray,0,0)[0]
            with self.assertRaisesRegex(RuntimeError,'detector failure'):
                state.visible.VisiblePointDetector().update(image)
            self.assertEqual(binding.gpu_release.completed,-1)
            try:reader.close()
            except StageCancelled:pass  # Expected cancellation, never a successful media receipt.
            self.assertTrue(binding.engine.stats['joined'])
            self.assertTrue(state.capture.released)
            self.assertFalse(binding.engine._thread.is_alive())

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):StageBinding('unknown')

    def test_receipt_rejects_missing_releases_hidden_wait_and_bad_stage_order(self):
        _,original,_=consume('staged')
        for mutate in (
            lambda s:s['admission'].update(maximum=3),
            lambda s:s['admission']['events'][0].update(disposition='error'),
            lambda s:s['frames'][0].update(request_ns=s['frames'][0]['admitted_ns']+1),
            lambda s:s['frames'][1].update(warp_gpu_start_ns=s['frames'][0]['gpu_release_ns']-1),
            lambda s:s['gpu_release'].update(completed=0),
            lambda s:s['admission'].update(held=[0]),
        ):
            bad=deepcopy(original);mutate(bad)
            with self.assertRaises(AssertionError):validate_snapshot(bad,6)


class PerformanceGateTests(unittest.TestCase):
    def test_each_latency_boundary_and_each_workload_can_block_adoption(self):
        from batch_visible_stage_v24 import speed_gate
        def result(candidate_ready=1,candidate_request=2,candidate_cadence=1,second_wall=80):
            def read(path):
                name=path.name;staged='staged' in name
                if path.suffixes[-2:]==['.v17','.json']:
                    return {'consumer_frame_ms':[candidate_cadence if staged else 2]*128}
                wall=(second_wall if name.startswith('0082') else 80) if staged else 100
                end=10_000_000
                return dict(wall_s=wall,fps=128/wall,execution={'frames':[
                    dict(consumer_complete_ns=end,ready_ns=end-int((candidate_ready if staged else 2)*1e6),
                         request_ns=end-int((candidate_request if staged else 3)*1e6))]*128})
            with patch('batch_visible_stage_v24.read',read):return speed_gate(Path('/synthetic'))
        self.assertTrue(result()['passed'])
        for kw in (dict(candidate_ready=3),dict(candidate_request=4),dict(candidate_cadence=3),
                   dict(second_wall=90)):
            self.assertFalse(result(**kw)['passed'])


if __name__=='__main__':unittest.main()
