from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from visible_front_v26 import pack_learning,ResidentFrontV26,attach_warp,CUDA_SOURCES
from tiny_target.visible_learning import shape_learning_mask


class FrontContractTests(unittest.TestCase):
    def test_journal_dependencies_are_packaged_and_preflighted(self):
        from run_visible_front_v26 import verifier_dependencies,CHECK_MODULES,FROZEN
        self.assertTrue(callable(verifier_dependencies()))
        self.assertTrue(all(n+'.py' in FROZEN for n in CHECK_MODULES))
        with patch('run_visible_front_v26.sha',side_effect=('loaded','packaged')):
            with self.assertRaisesRegex(ValueError,'dependency changed'):verifier_dependencies()

    def test_compact_learning_matches_reference_sparse_and_dense(self):
        rng=np.random.default_rng(2630)
        for shape in ((1,1),(7,13),(67,99)):
            for margin in (.5,1,1.5,2,4.25,16):
                for n in (0,1,99,1024):
                    with self.subTest(shape=shape,margin=margin,n=n):
                        support=rng.random(shape)>.15
                        regions=[] if n==0 else [dict(support_reference_xy=rng.uniform(-3,max(shape)+3,(n,2)).tolist())]
                        xy,offsets=pack_learning(shape,regions,margin);actual=support.copy()
                        for x,y in xy:
                            for dx,dy in offsets:
                                xx,yy=x+dx,y+dy
                                if 0<=xx<shape[1] and 0<=yy<shape[0]:actual[yy,xx]=False
                        np.testing.assert_array_equal(actual,shape_learning_mask(support,regions,margin))
                        self.assertEqual(xy.dtype,np.int32);self.assertTrue(xy.flags.c_contiguous)

    def test_round_to_even_duplicates_and_outside(self):
        xy,_=pack_learning((5,5),[dict(support_reference_xy=[[.5,1.5],[.5,1.5],[-.6,2],[4.6,2]])],2)
        np.testing.assert_array_equal(xy,[[0,2],[0,2]])

    def test_invalid_learning_inputs_fail_closed(self):
        for regions in ([dict(support_reference_xy=[])],[dict(support_reference_xy=[[np.nan,1]])],
                        [dict(support_reference_xy=[[1,2,3]])],[dict(support_reference_xy=[[1,2]]*1025)]):
            with self.assertRaises(ValueError):pack_learning((7,9),regions,2)
        for margin in (0,-1,17,np.inf,np.nan):
            with self.assertRaises(ValueError):pack_learning((7,9),[],margin)
        with self.assertRaises(ValueError):pack_learning((7,9),[{}]*513,2)

    def test_owner_and_active_close_guards(self):
        front=object.__new__(ResidentFrontV26);front.owner_thread=threading.get_ident();front.busy=True
        with self.assertRaisesRegex(RuntimeError,'active'):front.close()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError,'single-owner'):pool.submit(front._owned).result()

    def test_warp_adapter_only_annotates_device_frames(self):
        class Frame:pass
        frame=Frame();valid=object()
        def original(*a,**kw):return frame,valid
        call=attach_warp(original)
        self.assertEqual(call(None,None,None,None),(frame,valid));self.assertFalse(hasattr(frame,'front_erosion_v26'))
        self.assertEqual(call(None,None,None,None,device=True,erosion_px=7),(frame,valid))
        self.assertEqual(frame.front_erosion_v26,7)

    def test_original_kernel_hashes_match_archived_baseline(self):
        import hashlib
        here=Path(__file__).resolve().parent
        root=here/'baseline_cuda' if (here/'baseline_cuda').is_dir() else (
            here.parents[1]/'results/tiny_target/visible_speed_v19_20260918/evidence/build/reference_probe')
        self.assertEqual(len(CUDA_SOURCES),4)
        for name,digest in CUDA_SOURCES.items():
            self.assertEqual(len(digest),64)
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)

    def test_native_error_poison_requires_close_and_releases_handle(self):
        front=object.__new__(ResidentFrontV26);front.owner_thread=threading.get_ident()
        front.busy=front.poisoned=False;front.front=1;front.handle=2;front.shape=(7,13);front.segment=0;front.count=0
        front.config=SimpleNamespace(max_active_tracks_per_polarity=256,position_sigma_px=2,warmup_frames=8,
            noise_floor_dn=.5,max_candidates_per_tile_polarity=12,temporal_threshold_sigma=4,spatial_threshold_sigma=3)
        front.eligible=np.empty(front.shape,np.bool_);front.sigmas=np.empty(1);front.peaks=np.empty(1)
        front.counts=np.empty(2,np.int32);front.searchable=np.empty(1,np.int32)
        closed=[];front.lib=SimpleNamespace(seaqr_front_v26_prepare_host=lambda *a:999,
                                          seaqr_front_v26_destroy=closed.append)
        image=np.ones(front.shape,np.float32);valid=np.ones(front.shape,np.bool_)
        with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):front.update(image,valid,0)
        self.assertTrue(front.poisoned);self.assertFalse(front.busy)
        with self.assertRaisesRegex(RuntimeError,'close before reuse'):front.update(image,valid,0)
        front.close();front.close()
        self.assertEqual(closed,[1]);self.assertFalse(front.poisoned);self.assertIsNone(front.handle)

    def test_host_allocation_error_releases_new_device_resources(self):
        front=object.__new__(ResidentFrontV26);front.owner_thread=threading.get_ident();front.busy=False
        front.front=front.handle=front.shape=front.segment=None;front.config=SimpleNamespace(
            tile_size=32,noise_sample_stride=4,max_candidates_per_tile_polarity=3)
        closed=[];front.lib=SimpleNamespace(seaqr_front_v26_create=lambda *a:1,
            seaqr_front_v26_core=lambda *a:2,seaqr_front_v26_destroy=closed.append)
        with patch('visible_front_v26.np.empty',side_effect=MemoryError('test allocation')):
            with self.assertRaises(MemoryError):front._allocate((67,99),0)
        self.assertEqual(closed,[1]);self.assertIsNone(front.front);self.assertIsNone(front.handle)

    def test_paired_gate_rejects_hidden_request_wait_and_missing_frames(self):
        from batch_visible_front_v26 import speed_gate,initial_schedule
        self.assertEqual(len(initial_schedule()),14)
        def evaluate(delay,count=128):
            def read(path):
                candidate='candidate' in path.name;fps=1.3 if candidate else 1.
                return dict(passed=True,error=None,processed_frames=count,fps=fps,wall_s=128/fps,
                    consumer_frame_ms=[1.]*128,execution={'frames':[dict(consumer_complete_ns=10000000,
                        ready_ns=8000000,request_ns=10000000-int((delay if candidate else 3)*1e6))]*128})
            with patch('batch_visible_front_v26.read',read):return speed_gate(Path('/synthetic'))
        self.assertTrue(evaluate(2)['passed']);self.assertFalse(evaluate(4)['passed'])
        with self.assertRaisesRegex(ValueError,'Incomplete'):evaluate(2,127)


if __name__=='__main__':unittest.main()
