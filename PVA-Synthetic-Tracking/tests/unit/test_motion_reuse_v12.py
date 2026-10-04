from dataclasses import replace
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import motion_reuse_v12 as reuse
from raw16_speed_v8_common import FeaturePixelsCache
from tiny_target.motion.pva_pyrlk import PvaMotionConfig,PvaMotionError
from tiny_target.types import Frame,TimestampSource,Discontinuity
from verify_motion_reuse_v12 import stats,coverage


def frame(i,**kwargs):
    return Frame(np.full((12,16),i,np.uint16),i*100000000,i,'generated-reuse',16,TimestampSource.MANIFEST,**kwargs)


class MotionReuseV12Tests(unittest.TestCase):
    def fake(self):
        p=object.__new__(reuse.ReuseMotionV12)
        p.owner=threading.get_ident();p._stream=Mock();p.owner_stream=p._stream
        p.config=PvaMotionConfig();p.closed=False;p.failed=False;p._cached=None;p._pending=None;p._active_hit=False
        p.hits=p.misses=p.resets=0;p._pixel_cache=FeaturePixelsCache(reuse.REFERENCE_PIXELS)
        p._vpi=types.SimpleNamespace(Format=types.SimpleNamespace(U16='u16',U8='u8'),Backend=types.SimpleNamespace(CUDA='cuda'),asimage=Mock(side_effect=lambda *a:Mock()))
        return p

    def prepare(self,p,a,b):
        images=p._prepare_images(a,b,a.image,b.image,(8,6),(16,12),True)
        p._prepare_pyramids(images[2],images[3],'backend')
        p._cached=p._pending;p._pending=None

    def test_adjacent_identity_owns_pixels_and_reuses(self):
        p=self.fake();a,b,c=frame(0),frame(1),frame(2)
        self.prepare(p,a,b);cached=p._cached
        self.assertIs(cached.frame,b);self.assertIs(cached.pixels,b.image)
        self.prepare(p,b,c)
        self.assertEqual((p.hits,p.misses),(1,1));self.assertEqual(p._vpi.asimage.call_count,3)
        cached.motion.gaussian_pyramid.assert_called_once();p.close()

    def test_clone_is_not_identity(self):
        p=self.fake();a,b,c=frame(0),frame(1),frame(2);self.prepare(p,a,b)
        self.prepare(p,frame(1),c);self.assertEqual(p.hits,0);p.close()

    def test_configuration_source_gap_and_discontinuity_invalidate(self):
        for change in ('config','source','index_gap','sequence_gap','discontinuity'):
            p=self.fake();a,b,c=frame(0),frame(1),frame(2);self.prepare(p,a,b)
            if change=='config':p.config=replace(p.config,max_features=999)
            if change=='source':c=replace(c,source_id='other')
            if change=='index_gap':c=frame(3)
            if change=='sequence_gap':c=replace(c,sequence=7)
            if change=='discontinuity':c=replace(c,discontinuities=(Discontinuity.CHUNK_BOUNDARY,))
            self.prepare(p,b,c);self.assertEqual(p.hits,0);p.close()

    def test_reset_close_and_thread_ownership(self):
        p=self.fake();self.prepare(p,frame(0),frame(1));p.reset();self.assertIsNone(p._cached)
        p.owner=-1
        with self.assertRaises(RuntimeError):p.reset()
        with self.assertRaises(RuntimeError):p.close()
        p.owner=threading.get_ident();p.close();p.close()
        with self.assertRaises(RuntimeError):p.estimate(frame(0),frame(1))

    def test_stream_substitution_rejected(self):
        p=self.fake();old=p._stream;p._stream=Mock()
        with self.assertRaises(RuntimeError):p.reset()
        p._stream=old;p.close()

    def test_failure_discards_and_poison_cache(self):
        p=self.fake();self.prepare(p,frame(0),frame(1))
        with patch.object(reuse,'_ESTIMATE',side_effect=RuntimeError('device failure')):
            with self.assertRaises(RuntimeError):p.estimate(frame(1),frame(2))
        self.assertIsNone(p._cached);self.assertIsNone(p._pending);self.assertTrue(p.failed)
        with self.assertRaises(RuntimeError):p.estimate(frame(2),frame(3))
        p.close()

    def test_unobservable_failure_allows_clean_recovery(self):
        p=self.fake();self.prepare(p,frame(0),frame(1))
        with patch.object(reuse,'_ESTIMATE',side_effect=PvaMotionError('PVA Harris returned zero features')):
            with self.assertRaises(PvaMotionError):p.estimate(frame(1),frame(2))
        self.assertFalse(p.failed);self.assertIsNone(p._cached);p._live();p.close()

    def test_original_reset_harris_flow_unchanged_in_transform(self):
        source=reuse.generated_method()
        self.assertIn('vpi.clear_cache()',source)
        self.assertIn('forward_options["kptstatus"] = _fresh_flow_status',source)
        self.assertIn('backward_initial_status = _fresh_flow_status',source)
        self.assertIn('previous_s16.harriscorners(',source)
        self.assertEqual(source.count('self._prepare_images('),1)
        self.assertEqual(source.count('self._prepare_pyramids('),1)

    def test_verifier_rejects_incomplete_or_nonfinite_samples(self):
        for values in ([],[1],[True]*4,[float('nan')]*4,[0]*4):
            with self.assertRaises(ValueError):stats(values,4)
        self.assertEqual(stats([1,2,3,4],4)['median'],2.5)
        with self.assertRaises(ValueError):coverage([{'id':0},{'id':0}],('id',),[(0,),(1,)])


if __name__=='__main__':unittest.main()
