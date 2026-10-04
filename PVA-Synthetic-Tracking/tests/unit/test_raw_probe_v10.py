from collections import deque
import ctypes as C
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from resident_frontend_v10 import ResidentRawProbe
from resident_tracking_v10 import ResidentTracker


class RawProbeV10Tests(unittest.TestCase):
    def fake(self):
        p=object.__new__(ResidentRawProbe)
        p.handle=C.c_void_p(2);p.failed=False;p.owner=threading.get_ident()
        p.shape=(7,11);p.count=0;p.last_metadata=None;p.lib=Mock()
        p.lib.seaqr_front_v10_step.return_value=0
        p.lib.seaqr_front_v10_reset.return_value=0
        r=object.__new__(ResidentTracker)
        r.handle=C.c_void_p(1);r.failed=False;r.owner=p.owner;r.shape=p.shape
        r.segment=0;r.polarity='bright';r.count=0;r.rows=deque(maxlen=16)
        r.output_generation=None;r.lib=Mock();r.lib.seaqr_ring_v10_reset.return_value=0
        p.ring=r
        return p

    def test_raw_precision_and_shape_guards(self):
        p=self.fake();raw=np.zeros(p.shape,np.uint16)
        for wrong in (raw.astype(np.uint8),raw.astype(np.float32),raw.T):
            with self.assertRaises(ValueError):p.push(wrong,0,0,np.eye(3))
        p.lib.seaqr_front_v10_step.assert_not_called();p.close()

    def test_mask_and_transform_guards(self):
        p=self.fake();raw=np.zeros(p.shape,np.uint16)
        for mask in (np.ones(p.shape,np.uint8),np.ones((3,7),bool)):
            with self.assertRaises(ValueError):p.push(raw,0,0,np.eye(3),mask)
        for matrix in (np.eye(3)*2,np.eye(3)+np.nan,np.ones((2,3))):
            with self.assertRaises(ValueError):p.push(raw,0,0,matrix)
        p.lib.seaqr_front_v10_step.assert_not_called();p.close()

    def test_warmup_metadata_and_reset(self):
        p=self.fake();raw=np.zeros(p.shape,np.uint16)
        p.push(raw,0,0,np.eye(3));self.assertEqual(p.ring.count,0)
        for index,timestamp in ((0,1),(1,0)):
            with self.assertRaises(ValueError):p.push(raw,index,timestamp,np.eye(3))
        p.reset();self.assertIsNone(p.last_metadata);self.assertEqual(p.count,0)
        p.push(raw,0,0,np.eye(3));p.close()

    def test_emitted_state_mismatch_poison_both(self):
        p=self.fake();p.count=4;raw=np.zeros(p.shape,np.uint16)
        with self.assertRaises(RuntimeError):p.push(raw,4,400000000,np.eye(3))
        self.assertTrue(p.failed);self.assertTrue(p.ring.failed)
        with self.assertRaises(RuntimeError):p.ring.run()
        p.close()

    def test_native_failure_poison_both(self):
        p=self.fake();p.lib.seaqr_front_v10_step.return_value=2
        p.lib.seaqr_front_v10_error.return_value=b'injected allocation failure'
        with self.assertRaises(RuntimeError):p.push(np.zeros(p.shape,np.uint16),0,0,np.eye(3))
        self.assertTrue(p.failed);self.assertTrue(p.ring.failed)
        self.assertEqual(p.count,0);self.assertEqual(p.ring.count,0)
        with self.assertRaises(RuntimeError):p.reset()
        p.close();p.close();p.lib.seaqr_front_v10_destroy.assert_called_once()

    def test_debug_and_thread_guard(self):
        p=self.fake()
        with self.assertRaises(ValueError):p.debug()
        p.owner=-1
        with self.assertRaises(RuntimeError):p.reset()
        with self.assertRaises(RuntimeError):p.close()
        p.owner=threading.get_ident();p.close()


if __name__=='__main__':unittest.main()
