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
from resident_tracking_v10 import ResidentTracker


class ResidentV10Tests(unittest.TestCase):
    def fake(self):
        p=object.__new__(ResidentTracker)
        p.handle=C.c_void_p(1);p.failed=False;p.owner=threading.get_ident()
        p.shape=(3,7);p.segment=0;p.polarity='bright';p.count=0;p.rows=deque(maxlen=16)
        p.output_generation=None;p.lib=Mock()
        p.lib.seaqr_ring_v10_push.return_value=0
        p.lib.seaqr_ring_v10_reset.return_value=0
        return p

    def test_bad_geometry_rejected_before_library(self):
        for shape in ((0,1),(True,1),(2**32,2**32),(32000001,1),(3,)):
            with self.assertRaises(ValueError):
                ResidentTracker(shape,np.zeros((48,2),np.float32),'/not/a/library')

    def test_metadata_strict_and_nonmutating(self):
        p=self.fake();a=np.zeros(p.shape,np.float32);m=np.ones(p.shape,bool)
        p.push(a,m,0,0)
        for index,timestamp,segment,polarity in (
            (0,1,0,'bright'),(1,0,0,'bright'),(1,1,1,'bright'),(1,1,0,'dark'),
            (True,1,0,'bright'),(1,2**63,0,'bright')):
            with self.assertRaises(ValueError):p.push(a,m,index,timestamp,segment=segment,polarity=polarity)
        self.assertEqual(p.count,1);p.lib.seaqr_ring_v10_push.assert_called_once();p.close()

    def test_bad_arrays_do_not_append(self):
        p=self.fake();a=np.zeros(p.shape,np.float32);m=np.ones(p.shape,bool)
        for response,mask in ((a.astype(np.float64),m),(a+np.nan,m),(a,m.astype(np.uint8)),(a.T,m)):
            with self.assertRaises(ValueError):p.push(response,mask,0,0)
        self.assertEqual(p.count,0);p.lib.seaqr_ring_v10_push.assert_not_called();p.close()

    def test_wrap_metadata_and_stale_output(self):
        p=self.fake()
        for i in range(40):p.commit_metadata(i,i*100000000)
        self.assertEqual([r[0] for r in p.rows],list(range(24,40)))
        with self.assertRaises(RuntimeError):p.download()
        p.reset(segment=2,polarity='dark')
        self.assertEqual(p.count,0);self.assertEqual(len(p.rows),0)
        with self.assertRaises(ValueError):p.run()
        self.assertEqual((p.segment,p.polarity),(2,'dark'));p.close()

    def test_failure_is_sticky(self):
        p=self.fake();p.lib.seaqr_ring_v10_push.return_value=2
        p.lib.seaqr_ring_v10_error.return_value=b'injected failure'
        a=np.zeros(p.shape,np.float32);m=np.ones(p.shape,bool)
        with self.assertRaises(RuntimeError):p.push(a,m,0,0)
        self.assertTrue(p.failed);self.assertEqual(p.count,0)
        with self.assertRaises(RuntimeError):p.reset()
        p.close();p.close();p.lib.seaqr_ring_v10_destroy.assert_called_once()

    def test_thread_guard(self):
        p=self.fake();p.owner=-1
        with self.assertRaises(RuntimeError):p.reset()
        with self.assertRaises(RuntimeError):p.close()
        p.owner=threading.get_ident();p.close()


if __name__=='__main__':unittest.main()
