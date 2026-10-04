from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from build_tracking_batch_v27 import build
from build_tracking_geometry_v20 import build as build_scalar
from tracking_batch_v27 import BatchGeometryV27
from tracking_geometry_v20 import GeometryV20,reference
from check_tracking_geometry_v20 import same
from check_tracking_batch_v27 import generated
from replay_tracking_v27 import digest


class BatchTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='seaqr_v27_test_');root=Path(cls.temp.name)
        build(root/'batch');cls.library=root/'batch/libtracking_batch_v27.so'
        cls.geometry=build_scalar(root/'scalar')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def adapter(self):
        a=BatchGeometryV27(self.library);a.fallback=GeometryV20(self.geometry);return a

    def test_generated_complete_geometry_and_tracker_parity(self):
        g=generated(self.library,self.geometry)
        self.assertEqual((len(g['cases']),len(g['replays'])),(144,12))
        self.assertTrue(all(c['exact'] for c in g['cases']+g['replays']))
        self.assertTrue(any(c['native'] for c in g['cases']))
        self.assertTrue(any(not c['native'] for c in g['cases']))

    def test_results_are_owned_and_survive_later_calls(self):
        a=self.adapter();values=np.ones((10,4));tracks={2:SimpleNamespace(mean=np.zeros(4)),1:SimpleNamespace(mean=np.ones(4))}
        result=a(values,tracks,4,5.,5.);before=[x.tobytes() for x in result.arrays]
        a(values*20,tracks,4,5.,5.)
        self.assertEqual(before,[x.tobytes() for x in result.arrays])
        self.assertFalse(np.shares_memory(result.arrays[0][0],result.arrays[0][1]))

    def test_native_error_never_falls_back_silently(self):
        a=self.adapter();a.fn=lambda *args:99
        with self.assertRaisesRegex(RuntimeError,'failed: 99'):
            a(np.ones((2,4)),{0:SimpleNamespace(mean=np.zeros(4))},2,5.,5.)
        self.assertEqual(a.fallbacks,0)

    def test_fallback_keeps_numpy_error_policy(self):
        for value,policy in ((1e200,'over'),(1e-200,'under'),(np.inf,'invalid')):
            a=self.adapter();values=np.full((2,4),value);mean=np.full(4,value) if policy=='invalid' else np.zeros(4)
            with np.errstate(**{policy:'raise'}):
                r=a(values,{0:SimpleNamespace(mean=mean)},4,5.,5.)
                with self.assertRaises(FloatingPointError):r.get(0,values,mean,4,5.,5.)

    def test_no_track_batch_does_not_enter_native(self):
        a=self.adapter();a.fn=lambda *args:self.fail('Unexpected native call')
        a(np.empty((0,4)),{},2,None,None)
        self.assertEqual(a.calls,0)

    def test_unbound_adapter_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'not bound'):BatchGeometryV27(self.library)(np.empty((0,4)),{},2,5.,5.)

    def test_parallel_calls_have_independent_scratch(self):
        a=self.adapter()
        def run(seed):
            values=np.random.default_rng(seed).normal(size=(20,4));tracks={i:SimpleNamespace(mean=np.ones(4)*i) for i in range(4)}
            r=a(values,tracks,4,5.,5.)
            for i,m in tracks.items():same(reference(values,m.mean,4,5.,5.),r.get(i,values,m.mean,4,5.,5.))
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,range(6)))

    def test_unrecognized_scalar_method_rejected(self):
        from tiny_target.tracking.kalman import KalmanTrackManager
        with self.assertRaisesRegex(ValueError,'original v20'):
            self.adapter().adapt(KalmanTrackManager.update)

    def test_state_digest_keeps_signed_zero_nan_and_array_layout(self):
        self.assertNotEqual(digest(-0.),digest(0.))
        self.assertNotEqual(digest(np.ones((2,2))),digest(np.ones((4,))))
        self.assertEqual(digest(np.inf),digest(float('inf')))


if __name__=='__main__':unittest.main()
