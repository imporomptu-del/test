import ctypes as C
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from build_native_motion_v25 import build
from native_motion_v25 import NativeMotionV25,reference_samples,gm
from check_native_motion_v25 import generated,gil_probe,equal_array,equal_estimate,correspondence


class NativeMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='seaqr_native_v25_test_')
        cls.library=build(Path(cls.directory.name)/'build')

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_generated_full_fit_and_primitive_contracts(self):
        result=generated(self.library)
        self.assertEqual(len(result['cases']),47);self.assertEqual(len(result['primitive']),11)
        self.assertGreater(result['native_calls'],0);self.assertGreater(result['fallbacks'],0)
        self.assertEqual(result['passthroughs'],2);self.assertTrue(result['independent_outputs'])

    def test_python_thread_progresses_inside_native_call(self):
        self.assertTrue(gil_probe(self.library)['passed'])

    def test_caps_and_layout_fall_back_without_changing_results(self):
        helper=NativeMotionV25(self.library)
        for n,samples in ((4097,[(0,)]),(2,[(0,)]*1025)):
            p=np.full((n,2),64.,np.float64);c=p+[1.,2.]
            before=helper.fallbacks;a=reference_samples(p,c,samples,.5);b=helper(p,c,samples,.5)
            self.assertEqual(helper.fallbacks,before+1)
            equal_array(a[0],b[0]);equal_array(a[1],b[1]);self.assertEqual(a[2],b[2])

    def test_error_policy_and_signed_zero_are_preserved(self):
        helper=NativeMotionV25(self.library);p=np.array([[64.,64.],[65.,65.]]);c=p+1
        p[0,0]=-0.;a=reference_samples(p,c,[(0,),(1,)],.5);b=helper(p,c,[(0,),(1,)],.5)
        equal_array(a[0],b[0]);equal_array(a[1],b[1]);self.assertEqual(helper.fallbacks,1)
        p[0,0]=np.inf;c[0,0]=np.inf  # inf - inf in the unchanged fitter must raise.
        with np.errstate(invalid='raise'):
            for score in (reference_samples,helper):
                with self.assertRaises(FloatingPointError):score(p,c,[(0,)],.5)

    def test_native_errors_do_not_silently_fall_back(self):
        helper=NativeMotionV25(self.library);p=np.ones((2,2),np.float64)
        with patch.object(helper,'fn',return_value=3):
            with self.assertRaisesRegex(RuntimeError,'scoring failed'):helper(p,p,[(0,)],.5)
        self.assertEqual(helper.fallbacks,0)

    def test_source_identity_and_exact_anchor_are_required(self):
        helper=NativeMotionV25(self.library)
        with patch('native_motion_v25.REFERENCE_SHA','unknown'):
            with self.assertRaisesRegex(ValueError,'source'):helper.adapter(gm.fit_global_motion)
        with patch('native_motion_v25.OLD','not-an-anchor'):
            with self.assertRaisesRegex(ValueError,'loop'):helper.adapter(gm.fit_global_motion)

    def test_native_abi_rejects_invalid_pointers_and_bounds(self):
        helper=NativeMotionV25(self.library)
        self.assertEqual(helper.fn(None,None,1,None,1,.5,None,None,None),2)

    def test_batch_reads_latency_from_linked_v24_and_rejects_hidden_wait(self):
        from batch_visible_native_v25 import speed_gate,initial_schedule
        self.assertEqual(len(initial_schedule()),14)
        def evaluate(delay):
            def read(path):
                candidate='native_staged' in path.name
                if path.name.endswith('.v25.json'):return dict(fps=1.3 if candidate else 1,wall_s=128/(1.3 if candidate else 1))
                if path.name.endswith('.v17.json'):return {'consumer_frame_ms':[1 if candidate else 2]*128}
                return {'execution':{'frames':[dict(consumer_complete_ns=10_000_000,ready_ns=8_000_000,
                    request_ns=10_000_000-int((delay if candidate else 3)*1e6))]*128}}
            with patch('batch_visible_native_v25.read',read):return speed_gate(Path('/synthetic'))
        self.assertTrue(evaluate(2)['passed']);self.assertFalse(evaluate(4)['passed'])

    def test_complete_transform_chain_preserves_reset_and_reuse(self):
        from dataclasses import replace
        helper=NativeMotionV25(self.library);fit=helper.adapter(gm.fit_global_motion)
        p=np.array([[64+x*64,64+y*64] for y in range(4) for x in range(4)],np.float32)
        base=gm.GlobalMotionConfig(minimum_correspondences=4,minimum_inliers=4,minimum_inlier_grid_coverage=0)
        for policy in ('reset_reference','reuse_previous'):
            config=replace(base,failure_policy=policy)
            left=gm.GlobalMotionTracker(config,0);right=gm.GlobalMotionTracker(config,0)
            for index in range(1,9):
                points=p[:1] if index in (3,4) else p
                corr=correspondence(points,points+[1,2],index=index)
                a=gm.fit_global_motion(corr,config);b=fit(corr,config);equal_estimate(a,b)
                self.assertEqual(left.update(a).to_dict(),right.update(b).to_dict())


if __name__=='__main__':unittest.main()
