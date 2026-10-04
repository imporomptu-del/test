from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_tracking_geometry_v20 import build
from tracking_geometry_v20 import GeometryV20,reference
from check_tracking_geometry_v20 import primitive_cases,same,replay


class TrackingGeometryV20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='seaqr_v20_test_')
        cls.helper=GeometryV20(build(Path(cls.temp.name)/'build'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_generated_bytes_gates_and_nonmutation(self):
        for name,a,m,d,p,v in primitive_cases():
            with self.subTest(name=name):
                before=(a.tobytes(),m.tobytes())
                with np.errstate(all='ignore'):same(reference(a,m,d,p,v),self.helper(a,m,d,p,v))
                self.assertEqual(before,(a.tobytes(),m.tobytes()))
        self.assertGreater(self.helper.calls,0);self.assertGreater(self.helper.fallbacks,0)

    def test_full_tracker_replay(self):
        rows=replay(self.helper)
        self.assertEqual(len(rows),12);self.assertTrue(all(r['exact'] for r in rows))

    def test_outputs_do_not_alias_next_call(self):
        a=np.ones((3,4));m=np.zeros(4);r=self.helper(a,m,4,5.,5.)
        before=[x.tobytes() for x in r[:5]]
        self.helper(a*10,m,4,5.,5.)
        self.assertEqual(before,[x.tobytes() for x in r[:5]])

    def test_fallback_preserves_numpy_floating_error_policy(self):
        for value,policy in ((1e200,'over'),(1e-200,'under'),(np.inf,'invalid')):
            a=np.full((2,4),value);m=np.full(4,value) if policy=='invalid' else np.zeros(4)
            for fn in (reference,self.helper):
                with self.subTest(value=value,function=fn),np.errstate(**{policy:'raise'}):
                    with self.assertRaises(FloatingPointError):fn(a,m,4,5.,5.)

    def test_signed_zero_and_adjacent_residuals(self):
        for d in (2,4):
            a=np.array([[-0.,0.,-0.,0.],[np.nextafter(1.,0.),1.,0.,0.]])
            for m in (np.zeros(4),np.full(4,-0.0),np.ones(4)):
                same(reference(a,m,d,1.,1.),self.helper(a,m,d,1.,1.))

    def test_unknown_tracker_source_rejected(self):
        def changed_update(self,batch):return batch
        with self.assertRaisesRegex(ValueError,'Unknown frozen tracker'):
            self.helper.adapter(changed_update)


if __name__=='__main__':unittest.main()
