"""Native noise equivalence, cache invalidation and error-path regression."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_noise_v18 import build
from noise_v18 import NoiseV18
from check_noise_v18 import cases, compare
from tiny_target.visible_resident import sample_layout


class NoiseV18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='seaqr_noise_v18_test_')
        cls.library = build(Path(cls.temp.name)/'build')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generated_exact_native_and_explicit_fallback(self):
        c=NoiseV18(self.library)
        rows=cases(c,include_large=False)
        self.assertEqual(len(rows),350)
        self.assertTrue(all(r['exact'] and r['inputs_unchanged'] for r in rows))
        self.assertGreater(c.calls,300); self.assertGreater(c.fallbacks,10)

    def test_cache_reuse_does_not_cache_mask_values(self):
        c=NoiseV18(self.library)
        layout,ids=sample_layout((15,18),8,2)
        samples=np.arange(len(ids),dtype=np.float32)
        mask=np.ones((15,18),bool)
        compare(c,samples,mask,layout,2,.5)
        self.assertEqual(c.geometry_builds,1)
        mask[:,1:8]=False; samples[:]*=-1; samples[0]=0
        compare(c,samples,mask,layout,2,.5)
        self.assertEqual(c.geometry_builds,1)
        layout2,ids2=sample_layout((15,18),9,2)
        compare(c,np.ones(len(ids2),np.float32),mask,layout2,2,.5)
        self.assertEqual(c.geometry_builds,2)

    def test_native_abi_error_is_not_silently_fallback(self):
        c=NoiseV18(self.library)
        layout,ids=sample_layout((8,8),8,1)
        with patch.object(c,'fn',return_value=-4):
            with self.assertRaisesRegex(RuntimeError,'no error fallback'):
                c(np.ones(len(ids),np.float32),np.ones((8,8),bool),layout,1,.5)
        self.assertEqual(c.fallbacks,0)

    def test_noncanonical_layout_retains_reference(self):
        c=NoiseV18(self.library)
        mask=np.ones((8,8),bool)
        layout=[(slice(None),slice(None),0,64)]
        row=compare(c,np.ones(64,np.float32),mask,layout,1,.5)
        self.assertTrue(row['fallback'])

    def test_masked_special_values_do_not_force_fallback(self):
        c=NoiseV18(self.library)
        layout,ids=sample_layout((8,8),8,1)
        values=np.ones(len(ids),np.float32);values[0]=np.nan
        mask=np.ones((8,8),bool);mask[0,0]=False
        row=compare(c,values,mask,layout,1,.5)
        self.assertTrue(row['native'])

    def test_native_rejects_bad_bounds_without_input_mutation(self):
        c=NoiseV18(self.library)
        samples=np.ones(4,np.float32);mask=np.ones(4,np.bool_)
        ids=np.arange(4,dtype=np.int64);bounds=np.asarray([0,4],np.int64)
        stats=np.empty((1,2),np.float32);sigmas=np.empty(1,np.float64)
        def call():
            return c.fn(samples.ctypes.data,4,mask.ctypes.data,4,ids.ctypes.data,
                        bounds.ctypes.data,1,.5,stats.ctypes.data,sigmas.ctypes.data)
        self.assertEqual(call(),0)
        bounds[0]=1;self.assertEqual(call(),-2)
        bounds[0]=0;ids[1]=4;self.assertEqual(call(),-4)
        np.testing.assert_array_equal(samples,np.ones(4,np.float32))
        np.testing.assert_array_equal(mask,np.ones(4,bool))


if __name__=='__main__':
    unittest.main()
