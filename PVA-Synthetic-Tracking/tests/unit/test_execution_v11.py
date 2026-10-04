from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from build_resident_v11 import transformed,once
from motion_lut_v11 import feature_pixels_lut,REFERENCE_FEATURE_PIXELS
from tiny_target.types import Frame,TimestampSource
from tiny_target.motion import pva_pyrlk as pva
from verify_execution_v11 import numbers,coverage


def fixture(image,depth=16,mask=None):
    return Frame(image,0,0,'generated-v11',depth,TimestampSource.MANIFEST,valid_mask=mask)


class ExecutionV11Tests(unittest.TestCase):
    def check(self,frame,mapping='raw_robust_u16_v1'):
        before=frame.pixel_sha256()
        a=REFERENCE_FEATURE_PIXELS(frame,mapping);b=feature_pixels_lut(frame,mapping)
        self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape)
        self.assertEqual(a.tobytes(),b.tobytes());self.assertTrue(b.flags.c_contiguous)
        self.assertEqual(before,frame.pixel_sha256())

    def test_all_codes_masks_and_depths(self):
        image=np.arange(65536,dtype=np.uint16).reshape(256,256)
        for depth in (9,10,12,14,16):
            values=image%2**depth if depth<16 else image
            for mask in (None,np.zeros(image.shape,bool),values%7!=0):self.check(fixture(values,depth,mask))

    def test_extremes_flat_and_noncontiguous(self):
        for value in (0,1,32767,65535):self.check(fixture(np.full((33,41),value,np.uint16)))
        image=np.arange(65536,dtype=np.uint16).reshape(256,256)
        self.check(fixture(image[::-1,::2]))
        self.check(fixture(np.array([[1]],np.uint16)))

    def test_float_rounding_boundaries(self):
        frame=fixture(np.arange(65536,dtype=np.uint16).reshape(256,256))
        for scale,offset in ((1,.5),(1.00000006,-.5),(.333333333333,8192.25),(40960.,-40960.),(0,0)):
            with patch.object(pva,'_raw_affine_parameters',return_value=(scale,offset)):self.check(frame)

    def test_eight_bit_and_mapping_fallback(self):
        image=np.arange(256,dtype=np.uint8).reshape(16,16)
        for mapping in ('bit_shift','raw_asinh_v1','raw_linear_u16_v1','raw_robust_u16_v1'):
            self.check(fixture(image,8),mapping)

    def test_invalid_declaration_preserved(self):
        frame=fixture(np.full((10,10),4096,np.uint16),12)
        for function in (REFERENCE_FEATURE_PIXELS,feature_pixels_lut):
            with self.assertRaises(pva.PvaMotionError):function(frame,'raw_robust_u16_v1')
            with self.assertRaises(ValueError):function(frame,'unknown')

    def test_pinned_transform_is_narrow(self):
        source=transformed()
        self.assertIn('if (weight[neighbor] <= 1.0e-7F)',source)
        self.assertIn('value += weight[neighbor] * frame[source_index]',source)
        self.assertIn('if (score > local_best_score)',source)
        self.assertIn('accumulator += static_cast<float>(polarity_sign) * sample',source)
        self.assertIn('frames*velocities*sizeof(StencilV11)',source)
        self.assertNotIn('fast_math',source)
        with self.assertRaises(ValueError):once('aa','a','b')

    def test_verifier_rejects_missing_nonfinite_or_duplicate_evidence(self):
        for values in ([],[1]*11,[float('nan')]*12,[-1]*12,[True]*12):
            with self.assertRaises(ValueError):numbers(values)
        self.assertEqual(numbers([1]*12)['median'],1)
        with self.assertRaises(ValueError):coverage([{'i':1},{'i':1}],('i',),[(1,),(2,)])


if __name__=='__main__':unittest.main()
