"""Independently check diagnostic guard counts against clamped scalar windows."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_median_v19 import guard_counts


class MedianGuardV19Tests(unittest.TestCase):
    def test_clamped_window_counts(self):
        rng=np.random.default_rng(1941)
        special=np.asarray([0x80000000,1,0x807fffff,0x7f800000,0x7fc12345],np.uint32)
        for h,w in ((1,1),(1,19),(4,3),(17,31)):
            for seed in range(5):
                a=np.ones((h,w),np.float32);bits=a.view(np.uint32)
                bits[int(rng.integers(h)),int(rng.integers(w))]=special[seed]
                total=0
                for y in range(h):
                    for x in range(w):
                        flag=False
                        for dy in range(-2,3):
                            for dx in range(-2,3):
                                v=int(bits[min(h-1,max(0,y+dy)),min(w-1,max(0,x+dx))]);m=v&0x7fffffff
                                flag |= m>=0x7f800000 or v==0x80000000 or 0<m<0x00800000
                        total+=flag
                self.assertEqual(guard_counts(a)['guarded_windows'],total)

    def test_positive_zero_is_not_guarded(self):
        r=guard_counts(np.zeros((9,13),np.float32))
        self.assertEqual(r['guarded_windows'],0)
        self.assertEqual(r['negative_zero_pixels'],0)


if __name__=='__main__':unittest.main()
