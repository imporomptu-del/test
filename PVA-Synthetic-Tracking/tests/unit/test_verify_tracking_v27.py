from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from verify_tracking_v27 import summarize


def rows():
    return [dict(clip='chunk_'+c,repeat=i,mode=m,frames=128,profiled=False,
                 tracking_ms=[10. if m=='reference' else 8.]*128)
            for c in ('0126','0082') for i in range(2)
            for m in (('reference','candidate') if i==0 else ('candidate','reference'))]


class TrackingReplayAuditTests(unittest.TestCase):
    def test_equal_extent_pooled_tracking_time(self):
        r=summarize(rows());self.assertEqual(r['0126']['speedup'],1.25)
        self.assertEqual(r['0126']['saved_ms_per_frame'],2.)
        self.assertEqual(r['0082']['tracking_ms']['reference']['count'],256)

    def test_no_missing_duplicate_or_reordered_sample(self):
        a=rows()
        for invalid in (a[:-1],a+[a[-1]],list(reversed(a))):
            with self.assertRaisesRegex(ValueError,'schedule'):summarize(invalid)

    def test_profiled_or_wrong_extent_rejected(self):
        for field,value in (('profiled',True),('frames',127),('tracking_ms',[1.])):
            a=rows();a[0][field]=value
            with self.assertRaisesRegex(ValueError,'timing scope'):summarize(a)

    def test_bad_durations_rejected(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            a=rows();a[0]['tracking_ms'][2]=value
            with self.subTest(value=value),self.assertRaises(ValueError):summarize(a)

    def test_pool_durations_not_individual_speedups(self):
        a=rows();a[0]['tracking_ms']=[20.]*128
        r=summarize(a)['0126']
        self.assertEqual(r['speedup'],15/8)
        self.assertEqual(r['paired_speedups'],[20/8,10/8])


if __name__=='__main__':unittest.main()
