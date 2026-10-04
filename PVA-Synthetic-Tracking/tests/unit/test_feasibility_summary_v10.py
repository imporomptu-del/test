import copy
import itertools
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_feasibility_v10 import exact_keys,stats,validate_timing


class FeasibilitySummaryV10Tests(unittest.TestCase):
    def record(self):
        modes=('resident_device','resident_host')
        schedule=[dict(scene=s,repeat=r,mode=m) for s in ('dense','holes') for r in range(4)
            for m in (modes if r%2==0 else modes[::-1])]
        return dict(passed=True,pipeline_benchmark=False,shape=[3190,4784],window_frames=16,
            stride_frames=8,velocity_count=48,schedule=schedule,trials=[dict(**t,
            initialization_and_warmup_s=1,samples=[dict(last_frame=end,advance_frames=8,
                host_s=.91,kernel_ms=665,append_host_s=.244,tracking_host_s=.666,
                download_host_s=0,outputs_exact=True,frames=[dict(index=i,host_s=.0305,
                    frontend_compute_ms=18) for i in range(end-7,end+1)])
                for end in (27,35,43)]) for t in schedule])

    def test_finite_nonempty_timings(self):
        for values in ([],[float('nan')],[float('inf')],[-1],[True]):
            with self.assertRaises(ValueError):stats(values)
        self.assertEqual(stats([1,3])['median'],2)

    def test_exact_coverage_rejects_duplicates(self):
        expected=list(itertools.product(('dense','holes'),(0,1)))
        rows=[dict(scene=s,index=i) for s,i in expected]
        exact_keys(rows,('scene','index'),expected)
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):exact_keys(rows,('scene','index'),expected)

    def test_overbudget_is_not_pipeline_success(self):
        groups=validate_timing(self.record(),('resident_device','resident_host'),(27,35,43))
        self.assertEqual(groups['dense/resident_device']['host_s']['n'],12)
        self.assertFalse(groups['dense/resident_device']['all_measured_cycles_within_800ms'])
        self.assertLess(groups['dense/resident_device']['remaining_800ms_budget_s'],0)

    def test_missing_failed_reordered_or_nan_rejected(self):
        original=self.record()
        for mutate in (
            lambda r:r.update(passed=False),lambda r:r.update(pipeline_benchmark=True),
            lambda r:r['trials'].pop(),lambda r:r['trials'].reverse(),
            lambda r:r['trials'][0]['samples'].pop(),
            lambda r:r['trials'][0]['samples'][0].update(outputs_exact=False),
            lambda r:r['trials'][0]['samples'][0].update(host_s=float('nan')),
            lambda r:r['trials'][0]['samples'][0]['frames'].pop(),
            lambda r:r.update(velocity_count=24)):
            r=copy.deepcopy(original);mutate(r)
            with self.assertRaises(ValueError):validate_timing(r,('resident_device','resident_host'),(27,35,43))


if __name__=='__main__':unittest.main()
