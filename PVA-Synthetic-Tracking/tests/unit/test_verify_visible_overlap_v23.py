"""Generated checks for the independent, queue-aware v23 acceptance decision."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from verify_visible_overlap_v23 import distribution, intersection_ns, performance


def trials(speedup=1.3, candidate_age=600):
    rows, samples = {}, {}
    for clip in ('0126', '0082'):
        for i in range(3):
            for mode in ('reference', 'overlap'):
                name=f'{clip}_repeat{i}_{mode}'
                fps=2.5*(speedup if mode=='overlap' else 1)
                times=dict(cadence=[1000/fps]*128,
                           queue_aware=[candidate_age if mode=='overlap' else 700]*128,
                           preparation=[100]*128)
                rows[name]=dict(name=name,count=128,wall_s=128/fps,fps=fps,traced=False,
                    cadence=distribution(times['cadence']),queue_aware_latency=distribution(times['queue_aware']),
                    preparation=distribution(times['preparation']),process_peak_rss_kib=1234)
                samples[name]=times
    return rows,samples


class OverlapVerifierTests(unittest.TestCase):
    def test_substantial_gain_and_lower_latency_pass(self):
        result=performance(*trials())
        self.assertTrue(result['complete'])
        self.assertTrue(result['passed'])

    def test_gain_below_twenty_percent_is_rejected(self):
        self.assertFalse(performance(*trials(speedup=1.19))['passed'])

    def test_higher_throughput_does_not_hide_queue_age_regression(self):
        result=performance(*trials(candidate_age=900))
        self.assertFalse(result['passed'])
        for clip in result['by_clip'].values():
            self.assertTrue(clip['p95_regressions']['queue_aware_latency']['consistent'])
            self.assertFalse(clip['p95_regressions']['cadence']['consistent'])

    def test_one_paired_slowdown_rejects_even_if_pooled_gain_passes(self):
        rows,samples=trials(speedup=2)
        for mode,fps in [('reference',2.5),('overlap',2.49)]:
            row=rows['0126_repeat0_'+mode]
            row.update(fps=fps,wall_s=128/fps)
        result=performance(rows,samples)
        self.assertGreater(result['by_clip']['0126']['speedup'],1.2)
        self.assertFalse(result['passed'])

    def test_missing_workload_cannot_pass(self):
        rows,samples=trials()
        del rows['0082_repeat2_overlap']
        result=performance(rows,samples)
        self.assertFalse(result['complete'])
        self.assertFalse(result['passed'])

    def test_overlap_is_unioned_not_double_counted(self):
        self.assertEqual(intersection_ns([(0,20),(5,15)],[(10,30),(12,25)]),10)


if __name__=='__main__':
    unittest.main()
