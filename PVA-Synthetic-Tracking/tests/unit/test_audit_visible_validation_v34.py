"""Report-only v34 audit maths, scope and negative checks; no hardware/media."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import audit_visible_validation_v34 as a
from export_visible_validation_v34 import files_for
import visible_validation_v34 as supervisor


class AuditTests(unittest.TestCase):
    def test_independent_schedule(self):
        self.assertEqual(a.specs(), supervisor.schedule())
        self.assertEqual(len(a.specs()), 16)
        self.assertEqual(sum(s['kind'] == 'full' for s in a.specs()), 12)

    def test_allowlist(self):
        paths = files_for(a.specs())
        self.assertEqual(len(paths), 160)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertFalse(any(Path(p).suffix in ('.avi', '.raw16', '.so') for p in paths))
        self.assertEqual(sum(p.endswith('.sqlite') for p in paths), 2)
        for bad in ('../escape', '/tmp/x', '..', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                files_for([dict(name=bad, traced=False)])

    def test_percentiles_and_strict_deadline(self):
        d = a.distribution([0, 100, 200])
        self.assertEqual(d['median'], 100)
        self.assertEqual(d['p95'], 190)
        self.assertEqual(d['p99'], 198)
        self.assertEqual(d['fraction_over_100_ms'], 1/3)
        self.assertEqual(a.distribution([12])['p99'], 12)

    def test_invalid_samples(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(AssertionError):
                a.distribution(values)

    def test_occupancy_not_just_peak(self):
        admission = dict(maximum=2, events=[dict(admitted_ns=1, released_ns=11),
            dict(admitted_ns=6, released_ns=16)])
        before = copy.deepcopy(admission)
        r = a.occupancy(admission)
        self.assertEqual(r['time_fraction'], {'0': 0, '1': 2/3, '2': 1/3})
        self.assertAlmostEqual(r['mean_held'], 4/3)
        self.assertEqual(admission, before)

    def test_occupancy_tied_release_admit(self):
        r = a.occupancy(dict(maximum=1, events=[dict(admitted_ns=1, released_ns=6),
            dict(admitted_ns=6, released_ns=11)]))
        self.assertEqual(r['maximum'], 1)
        self.assertEqual(r['mean_held'], 1)

    def test_bad_occupancy(self):
        for v in (dict(maximum=2, events=[dict(admitted_ns=1, released_ns=2)]),
                dict(maximum=3, events=[dict(admitted_ns=1, released_ns=2)]*3),
                dict(maximum=1, events=[dict(admitted_ns=2, released_ns=1)])):
            with self.subTest(v=v), self.assertRaises(AssertionError):
                a.occupancy(v)

    def test_windows(self):
        frames = [dict(consumer_complete_ns=(i+1)*100_000_000) for i in range(30)]
        self.assertEqual(a.windows(frames), [10.0]*20)
        frames[5]['consumer_complete_ns'] = frames[4]['consumer_complete_ns']
        with self.assertRaises(AssertionError):
            a.windows(frames)

    def test_runtime_rejects_policy_changes(self):
        reference = dict(blas=[dict(threads=12)], affinity=[0, 1], numpy='x', opencv='y',
            thread_environment=dict.fromkeys(('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'GOTO_NUM_THREADS')), clock_ticks=100, opencv_threads=12)
        a.runtime(reference, reference)
        after = copy.deepcopy(reference)
        after['opencv_threads'] = 2
        a.runtime(after, reference, after=True)
        for key, value in (('affinity', [0]), ('opencv_threads', 1), ('blas', [dict(threads=1)])):
            changed = copy.deepcopy(after)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                a.runtime(changed, reference, after=True)

    def fixture(self):
        trials, samples, executions = {}, {}, {}
        for c, count in a.COUNTS.items():
            for i in range(3):
                n = f'full_repeat{i}_{c}'
                # Different wall times catch incorrect arithmetic averaging of FPS.
                wall = (i+1)*count/10
                trials[n] = dict(kind='full', traced=False, state_audit=False, frames=None,
                    count=count, fps=count/wall, wall_s=wall, process_peak_rss_kib=1024)
                samples[n] = dict(consumer_cadence=[100*(i+1)]*count)
                executions[n] = dict(frames=[dict(consumer_complete_ns=(j+1)*100_000_000) for j in range(count)],
                    admission=dict(maximum=1, events=[dict(admitted_ns=1, released_ns=11)]))
        return trials, samples, executions

    def test_pooled_not_arithmetic_fps_and_excludes_traces(self):
        trials, samples, executions = self.fixture()
        trials['trace_0082'] = dict(kind='trace', fps=9999)
        result = a.summarize(trials, samples, executions)
        for c, r in result.items():
            self.assertEqual(r['pooled_fps'], 5)
            self.assertEqual(r['frame_and_stage_ms']['consumer_cadence']['count'], 3*a.COUNTS[c])

    def test_contaminated_or_truncated_clean_run_rejected(self):
        for key, value in (('kind', 'trace'), ('traced', True), ('state_audit', True), ('count', 128)):
            trials, samples, executions = self.fixture()
            trials['full_repeat0_0126'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                a.summarize(trials, samples, executions)


if __name__ == '__main__':
    unittest.main()
