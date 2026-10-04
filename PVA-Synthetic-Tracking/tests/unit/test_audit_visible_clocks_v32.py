"""Read-only summary tests and delayed-policy diagnostic; fake hardware only."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import audit_visible_clocks_v32 as audit
import visible_clocks_v32 as frozen


def fixture():
    trials, samples = {}, {}
    for s in audit.specs()[:46]:
        if s['audit']:
            continue
        speed = 2.0 if s['fixed'] else 1.0
        trials[s['name']] = dict(count=128, fps=speed, wall_s=128/speed)
        samples[s['name']] = dict(consumer_cadence=[1000/speed]*128)
    return trials, samples


class AuditTests(unittest.TestCase):
    def test_independent_schedule_agrees_with_frozen_protocol(self):
        self.assertEqual(audit.specs(), frozen.schedule())
        self.assertEqual(audit.specs()[46]['name'], '0082_repeat2_auto_combined_default')

    def test_partial_matched_repeats_not_biased_by_extra_fixed_repeat(self):
        trials, samples = fixture()
        # An unmatched, dramatically faster third fixed-light run must not
        # contaminate the paired two-repeat comparison with automatic clocks.
        trials['0082_repeat2_fixed_combined']['fps'] = 999
        trials['0082_repeat2_fixed_combined']['wall_s'] = 128/999
        result = audit.summarize(trials, samples)
        self.assertEqual(result['0126']['matched_repeats'], [0,1,2])
        self.assertEqual(result['0082']['matched_repeats'], [0,1])
        self.assertEqual(result['0082']['cells']['fixed']['combined']['pooled_fps'], 2.0)
        self.assertEqual(result['0082']['clock_speedups']['combined']['pooled'], 2.0)

    def test_pooled_fps_uses_total_time_not_mean_fps(self):
        trials, samples = fixture()
        n = '0082_repeat1_fixed_combined'
        trials[n].update(fps=4.0, wall_s=32.0)
        result = audit.summarize(trials, samples)
        self.assertAlmostEqual(result['0082']['cells']['fixed']['combined']['pooled_fps'], 256/96)

    def test_no_complete_pairs_refused(self):
        trials, samples = fixture()
        for i in range(3):
            trials.pop(f'0126_repeat{i}_auto_combined')
        with self.assertRaises(AssertionError):
            audit.summarize(trials, samples)

    def test_delayed_readback_reproduces_false_failure_without_sysfs(self):
        values, pending = {}, {}
        for p in frozen.POLICIES.values():
            for key, value in ((p['minimum'], p['expected_min']), (p['maximum'], p['expected_max']),
                               (p['governor'], p['expected_governor'])):
                values[p['path']+'/'+key] = str(value)
        reader = lambda path: values[str(path)]
        def deferred_writer(path, value):
            self.assertIn(path, frozen.MIN_PATHS)
            pending[path] = str(value)
        def commit():
            values.update(pending)
            pending.clear()
        saved = frozen.snapshot(reader)
        # The unchanged frozen supervisor assumes sysfs readback is immediate.
        with self.assertRaisesRegex(ValueError, 'Clock policy no longer matches'):
            frozen.set_policy(saved, True, reader, deferred_writer)
        commit()
        self.assertEqual(frozen.snapshot(reader), frozen.expected_policy(saved, True))
        immediate = frozen.restore(saved, reader, deferred_writer)
        self.assertFalse(immediate['restored'])
        self.assertEqual(immediate['errors'], [])
        commit()
        later = frozen.restore(saved, reader, deferred_writer)
        self.assertTrue(later['restored'])
        self.assertEqual(later['actual'], saved)


if __name__ == '__main__':
    unittest.main()
