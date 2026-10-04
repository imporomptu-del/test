import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from run_motion_video_v13 import validate_request
from batch_motion_video_v13 import schedule, check_pair


class MotionVideoV13Tests(unittest.TestCase):
    def test_scope_rejects_unknown_media_and_extents(self):
        for branch, clip, frames, injected in (
            ('raw', '0126', 64, False), ('raw', '0029', 65, False),
            ('visible', '0027', None, False), ('visible', '0126', 32, False),
            ('raw', '0029', 64, True), ('visible', '0029', 96, True),
            ('invalid', '0029', 64, False)):
            with self.assertRaises(ValueError):
                validate_request(branch, clip, frames, injected)
        validate_request('raw', '0040', 64, True)
        validate_request('visible', '0126', None, False)

    def test_schedules_are_bounded_and_alternating(self):
        self.assertEqual(len(schedule('smoke')), 2)
        self.assertEqual(len(schedule('raw_checks')), 3)
        for stage in ('full', 'visible_repeats', 'raw_repeats'):
            rows = schedule(stage)
            self.assertEqual(len(rows), 8)
            self.assertEqual({row[-1] for row in rows}, {'reference', 'reuse'})
            for i in range(0, len(rows), 2):
                self.assertEqual(rows[i][:-1], rows[i+1][:-1])
                self.assertNotEqual(rows[i][-1], rows[i+1][-1])
            for branch, clip, frames, injected, _, _ in rows:
                validate_request(branch, clip, frames, injected)

    def records(self):
        a = dict(passed=True, closed=True, error=None, processed_frames=3,
                 mode='reference', pipeline_fps=1., reuse_hits=0, reuse_misses=0,
                 runtime_sha256={'test': 'hash'}, wrapper_sha256='x', adapter_sha256='y',
                 method_sha256='z', branch='raw', clip='0040', frames=64, injected=False,
                 motion=[dict(frame=i, identity={'point': i}, estimator_s=.3, rss=None)
                         for i in (1, 2)])
        b = copy.deepcopy(a)
        b.update(mode='reuse', reuse_hits=1, reuse_misses=1, pipeline_fps=1.2)
        return a, b

    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp)/name for name in ('before', 'after')]
            for path, value in zip(paths, (a, b)):
                path.with_suffix('.execution.json').write_text(json.dumps(value))
            return check_pair(*paths)

    def test_equal_outputs_different_timings_pass(self):
        a, b = self.records()
        b['motion'][0]['estimator_s'] = .2
        self.assertTrue(self.compare(a, b)['exact'])

    def test_lifecycle_identity_and_missing_reuse_fail(self):
        for key, value in (('closed', False), ('passed', False), ('error', 'failure'),
                           ('runtime_sha256', {}), ('reuse_hits', 0), ('mode', 'reference')):
            a, b = self.records()
            b[key] = value
            with self.assertRaises(AssertionError):
                self.compare(a, b)
        a, b = self.records()
        b['motion'][0]['identity']['point'] += 1
        with self.assertRaises(AssertionError):
            self.compare(a, b)
        a, b = self.records()
        b['motion'].pop()
        with self.assertRaises(AssertionError):
            self.compare(a, b)


if __name__ == '__main__':
    unittest.main()
