import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT/'scripts' if (ROOT/'scripts').is_dir() else HERE
spec = importlib.util.spec_from_file_location('shadow_runner_tested', SCRIPTS/'run_weak_continuation_shadow_v1.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PLAN = ROOT/'configs/evaluation/weak_continuation_shadow_v1.json'
if not PLAN.is_file():
    PLAN = HERE/'plan.json'


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(PLAN.read_text())

    def test_frozen_schedule(self):
        runner.validate_plan(self.plan)
        self.assertEqual(sum(last-first+1 for v in self.plan['clips'].values()
                             for first,last in v['weak_windows_inclusive']), 182)
        self.assertTrue(runner.scheduled(self.plan['clips']['0126'], 216))
        self.assertFalse(runner.scheduled(self.plan['clips']['0126'], 219))

    def test_reject_scope_and_identity_change(self):
        for field,value in [('frames',675),('source_sha256','0'*64)]:
            plan = copy.deepcopy(self.plan)
            plan['clips']['0126'][field] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)
        self.plan['clips']['0240'] = self.plan['clips']['0126']
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan)

    def test_reject_overlapping_windows(self):
        self.plan['clips']['0029']['weak_windows_inclusive'] = [[70,81],[81,90]]
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan)

    def test_reject_tuned_policy(self):
        self.plan['shadow']['weak_budget_per_strong_gap'] = 2
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan)

    def test_exclusive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            runner.write(path, {'ok':True})
            with self.assertRaises(FileExistsError):
                runner.write(path, {'ok':False})
            self.assertEqual(runner.read(path), {'ok':True})

    def test_reject_duplicate_json(self):
        with patch.object(Path, 'read_text', return_value='{"a":1,"a":2}'):
            with self.assertRaises(ValueError):
                runner.read(Path('fixture.json'))

    def test_rectangle_dedup_ignores_selection_center_not_bounds(self):
        a = {'capture_bounds_exclusive_xyxy':[254,254,514,514],
             'tile_bounds_exclusive_xyxy':[256,256,512,512], 'probe_xy':[300.,300.]}
        b = copy.deepcopy(a)
        b['probe_xy'] = [310.,310.]
        self.assertEqual(runner.rectangle_key(a), runner.rectangle_key(b))
        b['tile_bounds_exclusive_xyxy'][0] = 255
        self.assertNotEqual(runner.rectangle_key(a), runner.rectangle_key(b))

    def test_wrong_clip_fails_before_dependency_or_media(self):
        with patch.object(runner, 'verify') as verify:
            with self.assertRaises(ValueError):
                runner.run('0240', 'shadow')
            verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
