from dataclasses import replace
import json
from pathlib import Path
import unittest

from tiny_target.motion.pva_pyrlk import PvaMotionConfig, _harris_output_capacity


class HarrisCapacityTests(unittest.TestCase):
    def test_native_capacity_covers_every_nms_cell(self):
        self.assertEqual(_harris_output_capacity((2392, 1595)), 60300)
        for w, h in ((1, 1), (640, 480), (2392, 1595), (3264, 2448)):
            self.assertGreater(_harris_output_capacity((w, h)), ((w + 7) // 8) * ((h + 7) // 8))

    def test_invalid_geometry_rejected(self):
        for size in ((0, 10), (-1, 8), (20., 10), (True, 10)):
            with self.assertRaises(ValueError):
                _harris_output_capacity(size)

    def test_legacy_and_positional_contract_unchanged(self):
        self.assertEqual(PvaMotionConfig(.5).harris_capacity_policy, 'legacy_default')
        cfg = PvaMotionConfig(harris_capacity_policy='complete_grid')
        self.assertEqual(cfg.max_features, 1000)
        for kwargs in ({'harris_capacity_policy': 'automatic'}, {'harris_min_nms_distance': 0}):
            with self.assertRaises(ValueError):
                replace(cfg, **kwargs)

    def test_only_capacity_policy_changes_from_v3(self):
        root = Path(__file__).resolve().parents[2]
        base = json.loads((root/'configs/evaluation/raw16_motion_v3.json').read_text())
        candidate = json.loads((root/'configs/evaluation/raw16_motion_v4.json').read_text())
        self.assertEqual(candidate['motion'].pop('harris_capacity_policy'), 'complete_grid')
        self.assertEqual(candidate, base)


if __name__ == '__main__':
    unittest.main()
