import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('nsys_timeline_v22',
    Path(__file__).resolve().parents[2]/'scripts/nsys_timeline_v22.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntervalTests(unittest.TestCase):
    def test_union_does_not_sum_overlap(self):
        self.assertEqual(module.merge([(0, 10), (5, 8), (10, 12), (15, 17)]), [(0, 12), (15, 17)])
        self.assertEqual(module.length([(0, 10), (5, 8), (10, 12)]), 12)

    def test_clipping_and_empty(self):
        self.assertEqual(module.clip([(-3, 4), (5, 7), (12, 20)], 0, 6), [(0, 4), (5, 6)])
        self.assertEqual(module.merge([]), [])
        self.assertEqual(module.merge([(2, 2)]), [])

    def test_intersection(self):
        self.assertEqual(module.intersection([(0, 5), (8, 12)], [(3, 10)]), [(3, 5), (8, 10)])
        self.assertEqual(module.intersection([(0, 3)], [(3, 4)]), [])

    def test_invalid_intervals(self):
        with self.assertRaises(ValueError):
            module.merge([(2, 1)])
        with self.assertRaises(ValueError):
            module.clip([], 1, 1)


if __name__ == '__main__':
    unittest.main()
