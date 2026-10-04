"""Additional targeted shape checks; do not alter the frozen runtime package."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tiny_target.visible_baseline import sha256
from tiny_target.visible_resident import SparseSpatial
from tiny_target.visible_shapes import consolidate_half_height
from tiny_target.visible_shapes_native import NativeShapes

ROOT = Path(__file__).resolve().parents[2]


class NativeShapeEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='seaqr_native_shape_edges_')
        path = Path(cls.tmp.name)/'native.so'
        subprocess.run([sys.executable, str(ROOT/'scripts/build_phase20_native_shapes.py'),
            '--output', str(path)], check=True, stdout=subprocess.DEVNULL)
        cls.native = NativeShapes(path, sha256(path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pairwise_chain_must_not_merge_transitively(self):
        im = np.zeros((64, 64), np.float32)
        im[30, 24:29] = [20, 10, 10, 5, 5]
        seeds = np.asarray([[24, 30], [26, 30], [28, 30]], np.int32)
        patches = np.stack([im[y-8:y+9, x-8:x+9] for x, y in seeds])
        ps = [dict(x=int(x), y=int(y), polarity='bright', score=float(im[y, x])) for x, y in seeds]
        mask = np.ones_like(im, bool)
        expected = consolidate_half_height(ps, SparseSpatial(im.shape, seeds, patches), mask, include_support=True)
        actual = self.native.consolidate(ps, im.shape, seeds, patches, mask, include_support=True)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], ps)
        self.assertEqual(actual[1]['merged_peak_count'], 0)
        self.assertEqual(actual[1]['unmodified_unbounded_or_unsupported_peaks'], 3)

    def test_first_pixel_last_window_precedence_preserves_zero_weight_error(self):
        seeds = np.asarray([[16, 16], [16, 16]], np.int32)
        patches = np.zeros((2, 17, 17), np.float32)
        patches[1, 8, 8] = 10
        ps = [dict(x=16, y=16, polarity='bright', score=10.) for _ in seeds]
        mask = np.ones((33, 33), bool)
        with self.assertRaisesRegex(ZeroDivisionError, 'Weights sum to zero'):
            consolidate_half_height(ps, SparseSpatial(mask.shape, seeds, patches), mask)
        with self.assertRaisesRegex(ZeroDivisionError, 'Weights sum to zero'):
            self.native.consolidate(ps, mask.shape, seeds, patches, mask)

    def test_noncontiguous_seed_stride_and_single_peak_rounding(self):
        padded = np.zeros((1, 4), np.int32); padded[:, ::2] = [[16, 16]]
        seeds = padded[:, ::2]
        self.assertFalse(seeds.flags.c_contiguous)
        patches = np.zeros((1, 17, 17), np.float32); patches[0, 8, 8:10] = 10
        ps = [dict(x=16, y=16, polarity='bright', score=10.)]
        mask = np.ones((33, 33), bool)
        actual = self.native.consolidate(ps, mask.shape, seeds, patches, mask, include_support=True)
        self.assertEqual(actual, consolidate_half_height(ps, SparseSpatial(mask.shape, seeds, patches), mask, include_support=True))
        self.assertEqual(actual[0][0]['shape']['centroid_reference_xy'], [16.5, 16.])
        self.assertEqual(actual[0][0]['x'], 16)


if __name__ == '__main__':
    unittest.main()
