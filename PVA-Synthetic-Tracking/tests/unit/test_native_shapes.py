"""Exact compiled shape conformance, including inconsistent sparse overlaps."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tiny_target.visible_resident import SparseSpatial
from tiny_target.visible_shapes import consolidate_half_height
from tiny_target.visible_shapes_native import NativeShapes

ROOT = Path(__file__).resolve().parents[2]


def patch_input(image, points, *, scores=None):
    seeds = np.asarray(points, np.int32).reshape(-1, 2)
    patches = np.zeros((len(seeds), 17, 17), np.float32)
    proposals = []
    h, w = image.shape
    for i, (x, y) in enumerate(seeds):
        if x >= 8 and y >= 8 and x+8 < w and y+8 < h:
            patches[i] = image[y-8:y+9, x-8:x+9]
        proposals.append(dict(x=int(x), y=int(y), polarity='bright' if image[y, x] >= 0 else 'dark',
            score=float(abs(image[y, x])) if scores is None else float(scores[i]),
            response_dn=float(image[y, x]), noise_sigma_dn=1.))
    return proposals, seeds, patches


def random_case(rng, case):
    image = rng.normal(0, .2, (71, 103)).astype(np.float32)
    image[30, 16:33] = 10; image[40, 33:42] = -8
    image[20:26, 60:65] = 12
    if case % 3 == 0:
        image[21:25, 61:64] = 0
    image *= 10. ** ((case % 13)-6)
    mask = rng.random(image.shape) > (0 if case % 5 == 0 else .003)
    seeds = rng.integers([0, 0], [103, 71], (90, 2))
    seeds = np.concatenate((seeds, [[x, 30] for x in range(16, 33, 2)],
        [[x, 40] for x in range(33, 42, 2)], [[62, 20], [16, 30], [16, 30]]))
    rng.shuffle(seeds)
    proposals, seeds, patches = patch_input(image, seeds)
    return proposals, image.shape, seeds, patches, mask


class NativeShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='seaqr_native_shape_tests_')
        cls.library = Path(cls.tmp.name) / 'libseaqr_shapes.so'
        subprocess.run([sys.executable, str(ROOT/'scripts/build_phase20_native_shapes.py'),
            '--output', str(cls.library)], check=True, stdout=subprocess.DEVNULL)
        cls.native = NativeShapes(cls.library, hashlib.sha256(cls.library.read_bytes()).hexdigest())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, proposals, shape, seeds, patches, mask, support=True):
        inputs = copy.deepcopy(proposals), seeds.copy(), patches.copy(), mask.copy()
        expected = consolidate_half_height(proposals, SparseSpatial(shape, seeds, patches), mask, include_support=support)
        actual = self.native.consolidate(proposals, shape, seeds, patches, mask, include_support=support)
        self.assertEqual(expected, actual)
        self.assertEqual(proposals, inputs[0])
        for current, before in zip((seeds, patches, mask), inputs[1:]):
            np.testing.assert_array_equal(current, before)
        return actual

    def test_random_shapes_exact(self):
        rng = np.random.default_rng(917013)
        for case in range(400):
            with self.subTest(case=case):
                self.check_case(*random_case(rng, case), support=bool(case % 2))

    def test_inconsistent_overlapping_and_duplicate_patches(self):
        rng = np.random.default_rng(926014)
        for case in range(160):
            args = list(random_case(rng, case))
            patches = args[3].copy()
            # Deliberately different values at the same absolute coordinate:
            # windows use the last duplicate, coordinate values use the first.
            patches += rng.normal(0, .1, patches.shape).astype(np.float32)
            args[3] = patches
            self.check_case(*args, support=bool(case % 2))

    def test_strongest_ties_member_order_and_polarities(self):
        image = np.zeros((64, 64), np.float32)
        image[30, 25:33] = 10; image[40, 30:36] = -10
        points = [[32, 30], [25, 30], [32, 30], [35, 40], [30, 40]]
        ps, seeds, patches = patch_input(image, points, scores=[10]*5)
        ps[2]['tag'] = 'later duplicate'
        a, m = self.check_case(ps, image.shape, seeds, patches, np.ones_like(image, bool))
        self.assertEqual(m['merged_peak_count'], 3)
        self.assertEqual(a[0]['shape']['peak_reference_xy'], [25, 30])
        self.assertEqual(a[0]['shape']['member_peak_reference_xy'], points[:3])

    def test_asymmetric_valley_hollow_and_unsupported(self):
        image = np.zeros((64, 64), np.float32)
        image[30, 26:31] = 2; image[30, 26] = 20; image[30, 30] = 3
        image[47:54, 27:34] = 10; image[48:53, 28:33] = 0
        mask = np.ones_like(image, bool)
        ps, seeds, patches = patch_input(image, [[26, 30], [30, 30], [30, 47], [0, 0]])
        actual, _ = self.check_case(ps, image.shape, seeds, patches, mask)
        self.assertEqual([(p['x'], p['y']) for p in actual], [(p['x'], p['y']) for p in ps])
        self.assertEqual(actual[2:], ps[2:])
        mask[30, 28] = False
        self.check_case(ps, image.shape, seeds, patches, mask)

    def test_nontransitive_group_membership(self):
        image = np.zeros((64, 64), np.float32)
        image[30, 24:39] = 8
        image[30, 24], image[30, 31], image[30, 38] = 10, 16, 20
        ps, seeds, patches = patch_input(image, [[24, 30], [31, 30], [38, 30]])
        self.check_case(ps, image.shape, seeds, patches, np.ones_like(image, bool))

    def test_native_coordinates_and_full_candidate_cap(self):
        shape = (3190, 4784)
        seeds = np.asarray([[24+(i % 64)*64, 24+(i//64)*64] for i in range(512)], np.int32)
        dy, dx = np.mgrid[-8:9, -8:9]
        patches = np.repeat((20*np.exp(-(dx*dx+dy*dy)/5))[None], 512, axis=0).astype(np.float32)
        ps = [dict(x=int(x), y=int(y), polarity='bright', score=20., response_dn=20., noise_sigma_dn=1.) for x, y in seeds]
        self.check_case(ps, shape, seeds, patches, np.ones(shape, bool))

    def test_empty_and_noncontiguous_arrays(self):
        image = np.zeros((64, 64), np.float32); image[30, 30] = 10
        for points in ([], [[30, 30], [31, 30]]):
            ps, seeds, patches = patch_input(image, points)
            self.check_case(ps, image.shape, seeds[:, ::-1][:, ::-1], patches[:, :, ::-1], np.ones_like(image, bool).T)

    def test_subnormal_threshold_and_half_pixel_rounding(self):
        for height in (np.nextafter(np.float32(0), np.float32(1)), np.float32(1e-38), np.float32(1e30), np.float32(10)):
            image = np.zeros((64, 64), np.float32)
            image[30, 28:30] = height
            ps, seeds, patches = patch_input(image, [[28, 30], [29, 30]])
            self.check_case(ps, image.shape, seeds, patches, np.ones_like(image, bool))

    def test_invalid_inputs_fail_closed(self):
        image = np.zeros((32, 32), np.float32)
        ps, seeds, patches = patch_input(image, [[16, 16]])
        mask = np.ones_like(image, bool)
        for bad in (patches.astype(np.float64), patches[:, :16]):
            with self.assertRaises(ValueError):
                self.native.consolidate(ps, image.shape, seeds, bad, mask)
        with self.assertRaises(ValueError):
            self.native.consolidate(ps, image.shape, seeds, patches, mask.astype(np.uint8))
        with self.assertRaises(ValueError):
            self.native.consolidate([dict(ps[0], x=17)], image.shape, seeds, patches, mask)
        with self.assertRaises(RuntimeError):
            self.native.consolidate([dict(ps[0], x=-1)], image.shape, np.asarray([[-1, 16]], np.int32), patches, mask)
        with self.assertRaises(FileNotFoundError):
            NativeShapes(Path(self.tmp.name)/'missing.so', '0'*64)
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            NativeShapes(self.library, '0'*64)


if __name__ == '__main__':
    unittest.main()
