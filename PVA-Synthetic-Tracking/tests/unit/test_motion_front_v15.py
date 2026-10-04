import copy
from pathlib import Path
import sys
import unittest
import tempfile
from functools import partial
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from motion_front_pixels_v15 import proxy_pixels, logical_identity, FrontLibrary
from build_motion_front_v15 import build
from summarize_motion_front_v15 import validate_execution_fields
from tiny_target.types import Frame, TimestampSource
from tiny_target.motion.pva_pyrlk import _feature_pixels, PvaMotionError


def frame(image, depth, mask=None):
    return Frame(image, 0, 0, 'v15-unit', depth, TimestampSource.MANIFEST, valid_mask=mask)


class MotionFrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='seaqr-v15-test-')
        cls.library = FrontLibrary(build(Path(cls.temp.name)/'build'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.proxy = partial(proxy_pixels, library=self.library)

    def expected(self, f, mapping):
        mapped = _feature_pixels(f, mapping)
        work = mapped.astype(np.uint32)
        return ((work[::2, ::2]+work[::2, 1::2]+work[1::2, ::2]+work[1::2, 1::2]+2)//4).astype(mapped.dtype)

    def test_same_affine_rounding_all_codes_and_masks(self):
        codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        for depth in (9, 10, 12, 14, 16):
            image = (codes % (1 << depth)).astype(np.uint16) if depth < 16 else codes
            for mask in (None, image % 7 != 0, np.zeros(image.shape, bool)):
                f = frame(image, depth, mask)
                original = f.pixel_sha256()
                actual = self.proxy(f, 'raw_robust_u16_v1')
                expected = self.expected(f, 'raw_robust_u16_v1')
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertEqual(f.pixel_sha256(), original)

    def test_u8_and_flat_u16(self):
        for depth, image, mapping in ((8, np.arange(256, dtype=np.uint8).reshape(16, 16), 'bit_shift'),
                                      (16, np.full((16, 16), 3000, np.uint16), 'raw_robust_u16_v1')):
            f = frame(image, depth)
            np.testing.assert_array_equal(self.proxy(f, mapping), self.expected(f, mapping))

    def test_odd_geometry_other_scale_invalid_codes_fail(self):
        f = frame(np.zeros((16, 16), np.uint16), 12)
        for mapping, scale in (('raw_robust_u16_v1', .25), ('raw_asinh_v1', .5)):
            with self.assertRaises(ValueError):
                self.proxy(f, mapping, scale)
        with self.assertRaises(ValueError):
            self.proxy(frame(np.zeros((15, 16), np.uint8), 8), 'bit_shift')
        bad = np.zeros((16, 16), np.uint16)
        bad[1, 1] = 65535  # invalid even though this sample is NOT in the proxy
        with self.assertRaises(PvaMotionError):
            self.proxy(frame(bad, 12), 'raw_robust_u16_v1')

    def test_native_boundary_rejects_strides_and_formats(self):
        for image in (np.zeros((16, 16), np.uint16)[:, ::2], np.zeros((16, 16), np.float32),
                      np.empty((0, 16), np.uint16), np.zeros((15, 16), np.uint16)):
            with self.assertRaises(ValueError):
                self.library.prepare(image, 1., 0.)
        with self.assertRaises(RuntimeError):
            self.library.prepare(np.zeros((16, 16), np.uint16), float('nan'), 0.)

    def test_identity_excludes_only_declared_execution_fields(self):
        ref = dict(previous_points={'sha256': 'x'}, current_points={'sha256': 'y'},
                   motion_image_size=[8, 8], full_image_size=[16, 16],
                   backends={'motion_image_rescale': 'CUDA', 'flow': 'CUDA'},
                   metrics={'accepted_count': 4, 'memory_bytes': {'source_frames_read': 1024,
                                                                'motion_u16_frames_created': 1024}})
        other = copy.deepcopy(ref)
        other['backends']['motion_image_rescale'] = 'CPU_fused_half_linear_v15'
        other['metrics']['memory_bytes']['motion_u16_frames_created'] = 256
        validate_execution_fields(ref, 'reference')
        validate_execution_fields(other, 'candidate')
        broken_memory = copy.deepcopy(other)
        broken_memory['metrics']['memory_bytes']['motion_u16_frames_created'] = 1024
        with self.assertRaises(ValueError):
            validate_execution_fields(broken_memory, 'candidate')
        with self.assertRaises(ValueError):
            validate_execution_fields(ref, 'candidate')
        self.assertEqual(logical_identity(ref), logical_identity(other))
        self.assertEqual(ref['backends']['motion_image_rescale'], 'CUDA')
        for key in ('flow',):
            broken = copy.deepcopy(other)
            broken['backends'][key] = 'PVA'
            self.assertNotEqual(logical_identity(ref), logical_identity(broken))
        other['previous_points']['sha256'] = 'changed'
        self.assertNotEqual(logical_identity(ref), logical_identity(other))
        arbitrary = {'backends': {'rescale': 'CUDA'}, 'unrelated': True}
        self.assertEqual(logical_identity(arbitrary), arbitrary)


if __name__ == '__main__':
    unittest.main()
