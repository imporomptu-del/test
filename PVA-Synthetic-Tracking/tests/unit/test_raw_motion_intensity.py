from dataclasses import replace
import unittest

import numpy as np

from tiny_target.motion.pva_pyrlk import PvaMotionConfig, PvaMotionError, _feature_u8, _feature_pixels, _motion_u8, _raw_asinh_lut
from tiny_target.types import Frame, TimestampSource


def frame(image, depth=16):
    return Frame(image, 0, 0, 'generated-raw-feature-mapping', depth, TimestampSource.MANIFEST)


class RawMotionIntensityTests(unittest.TestCase):
    def test_robust_u16_is_affine_invariant_and_flat_safe(self):
        raw = np.arange(1024, 5120, dtype=np.uint16).reshape(64, 64)
        original = frame(raw)
        reference = _feature_pixels(original, 'raw_robust_u16_v1')
        changed = _feature_pixels(frame(raw * 2 + 128), 'raw_robust_u16_v1')
        # Float32 normalization can differ by one rounded U16 code; no
        # detection or geometric acceptance tolerance is changed here.
        np.testing.assert_allclose(reference, changed, atol=1, rtol=0)
        np.testing.assert_array_equal(original.image, raw)
        self.assertGreater(len(np.unique(reference)), 256)
        for pixels in (np.full((64, 64), 2048, np.uint16), np.full((64, 64), 65535, np.uint16)):
            self.assertFalse(_feature_pixels(frame(pixels), 'raw_robust_u16_v1').any())
        self.assertFalse(_feature_pixels(replace(original, valid_mask=np.zeros(raw.shape, bool)), 'raw_robust_u16_v1').any())
        with self.assertRaises(ValueError):
            PvaMotionConfig(optical_flow_backend='CPU')
        with self.assertRaises(ValueError):
            PvaMotionConfig(feature_intensity_mapping='raw_robust_u16_v1')
        self.assertEqual(PvaMotionConfig(feature_intensity_mapping='raw_robust_u16_v1',
                         optical_flow_backend='CUDA').optical_flow_backend, 'CUDA')

    def test_linear_u16_retains_every_source_bit_and_does_not_wrap(self):
        for depth in (9, 10, 12, 14, 16):
            raw = np.arange(1 << depth, dtype=np.uint16).reshape(-1, 16)
            f = frame(raw, depth)
            before = f.pixel_sha256()
            pixels = _feature_pixels(f, 'raw_linear_u16_v1')
            np.testing.assert_array_equal(pixels >> (16 - depth), raw)
            self.assertEqual(pixels.dtype, np.uint16)
            self.assertTrue(pixels.flags.c_contiguous)
            self.assertEqual(before, f.pixel_sha256())
            signed = (pixels.astype(np.int32) - 32768).astype(np.int16)
            np.testing.assert_array_equal(signed.astype(np.int32) + 32768, pixels)
        f = frame(np.arange(256, dtype=np.uint8).reshape(16, 16), 8)
        np.testing.assert_array_equal(_feature_pixels(f, 'raw_linear_u16_v1'), _motion_u8(f))
        with self.assertRaises(PvaMotionError):
            _feature_pixels(frame(np.full((8, 8), 4096, np.uint16), 12), 'raw_linear_u16_v1')

    def test_default_and_positional_configuration_are_compatible(self):
        self.assertEqual(PvaMotionConfig().feature_intensity_mapping, 'bit_shift')
        self.assertEqual(PvaMotionConfig(.25).feature_image_scale, .25)
        with self.assertRaises(ValueError):
            PvaMotionConfig(feature_intensity_mapping='adaptive')

    def test_mapping_monotonic_bounded_deterministic_and_cached(self):
        for depth in (9, 10, 12, 14, 16):
            lut = _raw_asinh_lut(depth)
            self.assertIs(lut, _raw_asinh_lut(depth))
            self.assertEqual(lut.dtype, np.uint8)
            self.assertFalse(lut.flags.writeable)
            self.assertEqual((lut[0], lut[-1]), (0, 255))
            self.assertTrue(np.all(np.diff(lut.astype(int)) >= 0))
        for bad in (8, 17, 12.0, True):
            with self.assertRaises(ValueError):
                _raw_asinh_lut(bad)

    def test_legacy_mapping_is_byte_exact_for_entire_code_range(self):
        raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        f = frame(raw)
        self.assertEqual(_feature_u8(f, 'bit_shift').tobytes(), _motion_u8(f).tobytes())
        self.assertEqual(_feature_u8(f, 'bit_shift').tobytes(), (raw >> 8).astype(np.uint8).tobytes())

    def test_raw_bits_and_masks_are_not_mutated_or_used_to_tune_mapping(self):
        raw = np.arange(1536, 1664, dtype=np.uint16).reshape(8, 16)
        f = frame(raw)
        checksum = f.pixel_sha256()
        old, new = _feature_u8(f, 'bit_shift'), _feature_u8(f, 'raw_asinh_v1')
        self.assertEqual(np.ptp(old), 0)
        self.assertGreater(np.ptp(new), 0)
        self.assertEqual(checksum, f.pixel_sha256())
        self.assertTrue(new.flags.c_contiguous)
        masked = replace(f, valid_mask=np.zeros(raw.shape, bool))
        self.assertEqual(_feature_u8(masked, 'raw_asinh_v1').tobytes(), new.tobytes())
        self.assertFalse(f.image.flags.writeable)

    def test_8bit_passthrough_and_invalid_raw_contract(self):
        for dtype in (np.uint8, np.uint16):
            f = frame(np.arange(256, dtype=dtype).reshape(16, 16), 8)
            self.assertEqual(_feature_u8(f, 'raw_asinh_v1').tobytes(), _motion_u8(f).tobytes())
        with self.assertRaises(PvaMotionError):
            _feature_u8(frame(np.full((8, 8), 4096, np.uint16), 12), 'raw_asinh_v1')
        with self.assertRaises(PvaMotionError):
            _feature_u8(frame(np.zeros((8, 8), np.float32)), 'raw_asinh_v1')
        with self.assertRaises(ValueError):
            _feature_u8(frame(np.zeros((8, 8), np.uint16)), 'unknown')


if __name__ == '__main__':
    unittest.main()
