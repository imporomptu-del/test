"""Bit-exact execution checks, independent of real media or the CUDA runtime."""
from dataclasses import replace
import unittest

import numpy as np

from tiny_target.dense_screen import DensePointScreener, DenseScreenConfig
from tiny_target.types import Frame, TimestampSource


class DenseBackgroundExecutionTests(unittest.TestCase):
    def test_reference_remains_default_and_unknown_mode_rejected(self):
        self.assertEqual(DenseScreenConfig().background_execution, 'indexed_reference')
        with self.assertRaises(ValueError):
            DenseScreenConfig(background_execution='auto')

    def exercise(self, dtype, normal_rate, outlier_rate, noise_floor):
        cfg = DenseScreenConfig(crop_width=64, crop_height=48, opencv_threads=1,
            background_update_rate=normal_rate, background_outlier_update_rate=outlier_rate,
            noise_sigma_floor_dn=noise_floor, max_shortlist_tracks_per_clip=4)
        reference = DensePointScreener(cfg)
        candidate = DensePointScreener(replace(cfg, background_execution='masked_ufunc'))
        rng = np.random.default_rng(1845)
        for index in range(48):
            image = rng.integers(1000, 1200, (48, 64)).astype(dtype)
            # Adjacent RAW16 levels; dark/saturated samples; bright and dark
            # outliers; sub-DN interpolated values on the stabilized float path.
            image[12, 12:15] = [32768, 32769, 32770]
            image[5:8, 4:7] = 0
            image[20:24, 33:38] = 65535
            image[30, 10 + index % 30] = 50000
            image[31, 10 + index % 30] = 2
            if dtype == np.float32:
                image[10:12, :] += np.float32(0.03125)
            valid = rng.random(image.shape) > .17
            if index in (7, 20):
                valid[:] = False
            if index in (8, 21):
                valid[:] = True
            # Gaps/segment resets must reset the model without stale scratch data.
            frame = Frame(image=image, timestamp_ns=index * 173_000_001,
                frame_index=index, source_id='generated-raw16-equivalence', bit_depth=16,
                timestamp_source=TimestampSource.SIDECAR_UNIX_NS,
                source_timestamp_ns=1_700_000_000_000_000_000 + index * 173_000_001,
                valid_mask=valid)
            for screener in (reference, candidate):
                screener.process(frame, segment_index=index // 16)
            for name in ('_background_location', '_background_variance', '_background_support'):
                a, b = getattr(reference, name), getattr(candidate, name)
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.tobytes(), b.tobytes(), (index, name, dtype))
            a, b = reference._last_synthetic_frame, candidate._last_synthetic_frame
            self.assertEqual(a is None, b is None)
            if a is not None:
                self.assertEqual(a.response.tobytes(), b.response.tobytes(), index)
                self.assertEqual(a.valid_mask.tobytes(), b.valid_mask.tobytes(), index)
                self.assertEqual(a.timestamp_ns, b.timestamp_ns)
                self.assertEqual(a.detection_ready, b.detection_ready)
            if index in (10, 25):
                for screener in (reference, candidate):
                    screener._background_support[13:16, 13:16] = np.iinfo(np.uint16).max
        self.assertEqual(reference.finalize(), candidate.finalize())

    def test_uint16_and_interpolated_float_are_bit_exact(self):
        for dtype in (np.uint16, np.float32):
            for rates in ((.2, .03, 16.), (1., 1., 1.), (.7, .13, .03125)):
                with self.subTest(dtype=dtype, rates=rates):
                    self.exercise(dtype, *rates)


if __name__ == '__main__':
    unittest.main()
