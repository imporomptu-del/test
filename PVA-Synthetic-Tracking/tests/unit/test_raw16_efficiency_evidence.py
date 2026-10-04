"""Media-free guardrails for bounded RAW16 execution evidence."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/profile_raw16_efficiency.py'
SPEC = importlib.util.spec_from_file_location('raw16_efficiency', SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Raw16EvidenceTests(unittest.TestCase):
    def test_only_explicit_development_inputs(self):
        for clip in ('0029', '0040'):
            video, sidecar = module.source_paths(clip)
            self.assertEqual(video.name, f'chunk_{clip}.mkv')
            self.assertEqual(sidecar.name, f'chunk_{clip}_timestamps.csv')
        for invalid in ('other', '../0029', '29', '0126', '', '0029/..'):
            with self.assertRaises(ValueError):
                module.source_paths(invalid)

    def test_hash_preserves_sub_8bit_intensity_differences(self):
        a = np.array([[32768, 32769]], dtype=np.uint16)
        b = np.array([[32768, 32770]], dtype=np.uint16)
        self.assertNotEqual(module.compact(a), module.compact(b))
        self.assertNotEqual(module.compact(a), module.compact(a.astype(np.float32)))
        self.assertNotEqual(module.compact(a), module.compact(a.reshape(2, 1)))

    def test_strided_array_is_hashed_without_mutation(self):
        a = np.arange(32, dtype=np.float32).reshape(4, 8)[:, ::2]
        before = a.copy()
        self.assertEqual(module.compact(a), module.compact(a.copy()))
        np.testing.assert_array_equal(a, before)

    def test_empty_motion_correspondences_are_supported(self):
        empty = module.compact(np.empty((0, 2), dtype=np.float32))
        self.assertEqual(empty['shape'], [0, 2])
        self.assertEqual(len(empty['sha256']), 64)

    def test_only_explicit_nondeterministic_fields_removed(self):
        value = dict(timings_ms={'total': 5}, timestamp_ns=7,
                     window_duration_s=1.5, valid_support_count=16,
                     cuda={'free_device_bytes_before': 1, 'library_sha256': 'abc',
                           'kernel_launch_count': 3}, unknown_runtime_metric=9)
        self.assertEqual(module.compact(value), dict(timestamp_ns=7,
            window_duration_s=1.5, valid_support_count=16,
            cuda={'library_sha256': 'abc', 'kernel_launch_count': 3}, unknown_runtime_metric=9))

    def test_default_mode_does_not_instrument_timing(self):
        instrument = module.Instrumentation('timed', None)
        self.assertEqual(instrument.call('probe', lambda: 5), 5)
        self.assertEqual(instrument.stages, {})

    def test_rejected_fit_nonfinite_state_is_preserved(self):
        self.assertEqual(module.compact(float('inf')), {'nonfinite_float': 'inf'})
        self.assertNotEqual(module.compact(float('inf')), module.compact(float('-inf')))


if __name__ == '__main__':
    unittest.main()
