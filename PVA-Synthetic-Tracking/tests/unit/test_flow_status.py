from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import numpy as np

from tiny_target.motion.pva_pyrlk import PvaMotionConfig, PvaMotionError, _fresh_flow_status


class FakeArray:
    def __init__(self, count, dtype):
        self.data = np.full(count, 255, dtype=np.uint8)  # stale/lost storage
        self.size = 0

    @staticmethod
    def zeros(count, dtype):
        array = FakeArray(count, dtype)
        array.data[:] = 0
        return array

    @contextmanager
    def rwlock_cpu(self):
        yield self.data[:self.size]


class FlowStatusTests(unittest.TestCase):
    def setUp(self):
        self.vpi = SimpleNamespace(Array=FakeArray, Type=SimpleNamespace(U8='u8'))

    def test_new_features_explicitly_clear_all_stale_slots(self):
        status = _fresh_flow_status(self.vpi, 12)
        self.assertEqual(status.size, 12)
        np.testing.assert_array_equal(status.data, np.zeros(12, np.uint8))

    def test_backward_keeps_forward_failures_without_aliasing(self):
        flags = np.array([0, 1, 0, 255], np.uint8)
        copied = _fresh_flow_status(self.vpi, 4, flags)
        np.testing.assert_array_equal(copied.data, flags)
        copied.data[0] = 1
        self.assertEqual(flags[0], 0)
        self.assertEqual(copied.data[1], 1)

    def test_size_mismatch_fails_closed(self):
        with self.assertRaises(PvaMotionError):
            _fresh_flow_status(self.vpi, 3, np.zeros(2, np.uint8))

    def test_opt_in_and_backend_guard(self):
        self.assertEqual(PvaMotionConfig().flow_status_policy, 'legacy_default')
        PvaMotionConfig(flow_status_policy='fresh_per_pair', optical_flow_backend='CUDA')
        with self.assertRaises(ValueError):
            PvaMotionConfig(flow_status_policy='fresh_per_pair')
        with self.assertRaises(ValueError):
            PvaMotionConfig(flow_status_policy='always_accept', optical_flow_backend='CUDA')


if __name__ == '__main__':
    unittest.main()
