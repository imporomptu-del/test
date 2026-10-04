"""Ownership, boundedness, errors and closed-loop equivalence without real media."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tiny_target.visible_baseline import VisibleConfig, run
from tiny_target.visible_decode import VisibleFrameReader, decode_contract


class Capture:
    def __init__(self, count=6, *, fail_at=None, blocked=None, wrong_shape=False):
        self.count, self.fail_at, self.blocked = count, fail_at, blocked
        self.buffer = np.zeros((9 if wrong_shape else 8, 12, 3), np.uint8)
        self.index = self.releases = 0
        self.owners = []
        self.entered = threading.Event()
        self.fps = 10.

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.count

    def read(self):
        self.owners.append(threading.get_ident())
        self.entered.set()
        if self.blocked is not None:
            if not self.blocked.wait(3):
                raise RuntimeError('Test decoder was never unblocked')
        if self.index == self.fail_at:
            raise ValueError('decoder failed')
        if self.index == self.count:
            return False, None
        self.buffer.fill(self.index * 10)
        self.index += 1
        return True, self.buffer  # Deliberately reuse the decoder's buffer.

    def release(self):
        self.owners.append(threading.get_ident())
        self.releases += 1


class VisibleDecodeTests(unittest.TestCase):
    def reader(self, cap, mode='prefetch_one', limit=None):
        mock = patch('tiny_target.visible_decode.cv2.VideoCapture', return_value=cap)
        mock.start()
        self.addCleanup(mock.stop)
        return VisibleFrameReader('synthetic', (8, 12), mode, limit)

    def wait_slot(self, reader):
        with reader._condition:
            self.assertTrue(reader._condition.wait_for(
                lambda: reader._slot is not None or reader._done, timeout=2))

    def test_default_and_invalid_execution(self):
        self.assertEqual(VisibleConfig().frame_decode_execution, 'sequential')
        for mode in ('unbounded', 'drop_oldest', True, None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                VisibleConfig(frame_decode_execution=mode)

    def test_prefix_limit_validation(self):
        for limit in (0, -1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                VisibleFrameReader('synthetic', (8, 12), max_frames=limit)

    def test_order_independent_gray_ownership_and_single_capture_owner(self):
        for mode in ('sequential', 'prefetch_one'):
            with self.subTest(mode=mode):
                cap = Capture()
                with self.reader(cap, mode) as reader:
                    frames = []
                    while True:
                        frame, wait = reader.read()
                        self.assertGreaterEqual(wait, 0)
                        if frame is None:
                            break
                        frames.append(frame)
                    reader.close()
                    self.assertEqual(reader.completed_stats()['consumed_frames'], 6)
                self.assertEqual(cap.releases, 1)
                self.assertEqual(len(set(cap.owners)), 1)
                self.assertEqual(cap.owners[0] == threading.get_ident(), mode == 'sequential')
                for i, frame in enumerate(frames):
                    self.assertEqual(frame.index, i)
                    np.testing.assert_array_equal(frame.gray, np.full((8, 12), i*10, np.uint8))
                    self.assertFalse(np.shares_memory(frame.gray, cap.buffer))

    def test_one_slot_reserves_before_decode_and_cancel_wakes_producer(self):
        cap = Capture(100)
        with self.reader(cap) as reader:
            reader.start()
            self.wait_slot(reader)
            with reader._condition:
                self.assertEqual(cap.index, 1)
                self.assertEqual(reader.decoded, 1)
            reader.read()
            self.wait_slot(reader)
            self.assertEqual(reader.decoded, 2)
            self.assertEqual(reader.maximum_observed_frames_ahead, 1)
            reader.close()  # Buffered frame can be abandoned only on cancellation.
            self.assertFalse(reader._thread.is_alive())
            with self.assertRaisesRegex(RuntimeError, 'drain'):
                reader.completed_stats()
        self.assertEqual(cap.releases, 1)

    def test_prefix_does_not_decode_an_extra_frame(self):
        for mode in ('sequential', 'prefetch_one'):
            cap = Capture(100)
            with self.reader(cap, mode, 3) as reader:
                for _ in range(3):
                    self.assertIsNotNone(reader.read()[0])
                reader.close()
                stats = reader.completed_stats()
                self.assertEqual(stats['decoded_frames'], 3)
                self.assertEqual(stats['read_calls'], 3)
            self.assertEqual(cap.index, 3)

    def test_empty_and_repeated_eof(self):
        for mode in ('sequential', 'prefetch_one'):
            cap = Capture(0)
            with self.reader(cap, mode) as reader:
                self.assertIsNone(reader.read()[0])
                self.assertIsNone(reader.read()[0])
                reader.close()
                self.assertEqual(reader.completed_stats()['read_calls'], 1)

    def test_decode_failure_is_not_eof_and_is_released(self):
        for mode in ('sequential', 'prefetch_one'):
            cap = Capture(fail_at=2)
            with self.assertRaisesRegex(ValueError, 'decoder failed'):
                with self.reader(cap, mode) as reader:
                    self.assertEqual(reader.read()[0].index, 0)
                    self.assertEqual(reader.read()[0].index, 1)
                    reader.read()
            self.assertEqual(cap.releases, 1)

    def test_dimension_or_conversion_failure_propagates(self):
        cap = Capture(wrong_shape=True)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            with self.reader(cap) as reader:
                reader.read()
        self.assertEqual(cap.releases, 1)
        cap = Capture()
        with patch('tiny_target.visible_decode.cv2.cvtColor', side_effect=ValueError('conversion failed')):
            with self.assertRaisesRegex(ValueError, 'conversion failed'):
                with self.reader(cap) as reader:
                    reader.read()
        self.assertEqual(cap.releases, 1)

    def test_consumer_failure_joins_and_releases(self):
        cap = Capture(100)
        with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
            with self.reader(cap) as reader:
                reader.read()
                raise RuntimeError('consumer failed')
        self.assertFalse(reader._thread.is_alive())
        self.assertEqual(cap.releases, 1)

    def test_metadata_and_prestart_failure_release_on_caller(self):
        cap = Capture()
        cap.fps = float('nan')
        with self.assertRaisesRegex(ValueError, 'fps'):
            with self.reader(cap):
                self.fail('Must not enter')
        self.assertEqual(cap.releases, 1)
        cap = Capture()
        with self.assertRaisesRegex(ValueError, 'initialization'):
            with self.reader(cap):
                raise ValueError('initialization failed')
        self.assertEqual(cap.owners, [threading.get_ident()])

    def test_thread_start_failure_releases_without_joining_unstarted_thread(self):
        cap = Capture()
        with patch('threading.Thread.start', side_effect=RuntimeError('start failed')):
            with self.assertRaisesRegex(RuntimeError, 'start failed'):
                with self.reader(cap) as reader:
                    reader.start()
        self.assertEqual(cap.releases, 1)

    def test_native_stall_cannot_claim_cleanup_or_cross_thread_release(self):
        unblock = threading.Event()
        cap = Capture(blocked=unblock)
        reader = self.reader(cap)
        reader.__enter__()
        reader._join_timeout = .02
        try:
            reader.start()
            self.assertTrue(cap.entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'did not stop'):
                reader.close()
            self.assertEqual(cap.releases, 0)
            with self.assertRaisesRegex(RuntimeError, 'drain'):
                reader.completed_stats()
        finally:
            unblock.set()
            reader._join_timeout = 2
            reader.close()
        self.assertEqual(len(set(cap.owners)), 1)
        self.assertEqual(cap.releases, 1)

    def test_read_after_close_is_rejected(self):
        with self.reader(Capture()) as reader:
            reader.close()
            with self.assertRaisesRegex(RuntimeError, 'not open'):
                reader.read()

    def test_real_mjpeg_decode_and_closed_loop_are_exact(self):
        # Synthetic pixels only; no development or holdout media is accessed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'synthetic.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (192, 128))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(91)
            background = rng.integers(20, 70, (128, 192, 3), dtype=np.uint8)
            for i in range(40):
                frame = background.copy()
                frame[55:58, 50+i:53+i] = 220
                writer.write(frame)
            writer.release()
            from types import SimpleNamespace
            probe = SimpleNamespace(pixel_format='yuvj420p', height=128, width=192,
                to_dict=lambda: dict(width=192, height=128, pixel_format='yuvj420p'))
            journals = []
            for mode in ('sequential', 'prefetch_one'):
                config = root / (mode + '.json')
                config.write_text(json.dumps(asdict(VisibleConfig(frame_decode_execution=mode))))
                with patch('tiny_target.visible_baseline.probe_video', return_value=probe):
                    report = run(source, config, root / mode)
                self.assertTrue(report['completed'])
                self.assertEqual(report['frames'], 40)
                self.assertEqual(report['frame_decode']['contract'], decode_contract(mode))
                rows = [json.loads(s) for s in (root / mode / 'frames.jsonl').read_text().splitlines()]
                def strip(value):
                    if isinstance(value, list):
                        return [strip(v) for v in value]
                    if isinstance(value, dict):
                        return {k:strip(v) for k,v in value.items() if k not in {'timings_ms','detection_ms'}}
                    return value
                journals.append(strip(rows))
            self.assertEqual(journals[0], journals[1])
