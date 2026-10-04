"""Pipeline-level failure reporting for the already frozen decode implementation."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from tiny_target.visible_baseline import VisibleConfig, run


class DecodePipelineFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'synthetic_input'
        self.source.write_bytes(b'Synthetic test fixture; capture is mocked.')
        self.config = self.root/'config.json'
        self.config.write_text(json.dumps(asdict(VisibleConfig(frame_decode_execution='prefetch_one'))))
        self.cap = MagicMock()
        self.cap.isOpened.return_value = True
        self.cap.get.side_effect = lambda prop: 10. if prop == cv2.CAP_PROP_FPS else 3
        self.frame = np.full((64, 64, 3), 80, np.uint8)
        probe = SimpleNamespace(pixel_format='bgr24', height=64, width=64, to_dict=lambda: {})
        for mock in (patch('tiny_target.visible_decode.cv2.VideoCapture', return_value=self.cap),
                     patch('tiny_target.visible_baseline.probe_video', return_value=probe)):
            mock.start()
            self.addCleanup(mock.stop)

    def assert_failed(self, kind, message):
        with self.assertRaisesRegex(kind, message):
            run(self.source, self.config, self.root/'output')
        self.assertFalse((self.root/'output/report.json').exists())
        failure = json.loads((self.root/'output/failure.json').read_text())
        self.assertIs(failure['completed'], False)
        self.cap.release.assert_called_once()
        return failure

    def test_decoder_exception_never_becomes_successful_short_clip(self):
        self.cap.read.side_effect = [(True, self.frame), ValueError('decoder fixture failure')]
        failure = self.assert_failed(ValueError, 'decoder fixture failure')
        self.assertEqual(failure['frames'], 1)

    def test_premature_eof_fails_expected_frame_count(self):
        self.cap.read.side_effect = [(True, self.frame), (False, None)]
        failure = self.assert_failed(RuntimeError, 'Incomplete decode: 1/3')
        self.assertEqual(failure['frames'], 1)

    def test_consumer_failure_does_not_leave_capture_open(self):
        self.cap.read.side_effect = [(True, self.frame)]*3 + [(False, None)]
        with patch('tiny_target.visible_baseline.CpuTranslation.update', side_effect=ValueError('motion fixture failure')):
            self.assert_failed(ValueError, 'motion fixture failure')

    def test_failure_before_worker_start_still_releases_capture(self):
        with patch('tiny_target.visible_baseline.VisiblePointDetector', side_effect=ValueError('detector init failure')):
            with self.assertRaisesRegex(ValueError, 'detector init failure'):
                run(self.source, self.config, self.root/'output')
        self.cap.read.assert_not_called()
        self.cap.release.assert_called_once()
        self.assertFalse((self.root/'output/report.json').exists())
