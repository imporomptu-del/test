"""Generated-only native-size AOT input packaging checks, not detector validation.

No publisher imagery, annotations, acquisition timestamps, PVA, or CUDA are used.
The unchanged visible reader consumes a lossless gray FFV1 container at nominal
10 Hz; this does not establish support for irregular acquisition timestamps.
"""

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tiny_target.frame_source import probe_video
from tiny_target.visible_decode import VisibleFrameReader


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "Generated lossless decode test requires both ffmpeg and ffprobe",
)
class AotLosslessDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.height, cls.width = 2048, 2448
        cls.fps = 10
        x = np.arange(cls.width, dtype=np.uint16)[None, :]
        y = np.arange(cls.height, dtype=np.uint16)[:, None]
        shape = (cls.height, cls.width)
        cls.originals = [
            np.broadcast_to(x % 256, shape).astype(np.uint8),
            np.broadcast_to(y % 256, shape).astype(np.uint8),
            (((x // 7 + y // 11) % 2) * 255).astype(np.uint8),
            ((x * 13 + y * 17) % 256).astype(np.uint8),
        ]
        directory = tempfile.TemporaryDirectory(prefix="seaqr_generated_aot_decode_")
        cls.addClassCleanup(directory.cleanup)
        cls.video = Path(directory.name) / "generated_native_gray.avi"
        # AVI exposes an explicit frame count to both existing probe paths.
        # Raw uint8 pixels avoid any image-loader, colorspace, or scaling step.
        encoded = subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-nostdin", "-v", "error",
                "-f", "rawvideo", "-pixel_format", "gray",
                "-video_size", f"{cls.width}x{cls.height}",
                "-framerate", str(cls.fps), "-i", "pipe:0",
                "-an", "-c:v", "ffv1", "-level", "3",
                "-pix_fmt", "gray", "-threads", "1",
                str(cls.video),
            ],
            input=b"".join(frame.tobytes(order="C") for frame in cls.originals),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=45,
        )
        if encoded.returncode:
            raise AssertionError(
                "Generated FFV1 encoding failed: "
                + encoded.stderr.decode("utf-8", errors="replace")
            )

    def test_probe_preserves_native_size_gray_format_count_and_cadence(self):
        np.testing.assert_array_equal(
            np.unique(self.originals[0]), np.arange(256, dtype=np.uint8)
        )
        probe = probe_video(self.video)
        self.assertEqual(probe.codec, "ffv1")
        self.assertIn(probe.pixel_format, {"gray", "gray8"})
        self.assertEqual((probe.width, probe.height), (self.width, self.height))
        self.assertEqual(probe.frame_rate, Fraction(self.fps, 1))
        self.assertEqual(probe.interval_ns, 100_000_000)
        self.assertEqual(probe.declared_frame_count, len(self.originals))

    def check_reader(self, execution):
        decoded = []
        with VisibleFrameReader(
            self.video, (self.height, self.width), execution=execution
        ) as reader:
            self.assertEqual(reader.fps, self.fps)
            self.assertEqual(reader.expected, len(self.originals))
            while True:
                frame, wait_ms = reader.read()
                self.assertGreaterEqual(wait_ms, 0)
                if frame is None:
                    break
                decoded.append(frame)
            self.assertIsNone(reader.read()[0])
        # Verify after full decoding as well, catching reused/overwritten buffers.
        self.assertEqual(len(decoded), len(self.originals))
        for index, (frame, original) in enumerate(zip(decoded, self.originals)):
            self.assertEqual(frame.index, index)
            self.assertEqual(frame.gray.dtype, np.uint8)
            self.assertEqual(frame.gray.shape, (self.height, self.width))
            np.testing.assert_array_equal(frame.gray, original)
        stats = reader.completed_stats()
        self.assertEqual(stats["decoded_frames"], len(self.originals))
        self.assertEqual(stats["consumed_frames"], len(self.originals))
        self.assertEqual(stats["read_calls"], len(self.originals) + 1)
        self.assertEqual(stats["dropped_frames"], 0)
        self.assertLessEqual(
            stats["maximum_observed_frames_ahead"], int(execution == "prefetch_one")
        )
        self.assertTrue(stats["worker_joined"])
        self.assertTrue(stats["capture_released"])

    def test_sequential_reader_preserves_exact_pixels_and_order(self):
        self.check_reader("sequential")

    def test_prefetch_reader_preserves_exact_pixels_and_order(self):
        self.check_reader("prefetch_one")


if __name__ == "__main__":
    unittest.main()
