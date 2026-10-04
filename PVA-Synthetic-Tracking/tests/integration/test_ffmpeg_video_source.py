from __future__ import annotations

import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tiny_target.frame_source import FfmpegVideoSource
from tiny_target.types import Discontinuity, TimestampSource


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg tools are required",
)
class FfmpegVideoSourceTests(unittest.TestCase):
    def test_decodes_gray16_and_uses_sidecar_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "fixture.mkv"
            timestamp_path = root / "fixture_timestamps.csv"
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=8x6:rate=10",
                    "-frames:v",
                    "5",
                    "-pix_fmt",
                    "gray16le",
                    "-c:v",
                    "ffv1",
                    str(video_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with timestamp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_index", "unix_time_ns"])
                writer.writerow([0, 1_000_000_000])
                writer.writerow([1, 1_100_000_000])
                writer.writerow([2, 1_200_000_000])
                writer.writerow([3, 1_300_000_000])
                writer.writerow([4, 1_700_000_000])

            source = FfmpegVideoSource(
                video_path,
                timestamp_csv=timestamp_path,
                timestamp_policy="require_sidecar",
                bit_depth=16,
                max_frames=5,
            )
            frames = list(source)

            self.assertEqual(len(frames), 5)
            self.assertTrue(all(frame.image.dtype == np.uint16 for frame in frames))
            self.assertTrue(all(frame.shape == (6, 8) for frame in frames))
            self.assertEqual(
                [frame.timestamp_ns for frame in frames],
                [0, 100_000_000, 200_000_000, 300_000_000, 700_000_000],
            )
            self.assertTrue(
                all(
                    frame.timestamp_source == TimestampSource.SIDECAR_UNIX_NS
                    for frame in frames
                )
            )
            self.assertIn(Discontinuity.TIMESTAMP_GAP, frames[-1].discontinuities)
            self.assertEqual(source.expected_interval_ns, 100_000_000)
            self.assertEqual(
                source.expected_interval_source,
                "sidecar_median_positive_delta",
            )


if __name__ == "__main__":
    unittest.main()
