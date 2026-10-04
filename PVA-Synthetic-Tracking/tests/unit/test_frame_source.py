from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tiny_target.frame_source import (
    FrameSourceError,
    NpyManifestSource,
    load_timestamp_csv,
)
from tiny_target.types import Discontinuity


class RecordedSourceTests(unittest.TestCase):
    def test_npy_manifest_replays_identically_and_keeps_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for index, timestamp_ns in enumerate((0, 100, 300)):
                image = np.full((3, 4), 1000 + index, dtype=np.uint16)
                image_path = root / f"frame_{index}.npy"
                np.save(image_path, image)
                entries.append(
                    {
                        "path": image_path.name,
                        "frame_index": index,
                        "timestamp_ns": timestamp_ns,
                        "bit_depth": 16,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "expected_interval_ns": 100,
                        "frames": entries,
                    }
                ),
                encoding="utf-8",
            )

            first = list(NpyManifestSource(manifest_path))
            second = list(NpyManifestSource(manifest_path))

            self.assertEqual(
                [frame.pixel_sha256() for frame in first],
                [frame.pixel_sha256() for frame in second],
            )
            self.assertTrue(all(frame.image.dtype == np.uint16 for frame in first))
            self.assertIn(Discontinuity.TIMESTAMP_GAP, first[-1].discontinuities)

    def test_timestamp_csv_requires_contiguous_frame_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timestamps.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_index", "unix_time_ns"])
                writer.writerow([0, 100])
                writer.writerow([2, 200])
            with self.assertRaisesRegex(FrameSourceError, "contiguous"):
                load_timestamp_csv(path)


if __name__ == "__main__":
    unittest.main()
