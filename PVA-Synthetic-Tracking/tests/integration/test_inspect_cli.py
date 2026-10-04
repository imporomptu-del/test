from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tiny_target.cli import inspect_config


class InspectConfigTests(unittest.TestCase):
    def test_report_has_deterministic_frame_set_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for index in range(2):
                image_path = root / f"frame_{index}.npy"
                np.save(image_path, np.full((2, 3), index, dtype=np.uint8))
                entries.append(
                    {
                        "path": image_path.name,
                        "frame_index": index,
                        "timestamp_ns": index * 100,
                        "bit_depth": 8,
                    }
                )
            manifest_path = root / "frames.json"
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
            config_path = root / "config.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input": {
                            "source": "npy_manifest",
                            "path": manifest_path.name,
                        },
                        "output": {},
                    }
                ),
                encoding="utf-8",
            )

            first = inspect_config(config_path)
            second = inspect_config(config_path)

            self.assertEqual(first["inspection"]["frame_count"], 2)
            self.assertEqual(
                first["inspection"]["frame_set_sha256"],
                second["inspection"]["frame_set_sha256"],
            )
            self.assertEqual(first["inspection"]["dtypes"], {"|u1": 2})
            self.assertEqual(first["warnings"], [])


if __name__ == "__main__":
    unittest.main()
