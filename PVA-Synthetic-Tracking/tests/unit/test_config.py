from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tiny_target.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_json_compatible_yaml_resolves_relative_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.yaml"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input": {
                            "source": "video",
                            "path": "recording.mkv",
                            "max_frames": 4,
                        },
                        "output": {},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)

            self.assertEqual(
                config.input["path"],
                str((root / "recording.mkv").resolve()),
            )
            self.assertEqual(len(config.sha256), 64)
            self.assertEqual(config, load_config(path))

    def test_rejects_unversioned_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"input": {"source": "video", "path": "x.avi"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "schema_version"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
