"""Evaluation reporting must distinguish execution from usable detection coverage."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "transfer_summary", SCRIPTS / "summarize_phase20_frozen_transfer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenTransferTests(unittest.TestCase):
    def make_run(self, path, ready=False):
        freeze = dict(
            sources=[dict(clip_id="0027", sha256="source")],
            implementation_sha256={
                "configs/evaluation/phase20_visible_v7_pva.json": "config",
                "configs/tiny_target_phase12_cfar_test.yaml": "motion",
            },
        )
        launch = dict(
            source_sha256="source",
            config_sha256="config",
            motion_config_sha256="motion",
            code_sha256={},
            expected_frames=4,
        )
        report = dict(
            source_sha256="source",
            completed=True,
            full_clip=True,
            frames=4,
            qualified_track_count=0,
            counts={},
            processed_fps=1.0,
        )
        rows = [
            dict(
                frame_index=i,
                coverage=dict(warmup=not ready),
                candidates=[],
                motion=dict(reset=i > 0, pva_failure=False, accepted=False),
            )
            for i in range(4)
        ]
        for name, obj in (("launch.json", launch), ("report.json", report)):
            (path / name).write_text(json.dumps(obj))
        (path / "frames.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        return freeze

    def test_completed_pva_execution_without_ready_frames_is_unusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = module.summarize(path, "0027", "pva", self.make_run(path))
            self.assertEqual(result["verdict"], "unusable_no_detection_ready_frames")
            self.assertTrue(result["completed_full_clip"])
            self.assertFalse(result["usable_detection_coverage"])
            self.assertEqual(result["counts"]["pva_runtime_errors"], 0)
            self.assertIsNone(result["recall"])
            self.assertIsNone(result["false_positives_per_minute"])

    def test_no_proposals_with_ready_frames_does_not_prove_empty_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = module.summarize(
                path, "0027", "pva", self.make_run(path, ready=True)
            )
            self.assertEqual(result["verdict"], "unlabeled_review_required")
            self.assertIsNone(result["precision"])

    def test_changed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            freeze = self.make_run(path)
            launch = json.loads((path / "launch.json").read_text())
            launch["config_sha256"] = "retuned"
            (path / "launch.json").write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, "Configuration"):
                module.summarize(path, "0027", "pva", freeze)

    def test_partial_journal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            freeze = self.make_run(path)
            (path / "frames.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "Frame count"):
                module.summarize(path, "0027", "pva", freeze)


if __name__ == "__main__":
    unittest.main()
