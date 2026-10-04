import json
from pathlib import Path
import tempfile
import unittest

from tiny_target.visible_regression import score


class RegressionTests(unittest.TestCase):
    def fixture(self, root, measured=True):
        annotation = root / "annotations.json"
        annotation.write_text(
            json.dumps(
                dict(
                    schema="manual_visual_reference_v1",
                    source_sha256="abc",
                    events=[
                        dict(
                            event_id="target",
                            anchors=[
                                dict(
                                    frame_index=i,
                                    x=20 + i,
                                    y=30,
                                    position_uncertainty_px=2,
                                    visibility="visible",
                                )
                                for i in range(3)
                            ],
                        )
                    ],
                )
            )
        )
        (root / "launch.json").write_text(
            json.dumps(dict(source_sha256="abc", expected_frames=3))
        )
        report = dict(
            source_sha256="abc",
            completed=True,
            full_clip=True,
            frames=3,
            qualified_tracks=[dict(track_id="bright:0"), dict(track_id="bright:1")],
        )
        (root / "report.json").write_text(json.dumps(report))
        rows = []
        for i in range(3):
            xy = [20 + i, 30]
            rows.append(
                dict(
                    frame_index=i,
                    segment=0,
                    coverage={"warmup": False},
                    candidates=[dict(polarity="bright", source_xy=xy)],
                    tracks=[
                        dict(
                            track_id="bright:0",
                            measured=measured,
                            qualified_moving=True,
                            measurement_source_xy=xy if measured else None,
                            source_xy=xy,
                        )
                    ],
                )
            )
        (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return annotation, report

    def test_measured_track_matches_but_unmatched_is_not_false_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, _ = self.fixture(root)
            result = score(root, annotation)
            self.assertTrue(result["all_events_pass"])
            self.assertIsNone(result["false_tracks_per_minute"])
            self.assertEqual(result["unmatched_qualified_track_count"], 1)

    def test_coasting_predictions_cannot_pass_as_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, _ = self.fixture(root, measured=False)
            result = score(root, annotation)
            self.assertFalse(result["all_events_pass"])
            self.assertEqual(result["events"][0]["qualified_measured_anchor_hits"], 0)

    def test_wrong_source_and_partial_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, report = self.fixture(root)
            for change in (
                {"source_sha256": "wrong"},
                {"full_clip": False},
                {"frames": 2},
            ):
                (root / "report.json").write_text(json.dumps({**report, **change}))
                with self.assertRaises(ValueError):
                    score(root, annotation)

    def test_missing_journal_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, _ = self.fixture(root)
            journal = root / "frames.jsonl"
            journal.write_text("\n".join(journal.read_text().splitlines()[:2]) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                score(root, annotation)


if __name__ == "__main__":
    unittest.main()
