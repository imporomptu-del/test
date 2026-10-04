"""Safeguards against manufacturing ground truth or double-counting matches."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from score_phase20_accuracy import (
    assign_one_to_one,
    digest,
    read_verified_run,
    score_rows,
)
from prepare_phase20_accuracy_review import validate_plan


def packet():
    return {
        "windows": [
            {
                "id": "w",
                "clip_id": "0029",
                "first": 0,
                "last": 2,
                "crop_xywh": [0, 0, 100, 100],
            }
        ]
    }


def labels():
    return {
        "schema": "seaqr.bounded-visual-accuracy-labels.v1",
        "policy": {"extra_localization_tolerance_px": 2.0},
        "positive_windows": [
            {
                "window_id": "w",
                "event_id": "e",
                "polarity": "bright",
                "motion_confirmed": True,
                "visible_samples": [
                    {"frame_index": i, "xy": [20 + i, 30], "uncertainty_px": 2}
                    for i in range(3)
                ],
                "unknown_frames": [],
                "airborne_target_verified": False,
            }
        ],
        "negative_windows": [],
        "unresolved_windows": [],
    }


def rows():
    return [
        {
            "frame_index": i,
            "segment": 0,
            "coverage": {"warmup": False, "detection_ready": True},
            "candidates": [{"source_xy": [20 + i, 30], "polarity": "bright"}],
            "tracks": [
                {
                    "track_id": "bright:1",
                    "segment": 0,
                    "measured": True,
                    "qualified_moving": True,
                    "measurement_source_xy": [20 + i, 30],
                }
            ],
        }
        for i in range(3)
    ]


class AccuracyPilotTests(unittest.TestCase):
    def result(self, rr=None, ll=None):
        return score_rows(
            rows() if rr is None else rr,
            labels() if ll is None else ll,
            packet(),
            "0029",
            10,
        )

    def test_dense_same_id_is_window_only_not_airborne_accuracy(self):
        result = self.result()
        self.assertTrue(
            result["positive_windows"][0]["continuous_same_id_in_reviewed_window"]
        )
        self.assertEqual(result["verified_airborne_windows"], 0)
        self.assertIsNone(result["airborne_recall"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["positive_windows"][0]["onset_to_detection_seconds"])

    def test_prediction_is_a_miss_even_at_exact_position(self):
        rr = rows()
        rr[1]["tracks"][0]["measured"] = False
        result = self.result(rr)["positive_windows"][0]
        self.assertEqual(result["missed_visible_frames"], [1])
        self.assertEqual(result["candidate_hits"], 3)
        self.assertIsNone(result["continuous_same_id_in_reviewed_window"])

    def test_unqualified_measurement_not_a_track_hit(self):
        rr = rows()
        rr[1]["tracks"][0]["qualified_moving"] = False
        self.assertEqual(
            self.result(rr)["positive_windows"][0]["qualified_measured_hits"], 2
        )

    def test_unknown_frame_not_a_miss_or_negative(self):
        ll, rr = labels(), rows()
        ll["positive_windows"][0]["visible_samples"].pop(1)
        ll["positive_windows"][0]["unknown_frames"] = [1]
        rr[1]["tracks"] = []
        result = self.result(rr, ll)["positive_windows"][0]
        self.assertEqual(result["visible_samples"], 2)
        self.assertEqual(result["missed_visible_frames"], [])
        self.assertIsNone(result["continuous_same_id_in_reviewed_window"])

    def test_unavailable_frame_remains_in_denominator(self):
        rr = rows()
        rr[1]["tracks"] = []
        rr[1]["coverage"] = {"warmup": True, "detection_ready": False}
        result = self.result(rr)["positive_windows"][0]
        self.assertEqual(result["visible_samples"], 3)
        self.assertEqual(result["unavailable_visible_frames"], 1)

    def test_missing_or_duplicate_frames_rejected(self):
        for rr in (rows()[:2], rows() + rows()[:1]):
            with self.assertRaises(ValueError):
                self.result(rr)

    def test_nonfinite_or_outside_annotation_rejected(self):
        for xy in ([float("nan"), 30], [100, 30]):
            ll = labels()
            ll["positive_windows"][0]["visible_samples"][0]["xy"] = xy
            with self.assertRaises(ValueError):
                self.result(ll=ll)

    def test_duplicate_visibility_or_missing_disposition_rejected(self):
        for value in ([0], []):
            ll = labels()
            if value:
                ll["positive_windows"][0]["unknown_frames"] = value
            else:
                ll["positive_windows"][0]["visible_samples"].pop()
            with self.assertRaises(ValueError):
                self.result(ll=ll)

    def test_duplicate_tracks_do_not_prove_continuity(self):
        rr = rows()
        extra = dict(rr[1]["tracks"][0], track_id="bright:2")
        rr[1]["tracks"].append(extra)
        result = self.result(rr)["positive_windows"][0]
        self.assertEqual(result["qualified_measured_hits"], 3)
        self.assertEqual(result["ambiguity_frames"], 1)
        self.assertIsNone(result["continuous_same_id_in_reviewed_window"])

    def test_segment_change_breaks_identity(self):
        rr = rows()
        rr[2]["tracks"][0]["segment"] = 1
        result = self.result(rr)["positive_windows"][0]
        self.assertFalse(result["continuous_same_id_in_reviewed_window"])

    def test_one_observation_cannot_count_for_two_objects(self):
        samples = [
            {"xy": [20, 30], "polarity": "bright", "radius": 4} for _ in range(2)
        ]
        obs = [{"xy": [20, 30], "polarity": "bright"}]
        assignment, _ = assign_one_to_one(samples, obs)
        self.assertEqual(len(assignment), 1)

    def test_augmenting_assignment_avoids_greedy_avoidable_miss(self):
        samples = [
            {"xy": [1, 0], "polarity": "bright", "radius": 2},
            {"xy": [0, 0], "polarity": "bright", "radius": 0.5},
        ]
        obs = [
            {"xy": [0, 0], "polarity": "bright"},
            {"xy": [3, 0], "polarity": "bright"},
        ]
        self.assertEqual(len(assign_one_to_one(samples, obs)[0]), 2)

    def test_negatives_need_adjudication_and_stay_roi_scoped(self):
        ll = labels()
        ll["positive_windows"] = []
        ll["negative_windows"] = [{"window_id": "w", "status": "verified_no_target"}]
        with self.assertRaises(ValueError):
            self.result(ll=ll)
        ll["negative_windows"][0].update(
            every_frame_reviewed=True,
            native_pixels_reviewed=True,
            exhaustive_roi=True,
            target_definition_adjudicated=True,
        )
        result = self.result(ll=ll)
        self.assertEqual(
            result["verified_negative_windows"][0]["measured_qualified_track_frames"], 3
        )
        self.assertEqual(
            result["verified_negative_windows"][0]["exposure_roi_seconds"], 0.3
        )
        self.assertIsNone(result["false_alarms_per_full_frame_minute"])

    def test_unknown_label_does_not_count_false_alarms(self):
        ll = labels()
        ll["positive_windows"] = []
        ll["unresolved_windows"] = [{"window_id": "w"}]
        self.assertEqual(self.result(ll=ll)["verified_negative_windows"], [])

    def test_unapproved_media_rejected_before_media_access(self):
        plan = {
            "schema": "seaqr.source-only-review-plan.v1",
            "sources": [
                {
                    "clip_id": "9999",
                    "sha256": "bad",
                    "path": "/does/not/exist/chunk_9999.avi",
                }
            ],
            "windows": [],
        }
        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_plan(plan)

    def test_verified_run_checks_identity_backend_and_complete_journal(self):
        for mutation in (
            None,
            "source",
            "incomplete",
            "prefix",
            "annotations",
            "package",
            "backend",
            "missing_frame",
            "duplicate_frame",
            "fps",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "implementation").mkdir()
                module = root / "implementation" / "example.py"
                module.write_text("# immutable synthetic test fixture\n")
                package = {"example.py": digest(module)}
                freeze = {
                    "files_sha256": {
                        "tiny_target/example.py": digest(module),
                        "configs/evaluation/phase20_visible_v7.json": "cfg",
                    }
                }
                launch = {
                    "package_sha256": package,
                    "config_sha256": "cfg",
                    "expected_frames": 3,
                    "max_frames": None,
                    "source_sha256": "source",
                    "annotations_supplied_to_detector": False,
                    "configuration": {"motion_backend": "cpu_translation"},
                    "fps": 10,
                }
                report = {
                    "completed": True,
                    "frames": 3,
                    "full_clip": True,
                    "source_sha256": "source",
                }
                rr = rows()
                if mutation == "source":
                    report["source_sha256"] = "wrong"
                if mutation == "incomplete":
                    report["completed"] = False
                if mutation == "prefix":
                    report["full_clip"] = False
                if mutation == "annotations":
                    launch["annotations_supplied_to_detector"] = True
                if mutation == "package":
                    module.write_text("# changed\n")
                if mutation == "backend":
                    launch["configuration"]["motion_backend"] = "pva"
                if mutation == "missing_frame":
                    rr.pop()
                if mutation == "duplicate_frame":
                    rr[2]["frame_index"] = 1
                if mutation == "fps":
                    launch["fps"] = 30
                for name, value in (
                    ("freeze.json", freeze),
                    ("launch.json", launch),
                    ("report.json", report),
                ):
                    (root / name).write_text(json.dumps(value))
                (root / "frames.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rr)
                )
                spec = {
                    "run": str(root),
                    "freeze": str(root / "freeze.json"),
                    "backend": "cpu_translation",
                    "full_clip": True,
                }
                source = {"sha256": "source", "frames": 3, "fps": 10}

                def consume():
                    journal, _ = read_verified_run(spec, source)
                    return list(journal)

                if mutation is None:
                    self.assertEqual(len(consume()), 3)
                else:
                    with self.assertRaises(ValueError):
                        consume()


if __name__ == "__main__":
    unittest.main()
