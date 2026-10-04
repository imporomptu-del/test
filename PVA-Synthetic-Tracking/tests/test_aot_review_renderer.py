"""Generated-only checks for a review renderer; not detector accuracy evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import render_aot_pilot_review as review


def manifest_fixture():
    flight, digest = "a" * 32, "b" * 64
    rows, images = [], []
    for index in range(300):
        timestamp = 1600000000000000000 + index * 100000000
        name = f"{timestamp}{flight}.png"
        entity = {"flight_id": flight, "img_name": name, "time": timestamp,
                  "blob": {"frame": index + 1}, "id": "aircraft1", "bb": [40, 60, 8, 6]}
        row = {"source_frame": index + 1, "timestamp_ns": str(timestamp), "img_name": name,
               "entities": [entity], "airborne_label_count": 1}
        rows.append(row)
        images.append({key: row[key] for key in ("source_frame", "timestamp_ns", "img_name")})
        images[-1]["pixel_sha256"] = "c" * 64
    return ({"part": "part1", "flight_id": flight, "frames": rows},
            {"manifest_sha256": digest, "frames": 300, "resolution": [2448, 2048],
             "dtype": "uint8", "grayscale": True, "detector_run": False,
             "source_frame_range": [1, 300], "images": images}, digest)


def journal_fixture():
    tracks = [
        {"track_id": "dark:1", "segment": 0, "measured": True, "qualified_moving": True,
         "measurement_source_xy": [200.5, 100.5], "source_xy": [500.0, 700.0]},
        {"track_id": "dark:2", "segment": 0, "measured": False, "qualified_moving": True,
         "measurement_source_xy": None, "source_xy": [400.0, 600.0]},
        {"track_id": "dark:3", "segment": 0, "measured": True, "qualified_moving": False,
         "measurement_source_xy": [600.0, 800.0], "source_xy": [700.0, 900.0]},
    ]
    return [{"frame_index": index, "timestamp_ns": index * 100000000,
             "segment": 0, "tracks": copy.deepcopy(tracks)} for index in range(300)]


class ManifestValidationTests(unittest.TestCase):
    def test_exact_300_manifest_with_bound_validation(self):
        review.validate_manifest(*manifest_fixture())

    def test_reject_mismatched_pixel_ledger(self):
        manifest, validation, digest = manifest_fixture()
        validation["images"][15]["source_frame"] = 100
        with self.assertRaisesRegex(ValueError, "inventory"):
            review.validate_manifest(manifest, validation, digest)

    def test_reject_detached_annotation(self):
        manifest, validation, digest = manifest_fixture()
        manifest["frames"][30]["entities"][0]["time"] += 1
        with self.assertRaisesRegex(ValueError, "source frame"):
            review.validate_manifest(manifest, validation, digest)

    def test_reject_box_boolean_or_unbounded(self):
        for box in ([True, 3, 4, 5], [3, 4, float("inf"), 5], [3, 4, 0, 5], [5000, 5000, 5, 5]):
            manifest, validation, digest = manifest_fixture()
            manifest["frames"][0]["entities"][0]["bb"] = box
            with self.subTest(box=box), self.assertRaises(ValueError):
                review.validate_manifest(manifest, validation, digest)


class JournalValidationTests(unittest.TestCase):
    def parse(self, rows):
        with tempfile.TemporaryDirectory(prefix="seaqr_review_metadata_") as directory:
            path = Path(directory) / "journal.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            return list(review.journal_rows(path))

    def test_only_actual_qualified_measurements_are_selected(self):
        result = self.parse(journal_fixture())
        self.assertEqual(len(result), 300)
        self.assertEqual(result[0], (0, [(200.5, 100.5)], 1, 1))
        self.assertEqual(result[-1][0], 299)

    def test_reject_short_or_long_or_reordered_journal(self):
        fixtures = [journal_fixture()[:-1], journal_fixture() + [journal_fixture()[-1]]]
        reordered = journal_fixture()
        reordered[1], reordered[2] = reordered[2], reordered[1]
        fixtures.append(reordered)
        for rows in fixtures:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                self.parse(rows)

    def test_reject_wrong_timestamp_and_boolean_frame(self):
        for key, value in (("timestamp_ns", 1), ("frame_index", False)):
            rows = journal_fixture()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(rows)

    def test_reject_prediction_masquerading_as_measurement(self):
        rows = journal_fixture()
        rows[0]["tracks"][1]["measurement_source_xy"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "Prediction/coast"):
            self.parse(rows)

    def test_reject_boolean_qualification_and_nonfinite_measurement(self):
        for key, value in (("qualified_moving", 1), ("measurement_source_xy", [float("nan"), 20]),
                           ("measurement_source_xy", [True, 20])):
            rows = journal_fixture()
            rows[0]["tracks"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(rows)

    def test_reject_duplicate_json_keys_and_nonfinite_constants(self):
        for raw in ('{"measured":true,"measured":false}', '{"x":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                review.decode(raw)

    def test_finite_offscreen_measurements_preserved_for_diagnostic_counts(self):
        rows = journal_fixture()
        rows[0]["tracks"][0]["measurement_source_xy"] = [-1, 20]
        rows[0]["tracks"][2]["measurement_source_xy"] = [review.WIDTH + 5, 20]
        self.assertEqual(self.parse(rows)[0], (0, [(-1, 20)], 1, 1))


class RenderPanelTests(unittest.TestCase):
    def test_source_is_untouched_except_global_half_resize_and_captions_are_external(self):
        x = np.arange(review.WIDTH, dtype=np.uint16)[None, :]
        gray = np.broadcast_to(x % 256, (review.HEIGHT, review.WIDTH)).astype(np.uint8)
        expected = cv2.cvtColor(cv2.resize(gray, (1224, 1024), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        source = gray.copy()
        row = {"entities": [{"bb": [40, 60, 8, 6]}]}
        canvas = review.render_frame(gray, row, [(200, 100)], 0)
        self.assertEqual(canvas.shape, (1104, 2448, 3))
        np.testing.assert_array_equal(gray, source)
        np.testing.assert_array_equal(canvas[:1024, :1224], expected)
        self.assertTrue(np.any(np.all(canvas[:1024, 1224:] == review.GT_COLOR, axis=2)))
        self.assertTrue(np.any(np.all(canvas[:1024, 1224:] == review.MEASUREMENT_COLOR, axis=2)))
        self.assertTrue(np.any(canvas[1024:]))

    def test_empty_annotations_and_observations_leave_both_panels_identical(self):
        gray = np.full((review.HEIGHT, review.WIDTH), 81, dtype=np.uint8)
        canvas = review.render_frame(gray, {"entities": [{}]}, [], 299)
        np.testing.assert_array_equal(canvas[:1024, :1224], canvas[:1024, 1224:])

    def test_offscreen_measurements_never_create_false_edge_markers(self):
        gray = np.full((review.HEIGHT, review.WIDTH), 81, dtype=np.uint8)
        points = [(-1, 20), (20, -1), (review.WIDTH, 20), (20, review.HEIGHT)]
        canvas = review.render_frame(gray, {"entities": []}, points, 0)
        np.testing.assert_array_equal(canvas[:1024, :1224], canvas[:1024, 1224:])

    def test_all_captions_fit_their_external_bar_regions(self):
        captions = [
            ("SOURCE | full field, unmarked, 0.5x", .55, 1224 - 24),
            ("ANNOTATIONS + MEASURED OUTPUT | frame 299/299 | t=29.9s", .55, 1224 - 24),
            ("Yellow GT = dataset annotation, not a detector result", .50, 1224 - 24),
            ("Cyan = qualified actual measurement; enlarged marker, not object extent", .50, 1224 - 24),
            ("DISPLAY ONLY - NOT DETECTOR INPUT | 0.5x downscaling can hide few-pixel targets | no predictions/coasts | no true/false-object labels", .50, 2448 - 24),
        ]
        for text, scale, available in captions:
            with self.subTest(text=text):
                (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                self.assertLessEqual(width, available)
                self.assertLessEqual(height + baseline, 24)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires ffmpeg and ffprobe")
class GeneratedEncodingTests(unittest.TestCase):
    def test_generated_stream_encodes_verifies_and_refuses_overwrite(self):
        # Only this synthetic test changes constants; production CLI has no pin/extent bypass.
        with tempfile.TemporaryDirectory(prefix="seaqr_review_encoding_") as directory:
            base = Path(directory)
            video, output, receipt = (base / name for name in ("generated.avi", "review.mp4", "receipt.json"))
            pixels = [np.full((48, 64), value, np.uint8).tobytes() for value in (31, 67, 129)]
            subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-f", "rawvideo",
                            "-pixel_format", "gray", "-video_size", "64x48", "-framerate", "10", "-i", "pipe:0",
                            "-an", "-c:v", "ffv1", "-pix_fmt", "gray", "-threads", "1", str(video)],
                           input=b"".join(pixels), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
            manifest, validation, _ = manifest_fixture()
            manifest["frames"] = manifest["frames"][:3]
            for row in manifest["frames"]:
                row["entities"][0]["bb"] = [4, 4, 4, 4]
            manifest_path, validation_path, journal_path = (base / name for name in ("manifest.json", "validation.json", "journal.jsonl"))
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            validation.update(manifest_sha256=review.sha256(manifest_path), frames=3, resolution=[64, 48], source_frame_range=[1, 3])
            validation["images"] = validation["images"][:3]
            for image, original in zip(validation["images"], pixels):
                image["pixel_sha256"] = hashlib.sha256(original).hexdigest()
            with validation_path.open("x", encoding="utf-8") as stream:
                json.dump(validation, stream)
            rows = journal_fixture()[:3]
            for row in rows:
                row["tracks"][0]["measurement_source_xy"] = [20, 10]
            with journal_path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            with patch.multiple(review, WIDTH=64, HEIGHT=48, FRAME_COUNT=3,
                                EXPECTED_VIDEO_SHA256=review.sha256(video),
                                EXPECTED_MANIFEST_SHA256=review.sha256(manifest_path)):
                result = review.render(video, manifest_path, journal_path, validation_path, output, receipt)
                self.assertTrue(result["passed"])
                self.assertEqual(result["counts"]["frames"], 3)
                self.assertEqual(result["counts"]["rendered_qualified_measurements"], 3)
                self.assertEqual(result["counts"]["excluded_prediction_or_coast_states"], 3)
                self.assertEqual(result["pixel_hashes_verified"], 3)
                self.assertEqual(result["output_sha256"], review.sha256(output))
                self.assertEqual(len(result["output_probe"]["frames"]), 3)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    review.render(video, manifest_path, journal_path, validation_path, output, receipt)


if __name__ == "__main__":
    unittest.main()
