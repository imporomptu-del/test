"""Synthetic-only coverage tests: never open real media or detector outputs."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import prepare_accuracy_v40_coverage as runner


class FakeCapture:
    """Generated BGR values encode the index; only retrieved crops allocate."""
    def __init__(self, *, count=40, reported_count=40, bad_shape=False,
                 bad_dtype=False, fail_at=None, bad_position_at=None, fps=10):
        self.count, self.reported_count, self.fps = count, reported_count, fps
        self.bad_shape, self.bad_dtype = bad_shape, bad_dtype
        self.fail_at, self.bad_position_at = fail_at, bad_position_at
        self.position = 0
        self.grabbed, self.retrieved = [], []

    def isOpened(self):
        return True

    def getBackendName(self):
        return "synthetic-only"

    def get(self, prop):
        return {runner.cv2.CAP_PROP_FRAME_WIDTH: 300,
                runner.cv2.CAP_PROP_FRAME_HEIGHT: 220,
                runner.cv2.CAP_PROP_FRAME_COUNT: self.reported_count,
                runner.cv2.CAP_PROP_FPS: self.fps,
                runner.cv2.CAP_PROP_POS_FRAMES:
                    self.position + int(self.position == self.bad_position_at)}[prop]

    def grab(self):
        frame = self.position
        self.grabbed.append(frame)
        if frame >= self.count or frame == self.fail_at:
            return False
        self.position += 1
        return True

    def retrieve(self):
        frame = self.position - 1
        self.retrieved.append(frame)
        shape = (2, 3, 3) if self.bad_shape else (220, 300, 3)
        dtype = np.float32 if self.bad_dtype else np.uint8
        # Different channels detect accidental RGB conversion/gray transforms.
        pixel = np.asarray([frame, frame + 40, frame + 80], dtype=dtype)
        return True, np.broadcast_to(pixel, shape)


def fixture_window():
    return dict(window_id="synthetic", clip="0029", temporal_stratum="early",
                spatial_row=0, spatial_column=0, frame_start=5,
                frame_end_inclusive=24, frame_count=20, crop_xywh=[3, 4, 256, 192])


def fixture_claim():
    return dict(width=300, height=220, declared_frame_count=40, nominal_container_fps=10)


class AccuracyV40CoverageTests(unittest.TestCase):
    def setUp(self):
        # Preregistration metadata is allowed; no source bytes are accessed.
        self.manifest = runner.load_manifest()

    def test_frozen_manifest_contains_exact_108_windows_and_2160_frames(self):
        windows = runner.validate_manifest(self.manifest)
        self.assertEqual(len(windows), 108)
        self.assertEqual(sum(w["frame_count"] for w in windows), 2160)
        for clip in runner.CLIPS:
            self.assertEqual(sum(w["clip"] == clip for w in windows), 27)

    def test_invalid_manifest_or_missing_duplicate_reordered_window_rejected(self):
        mutations = [lambda m: m.update(schema="wrong"),
                     lambda m: m["allowlisted_clips"].append("9999"),
                     lambda m: m["windows"].pop(),
                     lambda m: m["windows"].__setitem__(1, copy.deepcopy(m["windows"][0])),
                     lambda m: m["windows"].reverse(),
                     lambda m: m["windows"][0].update(labels=[]),
                     lambda m: m["source_claims"]["0029"].update(width=4783)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(self.manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    runner.validate_manifest(changed)

    def test_invalid_frame_or_crop_range_and_noninteger_rejected(self):
        for values in (dict(frame_start=-1), dict(frame_end_inclusive=40),
                       dict(frame_end_inclusive=23), dict(frame_count=19),
                       dict(frame_start=True), dict(crop_xywh=[45, 4, 256, 192]),
                       dict(crop_xywh=[3, 29, 256, 192]), dict(crop_xywh=[3, 4, 255, 192]),
                       dict(crop_xywh=[3., 4, 256, 192])):
            with self.subTest(values=values):
                window = fixture_window()
                window.update(values)
                with self.assertRaises(ValueError):
                    runner.extract_clip(FakeCapture(), [window], fixture_claim())

    def test_exact_20_frame_slicing_preserves_bgr_and_only_retrieves_interval(self):
        capture = FakeCapture()
        window = fixture_window()
        arrays, metadata = runner.extract_clip(capture, [window], fixture_claim())
        frames = arrays["synthetic"]
        self.assertEqual(frames.shape, (20, 192, 256, 3))
        self.assertEqual(frames.dtype, np.uint8)
        for index in range(20):
            np.testing.assert_array_equal(frames[index, 0, 0], [index + 5, index + 45, index + 85])
            np.testing.assert_array_equal(frames[index, -1, -1], frames[index, 0, 0])
        self.assertEqual(capture.retrieved, list(range(5, 25)))
        self.assertEqual(capture.grabbed, list(range(41)))
        self.assertEqual(metadata["actual_sequential_frame_count"], 40)
        self.assertTrue(metadata["eof_verified"])
        self.assertEqual(metadata["retained_full_frames"], 0)

    def test_decoder_metadata_truncation_extra_frame_index_shape_and_dtype_fail(self):
        for options in (dict(reported_count=41), dict(fps=11), dict(count=39),
                        dict(count=41), dict(fail_at=7), dict(bad_position_at=7),
                        dict(bad_shape=True), dict(bad_dtype=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.extract_clip(FakeCapture(**options), [fixture_window()], fixture_claim())

    def test_contact_sheet_has_all_20_native_crops_in_five_by_four_with_no_pixel_edits(self):
        frames, _ = runner.extract_clip(FakeCapture(), [fixture_window()], fixture_claim())
        native = frames["synthetic"]
        sheet = runner.contact_sheet(native, fixture_window())
        self.assertEqual(sheet.shape, (956, 1312, 3))
        self.assertEqual(len(runner.sheet_origins()), 20)
        for index, (x, y) in enumerate(runner.sheet_origins()):
            np.testing.assert_array_equal(sheet[y:y + 192, x:x + 256], native[index])
        with self.assertRaises(ValueError):
            runner.contact_sheet(native[:19], fixture_window())

    def test_archive_and_lossless_sheet_hashes_bind_every_window_and_frame(self):
        frames, _ = runner.extract_clip(FakeCapture(), [fixture_window()], fixture_claim())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "native").mkdir()
            (output / "sheets").mkdir()
            record = runner.export_window(output, fixture_window(), frames["synthetic"])
            archive = output / record["native_archive"]["path"]
            self.assertEqual(runner.sha256_file(archive), record["native_archive"]["sha256"])
            with np.load(archive, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["native_bgr"], frames["synthetic"])
                np.testing.assert_array_equal(data["frame_indices"], np.arange(5, 25))
            image = runner.cv2.imread(str(output / record["contact_sheet"]["path"]))
            np.testing.assert_array_equal(image, runner.contact_sheet(frames["synthetic"], fixture_window()))
            self.assertEqual(len(record["native_array"]["per_frame_sha256"]), 20)
            self.assertEqual(record["native_array"]["sha256"], runner.array_sha256(frames["synthetic"]))
            with self.assertRaises(FileExistsError):
                runner.export_window(output, fixture_window(), frames["synthetic"])

    def test_existing_output_rejected_before_manifest_or_source_read(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "load_manifest") as load:
            with self.assertRaises(FileExistsError):
                runner.run(Path(tmp))
            load.assert_not_called()

    def test_hash_mismatch_and_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.verify_hash(path, "0" * 64)
            self.assertEqual(runner.verify_hash(path, hashlib.sha256(b"{}").hexdigest()),
                             hashlib.sha256(b"{}").hexdigest())
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                runner.load_manifest(path)

    def test_source_hash_mismatch_aborts_before_output_creation_and_decode(self):
        original = runner.verify_hash
        def verify(path, expected):
            if Path(path) in runner.SOURCES.values():
                raise ValueError("SHA-256 mismatch: synthetic source")
            return original(path, expected)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "verify_hash", side_effect=verify), \
                mock.patch.object(runner.cv2, "VideoCapture") as capture:
            output = Path(tmp) / "fresh"
            with self.assertRaisesRegex(ValueError, "synthetic source"):
                runner.run(output)
            self.assertFalse(output.exists())
            capture.assert_not_called()

    def test_freeze_binds_code_tests_manifest_and_plan_before_decoder_opens(self):
        original = runner.verify_hash
        def synthetic_source_hash(path, expected):
            # No source is opened; these are fixture provenance claims only.
            return expected if Path(path) in runner.SOURCES.values() else original(path, expected)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "synthetic_packet"
            def decoder_assertion(path):
                freeze = json.loads((output / "freeze.json").read_text())
                self.assertEqual(freeze["native_crop_frames"], 2160)
                self.assertEqual(len(freeze["windows"]), 108)
                for bound in (Path(runner.__file__).resolve(), runner.TEST, runner.MANIFEST, runner.PLAN):
                    self.assertEqual(freeze["inputs_sha256"][str(bound)], runner.sha256_file(bound))
                    self.assertEqual(runner.sha256_file(output / "implementation" / bound.name),
                                     freeze["inputs_sha256"][str(bound)])
                self.assertTrue(freeze["frozen_at_utc"])
                raise RuntimeError("Synthetic stop before any decoder opens")
            with mock.patch.object(runner, "verify_hash", side_effect=synthetic_source_hash), \
                    mock.patch.object(runner.cv2, "VideoCapture", side_effect=decoder_assertion):
                with self.assertRaisesRegex(RuntimeError, "Synthetic stop"):
                    runner.run(output)
            self.assertFalse((output / "completion.json").exists())
            self.assertFalse(json.loads((output / "failure.json").read_text())["completed"])


if __name__ == "__main__":
    unittest.main()
