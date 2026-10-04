"""Media/GPU-free native-export mocks for passive nuisance origin capture."""
from __future__ import annotations

import base64
import copy
import ctypes as C
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
PATH = HERE / "nuisance_origin_capture_v1.py"
if not PATH.is_file():
    PATH = Path(__file__).resolve().parents[2] / "scripts/nuisance_origin_capture_v1.py"
SPEC = importlib.util.spec_from_file_location("nuisance_origin_capture_tested", PATH)
capture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_module)
PEAK_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("score", "<f4"),
                       ("response", "<f4"), ("noise", "<f4")])


def plan(comparators=True):
    frames = {}
    episode = 0
    for group, count in zip(capture_module.FRAME_GROUPS, (2, 2, 2, 1, 1)):
        for frame in group:
            frames[str(frame)] = []
            for offset in range(count):
                name = f"episode-{episode + offset}"
                roles = ("sample", "comparator") if comparators else ("sample",)
                for role in roles:
                    xy = [16, 16] if role == "sample" else [32, 32]
                    frames[str(frame)].append(dict(point_id=name + "/" + role, episode_id=name,
                        role=role, reference_xy=xy, source_xy=[float(v) for v in xy]))
        episode += count
    return {"frame_points": frames}


def unpack(item):
    raw = base64.b64decode(item["data_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise AssertionError("Descriptor hash differs")
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"])


def emit(pointer, value):
    value = np.ascontiguousarray(value)
    C.memmove(pointer, value.ctypes.data, value.nbytes)


class Export:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []
        self.argtypes = ["original-argument-types"]
        self.restype = "original-return-type"

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class NativeLibrary:
    _name = "frozen-test-library"

    def __init__(self, shape):
        self.shape = shape
        self.events = []
        self.bad_score = False
        self.debug_error = False
        self.patch_error = False
        self.finish_error = False
        self.finish_changes_stats = False
        self.consolidated = False
        self.seaqr_resident_debug = Export(self.debug)
        self.seaqr_front_v26_debug = Export(self.front_debug)
        self.seaqr_resident_patches = Export(self.patches)
        self.seaqr_front_v26_finish = Export(self.finish)

    def prepare(self):
        self.background = np.full(self.shape, .5, np.float32)
        self.variance = np.full(self.shape, .25, np.float32)
        self.spatial = np.full(self.shape, 3, np.float32)
        self.support = np.ones(self.shape, np.bool_)
        self.mask = self.support.copy()
        self.stats = np.tile(np.array([.5, .5], np.float32), (4, 1))
        self.sigmas = np.full(4, .5, np.float64)
        self.consolidated = False

    def debug(self, handle, background, variance):
        self.events.append("state-read")
        if self.debug_error:
            return 7
        if handle != 202:
            raise AssertionError("Wrong core handle")
        emit(background, self.background)
        emit(variance, self.variance)
        return 0

    def front_debug(self, front, support, mask, stats, sigmas):
        self.events.append("front-read")
        if front != 101:
            raise AssertionError("Wrong front handle")
        for pointer, value in ((support, self.support), (mask, self.mask),
                               (stats, self.stats), (sigmas, self.sigmas)):
            emit(pointer, value)
        return 0

    def patches(self, handle, seeds, count, output):
        self.events.append("scratch-patches")
        if not self.consolidated:
            raise AssertionError("Additional gather must follow original consolidation")
        if self.patch_error:
            return 3
        xy = np.frombuffer(C.string_at(seeds, count * 2 * 4), np.int32).reshape(count, 2)
        values = np.stack([self.spatial[y-8:y+9, x-8:x+9] for x, y in xy])
        emit(output, values)
        return 0

    def finish(self, front, points, count, offsets, offset_count, alpha, noise_alpha, clip2, floor2, variance_only):
        self.events.append("original-finish")
        if self.finish_error:
            return 11
        self.mask = self.support.copy()
        self.mask[16, 16] = False
        temporal = self.spatial - self.background
        learn_background = self.support if variance_only else self.mask
        self.background[learn_background] += np.float32(alpha) * temporal[learn_background]
        observed = np.minimum(temporal * temporal, self.variance * np.float32(clip2))
        self.variance[self.mask] += np.float32(noise_alpha) * (observed[self.mask] - self.variance[self.mask])
        np.maximum(self.variance, np.float32(floor2), out=self.variance)
        if self.finish_changes_stats:
            self.stats[0, 0] = 8
        return 0


class DeviceImage:
    def __init__(self, valid):
        self.shape = valid.shape
        self.valid = valid
        self.consumed = False
        self.downloads = 0

    def validate(self, valid):
        if valid is not self.valid or self.consumed:
            raise ValueError("Original warp validation")

    def download_for_verification(self):
        self.validate(self.valid)
        self.downloads += 1
        return np.full(self.shape, 20, np.float32), np.full(self.shape, 23, np.float32)


class Front:
    def __init__(self):
        self.config = SimpleNamespace(tile_size=32)
        self.shape = (48, 64)
        self.lib = NativeLibrary(self.shape)
        self.front, self.handle = 101, 202
        self.sigmas = np.empty(4, np.float64)
        self.busy = False
        self.calls = self.device_calls = self.host_calls = self.finish_calls = self.learning_points = 0
        self.double_finish = False
        self.skip_finish = False
        self.mutate_valid = False
        self.mutate_learning = False
        self.source_args = None
        self.result = None

    def update(self, image, valid, segment, learning_centers=()):
        image.validate(valid)
        if self.busy:
            raise RuntimeError("Original ownership guard")
        self.busy = True
        native = getattr(self.lib, "original", self.lib)
        try:
            native.prepare()
            self.eligible = native.mask.copy()
            self.peaks = np.zeros((8, 1), PEAK_DTYPE)
            self.peaks["x"] = -1
            self.peaks[0, 0] = (16, 16, 4.1 if native.bad_score else 4, 2.5, .5)
            proposals = [dict(x=17, y=16, score=4., response_dn=2.5, noise_sigma_dn=.5,
                polarity="bright", shape=dict(peak_reference_xy=[16, 16],
                    centroid_reference_xy=[16.6, 16.], support_reference_xy=[[16, 16], [17, 16]]))]
            native.consolidated = True
            image.consumed = True
            points = np.empty((0, 2), np.int32)
            offsets = np.zeros((1, 2), np.int32)
            self.source_args = (self.front, points.ctypes.data, 0, offsets.ctypes.data, 1, .1, .2, 16., .25, 1)
            if not self.skip_finish:
                status = self.lib.seaqr_front_v26_finish(*self.source_args)
                if status:
                    raise RuntimeError(f"Original native finish error {status}")
                if self.double_finish:
                    self.lib.seaqr_front_v26_finish(*self.source_args)
            self.finish_calls += 1
            self.calls += 1
            self.device_calls += 1
            if self.mutate_valid:
                valid[0, 0] = False
            if self.mutate_learning:
                learning_centers.append({"unexpected": True})
            self.result = (proposals, dict(searchable_pixels=int(valid.sum()), detection_ms=9.5))
            return self.result
        finally:
            self.busy = False


class OriginCaptureTests(unittest.TestCase):
    def make(self, selected=False, comparators=True):
        capture = capture_module.OriginCapture(plan(comparators))
        front = capture.front_class(Front)()
        if selected:
            # Run ordinary earlier frames to retain real lifecycle validation.
            for unused in range(58):
                valid = np.ones(front.shape, np.bool_)
                front.update(DeviceImage(valid), valid, 0)
        return capture, front

    def call(self, front, learning=()):
        valid = np.ones(front.shape, np.bool_)
        image = DeviceImage(valid)
        result = front.update(image, valid, 0, learning)
        return result, image, valid

    def test_full_capture_original_results_and_exact_snapshots(self):
        capture, front = self.make()
        native = front.lib.original
        for frame in range(673):
            result, image, valid = self.call(front)
            self.assertIs(result, front.result)
            self.assertEqual(image.downloads, int(frame in capture_module.CAPTURE_FRAMES))
            self.assertEqual(native.seaqr_front_v26_finish.calls[-1], front.source_args)
            self.assertFalse(front.busy)
        record = capture.finish()
        self.assertEqual(record["processed_frames"], 673)
        self.assertEqual(record["point_snapshots"], 80)
        self.assertEqual(len(native.seaqr_front_v26_finish.calls), 673)
        self.assertEqual(len(native.seaqr_resident_patches.calls), 25)
        self.assertEqual(len(native.seaqr_resident_debug.calls), 50)
        row = record["records"][0]
        sample, comparator = row["points"][:2]
        self.assertEqual(row["timestamp_ns"], 5800000000)
        self.assertEqual(unpack(sample["warped_image"])[8, 8], 20)
        self.assertEqual(unpack(sample["spatial"])[8, 8], 3)
        self.assertEqual(unpack(sample["temporal_pre_learning"])[8, 8], 2.5)
        self.assertTrue(unpack(sample["eligible_pre_finish"])[8, 8])
        self.assertFalse(unpack(sample["learning_mask_post_finish"])[8, 8])
        self.assertTrue(unpack(comparator["learning_mask_post_finish"])[8, 8])
        self.assertEqual(unpack(sample["variance_pre_finish"])[8, 8], .25)
        self.assertEqual(unpack(sample["variance_post_finish"])[8, 8], .25)
        self.assertEqual(unpack(comparator["variance_post_finish"])[8, 8], 1)
        self.assertEqual(sample["center_values"]["exact_peak_reconstruction_count"], 1)
        self.assertEqual(comparator["center_values"]["exact_peak_reconstruction_count"], 0)
        self.assertEqual(sample["consolidated_candidates_at_original_peak"][0]["x"], 17)
        self.assertFalse(record["busy_flag_overridden"])
        self.assertFalse(record["full_journal_parity_verified_by_this_module"])

    def test_optional_comparators_have_exact_inventory(self):
        capture, front = self.make(comparators=False)
        for unused in range(673):
            self.call(front)
        self.assertEqual(capture.finish()["point_snapshots"], 40)

    def test_plan_is_detached_from_caller_mutations(self):
        value = plan()
        capture = capture_module.OriginCapture(value)
        value["frame_points"]["58"][0]["reference_xy"][0] = 1
        self.assertEqual(capture.frame_points[58][0]["reference_xy"], [16, 16])

    def test_attribute_plan_supported(self):
        capture = capture_module.OriginCapture(SimpleNamespace(**plan()))
        self.assertEqual(capture.expected_points, 80)

    def test_bad_plan_frames_roles_duplicates_and_types_rejected(self):
        edits = (
            lambda p: p["frame_points"].pop("58"),
            lambda p: p["frame_points"]["58"][0].update(role="negative"),
            lambda p: p["frame_points"]["58"][0].update(reference_xy=[True, 16]),
            lambda p: p["frame_points"]["58"][0].update(source_xy=[float("nan"), 16.]),
            lambda p: p["frame_points"]["58"].append(copy.deepcopy(p["frame_points"]["58"][0])),
            lambda p: p["frame_points"]["58"].pop(0),
            lambda p: p["frame_points"]["58"][0].update(extra=3),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                value = plan()
                edit(value)
                with self.assertRaises(ValueError):
                    capture_module.OriginCapture(value)

    def test_descriptor_preserves_float_bits_and_owns_bytes(self):
        array = np.array([-0., np.nextafter(np.float32(1), np.float32(2))], np.float32)
        original = array.tobytes()
        record = capture_module.descriptor(array)
        array[:] = 9
        self.assertEqual(unpack(record).tobytes(), original)
        for bad in (np.array([np.inf]), np.array([object()]), np.zeros(32769)):
            with self.assertRaises(ValueError):
                capture_module.descriptor(bad)

    def test_raw_finish_proxy_keeps_callable_metadata(self):
        capture, front = self.make(selected=True)
        original = front.lib.original.seaqr_front_v26_finish
        proxy = front.lib.seaqr_front_v26_finish
        self.assertIs(proxy.argtypes, original.argtypes)
        self.assertEqual(proxy.restype, original.restype)
        self.call(front)
        self.assertEqual(original.calls[-1], front.source_args)
        self.assertEqual(original.argtypes, ["original-argument-types"])

    def test_wrong_peak_score_fails_before_native_finish(self):
        capture, front = self.make(selected=True)
        native = front.lib.original
        native.bad_score = True
        before = len(native.seaqr_front_v26_finish.calls)
        with self.assertRaisesRegex(ValueError, "reconstruct exactly"):
            self.call(front)
        self.assertEqual(len(native.seaqr_front_v26_finish.calls), before)
        with self.assertRaisesRegex(ValueError, "failed"):
            capture.finish()

    def test_native_error_is_original_error_no_retry(self):
        capture, front = self.make(selected=True)
        native = front.lib.original
        native.finish_error = True
        before = len(native.seaqr_front_v26_finish.calls)
        with self.assertRaisesRegex(RuntimeError, "Original native finish error 11"):
            self.call(front)
        self.assertEqual(len(native.seaqr_front_v26_finish.calls), before + 1)
        self.assertFalse(front.busy)

    def test_debug_or_patch_error_prevents_interpretation(self):
        for flag in ("debug_error", "patch_error"):
            capture, front = self.make(selected=True)
            setattr(front.lib.original, flag, True)
            with self.assertRaises(ValueError):
                self.call(front)
            with self.assertRaises(ValueError):
                capture.finish()

    def test_original_finish_called_only_once_even_on_duplicate(self):
        capture, front = self.make(selected=True)
        front.double_finish = True
        before = len(front.lib.original.seaqr_front_v26_finish.calls)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.call(front)
        self.assertEqual(len(front.lib.original.seaqr_front_v26_finish.calls), before + 1)

    def test_skipped_finish_rejected(self):
        capture, front = self.make()
        front.skip_finish = True
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.call(front)

    def test_support_or_statistics_change_not_ignored(self):
        capture, front = self.make(selected=True)
        front.lib.original.finish_changes_stats = True
        with self.assertRaisesRegex(ValueError, "tile statistics"):
            self.call(front)

    def test_valid_and_learning_inputs_must_remain_unchanged(self):
        for field in ("mutate_valid", "mutate_learning"):
            capture, front = self.make(selected=True)
            setattr(front, field, True)
            with self.assertRaisesRegex(ValueError, "input was mutated"):
                self.call(front, [])

    def test_edge_support_is_unavailable_not_padded(self):
        value = plan()
        value["frame_points"]["58"][0]["reference_xy"] = [7, 16]
        capture = capture_module.OriginCapture(value)
        front = capture.front_class(Front)()
        for unused in range(58):
            self.call(front)
        with self.assertRaisesRegex(ValueError, "no clipping/padding"):
            self.call(front)

    def test_no_destructive_noise_probe_exposed(self):
        capture, front = self.make()
        with self.assertRaisesRegex(ValueError, "destructive noise probe"):
            getattr(front.lib, "seaqr_front_v26_noise_probe")

    def test_second_instance_or_class_binding_rejected(self):
        capture = capture_module.OriginCapture(plan())
        klass = capture.front_class(Front)
        klass()
        with self.assertRaisesRegex(ValueError, "one resident"):
            klass()
        with self.assertRaisesRegex(ValueError, "one front-class"):
            capture.front_class(Front)

    def test_finish_partial_or_repeated_and_extra_frame_rejected(self):
        capture, front = self.make()
        with self.assertRaisesRegex(ValueError, "complete original"):
            capture.finish()
        for unused in range(673):
            self.call(front)
        delivered = capture.finish()
        with self.assertRaises(ValueError):
            capture.finish()
        with self.assertRaises(ValueError):
            self.call(front)
        self.assertEqual(delivered["processed_frames"], 673)


if __name__ == "__main__":
    unittest.main()
