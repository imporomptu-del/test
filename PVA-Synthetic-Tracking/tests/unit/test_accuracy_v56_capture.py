"""CPU/generated tests only: no GPU, media, Jetson, or archived measurements."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]/"scripts"
if not (SCRIPTS/"accuracy_v56_capture.py").is_file():
    SCRIPTS = Path(__file__).resolve().parent  # flat frozen Jetson diagnostic package
sys.path.insert(0, str(SCRIPTS))
import accuracy_v56_capture as capture


def snapshot(shape=(8, 8), tile=4, center=(2.0, 2.0), radius=1, k=2):
    rect = capture.tile_rectangle(shape, tile, center, radius)
    x0, y0, x1, y1 = rect["capture_bounds_exclusive_xyxy"]
    h, w = y1-y0, x1-x0
    values = np.zeros((h, w, len(capture.FLOAT_FIELDS)), np.float32)
    values[..., 9] = 1.0
    flags = np.zeros((h, w, len(capture.FLAG_FIELDS)), np.uint8)
    cells = 2*((shape[0]+tile-1)//tile)*((shape[1]+tile-1)//tile)
    peaks = np.zeros((cells, k), capture.PEAK_DTYPE)
    peaks["x"] = -1
    peaks["y"] = -1
    return {"metadata": {"rectangle": rect}, "values": values, "flags": flags,
            "precise_sigmas": np.ones((h, w), np.float64),
            "native_counts": np.zeros(cells, np.int32), "native_peaks": peaks}


def add_candidate(snap, x, y, score, polarity=0):
    rect = snap["metadata"]["rectangle"]
    x0, y0, _, _ = rect["capture_bounds_exclusive_xyxy"]
    yy, xx = y-y0, x-x0
    snap["flags"][yy, xx, [0, 1, 2, 3, 4, 9, 10+polarity]] = 1
    snap["values"][yy, xx, 13+polarity] = score
    snap["values"][yy, xx, 5] = score*(1 if polarity == 0 else -1)


def fill_native(snap, k=2):
    ranks = capture.rank_candidates(snap, max_candidates_per_tile_polarity=k, verify_native=False)
    rect = snap["metadata"]["rectangle"]
    tile = rect["tile_size"]
    nx = (rect["shape_hw"][1]+tile-1)//tile
    x0, y0, _, _ = rect["capture_bounds_exclusive_xyxy"]
    for item in ranks["tile_polarity_summary"]:
        cell = 2*item["tile_id"]+item["polarity"]
        snap["native_counts"][cell] = item["prequota_count"]
    for polarity, label in enumerate(("positive", "negative")):
        rank_array = ranks[label+"_prequota_rank"]
        for yy, xx in np.argwhere((rank_array > 0) & (rank_array <= k)):
            x, y = int(xx+x0), int(yy+y0)
            cell = 2*((y//tile)*nx+x//tile)+polarity
            rank = int(rank_array[yy, xx])-1
            snap["native_peaks"][cell, rank] = (x, y, snap["values"][yy, xx, 13+polarity],
                                                snap["values"][yy, xx, 5], snap["values"][yy, xx, 9])


class RectangleTests(unittest.TestCase):
    def test_interior_whole_tile_with_halo(self):
        r = capture.tile_rectangle((100, 100), 16, (24.0, 24.0), 3)
        self.assertEqual(r["tile_bounds_exclusive_xyxy"], [16, 16, 32, 32])
        self.assertEqual(r["capture_bounds_exclusive_xyxy"], [14, 14, 34, 34])
        self.assertEqual(r["full_tile_ids"], [8])

    def test_boundary_intersects_four_tiles(self):
        r = capture.tile_rectangle((100, 100), 16, (16.0, 16.0), 3)
        self.assertEqual(r["full_tile_ids"], [0, 1, 7, 8])
        self.assertEqual(r["capture_bounds_exclusive_xyxy"], [0, 0, 34, 34])

    def test_fractional_probe_not_rounded(self):
        r = capture.tile_rectangle((20, 20), 8, (8.2, 7.8), 2)
        self.assertEqual(r["probe_bounds_inclusive_xyxy"], [7, 6, 10, 9])
        self.assertEqual(r["probe_xy"], [8.2, 7.8])

    def test_partial_border_tile(self):
        r = capture.tile_rectangle((17, 19), 8, (18.0, 16.0), 2)
        self.assertEqual(r["tile_bounds_exclusive_xyxy"], [16, 8, 19, 17])
        self.assertEqual(r["capture_bounds_exclusive_xyxy"], [14, 6, 19, 17])

    def test_missing_integer_pixel_rejected(self):
        with self.assertRaises(ValueError):
            capture.tile_rectangle((20, 20), 8, (8.2, 7.8), 0)

    def test_excess_tile_span_rejected(self):
        with self.assertRaises(ValueError):
            capture.tile_rectangle((100, 100), 8, (40.0, 40.0), 20)

    def test_three_by_three_large_tiles_within_pixel_bound(self):
        r = capture.tile_rectangle((2048, 2048), 256, (768.0, 768.0), 256)
        x0, y0, x1, y1 = r["capture_bounds_exclusive_xyxy"]
        self.assertEqual(len(r["full_tile_ids"]), 9)
        self.assertEqual((x1-x0)*(y1-y0), 772*772)

    def test_bad_parameters(self):
        cases = [((0, 8), 4, (2.0, 2.0), 1), ((8, 8), True, (2.0, 2.0), 1),
                 ((8, 8), 4, (float("nan"), 2.0), 1), ((8, 8), 4, (8.0, 2.0), 1),
                 ((8, 8), 4, (True, 2.0), 1), ((8, 8), 4, (2.0, 2.0), True),
                 ((8, 8), 4, (2.0, 2.0), -1), ((6000, 6000), 4, (2.0, 2.0), 1)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                capture.tile_rectangle(*args)

    def test_descriptor_tampering_rejected(self):
        r = capture.tile_rectangle((8, 8), 4, (2.0, 2.0), 1)
        r["capture_bounds_exclusive_xyxy"][2] = 5
        with self.assertRaises(ValueError):
            capture.validate_rectangle(r)

    def test_extra_descriptor_fields_rejected(self):
        r = capture.tile_rectangle((8, 8), 4, (2.0, 2.0), 1)
        r["extra"] = 1
        with self.assertRaises(ValueError):
            capture.validate_rectangle(r)


class RankTests(unittest.TestCase):
    def test_no_candidates(self):
        s = snapshot()
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=2)
        self.assertFalse(r["positive_prequota_rank"].any())
        self.assertEqual(len(r["tile_polarity_summary"]), 2)

    def test_descending_scores_and_y_x_tie_break(self):
        s = snapshot(k=4)
        for x, y, score in ((2, 2, 4), (1, 2, 4), (3, 0, 5), (0, 1, 4)):
            add_candidate(s, x, y, score)
        fill_native(s, k=4)
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=4)["positive_prequota_rank"]
        self.assertEqual([r[0, 3], r[1, 0], r[2, 1], r[2, 2]], [1, 2, 3, 4])

    def test_prequota_includes_dropped_candidates(self):
        s = snapshot(k=1)
        for x in range(4):
            add_candidate(s, x, 1, x+1)
        fill_native(s, k=1)
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=1)
        self.assertEqual(r["positive_prequota_rank"][1, :4].tolist(), [4, 3, 2, 1])
        self.assertEqual(r["tile_polarity_summary"][0]["prequota_count"], 4)

    def test_polarities_separate(self):
        s = snapshot()
        add_candidate(s, 1, 1, 6, 0)
        add_candidate(s, 2, 2, 5, 1)
        fill_native(s)
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=2)
        self.assertEqual(r["positive_prequota_rank"][1, 1], 1)
        self.assertEqual(r["negative_prequota_rank"][2, 2], 1)
        self.assertEqual(r["positive_prequota_rank"][2, 2], 0)

    def test_halo_candidates_not_ranked(self):
        s = snapshot()
        add_candidate(s, 5, 1, 99)
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=2)
        self.assertEqual(r["positive_prequota_rank"][1, 5], 0)

    def test_multiple_tiles_independent_ranks(self):
        s = snapshot(center=(4.0, 4.0), radius=1)
        for x, y in ((1, 1), (5, 1), (1, 5), (5, 5)):
            add_candidate(s, x, y, 5)
        fill_native(s)
        r = capture.rank_candidates(s, max_candidates_per_tile_polarity=2)
        self.assertEqual(np.count_nonzero(r["positive_prequota_rank"] == 1), 4)

    def test_native_count_mismatch_fails(self):
        s = snapshot()
        add_candidate(s, 1, 1, 6)
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_native_top_k_mismatch_fails(self):
        s = snapshot()
        add_candidate(s, 1, 1, 6)
        fill_native(s)
        s["native_peaks"][0, 0]["score"] = 6.001
        with self.assertRaisesRegex(RuntimeError, "top-k mismatch"):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_empty_sentinel_mismatch_fails(self):
        s = snapshot()
        s["native_peaks"][0, 0]["score"] = 1
        with self.assertRaisesRegex(RuntimeError, "sentinel mismatch"):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_nonfinite_candidate_score_fails(self):
        s = snapshot()
        add_candidate(s, 1, 1, float("nan"))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2, verify_native=False)

    def test_eligibility_mismatch_fails(self):
        s = snapshot()
        s["flags"][0, 0, 2] = 1
        with self.assertRaisesRegex(RuntimeError, "eligibility"):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_nonboolean_flag_fails(self):
        s = snapshot()
        s["flags"][0, 0, 0] = 2
        with self.assertRaises(ValueError):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_wrong_value_dtype_fails(self):
        s = snapshot()
        s["values"] = s["values"].astype(np.float64)
        with self.assertRaises(ValueError):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_wrong_native_peak_dtype_fails(self):
        s = snapshot()
        s["native_peaks"] = np.zeros((8, 2), [("x", np.int64), ("y", np.int64),
                                             ("score", np.float32), ("response", np.float32), ("noise", np.float32)])
        with self.assertRaises(ValueError):
            capture.rank_candidates(s, max_candidates_per_tile_polarity=2)

    def test_rank_does_not_mutate_inputs(self):
        s = snapshot()
        add_candidate(s, 1, 1, 6)
        fill_native(s)
        copies = {key: value.copy() for key, value in s.items() if isinstance(value, np.ndarray)}
        capture.rank_candidates(s, max_candidates_per_tile_polarity=2)
        for key, value in copies.items():
            np.testing.assert_array_equal(s[key], value)


class Callable:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


class NativeBoundaryTests(unittest.TestCase):
    def setup_fake(self):
        s = snapshot()
        cfg = SimpleNamespace(tile_size=4, temporal_threshold_sigma=3.0,
                              spatial_threshold_sigma=2.0, max_candidates_per_tile_polarity=2)
        detector = SimpleNamespace(_owned=mock.Mock(), front=1234, busy=True, poisoned=False,
                                   shape=(8, 8), config=cfg, counts=s["native_counts"],
                                   peaks=s["native_peaks"], lib=object())
        calls = []

        def native(*args):
            calls.append(args[:9])
            for ptr, field in zip(args[9:], ("values", "flags", "precise_sigmas")):
                array = s[field]
                C.memmove(ptr, array.ctypes.data, array.nbytes)
            return 0

        lib = SimpleNamespace(seaqr_accuracy_v56_abi=Callable(lambda: 1),
                              seaqr_accuracy_v56_capture=Callable(native))
        return s, detector, lib, calls

    def test_separate_bridge_and_original_pointer(self):
        s, detector, lib, calls = self.setup_fake()
        original = detector.lib
        out = capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False,
                                       warp_handle=9876)
        self.assertIs(detector.lib, original)
        self.assertEqual(calls[0][:2], (1234, 9876))
        self.assertEqual(calls[0][6:9], (0, 3.0, 2.0))
        self.assertTrue(out["metadata"]["warp_buffers"])
        for value in out.values():
            if isinstance(value, np.ndarray):
                self.assertFalse(value.flags.writeable)
        detector._owned.assert_called_once()

    def test_host_pointer_is_null(self):
        s, detector, lib, calls = self.setup_fake()
        capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)
        self.assertIsNone(calls[0][1])

    def test_same_library_rejected(self):
        s, detector, lib, _ = self.setup_fake()
        detector.lib = lib
        with self.assertRaisesRegex(ValueError, "separate bridge"):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)

    def test_inactive_or_poisoned_rejected(self):
        for field, value in (("busy", False), ("poisoned", True), ("front", None)):
            with self.subTest(field=field):
                s, detector, lib, _ = self.setup_fake()
                setattr(detector, field, value)
                with self.assertRaises(RuntimeError):
                    capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)

    def test_geometry_mismatch_rejected(self):
        s, detector, lib, _ = self.setup_fake()
        detector.shape = (9, 8)
        with self.assertRaises(ValueError):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)

    def test_nonboolean_ready_rejected(self):
        s, detector, lib, _ = self.setup_fake()
        with self.assertRaises(ValueError):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=0)

    def test_bad_abi_rejected(self):
        s, detector, lib, _ = self.setup_fake()
        lib.seaqr_accuracy_v56_abi = Callable(lambda: 2)
        with self.assertRaisesRegex(RuntimeError, "ABI"):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)

    def test_cuda_failure_propagates(self):
        s, detector, lib, _ = self.setup_fake()
        lib.seaqr_accuracy_v56_capture = Callable(lambda *args: 700)
        with self.assertRaisesRegex(RuntimeError, "700"):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)

    def test_owned_failure_prevents_native_access(self):
        s, detector, lib, calls = self.setup_fake()
        detector._owned.side_effect = RuntimeError("wrong thread")
        with self.assertRaisesRegex(RuntimeError, "wrong thread"):
            capture.capture_prepared(detector, lib, rectangle=s["metadata"]["rectangle"], ready=False)
        self.assertEqual(calls, [])


class SourceContractTests(unittest.TestCase):
    def test_only_archived_include_and_new_exports(self):
        source = (SCRIPTS/"accuracy_v56_capture.cu").read_text()
        self.assertIn('#include "visible_front_v26.cu"', source)
        self.assertNotIn("seaqr_front_v26_prepare_warp(", source)
        self.assertNotIn("seaqr_front_v26_finish(", source)
        self.assertIn("const float* image=s.image;const float* blur=s.blur", source)
        self.assertIn("image=w.output;blur=w.blur", source)

    def test_channels_have_unique_names(self):
        self.assertEqual(len(capture.FLOAT_FIELDS), 20)
        self.assertEqual(len(capture.FLAG_FIELDS), 13)
        self.assertEqual(len(set(capture.FLOAT_FIELDS)), 20)
        self.assertEqual(len(set(capture.FLAG_FIELDS)), 13)


def run_cuda_smoke(original_library_path, diagnostic_library_path):
    """Actual CUDA arithmetic test on generated arrays only; never opens media.

    The original frozen library owns every detector allocation and runs every
    prepare/finish. The separate bridge only observes those original buffers.
    This deliberately supplies controlled host image/blur arrays to the native
    ABI, not a claimed physical sensor or Gaussian-consistency simulation.
    """
    original_path = Path(original_library_path).resolve(strict=True)
    diagnostic_path = Path(diagnostic_library_path).resolve(strict=True)
    if original_path == diagnostic_path:
        raise ValueError("Original and diagnostic libraries must be separate")
    original = C.CDLL(str(original_path), mode=C.RTLD_LOCAL)
    bridge = C.CDLL(str(diagnostic_path), mode=C.RTLD_LOCAL)
    p, i, f, d = C.c_void_p, C.c_int, C.c_float, C.c_double
    specs = {
        "create": ([i]*4, p), "destroy": ([p], None),
        "prepare_host": ([p]*4+[i, i, f, d, i, f, f]+[p]*5, i),
        "finish": ([p, p, i, p, i, f, f, f, f, i], i),
    }
    for name, (args, result) in specs.items():
        fn = getattr(original, "seaqr_front_v26_"+name)
        fn.argtypes, fn.restype = args, result
    handle = original.seaqr_front_v26_create(64, 64, 16, 2)
    if not handle:
        raise RuntimeError("Generated CUDA smoke allocation failed")
    cfg = SimpleNamespace(tile_size=16, temporal_threshold_sigma=3.0,
                          spatial_threshold_sigma=2.0, max_candidates_per_tile_polarity=1)
    detector = SimpleNamespace(_owned=lambda: None, front=handle, busy=True, poisoned=False,
                               shape=(64, 64), config=cfg, lib=original,
                               counts=np.empty(32, np.int32), peaks=np.empty((32, 1), capture.PEAK_DTYPE))
    image = np.full((64, 64), 96.0, np.float32)
    blur, valid = image.copy(), np.ones((64, 64), np.uint8)
    eligible, sigmas, searchable = np.empty((64, 64), np.uint8), np.empty(16, np.float64), np.empty(1, np.int32)
    rectangle = capture.tile_rectangle((64, 64), 16, (31.5, 31.5), 12)
    signatures = []

    def call_prepare(reset, ready):
        rc = original.seaqr_front_v26_prepare_host(
            handle, image.ctypes.data, blur.ctypes.data, valid.ctypes.data, reset, ready,
            1.0, 1.0, 1, 3.0, 2.0, eligible.ctypes.data, sigmas.ctypes.data,
            detector.peaks.ctypes.data, detector.counts.ctypes.data, searchable.ctypes.data)
        if rc:
            raise RuntimeError(f"Original prepare failed in generated smoke: {rc}")
        before = {"counts": detector.counts.tobytes(), "peaks": detector.peaks.tobytes(),
                  "eligible": eligible.tobytes(), "sigmas": sigmas.tobytes()}
        snap = capture.capture_prepared(detector, bridge, rectangle=rectangle, ready=bool(ready))
        after = {"counts": detector.counts.tobytes(), "peaks": detector.peaks.tobytes(),
                 "eligible": eligible.tobytes(), "sigmas": sigmas.tobytes()}
        if before != after:
            raise AssertionError("Read-only capture modified generated host outputs")
        signatures.append({name: hashlib.sha256(value).hexdigest() for name, value in before.items()})
        return snap

    def finish():
        offsets = np.array([[0, 0]], np.int32)
        rc = original.seaqr_front_v26_finish(handle, None, 0, offsets.ctypes.data, 1,
                                            0.2, 0.1, 25.0, 1.0, 0)
        if rc:
            raise RuntimeError(f"Original finish failed in generated smoke: {rc}")

    def at(snap, x, y, fields, name):
        x0, y0, _, _ = rectangle["capture_bounds_exclusive_xyxy"]
        array = snap["flags"] if fields == capture.FLAG_FIELDS else snap["values"]
        return array[y-y0, x-x0, fields.index(name)].item()

    try:
        warmup = call_prepare(1, 0)
        if detector.counts.any() or warmup["flags"][..., 4].any():
            raise AssertionError("Warmup produced eligible candidates")
        finish()
        for x, y, response in ((20, 20, 5.0), (22, 20, 5.0), (42, 42, -6.0),
                               (31, 32, 4.0), (32, 32, -7.0),
                               (20, 28, 3.0), (28, 20, 2.0), (44, 36, -3.0)):
            blur[y, x] += response
        valid[20, 40] = 0
        snap = call_prepare(0, 1)
        checks = {
            "positive_temporal_pass": at(snap, 20, 20, capture.FLAG_FIELDS, "positive_temporal_pass") == 1,
            "positive_spatial_pass": at(snap, 20, 20, capture.FLAG_FIELDS, "positive_spatial_pass") == 1,
            "tie_raw_peaks_survive": all(at(snap, x, 20, capture.FLAG_FIELDS, "raw_absolute_peak") == 1 for x in (20, 22)),
            "opposite_polarity_tile_edge_neighbor_suppresses": at(snap, 31, 32, capture.FLAG_FIELDS, "raw_absolute_peak") == 0,
            "edge_temporal_and_spatial_pass": at(snap, 31, 32, capture.FLAG_FIELDS, "positive_temporal_pass") == 1
                and at(snap, 31, 32, capture.FLAG_FIELDS, "positive_spatial_pass") == 1,
            "edge_neighbor_max_is_seven": at(snap, 31, 32, capture.FLOAT_FIELDS, "neighborhood_max_abs") == 7.0,
            "negative_candidate": at(snap, 42, 42, capture.FLAG_FIELDS, "negative_candidate") == 1,
            "positive_threshold_equality_passes": at(snap, 20, 28, capture.FLAG_FIELDS, "positive_candidate") == 1,
            "negative_threshold_equality_passes": at(snap, 44, 36, capture.FLAG_FIELDS, "negative_candidate") == 1,
            "spatial_equality_passes_but_temporal_fails": at(snap, 28, 20, capture.FLAG_FIELDS, "positive_spatial_pass") == 1
                and at(snap, 28, 20, capture.FLAG_FIELDS, "positive_temporal_pass") == 0,
            "current_support_hole": at(snap, 40, 20, capture.FLAG_FIELDS, "support") == 0,
            "previous_support_preserved": at(snap, 40, 20, capture.FLAG_FIELDS, "previous_support") == 1,
            "support_hole_ineligible": at(snap, 40, 20, capture.FLAG_FIELDS, "eligible") == 0,
        }
        x0, y0, _, _ = rectangle["capture_bounds_exclusive_xyxy"]
        checks["prequota_ranks_include_dropped_tie"] = (
            snap["positive_prequota_rank"][20-y0, 20-x0] == 1
            and snap["positive_prequota_rank"][20-y0, 22-x0] == 2)
        checks = {key: bool(value) for key, value in checks.items()}
        if not all(checks.values()):
            raise AssertionError("Generated CUDA smoke predicate failure: "+repr(checks))
        finish()
        return {"schema": "accuracy_v56_generated_cuda_smoke_v1", "passed": True,
                "generated_only": True, "media_opened": False, "physical_sensor_simulation": False,
                "original_library_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                "diagnostic_library_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
                "checks": checks, "host_output_sha256": signatures,
                "tile_polarity_summary": snap["metadata"]["tile_polarity_summary"]}
    finally:
        original.seaqr_front_v26_destroy(handle)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--cuda-smoke":
        print(json.dumps(run_cuda_smoke(sys.argv[2], sys.argv[3]), indent=2, allow_nan=False))
    else:
        unittest.main()
