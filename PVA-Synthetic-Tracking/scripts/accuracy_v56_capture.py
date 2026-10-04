"""Bounded, read-only V56 pre-learning diagnostics for hash-pinned V26.

This module does not select candidates, mutate detector state, decode media, or
load a native library. The caller supplies the already-owned detector and fixes
the capture coordinates before replay. Native diagnostic arithmetic is captured
on the GPU; Python only orders already-qualified pixels with the original
score/y/x tie break and checks them against the original selection outputs.
"""
from __future__ import annotations

import ctypes as C
import math
from typing import Any

import numpy as np


FLOAT_FIELDS = (
    "image", "blur", "median", "spatial", "background", "temporal", "variance",
    "tile_center", "tile_sigma_float", "noise", "centered_temporal",
    "temporal_threshold_dn", "spatial_threshold_dn", "positive_score",
    "negative_score", "neighborhood_max_abs", "positive_signed_temporal",
    "negative_signed_temporal", "positive_signed_spatial", "negative_signed_spatial",
)
FLAG_FIELDS = (
    "support", "previous_support", "native_eligible", "ready", "eligible",
    "positive_temporal_pass", "negative_temporal_pass", "positive_spatial_pass",
    "negative_spatial_pass", "raw_absolute_peak", "positive_candidate",
    "negative_candidate", "finite_neighborhood",
)
MAX_PIXELS = 600_000
MAX_TILE_SPAN = 3
HALO = 2
PEAK_DTYPE = np.dtype([("x", np.int32), ("y", np.int32), ("score", np.float32),
                       ("response", np.float32), ("noise", np.float32)])


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def tile_rectangle(shape, tile_size: int, probe_xy, radius: int) -> dict:
    """Enclose a fixed inclusive pixel probe box in full tiles plus peak halo.

    Fractional reference centers are retained exactly in metadata. The inclusive
    pixel box uses ceil(center-radius)..floor(center+radius), not rounding the
    reference position or silently widening the requested spatial gate.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("A bounded (height, width) shape is required")
    h = _integer(shape[0], "height", 1, 32766)
    w = _integer(shape[1], "width", 1, 32766)
    if h * w > 32_000_000:
        raise ValueError("Image exceeds the frozen front-end bound")
    tile = _integer(tile_size, "tile_size", 1, 256)
    radius = _integer(radius, "radius", 0, 256)
    if not isinstance(probe_xy, (tuple, list)) or len(probe_xy) != 2:
        raise ValueError("A finite fixed (x, y) reference center is required")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in probe_xy):
        raise ValueError("A finite fixed (x, y) reference center is required")
    x, y = map(float, probe_xy)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Reference center must be inside the frame")
    px0, py0 = max(0, math.ceil(x-radius)), max(0, math.ceil(y-radius))
    px1, py1 = min(w-1, math.floor(x+radius)), min(h-1, math.floor(y+radius))
    if px0 > px1 or py0 > py1:
        raise ValueError("Reference box contains no integer pixels")
    tx0, ty0, tx1, ty1 = px0//tile, py0//tile, px1//tile, py1//tile
    if tx1-tx0+1 > MAX_TILE_SPAN or ty1-ty0+1 > MAX_TILE_SPAN:
        raise ValueError("Capture exceeds the fixed 3x3-tile bound")
    cx0, cy0 = tx0*tile, ty0*tile
    cx1, cy1 = min(w, (tx1+1)*tile), min(h, (ty1+1)*tile)
    x0, y0, x1, y1 = max(0, cx0-HALO), max(0, cy0-HALO), min(w, cx1+HALO), min(h, cy1+HALO)
    if (x1-x0)*(y1-y0) > MAX_PIXELS:
        raise ValueError("Capture exceeds the native pixel bound")
    nx = math.ceil(w/tile)
    return {
        "schema": "accuracy_v56_fixed_rectangle_v1", "shape_hw": [h, w],
        "tile_size": tile, "probe_xy": [x, y], "radius_px": radius,
        "probe_bounds_inclusive_xyxy": [px0, py0, px1, py1],
        "tile_bounds_exclusive_xyxy": [cx0, cy0, cx1, cy1],
        "capture_bounds_exclusive_xyxy": [x0, y0, x1, y1],
        "full_tile_ids": [ty*nx+tx for ty in range(ty0, ty1+1) for tx in range(tx0, tx1+1)],
        "halo_px": HALO,
    }


def validate_rectangle(rectangle: dict) -> dict:
    if not isinstance(rectangle, dict):
        raise ValueError("A frozen rectangle descriptor is required")
    try:
        expected = tile_rectangle(rectangle["shape_hw"], rectangle["tile_size"],
                                  rectangle["probe_xy"], rectangle["radius_px"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete rectangle descriptor") from exc
    if rectangle != expected:
        raise ValueError("Rectangle descriptor changed or is not canonical")
    return expected


def signatures(lib) -> None:
    abi = lib.seaqr_accuracy_v56_abi
    abi.argtypes = []
    abi.restype = C.c_int
    if abi() != 1:
        raise RuntimeError("Unsupported V56 capture ABI")
    fn = lib.seaqr_accuracy_v56_capture
    fn.argtypes = [C.c_void_p, C.c_void_p] + [C.c_int]*5 + [C.c_float]*2 + [C.c_void_p]*3
    fn.restype = C.c_int


def rank_candidates(snapshot: dict, *, max_candidates_per_tile_polarity: int,
                    verify_native: bool = True) -> dict:
    """One-based prequota ranks; zero means not an eligible tile candidate.

    Only full captured tiles are ranked. Halo pixels remain zero. Native counts
    and top-k outputs are checked exactly, so a diagnostic predicate discrepancy
    cannot silently become an explanation for a production miss.
    """
    rectangle = validate_rectangle(snapshot["metadata"]["rectangle"])
    k = _integer(max_candidates_per_tile_polarity, "tile cap", 1, 16)
    values, flags = snapshot["values"], snapshot["flags"]
    x0, y0, x1, y1 = rectangle["capture_bounds_exclusive_xyxy"]
    height, width = y1-y0, x1-x0
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32
            or values.shape != (height, width, len(FLOAT_FIELDS))
            or not isinstance(flags, np.ndarray) or flags.dtype != np.uint8
            or flags.shape != (height, width, len(FLAG_FIELDS))
            or np.any(flags > 1)):
        raise ValueError("Capture arrays have invalid shapes, types, or flags")
    if not np.array_equal(flags[..., 2], flags[..., 4]):
        raise RuntimeError("Captured native eligibility differs from exact diagnostic eligibility")
    h, w = rectangle["shape_hw"]
    tile = rectangle["tile_size"]
    nx = math.ceil(w/tile)
    result = {"positive_prequota_rank": np.zeros((height, width), np.int32),
              "negative_prequota_rank": np.zeros((height, width), np.int32)}
    summaries = []
    if verify_native:
        counts, peaks = snapshot["native_counts"], snapshot["native_peaks"]
        cells = 2*math.ceil(h/tile)*nx
        if (counts.dtype != np.int32 or counts.shape != (cells,)
                or peaks.shape != (cells, k) or peaks.dtype != PEAK_DTYPE):
            raise ValueError("Invalid native count or peak arrays")
    for tid in rectangle["full_tile_ids"]:
        tx, ty = tid%nx, tid//nx
        left, top = tx*tile-x0, ty*tile-y0
        right, bottom = min(w, (tx+1)*tile)-x0, min(h, (ty+1)*tile)-y0
        for polarity, name in enumerate(("positive", "negative")):
            local_y, local_x = np.nonzero(flags[top:bottom, left:right, 10+polarity])
            yy, xx = local_y+top, local_x+left
            scores = values[yy, xx, 13+polarity]
            if not np.isfinite(scores).all():
                raise RuntimeError("Qualified candidate has a nonfinite diagnostic score")
            order = np.lexsort((xx+x0, yy+y0, -scores))
            xx, yy = xx[order], yy[order]
            result[name+"_prequota_rank"][yy, xx] = np.arange(1, len(xx)+1, dtype=np.int32)
            cell = 2*tid+polarity
            if verify_native:
                if int(counts[cell]) != len(xx):
                    raise RuntimeError(f"Tile {tid} polarity {polarity}: diagnostic/native candidate count mismatch")
                for j in range(k):
                    peak = peaks[cell, j]
                    if j >= len(xx):
                        if tuple(peak.tolist()) != (-1, -1, 0.0, 0.0, 0.0):
                            raise RuntimeError("Native empty peak sentinel mismatch")
                        continue
                    actual = (int(peak["x"]), int(peak["y"]), peak["score"], peak["response"], peak["noise"])
                    expected = (int(xx[j]+x0), int(yy[j]+y0), values[yy[j], xx[j], 13+polarity],
                                values[yy[j], xx[j], 5], values[yy[j], xx[j], 9])
                    if actual != expected:
                        raise RuntimeError(f"Tile {tid} polarity {polarity}: diagnostic/native top-k mismatch")
            summaries.append({"tile_id": tid, "polarity": polarity,
                              "prequota_count": len(xx), "tile_cap": k,
                              "native_selection_verified": verify_native})
    result["tile_polarity_summary"] = summaries
    return result


def capture_prepared(detector, diagnostic_lib, *, rectangle: dict, ready: bool, warp_handle=None) -> dict:
    """Capture one successful native prepare before native finish.

    ``diagnostic_lib`` is a separate, caller-verified CDLL. The original detector
    and warp CDLL identities remain untouched; only layout-compatible pointers
    cross into this read-only bridge. Requires the owning thread and active
    update. The caller must not invoke
    this after finish: ``learn`` is no longer the native eligibility array then.
    """
    rectangle = validate_rectangle(rectangle)
    if type(ready) is not bool:
        raise ValueError("ready must be a boolean copied from the prepare call")
    detector._owned()
    if not detector.front or not detector.busy or detector.poisoned:
        raise RuntimeError("Capture requires an active, successfully prepared owned front")
    if list(detector.shape) != rectangle["shape_hw"] or detector.config.tile_size != rectangle["tile_size"]:
        raise ValueError("Capture geometry does not match the prepared detector")
    cfg = detector.config
    thresholds = (cfg.temporal_threshold_sigma, cfg.spatial_threshold_sigma)
    if not all(math.isfinite(v) and np.isfinite(np.float32(v)) for v in thresholds):
        raise ValueError("Finite float32 thresholds required")
    if diagnostic_lib is detector.lib:
        raise ValueError("Diagnostics require a separate bridge library; retain the original detector library")
    signatures(diagnostic_lib)
    x0, y0, x1, y1 = rectangle["capture_bounds_exclusive_xyxy"]
    shape = (y1-y0, x1-x0)
    values = np.empty((*shape, len(FLOAT_FIELDS)), np.float32)
    flags = np.empty((*shape, len(FLAG_FIELDS)), np.uint8)
    sigmas = np.empty(shape, np.float64)
    rc = diagnostic_lib.seaqr_accuracy_v56_capture(
        detector.front, warp_handle, x0, y0, shape[1], shape[0], int(ready),
        *thresholds, values.ctypes.data, flags.ctypes.data, sigmas.ctypes.data)
    if rc:
        raise RuntimeError(f"Read-only V56 native capture failed with CUDA status {rc}")
    snapshot = {
        "metadata": {"schema": "accuracy_v56_prelearning_capture_v1", "rectangle": rectangle,
                     "float_fields": list(FLOAT_FIELDS), "flag_fields": list(FLAG_FIELDS),
                     "ready": ready, "warp_buffers": warp_handle is not None,
                     "temporal_threshold_sigma_float32": float(np.float32(thresholds[0])),
                     "spatial_threshold_sigma_float32": float(np.float32(thresholds[1])),
                     "read_only_capture": True, "production_selection_unchanged": True},
        "values": values, "flags": flags, "precise_sigmas": sigmas,
        "native_counts": detector.counts.copy(), "native_peaks": detector.peaks.copy(),
    }
    ranks = rank_candidates(snapshot, max_candidates_per_tile_polarity=cfg.max_candidates_per_tile_polarity)
    snapshot["metadata"]["tile_polarity_summary"] = ranks.pop("tile_polarity_summary")
    snapshot.update(ranks)
    for value in snapshot.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return snapshot
