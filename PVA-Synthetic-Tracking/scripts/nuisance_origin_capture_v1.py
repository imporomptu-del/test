"""Bounded passive observation of the frozen U8 resident detector.

This module never selects media, changes detector settings, or writes files. The
caller must verify a complete causal replay against its frozen original journal.
Extra patch reads use the existing scratch-only gather; no device state is set.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import math
from collections.abc import Mapping

import numpy as np


SCHEMA = "seaqr.nuisance-origin-capture.v1"
FRAME_GROUPS = ((58, 59, 60, 61, 62), (73, 74, 75, 76, 77),
                (88, 89, 90, 91, 92), (433, 434, 435, 436, 437),
                (448, 449, 450, 451, 452))
CAPTURE_FRAMES = frozenset(frame for group in FRAME_GROUPS for frame in group)
TOTAL_FRAMES = 673
PATCH_RADIUS = 8
PATCH_SIZE = 17
POINT_KEYS = frozenset(("point_id", "episode_id", "role", "reference_xy", "source_xy"))
PEAK_NAMES = ("x", "y", "score", "response", "noise")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(array):
    """Own immutable bytes; preserve dtype, shape and every floating-point bit."""
    value = np.asarray(array)
    require(value.dtype.kind in "buif" and not value.dtype.hasobject,
            "Only plain numerical diagnostic arrays are allowed")
    require(value.size <= 32768, "Unbounded diagnostic array")
    require(value.dtype.kind != "f" or np.isfinite(value).all(), "Nonfinite diagnostic array")
    raw = value.tobytes(order="C")
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                data_base64=base64.b64encode(raw).decode("ascii"),
                sha256=hashlib.sha256(raw).hexdigest())


def _identity(value):
    """Accept both ordinary integer handles and ctypes c_void_p arguments."""
    return getattr(value, "value", value)


def _f32_bits(value):
    return np.asarray(value, dtype=np.float32).tobytes()


def validate_plan(plan):
    points = plan.get("frame_points") if isinstance(plan, Mapping) else getattr(plan, "frame_points", None)
    require(isinstance(points, Mapping), "Plan.frame_points must be a mapping")
    require(set(points) == {str(frame) for frame in CAPTURE_FRAMES},
            "Exactly the declared 25 capture frames are required")
    result, episodes = {}, {}
    for frame in sorted(CAPTURE_FRAMES):
        rows = points[str(frame)]
        require(isinstance(rows, list) and 1 <= len(rows) <= 16, "Bounded diagnostic points required")
        seen_ids, seen_pairs, paired = set(), set(), {}
        for row in rows:
            require(isinstance(row, dict) and set(row) == POINT_KEYS, "Unexpected point fields")
            require(all(isinstance(row[key], str) and 0 < len(row[key]) <= 160
                        for key in ("point_id", "episode_id")), "Invalid point identity")
            require(row["role"] in {"sample", "comparator"}, "Invalid diagnostic point role")
            xy, source = row["reference_xy"], row["source_xy"]
            require(isinstance(xy, list) and len(xy) == 2
                    and all(type(v) is int and 0 <= v < 32767 for v in xy),
                    "Reference sites must be bounded integer native pixels")
            require(isinstance(source, list) and len(source) == 2
                    and all(type(v) in (float, int) and math.isfinite(v) for v in source),
                    "Planned source coordinates must be finite numbers")
            pair = (row["episode_id"], row["role"])
            require(row["point_id"] not in seen_ids and pair not in seen_pairs, "Duplicate planned point")
            seen_ids.add(row["point_id"])
            seen_pairs.add(pair)
            paired.setdefault(row["episode_id"], set()).add(row["role"])
            episodes.setdefault(row["episode_id"], set()).add(frame)
        require(all("sample" in roles for roles in paired.values()),
                "Every episode/frame needs its sample; comparators may be unavailable")
        result[frame] = copy.deepcopy(rows)
    require(len(episodes) == 8 and 40 <= sum(map(len, result.values())) <= 80,
            "Exactly eight five-frame episodes with optional comparators are required")
    require(all(tuple(sorted(frames)) in FRAME_GROUPS for frames in episodes.values()),
            "Each episode must cover exactly one declared five-frame group")
    return result


class _FunctionFacade:
    def __init__(self, original, callback):
        self.original, self.callback = original, callback

    def __call__(self, *args):
        return self.callback(self.original, args)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _LibraryFacade:
    def __init__(self, original, capture, instance):
        self.original = original
        self.finish = _FunctionFacade(original.seaqr_front_v26_finish,
            lambda function, args: capture._native_finish(instance, function, args))

    def __getattr__(self, name):
        if name == "seaqr_front_v26_finish":
            return self.finish
        if name == "seaqr_front_v26_noise_probe":
            raise ValueError("A destructive noise probe is forbidden on a live diagnostic front")
        return getattr(self.original, name)


class OriginCapture:
    """Observe one original front from frame zero; return original outputs.

    Usage: capture = OriginCapture(plan); Wrapped = capture.front_class(Front).
    Install Wrapped through the caller's module facade. After the full replay,
    capture.finish() returns JSON-compatible bounded evidence. It does not itself
    certify journal parity, detector accuracy, or production performance.
    """

    def __init__(self, plan):
        self.frame_points = validate_plan(plan)
        self.expected_points = sum(map(len, self.frame_points.values()))
        self.instance = None
        self._wrapped = False
        self._active = None
        self._records = []
        self._updates = 0
        self._finishes = 0
        self._error = None
        self._sealed = False

    def front_class(self, original):
        require(not self._wrapped and self.instance is None, "Only one front-class binding is allowed")
        require(isinstance(original, type), "Original front must be a class")
        self._wrapped = True
        capture = self

        class CapturedOriginFront(original):
            def __init__(self, *args, **kwargs):
                require(capture.instance is None and not capture._sealed,
                        "Only one resident front instance is allowed")
                super().__init__(*args, **kwargs)
                capture.instance = self
                self.lib = _LibraryFacade(self.lib, capture, self)

            def update(self, image, valid, segment, learning_centers=()):
                try:
                    capture._begin(self, image, valid, segment, learning_centers)
                    result = super().update(image, valid, segment, learning_centers)
                    capture._end(self, image, valid, learning_centers, result)
                    return result
                except BaseException as exc:
                    capture._error = repr(exc)
                    raise
                finally:
                    capture._active = None

        return CapturedOriginFront

    def _begin(self, instance, image, valid, segment, learning_centers):
        require(not self._sealed and self._error is None, "Capture already failed or finished")
        require(self.instance is instance and self._active is None, "Unexpected/reentrant front update")
        require(self._updates < TOTAL_FRAMES, "Extra frame beyond frozen full-clip scope")
        require(not getattr(instance, "busy", False), "Cannot observe an active front")
        require(isinstance(instance.lib, _LibraryFacade) and instance.lib.original is not None,
                "Diagnostic front library was replaced")
        frame = self._updates
        active = dict(frame=frame, native_finishes=0, selected=frame in self.frame_points,
                      segment=copy.deepcopy(segment), before_counters=self._counters(instance))
        self._active = active
        if not active["selected"]:
            return
        require(hasattr(image, "download_for_verification") and not getattr(image, "consumed", True),
                "Selected frames require the unconsumed original device warp")
        image.validate(valid)
        require(valid.dtype == np.bool_ and valid.ndim == 2 and tuple(image.shape) == valid.shape,
                "Original device frame and boolean validity are required")
        shape = tuple(valid.shape)
        require(min(shape) >= PATCH_SIZE and max(shape) < 32767 and valid.size <= 32000000,
                "Unexpected native image shape")
        for point in self.frame_points[frame]:
            x, y = point["reference_xy"]
            require(PATCH_RADIUS <= x < shape[1] - PATCH_RADIUS
                    and PATCH_RADIUS <= y < shape[0] - PATCH_RADIUS,
                    "Planned point lacks full native patch support; no clipping/padding")
        active.update(shape=shape, valid_bytes=valid.tobytes(),
                      learning=copy.deepcopy(learning_centers), points=[])
        warped, blur = image.download_for_verification()
        require(not image.consumed, "Diagnostic download consumed the production warp")
        for array in (warped, blur):
            require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype == np.float32,
                    "Unexpected actual warp download dtype/shape")
        for point in self.frame_points[frame]:
            entry = copy.deepcopy(point)
            entry["warped_image"] = descriptor(self._patch(warped, point))
            entry["warped_blur"] = descriptor(self._patch(blur, point))
            active["points"].append(entry)

    @staticmethod
    def _counters(instance):
        names = ("calls", "device_calls", "host_calls", "finish_calls", "learning_points")
        require(all(type(getattr(instance, key, None)) is int for key in names),
                "Missing original front lifecycle counters")
        return {key: getattr(instance, key) for key in names}

    @staticmethod
    def _patch(array, point):
        x, y = point["reference_xy"]
        return array[y - PATCH_RADIUS:y + PATCH_RADIUS + 1,
                     x - PATCH_RADIUS:x + PATCH_RADIUS + 1]

    def _read_native(self, instance):
        """Read-only exports work inside native-finish interception; no busy override."""
        active = self._active
        shape = active["shape"]
        require(tuple(instance.shape) == shape and instance.front and instance.handle,
                "Uninitialized/mismatched resident diagnostic handles")
        library = instance.lib.original
        background = np.empty(shape, np.float32)
        variance = np.empty(shape, np.float32)
        support = np.empty(shape, np.bool_)
        phase_mask = np.empty(shape, np.bool_)
        sigmas = np.empty_like(instance.sigmas)
        stats = np.empty((len(sigmas), 2), np.float32)
        require(sigmas.dtype == np.float64 and sigmas.ndim == 1 and len(sigmas) <= 16384,
                "Unexpected original tile-sigma buffer")
        code = library.seaqr_resident_debug(instance.handle, background.ctypes.data, variance.ctypes.data)
        require(code == 0, "Resident state debug read failed")
        code = library.seaqr_front_v26_debug(instance.front, support.ctypes.data, phase_mask.ctypes.data,
                                           stats.ctypes.data, sigmas.ctypes.data)
        require(code == 0, "Resident front debug read failed")
        return dict(background=background, variance=variance, support=support,
                    phase_mask=phase_mask, stats=stats, sigmas=sigmas)

    @staticmethod
    def _peak_records(peaks, shape):
        require(isinstance(peaks, np.ndarray) and peaks.ndim == 2
                and peaks.dtype.names == PEAK_NAMES and peaks.size <= 32768,
                "Unexpected bounded original peak ABI")
        rows = []
        for cell, values in enumerate(peaks):
            for rank, record in enumerate(values):
                x, y = int(record["x"]), int(record["y"])
                if x < 0:
                    continue
                require(0 <= x < shape[1] and 0 <= y < shape[0], "Out-of-bounds original peak")
                row = dict(cell_index=cell, rank=rank, x=x, y=y,
                           polarity="bright" if cell % 2 == 0 else "dark",
                           score=float(record["score"]), response_dn=float(record["response"]),
                           noise_sigma_dn=float(record["noise"]))
                require(all(math.isfinite(row[k]) for k in ("score", "response_dn", "noise_sigma_dn"))
                        and row["noise_sigma_dn"] > 0, "Invalid original peak value")
                rows.append(row)
        return rows

    def _native_finish(self, instance, original, args):
        active = self._active
        require(active is not None and self.instance is instance, "Native finish outside an update")
        require(active["native_finishes"] == 0, "Repeated original native finish")
        require(len(args) == 10 and _identity(args[0]) == _identity(instance.front),
                "Unexpected original finish signature/handle")
        active["native_finishes"] += 1
        self._finishes += 1
        if not active["selected"]:
            return original(*args)
        require(getattr(instance, "busy", False), "Native finish is not inside original update")
        active["finish_parameters"] = dict(background_alpha=float(np.float32(args[5])),
            pixel_noise_alpha=float(np.float32(args[6])), pixel_noise_clip_sigma_squared=float(np.float32(args[7])),
            noise_floor_squared=float(np.float32(args[8])), variance_only=bool(args[9]),
            learning_point_count=int(args[2]), learning_offset_count=int(args[4]))
        before = self._read_native(instance)
        require(np.array_equal(before["phase_mask"], instance.eligible),
                "Pre-finish phase mask is not original eligibility")
        peaks = self._peak_records(instance.peaks, active["shape"])
        seeds = np.ascontiguousarray([p["reference_xy"] for p in active["points"]], dtype=np.int32)
        patches = np.empty((len(seeds), PATCH_SIZE, PATCH_SIZE), np.float32)
        code = instance.lib.original.seaqr_resident_patches(instance.handle, seeds.ctypes.data,
                                                          len(seeds), patches.ctypes.data)
        require(code == 0, "Bounded spatial scratch gather failed")
        require(np.isfinite(patches).all(), "Nonfinite actual spatial patch")
        active["pre_tile_statistics"] = descriptor(before["stats"])
        active["pre_tile_sigmas_float64"] = descriptor(before["sigmas"])
        active["raw_peaks"] = peaks
        tile = instance.config.tile_size
        require(type(tile) is int and 1 <= tile <= 256, "Unexpected tile size")
        nx = (active["shape"][1] + tile - 1) // tile
        for index, point in enumerate(active["points"]):
            x, y = point["reference_xy"]
            tile_index = (y // tile) * nx + x // tile
            require(0 <= tile_index < len(before["stats"]), "Point tile absent")
            temporal = np.subtract(patches[index], self._patch(before["background"], point), dtype=np.float32)
            point.update(tile_index=tile_index, spatial=descriptor(patches[index]),
                         temporal_pre_learning=descriptor(temporal))
            for field in ("background", "variance", "support", "phase_mask"):
                key = "eligible_pre_finish" if field == "phase_mask" else field + "_pre_finish"
                point[key] = descriptor(self._patch(before[field], point))
            variance = np.float32(before["variance"][y, x])
            require(np.isfinite(variance) and variance >= 0, "Invalid pre-learning variance")
            center, tile_sigma = before["stats"][tile_index]
            pixel_sigma = np.sqrt(variance, dtype=np.float32)
            effective = np.maximum(tile_sigma, pixel_sigma)
            response = temporal[PATCH_RADIUS, PATCH_RADIUS]
            require(np.isfinite(effective) and effective > 0, "Invalid effective detector sigma")
            matches = [dict(p) for p in peaks if p["x"] == x and p["y"] == y]
            for peak in matches:
                sign = np.float32(1 if peak["polarity"] == "bright" else -1)
                numerator = np.multiply(sign, np.subtract(response, center, dtype=np.float32), dtype=np.float32)
                score = np.divide(numerator, effective, dtype=np.float32)
                require(_f32_bits(peak["response_dn"]) == _f32_bits(response)
                        and _f32_bits(peak["noise_sigma_dn"]) == _f32_bits(effective)
                        and _f32_bits(peak["score"]) == _f32_bits(score),
                        "Captured original peak score/response/noise does not reconstruct exactly")
            point["center_values"] = dict(spatial_dn=float(patches[index, PATCH_RADIUS, PATCH_RADIUS]),
                temporal_dn=float(response), background_dn=float(before["background"][y, x]),
                variance_dn_squared=float(variance), tile_center_dn=float(center),
                tile_sigma_float32_dn=float(tile_sigma), tile_sigma_float64_dn=float(before["sigmas"][tile_index]),
                pixel_sigma_dn=float(pixel_sigma), effective_sigma_dn=float(effective),
                tile_floor_or_variance_comparison="tile" if tile_sigma > pixel_sigma else
                    "pixel_variance" if tile_sigma < pixel_sigma else "equal", raw_peak_matches=matches,
                exact_peak_reconstruction_count=len(matches))
        # The original function receives exactly the original argument objects once.
        result = original(*args)
        if result != 0:
            return result
        after = self._read_native(instance)
        require(np.array_equal(before["support"], after["support"])
                and before["stats"].tobytes() == after["stats"].tobytes()
                and before["sigmas"].tobytes() == after["sigmas"].tobytes(),
                "Finish unexpectedly changed support or tile statistics")
        for point in active["points"]:
            for field in ("background", "variance", "support", "phase_mask"):
                key = "learning_mask_post_finish" if field == "phase_mask" else field + "_post_finish"
                point[key] = descriptor(self._patch(after[field], point))
            x, y = point["reference_xy"]
            point["center_values"].update(background_after_dn=float(after["background"][y, x]),
                variance_after_dn_squared=float(after["variance"][y, x]),
                supported=bool(after["support"][y, x]), eligible=bool(before["phase_mask"][y, x]),
                learn_variance=bool(after["phase_mask"][y, x]),
                variance_protected=bool(after["support"][y, x] and not after["phase_mask"][y, x]))
        active["native_capture_complete"] = True
        return result

    def _end(self, instance, image, valid, learning_centers, result):
        active = self._active
        require(active is not None and active["native_finishes"] == 1,
                "Original update did not finish exactly once")
        counters = self._counters(instance)
        old = active["before_counters"]
        require(counters["calls"] == old["calls"] + 1
                and counters["finish_calls"] == old["finish_calls"] + 1
                and counters["device_calls"] == old["device_calls"] + 1
                and counters["host_calls"] == old["host_calls"], "Original front lifecycle/call count changed")
        require(not instance.busy and getattr(image, "consumed", False),
                "Original front failed to become idle/consume its frame")
        if active["selected"]:
            require(active.get("native_capture_complete") is True, "Incomplete selected-frame capture")
            require(valid.tobytes() == active["valid_bytes"] and learning_centers == active["learning"],
                    "Production validity or learning input was mutated")
            require(isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list)
                    and len(result[0]) <= 512, "Unexpected original detector return contract")
            for point in active["points"]:
                point["consolidated_candidates_at_original_peak"] = [copy.deepcopy(p) for p in result[0]
                    if p.get("shape", {}).get("peak_reference_xy", [p["x"], p["y"]]) == point["reference_xy"]]
            self._records.append(dict(frame_index=active["frame"], timestamp_ns=active["frame"] * 100000000,
                timestamp_basis="frame index / nominal 10 Hz; caller verifies archived cadence",
                segment=active["segment"],
                shape_hw=list(active["shape"]), source_pixels_captured=False,
                original_device_warp_downloaded=True, points=active["points"],
                pre_tile_statistics=active["pre_tile_statistics"],
                pre_tile_sigmas_float64=active["pre_tile_sigmas_float64"], raw_peaks=active["raw_peaks"],
                finish_parameters=active["finish_parameters"],
                original_output_candidate_count=len(result[0]), original_coverage=copy.deepcopy(result[1]),
                lifecycle_before=old, lifecycle_after=counters))
        self._updates += 1

    def finish(self):
        require(not self._sealed and self._error is None and self._active is None,
                "Capture incomplete, failed, active, or already delivered")
        require(self.instance is not None and self._updates == self._finishes == TOTAL_FRAMES,
                "A complete original 673-frame replay is required")
        require([r["frame_index"] for r in self._records] == sorted(CAPTURE_FRAMES)
                and sum(len(r["points"]) for r in self._records) == self.expected_points,
                "Missing or duplicate planned frame/point evidence")
        self._sealed = True
        return dict(schema=SCHEMA, processed_frames=self._updates, native_finish_calls=self._finishes,
            capture_frames=sorted(CAPTURE_FRAMES), point_snapshots=self.expected_points, front_instances=1,
            original_update_results_returned_unchanged=True, busy_flag_overridden=False,
            diagnostic_device_state_set=False, extra_spatial_gather_mutates_scratch_only=True,
            full_journal_parity_verified_by_this_module=False, production_performance_claim=False,
            airborne_class_verified=False, records=copy.deepcopy(self._records),
            limitations=["Caller must independently verify the complete non-timing journal and feedback parity.",
                "Native source coordinates are predeclared metadata; this module captures actual warped pixels, not source pixels.",
                "Pre-finish phase masks mean eligibility; post-finish masks mean variance learning permission.",
                "Score reconstruction uses original peak coordinates, not a consolidated centroid.",
                "Extra copies/synchronization and scratch gathers make diagnostic timings unsuitable for pipeline FPS.",
                "Sample/comparator roles are diagnostic selection roles, not target/negative class labels."])
