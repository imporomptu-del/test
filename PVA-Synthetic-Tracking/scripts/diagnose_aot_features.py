#!/usr/bin/env python3
"""Frozen, baseline-only AOT feature diagnostic: 8 real + 8 static + 5 controls.

This is not a detector run or an accuracy/speed benchmark. Fresh pair-local
estimator state differs from full-clip preparation reuse. No VPI global cache
reset, threshold change, gain/scale ablation, clock write, or CPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from unittest.mock import patch

SCHEMA = "seaqr.aot.feature-diagnostic.v1"
PLAN_SCHEMA = "seaqr.aot.feature-diagnostic-plan.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_aot_features_20260927_[A-Za-z0-9]{6}"
INPUT_WORKSPACE = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
HARNESS_SHA = "ccbe0ddca4b15b7cbb05fcc3a1b7f05dcb099e90e34c296b16147aeac842c093"
METHOD_SHA = "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733"
CURRENT_INDICES = (1, 43, 86, 128, 171, 213, 256, 299)
CONTROL_SEED = 20260927
CONTROL_NAMES = ("high_contrast_static", "high_contrast_translated",
                 "low_contrast_static", "low_contrast_translated", "flat_static")
WIDTH, HEIGHT, FRAME_COUNT, PAIR_COUNT = 2448, 2048, 300, 21
MAX_HARRIS = 8192  # Frozen legacy-default output capacity, not a new feature cap.
CONTROL_DX, CONTROL_DY, CONTROL_FILL, CONTROL_INTERIOR = 4, -2, 128, 128


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/linked file: {path}")
    require(path.stat().st_size <= 32 * 1024 * 1024, f"oversize metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def scope_path(workspace):
    text = str(workspace)
    require(re.fullmatch(WORKSPACE_PATTERN, text), "outside exact feature workspace scope")
    return Path(text)


def validate_manifest(manifest, hashes):
    expected = dict(schema=PLAN_SCHEMA, input_workspace=str(INPUT_WORKSPACE),
                    current_indices=list(CURRENT_INDICES), control_seed=CONTROL_SEED,
                    motion_pair_calls=PAIR_COUNT, ablation="none",
                    include_stationary_counterfactuals=True,
                    baseline_harness_sha256=HARNESS_SHA)
    require(all(manifest.get(key) == value for key, value in expected.items()),
            "feature plan differs from fixed baseline-only 21-pair scope")
    require(type(manifest["control_seed"]) is int and type(manifest["motion_pair_calls"]) is int
            and manifest["include_stationary_counterfactuals"] is True
            and all(type(index) is int for index in manifest["current_indices"]),
            "invalid fixed plan numeric/bool types")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        require(isinstance(hashes.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", hashes[key])
                and manifest.get(key) == hashes[key], f"feature plan identity differs: {key}")


def artifact_hashes(workspace):
    files = {"script_sha256": workspace / "diagnose_aot_features.py",
             "tests_sha256": workspace / "test_aot_feature_diagnostic.py",
             "plan_sha256": workspace / "plan.md"}
    for path in files.values():
        require(path.is_file() and not path.is_symlink(), f"missing/linked artifact: {path}")
    require(Path(__file__).resolve() == files["script_sha256"], "script must run from guarded workspace")
    return {key: sha(path) for key, path in files.items()}


def workspace_guard(workspace):
    workspace = scope_path(workspace)
    require(os.geteuid() != 0, "never run diagnostic as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked feature workspace")
    for name in ("result.json", "failure.json"):
        require(not (workspace / name).exists() and not (workspace / name).is_symlink(),
                "existing result/failure; refusing overwrite")
    require(INPUT_WORKSPACE.is_dir() and not INPUT_WORKSPACE.is_symlink()
            and (INPUT_WORKSPACE / "input").is_dir()
            and not (INPUT_WORKSPACE / "input").is_symlink(), "missing/linked fixed pilot workspace")
    return workspace


def load_harness():
    path = INPUT_WORKSPACE / "run_aot_frozen_baseline.py"
    require(path.is_file() and not path.is_symlink() and sha(path) == HARNESS_SHA,
            "approved baseline harness identity differs")
    spec = importlib.util.spec_from_file_location("_aot_approved_baseline_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    """JSON-safe diagnostics without changing numeric arrays used by the estimator."""
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return finite(value.tolist())
    return value


def raw_array(array):
    import numpy as np
    array = np.asarray(array)
    require(array.size <= MAX_HARRIS * 2 and array.dtype.kind in "uifb",
            "unbounded/non-numerical diagnostic point array")
    return dict(dtype=str(array.dtype), shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                values=finite(array.tolist()))


def image_summary(array):
    import numpy as np
    array = np.asarray(array)
    require(array.ndim == 2 and 0 < array.size <= WIDTH * HEIGHT
            and array.dtype in (np.dtype("uint8"), np.dtype("int16")),
            "image diagnostic requires bounded native U8/proxy U8/S16")
    gradients = {}
    signed = array.astype(np.int32)
    for name, axis in (("x", 1), ("y", 0)):
        delta = np.abs(np.diff(signed, axis=axis))
        gradients[name] = dict(count=int(delta.size), nonzero=int(np.count_nonzero(delta)),
            maximum=int(delta.max()) if delta.size else 0,
            mean_abs=float(delta.mean()) if delta.size else 0.0,
            rms=float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)))) if delta.size else 0.0)
    return dict(dtype=str(array.dtype), shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                minimum=int(array.min()), maximum=int(array.max()), mean=float(array.mean()),
                standard_deviation=float(array.std()),
                percentiles=dict(zip(("p0", "p1", "p50", "p99", "p100"),
                                     np.percentile(array, [0, 1, 50, 99, 100]).tolist())),
                gradients=gradients)


def conversion_comparison(proxy_u8, s16):
    import numpy as np
    require(proxy_u8.shape == s16.shape and proxy_u8.dtype == np.uint8 and s16.dtype == np.int16,
            "unexpected U8-to-S16 diagnostic representation")
    difference = s16.astype(np.int32) - proxy_u8.astype(np.int32)
    return dict(exact_value_equality=bool(np.all(difference == 0)),
                unequal_pixels=int(np.count_nonzero(difference)),
                maximum_absolute_error=int(np.abs(difference).max()),
                signed_difference_minimum=int(difference.min()),
                signed_difference_maximum=int(difference.max()))


# Each hook is inserted after a completed production stage, before its early
# return/error where applicable. Removing these lines recovers the exact source.
HOOKS = (
    ('    timings["intensity_conversion_cpu"] = _elapsed_ms(started)\n', "pixels", 4),
    ('        timings["motion_image_prepare"] = (completed - started) / 1_000_000\n', "proxy", 8),
    ('        timings["harris_input_conversion_cuda"] = (completed - started) / 1_000_000\n', "s16", 8),
    ('        detected_count = int(features.size)\n', "harris", 8),
    ('        timings["spatial_quota_cpu"] = _elapsed_ms(selection_started)\n', "selection", 8),
    ('        timings["flow_readback_cpu"] = _elapsed_ms(started)\n', "flow", 8),
)


def instrument_source(source):
    require(hashlib.sha256(source.encode()).hexdigest() == METHOD_SHA,
            "frozen generated estimator source identity differs")
    transformed = source
    for anchor, stage, indent in HOOKS:
        require(transformed.count(anchor) == 1, f"diagnostic source anchor differs: {stage}")
        transformed = transformed.replace(anchor, anchor + " " * indent
                                          + f'self._aot_capture.record("{stage}", locals())\n')
    recovered = transformed
    for _, stage, indent in HOOKS:
        recovered = recovered.replace(" " * indent + f'self._aot_capture.record("{stage}", locals())\n', "")
    require(recovered == source, "diagnostic instrumentation altered original estimator statements")
    return transformed


def translate_no_wrap(image, dx, dy, fill):
    import numpy as np
    require(image.ndim == 2 and image.dtype == np.uint8, "control image must be gray8")
    height, width = image.shape
    require(abs(dx) < width and abs(dy) < height and 0 <= fill <= 255, "invalid control translation")
    result = np.full_like(image, fill)
    x0, x1, y0, y1 = max(0, -dx), min(width, width - dx), max(0, -dy), min(height, height - dy)
    result[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = image[y0:y1, x0:x1]
    return result


def generated_controls():
    import numpy as np
    rng = np.random.default_rng(CONTROL_SEED)
    cells = rng.integers(0, 16, size=((HEIGHT + 31) // 32, (WIDTH + 31) // 32), dtype=np.uint8)
    pattern = cells.repeat(32, axis=0).repeat(32, axis=1)[:HEIGHT, :WIDTH]
    high, low = (16 + 14 * pattern).astype(np.uint8), (120 + pattern).astype(np.uint8)
    flat = np.full((HEIGHT, WIDTH), CONTROL_FILL, dtype=np.uint8)
    return [(CONTROL_NAMES[0], high, high, (0, 0)),
            (CONTROL_NAMES[1], high, translate_no_wrap(high, CONTROL_DX, CONTROL_DY, CONTROL_FILL),
             (CONTROL_DX, CONTROL_DY)),
            (CONTROL_NAMES[2], low, low, (0, 0)),
            (CONTROL_NAMES[3], low, translate_no_wrap(low, CONTROL_DX, CONTROL_DY, CONTROL_FILL),
             (CONTROL_DX, CONTROL_DY)),
            (CONTROL_NAMES[4], flat, flat, (0, 0))]


class Capture:
    """Read-only copies at existing synchronized stage boundaries; no writes to VPI."""

    def __init__(self):
        self.data = {}
        self.stages = []
        self.error = None
        self.previous_proxy = None

    @staticmethod
    def copy_vpi(value):
        import numpy as np
        with value.rlock_cpu() as data:
            return np.array(data, copy=True)

    def record(self, stage, state):
        import numpy as np
        try:
            require(stage not in self.stages, "duplicate diagnostic stage")
            if stage == "pixels":
                self.data[stage] = {
                    name: dict(native=image_summary(state[name].image),
                               feature=image_summary(state[name + "_pixels"]))
                    for name in ("previous", "current")}
                require(not state["uses_u16"], "native AOT diagnostic unexpectedly uses U16")
            elif stage == "proxy":
                previous = self.copy_vpi(state["previous_motion"])
                current = self.copy_vpi(state["current_motion"])
                require(previous.dtype == current.dtype == np.uint8
                        and previous.shape == current.shape == (HEIGHT // 2, WIDTH // 2),
                        "frozen half-resolution U8 proxy differs")
                self.previous_proxy = previous
                self.data[stage] = dict(previous=image_summary(previous), current=image_summary(current))
                self.data["backends_requested"] = dict(rescale=state["rescale_backend"],
                    pyramid=state["pyramid_backend_name"], conversion="CUDA", harris="PVA",
                    optical_flow=state["config"].optical_flow_backend, cpu_fallback=False)
                require(self.data["backends_requested"] == dict(rescale="CUDA", pyramid="PVA",
                    conversion="CUDA", harris="PVA", optical_flow="PVA", cpu_fallback=False),
                    "unexpected feature backend selection")
            elif stage == "s16":
                previous = self.copy_vpi(state["previous_s16"])
                self.data[stage] = dict(previous=image_summary(previous),
                    conversion=conversion_comparison(self.previous_proxy, previous))
                self.previous_proxy = None
            elif stage == "harris":
                count = state["detected_count"]
                require(0 <= count <= MAX_HARRIS and state["harris_capacity"] is None,
                        "frozen legacy Harris output capacity changed")
                # An empty VPI array need not be CPU-lockable. Never read it.
                points = self.copy_vpi(state["features"]) if count else np.empty((0, 2), np.float32)
                scores = self.copy_vpi(state["scores"]).reshape(-1) if count else np.empty(0, np.uint32)
                require(points.shape == (count, 2) and points.dtype == np.float32
                        and scores.shape == (count,) and scores.dtype == np.uint32,
                        "unexpected raw Harris coordinate/score representation")
                self.data[stage] = dict(raw_count=count, legacy_default_capacity=MAX_HARRIS,
                    capacity_saturation_observed=count == MAX_HARRIS,
                    coordinates=raw_array(points), scores=raw_array(scores),
                    score_capture="exact U32 before production float32 conversion")
            elif stage == "selection":
                indices = np.asarray(state["selected_indices"])
                self.data[stage] = dict(selected_count=len(indices),
                    eligible_mask=raw_array(state["eligible_mask"]), selected_indices=raw_array(indices),
                    coordinates=raw_array(state["detected_points"][indices]),
                    scores_float32=raw_array(state["detected_scores"][indices]),
                    exclusions=finite(state["feature_exclusions"]))
            elif stage == "flow":
                self.data[stage] = {name: raw_array(state[name]) if state[name] is not None else None
                    for name in ("current_motion_points", "forward_status_array",
                                 "backward_motion_points", "backward_status_array")}
            else:
                raise ValueError("unknown diagnostic stage")
            self.stages.append(stage)
            self.data["last_production_timings_ms"] = dict(state["timings"])
        except BaseException as exc:
            self.error = repr(exc)
            raise RuntimeError(f"diagnostic capture failed at {stage}: {exc}") from exc


def compare_counterfactuals(rows):
    """Observe (do not enforce) exact previous-frame representations/points/scores."""
    result = []
    for index in CURRENT_INDICES:
        real = [row for row in rows if row["kind"] == "aot_adjacent" and row["current_index"] == index]
        static = [row for row in rows if row["kind"] == "aot_stationary_counterfactual"
                  and row["current_index"] == index]
        require(len(real) == len(static) == 1, "missing/duplicate counterfactual comparison pair")
        fields = {}
        for label, stage, key in (("previous_proxy", "proxy", "previous"),
                                  ("previous_s16", "s16", "previous"),
                                  ("harris_coordinates", "harris", "coordinates"),
                                  ("harris_scores", "harris", "scores")):
            a, b = real[0]["capture"][stage][key], static[0]["capture"][stage][key]
            fields[label] = all(a[item] == b[item] for item in ("dtype", "shape", "sha256"))
        fields["harris_count"] = (real[0]["capture"]["harris"]["raw_count"]
                                  == static[0]["capture"]["harris"]["raw_count"])
        result.append(dict(current_index=index, equal=fields,
                           all_previous_feature_observations_equal=all(fields.values())))
    return result


def decode_selected(harness, images):
    """One sequential read verifies all 300 frames; retains only fixed pair frames."""
    from fractions import Fraction
    import numpy as np
    from tiny_target.frame_source import probe_video
    from tiny_target.visible_decode import VisibleFrameReader
    video = INPUT_WORKSPACE / "input/pilot_gray8_ffv1_10fps.avi"
    probe = probe_video(video)
    require(probe.codec == "ffv1" and probe.pixel_format in {"gray", "gray8"}
            and (probe.width, probe.height) == (WIDTH, HEIGHT)
            and probe.frame_rate == Fraction(10) and probe.declared_frame_count == FRAME_COUNT,
            "pilot video probe differs")
    wanted = {index for current in CURRENT_INDICES for index in (current - 1, current)}
    selected, count = {}, 0
    with VisibleFrameReader(video, (HEIGHT, WIDTH), "prefetch_one") as reader:
        require(reader.fps == 10 and reader.expected == FRAME_COUNT, "pilot reader cadence/count differs")
        while True:
            frame, _ = reader.read()
            if frame is None:
                break
            harness.check_frame(frame, count, images)
            if count in wanted:
                pixels = np.array(frame.gray, copy=True)
                pixels.setflags(write=False)
                selected[count] = pixels
            count += 1
    stats = reader.completed_stats()
    require(count == FRAME_COUNT and stats["decoded_frames"] == stats["consumed_frames"] == FRAME_COUNT
            and stats["dropped_frames"] == 0 and stats["maximum_observed_frames_ahead"] <= 1
            and set(selected) == wanted, "incomplete/changed pilot decode")
    return selected, dict(pixel_hashes_verified=count, retained_indices=sorted(selected),
                           probe=probe.to_dict(), stats=stats)


def control_error(correspondence, expected):
    """Post-hoc interior error only; this does not modify tracking or the fit."""
    import numpy as np
    p, q = correspondence.previous_points, correspondence.current_points
    dx, dy = expected
    overlap = (max(0, -dx), max(0, -dy), min(WIDTH, WIDTH - dx), min(HEIGHT, HEIGHT - dy))
    margin = CONTROL_INTERIOR
    mask = ((p[:, 0] >= overlap[0] + margin) & (p[:, 0] < overlap[2] - margin)
            & (p[:, 1] >= overlap[1] + margin) & (p[:, 1] < overlap[3] - margin)
            & (q[:, 0] >= margin) & (q[:, 0] < WIDTH - margin)
            & (q[:, 1] >= margin) & (q[:, 1] < HEIGHT - margin))
    error = q[mask].astype(np.float64) - p[mask].astype(np.float64) - np.asarray(expected)
    norms = np.linalg.norm(error, axis=1)
    return dict(expected_displacement_xy=list(expected), valid_previous_overlap_xyxy=list(overlap),
                additional_interior_margin_px=margin, accepted_interior_points=int(mask.sum()),
                all_accepted_points=correspondence.count, error_xy_px=raw_array(error),
                median_error_px=float(np.median(norms)) if len(norms) else None,
                maximum_error_px=float(norms.max()) if len(norms) else None,
                note="Only this post-hoc summary excludes borders. The unchanged baseline fit uses its own original accepted points.")


def evaluate_pair(reuse, motion_config, global_config, previous_pixels, current_pixels,
                  row, expected=None):
    from tiny_target.types import Frame, TimestampSource
    from tiny_target.motion import PvaMotionError, fit_global_motion
    capture, estimator = Capture(), None
    previous_index, current_index = row["previous_index"], row["current_index"]
    previous = Frame(previous_pixels, previous_index * 100_000_000, previous_index,
                     "visible-baseline", 8, TimestampSource.CONTAINER_RATE)
    current = Frame(current_pixels, current_index * 100_000_000, current_index,
                    "visible-baseline", 8, TimestampSource.CONTAINER_RATE)
    row.update(previous_metadata=previous.metadata_dict(), current_metadata=current.metadata_dict(),
               capture=capture.data, expected_unavailable=False, error=None, completed=False)
    before = [hashlib.sha256(frame.image.tobytes(order="C")).hexdigest() for frame in (previous, current)]
    row["native_pixel_sha256"] = dict(previous=before[0], current=before[1])
    try:
        estimator = reuse.ReuseMotionV12(motion_config)
        estimator._aot_capture = capture
        try:
            correspondence = estimator.estimate(previous, current)
        except PvaMotionError as exc:
            expected_error = any(text in str(exc) for text in ("zero features", "No finite in-bounds"))
            row.update(error=repr(exc), expected_unavailable=expected_error and capture.error is None,
                       outcome="motion_unavailable", global_fit=None)
            require(row["expected_unavailable"] and not estimator.failed, "unexpected backend/capture failure")
        else:
            require(correspondence.backends == dict(intensity_conversion="CPU", motion_image_rescale="CUDA",
                    gaussian_pyramid="PVA", harris_input_conversion="CUDA", harris="PVA",
                    optical_flow_pyrlk="PVA", cpu_fallback=False), "unexpected motion backend fallback")
            row.update(outcome="correspondences_returned", correspondence=finite(correspondence.to_dict()),
                       global_fit=finite(fit_global_motion(correspondence, global_config).to_dict()))
            if expected is not None:
                row["generated_control_error"] = control_error(correspondence, expected)
        require(capture.error is None and capture.stages[:4] == ["pixels", "proxy", "s16", "harris"],
                "required early-stage observations missing")
        required = (["pixels", "proxy", "s16", "harris"] if capture.data["harris"]["raw_count"] == 0
                    else ["pixels", "proxy", "s16", "harris", "selection"])
        if not row["expected_unavailable"]:
            required.append("flow")
        require(capture.stages == required, "incomplete/extra feature-stage execution")
        require(not estimator.failed and estimator.hits == 0 and estimator.misses == 1,
                "unexpected pair-local reuse state")
        require(before == [hashlib.sha256(frame.image.tobytes(order="C")).hexdigest()
                           for frame in (previous, current)], "diagnostic mutated source pixels")
        row["completed"] = True
    except BaseException as exc:
        row.update(completed=False, unexpected_error=repr(exc))
        raise
    finally:
        row.update(captured_stages=capture.stages, capture_error=capture.error)
        capture.previous_proxy = None
        if estimator is not None:
            try:
                estimator.close()
                require(estimator.closed, "pair-local motion estimator did not close")
            except BaseException as exc:
                row.update(completed=False, cleanup_error=repr(exc))
                raise
            finally:
                row["lifecycle"] = dict(closed=estimator.closed, failed=estimator.failed,
                    hits=estimator.hits, misses=estimator.misses, resets=estimator.resets)
    return row


def run(workspace):
    workspace = workspace_guard(workspace)
    manifest = read(workspace / "manifest.json")
    artifact_before = artifact_hashes(workspace)
    validate_manifest(manifest, artifact_before)
    manifest_sha = sha(workspace / "manifest.json")
    receipt = dict(schema=SCHEMA, passed=False, error=None, workspace=str(workspace),
        input_workspace=str(INPUT_WORKSPACE), manifest_sha256=manifest_sha,
        artifact_sha256_before=artifact_before, planned_motion_pair_calls=PAIR_COUNT,
        completed_motion_pair_calls=0, rows=[], current_indices=list(CURRENT_INDICES),
        control_names=list(CONTROL_NAMES), control_seed=CONTROL_SEED,
        include_stationary_counterfactuals=True, ablation="none",
        algorithm_changed=False, detector_configuration_changed=False, detector_run=False,
        annotations_used=False, raw16_accessed=False, private_camera_media_accessed=False,
        interpretation=dict(
            camera_movement="Harris uses the previous image only; interframe movement cannot explain zero raw Harris features.",
            pair_state="One fresh estimator/cache per pair, without global VPI-cache reset. Not full-clip temporal reuse parity or global-cache isolation.",
            stationary_counterfactual="Synthetic reuse of previous pixels with distinct frame indices/timestamps; not evidence of real stationary footage.",
            timing="Nominal 10 Hz container timestamps, not acquisition timing. Extra host readbacks/hashes/statistics make timings non-comparable to the baseline.",
            result="Passed means scope/integrity/execution completed, not successful motion fits or aircraft detection. Failed positive controls remain outcomes; no retuning.",
            method="Only read-only observation hooks added after synchronized stages. Original method statements and all settings preserved."))
    harness = runtime_info = cv2 = None
    original_threads = None
    # Reserve the sole result before decoding/estimating: concurrent or repeated
    # invocation cannot overwrite an experiment. A killed process leaves evidence.
    with (workspace / "result.json").open("x", encoding="utf-8") as result_file:
        try:
            harness = load_harness()
            receipt["baseline_harness_sha256_before"] = sha(INPUT_WORKSPACE / "run_aot_frozen_baseline.py")
            values, input_hashes = harness.inputs(INPUT_WORKSPACE)
            modules, identities = harness.dependencies(values["reference_launch_v34.json"])
            preflight = harness.read(INPUT_WORKSPACE / "preflight.json")
            harness.validate_preflight(preflight, input_hashes, identities, HARNESS_SHA, INPUT_WORKSPACE)
            receipt.update(input_sha256_before=input_hashes, frozen_dependencies=identities,
                           baseline_preflight_sha256=sha(INPUT_WORKSPACE / "preflight.json"))
            runtime_info = modules["profile_visible_interaction_v30"].runtime_info
            before = runtime_info()
            harness.runtime_check(before, values["reference_runtime.json"])
            clock_before = harness.clock_policy_snapshot()
            receipt.update(runtime_before=before, clock_policy_before=clock_before,
                           remote_clocks_unchanged=None, clock_writes=False)
            import cv2
            from dataclasses import asdict
            from tiny_target.config import load_config
            from tiny_target.motion import PvaMotionConfig, GlobalMotionConfig
            original_threads = cv2.getNumThreads()
            # This is the same process-local numerical policy set by visible.run.
            # It is explicitly recorded and restored, not a hardware clock control.
            cv2.setNumThreads(2)
            receipt["opencv_thread_policy"] = dict(original=original_threads, diagnostic=cv2.getNumThreads(),
                                                   restored=None, process_local_only=True)
            harness.runtime_check(runtime_info(), values["reference_runtime.json"], after=True)
            raw = load_config(harness.MOTION_CONFIG).raw
            motion_config = PvaMotionConfig.from_mapping(raw.get("motion"))
            global_config = GlobalMotionConfig.from_mapping(raw.get("global_motion"))
            require(motion_config.feature_image_scale == 0.5 and motion_config.minimum_accepted_features == 30
                    and motion_config.harris_capacity_policy == "legacy_default"
                    and motion_config.flow_status_policy == "legacy_default"
                    and global_config.minimum_correspondences == global_config.minimum_inliers == 30,
                    "baseline feature/fit settings changed")
            receipt.update(resolved_motion_configuration=asdict(motion_config),
                           resolved_global_configuration=asdict(global_config))
            reuse = modules["motion_reuse_v12"]
            original_source = reuse.generated_method()
            transformed = instrument_source(original_source)
            namespace = dict(vars(reuse.pva))
            exec(compile(transformed, str(workspace / "diagnose_aot_features.py") + ":observation-hooks", "exec"), namespace)
            receipt["source_instrumentation"] = dict(
                original_method_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                transformed_method_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                exact_original_recovered_after_removing_hooks=True,
                hooks=[stage for _, stage, _ in HOOKS], instrumented_method_source=transformed)
            frames, decoded = decode_selected(harness, values["download_validation.json"]["images"])
            receipt["decode"] = decoded
            source_rows = values["frozen_image_manifest.json"]["frames"]
            with patch.object(reuse, "_ESTIMATE", namespace["_estimate_v12"]):
                # Fixed interleaved order keeps the real/counterfactual comparison
                # adjacent while retaining all inherited process-local VPI state.
                for current in CURRENT_INDICES:
                    for stationary in (False, True):
                        row = dict(id=f"aot_{current:03d}_" + ("stationary" if stationary else "adjacent"),
                            kind="aot_stationary_counterfactual" if stationary else "aot_adjacent",
                            synthetic=stationary, previous_index=current - 1, current_index=current,
                            source_previous={key: source_rows[current - 1][key]
                                             for key in ("source_frame", "timestamp_ns", "img_name")},
                            source_current={key: source_rows[current][key]
                                            for key in ("source_frame", "timestamp_ns", "img_name")},
                            current_pixels_from_index=current - 1 if stationary else current,
                            acquisition_metadata_note="For synthetic rows current source metadata identifies the replaced frame; actual pixels come from current_pixels_from_index.")
                        receipt["rows"].append(row)
                        evaluate_pair(reuse, motion_config, global_config, frames[current - 1],
                                      frames[current - 1 if stationary else current], row)
                        receipt["completed_motion_pair_calls"] += 1
                        print(json.dumps(dict(pair=row["id"], completed=receipt["completed_motion_pair_calls"],
                            raw_harris=row["capture"]["harris"]["raw_count"], outcome=row["outcome"])), flush=True)
                for name, previous, current, expected in generated_controls():
                    row = dict(id=name, kind="generated_control", synthetic=True, previous_index=0,
                        current_index=1, generator=dict(seed=CONTROL_SEED, block_size_px=32,
                            pattern_levels=16, high_mapping="16 + 14 * pattern", low_mapping="120 + pattern",
                            translation_xy_px=list(expected), wrapping=False, border_fill=CONTROL_FILL,
                            native_resolution=[WIDTH, HEIGHT], native_dtype="uint8"))
                    receipt["rows"].append(row)
                    evaluate_pair(reuse, motion_config, global_config, previous, current, row, expected)
                    receipt["completed_motion_pair_calls"] += 1
                    print(json.dumps(dict(pair=name, completed=receipt["completed_motion_pair_calls"],
                        raw_harris=row["capture"]["harris"]["raw_count"], outcome=row["outcome"])), flush=True)
            require(receipt["completed_motion_pair_calls"] == len(receipt["rows"]) == PAIR_COUNT,
                    "incomplete/excess motion pair calls")
            receipt["counterfactual_comparisons"] = compare_counterfactuals(receipt["rows"])
            after = runtime_info()
            harness.runtime_check(after, values["reference_runtime.json"], after=True)
            receipt["runtime_after"] = after
            after_dependencies = harness.dependencies(values["reference_launch_v34.json"])[1]
            require(after_dependencies == identities, "frozen dependencies changed during diagnostic")
            receipt["frozen_dependencies_reverified"] = True
            receipt["input_sha256_after"] = harness.inputs(INPUT_WORKSPACE)[1]
            require(receipt["input_sha256_after"] == input_hashes, "fixed pilot inputs changed")
            require(sha(INPUT_WORKSPACE / "preflight.json") == receipt["baseline_preflight_sha256"],
                    "baseline preflight changed")
            receipt["baseline_harness_sha256_after"] = sha(INPUT_WORKSPACE / "run_aot_frozen_baseline.py")
            require(receipt["baseline_harness_sha256_after"] == HARNESS_SHA, "baseline harness changed")
            receipt["artifact_sha256_after"] = artifact_hashes(workspace)
            require(receipt["artifact_sha256_after"] == artifact_before
                    and sha(workspace / "manifest.json") == manifest_sha, "diagnostic artifacts/plan changed")
            receipt["clock_policy_after"] = harness.clock_policy_snapshot()
            receipt["remote_clocks_unchanged"] = receipt["clock_policy_after"] == clock_before
            require(receipt["remote_clocks_unchanged"], "clock controls changed during diagnostic")
            receipt["passed"] = True
        except BaseException as exc:
            receipt.update(error=repr(exc), passed=False)
            raise
        finally:
            try:
                if cv2 is not None and original_threads is not None:
                    cv2.setNumThreads(original_threads)
                    receipt["opencv_thread_policy"]["restored"] = cv2.getNumThreads()
                    require(cv2.getNumThreads() == original_threads, "OpenCV thread restoration failed")
                    receipt["runtime_after_thread_restore"] = runtime_info()
                    harness.runtime_check(receipt["runtime_after_thread_restore"],
                                          values["reference_runtime.json"])
                if harness is not None and "clock_policy_before" in receipt:
                    receipt["clock_policy_final"] = harness.clock_policy_snapshot()
                    require(receipt["clock_policy_final"] == receipt["clock_policy_before"],
                            "clock controls changed before diagnostic cleanup")
            except BaseException as exc:
                receipt.update(passed=False, cleanup_error=repr(exc))
            json.dump(finite(receipt), result_file, indent=2, allow_nan=False)
            result_file.write("\n")
            result_file.flush()
    require(receipt["passed"], "feature diagnostic did not pass integrity/cleanup gates")
    print(json.dumps(dict(passed=True, completed_motion_pair_calls=PAIR_COUNT,
                          result=str(workspace / "result.json"))), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    run(args.workspace)


if __name__ == "__main__":
    main()
