#!/usr/bin/env python3
"""Passive, full-causal forensic replay of the frozen failed selection candidate.

This is not a new algorithm, a speed benchmark, or a fresh pair-local replay.
Only accepted CPU correspondence arrays and the original single fit are copied.
No pre-filter/status-rejected coordinates or extra VPI readbacks are captured.
Every non-timing journal field must match the original before interpretation.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

SCHEMA = "seaqr.discovery-feature-residual-trace.v1"
TRACE_SCHEMA = "seaqr.feature-residual-trace.v1.trace"
PARITY_SCHEMA = "seaqr.feature-residual-trace.v1.parity"
WORKSPACE_PATTERN = r"/tmp/seaqr_feature_residual_trace_20260930_[A-Za-z0-9]{6}"
ORIGINAL_WORKSPACE = Path("/tmp/seaqr_feature_selection_20260929_q4iI5B")
ORIGINAL_FREEZE_SHA = "9ae62028ca25fe063241e62697cb3578a708078a1b03969061b4736f0c9c6b7e"
SELECTION_RUNNER_SHA = "db5a92d69fb7fc503a3ef2d56236460d1bb32656111ad3fd170203b9ab53bc4d"
PLAN_NAME = "feature_residual_trace_plan.json"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=0.5,
                 feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
                 grid_rows=6, grid_cols=8)
SOURCE_HASHES = {"0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
                 "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"}
FAILED = {"0170": [5, 35, 118, 161, 165, 170, 322, 452, 455, 475, 478, 481, 502, 527, 601, 614, 641],
          "0240": [344]}
TIMING_PATHS = (("timings_ms",), ("motion", "pva_timings_ms"),
                ("motion", "motion_fit", "timing_ms"), ("motion", "warp_timings_ms"),
                ("coverage", "detection_ms"))
ARTIFACT_PATHS = dict(execution_receipt="execution_receipt.json", preflight="preflight.json",
                      journal="run/frames.jsonl", report="run/report.json", launch="run/launch.json")
ARRAY_NAMES = ("previous_points", "current_points", "harris_scores", "forward_backward_error_px")
LIMITATIONS = ["Only returned accepted correspondence arrays are captured; LK/status/FB-rejected coordinates are absent.",
              "Original aggregate rejection counters are preserved, not expanded into rejected-point evidence.",
              "Array indices are pair-local, not persistent feature identities.",
              "Fit residual metrics are inlier-only; saved residual arrays also contain outliers.",
              "Native CPU gray hashes are not hashes of the CUDA/PVA half-scale image.",
              "Diagnostic copying and hashing add overhead; these timings are not a performance benchmark."]


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
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024 * 1024,
            "missing/linked/oversize metadata: " + str(path))
    with path.open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_spec(clip):
    require(clip in SOURCE_HASHES, "unapproved source")
    return dict(path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
                sha256=SOURCE_HASHES[clip], frames=673, width=4784, height=3190,
                fps=10, codec="mjpeg", pixel_format="yuvj420p")


def fixed_groups(clip):
    source_spec(clip)
    return dict(failed=FAILED[clip], adjacent=sorted({j for i in FAILED[clip] for j in (i - 1, i + 1)}),
                temporal=[100, 200, 300, 400, 500, 600], positive=list(range(430, 465)) if clip == "0240" else [])


def fixed_pairs(clip):
    return sorted(set(itertools.chain.from_iterable(fixed_groups(clip).values())))


def workspace_guard(workspace, clip, mode):
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside residual-trace workspace")
    workspace = Path(workspace)
    source_spec(clip)
    require(mode in {"preflight", "run"} and os.geteuid() != 0, "invalid mode/root execution")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked workspace")
    directory = workspace / clip
    require(not directory.is_symlink() and (not directory.exists() or directory.is_dir()), "invalid clip directory")
    names = ("run", "execution_receipt.json", "trace.json", "parity.json")
    if mode == "preflight":
        names += ("preflight.json",)
    for name in names:
        path = directory / name
        require(not path.exists() and not path.is_symlink(), "existing output; no overwrite")
    return workspace


def validate_plan(plan):
    require(plan.get("schema") == "seaqr.feature-residual-trace.plan.v1"
            and plan.get("original_workspace") == str(ORIGINAL_WORKSPACE)
            and plan.get("original_freeze_sha256") == ORIGINAL_FREEZE_SHA
            and plan.get("candidate") == CANDIDATE
            and plan.get("sources") == {c: source_spec(c) for c in SOURCE_HASHES}, "plan identity differs")
    require(plan.get("capture_pairs") == {c: fixed_pairs(c) for c in SOURCE_HASHES}
            and plan.get("capture_groups") == {c: fixed_groups(c) for c in SOURCE_HASHES}
            and plan.get("capture_pair_counts") == {"0170": 56, "0240": 44}, "fixed pair inventory differs")
    require(all(type(i) is int for rows in plan["capture_pairs"].values() for i in rows), "invalid pair index type")
    require(plan.get("parity", {}).get("full_frames_each") == 673
            and plan["parity"].get("ignored_exact_paths") == [list(p) for p in TIMING_PATHS], "parity policy differs")
    require(plan.get("execution") == dict(workers=1, phase_deadline_seconds=900,
            batch_deadline_seconds=3600, start_below_celsius=65, stop_at_celsius=75,
            automatic_retries=0, detached_tmux=True), "execution safety policy differs")
    artifacts = plan.get("original_artifacts", {})
    require(set(artifacts) == set(SOURCE_HASHES), "missing original artifacts")
    require(all(set(row) == set(ARTIFACT_PATHS) and all(isinstance(d, str) and re.fullmatch(r"[a-f0-9]{64}", d)
            for d in row.values()) for row in artifacts.values()), "invalid original artifact hashes")


def validate_freeze(value, hashes):
    require(value.get("schema") == "feature_residual_trace.v1"
            and value.get("candidate") == CANDIDATE
            and value.get("sources") == {c: source_spec(c) for c in SOURCE_HASHES}
            and value.get("original_workspace") == str(ORIGINAL_WORKSPACE)
            and value.get("original_freeze_sha256") == ORIGINAL_FREEZE_SHA, "diagnostic freeze differs")
    files = value.get("files", {})
    require(isinstance(files, dict) and {"run_feature_residual_trace.py", PLAN_NAME} <= set(files), "incomplete trace bundle")
    require(all(isinstance(n, str) and Path(n).name == n and n not in {"", ".", "..", "freeze.json"}
            and isinstance(d, str) and re.fullmatch(r"[a-f0-9]{64}", d) for n, d in files.items()), "unsafe frozen file names/hashes")
    require(files == hashes, "transferred diagnostic artifacts differ")


def load_selection():
    path = ORIGINAL_WORKSPACE / "run_discovery_feature_selection.py"
    require(path.is_file() and not path.is_symlink() and sha(path) == SELECTION_RUNNER_SHA,
            "original candidate runner changed")
    spec = importlib.util.spec_from_file_location("residual_trace_frozen_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.CANDIDATE == CANDIDATE, "original candidate contract differs")
    return module


def inputs(workspace, clip, selection, baseline):
    freeze = read(workspace / "freeze.json")
    validate_freeze(freeze, freeze.get("files", {}))
    hashes = {}
    for name in freeze["files"]:
        path = workspace / name
        require(path.is_file() and not path.is_symlink(), "missing/linked trace bundle member")
        hashes[name] = sha(path)
    validate_freeze(freeze, hashes)
    require(Path(__file__).name == "run_feature_residual_trace.py"
            and sha(Path(__file__)) == hashes["run_feature_residual_trace.py"], "executed diagnostic runner differs")
    plan = read(workspace / PLAN_NAME)
    validate_plan(plan)
    require((ORIGINAL_WORKSPACE / "freeze.json").is_file()
            and not (ORIGINAL_WORKSPACE / "freeze.json").is_symlink()
            and sha(ORIGINAL_WORKSPACE / "freeze.json") == ORIGINAL_FREEZE_SHA, "original candidate freeze differs")
    require(sha(ORIGINAL_WORKSPACE / "run_discovery_feature_selection.py") == SELECTION_RUNNER_SHA,
            "original candidate runner changed")
    reference, runtime, selection_hashes = selection.inputs(ORIGINAL_WORKSPACE, clip, baseline)
    for c in SOURCE_HASHES:
        for key, relative in ARTIFACT_PATHS.items():
            path = ORIGINAL_WORKSPACE / c / relative
            require(path.is_file() and not path.is_symlink() and sha(path) == plan["original_artifacts"][c][key],
                    "original candidate artifact changed: " + c + "/" + key)
        receipt = read(ORIGINAL_WORKSPACE / c / "execution_receipt.json")
        require(receipt.get("schema") == selection.SCHEMA and receipt.get("passed") is True
                and receipt.get("processed_frames") == receipt.get("decoded_frames_verified") == 673
                and receipt.get("source") == source_spec(c) and receipt.get("candidate") == CANDIDATE
                and receipt.get("workspace") == str(ORIGINAL_WORKSPACE)
                and receipt.get("input_sha256", {}).get("freeze_sha256") == ORIGINAL_FREEZE_SHA,
                "original candidate did not complete under matching freeze")
        require(all(receipt.get(key + "_sha256") == plan["original_artifacts"][c][key]
                    for key in ("preflight", "journal", "report", "launch")), "original receipt artifact binding differs")
    return reference, runtime, dict(files=hashes, freeze_sha256=sha(workspace / "freeze.json"),
        plan_sha256=hashes[PLAN_NAME], original_workspace=str(ORIGINAL_WORKSPACE),
        original_freeze_sha256=ORIGINAL_FREEZE_SHA, original_runner_sha256=SELECTION_RUNNER_SHA,
        original_artifacts=plan["original_artifacts"], selection_inputs=selection_hashes), plan


def finite_json(value):
    """Non-finite scalar metadata is explicit null; array descriptors retain every bit."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def array_descriptor(array):
    import numpy as np
    array = np.asarray(array)
    require(array.dtype.kind in "buif" and array.dtype.hasobject is False, "unexpected captured array dtype")
    raw = array.tobytes(order="C")
    return dict(dtype=array.dtype.str, shape=list(array.shape), data_base64=base64.b64encode(raw).decode("ascii"),
                sha256=hashlib.sha256(raw).hexdigest())


def snapshot(array):
    import numpy as np
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def fingerprints(correspondence):
    return tuple((getattr(correspondence, n).dtype.str, getattr(correspondence, n).shape,
                  getattr(correspondence, n).tobytes(order="C")) for n in ARRAY_NAMES)


class TraceCapture:
    """Bounded CPU snapshots; never calls an estimator, fit, or VPI API itself."""

    def __init__(self, clip, pairs, global_configuration, *, full_size=(4784, 3190), total_frames=673):
        require(pairs == sorted(set(pairs)) and all(type(i) is int and 1 <= i < total_frames for i in pairs),
                "invalid capture pair inventory")
        self.clip, self.pairs = clip, list(pairs)
        self.wanted = set(pairs)
        self.global_configuration = copy.deepcopy(global_configuration)
        self.full_size, self.total_frames = tuple(full_size), total_frames
        self.pending, self.rows, self.fit_indices = {}, {}, []
        self.successful_estimates = 0

    def estimator_class(self, candidate):
        owner = self

        class PassiveEstimator(candidate):
            def estimate(self, previous, current):
                result = super().estimate(previous, current)
                owner.capture_source(result, previous, current)
                return result

        return PassiveEstimator

    def capture_source(self, correspondence, previous, current):
        self.successful_estimates += 1
        frame = correspondence.current_frame_index
        if frame not in self.wanted:
            return
        require(frame not in self.pending and frame not in self.rows, "duplicate source pair capture")
        require(correspondence.previous_frame_index == frame - 1 == previous.frame_index
                and current.frame_index == frame
                and correspondence.previous_timestamp_ns == previous.timestamp_ns == (frame - 1) * 100_000_000
                and correspondence.current_timestamp_ns == current.timestamp_ns == frame * 100_000_000,
                "non-adjacent capture or changed timestamps")
        require(previous.shape == current.shape == self.full_size[::-1]
                and previous.bit_depth == current.bit_depth == 8
                and str(previous.image.dtype) == str(current.image.dtype) == "uint8", "capture is not native gray8")
        self.pending[frame] = (correspondence,
            dict(previous=previous.pixel_sha256(), current=current.pixel_sha256()))

    def wrap_fit(self, original):
        owner = self

        def captured_fit(correspondence, config=None):
            frame = correspondence.current_frame_index
            require(frame == len(owner.fit_indices) + 1 and correspondence.previous_frame_index == frame - 1,
                    "fit path is not full-causal adjacent sequence")
            require(asdict(config) == owner.global_configuration, "fit configuration changed")
            before = fingerprints(correspondence)
            result = original(correspondence, config)  # Exactly one original fit, identical arguments.
            require(fingerprints(correspondence) == before, "original fit mutated correspondence arrays")
            owner.fit_indices.append(frame)
            require(result.previous_frame_index == frame - 1 and result.current_frame_index == frame,
                    "fit result frame identity differs")
            if frame in owner.wanted:
                require(frame in owner.pending and frame not in owner.rows, "missing/duplicate source capture")
                pending, pixels = owner.pending.pop(frame)
                require(pending is correspondence, "fit did not receive original estimator return object")
                require(tuple(correspondence.full_image_size) == owner.full_size
                        and tuple(correspondence.motion_image_size) == tuple(round(s * .5) for s in owner.full_size)
                        and 0 < correspondence.count <= 384
                        and result.inlier_mask.shape == result.residuals_px.shape == (correspondence.count,),
                        "invalid native accepted correspondence/fit dimensions")
                owner.rows[frame] = dict(previous_frame_index=frame - 1, current_frame_index=frame,
                    previous_timestamp_ns=correspondence.previous_timestamp_ns,
                    current_timestamp_ns=correspondence.current_timestamp_ns,
                    full_image_size=list(correspondence.full_image_size), motion_image_size=list(correspondence.motion_image_size),
                    native_gray_pixel_sha256=pixels,
                    correspondence=dict(**{n: snapshot(getattr(correspondence, n)) for n in ARRAY_NAMES},
                        metrics=copy.deepcopy(correspondence.metrics), backends=copy.deepcopy(correspondence.backends)),
                    fit=dict(model=result.model, quality_status=result.quality_status,
                        rejection_reasons=list(result.rejection_reasons), parameters=copy.deepcopy(result.parameters),
                        metrics=copy.deepcopy(result.metrics),
                        previous_to_current_matrix=None if result.previous_to_current_matrix is None else snapshot(result.previous_to_current_matrix),
                        inlier_mask=snapshot(result.inlier_mask), residuals_px=snapshot(result.residuals_px)))
            return result  # Same object: no synthetic fit, no reconstructed estimate.

        return captured_fit

    def finish(self):
        require(self.fit_indices == list(range(1, self.total_frames))
                and self.successful_estimates == self.total_frames - 1,
                "incomplete original full-causal estimator/fit path")
        require(not self.pending and sorted(self.rows) == self.pairs, "incomplete trace pair inventory")
        rows = []
        for frame in self.pairs:
            row = copy.deepcopy(self.rows[frame])
            for name in ARRAY_NAMES:
                row["correspondence"][name] = array_descriptor(row["correspondence"][name])
            for name in ("previous_to_current_matrix", "inlier_mask", "residuals_px"):
                if row["fit"][name] is not None:
                    row["fit"][name] = array_descriptor(row["fit"][name])
            rows.append(finite_json(row))
        return rows


@contextmanager
def passive_fit_capture(capture):
    import tiny_target.motion as facade
    from tiny_target.motion.global_motion import fit_global_motion as implementation
    require(facade.fit_global_motion is implementation, "fit facade was already replaced")
    with patch.object(facade, "fit_global_motion", capture.wrap_fit(implementation)):
        yield
    require(facade.fit_global_motion is implementation, "fit facade not restored")


def non_timing(row):
    result = copy.deepcopy(row)
    for path in TIMING_PATHS:
        node = result
        for key in path[:-1]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def journal_parity(original, diagnostic, expected_frames=673):
    """Stream all rows; no numeric tolerance, dropped frames, or broad timing filter."""
    mismatch, count = [], 0
    old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
    with Path(original).open("rb") as old, Path(diagnostic).open("rb") as new:
        for index, (left, right) in enumerate(itertools.zip_longest(old, new)):
            if left is not None:
                old_hash.update(left)
            if right is not None:
                new_hash.update(right)
            same = left is not None and right is not None
            if same:
                a, b = json.loads(left), json.loads(right)
                same = (index < expected_frames and a.get("frame_index") == b.get("frame_index") == index
                    and type(a.get("frame_index")) is type(b.get("frame_index")) is int
                    and canonical(non_timing(a)) == canonical(non_timing(b)))
            if not same and len(mismatch) < 20:
                mismatch.append(index)
            count += 1
    return dict(schema=PARITY_SCHEMA, passed=count == expected_frames and not mismatch,
                rows_compared=count, expected_frames=expected_frames, original_journal_sha256=old_hash.hexdigest(),
                diagnostic_journal_sha256=new_hash.hexdigest(), excluded_paths=[list(p) for p in TIMING_PATHS],
                mismatch_frames=mismatch, mismatches_limited_to_first=20, numeric_tolerance=False)


def generated_capture_check():
    """Exercise real fit facade on generated arrays only, never media or VPI."""
    import numpy as np
    from tiny_target.types import Frame, TimestampSource
    from tiny_target.motion import GlobalMotionConfig, MotionCorrespondences, fit_global_motion
    p = np.array([[x, y] for y in (8, 16, 24, 32, 40, 48) for x in (8, 16, 24, 32, 40, 48, 56, 60)], dtype=np.float32)
    corr = MotionCorrespondences(p, p + np.array([1, 0], np.float32), np.arange(len(p), dtype=np.float32),
        np.zeros(len(p), np.float32), 0, 1, 0, 100_000_000, (64, 64), (32, 32), {}, {}, {})
    cfg = GlobalMotionConfig()
    capture = TraceCapture("generated", [1], asdict(cfg), full_size=(64, 64), total_frames=2)
    pixels = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
    previous = Frame(pixels, 0, 0, "generated", 8, TimestampSource.CONTAINER_RATE)
    current = Frame(pixels, 100_000_000, 1, "generated", 8, TimestampSource.CONTAINER_RATE)
    capture.capture_source(corr, previous, current)
    calls, returned = [], []

    def once(c, config):
        calls.append(c)
        value = fit_global_motion(c, config)
        returned.append(value)
        return value

    result = capture.wrap_fit(once)(corr, cfg)
    rows = capture.finish()
    require(calls == [corr] and result is returned[0] and len(rows) == 1, "generated passive capture identity failed")
    canonical(rows)
    return dict(passed=True, generated_only=True, original_fit_calls=1, same_return_object=True,
                input_arrays_unchanged=True, exact_array_byte_serialization=True, native_gray_hashes=True)


def validate_preflight(pre, hashes, identities, workspace, clip, selection):
    require(pre.get("schema") == SCHEMA + ".preflight", "wrong diagnostic preflight schema")
    adapted = dict(pre, schema=selection.SCHEMA + ".preflight")
    selection.validate_preflight(adapted, hashes, identities, workspace, clip)
    require(pre.get("passive_capture_check") == dict(passed=True, generated_only=True, original_fit_calls=1,
            same_return_object=True, input_arrays_unchanged=True, exact_array_byte_serialization=True,
            native_gray_hashes=True), "generated passive capture preflight incomplete")


def preflight(workspace, clip):
    workspace = workspace_guard(workspace, clip, "preflight")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA + ".preflight", passed=False, workspace=str(workspace), clip=clip,
        source=source_spec(clip), candidate=dict(CANDIDATE), detector_run=False, controls=[], full_pixel_predecode=False)
    try:
        selection = load_selection()
        baseline = selection.load_baseline()
        reference, runtime, hashes, plan = inputs(workspace, clip, selection, baseline)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        from tiny_target.frame_source import probe_video
        import cv2
        probe = probe_video(source_spec(clip)["path"])
        baseline.validate_probe(probe)
        motion, global_config = selection.configurations(helper)
        receipt.update(input_sha256=hashes, runtime_before=before, clock_policy=clocks,
                       probe=probe.to_dict(), probe_passed=True, **identities)
        cv2.setNumThreads(2)
        receipt["cpu_parity"] = selection.generated_cpu_parity(motion)
        receipt["conversion"] = selection.conversion_check()
        receipt["feature_adapter"] = {}
        with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            selection.run_controls(candidate, motion, global_config, receipt["controls"])
        receipt["passive_capture_check"] = generated_capture_check()
        after = info()
        helper.runtime_check(after, runtime, after=True)
        require(helper.clock_policy_snapshot() == clocks, "clock controls changed in preflight")
        require(inputs(workspace, clip, selection, baseline)[2] == hashes
                and helper.dependencies(reference)[1] == identities, "inputs/dependencies changed")
        receipt.update(passed=True, runtime_after=after, global_configuration=asdict(global_config),
                       clocks_changed=False, generated_pair_calls=3)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "preflight.json", receipt)
    return receipt


def run(workspace, clip):
    workspace = workspace_guard(workspace, clip, "run")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA, passed=False, error=None, processed_frames=0,
        workspace=str(workspace), clip=clip, source=source_spec(clip), candidate=dict(CANDIDATE),
        algorithm_changed=True, feature_algorithm_changed=True,
        algorithm_change_reference="Older pre-selection baseline only; unchanged relative to original q4iI5B candidate.",
        candidate_algorithm_changed_relative_to_original=False, full_causal_history=True,
        detector_configuration_changed=False, tracker_configuration_changed=False, global_motion_gates_changed=False,
        estimator_method_source_changed=False, extra_vpi_readbacks=False, prefilter_points_captured=False,
        harris_score_precision_changed=False, production_promotion=False, annotations_supplied_to_detector=False,
        raw16_accessed=False, sealed_holdouts_accessed=False, airborne_class_verified=False,
        clocks_changed=None, remote_clocks_unchanged=None,
        timestamp_basis="consumer frame index / nominal10Hz container rate; physical acquisition cadence unverified",
        interpretation="Accepted-correspondence forensic trace is interpretable only after exact full-journal non-timing parity.",
        launch_config_semantics="Pinned original selection adapter supplies the same declared in-memory gain16/capacity/384x8/exact-batching overrides; source files and all gates remain fixed.",
        timing_instrumentation="Passive CPU array copies and captured-pair native gray hashing. No extra VPI readbacks. Diagnostic timings are not a performance result.")
    capture = None
    try:
        selection = load_selection()
        baseline = selection.load_baseline()
        reference, runtime, hashes, plan = inputs(workspace, clip, selection, baseline)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        pre = read(directory / "preflight.json")
        validate_preflight(pre, hashes, identities, workspace, clip, selection)
        both_preflight = {}
        for c in SOURCE_HASHES:
            bound_hashes = hashes if c == clip else inputs(workspace, c, selection, baseline)[2]
            other = read(workspace / c / "preflight.json")
            validate_preflight(other, bound_hashes, identities, workspace, c, selection)
            both_preflight[c] = sha(workspace / c / "preflight.json")
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        require(clocks == pre.get("clock_policy"), "clock policy differs from preflight")
        motion, global_config = selection.configurations(helper)
        receipt.update(input_sha256=hashes, preflight_sha256=both_preflight[clip],
            both_preflight_sha256=both_preflight, runtime_before=before, clock_policy_before=clocks,
            feature_adapter={}, global_configuration=asdict(global_config),
            tracking_transformed_sha256=helper.TRACKING_METHOD_SHA, **identities)
        capture = TraceCapture(clip, plan["capture_pairs"][clip], asdict(global_config))
        with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            changed_modules = dict(modules, motion_reuse_v12=SimpleNamespace(ReuseMotionV12=capture.estimator_class(candidate)))
            with passive_fit_capture(capture):
                report = baseline.execute_baseline(helper, changed_modules, Path(source_spec(clip)["path"]),
                                                   directory / "run", receipt)
        baseline.validate_output(report, read(directory / "run/launch.json"), reference, clip,
                                 receipt["decoded_frames_verified"])
        require(receipt["feature_adapter"]["estimator_instances"] == 1
                and receipt["feature_adapter"]["successful_pair_backend_checks"] == 672
                and all(row["error"] is None for row in receipt["motion_attempts"]), "original successful pair path differs")
        selection.validate_adapter_roundtrip(receipt["feature_adapter"], pre["feature_adapter"])
        after, clock_after = info(), helper.clock_policy_snapshot()
        helper.runtime_check(after, runtime, after=True)
        receipt.update(runtime_after=after, clock_policy_after=clock_after,
                       clocks_changed=clock_after != clocks, remote_clocks_unchanged=clock_after == clocks)
        require(clock_after == clocks, "clock controls changed")
        require(inputs(workspace, clip, selection, baseline)[2] == hashes
                and helper.dependencies(reference)[1] == identities, "frozen inputs/dependencies changed")
        require(all(sha(workspace / c / "preflight.json") == digest for c, digest in both_preflight.items()),
                "preflight evidence changed during run")
        trace = dict(schema=TRACE_SCHEMA, workspace=str(workspace), clip=clip, source=source_spec(clip),
            candidate=dict(CANDIDATE), input_sha256=hashes, capture_pairs=plan["capture_pairs"][clip],
            capture_groups=plan["capture_groups"][clip], limitations=LIMITATIONS,
            full_causal_history=True, global_configuration=asdict(global_config),
            captured_pairs=len(capture.rows), original_fit_calls=len(capture.fit_indices),
            rows=capture.finish())
        write(directory / "trace.json", trace)
        parity = journal_parity(ORIGINAL_WORKSPACE / clip / "run/frames.jsonl", directory / "run/frames.jsonl")
        write(directory / "parity.json", parity)
        receipt.update(trace_sha256=sha(directory / "trace.json"), parity_sha256=sha(directory / "parity.json"),
            captured_pairs=len(capture.rows), original_fit_calls=len(capture.fit_indices),
            journal_sha256=sha(directory / "run/frames.jsonl"), report_sha256=sha(directory / "run/report.json"),
            launch_sha256=sha(directory / "run/launch.json"), availability=report["availability"],
            detection_status=report["detection_status"], non_timing_journal_parity_passed=parity["passed"])
        require(parity["original_journal_sha256"] == plan["original_artifacts"][clip]["journal"]
                and parity["diagnostic_journal_sha256"] == receipt["journal_sha256"], "journal hash binding differs")
        require(parity["passed"] is True, "full-journal non-timing parity failed; trace must not be interpreted as original")
        receipt.update(passed=True, processed_frames=report["frames"])
    except BaseException as exc:
        receipt["error"] = repr(exc)
        if capture is not None:
            receipt.update(captured_pairs=len(capture.rows), original_fit_calls=len(capture.fit_indices))
        raise
    finally:
        write(directory / "execution_receipt.json", receipt)
    print(json.dumps(dict(clip=clip, passed=True, frames=receipt["processed_frames"],
                          captured_pairs=receipt["captured_pairs"], parity_passed=True)), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--clip", choices=tuple(SOURCE_HASHES), required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    (preflight if args.preflight else run)(args.workspace, args.clip)


if __name__ == "__main__":
    main()
