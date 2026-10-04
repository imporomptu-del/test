#!/usr/bin/env python3
"""Two-source human-review intake using the unchanged frozen v34 PVA/GPU stack.

No AOT labels/scoring, clock writes, native builds, fallback or source edits.
Each clip requires its own successful preflight and fresh-process full run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from unittest.mock import patch

SCHEMA = "seaqr.discovery-pair.baseline.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_discovery_pair_20260928_[A-Za-z0-9]{6}"
FRAMES, WIDTH, HEIGHT = 673, 4784, 3190
OLD_WORKSPACE = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
HELPER = OLD_WORKSPACE / "run_aot_frozen_baseline.py"
HELPER_SHA = "ccbe0ddca4b15b7cbb05fcc3a1b7f05dcb099e90e34c296b16147aeac842c093"
REFERENCE = OLD_WORKSPACE / "input/reference_launch_v34.json"
REFERENCE_SHA = "653c760112f01ebc074e23258104b5a3976aec8114abb26865cf211009c6fe5b"
REFERENCE_RUNTIME = OLD_WORKSPACE / "input/reference_runtime.json"
REFERENCE_RUNTIME_SHA = "57aa4d227d6b490bab3dbf4ddd1574542a042d7d29879ad1ddbe7ae48de55421"
SOURCE_ROOT = Path("/home/serg/project/camera_reader_sky/srcsky/chunks")
SOURCE_HASHES = {
    "0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
    "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585",
}


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
    require(path.stat().st_size <= 32 * 1024 * 1024, "oversize metadata")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scope_path(workspace):
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside discovery workspace scope")
    return Path(workspace)


def source_spec(clip):
    require(clip in SOURCE_HASHES, "unapproved clip")
    return dict(path=str(SOURCE_ROOT / f"chunk_{clip}.avi"), sha256=SOURCE_HASHES[clip],
                frames=FRAMES, width=WIDTH, height=HEIGHT, fps=10,
                codec="mjpeg", pixel_format="yuvj420p")


def workspace_guard(workspace, clip, mode):
    workspace = scope_path(workspace)
    source_spec(clip)
    require(mode in {"preflight", "run"}, "invalid execution mode")
    require(os.geteuid() != 0, "never run as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked workspace")
    directory = workspace / clip
    require(not directory.is_symlink() and (not directory.exists() or directory.is_dir()),
            "invalid/linked clip directory")
    for name in ("run", "execution_receipt.json") + (("preflight.json",) if mode == "preflight" else ()):
        path = directory / name
        require(not path.exists() and not path.is_symlink(), f"existing output: {path}")
    return workspace


def validate_freeze(value, hashes):
    require(value.get("schema") == "discovery_pair.v1", "wrong freeze schema")
    files = value.get("files", {})
    require(isinstance(files, dict) and "run_discovery_pair_baseline.py" in files,
            "missing frozen runner")
    require(all(isinstance(name, str) and Path(name).name == name and name not in {".", "..", "freeze.json"}
                and isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
                for name, digest in files.items()), "invalid frozen file names/hashes")
    require(files == hashes, "transferred files differ from freeze")
    require(value.get("sources") == {clip: source_spec(clip) for clip in SOURCE_HASHES},
            "source inventory outside fixed two-clip scope")
    for row in value["sources"].values():
        require(all(type(row[key]) is int for key in ("frames", "width", "height", "fps")),
                "source numerical metadata must be exact integers")


def inputs(workspace, clip):
    freeze = read(workspace / "freeze.json")
    files = freeze.get("files", {})
    # Validate path syntax before opening any manifest-controlled path.
    validate_freeze(freeze, files)
    file_hashes = {}
    for name in files:
        path = workspace / name
        require(path.is_file() and not path.is_symlink(), f"missing/linked frozen file: {name}")
        file_hashes[name] = sha(path)
    validate_freeze(freeze, file_hashes)
    require(sha(Path(__file__)) == file_hashes["run_discovery_pair_baseline.py"], "executed runner differs")
    pinned = {HELPER: HELPER_SHA, REFERENCE: REFERENCE_SHA,
              REFERENCE_RUNTIME: REFERENCE_RUNTIME_SHA,
              Path(source_spec(clip)["path"]): SOURCE_HASHES[clip]}
    for path, digest in pinned.items():
        require(path.is_file() and not path.is_symlink(), f"missing/linked pinned input: {path}")
        require(sha(path) == digest, f"pinned input identity changed: {path}")
    hashes = dict(files=file_hashes, freeze_sha256=sha(workspace / "freeze.json"),
                  baseline_harness_sha256=HELPER_SHA, reference_launch_sha256=REFERENCE_SHA,
                  reference_runtime_sha256=REFERENCE_RUNTIME_SHA, source_sha256=SOURCE_HASHES[clip])
    return read(REFERENCE), read(REFERENCE_RUNTIME), hashes


def load_helper():
    require(HELPER.is_file() and not HELPER.is_symlink() and sha(HELPER) == HELPER_SHA,
            "baseline helper identity changed before import")
    spec = importlib.util.spec_from_file_location("discovery_frozen_aot_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_probe(probe):
    require(probe.codec == "mjpeg" and probe.pixel_format == "yuvj420p"
            and (probe.width, probe.height) == (WIDTH, HEIGHT)
            and probe.frame_rate == Fraction(10) and probe.declared_frame_count == FRAMES,
            "unexpected native AVI probe contract")


def check_frame(frame, index):
    require(0 <= index < FRAMES and frame.index == index, "decoded frame order/count changed")
    require(frame.gray.shape == (HEIGHT, WIDTH) and str(frame.gray.dtype) == "uint8",
            "decoded frame is not native gray8")


def validate_preflight(pre, hashes, identities, workspace, clip):
    require(pre.get("schema") == SCHEMA + ".preflight" and pre.get("passed") is True
            and pre.get("workspace") == str(workspace) and pre.get("clip") == clip,
            "missing successful matching preflight")
    require(pre.get("input_sha256") == hashes and pre.get("source") == source_spec(clip),
            "preflight-bound inputs/source changed")
    require(pre.get("probe_passed") is True and pre.get("detector_run") is False,
            "incomplete metadata preflight")
    require(all(pre.get(key) == value for key, value in identities.items()),
            "dependencies differ from preflight")


def preflight(workspace, clip):
    workspace = workspace_guard(workspace, clip, "preflight")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA + ".preflight", passed=False, workspace=str(workspace),
                   clip=clip, detector_run=False, full_pixel_predecode=False)
    try:
        reference, runtime, hashes = inputs(workspace, clip)
        helper = load_helper()
        modules, identities = helper.dependencies(reference)
        runtime_info = modules["profile_visible_interaction_v30"].runtime_info
        before = runtime_info()
        helper.runtime_check(before, runtime)
        clocks = helper.clock_policy_snapshot()
        from tiny_target.frame_source import probe_video
        probe = probe_video(source_spec(clip)["path"])
        validate_probe(probe)
        after = runtime_info()
        helper.runtime_check(after, runtime)
        require(helper.clock_policy_snapshot() == clocks, "clock controls changed during preflight")
        require(inputs(workspace, clip)[2] == hashes, "inputs changed during preflight")
        receipt.update(passed=True, source=source_spec(clip), input_sha256=hashes,
                       probe=probe.to_dict(), probe_passed=True, runtime_before=before,
                       runtime_after=after, clock_policy=clocks, clocks_changed=False, **identities)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "preflight.json", receipt)
    print(json.dumps(dict(clip=clip, preflight_passed=True, detector_run=False)), flush=True)
    return receipt


def validate_output(report, launch, reference, clip, verified_count):
    require(report.get("completed") is True and report.get("full_clip") is True
            and report.get("frames") == verified_count == FRAMES,
            "incomplete full-source execution")
    require(report.get("source_sha256") == launch.get("source_sha256") == SOURCE_HASHES[clip]
            and launch.get("source") == source_spec(clip)["path"], "output source identity differs")
    require(report.get("configuration") == launch.get("configuration") == reference["configuration"]
            and launch.get("config_sha256") == reference["config_sha256"]
            and launch.get("motion_config_sha256") == reference["motion_config_sha256"]
            and launch.get("package_sha256") == reference["package_sha256"],
            "output frozen configuration/package differs")
    require(launch.get("expected_frames") == FRAMES and launch.get("max_frames") is None
            and launch.get("fps") == 10 and launch.get("annotations_supplied_to_detector") is False,
            "output count/cadence/annotation contract differs")
    probe = launch.get("source_probe", {})
    require(all(probe.get(k) == v for k, v in dict(codec="mjpeg", pixel_format="yuvj420p",
                width=WIDTH, height=HEIGHT, declared_frame_count=FRAMES, frame_rate="10").items()),
            "output probe contract differs")
    stats = report.get("frame_decode", {})
    require(stats.get("decoded_frames") == stats.get("consumed_frames") == FRAMES
            and stats.get("dropped_frames") == 0 and stats.get("worker_joined") is True
            and stats.get("capture_released") is True
            and stats.get("maximum_observed_frames_ahead", 2) <= 1
            and stats.get("contract", {}).get("execution") == "prefetch_one",
            "decode lifecycle/count differs")


def execute_baseline(helper, modules, source, output, receipt):
    """Exact frozen AOT adapter stack; only input/count integrity instrumentation differs."""
    from tiny_target import motion, visible_baseline as visible, visible_resident as resident
    from tiny_target import visible_warp_exact as warp, visible_decode as decode
    from tiny_target.tracking.kalman import KalmanTrackManager
    from run_visible_stage_v24 import validate_snapshot
    stage, front = modules["visible_stage_v24"], modules["visible_front_v26"]
    reuse = modules["motion_reuse_v12"]
    fronts, estimators, attempts, readers, cleanup_errors = [], [], [], [], []
    verified_count = 0
    mask = modules["learning_mask_v17"].LearningMaskV17(helper.V17 / "build/liblearning_mask_v17.so")
    geometry = modules["tracking_geometry_v20"].GeometryV20(helper.V20 / "build/libtracking_geometry_v20.so")
    optimized = modules["tracking_stage_v28"].TrackingStageV28(helper.V27 / "build_01/libtracking_batch_v27.so")
    method = optimized.adapt(geometry.adapter(KalmanTrackManager.update))
    require(optimized.transformed_sha256 == helper.TRACKING_METHOD_SHA, "tracking transformation changed")
    binding = stage.StageBinding("reference")

    class CapturedFront(front.ResidentFrontV26):
        def __init__(self, cfg):
            super().__init__(cfg)
            fronts.append(self)

    class CapturedMotion(reuse.ReuseMotionV12):
        def __init__(self, cfg):
            super().__init__(cfg)
            estimators.append(self)

        def estimate(self, previous, current):
            row = dict(frame=current.frame_index, error=None, expected_unavailable=False)
            attempts.append(row)
            try:
                return super().estimate(previous, current)
            except BaseException as exc:
                row.update(error=repr(exc), expected_unavailable=(isinstance(exc, motion.PvaMotionError)
                    and any(text in str(exc) for text in ("zero features", "No finite in-bounds"))))
                raise

    try:
        with ExitStack() as stack:
            stage.install(stack, binding)
            stack.enter_context(patch.object(resident, "shape_learning_mask", mask))
            stack.enter_context(patch.object(KalmanTrackManager, "update", method))
            stack.enter_context(patch.object(resident, "VisibleCudaResident", CapturedFront))
            stack.enter_context(patch.object(warp.CudaCubicTranslation, "__call__",
                front.attach_warp(warp.CudaCubicTranslation.__call__)))
            stack.enter_context(patch.object(motion, "PvaPyrLkMotionEstimator", CapturedMotion))
            original_read = decode.VisibleFrameReader.read

            def checked_read(reader):
                nonlocal verified_count
                if not readers:
                    require(reader.fps == 10 and reader.expected == FRAMES, "reader cadence/count differs")
                    readers.append(reader)
                require(readers[0] is reader, "unexpected second decoder")
                result = original_read(reader)
                if result[0] is not None:
                    check_frame(result[0], verified_count)
                    verified_count += 1
                return result

            stack.enter_context(patch.object(decode.VisibleFrameReader, "read", checked_read))
            report = visible.run(source, helper.CONFIG, output, helper.MOTION_CONFIG, None)
        receipt["processed_frames"] = report["frames"]
        require(report["completed"] and report["full_clip"] and report["frames"] == FRAMES
                and verified_count == FRAMES, "incomplete full clip execution")
        validate_snapshot(binding.snapshot(), FRAMES)
        require(len(fronts) == 1 and fronts[0].calls == fronts[0].device_calls
                == fronts[0].finish_calls == FRAMES and fronts[0].host_calls == mask.calls == 0
                and fronts[0].front is None and fronts[0].handle is None,
                "GPU front execution/lifecycle differs from combined baseline")
        require(geometry.calls == geometry.fallbacks == optimized.geometry.fallbacks
                == optimized.innovation_fallbacks == 0
                and optimized.innovation_tracks == optimized.geometry.track_rows,
                "tracking fallback/incomplete optimized tracking")
        require(len(estimators) == 1 and [row["frame"] for row in attempts] == list(range(1, FRAMES)),
                "missing/duplicate adjacent motion attempts")
        require(not estimators[0].failed and all(row["error"] is None or row["expected_unavailable"]
                for row in attempts), "unexpected PVA/backend failure")
        return report
    finally:
        for estimator in estimators:
            try:
                estimator.close()
            except BaseException as exc:
                cleanup_errors.append(repr(exc))
        receipt.update(decoded_frames_verified=verified_count, execution=binding.snapshot(),
                       motion_attempts=attempts, cleanup_errors=cleanup_errors,
                       motion_instances=[dict(hits=x.hits, misses=x.misses, failed=x.failed, closed=x.closed)
                                         for x in estimators],
                       gpu_fronts=[dict(calls=x.calls, device_calls=x.device_calls, host_calls=x.host_calls,
                           finish_calls=x.finish_calls, closed=x.front is None and x.handle is None) for x in fronts],
                       native_mask_calls=mask.calls,
                       tracking=dict(geometry_calls=geometry.calls, geometry_fallbacks=geometry.fallbacks,
                           batch_calls=optimized.geometry.calls, batch_fallbacks=optimized.geometry.fallbacks,
                           batch_track_rows=optimized.geometry.track_rows,
                           innovation_tracks=optimized.innovation_tracks,
                           innovation_fallbacks=optimized.innovation_fallbacks,
                           exercised=optimized.geometry.calls > 0))
        require(not cleanup_errors, "motion estimator cleanup failed: " + repr(cleanup_errors))


def run(workspace, clip):
    workspace = workspace_guard(workspace, clip, "run")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA, passed=False, error=None, processed_frames=0,
                   workspace=str(workspace), clip=clip, source=source_spec(clip),
                   algorithm_changed=False, detector_configuration_changed=False,
                   raw16_accessed=False, sealed_holdouts_accessed=False,
                   annotations_supplied_to_detector=False, airborne_class_verified=False,
                   purpose="Unlabeled candidate discovery for human review, not independent accuracy scoring.",
                   timestamp_basis="consumer index / nominal 10 Hz container fps; physical acquisition cadence unverified",
                   timing_instrumentation="Per-frame index/native shape/dtype checks only, no pixel hashing. Source/code hashing is outside timed detector loop. Current clocks: not a v34 fixed-clock speed comparison.",
                   clocks_changed=None, remote_clocks_unchanged=None)
    try:
        reference, runtime, hashes = inputs(workspace, clip)
        helper = load_helper()
        modules, identities = helper.dependencies(reference)
        pre = read(directory / "preflight.json")
        validate_preflight(pre, hashes, identities, workspace, clip)
        runtime_info = modules["profile_visible_interaction_v30"].runtime_info
        before = runtime_info()
        helper.runtime_check(before, runtime)
        clock_before = helper.clock_policy_snapshot()
        require(clock_before == pre.get("clock_policy"), "clock controls differ from preflight")
        receipt.update(input_sha256=hashes, preflight_sha256=sha(directory / "preflight.json"),
                       runtime_before=before, clock_policy_before=clock_before,
                       tracking_transformed_sha256=helper.TRACKING_METHOD_SHA, **identities)
        report = execute_baseline(helper, modules, Path(source_spec(clip)["path"]), directory / "run", receipt)
        launch = read(directory / "run/launch.json")
        validate_output(report, launch, reference, clip, receipt["decoded_frames_verified"])
        after = runtime_info()
        helper.runtime_check(after, runtime, after=True)
        clock_after = helper.clock_policy_snapshot()
        receipt.update(runtime_after=after, clock_policy_after=clock_after,
                       clocks_changed=clock_after != clock_before,
                       remote_clocks_unchanged=clock_after == clock_before)
        require(clock_after == clock_before, "clock controls changed during detector execution")
        require(inputs(workspace, clip)[2] == hashes, "frozen inputs changed during run")
        require(helper.dependencies(reference)[1] == identities, "frozen dependencies changed during run")
        receipt.update(passed=True, processed_frames=report["frames"],
                       journal_sha256=sha(directory / "run/frames.jsonl"),
                       report_sha256=sha(directory / "run/report.json"),
                       launch_sha256=sha(directory / "run/launch.json"),
                       availability=report["availability"], detection_status=report["detection_status"])
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "execution_receipt.json", receipt)
    print(json.dumps(dict(clip=clip, passed=True, processed_frames=receipt["processed_frames"],
                          journal_sha256=receipt["journal_sha256"])), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--clip", choices=tuple(SOURCE_HASHES), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    (preflight if args.preflight else run)(args.workspace, args.clip)


if __name__ == "__main__":
    main()
