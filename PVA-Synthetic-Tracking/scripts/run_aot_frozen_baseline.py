#!/usr/bin/env python3
"""Explicit AOT-only input harness for the frozen v34 combined PVA/CUDA stack.

No installation, clock writes, old-source wrapper calls, CPU substitution,
annotation-driven settings, training, or source modifications are performed.
Run preflight first, after freezing scoring policy, then run in a fresh process.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

SCHEMA = "seaqr.aot.frozen-baseline.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_aot_pilot_20260927_[A-Za-z0-9]{6}"
FRAMES, WIDTH, HEIGHT = 300, 2448, 2048
MANIFEST_SHA = "425b8220417e9853a8fbf272a1119d4e1989678984c84f295c9aeb2507b72d8d"
VIDEO_SHA = "869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f"
REFERENCE_SHA = "653c760112f01ebc074e23258104b5a3976aec8114abb26865cf211009c6fe5b"
CONFIG_SHA = "7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f"
MOTION_SHA = "fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1"
FREEZE_SHA = "60b79d450672d131b517e9ed5a33fdeb40a6a2a9b29c6c0f584a9a8e93cc0dbc"
REUSE_METHOD_SHA = "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733"
TRACKING_METHOD_SHA = "571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325"
V29 = Path("/tmp/seaqr_visible_combined_v29_s8XhL1")
V30 = Path("/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ")
V26 = Path("/tmp/seaqr_visible_front_v26_retry_24sEU7")
V28 = Path("/tmp/seaqr_tracking_v28_Nn629D")
V27 = Path("/tmp/seaqr_tracking_v27_pmUXGZ")
V24 = Path("/tmp/seaqr_visible_stage_v24_Uo3Tze")
V20 = Path("/tmp/seaqr_visible_speed_v20_XUf1LR")
V17 = Path("/tmp/seaqr_visible_speed_v17_EER6lm")
V13 = Path("/tmp/seaqr_video_v13_IyK7eQ")
RUNTIME = Path("/tmp/seaqr_exact_v9_rS2LFx")
VISIBLE = Path("/tmp/seaqr_phase20_decode_v10_Iz9RSF")
CONFIG = V26 / "candidate_config.json"
MOTION_CONFIG = RUNTIME / "configs/evaluation/phase20_motion_v8.json"
LIBRARIES = {
    V26 / "build_03/candidate.so": "fd689b653175eb9baf3e84259ccccd431ba3ec8aa6c6fa4378a596eb03f8a027",
    VISIBLE / "libseaqr_shapes.so": "ec3ef02040df013536a5610d78bcde9763c1bcade0ec8a43e6525a6b3f0bb7ca",
    V20 / "build/libtracking_geometry_v20.so": "1d81a0369a78ca462fce16e9655e9c81811d4544e071c013682f5aaf58821782",
    V27 / "build_01/libtracking_batch_v27.so": "bdabc75a633da7cb72c0565a3d2b87dcb6662b17b12ba96ccd7a7c586bebb644",
    V17 / "build/liblearning_mask_v17.so": "1eb3e13dc035644162e3dd8056c2ef5cc7ef40de33b25a82efaab491a3dd8c70",
}
ADAPTERS = {
    "motion_reuse_v12": "038e45d83c46909958fcbdf5b94d791f7a69733ef765851c7d0fdf77c19a48f8",
    "raw16_speed_v8_common": "09c89333193888e97bae1d1f90bed4ca3e33ea5a2a895c0bdc245dd4a84b997d",
    "visible_stage_v24": "d19fc73d4528e3783eb0f0b019a2c328d6ce21eee7df837cfdada56cc66c8f2a",
    "visible_overlap_v23": "b611ea55a68a67a727c2754751b478e8d3d6a4ff0b731af8a83c08612312426f",
    "frame_lookahead_v23": "8193584fda014cff16945d92fbe19ca89266fc4b713dae05ebf4672a41052ad8",
    "stage_control_v24": "47dbf736a65ca4c19c4010d6d6703cc7434eba0de39e98f23975341e04bc4328",
    "visible_front_v26": "59013c83604f636aa2239ccab5fcc362bacd2e1c741fb282d5bd732fb92d8790",
    "learning_mask_v17": "eb9f3869a5e54d883dde30136e9fc04d0ef4b9729e9b77e5404ae7cde622b5c8",
    "tracking_geometry_v20": "2021152f598082b88cabac762f12a7bb886def17a85b85cec57d1c0b31ad4e52",
    "tracking_batch_v27": "6ff824041e254bb7ada457263708e5210cb503f62d5691ee3af012564c7a4793",
    "tracking_stage_v28": "92b4e5a7b2be8556de434f1a216530d896e428e6c62e4879789ec7a9cc360975",
    "profile_visible_interaction_v30": "190ca65f9455138ba5b766238821873df07850c0a3b9436a0bd57612559931e0",
}
INPUT_NAMES = ("pilot_gray8_ffv1_10fps.avi", "frozen_image_manifest.json",
               "download_validation.json", "packaging_validation.json",
               "reference_launch_v34.json", "reference_runtime.json")


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
    require(path.is_file() and not path.is_symlink(), f"missing/linked file: {path}")
    require(path.stat().st_size <= 32 * 1024 * 1024, f"oversize metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scope_path(workspace):
    text = str(workspace)
    require(re.fullmatch(WORKSPACE_PATTERN, text), "outside exact AOT workspace scope")
    return Path(text)


def workspace_guard(workspace, mode):
    workspace = scope_path(workspace)
    require(os.geteuid() != 0, "never run intake/detector as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked workspace")
    require((workspace / "input").is_dir() and not (workspace / "input").is_symlink(),
            "missing/linked input directory")
    for name in ("run", "execution_receipt.json"):
        path = workspace / name
        require(not path.exists() and not path.is_symlink(), f"existing detector output: {name}")
    if mode == "preflight":
        path = workspace / "preflight.json"
        require(not path.exists() and not path.is_symlink(), "existing preflight; refusing overwrite")
    return workspace


def validate_inventory(manifest, download, packaging, hashes):
    require(hashes["frozen_image_manifest.json"] == MANIFEST_SHA
            and hashes["pilot_gray8_ffv1_10fps.avi"] == VIDEO_SHA
            and hashes["reference_launch_v34.json"] == REFERENCE_SHA,
            "source/reference identity outside fixed AOT pilot")
    rows, images = manifest.get("frames", []), download.get("images", [])
    require(len(rows) == len(images) == FRAMES and download.get("frames") == FRAMES,
            "exactly 300 causal input frames required")
    require(download.get("manifest_sha256") == MANIFEST_SHA
            and download.get("resolution") == [WIDTH, HEIGHT]
            and download.get("dtype") == "uint8" and download.get("grayscale") is True,
            "download validation contract differs")
    require(packaging.get("passed") is True and packaging.get("frames") == FRAMES
            and packaging.get("video_sha256") == VIDEO_SHA
            and packaging.get("pixel_hashes_verified") == FRAMES
            and packaging.get("detector_run") is False,
            "missing completed lossless packaging validation")
    require(packaging.get("inputs_sha256", {}).get("frozen_image_manifest.json") == MANIFEST_SHA
            and packaging.get("inputs_sha256", {}).get("download_validation.json")
            == hashes["download_validation.json"], "packaging input identities changed")
    for i, (row, image) in enumerate(zip(rows, images)):
        require(all(row.get(k) == image.get(k) for k in ("source_frame", "timestamp_ns", "img_name")),
                "download/frame manifest mapping differs")
        require(type(row.get("source_frame")) is int and isinstance(row.get("timestamp_ns"), str)
                and row["timestamp_ns"].isdigit(), "invalid exact source index/timestamp")
        require(isinstance(image.get("pixel_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", image["pixel_sha256"]), "missing exact pixel hash")
        if i:
            require(row["source_frame"] == rows[i - 1]["source_frame"] + 1
                    and int(row["timestamp_ns"]) > int(rows[i - 1]["timestamp_ns"]),
                    "non-contiguous/reordered causal prefix")


def inputs(workspace):
    directory = workspace / "input"
    hashes = {}
    for name in INPUT_NAMES:
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"missing/linked input: {name}")
        hashes[name] = sha(path)
    values = {name: read(directory / name) for name in INPUT_NAMES if name.endswith(".json")}
    policy = read(workspace / "scoring_freeze.json")
    require(isinstance(policy, dict) and bool(policy), "nonempty pre-run scoring freeze required")
    hashes["scoring_freeze.json"] = sha(workspace / "scoring_freeze.json")
    validate_inventory(values["frozen_image_manifest.json"], values["download_validation.json"],
                       values["packaging_validation.json"], hashes)
    return values, hashes


def runtime_check(actual, expected, after=False):
    keys = ("blas", "affinity", "numpy", "opencv", "thread_environment", "clock_ticks")
    require(all(actual.get(k) == expected.get(k) for k in keys), "numerical runtime differs from v34")
    require(actual["numpy"] == "1.26.1" and actual["opencv"] == "4.10.0"
            and len(actual["blas"]) == 1 and actual["blas"][0]["threads"] == 12,
            "frozen numerical versions/12-thread BLAS required")
    require(all(value is None for value in actual["thread_environment"].values()),
            "numerical thread environment changed")
    require(not after or actual["opencv_threads"] == 2, "post-run OpenCV workers differ")


def clock_policy_snapshot():
    """Read configured controls, not naturally varying instantaneous DVFS clocks."""
    policies = {
        **{f"cpu{i}": (Path(f"/sys/devices/system/cpu/cpufreq/policy{i}"),
                       ("scaling_min_freq", "scaling_max_freq", "scaling_governor"))
           for i in (0, 4, 8)},
        "gpu": (Path("/sys/class/devfreq/17000000.gpu"), ("min_freq", "max_freq", "governor")),
    }
    result = {}
    for name, (directory, fields) in policies.items():
        values = [(directory / field).read_text().strip() for field in fields]
        result[name] = dict(minimum=int(values[0]), maximum=int(values[1]), governor=values[2])
    return result


def dependencies(reference):
    paths = [V29, V30, V26, V28, V24, V20, V17, V13, RUNTIME, RUNTIME / "scripts"]
    sys.path[:0] = [str(path) for path in paths]  # Also works with python -I.
    require(sha(V29 / "freeze.json") == FREEZE_SHA, "v29 freeze identity changed")
    frozen = read(V29 / "freeze.json")
    require(sha(V29 / "run_visible_combined_v29.py") == frozen["files"]["run_visible_combined_v29.py"],
            "v29 runner changed before import")
    baseline = importlib.import_module("run_visible_combined_v29")
    require(Path(baseline.__file__).resolve() == V29 / "run_visible_combined_v29.py",
            "wrong baseline module origin")
    baseline.verify_freeze()  # No calls to any old-source run wrapper.
    package = importlib.import_module("tiny_target")
    require(Path(package.__file__).resolve() == RUNTIME / "tiny_target/__init__.py",
            "wrong frozen package origin")
    for name, digest in reference["package_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "invalid package hash path")
        require(sha(RUNTIME / "tiny_target" / relative) == digest, f"package changed: {name}")
    require(sha(CONFIG) == reference["config_sha256"] == CONFIG_SHA
            and sha(MOTION_CONFIG) == reference["motion_config_sha256"] == MOTION_SHA,
            "frozen detector/motion configuration changed")
    require(read(CONFIG) == reference["configuration"], "candidate configuration differs from v34")
    libraries = {}
    for path, digest in LIBRARIES.items():
        require(sha(path) == digest, f"native library identity changed: {path}")
        libraries[str(path)] = digest
    modules, adapters = {}, {}
    for name, digest in ADAPTERS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.parent in paths and sha(path) == digest, f"adapter origin/hash changed: {name}")
        modules[name] = module
        adapters[name] = dict(path=str(path), sha256=digest)
    method = modules["motion_reuse_v12"].generated_method()
    require(hashlib.sha256(method.encode()).hexdigest() == REUSE_METHOD_SHA,
            "motion reuse transformation changed")
    vpi = importlib.import_module("vpi")
    require(vpi.__version__ == "3.2.4" and all(hasattr(vpi.Backend, name) for name in ("CUDA", "PVA")),
            "frozen VPI 3.2.4 CUDA/PVA backends required")
    return modules, dict(adapters=adapters, libraries=libraries, config_sha256=CONFIG_SHA,
                         motion_config_sha256=MOTION_SHA, v29_freeze_sha256=FREEZE_SHA,
                         reuse_method_sha256=REUSE_METHOD_SHA, vpi_version=vpi.__version__)


def check_frame(frame, index, images):
    require(index < FRAMES and frame.index == index, "decoded frame order/count changed")
    require(frame.gray.shape == (HEIGHT, WIDTH) and str(frame.gray.dtype) == "uint8",
            "decoded frame is not native gray8")
    require(hashlib.sha256(frame.gray.tobytes(order="C")).hexdigest() == images[index]["pixel_sha256"],
            f"decoded native pixel hash differs at frame {index}")


def decode_preflight(video, images):
    from fractions import Fraction
    from tiny_target.frame_source import probe_video
    from tiny_target.visible_decode import VisibleFrameReader
    probe = probe_video(video)
    require(probe.codec == "ffv1" and probe.pixel_format in {"gray", "gray8"}
            and (probe.width, probe.height) == (WIDTH, HEIGHT)
            and probe.frame_rate == Fraction(10) and probe.declared_frame_count == FRAMES,
            "unexpected video probe contract")
    count = 0
    with VisibleFrameReader(video, (HEIGHT, WIDTH), "prefetch_one") as reader:
        require(reader.fps == 10 and reader.expected == FRAMES, "reader cadence/count differs")
        while True:
            frame, _ = reader.read()
            if frame is None:
                break
            check_frame(frame, count, images)
            count += 1
    stats = reader.completed_stats()
    require(count == FRAMES and stats["decoded_frames"] == stats["consumed_frames"] == FRAMES
            and stats["dropped_frames"] == 0 and stats["maximum_observed_frames_ahead"] <= 1,
            "incomplete native pixel decode preflight")
    return dict(passed=True, pixel_hashes_verified=count, probe=probe.to_dict(), stats=stats)


def preflight(workspace):
    workspace = workspace_guard(workspace, "preflight")
    values, hashes = inputs(workspace)
    modules, identities = dependencies(values["reference_launch_v34.json"])
    before = modules["profile_visible_interaction_v30"].runtime_info()
    runtime_check(before, values["reference_runtime.json"])
    clock_policy = clock_policy_snapshot()
    decoded = decode_preflight(workspace / "input" / INPUT_NAMES[0], values["download_validation.json"]["images"])
    # Decode must not silently change any numerical policy used by the baseline.
    after = modules["profile_visible_interaction_v30"].runtime_info()
    runtime_check(after, values["reference_runtime.json"])
    require(clock_policy_snapshot() == clock_policy, "clock controls changed during preflight")
    require(inputs(workspace)[1] == hashes, "inputs changed during preflight")
    receipt = dict(schema=SCHEMA + ".preflight", passed=True, workspace=str(workspace),
                   input_sha256=hashes, script_sha256=sha(Path(__file__)),
                   runtime_before=before, runtime_after=after, decode=decoded,
                   detector_run=False, clocks_changed=False, clock_policy=clock_policy, **identities)
    write(workspace / "preflight.json", receipt)
    print(json.dumps(dict(preflight_passed=True, frames=FRAMES, detector_run=False)), flush=True)
    return receipt


def validate_preflight(pre, hashes, identities, script_hash, workspace):
    require(pre.get("schema") == SCHEMA + ".preflight" and pre.get("passed") is True
            and pre.get("workspace") == str(workspace), "missing successful matching preflight")
    require(pre.get("script_sha256") == script_hash and pre.get("input_sha256") == hashes,
            "preflight-bound script/input/scoring freeze changed")
    require(pre.get("decode", {}).get("passed") is True
            and pre["decode"].get("pixel_hashes_verified") == FRAMES,
            "preflight did not verify every input frame")
    require(all(pre.get(key) == value for key, value in identities.items()),
            "dependencies differ from preflight")


def run(workspace):
    workspace = workspace_guard(workspace, "run")
    pre = read(workspace / "preflight.json")
    values, hashes = inputs(workspace)
    modules, identities = dependencies(values["reference_launch_v34.json"])
    validate_preflight(pre, hashes, identities, sha(Path(__file__)), workspace)
    runtime_info = modules["profile_visible_interaction_v30"].runtime_info
    before = runtime_info()
    runtime_check(before, values["reference_runtime.json"])
    clock_before = clock_policy_snapshot()
    require(clock_before == pre.get("clock_policy"), "clock controls differ from preflight")
    from tiny_target import motion, visible_baseline as visible, visible_resident as resident
    from tiny_target import visible_warp_exact as warp, visible_decode as decode
    from tiny_target.tracking.kalman import KalmanTrackManager
    from run_visible_stage_v24 import validate_snapshot
    stage, front = modules["visible_stage_v24"], modules["visible_front_v26"]
    reuse = modules["motion_reuse_v12"]
    mask = modules["learning_mask_v17"].LearningMaskV17(V17 / "build/liblearning_mask_v17.so")
    geometry = modules["tracking_geometry_v20"].GeometryV20(V20 / "build/libtracking_geometry_v20.so")
    optimized = modules["tracking_stage_v28"].TrackingStageV28(V27 / "build_01/libtracking_batch_v27.so")
    method = optimized.adapt(geometry.adapter(KalmanTrackManager.update))
    require(optimized.transformed_sha256 == TRACKING_METHOD_SHA, "tracking transformation changed")
    binding, fronts, estimators, attempts = stage.StageBinding("reference"), [], [], []
    verified_count = 0

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

    receipt = dict(schema=SCHEMA, passed=False, error=None, processed_frames=0,
                   workspace=str(workspace), input_sha256=hashes, runtime_before=before,
                   preflight_sha256=sha(workspace / "preflight.json"),
                   scoring_freeze_sha256=hashes["scoring_freeze.json"],
                   script_sha256=sha(Path(__file__)), tracking_transformed_sha256=TRACKING_METHOD_SHA,
                   algorithm_changed=False, detector_configuration_changed=False,
                   clocks_changed=None, remote_clocks_unchanged=None,
                   clock_policy_before=clock_before,
                   clock_policy="no clock writes; current clocks, not v34 fixed-clock speed comparison",
                   timing_instrumentation="Every consumed native frame is SHA-256 checked on the host before motion; measured processing time includes this added integrity work. Accuracy/compatibility run only, not speed-comparable to v34.",
                   timestamp_basis="nominal 10 Hz container CFR; acquisition timestamps retained in input manifest",
                   raw16_accessed=False, private_camera_media_accessed=False,
                   annotations_supplied_to_detector=False, **identities)
    cleanup_errors = []
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
                result = original_read(reader)
                if result[0] is not None:
                    check_frame(result[0], verified_count, values["download_validation.json"]["images"])
                    verified_count += 1
                return result

            stack.enter_context(patch.object(decode.VisibleFrameReader, "read", checked_read))
            report = visible.run(workspace / "input" / INPUT_NAMES[0], CONFIG,
                                 workspace / "run", MOTION_CONFIG, None)
        receipt["processed_frames"] = report["frames"]
        require(report["completed"] and report["full_clip"] and report["frames"] == FRAMES
                and verified_count == FRAMES, "incomplete full pilot execution")
        validate_snapshot(binding.snapshot(), FRAMES)
        require(len(fronts) == 1 and fronts[0].calls == fronts[0].device_calls
                == fronts[0].finish_calls == FRAMES and fronts[0].host_calls == mask.calls == 0
                and fronts[0].front is None and fronts[0].handle is None,
                "GPU front execution/lifecycle differs from combined baseline")
        require(geometry.calls == geometry.fallbacks == optimized.geometry.fallbacks
                == optimized.innovation_fallbacks == 0
                and optimized.innovation_tracks == optimized.geometry.track_rows,
                "tracking backend fallback or incomplete optimized tracking")
        require(len(estimators) == 1 and [row["frame"] for row in attempts] == list(range(1, FRAMES)),
                "missing/duplicate adjacent motion attempts")
        require(not estimators[0].failed and all(row["error"] is None or row["expected_unavailable"]
                for row in attempts), "unexpected PVA/backend failure")
        after = runtime_info()
        runtime_check(after, values["reference_runtime.json"], after=True)
        clock_after = clock_policy_snapshot()
        receipt.update(clock_policy_after=clock_after, clocks_changed=clock_after != clock_before,
                       remote_clocks_unchanged=clock_after == clock_before)
        require(clock_after == clock_before, "clock controls changed during detector execution")
        require(inputs(workspace)[1] == hashes, "inputs/scoring freeze changed during run")
        require(dependencies(values["reference_launch_v34.json"])[1] == identities,
                "frozen dependency identities changed during run")
        receipt.update(runtime_after=after, journal_sha256=sha(workspace / "run/frames.jsonl"),
                       report_sha256=sha(workspace / "run/report.json"),
                       launch_sha256=sha(workspace / "run/launch.json"),
                       availability=report["availability"], detection_status=report["detection_status"],
                       passed=True)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        for estimator in estimators:
            try:
                estimator.close()
            except BaseException as exc:
                cleanup_errors.append(repr(exc))
        if cleanup_errors:
            receipt.update(passed=False, error="motion cleanup failed: " + repr(cleanup_errors))
        receipt.update(pixel_hashes_verified=verified_count, execution=binding.snapshot(),
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
        write(workspace / "execution_receipt.json", receipt)
    require(not cleanup_errors, "motion estimator cleanup failed")
    print(json.dumps(dict(passed=receipt["passed"], processed_frames=receipt["processed_frames"],
                          journal_sha256=receipt.get("journal_sha256"))), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    (preflight if args.preflight else run)(args.workspace)


if __name__ == "__main__":
    main()
