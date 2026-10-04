#!/usr/bin/env python3
"""Supervised metadata-only baseline parity on the original Jetson runtime.

No source media, decoder, detector, PVA, GPU, native build, policy change or
hardware-control write. Original tracker binaries and saved journals are reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

BASE_SHA = "6a26937ef03821db05a96f680d899234436e49a864c9d287628872e471c20c73"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
PATTERN = r"/tmp/seaqr_tracker_baseline_20261001_[A-Za-z0-9]{6}"
TRACE = Path("/tmp/seaqr_feature_residual_trace_20260930_1lWybU")
RUNTIME = Path("/tmp/seaqr_exact_v9_rS2LFx")
CONFIG = Path("/tmp/seaqr_visible_front_v26_retry_24sEU7/candidate_config.json")
ADAPTER_PATHS = {
    "tracking_geometry_v20": Path("/tmp/seaqr_visible_speed_v20_XUf1LR/tracking_geometry_v20.py"),
    "tracking_batch_v27": Path("/tmp/seaqr_tracking_v28_Nn629D/tracking_batch_v27.py"),
    "tracking_stage_v28": Path("/tmp/seaqr_tracking_v28_Nn629D/tracking_stage_v28.py"),
}
PROFILER = Path("/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ/profile_visible_interaction_v30.py")
PROFILER_SHA = "190ca65f9455138ba5b766238821873df07850c0a3b9436a0bd57612559931e0"
LIBRARIES = {
    "geometry": (Path("/tmp/seaqr_visible_speed_v20_XUf1LR/build/libtracking_geometry_v20.so"),
                 "1d81a0369a78ca462fce16e9655e9c81811d4544e071c013682f5aaf58821782"),
    "batch": (Path("/tmp/seaqr_tracking_v27_pmUXGZ/build_01/libtracking_batch_v27.so"),
              "bdabc75a633da7cb72c0565a3d2b87dcb6662b17b12ba96ccd7a7c586bebb644"),
}
SCHEMA = "seaqr.tracker-baseline-jetson.v1"
FREEZE_SCHEMA = "tracker_baseline_jetson.v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def load_path(name, path, expected):
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, "module identity differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workspace_guard(workspace):
    workspace = Path(workspace)
    require(re.fullmatch(PATTERN, str(workspace)) and workspace.is_dir() and not workspace.is_symlink()
            and os.geteuid() != 0, "real unprivileged fixed-scope workspace required")
    return workspace


def bundle(workspace, expected_freeze_sha):
    freeze_path = workspace / "freeze.json"
    require(isinstance(expected_freeze_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha)
            and freeze_path.is_file() and not freeze_path.is_symlink()
            and sha(freeze_path) == expected_freeze_sha, "caller-bound freeze identity differs")
    base = load_path("frozen_tracker_capacity_base", workspace / "check_tracker_capacity_shadow.py", BASE_SHA)
    freeze = base.read(freeze_path)
    files = freeze.get("files", {})
    required = {"check_tracker_capacity_shadow.py", "check_tracker_capacity_jetson.py", "batch_discovery_pair.py"}
    require(freeze.get("schema") == FREEZE_SCHEMA and freeze.get("baseline_only") is True
            and set(freeze.get("sources", {})) == set(base.CLIPS)
            and required <= set(files) and 3 <= len(files) <= 8, "freeze scope differs")
    require(freeze["sources"] == {c: {"frames": 673, "source_sha256": base.SOURCE_SHA[c],
            "journal_sha256": base.FILES[c]["run/frames.jsonl"]} for c in base.CLIPS}, "source metadata differs")
    hashes = {}
    base.bind(freeze_path, expected_freeze_sha, hashes)
    for name, expected in files.items():
        require(Path(name).name == name and name not in (".", "..", "freeze.json")
                and Path(name).suffix == ".py" and re.fullmatch(r"[0-9a-f]{64}", expected), "unsafe bundle member")
        base.bind(workspace / name, expected, hashes)
    require(files["check_tracker_capacity_shadow.py"] == BASE_SHA
            and files["batch_discovery_pair.py"] == SAFETY_SHA
            and sha(Path(__file__).resolve()) == files["check_tracker_capacity_jetson.py"], "executed bundle differs")
    base.EVIDENCE = TRACE  # Only input location changes; replay_rows/run_clip stay byte-identical.
    return base, freeze, hashes


def dependencies(base, hashes):
    launches, receipts = {}, {}
    base.bind(CONFIG, base.CONFIG_SHA, hashes)
    configuration = base.read(CONFIG)
    for name, path in ADAPTER_PATHS.items():
        base.bind(path, base.ADAPTERS[name], hashes)
    base.bind(PROFILER, PROFILER_SHA, hashes)
    for path, expected in LIBRARIES.values():
        base.bind(path, expected, hashes)
    for clip in base.CLIPS:
        for relative, expected in base.FILES[clip].items():
            base.bind(TRACE / clip / relative, expected, hashes)
        launch = base.read(TRACE / clip / "run/launch.json")
        receipt = base.read(TRACE / clip / "execution_receipt.json")
        require(receipt["passed"] is True and receipt["non_timing_journal_parity_passed"] is True
                and receipt["processed_frames"] == 673 and receipt["tracker_configuration_changed"] is False,
                "original execution not complete/unchanged")
        require(launch["source_sha256"] == base.SOURCE_SHA[clip]
                and launch["configuration"] == configuration and launch["config_sha256"] == base.CONFIG_SHA
                and launch["fps"] == 10 and launch["expected_frames"] == 673 and launch["max_frames"] is None,
                "frozen source/configuration differs")
        require(receipt["journal_sha256"] == base.FILES[clip]["run/frames.jsonl"]
                and receipt["launch_sha256"] == base.FILES[clip]["run/launch.json"]
                and receipt["tracking_transformed_sha256"] == base.METHOD_SHA, "receipt identity differs")
        for name, path in ADAPTER_PATHS.items():
            require(receipt["adapters"][name] == dict(path=str(path), sha256=base.ADAPTERS[name]), "adapter provenance differs")
        for path, expected in LIBRARIES.values():
            require(receipt["libraries"][str(path)] == expected, "native provenance differs")
        for relative, expected in launch["package_sha256"].items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe package path")
            base.bind(RUNTIME / "tiny_target" / relative, expected, hashes)
        launches[clip], receipts[clip] = launch, receipt
    require(launches["0170"]["package_sha256"] == launches["0240"]["package_sha256"], "package differs across clips")
    paths = [RUNTIME, *dict.fromkeys(p.parent for p in ADAPTER_PATHS.values()), PROFILER.parent]
    sys.path[:0] = [str(p) for p in paths]
    package = importlib.import_module("tiny_target")
    require(Path(package.__file__).resolve() == RUNTIME / "tiny_target/__init__.py", "wrong package import")
    for name, path in ADAPTER_PATHS.items():
        imported = importlib.import_module(name)
        require(Path(imported.__file__).resolve() == path, "wrong adapter import")
    profiler = importlib.import_module("profile_visible_interaction_v30")
    require(Path(profiler.__file__).resolve() == PROFILER, "wrong runtime helper import")
    libraries = {name: path for name, (path, _) in LIBRARIES.items()}
    base.make_adapter(libraries)  # Source transformation proof only; no frame processing.
    return launches, receipts, libraries, profiler


def check_runtime(actual, expected, *, after=False):
    keys = ("blas", "affinity", "numpy", "opencv", "thread_environment", "clock_ticks")
    require(all(actual.get(key) == expected.get(key) for key in keys), "original numerical runtime differs")
    require(actual["numpy"] == "1.26.1" and actual["opencv"] == "4.10.0"
            and len(actual["blas"]) == 1 and actual["blas"][0]["threads"] == 12
            and actual["opencv_threads"] == (2 if after else 12), "original numerical/thread contract differs")
    require(sys.version_info[:2] == (3, 10) and sys.platform == "linux" and platform.machine() == "aarch64",
            "original Python3.10/Linux/aarch64 required")
    require((2*math.log1p(2)).hex() == "0x1.193ea7aad030ap+1", "original scalar maturity-prior arithmetic differs")


def clock_policy():
    """Read control settings, never instantaneous DVFS values or write controls."""
    controls = {
        **{f"cpu{i}": (Path(f"/sys/devices/system/cpu/cpufreq/policy{i}"),
                       ("scaling_min_freq", "scaling_max_freq", "scaling_governor")) for i in (0, 4, 8)},
        "gpu": (Path("/sys/class/devfreq/17000000.gpu"), ("min_freq", "max_freq", "governor")),
    }
    result = {}
    for label, (directory, names) in controls.items():
        values = [(directory/name).read_text().strip() for name in names]
        result[label] = dict(minimum=int(values[0]), maximum=int(values[1]), governor=values[2])
    return result


def child(workspace, *, freeze_sha, preflight=False, clip=None):
    workspace = workspace_guard(workspace)
    require(preflight != (clip in ("0170", "0240")), "exactly one preflight/run-clip mode required")
    output = workspace if preflight else workspace / clip
    receipt_path = output / ("preflight.json" if preflight else "result.json")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "refusing existing child receipt")
    if not preflight:
        require(not output.exists() and not output.is_symlink(), "refusing existing clip output")
        output.mkdir()
    began = time.monotonic()
    result = dict(schema=SCHEMA + (".preflight" if preflight else ".run"), passed=False,
                  baseline_only=True, candidate_implemented=False, detector_replayed=False,
                  source_media_opened=False, raw16_or_holdouts_accessed=False, native_build_performed=False,
                  original_native_binaries=True, original_archive_internal_state_parity_claimed=False,
                  workspace=str(workspace), freeze_sha256=freeze_sha, clip=clip, error=None,
                  note="Original observable tracks/metrics exact; dual-replay state and archive-derived learning checks only. No changed-policy candidate or full detector replay.")
    hashes, base = {}, None
    try:
        base, freeze, hashes = bundle(workspace, freeze_sha)
        launches, receipts, libraries, profiler = dependencies(base, hashes)
        result["inputs_sha256"] = hashes.copy()
        result["clock_policy_before"] = clock_policy()
        before = profiler.runtime_info()
        for reference in receipts.values():
            check_runtime(before, reference["runtime_before"])
        result["runtime_before"] = before
        result["python_version"] = sys.version
        result["tracking_transformed_sha256"] = base.METHOD_SHA
        if preflight:
            result["frames_processed"] = 0
            result["passed"] = True
        else:
            pre_path = workspace / "preflight.json"
            pre = base.read(pre_path)
            require(pre["passed"] is True and pre["schema"] == SCHEMA+".preflight"
                    and pre["workspace"] == str(workspace) and pre["inputs_sha256"] == hashes
                    and pre["runtime_before"] == before and pre["freeze_sha256"] == freeze_sha,
                    "matching completed preflight required")
            base.bind(pre_path, sha(pre_path), hashes)
            if clip == "0240":
                first = base.read(workspace / "0170/result.json")
                require(first["passed"] is True and first["clip"] == "0170"
                        and first["freeze_sha256"] == freeze_sha
                        and first["replay"]["exact_archive_frames"] == 673
                        and first["tracking_transformed_sha256"] == base.METHOD_SHA
                        and first["preflight_sha256"] == hashes[str(pre_path)], "0170 full baseline parity required before0240")
                base.bind(workspace / "0170/result.json", sha(workspace / "0170/result.json"), hashes)
            result["preflight_sha256"] = hashes[str(pre_path)]
            import cv2
            cv2.setNumThreads(launches[clip]["configuration"]["opencv_threads"])
            result["process_local_opencv_threads_set_to_original"] = 2
            result["replay"] = base.run_clip(clip, launches[clip], libraries, output)
            after = profiler.runtime_info()
            check_runtime(after, receipts[clip]["runtime_after"], after=True)
            result["runtime_after"] = after
            result["passed"] = result["replay"]["passed"]
    except BaseException as exc:
        result["passed"] = False
        result["error"] = repr(exc)
    finally:
        result["inputs_sha256"] = hashes.copy()
        if base is not None and hashes:
            try:
                base.verify_unchanged(hashes)
                result["inputs_unchanged_after_check"] = True
            except BaseException as exc:
                result["passed"] = False
                result["postcheck_error"] = repr(exc)
                result["inputs_unchanged_after_check"] = False
        if "clock_policy_before" in result:
            try:
                result["clock_policy_after"] = clock_policy()
                result["clock_controls_unchanged"] = result["clock_policy_after"] == result["clock_policy_before"]
                require(result["clock_controls_unchanged"], "clock controls changed")
            except BaseException as exc:
                result["passed"] = False
                result["clock_postcheck_error"] = repr(exc)
        result["elapsed_s"] = time.monotonic()-began
        with receipt_path.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(passed=result["passed"], clip=clip, preflight=preflight,
                         result=str(receipt_path), sha256=sha(receipt_path), error=result["error"])), flush=True)
    return result


def verify_child_receipt(base, workspace, phase_name, bundle_hashes):
    preflight = phase_name == "preflight"
    clip = None if preflight else phase_name.removeprefix("run_")
    path = workspace / "preflight.json" if preflight else workspace / clip / "result.json"
    result = base.read(path)
    require(result["schema"] == SCHEMA+(".preflight" if preflight else ".run")
            and result["passed"] is True and result["workspace"] == str(workspace) and result["clip"] == clip
            and result["freeze_sha256"] == bundle_hashes.get(str(workspace/"freeze.json"))
            and result["inputs_unchanged_after_check"] is True and result["clock_controls_unchanged"] is True
            and result["candidate_implemented"] is False and result["detector_replayed"] is False
            and result["source_media_opened"] is False and result["native_build_performed"] is False
            and result["original_native_binaries"] is True and result["tracking_transformed_sha256"] == base.METHOD_SHA,
            "incomplete or incorrect child receipt")
    inputs = result["inputs_sha256"]
    require(all(inputs.get(path) == expected for path, expected in bundle_hashes.items()), "child bundle provenance differs")
    for source in base.CLIPS:
        require(all(inputs.get(str(TRACE/source/relative)) == expected for relative, expected in base.FILES[source].items()),
                "child journal provenance differs")
    artifacts = {str(path): sha(path)}
    if preflight:
        require(result["frames_processed"] == 0, "preflight processed frames")
    else:
        pre_path = workspace / "preflight.json"
        require(result["preflight_sha256"] == inputs.get(str(pre_path)) == sha(pre_path), "child preflight link differs")
        replay = result["replay"]
        require(replay["passed"] is True and replay["first_difference"] is None
                and all(replay[key] == 673 for key in ("attempted_frames", "exact_archive_frames", "dual_state_exact_frames", "derived_learning_exact_frames")),
                "child did not prove complete exact baseline")
        state_path = workspace / clip / (clip+"_state_hashes.jsonl")
        require(replay["state_hashes_file"] == str(state_path), "unexpected state artifact path")
        base.bind(state_path, replay["state_hashes_sha256"], artifacts)
        require(len(replay["adapter_runs"]) == 2 and all(all(run[key] == 0 for key in (
            "geometry_calls", "geometry_fallbacks", "batch_fallbacks", "innovation_fallbacks"))
            and run["batch_calls"] > 0 and run["batch_track_rows"] == run["innovation_tracks"] for run in replay["adapter_runs"]),
            "child adapter execution/fallback differs")
        if clip == "0240":
            first = workspace / "0170/result.json"
            require(inputs.get(str(first)) == sha(first), "first-clip gate evidence differs")
    return artifacts


def supervise(workspace, freeze_sha):
    workspace = workspace_guard(workspace)
    base, _, hashes = bundle(workspace, freeze_sha)  # This module has no eager numerical imports.
    safety = load_path("frozen_tracker_safety", workspace / "batch_discovery_pair.py", SAFETY_SHA)
    for name in ("batch_status.json", "batch_status.tmp", "batch.lock", "telemetry.jsonl", "preflight.json", "0170", "0240"):
        require(not (workspace/name).exists() and not (workspace/name).is_symlink(), "existing batch evidence: " + name)
    began, process = time.monotonic(), None
    artifacts = {}
    status = dict(schema=SCHEMA+".batch", complete=False, passed=False, phases=[], current=None, error=None,
                  started_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
                  freeze_sha256=freeze_sha,
                  baseline_only=True, candidate_implemented=False, clock_writes_performed=False,
                  maximum_clip_seconds=900, maximum_total_seconds=1800, sources=["0170", "0240"])

    def save():
        status["elapsed_s"] = time.monotonic()-began
        temporary = workspace / "batch_status.tmp"
        base.write_new(temporary, status)
        os.replace(temporary, workspace / "batch_status.json")

    def interrupted(sig, frame):
        raise RuntimeError("supervisor interrupted by signal " + str(sig))

    with (workspace/"batch.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            save()
            status["clock_policy_before"] = clock_policy()
            with (workspace/"telemetry.jsonl").open("x") as telemetry:
                for phase_name, arguments, deadline in (("preflight", ["--preflight"], 300),
                        ("run_0170", ["--clip", "0170"], 900), ("run_0240", ["--clip", "0240"], 900)):
                    base.verify_unchanged(hashes)
                    require(max(safety.temperatures().values()) < 65, "too warm to start")
                    require(time.monotonic()-began < 1800, "batch deadline exceeded")
                    command = [sys.executable, "-I", "-u", str(workspace/"check_tracker_capacity_jetson.py"),
                               "--workspace", str(workspace), "--freeze-sha256", freeze_sha, *arguments]
                    phase = dict(name=phase_name, command=command, returncode=None)
                    status["phases"].append(phase)
                    status["current"] = phase_name
                    start = time.monotonic()
                    with (workspace/(phase_name+".log")).open("x") as log:
                        process = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL, start_new_session=True)
                        phase["pid"] = process.pid
                        save()
                        print(json.dumps(dict(phase=phase_name, status="starting")), flush=True)
                        while process.poll() is None:
                            temperatures = safety.temperatures()
                            telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                phase=phase_name, temperatures_c=temperatures))+"\n")
                            telemetry.flush()
                            require(max(temperatures.values()) < 75, "temperature stop threshold reached")
                            require(time.monotonic()-start < deadline, "phase deadline exceeded")
                            require(time.monotonic()-began < 1800, "batch deadline exceeded")
                            time.sleep(2)
                        phase.update(returncode=process.returncode, elapsed_s=time.monotonic()-start)
                        require(max(safety.temperatures().values()) < 75, "temperature stop threshold reached after child exit")
                        require(time.monotonic()-start < deadline, "phase deadline exceeded after child exit")
                        require(time.monotonic()-began < 1800, "batch deadline exceeded after child exit")
                        require(process.returncode == 0, "failed phase " + phase_name)
                        artifacts.update(verify_child_receipt(base, workspace, phase_name, hashes))
                        status["child_artifacts_sha256"] = artifacts.copy()
                    process = None
                    save()
                    print(json.dumps(dict(phase=phase_name, status="complete")), flush=True)
            base.verify_unchanged(hashes)
            base.verify_unchanged(artifacts)
            status.update(complete=True, passed=True, current=None)
        except BaseException as exc:
            status["error"] = repr(exc)
            if process is not None:
                safety.stop_owned(process)
                status["owned_child_stopped"] = True
        finally:
            try:
                base.verify_unchanged(hashes)
                base.verify_unchanged(artifacts)
                status["bundle_and_child_artifacts_unchanged"] = True
            except BaseException as exc:
                status["passed"] = False
                status["postcheck_error"] = repr(exc)
                status["bundle_and_child_artifacts_unchanged"] = False
            if "clock_policy_before" in status:
                try:
                    status["clock_policy_after"] = clock_policy()
                    status["clock_controls_unchanged"] = status["clock_policy_after"] == status["clock_policy_before"]
                    require(status["clock_controls_unchanged"], "clock controls changed")
                except BaseException as exc:
                    status["passed"] = False
                    status["clock_postcheck_error"] = repr(exc)
            save()
            for sig, previous in old.items():
                signal.signal(sig, previous)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--supervise", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--clip", choices=("0170", "0240"))
    args = parser.parse_args()
    result = supervise(args.workspace, args.freeze_sha256) if args.supervise else child(
        args.workspace, freeze_sha=args.freeze_sha256, preflight=args.preflight, clip=args.clip)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
