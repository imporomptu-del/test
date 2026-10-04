"""Bounded passive detector-origin replay; no candidate algorithm or CUDA changes.

The full causal 0240 U8 clip is replayed. Only declared diagnostic patches are
saved; every non-timing journal field must match the archived selection run.
This is forensic evidence, not a speed benchmark or accuracy improvement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

SCHEMA = "seaqr.nuisance-origin.run.v1"
PATTERN = r"/tmp/seaqr_nuisance_origin_20261001_[A-Za-z0-9]{6}"
TRACE = Path("/tmp/seaqr_feature_residual_trace_20260930_1lWybU")
TRACE_SHA = "5e2c6f6d0d946e326ab348a747ffd5987dbca97865f3eed49989248383e2b15a"
TRACE_FREEZE = "dd3a7f4a04bbafb2f68161794b70606060f9ae25e3b968144e4ac4c3b72d1fe6"
JOURNAL_SHA = "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
MEMBERS = {"run_nuisance_origin_v1.py", "nuisance_origin_capture_v1.py",
           "nuisance_origin_packet_v1.py", "packet_plan.json", "batch_discovery_pair.py",
           "test_nuisance_origin_capture_v1.py", "test_nuisance_origin_packet_v1.py",
           "test_nuisance_origin_runner_v1.py"}
FRAMES = sorted(i for anchor in (60, 75, 90, 435, 450) for i in range(anchor - 2, anchor + 3))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/linked metadata: " + str(path))
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        value = float(value)
        require(math.isfinite(value), "nonfinite JSON number")
        return value
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bound(path, digest):
    require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == digest,
            "input changed: " + str(path))


def load(path, name):
    """Register before execution so class source inspection has a real module."""
    path = Path(path)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__) == path, "module collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def bundle(workspace, digest):
    workspace = Path(workspace)
    require(re.fullmatch(PATTERN, str(workspace)) and workspace.is_dir()
            and not workspace.is_symlink() and os.geteuid() != 0, "invalid unprivileged workspace")
    require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest), "caller freeze required")
    bound(workspace / "freeze.json", digest)
    freeze = read(workspace / "freeze.json")
    require(freeze.get("schema") == SCHEMA + ".freeze" and set(freeze.get("files", {})) == MEMBERS,
            "frozen file inventory differs")
    for name, value in freeze["files"].items():
        require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "invalid file digest")
        bound(workspace / name, value)
    bound(workspace / "batch_discovery_pair.py", SAFETY_SHA)
    require(Path(__file__).resolve() == workspace / "run_nuisance_origin_v1.py", "wrong executing runner")
    packet = load(workspace / "nuisance_origin_packet_v1.py", "nuisance_packet_frozen")
    plan = read(workspace / "packet_plan.json")
    packet.validate_plan(plan)
    require(sorted(map(int, plan["frame_points"])) == FRAMES, "capture frame scope changed")
    require(all(1 <= len(points) <= 4 for points in plan["frame_points"].values()), "capture point bound changed")
    return freeze, plan


def dependencies(workspace, digest):
    freeze, plan = bundle(workspace, digest)
    bound(TRACE / "freeze.json", TRACE_FREEZE)
    bound(TRACE / "run_feature_residual_trace.py", TRACE_SHA)
    bound(TRACE / "0240/run/frames.jsonl", JOURNAL_SHA)
    trace = load(TRACE / "run_feature_residual_trace.py", "nuisance_frozen_trace")
    selection = trace.load_selection()
    baseline = selection.load_baseline()
    reference, runtime, hashes, _ = trace.inputs(TRACE, "0240", selection, baseline)
    helper = baseline.load_helper()
    modules, identities = helper.dependencies(reference)
    return SimpleNamespace(freeze=freeze, plan=plan, trace=trace, selection=selection,
        baseline=baseline, reference=reference, runtime=runtime, hashes=hashes,
        helper=helper, modules=modules, identities=identities)


def run(workspace, digest, preflight=False):
    workspace = Path(workspace)
    # Reject an existing result before executing any detector code.
    output = workspace / ("preflight.json" if preflight else "result.json")
    bundle(workspace, digest)
    require(not output.exists() and not output.is_symlink(), "existing result; no retry/overwrite")
    if not preflight:
        require(not (workspace / "run").exists() and not (workspace / "capture.json").exists()
                and not (workspace / "parity.json").exists(), "existing replay evidence")
    receipt = dict(schema=SCHEMA, passed=False, preflight=preflight, freeze_sha256=digest, workspace=str(workspace),
        source_clip="0240", full_frames=673, algorithm_changed=False, production_changed=False,
        raw16_accessed=False, sealed_holdouts_accessed=False, source_class_inferred=False,
        performance_benchmark=False, clocks_changed=None, error=None)
    try:
        dep = dependencies(workspace, digest)
        helper, modules, selection, baseline = dep.helper, dep.modules, dep.selection, dep.baseline
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, dep.runtime)
        motion, global_config = selection.configurations(helper)
        receipt.update(inputs=dep.hashes, identities=dep.identities, runtime_before=before,
                       clock_policy_before=clocks, source=dep.trace.source_spec("0240"))
        if preflight:
            from tiny_target.frame_source import probe_video
            import cv2
            probe = probe_video(receipt["source"]["path"])
            baseline.validate_probe(probe)
            cv2.setNumThreads(2)
            receipt.update(probe=probe.to_dict(), controls=[], feature_adapter={})
            receipt["cpu_parity"] = selection.generated_cpu_parity(motion)
            receipt["conversion"] = selection.conversion_check()
            with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
                selection.run_controls(candidate, motion, global_config, receipt["controls"])
        else:
            pre = read(workspace / "preflight.json")
            require(pre.get("passed") is True and pre.get("preflight") is True
                    and pre.get("freeze_sha256") == digest and pre.get("inputs") == dep.hashes
                    and pre.get("identities") == dep.identities and pre.get("clock_policy_before") == clocks,
                    "successful matching preflight required")
            capture_module = load(workspace / "nuisance_origin_capture_v1.py", "nuisance_capture_frozen")
            capture = capture_module.OriginCapture(dep.plan)
            front = modules["visible_front_v26"]
            changed = dict(modules, visible_front_v26=SimpleNamespace(
                ResidentFrontV26=capture.front_class(front.ResidentFrontV26), attach_warp=front.attach_warp))
            receipt["feature_adapter"] = {}
            with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
                changed["motion_reuse_v12"] = SimpleNamespace(ReuseMotionV12=candidate)
                report = baseline.execute_baseline(helper, changed, Path(receipt["source"]["path"]),
                                                   workspace / "run", receipt)
            baseline.validate_output(report, read(workspace / "run/launch.json"), dep.reference,
                                     "0240", receipt["decoded_frames_verified"])
            selection.validate_adapter_roundtrip(receipt["feature_adapter"], pre["feature_adapter"])
            evidence = capture.finish()
            write(workspace / "capture.json", evidence)
            parity = dep.trace.journal_parity(TRACE / "0240/run/frames.jsonl", workspace / "run/frames.jsonl")
            write(workspace / "parity.json", parity)
            receipt.update(capture_sha256=sha(workspace / "capture.json"), parity_sha256=sha(workspace / "parity.json"),
                journal_sha256=sha(workspace / "run/frames.jsonl"), report_sha256=sha(workspace / "run/report.json"),
                launch_sha256=sha(workspace / "run/launch.json"), non_timing_journal_parity=parity["passed"],
                preflight_sha256=sha(workspace / "preflight.json"))
            require(parity["passed"] and parity["original_journal_sha256"] == JOURNAL_SHA,
                    "full journal parity failed; capture is not interpretable as original")
        after, clock_after = info(), helper.clock_policy_snapshot()
        helper.runtime_check(after, dep.runtime, after=True)
        require(clock_after == clocks, "clock controls changed")
        final = dependencies(workspace, digest)
        require(final.hashes == dep.hashes and final.identities == dep.identities, "dependencies changed")
        receipt.update(passed=True, runtime_after=after, clock_policy_after=clock_after, clocks_changed=False)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(output, receipt)
    print(json.dumps(dict(passed=True, preflight=preflight, full_frames=0 if preflight else 673)), flush=True)


def validate_child(workspace, digest, mode):
    """Bind completed children and recheck their immutable evidence at batch end."""
    workspace = Path(workspace)
    require(mode in {"preflight", "run"}, "unknown child mode")
    path = workspace / ("preflight.json" if mode == "preflight" else "result.json")
    row = read(path)
    require(row.get("schema") == SCHEMA and row.get("passed") is True and row.get("error") is None
            and row.get("workspace") == str(workspace) and row.get("freeze_sha256") == digest
            and row.get("preflight") is (mode == "preflight") and row.get("source_clip") == "0240"
            and row.get("full_frames") == 673 and row.get("clocks_changed") is False
            and row.get("clock_policy_before") == row.get("clock_policy_after"), "invalid child receipt")
    require(all(row.get(key) is False for key in ("algorithm_changed", "production_changed", "raw16_accessed",
        "sealed_holdouts_accessed", "source_class_inferred", "performance_benchmark")), "child scope differs")
    require(row.get("source") == dict(path="/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0240.avi",
        sha256="2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585", frames=673,
        width=4784, height=3190, fps=10, codec="mjpeg", pixel_format="yuvj420p"), "child source differs")
    hashes = {str(path): sha(path)}
    if mode == "run":
        require(row.get("processed_frames") == row.get("decoded_frames_verified") == 673
                and row.get("non_timing_journal_parity") is True, "incomplete causal/parity result")
        names = {"capture": "capture.json", "parity": "parity.json", "journal": "run/frames.jsonl",
                 "report": "run/report.json", "launch": "run/launch.json", "preflight": "preflight.json"}
        for key, relative in names.items():
            file = workspace / relative
            bound(file, row.get(key + "_sha256"))
            hashes[str(file)] = sha(file)
        parity = read(workspace / "parity.json")
        require(parity.get("passed") is True and parity.get("rows_compared") == parity.get("expected_frames") == 673
            and parity.get("original_journal_sha256") == JOURNAL_SHA
            and parity.get("diagnostic_journal_sha256") == row["journal_sha256"]
            and parity.get("numeric_tolerance") is False and parity.get("mismatch_frames") == [],
            "invalid complete parity receipt")
    return hashes


def batch(workspace, digest):
    workspace = Path(workspace)
    bundle(workspace, digest)
    safety = load(workspace / "batch_discovery_pair.py", "nuisance_safety_frozen")
    for name in ("batch.lock", "batch_status.json", "batch_status.tmp", "telemetry.jsonl", "tests.log", "preflight.log", "run.log"):
        require(not (workspace / name).exists() and not (workspace / name).is_symlink(), "existing batch evidence")
    status = dict(schema=SCHEMA + ".batch", complete=False, passed=False, error=None,
        started_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=digest, supervisor_pid=os.getpid(),
        phases=[], current=None, workers=1, automatic_retries=0, production_changed=False)
    began, child, completed_hashes = time.monotonic(), None, {}

    def save():
        status["elapsed_s"] = time.monotonic() - began
        write(workspace / "batch_status.tmp", status)
        os.replace(workspace / "batch_status.tmp", workspace / "batch_status.json")

    def interrupt(sig, frame):
        raise RuntimeError("supervisor signal " + str(sig))

    commands = [("tests", 120, [sys.executable, "-m", "unittest", "-q",
        "test_nuisance_origin_capture_v1", "test_nuisance_origin_packet_v1", "test_nuisance_origin_runner_v1"])]
    commands += [(mode, limit, [sys.executable, "-I", "-u", str(workspace / "run_nuisance_origin_v1.py"),
        "--workspace", str(workspace), "--freeze-sha256", digest, "--" + mode])
        for mode, limit in (("preflight", 300), ("run", 1800))]
    with (workspace / "batch.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            save()
            with (workspace / "telemetry.jsonl").open("x") as telemetry:
                for name, limit, command in commands:
                    bundle(workspace, digest)
                    require(max(safety.temperatures().values()) < 65, "too warm to start")
                    require(time.monotonic() - began < 2400, "batch deadline before launch")
                    phase = dict(name=name, command=command, returncode=None, deadline_s=limit)
                    status["phases"].append(phase)
                    status["current"] = name
                    start = time.monotonic()
                    with (workspace / (name + ".log")).open("x") as log:
                        child = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
                        phase["pid"] = child.pid
                        save()
                        while child.poll() is None:
                            temps = safety.temperatures()
                            telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                phase=name, temperatures_c=temps)) + "\n")
                            telemetry.flush()
                            require(max(temps.values()) < 75, "temperature stop limit")
                            require(time.monotonic() - start < limit and time.monotonic() - began < 2400,
                                    "diagnostic deadline reached")
                            time.sleep(2)
                        phase.update(returncode=child.returncode, elapsed_s=time.monotonic() - start)
                        final_temps = safety.temperatures()
                        telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                            phase=name, final_sample=True, temperatures_c=final_temps)) + "\n")
                        telemetry.flush()
                        require(max(final_temps.values()) < 75, "temperature stop limit after exit")
                        require(phase["elapsed_s"] < limit and time.monotonic() - began < 2400,
                                "diagnostic deadline after exit")
                        require(child.returncode == 0, "failed child " + name)
                    child = None
                    if name != "tests":
                        new_hashes = validate_child(workspace, digest, name)
                        require(all(path not in completed_hashes or completed_hashes[path] == value
                                    for path, value in new_hashes.items()), "completed evidence changed")
                        completed_hashes.update(new_hashes)
                    save()
            bundle(workspace, digest)
            for path, value in completed_hashes.items():
                bound(path, value)
            status.update(complete=True, passed=True, current=None, child_artifacts_sha256=completed_hashes)
        except BaseException as exc:
            status["error"] = repr(exc)
            if child is not None:
                safety.stop_owned(child)
            raise
        finally:
            save()
            for sig, handler in old.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--batch", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    batch(args.workspace, args.freeze_sha256) if args.batch else run(args.workspace, args.freeze_sha256, args.preflight)
