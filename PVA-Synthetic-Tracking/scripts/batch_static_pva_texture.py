#!/usr/bin/env python3
"""Four fresh-process generated PVA probes; bounded, no camera input or retries."""
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

SCHEMA = "seaqr.static-pva-texture.batch.v1"
PHASES = (("bridge", "base"), ("bridge", "trace"), ("texture", "base"), ("texture", "trace"))
EXECUTION = dict(workers=1, phases=4, phase_deadline_seconds=900, batch_deadline_seconds=3600,
                 start_below_celsius=65, stop_at_celsius=75, automatic_retries=0)
FILES = {"probe_static_pva_texture.py", "batch_static_pva_texture.py", "test_static_pva_texture.py",
         "test_batch_static_pva_texture.py", "batch_discovery_pair.py"}
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
PHOTO_WORKSPACE = "/tmp/seaqr_motion_photometric_20260930_6RkevU"
PHOTO_RUNNER_SHA = "569220932c8132944120ffc6026f59e01c47cb81cfd11d9c2875f890566f32f1"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5,
                 feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
                 grid_rows=6, grid_cols=8)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024*1024,
            "Missing, linked or oversize metadata")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def number(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON number")
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: require(False, "Nonfinite JSON constant"))


def validate_freeze(freeze):
    require(set(freeze) == {"schema", "files", "cases", "modes", "shape_hw", "photometric_workspace",
                           "photometric_runner_sha256", "candidate", "no_preflight_pva_calls", "execution"},
            "Freeze fields differ")
    require(freeze["schema"] == "static_pva_texture.v1" and freeze["cases"] == ["bridge", "texture"]
            and freeze["modes"] == ["base", "trace"] and freeze["shape_hw"] == [512, 640]
            and freeze["photometric_workspace"] == PHOTO_WORKSPACE
            and freeze["photometric_runner_sha256"] == PHOTO_RUNNER_SHA
            and freeze["candidate"] == CANDIDATE and freeze["no_preflight_pva_calls"] is True
            and freeze["execution"] == EXECUTION, "Frozen generated scope differs")
    require(isinstance(freeze["files"], dict) and set(freeze["files"]) == FILES
            and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in freeze["files"].values()),
            "Frozen file inventory differs")
    require(freeze["files"]["batch_discovery_pair.py"] == SAFETY_SHA, "Safety helper differs")


def bundle(workspace, freeze_path, freeze_sha):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(r"/tmp/seaqr_static_pva_texture_20261001_[A-Za-z0-9]{6}", str(workspace))
            and workspace.is_dir() and not workspace.is_symlink() and os.geteuid() != 0,
            "Unprivileged isolated workspace required")
    require(freeze_path == workspace / "freeze.json" and freeze_path.is_file()
            and not freeze_path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", freeze_sha)
            and sha(freeze_path) == freeze_sha, "Caller-bound freeze differs")
    freeze = read(freeze_path)
    validate_freeze(freeze)
    for name, expected in freeze["files"].items():
        path = workspace / name
        require(path.is_file() and not path.is_symlink() and sha(path) == expected, "Bundle changed: " + name)
    require(sha(Path(__file__).resolve()) == freeze["files"]["batch_static_pva_texture.py"], "Supervisor identity differs")
    return freeze


def command(workspace, freeze_sha, case, mode):
    require((case, mode) in PHASES, "Unknown generated phase")
    result = [sys.executable, "-I", str(workspace / "probe_static_pva_texture.py"),
              "--workspace", str(workspace), "--freeze", str(workspace / "freeze.json"),
              "--freeze-sha256", freeze_sha, "--case", case]
    if mode == "trace":
        result.append("--trace")
    return result


def validate_result(result, case, mode, freeze_sha):
    require(result.get("schema") == "seaqr.static-pva-texture.v1"
            and result.get("completed") is True and result.get("passed_integrity") is True
            and result.get("case") == case and result.get("mode") == mode
            and result.get("input_sha256", {}).get("freeze_sha256") == freeze_sha,
            "Child result contract differs")
    require(isinstance(result.get("canonical_nontiming"), dict)
            and result.get("canonical_nontiming_sha256") == canonical_sha(result["canonical_nontiming"]),
            "Child canonical result hash differs")


def compare_pair(base, trace):
    require(base["case"] == trace["case"] and base["mode"] == "base" and trace["mode"] == "trace",
            "Pair roles differ")
    matched = base["canonical_nontiming"] == trace["canonical_nontiming"]
    require(matched == (base["canonical_nontiming_sha256"] == trace["canonical_nontiming_sha256"]),
            "Canonical hash/equality disagree")
    return dict(passed=matched, case=base["case"], exact_nontiming_parity=matched,
                baseline_sha256=base["canonical_nontiming_sha256"], trace_sha256=trace["canonical_nontiming_sha256"],
                timings_excluded=True, trace_may_explain_baseline_only_if_passed=matched)


def guard(temperatures, now, began, phase_began, starting=False):
    require(isinstance(temperatures, dict) and temperatures and all(math.isfinite(v) for v in temperatures.values()),
            "Missing or nonfinite temperature readings")
    require(max(temperatures.values()) < (65 if starting else 75), "Temperature threshold reached")
    require(now - began < 3600 and now - phase_began < 900, "Execution deadline exceeded")


def run(workspace, freeze_path, freeze_sha):
    workspace = Path(workspace)
    freeze = bundle(workspace, freeze_path, freeze_sha)
    names = {"batch.lock", "batch_status.json", "batch_status.tmp", "telemetry.jsonl", "parity.json"}
    names.update(case+"_"+mode+ext for case, mode in PHASES for ext in (".json", ".log"))
    require(all(not (workspace/name).exists() and not (workspace/name).is_symlink() for name in names),
            "Existing evidence: no retry or overwrite")
    spec = importlib.util.spec_from_file_location("static_pva_safety", workspace / "batch_discovery_pair.py")
    safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety)
    lock = (workspace / "batch.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    status = dict(schema=SCHEMA, complete=False, execution_passed=False, parity_passed=False,
        generated_only=True, camera_media_accessed=False, clock_writes=False,
        freeze_sha256=freeze_sha, files=freeze["files"], execution=EXECUTION,
        started_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
        phases=[], current=None, error=None)
    def save():
        status["elapsed_seconds"] = time.monotonic()-began
        temporary = workspace / "batch_status.tmp"
        with temporary.open("x") as stream:
            json.dump(status, stream, indent=2, allow_nan=False)
        os.replace(temporary, workspace / "batch_status.json")
    def interrupted(signum, frame):
        raise RuntimeError("Supervisor signal " + str(signum))
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    child = None
    results, parity = {}, []
    try:
        save()
        with (workspace / "telemetry.jsonl").open("x") as telemetry:
            for case, mode in PHASES:
                require(bundle(workspace, freeze_path, freeze_sha) == freeze, "Bundle changed before phase")
                phase_start = time.monotonic()
                guard(safety.temperatures(), phase_start, began, phase_start, starting=True)
                name = case + "_" + mode
                entry = dict(name=name, case=case, mode=mode, command=command(workspace, freeze_sha, case, mode),
                             pid=None, returncode=None)
                status["phases"].append(entry)
                status["current"] = name
                save()
                with (workspace / (name+".log")).open("x") as log:
                    child = subprocess.Popen(entry["command"], cwd=workspace, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    entry["pid"] = child.pid
                    save()
                    while child.poll() is None:
                        temps = safety.temperatures()
                        telemetry.write(json.dumps(dict(phase=name, utc=datetime.now(timezone.utc).isoformat(),
                                                        temperatures_c=temps))+"\n")
                        telemetry.flush()
                        guard(temps, time.monotonic(), began, phase_start)
                        time.sleep(2)
                    entry.update(returncode=child.returncode, elapsed_seconds=time.monotonic()-phase_start)
                    guard(safety.temperatures(), time.monotonic(), began, phase_start)
                    require(child.returncode == 0, "Child failed; retained evidence, no retries: " + name)
                child = None
                path = workspace / (name+".json")
                result = read(path)
                validate_result(result, case, mode, freeze_sha)
                entry["result_sha256"] = sha(path)
                results[name] = result
                if mode == "trace":
                    parity.append(compare_pair(results[case+"_base"], result))
                    # Keep the other independent fixed case even if scientific parity
                    # fails; do not retune, retry, or interpret a mismatched trace.
                save()
        require(bundle(workspace, freeze_path, freeze_sha) == freeze, "Bundle changed after batch")
        require(len(status["phases"]) == 4 and len({p["pid"] for p in status["phases"]}) == 4,
                "Fresh-process inventory differs")
        for entry in status["phases"]:
            require(sha(workspace / (entry["name"]+".json")) == entry["result_sha256"], "Child evidence changed")
        parity_path = workspace / "parity.json"
        with parity_path.open("x") as stream:
            json.dump(dict(schema=SCHEMA+".parity", cases=parity, passed=all(p["passed"] for p in parity)),
                      stream, indent=2, allow_nan=False)
        status.update(complete=True, execution_passed=True, parity_passed=all(p["passed"] for p in parity),
                      parity_sha256=sha(parity_path), current=None)
    except BaseException as exc:
        status["error"] = repr(exc)
        if child is not None:
            safety.stop_owned(child)
            status["phases"][-1]["returncode"] = child.returncode
            status["owned_child_stopped"] = True
        raise
    finally:
        save()
        lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    result = run(args.workspace, args.freeze, args.freeze_sha256)
    print(json.dumps({key: result[key] for key in ("complete", "execution_passed", "parity_passed")}))
