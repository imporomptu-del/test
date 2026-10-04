#!/usr/bin/env python3
"""One generated-only child, bounded deadlines/thermal checks, no clock writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(workspace, freeze_path, freeze_sha256):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(r"/tmp/seaqr_motion_photometric_20260930_[A-Za-z0-9]{6}", str(workspace))
            and workspace.is_dir() and not workspace.is_symlink() and os.geteuid() != 0,
            "unprivileged real scoped workspace required")
    require(freeze_path == workspace / "freeze.json" and not freeze_path.is_symlink()
            and re.fullmatch(r"[a-f0-9]{64}", freeze_sha256) and sha(freeze_path) == freeze_sha256,
            "explicit freeze identity differs")
    # The caller-supplied hash authenticates this manifest before any import.
    freeze = json.loads(freeze_path.read_text())
    runner = workspace / "run_motion_photometric_controls.py"
    require(runner.is_file() and not runner.is_symlink()
            and sha(runner) == freeze["files"][runner.name], "runner differs from frozen bundle")
    diagnostic = load(runner, "generated_photometric_runner")
    initial = diagnostic.bundle_inputs(workspace, freeze_path, freeze_sha256)
    diagnostic.pinned(Path(__file__), initial["files"]["batch_motion_photometric_controls.py"])
    safety_path = workspace / "batch_discovery_pair.py"
    diagnostic.pinned(safety_path, SAFETY_SHA)
    safety = load(safety_path, "generated_photometric_safety")
    require(max(safety.temperatures().values()) < 65, "too warm to start")
    for name in ("batch_status.json", "batch_status.tmp", "batch.lock", "telemetry.jsonl",
                 "generated_controls.log", "generated_pva_controls.json"):
        require(not (workspace/name).exists() and not (workspace/name).is_symlink(), "existing batch evidence")
    lock = (workspace / "batch.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    command = [sys.executable, "-I", str(runner), "--workspace", str(workspace),
               "--freeze", str(freeze_path), "--freeze-sha256", freeze_sha256]
    phase = dict(name="generated_controls", command=command, pid=None, returncode=None, elapsed_s=None)
    status = dict(schema="seaqr.motion-photometric-controls.batch.v1", complete=False, passed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), input_sha256=initial,
                  freeze_sha256=freeze_sha256, supervisor_pid=os.getpid(), phases=[phase],
                  current=None, error=None, generated_only=True, source_media_accessed=False,
                  clock_writes_performed=False, execution=diagnostic.EXECUTION)
    def save():
        status["elapsed_s"] = time.monotonic()-began
        temporary = workspace / "batch_status.tmp"
        with temporary.open("x") as stream:
            json.dump(status, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, workspace / "batch_status.json")
    def interrupted(signum, frame):
        raise RuntimeError("supervisor interrupted by signal " + str(signum))
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    child = None
    try:
        save()
        require(diagnostic.bundle_inputs(workspace, freeze_path, freeze_sha256) == initial, "bundle changed before phase")
        require(max(safety.temperatures().values()) < 65, "too warm to start phase")
        start = time.monotonic()
        status["current"] = phase["name"]
        with (workspace/"telemetry.jsonl").open("x") as telemetry, (workspace/"generated_controls.log").open("x") as log:
            child = subprocess.Popen(command, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            phase["pid"] = child.pid
            save()
            while child.poll() is None:
                temps = safety.temperatures()
                telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                               phase=phase["name"], temperatures_c=temps))+"\n")
                telemetry.flush()
                require(max(temps.values()) < 75, "temperature stop threshold reached")
                require(time.monotonic()-start < 900, "phase deadline exceeded")
                require(time.monotonic()-began < 3600, "batch deadline exceeded")
                time.sleep(2)
            phase.update(returncode=child.returncode, elapsed_s=time.monotonic()-start)
            require(time.monotonic()-start < 900 and time.monotonic()-began < 3600, "completed after deadline")
            require(max(safety.temperatures().values()) < 75, "temperature stop threshold reached at completion")
            require(child.returncode == 0, "generated diagnostic execution failed; inspect retained case evidence")
        child = None
        require(diagnostic.bundle_inputs(workspace, freeze_path, freeze_sha256) == initial, "bundle changed during phase")
        result_path = workspace / "generated_pva_controls.json"
        result = diagnostic.read(result_path)
        require(result.get("schema") == diagnostic.SCHEMA and result.get("completed") is True
                and result.get("passed_integrity") is True and result.get("execution_passed") is True
                and result.get("generated_only") is True and result.get("source_media_accessed") is False
                and result.get("input_sha256", {}).get("files") == initial["files"]
                and result["input_sha256"].get("freeze_sha256") == freeze_sha256
                and len(result.get("cases", [])) == 20, "child result failed contract")
        status.update(complete=True, passed=True, current=None, result_sha256=sha(result_path),
                      scientific_accuracy_gate_applied=False)
    except BaseException as exc:
        status["error"] = repr(exc)
        if child is not None:
            safety.stop_owned(child)
            phase["returncode"] = child.returncode
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
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    main(args.workspace, args.freeze, args.freeze_sha256)
