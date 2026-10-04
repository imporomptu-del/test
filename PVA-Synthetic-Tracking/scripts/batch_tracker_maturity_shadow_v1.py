#!/usr/bin/env python3
"""Serial bounded saved-proposal shadow; no camera media or hardware writes."""
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

SCHEMA = "seaqr.tracker-maturity-shadow.batch.v1"
PATTERN = r"/tmp/seaqr_tracker_maturity_20261001_[A-Za-z0-9]{6}"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
PHASES = (("preflight", ("--preflight",), 300),
          ("run_0170", ("--clip", "0170"), 1800),
          ("run_0240", ("--clip", "0240"), 1800))
EXECUTION = dict(workers=1, children=3, start_below_celsius=65,
                 stop_at_celsius=75, batch_deadline_seconds=3900, automatic_retries=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024*1024, "Invalid metadata file")
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON")
        return value
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda _: require(False, "Nonfinite JSON"))


def imported(path, digest, name):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Source identity differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(workspace, digest):
    workspace = Path(workspace)
    require(re.fullmatch(PATTERN, str(workspace)) and workspace.is_dir()
            and not workspace.is_symlink() and os.geteuid() != 0, "Unprivileged scoped workspace required")
    require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
            and sha(workspace/"freeze.json") == digest, "Caller-bound freeze differs")
    freeze = read(workspace/"freeze.json")
    files = freeze["files"]
    require(files.get(Path(__file__).name) == sha(Path(__file__)), "Supervisor identity differs")
    runner = imported(workspace/"check_tracker_maturity_shadow_v1.py", files.get("check_tracker_maturity_shadow_v1.py"), "maturity_shadow_worker")
    base, checked, hashes = runner.load(workspace, digest)
    require(checked == freeze and files.get("batch_discovery_pair.py") == SAFETY_SHA, "Frozen bundle differs")
    safety = imported(workspace/"batch_discovery_pair.py", SAFETY_SHA, "maturity_shadow_safety")
    return base, runner, safety, hashes


def command(workspace, digest, phase):
    matches = [args for name, args, _ in PHASES if name == phase]
    require(len(matches) == 1, "Unknown phase")
    return [sys.executable, "-I", "-u", str(Path(workspace)/"check_tracker_maturity_shadow_v1.py"),
            "--workspace", str(workspace), "--freeze-sha256", digest, *matches[0]]


def guard(temperatures, now, began, phase_began, deadline, *, starting=False):
    require(isinstance(temperatures, dict) and temperatures and all(math.isfinite(v) for v in temperatures.values()), "Missing/nonfinite temperatures")
    require(max(temperatures.values()) < (65 if starting else 75), "Temperature threshold reached")
    require(now-began < 3900 and now-phase_began < deadline, "Execution deadline exceeded")


def run(workspace, digest):
    workspace = Path(workspace)
    base, runner, safety, hashes = load(workspace, digest)
    reserved = ["batch_status.json", "batch_status.tmp", "batch.lock", "telemetry.jsonl", "preflight.json", "0170", "0240"]
    reserved += [name+".log" for name, _, _ in PHASES]
    require(all(not (workspace/name).exists() and not (workspace/name).is_symlink() for name in reserved), "Existing evidence; no overwrite")
    began, process, artifacts = time.monotonic(), None, {}
    status = dict(schema=SCHEMA, complete=False, execution_passed=False, error=None,
                  started_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
                  freeze_sha256=digest, execution=EXECUTION, phases=[], current=None,
                  not_run=[name for name, _, _ in PHASES], source_media_accessed=False,
                  production_changed=False, shadow_only=True, automatic_retries=0,
                  scientific_improvement_claimed=False)
    def save():
        status["elapsed_seconds"] = time.monotonic()-began
        temporary = workspace/"batch_status.tmp"
        base.write_new(temporary, status)
        os.replace(temporary, workspace/"batch_status.json")
    def interrupted(sig, frame):
        raise RuntimeError("Supervisor interrupted by signal " + str(sig))
    with (workspace/"batch.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            save()
            with (workspace/"telemetry.jsonl").open("x") as telemetry:
                for name, _, deadline in PHASES:
                    base.verify_unchanged(hashes)
                    guard(safety.temperatures(), time.monotonic(), began, time.monotonic(), deadline, starting=True)
                    phase = dict(name=name, command=command(workspace, digest, name), returncode=None)
                    status["phases"].append(phase)
                    status["not_run"].remove(name)
                    status["current"] = name
                    start = time.monotonic()
                    with (workspace/(name+".log")).open("x") as log:
                        process = subprocess.Popen(phase["command"], cwd=workspace, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                        phase["pid"] = process.pid
                        save()
                        print(json.dumps(dict(phase=name, status="starting")), flush=True)
                        while process.poll() is None:
                            temperatures = safety.temperatures()
                            telemetry.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), phase=name,
                                temperatures_c=temperatures), allow_nan=False)+"\n")
                            telemetry.flush()
                            guard(temperatures, time.monotonic(), began, start, deadline)
                            time.sleep(2)
                        phase.update(returncode=process.returncode, elapsed_seconds=time.monotonic()-start)
                        guard(safety.temperatures(), time.monotonic(), began, start, deadline)
                        require(process.returncode == 0, "Failed phase " + name)
                        artifacts.update(runner.validate_child_receipt(workspace, digest, name))
                        status["child_artifacts_sha256"] = dict(artifacts)
                    process = None
                    save()
                    print(json.dumps(dict(phase=name, status="complete")), flush=True)
            require(len({phase["pid"] for phase in status["phases"]}) == 3, "Fresh-process identity differs")
            base.verify_unchanged(hashes)
            base.verify_unchanged(artifacts)
            status.update(complete=True, execution_passed=True, current=None)
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
                status.update(execution_passed=False, postcheck_error=repr(exc), bundle_and_child_artifacts_unchanged=False)
            save()
            for sig, previous in old.items():
                signal.signal(sig, previous)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    result = run(args.workspace, args.freeze_sha256)
    raise SystemExit(0 if result["execution_passed"] else 1)
