#!/usr/bin/env python3
"""Bounded generated-only PVA robustness experiment; no retry or tuning."""
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

SCHEMA = "seaqr.pva-robustness.batch.v1"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
EXECUTION = dict(workers=1, phases=104, phase_deadline_seconds=900,
                 batch_deadline_seconds=3600, start_below_celsius=65,
                 stop_at_celsius=75, automatic_retries=0)


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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


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


def imported(path, digest, name):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest,
            "Pinned executable differs: " + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(workspace, freeze_path, digest):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(r"/tmp/seaqr_pva_robustness_20261001_[A-Za-z0-9]{6}", str(workspace))
            and workspace.is_dir() and not workspace.is_symlink() and os.geteuid() != 0,
            "Unprivileged scoped workspace required")
    require(freeze_path == workspace / "freeze.json" and freeze_path.is_file()
            and not freeze_path.is_symlink() and re.fullmatch(r"[a-f0-9]{64}", digest)
            and sha(freeze_path) == digest, "Caller-bound freeze differs")
    frozen = read(freeze_path)
    files = frozen.get("source_sha256", {})
    require(files.get("batch_pva_robustness.py") == sha(Path(__file__)), "Supervisor identity differs")
    probe = imported(workspace / "probe_pva_robustness.py", files.get("probe_pva_robustness.py"),
                     "robustness_worker")
    require(probe.bundle(workspace, freeze_path, digest) == frozen, "Worker bundle differs")
    require(files.get("batch_discovery_pair.py") == SAFETY_SHA, "Safety helper identity differs")
    safety = imported(workspace / "batch_discovery_pair.py", SAFETY_SHA, "robustness_safety")
    return frozen, probe, safety


def phases(cases):
    ids = [item["id"] for item in cases]
    require(len(ids) == 26 and len(set(ids)) == 26
            and all(re.fullmatch(r"[a-zA-Z0-9_]+", case) for case in ids), "Case inventory differs")
    return [(case, depth, mode) for case in ids for depth in (4, 2) for mode in ("base", "trace")]


def phase_name(case, depth, mode):
    return case + "_depth" + str(depth) + "_" + mode


def command(workspace, digest, case, depth, mode):
    require(re.fullmatch(r"[a-zA-Z0-9_]+", case) and type(depth) is int and depth in (4, 2)
            and mode in ("base", "trace"), "Invalid phase role")
    result = [sys.executable, "-I", str(Path(workspace) / "probe_pva_robustness.py"),
              "--workspace", str(workspace), "--freeze", str(Path(workspace) / "freeze.json"),
              "--freeze-sha256", digest, "--case", case, "--depth", str(depth)]
    if mode == "trace":
        result.append("--trace")
    return result


def validate_result(row, case, depth, mode, digest):
    require(row.get("schema") == "seaqr.pva-robustness.v1" and row.get("completed") is True
            and row.get("passed_integrity") is True and row.get("case") == case
            and row.get("depth") == depth and row.get("mode") == mode
            and row.get("input_sha256", {}).get("freeze_sha256") == digest,
            "Child result contract differs")
    canonical = row.get("canonical_nontiming")
    require(isinstance(canonical, dict) and canonical.get("effective_motion_configuration", {}).get("pyramid_levels") == depth
            and canonical_sha(canonical) == row.get("canonical_nontiming_sha256"), "Canonical output differs")


def compare_pair(base, trace):
    require(base["case"] == trace["case"] and base["depth"] == trace["depth"]
            and base["mode"] == "base" and trace["mode"] == "trace", "Pair roles differ")
    matched = base["canonical_nontiming"] == trace["canonical_nontiming"]
    require(matched == (base["canonical_nontiming_sha256"] == trace["canonical_nontiming_sha256"]),
            "Canonical hash/equality disagree")
    return dict(case=base["case"], depth=base["depth"], passed=matched, exact_nontiming_parity=matched,
                baseline_sha256=base["canonical_nontiming_sha256"], trace_sha256=trace["canonical_nontiming_sha256"],
                timings_excluded=True, trace_may_explain_baseline_only_if_passed=matched)


def guard(temperatures, now, began, phase_began, starting=False):
    require(isinstance(temperatures, dict) and temperatures
            and all(math.isfinite(v) for v in temperatures.values()), "Missing or nonfinite temperatures")
    require(max(temperatures.values()) < (65 if starting else 75), "Temperature threshold reached")
    require(now - began < 3600 and now - phase_began < 900, "Execution deadline exceeded")


def completed_inventory(entries, inventory):
    require([(row["case"], row["depth"], row["mode"]) for row in entries] == inventory
            and len(entries) == 104 and len({row["pid"] for row in entries}) == 104
            and all(type(row["pid"]) is int and row["pid"] > 0 and row["returncode"] == 0
                    for row in entries), "Completed fresh-process inventory differs")


def run(workspace, freeze_path, digest):
    workspace = Path(workspace)
    frozen, probe, safety = load(workspace, freeze_path, digest)
    inventory = phases(probe.inventory())
    names = {"batch.lock", "batch_status.json", "batch_status.tmp", "telemetry.jsonl", "parity.json"}
    names.update(phase_name(*phase)+ext for phase in inventory for ext in (".json", ".log", ".failure.json"))
    require(all(not (workspace/name).exists() and not (workspace/name).is_symlink() for name in names),
            "Existing evidence: no retry or overwrite")
    lock = (workspace / "batch.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    status = dict(schema=SCHEMA, complete=False, execution_passed=False, parity_passed=False,
        generated_only=True, camera_media_accessed=False, clock_writes=False,
        freeze_sha256=digest, source_sha256=frozen["source_sha256"], execution=EXECUTION,
        started_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
        phases=[], current=None, error=None, not_run=[phase_name(*p) for p in inventory])
    def save():
        status["elapsed_seconds"] = time.monotonic()-began
        temporary = workspace / "batch_status.tmp"
        with temporary.open("x") as stream:
            json.dump(status, stream, indent=2, allow_nan=False)
        os.replace(temporary, workspace / "batch_status.json")
    def interrupted(signum, frame):
        raise RuntimeError("Supervisor signal " + str(signum))
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    child, base = None, None
    parity = []
    try:
        save()
        with (workspace / "telemetry.jsonl").open("x") as telemetry:
            for case, depth, mode in inventory:
                require(probe.bundle(workspace, freeze_path, digest) == frozen, "Bundle changed before phase")
                phase_start = time.monotonic()
                guard(safety.temperatures(), phase_start, began, phase_start, starting=True)
                name = phase_name(case, depth, mode)
                entry = dict(name=name, case=case, depth=depth, mode=mode,
                             command=command(workspace, digest, case, depth, mode), pid=None, returncode=None)
                status["phases"].append(entry)
                status["not_run"].remove(name)
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
                    require(child.returncode == 0, "Child failed; retained evidence, no retry: " + name)
                child = None
                path = workspace / (name+".json")
                row = read(path)
                validate_result(row, case, depth, mode, digest)
                entry["result_sha256"] = sha(path)
                if mode == "base":
                    base = row
                else:
                    parity.append(compare_pair(base, row))
                    base = None
                # Scientific failures are results, not permission to retune or retry.
                save()
        require(probe.bundle(workspace, freeze_path, digest) == frozen, "Bundle changed after batch")
        completed_inventory(status["phases"], inventory)
        for entry in status["phases"]:
            require(sha(workspace / (entry["name"]+".json")) == entry["result_sha256"], "Child evidence changed")
        require(len(parity) == 52, "Parity inventory differs")
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
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    result = run(args.workspace, args.freeze, args.freeze_sha256)
    print(json.dumps({key: result[key] for key in ("complete", "execution_passed", "parity_passed")}))
