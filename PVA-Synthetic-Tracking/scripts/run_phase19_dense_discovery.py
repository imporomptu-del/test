#!/usr/bin/env python3
"""Run the frozen Phase 19 dense screen over only the Phase 18 discovery split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


EXPECTED_SPLIT_SCHEMA = "seaqr.tiny-target.raw16-discovery-split.v1"
EXPECTED_REPORT_SCHEMA = "seaqr.tiny-target.dense-screen.v1"


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_split(path: Path) -> tuple[list[str], set[str], dict[str, Any]]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError(f"unexpected split schema in {path}")
    discovery = value.get("discovery")
    holdout = value.get("untouched_holdout")
    if not isinstance(discovery, list) or not isinstance(holdout, list):
        raise ValueError("split discovery and holdout values must be lists")
    if len(discovery) != 80 or len(holdout) != 20:
        raise ValueError("Phase 19 requires the frozen 80/20 discovery split")
    discovery_ids = [str(item) for item in discovery]
    holdout_ids = {str(item) for item in holdout}
    if len(set(discovery_ids)) != len(discovery_ids):
        raise ValueError("discovery split contains duplicate clip IDs")
    if set(discovery_ids) & holdout_ids:
        raise ValueError("discovery and holdout clip IDs overlap")
    return discovery_ids, holdout_ids, value


def _completed_report(path: Path, expected_source: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("schema_version") == EXPECTED_REPORT_SCHEMA
        and value.get("source", {}).get("identity", {}).get("path")
        == str(expected_source.resolve())
    )


def _write_status(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--motion-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    dataset_root = args.dataset_root.resolve()
    split_path = args.split.resolve()
    config_path = args.config.resolve()
    motion_config_path = args.motion_config.resolve()
    output_dir = args.output_dir.resolve()
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    discovery, holdout, split = _load_split(split_path)
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("--max-clips must be positive")
        discovery = discovery[: args.max_clips]
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    status_path = output_dir / "phase19_dense_discovery_status.json"
    status: dict[str, Any] = {
        "schema_version": "seaqr.tiny-target.phase19-batch-status.v1",
        "state": "running",
        "workspace": str(workspace),
        "dataset_root": str(dataset_root),
        "split": _identity(split_path),
        "split_id": split.get("split_id"),
        "configuration": _identity(config_path),
        "motion_configuration": _identity(motion_config_path),
        "requested_discovery_clip_count": len(discovery),
        "sealed_holdout_clip_count": len(holdout),
        "sealed_holdout_ids": sorted(holdout),
        "max_frames_per_clip": args.max_frames,
        "started_unix_s": time.time(),
        "updated_unix_s": time.time(),
        "completed": [],
        "skipped_existing": [],
        "failed": [],
        "current_clip_id": None,
    }
    _write_status(status_path, status)

    for ordinal, clip_id in enumerate(discovery, 1):
        if clip_id in holdout:
            raise RuntimeError(f"refusing sealed holdout clip {clip_id}")
        video = dataset_root / f"chunk_{clip_id}.mkv"
        timestamps = dataset_root / f"chunk_{clip_id}_timestamps.csv"
        if not video.is_file() or not timestamps.is_file():
            status["failed"].append(
                {"clip_id": clip_id, "reason": "source_or_timestamp_missing"}
            )
            status["updated_unix_s"] = time.time()
            _write_status(status_path, status)
            print(f"[{ordinal}/{len(discovery)}] {clip_id}: missing input", flush=True)
            continue
        report = reports_dir / f"chunk_{clip_id}_dense_screen.json"
        if _completed_report(report, video):
            status["skipped_existing"].append(clip_id)
            status["updated_unix_s"] = time.time()
            _write_status(status_path, status)
            print(f"[{ordinal}/{len(discovery)}] {clip_id}: valid report exists", flush=True)
            continue
        status["current_clip_id"] = clip_id
        status["updated_unix_s"] = time.time()
        _write_status(status_path, status)
        command = [
            sys.executable,
            "-m",
            "tiny_target.dense_screen",
            "--config",
            str(config_path),
            "--motion-config",
            str(motion_config_path),
            "--input-video",
            str(video),
            "--timestamp-csv",
            str(timestamps),
            "--bit-depth",
            "16",
            "--output",
            str(report),
        ]
        if args.max_frames is not None:
            command.extend(("--max-frames", str(args.max_frames)))
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.monotonic() - started
        if result.returncode == 0 and _completed_report(report, video):
            status["completed"].append(
                {"clip_id": clip_id, "elapsed_seconds": elapsed}
            )
            outcome = "complete"
        else:
            status["failed"].append(
                {
                    "clip_id": clip_id,
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                    "output_tail": result.stdout[-4000:],
                }
            )
            outcome = f"failed ({result.returncode})"
        status["current_clip_id"] = None
        status["updated_unix_s"] = time.time()
        _write_status(status_path, status)
        print(
            f"[{ordinal}/{len(discovery)}] {clip_id}: {outcome} in {elapsed:.1f}s",
            flush=True,
        )

    status["state"] = "complete" if not status["failed"] else "complete_with_failures"
    status["finished_unix_s"] = time.time()
    status["updated_unix_s"] = time.time()
    status["current_clip_id"] = None
    _write_status(status_path, status)
    return 0 if not status["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
