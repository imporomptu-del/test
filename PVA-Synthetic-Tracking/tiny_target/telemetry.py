"""Minimal run identity and stage timing for the no-op Phase 1 pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Iterator


@dataclass(slots=True)
class StageTimings:
    samples_ns: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.samples_ns[stage].append(time.perf_counter_ns() - started)

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for stage, values_ns in sorted(self.samples_ns.items()):
            values_ms = [value / 1_000_000 for value in values_ns]
            result[stage] = {
                "count": len(values_ms),
                "mean_ms": statistics.mean(values_ms),
                "min_ms": min(values_ms),
                "max_ms": max(values_ms),
            }
        return result


def git_identity(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def file_identity(path: str | Path) -> dict[str, Any]:
    input_path = Path(path).expanduser().resolve()
    stat = input_path.stat()
    return {
        "path": str(input_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_hash": None,
        "content_hash_note": (
            "Large recordings are identified by path, size, and mtime in Phase 1; "
            "decoded frame hashes identify the inspected content."
        ),
    }


def run_identity(repository: Path) -> dict[str, Any]:
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": git_identity(repository),
    }


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path

