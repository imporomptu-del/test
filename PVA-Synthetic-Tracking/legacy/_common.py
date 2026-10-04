"""Shared helpers for GPU technology study probes (pointer_bench baseline)."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# SkyEye62AM full frame — same resolution as save_video_of_two.py (RESOLUTION_INDEX=2)
WIDTH = 3184
HEIGHT = 2124

# SkyEye62AM baseline (~10 fps, 3184×2124; adjust after first runs on hardware)
BASELINE = {
    "mog2_ms": 55.0,
    "video_write_ms": 250.0,
    "worker_frame_ms_avi": 320.0,
    "worker_frame_ms_novideo": 65.0,
    "processed_fps_novideo": 10.0,
    "camera_fps": 10.0,
}

DEFAULT_OUTPUT = Path("/home/a/projects/SkyFortress/skymove/results/gpu_study")
RESULTS_ROOT = Path("/home/a/projects/SkyFortress/skymove/results")


@dataclass
class TimingResult:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(self.samples_ms)
        p95_idx = max(0, int(len(ordered) * 0.95) - 1)
        return {
            "count": len(ordered),
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p95": ordered[p95_idx],
            "max": max(ordered),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synthetic_gray(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 80, size=(height, width), dtype=np.uint8)
    # A few bright point sources
    for y, x in [(400, 1200), (2000, 2800), (900, 900)]:
        base[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 220
    return base


def synthetic_bgr(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> np.ndarray:
    gray = synthetic_gray(width, height, seed)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def time_call(fn, warmup: int = 2, iterations: int = 10) -> TimingResult:
    result = TimingResult(name=getattr(fn, "__name__", "call"))
    for _ in range(warmup):
        fn()
    for _ in range(iterations):
        t0 = time.monotonic()
        fn()
        result.samples_ms.append((time.monotonic() - t0) * 1000.0)
    return result


def compare_to_baseline(metric: str, measured_ms: float) -> str:
    ref = BASELINE.get(metric)
    if ref is None or ref <= 0:
        return "no baseline"
    ratio = measured_ms / ref
    if ratio < 0.9:
        return f"{ratio:.2f}x faster than baseline ({ref:.0f} ms)"
    if ratio > 1.1:
        return f"{ratio:.2f}x slower than baseline ({ref:.0f} ms)"
    return f"~same as baseline ({ref:.0f} ms)"


def write_report(study_id: str, title: str, sections: dict, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_id": study_id,
        "title": title,
        "generated_at_utc": utc_now(),
        "baseline": BASELINE,
        "sections": sections,
    }
    json_path = out_dir / f"{study_id}_report.json"
    md_path = out_dir / f"{study_id}_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "## Baseline (pointer_bench)",
        "",
        f"- mog2_ms: {BASELINE['mog2_ms']}",
        f"- video_write_ms (AVI): {BASELINE['video_write_ms']}",
        f"- worker_frame_ms (--no-video): {BASELINE['worker_frame_ms_novideo']}",
        "",
    ]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", str(body), ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path
