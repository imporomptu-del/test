"""Deterministic Phase 5 synthetic preprocessing characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np

from .preprocessing import BackgroundConfig, NoiseConfig, RobustPreprocessor
from .stabilization import StabilizedFrame
from .telemetry import run_identity, write_json_exclusive
from .types import Frame, TimestampSource


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.preprocessing-benchmark.v1"


def _implementation_identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "preprocessing" / "model.py",
        Path(__file__).parent / "preprocessing" / "types.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _stabilized(image: np.ndarray, index: int, bad: np.ndarray) -> StabilizedFrame:
    valid = np.ones(image.shape, bool)
    valid[:2] = False
    valid[-2:] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    frame = Frame(
        image=np.asarray(image, np.float32),
        timestamp_ns=index * 50_000_000,
        frame_index=index,
        source_id="phase5_synthetic",
        bit_depth=16,
        timestamp_source=TimestampSource.MANIFEST,
        valid_mask=valid,
    )
    return StabilizedFrame(
        frame=frame,
        reference_frame_index=0,
        segment_index=0,
        source_to_reference_matrix=np.eye(3),
        interpolation="cubic",
        backend="synthetic_identity",
        resampling_count=0,
        metrics={"bad_pixel_count": int(np.count_nonzero(bad))},
        timings_ms={},
    )


def _psf(shape: tuple[int, int], x: float, y: float, flux: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    value = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.8**2))
    return (value / value.sum() * flux).astype(np.float32)


def _scale(values: np.ndarray) -> float:
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def _run_method(
    method: str,
    frames: list[np.ndarray],
    targets: list[np.ndarray],
    bad: np.ndarray,
    warmup: int,
) -> dict[str, Any]:
    noise_method = "temporal_mad" if method == "temporal_median" else "robust_ewma"
    background = BackgroundConfig(
        method=method,
        warmup_frames=warmup,
        history_frames=20,
        minimum_history_samples=12,
        update_rate=0.2,
        outlier_clip_sigma=4,
        update_exclusion_sigma=3,
    )
    noise = NoiseConfig(
        method=noise_method,
        sigma_floor=0.5,
        saturation_value=4095,
        dead_level_max=0,
    )
    control = RobustPreprocessor(background, noise, bad_pixel_mask=bad)
    injected = RobustPreprocessor(background, noise, bad_pixel_mask=bad)
    retained: list[float] = []
    scale_ratios: list[float] = []
    false_maxima: list[float] = []
    false_tail_5: list[float] = []
    residual_biases: list[float] = []
    latencies: list[float] = []
    ready_frames = 0
    for index, (base, target) in enumerate(zip(frames, targets)):
        base_stabilized = _stabilized(base, index, bad)
        target_stabilized = _stabilized(base + target, index, bad)
        reference_result = control.process(base_stabilized)
        result = injected.process(target_stabilized)
        latencies.append(result.timings_ms["total"])
        if not result.detection_ready:
            continue
        ready_frames += 1
        valid = reference_result.valid_mask
        left = reference_result.whitened[:, : base.shape[1] // 2]
        left_valid = valid[:, : base.shape[1] // 2]
        right = reference_result.whitened[:, base.shape[1] // 2 :]
        right_valid = valid[:, base.shape[1] // 2 :]
        left_scale = _scale(left[left_valid])
        right_scale = _scale(right[right_valid])
        scale_ratios.append(right_scale / left_scale)
        values = reference_result.whitened[valid]
        false_maxima.append(float(np.max(values)))
        false_tail_5.append(float(np.mean(np.abs(values) > 5)))
        residual_biases.append(float(np.median(reference_result.value[valid])))
        if np.any(target):
            target_support = target > target.max() * 0.001
            recovered = np.sum(
                result.value[target_support] - reference_result.value[target_support]
            )
            retained.append(float(recovered / np.sum(target[target_support])))
    return {
        "background_method": method,
        "noise_method": noise_method,
        "ready_frames": ready_frames,
        "warmup_frames": warmup,
        "target_flux_retention": {
            "count": len(retained),
            "minimum": min(retained),
            "median": statistics.median(retained),
            "maximum": max(retained),
        },
        "bright_to_dark_whitened_scale_ratio": {
            "median": statistics.median(scale_ratios),
            "p90": float(np.percentile(scale_ratios, 90)),
        },
        "noise_only_spatial_false_peak": {
            "median_frame_maximum": statistics.median(false_maxima),
            "p90_frame_maximum": float(np.percentile(false_maxima, 90)),
            "mean_fraction_abs_gt_5": statistics.mean(false_tail_5),
        },
        "slow_drift_residual_median": {
            "median": statistics.median(residual_biases),
            "last": residual_biases[-1],
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p90": float(np.percentile(latencies, 90)),
        },
    }


def benchmark(*, seed: int = 75) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shape = (96, 128)
    count = 48
    warmup = 16
    bad = np.zeros(shape, bool)
    bad[8, 8] = True
    frames: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    started = time.perf_counter_ns()
    for index in range(count):
        image = np.empty(shape, np.float32)
        drift = index * 0.12
        image[:, : shape[1] // 2] = 120 + drift + rng.normal(
            0, 2, (shape[0], shape[1] // 2)
        )
        image[:, shape[1] // 2 :] = 1200 + drift + rng.normal(
            0, 8, (shape[0], shape[1] // 2)
        )
        image[8, 8] = 4095
        frames.append(image)
        if index >= warmup:
            elapsed = index - warmup
            targets.append(
                _psf(shape, 25.25 + 0.65 * elapsed, 42.75 + 0.18 * elapsed, 240)
            )
        else:
            targets.append(np.zeros(shape, np.float32))
    results = [
        _run_method(method, frames, targets, bad, warmup)
        for method in ("temporal_median", "robust_running")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _implementation_identity(),
        "configuration": {
            "seed": seed,
            "shape": list(shape),
            "frame_count": count,
            "warmup_frames": warmup,
            "backgrounds": [120, 1200],
            "noise_sigmas": [2, 8],
            "illumination_drift_per_frame": 0.12,
            "target_flux": 240,
            "target_sigma_px": 0.8,
            "target_velocity_px_per_frame": [0.65, 0.18],
        },
        "results": results,
        "total_ms": (time.perf_counter_ns() - started) / 1_000_000,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = benchmark(seed=args.seed)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
