"""Deterministic Phase 7 reference shift-and-stack characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np

from .detection import (
    MatchedFilterFrame,
    ReferenceShiftAndStack,
    SyntheticTrackingConfig,
)
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.synthetic-reference-benchmark.v1"


def _implementation_identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "detection" / "synthetic_reference.py",
        Path(__file__).parent / "detection" / "synthetic_types.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _matched(response: np.ndarray, index: int, timestamp_ns: int) -> MatchedFilterFrame:
    valid = np.ones(response.shape, bool)
    return MatchedFilterFrame(
        response=np.asarray(response, np.float32),
        phase_index=np.zeros(response.shape, np.uint16),
        valid_mask=valid,
        valid_support_count=np.ones(response.shape, np.uint16),
        timestamp_ns=timestamp_ns,
        frame_index=index,
        reference_frame_index=0,
        segment_index=0,
        detection_ready=True,
        polarity="bright",
        backend="synthetic",
        kernel_metadata={},
        metrics={},
        timings_ms={},
    )


def _config(
    count: int,
    *,
    velocity_min: float,
    velocity_max: float,
    step: float,
    tile_rows: int = 16,
) -> SyntheticTrackingConfig:
    return SyntheticTrackingConfig(
        window_frames=count,
        window_stride_frames=1,
        vx_min_px_s=velocity_min,
        vx_max_px_s=velocity_max,
        vy_min_px_s=velocity_min,
        vy_max_px_s=velocity_max,
        velocity_step_px_s=step,
        min_valid_fraction=1.0,
        tile_rows=tile_rows,
    )


def _gaussian(
    shape: tuple[int, int], x: float, y: float, amplitude: float
) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (
        amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.75**2))
    ).astype(np.float32)


def _frames_with_track(
    rng: np.random.Generator,
    timestamps_ns: Sequence[int],
    shape: tuple[int, int],
    reference_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    amplitude: float,
    *,
    include_target: bool = True,
) -> list[MatchedFilterFrame]:
    reference_ns = (timestamps_ns[0] + timestamps_ns[-1]) // 2
    frames: list[MatchedFilterFrame] = []
    for index, timestamp_ns in enumerate(timestamps_ns):
        response = rng.normal(0, 1, shape).astype(np.float32)
        if include_target:
            offset_s = (timestamp_ns - reference_ns) / 1e9
            response += _gaussian(
                shape,
                reference_xy[0] + velocity_xy[0] * offset_s,
                reference_xy[1] + velocity_xy[1] * offset_s,
                amplitude,
            )
        frames.append(_matched(response, index, timestamp_ns))
    return frames


def _sqrt_n_scaling() -> dict[str, Any]:
    rows = []
    for count in (1, 2, 4, 8):
        frames = []
        for index in range(count):
            image = np.zeros((17, 19), np.float32)
            image[8, 9] = 3
            frames.append(_matched(image, index, index * 100_000_000))
        result = ReferenceShiftAndStack(
            _config(count, velocity_min=0, velocity_max=0, step=1)
        ).integrate(frames)
        measured = float(result.score[8, 9])
        expected = 3 * np.sqrt(count)
        rows.append(
            {
                "frame_count": count,
                "measured_score": measured,
                "expected_score": float(expected),
                "relative_error": abs(measured - expected) / expected,
            }
        )
    return {"per_frame_score": 3.0, "results": rows}


def _golden_output() -> dict[str, Any]:
    """Return a compact, fully serialized oracle for later CUDA comparison."""

    timestamps = (
        0,
        500_000_000,
        1_000_000_000,
        1_500_000_000,
        2_000_000_000,
    )
    shape = (13, 15)
    reference_xy = (7.0, 6.0)
    velocity_xy = (1.0, -1.0)
    reference_ns = (timestamps[0] + timestamps[-1]) // 2
    frames = []
    for index, timestamp_ns in enumerate(timestamps):
        offset_s = (timestamp_ns - reference_ns) / 1e9
        frames.append(
            _matched(
                _gaussian(
                    shape,
                    reference_xy[0] + velocity_xy[0] * offset_s,
                    reference_xy[1] + velocity_xy[1] * offset_s,
                    4.0,
                ),
                index,
                timestamp_ns,
            )
        )
    result = ReferenceShiftAndStack(
        _config(len(frames), velocity_min=-1, velocity_max=1, step=1)
    ).integrate(frames)
    peak_y, peak_x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
    return {
        "purpose": "elementwise_cuda_comparison_with_float_tolerance",
        "shape": list(shape),
        "timestamps_ns": list(timestamps),
        "reference_xy": list(reference_xy),
        "true_velocity_xy_px_s": list(velocity_xy),
        "velocity_grid_xy_px_s": result.velocity_grid_xy_px_s.tolist(),
        "peak_xy": [int(peak_x), int(peak_y)],
        "selected_peak_velocity_xy_px_s": result.velocity_grid_xy_px_s[
            result.velocity_index[peak_y, peak_x]
        ].tolist(),
        "score_float32": result.score.tolist(),
        "velocity_index_uint16": result.velocity_index.tolist(),
        "valid_support_count_uint16": result.valid_support_count.tolist(),
        "valid_mask_bool": result.valid_mask.tolist(),
    }


def _recovery(seed: int) -> dict[str, Any]:
    timestamps = (
        0,
        80_000_000,
        210_000_000,
        350_000_000,
        520_000_000,
        740_000_000,
        910_000_000,
        1_200_000_000,
    )
    shape = (48, 56)
    reference_xy = (28.0, 24.0)
    velocity_xy = (2.0, -1.0)
    tracker = ReferenceShiftAndStack(
        _config(len(timestamps), velocity_min=-3, velocity_max=3, step=1)
    )
    velocity_exact = 0
    velocity_within_one_step = 0
    velocity_errors = []
    selected_velocity_counts: dict[str, int] = {}
    localization_errors = []
    scores = []
    for trial in range(20):
        rng = np.random.default_rng(seed + trial)
        frames = _frames_with_track(
            rng,
            timestamps,
            shape,
            reference_xy,
            velocity_xy,
            amplitude=3.5,
        )
        result = tracker.integrate(frames)
        peak_y, peak_x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
        selected_velocity = result.velocity_grid_xy_px_s[
            result.velocity_index[peak_y, peak_x]
        ]
        velocity_exact += int(np.array_equal(selected_velocity, velocity_xy))
        velocity_error = float(
            np.linalg.norm(selected_velocity - np.asarray(velocity_xy))
        )
        velocity_errors.append(velocity_error)
        velocity_within_one_step += int(velocity_error <= 1.0)
        velocity_key = f"{float(selected_velocity[0]):g},{float(selected_velocity[1]):g}"
        selected_velocity_counts[velocity_key] = (
            selected_velocity_counts.get(velocity_key, 0) + 1
        )
        localization_errors.append(
            float(np.hypot(peak_x - reference_xy[0], peak_y - reference_xy[1]))
        )
        scores.append(float(result.score[peak_y, peak_x]))
    return {
        "trial_count": 20,
        "timestamps_ns": list(timestamps),
        "timestamp_intervals_ms": [
            (timestamps[index] - timestamps[index - 1]) / 1e6
            for index in range(1, len(timestamps))
        ],
        "true_reference_xy": list(reference_xy),
        "true_velocity_xy_px_s": list(velocity_xy),
        "single_frame_peak_snr": 3.5,
        "exact_velocity_recovery_fraction": velocity_exact / 20,
        "velocity_within_one_step_fraction": velocity_within_one_step / 20,
        "velocity_error_px_s": {
            "median": statistics.median(velocity_errors),
            "p90": float(np.percentile(velocity_errors, 90)),
            "maximum": max(velocity_errors),
        },
        "selected_velocity_counts": selected_velocity_counts,
        "localization_error_px": {
            "median": statistics.median(localization_errors),
            "p90": float(np.percentile(localization_errors, 90)),
            "maximum": max(localization_errors),
        },
        "recovered_peak_score": {
            "median": statistics.median(scores),
            "minimum": min(scores),
            "maximum": max(scores),
        },
    }


def _half_step_case(seed: int) -> dict[str, Any]:
    timestamps = (0, 250_000_000, 500_000_000, 750_000_000, 1_000_000_000)
    reference_xy = (20.0, 19.0)
    true_velocity = (1.5, -0.5)
    frames = _frames_with_track(
        np.random.default_rng(seed),
        timestamps,
        (39, 41),
        reference_xy,
        true_velocity,
        amplitude=6,
    )
    result = ReferenceShiftAndStack(
        _config(len(timestamps), velocity_min=-2, velocity_max=2, step=1)
    ).integrate(frames)
    peak_y, peak_x = np.unravel_index(int(np.argmax(result.score)), result.score.shape)
    selected = result.velocity_grid_xy_px_s[result.velocity_index[peak_y, peak_x]]
    endpoint_error = float(
        np.linalg.norm(selected - np.asarray(true_velocity))
        * result.metrics["window_duration_s"]
        / 2
    )
    return {
        "true_velocity_xy_px_s": list(true_velocity),
        "selected_velocity_xy_px_s": selected.tolist(),
        "velocity_step_px_s": 1.0,
        "window_duration_s": result.metrics["window_duration_s"],
        "half_window_endpoint_error_px": endpoint_error,
        "score": float(result.score[peak_y, peak_x]),
    }


def _noise_only(seed: int) -> dict[str, Any]:
    timestamps = tuple(index * 150_000_000 for index in range(6))
    tracker = ReferenceShiftAndStack(
        _config(len(timestamps), velocity_min=-2, velocity_max=2, step=1)
    )
    values = []
    maxima = []
    for trial in range(20):
        frames = _frames_with_track(
            np.random.default_rng(seed + trial),
            timestamps,
            (40, 44),
            (22, 20),
            (0, 0),
            amplitude=0,
            include_target=False,
        )
        result = tracker.integrate(frames)
        sample = result.score[result.valid_mask]
        values.append(sample)
        maxima.append(float(np.max(sample)))
    combined = np.concatenate(values)
    median = float(np.median(combined))
    robust_scale = 1.4826 * float(np.median(np.abs(combined - median)))
    return {
        "trial_count": 20,
        "velocity_trial_count": len(tracker.velocity_grid),
        "sample_count": int(combined.size),
        "median": median,
        "robust_scale": robust_scale,
        "fraction_gt_5": float(np.mean(combined > 5)),
        "median_window_maximum": statistics.median(maxima),
        "p90_window_maximum": float(np.percentile(maxima, 90)),
        "maximum": max(maxima),
    }


def _performance(seed: int) -> dict[str, Any]:
    shape = (256, 320)
    count = 4
    timestamps = tuple(index * 200_000_000 for index in range(count))
    frames = _frames_with_track(
        np.random.default_rng(seed),
        timestamps,
        shape,
        (160, 128),
        (0, 0),
        amplitude=0,
        include_target=False,
    )
    tracker = ReferenceShiftAndStack(
        _config(count, velocity_min=-2, velocity_max=2, step=1, tile_rows=32)
    )
    result = tracker.integrate(frames)
    output_bytes = (
        result.score.nbytes
        + result.velocity_index.nbytes
        + result.valid_support_count.nbytes
        + result.valid_mask.nbytes
    )
    full_volume_bytes = len(tracker.velocity_grid) * shape[0] * shape[1] * 4
    return {
        "shape": list(shape),
        "frame_count": count,
        "velocity_trial_count": len(tracker.velocity_grid),
        "tile_rows": tracker.config.tile_rows,
        "total_ms": result.timings_ms["total"],
        "retained_output_bytes": output_bytes,
        "avoided_full_score_volume_bytes": full_volume_bytes,
        "full_volume_to_retained_output_ratio": full_volume_bytes / output_bytes,
    }


def benchmark(seed: int = 75) -> dict[str, Any]:
    started = time.perf_counter_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _implementation_identity(),
        "configuration": {
            "seed": seed,
            "fractional_sampling": "bilinear",
            "reference_time": "temporal_midpoint",
            "score_normalization": "sum_valid_responses_div_sqrt_valid_count",
        },
        "golden_output": _golden_output(),
        "sqrt_n_scaling": _sqrt_n_scaling(),
        "injected_track_recovery": _recovery(seed),
        "half_velocity_step": _half_step_case(seed + 100),
        "noise_only": _noise_only(seed + 200),
        "performance": _performance(seed + 300),
        "total_ms": (time.perf_counter_ns() - started) / 1_000_000,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = benchmark(args.seed)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
