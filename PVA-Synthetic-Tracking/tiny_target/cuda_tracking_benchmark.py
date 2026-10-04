"""Phase 8 CUDA/reference accuracy, throughput, memory, and stress benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from .detection import (
    CudaShiftAndStack,
    MatchedFilterFrame,
    ReferenceShiftAndStack,
    SyntheticTrackingConfig,
)
from .detection.synthetic_cuda import DEFAULT_CUDA_LIBRARY
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.cuda-tracking-benchmark.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_identity(library_path: Path) -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).with_name("cuda_build.py"),
        Path(__file__).parent / "detection" / "synthetic_cuda.py",
        Path(__file__).parent / "detection" / "synthetic_reference.py",
        Path(__file__).parent / "detection" / "cuda" / "synthetic_tracking.cu",
    )
    identity = {
        str(path.relative_to(REPOSITORY)): _sha256(path) for path in files
    }
    identity[str(library_path)] = _sha256(library_path)
    return identity


def _matched(
    response: np.ndarray,
    valid: np.ndarray,
    index: int,
    timestamp_ns: int,
) -> MatchedFilterFrame:
    return MatchedFilterFrame(
        response=np.asarray(response, np.float32),
        phase_index=np.zeros(response.shape, np.uint16),
        valid_mask=np.asarray(valid, bool),
        valid_support_count=np.asarray(valid, np.uint16),
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
    frame_count: int,
    library_path: Path,
    *,
    velocity_bound: int,
    batch_size: int,
    minimum_valid_fraction: float = 1.0,
    threads_per_block: int = 256,
) -> SyntheticTrackingConfig:
    return SyntheticTrackingConfig(
        backend="cuda",
        window_frames=frame_count,
        window_stride_frames=1,
        vx_min_px_s=-velocity_bound,
        vx_max_px_s=velocity_bound,
        vy_min_px_s=-velocity_bound,
        vy_max_px_s=velocity_bound,
        velocity_step_px_s=1,
        min_valid_fraction=minimum_valid_fraction,
        velocity_batch_size=batch_size,
        cuda_threads_per_block=threads_per_block,
        cuda_library_path=str(library_path),
        tile_rows=32,
    )


def _comparison(reference: Any, cuda: Any) -> dict[str, Any]:
    score_difference = np.abs(cuda.score - reference.score)
    valid_count = int(np.count_nonzero(reference.valid_mask))
    return {
        "score_maximum_absolute_error": float(np.max(score_difference)),
        "score_p99_absolute_error": float(np.percentile(score_difference, 99)),
        "velocity_index_mismatch_count": int(
            np.count_nonzero(cuda.velocity_index != reference.velocity_index)
        ),
        "support_mismatch_count": int(
            np.count_nonzero(
                cuda.valid_support_count != reference.valid_support_count
            )
        ),
        "valid_mask_mismatch_count": int(
            np.count_nonzero(cuda.valid_mask != reference.valid_mask)
        ),
        "valid_pixel_count": valid_count,
    }


def _golden_cases(library_path: Path) -> dict[str, Any]:
    cases = []
    definitions = (
        {
            "name": "integer_axis_aligned",
            "timestamps": (0, 1_000_000_000, 2_000_000_000),
            "shape": (13, 17),
            "velocity": (1.0, -1.0),
            "reference_xy": (8.0, 6.0),
        },
        {
            "name": "fractional_irregular_diagonal",
            "timestamps": (
                0,
                410_000_000,
                1_030_000_000,
                1_720_000_000,
                2_510_000_000,
            ),
            "shape": (21, 25),
            "velocity": (1.0, -1.0),
            "reference_xy": (12.0, 10.0),
        },
    )
    for definition in definitions:
        timestamps = definition["timestamps"]
        reference_ns = (timestamps[0] + timestamps[-1]) // 2
        yy, xx = np.mgrid[: definition["shape"][0], : definition["shape"][1]]
        frames = []
        for index, timestamp_ns in enumerate(timestamps):
            offset_s = (timestamp_ns - reference_ns) / 1e9
            x = definition["reference_xy"][0] + definition["velocity"][0] * offset_s
            y = definition["reference_xy"][1] + definition["velocity"][1] * offset_s
            response = 5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.7**2))
            valid = np.ones(definition["shape"], bool)
            valid[0, :] = False
            valid[:, -1] = False
            frames.append(
                _matched(response.astype(np.float32), valid, index, timestamp_ns)
            )
        settings = _config(
            len(frames), library_path, velocity_bound=1, batch_size=2
        )
        reference = ReferenceShiftAndStack(settings).integrate(frames)
        cuda = CudaShiftAndStack(settings).integrate(frames)
        peak_y, peak_x = np.unravel_index(int(np.argmax(cuda.score)), cuda.score.shape)
        cases.append(
            {
                "name": definition["name"],
                "shape": list(definition["shape"]),
                "frame_count": len(frames),
                "peak_xy": [int(peak_x), int(peak_y)],
                "selected_velocity_xy_px_s": cuda.velocity_grid_xy_px_s[
                    cuda.velocity_index[peak_y, peak_x]
                ].tolist(),
                "comparison": _comparison(reference, cuda),
            }
        )
    return {"cases": cases}


def _randomized_accuracy(library_path: Path, seed: int) -> dict[str, Any]:
    comparisons = []
    for trial in range(24):
        rng = np.random.default_rng(seed + trial)
        frame_count = 2 + trial % 7
        height = 9 + trial % 8
        width = 10 + (trial * 3) % 11
        intervals = rng.integers(50_000_000, 400_000_000, frame_count - 1)
        timestamps = np.concatenate(([0], np.cumsum(intervals))).astype(np.int64)
        frames = []
        for index, timestamp_ns in enumerate(timestamps):
            response = rng.normal(0, 1, (height, width)).astype(np.float32)
            valid = rng.random((height, width)) > (0.03 + 0.03 * (trial % 4))
            frames.append(_matched(response, valid, index, int(timestamp_ns)))
        settings = _config(
            frame_count,
            library_path,
            velocity_bound=1 + trial % 2,
            batch_size=1 + trial % 9,
            minimum_valid_fraction=(0.5, 0.75, 1.0)[trial % 3],
            threads_per_block=(64, 128, 256)[trial % 3],
        )
        reference = ReferenceShiftAndStack(settings).integrate(frames)
        cuda = CudaShiftAndStack(settings).integrate(frames)
        comparison = _comparison(reference, cuda)
        comparison.update(
            {
                "trial": trial,
                "shape": [height, width],
                "frame_count": frame_count,
                "velocity_trial_count": len(cuda.velocity_grid_xy_px_s),
                "batch_size": settings.velocity_batch_size,
                "threads_per_block": settings.cuda_threads_per_block,
            }
        )
        comparisons.append(comparison)
    return {
        "seed": seed,
        "trial_count": len(comparisons),
        "maximum_score_absolute_error": max(
            item["score_maximum_absolute_error"] for item in comparisons
        ),
        "total_velocity_index_mismatches": sum(
            item["velocity_index_mismatch_count"] for item in comparisons
        ),
        "total_support_mismatches": sum(
            item["support_mismatch_count"] for item in comparisons
        ),
        "total_valid_mask_mismatches": sum(
            item["valid_mask_mismatch_count"] for item in comparisons
        ),
        "cases": comparisons,
    }


def _output_digest(result: Any) -> str:
    digest = hashlib.sha256()
    for value in (
        result.score,
        result.velocity_index,
        result.valid_support_count,
        result.valid_mask,
    ):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _performance_case(
    library_path: Path,
    *,
    shape: tuple[int, int],
    frame_count: int,
    velocity_bound: int,
    batch_size: int,
    repeats: int,
    include_reference: bool,
) -> dict[str, Any]:
    timestamps = tuple(index * 200_000_000 for index in range(frame_count))
    valid = np.ones(shape, bool)
    frames = [
        _matched(
            np.zeros(shape, np.float32),
            valid,
            index,
            timestamp_ns,
        )
        for index, timestamp_ns in enumerate(timestamps)
    ]
    settings = _config(
        frame_count,
        library_path,
        velocity_bound=velocity_bound,
        batch_size=batch_size,
    )
    tracker = CudaShiftAndStack(settings)
    warmup = tracker.integrate(frames)
    warmup_timings = dict(warmup.timings_ms)
    del warmup
    gc.collect()
    runs = []
    digests = []
    last_metrics = None
    for _ in range(repeats):
        result = tracker.integrate(frames)
        runs.append(dict(result.timings_ms))
        digests.append(_output_digest(result))
        last_metrics = result.metrics["cuda"]
        del result
        gc.collect()
    comparison = None
    reference_total_ms = None
    if include_reference:
        reference = ReferenceShiftAndStack(settings).integrate(frames)
        cuda = tracker.integrate(frames)
        comparison = _comparison(reference, cuda)
        reference_total_ms = float(reference.timings_ms["total"])
    kernel_times = [run["cuda_kernel"] for run in runs]
    total_times = [run["total"] for run in runs]
    timing_components = {
        key: statistics.median(float(run[key]) for run in runs)
        for key in (
            "cuda_h2d",
            "cuda_displacement",
            "cuda_kernel",
            "cuda_d2h",
            "cuda_gpu_total",
            "native_call_host",
            "total",
        )
    }
    assert last_metrics is not None
    return {
        "shape": list(shape),
        "frame_count": frame_count,
        "velocity_trial_count": len(tracker.velocity_grid),
        "velocity_batch_size": batch_size,
        "repeat_count": repeats,
        "warmup_timings_ms": warmup_timings,
        "kernel_ms": {
            "minimum": min(kernel_times),
            "median": statistics.median(kernel_times),
            "maximum": max(kernel_times),
        },
        "total_ms": {
            "minimum": min(total_times),
            "median": statistics.median(total_times),
            "maximum": max(total_times),
        },
        "median_timing_breakdown_ms": timing_components,
        "median_non_gpu_host_overhead_ms": (
            timing_components["total"] - timing_components["cuda_gpu_total"]
        ),
        "repeatable_output": len(set(digests)) == 1,
        "output_sha256": digests[0],
        "cuda_metrics": last_metrics,
        "reference_comparison": comparison,
        "reference_total_ms": reference_total_ms,
        "reference_to_cuda_kernel_speedup": (
            reference_total_ms / statistics.median(kernel_times)
            if reference_total_ms is not None
            else None
        ),
        "reference_to_cuda_total_speedup": (
            reference_total_ms / statistics.median(total_times)
            if reference_total_ms is not None
            else None
        ),
    }


def _temperatures(lines: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, temperature in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", lines):
        values[name] = max(values.get(name, float("-inf")), float(temperature))
    return values


def _batch_sweep(
    library_path: Path,
    shape: tuple[int, int],
) -> dict[str, Any]:
    frame_count = 4
    valid = np.ones(shape, bool)
    frames = [
        _matched(
            np.zeros(shape, np.float32),
            valid,
            index,
            index * 200_000_000,
        )
        for index in range(frame_count)
    ]
    batch_sizes = (1, 2, 4, 8, 16, 32)
    trackers = {}
    for batch_size in batch_sizes:
        settings = _config(
            frame_count,
            library_path,
            velocity_bound=2,
            batch_size=batch_size,
        )
        trackers[batch_size] = CudaShiftAndStack(settings)
    warmup = trackers[8].integrate(frames)
    del warmup
    gc.collect()
    measurements: dict[int, list[dict[str, float]]] = {
        batch_size: [] for batch_size in batch_sizes
    }
    orders = (
        (1, 8, 2, 16, 4, 32),
        (32, 4, 16, 2, 8, 1),
        (2, 4, 8, 16, 32, 1),
    )
    for order in orders:
        for batch_size in order:
            result = trackers[batch_size].integrate(frames)
            measurements[batch_size].append(
                {
                    "cuda_kernel_ms": result.timings_ms["cuda_kernel"],
                    "cuda_gpu_total_ms": result.timings_ms["cuda_gpu_total"],
                    "total_ms": result.timings_ms["total"],
                }
            )
            del result
            gc.collect()
    rows = []
    for batch_size in batch_sizes:
        values = measurements[batch_size]
        tracker = trackers[batch_size]
        rows.append(
            {
                "velocity_batch_size": batch_size,
                "kernel_launch_count": (
                    math.ceil(len(tracker.velocity_grid) / batch_size) + 3
                ),
                "cuda_kernel_ms": {
                    "minimum": min(item["cuda_kernel_ms"] for item in values),
                    "median": statistics.median(
                        item["cuda_kernel_ms"] for item in values
                    ),
                    "maximum": max(item["cuda_kernel_ms"] for item in values),
                },
                "cuda_gpu_total_ms": {
                    "minimum": min(item["cuda_gpu_total_ms"] for item in values),
                    "median": statistics.median(
                        item["cuda_gpu_total_ms"] for item in values
                    ),
                    "maximum": max(item["cuda_gpu_total_ms"] for item in values),
                },
                "total_ms": {
                    "minimum": min(item["total_ms"] for item in values),
                    "median": statistics.median(
                        item["total_ms"] for item in values
                    ),
                    "maximum": max(item["total_ms"] for item in values),
                },
            }
        )
    selected = min(rows, key=lambda item: item["cuda_kernel_ms"]["median"])
    return {
        "shape": list(shape),
        "frame_count": frame_count,
        "velocity_trial_count": 25,
        "interleaved_repeat_count_per_batch_size": len(orders),
        "results": rows,
        "fastest_median_velocity_batch_size": selected["velocity_batch_size"],
        "configured_default_velocity_batch_size": 32,
    }


def _tegrastats_summary(lines: str) -> dict[str, Any]:
    gpu_utilization = [
        int(value) for value in re.findall(r"GR3D_FREQ\s+(\d+)%", lines)
    ]
    ram_used = [
        int(value) for value in re.findall(r"RAM\s+(\d+)/\d+MB", lines)
    ]
    gpu_power = [
        int(value)
        for value in re.findall(r"VDD_GPU_SOC\s+(\d+)mW/\d+mW", lines)
    ]
    return {
        "samples": len(lines.splitlines()),
        "maximum_temperature_c": _temperatures(lines),
        "gpu_utilization_percent": {
            "median": (
                statistics.median(gpu_utilization) if gpu_utilization else None
            ),
            "maximum": max(gpu_utilization) if gpu_utilization else None,
        },
        "maximum_ram_used_mb": max(ram_used) if ram_used else None,
        "gpu_soc_power_mw": {
            "median": statistics.median(gpu_power) if gpu_power else None,
            "maximum": max(gpu_power) if gpu_power else None,
        },
    }


def _read_text_if_available(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def benchmark(
    library_path: Path,
    *,
    seed: int = 75,
    stress_shape: tuple[int, int] = (3190, 4784),
    stress_repeats: int = 3,
    maximum_score_absolute_error: float = 1e-5,
) -> dict[str, Any]:
    resolved_library = library_path.expanduser().resolve()
    started = time.perf_counter_ns()
    tegrastats = shutil.which("tegrastats")
    sampler = None
    if tegrastats is not None:
        sampler = subprocess.Popen(
            [tegrastats, "--interval", "250"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        golden = _golden_cases(resolved_library)
        randomized = _randomized_accuracy(resolved_library, seed)
        representative = _performance_case(
            resolved_library,
            shape=(256, 320),
            frame_count=4,
            velocity_bound=2,
            batch_size=32,
            repeats=5,
            include_reference=True,
        )
        batch_sweep = _batch_sweep(resolved_library, stress_shape)
        stress = _performance_case(
            resolved_library,
            shape=stress_shape,
            frame_count=4,
            velocity_bound=2,
            batch_size=32,
            repeats=stress_repeats,
            include_reference=False,
        )
    finally:
        tegrastats_output = ""
        if sampler is not None:
            sampler.terminate()
            try:
                tegrastats_output = sampler.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                sampler.kill()
                tegrastats_output = sampler.communicate()[0]
    comparisons = [
        item["comparison"] for item in golden["cases"]
    ] + randomized["cases"]
    if representative["reference_comparison"] is not None:
        comparisons.append(representative["reference_comparison"])
    observed_maximum_error = max(
        item["score_maximum_absolute_error"] for item in comparisons
    )
    exact_discrete_outputs = all(
        item["velocity_index_mismatch_count"] == 0
        and item["support_mismatch_count"] == 0
        and item["valid_mask_mismatch_count"] == 0
        for item in comparisons
    )
    accuracy_passed = (
        observed_maximum_error <= maximum_score_absolute_error
        and exact_discrete_outputs
    )
    stress_passed = bool(stress["repeatable_output"])
    nvidia_parameters = _read_text_if_available(
        Path("/proc/driver/nvidia/params")
    )
    profiling_admin_only = None
    if nvidia_parameters is not None:
        match = re.search(r"^RmProfilingAdminOnly:\s*(\d+)", nvidia_parameters, re.M)
        if match is not None:
            profiling_admin_only = bool(int(match.group(1)))
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _implementation_identity(resolved_library),
        "configuration": {
            "seed": seed,
            "stress_shape": list(stress_shape),
            "stress_repeats": stress_repeats,
            "accumulation_dtype": "float32",
            "fractional_sampling": "bilinear",
            "maximum_score_absolute_error": maximum_score_absolute_error,
        },
        "golden": golden,
        "randomized_accuracy": randomized,
        "representative_performance": representative,
        "velocity_batch_sweep": batch_sweep,
        "maximum_intended_workload": stress,
        "tegrastats": {
            "available": tegrastats is not None,
            **_tegrastats_summary(tegrastats_output),
        },
        "profiler": {
            "nsight_systems_available": shutil.which("nsys") is not None,
            "nsight_compute_available": shutil.which("ncu") is not None,
            "rm_profiling_admin_only": profiling_admin_only,
            "perf_event_paranoid": _read_text_if_available(
                Path("/proc/sys/kernel/perf_event_paranoid")
            ),
            "note": (
                "CUDA-event timing and runtime occupancy attributes are recorded; "
                "Nsight CLI tools were not installed and unprivileged CUPTI "
                "hardware profiling is restricted on the target."
            ),
        },
        "acceptance": {
            "observed_maximum_score_absolute_error": observed_maximum_error,
            "exact_velocity_support_and_validity": exact_discrete_outputs,
            "accuracy_passed": accuracy_passed,
            "maximum_workload_completed_repeatably": stress_passed,
            "passed": accuracy_passed and stress_passed,
        },
        "total_ms": (time.perf_counter_ns() - started) / 1_000_000,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_CUDA_LIBRARY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--stress-height", type=int, default=3190)
    parser.add_argument("--stress-width", type=int, default=4784)
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument(
        "--maximum-score-absolute-error", type=float, default=1e-5
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stress_height <= 0 or args.stress_width <= 0:
        raise SystemExit("stress dimensions must be positive")
    if args.stress_repeats <= 0:
        raise SystemExit("stress repeats must be positive")
    if args.maximum_score_absolute_error < 0:
        raise SystemExit("maximum score absolute error must be non-negative")
    report = benchmark(
        args.library,
        seed=args.seed,
        stress_shape=(args.stress_height, args.stress_width),
        stress_repeats=args.stress_repeats,
        maximum_score_absolute_error=args.maximum_score_absolute_error,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
