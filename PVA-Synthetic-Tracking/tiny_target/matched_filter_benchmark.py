"""Deterministic Phase 6 PSF phase-bank and backend characterization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np

from .detection import (
    MatchedFilterConfig,
    PsfMatchedFilter,
    integrated_gaussian_kernel,
)
from .preprocessing import ResidualFrame
from .telemetry import run_identity, write_json_exclusive


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.matched-filter-benchmark.v1"


def _implementation_identity() -> dict[str, str]:
    files = (
        Path(__file__),
        Path(__file__).parent / "detection" / "matched_filter.py",
        Path(__file__).parent / "detection" / "types.py",
    )
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _residual(image: np.ndarray, valid: np.ndarray | None = None) -> ResidualFrame:
    if valid is None:
        valid = np.ones(image.shape, bool)
    return ResidualFrame(
        value=np.asarray(image, np.float32),
        sigma=np.ones(image.shape, np.float32),
        whitened=np.asarray(image, np.float32),
        valid_mask=np.asarray(valid, bool),
        timestamp_ns=0,
        frame_index=0,
        reference_frame_index=0,
        segment_index=0,
        detection_ready=True,
        history_frames=16,
        background_method="synthetic",
        noise_method="unit_variance",
        metrics={},
        timings_ms={},
    )


def _inject(
    shape: tuple[int, int],
    x: int,
    y: int,
    offset_x: float,
    offset_y: float,
    sigma: float,
    radius: int,
    flux: float,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = integrated_gaussian_kernel(sigma, radius, offset_x, offset_y)
    image = np.zeros(shape, np.float32)
    image[
        y - radius : y + radius + 1,
        x - radius : x + radius + 1,
    ] = flux * kernel
    return image, kernel


def _phase_characterization(phases: int) -> dict[str, Any]:
    sigma = 0.8
    radius = 3
    flux = 20.0
    shape = (35, 37)
    center_y, center_x = 17, 18
    detector = PsfMatchedFilter(
        MatchedFilterConfig(
            backend="numpy_reference",
            gaussian_sigma_px=sigma,
            radius_px=radius,
            phases_per_axis=phases,
            polarity="bright",
        )
    )
    response_retention: list[float] = []
    single_pixel_gains: list[float] = []
    localization_errors: list[float] = []
    offsets = np.linspace(-0.45, 0.45, 7)
    for offset_y in offsets:
        for offset_x in offsets:
            image, true_kernel = _inject(
                shape,
                center_x,
                center_y,
                float(offset_x),
                float(offset_y),
                sigma,
                radius,
                flux,
            )
            result = detector.process(_residual(image))
            score = result.score().copy()
            score[~result.valid_mask] = -np.inf
            peak_y, peak_x = np.unravel_index(int(np.argmax(score)), score.shape)
            phase_index = int(result.phase_index[peak_y, peak_x])
            phase_offset = detector.bank.phase_offsets_xy[phase_index]
            estimated_x = peak_x + float(phase_offset[0])
            estimated_y = peak_y + float(phase_offset[1])
            localization_errors.append(
                float(
                    np.hypot(
                        estimated_x - (center_x + offset_x),
                        estimated_y - (center_y + offset_y),
                    )
                )
            )
            peak = float(score[peak_y, peak_x])
            ideal = flux * float(
                np.sqrt(np.sum(true_kernel.astype(np.float64) ** 2))
            )
            response_retention.append(peak / ideal)
            single_pixel = flux * float(np.max(true_kernel))
            single_pixel_gains.append(peak / single_pixel)
    return {
        "phases_per_axis": phases,
        "phase_count": detector.bank.phase_count,
        "response_retention_vs_exact_template": {
            "minimum": min(response_retention),
            "median": statistics.median(response_retention),
            "maximum": max(response_retention),
        },
        "snr_gain_over_single_pixel": {
            "minimum": min(single_pixel_gains),
            "median": statistics.median(single_pixel_gains),
            "maximum": max(single_pixel_gains),
        },
        "localization_error_px": {
            "median": statistics.median(localization_errors),
            "p90": float(np.percentile(localization_errors, 90)),
            "maximum": max(localization_errors),
        },
    }


def _noise_characterization(phases: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    detector = PsfMatchedFilter(
        MatchedFilterConfig(
            backend="numpy_reference",
            gaussian_sigma_px=0.8,
            radius_px=3,
            phases_per_axis=phases,
            polarity="bright",
        )
    )
    values: list[np.ndarray] = []
    maxima: list[float] = []
    for _ in range(24):
        image = rng.normal(0, 1, (40, 44)).astype(np.float32)
        result = detector.process(_residual(image))
        sample = result.response[result.valid_mask]
        values.append(sample)
        maxima.append(float(np.max(sample)))
    combined = np.concatenate(values)
    median = float(np.median(combined))
    scale = 1.4826 * float(np.median(np.abs(combined - median)))
    return {
        "phases_per_axis": phases,
        "sample_count": int(combined.size),
        "median": median,
        "robust_scale": scale,
        "fraction_gt_5": float(np.mean(combined > 5)),
        "median_frame_maximum": statistics.median(maxima),
        "p90_frame_maximum": float(np.percentile(maxima, 90)),
    }


def _backend_agreement(seed: int) -> dict[str, Any]:
    if importlib.util.find_spec("cv2") is None:
        return {"available": False, "reason": "OpenCV is unavailable"}
    rng = np.random.default_rng(seed)
    image = rng.normal(0, 1, (256, 320)).astype(np.float32)
    valid = rng.random(image.shape) > 0.01
    common = dict(
        gaussian_sigma_px=0.8,
        radius_px=3,
        phases_per_axis=2,
        polarity="both",
    )
    outputs = {}
    timings = {}
    for backend in ("numpy_reference", "opencv_cpu"):
        detector = PsfMatchedFilter(MatchedFilterConfig(backend=backend, **common))
        started = time.perf_counter_ns()
        outputs[backend] = detector.process(_residual(image, valid))
        timings[backend] = (time.perf_counter_ns() - started) / 1_000_000
    reference = outputs["numpy_reference"]
    optimized = outputs["opencv_cpu"]
    masks_equal = bool(np.array_equal(reference.valid_mask, optimized.valid_mask))
    common_valid = reference.valid_mask & optimized.valid_mask
    difference = np.abs(reference.response[common_valid] - optimized.response[common_valid])
    return {
        "available": True,
        "masks_equal": masks_equal,
        "phase_indices_equal": bool(
            np.array_equal(
                reference.phase_index[common_valid], optimized.phase_index[common_valid]
            )
        ),
        "maximum_absolute_error": float(np.max(difference)),
        "p99_absolute_error": float(np.percentile(difference, 99)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "latency_ms": timings,
        "speedup": timings["numpy_reference"] / timings["opencv_cpu"],
    }


def _opencv_phase_scaling(seed: int) -> dict[str, Any]:
    if importlib.util.find_spec("cv2") is None:
        return {"available": False, "reason": "OpenCV is unavailable"}
    rng = np.random.default_rng(seed)
    shape = (512, 640)
    image = rng.normal(0, 1, shape).astype(np.float32)
    source = _residual(image)
    rows: list[dict[str, Any]] = []
    for phases in (1, 2, 4):
        detector = PsfMatchedFilter(
            MatchedFilterConfig(
                backend="opencv_cpu",
                gaussian_sigma_px=0.8,
                radius_px=3,
                phases_per_axis=phases,
                polarity="bright",
            )
        )
        detector.process(source)
        samples = [detector.process(source).timings_ms["total"] for _ in range(3)]
        median_ms = statistics.median(samples)
        rows.append(
            {
                "phases_per_axis": phases,
                "phase_count": phases**2,
                "shape": list(shape),
                "median_total_ms": median_ms,
                "estimated_megapixels_per_second": (
                    shape[0] * shape[1] / 1_000_000 / (median_ms / 1000)
                ),
            }
        )
    return {"available": True, "results": rows}


def benchmark(seed: int = 75) -> dict[str, Any]:
    started = time.perf_counter_ns()
    phase_counts = (1, 2, 4)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": _implementation_identity(),
        "configuration": {
            "seed": seed,
            "psf": "provisional pixel-integrated Gaussian",
            "sigma_px": 0.8,
            "radius_px": 3,
            "tested_phases_per_axis": list(phase_counts),
            "polarity": "bright",
        },
        "phase_characterization": [
            _phase_characterization(count) for count in phase_counts
        ],
        "noise_only": [
            _noise_characterization(count, seed + count) for count in phase_counts
        ],
        "backend_agreement": _backend_agreement(seed),
        "opencv_phase_scaling": _opencv_phase_scaling(seed),
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
