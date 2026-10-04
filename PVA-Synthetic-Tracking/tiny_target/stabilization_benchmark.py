"""Synthetic one-pixel/PSF interpolation benchmark for the warp stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .motion import ComposedMotionState
from .stabilization import (
    FullResolutionStabilizer,
    StabilizationConfig,
    StabilizationError,
    alignment_improvement_metrics,
    signal_preservation_metrics,
)
from .telemetry import run_identity, write_json_exclusive
from .types import Frame, TimestampSource


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.stabilization-benchmark.v1"


def _profile(
    name: str,
    width: int,
    height: int,
    signal_flux: float,
) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float32)
    center_x = width // 2
    center_y = height // 2
    if name == "impulse":
        image[center_y, center_x] = signal_flux
        return image
    if name == "gaussian_sigma_0.8":
        radius = 5
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * 0.8**2))
        kernel = (kernel / kernel.sum() * signal_flux).astype(np.float32)
        image[
            center_y - radius : center_y + radius + 1,
            center_x - radius : center_x + radius + 1,
        ] = kernel
        return image
    raise ValueError(f"Unknown signal profile: {name}")


def _frame(image: np.ndarray) -> Frame:
    return Frame(
        image=image,
        timestamp_ns=0,
        frame_index=0,
        source_id="synthetic_psf_benchmark",
        bit_depth=16,
        timestamp_source=TimestampSource.MANIFEST,
    )


def benchmark_interpolations(
    *,
    width: int = 4784,
    height: int = 3190,
    shift_x_px: float = 0.5,
    shift_y_px: float = 0.5,
    signal_flux: float = 10_000.0,
) -> dict[str, Any]:
    if width < 32 or height < 32:
        raise ValueError("benchmark dimensions must be at least 32x32")
    matrix = np.array(
        [[1, 0, shift_x_px], [0, 1, shift_y_px], [0, 0, 1]], np.float64
    )
    state = ComposedMotionState(
        reference_frame_index=0,
        current_frame_index=0,
        segment_index=0,
        reference_from_current_matrix=matrix,
        status="benchmark",
        window_reset=False,
        reused_pairs=0,
        pair_parameter_delta=None,
    )
    cv2 = __import__("cv2")
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    results: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    for profile_name in ("impulse", "gaussian_sigma_0.8"):
        source = _profile(profile_name, width, height, signal_flux)
        source_frame = _frame(source)
        for interpolation in ("linear", "cubic", "lanczos4"):
            cpu = FullResolutionStabilizer(
                StabilizationConfig(
                    backend="opencv_cpu",
                    interpolation=interpolation,
                    valid_mask_erosion_px=0,
                    skip_exact_identity_warp=False,
                )
            ).stabilize(source_frame, state)
            assert cpu.frame.valid_mask is not None
            results.append(
                {
                    "profile": profile_name,
                    "backend": "opencv_cpu",
                    "interpolation": interpolation,
                    "signal": signal_preservation_metrics(
                        source, cpu.frame.image, cpu.frame.valid_mask
                    ),
                    "timings_ms": cpu.timings_ms,
                }
            )
            if not cuda_available or interpolation == "lanczos4":
                continue
            cuda = FullResolutionStabilizer(
                StabilizationConfig(
                    backend="opencv_cuda",
                    interpolation=interpolation,
                    valid_mask_erosion_px=0,
                    skip_exact_identity_warp=False,
                )
            ).stabilize(source_frame, state)
            assert cuda.frame.valid_mask is not None
            results.append(
                {
                    "profile": profile_name,
                    "backend": "opencv_cuda",
                    "interpolation": interpolation,
                    "signal": signal_preservation_metrics(
                        source, cuda.frame.image, cuda.frame.valid_mask
                    ),
                    "timings_ms": cuda.timings_ms,
                }
            )
            common = cpu.frame.valid_mask & cuda.frame.valid_mask
            difference = np.abs(cpu.frame.image[common] - cuda.frame.image[common])
            agreement.append(
                {
                    "profile": profile_name,
                    "interpolation": interpolation,
                    "maximum_absolute_difference": float(np.max(difference)),
                    "mean_absolute_difference": float(np.mean(difference)),
                    "masks_equal": bool(
                        np.array_equal(cpu.frame.valid_mask, cuda.frame.valid_mask)
                    ),
                }
            )
    implementation_path = Path(__file__).parent / "stabilization" / "warp.py"
    rng = np.random.default_rng(75)
    alignment_height = min(height, 768)
    alignment_width = min(width, 1024)
    reference_image = cv2.GaussianBlur(
        rng.uniform(0, 4096, (alignment_height, alignment_width)).astype(np.float32),
        (0, 0),
        2,
    )
    reference_frame = _frame(reference_image)
    current_image = cv2.warpPerspective(
        reference_image,
        matrix,
        (alignment_width, alignment_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    current_frame = Frame(
        image=current_image,
        timestamp_ns=100_000_000,
        frame_index=1,
        source_id="synthetic_alignment_benchmark",
        bit_depth=16,
        timestamp_source=TimestampSource.MANIFEST,
    )
    cpu_cubic = FullResolutionStabilizer(
        StabilizationConfig(
            backend="opencv_cpu",
            interpolation="cubic",
            valid_mask_erosion_px=2,
        )
    )
    reference_state = ComposedMotionState(
        reference_frame_index=0,
        current_frame_index=0,
        segment_index=0,
        reference_from_current_matrix=np.eye(3),
        status="benchmark_reference",
        window_reset=False,
        reused_pairs=0,
        pair_parameter_delta=None,
    )
    current_state = ComposedMotionState(
        reference_frame_index=0,
        current_frame_index=1,
        segment_index=0,
        reference_from_current_matrix=np.linalg.inv(matrix),
        status="benchmark",
        window_reset=False,
        reused_pairs=0,
        pair_parameter_delta=None,
    )
    stabilized_reference = cpu_cubic.stabilize(reference_frame, reference_state)
    stabilized_current = cpu_cubic.stabilize(current_frame, current_state)
    synthetic_alignment = alignment_improvement_metrics(
        reference_frame,
        current_frame,
        stabilized_reference,
        stabilized_current,
        sample_stride=1,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "implementation_sha256": {
            str(implementation_path.relative_to(REPOSITORY)): hashlib.sha256(
                implementation_path.read_bytes()
            ).hexdigest(),
            str(Path(__file__).relative_to(REPOSITORY)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "configuration": {
            "width": width,
            "height": height,
            "shift_x_px": shift_x_px,
            "shift_y_px": shift_y_px,
            "signal_flux": signal_flux,
            "output_dtype": "float32",
            "border_policy": "constant_zero_with_separate_valid_mask",
        },
        "cuda_available": cuda_available,
        "results": results,
        "cpu_cuda_agreement": agreement,
        "synthetic_alignment": {
            "size": [alignment_width, alignment_height],
            "backend": "opencv_cpu",
            "interpolation": "cubic",
            "metrics": synthetic_alignment,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PSF preservation by warp")
    parser.add_argument("--width", type=int, default=4784)
    parser.add_argument("--height", type=int, default=3190)
    parser.add_argument("--shift-x", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = benchmark_interpolations(
            width=args.width,
            height=args.height,
            shift_x_px=args.shift_x,
            shift_y_px=args.shift_y,
        )
        if args.output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            output = write_json_exclusive(args.output, report)
            print(f"Wrote {output}")
    except (ImportError, OSError, StabilizationError, ValueError) as exc:
        raise SystemExit(f"stabilization benchmark failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
