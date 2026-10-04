"""ctypes wrapper for the dependency-free CUDA shift-and-stack backend."""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .synthetic_reference import (
    SyntheticTrackingConfig,
    SyntheticTrackingError,
    _robust_scale,
)
from .synthetic_types import SyntheticTrackWindow
from .types import MatchedFilterFrame


CUDA_ABI_VERSION = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_LIBRARY = REPOSITORY / "build" / "cuda" / "libtiny_target_cuda.so"


class CudaSyntheticTrackingError(SyntheticTrackingError):
    """The native CUDA tracker is unavailable or failed explicitly."""


def _pointer(value: np.ndarray) -> ctypes.c_void_p:
    return value.ctypes.data_as(ctypes.c_void_p)


class CudaShiftAndStack:
    """Native CUDA shift-and-stack with one resident temporal frame stack."""

    backend = "cuda"

    def __init__(
        self,
        config: SyntheticTrackingConfig | Mapping[str, Any],
        *,
        base_path: Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, SyntheticTrackingConfig)
            else SyntheticTrackingConfig.from_mapping(config)
        )
        if self.config.backend != "cuda":
            raise ValueError("CudaShiftAndStack requires synthetic_tracking.backend=cuda")
        self.velocity_grid = np.ascontiguousarray(
            self.config.velocity_grid(), dtype=np.float32
        )
        library_value = self.config.cuda_library_path
        if library_value is None:
            library_path = DEFAULT_CUDA_LIBRARY
        else:
            library_path = Path(library_value).expanduser()
            if not library_path.is_absolute():
                library_path = (base_path or Path.cwd()) / library_path
        self.library_path = library_path.resolve()
        if not self.library_path.is_file():
            raise CudaSyntheticTrackingError(
                f"CUDA synthetic-tracking library is missing: {self.library_path}. "
                "Build it with python3 -m tiny_target.cuda_build."
            )
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            raise CudaSyntheticTrackingError(
                f"Cannot load CUDA synthetic-tracking library {self.library_path}: {exc}"
            ) from exc
        self._library.tt_cuda_abi_version.argtypes = []
        self._library.tt_cuda_abi_version.restype = ctypes.c_int
        abi_version = int(self._library.tt_cuda_abi_version())
        if abi_version != CUDA_ABI_VERSION:
            raise CudaSyntheticTrackingError(
                f"CUDA library ABI {abi_version} does not match Python ABI "
                f"{CUDA_ABI_VERSION}"
            )
        function = self._library.tt_synthetic_track
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        self._function = function
        self.library_sha256 = hashlib.sha256(self.library_path.read_bytes()).hexdigest()

    def _validate_frames(
        self, frames: Sequence[MatchedFilterFrame]
    ) -> tuple[tuple[int, int], int, str, np.ndarray, int, np.ndarray]:
        assert self.config.window_frames is not None
        if len(frames) != self.config.window_frames:
            raise CudaSyntheticTrackingError(
                f"Expected {self.config.window_frames} frames, got {len(frames)}"
            )
        if any(not frame.detection_ready for frame in frames):
            raise CudaSyntheticTrackingError(
                "synthetic window contains a suppressed frame"
            )
        shape = frames[0].response.shape
        segment = frames[0].segment_index
        polarity = frames[0].polarity
        if polarity == "both":
            raise CudaSyntheticTrackingError(
                "both-polarity per-frame maxima cannot be coherently integrated"
            )
        for frame in frames:
            if frame.response.shape != shape:
                raise CudaSyntheticTrackingError(
                    "frame shape changed within synthetic window"
                )
            if frame.segment_index != segment:
                raise CudaSyntheticTrackingError(
                    "synthetic window crosses stabilization segments"
                )
            if frame.polarity != polarity:
                raise CudaSyntheticTrackingError(
                    "matched-filter polarity changed within window"
                )
        timestamps = np.array([frame.timestamp_ns for frame in frames], np.int64)
        if np.any(np.diff(timestamps) <= 0):
            raise CudaSyntheticTrackingError(
                "synthetic window timestamps must increase strictly"
            )
        reference_timestamp = int((int(timestamps[0]) + int(timestamps[-1])) // 2)
        time_offsets_s = np.ascontiguousarray(
            (timestamps.astype(np.float64) - reference_timestamp) / 1e9,
            dtype=np.float64,
        )
        return (
            shape,
            segment,
            polarity,
            timestamps,
            reference_timestamp,
            time_offsets_s,
        )

    def integrate(self, frames: Sequence[MatchedFilterFrame]) -> SyntheticTrackWindow:
        (
            shape,
            segment,
            polarity,
            timestamps,
            reference_timestamp,
            time_offsets_s,
        ) = self._validate_frames(frames)
        started_total = time.perf_counter_ns()
        assert self.config.min_valid_fraction is not None
        minimum_support = math.ceil(
            len(frames) * float(self.config.min_valid_fraction)
        )
        height, width = shape
        frame_stack = np.ascontiguousarray(
            np.stack([frame.response for frame in frames]), dtype=np.float32
        )
        mask_stack = np.ascontiguousarray(
            np.stack([frame.valid_mask for frame in frames]), dtype=np.uint8
        )
        best_score = np.empty(shape, np.float32)
        best_velocity = np.empty(shape, np.uint16)
        best_support = np.empty(shape, np.uint16)
        valid_u8 = np.empty(shape, np.uint8)
        native_timings = np.zeros(5, np.float32)
        counters = np.zeros(7, np.uint64)
        launch_metrics = np.zeros(12, np.int32)
        error = ctypes.create_string_buffer(2048)
        native_started = time.perf_counter_ns()
        status = int(
            self._function(
                _pointer(frame_stack),
                _pointer(mask_stack),
                _pointer(time_offsets_s),
                _pointer(self.velocity_grid),
                len(frames),
                len(self.velocity_grid),
                height,
                width,
                minimum_support,
                1 if polarity == "bright" else -1,
                self.config.velocity_batch_size,
                self.config.cuda_threads_per_block,
                self.config.cuda_device_index,
                _pointer(best_score),
                _pointer(best_velocity),
                _pointer(best_support),
                _pointer(valid_u8),
                _pointer(native_timings),
                _pointer(counters),
                _pointer(launch_metrics),
                error,
                len(error),
            )
        )
        native_call_ms = (time.perf_counter_ns() - native_started) / 1_000_000
        if status != 0:
            detail = error.value.decode("utf-8", errors="replace") or "unknown error"
            raise CudaSyntheticTrackingError(f"CUDA shift-and-stack failed: {detail}")
        valid = valid_u8.astype(bool)
        if np.any(best_velocity[valid] >= len(self.velocity_grid)):
            raise CudaSyntheticTrackingError(
                "CUDA returned a velocity index outside the configured grid"
            )
        duration_s = (int(timestamps[-1]) - int(timestamps[0])) / 1e9
        x_trial_count = len(np.unique(self.velocity_grid[:, 0]))
        y_trial_count = len(np.unique(self.velocity_grid[:, 1]))
        quantization_error_x = (
            0.5 * float(self.config.velocity_step_px_s) * duration_s
            if x_trial_count > 1
            else 0.0
        )
        quantization_error_y = (
            0.5 * float(self.config.velocity_step_px_s) * duration_s
            if y_trial_count > 1
            else 0.0
        )
        values = best_score[valid]
        velocity_counts = np.bincount(
            best_velocity[valid], minlength=len(self.velocity_grid)
        )
        kernel_s = float(native_timings[2]) / 1000
        attempted_samples = int(counters[3])
        theoretical_occupancy = min(
            1.0,
            (
                int(launch_metrics[4]) * int(launch_metrics[5])
                / int(launch_metrics[9])
            ),
        )
        retained_output_bytes = (
            best_score.nbytes
            + best_velocity.nbytes
            + best_support.nbytes
            + valid_u8.nbytes
        )
        full_score_volume_bytes = (
            len(self.velocity_grid) * height * width * np.dtype(np.float32).itemsize
        )
        timings = {
            "total": (time.perf_counter_ns() - started_total) / 1_000_000,
            "native_call_host": native_call_ms,
            "cuda_h2d": float(native_timings[0]),
            "cuda_displacement": float(native_timings[1]),
            "cuda_kernel": float(native_timings[2]),
            "cuda_d2h": float(native_timings[3]),
            "cuda_gpu_total": float(native_timings[4]),
        }
        metrics: dict[str, Any] = {
            "backend": "cuda",
            "frame_count": len(frames),
            "window_duration_s": duration_s,
            "velocity_trial_count": len(self.velocity_grid),
            "velocity_trial_order": "vy_outer_vx_inner",
            "velocity_step_px_s": self.config.velocity_step_px_s,
            "maximum_velocity_quantization_endpoint_error_xy_px": (
                quantization_error_x,
                quantization_error_y,
            ),
            "maximum_velocity_quantization_endpoint_error_px": math.hypot(
                quantization_error_x,
                quantization_error_y,
            ),
            "minimum_temporal_support": minimum_support,
            "valid_fraction": float(np.mean(valid)),
            "score": {
                "median": float(np.median(values)) if values.size else None,
                "robust_scale": _robust_scale(values),
                "p99": float(np.percentile(values, 99)) if values.size else None,
                "maximum": float(np.max(values)) if values.size else None,
                "fraction_gt_5": float(np.mean(values > 5)) if values.size else None,
            },
            "selected_velocity_counts": velocity_counts.tolist(),
            "working_memory_policy": (
                "resident_frame_stack_streamed_velocity_batches_best_full_frame"
            ),
            "cuda": {
                "abi_version": CUDA_ABI_VERSION,
                "library_path": str(self.library_path),
                "library_sha256": self.library_sha256,
                "device_index": int(launch_metrics[0]),
                "compute_capability": [
                    int(launch_metrics[1]),
                    int(launch_metrics[2]),
                ],
                "multiprocessor_count": int(launch_metrics[3]),
                "threads_per_block": int(launch_metrics[4]),
                "active_blocks_per_sm": int(launch_metrics[5]),
                "registers_per_thread": int(launch_metrics[6]),
                "static_shared_bytes": int(launch_metrics[7]),
                "local_bytes_per_thread": int(launch_metrics[8]),
                "max_threads_per_sm": int(launch_metrics[9]),
                "theoretical_occupancy_fraction": theoretical_occupancy,
                "velocity_batch_count": int(launch_metrics[10]),
                "kernel_launch_count": int(launch_metrics[11]),
                "host_to_device_bytes": int(counters[0]),
                "device_to_host_bytes": int(counters[1]),
                "allocated_device_bytes": int(counters[2]),
                "retained_output_bytes": retained_output_bytes,
                "avoided_full_score_volume_bytes": full_score_volume_bytes,
                "full_score_volume_to_retained_output_ratio": (
                    full_score_volume_bytes / retained_output_bytes
                ),
                "displacement_table_bytes": int(counters[4]),
                "free_device_bytes_before": int(counters[5]),
                "free_device_bytes_after_allocations": int(counters[6]),
                "attempted_pixel_samples": attempted_samples,
                "velocity_hypotheses_per_second": (
                    len(self.velocity_grid) / kernel_s if kernel_s > 0 else None
                ),
                "effective_pixel_samples_per_second": (
                    attempted_samples / kernel_s if kernel_s > 0 else None
                ),
                "timing_source": "cuda_events_with_explicit_kernel_sync",
                "occupancy_source": "cuda_runtime_occupancy_api",
            },
        }
        return SyntheticTrackWindow(
            score=best_score,
            velocity_index=best_velocity,
            valid_support_count=best_support,
            valid_mask=valid,
            velocity_grid_xy_px_s=self.velocity_grid,
            frame_indices=tuple(frame.frame_index for frame in frames),
            window_start_timestamp_ns=int(timestamps[0]),
            window_end_timestamp_ns=int(timestamps[-1]),
            reference_timestamp_ns=reference_timestamp,
            segment_index=segment,
            metrics=metrics,
            timings_ms=timings,
        )
