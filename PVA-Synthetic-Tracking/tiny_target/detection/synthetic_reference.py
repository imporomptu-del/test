"""Transparent timestamp-aware reference shift-and-stack implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .synthetic_types import SyntheticTrackWindow
from .types import MatchedFilterFrame


class SyntheticTrackingError(RuntimeError):
    """A reference synthetic-tracking window violates its contract."""


@dataclass(frozen=True, slots=True)
class SyntheticTrackingConfig:
    backend: str = "reference"
    window_frames: int | None = None
    window_stride_frames: int | None = None
    vx_min_px_s: float | None = None
    vx_max_px_s: float | None = None
    vy_min_px_s: float | None = None
    vy_max_px_s: float | None = None
    velocity_step_px_s: float | None = None
    fractional_sampling: str = "bilinear"
    min_valid_fraction: float | None = None
    reference_time: str = "midpoint"
    tile_rows: int = 64
    cuda_library_path: str | None = None
    velocity_batch_size: int = 32
    cuda_threads_per_block: int = 256
    cuda_device_index: int = 0

    def __post_init__(self) -> None:
        required = (
            "window_frames",
            "window_stride_frames",
            "vx_min_px_s",
            "vx_max_px_s",
            "vy_min_px_s",
            "vy_max_px_s",
            "velocity_step_px_s",
            "min_valid_fraction",
        )
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Synthetic tracking is not calibrated; set " + ", ".join(missing)
            )
        assert self.window_frames is not None
        assert self.window_stride_frames is not None
        if self.backend not in {"reference", "cuda"}:
            raise ValueError("synthetic_tracking.backend must be reference or cuda")
        for name in (
            "window_frames",
            "window_stride_frames",
            "tile_rows",
            "velocity_batch_size",
            "cuda_threads_per_block",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"synthetic_tracking.{name} must be a positive integer")
        if self.window_frames > np.iinfo(np.uint16).max:
            raise ValueError("synthetic tracking window exceeds uint16 support capacity")
        if self.window_stride_frames > self.window_frames:
            raise ValueError(
                "synthetic_tracking.window_stride_frames cannot exceed window_frames"
            )
        values = (
            self.vx_min_px_s,
            self.vx_max_px_s,
            self.vy_min_px_s,
            self.vy_max_px_s,
            self.velocity_step_px_s,
            self.min_valid_fraction,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("synthetic tracking numeric values must be finite")
        assert self.vx_min_px_s is not None and self.vx_max_px_s is not None
        assert self.vy_min_px_s is not None and self.vy_max_px_s is not None
        assert self.velocity_step_px_s is not None
        assert self.min_valid_fraction is not None
        if self.vx_min_px_s > self.vx_max_px_s or self.vy_min_px_s > self.vy_max_px_s:
            raise ValueError("synthetic tracking velocity minima cannot exceed maxima")
        if self.velocity_step_px_s <= 0:
            raise ValueError("synthetic_tracking.velocity_step_px_s must be positive")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("synthetic_tracking.min_valid_fraction must be in (0, 1]")
        if self.fractional_sampling != "bilinear":
            raise ValueError("synthetic_tracking.fractional_sampling must be bilinear")
        if self.reference_time != "midpoint":
            raise ValueError("synthetic_tracking.reference_time must be midpoint")
        if self.cuda_threads_per_block > 1024 or self.cuda_threads_per_block % 32:
            raise ValueError(
                "synthetic_tracking.cuda_threads_per_block must be a multiple of 32 "
                "and no greater than 1024"
            )
        if (
            isinstance(self.cuda_device_index, bool)
            or not isinstance(self.cuda_device_index, int)
            or self.cuda_device_index < 0
        ):
            raise ValueError(
                "synthetic_tracking.cuda_device_index must be a non-negative integer"
            )
        if self.cuda_library_path is not None and (
            not isinstance(self.cuda_library_path, str) or not self.cuda_library_path
        ):
            raise ValueError(
                "synthetic_tracking.cuda_library_path must be a non-empty string or null"
            )
        self._axis_values(self.vx_min_px_s, self.vx_max_px_s)
        self._axis_values(self.vy_min_px_s, self.vy_max_px_s)
        if self.velocity_grid().shape[0] > np.iinfo(np.uint16).max:
            raise ValueError("synthetic tracking velocity grid has more than 65535 trials")

    def _axis_values(self, minimum: float, maximum: float) -> np.ndarray:
        assert self.velocity_step_px_s is not None
        span = maximum - minimum
        steps = round(span / self.velocity_step_px_s)
        if not math.isclose(
            minimum + steps * self.velocity_step_px_s,
            maximum,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "velocity range must be an integer multiple of velocity_step_px_s"
            )
        return minimum + np.arange(steps + 1, dtype=np.float64) * self.velocity_step_px_s

    def velocity_grid(self) -> np.ndarray:
        assert self.vx_min_px_s is not None and self.vx_max_px_s is not None
        assert self.vy_min_px_s is not None and self.vy_max_px_s is not None
        x_values = self._axis_values(self.vx_min_px_s, self.vx_max_px_s)
        y_values = self._axis_values(self.vy_min_px_s, self.vy_max_px_s)
        return np.array([(x, y) for y in y_values for x in x_values], np.float32)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SyntheticTrackingConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown synthetic tracking configuration keys: {unknown}")
        return cls(**data)


def _shift_sample_tile(
    image: np.ndarray,
    valid_mask: np.ndarray,
    shift_x: float,
    shift_y: float,
    row_start: int,
    row_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``image[y + shift_y, x + shift_x]`` with bilinear support."""

    height, width = image.shape
    tile_height = row_end - row_start
    sampled = np.zeros((tile_height, width), np.float32)
    sampled_valid = np.ones((tile_height, width), bool)
    base_x = math.floor(shift_x)
    base_y = math.floor(shift_y)
    fraction_x = shift_x - base_x
    fraction_y = shift_y - base_y
    neighbors = (
        (base_x, base_y, (1.0 - fraction_x) * (1.0 - fraction_y)),
        (base_x + 1, base_y, fraction_x * (1.0 - fraction_y)),
        (base_x, base_y + 1, (1.0 - fraction_x) * fraction_y),
        (base_x + 1, base_y + 1, fraction_x * fraction_y),
    )
    used_neighbor = False
    for offset_x, offset_y, weight in neighbors:
        if weight <= 1e-7:
            continue
        used_neighbor = True
        neighbor_valid = np.zeros((tile_height, width), bool)
        destination_x_start = max(0, -offset_x)
        destination_x_end = min(width, width - offset_x)
        global_destination_y_start = max(row_start, -offset_y)
        global_destination_y_end = min(row_end, height - offset_y)
        if (
            destination_x_start < destination_x_end
            and global_destination_y_start < global_destination_y_end
        ):
            tile_y_start = global_destination_y_start - row_start
            tile_y_end = global_destination_y_end - row_start
            source_y_start = global_destination_y_start + offset_y
            source_y_end = global_destination_y_end + offset_y
            source_x_start = destination_x_start + offset_x
            source_x_end = destination_x_end + offset_x
            destination = (
                slice(tile_y_start, tile_y_end),
                slice(destination_x_start, destination_x_end),
            )
            source = (
                slice(source_y_start, source_y_end),
                slice(source_x_start, source_x_end),
            )
            sampled[destination] += weight * image[source]
            neighbor_valid[destination] = valid_mask[source]
        sampled_valid &= neighbor_valid
    if not used_neighbor:
        raise SyntheticTrackingError("bilinear sampler received no nonzero weights")
    return sampled, sampled_valid


def _robust_scale(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


class ReferenceShiftAndStack:
    """Slow, tiled golden implementation for future CUDA comparison."""

    backend = "reference"

    def __init__(self, config: SyntheticTrackingConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, SyntheticTrackingConfig)
            else SyntheticTrackingConfig.from_mapping(config)
        )
        self.velocity_grid = self.config.velocity_grid()

    def integrate(self, frames: Sequence[MatchedFilterFrame]) -> SyntheticTrackWindow:
        assert self.config.window_frames is not None
        if len(frames) != self.config.window_frames:
            raise SyntheticTrackingError(
                f"Expected {self.config.window_frames} frames, got {len(frames)}"
            )
        if any(not frame.detection_ready for frame in frames):
            raise SyntheticTrackingError("synthetic window contains a suppressed frame")
        shape = frames[0].response.shape
        segment = frames[0].segment_index
        polarity = frames[0].polarity
        if polarity == "both":
            raise SyntheticTrackingError(
                "both-polarity per-frame maxima cannot be coherently integrated"
            )
        for frame in frames:
            if frame.response.shape != shape:
                raise SyntheticTrackingError("frame shape changed within synthetic window")
            if frame.segment_index != segment:
                raise SyntheticTrackingError("synthetic window crosses stabilization segments")
            if frame.polarity != polarity:
                raise SyntheticTrackingError("matched-filter polarity changed within window")
        timestamps = np.array([frame.timestamp_ns for frame in frames], np.int64)
        if np.any(np.diff(timestamps) <= 0):
            raise SyntheticTrackingError("synthetic window timestamps must increase strictly")
        reference_timestamp = int((int(timestamps[0]) + int(timestamps[-1])) // 2)
        time_offsets_s = (timestamps.astype(np.float64) - reference_timestamp) / 1e9
        minimum_support = math.ceil(
            len(frames) * float(self.config.min_valid_fraction)
        )
        height, width = shape
        best_score = np.full(shape, -np.inf, np.float32)
        best_velocity = np.zeros(shape, np.uint16)
        best_support = np.zeros(shape, np.uint16)
        started_total = time.perf_counter_ns()
        sample_operations = 0
        for row_start in range(0, height, self.config.tile_rows):
            row_end = min(height, row_start + self.config.tile_rows)
            tile_score = best_score[row_start:row_end]
            tile_velocity = best_velocity[row_start:row_end]
            tile_support = best_support[row_start:row_end]
            for velocity_index, (velocity_x, velocity_y) in enumerate(self.velocity_grid):
                accumulator = np.zeros(tile_score.shape, np.float32)
                support = np.zeros(tile_score.shape, np.uint16)
                for frame, offset_s in zip(frames, time_offsets_s):
                    value = frame.response if polarity == "bright" else -frame.response
                    sampled, sample_valid = _shift_sample_tile(
                        value,
                        frame.valid_mask,
                        float(velocity_x) * float(offset_s),
                        float(velocity_y) * float(offset_s),
                        row_start,
                        row_end,
                    )
                    accumulator[sample_valid] += sampled[sample_valid]
                    support[sample_valid] += 1
                    sample_operations += 1
                trial_valid = support >= minimum_support
                score = np.zeros(tile_score.shape, np.float32)
                score[trial_valid] = accumulator[trial_valid] / np.sqrt(
                    support[trial_valid].astype(np.float32)
                )
                better = trial_valid & (score > tile_score)
                tile_score[better] = score[better]
                tile_velocity[better] = velocity_index
                tile_support[better] = support[better]
        valid = np.isfinite(best_score)
        best_score[~valid] = 0
        duration_s = (int(timestamps[-1]) - int(timestamps[0])) / 1e9
        values = best_score[valid]
        velocity_counts = np.bincount(
            best_velocity[valid], minlength=len(self.velocity_grid)
        )
        x_trial_count = len(
            np.unique(self.velocity_grid[:, 0])
        )
        y_trial_count = len(
            np.unique(self.velocity_grid[:, 1])
        )
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
        timings = {
            "total": (time.perf_counter_ns() - started_total) / 1_000_000,
        }
        metrics: dict[str, Any] = {
            "backend": "reference",
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
                "fraction_gt_5": (
                    float(np.mean(values > 5)) if values.size else None
                ),
            },
            "selected_velocity_counts": velocity_counts.tolist(),
            "sample_tile_operations": sample_operations,
            "working_memory_policy": "one_velocity_one_row_tile_plus_best_full_frame",
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


class ReferenceSyntheticWindow:
    """Form bounded overlapping windows for either computational backend."""

    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        assert tracker.config.window_frames is not None
        self._frames: deque[MatchedFilterFrame] = deque(
            maxlen=tracker.config.window_frames
        )
        self._segment_index: int | None = None
        self._emitted = False
        self._frames_since_emit = 0

    def _clear(self, segment_index: int | None) -> None:
        self._frames.clear()
        self._segment_index = segment_index
        self._emitted = False
        self._frames_since_emit = 0

    def update(self, frame: MatchedFilterFrame) -> SyntheticTrackWindow | None:
        if not frame.detection_ready:
            self._clear(frame.segment_index)
            return None
        if self._segment_index != frame.segment_index:
            self._clear(frame.segment_index)
        self._frames.append(frame)
        assert self.tracker.config.window_frames is not None
        assert self.tracker.config.window_stride_frames is not None
        if len(self._frames) < self.tracker.config.window_frames:
            return None
        if self._emitted:
            self._frames_since_emit += 1
            if self._frames_since_emit < self.tracker.config.window_stride_frames:
                return None
        result = self.tracker.integrate(tuple(self._frames))
        self._emitted = True
        self._frames_since_emit = 0
        return result
