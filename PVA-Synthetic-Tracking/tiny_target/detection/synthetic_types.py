"""Typed products of reference motion-consistent temporal integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SyntheticTrackWindow:
    """Best shift-and-stack score and velocity hypothesis at every pixel."""

    score: np.ndarray
    velocity_index: np.ndarray
    valid_support_count: np.ndarray
    valid_mask: np.ndarray
    velocity_grid_xy_px_s: np.ndarray
    frame_indices: tuple[int, ...]
    window_start_timestamp_ns: int
    window_end_timestamp_ns: int
    reference_timestamp_ns: int
    segment_index: int
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        score = _readonly(self.score, np.float32)
        velocity_index = _readonly(self.velocity_index, np.uint16)
        support = _readonly(self.valid_support_count, np.uint16)
        valid = _readonly(self.valid_mask, bool)
        grid = _readonly(self.velocity_grid_xy_px_s, np.float32)
        if score.ndim != 2 or score.size == 0:
            raise ValueError("synthetic score must be a non-empty 2-D array")
        if not (score.shape == velocity_index.shape == support.shape == valid.shape):
            raise ValueError("synthetic output arrays must share a shape")
        if grid.ndim != 2 or grid.shape[1] != 2 or grid.shape[0] == 0:
            raise ValueError("velocity grid must have shape [trial, 2]")
        if grid.shape[0] > np.iinfo(np.uint16).max:
            raise ValueError("velocity grid exceeds uint16 index capacity")
        if not np.isfinite(score).all() or not np.isfinite(grid).all():
            raise ValueError("synthetic scores and velocities must be finite")
        if np.any(velocity_index[valid] >= grid.shape[0]):
            raise ValueError("velocity index is outside the velocity grid")
        if len(self.frame_indices) == 0:
            raise ValueError("synthetic window must contain frames")
        if not (
            self.window_start_timestamp_ns
            <= self.reference_timestamp_ns
            <= self.window_end_timestamp_ns
        ):
            raise ValueError("reference timestamp must be inside the window")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "velocity_index", velocity_index)
        object.__setattr__(self, "valid_support_count", support)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "velocity_grid_xy_px_s", grid)
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))

    def selected_velocity_xy_px_s(self) -> np.ndarray:
        """Materialize selected velocities only when a downstream stage needs them."""

        return self.velocity_grid_xy_px_s[self.velocity_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_indices": list(self.frame_indices),
            "window_start_timestamp_ns": self.window_start_timestamp_ns,
            "window_end_timestamp_ns": self.window_end_timestamp_ns,
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "reference_time_policy": "temporal_midpoint",
            "segment_index": self.segment_index,
            "velocity_grid_xy_px_s": self.velocity_grid_xy_px_s.tolist(),
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }
