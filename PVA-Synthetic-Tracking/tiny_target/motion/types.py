"""Typed output of sparse background-feature tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _points(value: np.ndarray, label: str) -> np.ndarray:
    array = np.array(value, dtype=np.float32, order="C", copy=True)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} must have shape (N, 2), got {array.shape}")
    array.setflags(write=False)
    return array


def _vector(value: np.ndarray, length: int, label: str) -> np.ndarray:
    array = np.array(value, order="C", copy=True)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{label} must have shape ({length},), got {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MotionCorrespondences:
    """Accepted background point pairs in full-resolution coordinates."""

    previous_points: np.ndarray
    current_points: np.ndarray
    harris_scores: np.ndarray
    forward_backward_error_px: np.ndarray
    previous_frame_index: int
    current_frame_index: int
    previous_timestamp_ns: int
    current_timestamp_ns: int
    full_image_size: tuple[int, int]
    motion_image_size: tuple[int, int]
    metrics: dict[str, Any]
    timings_ms: dict[str, float]
    backends: dict[str, str | bool]

    def __post_init__(self) -> None:
        previous = _points(self.previous_points, "previous_points")
        current = _points(self.current_points, "current_points")
        if previous.shape != current.shape:
            raise ValueError("previous_points and current_points must have equal shape")
        count = len(previous)
        scores = _vector(self.harris_scores, count, "harris_scores")
        fb_error = _vector(
            self.forward_backward_error_px,
            count,
            "forward_backward_error_px",
        ).astype(np.float32, copy=False)
        fb_error.setflags(write=False)
        if self.current_frame_index <= self.previous_frame_index:
            raise ValueError("current_frame_index must follow previous_frame_index")
        if self.current_timestamp_ns <= self.previous_timestamp_ns:
            raise ValueError("motion pair timestamps must be strictly increasing")
        for label, size in (
            ("full_image_size", self.full_image_size),
            ("motion_image_size", self.motion_image_size),
        ):
            if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"{label} must be positive (width, height)")
        object.__setattr__(self, "previous_points", previous)
        object.__setattr__(self, "current_points", current)
        object.__setattr__(self, "harris_scores", scores)
        object.__setattr__(self, "forward_backward_error_px", fb_error)

    @property
    def count(self) -> int:
        return len(self.previous_points)

    @property
    def delta_seconds(self) -> float:
        return (self.current_timestamp_ns - self.previous_timestamp_ns) / 1e9

    def to_dict(self, *, include_points: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "previous_frame_index": self.previous_frame_index,
            "current_frame_index": self.current_frame_index,
            "previous_timestamp_ns": self.previous_timestamp_ns,
            "current_timestamp_ns": self.current_timestamp_ns,
            "delta_seconds": self.delta_seconds,
            "full_image_size": list(self.full_image_size),
            "motion_image_size": list(self.motion_image_size),
            "accepted_count": self.count,
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
            "backends": self.backends,
        }
        if include_points:
            result["correspondences"] = [
                {
                    "previous_xy": previous.tolist(),
                    "current_xy": current.tolist(),
                    "harris_score": float(score),
                    "forward_backward_error_px": (
                        float(fb_error) if np.isfinite(fb_error) else None
                    ),
                }
                for previous, current, score, fb_error in zip(
                    self.previous_points,
                    self.current_points,
                    self.harris_scores,
                    self.forward_backward_error_px,
                )
            ]
        return result
