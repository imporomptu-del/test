"""Typed products of full-resolution stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import Frame


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class StabilizedFrame:
    """A full-resolution float frame mapped directly into one reference."""

    frame: Frame
    reference_frame_index: int
    segment_index: int
    source_to_reference_matrix: np.ndarray
    interpolation: str
    backend: str
    resampling_count: int
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        if self.frame.image.dtype != np.float32:
            raise TypeError("stabilized frame image must be float32")
        if self.frame.valid_mask is None:
            raise ValueError("stabilized frame requires a valid-pixel mask")
        matrix = _readonly(self.source_to_reference_matrix, np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("source_to_reference_matrix must be finite 3x3")
        if self.resampling_count not in {0, 1}:
            raise ValueError("an image may be resampled at most once")
        object.__setattr__(self, "source_to_reference_matrix", matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame.frame_index,
            "reference_frame_index": self.reference_frame_index,
            "segment_index": self.segment_index,
            "mapping": "source_frame_pixels_to_segment_reference_pixels",
            "source_to_reference_matrix": self.source_to_reference_matrix.tolist(),
            "interpolation": self.interpolation,
            "backend": self.backend,
            "resampling_count": self.resampling_count,
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }


@dataclass(frozen=True, slots=True)
class ValidSupport:
    """Per-pixel support for one bounded integration window."""

    common_valid_mask: np.ndarray
    support_count: np.ndarray
    frame_count: int
    segment_index: int
    first_frame_index: int
    last_frame_index: int

    def __post_init__(self) -> None:
        common = _readonly(self.common_valid_mask, bool)
        support = _readonly(self.support_count, np.uint16)
        if common.shape != support.shape or common.ndim != 2:
            raise ValueError("common mask and support count must share a 2-D shape")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if np.any(support > self.frame_count):
            raise ValueError("support_count cannot exceed frame_count")
        if not np.array_equal(common, support == self.frame_count):
            raise ValueError("common_valid_mask must equal full-window support")
        object.__setattr__(self, "common_valid_mask", common)
        object.__setattr__(self, "support_count", support)

    def metrics(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "segment_index": self.segment_index,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "common_valid_fraction": float(np.mean(self.common_valid_mask)),
            "minimum_support": int(np.min(self.support_count)),
            "maximum_support": int(np.max(self.support_count)),
        }
