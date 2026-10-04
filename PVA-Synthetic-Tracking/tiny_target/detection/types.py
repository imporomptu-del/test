"""Typed point-spread-function and matched-filter products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PsfKernelBank:
    """Flux-normalized PSF templates at one or more subpixel phases."""

    kernels: np.ndarray
    phase_offsets_xy: np.ndarray
    source: str
    source_identity: dict[str, Any]
    provisional: bool

    def __post_init__(self) -> None:
        kernels = _readonly(self.kernels, np.float32)
        offsets = _readonly(self.phase_offsets_xy, np.float32)
        if kernels.ndim != 3 or kernels.shape[0] == 0:
            raise ValueError("PSF kernels must have shape [phase, height, width]")
        if kernels.shape[1] % 2 != 1 or kernels.shape[2] % 2 != 1:
            raise ValueError("PSF kernel dimensions must be odd")
        if offsets.shape != (kernels.shape[0], 2):
            raise ValueError("phase_offsets_xy must have shape [phase, 2]")
        if not np.isfinite(kernels).all() or not np.isfinite(offsets).all():
            raise ValueError("PSF kernels and phase offsets must be finite")
        sums = np.sum(kernels, axis=(1, 2), dtype=np.float64)
        if np.any(sums <= 0):
            raise ValueError("every PSF kernel must have positive signed flux")
        if not np.allclose(sums, 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("every PSF kernel must be normalized to unit flux")
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "phase_offsets_xy", offsets)

    @property
    def phase_count(self) -> int:
        return int(self.kernels.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.kernels.shape[1]), int(self.kernels.shape[2])

    def metadata(self) -> dict[str, Any]:
        l2 = np.sqrt(np.sum(self.kernels.astype(np.float64) ** 2, axis=(1, 2)))
        return {
            "source": self.source,
            "source_identity": self.source_identity,
            "provisional": self.provisional,
            "phase_count": self.phase_count,
            "kernel_shape": list(self.shape),
            "phase_offsets_xy": self.phase_offsets_xy.tolist(),
            "normalization": "unit_flux_templates_l2_normalized_during_correlation",
            "l2_norms": l2.tolist(),
        }


@dataclass(frozen=True, slots=True)
class MatchedFilterFrame:
    """Best signed PSF response and selected phase at every valid pixel."""

    response: np.ndarray
    phase_index: np.ndarray
    valid_mask: np.ndarray
    valid_support_count: np.ndarray
    timestamp_ns: int
    frame_index: int
    reference_frame_index: int
    segment_index: int
    detection_ready: bool
    polarity: str
    backend: str
    kernel_metadata: dict[str, Any]
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        response = _readonly(self.response, np.float32)
        phase = _readonly(self.phase_index, np.uint16)
        valid = _readonly(self.valid_mask, bool)
        support = _readonly(self.valid_support_count, np.uint16)
        if response.ndim != 2 or response.size == 0:
            raise ValueError("matched-filter response must be a non-empty 2-D array")
        if not (response.shape == phase.shape == valid.shape == support.shape):
            raise ValueError("matched-filter arrays must share a shape")
        if not np.isfinite(response).all():
            raise ValueError("matched-filter response must be finite")
        if not self.detection_ready and np.any(valid):
            raise ValueError("suppressed matched-filter frames cannot be valid")
        if self.polarity not in {"bright", "dark", "both"}:
            raise ValueError("matched-filter polarity is invalid")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "phase_index", phase)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "valid_support_count", support)

    def score(self) -> np.ndarray:
        if self.polarity == "bright":
            return self.response
        if self.polarity == "dark":
            return -self.response
        return np.abs(self.response)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "reference_frame_index": self.reference_frame_index,
            "segment_index": self.segment_index,
            "detection_ready": self.detection_ready,
            "polarity": self.polarity,
            "backend": self.backend,
            "kernel": self.kernel_metadata,
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }
