"""Typed products of background subtraction and noise normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly_2d(value: np.ndarray, dtype: Any, label: str) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if result.ndim != 2 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty 2-D array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ResidualFrame:
    """One signed, noise-normalized frame ready for later PSF filtering.

    ``valid_mask`` is the detection mask, not merely the set of pixels that may
    update the background. It is empty during warm-up by construction.
    """

    value: np.ndarray
    sigma: np.ndarray
    whitened: np.ndarray
    valid_mask: np.ndarray
    timestamp_ns: int
    frame_index: int
    reference_frame_index: int
    segment_index: int
    detection_ready: bool
    history_frames: int
    background_method: str
    noise_method: str
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        value = _readonly_2d(self.value, np.float32, "value")
        sigma = _readonly_2d(self.sigma, np.float32, "sigma")
        whitened = _readonly_2d(self.whitened, np.float32, "whitened")
        valid = _readonly_2d(self.valid_mask, bool, "valid_mask")
        if not (value.shape == sigma.shape == whitened.shape == valid.shape):
            raise ValueError("residual arrays must share a shape")
        if not np.isfinite(value).all() or not np.isfinite(whitened).all():
            raise ValueError("residual and whitened arrays must be finite")
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("sigma must be finite and strictly positive")
        if self.frame_index < 0 or self.reference_frame_index < 0:
            raise ValueError("frame indices must be non-negative")
        if self.segment_index < 0 or self.history_frames < 0:
            raise ValueError("segment and history counts must be non-negative")
        if not self.detection_ready and np.any(valid):
            raise ValueError("warm-up residuals cannot expose detection-valid pixels")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "whitened", whitened)
        object.__setattr__(self, "valid_mask", valid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "reference_frame_index": self.reference_frame_index,
            "segment_index": self.segment_index,
            "detection_ready": self.detection_ready,
            "history_frames": self.history_frames,
            "background_method": self.background_method,
            "noise_method": self.noise_method,
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }
