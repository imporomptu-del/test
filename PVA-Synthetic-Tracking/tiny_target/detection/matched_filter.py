"""Support-aware PSF matched filtering with a deterministic NumPy reference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from ..preprocessing import ResidualFrame
from .types import MatchedFilterFrame, PsfKernelBank


class MatchedFilterError(RuntimeError):
    """The configured PSF matched filter cannot run."""


@dataclass(frozen=True, slots=True)
class MatchedFilterConfig:
    source: str = "gaussian"
    kernel_path: str | None = None
    gaussian_sigma_px: float = 0.8
    radius_px: int = 3
    phases_per_axis: int = 1
    normalize: str = "snr_l2"
    polarity: str = "bright"
    backend: str = "numpy_reference"
    minimum_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.source not in {"gaussian", "npy"}:
            raise ValueError("psf.source must be gaussian or npy")
        if self.source == "npy" and not self.kernel_path:
            raise ValueError("psf.kernel_path is required when source=npy")
        if not math.isfinite(self.gaussian_sigma_px) or self.gaussian_sigma_px <= 0:
            raise ValueError("psf.gaussian_sigma_px must be finite and positive")
        if isinstance(self.radius_px, bool) or not isinstance(self.radius_px, int) or self.radius_px < 1:
            raise ValueError("psf.radius_px must be an integer >= 1")
        if (
            isinstance(self.phases_per_axis, bool)
            or not isinstance(self.phases_per_axis, int)
            or not 1 <= self.phases_per_axis <= 8
        ):
            raise ValueError("psf.phases_per_axis must be an integer in [1, 8]")
        if self.normalize != "snr_l2":
            raise ValueError("psf.normalize must be snr_l2")
        if self.polarity not in {"bright", "dark", "both"}:
            raise ValueError("psf.polarity must be bright, dark, or both")
        if self.backend not in {"numpy_reference", "opencv_cpu"}:
            raise ValueError("psf.backend must be numpy_reference or opencv_cpu")
        if (
            not math.isfinite(self.minimum_valid_fraction)
            or not 0 < self.minimum_valid_fraction <= 1
        ):
            raise ValueError("psf.minimum_valid_fraction must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MatchedFilterConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown PSF configuration keys: {unknown}")
        return cls(**data)


def _phase_offsets(count: int) -> np.ndarray:
    if count == 1:
        values = np.array([0.0], np.float32)
    else:
        values = ((np.arange(count, dtype=np.float32) + 0.5) / count) - 0.5
    return np.array([(x, y) for y in values for x in values], np.float32)


def integrated_gaussian_kernel(
    sigma: float,
    radius: int,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    denominator = math.sqrt(2.0) * sigma

    def axis(offset: float) -> np.ndarray:
        return np.array(
            [
                0.5
                * (
                    math.erf((coordinate + 0.5 - offset) / denominator)
                    - math.erf((coordinate - 0.5 - offset) / denominator)
                )
                for coordinate in coordinates
            ],
            np.float64,
        )

    kernel = np.outer(axis(offset_y), axis(offset_x))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def build_kernel_bank(
    config: MatchedFilterConfig | Mapping[str, Any] | None = None,
    *,
    base_path: str | Path | None = None,
) -> PsfKernelBank:
    resolved = (
        config
        if isinstance(config, MatchedFilterConfig)
        else MatchedFilterConfig.from_mapping(config)
    )
    if resolved.source == "gaussian":
        offsets = _phase_offsets(resolved.phases_per_axis)
        kernels = np.stack(
            [
                integrated_gaussian_kernel(
                    resolved.gaussian_sigma_px,
                    resolved.radius_px,
                    float(offset[0]),
                    float(offset[1]),
                )
                for offset in offsets
            ]
        )
        identity = {
            "model": "pixel_integrated_isotropic_gaussian",
            "sigma_px": resolved.gaussian_sigma_px,
            "radius_px": resolved.radius_px,
        }
        return PsfKernelBank(
            kernels=kernels,
            phase_offsets_xy=offsets,
            source="gaussian",
            source_identity=identity,
            provisional=True,
        )

    assert resolved.kernel_path is not None
    path = Path(resolved.kernel_path).expanduser()
    if not path.is_absolute() and base_path is not None:
        path = Path(base_path) / path
    path = path.resolve()
    try:
        value = np.load(path, allow_pickle=False)
    except OSError as exc:
        raise MatchedFilterError(f"Cannot load PSF kernel {path}: {exc}") from exc
    kernels = np.asarray(value, dtype=np.float64)
    if kernels.ndim == 2:
        kernels = kernels[None, ...]
    if kernels.ndim != 3:
        raise MatchedFilterError("NPY PSF must be 2-D or [phase, height, width]")
    sums = np.sum(kernels, axis=(1, 2))
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
        raise MatchedFilterError("NPY PSF kernels must have finite positive flux")
    kernels = kernels / sums[:, None, None]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    offsets = (
        _phase_offsets(resolved.phases_per_axis)
        if resolved.phases_per_axis**2 == kernels.shape[0]
        else np.zeros((kernels.shape[0], 2), np.float32)
    )
    return PsfKernelBank(
        kernels=kernels.astype(np.float32),
        phase_offsets_xy=offsets,
        source="npy",
        source_identity={"path": str(path), "sha256": digest},
        provisional=False,
    )


def correlate_reference(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Transparent zero-border 2-D correlation used as the golden reference."""

    source = np.asarray(image, dtype=np.float32)
    weights = np.asarray(kernel, dtype=np.float32)
    if source.ndim != 2 or weights.ndim != 2:
        raise ValueError("image and kernel must be 2-D")
    if weights.shape[0] % 2 != 1 or weights.shape[1] % 2 != 1:
        raise ValueError("kernel dimensions must be odd")
    y_radius = weights.shape[0] // 2
    x_radius = weights.shape[1] // 2
    padded = np.pad(
        source,
        ((y_radius, y_radius), (x_radius, x_radius)),
        mode="constant",
    )
    result = np.zeros(source.shape, np.float32)
    for kernel_y in range(weights.shape[0]):
        for kernel_x in range(weights.shape[1]):
            weight = weights[kernel_y, kernel_x]
            if weight != 0:
                result += weight * padded[
                    kernel_y : kernel_y + source.shape[0],
                    kernel_x : kernel_x + source.shape[1],
                ]
    return result


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise MatchedFilterError(
            "OpenCV is unavailable; use numpy_reference or install the vision extra"
        ) from exc


def _robust_scale(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


class PsfMatchedFilter:
    """Correlate whitened residuals against a PSF phase bank in SNR units."""

    def __init__(
        self,
        config: MatchedFilterConfig | Mapping[str, Any] | None = None,
        *,
        base_path: str | Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, MatchedFilterConfig)
            else MatchedFilterConfig.from_mapping(config)
        )
        self.bank = build_kernel_bank(self.config, base_path=base_path)
        self._cv2 = _load_cv2() if self.config.backend == "opencv_cpu" else None
        footprint = np.any(self.bank.kernels != 0, axis=0).astype(np.float32)
        self._footprint = footprint
        self._full_support = int(np.count_nonzero(footprint))
        self._minimum_support = int(
            math.ceil(self._full_support * self.config.minimum_valid_fraction)
        )

    def _correlate(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        if self._cv2 is None:
            return correlate_reference(image, kernel)
        return self._cv2.filter2D(
            np.asarray(image, np.float32),
            self._cv2.CV_32F,
            np.asarray(kernel, np.float32),
            borderType=self._cv2.BORDER_CONSTANT,
        )

    def process(self, residual: ResidualFrame) -> MatchedFilterFrame:
        started_total = time.perf_counter_ns()
        shape = residual.whitened.shape
        timings: dict[str, float] = {}
        if not residual.detection_ready or not np.any(residual.valid_mask):
            zeros = np.zeros(shape, np.float32)
            valid = np.zeros(shape, bool)
            support = np.zeros(shape, np.uint16)
            phase = np.zeros(shape, np.uint16)
            timings["suppressed_no_filter"] = 0.0
            timings["total"] = (time.perf_counter_ns() - started_total) / 1_000_000
            return MatchedFilterFrame(
                response=zeros,
                phase_index=phase,
                valid_mask=valid,
                valid_support_count=support,
                timestamp_ns=residual.timestamp_ns,
                frame_index=residual.frame_index,
                reference_frame_index=residual.reference_frame_index,
                segment_index=residual.segment_index,
                detection_ready=False,
                polarity=self.config.polarity,
                backend=self.config.backend,
                kernel_metadata=self.bank.metadata(),
                metrics={
                    "valid_fraction": 0.0,
                    "suppressed": True,
                    "suppression_reason": "preprocessing_not_detection_ready",
                },
                timings_ms=timings,
            )

        started = time.perf_counter_ns()
        support_float = self._correlate(
            residual.valid_mask.astype(np.float32), self._footprint
        )
        support = np.rint(support_float).astype(np.uint16)
        valid = support >= self._minimum_support
        timings["valid_support"] = (time.perf_counter_ns() - started) / 1_000_000

        started = time.perf_counter_ns()
        response: np.ndarray | None = None
        phase = np.zeros(shape, np.uint16)
        for phase_index, kernel in enumerate(self.bank.kernels):
            l2_norm = float(np.sqrt(np.sum(kernel.astype(np.float64) ** 2)))
            candidate = np.asarray(
                self._correlate(residual.whitened, kernel) / l2_norm,
                np.float32,
            )
            if response is None:
                response = candidate
                continue
            if self.config.polarity == "bright":
                better = candidate > response
            elif self.config.polarity == "dark":
                better = candidate < response
            else:
                better = np.abs(candidate) > np.abs(response)
            response[better] = candidate[better]
            phase[better] = phase_index
        assert response is not None
        timings["phase_bank_correlation_and_selection"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000

        started = time.perf_counter_ns()
        response[~valid] = 0
        phase[~valid] = 0
        timings["mask_output"] = (time.perf_counter_ns() - started) / 1_000_000

        values = response[valid]
        if self.config.polarity == "bright":
            scores = values
        elif self.config.polarity == "dark":
            scores = -values
        else:
            scores = np.abs(values)
        metrics: dict[str, Any] = {
            "valid_fraction": float(np.mean(valid)),
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "support": {
                "kernel_footprint_pixels": self._full_support,
                "minimum_required": self._minimum_support,
                "minimum_observed_valid": (
                    int(np.min(support[valid])) if values.size else None
                ),
            },
            "response": {
                "median": float(np.median(values)) if values.size else None,
                "robust_scale": _robust_scale(values),
                "p99_absolute": (
                    float(np.percentile(np.abs(values), 99)) if values.size else None
                ),
                "maximum": float(np.max(values)) if values.size else None,
                "minimum": float(np.min(values)) if values.size else None,
            },
            "score": {
                "maximum": float(np.max(scores)) if scores.size else None,
                "p99": float(np.percentile(scores, 99)) if scores.size else None,
                "fraction_gt_5": (
                    float(np.mean(scores > 5)) if scores.size else None
                ),
            },
            "selected_phase_counts": np.bincount(
                phase[valid], minlength=self.bank.phase_count
            ).tolist(),
            "suppressed": False,
        }
        timings["total"] = (time.perf_counter_ns() - started_total) / 1_000_000
        return MatchedFilterFrame(
            response=response,
            phase_index=phase,
            valid_mask=valid,
            valid_support_count=support,
            timestamp_ns=residual.timestamp_ns,
            frame_index=residual.frame_index,
            reference_frame_index=residual.reference_frame_index,
            segment_index=residual.segment_index,
            detection_ready=True,
            polarity=self.config.polarity,
            backend=self.config.backend,
            kernel_metadata=self.bank.metadata(),
            metrics=metrics,
            timings_ms=timings,
        )
