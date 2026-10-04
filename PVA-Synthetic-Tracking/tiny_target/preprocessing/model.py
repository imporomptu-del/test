"""Deterministic reference and bounded-state streaming preprocessing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any, Mapping
import warnings

import numpy as np

from ..stabilization import StabilizedFrame
from ..types import Frame
from .types import ResidualFrame


class PreprocessingError(RuntimeError):
    """A background/noise preprocessing contract cannot be satisfied."""


def _positive_finite(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    method: str = "temporal_median"
    warmup_frames: int = 4
    history_frames: int = 8
    minimum_history_samples: int = 3
    update_rate: float = 0.05
    outlier_clip_sigma: float = 4.0
    update_exclusion_sigma: float = 3.0
    global_change_median_sigma: float | None = None
    global_change_robust_scale: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"temporal_median", "robust_running"}:
            raise ValueError(
                "background.method must be temporal_median or robust_running"
            )
        for name in ("warmup_frames", "history_frames", "minimum_history_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"background.{name} must be a positive integer")
        if self.warmup_frames > self.history_frames:
            raise ValueError("background.warmup_frames cannot exceed history_frames")
        if self.minimum_history_samples > self.warmup_frames:
            raise ValueError(
                "background.minimum_history_samples cannot exceed warmup_frames"
            )
        _positive_finite(self.outlier_clip_sigma, "background.outlier_clip_sigma")
        _positive_finite(
            self.update_exclusion_sigma, "background.update_exclusion_sigma"
        )
        for name in ("global_change_median_sigma", "global_change_robust_scale"):
            value = getattr(self, name)
            if value is not None:
                _positive_finite(value, f"background.{name}")
        if not math.isfinite(self.update_rate) or not 0 < self.update_rate <= 1:
            raise ValueError("background.update_rate must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BackgroundConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown background configuration keys: {unknown}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    method: str = "temporal_mad"
    sigma_floor: float = 1.0
    mad_scale: float = 1.4826
    mask_saturated: bool = True
    saturation_value: float | None = None
    dead_level_max: float | None = None
    unstable_sigma_ceiling: float | None = None
    bad_pixel_map: str | None = None

    def __post_init__(self) -> None:
        if self.method not in {"temporal_mad", "robust_ewma"}:
            raise ValueError("noise.method must be temporal_mad or robust_ewma")
        _positive_finite(self.sigma_floor, "noise.sigma_floor")
        _positive_finite(self.mad_scale, "noise.mad_scale")
        if not isinstance(self.mask_saturated, bool):
            raise ValueError("noise.mask_saturated must be boolean")
        for name in ("saturation_value", "unstable_sigma_ceiling"):
            value = getattr(self, name)
            if value is not None:
                _positive_finite(value, f"noise.{name}")
        if self.dead_level_max is not None and (
            not math.isfinite(self.dead_level_max) or self.dead_level_max < 0
        ):
            raise ValueError("noise.dead_level_max must be finite and non-negative")
        if self.bad_pixel_map is not None and not isinstance(self.bad_pixel_map, str):
            raise ValueError("noise.bad_pixel_map must be a path string or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "NoiseConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown noise configuration keys: {unknown}")
        return cls(**data)


def load_bad_pixel_map(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a boolean NPY mask where true means the pixel must be excluded."""

    value = np.load(Path(path).expanduser(), allow_pickle=False)
    if value.shape != shape:
        raise PreprocessingError(
            f"Bad-pixel map shape {value.shape} does not match frame shape {shape}"
        )
    if value.dtype != np.bool_:
        raise PreprocessingError("Bad-pixel map must have boolean dtype")
    return np.ascontiguousarray(value)


def _robust_scale(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


class RobustPreprocessor:
    """Create signed residuals without allowing a frame to model itself."""

    def __init__(
        self,
        background: BackgroundConfig | Mapping[str, Any] | None = None,
        noise: NoiseConfig | Mapping[str, Any] | None = None,
        *,
        bad_pixel_mask: np.ndarray | None = None,
    ) -> None:
        self.background = (
            background
            if isinstance(background, BackgroundConfig)
            else BackgroundConfig.from_mapping(background)
        )
        self.noise = (
            noise if isinstance(noise, NoiseConfig) else NoiseConfig.from_mapping(noise)
        )
        self._bad_pixel_mask = (
            np.ascontiguousarray(bad_pixel_mask, dtype=bool)
            if bad_pixel_mask is not None
            else None
        )
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.background.history_frames
        )
        self._segment_index: int | None = None
        self._shape: tuple[int, int] | None = None
        self._location: np.ndarray | None = None
        self._variance: np.ndarray | None = None
        self._support: np.ndarray | None = None

    def _reset(self, segment_index: int, shape: tuple[int, int]) -> None:
        self._history.clear()
        self._segment_index = segment_index
        self._shape = shape
        self._location = np.zeros(shape, np.float32)
        self._variance = np.full(shape, self.noise.sigma_floor**2, np.float32)
        self._support = np.zeros(shape, np.uint16)
        if self._bad_pixel_mask is not None and self._bad_pixel_mask.shape != shape:
            raise PreprocessingError(
                f"Bad-pixel mask shape {self._bad_pixel_mask.shape} != {shape}"
            )

    def _input_valid(self, stabilized: StabilizedFrame) -> np.ndarray:
        image = stabilized.frame.image
        assert stabilized.frame.valid_mask is not None
        return self._radiometric_valid(
            image,
            stabilized.frame.valid_mask,
            stabilized.frame.bit_depth,
        )

    def _radiometric_valid(
        self,
        image: np.ndarray,
        base_valid: np.ndarray,
        bit_depth: int,
    ) -> np.ndarray:
        valid = np.asarray(base_valid, dtype=bool) & np.isfinite(image)
        if self.noise.mask_saturated:
            threshold = self.noise.saturation_value
            if threshold is None:
                threshold = float((1 << bit_depth) - 1)
            valid &= image < threshold
        if self.noise.dead_level_max is not None:
            valid &= image > self.noise.dead_level_max
        if self._bad_pixel_mask is not None:
            if self._bad_pixel_mask.shape != image.shape:
                raise PreprocessingError(
                    f"Bad-pixel mask shape {self._bad_pixel_mask.shape} != {image.shape}"
                )
            valid &= ~self._bad_pixel_mask
        return np.ascontiguousarray(valid)

    def prepare_source(self, source: Frame) -> Frame:
        """Attach radiometric validity before the mask is geometrically warped."""

        base_valid = (
            source.valid_mask
            if source.valid_mask is not None
            else np.ones(source.shape, bool)
        )
        valid = self._radiometric_valid(source.image, base_valid, source.bit_depth)
        return replace(source, valid_mask=valid)

    def _temporal_statistics(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._shape is not None
        if not self._history:
            return (
                np.zeros(self._shape, np.float32),
                np.full(self._shape, self.noise.sigma_floor, np.float32),
                np.zeros(self._shape, np.uint16),
            )
        images = np.stack([item[0] for item in self._history])
        masks = np.stack([item[1] for item in self._history])
        samples = np.where(masks, images, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            location = np.nanmedian(samples, axis=0)
            deviation = np.abs(samples - location)
            sigma = self.noise.mad_scale * np.nanmedian(deviation, axis=0)
        # OpenCV's Jetson build may enable flush-to-zero globally. Avoid
        # ``nan_to_num``, whose dtype-limit lookup warns under that FP mode.
        location = np.where(np.isfinite(location), location, 0.0).astype(np.float32)
        sigma = np.where(
            np.isfinite(sigma), sigma, self.noise.sigma_floor
        ).astype(np.float32)
        np.maximum(sigma, self.noise.sigma_floor, out=sigma)
        support = np.sum(masks, axis=0, dtype=np.uint16)
        return location, sigma, support

    def _prior_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._location is not None
        assert self._variance is not None
        assert self._support is not None
        if self.background.method == "temporal_median":
            location, temporal_sigma, support = self._temporal_statistics()
        else:
            location = self._location.copy()
            temporal_sigma = np.sqrt(
                np.maximum(self._variance, self.noise.sigma_floor**2)
            ).astype(np.float32)
            support = self._support.copy()
        if self.noise.method == "temporal_mad":
            _ignored, sigma, temporal_support = self._temporal_statistics()
            support = np.minimum(support, temporal_support)
        else:
            sigma = np.sqrt(
                np.maximum(self._variance, self.noise.sigma_floor**2)
            ).astype(np.float32)
        return location, sigma, support

    def _update_running(
        self,
        image: np.ndarray,
        valid: np.ndarray,
        whitened: np.ndarray,
        model_ready: np.ndarray,
    ) -> float:
        assert self._location is not None
        assert self._variance is not None
        assert self._support is not None
        unseen = valid & (self._support == 0)
        self._location[unseen] = image[unseen]
        self._variance[unseen] = self.noise.sigma_floor**2

        update = valid & ~unseen
        outlier = update & model_ready & (
            np.abs(whitened) >= self.background.update_exclusion_sigma
        )
        update &= ~outlier
        innovation = image - self._location
        scale = np.sqrt(np.maximum(self._variance, self.noise.sigma_floor**2))
        clipped = np.clip(
            innovation,
            -self.background.outlier_clip_sigma * scale,
            self.background.outlier_clip_sigma * scale,
        )
        rate = self.background.update_rate
        self._location[update] += rate * clipped[update]
        self._variance[update] = (
            (1.0 - rate) * self._variance[update] + rate * clipped[update] ** 2
        )
        increment = valid & (self._support < np.iinfo(np.uint16).max)
        self._support[increment] += 1
        return float(np.mean(outlier))

    def process(self, stabilized: StabilizedFrame) -> ResidualFrame:
        image = np.asarray(stabilized.frame.image, dtype=np.float32)
        if (
            self._segment_index != stabilized.segment_index
            or self._shape != stabilized.frame.shape
        ):
            self._reset(stabilized.segment_index, stabilized.frame.shape)
        total_started = time.perf_counter_ns()
        timings: dict[str, float] = {}

        started = time.perf_counter_ns()
        input_valid = self._input_valid(stabilized)
        timings["input_mask"] = (time.perf_counter_ns() - started) / 1_000_000

        started = time.perf_counter_ns()
        location, sigma, support = self._prior_model()
        timings["background_and_noise_estimate"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000
        history_count = len(self._history)
        globally_ready = history_count >= self.background.warmup_frames
        model_ready = support >= self.background.minimum_history_samples
        detection_valid = input_valid & model_ready
        if self.noise.unstable_sigma_ceiling is not None:
            detection_valid &= sigma <= self.noise.unstable_sigma_ceiling
        if not globally_ready:
            detection_valid[:] = False

        started = time.perf_counter_ns()
        residual = np.zeros_like(image, dtype=np.float32)
        whitened = np.zeros_like(image, dtype=np.float32)
        estimable = input_valid & (support > 0)
        residual[estimable] = image[estimable] - location[estimable]
        whitened[estimable] = residual[estimable] / sigma[estimable]
        timings["subtract_and_normalize"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000

        model_values = whitened[input_valid & model_ready]
        global_whitened_median = (
            float(np.median(model_values)) if model_values.size else None
        )
        global_whitened_scale = (
            _robust_scale(model_values) if model_values.size else None
        )
        global_change = bool(
            globally_ready
            and (
                (
                    global_whitened_median is not None
                    and self.background.global_change_median_sigma is not None
                    and abs(global_whitened_median)
                    >= self.background.global_change_median_sigma
                )
                or (
                    global_whitened_scale is not None
                    and self.background.global_change_robust_scale is not None
                    and global_whitened_scale
                    >= self.background.global_change_robust_scale
                )
            )
        )
        if global_change:
            detection_valid[:] = False
        detection_ready = globally_ready and not global_change

        started = time.perf_counter_ns()
        excluded_fraction = 0.0
        if self.background.method == "robust_running":
            excluded_fraction = self._update_running(
                image,
                input_valid,
                whitened,
                model_ready & globally_ready & (not global_change),
            )
        elif self.noise.method == "robust_ewma":
            # Maintain bounded-state scale even when the median is the location.
            assert self._variance is not None and self._support is not None
            rate = self.background.update_rate
            update = input_valid & (self._support > 0)
            innovation = image - location
            clip = self.background.outlier_clip_sigma * sigma
            clipped = np.clip(innovation, -clip, clip)
            self._variance[update] = (
                (1.0 - rate) * self._variance[update] + rate * clipped[update] ** 2
            )
            unseen = input_valid & (self._support == 0)
            self._variance[unseen] = self.noise.sigma_floor**2
            increment = input_valid & (self._support < np.iinfo(np.uint16).max)
            self._support[increment] += 1
        self._history.append((image.copy(), input_valid.copy()))
        timings["model_update"] = (time.perf_counter_ns() - started) / 1_000_000

        analysis = detection_valid if np.any(detection_valid) else estimable
        residual_values = residual[analysis]
        whitened_values = whitened[analysis]
        residual_median = (
            float(np.median(residual_values)) if residual_values.size else None
        )
        whitened_median = (
            float(np.median(whitened_values)) if whitened_values.size else None
        )
        metrics: dict[str, Any] = {
            "source_valid_fraction": float(np.mean(stabilized.frame.valid_mask)),
            "update_valid_fraction": float(np.mean(input_valid)),
            "detection_valid_fraction": float(np.mean(detection_valid)),
            "valid_pixel_count": int(np.count_nonzero(detection_valid)),
            "warmup_complete": globally_ready,
            "global_change_suppressed": global_change,
            "global_whitened_median": global_whitened_median,
            "global_whitened_robust_scale": global_whitened_scale,
            "model_valid_fraction": float(np.mean(model_ready)),
            "outlier_update_excluded_fraction": excluded_fraction,
            "residual": {
                "median": residual_median,
                "robust_scale": (
                    _robust_scale(residual_values) if residual_values.size else None
                ),
                "p95_absolute": (
                    float(np.percentile(np.abs(residual_values), 95))
                    if residual_values.size
                    else None
                ),
                "p99_absolute": (
                    float(np.percentile(np.abs(residual_values), 99))
                    if residual_values.size
                    else None
                ),
            },
            "whitened": {
                "median": whitened_median,
                "robust_scale": (
                    _robust_scale(whitened_values) if whitened_values.size else None
                ),
                "p99_absolute": (
                    float(np.percentile(np.abs(whitened_values), 99))
                    if whitened_values.size
                    else None
                ),
                "maximum": (
                    float(np.max(whitened_values)) if whitened_values.size else None
                ),
                "minimum": (
                    float(np.min(whitened_values)) if whitened_values.size else None
                ),
                "fraction_abs_gt_3": (
                    float(np.mean(np.abs(whitened_values) > 3))
                    if whitened_values.size
                    else None
                ),
                "fraction_abs_gt_5": (
                    float(np.mean(np.abs(whitened_values) > 5))
                    if whitened_values.size
                    else None
                ),
            },
            "sigma": {
                "median": (
                    float(np.median(sigma[model_ready]))
                    if np.any(model_ready)
                    else None
                ),
                "p90": (
                    float(np.percentile(sigma[model_ready], 90))
                    if np.any(model_ready)
                    else None
                ),
                "floor_fraction": (
                    float(np.mean(sigma[model_ready] <= self.noise.sigma_floor))
                    if np.any(model_ready)
                    else None
                ),
            },
        }
        timings["total"] = (time.perf_counter_ns() - total_started) / 1_000_000
        return ResidualFrame(
            value=residual,
            sigma=sigma,
            whitened=whitened,
            valid_mask=detection_valid,
            timestamp_ns=stabilized.frame.timestamp_ns,
            frame_index=stabilized.frame.frame_index,
            reference_frame_index=stabilized.reference_frame_index,
            segment_index=stabilized.segment_index,
            detection_ready=detection_ready,
            history_frames=history_count,
            background_method=self.background.method,
            noise_method=self.noise.method,
            metrics=metrics,
            timings_ms=timings,
        )
