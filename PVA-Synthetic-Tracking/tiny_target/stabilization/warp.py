"""OpenCV CPU reference and CUDA full-resolution stabilization paths."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import importlib
import math
import time
from typing import Any, Mapping

import numpy as np

from ..motion import ComposedMotionState
from ..types import Frame
from .types import StabilizedFrame, ValidSupport


class StabilizationError(RuntimeError):
    """A requested full-resolution stabilization operation cannot run."""


@dataclass(frozen=True, slots=True)
class StabilizationConfig:
    backend: str = "opencv_cuda"
    interpolation: str = "linear"
    border_value: float = 0.0
    valid_mask_erosion_px: int = 2
    integration_window_frames: int = 8
    alignment_sample_stride: int = 8
    skip_exact_identity_warp: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "opencv_cpu", "opencv_cuda"}:
            raise ValueError(
                "stabilization.backend must be auto, opencv_cpu, or opencv_cuda"
            )
        if self.interpolation not in {"linear", "cubic", "lanczos4"}:
            raise ValueError(
                "stabilization.interpolation must be linear, cubic, or lanczos4"
            )
        if self.backend == "opencv_cuda" and self.interpolation == "lanczos4":
            raise ValueError("OpenCV CUDA warp does not support lanczos4")
        if not math.isfinite(self.border_value):
            raise ValueError("stabilization.border_value must be finite")
        for name, minimum in (
            ("valid_mask_erosion_px", 0),
            ("integration_window_frames", 1),
            ("alignment_sample_stride", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"stabilization.{name} must be an integer >= {minimum}")
        if not isinstance(self.skip_exact_identity_warp, bool):
            raise ValueError("stabilization.skip_exact_identity_warp must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "StabilizationConfig":
        data = dict(value or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown stabilization configuration keys: {unknown}")
        return cls(**data)


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise StabilizationError(
            "OpenCV is unavailable; install the vision extra for CPU development "
            "or use the Jetson OpenCV build"
        ) from exc


def _interpolation_flag(cv2: Any, name: str) -> int:
    return {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[name]


def _is_exact_identity(matrix: np.ndarray) -> bool:
    return np.array_equal(np.asarray(matrix, dtype=np.float64), np.eye(3))


class FullResolutionStabilizer:
    """Warp original source pixels directly into a composed reference frame."""

    def __init__(self, config: StabilizationConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            config
            if isinstance(config, StabilizationConfig)
            else StabilizationConfig.from_mapping(config)
        )
        self._cv2 = _load_cv2()
        backend = self.config.backend
        cuda_devices = int(self._cv2.cuda.getCudaEnabledDeviceCount())
        if backend == "auto":
            backend = "opencv_cuda" if cuda_devices > 0 else "opencv_cpu"
        if backend == "opencv_cuda" and cuda_devices <= 0:
            raise StabilizationError("OpenCV reports no CUDA-enabled device")
        if backend == "opencv_cuda" and self.config.interpolation == "lanczos4":
            raise StabilizationError("OpenCV CUDA warp does not support lanczos4")
        self.backend = backend

    def _warp_cpu(
        self,
        image: np.ndarray,
        source_mask: np.ndarray,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        cv2 = self._cv2
        height, width = image.shape
        started = time.perf_counter_ns()
        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=_interpolation_flag(cv2, self.config.interpolation),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.config.border_value,
        )
        image_ms = (time.perf_counter_ns() - started) / 1_000_000
        started = time.perf_counter_ns()
        mask = cv2.warpPerspective(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_ms = (time.perf_counter_ns() - started) / 1_000_000
        return warped, mask, {
            "image_warp": image_ms,
            "valid_mask_warp": mask_ms,
        }

    def _warp_cuda(
        self,
        image: np.ndarray,
        source_mask: np.ndarray,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        cv2 = self._cv2
        height, width = image.shape
        image_gpu = cv2.cuda_GpuMat()
        started = time.perf_counter_ns()
        image_gpu.upload(image)
        upload_ms = (time.perf_counter_ns() - started) / 1_000_000

        started = time.perf_counter_ns()
        warped_gpu = cv2.cuda.warpPerspective(
            image_gpu,
            matrix,
            (width, height),
            flags=_interpolation_flag(cv2, self.config.interpolation),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.config.border_value,
        )
        warped = warped_gpu.download()
        image_ms = (time.perf_counter_ns() - started) / 1_000_000

        started = time.perf_counter_ns()
        mask = cv2.warpPerspective(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_ms = (time.perf_counter_ns() - started) / 1_000_000
        return warped, mask, {
            "host_to_device_upload": upload_ms,
            "image_warp_and_download": image_ms,
            "valid_mask_warp_cpu": mask_ms,
        }

    def stabilize(self, source: Frame, state: ComposedMotionState) -> StabilizedFrame:
        if source.frame_index != state.current_frame_index:
            raise StabilizationError(
                f"Frame {source.frame_index} does not match transform frame "
                f"{state.current_frame_index}"
            )
        matrix = np.asarray(state.reference_from_current_matrix, dtype=np.float64)
        timings: dict[str, float] = {}
        total_started = time.perf_counter_ns()

        started = time.perf_counter_ns()
        source_float = np.ascontiguousarray(source.image, dtype=np.float32)
        source_mask = (
            np.ascontiguousarray(source.valid_mask, dtype=np.uint8)
            if source.valid_mask is not None
            else np.ones(source.shape, dtype=np.uint8)
        )
        timings["float_and_mask_prepare"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000

        exact_identity = _is_exact_identity(matrix)
        if exact_identity and self.config.skip_exact_identity_warp:
            warped = source_float
            warped_mask = source_mask
            resampling_count = 0
            timings["identity_copy_no_resample"] = 0.0
            backend_timings: dict[str, float] = {}
        else:
            if self.backend == "opencv_cuda":
                warped, warped_mask, backend_timings = self._warp_cuda(
                    source_float, source_mask, matrix
                )
            else:
                warped, warped_mask, backend_timings = self._warp_cpu(
                    source_float, source_mask, matrix
                )
            timings.update(backend_timings)
            resampling_count = 1

        pre_erosion_fraction = float(np.mean(warped_mask != 0))
        started = time.perf_counter_ns()
        radius = self.config.valid_mask_erosion_px
        if radius:
            kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
            warped_mask = self._cv2.erode(
                warped_mask,
                kernel,
                borderType=self._cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        valid_mask = np.ascontiguousarray(warped_mask != 0)
        timings["valid_mask_erosion"] = (
            time.perf_counter_ns() - started
        ) / 1_000_000
        warped = np.ascontiguousarray(warped, dtype=np.float32)
        warped.setflags(write=False)
        valid_mask.setflags(write=False)
        started = time.perf_counter_ns()
        stabilized_frame = replace(source, image=warped, valid_mask=valid_mask)
        timings["frame_contract"] = (time.perf_counter_ns() - started) / 1_000_000
        timings["total"] = (time.perf_counter_ns() - total_started) / 1_000_000
        transferred_bytes = (
            source_float.nbytes
            + warped.nbytes
            + source_mask.nbytes
            + warped_mask.nbytes
        )
        effective_ms = sum(backend_timings.values())
        metrics: dict[str, Any] = {
            "full_image_size": [source.shape[1], source.shape[0]],
            "source_dtype": source.image.dtype.str,
            "output_dtype": stabilized_frame.image.dtype.str,
            "source_valid_fraction": float(np.mean(source_mask != 0)),
            "warped_valid_fraction_before_erosion": pre_erosion_fraction,
            "valid_fraction": float(np.mean(valid_mask)),
            "invalid_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
            "transferred_bytes_estimate": transferred_bytes,
            "effective_memory_bandwidth_gib_s": (
                transferred_bytes / (1024**3) / (effective_ms / 1000)
                if resampling_count == 1 and effective_ms > 0
                else None
            ),
            "exact_identity": exact_identity,
            "window_reset": state.window_reset,
        }
        return StabilizedFrame(
            frame=stabilized_frame,
            reference_frame_index=state.reference_frame_index,
            segment_index=state.segment_index,
            source_to_reference_matrix=matrix,
            interpolation=self.config.interpolation,
            backend=self.backend,
            resampling_count=resampling_count,
            metrics=metrics,
            timings_ms=timings,
        )


class ValidMaskWindow:
    """Maintain valid support for a bounded single-segment integration window."""

    def __init__(self, maximum_frames: int) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self.maximum_frames = maximum_frames
        self._items: deque[tuple[int, np.ndarray]] = deque(maxlen=maximum_frames)
        self._segment_index: int | None = None

    def update(self, stabilized: StabilizedFrame) -> ValidSupport:
        if self._segment_index != stabilized.segment_index:
            self._items.clear()
            self._segment_index = stabilized.segment_index
        assert stabilized.frame.valid_mask is not None
        self._items.append(
            (stabilized.frame.frame_index, stabilized.frame.valid_mask.copy())
        )
        shape = self._items[0][1].shape
        support = np.zeros(shape, dtype=np.uint16)
        for _frame_index, mask in self._items:
            if mask.shape != shape:
                raise StabilizationError("valid-mask shape changed within a window")
            support += mask
        count = len(self._items)
        return ValidSupport(
            common_valid_mask=support == count,
            support_count=support,
            frame_count=count,
            segment_index=stabilized.segment_index,
            first_frame_index=self._items[0][0],
            last_frame_index=self._items[-1][0],
        )


def alignment_improvement_metrics(
    previous_source: Frame,
    current_source: Frame,
    previous_stabilized: StabilizedFrame,
    current_stabilized: StabilizedFrame,
    *,
    sample_stride: int = 8,
) -> dict[str, float | int | None]:
    """Compare robust pair differences before and after actual image warps."""

    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if previous_source.shape != current_source.shape:
        raise ValueError("source frame shapes differ")
    if previous_stabilized.frame.shape != current_stabilized.frame.shape:
        raise ValueError("stabilized frame shapes differ")
    if previous_stabilized.segment_index != current_stabilized.segment_index:
        return {
            "sample_count": 0,
            "median_absolute_difference_before": None,
            "median_absolute_difference_after": None,
            "p90_absolute_difference_before": None,
            "p90_absolute_difference_after": None,
            "median_reduction_fraction": None,
        }
    previous_mask = previous_stabilized.frame.valid_mask
    current_mask = current_stabilized.frame.valid_mask
    assert previous_mask is not None and current_mask is not None
    valid = (previous_mask & current_mask)[::sample_stride, ::sample_stride]
    before = np.abs(
        current_source.image[::sample_stride, ::sample_stride].astype(np.float32)
        - previous_source.image[::sample_stride, ::sample_stride].astype(np.float32)
    )[valid]
    after = np.abs(
        current_stabilized.frame.image[::sample_stride, ::sample_stride]
        - previous_stabilized.frame.image[::sample_stride, ::sample_stride]
    )[valid]
    if len(after) == 0:
        raise StabilizationError("no common valid samples for alignment measurement")
    before_median = float(np.median(before))
    after_median = float(np.median(after))
    return {
        "sample_count": len(after),
        "median_absolute_difference_before": before_median,
        "median_absolute_difference_after": after_median,
        "p90_absolute_difference_before": float(np.percentile(before, 90)),
        "p90_absolute_difference_after": float(np.percentile(after, 90)),
        "median_reduction_fraction": (
            (before_median - after_median) / before_median
            if before_median > 0
            else None
        ),
    }


def signal_preservation_metrics(
    source: np.ndarray,
    warped: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """Measure peak, signed flux, and L2 energy retained by one image warp."""

    before = np.asarray(source, dtype=np.float64)
    after = np.asarray(warped, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if before.shape != after.shape or before.shape != valid.shape or before.ndim != 2:
        raise ValueError("source, warped, and valid_mask must share a 2-D shape")
    source_peak = float(np.max(before))
    source_flux = float(np.sum(before))
    source_energy = float(np.sum(before * before))
    valid_after = after[valid]
    output_peak = float(np.max(valid_after))
    output_flux = float(np.sum(valid_after))
    output_energy = float(np.sum(valid_after * valid_after))
    return {
        "source_peak": source_peak,
        "output_peak": output_peak,
        "peak_retention": output_peak / source_peak if source_peak else math.nan,
        "source_flux": source_flux,
        "output_flux": output_flux,
        "flux_retention": output_flux / source_flux if source_flux else math.nan,
        "source_l2_energy": source_energy,
        "output_l2_energy": output_energy,
        "l2_energy_retention": (
            output_energy / source_energy if source_energy else math.nan
        ),
        "output_minimum": float(np.min(valid_after)),
        "output_maximum": output_peak,
    }
