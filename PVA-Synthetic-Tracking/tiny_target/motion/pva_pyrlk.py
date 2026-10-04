"""Jetson VPI implementation of sparse background-feature tracking.

The module deliberately does not import :mod:`vpi` at import time. Geometry,
configuration, and report code can therefore run on development machines; the
hardware dependency is resolved only when ``PvaPyrLkMotionEstimator`` is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from functools import lru_cache
import importlib
import math
import time
from typing import Any, Mapping

import numpy as np

from ..types import Frame
from .geometry import (
    correspondence_acceptance_mask,
    grid_coverage,
    lift_points_to_full_resolution,
    select_spatially_distributed,
)
from .types import MotionCorrespondences


class PvaMotionError(RuntimeError):
    """The requested PVA motion operation cannot produce correspondences."""


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"motion.{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"motion.{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"motion.{name} must be >= {minimum}")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"motion.{name} must be a number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"motion.{name} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class PvaMotionConfig:
    feature_intensity_mapping: str = field(default="bit_shift", kw_only=True)
    optical_flow_backend: str = field(default="PVA", kw_only=True)
    harris_capacity_policy: str = field(default="legacy_default", kw_only=True)
    flow_status_policy: str = field(default="legacy_default", kw_only=True)
    feature_cpu_policy: str = field(default="reference", kw_only=True)
    feature_image_scale: float = 0.5
    max_features: int = 1000
    grid_rows: int = 6
    grid_cols: int = 8
    max_features_per_cell: int | None = None
    feature_border_px: float = 16.0
    saturation_fraction: float = 0.995
    saturated_neighborhood_radius_px: int = 2
    exclusion_regions_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    harris_strength: float = 1.0
    harris_sensitivity: float = 0.0625
    harris_gradient_size: int = 3
    harris_block_size: int = 3
    harris_min_nms_distance: int = 8
    pyramid_levels: int = 4
    pyramid_scale: float = 0.5
    pyramid_backend: str = "auto"
    window_size: int = 11
    max_iterations: int = 6
    forward_backward_check: bool = True
    max_forward_backward_error_px: float = 3.0
    max_displacement_px: float = 120.0
    minimum_accepted_features: int = 30
    minimum_grid_coverage: float = 0.25

    def __post_init__(self) -> None:
        if self.feature_cpu_policy not in {"reference", "batched_exact_v1"}:
            raise ValueError("Unknown motion.feature_cpu_policy")
        if self.flow_status_policy not in {"legacy_default", "fresh_per_pair"}:
            raise ValueError("Unknown motion.flow_status_policy")
        if self.flow_status_policy == "fresh_per_pair" and self.optical_flow_backend != "CUDA":
            raise ValueError("fresh_per_pair is currently validated only for CUDA flow")
        if self.harris_capacity_policy not in {"legacy_default", "complete_grid"}:
            raise ValueError("Unknown motion.harris_capacity_policy")
        if self.harris_capacity_policy == "complete_grid" and self.harris_min_nms_distance != 8:
            raise ValueError("complete_grid requires the PVA eight-pixel NMS cell")
        if self.feature_intensity_mapping not in {"bit_shift", "raw_asinh_v1", "raw_linear_u16_v1", "raw_robust_u16_v1"}:
            raise ValueError("Unknown motion.feature_intensity_mapping")
        if self.optical_flow_backend not in {"PVA", "CUDA"}:
            raise ValueError("motion.optical_flow_backend must be PVA or CUDA")
        if self.feature_intensity_mapping == "raw_robust_u16_v1" and self.optical_flow_backend != "CUDA":
            raise ValueError("raw_robust_u16_v1 requires the validated CUDA optical-flow path")
        if not 0 < self.feature_image_scale <= 1:
            raise ValueError("motion.feature_image_scale must be in (0, 1]")
        for name in (
            "max_features",
            "grid_rows",
            "grid_cols",
            "harris_gradient_size",
            "harris_block_size",
            "harris_min_nms_distance",
            "pyramid_levels",
            "window_size",
            "max_iterations",
            "minimum_accepted_features",
        ):
            _integer(getattr(self, name), name)
        _integer(
            self.saturated_neighborhood_radius_px,
            "saturated_neighborhood_radius_px",
            minimum=0,
        )
        if self.max_features_per_cell is not None:
            _integer(self.max_features_per_cell, "max_features_per_cell")
        for name in (
            "harris_strength",
            "harris_sensitivity",
            "pyramid_scale",
            "max_forward_backward_error_px",
            "max_displacement_px",
        ):
            _positive_float(getattr(self, name), name)
        if not 0 < self.pyramid_scale < 1:
            raise ValueError("motion.pyramid_scale must be in (0, 1)")
        if self.pyramid_backend not in {"auto", "PVA", "CUDA"}:
            raise ValueError("motion.pyramid_backend must be auto, PVA, or CUDA")
        if not 0 <= self.minimum_grid_coverage <= 1:
            raise ValueError("motion.minimum_grid_coverage must be in [0, 1]")
        if self.feature_border_px < 0 or not math.isfinite(self.feature_border_px):
            raise ValueError("motion.feature_border_px must be non-negative and finite")
        if not 0 < self.saturation_fraction <= 1:
            raise ValueError("motion.saturation_fraction must be in (0, 1]")
        for region in self.exclusion_regions_xyxy:
            if len(region) != 4 or not all(math.isfinite(item) for item in region):
                raise ValueError(
                    "each motion exclusion region must be finite x1,y1,x2,y2"
                )
            if region[2] <= region[0] or region[3] <= region[1]:
                raise ValueError("motion exclusion regions must have positive area")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PvaMotionConfig":
        data = dict(value or {})
        backend = str(data.pop("backend", "PVA")).upper()
        if backend != "PVA":
            raise ValueError("Phase 2 motion.backend must be PVA")
        aliases = {
            "feature_scale": "feature_image_scale",
            "winsize": "window_size",
            "maxiter": "max_iterations",
            "min_features": "minimum_accepted_features",
            "min_grid_coverage": "minimum_grid_coverage",
        }
        for old, new in aliases.items():
            if old in data:
                if new in data:
                    raise ValueError(f"motion supplies both {old} and {new}")
                data[new] = data.pop(old)
        regions = data.get("exclusion_regions_xyxy")
        if regions is not None:
            if not isinstance(regions, (list, tuple)):
                raise ValueError("motion.exclusion_regions_xyxy must be a list")
            try:
                data["exclusion_regions_xyxy"] = tuple(
                    tuple(float(item) for item in region) for region in regions
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "motion.exclusion_regions_xyxy entries must be x1,y1,x2,y2"
                ) from exc
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown motion configuration keys: {unknown}")
        return cls(**data)


def _motion_u8(frame: Frame) -> np.ndarray:
    """Produce the explicitly lossy 8-bit feature image used only for motion."""

    image = frame.image
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if image.dtype.kind != "u" or image.dtype.itemsize > 2:
        raise PvaMotionError(
            f"PVA motion supports uint8/uint16 frames, got {image.dtype}"
        )
    shift = max(frame.bit_depth - 8, 0)
    if shift:
        converted = np.empty(image.shape, dtype=np.uint8)
        np.right_shift(image, shift, out=converted, casting="unsafe")
        return converted
    return np.ascontiguousarray(np.clip(image, 0, 255), dtype=np.uint8)


@lru_cache(maxsize=8)
def _raw_asinh_lut(bit_depth: int) -> np.ndarray:
    """Fixed monotonic feature-only mapping, with knee at 1/64 of code range.

    There are no image-derived percentiles, local tiles, temporal state or
    clip-specific parameters. The linear toe avoids an unbounded derivative
    near black; the logarithmic shoulder reserves codes for dim texture.
    """
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or not 9 <= bit_depth <= 16:
        raise ValueError("RAW feature LUT requires an integer bit depth in 9..16")
    limit = (1 << bit_depth) - 1
    code = np.arange(limit + 1, dtype=np.float64)
    mapped = 255.0 * np.arcsinh(64.0 * code / limit) / np.arcsinh(64.0)
    lut = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
    lut.setflags(write=False)
    return lut


def _feature_u8(frame: Frame, mapping: str) -> np.ndarray:
    if mapping not in {"bit_shift", "raw_asinh_v1"}:
        raise ValueError(f"Unknown feature intensity mapping: {mapping}")
    if mapping == "bit_shift" or frame.bit_depth <= 8:
        return _motion_u8(frame)
    image = frame.image
    if image.dtype.kind != "u" or image.dtype.itemsize != 2 or frame.bit_depth > 16:
        raise PvaMotionError("raw_asinh_v1 requires unsigned 9..16-bit source samples")
    if frame.bit_depth < 16 and np.any(image > (1 << frame.bit_depth) - 1):
        raise PvaMotionError("RAW source samples exceed the declared bit depth")
    return np.ascontiguousarray(_raw_asinh_lut(frame.bit_depth)[image])


def _raw_affine_parameters(frame: Frame) -> tuple[float, float]:
    """Fixed-grid robust global photometric normalization, motion images only.

    Use valid, unsaturated 5th/95th percentiles, sampled every eight pixels.
    This policy is identical for every clip and frame. It handles affine gain
    and offset changes; it cannot identify scene texture versus sensor pattern.
    A flat/degenerate sample becomes a flat image, never amplified noise.
    """
    samples = frame.image[::8, ::8]
    eligible = samples < .995 * ((1 << frame.bit_depth) - 1)
    if frame.valid_mask is not None:
        eligible &= frame.valid_mask[::8, ::8]
    samples = samples[eligible]
    if not samples.size:
        return 0.0, 0.0
    lo, hi = np.percentile(samples, (5, 95))
    if hi <= lo:
        return 0.0, 0.0
    scale = 40960.0 / (hi - lo)
    return float(scale), float(8192.0 - lo * scale)


def _feature_pixels(frame: Frame, mapping: str) -> np.ndarray:
    """Prepare a motion-only image; never overwrite the original RAW samples.

    Linear-U16 preserves source codes; robust-U16 applies a documented global
    affine mapping and can clip/round working-image values. Eight-bit inputs
    retain the historical mapping when a RAW-only option is requested.
    """
    if mapping not in {"raw_linear_u16_v1", "raw_robust_u16_v1"}:
        return _feature_u8(frame, mapping)
    if frame.bit_depth <= 8:
        return _motion_u8(frame)
    image = frame.image
    if image.dtype.kind != "u" or image.dtype.itemsize != 2 or not 9 <= frame.bit_depth <= 16:
        raise PvaMotionError("RAW U16 motion requires unsigned 9..16-bit source samples")
    shift = 16 - frame.bit_depth
    if shift and np.any(image > (1 << frame.bit_depth) - 1):
        raise PvaMotionError("RAW source samples exceed the declared bit depth")
    if mapping == "raw_robust_u16_v1":
        scale, offset = _raw_affine_parameters(frame)
        work = image.astype(np.float32)
        np.multiply(work, scale, out=work)
        np.add(work, offset, out=work)
        np.clip(work, 0, 65535, out=work)
        np.rint(work, out=work)
        return work.astype(np.uint16)
    if shift:
        return np.ascontiguousarray(np.left_shift(image, shift), dtype=np.uint16)
    return np.ascontiguousarray(image, dtype=np.uint16)


def _feature_eligibility(
    points_motion: np.ndarray,
    frame: Frame,
    motion_size: tuple[int, int],
    config: PvaMotionConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """Evaluate source masks and exclusion policies at Harris locations."""

    full_height, full_width = frame.shape
    points_full = lift_points_to_full_resolution(
        points_motion, motion_size, (full_width, full_height)
    )
    finite = np.isfinite(points_full).all(axis=1)
    border = config.feature_border_px
    inside_border = (
        (points_full[:, 0] >= border)
        & (points_full[:, 0] < full_width - border)
        & (points_full[:, 1] >= border)
        & (points_full[:, 1] < full_height - border)
    )
    outside_regions = np.ones(len(points_full), dtype=bool)
    for x1, y1, x2, y2 in config.exclusion_regions_xyxy:
        outside_regions &= ~(
            (points_full[:, 0] >= x1)
            & (points_full[:, 0] < x2)
            & (points_full[:, 1] >= y1)
            & (points_full[:, 1] < y2)
        )

    rounded = np.zeros(points_full.shape, dtype=np.int64)
    rounded[finite] = np.rint(points_full[finite]).astype(np.int64, copy=False)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, full_width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, full_height - 1)
    valid_mask_ok = np.ones(len(points_full), dtype=bool)
    if frame.valid_mask is not None:
        valid_mask_ok = frame.valid_mask[rounded[:, 1], rounded[:, 0]]

    saturation_ok = np.ones(len(points_full), dtype=bool)
    saturation_level = ((1 << frame.bit_depth) - 1) * config.saturation_fraction
    radius = config.saturated_neighborhood_radius_px
    candidate_indices = np.flatnonzero(finite & inside_border & outside_regions)
    if config.feature_cpu_policy == "batched_exact_v1":
        saturation_ok[candidate_indices] = _unsaturated_neighborhoods(
            frame.image, rounded[candidate_indices], radius, saturation_level,
            execution=config.feature_cpu_policy,
        )
    else:
        for index in candidate_indices:
            x = int(rounded[index, 0])
            y = int(rounded[index, 1])
            patch = frame.image[
                max(0, y - radius):min(full_height, y + radius + 1),
                max(0, x - radius):min(full_width, x + radius + 1),
            ]
            if np.any(patch >= saturation_level):
                saturation_ok[index] = False

    eligible = finite & inside_border & outside_regions & valid_mask_ok & saturation_ok
    reasons = {
        "nonfinite": int(np.count_nonzero(~finite)),
        "unreliable_border": int(np.count_nonzero(finite & ~inside_border)),
        "exclusion_region": int(
            np.count_nonzero(finite & inside_border & ~outside_regions)
        ),
        "invalid_source_mask": int(
            np.count_nonzero(finite & inside_border & outside_regions & ~valid_mask_ok)
        ),
        "saturated_neighborhood": int(
            np.count_nonzero(
                finite
                & inside_border
                & outside_regions
                & valid_mask_ok
                & ~saturation_ok
            )
        ),
    }
    return eligible, reasons


def _unsaturated_neighborhoods(image, centers, radius, level, *,
                              execution="reference", batch_points=4096):
    """Exact cropped-neighborhood OR, with bounded temporary storage.

    Clamping repeats edge pixels instead of padding: repetition cannot change
    an OR/max predicate. Large or unusual radii retain the scalar implementation
    to avoid allocating a points-by-large-kernel tensor.
    """
    height, width = image.shape
    result = np.ones(len(centers), dtype=bool)
    if (execution == "batched_exact_v1" and type(radius) is int and 0 <= radius <= 8):
        if type(batch_points) is not int or batch_points <= 0:
            raise ValueError("batch_points must be a positive integer")
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()
        for start in range(0, len(centers), batch_points):
            chunk = centers[start:start + batch_points]
            yy = np.clip(chunk[:, 1, None] + dy, 0, height - 1)
            xx = np.clip(chunk[:, 0, None] + dx, 0, width - 1)
            result[start:start + len(chunk)] = ~np.any(image[yy, xx] >= level, axis=1)
        return result
    for index, (x, y) in enumerate(centers):
        x, y = int(x), int(y)
        patch = image[max(0, y - radius):min(height, y + radius + 1),
                      max(0, x - radius):min(width, x + radius + 1)]
        if np.any(patch >= level):
            result[index] = False
    return result


def _elapsed_ms(started: int) -> float:
    return (time.perf_counter_ns() - started) / 1_000_000


def _harris_output_capacity(motion_size: tuple[int, int]) -> int:
    """Cover every PVA 8x8 NMS cell, plus one boundary cell per dimension.

    The Python API's default 8192-element array truncates dense U16 corner
    results in raster order, before our spatial quotas can see the lower image.
    This is an output-storage bound, not a tracked-point budget or a threshold.
    """
    width, height = motion_size
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in motion_size):
        raise ValueError("Motion image dimensions must be positive integers")
    return ((width + 7) // 8 + 1) * ((height + 7) // 8 + 1)


def _fresh_flow_status(vpi: Any, count: int, initial: np.ndarray | None = None) -> Any:
    """Explicitly initialize an in/out status array for a newly seeded set.

    A lost flag refers to a particular tracked feature, not to its array slot.
    New Harris features must not inherit flags from earlier calls or cached
    allocations. Backward flow receives a separate copy of forward flags.
    """
    # Use VPI's zero-initialized allocation, not an uninitialized cached
    # allocation followed only by a CPU write. The latter retained CUDA-side
    # lost flags on the tested VPI 3.2 installation despite zero CPU readback.
    status = vpi.Array.zeros(count, vpi.Type.U8)
    status.size = count
    if initial is not None:
        incoming = np.asarray(initial, dtype=np.uint8).reshape(-1)
        if len(incoming) != count:
            raise PvaMotionError("Flow status count does not match the newly seeded features")
        with status.rwlock_cpu() as data:
            values = np.asarray(data).reshape(-1)
            values[:] = incoming
    return status


class PvaPyrLkMotionEstimator:
    """Estimate full-resolution background correspondences with PVA PyrLK."""

    PVA_PYRAMID_MAX_SIZE = (3264, 2048)

    def __init__(
        self, config: PvaMotionConfig | Mapping[str, Any] | None = None
    ) -> None:
        self.config = (
            config
            if isinstance(config, PvaMotionConfig)
            else PvaMotionConfig.from_mapping(config)
        )
        try:
            self._vpi = importlib.import_module("vpi")
        except ImportError as exc:
            raise PvaMotionError(
                "NVIDIA VPI Python bindings are unavailable; run this estimator "
                "on the Jetson image with VPI installed"
            ) from exc
        missing = [
            name for name in ("CUDA", "PVA") if not hasattr(self._vpi.Backend, name)
        ]
        if missing:
            raise PvaMotionError(f"Required VPI backends are unavailable: {missing}")
        try:
            self._stream = self._vpi.Stream(
                self._vpi.Backend.CUDA | self._vpi.Backend.PVA
            )
        except Exception as exc:
            raise PvaMotionError(f"Cannot create CUDA/PVA VPI stream: {exc}") from exc

    def _pyramid_backend(self, motion_size: tuple[int, int]) -> tuple[Any, str]:
        requested = self.config.pyramid_backend
        max_width, max_height = self.PVA_PYRAMID_MAX_SIZE
        if requested == "auto":
            requested = (
                "PVA"
                if motion_size[0] <= max_width and motion_size[1] <= max_height
                else "CUDA"
            )
        elif requested == "PVA" and (
            motion_size[0] > max_width or motion_size[1] > max_height
        ):
            raise PvaMotionError(
                f"PVA pyramid size {motion_size} exceeds the verified "
                f"{self.PVA_PYRAMID_MAX_SIZE} limit; use auto or CUDA"
            )
        return getattr(self._vpi.Backend, requested), requested

    def estimate(self, previous: Frame, current: Frame) -> MotionCorrespondences:
        if previous.shape != current.shape:
            raise PvaMotionError(
                f"Frame shapes differ: {previous.shape} != {current.shape}"
            )
        if current.frame_index <= previous.frame_index:
            raise PvaMotionError("Current frame index must follow previous frame index")
        if current.timestamp_ns <= previous.timestamp_ns:
            raise PvaMotionError("Motion pair timestamps must be strictly increasing")
        if previous.bit_depth != current.bit_depth:
            raise PvaMotionError("Motion pair bit depths differ")

        config = self.config
        vpi = self._vpi
        full_height, full_width = previous.shape
        full_size = (full_width, full_height)
        motion_size = (
            max(1, round(full_width * config.feature_image_scale)),
            max(1, round(full_height * config.feature_image_scale)),
        )
        pyramid_backend, pyramid_backend_name = self._pyramid_backend(motion_size)
        timings: dict[str, float] = {}
        total_started = time.perf_counter_ns()

        if config.flow_status_policy == "fresh_per_pair":
            # VPI 3.2 reused lost-point state even with zeroed status arrays,
            # explicit stream ordering and freshly wrapped host buffers. The
            # generated clean/noise/clean test reproduces that. Evict unused
            # process-local VPI objects at the pair boundary for this opt-in,
            # single-worker correctness path. This does not reset the device.
            started = time.perf_counter_ns()
            try:
                self._stream.sync()
                vpi.clear_cache()
            except Exception as exc:
                raise PvaMotionError(f"Cannot isolate fresh per-pair flow state: {exc}") from exc
            timings["flow_state_cache_reset"] = _elapsed_ms(started)

        started = time.perf_counter_ns()
        previous_pixels = _feature_pixels(previous, config.feature_intensity_mapping)
        current_pixels = _feature_pixels(current, config.feature_intensity_mapping)
        uses_u16 = previous_pixels.dtype == np.uint16
        timings["intensity_conversion_cpu"] = _elapsed_ms(started)

        try:
            started = time.perf_counter_ns()
            image_format = vpi.Format.U16 if uses_u16 else vpi.Format.U8
            previous_image = vpi.asimage(previous_pixels, image_format)
            current_image = vpi.asimage(current_pixels, image_format)
            if motion_size != full_size:
                previous_motion = previous_image.rescale(
                    motion_size, backend=vpi.Backend.CUDA, stream=self._stream
                )
                current_motion = current_image.rescale(
                    motion_size, backend=vpi.Backend.CUDA, stream=self._stream
                )
                submitted = time.perf_counter_ns()
                self._stream.sync()
                rescale_backend = "CUDA"
            else:
                previous_motion = previous_image
                current_motion = current_image
                submitted = time.perf_counter_ns()
                rescale_backend = "none"
            completed = time.perf_counter_ns()
            timings["motion_image_prepare_submit"] = (submitted - started) / 1_000_000
            timings["motion_image_prepare_sync"] = (completed - submitted) / 1_000_000
            timings["motion_image_prepare"] = (completed - started) / 1_000_000

            started = time.perf_counter_ns()
            previous_pyramid = previous_motion.gaussian_pyramid(
                config.pyramid_levels,
                config.pyramid_scale,
                backend=pyramid_backend,
                stream=self._stream,
            )
            current_pyramid = current_motion.gaussian_pyramid(
                config.pyramid_levels,
                config.pyramid_scale,
                backend=pyramid_backend,
                stream=self._stream,
            )
            submitted = time.perf_counter_ns()
            self._stream.sync()
            completed = time.perf_counter_ns()
            timings["gaussian_pyramids_submit"] = (submitted - started) / 1_000_000
            timings["gaussian_pyramids_sync"] = (completed - submitted) / 1_000_000
            timings["gaussian_pyramids"] = (completed - started) / 1_000_000

            started = time.perf_counter_ns()
            # A signed offset preserves the U16 range without clipping or
            # discarding bits. Harris depends on gradients, not the DC level.
            conversion = {"offset": -32768.0} if uses_u16 else {}
            previous_s16 = previous_motion.convert(
                vpi.Format.S16, backend=vpi.Backend.CUDA, stream=self._stream,
                **conversion,
            )
            submitted = time.perf_counter_ns()
            self._stream.sync()
            completed = time.perf_counter_ns()
            timings["harris_input_conversion_cuda_submit"] = (
                submitted - started
            ) / 1_000_000
            timings["harris_input_conversion_cuda_sync"] = (
                completed - submitted
            ) / 1_000_000
            timings["harris_input_conversion_cuda"] = (completed - started) / 1_000_000

            started = time.perf_counter_ns()
            harris_outputs = {}
            harris_capacity = None
            if config.harris_capacity_policy == "complete_grid":
                harris_capacity = _harris_output_capacity(motion_size)
                harris_outputs = {
                    "out_features": vpi.Array(harris_capacity, vpi.Type.KEYPOINT_F32),
                    "out_scores": vpi.Array(harris_capacity, vpi.Type.U32),
                }
            features, scores = previous_s16.harriscorners(
                backend=vpi.Backend.PVA,
                gradient_size=config.harris_gradient_size,
                block_size=config.harris_block_size,
                strength=config.harris_strength,
                sensitivity=config.harris_sensitivity,
                min_nms_distance=config.harris_min_nms_distance,
                stream=self._stream,
                **harris_outputs,
            )
            submitted = time.perf_counter_ns()
            self._stream.sync()
            completed = time.perf_counter_ns()
            timings["harris_pva_submit"] = (submitted - started) / 1_000_000
            timings["harris_pva_sync"] = (completed - submitted) / 1_000_000
            timings["harris_pva"] = (completed - started) / 1_000_000
            detected_count = int(features.size)
            if harris_capacity is not None and detected_count >= harris_capacity:
                raise PvaMotionError("Harris output capacity exhausted; full-image feature coverage is unknown")
            if detected_count == 0:
                raise PvaMotionError(
                    "PVA Harris returned zero features; motion is unavailable"
                )

            started = time.perf_counter_ns()
            with features.rlock_cpu() as feature_data:
                detected_points = np.array(feature_data, dtype=np.float32, copy=True)
            with scores.rlock_cpu() as score_data:
                detected_scores = np.array(
                    score_data, dtype=np.float32, copy=True
                ).reshape(-1)
            timings["harris_readback_cpu"] = _elapsed_ms(started)
            eligibility_started = time.perf_counter_ns()
            eligible_mask, feature_exclusions = _feature_eligibility(
                detected_points, previous, motion_size, config
            )
            timings["feature_eligibility_cpu"] = _elapsed_ms(eligibility_started)
            selection_started = time.perf_counter_ns()
            selected_indices = select_spatially_distributed(
                detected_points,
                detected_scores,
                motion_size,
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                max_features=config.max_features,
                max_per_cell=config.max_features_per_cell,
                eligible_mask=eligible_mask,
                execution=config.feature_cpu_policy,
            )
            timings["spatial_quota_cpu"] = _elapsed_ms(selection_started)
            if len(selected_indices) == 0:
                raise PvaMotionError(
                    "No finite in-bounds Harris features survived selection"
                )
            selected_points = detected_points[selected_indices]
            selected_scores = detected_scores[selected_indices]
            upload_started = time.perf_counter_ns()
            with features.rwlock_cpu() as feature_data:
                feature_data[: len(selected_indices)] = selected_points
            features.size = len(selected_indices)
            with scores.rwlock_cpu() as score_data:
                score_data[: len(selected_indices)] = selected_scores
            scores.size = len(selected_indices)
            timings["selected_feature_upload_cpu"] = _elapsed_ms(upload_started)
            timings["feature_readback_and_grid_selection"] = _elapsed_ms(started)

            started = time.perf_counter_ns()
            forward_options = {}
            if config.flow_status_policy == "fresh_per_pair":
                forward_options["kptstatus"] = _fresh_flow_status(vpi, len(selected_points))
            # Constructors have no stream keyword; use the same current
            # stream for their state initialization as for the flow submit.
            # Otherwise initialization may run on VPI's asynchronous default
            # stream while the computation runs on our private stream.
            with self._stream if config.flow_status_policy == "fresh_per_pair" else nullcontext():
                forward_flow = vpi.OpticalFlowPyrLK(
                    previous_pyramid, features, backend=getattr(vpi.Backend, config.optical_flow_backend),
                    **forward_options,
                )
            tracked_points, forward_status = forward_flow(
                current_pyramid,
                winsize=config.window_size,
                maxiter=config.max_iterations,
                stream=self._stream,
            )
            submitted = time.perf_counter_ns()
            self._stream.sync()
            completed = time.perf_counter_ns()
            flow_key = "forward_pyrlk_" + config.optical_flow_backend.lower()
            timings[flow_key + "_submit"] = (submitted - started) / 1_000_000
            timings[flow_key + "_sync"] = (completed - submitted) / 1_000_000
            timings[flow_key] = (completed - started) / 1_000_000

            backward_points_vpi = None
            backward_status_vpi = None
            if config.forward_backward_check:
                started = time.perf_counter_ns()
                backward_initial_status = forward_status
                if config.flow_status_policy == "fresh_per_pair":
                    with forward_status.rlock_cpu() as data:
                        flags = np.array(data, dtype=np.uint8, copy=True).reshape(-1)
                    backward_initial_status = _fresh_flow_status(vpi, len(selected_points), flags)
                with self._stream if config.flow_status_policy == "fresh_per_pair" else nullcontext():
                    backward_flow = vpi.OpticalFlowPyrLK(
                        current_pyramid,
                        tracked_points,
                        kptstatus=backward_initial_status,
                        backend=getattr(vpi.Backend, config.optical_flow_backend),
                    )
                backward_points_vpi, backward_status_vpi = backward_flow(
                    previous_pyramid,
                    winsize=config.window_size,
                    maxiter=config.max_iterations,
                    stream=self._stream,
                )
                submitted = time.perf_counter_ns()
                self._stream.sync()
                completed = time.perf_counter_ns()
                flow_key = "backward_pyrlk_" + config.optical_flow_backend.lower()
                timings[flow_key + "_submit"] = (submitted - started) / 1_000_000
                timings[flow_key + "_sync"] = (completed - submitted) / 1_000_000
                timings[flow_key] = (completed - started) / 1_000_000

            started = time.perf_counter_ns()
            with tracked_points.rlock_cpu() as data:
                current_motion_points = np.array(data, dtype=np.float32, copy=True)
            with forward_status.rlock_cpu() as data:
                forward_status_array = np.array(
                    data, dtype=np.uint8, copy=True
                ).reshape(-1)
            backward_motion_points: np.ndarray | None = None
            backward_status_array: np.ndarray | None = None
            if backward_points_vpi is not None and backward_status_vpi is not None:
                with backward_points_vpi.rlock_cpu() as data:
                    backward_motion_points = np.array(data, dtype=np.float32, copy=True)
                with backward_status_vpi.rlock_cpu() as data:
                    backward_status_array = np.array(
                        data, dtype=np.uint8, copy=True
                    ).reshape(-1)
            timings["flow_readback_cpu"] = _elapsed_ms(started)
        except PvaMotionError:
            raise
        except Exception as exc:
            raise PvaMotionError(f"VPI PVA motion estimation failed: {exc}") from exc

        selected_count = len(selected_points)
        if (
            len(current_motion_points) != selected_count
            or len(forward_status_array) != selected_count
        ):
            raise PvaMotionError(
                "PyrLK result count does not match selected feature count: "
                f"selected={selected_count}, points={len(current_motion_points)}, "
                f"status={len(forward_status_array)}"
            )

        started = time.perf_counter_ns()
        previous_full_points = lift_points_to_full_resolution(
            selected_points, motion_size, full_size
        )
        current_full_points = lift_points_to_full_resolution(
            current_motion_points, motion_size, full_size
        )
        backward_full_points = (
            lift_points_to_full_resolution(
                backward_motion_points, motion_size, full_size
            )
            if backward_motion_points is not None
            else None
        )
        accepted_mask, rejection_counts, fb_error = correspondence_acceptance_mask(
            previous_full_points,
            current_full_points,
            forward_status_array,
            image_size=full_size,
            max_displacement_px=config.max_displacement_px,
            backward_points=backward_full_points,
            backward_status=backward_status_array,
            max_forward_backward_error_px=(
                config.max_forward_backward_error_px
                if config.forward_backward_check
                else None
            ),
        )
        accepted_previous = previous_full_points[accepted_mask]
        accepted_current = current_full_points[accepted_mask]
        accepted_scores = selected_scores[accepted_mask]
        accepted_fb_error = fb_error[accepted_mask]
        coverage = grid_coverage(
            accepted_previous,
            full_size,
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            execution=config.feature_cpu_policy,
        )
        displacement = np.linalg.norm(accepted_current - accepted_previous, axis=1)
        usable = (
            len(accepted_previous) >= config.minimum_accepted_features
            and coverage["fraction"] >= config.minimum_grid_coverage
        )
        timings["coordinate_lift_and_filter_cpu"] = _elapsed_ms(started)
        timings["total"] = _elapsed_ms(total_started)

        metrics_started = time.perf_counter_ns()
        metrics: dict[str, Any] = {
            "detected_count": detected_count,
            "selected_count": selected_count,
            "accepted_count": len(accepted_previous),
            "rejected_count": selected_count - len(accepted_previous),
            "rejections": rejection_counts,
            "feature_exclusions_before_tracking": feature_exclusions,
            "track_survival_ratio": len(accepted_previous) / selected_count,
            "grid_coverage": coverage,
            "median_displacement_px": (
                float(np.median(displacement)) if len(displacement) else None
            ),
            "p95_displacement_px": (
                float(np.percentile(displacement, 95)) if len(displacement) else None
            ),
            "median_forward_backward_error_px": (
                float(np.median(accepted_fb_error))
                if config.forward_backward_check and len(accepted_fb_error)
                else None
            ),
            "usable_for_transform": bool(usable),
            "quality_rejection_reasons": (
                (
                    ["insufficient_accepted_features"]
                    if len(accepted_previous) < config.minimum_accepted_features
                    else []
                )
                + (
                    ["low_grid_coverage"]
                    if coverage["fraction"] < config.minimum_grid_coverage
                    else []
                )
            ),
            "minimum_accepted_features": config.minimum_accepted_features,
            "minimum_grid_coverage": config.minimum_grid_coverage,
            "memory_bytes": {
                "source_frames_read": previous.image.nbytes + current.image.nbytes,
                "motion_u16_frames_created" if uses_u16 else "motion_u8_frames_created":
                    previous_pixels.nbytes + current_pixels.nbytes,
                "feature_readback": detected_points.nbytes + detected_scores.nbytes,
                "flow_readback": (
                    current_motion_points.nbytes
                    + forward_status_array.nbytes
                    + (
                        backward_motion_points.nbytes
                        if backward_motion_points is not None
                        else 0
                    )
                    + (
                        backward_status_array.nbytes
                        if backward_status_array is not None
                        else 0
                    )
                ),
            },
        }
        if config.feature_intensity_mapping != "bit_shift":
            metrics["feature_intensity_mapping"] = {
                "requested": config.feature_intensity_mapping,
                "effective": config.feature_intensity_mapping if previous.bit_depth > 8 else "8bit_passthrough",
                "source_pixels_modified": False,
                "working_image_format": "U16" if uses_u16 else "U8",
            }
            if config.feature_intensity_mapping == "raw_asinh_v1":
                metrics["feature_intensity_mapping"]["knee_fraction_of_declared_code_range"] = 1.0 / 64.0
            elif uses_u16 and config.feature_intensity_mapping == "raw_linear_u16_v1":
                metrics["feature_intensity_mapping"].update(
                    source_code_left_shift=16 - previous.bit_depth,
                    harris_signed_offset=-32768,
                )
            elif uses_u16:
                metrics["feature_intensity_mapping"].update(
                    normalization="per_frame_valid_unsaturated_percentiles_5_95_stride8",
                    normalized_percentile_codes=[8192, 49152],
                    harris_signed_offset=-32768,
                )
        if harris_capacity is not None:
            metrics["harris_output"] = {
                "capacity_policy": config.harris_capacity_policy,
                "capacity": harris_capacity,
                "capacity_exhausted": False,
                "detected_grid_coverage": grid_coverage(
                    detected_points, motion_size,
                    grid_rows=config.grid_rows, grid_cols=config.grid_cols,
                    execution=config.feature_cpu_policy,
                ),
            }
        if config.flow_status_policy != "legacy_default":
            metrics["flow_status"] = {
                "policy": config.flow_status_policy,
                "forward_initial_flags": "all_zero_for_new_harris_features",
                "backward_initial_flags": "separate_copy_of_forward_result",
                "post_tracking_failure_flags_preserved": True,
                "cached_state_isolation": "clear_unused_process_local_vpi_cache_at_pair_boundary",
            }
        timings["metrics_and_diagnostics_cpu"] = _elapsed_ms(metrics_started)
        timings["total_including_metrics"] = _elapsed_ms(total_started)
        return MotionCorrespondences(
            previous_points=accepted_previous,
            current_points=accepted_current,
            harris_scores=accepted_scores,
            forward_backward_error_px=accepted_fb_error,
            previous_frame_index=previous.frame_index,
            current_frame_index=current.frame_index,
            previous_timestamp_ns=previous.timestamp_ns,
            current_timestamp_ns=current.timestamp_ns,
            full_image_size=full_size,
            motion_image_size=motion_size,
            metrics=metrics,
            timings_ms=timings,
            backends={
                "intensity_conversion": "CPU",
                "motion_image_rescale": rescale_backend,
                "gaussian_pyramid": pyramid_backend_name,
                "harris_input_conversion": "CUDA",
                "harris": "PVA",
                "optical_flow_pyrlk": config.optical_flow_backend,
                "cpu_fallback": False,
            },
        )
