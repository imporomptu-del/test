"""Full-frame 8-bit visible-point development baseline; no truth inputs.

This complements, and does not replace, independent faint-target discovery.
All proposals and measured/predicted track states are journaled separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .detection import CandidateBatch, CandidateRecord
from .frame_source import probe_video
from .tracking import KalmanTrackManager, KalmanTrackingConfig
from .visible_quality import CausalMotionQuality, suppress_nearby
from .visible_shapes import consolidate_half_height
from .visible_learning import shape_learning_mask
from .visible_coverage import DetectionAvailability, finite_json
from .visible_decode import VisibleFrameReader, decode_contract


@dataclass(frozen=True)
class VisibleConfig:
    schema_version: int = 1
    input_bit_depth: int = 8
    tile_size: int = 256
    noise_sample_stride: int = 4
    noise_floor_dn: float = 0.5
    temporal_threshold_sigma: float = 4.0
    spatial_threshold_sigma: float = 3.0
    spatial_background: str = "box13"
    pixel_noise_enabled: bool = False
    pixel_noise_alpha: float = 0.1
    pixel_noise_clip_sigma: float = 4.0
    pixel_noise_model: str = "frame_difference"
    learning_exclusion_radius_px: int = 0
    learning_protection_mode: str = "background_and_variance"
    learning_protection_geometry: str = "circle"
    background_alpha: float = 0.05
    warmup_frames: int = 8
    max_candidates_per_tile_polarity: int = 12
    max_candidates_per_frame: int = 512
    max_active_tracks_per_polarity: int = 256
    position_sigma_px: float = 2.0
    initial_velocity_sigma_px_s: float = 150.0
    acceleration_sigma_px_s2: float = 60.0
    position_gate_px: float = 45.0
    mahalanobis_gate_squared: float = 25.0
    confirmation_hits: int = 4
    coast_seconds: float = 0.7
    minimum_moving_excursion_px: float = 12.0
    tracking_peak_nms_radius_px: float = 0.0
    shape_measurement_mode: str = "none"
    motion_quality_enabled: bool = False
    motion_quality_window_hits: int = 8
    motion_quality_minimum_hits: int = 5
    motion_quality_maximum_rmse_px: float = 3.0
    tracking_association_cost: str = "mahalanobis"
    tracking_association_cascade: str = "none"
    tracking_association_assignment: str = "greedy"
    tracking_association_prior: str = "none"
    tracking_association_appearance: str = "none"
    state_update_backend: str = "indexed"
    spatial_filter_backend: str = "cpu"
    cuda_median_library: str | None = None
    native_shape_library: str | None = None
    native_shape_library_sha256: str | None = None
    stabilization_execution: str = "reference"
    tracking_birth_policy: str = "input_order"
    tracking_birth_cell_size_px: float = 256.0
    motion_backend: str = "cpu_translation"
    motion_proxy_max_dimension: int = 1024
    motion_min_response: float = 0.2
    motion_max_pair_shift_px: float = 80.0
    opencv_threads: int = 2
    frame_decode_execution: str = "sequential"

    def __post_init__(self):
        decode_contract(self.frame_decode_execution)
        if self.native_shape_library is not None:
            digest = self.native_shape_library_sha256
            if (not isinstance(self.native_shape_library, str) or not self.native_shape_library
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or self.state_update_backend != 'cuda_resident'
                    or self.shape_measurement_mode != 'mutual_half_height_r8'):
                raise ValueError('Native shapes require an explicit hashed library and resident r8 measurements')
        elif self.native_shape_library_sha256 is not None:
            raise ValueError('Native shape hash requires its library')
        if self.stabilization_execution not in {"reference", "cuda_cubic_host", "cuda_cubic_resident"}:
            raise ValueError("Unknown stabilization_execution")
        if self.stabilization_execution != "reference" and (
            self.motion_backend != "pva" or self.state_update_backend != "cuda_resident"
            or self.spatial_filter_backend != "cuda_median5"
        ):
            raise ValueError("Exact CUDA stabilization requires explicit PVA and resident CUDA execution")
        if self.spatial_filter_backend not in {"cpu", "cuda_median5"}:
            raise ValueError("Unknown spatial_filter_backend")
        if self.spatial_filter_backend == "cuda_median5":
            if self.spatial_background != "median5" or not isinstance(self.cuda_median_library,str) or not self.cuda_median_library:
                raise ValueError("CUDA median requires median5 and an explicit compiled library")
        elif self.cuda_median_library is not None:
            raise ValueError("CPU filter cannot specify a CUDA library")
        if self.state_update_backend not in {"indexed", "inplace", "cuda_resident"}:
            raise ValueError("Unknown state_update_backend")
        if self.state_update_backend == "cuda_resident" and (
            self.spatial_filter_backend != "cuda_median5" or not self.pixel_noise_enabled
            or self.pixel_noise_model != "background_residual" or self.tile_size > 256
            or self.max_candidates_per_tile_polarity > 16 or self.max_candidates_per_frame > 512
        ):
            raise ValueError("Resident CUDA requires median5, residual pixel noise, tiles<=256, tile quota<=16 and frame cap<=512")
        if self.tracking_association_appearance not in {"none", "log_response", "log_response_coast"}:
            raise ValueError("Unknown tracking_association_appearance")
        if self.tracking_association_appearance != "none" and self.tracking_association_cost != "gaussian_nll":
            raise ValueError("Appearance experiment requires Gaussian association")
        if self.learning_protection_geometry not in {"circle", "observed_shape"}:
            raise ValueError("Unknown learning_protection_geometry")
        if self.learning_protection_geometry == "observed_shape" and (
            self.learning_protection_mode != "variance_only"
            or self.shape_measurement_mode != "mutual_half_height_r8"
            or self.learning_exclusion_radius_px != 0
            or not self.pixel_noise_enabled
            or not 0 < self.position_sigma_px <= 16
        ):
            raise ValueError("Observed-shape protection requires shape measurements, variance-only learning, zero circle radius and bounded position sigma")
        if self.tracking_association_prior not in {"none", "hit_maturity"}:
            raise ValueError("Unknown tracking_association_prior")
        if self.tracking_association_prior != "none" and (
            self.tracking_association_cost != "gaussian_nll"
            or self.tracking_association_cascade != "none"
        ):
            raise ValueError("Hit maturity requires Gaussian cost without a cascade")
        if self.shape_measurement_mode not in {"none", "mutual_half_height_r8"}:
            raise ValueError("Unknown shape_measurement_mode")
        if self.tracking_association_assignment not in {"greedy", "global_min_cost"}:
            raise ValueError("Unknown tracking_association_assignment")
        if (
            self.tracking_association_assignment != "greedy"
            and self.tracking_association_cascade != "none"
        ):
            raise ValueError(
                "Global assignment cannot be combined with a priority cascade"
            )
        if self.learning_protection_mode not in {
            "background_and_variance",
            "variance_only",
        }:
            raise ValueError("Unknown learning_protection_mode")
        if (
            isinstance(self.learning_exclusion_radius_px, bool)
            or not isinstance(self.learning_exclusion_radius_px, int)
            or not 0 <= self.learning_exclusion_radius_px <= 16
        ):
            raise ValueError(
                "learning_exclusion_radius_px must be an integer in [0,16]"
            )
        if self.tracking_association_cascade not in {"none", "confirmed_first"}:
            raise ValueError("Unknown tracking_association_cascade")
        if self.tracking_association_cost not in {"mahalanobis", "gaussian_nll"}:
            raise ValueError("Unknown tracking_association_cost")
        if self.tracking_birth_policy not in {"input_order", "spatial_fair"}:
            raise ValueError("Unknown tracking_birth_policy")
        if (
            not math.isfinite(self.tracking_birth_cell_size_px)
            or self.tracking_birth_cell_size_px <= 0
        ):
            raise ValueError("tracking_birth_cell_size_px must be finite and positive")
        if not isinstance(self.motion_quality_enabled, bool):
            raise ValueError("motion_quality_enabled must be boolean")
        if (
            not math.isfinite(self.tracking_peak_nms_radius_px)
            or self.tracking_peak_nms_radius_px < 0
        ):
            raise ValueError(
                "tracking_peak_nms_radius_px must be finite and nonnegative"
            )
        for name in ("motion_quality_window_hits", "motion_quality_minimum_hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        CausalMotionQuality(
            self.motion_quality_window_hits,
            self.motion_quality_minimum_hits,
            self.motion_quality_maximum_rmse_px,
        )
        if not isinstance(self.pixel_noise_enabled, bool):
            raise ValueError("pixel_noise_enabled must be boolean")
        if self.pixel_noise_model not in {"frame_difference", "background_residual"}:
            raise ValueError("invalid pixel_noise_model")
        if (
            not 0 < self.pixel_noise_alpha <= 1
            or not math.isfinite(self.pixel_noise_clip_sigma)
            or self.pixel_noise_clip_sigma <= 0
        ):
            raise ValueError("invalid pixel-noise learning controls")
        if self.schema_version != 1 or self.input_bit_depth != 8:
            raise ValueError(
                "This version requires schema 1 and native 8-bit input; RAW16 is not calibrated"
            )
        for name in (
            "tile_size",
            "noise_sample_stride",
            "warmup_frames",
            "max_candidates_per_tile_polarity",
            "max_candidates_per_frame",
            "max_active_tracks_per_polarity",
            "confirmation_hits",
            "motion_proxy_max_dimension",
            "opencv_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.confirmation_hits < 2:
            raise ValueError("confirmation_hits must be >=2")
        for name in (
            "noise_floor_dn",
            "temporal_threshold_sigma",
            "spatial_threshold_sigma",
            "background_alpha",
            "position_sigma_px",
            "initial_velocity_sigma_px_s",
            "acceleration_sigma_px_s2",
            "position_gate_px",
            "mahalanobis_gate_squared",
            "coast_seconds",
            "minimum_moving_excursion_px",
            "motion_min_response",
            "motion_max_pair_shift_px",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.background_alpha > 1:
            raise ValueError("background_alpha must be <=1")
        if self.motion_backend not in {"cpu_translation", "pva"}:
            raise ValueError(
                "motion_backend must be cpu_translation or pva; no silent identity fallback"
            )
        if self.spatial_background not in {"median5", "box13"}:
            raise ValueError("spatial_background must be median5 or box13")

    @property
    def learning_protection_enabled(self):
        return bool(self.learning_exclusion_radius_px) or self.learning_protection_geometry == "observed_shape"

    def tracker(self, fps):
        return KalmanTrackingConfig(
            measurement_model="position_only",
            association_cost=self.tracking_association_cost,
            association_cascade=self.tracking_association_cascade,
            association_assignment=self.tracking_association_assignment,
            association_prior=self.tracking_association_prior,
            association_appearance=self.tracking_association_appearance,
            birth_policy=self.tracking_birth_policy,
            birth_cell_size_px=self.tracking_birth_cell_size_px,
            position_measurement_sigma_px=self.position_sigma_px,
            velocity_measurement_sigma_px_s=1.0,
            acceleration_process_sigma_px_s2=self.acceleration_sigma_px_s2,
            initial_position_sigma_px=self.position_sigma_px,
            initial_velocity_sigma_px_s=self.initial_velocity_sigma_px_s,
            mahalanobis_gate_squared=self.mahalanobis_gate_squared,
            maximum_position_residual_px=self.position_gate_px,
            maximum_velocity_residual_px_s=1.0,
            confirmation_independent_hits=self.confirmation_hits,
            max_missed_windows=max(1, int(math.floor(self.coast_seconds * fps))),
            maximum_timestamp_gap_s=max(1.0, 3 / fps),
            measurement_noise_source="provisional",
            max_active_tracks=self.max_active_tracks_per_polarity,
        )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def map_point(matrix, x, y):
    p = np.asarray(matrix, dtype=float) @ np.array([x, y, 1.0])
    return [float(p[0] / p[2]), float(p[1] / p[2])]


class CpuTranslation:
    """Development motion backend. Only the camera-motion proxy is downsampled."""

    def __init__(self, config):
        self.config = config
        self.previous = None
        self.matrix = np.eye(3)
        self.segment = 0

    def update(self, gray, frame_index, timestamp_ns):
        h, w = gray.shape
        scale = min(1.0, self.config.motion_proxy_max_dimension / max(h, w))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        proxy = cv2.resize(gray.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        proxy -= cv2.GaussianBlur(proxy, (0, 0), 3)
        response = None
        shift = (0.0, 0.0)
        reset = False
        if self.previous is not None:
            delta, response = cv2.phaseCorrelate(self.previous, proxy)
            shift = (delta[0] * w / size[0], delta[1] * h / size[1])
            if (
                not np.isfinite((*shift, response)).all()
                or response < self.config.motion_min_response
                or math.hypot(*shift) > self.config.motion_max_pair_shift_px
            ):
                self.segment += 1
                self.matrix = np.eye(3)
                reset = True
            else:
                # phaseCorrelate gives current displacement; inverse maps to reference.
                self.matrix[0, 2] -= shift[0]
                self.matrix[1, 2] -= shift[1]
        self.previous = proxy
        raw = gray.astype(np.float32)
        if np.array_equal(self.matrix, np.eye(3)):
            image = raw
            valid = np.ones(gray.shape, np.uint8)
        else:
            image = cv2.warpAffine(raw, self.matrix[:2], (w, h), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(
                np.ones(gray.shape, np.uint8),
                self.matrix[:2],
                (w, h),
                flags=cv2.INTER_NEAREST,
            )
        return (
            image,
            valid.astype(bool),
            self.matrix.copy(),
            self.segment,
            dict(
                backend="cpu_translation",
                response=response,
                pair_shift_xy_px=list(shift),
                reset=reset,
                proxy_size=list(size),
            ),
        )


class PvaMotion:
    """Use existing PVA motion + global fit + full-resolution warp, retaining mapping."""

    def __init__(self, path, execution="reference", library=None):
        from .config import load_config
        from .motion import PvaMotionConfig, PvaPyrLkMotionEstimator, GlobalMotionConfig
        from .stabilization import FullResolutionStabilizer, StabilizationConfig

        raw = load_config(path).raw
        self.estimator = PvaPyrLkMotionEstimator(
            PvaMotionConfig.from_mapping(raw.get("motion"))
        )
        self.global_config = GlobalMotionConfig.from_mapping(raw.get("global_motion"))
        self.stabilizer = FullResolutionStabilizer(
            StabilizationConfig.from_mapping(raw.get("stabilization"))
        )
        self.previous = None
        self.tracker = None
        self.cuda_warp = None
        self.execution = execution
        self.conformance = None
        self.source_mask = None
        if execution != "reference":
            if execution not in {"cuda_cubic_host", "cuda_cubic_resident"}:
                raise ValueError("Unsupported stabilization execution")
            cfg = self.stabilizer.config
            if self.global_config.model != "translation" or cfg.interpolation != "cubic" or cfg.border_value != 0 or cfg.backend != "opencv_cpu":
                raise ValueError("Exact CUDA replacement requires the frozen CPU cubic translation reference")
            from .visible_warp_exact import CudaCubicTranslation
            self.cuda_warp = CudaCubicTranslation(library)
            self.conformance = self.cuda_warp.verify_reference(include_gaussian=execution == "cuda_cubic_resident")

    def close(self):
        if self.cuda_warp is not None:
            self.cuda_warp.close()

    def update(self, gray, frame_index, timestamp_ns):
        from .types import Frame, TimestampSource
        from .motion import (
            GlobalMotionTracker,
            ComposedMotionState,
            fit_global_motion,
            PvaMotionError,
        )

        current = Frame(
            gray,
            timestamp_ns,
            frame_index,
            "visible-baseline",
            8,
            TimestampSource.CONTAINER_RATE,
        )
        metadata = {
            "backend": "pva",
            "reset": False,
            "pva_failure": False,
            "rejection_reasons": [],
        }
        if self.previous is None:
            self.tracker = GlobalMotionTracker(
                self.global_config, initial_frame_index=frame_index
            )
            chain = ComposedMotionState(
                reference_frame_index=frame_index,
                current_frame_index=frame_index,
                segment_index=0,
                reference_from_current_matrix=np.eye(3),
                status="initial_reference",
                window_reset=False,
                reused_pairs=0,
                pair_parameter_delta=None,
            )
        else:
            try:
                correspondence = self.estimator.estimate(self.previous, current)
                estimate = fit_global_motion(correspondence, self.global_config)
                chain = self.tracker.update(estimate)
                metadata.update(
                    accepted=estimate.accepted,
                    pva_timings_ms=correspondence.timings_ms,
                    rejection_reasons=list(estimate.rejection_reasons),
                    motion_fit=finite_json(
                        estimate.to_dict(include_inlier_indices=False)
                    ),
                    correspondence_metrics=finite_json(correspondence.metrics),
                    motion_backends=correspondence.backends,
                )
            except PvaMotionError as exc:
                chain = self.tracker.reset(frame_index)
                metadata.update(
                    pva_failure=True,
                    error=str(exc),
                    rejection_reasons=["pva_runtime_error"],
                )
        if self.cuda_warp is not None:
            start = time.perf_counter()
            if self.source_mask is None or self.source_mask.shape != gray.shape:
                self.source_mask = np.ones(gray.shape, np.uint8)
            device = self.execution == "cuda_cubic_resident"
            matrix = chain.reference_from_current_matrix
            image, valid = self.cuda_warp(gray, self.source_mask, matrix, device=device,
                                         erosion_px=self.stabilizer.config.valid_mask_erosion_px)
            if not device:
                radius = self.stabilizer.config.valid_mask_erosion_px
                if radius:
                    valid = cv2.erode(valid, np.ones((radius*2+1, radius*2+1), np.uint8),
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
                valid = valid.astype(bool)
            self.previous = current
            metadata.update(reset=chain.window_reset, status=chain.status,
                            warp_timings_ms={"exact_cuda_total": 1000*(time.perf_counter()-start)},
                            stabilization_execution=self.execution,
                            gaussian_in_warp_stage=device, cpu_warp_fallback=False)
            return image, valid, matrix, chain.segment_index, metadata
        stabilized = self.stabilizer.stabilize(current, chain)
        self.previous = current
        metadata.update(
            reset=chain.window_reset,
            warp_timings_ms=stabilized.timings_ms,
            status=chain.status,
        )
        return (
            stabilized.frame.image,
            stabilized.frame.valid_mask,
            stabilized.source_to_reference_matrix,
            stabilized.segment_index,
            metadata,
        )


class VisiblePointDetector:
    """Shared full-frame filters, spatially balanced quotas, explicit coverage loss."""

    margin = 6  # support of the 13x13 spatial background kernel

    def __init__(self, config):
        self.config = config
        self.background = None
        self.previous_valid = None
        self.segment = None
        self.count = 0
        self._state_scratch = None
        self._state_observed = None
        self._cuda_median = None
        self._resident = None
        if config.state_update_backend == "cuda_resident":
            from .visible_resident import VisibleCudaResident
            self._resident = VisibleCudaResident(config)
        elif config.spatial_filter_backend == "cuda_median5":
            from .visible_cuda import CudaMedian5
            self._cuda_median = CudaMedian5(config.cuda_median_library)

    def close(self):
        if self._resident is not None:
            self._resident.close()
        if self._cuda_median is not None:
            self._cuda_median.close()

    def update(self, image, valid, segment, learning_centers=()):
        if self._resident is not None:
            return self._resident.update(image,valid,segment,learning_centers)
        cfg = self.config
        if (
            image.ndim != 2
            or image.shape != valid.shape
            or not np.isfinite(image).all()
        ):
            raise ValueError("finite grayscale image and matching mask required")
        started = time.perf_counter()
        image = image.astype(np.float32, copy=False)
        local_background = (
            (self._cuda_median(image) if self._cuda_median is not None else cv2.medianBlur(image, 5))
            if cfg.spatial_background == "median5"
            else cv2.boxFilter(image, -1, (13, 13), normalize=True)
        )
        spatial = cv2.GaussianBlur(image, (5, 5), 0.8) - local_background
        # Filtering happens before tile splitting, so quota seams do not lose PSF support.
        support = cv2.erode(
            valid.astype(np.uint8),
            np.ones((13, 13), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        if self.background is None or segment != self.segment:
            self.background = spatial.copy()
            self.previous_valid = support.copy()
            self.count = 0
            self.segment = segment
            self.variance = (
                np.full_like(spatial, cfg.noise_floor_dn ** 2)
                if cfg.pixel_noise_enabled
                else None
            )
            self.previous_spatial = (
                spatial.copy()
                if cfg.pixel_noise_enabled
                and cfg.pixel_noise_model == "frame_difference"
                else None
            )
        newly_valid = support & ~self.previous_valid
        self.background[newly_valid] = spatial[newly_valid]
        if cfg.pixel_noise_enabled:
            self.variance[newly_valid] = cfg.noise_floor_dn ** 2
        temporal = spatial - self.background
        self.count += 1
        ready = self.count > cfg.warmup_frames
        eligible = support & self.previous_valid if ready else np.zeros_like(support)
        absolute = np.abs(temporal)
        peaks = absolute == cv2.dilate(absolute, np.ones((5, 5), np.uint8))
        cells = []
        above_threshold = 0
        tile_dropped = 0
        noise_values = []
        h, w = image.shape
        for y in range(0, h, cfg.tile_size):
            for x in range(0, w, cfg.tile_size):
                ys = slice(y, min(y + cfg.tile_size, h))
                xs = slice(x, min(x + cfg.tile_size, w))
                r = temporal[ys, xs]
                s = spatial[ys, xs]
                mask = eligible[ys, xs]
                sample = r[:: cfg.noise_sample_stride, :: cfg.noise_sample_stride]
                sample_mask = support[ys, xs][
                    :: cfg.noise_sample_stride, :: cfg.noise_sample_stride
                ]
                sample = sample[sample_mask]
                center = float(np.median(sample)) if sample.size else 0.0
                sigma = max(
                    cfg.noise_floor_dn,
                    1.4826 * float(np.median(np.abs(sample - center)))
                    if sample.size
                    else 0.0,
                )
                noise_values.append(sigma)
                pixel_sigma = (
                    np.maximum(sigma, np.sqrt(self.variance[ys, xs]))
                    if cfg.pixel_noise_enabled
                    else np.full(r.shape, sigma, dtype=np.float32)
                )
                for polarity, sign in [("bright", 1), ("dark", -1)]:
                    yy, xx = np.nonzero(
                        mask
                        & peaks[ys, xs]
                        & (
                            sign * (r - center)
                            >= cfg.temporal_threshold_sigma * pixel_sigma
                        )
                        & (sign * s >= cfg.spatial_threshold_sigma * pixel_sigma)
                    )
                    scores = sign * (r[yy, xx] - center) / pixel_sigma[yy, xx]
                    order = np.lexsort((xx, yy, -scores))
                    above_threshold += len(order)
                    tile_dropped += max(
                        0, len(order) - cfg.max_candidates_per_tile_polarity
                    )
                    selected = []
                    for k in order[: cfg.max_candidates_per_tile_polarity]:
                        selected.append(
                            dict(
                                x=int(x + xx[k]),
                                y=int(y + yy[k]),
                                polarity=polarity,
                                score=float(scores[k]),
                                response_dn=float(r[yy[k], xx[k]]),
                                noise_sigma_dn=float(pixel_sigma[yy[k], xx[k]]),
                            )
                        )
                    if selected:
                        cells.append(selected)
        # Round-robin by tile/polarity: a bright city region cannot take all slots.
        proposals = [
            cell[rank]
            for rank in range(cfg.max_candidates_per_tile_polarity)
            for cell in cells
            if rank < len(cell)
        ]
        global_dropped = max(0, len(proposals) - cfg.max_candidates_per_frame)
        proposals = proposals[: cfg.max_candidates_per_frame]
        shape_metrics = {}
        if cfg.shape_measurement_mode == "mutual_half_height_r8":
            proposals, shape_metrics = consolidate_half_height(
                proposals, spatial, eligible,
                include_support=cfg.learning_protection_geometry == "observed_shape",
            )
        learn = support
        if cfg.learning_protection_geometry == "observed_shape" and learning_centers:
            if len(learning_centers) > 2 * cfg.max_active_tracks_per_polarity:
                raise ValueError("Learning protection exceeds active-track bound")
            learn = shape_learning_mask(support, learning_centers, cfg.position_sigma_px)
        if cfg.learning_exclusion_radius_px and learning_centers:
            if len(learning_centers) > 2 * cfg.max_active_tracks_per_polarity:
                raise ValueError("Learning protection exceeds active-track bound")
            learn = support.copy()
            radius = cfg.learning_exclusion_radius_px
            for cx, cy in learning_centers:
                if not math.isfinite(cx) or not math.isfinite(cy):
                    raise ValueError("Finite causal learning center required")
                x, y = round(cx), round(cy)
                x0, x1 = max(0, x - radius), min(w, x + radius + 1)
                y0, y1 = max(0, y - radius), min(h, y + radius + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.ogrid[y0:y1, x0:x1]
                learn[y0:y1, x0:x1] &= (xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2
        # Only state learning is protected. Current scoring, support, candidate
        # quotas and thresholds remain untouched; no predicted detections added.
        background_learn = (
            support if cfg.learning_protection_mode == "variance_only" else learn
        )
        inplace = cfg.state_update_backend == "inplace"
        if inplace:
            if self._state_scratch is None or self._state_scratch.shape != temporal.shape:
                self._state_scratch = np.empty_like(temporal)
                self._state_observed = np.empty_like(temporal) if cfg.pixel_noise_enabled else None
            np.multiply(temporal, cfg.background_alpha, out=self._state_scratch)
            np.add(self.background, self._state_scratch, out=self.background, where=background_learn)
        else:
            self.background[background_learn] += (
                cfg.background_alpha * temporal[background_learn]
            )
        if cfg.pixel_noise_enabled:
            # Causal estimate: score before learning from this frame. Clip a new
            # target's influence, while learning repeatedly unstable source locations.
            if cfg.pixel_noise_model == "frame_difference":
                change = spatial - self.previous_spatial
            if inplace:
                observed = self._state_observed
                if cfg.pixel_noise_model == "background_residual":
                    np.multiply(temporal, temporal, out=observed)
                else:
                    np.multiply(change, 0.5, out=observed)
                    np.multiply(observed, change, out=observed)
                np.multiply(self.variance, cfg.pixel_noise_clip_sigma ** 2, out=self._state_scratch)
                np.minimum(observed, self._state_scratch, out=observed)
                np.subtract(observed, self.variance, out=observed)
                np.multiply(observed, cfg.pixel_noise_alpha, out=observed)
                np.add(self.variance, observed, out=self.variance, where=learn)
            else:
                observed = np.minimum(
                    temporal * temporal
                    if cfg.pixel_noise_model == "background_residual"
                    else 0.5 * change * change,
                    self.variance * cfg.pixel_noise_clip_sigma ** 2,
                )
                self.variance[learn] += cfg.pixel_noise_alpha * (
                    observed[learn] - self.variance[learn]
                )
            np.maximum(self.variance, cfg.noise_floor_dn ** 2, out=self.variance)
            if cfg.pixel_noise_model == "frame_difference":
                self.previous_spatial = spatial.copy()
        self.previous_valid = support
        return (
            proposals,
            dict(
                full_shape_hw=[h, w],
                configured_crop=None,
                native_pixel_sampling=True,
                searchable_pixels=int(eligible.sum()),
                total_pixels=int(image.size),
                warmup=not ready,
                filter_support_margin_px=self.margin,
                above_threshold_count=above_threshold,
                dropped_at_tile_cap=tile_dropped,
                dropped_at_frame_cap=global_dropped,
                noise_sigma_median_dn=float(np.median(noise_values)),
                detection_ms=1000 * (time.perf_counter() - started),
                **({"shape_measurement": shape_metrics} if shape_metrics else {}),
            ),
        )


def candidate(index, p, h, w):
    return CandidateRecord(
        candidate_index=index,
        x_px=p["x"],
        y_px=p["y"],
        velocity_index=-1,
        velocity_xy_px_s=(0.0, 0.0),
        normalized_score_snr=p["score"],
        raw_sum_score=p["response_dn"],
        supporting_frame_count=1,
        support_weight=1.0,
        peak_neighbor_max_score_snr=None,
        peak_contrast_snr=None,
        peak_to_neighbor_ratio=None,
        distance_to_border_px=min(p["x"], p["y"], w - 1 - p["x"], h - 1 - p["y"]),
        distance_to_invalid_chebyshev_px=None,
        distance_to_invalid_is_lower_bound=True,
        ranking_score=p["score"],
        ranking_score_units="local_temporal_sigma",
    )


class VisibleTracks:
    def __init__(self, config, fps):
        self.config = config
        self.managers = {
            p: KalmanTrackManager(config.tracker(fps)) for p in ("bright", "dark")
        }
        self.extents = {}
        self.qualified = set()
        self.summary = {}
        self.ever_qualified = set()
        self.quality = {}
        self.previous_records = []
        self.previous_timestamp_ns = None

    def learning_centers(self, timestamp_ns, segment):
        """One-step predictions of last frame's qualified MEASURED tracks only.

        A reset, stale timestamp or missed observation cancels protection. This
        cannot supply a current measurement or extend a coast indefinitely.
        """
        if (
            not self.config.learning_protection_enabled
            or self.previous_timestamp_ns is None
        ):
            return []
        dt = (timestamp_ns - self.previous_timestamp_ns) / 1e9
        if not 0 < dt <= self.config.coast_seconds:
            return []
        if self.config.learning_protection_geometry == "observed_shape":
            return [dict(support_reference_xy=[
                [p[j] + dt * t["velocity_reference_xy_px_s"][j] for j in (0, 1)]
                for p in t["learning_shape_reference_xy"]])
                for t in self.previous_records
                if t["segment"] == segment and t["measured"] and t["qualified_moving"]
                and t.get("learning_shape_reference_xy")]
        return [
            [
                t["reference_xy"][j] + dt * t["velocity_reference_xy_px_s"][j]
                for j in (0, 1)
            ]
            for t in self.previous_records
            if t["segment"] == segment and t["measured"] and t["qualified_moving"]
        ]

    def update(self, proposals, frame_index, timestamp_ns, segment, matrix, shape):
        h, w = shape
        inverse = np.linalg.inv(matrix)
        records = []
        metrics = {}
        proposals, suppressed = suppress_nearby(
            proposals, self.config.tracking_peak_nms_radius_px
        )
        metrics["resolution_nms"] = dict(
            radius_px=self.config.tracking_peak_nms_radius_px,
            suppressed=suppressed,
            dropped_candidate_count=len(suppressed),
        )
        for polarity, manager in self.managers.items():
            selected = [p for p in proposals if p["polarity"] == polarity]
            batch = CandidateBatch(
                candidates=tuple(candidate(i, p, h, w) for i, p in enumerate(selected)),
                frame_indices=(frame_index,),
                reference_timestamp_ns=timestamp_ns,
                segment_index=segment,
                metrics={},
                timings_ms={},
            )
            # The visible journal does not consume the generic quality summary.
            # Running observations are still accumulated, and all state, gates,
            # association diagnostics and output measurements are unchanged.
            tracked = manager.update(batch, include_quality_evidence=False)
            metrics[polarity] = tracked.metrics
            for deleted in tracked.deleted_tracks:
                key = f'{polarity}:{deleted["track_id"]}'
                self.extents.pop(key, None)
                self.qualified.discard(key)
                self.quality.pop(key, None)
            for t in tracked.tracks:
                key = f"{polarity}:{t.track_id}"
                measured = t.last_measurement_timestamp_ns == timestamp_ns
                observation = selected[t.last_candidate_index] if measured else None
                if measured:
                    px, py = observation["x"], observation["y"]
                    bounds = self.extents.setdefault(key, [px, py, px, py])
                    bounds[:] = [
                        min(bounds[0], px),
                        min(bounds[1], py),
                        max(bounds[2], px),
                        max(bounds[3], py),
                    ]
                    if self.config.motion_quality_enabled:
                        quality = self.quality.setdefault(
                            key,
                            CausalMotionQuality(
                                self.config.motion_quality_window_hits,
                                self.config.motion_quality_minimum_hits,
                                self.config.motion_quality_maximum_rmse_px,
                            ),
                        )
                        quality.observe(timestamp_ns, px, py)
                bounds = self.extents[key]
                excursion = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
                consistent = (
                    not self.config.motion_quality_enabled
                    or self.quality[key].latest["passed"]
                )
                if (
                    t.confirmation_timestamp_ns is not None
                    and excursion >= self.config.minimum_moving_excursion_px
                    and consistent
                ):
                    self.qualified.add(key)
                    self.ever_qualified.add(key)
                else:
                    self.qualified.discard(key)
                record = dict(
                    track_id=key,
                    segment=segment,
                    measured=measured,
                    lifecycle=t.lifecycle_state,
                    qualified_moving=key in self.qualified,
                    source_xy=map_point(inverse, *t.state_xy_vx_vy[:2]),
                    reference_xy=list(t.state_xy_vx_vy[:2]),
                    velocity_reference_xy_px_s=list(t.state_xy_vx_vy[2:]),
                    measurement_source_xy=map_point(
                        inverse, observation["x"], observation["y"]
                    )
                    if measured
                    else None,
                    measurement_score=observation["score"] if measured else None,
                    hits=t.associated_update_count,
                    independent_hits=t.independent_confirmation_hits,
                    excursion_px=excursion,
                    confirmation_timestamp_ns=t.confirmation_timestamp_ns,
                    motion_quality=self.quality[key].latest
                    if self.config.motion_quality_enabled
                    else None,
                )
                records.append(record)
                if self.config.learning_protection_geometry == "observed_shape":
                    record["learning_shape_reference_xy"] = (
                        observation.get("shape", {}).get("support_reference_xy")
                        if measured else None)
                if key in self.qualified:
                    info = self.summary.setdefault(
                        key,
                        dict(
                            track_id=key,
                            polarity=polarity,
                            first_qualified_frame=frame_index,
                            first_qualified_timestamp_ns=timestamp_ns,
                            birth_timestamp_ns=t.birth_timestamp_ns,
                        ),
                    )
                    info.update(
                        last_frame=frame_index,
                        last_measurement_timestamp_ns=t.last_measurement_timestamp_ns,
                        hits=t.associated_update_count,
                        excursion_px=excursion,
                    )
        self.previous_records = records
        self.previous_timestamp_ns = timestamp_ns
        return records, metrics


def run(source, config_path, output, motion_config=None, max_frames=None):
    cfg = VisibleConfig(**json.loads(Path(config_path).read_text()))
    if cfg.motion_backend == "pva" and motion_config is None:
        raise ValueError("--motion-config is required for PVA")
    probe = probe_video(source)
    if probe.pixel_format not in {
        "yuvj420p",
        "yuv420p",
        "gray",
        "gray8",
        "bgr24",
        "rgb24",
        "yuvj422p",
        "yuv422p",
    }:
        raise ValueError(
            f"Uncalibrated input format {probe.pixel_format}: refusing implicit 16-to-8 conversion"
        )
    cv2.setNumThreads(cfg.opencv_threads)
    with VisibleFrameReader(source, (probe.height, probe.width),
            cfg.frame_decode_execution, max_frames) as reader:
        return _run_open_video(source, config_path, output, motion_config,
            max_frames, cfg, probe, reader)


def _run_open_video(source, config_path, output, motion_config, max_frames, cfg, probe, reader):
    fps, expected = reader.fps, reader.expected
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source)
    launch = dict(
        source=str(Path(source).resolve()),
        source_sha256=source_hash,
        source_probe=probe.to_dict(),
        configuration=asdict(cfg),
        config_sha256=sha256(config_path),
        motion_config_sha256=sha256(motion_config) if motion_config else None,
        code_sha256={
            p: sha256(Path(__file__).parent / p)
            for p in ("visible_baseline.py", "tracking/kalman.py", "visible_quality.py")
        },
        package_sha256={
            str(p.relative_to(Path(__file__).parent)): sha256(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
        timestamp_basis="container playback fps; physical acquisition cadence unverified",
        fps=fps,
        expected_frames=expected,
        max_frames=max_frames,
        annotations_supplied_to_detector=False,
        frame_decode=reader.contract,
        **({"external_accelerators": {
            "median": dict(backend="CUDA",library_path=cfg.cuda_median_library,
                           library_sha256=sha256(cfg.cuda_median_library),
                           input_dtype="float32",cpu_fallback=False,
                           host_device_copies_included_in_detection_timing=True)}}
           if cfg.spatial_filter_backend == "cuda_median5" else {}),
    )
    if cfg.native_shape_library is not None:
        if sha256(cfg.native_shape_library) != cfg.native_shape_library_sha256:
            raise ValueError('Native shape library hash changed')
        launch.setdefault('external_accelerators', {})['shape'] = dict(
            backend='native_cpu_bookkeeping', abi=1, library_path=cfg.native_shape_library,
            library_sha256=sha256(cfg.native_shape_library), fallback=False,
            centroid_reductions='NumPy float64 reference order', radius_px=8)
    (output / "launch.json").write_text(json.dumps(launch, indent=2))
    # Snapshot implementation so later development cannot make a run irreproducible.
    snapshots = output / "implementation"
    snapshots.mkdir()
    for relative in launch["package_sha256"]:
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(__file__).parent / relative).read_bytes())
    motion = (
        CpuTranslation(cfg)
        if cfg.motion_backend == "cpu_translation"
        else PvaMotion(motion_config, cfg.stabilization_execution, cfg.cuda_median_library)
    )
    detector = VisiblePointDetector(cfg)
    if cfg.stabilization_execution != "reference":
        launch["exact_cuda_stabilization"] = dict(
            execution=cfg.stabilization_execution, conformance=motion.conformance,
            library_sha256=sha256(cfg.cuda_median_library), camera_motion_model="translation",
            cpu_fallback=False, full_float_image_roundtrip=cfg.stabilization_execution == "cuda_cubic_host",
            validity_masks_on_cpu=True)
        (output / "launch.json").write_text(json.dumps(launch, indent=2))
    tracks = VisibleTracks(cfg, fps)
    totals = defaultdict(float)
    timings = defaultdict(list)
    count = 0
    availability = DetectionAvailability()
    started = time.perf_counter()
    try:
        reader.start()
        with (output / "frames.jsonl").open("x") as journal:
            while max_frames is None or count < max_frames:
                frame, decode_wait_ms = reader.read()
                if frame is None:
                    break
                if frame.index != count:
                    raise RuntimeError("Decoded frame order changed")
                gray = frame.gray
                timestamp_ns = round(count / fps * 1e9)
                t = time.perf_counter()
                image, valid, matrix, segment, motion_meta = motion.update(
                    gray, count, timestamp_ns
                )
                motion_ms = 1000 * (time.perf_counter() - t)
                centers = tracks.learning_centers(timestamp_ns, segment)
                proposals, coverage = detector.update(image, valid, segment, centers)
                if cfg.learning_protection_enabled:
                    coverage["learning_protection"] = dict(
                        causal_measured_track_centers=len(centers),
                        radius_px=cfg.learning_exclusion_radius_px,
                        mode=cfg.learning_protection_mode,
                        thresholds_changed=False,
                        **({"geometry": "observed_shape",
                            "margin_px": cfg.position_sigma_px,
                            "no_shape_fallback": True}
                           if cfg.learning_protection_geometry == "observed_shape" else {}),
                    )
                availability.update(coverage, motion_meta)
                t = time.perf_counter()
                records, track_metrics = tracks.update(
                    proposals, count, timestamp_ns, segment, matrix, image.shape
                )
                tracking_ms = 1000 * (time.perf_counter() - t)
                inverse = np.linalg.inv(matrix)
                for p in proposals:
                    p["source_xy"] = map_point(inverse, p["x"], p["y"])
                row = dict(
                    frame_index=count,
                    timestamp_ns=timestamp_ns,
                    segment=segment,
                    source_to_reference=matrix.tolist(),
                    motion=motion_meta,
                    coverage=coverage,
                    candidates=proposals,
                    tracks=records,
                    tracking_metrics=track_metrics,
                    timings_ms=dict(
                        decode=frame.decode_ms,
                        grayscale=frame.grayscale_ms,
                        decode_wait=decode_wait_ms,
                        motion_and_warp=motion_ms,
                        detection=coverage["detection_ms"],
                        tracking=tracking_ms,
                    ),
                )
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                for key, value in row["timings_ms"].items():
                    timings[key].append(value)
                for key in ("dropped_at_tile_cap", "dropped_at_frame_cap"):
                    totals[key] += coverage[key]
                totals["dropped_track_births"] += sum(
                    m.get("dropped_birth_count_at_active_track_cap", 0)
                    for m in track_metrics.values()
                )
                totals["dropped_at_tracking_resolution_nms"] += track_metrics[
                    "resolution_nms"
                ]["dropped_candidate_count"]
                totals["candidate_count"] += len(proposals)
                totals["motion_resets"] += bool(motion_meta["reset"])
                totals["warmup_frames"] += coverage["warmup"]
                count += 1
                # The reader owns at most one queued/in-flight gray frame;
                # release consumer references before requesting the next one.
                gray = frame = None
                if count % 50 == 0:
                    journal.flush()
                    print(
                        json.dumps(
                            dict(
                                frames=count,
                                expected=expected,
                                fps=count / (time.perf_counter() - started),
                                qualified_tracks=len(tracks.ever_qualified),
                                detection_ready_frames=availability.counts[
                                    "detection_ready_frames"
                                ],
                                motion_resets=availability.counts["motion_resets"],
                            )
                        ),
                        flush=True,
                    )
        reader.close()
        decode_stats = reader.completed_stats()
        if max_frames is None and count != expected:
            raise RuntimeError(f"Incomplete decode: {count}/{expected} frames")
        elapsed = time.perf_counter() - started
        report = dict(
            schema="seaqr.visible-baseline.v1",
            completed=True,
            full_clip=max_frames is None,
            frames=count,
            source_sha256=source_hash,
            configuration=asdict(cfg),
            elapsed_seconds=elapsed,
            processed_fps=count / elapsed,
            frame_decode=dict(contract=reader.contract, **decode_stats),
            timing_semantics=("decode/grayscale are work durations on the decode owner; "
                "decode_wait is consumer blocking time (zero for sequential). "
                "Overlapping durations must not be added to estimate wall latency; "
                "processed_fps includes startup of the worker, draining/join, decode and journaling."),
            counts=dict(totals),
            qualified_tracks=list(tracks.summary.values()),
            qualified_track_count=len(tracks.ever_qualified),
            timings_ms={
                k: dict(
                    mean=float(np.mean(v)),
                    median=float(np.median(v)),
                    p95=float(np.percentile(v, 95)),
                )
                for k, v in timings.items()
            },
            interpretation="Unlabeled automatic moving-track proposals, not verified objects or false positives. Score separately.",
            faint_target_synthetic_branch_enabled=False,
            availability=availability.report(),
            detection_status=availability.report()["detection_status"],
        )
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        return report
    except BaseException as exc:
        (output / "failure.json").write_text(
            json.dumps(dict(completed=False, frames=count, error=repr(exc)), indent=2)
        )
        raise
    finally:
        try:
            reader.close()
        finally:
            detector.close()
            if isinstance(motion, PvaMotion):
                motion.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-config", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    report = run(
        args.source, args.config, args.output, args.motion_config, args.max_frames
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "frames",
                    "processed_fps",
                    "qualified_track_count",
                    "detection_status",
                    "availability",
                )
            },
            indent=2,
        )
    )
    if report["full_clip"] and report["detection_status"] == "unavailable":
        # Preserve completed execution artifacts but fail the usable-coverage gate.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
