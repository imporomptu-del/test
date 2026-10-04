"""Cheap native-resolution discovery screen for point-like moving targets.

This module is intentionally a funnel, not a detector.  It decodes a native
pixel crop, removes broad temporal/spatial structure, emits a bounded number of
tile extrema, and links those extrema into review windows.  Unlabeled outputs
remain review workload and are never reported as real objects or false alarms.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .config import ConfigError, load_config
from .dense_review import describe_track_outputs
from .detection import (
    CandidateExtractionConfig,
    CandidateExtractionError,
    CandidateExtractor,
    CudaShiftAndStack,
    ReferenceSyntheticWindow,
    SyntheticTrackingConfig,
    SyntheticTrackingError,
    integrated_gaussian_kernel,
)
from .evaluation import (
    EvaluationError,
    SyntheticInjectionSpec,
    SyntheticInjector,
    SyntheticTarget,
    load_injection_spec,
)
from .frame_source import (
    FfmpegVideoSource,
    FrameSourceError,
    load_timestamp_csv,
    probe_video,
    sidecar_expected_interval_ns,
)
from .motion import (
    ComposedMotionState,
    GlobalMotionConfig,
    GlobalMotionTracker,
    PvaMotionConfig,
    PvaMotionError,
    PvaPyrLkMotionEstimator,
    fit_global_motion,
)
from .stabilization import (
    FullResolutionStabilizer,
    StabilizationConfig,
    StabilizationError,
)
from .telemetry import file_identity, run_identity, write_json_exclusive
from .types import Frame, TimestampSource


REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "seaqr.tiny-target.dense-screen.v1"


class DenseScreenError(RuntimeError):
    """The native-resolution discovery screen cannot run."""


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class DenseScreenConfig:
    """Frozen controls for the sparse native-resolution discovery funnel."""

    coverage_mode: str = field(default="configured_crop", kw_only=True)
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 4784
    crop_height: int = 1920
    background_warmup_frames: int = 4
    background_update_rate: float = 0.2
    background_outlier_update_rate: float = 0.03
    # Execution only; the indexed implementation remains the default reference.
    background_execution: str = "indexed_reference"
    background_update_exclusion_sigma: float = 3.0
    background_outlier_clip_sigma: float = 4.0
    noise_sigma_floor_dn: float = 16.0
    psf_sigma_px: float = 0.8
    psf_radius_px: int = 3
    spatial_background_radius_px: int = 4
    minimum_spatial_support_fraction: float = 0.8
    tile_size_px: int = 512
    cfar_sample_stride_px: int = 8
    threshold_sigma: float = 6.0
    sigma_floor_dn: float = 2.0
    dark_floor_dn: float = 1.0
    saturation_fraction: float = 0.995
    spatial_nms_radius_px: float = 4.0
    max_events_per_tile_per_polarity: int = 4
    max_events_per_frame: int = 128
    association_radius_px_per_frame: float = 3.0
    max_track_gap_frames: int = 2
    minimum_track_hits: int = 5
    minimum_track_span_px: float = 1.0
    maximum_track_fit_rmse_px: float = 2.0
    retained_track_pool_size: int = 128
    max_shortlist_tracks_per_clip: int = 8
    shortlist_nms_radius_px: float = 8.0
    shortlist_nms_frame_radius: int = 12
    followup_half_window_frames: int = 32
    per_frame_event_screen_enabled: bool = True
    synthetic_tracking_enabled: bool = False
    synthetic_window_frames: int = 16
    synthetic_window_stride_frames: int = 8
    synthetic_velocity_min_px_s: float = -3.0
    synthetic_velocity_max_px_s: float = 3.0
    synthetic_velocity_step_px_s: float = 1.0
    synthetic_minimum_speed_px_s: float = 0.5
    synthetic_min_valid_fraction: float = 0.75
    synthetic_raw_threshold_snr: float = 4.0
    synthetic_cfar_threshold_sigma: float = 4.0
    synthetic_max_candidates_per_window: int = 128
    synthetic_retained_candidate_pool_size: int = 256
    synthetic_track_max_gap_windows: int = 2
    synthetic_minimum_track_hits: int = 3
    synthetic_track_position_gate_px: float = 8.0
    synthetic_track_velocity_gate_px_s: float = 1.5
    synthetic_retained_track_pool_size: int = 512
    decoder_threads: int = 4
    opencv_threads: int = 2
    truth_match_radius_px: float = 3.0

    def __post_init__(self) -> None:
        if self.coverage_mode not in {"configured_crop", "full_frame"}:
            raise ValueError("coverage_mode must be configured_crop or full_frame")
        if self.coverage_mode == "full_frame" and (self.crop_x != 0 or self.crop_y != 0):
            raise ValueError("full_frame coverage requires a zero crop origin")
        if self.background_execution not in {"indexed_reference", "masked_ufunc", "cuda_temporal_exact_v1"}:
            raise ValueError("Unknown background execution policy")
        if self.background_execution == "cuda_temporal_exact_v1" and (
            self.spatial_background_radius_px != 4 or self.per_frame_event_screen_enabled
        ):
            raise ValueError("Opt-in RAW GPU background requires radius 4 and the synthetic-only screen")
        for name in (
            "crop_x",
            "crop_y",
            "max_track_gap_frames",
            "synthetic_track_max_gap_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "crop_width",
            "crop_height",
            "background_warmup_frames",
            "psf_radius_px",
            "spatial_background_radius_px",
            "tile_size_px",
            "cfar_sample_stride_px",
            "max_events_per_tile_per_polarity",
            "max_events_per_frame",
            "minimum_track_hits",
            "retained_track_pool_size",
            "max_shortlist_tracks_per_clip",
            "followup_half_window_frames",
            "shortlist_nms_frame_radius",
            "synthetic_window_frames",
            "synthetic_window_stride_frames",
            "synthetic_max_candidates_per_window",
            "synthetic_retained_candidate_pool_size",
            "synthetic_minimum_track_hits",
            "synthetic_retained_track_pool_size",
            "decoder_threads",
            "opencv_threads",
        ):
            _positive_int(getattr(self, name), name)
        for name in (
            "threshold_sigma",
            "sigma_floor_dn",
            "psf_sigma_px",
            "background_update_exclusion_sigma",
            "background_outlier_clip_sigma",
            "noise_sigma_floor_dn",
            "spatial_nms_radius_px",
            "association_radius_px_per_frame",
            "minimum_track_span_px",
            "maximum_track_fit_rmse_px",
            "shortlist_nms_radius_px",
            "truth_match_radius_px",
            "synthetic_velocity_step_px_s",
            "synthetic_raw_threshold_snr",
            "synthetic_cfar_threshold_sigma",
            "synthetic_track_position_gate_px",
            "synthetic_track_velocity_gate_px_s",
        ):
            _finite_positive(float(getattr(self, name)), name)
        for name in (
            "per_frame_event_screen_enabled",
            "synthetic_tracking_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in (
            "synthetic_velocity_min_px_s",
            "synthetic_velocity_max_px_s",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.synthetic_velocity_min_px_s > self.synthetic_velocity_max_px_s:
            raise ValueError("synthetic velocity minimum cannot exceed maximum")
        if (
            not math.isfinite(self.synthetic_minimum_speed_px_s)
            or self.synthetic_minimum_speed_px_s < 0
        ):
            raise ValueError("synthetic minimum speed must be finite and non-negative")
        if self.synthetic_window_stride_frames > self.synthetic_window_frames:
            raise ValueError("synthetic window stride cannot exceed its length")
        if (
            self.synthetic_retained_candidate_pool_size
            < self.max_shortlist_tracks_per_clip
        ):
            raise ValueError(
                "synthetic retained pool cannot be smaller than the shortlist"
            )
        if self.synthetic_retained_track_pool_size < self.max_shortlist_tracks_per_clip:
            raise ValueError(
                "synthetic retained track pool cannot be smaller than the shortlist"
            )
        if (
            not math.isfinite(self.background_update_rate)
            or not 0 < self.background_update_rate <= 1
        ):
            raise ValueError("background_update_rate must be finite and in (0, 1]")
        if (
            not math.isfinite(self.background_outlier_update_rate)
            or not 0 < self.background_outlier_update_rate <= 1
        ):
            raise ValueError(
                "background_outlier_update_rate must be finite and in (0, 1]"
            )
        if self.background_outlier_update_rate > self.background_update_rate:
            raise ValueError(
                "background outlier update rate cannot exceed the normal rate"
            )
        if not math.isfinite(self.dark_floor_dn) or self.dark_floor_dn < 0:
            raise ValueError("dark_floor_dn must be finite and non-negative")
        for name in ("minimum_spatial_support_fraction", "saturation_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.retained_track_pool_size < self.max_shortlist_tracks_per_clip:
            raise ValueError(
                "retained_track_pool_size cannot be smaller than "
                "max_shortlist_tracks_per_clip"
            )
        if self.spatial_background_radius_px < self.psf_radius_px:
            raise ValueError(
                "spatial_background_radius_px cannot be smaller than psf_radius_px"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DenseScreenConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown dense-screen configuration keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        return result


def resolve_dense_geometry(
    config: DenseScreenConfig, width: int, height: int,
) -> DenseScreenConfig:
    """Resolve opt-in full-frame coverage from the source, without resampling.

    Historical configurations retain their exact crop. Full-frame coverage is
    still subject to stabilization borders, sensor validity and filter support;
    requesting all pixels does not claim that all pixels are observable.
    """
    _positive_int(width, "source width")
    _positive_int(height, "source height")
    if config.coverage_mode == "full_frame":
        config = replace(config, crop_width=width, crop_height=height)
    if config.crop_x + config.crop_width > width or config.crop_y + config.crop_height > height:
        raise DenseScreenError("dense-screen crop exceeds source bounds")
    return config


def _support_counts_3x3(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    return [int(np.count_nonzero(mask[r * height // 3:(r + 1) * height // 3,
                                      c * width // 3:(c + 1) * width // 3]))
            for r in range(3) for c in range(3)]


def load_dense_screen_config(
    path: str | Path,
) -> tuple[DenseScreenConfig, dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseScreenError(f"Cannot load dense-screen config {resolved}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise DenseScreenError("dense-screen config schema_version must be exactly 1")
    data = dict(value)
    data.pop("schema_version")
    return DenseScreenConfig.from_mapping(data), {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class PointEvent:
    frame_index: int
    timestamp_ns: int
    x: float
    y: float
    polarity: str
    snr: float
    response_dn: float
    background_model_age_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "position_xy_px": [self.x, self.y],
            "polarity": self.polarity,
            "score_sigma": self.snr,
            "response_dn": self.response_dn,
            "background_model_age_frames": self.background_model_age_frames,
        }


@dataclass(frozen=True, slots=True)
class _DenseMatchedFrame:
    """Minimal duck-typed input retained by the CUDA shift-and-stack window."""

    response: np.ndarray
    valid_mask: np.ndarray
    timestamp_ns: int
    frame_index: int
    segment_index: int
    detection_ready: bool
    polarity: str = "bright"


@dataclass(slots=True)
class _DenseSyntheticTrack:
    track_id: int
    segment_index: int
    hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last(self) -> dict[str, Any]:
        return self.hits[-1]


@dataclass(slots=True)
class _ActiveTrack:
    track_id: int
    events: list[PointEvent] = field(default_factory=list)

    @property
    def last(self) -> PointEvent:
        return self.events[-1]

    def predicted_position(self, frame_index: int) -> tuple[float, float]:
        last = self.last
        gap = frame_index - last.frame_index
        if len(self.events) < 2:
            return last.x, last.y
        previous = self.events[-2]
        baseline = last.frame_index - previous.frame_index
        if baseline <= 0:
            return last.x, last.y
        return (
            last.x + (last.x - previous.x) * gap / baseline,
            last.y + (last.y - previous.y) * gap / baseline,
        )


def _load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise DenseScreenError(
            "OpenCV is required for dense screening; install the vision extra"
        ) from exc


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _translated_injection_spec(
    spec: SyntheticInjectionSpec, crop_x: int, crop_y: int
) -> SyntheticInjectionSpec:
    return replace(
        spec,
        targets=tuple(
            replace(
                target,
                reference_position_xy_px=(
                    target.reference_position_xy_px[0] - crop_x,
                    target.reference_position_xy_px[1] - crop_y,
                ),
            )
            for target in spec.targets
        ),
    )


class CroppedVideoSource:
    """Sequential FFmpeg source that pipes only a native-resolution crop."""

    def __init__(
        self,
        path: str | Path,
        config: DenseScreenConfig,
        *,
        timestamp_csv: str | Path | None,
        bit_depth: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.probe = probe_video(path)
        config = resolve_dense_geometry(config, self.probe.width, self.probe.height)
        self.config = config
        self.max_frames = max_frames
        if max_frames is not None:
            _positive_int(max_frames, "max_frames")
        if (
            config.crop_x + config.crop_width > self.probe.width
            or config.crop_y + config.crop_height > self.probe.height
        ):
            raise DenseScreenError(
                "dense-screen crop exceeds source bounds: "
                f"crop=({config.crop_x},{config.crop_y},"
                f"{config.crop_width},{config.crop_height}), "
                f"source=({self.probe.width},{self.probe.height})"
            )
        self.timestamps = (
            load_timestamp_csv(timestamp_csv) if timestamp_csv is not None else None
        )
        self.expected_interval_ns = (
            sidecar_expected_interval_ns(self.timestamps)
            if self.timestamps is not None
            else self.probe.interval_ns
        )
        if self.expected_interval_ns is None:
            self.expected_interval_ns = self.probe.interval_ns
        if "16" in self.probe.pixel_format:
            self.pixel_format = "gray16le"
            self.dtype = np.dtype("<u2")
            inferred_depth = 16
        else:
            self.pixel_format = "gray8"
            self.dtype = np.dtype("u1")
            inferred_depth = 8
        self.bit_depth = inferred_depth if bit_depth is None else bit_depth
        if self.bit_depth <= 0 or self.bit_depth > self.dtype.itemsize * 8:
            raise DenseScreenError(
                f"bit depth {self.bit_depth} is incompatible with {self.dtype}"
            )

    def __iter__(self) -> Iterator[Frame]:
        config = self.config
        frame_bytes = config.crop_width * config.crop_height * self.dtype.itemsize
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            str(config.decoder_threads),
            "-i",
            str(self.probe.path),
            "-map",
            "0:v:0",
            "-vf",
            (
                f"crop={config.crop_width}:{config.crop_height}:"
                f"{config.crop_x}:{config.crop_y}"
            ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pixel_format,
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        yielded = 0
        frame_index = 0
        stopped_early = False
        stderr_text = ""
        source_base_ns = self.timestamps[0] if self.timestamps is not None else 0
        try:
            while True:
                payload = _read_exact(process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise FrameSourceError(
                        f"Truncated dense-screen frame {frame_index}: "
                        f"{len(payload)} of {frame_bytes} bytes"
                    )
                if self.timestamps is not None and frame_index >= len(self.timestamps):
                    raise FrameSourceError(
                        "Video contains more frames than the timestamp sidecar"
                    )
                image = np.frombuffer(payload, self.dtype).reshape(
                    config.crop_height, config.crop_width
                )
                if self.timestamps is None:
                    timestamp_ns = frame_index * self.probe.interval_ns
                    source_timestamp_ns = None
                    timestamp_source = TimestampSource.CONTAINER_RATE
                else:
                    source_timestamp_ns = self.timestamps[frame_index]
                    timestamp_ns = source_timestamp_ns - source_base_ns
                    timestamp_source = TimestampSource.SIDECAR_UNIX_NS
                yield Frame(
                    image=image,
                    timestamp_ns=timestamp_ns,
                    source_timestamp_ns=source_timestamp_ns,
                    timestamp_source=timestamp_source,
                    frame_index=frame_index,
                    sequence=frame_index,
                    source_id=str(self.probe.path),
                    bit_depth=self.bit_depth,
                )
                yielded += 1
                frame_index += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    stopped_early = True
                    break
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            if process.stderr is not None:
                stderr_text = process.stderr.read().decode(
                    "utf-8", errors="replace"
                ).strip()
                process.stderr.close()
        if not stopped_early and process.returncode not in {0, -15}:
            raise FrameSourceError(f"ffmpeg crop decode failed: {stderr_text}")
        if yielded == 0:
            raise FrameSourceError("dense-screen video source produced no frames")


class StabilizedCropSource:
    """PVA-stabilized source that yields only the configured native crop."""

    def __init__(
        self,
        path: str | Path,
        config: DenseScreenConfig,
        motion_config_path: str | Path,
        *,
        timestamp_csv: str | Path | None,
        bit_depth: int | None,
        max_frames: int | None,
        injector: SyntheticInjector | None,
    ) -> None:
        loaded = load_config(motion_config_path)
        motion_value = loaded.raw.get("motion")
        global_value = loaded.raw.get("global_motion")
        stabilization_value = loaded.raw.get("stabilization")
        for name, value in (
            ("motion", motion_value),
            ("global_motion", global_value),
            ("stabilization", stabilization_value),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise DenseScreenError(f"{name} must be a mapping")
        self.motion_config = PvaMotionConfig.from_mapping(motion_value)
        self.global_config = GlobalMotionConfig.from_mapping(global_value)
        self.stabilization_config = StabilizationConfig.from_mapping(
            stabilization_value
        )
        self.motion_config_identity = {
            "path": str(loaded.path),
            "sha256": hashlib.sha256(loaded.path.read_bytes()).hexdigest(),
        }
        self.injector = injector
        self.source = FfmpegVideoSource(
            path,
            timestamp_csv=timestamp_csv,
            timestamp_policy=(
                "require_sidecar" if timestamp_csv is not None else "container_rate"
            ),
            bit_depth=bit_depth,
            max_frames=max_frames,
            timestamp_gap_factor=4.0,
        )
        self.probe = self.source.probe
        config = resolve_dense_geometry(config, self.probe.width, self.probe.height)
        self.config = config
        if (
            config.crop_x + config.crop_width > self.probe.width
            or config.crop_y + config.crop_height > self.probe.height
        ):
            raise DenseScreenError("dense-screen crop exceeds stabilized source bounds")
        self.metrics: dict[str, Any] = {}

    def _crop(self, frame: Frame) -> Frame:
        config = self.config
        ys = slice(config.crop_y, config.crop_y + config.crop_height)
        xs = slice(config.crop_x, config.crop_x + config.crop_width)
        mask = frame.valid_mask
        return replace(
            frame,
            image=np.ascontiguousarray(frame.image[ys, xs]),
            valid_mask=(
                np.ascontiguousarray(mask[ys, xs]) if mask is not None else None
            ),
        )

    def __iter__(self) -> Iterator[tuple[Frame, int]]:
        estimator = PvaPyrLkMotionEstimator(self.motion_config)
        stabilizer = FullResolutionStabilizer(self.stabilization_config)
        tracker: GlobalMotionTracker | None = None
        previous: Frame | None = None
        accepted = 0
        rejected = 0
        pva_failures = 0
        motion_times = []
        stabilization_times = []
        motion_pairs: list[dict[str, Any]] = []
        frames = 0
        for current in self.source:
            frames += 1
            pair: dict[str, Any] = {
                "frame_index": current.frame_index,
                "fit_accepted": None,
                "rejection_reasons": [],
                "discontinuities": [item.value for item in current.discontinuities],
            }
            if previous is None:
                tracker = GlobalMotionTracker(
                    self.global_config,
                    initial_frame_index=current.frame_index,
                )
                chain = ComposedMotionState(
                    reference_frame_index=current.frame_index,
                    current_frame_index=current.frame_index,
                    segment_index=0,
                    reference_from_current_matrix=np.eye(3),
                    status="initial_reference",
                    window_reset=False,
                    reused_pairs=0,
                    pair_parameter_delta=None,
                )
            else:
                assert tracker is not None
                if current.timestamp_ns <= previous.timestamp_ns:
                    chain = tracker.reset(current.frame_index)
                    rejected += 1
                    pair["rejection_reasons"] = ["nonincreasing_timestamp"]
                else:
                    try:
                        correspondence = estimator.estimate(previous, current)
                    except PvaMotionError as exc:
                        chain = tracker.reset(current.frame_index)
                        pva_failures += 1
                        rejected += 1
                        pair["rejection_reasons"] = ["pva_failure"]
                        pair["pva_error"] = str(exc)
                    else:
                        motion_times.append(
                            float(correspondence.timings_ms.get("total", 0.0))
                        )
                        estimate = fit_global_motion(
                            correspondence,
                            self.global_config,
                        )
                        pair.update(
                            accepted_correspondences=correspondence.count,
                            fit_accepted=estimate.accepted,
                            rejection_reasons=list(estimate.rejection_reasons),
                        )
                        chain = tracker.update(
                            estimate,
                            force_reset=bool(current.discontinuities),
                        )
                        if estimate.accepted:
                            accepted += 1
                        else:
                            rejected += 1
            pair.update(chain_status=chain.status, reference_reset=chain.window_reset,
                        segment_index=chain.segment_index)
            motion_pairs.append(pair)
            stabilized = stabilizer.stabilize(current, chain)
            stabilization_times.append(
                float(stabilized.timings_ms.get("total", 0.0))
            )
            stabilized_frame = stabilized.frame
            if self.injector is not None:
                # Inject after stabilization so the synthetic truth remains in the
                # coordinate system used by the screen and its evaluator. This
                # tests the downstream screen, NOT target interference with PVA.
                stabilized_frame = self.injector.inject(stabilized_frame)
            yield self._crop(stabilized_frame), stabilized.segment_index
            previous = current
        self.metrics = {
            "frames": frames,
            "accepted_global_transforms": accepted,
            "rejected_global_transforms": rejected,
            "pva_failures": pva_failures,
            "accepted_transforms_applied": sum(p["chain_status"] == "accepted" for p in motion_pairs),
            "reference_resets": sum(p["reference_reset"] for p in motion_pairs),
            "motion_pairs": motion_pairs,
            "median_pva_total_ms": (
                float(statistics.median(motion_times)) if motion_times else None
            ),
            "median_stabilization_total_ms": (
                float(statistics.median(stabilization_times))
                if stabilization_times
                else None
            ),
        }


class DensePointScreener:
    """Extract and link bounded point events from native-resolution frames."""

    def __init__(
        self,
        config: DenseScreenConfig,
        *,
        crop_origin_xy: tuple[int, int] = (0, 0),
        truth_targets: Sequence[SyntheticTarget] = (),
    ) -> None:
        self.config = config
        self.crop_origin_xy = crop_origin_xy
        self._truth_targets = tuple(truth_targets)
        self._cv2 = _load_cv2()
        self._cv2.setNumThreads(config.opencv_threads)
        kernel_size = 2 * config.spatial_background_radius_px + 1
        kernel = np.full(
            (kernel_size, kernel_size),
            -1.0 / (kernel_size * kernel_size),
            np.float32,
        )
        psf = integrated_gaussian_kernel(
            config.psf_sigma_px,
            config.psf_radius_px,
            0.0,
            0.0,
        )
        offset = config.spatial_background_radius_px - config.psf_radius_px
        kernel[
            offset : offset + psf.shape[0],
            offset : offset + psf.shape[1],
        ] += psf
        self._point_kernel = kernel
        self._point_kernel_l2 = float(
            np.sqrt(np.sum(kernel.astype(np.float64) ** 2))
        )
        self._background_location: np.ndarray | None = None
        self._background_variance: np.ndarray | None = None
        self._background_support: np.ndarray | None = None
        self._background_scratch: tuple[np.ndarray, np.ndarray] | None = None
        self._background_cuda: Any | None = None
        self._background_frame_count = 0
        self._shape: tuple[int, int] | None = None
        self._active: list[_ActiveTrack] = []
        self._qualified: list[dict[str, Any]] = []
        self._next_track_id = 1
        self._events_per_frame: list[int] = []
        self._valid_fraction_per_frame: list[float] = []
        self._frames_seen = 0
        self._frames_screened = 0
        self._availability: list[dict[str, Any]] = []
        self._last_timestamp_ns: int | None = None
        self._segment_index: int | None = None
        self._truth_surface_probes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._truth_raw_event_distances: dict[str, list[float]] = defaultdict(list)
        self._truth_bounded_event_distances: dict[str, list[float]] = defaultdict(list)
        self._last_synthetic_frame: _DenseMatchedFrame | None = None
        self._synthetic_window: ReferenceSyntheticWindow | None = None
        self._synthetic_extractor: CandidateExtractor | None = None
        self._synthetic_window_summaries: list[dict[str, Any]] = []
        self._synthetic_candidate_count = 0
        self._synthetic_candidate_pool: list[dict[str, Any]] = []
        self._synthetic_active_tracks: list[_DenseSyntheticTrack] = []
        self._synthetic_qualified_tracks: list[dict[str, Any]] = []
        self._next_synthetic_track_id = 1
        self._synthetic_segment_index: int | None = None
        self._synthetic_truth_probes: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )
        if config.synthetic_tracking_enabled:
            synthetic_config = SyntheticTrackingConfig(
                backend="cuda",
                window_frames=config.synthetic_window_frames,
                window_stride_frames=config.synthetic_window_stride_frames,
                vx_min_px_s=config.synthetic_velocity_min_px_s,
                vx_max_px_s=config.synthetic_velocity_max_px_s,
                vy_min_px_s=config.synthetic_velocity_min_px_s,
                vy_max_px_s=config.synthetic_velocity_max_px_s,
                velocity_step_px_s=config.synthetic_velocity_step_px_s,
                fractional_sampling="bilinear",
                min_valid_fraction=config.synthetic_min_valid_fraction,
                reference_time="midpoint",
                cuda_library_path="build/cuda/libtiny_target_cuda.so",
            )
            synthetic_backend = CudaShiftAndStack(
                synthetic_config,
                base_path=REPOSITORY,
            )
            if config.synthetic_minimum_speed_px_s > 0:
                velocities = synthetic_backend.velocity_grid
                retained = np.linalg.norm(velocities, axis=1) >= (
                    config.synthetic_minimum_speed_px_s - 1e-6
                )
                if not np.any(retained):
                    raise ValueError(
                        "synthetic minimum speed removes every velocity hypothesis"
                    )
                synthetic_backend.velocity_grid = np.ascontiguousarray(
                    velocities[retained],
                    dtype=np.float32,
                )
            self._synthetic_window = ReferenceSyntheticWindow(synthetic_backend)
            minimum_support = math.ceil(
                config.synthetic_window_frames
                * config.synthetic_min_valid_fraction
            )
            self._synthetic_extractor = CandidateExtractor(
                CandidateExtractionConfig(
                    score_threshold_snr=config.synthetic_raw_threshold_snr,
                    minimum_support_frames=minimum_support,
                    local_maximum_radius_px=2,
                    spatial_nms_radius_px=4.0,
                    velocity_nms_radius_px_s=1.5,
                    border_margin_px=8,
                    invalid_margin_px=1,
                    diagnostic_distance_limit_px=16,
                    pre_nms_candidate_limit=32768,
                    max_candidates_per_window=(
                        config.synthetic_max_candidates_per_window
                    ),
                    ranking_mode="tile_robust_cfar",
                    cfar_threshold_sigma=(
                        config.synthetic_cfar_threshold_sigma
                    ),
                    cfar_tile_height_px=256,
                    cfar_tile_width_px=256,
                    cfar_minimum_samples=4096,
                    cfar_scale_floor_snr=1.0,
                    quota_grid_rows=64,
                    quota_grid_cols=64,
                    pre_nms_candidates_per_cell=4,
                    max_candidates_per_cell=1,
                )
            )

    def _record_availability(
        self, frame: Frame, input_valid: np.ndarray,
        filter_valid: np.ndarray | None, warmed_up: bool,
    ) -> None:
        """Record support, not detections; includes initial/reset/warmup frames."""
        height, width = frame.shape
        counts = _support_counts_3x3(filter_valid) if filter_valid is not None else [0] * 9
        count = sum(counts)
        state = ("background_warmup" if not warmed_up else
                 "ready" if count else "no_valid_filter_pixels")
        self._availability.append({
            "frame_index": frame.frame_index,
            "timestamp_ns": frame.timestamp_ns,
            "segment_index": self._segment_index,
            "filter_state": state,
            "input_valid_pixels": int(np.count_nonzero(input_valid)),
            "filter_valid_pixels": count,
            "filter_valid_fraction": count / (width * height),
            "filter_valid_pixels_3x3_row_major": counts,
        })

    def _events_for_frame(self, current: Frame) -> list[PointEvent]:
        if self.config.background_execution == "cuda_temporal_exact_v1":
            return self._events_for_frame_cuda(current)
        config = self.config
        self._last_synthetic_frame = None
        image = np.asarray(current.image, dtype=np.float32)
        sensor_limit = float((1 << current.bit_depth) - 1)
        saturation = sensor_limit * config.saturation_fraction
        input_valid = (
            (current.image > config.dark_floor_dn)
            & (current.image < saturation)
        )
        if current.valid_mask is not None:
            input_valid &= current.valid_mask
        if self._background_location is None:
            self._background_location = np.where(input_valid, image, 0.0).astype(
                np.float32
            )
            self._background_variance = np.full(
                image.shape,
                config.noise_sigma_floor_dn**2,
                np.float32,
            )
            self._background_support = input_valid.astype(np.uint16)
            self._background_frame_count = 1
            self._record_availability(current, input_valid, None, False)
            return []
        assert self._background_variance is not None
        assert self._background_support is not None
        location = self._background_location
        variance = self._background_variance
        history_support = self._background_support
        masked_execution = config.background_execution == "masked_ufunc"
        if masked_execution:
            sigma = np.maximum(variance, config.noise_sigma_floor_dn**2)
            np.sqrt(sigma, out=sigma)
        else:
            sigma = np.sqrt(
                np.maximum(variance, config.noise_sigma_floor_dn**2)
            ).astype(np.float32)
        model_ready = history_support >= config.background_warmup_frames
        estimable = input_valid & (history_support > 0)
        whitened = np.zeros_like(image)
        if masked_execution:
            np.subtract(image, location, out=whitened, where=estimable)
            np.divide(whitened, sigma, out=whitened, where=estimable)
        else:
            whitened[estimable] = (
                image[estimable] - location[estimable]
            ) / sigma[estimable]

        unseen = input_valid & (history_support == 0)
        if masked_execution:
            np.copyto(location, image, where=unseen)
            np.copyto(variance, config.noise_sigma_floor_dn**2, where=unseen)
        else:
            location[unseen] = image[unseen]
            variance[unseen] = config.noise_sigma_floor_dn**2
        update = input_valid & ~unseen
        protected_outlier = update & (
            model_ready
            & (np.abs(whitened) >= config.background_update_exclusion_sigma)
        )
        innovation = image - location
        clipped = np.clip(
            innovation,
            -config.background_outlier_clip_sigma * sigma,
            config.background_outlier_clip_sigma * sigma,
        )
        normal_update = update & ~protected_outlier
        if masked_execution:
            if self._background_scratch is None or self._background_scratch[0].shape != image.shape:
                self._background_scratch = (np.empty_like(image), np.empty_like(image))
            product, squared = self._background_scratch
            for mask, rate in (
                (normal_update, config.background_update_rate),
                (protected_outlier, config.background_outlier_update_rate),
            ):
                # Keep every float32 rounding boundary and update order of the
                # reference. No fused arithmetic, quantization or policy change.
                np.multiply(clipped, rate, out=product, where=mask)
                np.add(location, product, out=location, where=mask)
                np.multiply(variance, 1.0 - rate, out=product, where=mask)
                np.square(clipped, out=squared, where=mask)
                np.multiply(squared, rate, out=squared, where=mask)
                np.add(product, squared, out=variance, where=mask)
        else:
            rate = config.background_update_rate
            location[normal_update] += rate * clipped[normal_update]
            variance[normal_update] = (
                (1.0 - rate) * variance[normal_update]
                + rate * clipped[normal_update] ** 2
            )
            outlier_rate = config.background_outlier_update_rate
            location[protected_outlier] += (
                outlier_rate * clipped[protected_outlier]
            )
            variance[protected_outlier] = (
                (1.0 - outlier_rate) * variance[protected_outlier]
                + outlier_rate * clipped[protected_outlier] ** 2
            )
        increment = input_valid & (history_support < np.iinfo(np.uint16).max)
        if masked_execution:
            np.add(history_support, np.uint16(1), out=history_support, where=increment)
        else:
            history_support[increment] += 1
        detection_ready = (
            self._background_frame_count >= config.background_warmup_frames
        )
        self._background_frame_count += 1
        if detection_ready:
            self._frames_screened += 1

        kernel_size = 2 * config.spatial_background_radius_px + 1
        detection_valid = input_valid & model_ready
        weights = detection_valid.astype(np.float32)
        point_response = self._cv2.filter2D(
            whitened,
            self._cv2.CV_32F,
            self._point_kernel,
            borderType=self._cv2.BORDER_CONSTANT,
        )
        point_response /= self._point_kernel_l2
        support = self._cv2.boxFilter(
            weights,
            self._cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=self._cv2.BORDER_CONSTANT,
        )
        required_support = (
            kernel_size * kernel_size * config.minimum_spatial_support_fraction
        )
        filter_valid = detection_valid & (support >= required_support)
        if not detection_ready:
            filter_valid[:] = False
        margin = config.spatial_background_radius_px
        filter_valid[:margin] = False
        filter_valid[-margin:] = False
        filter_valid[:, :margin] = False
        filter_valid[:, -margin:] = False
        valid_mask = filter_valid.astype(np.uint8)
        self._valid_fraction_per_frame.append(float(np.mean(filter_valid)))
        self._record_availability(current, input_valid, filter_valid, detection_ready)
        self._last_synthetic_frame = _DenseMatchedFrame(
            response=point_response,
            valid_mask=filter_valid,
            timestamp_ns=current.timestamp_ns,
            frame_index=current.frame_index,
            segment_index=(self._segment_index or 0),
            detection_ready=detection_ready,
        )
        if not config.per_frame_event_screen_enabled:
            return []

        nms_size = 2 * int(math.ceil(config.spatial_nms_radius_px)) + 1
        nms_kernel = np.ones((nms_size, nms_size), np.uint8)
        local_maximum = self._cv2.dilate(point_response, nms_kernel)
        local_minimum = self._cv2.erode(point_response, nms_kernel)

        height, width = point_response.shape
        origin_x, origin_y = self.crop_origin_xy
        events: list[PointEvent] = []
        for y0 in range(0, height, config.tile_size_px):
            y1 = min(height, y0 + config.tile_size_px)
            for x0 in range(0, width, config.tile_size_px):
                x1 = min(width, x0 + config.tile_size_px)
                tile = point_response[y0:y1, x0:x1]
                tile_mask = valid_mask[y0:y1, x0:x1]
                sampled = tile[:: config.cfar_sample_stride_px, :: config.cfar_sample_stride_px]
                sampled_mask = tile_mask[
                    :: config.cfar_sample_stride_px, :: config.cfar_sample_stride_px
                ].astype(bool)
                values = sampled[sampled_mask]
                if values.size < 32:
                    continue
                center = float(np.median(values))
                sigma = 1.4826 * float(np.median(np.abs(values - center)))
                sigma = max(sigma, config.sigma_floor_dn)
                maxima = local_maximum[y0:y1, x0:x1]
                minima = local_minimum[y0:y1, x0:x1]
                positive_indices = np.flatnonzero(
                    tile_mask.astype(bool)
                    & (tile >= maxima)
                    & (tile >= center + config.threshold_sigma * sigma)
                )
                if positive_indices.size:
                    positive_scores = (tile.flat[positive_indices] - center) / sigma
                    order = np.argsort(-positive_scores, kind="stable")[
                        : config.max_events_per_tile_per_polarity
                    ]
                else:
                    order = np.empty(0, np.int64)
                for selected in order:
                    flat_index = int(positive_indices[int(selected)])
                    local_y, local_x = divmod(flat_index, tile.shape[1])
                    response = float(tile[local_y, local_x])
                    events.append(
                        PointEvent(
                            frame_index=current.frame_index,
                            timestamp_ns=current.timestamp_ns,
                            x=float(origin_x + x0 + local_x),
                            y=float(origin_y + y0 + local_y),
                            polarity="bright",
                            snr=(response - center) / sigma,
                            response_dn=response,
                            background_model_age_frames=self._background_frame_count,
                        )
                    )
                negative_indices = np.flatnonzero(
                    tile_mask.astype(bool)
                    & (tile <= minima)
                    & (tile <= center - config.threshold_sigma * sigma)
                )
                if negative_indices.size:
                    negative_scores = (center - tile.flat[negative_indices]) / sigma
                    order = np.argsort(-negative_scores, kind="stable")[
                        : config.max_events_per_tile_per_polarity
                    ]
                else:
                    order = np.empty(0, np.int64)
                for selected in order:
                    flat_index = int(negative_indices[int(selected)])
                    local_y, local_x = divmod(flat_index, tile.shape[1])
                    response = float(tile[local_y, local_x])
                    events.append(
                        PointEvent(
                            frame_index=current.frame_index,
                            timestamp_ns=current.timestamp_ns,
                            x=float(origin_x + x0 + local_x),
                            y=float(origin_y + y0 + local_y),
                            polarity="dark",
                            snr=(center - response) / sigma,
                            response_dn=response,
                            background_model_age_frames=self._background_frame_count,
                        )
                    )
                for target in self._truth_targets:
                    if not target.active(current.frame_index):
                        continue
                    truth_x, truth_y = target.position_at(current.timestamp_ns)
                    local_x = int(round(truth_x - origin_x)) - x0
                    local_y = int(round(truth_y - origin_y)) - y0
                    if not (
                        0 <= local_x < tile.shape[1]
                        and 0 <= local_y < tile.shape[0]
                    ):
                        continue
                    truth_valid = bool(tile_mask[local_y, local_x])
                    response = float(tile[local_y, local_x])
                    score = abs(response - center) / sigma
                    polarity = "bright" if response >= center else "dark"
                    if polarity == "bright":
                        local_extremum = bool(
                            response >= maxima[local_y, local_x]
                        )
                    else:
                        local_extremum = bool(
                            response <= minima[local_y, local_x]
                        )
                    stronger = int(
                        np.count_nonzero(
                            tile_mask.astype(bool)
                            & (np.abs(tile - center) > abs(response - center))
                        )
                    )
                    self._truth_surface_probes[target.target_id].append(
                        {
                            "frame_index": current.frame_index,
                            "background_model_age_frames": self._background_frame_count,
                            "valid": truth_valid,
                            "score_sigma": score,
                            "polarity": polarity,
                            "local_extremum": local_extremum,
                            "tile_rank_lower_bound": stronger + 1,
                        }
                    )
        return events

    def _events_for_frame_cuda(self, current: Frame) -> list[PointEvent]:
        """Execution-only GPU state/support; leave the CPU FFT response unchanged."""
        if current.bit_depth != 16:
            raise DenseScreenError("The opt-in GPU background path is validated for RAW16 only")
        from .raw_background_cuda import RawBackgroundCuda

        config = self.config
        self._last_synthetic_frame = None
        image = np.asarray(current.image, dtype=np.float32)
        # Evaluate sensor thresholds on the original dtype exactly as the CPU
        # reference, including uint16 versus interpolated float32 comparison.
        input_valid = ((current.image > config.dark_floor_dn)
                       & (current.image < float((1 << current.bit_depth) - 1) * config.saturation_fraction))
        if current.valid_mask is not None:
            input_valid &= current.valid_mask
        if self._background_cuda is None:
            self._background_cuda = RawBackgroundCuda(config, current.shape)
        product = self._background_cuda.step(image, input_valid)
        self._background_frame_count = self._background_cuda.count
        if product is None:
            self._record_availability(current, input_valid, None, False)
            return []
        whitened, filter_valid, detection_ready = product
        point_response = self._cv2.filter2D(
            whitened, self._cv2.CV_32F, self._point_kernel,
            borderType=self._cv2.BORDER_CONSTANT,
        )
        point_response /= self._point_kernel_l2
        if detection_ready:
            self._frames_screened += 1
        self._valid_fraction_per_frame.append(float(np.mean(filter_valid)))
        self._record_availability(current, input_valid, filter_valid, detection_ready)
        self._last_synthetic_frame = _DenseMatchedFrame(
            response=point_response, valid_mask=filter_valid,
            timestamp_ns=current.timestamp_ns, frame_index=current.frame_index,
            segment_index=(self._segment_index or 0), detection_ready=detection_ready,
        )
        return []

    def close(self) -> None:
        """Release the optional GPU workspace; a closed instance cannot restart it."""
        if self._background_cuda is not None:
            self._background_cuda.close()

    def _probe_synthetic_truth(
        self,
        window: Any,
        ranking: Any,
        candidates: Sequence[Any],
    ) -> None:
        if not self._truth_targets:
            return
        height, width = window.score.shape
        origin_x, origin_y = self.crop_origin_xy
        score_values = window.score[window.valid_mask]
        selection_valid = window.valid_mask & np.isfinite(ranking.score)
        selection_values = ranking.score[selection_valid]
        radius = int(math.ceil(self.config.truth_match_radius_px))
        for target in self._truth_targets:
            if not all(target.active(index) for index in window.frame_indices):
                continue
            truth_x, truth_y = target.position_at(window.reference_timestamp_ns)
            local_truth_x = truth_x - origin_x
            local_truth_y = truth_y - origin_y
            center_x = int(round(local_truth_x))
            center_y = int(round(local_truth_y))
            x0 = max(0, center_x - radius)
            x1 = min(width, center_x + radius + 1)
            y0 = max(0, center_y - radius)
            y1 = min(height, center_y + radius + 1)
            local_valid = window.valid_mask[y0:y1, x0:x1]
            probe: dict[str, Any] = {
                "frame_indices": list(window.frame_indices),
                "reference_timestamp_ns": window.reference_timestamp_ns,
                "truth_position_xy_px": [truth_x, truth_y],
                "truth_velocity_xy_px_s": list(target.velocity_xy_px_s),
                "valid_score_available": bool(
                    local_valid.size and np.any(local_valid)
                ),
            }
            if not probe["valid_score_available"]:
                self._synthetic_truth_probes[target.target_id].append(probe)
                continue
            local_score = np.where(
                local_valid,
                window.score[y0:y1, x0:x1],
                -np.inf,
            )
            peak_y_local, peak_x_local = np.unravel_index(
                int(np.argmax(local_score)), local_score.shape
            )
            peak_x = x0 + int(peak_x_local)
            peak_y = y0 + int(peak_y_local)
            peak_score = float(window.score[peak_y, peak_x])
            velocity_index = int(window.velocity_index[peak_y, peak_x])
            selected_velocity = window.velocity_grid_xy_px_s[velocity_index]
            probe.update(
                {
                    "local_peak_position_xy_px": [
                        origin_x + peak_x,
                        origin_y + peak_y,
                    ],
                    "local_peak_score_snr": peak_score,
                    "local_peak_position_error_px": float(
                        math.hypot(peak_x - local_truth_x, peak_y - local_truth_y)
                    ),
                    "local_peak_selected_velocity_xy_px_s": (
                        selected_velocity.tolist()
                    ),
                    "local_peak_velocity_error_px_s": float(
                        np.linalg.norm(
                            selected_velocity
                            - np.asarray(target.velocity_xy_px_s, np.float32)
                        )
                    ),
                    "surface_rank_lower_bound": (
                        1 + int(np.count_nonzero(score_values > peak_score))
                    ),
                }
            )
            local_selection_valid = selection_valid[y0:y1, x0:x1]
            if np.any(local_selection_valid):
                local_selection = np.where(
                    local_selection_valid,
                    ranking.score[y0:y1, x0:x1],
                    -np.inf,
                )
                selection_y_local, selection_x_local = np.unravel_index(
                    int(np.argmax(local_selection)), local_selection.shape
                )
                selection_x = x0 + int(selection_x_local)
                selection_y = y0 + int(selection_y_local)
                selection_score = float(ranking.score[selection_y, selection_x])
                probe.update(
                    {
                        "selection_score_available": True,
                        "local_peak_selection_score": selection_score,
                        "selection_score_units": ranking.units,
                        "selection_surface_rank_lower_bound": (
                            1
                            + int(
                                np.count_nonzero(
                                    selection_values > selection_score
                                )
                            )
                        ),
                    }
                )
            else:
                probe["selection_score_available"] = False
            matches = []
            for candidate in candidates:
                position_error = math.hypot(
                    candidate.x_px - local_truth_x,
                    candidate.y_px - local_truth_y,
                )
                velocity_error = float(
                    np.linalg.norm(
                        np.asarray(candidate.velocity_xy_px_s, np.float32)
                        - np.asarray(target.velocity_xy_px_s, np.float32)
                    )
                )
                if (
                    position_error <= self.config.truth_match_radius_px
                    and velocity_error
                    <= 1.5 * self.config.synthetic_velocity_step_px_s
                ):
                    matches.append(
                        {
                            "candidate_index": candidate.candidate_index,
                            "position_error_px": position_error,
                            "velocity_error_px_s": velocity_error,
                        }
                    )
            probe["candidate_matches"] = matches
            self._synthetic_truth_probes[target.target_id].append(probe)

    def _close_synthetic_track(self, track: _DenseSyntheticTrack) -> None:
        if len(track.hits) < self.config.synthetic_minimum_track_hits:
            return
        timestamps = np.asarray(
            [item["reference_timestamp_ns"] for item in track.hits],
            np.int64,
        )
        times = (timestamps - timestamps[0]).astype(np.float64) / 1e9
        positions = np.asarray(
            [item["candidate"]["discrete_position_xy_px"] for item in track.hits],
            np.float64,
        )
        design = np.column_stack((np.ones(len(times)), times))
        coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
        fitted = design @ coefficients
        rmse = float(
            np.sqrt(np.mean(np.sum((positions - fitted) ** 2, axis=1)))
        )
        selection_scores = [float(item["ranking_score"]) for item in track.hits]
        raw_scores = [
            float(item["candidate"]["normalized_score_snr"])
            for item in track.hits
        ]
        independent_hits = 0
        last_end = -1
        for item in track.hits:
            start, end = item["frame_range"]
            if start > last_end:
                independent_hits += 1
                last_end = end
        ranking_score = (
            100.0 * len(track.hits)
            + 20.0 * independent_hits
            + math.log1p(max(0.0, float(statistics.median(selection_scores))))
            - rmse
        )
        summary = {
            "track_id": track.track_id,
            "segment_index": track.segment_index,
            "first_reference_frame_index": track.hits[0][
                "reference_frame_index"
            ],
            "last_reference_frame_index": track.hits[-1][
                "reference_frame_index"
            ],
            "hit_count": len(track.hits),
            "independent_nonoverlapping_hit_count": independent_hits,
            "fit_rmse_px": rmse,
            "fitted_velocity_xy_px_s": coefficients[1].tolist(),
            "median_discrete_velocity_xy_px_s": np.median(
                np.asarray(
                    [
                        item["candidate"]["discrete_velocity_xy_px_s"]
                        for item in track.hits
                    ],
                    np.float64,
                ),
                axis=0,
            ).tolist(),
            "median_selection_score": float(statistics.median(selection_scores)),
            "maximum_selection_score": max(selection_scores),
            "median_raw_score_snr": float(statistics.median(raw_scores)),
            "maximum_raw_score_snr": max(raw_scores),
            "ranking_score": ranking_score,
            "followup_frame_range": [
                max(
                    0,
                    track.hits[0]["frame_range"][0]
                    - self.config.followup_half_window_frames,
                ),
                track.hits[-1]["frame_range"][1]
                + self.config.followup_half_window_frames,
            ],
            "hits": track.hits,
        }
        self._synthetic_qualified_tracks.append(summary)
        limit = self.config.synthetic_retained_track_pool_size
        if len(self._synthetic_qualified_tracks) > 2 * limit:
            self._synthetic_qualified_tracks = sorted(
                self._synthetic_qualified_tracks,
                key=lambda item: (-item["ranking_score"], item["track_id"]),
            )[:limit]

    def _associate_synthetic_candidates(
        self,
        measurements: list[dict[str, Any]],
        *,
        window_index: int,
        reference_timestamp_ns: int,
        segment_index: int,
    ) -> None:
        if (
            self._synthetic_segment_index is not None
            and segment_index != self._synthetic_segment_index
        ):
            for track in self._synthetic_active_tracks:
                self._close_synthetic_track(track)
            self._synthetic_active_tracks.clear()
        self._synthetic_segment_index = segment_index
        active = []
        for track in self._synthetic_active_tracks:
            gap = window_index - int(track.last["window_index"])
            if gap <= self.config.synthetic_track_max_gap_windows + 1:
                active.append(track)
            else:
                self._close_synthetic_track(track)
        self._synthetic_active_tracks = active

        position_gate = self.config.synthetic_track_position_gate_px
        velocity_gate = self.config.synthetic_track_velocity_gate_px_s
        bucket_size = max(1.0, position_gate)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(measurements):
            buckets[
                (
                    math.floor(item["local_x_px"] / bucket_size),
                    math.floor(item["local_y_px"] / bucket_size),
                )
            ].append(index)
        options: list[tuple[float, float, int, int]] = []
        for track_index, track in enumerate(self._synthetic_active_tracks):
            last = track.last
            delta_t = (
                reference_timestamp_ns - last["reference_timestamp_ns"]
            ) / 1e9
            predicted_x = last["local_x_px"] + last["velocity_x_px_s"] * delta_t
            predicted_y = last["local_y_px"] + last["velocity_y_px_s"] * delta_t
            center_x = math.floor(predicted_x / bucket_size)
            center_y = math.floor(predicted_y / bucket_size)
            for bucket_y in range(center_y - 1, center_y + 2):
                for bucket_x in range(center_x - 1, center_x + 2):
                    for measurement_index in buckets.get(
                        (bucket_x, bucket_y), ()
                    ):
                        item = measurements[measurement_index]
                        position_error = math.hypot(
                            item["local_x_px"] - predicted_x,
                            item["local_y_px"] - predicted_y,
                        )
                        velocity_error = math.hypot(
                            item["velocity_x_px_s"] - last["velocity_x_px_s"],
                            item["velocity_y_px_s"] - last["velocity_y_px_s"],
                        )
                        if (
                            position_error <= position_gate
                            and velocity_error <= velocity_gate
                        ):
                            normalized = (
                                position_error / position_gate
                                + velocity_error / velocity_gate
                            )
                            options.append(
                                (
                                    normalized,
                                    -item["ranking_score"],
                                    track_index,
                                    measurement_index,
                                )
                            )
        options.sort()
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        for _cost, _score, track_index, measurement_index in options:
            if (
                track_index in used_tracks
                or measurement_index in used_measurements
            ):
                continue
            self._synthetic_active_tracks[track_index].hits.append(
                measurements[measurement_index]
            )
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)
        for measurement_index, item in enumerate(measurements):
            if measurement_index in used_measurements:
                continue
            self._synthetic_active_tracks.append(
                _DenseSyntheticTrack(
                    track_id=self._next_synthetic_track_id,
                    segment_index=segment_index,
                    hits=[item],
                )
            )
            self._next_synthetic_track_id += 1

    def _update_synthetic_tracking(self) -> None:
        if self._synthetic_window is None or self._last_synthetic_frame is None:
            return
        window = self._synthetic_window.update(self._last_synthetic_frame)
        if window is None:
            return
        assert self._synthetic_extractor is not None
        ranking = self._synthetic_extractor.ranking_surface(window)
        batch = self._synthetic_extractor.extract(
            window,
            ranking_surface=ranking,
        )
        self._synthetic_candidate_count += len(batch.candidates)
        window_index = len(self._synthetic_window_summaries)
        window_support = _support_counts_3x3(window.valid_mask)
        ranking_support = _support_counts_3x3(window.valid_mask & np.isfinite(ranking.score))
        self._synthetic_window_summaries.append(
            {
                "window_index": window_index,
                "frame_indices": list(window.frame_indices),
                "reference_timestamp_ns": window.reference_timestamp_ns,
                "segment_index": window.segment_index,
                "availability": {
                    "valid_score_pixels": sum(window_support),
                    "valid_score_pixels_3x3_row_major": window_support,
                    "valid_ranking_pixels": sum(ranking_support),
                    "valid_ranking_pixels_3x3_row_major": ranking_support,
                },
                "synthetic_tracking_metrics": window.metrics,
                "synthetic_tracking_timings_ms": window.timings_ms,
                "candidate_count": len(batch.candidates),
                "candidate_metrics": batch.metrics,
                "candidate_timings_ms": batch.timings_ms,
            }
        )
        self._probe_synthetic_truth(window, ranking, batch.candidates)
        origin_x, origin_y = self.crop_origin_xy
        measurements: list[dict[str, Any]] = []
        center_frame = window.frame_indices[len(window.frame_indices) // 2]
        for candidate in batch.candidates:
            candidate_dict = candidate.to_dict()
            candidate_dict["discrete_position_xy_px"] = [
                origin_x + candidate.x_px,
                origin_y + candidate.y_px,
            ]
            ranking_score = (
                candidate.normalized_score_snr
                if candidate.ranking_score is None
                else candidate.ranking_score
            )
            entry = {
                "window_index": window_index,
                "frame_range": [
                    window.frame_indices[0],
                    window.frame_indices[-1],
                ],
                "reference_frame_index": center_frame,
                "reference_timestamp_ns": window.reference_timestamp_ns,
                "segment_index": window.segment_index,
                "ranking_score": float(ranking_score),
                "candidate": candidate_dict,
                "followup_frame_range": [
                    max(
                        0,
                        center_frame - self.config.followup_half_window_frames,
                    ),
                    center_frame + self.config.followup_half_window_frames,
                ],
            }
            self._synthetic_candidate_pool.append(entry)
            measurements.append(
                {
                    **entry,
                    "local_x_px": float(candidate.x_px),
                    "local_y_px": float(candidate.y_px),
                    "velocity_x_px_s": float(candidate.velocity_xy_px_s[0]),
                    "velocity_y_px_s": float(candidate.velocity_xy_px_s[1]),
                }
            )
        self._associate_synthetic_candidates(
            measurements,
            window_index=window_index,
            reference_timestamp_ns=window.reference_timestamp_ns,
            segment_index=window.segment_index,
        )
        pool_limit = self.config.synthetic_retained_candidate_pool_size
        if len(self._synthetic_candidate_pool) > 2 * pool_limit:
            self._synthetic_candidate_pool = sorted(
                self._synthetic_candidate_pool,
                key=lambda item: (
                    -item["ranking_score"],
                    item["reference_frame_index"],
                    item["candidate"]["candidate_index"],
                ),
            )[:pool_limit]

    def _bounded_events(self, events: Iterable[PointEvent]) -> list[PointEvent]:
        kept: list[PointEvent] = []
        radius_squared = self.config.spatial_nms_radius_px**2
        for event in sorted(events, key=lambda item: (-item.snr, item.y, item.x)):
            duplicate = any(
                event.polarity == existing.polarity
                and (event.x - existing.x) ** 2 + (event.y - existing.y) ** 2
                <= radius_squared
                for existing in kept
            )
            if duplicate:
                continue
            kept.append(event)
            if len(kept) >= self.config.max_events_per_frame:
                break
        return kept

    def _track_summary(self, track: _ActiveTrack) -> dict[str, Any] | None:
        events = track.events
        config = self.config
        if len(events) < config.minimum_track_hits:
            return None
        times = np.asarray(
            [(event.timestamp_ns - events[0].timestamp_ns) / 1e9 for event in events],
            np.float64,
        )
        if np.ptp(times) <= 0:
            times = np.asarray(
                [event.frame_index - events[0].frame_index for event in events],
                np.float64,
            )
        positions = np.asarray([(event.x, event.y) for event in events], np.float64)
        design = np.column_stack((np.ones(len(events)), times))
        coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
        fitted = design @ coefficients
        residuals = np.linalg.norm(fitted - positions, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        span = float(np.linalg.norm(positions[-1] - positions[0]))
        if span < config.minimum_track_span_px or rmse > config.maximum_track_fit_rmse_px:
            return None
        scores = [event.snr for event in events]
        peak = max(events, key=lambda item: (item.snr, -item.frame_index))
        ranking_score = (
            math.log1p(float(statistics.median(scores)))
            + 0.35 * min(len(events), 30)
            + 0.5 * math.log1p(span)
            - 0.5 * rmse
        )
        return {
            "track_id": track.track_id,
            "polarity": events[0].polarity,
            "first_frame_index": events[0].frame_index,
            "last_frame_index": events[-1].frame_index,
            "hit_count": len(events),
            "span_px": span,
            "fit_rmse_px": rmse,
            "fitted_velocity_xy_px_s": coefficients[1].tolist(),
            "median_score_sigma": float(statistics.median(scores)),
            "maximum_score_sigma": max(scores),
            "ranking_score": ranking_score,
            "peak_event": peak.to_dict(),
            "followup_frame_range": [
                max(0, peak.frame_index - config.followup_half_window_frames),
                peak.frame_index + config.followup_half_window_frames,
            ],
            "events": [event.to_dict() for event in events],
        }

    def _close_track(self, track: _ActiveTrack) -> None:
        summary = self._track_summary(track)
        if summary is None:
            return
        self._qualified.append(summary)
        limit = self.config.retained_track_pool_size
        if len(self._qualified) > 2 * limit:
            self._qualified = sorted(
                self._qualified,
                key=lambda item: (-item["ranking_score"], item["track_id"]),
            )[:limit]

    def _associate(self, frame_index: int, events: list[PointEvent]) -> None:
        config = self.config
        still_active = []
        for track in self._active:
            if frame_index - track.last.frame_index <= config.max_track_gap_frames + 1:
                still_active.append(track)
            else:
                self._close_track(track)
        self._active = still_active

        maximum_gate = (
            config.association_radius_px_per_frame
            * (config.max_track_gap_frames + 1)
        )
        buckets: dict[tuple[str, int, int], list[int]] = {}
        for event_index, event in enumerate(events):
            key = (
                event.polarity,
                math.floor(event.x / maximum_gate),
                math.floor(event.y / maximum_gate),
            )
            buckets.setdefault(key, []).append(event_index)

        options: list[tuple[float, float, int, int]] = []
        for track_index, track in enumerate(self._active):
            predicted_x, predicted_y = track.predicted_position(frame_index)
            gap = max(1, frame_index - track.last.frame_index)
            gate = config.association_radius_px_per_frame * gap
            center_x = math.floor(predicted_x / maximum_gate)
            center_y = math.floor(predicted_y / maximum_gate)
            for bucket_y in range(center_y - 1, center_y + 2):
                for bucket_x in range(center_x - 1, center_x + 2):
                    for event_index in buckets.get(
                        (track.last.polarity, bucket_x, bucket_y), ()
                    ):
                        event = events[event_index]
                        distance = math.hypot(
                            event.x - predicted_x, event.y - predicted_y
                        )
                        if distance <= gate:
                            options.append(
                                (distance / gate, -event.snr, track_index, event_index)
                            )
        options.sort()
        used_tracks: set[int] = set()
        used_events: set[int] = set()
        for _distance, _score, track_index, event_index in options:
            if track_index in used_tracks or event_index in used_events:
                continue
            self._active[track_index].events.append(events[event_index])
            used_tracks.add(track_index)
            used_events.add(event_index)
        for event_index, event in enumerate(events):
            if event_index in used_events:
                continue
            self._active.append(
                _ActiveTrack(track_id=self._next_track_id, events=[event])
            )
            self._next_track_id += 1

    def process(self, frame: Frame, *, segment_index: int = 0) -> None:
        reference_reset = self._segment_index is not None and segment_index != self._segment_index
        if reference_reset:
            for track in self._active:
                self._close_track(track)
            self._active.clear()
            self._background_location = None
            self._background_variance = None
            self._background_support = None
            self._background_frame_count = 0
            if self._background_cuda is not None:
                self._background_cuda.reset()
        self._segment_index = segment_index
        if self._last_timestamp_ns is not None and frame.timestamp_ns <= self._last_timestamp_ns:
            raise DenseScreenError("dense-screen timestamps must increase strictly")
        if self._shape is not None and frame.shape != self._shape:
            raise DenseScreenError("dense-screen frame shape changed")
        self._shape = frame.shape
        self._last_timestamp_ns = frame.timestamp_ns
        self._frames_seen += 1
        events = self._events_for_frame(frame)
        bounded = self._bounded_events(events)
        for target in self._truth_targets:
            if not target.active(frame.frame_index):
                continue
            truth_x, truth_y = target.position_at(frame.timestamp_ns)
            raw_distances = [
                math.hypot(event.x - truth_x, event.y - truth_y) for event in events
            ]
            bounded_distances = [
                math.hypot(event.x - truth_x, event.y - truth_y) for event in bounded
            ]
            if raw_distances:
                self._truth_raw_event_distances[target.target_id].append(
                    min(raw_distances)
                )
            if bounded_distances:
                self._truth_bounded_event_distances[target.target_id].append(
                    min(bounded_distances)
                )
        self._events_per_frame.append(len(bounded))
        prior_windows = len(self._synthetic_window_summaries)
        self._update_synthetic_tracking()
        emitted = len(self._synthetic_window_summaries) > prior_windows
        self._availability[-1].update(
            reference_reset=reference_reset,
            synthetic_window_emitted=emitted,
            synthetic_window_state=(
                "disabled" if not self.config.synthetic_tracking_enabled else
                "emitted" if emitted else
                "background_warmup" if self._availability[-1]["filter_state"] == "background_warmup" else
                "accumulating_or_stride_wait"
            ),
        )
        self._associate(frame.frame_index, bounded)

    def availability_summary(self) -> dict[str, Any]:
        """Separate unavailable processing from a supported zero-candidate run."""
        counts: dict[str, int] = defaultdict(int)
        for frame in self._availability:
            counts[frame["filter_state"]] += 1
        return {
            "frames_by_filter_state": dict(counts),
            "reference_reset_count": sum(f["reference_reset"] for f in self._availability),
            "frames_with_valid_filter_support": counts.get("ready", 0),
            "frames_emitting_synthetic_windows": sum(f["synthetic_window_emitted"] for f in self._availability),
            "synthetic_windows_with_valid_scores": sum(w["availability"]["valid_score_pixels"] > 0 for w in self._synthetic_window_summaries),
            "synthetic_windows_with_valid_ranking": sum(w["availability"]["valid_ranking_pixels"] > 0 for w in self._synthetic_window_summaries),
            "per_frame_event_screen_enabled": self.config.per_frame_event_screen_enabled,
            "synthetic_tracking_enabled": self.config.synthetic_tracking_enabled,
            "region_grid": {
                "rows": 3, "columns": 3, "ordering": "row_major",
                "image_shape_hw": list(self._shape) if self._shape is not None else None,
                "coordinate_system": "native_stabilized_crop",
                "crop_origin_xy": list(self.crop_origin_xy),
                "boundaries": "integer floor: i*dimension//3",
            },
            "frames": self._availability,
            "warning": "Filter support and emitted windows are availability evidence, not real-object recall or full coverage of all target speeds/polarities.",
        }

    def _synthetic_tracking_result(self) -> dict[str, Any] | None:
        if self._synthetic_window is None:
            return None
        for track in self._synthetic_active_tracks:
            self._close_synthetic_track(track)
        self._synthetic_active_tracks.clear()
        retained_tracks = sorted(
            self._synthetic_qualified_tracks,
            key=lambda item: (-item["ranking_score"], item["track_id"]),
        )[: self.config.synthetic_retained_track_pool_size]
        self._synthetic_qualified_tracks = retained_tracks
        track_shortlist = retained_tracks[
            : self.config.max_shortlist_tracks_per_clip
        ]
        pool_limit = self.config.synthetic_retained_candidate_pool_size
        ranked = sorted(
            self._synthetic_candidate_pool,
            key=lambda item: (
                -item["ranking_score"],
                item["reference_frame_index"],
                item["candidate"]["candidate_index"],
            ),
        )[:pool_limit]
        self._synthetic_candidate_pool = ranked
        single_window_shortlist: list[dict[str, Any]] = []
        radius_squared = self.config.shortlist_nms_radius_px**2
        velocity_radius_squared = (
            1.5 * self.config.synthetic_velocity_step_px_s
        ) ** 2
        for item in ranked:
            candidate = item["candidate"]
            x, y = candidate["discrete_position_xy_px"]
            vx, vy = candidate["discrete_velocity_xy_px_s"]
            duplicate = False
            for existing in single_window_shortlist:
                prior = existing["candidate"]
                px, py = prior["discrete_position_xy_px"]
                pvx, pvy = prior["discrete_velocity_xy_px_s"]
                delta_t = (
                    item["reference_timestamp_ns"]
                    - existing["reference_timestamp_ns"]
                ) / 1e9
                predicted_x = px + pvx * delta_t
                predicted_y = py + pvy * delta_t
                if (
                    (x - predicted_x) ** 2 + (y - predicted_y) ** 2
                    <= radius_squared
                    and (vx - pvx) ** 2 + (vy - pvy) ** 2
                    <= velocity_radius_squared
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            single_window_shortlist.append(item)
            if (
                len(single_window_shortlist)
                >= self.config.max_shortlist_tracks_per_clip
            ):
                break
        window_times = [
            float(item["synthetic_tracking_timings_ms"]["total"])
            for item in self._synthetic_window_summaries
        ]
        candidate_times = [
            float(item["candidate_timings_ms"]["total"])
            for item in self._synthetic_window_summaries
        ]
        return {
            "window_count": len(self._synthetic_window_summaries),
            "candidate_count_before_clip_pool": self._synthetic_candidate_count,
            "retained_candidate_pool_count": len(ranked),
            "single_window_shortlist_count": len(single_window_shortlist),
            "single_window_shortlist": single_window_shortlist,
            "qualified_track_pool_count": len(retained_tracks),
            "track_pool": retained_tracks,
            "output_contract": describe_track_outputs(
                retained_tracks, track_shortlist,
                capacity=self.config.synthetic_retained_track_pool_size,
                preview_limit=self.config.max_shortlist_tracks_per_clip,
            ),
            "shortlist_count": len(track_shortlist),
            "shortlist": track_shortlist,
            "window_timing_ms": {
                "median": (
                    float(statistics.median(window_times))
                    if window_times
                    else None
                ),
                "maximum": max(window_times) if window_times else None,
            },
            "candidate_timing_ms": {
                "median": (
                    float(statistics.median(candidate_times))
                    if candidate_times
                    else None
                ),
                "maximum": max(candidate_times) if candidate_times else None,
            },
            "windows": self._synthetic_window_summaries,
        }

    def finalize(self) -> dict[str, Any]:
        for track in self._active:
            self._close_track(track)
        self._active.clear()
        ranked = sorted(
            self._qualified,
            key=lambda item: (-item["ranking_score"], item["track_id"]),
        )[: self.config.retained_track_pool_size]
        self._qualified = ranked
        shortlist: list[dict[str, Any]] = []
        radius_squared = self.config.shortlist_nms_radius_px**2
        for candidate in ranked:
            peak = candidate["peak_event"]
            px, py = peak["position_xy_px"]
            duplicate = False
            for existing in shortlist:
                other = existing["peak_event"]
                ox, oy = other["position_xy_px"]
                if (
                    abs(peak["frame_index"] - other["frame_index"])
                    <= self.config.shortlist_nms_frame_radius
                    and (px - ox) ** 2 + (py - oy) ** 2 <= radius_squared
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            shortlist.append(candidate)
            if len(shortlist) >= self.config.max_shortlist_tracks_per_clip:
                break
        counts = self._events_per_frame or [0]
        valid = self._valid_fraction_per_frame or [0.0]
        return {
            "frames_seen": self._frames_seen,
            "frames_screened_after_background_warmup": self._frames_screened,
            "availability": self.availability_summary(),
            "point_event_count": int(sum(counts)),
            "point_events_per_frame": {
                "minimum": min(counts),
                "median": float(statistics.median(counts)),
                "maximum": max(counts),
            },
            "valid_fraction_per_screened_frame": {
                "minimum": min(valid),
                "median": float(statistics.median(valid)),
                "maximum": max(valid),
            },
            "qualified_track_pool_count": len(self._qualified),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "synthetic_tracking": self._synthetic_tracking_result(),
        }

    @property
    def retained_tracks(self) -> tuple[Mapping[str, Any], ...]:
        """Expose the bounded post-finalize pool for synthetic-only diagnostics."""

        return tuple(self._qualified)

    def truth_probe_summary(self) -> dict[str, Any]:
        """Summarize synthetic truth visibility before temporal track ranking."""

        result = {}
        radius = self.config.truth_match_radius_px
        for target in self._truth_targets:
            probes = self._truth_surface_probes.get(target.target_id, [])
            valid = [item for item in probes if item["valid"]]
            scores = [float(item["score_sigma"]) for item in valid]
            ranks = [int(item["tile_rank_lower_bound"]) for item in valid]
            raw = self._truth_raw_event_distances.get(target.target_id, [])
            bounded = self._truth_bounded_event_distances.get(target.target_id, [])
            result[target.target_id] = {
                "probe_count": len(probes),
                "valid_probe_count": len(valid),
                "local_extremum_probe_count": sum(
                    bool(item["local_extremum"]) for item in valid
                ),
                "score_sigma": (
                    {
                        "maximum": max(scores),
                        "median": float(statistics.median(scores)),
                    }
                    if scores
                    else None
                ),
                "best_tile_rank_lower_bound": min(ranks) if ranks else None,
                "raw_event_match_frame_count": sum(value <= radius for value in raw),
                "bounded_event_match_frame_count": sum(
                    value <= radius for value in bounded
                ),
                "minimum_raw_event_distance_px": min(raw) if raw else None,
                "minimum_bounded_event_distance_px": min(bounded) if bounded else None,
            }
        return result

    def synthetic_truth_probe_summary(self) -> dict[str, Any]:
        """Summarize injected truth on the dense shift-and-stack surfaces."""

        result = {}
        for target in self._truth_targets:
            probes = self._synthetic_truth_probes.get(target.target_id, [])
            scored = [
                item for item in probes if item.get("valid_score_available")
            ]
            scores = [float(item["local_peak_score_snr"]) for item in scored]
            selection_scores = [
                float(item["local_peak_selection_score"])
                for item in scored
                if item.get("selection_score_available")
            ]
            result[target.target_id] = {
                "window_probe_count": len(probes),
                "valid_window_probe_count": len(scored),
                "candidate_matched_window_count": sum(
                    bool(item.get("candidate_matches")) for item in scored
                ),
                "local_peak_score_snr": (
                    {
                        "maximum": max(scores),
                        "median": float(statistics.median(scores)),
                    }
                    if scores
                    else None
                ),
                "local_peak_selection_score": (
                    {
                        "maximum": max(selection_scores),
                        "median": float(statistics.median(selection_scores)),
                    }
                    if selection_scores
                    else None
                ),
                "best_surface_rank_lower_bound": (
                    min(int(item["surface_rank_lower_bound"]) for item in scored)
                    if scored
                    else None
                ),
                "probes": probes,
            }
        return result


def _evaluate_injected_targets(
    shortlist: Sequence[Mapping[str, Any]],
    spec: SyntheticInjectionSpec,
    radius_px: float,
) -> dict[str, Any]:
    options: list[tuple[float, str, int, dict[str, Any]]] = []
    for target in spec.targets:
        for track_index, track in enumerate(shortlist):
            distances = []
            for event in track["events"]:
                frame_index = int(event["frame_index"])
                if not target.active(frame_index):
                    continue
                truth_x, truth_y = target.position_at(int(event["timestamp_ns"]))
                event_x, event_y = event["position_xy_px"]
                distances.append(math.hypot(event_x - truth_x, event_y - truth_y))
            if not distances:
                continue
            median_distance = float(statistics.median(distances))
            if median_distance <= radius_px:
                options.append(
                    (
                        median_distance,
                        target.target_id,
                        track_index,
                        {
                            "target_id": target.target_id,
                            "track_id": track["track_id"],
                            "median_position_error_px": median_distance,
                            "maximum_position_error_px": max(distances),
                        },
                    )
                )
    options.sort(key=lambda item: item[:3])
    used_targets: set[str] = set()
    used_tracks: set[int] = set()
    matches = []
    for _distance, target_id, track_index, match in options:
        if target_id in used_targets or track_index in used_tracks:
            continue
        used_targets.add(target_id)
        used_tracks.add(track_index)
        matches.append(match)
    return {
        "matched_targets": matches,
        "detected_target_ids": sorted(used_targets),
        "missed_target_ids": sorted(
            target.target_id for target in spec.targets if target.target_id not in used_targets
        ),
    }


def _evaluate_synthetic_tracks(
    tracks: Sequence[Mapping[str, Any]],
    spec: SyntheticInjectionSpec,
    position_radius_px: float,
    velocity_radius_px_s: float,
) -> dict[str, Any]:
    options = []
    for target in spec.targets:
        truth_velocity = np.asarray(target.velocity_xy_px_s, np.float64)
        for track_index, track in enumerate(tracks):
            position_errors = []
            velocity_errors = []
            for hit in track["hits"]:
                frame_index = int(hit["reference_frame_index"])
                if not target.active(frame_index):
                    continue
                truth_x, truth_y = target.position_at(
                    int(hit["reference_timestamp_ns"])
                )
                x, y = hit["candidate"]["discrete_position_xy_px"]
                position_errors.append(math.hypot(x - truth_x, y - truth_y))
                velocity_errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                hit["candidate"]["discrete_velocity_xy_px_s"],
                                np.float64,
                            )
                            - truth_velocity
                        )
                    )
                )
            if not position_errors:
                continue
            median_position = float(statistics.median(position_errors))
            median_velocity = float(statistics.median(velocity_errors))
            if (
                median_position <= position_radius_px
                and median_velocity <= velocity_radius_px_s
            ):
                options.append(
                    (
                        median_position + median_velocity,
                        target.target_id,
                        track_index,
                        {
                            "target_id": target.target_id,
                            "track_id": track["track_id"],
                            "hit_count": track["hit_count"],
                            "median_position_error_px": median_position,
                            "median_velocity_error_px_s": median_velocity,
                        },
                    )
                )
    options.sort(key=lambda item: item[:3])
    used_targets: set[str] = set()
    used_tracks: set[int] = set()
    matches = []
    for _cost, target_id, track_index, match in options:
        if target_id in used_targets or track_index in used_tracks:
            continue
        used_targets.add(target_id)
        used_tracks.add(track_index)
        matches.append(match)
    return {
        "matched_targets": matches,
        "detected_target_ids": sorted(used_targets),
        "missed_target_ids": sorted(
            target.target_id
            for target in spec.targets
            if target.target_id not in used_targets
        ),
    }


def screen_video(
    config_path: str | Path,
    input_video: str | Path,
    *,
    timestamp_csv: str | Path | None,
    motion_config_path: str | Path | None = None,
    max_frames: int | None = None,
    bit_depth: int | None = None,
    injection_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    config, config_identity = load_dense_screen_config(config_path)
    injection_spec = None
    injection_identity = None
    injector = None
    if injection_spec_path is not None:
        injection_spec, injection_identity = load_injection_spec(injection_spec_path)
    if motion_config_path is not None:
        injector = (
            SyntheticInjector(injection_spec) if injection_spec is not None else None
        )
        source: CroppedVideoSource | StabilizedCropSource = StabilizedCropSource(
            input_video,
            config,
            motion_config_path,
            timestamp_csv=timestamp_csv,
            bit_depth=bit_depth,
            max_frames=max_frames,
            injector=injector,
        )
    else:
        source = CroppedVideoSource(
            input_video,
            config,
            timestamp_csv=timestamp_csv,
            bit_depth=bit_depth,
            max_frames=max_frames,
        )
        config = source.config
        injector = (
            SyntheticInjector(
                _translated_injection_spec(
                    injection_spec,
                    config.crop_x,
                    config.crop_y,
                )
            )
            if injection_spec is not None
            else None
        )
    config = source.config
    screener = DensePointScreener(
        config,
        crop_origin_xy=(config.crop_x, config.crop_y),
        truth_targets=(injection_spec.targets if injection_spec is not None else ()),
    )
    started = time.perf_counter()
    if isinstance(source, StabilizedCropSource):
        for frame, segment_index in source:
            screener.process(frame, segment_index=segment_index)
    else:
        for frame in source:
            screener.process(
                injector.inject(frame) if injector is not None else frame
            )
    elapsed = time.perf_counter() - started
    result = screener.finalize()
    injection = None
    if injector is not None and injection_spec is not None:
        injection = {
            "injection_stage": "after_stabilization_before_screen" if motion_config_path is not None else "after_decode_before_screen",
            "validates_motion_under_target_interference": False,
            "identity": injection_identity,
            "specification": injection_spec.to_dict(),
            "coverage": injector.validate_coverage(),
            "truth_surface_probes": screener.truth_probe_summary(),
            "synthetic_tracking_truth_probes": (
                screener.synthetic_truth_probe_summary()
            ),
            "synthetic_track_pool_evaluation": (
                _evaluate_synthetic_tracks(
                    result["synthetic_tracking"]["track_pool"],
                    injection_spec,
                    config.truth_match_radius_px,
                    1.5 * config.synthetic_velocity_step_px_s,
                )
                if result["synthetic_tracking"] is not None
                else None
            ),
            "synthetic_track_shortlist_evaluation": (
                _evaluate_synthetic_tracks(
                    result["synthetic_tracking"]["shortlist"],
                    injection_spec,
                    config.truth_match_radius_px,
                    1.5 * config.synthetic_velocity_step_px_s,
                )
                if result["synthetic_tracking"] is not None
                else None
            ),
            "shortlist_evaluation": _evaluate_injected_targets(
                result["shortlist"],
                injection_spec,
                config.truth_match_radius_px,
            ),
            "retained_pool_evaluation": _evaluate_injected_targets(
                screener.retained_tracks,
                injection_spec,
                config.truth_match_radius_px,
            ),
        }
    timestamp_identity = (
        file_identity(timestamp_csv) if timestamp_csv is not None else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_identity(REPOSITORY),
        "configuration": {
            "identity": config_identity,
            "effective": config.to_dict(),
        },
        "source": {
            "identity": file_identity(input_video),
            "probe": source.probe.to_dict(),
            "timestamp_sidecar": timestamp_identity,
            "crop_xywh": [
                config.crop_x,
                config.crop_y,
                config.crop_width,
                config.crop_height,
            ],
            "native_pixel_sampling": True,
            "requested_image_area_fraction": (
                config.crop_width * config.crop_height / (source.probe.width * source.probe.height)
            ),
            "spatial_downscaling": False,
            "pva_stabilization": (
                {
                    "configuration": source.motion_config_identity,
                    "metrics": source.metrics,
                }
                if isinstance(source, StabilizedCropSource)
                else None
            ),
        },
        "performance": {
            "elapsed_seconds": elapsed,
            "processed_frames_per_second": (
                result["frames_seen"] / elapsed if elapsed > 0 else None
            ),
        },
        "screening": result,
        "injection": injection,
        "interpretation": {
            "real_object_count": None,
            "false_alarm_count": None,
            "warning": (
                "Unlabeled shortlist entries are bounded follow-up workload, not "
                "detections of real objects or measured false alarms."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight every-frame native-resolution point-target screen"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--timestamp-csv", type=Path)
    parser.add_argument(
        "--motion-config",
        type=Path,
        help="Use the PVA/global-motion/stabilization sections from this config",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--bit-depth", type=int)
    parser.add_argument("--injection-spec", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = screen_video(
            args.config,
            args.input_video,
            timestamp_csv=args.timestamp_csv,
            motion_config_path=args.motion_config,
            max_frames=args.max_frames,
            bit_depth=args.bit_depth,
            injection_spec_path=args.injection_spec,
        )
        output = write_json_exclusive(args.output, report)
        print(f"Wrote {output}")
    except (
        DenseScreenError,
        ConfigError,
        EvaluationError,
        CandidateExtractionError,
        FrameSourceError,
        PvaMotionError,
        StabilizationError,
        SyntheticTrackingError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"tiny-target dense screen failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
