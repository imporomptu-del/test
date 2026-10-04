"""Deterministic candidate extraction from retained synthetic-tracking maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .synthetic_types import SyntheticTrackWindow


class CandidateExtractionError(RuntimeError):
    """A candidate window violates the extraction contract."""


@dataclass(frozen=True, slots=True)
class CandidateExtractionConfig:
    """Explicit detection operating point and bounded-output policy."""

    score_threshold_snr: float | None = None
    minimum_support_frames: int | None = None
    local_maximum_radius_px: int = 1
    spatial_nms_radius_px: float = 2.0
    velocity_nms_radius_px_s: float = 1.0
    border_margin_px: int = 0
    invalid_margin_px: int = 0
    diagnostic_distance_limit_px: int = 32
    pre_nms_candidate_limit: int = 4096
    max_candidates_per_window: int = 256
    ranking_mode: str = "raw_snr"
    cfar_threshold_sigma: float = 6.0
    cfar_tile_height_px: int = 256
    cfar_tile_width_px: int = 256
    cfar_minimum_samples: int = 1024
    cfar_scale_floor_snr: float = 1.0
    quota_grid_rows: int = 0
    quota_grid_cols: int = 0
    pre_nms_candidates_per_cell: int = 0
    max_candidates_per_cell: int = 0
    track_guided_reservation_position_radius_px: float = 0.0
    track_guided_reservation_velocity_radius_px_s: float = 0.0
    track_guided_reservation_minimum_mean_speed_px_s: float = 0.0
    max_track_guided_reservations_per_window: int = 0

    def __post_init__(self) -> None:
        if self.score_threshold_snr is None or self.minimum_support_frames is None:
            raise ValueError(
                "Candidate extraction is not calibrated; set score_threshold_snr "
                "and minimum_support_frames"
            )
        if not math.isfinite(float(self.score_threshold_snr)):
            raise ValueError("candidates.score_threshold_snr must be finite")
        if self.score_threshold_snr <= 0:
            raise ValueError("candidates.score_threshold_snr must be positive")
        integer_fields = (
            "minimum_support_frames",
            "local_maximum_radius_px",
            "border_margin_px",
            "invalid_margin_px",
            "diagnostic_distance_limit_px",
            "pre_nms_candidate_limit",
            "max_candidates_per_window",
            "cfar_tile_height_px",
            "cfar_tile_width_px",
            "cfar_minimum_samples",
            "quota_grid_rows",
            "quota_grid_cols",
            "pre_nms_candidates_per_cell",
            "max_candidates_per_cell",
            "max_track_guided_reservations_per_window",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"candidates.{name} must be an integer")
        if self.minimum_support_frames <= 0:
            raise ValueError("candidates.minimum_support_frames must be positive")
        if self.local_maximum_radius_px <= 0:
            raise ValueError("candidates.local_maximum_radius_px must be positive")
        if min(self.border_margin_px, self.invalid_margin_px) < 0:
            raise ValueError("candidate margins cannot be negative")
        if self.diagnostic_distance_limit_px <= 0:
            raise ValueError("candidates.diagnostic_distance_limit_px must be positive")
        if min(self.pre_nms_candidate_limit, self.max_candidates_per_window) <= 0:
            raise ValueError("candidate limits must be positive")
        if self.pre_nms_candidate_limit < self.max_candidates_per_window:
            raise ValueError(
                "candidates.pre_nms_candidate_limit cannot be smaller than "
                "max_candidates_per_window"
            )
        for name in ("spatial_nms_radius_px", "velocity_nms_radius_px_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"candidates.{name} must be finite and non-negative")
        if self.ranking_mode not in {"raw_snr", "tile_robust_cfar"}:
            raise ValueError(
                "candidates.ranking_mode must be raw_snr or tile_robust_cfar"
            )
        if not math.isfinite(self.cfar_threshold_sigma) or self.cfar_threshold_sigma <= 0:
            raise ValueError("candidates.cfar_threshold_sigma must be positive")
        if not math.isfinite(self.cfar_scale_floor_snr) or self.cfar_scale_floor_snr <= 0:
            raise ValueError("candidates.cfar_scale_floor_snr must be positive")
        if min(
            self.cfar_tile_height_px,
            self.cfar_tile_width_px,
            self.cfar_minimum_samples,
        ) <= 0:
            raise ValueError("candidate CFAR tile/sample settings must be positive")
        quota_values = (
            self.quota_grid_rows,
            self.quota_grid_cols,
            self.pre_nms_candidates_per_cell,
            self.max_candidates_per_cell,
        )
        if any(value < 0 for value in quota_values):
            raise ValueError("candidate quota settings cannot be negative")
        quota_enabled = any(quota_values)
        if quota_enabled and not all(quota_values):
            raise ValueError("candidate quota settings must all be zero or all positive")
        if quota_enabled:
            if self.pre_nms_candidates_per_cell < self.max_candidates_per_cell:
                raise ValueError(
                    "pre_nms_candidates_per_cell cannot be smaller than "
                    "max_candidates_per_cell"
                )
            if (
                self.quota_grid_rows
                * self.quota_grid_cols
                * self.max_candidates_per_cell
                < self.max_candidates_per_window
            ):
                raise ValueError(
                    "candidate quota grid cannot satisfy max_candidates_per_window"
                )
        reservation_float_names = (
            "track_guided_reservation_position_radius_px",
            "track_guided_reservation_velocity_radius_px_s",
        )
        reservation_floats = tuple(
            float(getattr(self, name)) for name in reservation_float_names
        )
        for name, value in zip(reservation_float_names, reservation_floats):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"candidates.{name} must be finite and non-negative")
        if (
            not math.isfinite(
                self.track_guided_reservation_minimum_mean_speed_px_s
            )
            or self.track_guided_reservation_minimum_mean_speed_px_s < 0
        ):
            raise ValueError(
                "candidates.track_guided_reservation_minimum_mean_speed_px_s "
                "must be finite and non-negative"
            )
        reservation_values = reservation_floats + (
            self.max_track_guided_reservations_per_window,
        )
        reservation_enabled = any(reservation_values)
        if reservation_enabled and not all(value > 0 for value in reservation_values):
            raise ValueError(
                "track-guided reservation settings must all be zero or all positive"
            )
        if (
            self.max_track_guided_reservations_per_window
            > self.max_candidates_per_window
        ):
            raise ValueError(
                "max_track_guided_reservations_per_window cannot exceed "
                "max_candidates_per_window"
            )
        if reservation_enabled and not quota_enabled:
            raise ValueError("track-guided reservation requires spatial quotas")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "CandidateExtractionConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown candidate configuration keys: {unknown}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TrackPredictionHint:
    """Causal, read-only track prediction used only for bounded candidate rescue."""

    track_id: int
    position_xy_px: tuple[float, float]
    velocity_xy_px_s: tuple[float, float]
    lifecycle_state: str
    age_windows: int
    associated_update_count: int
    independent_confirmation_hits: int
    missed_windows: int
    mean_measurement_speed_px_s: float

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError("track prediction hint ID must be an integer")
        if self.track_id < 0:
            raise ValueError("track prediction hint ID cannot be negative")
        values = self.position_xy_px + self.velocity_xy_px_s
        if len(self.position_xy_px) != 2 or len(self.velocity_xy_px_s) != 2:
            raise ValueError("track prediction position and velocity must have two values")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("track prediction position and velocity must be finite")
        if self.lifecycle_state not in {"tentative", "confirmed", "coasted"}:
            raise ValueError("track prediction lifecycle state is not active")
        for name in (
            "age_windows",
            "associated_update_count",
            "independent_confirmation_hits",
            "missed_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"track prediction {name} must be an integer")
        if min(
            self.age_windows,
            self.associated_update_count,
            self.independent_confirmation_hits,
        ) <= 0:
            raise ValueError("track prediction observation counts must be positive")
        if self.missed_windows < 0:
            raise ValueError("track prediction missed_windows cannot be negative")
        if self.associated_update_count > self.age_windows:
            raise ValueError("track prediction updates cannot exceed age")
        if (
            not math.isfinite(self.mean_measurement_speed_px_s)
            or self.mean_measurement_speed_px_s < 0
        ):
            raise ValueError(
                "track prediction mean measurement speed must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_index: int
    x_px: int
    y_px: int
    velocity_index: int
    velocity_xy_px_s: tuple[float, float]
    normalized_score_snr: float
    raw_sum_score: float
    supporting_frame_count: int
    support_weight: float
    peak_neighbor_max_score_snr: float | None
    peak_contrast_snr: float | None
    peak_to_neighbor_ratio: float | None
    distance_to_border_px: int
    distance_to_invalid_chebyshev_px: int | None
    distance_to_invalid_is_lower_bound: bool
    ranking_score: float | None = None
    ranking_score_units: str = "normalized_shift_and_stack_snr"
    local_clutter_center_snr: float | None = None
    local_clutter_scale_snr: float | None = None
    quota_cell_row_col: tuple[int, int] | None = None
    track_guided_reservation_track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        selection = {
            "ranking_score": (
                self.normalized_score_snr
                if self.ranking_score is None
                else self.ranking_score
            ),
            "ranking_score_units": self.ranking_score_units,
            "local_clutter_center_snr": self.local_clutter_center_snr,
            "local_clutter_scale_snr": self.local_clutter_scale_snr,
            "quota_cell_row_col": (
                list(self.quota_cell_row_col)
                if self.quota_cell_row_col is not None
                else None
            ),
        }
        if self.track_guided_reservation_track_id is not None:
            selection["track_guided_reservation"] = True
            selection["reservation_track_id"] = (
                self.track_guided_reservation_track_id
            )
        return {
            "candidate_index": self.candidate_index,
            "discrete_position_xy_px": [self.x_px, self.y_px],
            "refined_position_xy_px": None,
            "velocity_index": self.velocity_index,
            "discrete_velocity_xy_px_s": list(self.velocity_xy_px_s),
            "refined_velocity_xy_px_s": None,
            "normalized_score_snr": self.normalized_score_snr,
            "raw_sum_score": self.raw_sum_score,
            "selection": selection,
            "temporal_support": {
                "supporting_frame_count": self.supporting_frame_count,
                "support_weight": self.support_weight,
                "supporting_frame_indices": None,
                "identity_availability": "count_only_in_retained_best_map",
            },
            "peak": {
                "neighbor_max_score_snr": self.peak_neighbor_max_score_snr,
                "contrast_snr": self.peak_contrast_snr,
                "peak_to_neighbor_ratio": self.peak_to_neighbor_ratio,
            },
            "validity": {
                "distance_to_border_px": self.distance_to_border_px,
                "distance_to_invalid_chebyshev_px": (
                    self.distance_to_invalid_chebyshev_px
                ),
                "distance_to_invalid_is_lower_bound": (
                    self.distance_to_invalid_is_lower_bound
                ),
            },
            "source_quality_flags": {
                "saturated_pixel": None,
                "configured_hot_or_bad_pixel": None,
                "direct_flags_available": False,
                "policy": "excluded_upstream_by_detection_validity_mask",
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    candidates: tuple[CandidateRecord, ...]
    frame_indices: tuple[int, ...]
    reference_timestamp_ns: int
    segment_index: int
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_indices": list(self.frame_indices),
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "segment_index": self.segment_index,
            "candidate_count": len(self.candidates),
            "candidates": [item.to_dict() for item in self.candidates],
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }


def _erode_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    result = np.ones_like(mask)
    for dy in range(-radius, radius + 1):
        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        destination_y0 = source_y0 + dy
        destination_y1 = source_y1 + dy
        for dx in range(-radius, radius + 1):
            shifted = np.zeros_like(mask)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            destination_x0 = source_x0 + dx
            destination_x1 = source_x1 + dx
            shifted[destination_y0:destination_y1, destination_x0:destination_x1] = (
                mask[source_y0:source_y1, source_x0:source_x1]
            )
            result &= shifted
    return result


def _local_maxima(score: np.ndarray, eligible: np.ndarray, radius: int) -> np.ndarray:
    """Return maxima with smallest-flat-index ownership of equal plateaus."""

    height, width = score.shape
    result = eligible.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            center_y0 = max(0, -dy)
            center_y1 = min(height, height - dy)
            center_x0 = max(0, -dx)
            center_x1 = min(width, width - dx)
            neighbor_y0 = center_y0 + dy
            neighbor_y1 = center_y1 + dy
            neighbor_x0 = center_x0 + dx
            neighbor_x1 = center_x1 + dx
            center = score[center_y0:center_y1, center_x0:center_x1]
            neighbor = score[neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1]
            neighbor_eligible = eligible[
                neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1
            ]
            # A negative offset has the smaller flat index and wins an equal tie.
            comparison = (
                center > neighbor
                if (dy < 0 or (dy == 0 and dx < 0))
                else center >= neighbor
            )
            comparison |= ~neighbor_eligible
            result[center_y0:center_y1, center_x0:center_x1] &= comparison
    return result


def _bounded_top_indices(
    score: np.ndarray, flat_indices: np.ndarray, limit: int
) -> tuple[np.ndarray, bool]:
    """Choose deterministic score-descending/flat-index-ascending entries."""

    if flat_indices.size <= limit:
        chosen = flat_indices
        truncated = False
    else:
        values = score.ravel()[flat_indices]
        partition = np.argpartition(values, values.size - limit)[values.size - limit :]
        cutoff = float(np.min(values[partition]))
        above = flat_indices[values > cutoff]
        ties = np.sort(flat_indices[values == cutoff])
        chosen = np.concatenate((above, ties[: limit - above.size]))
        truncated = True
    values = score.ravel()[chosen]
    order = np.lexsort((chosen, -values))
    return chosen[order], truncated


@dataclass(frozen=True, slots=True)
class CandidateRankingSurface:
    score: np.ndarray
    units: str
    tile_center_snr: np.ndarray | None
    tile_scale_snr: np.ndarray | None
    tile_sample_count: np.ndarray | None
    tile_height_px: int | None
    tile_width_px: int | None
    metrics: dict[str, Any]


def _summary(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def _tile_robust_cfar(
    score: np.ndarray,
    statistics_valid: np.ndarray,
    config: CandidateExtractionConfig,
) -> CandidateRankingSurface:
    height, width = score.shape
    tile_height = config.cfar_tile_height_px
    tile_width = config.cfar_tile_width_px
    tile_rows = math.ceil(height / tile_height)
    tile_cols = math.ceil(width / tile_width)
    center = np.full((tile_rows, tile_cols), np.nan, np.float32)
    scale = np.full((tile_rows, tile_cols), np.nan, np.float32)
    samples = np.zeros((tile_rows, tile_cols), np.int32)
    ranking = np.full(score.shape, -np.inf, np.float32)
    for tile_y in range(tile_rows):
        y0 = tile_y * tile_height
        y1 = min(height, y0 + tile_height)
        for tile_x in range(tile_cols):
            x0 = tile_x * tile_width
            x1 = min(width, x0 + tile_width)
            valid = statistics_valid[y0:y1, x0:x1]
            values = score[y0:y1, x0:x1][valid]
            samples[tile_y, tile_x] = values.size
            if values.size < config.cfar_minimum_samples:
                continue
            tile_center = float(np.median(values))
            tile_scale = max(
                1.4826 * float(np.median(np.abs(values - tile_center))),
                config.cfar_scale_floor_snr,
            )
            center[tile_y, tile_x] = tile_center
            scale[tile_y, tile_x] = tile_scale
            tile_score = ranking[y0:y1, x0:x1]
            tile_score[valid] = (
                score[y0:y1, x0:x1][valid] - tile_center
            ) / tile_scale
    valid_tiles = np.isfinite(center)
    return CandidateRankingSurface(
        score=ranking,
        units="local_robust_sigma",
        tile_center_snr=center,
        tile_scale_snr=scale,
        tile_sample_count=samples,
        tile_height_px=tile_height,
        tile_width_px=tile_width,
        metrics={
            "mode": "tile_robust_cfar",
            "tile_grid_rows_cols": [tile_rows, tile_cols],
            "tile_size_px": [tile_height, tile_width],
            "minimum_samples_per_tile": config.cfar_minimum_samples,
            "scale_floor_snr": config.cfar_scale_floor_snr,
            "valid_tile_count": int(np.count_nonzero(valid_tiles)),
            "invalid_tile_count": int(valid_tiles.size - np.count_nonzero(valid_tiles)),
            "tile_center_snr": _summary(center),
            "tile_scale_snr": _summary(scale),
            "tile_sample_count": _summary(samples.astype(np.float32)),
        },
    )


def _quota_cell(
    y: int,
    x: int,
    height: int,
    width: int,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    return min(rows - 1, y * rows // height), min(cols - 1, x * cols // width)


def _balanced_top_indices(
    ranking_score: np.ndarray,
    flat_indices: np.ndarray,
    config: CandidateExtractionConfig,
) -> tuple[np.ndarray, bool, int, int]:
    if config.quota_grid_rows == 0:
        ordered, truncated = _bounded_top_indices(
            ranking_score, flat_indices, config.pre_nms_candidate_limit
        )
        return ordered, truncated, 0, 0
    height, width = ranking_score.shape
    ys = flat_indices // width
    xs = flat_indices % width
    cell_ids = (
        np.minimum(config.quota_grid_rows - 1, ys * config.quota_grid_rows // height)
        * config.quota_grid_cols
        + np.minimum(config.quota_grid_cols - 1, xs * config.quota_grid_cols // width)
    )
    selected = []
    cells_truncated = 0
    for cell_id in range(config.quota_grid_rows * config.quota_grid_cols):
        cell_indices = flat_indices[cell_ids == cell_id]
        if not cell_indices.size:
            continue
        cell_selected, truncated = _bounded_top_indices(
            ranking_score,
            cell_indices,
            config.pre_nms_candidates_per_cell,
        )
        selected.append(cell_selected)
        cells_truncated += int(truncated)
    balanced = (
        np.concatenate(selected) if selected else np.empty(0, dtype=flat_indices.dtype)
    )
    ordered, globally_truncated = _bounded_top_indices(
        ranking_score, balanced, config.pre_nms_candidate_limit
    )
    dropped_by_cell_quotas = int(flat_indices.size - balanced.size)
    return (
        ordered,
        bool(cells_truncated or globally_truncated),
        cells_truncated,
        dropped_by_cell_quotas,
    )


def _invalid_distance(
    valid: np.ndarray, y: int, x: int, limit: int
) -> tuple[int | None, bool]:
    height, width = valid.shape
    for radius in range(1, limit + 1):
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        ring = valid[y0:y1, x0:x1]
        if not np.all(ring):
            return radius, False
    return None, True


class CandidateExtractor:
    """Extract reproducible, bounded candidates from a best-velocity surface."""

    def __init__(
        self, config: CandidateExtractionConfig | Mapping[str, Any]
    ) -> None:
        self.config = (
            config
            if isinstance(config, CandidateExtractionConfig)
            else CandidateExtractionConfig.from_mapping(config)
        )

    def ranking_surface(self, window: SyntheticTrackWindow) -> CandidateRankingSurface:
        config = self.config
        assert config.minimum_support_frames is not None
        statistics_valid = window.valid_mask & (
            window.valid_support_count >= config.minimum_support_frames
        )
        if config.ranking_mode == "raw_snr":
            return CandidateRankingSurface(
                score=window.score,
                units="normalized_shift_and_stack_snr",
                tile_center_snr=None,
                tile_scale_snr=None,
                tile_sample_count=None,
                tile_height_px=None,
                tile_width_px=None,
                metrics={"mode": "raw_snr"},
            )
        return _tile_robust_cfar(window.score, statistics_valid, config)

    def extract(
        self,
        window: SyntheticTrackWindow,
        *,
        ranking_surface: CandidateRankingSurface | None = None,
        track_prediction_hints: Sequence[TrackPredictionHint] = (),
    ) -> CandidateBatch:
        started = time.perf_counter_ns()
        config = self.config
        assert config.minimum_support_frames is not None
        assert config.score_threshold_snr is not None
        if config.minimum_support_frames > len(window.frame_indices):
            raise CandidateExtractionError(
                "minimum_support_frames exceeds the synthetic integration window"
            )
        height, width = window.score.shape
        ranking = ranking_surface or self.ranking_surface(window)
        if ranking.score.shape != window.score.shape:
            raise CandidateExtractionError(
                "candidate ranking surface shape does not match synthetic window"
            )
        raw_threshold = window.score >= config.score_threshold_snr
        ranking_threshold_value = (
            config.score_threshold_snr
            if config.ranking_mode == "raw_snr"
            else config.cfar_threshold_sigma
        )
        ranking_threshold = ranking.score >= ranking_threshold_value
        eligible = (
            window.valid_mask
            & (window.valid_support_count >= config.minimum_support_frames)
            & raw_threshold
            & ranking_threshold
        )
        raw_threshold_count = int(np.count_nonzero(window.valid_mask & raw_threshold))
        ranking_threshold_count = int(
            np.count_nonzero(window.valid_mask & ranking_threshold)
        )
        threshold_count = int(np.count_nonzero(eligible))
        if config.border_margin_px:
            margin = config.border_margin_px
            if 2 * margin >= min(height, width):
                eligible[:] = False
            else:
                eligible[:margin] = False
                eligible[-margin:] = False
                eligible[:, :margin] = False
                eligible[:, -margin:] = False
        after_border_count = int(np.count_nonzero(eligible))
        if config.invalid_margin_px:
            eligible &= _erode_square(window.valid_mask, config.invalid_margin_px)
        after_support_and_margin_count = int(np.count_nonzero(eligible))
        peaks = _local_maxima(
            window.score, eligible, config.local_maximum_radius_px
        )
        peak_indices = np.flatnonzero(peaks)
        spatial_peak_count = int(peak_indices.size)
        (
            ordered,
            pre_nms_truncated,
            pre_nms_cells_truncated,
            pre_nms_dropped_by_cell_quotas,
        ) = _balanced_top_indices(
            ranking.score, peak_indices, config
        )
        velocities = window.velocity_grid_xy_px_s
        kept: list[int] = []
        spatial_radius_squared = float(config.spatial_nms_radius_px) ** 2
        velocity_radius_squared = float(config.velocity_nms_radius_px_s) ** 2
        suppressed = 0
        quota_suppressed = 0
        quota_suppressed_indices: list[int] = []
        quota_counts: dict[tuple[int, int], int] = {}
        spatial_buckets: dict[tuple[int, int], list[int]] = {}
        for flat_index in ordered:
            y, x = divmod(int(flat_index), width)
            quota_cell = None
            if config.quota_grid_rows:
                quota_cell = _quota_cell(
                    y,
                    x,
                    height,
                    width,
                    config.quota_grid_rows,
                    config.quota_grid_cols,
                )
                if quota_counts.get(quota_cell, 0) >= config.max_candidates_per_cell:
                    quota_suppressed += 1
                    quota_suppressed_indices.append(int(flat_index))
                    continue
            velocity = velocities[int(window.velocity_index[y, x])]
            duplicate = False
            bucket_x = bucket_y = 0
            nearby: list[int] = []
            if config.spatial_nms_radius_px > 0:
                bucket_x = int(x // config.spatial_nms_radius_px)
                bucket_y = int(y // config.spatial_nms_radius_px)
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        nearby.extend(
                            spatial_buckets.get(
                                (bucket_x + offset_x, bucket_y + offset_y), ()
                            )
                        )
            for prior in nearby:
                prior_y, prior_x = divmod(prior, width)
                if (x - prior_x) ** 2 + (y - prior_y) ** 2 > spatial_radius_squared:
                    continue
                prior_velocity = velocities[int(window.velocity_index[prior_y, prior_x])]
                delta = velocity - prior_velocity
                if float(delta @ delta) <= velocity_radius_squared:
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                kept.append(int(flat_index))
                if quota_cell is not None:
                    quota_counts[quota_cell] = quota_counts.get(quota_cell, 0) + 1
                if config.spatial_nms_radius_px > 0:
                    spatial_buckets.setdefault((bucket_x, bucket_y), []).append(
                        int(flat_index)
                    )
        retained_before_cap = len(kept)
        output_truncated = retained_before_cap > config.max_candidates_per_window
        baseline_kept = kept[: config.max_candidates_per_window]
        reservation_enabled = bool(
            config.max_track_guided_reservations_per_window
        )
        reserved: list[int] = []
        reserved_track_by_flat_index: dict[int, int] = {}
        covered_hint_count_before_reservation = 0
        reservation_eligible_suppressed_count = 0
        reservation_suppressed_examined_count = 0
        eligible_tentative_hint_count = 0
        baseline_candidates_displaced = 0
        hints = (
            tuple(sorted(track_prediction_hints, key=lambda item: item.track_id))
            if reservation_enabled
            else ()
        )
        if len({hint.track_id for hint in hints}) != len(hints):
            raise CandidateExtractionError("track prediction hint IDs must be unique")
        if reservation_enabled and hints:
            eligible_hints = tuple(
                sorted(
                    (
                        hint
                        for hint in hints
                        if (
                            hint.lifecycle_state == "tentative"
                            and hint.missed_windows == 0
                            and hint.associated_update_count == hint.age_windows
                            and hint.associated_update_count
                            >= len(window.frame_indices)
                            and hint.mean_measurement_speed_px_s
                            >= config.track_guided_reservation_minimum_mean_speed_px_s
                        )
                    ),
                    key=lambda hint: (
                        hint.missed_windows,
                        -(hint.associated_update_count / hint.age_windows),
                        -hint.associated_update_count,
                        -hint.age_windows,
                        hint.track_id,
                    ),
                )
            )
            eligible_tentative_hint_count = len(eligible_hints)
            position_radius_squared = (
                config.track_guided_reservation_position_radius_px**2
            )
            reservation_velocity_radius_squared = (
                config.track_guided_reservation_velocity_radius_px_s**2
            )

            def matches_hint(
                flat_index: int, hint: TrackPredictionHint
            ) -> bool:
                candidate_y, candidate_x = divmod(flat_index, width)
                candidate_velocity = velocities[
                    int(window.velocity_index[candidate_y, candidate_x])
                ]
                position_dx = hint.position_xy_px[0] - candidate_x
                position_dy = hint.position_xy_px[1] - candidate_y
                velocity_dx = hint.velocity_xy_px_s[0] - candidate_velocity[0]
                velocity_dy = hint.velocity_xy_px_s[1] - candidate_velocity[1]
                return (
                    position_dx * position_dx + position_dy * position_dy
                    <= position_radius_squared
                    and velocity_dx * velocity_dx + velocity_dy * velocity_dy
                    <= reservation_velocity_radius_squared
                )

            covered_hint_ids = {
                hint.track_id
                for hint in eligible_hints
                if any(matches_hint(flat_index, hint) for flat_index in baseline_kept)
            }
            covered_hint_count_before_reservation = len(covered_hint_ids)
            selected_for_nms = list(baseline_kept)
            eligible_suppressed_indices: set[int] = set()
            for hint in eligible_hints:
                if (
                    len(reserved)
                    >= config.max_track_guided_reservations_per_window
                ):
                    break
                if hint.track_id in covered_hint_ids:
                    continue
                for flat_index in quota_suppressed_indices:
                    reservation_suppressed_examined_count += 1
                    if not matches_hint(flat_index, hint):
                        continue
                    eligible_suppressed_indices.add(flat_index)
                    y, x = divmod(flat_index, width)
                    velocity = velocities[int(window.velocity_index[y, x])]
                    duplicate = False
                    if config.spatial_nms_radius_px > 0:
                        for prior in selected_for_nms:
                            prior_y, prior_x = divmod(prior, width)
                            if (
                                (x - prior_x) ** 2 + (y - prior_y) ** 2
                                > spatial_radius_squared
                            ):
                                continue
                            prior_velocity = velocities[
                                int(window.velocity_index[prior_y, prior_x])
                            ]
                            velocity_delta = velocity - prior_velocity
                            if (
                                float(velocity_delta @ velocity_delta)
                                <= velocity_radius_squared
                            ):
                                duplicate = True
                                break
                    if duplicate:
                        continue
                    reserved.append(flat_index)
                    selected_for_nms.append(flat_index)
                    reserved_track_by_flat_index[flat_index] = hint.track_id
                    for covered_hint in eligible_hints:
                        if matches_hint(flat_index, covered_hint):
                            covered_hint_ids.add(covered_hint.track_id)
                    break
            reservation_eligible_suppressed_count = len(
                eligible_suppressed_indices
            )

            baseline_capacity = (
                config.max_candidates_per_window - len(reserved)
            )
            baseline_candidates_displaced = max(
                0, len(baseline_kept) - baseline_capacity
            )
            kept = baseline_kept[:baseline_capacity] + reserved
            kept.sort(
                key=lambda flat_index: (
                    -float(ranking.score.flat[flat_index]),
                    flat_index,
                )
            )
        else:
            kept = baseline_kept
        output_quota_counts: dict[tuple[int, int], int] = {}
        if config.quota_grid_rows:
            for flat_index in kept:
                y, x = divmod(flat_index, width)
                cell = _quota_cell(
                    y,
                    x,
                    height,
                    width,
                    config.quota_grid_rows,
                    config.quota_grid_cols,
                )
                output_quota_counts[cell] = output_quota_counts.get(cell, 0) + 1
        records = []
        diagnostic_radius = config.local_maximum_radius_px
        for candidate_index, flat_index in enumerate(kept):
            y, x = divmod(flat_index, width)
            score = float(window.score[y, x])
            support = int(window.valid_support_count[y, x])
            y0, y1 = max(0, y - diagnostic_radius), min(height, y + diagnostic_radius + 1)
            x0, x1 = max(0, x - diagnostic_radius), min(width, x + diagnostic_radius + 1)
            neighborhood = window.score[y0:y1, x0:x1].copy()
            neighborhood[y - y0, x - x0] = -np.inf
            finite_neighbors = neighborhood[np.isfinite(neighborhood)]
            neighbor = float(np.max(finite_neighbors)) if finite_neighbors.size else None
            contrast = score - neighbor if neighbor is not None else None
            ratio = score / neighbor if neighbor is not None and neighbor > 0 else None
            invalid_distance, lower_bound = _invalid_distance(
                window.valid_mask,
                y,
                x,
                config.diagnostic_distance_limit_px,
            )
            velocity_index = int(window.velocity_index[y, x])
            velocity = velocities[velocity_index]
            local_center = None
            local_scale = None
            if ranking.tile_center_snr is not None:
                assert ranking.tile_height_px is not None
                assert ranking.tile_width_px is not None
                tile_y = min(
                    ranking.tile_center_snr.shape[0] - 1,
                    y // ranking.tile_height_px,
                )
                tile_x = min(
                    ranking.tile_center_snr.shape[1] - 1,
                    x // ranking.tile_width_px,
                )
                local_center = float(ranking.tile_center_snr[tile_y, tile_x])
                local_scale = float(ranking.tile_scale_snr[tile_y, tile_x])
            candidate_quota_cell = (
                _quota_cell(
                    y,
                    x,
                    height,
                    width,
                    config.quota_grid_rows,
                    config.quota_grid_cols,
                )
                if config.quota_grid_rows
                else None
            )
            records.append(
                CandidateRecord(
                    candidate_index=candidate_index,
                    x_px=x,
                    y_px=y,
                    velocity_index=velocity_index,
                    velocity_xy_px_s=(float(velocity[0]), float(velocity[1])),
                    normalized_score_snr=score,
                    raw_sum_score=score * math.sqrt(support),
                    supporting_frame_count=support,
                    support_weight=float(support),
                    peak_neighbor_max_score_snr=neighbor,
                    peak_contrast_snr=contrast,
                    peak_to_neighbor_ratio=ratio,
                    distance_to_border_px=min(x, y, width - 1 - x, height - 1 - y),
                    distance_to_invalid_chebyshev_px=invalid_distance,
                    distance_to_invalid_is_lower_bound=lower_bound,
                    ranking_score=float(ranking.score[y, x]),
                    ranking_score_units=ranking.units,
                    local_clutter_center_snr=local_center,
                    local_clutter_scale_snr=local_scale,
                    quota_cell_row_col=candidate_quota_cell,
                    track_guided_reservation_track_id=(
                        reserved_track_by_flat_index.get(flat_index)
                    ),
                )
            )
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        return CandidateBatch(
            candidates=tuple(records),
            frame_indices=window.frame_indices,
            reference_timestamp_ns=window.reference_timestamp_ns,
            segment_index=window.segment_index,
            metrics={
                "score_surface": "best_score_over_velocity_hypotheses",
                "per_velocity_local_maxima_available": False,
                "threshold_units": "normalized_shift_and_stack_snr",
                "score_threshold_snr": config.score_threshold_snr,
                "raw_score_threshold_eligible_count": raw_threshold_count,
                "ranking": ranking.metrics
                | {
                    "score_units": ranking.units,
                    "threshold": ranking_threshold_value,
                    "threshold_eligible_count": ranking_threshold_count,
                },
                "minimum_support_frames": config.minimum_support_frames,
                "threshold_eligible_count_before_margins": threshold_count,
                "eligible_count_after_border_margin": after_border_count,
                "eligible_count_after_all_margins": after_support_and_margin_count,
                "spatial_local_maximum_count": spatial_peak_count,
                "pre_nms_evaluated_count": int(ordered.size),
                "pre_nms_candidate_limit": config.pre_nms_candidate_limit,
                "pre_nms_truncated": pre_nms_truncated,
                "pre_nms_cells_truncated": pre_nms_cells_truncated,
                "pre_nms_dropped_by_cell_quotas": pre_nms_dropped_by_cell_quotas,
                "joint_nms_suppressed_count": suppressed,
                "spatial_quota": {
                    "enabled": bool(config.quota_grid_rows),
                    "grid_rows_cols": [
                        config.quota_grid_rows,
                        config.quota_grid_cols,
                    ],
                    "pre_nms_candidates_per_cell": (
                        config.pre_nms_candidates_per_cell
                        if config.quota_grid_rows
                        else None
                    ),
                    "max_candidates_per_cell": (
                        config.max_candidates_per_cell
                        if config.quota_grid_rows
                        else None
                    ),
                    "suppressed_at_final_cell_quota": quota_suppressed,
                    "occupied_output_cells": len(output_quota_counts),
                    "maximum_output_candidates_in_one_cell": (
                        max(output_quota_counts.values(), default=0)
                    ),
                },
                "retained_before_output_cap": retained_before_cap,
                "output_candidate_count": len(records),
                "max_candidates_per_window": config.max_candidates_per_window,
                "output_truncated": output_truncated,
                "counts_after_pre_nms_limit_are_lower_bounds": pre_nms_truncated,
                "plateau_tie_policy": "smallest_flat_index",
                "output_order": "ranking_score_descending_then_flat_index_ascending",
                "position_refinement": "not_applied",
                "velocity_refinement": "not_available_without_neighbor_trial_scores",
            }
            | (
                {
                    "track_guided_reservation": {
                        "enabled": True,
                        "configured_hint_count": len(hints),
                        "eligible_continuous_tentative_hint_count": (
                            eligible_tentative_hint_count
                        ),
                        "hint_eligibility_policy": (
                            "tentative_zero_miss_full_window_continuous_observation_"
                            "and_minimum_mean_measurement_speed"
                        ),
                        "minimum_mean_measurement_speed_px_s": (
                            config.track_guided_reservation_minimum_mean_speed_px_s
                        ),
                        "hint_priority": (
                            "missed_windows_ascending_then_observation_fraction_"
                            "descending_then_updates_descending_then_age_"
                            "descending_then_track_id_ascending"
                        ),
                        "covered_hint_count_before_reservation": (
                            covered_hint_count_before_reservation
                        ),
                        "available_quota_suppressed_candidate_count": len(
                            quota_suppressed_indices
                        ),
                        "examined_quota_suppressed_candidate_count": (
                            reservation_suppressed_examined_count
                        ),
                        "eligible_quota_suppressed_candidate_count": (
                            reservation_eligible_suppressed_count
                        ),
                        "retained_reservation_count": len(reserved),
                        "reserved_track_ids": [
                            reserved_track_by_flat_index[index]
                            for index in reserved
                        ],
                        "baseline_candidates_displaced_at_global_cap": (
                            baseline_candidates_displaced
                        ),
                        "position_radius_px": (
                            config.track_guided_reservation_position_radius_px
                        ),
                        "velocity_radius_px_s": (
                            config.track_guided_reservation_velocity_radius_px_s
                        ),
                        "maximum_reservations_per_window": (
                            config.max_track_guided_reservations_per_window
                        ),
                    }
                }
                if reservation_enabled
                else {}
            ),
            timings_ms={"total": total_ms},
        )
