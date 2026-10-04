"""Deterministic constant-velocity Kalman tracking and confirmation."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from functools import lru_cache
import math
import time
from typing import Any, Mapping

import numpy as np

from ..detection import CandidateBatch, CandidateRecord, TrackPredictionHint
from .quadratic import mahalanobis_values as _mahalanobis_values


class TemporalTrackingError(RuntimeError):
    """Candidate batches violate temporal tracking assumptions."""


@lru_cache(maxsize=64)
def _mahalanobis_einsum_path(candidate_count: int, dimension: int):
    """Reuse NumPy's same greedy contraction order for identical operand shapes.

    No covariance, measurement or result is cached. Shape-only path planning was
    repeated for every track even though all tracks in a batch share shapes.
    """
    residual = np.empty((candidate_count, dimension), dtype=np.float64)
    covariance = np.empty((dimension, dimension), dtype=np.float64)
    return np.einsum_path("ni,ij,nj->n", residual, covariance, residual, optimize=True)[0]


def minimum_cost_pairs(edges):
    """Maximum-cardinality, minimum-cost gated one-to-one assignment.

    Edges are (track ID, candidate offset, cost). Dummy columns represent
    unmatched tracks. The finite penalty guarantees cardinality takes priority
    over cost; no absent/gate-rejected edge can be selected. Deterministic ties.
    This solves assignment, not physical identity or duplicate detections.
    """
    if not edges:
        return set()
    if any(not math.isfinite(e[2]) for e in edges):
        raise ValueError("Finite association costs required")
    tracks = sorted({e[0] for e in edges})
    candidates = sorted({e[1] for e in edges})
    ti = {v: i for i, v in enumerate(tracks)}
    ci = {v: i for i, v in enumerate(candidates)}
    n, real = len(tracks), len(candidates)
    lo, hi = min(e[2] for e in edges), max(e[2] for e in edges)
    penalty = (n + 1) * (hi - lo + 1)
    matrix = np.full((n, real + n), penalty * (n + 2), dtype=np.float64)
    matrix[:, real:] = penalty
    allowed = set()
    for t, c, value in edges:
        matrix[ti[t], ci[c]] = value - lo
        allowed.add((t, c))
    # Rectangular Hungarian shortest augmenting path; rows <= columns.
    m = matrix.shape[1]
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, predecessor = np.zeros(m + 1, int), np.zeros(m + 1, int)
    for row in range(1, n + 1):
        owner[0] = row
        j = 0
        distance = np.full(m + 1, np.inf)
        visited = np.zeros(m + 1, bool)
        while True:
            visited[j] = True
            i = owner[j]
            available = np.flatnonzero(~visited[1:]) + 1
            reduced = matrix[i - 1, available - 1] - u[i] - v[available]
            improve = reduced < distance[available]
            better = available[improve]
            distance[better] = reduced[improve]
            predecessor[better] = j
            next_j = available[np.argmin(distance[available])]
            delta = distance[next_j]
            u[owner[visited]] += delta
            v[visited] -= delta
            distance[~visited] -= delta
            j = next_j
            if owner[j] == 0:
                break
        while j:
            previous = predecessor[j]
            owner[j] = owner[previous]
            j = previous
    result = {
        (tracks[owner[j] - 1], candidates[j - 1])
        for j in range(1, real + 1)
        if owner[j]
    }
    if not result <= allowed:
        raise TemporalTrackingError("Assignment selected a forbidden edge")
    return result


@dataclass(frozen=True, slots=True)
class KalmanTrackingConfig:
    """Calibrated motion/noise gates and explicit lifecycle policy."""

    position_measurement_sigma_px: float | None = None
    velocity_measurement_sigma_px_s: float | None = None
    acceleration_process_sigma_px_s2: float | None = None
    initial_position_sigma_px: float | None = None
    initial_velocity_sigma_px_s: float | None = None
    mahalanobis_gate_squared: float | None = None
    maximum_position_residual_px: float | None = None
    maximum_velocity_residual_px_s: float | None = None
    confirmation_independent_hits: int | None = None
    max_missed_windows: int | None = None
    maximum_timestamp_gap_s: float | None = None
    measurement_noise_source: str | None = None
    max_active_tracks: int = 512
    evidence_policy: str = "non_overlapping_frames"
    measurement_model: str = "position_velocity"
    association_cost: str = "mahalanobis"
    association_cascade: str = "none"
    association_assignment: str = "greedy"
    association_prior: str = "none"
    association_appearance: str = "none"
    birth_policy: str = "input_order"
    birth_cell_size_px: float = 256.0

    def __post_init__(self) -> None:
        if self.association_appearance not in {"none", "log_response", "log_response_coast"}:
            raise ValueError("Unknown association_appearance")
        if self.association_appearance != "none" and (
            self.measurement_model != "position_only" or self.association_cost != "gaussian_nll"
        ):
            raise ValueError("Appearance experiment requires position-only Gaussian association")
        if self.association_prior not in {"none", "hit_maturity"}:
            raise ValueError("Unknown association_prior")
        if self.association_prior != "none" and (
            self.association_cost != "gaussian_nll" or self.association_cascade != "none"
        ):
            raise ValueError("Hit maturity requires Gaussian cost without a cascade")
        if self.association_assignment not in {"greedy", "global_min_cost"}:
            raise ValueError("Unknown association_assignment")
        if (
            self.association_assignment != "greedy"
            and self.association_cascade != "none"
        ):
            raise ValueError(
                "Global assignment cannot be combined with a priority cascade"
            )
        if self.association_cascade not in {"none", "confirmed_first"}:
            raise ValueError("Unknown association_cascade")
        if self.association_cost not in {"mahalanobis", "gaussian_nll"}:
            raise ValueError("Unknown association_cost")
        if self.birth_policy not in {"input_order", "spatial_fair"}:
            raise ValueError("Unknown birth_policy")
        if not math.isfinite(self.birth_cell_size_px) or self.birth_cell_size_px <= 0:
            raise ValueError("birth_cell_size_px must be finite and positive")
        if self.measurement_model not in {"position_velocity", "position_only"}:
            raise ValueError(
                "tracking.measurement_model must be position_velocity or position_only"
            )
        required = (
            "position_measurement_sigma_px",
            "velocity_measurement_sigma_px_s",
            "acceleration_process_sigma_px_s2",
            "initial_position_sigma_px",
            "initial_velocity_sigma_px_s",
            "mahalanobis_gate_squared",
            "maximum_position_residual_px",
            "maximum_velocity_residual_px_s",
            "confirmation_independent_hits",
            "max_missed_windows",
            "maximum_timestamp_gap_s",
            "measurement_noise_source",
        )
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Temporal tracking is not calibrated; set " + ", ".join(missing)
            )
        positive_floats = (
            "position_measurement_sigma_px",
            "velocity_measurement_sigma_px_s",
            "acceleration_process_sigma_px_s2",
            "initial_position_sigma_px",
            "initial_velocity_sigma_px_s",
            "mahalanobis_gate_squared",
            "maximum_position_residual_px",
            "maximum_velocity_residual_px_s",
            "maximum_timestamp_gap_s",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"tracking.{name} must be finite and positive")
        integer_fields = (
            "confirmation_independent_hits",
            "max_missed_windows",
            "max_active_tracks",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"tracking.{name} must be an integer")
        assert self.confirmation_independent_hits is not None
        assert self.max_missed_windows is not None
        if self.confirmation_independent_hits < 2:
            raise ValueError(
                "tracking.confirmation_independent_hits must be at least 2"
            )
        if self.max_missed_windows < 0:
            raise ValueError("tracking.max_missed_windows cannot be negative")
        if self.max_active_tracks <= 0:
            raise ValueError("tracking.max_active_tracks must be positive")
        if self.evidence_policy != "non_overlapping_frames":
            raise ValueError(
                "tracking.evidence_policy must currently be non_overlapping_frames"
            )
        if self.measurement_noise_source not in {
            "provisional",
            "synthetic_characterization",
            "empirical_camera_calibration",
        }:
            raise ValueError(
                "tracking.measurement_noise_source must be provisional, "
                "synthetic_characterization, or empirical_camera_calibration"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "KalmanTrackingConfig":
        data = dict(value or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown temporal tracking configuration keys: {unknown}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TrackRecord:
    track_id: int
    lifecycle_state: str
    state_xy_vx_vy: tuple[float, float, float, float]
    covariance: tuple[tuple[float, float, float, float], ...]
    birth_timestamp_ns: int
    state_timestamp_ns: int
    last_measurement_timestamp_ns: int
    age_windows: int
    associated_update_count: int
    independent_confirmation_hits: int
    confirmation_required_hits: int
    missed_windows: int
    confirmation_timestamp_ns: int | None
    confirmation_latency_s: float | None
    last_detector_score_snr: float
    last_candidate_index: int
    last_credited_frame_indices: tuple[int, ...]
    quality_evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "lifecycle_state": self.lifecycle_state,
            "state": {
                "coordinate_order": ["x_px", "y_px", "vx_px_s", "vy_px_s"],
                "mean": list(self.state_xy_vx_vy),
                "covariance": [list(row) for row in self.covariance],
                "timestamp_ns": self.state_timestamp_ns,
                "model": "constant_velocity",
            },
            "birth_timestamp_ns": self.birth_timestamp_ns,
            "last_measurement_timestamp_ns": self.last_measurement_timestamp_ns,
            "age_windows": self.age_windows,
            "associated_update_count": self.associated_update_count,
            "confirmation": {
                "evidence_policy": "non_overlapping_frames",
                "independent_hits": self.independent_confirmation_hits,
                "required_hits": self.confirmation_required_hits,
                "progress_fraction": min(
                    1.0,
                    self.independent_confirmation_hits
                    / self.confirmation_required_hits,
                ),
                "timestamp_ns": self.confirmation_timestamp_ns,
                "latency_s": self.confirmation_latency_s,
                "last_credited_frame_indices": list(self.last_credited_frame_indices),
            },
            "missed_windows": self.missed_windows,
            "detector_evidence": {
                "last_normalized_score_snr": self.last_detector_score_snr,
                "last_candidate_index": self.last_candidate_index,
                "is_track_confidence": False,
            },
            "quality_evidence": self.quality_evidence,
        }


@dataclass(frozen=True, slots=True)
class TrackBatch:
    tracks: tuple[TrackRecord, ...]
    frame_indices: tuple[int, ...]
    reference_timestamp_ns: int
    segment_index: int
    associations: tuple[dict[str, Any], ...]
    born_track_ids: tuple[int, ...]
    deleted_tracks: tuple[dict[str, Any], ...]
    reset_reason: str | None
    metrics: dict[str, Any]
    timings_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_indices": list(self.frame_indices),
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "segment_index": self.segment_index,
            "tracks": [track.to_dict() for track in self.tracks],
            "associations": list(self.associations),
            "born_track_ids": list(self.born_track_ids),
            "deleted_tracks": list(self.deleted_tracks),
            "reset_reason": self.reset_reason,
            "metrics": self.metrics,
            "timings_ms": self.timings_ms,
        }


@dataclass(slots=True)
class _RunningMoments:
    """Fixed-memory scalar telemetry with JSON-safe population moments."""

    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        number = float(value)
        if not math.isfinite(number):
            return
        self.count += 1
        self.total += number
        self.total_squared += number * number
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "minimum": None,
                "mean": None,
                "rms": None,
                "standard_deviation": None,
                "maximum": None,
            }
        mean = self.total / self.count
        mean_square = self.total_squared / self.count
        return {
            "count": self.count,
            "minimum": self.minimum,
            "mean": mean,
            "rms": math.sqrt(max(0.0, mean_square)),
            "standard_deviation": math.sqrt(max(0.0, mean_square - mean * mean)),
            "maximum": self.maximum,
        }


@dataclass(slots=True)
class _MutableTrack:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    birth_timestamp_ns: int
    state_timestamp_ns: int
    last_measurement_timestamp_ns: int
    age_windows: int
    associated_update_count: int
    independent_confirmation_hits: int
    missed_windows: int
    lifecycle_state: str
    confirmation_timestamp_ns: int | None
    last_detector_score_snr: float
    last_candidate_index: int
    last_credited_frame_indices: tuple[int, ...] = field(default_factory=tuple)
    last_measurement_velocity_xy_px_s: tuple[float, float] | None = None
    last_raw_response: float | None = None
    selection_score_units: str | None = None
    detector_score_snr: _RunningMoments = field(default_factory=_RunningMoments)
    selection_score: _RunningMoments = field(default_factory=_RunningMoments)
    peak_contrast_snr: _RunningMoments = field(default_factory=_RunningMoments)
    peak_to_neighbor_ratio: _RunningMoments = field(default_factory=_RunningMoments)
    supporting_frame_count: _RunningMoments = field(default_factory=_RunningMoments)
    support_weight: _RunningMoments = field(default_factory=_RunningMoments)
    measurement_speed_px_s: _RunningMoments = field(default_factory=_RunningMoments)
    measurement_velocity_step_px_s: _RunningMoments = field(
        default_factory=_RunningMoments
    )
    mahalanobis_distance_squared: _RunningMoments = field(
        default_factory=_RunningMoments
    )
    position_residual_px: _RunningMoments = field(default_factory=_RunningMoments)
    velocity_residual_px_s: _RunningMoments = field(default_factory=_RunningMoments)


def _measurement(candidate: CandidateRecord) -> np.ndarray:
    return np.array(
        [
            candidate.x_px,
            candidate.y_px,
            candidate.velocity_xy_px_s[0],
            candidate.velocity_xy_px_s[1],
        ],
        np.float64,
    )


class KalmanTrackManager:
    """Bounded multi-target tracker with deterministic greedy association."""

    def __init__(self, config: KalmanTrackingConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, KalmanTrackingConfig)
            else KalmanTrackingConfig.from_mapping(config)
        )
        self._tracks: dict[int, _MutableTrack] = {}
        self._next_track_id = 0
        self._last_timestamp_ns: int | None = None
        self._segment_index: int | None = None

    def _measurement_covariance(self) -> np.ndarray:
        assert self.config.position_measurement_sigma_px is not None
        assert self.config.velocity_measurement_sigma_px_s is not None
        if self.config.measurement_model == "position_only":
            return (
                np.eye(2, dtype=np.float64)
                * self.config.position_measurement_sigma_px ** 2
            )
        return np.diag(
            [
                self.config.position_measurement_sigma_px ** 2,
                self.config.position_measurement_sigma_px ** 2,
                self.config.velocity_measurement_sigma_px_s ** 2,
                self.config.velocity_measurement_sigma_px_s ** 2,
            ]
        ).astype(np.float64)

    def _initial_covariance(self) -> np.ndarray:
        assert self.config.initial_position_sigma_px is not None
        assert self.config.initial_velocity_sigma_px_s is not None
        return np.diag(
            [
                self.config.initial_position_sigma_px ** 2,
                self.config.initial_position_sigma_px ** 2,
                self.config.initial_velocity_sigma_px_s ** 2,
                self.config.initial_velocity_sigma_px_s ** 2,
            ]
        ).astype(np.float64)

    def _predicted_state(
        self, track: _MutableTrack, timestamp_ns: int, *, cache: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        delta_ns = timestamp_ns - track.state_timestamp_ns
        dt = delta_ns / 1e9
        if dt < 0:
            raise TemporalTrackingError("track prediction timestamp moved backward")
        terms_key = ("terms", delta_ns)
        terms = None if cache is None else cache.get(terms_key)
        if terms is None:
            transition = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64,
            )
            assert self.config.acceleration_process_sigma_px_s2 is not None
            variance = self.config.acceleration_process_sigma_px_s2 ** 2
            process = variance * np.array(
                [
                    [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                    [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                    [dt ** 3 / 2, 0, dt ** 2, 0],
                    [0, dt ** 3 / 2, 0, dt ** 2],
                ],
                np.float64,
            )
            terms = transition, process
            if cache is not None:
                cache[terms_key] = terms
        transition, process = terms
        mean = transition @ track.mean
        covariance_key = ("covariance", delta_ns, track.covariance.tobytes())
        covariance = None if cache is None else cache.get(covariance_key)
        if covariance is None:
            covariance = transition @ track.covariance @ transition.T + process
            covariance = 0.5 * (covariance + covariance.T)
            if cache is not None:
                cache[covariance_key] = covariance
        if cache is not None:
            covariance = covariance.copy()  # Mutable per-track state must not alias.
        return mean, covariance

    def _predict(self, track: _MutableTrack, timestamp_ns: int, *, cache: dict | None = None) -> None:
        track.mean, track.covariance = self._predicted_state(track, timestamp_ns, cache=cache)
        track.state_timestamp_ns = timestamp_ns
        track.age_windows += 1

    def reservation_hints(
        self, *, reference_timestamp_ns: int, segment_index: int
    ) -> tuple[TrackPredictionHint, ...]:
        """Return prior-only predictions without advancing or otherwise mutating tracks."""

        if not self._tracks or self._segment_index != segment_index:
            return ()
        if (
            self._last_timestamp_ns is None
            or reference_timestamp_ns <= self._last_timestamp_ns
        ):
            return ()
        assert self.config.maximum_timestamp_gap_s is not None
        if (
            reference_timestamp_ns - self._last_timestamp_ns
        ) / 1e9 > self.config.maximum_timestamp_gap_s:
            return ()
        hints = []
        for track_id, track in sorted(self._tracks.items()):
            mean, _ = self._predicted_state(track, reference_timestamp_ns)
            hints.append(
                TrackPredictionHint(
                    track_id=track_id,
                    position_xy_px=(float(mean[0]), float(mean[1])),
                    velocity_xy_px_s=(float(mean[2]), float(mean[3])),
                    lifecycle_state=track.lifecycle_state,
                    age_windows=track.age_windows,
                    associated_update_count=track.associated_update_count,
                    independent_confirmation_hits=(track.independent_confirmation_hits),
                    missed_windows=track.missed_windows,
                    mean_measurement_speed_px_s=(
                        track.measurement_speed_px_s.total
                        / track.measurement_speed_px_s.count
                        if track.measurement_speed_px_s.count
                        else math.hypot(*mean[2:])
                    ),
                )
            )
        return tuple(hints)

    def _record(self, track: _MutableTrack, *, include_quality_evidence=True) -> TrackRecord:
        latency = (
            (track.confirmation_timestamp_ns - track.birth_timestamp_ns) / 1e9
            if track.confirmation_timestamp_ns is not None
            else None
        )
        return TrackRecord(
            track_id=track.track_id,
            lifecycle_state=track.lifecycle_state,
            state_xy_vx_vy=tuple(float(value) for value in track.mean),
            covariance=tuple(
                tuple(float(value) for value in row) for row in track.covariance
            ),
            birth_timestamp_ns=track.birth_timestamp_ns,
            state_timestamp_ns=track.state_timestamp_ns,
            last_measurement_timestamp_ns=track.last_measurement_timestamp_ns,
            age_windows=track.age_windows,
            associated_update_count=track.associated_update_count,
            independent_confirmation_hits=track.independent_confirmation_hits,
            confirmation_required_hits=int(self.config.confirmation_independent_hits),
            missed_windows=track.missed_windows,
            confirmation_timestamp_ns=track.confirmation_timestamp_ns,
            confirmation_latency_s=latency,
            last_detector_score_snr=track.last_detector_score_snr,
            last_candidate_index=track.last_candidate_index,
            last_credited_frame_indices=track.last_credited_frame_indices,
            quality_evidence={
                **self._quality_evidence(track),
                "measurement_model": self.config.measurement_model,
            } if include_quality_evidence else {"omitted": True, "reason": "caller_does_not_consume_quality_summary"},
        )

    @staticmethod
    def _quality_evidence(track: _MutableTrack) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "aggregation": "fixed_memory_running_population_moments",
            "used_for_association_or_confirmation": False,
            "observation_count": track.detector_score_snr.count,
            "observation_fraction_of_age_windows": (
                track.detector_score_snr.count / track.age_windows
                if track.age_windows
                else None
            ),
            "detector_score_snr": track.detector_score_snr.summary(),
            "selection_score": {
                "units": track.selection_score_units,
                **track.selection_score.summary(),
            },
            "peak_contrast_snr": track.peak_contrast_snr.summary(),
            "peak_to_neighbor_ratio": track.peak_to_neighbor_ratio.summary(),
            "supporting_frame_count": track.supporting_frame_count.summary(),
            "support_weight": track.support_weight.summary(),
            "measurement_speed_px_s": track.measurement_speed_px_s.summary(),
            "measurement_velocity_step_px_s": (
                track.measurement_velocity_step_px_s.summary()
            ),
            "kalman_innovation": {
                "association_count_excluding_birth": (
                    track.mahalanobis_distance_squared.count
                ),
                "mahalanobis_distance_squared": (
                    track.mahalanobis_distance_squared.summary()
                ),
                "position_residual_px": track.position_residual_px.summary(),
                "velocity_residual_px_s": track.velocity_residual_px_s.summary(),
            },
        }

    def _observe_candidate(
        self,
        track: _MutableTrack,
        candidate: CandidateRecord,
        *,
        mahalanobis_distance_squared: float | None = None,
        position_residual_px: float | None = None,
        velocity_residual_px_s: float | None = None,
    ) -> None:
        units = candidate.ranking_score_units
        if (
            track.selection_score_units is not None
            and units != track.selection_score_units
        ):
            raise TemporalTrackingError(
                "candidate ranking-score units changed within one track"
            )
        track.selection_score_units = units
        if self.config.association_appearance != "none":
            track.last_raw_response = float(candidate.raw_sum_score)
        velocity = tuple(float(value) for value in candidate.velocity_xy_px_s)
        if (
            self.config.measurement_model == "position_velocity"
            and track.last_measurement_velocity_xy_px_s is not None
        ):
            track.measurement_velocity_step_px_s.add(
                math.dist(velocity, track.last_measurement_velocity_xy_px_s)
            )
        if self.config.measurement_model == "position_velocity":
            track.last_measurement_velocity_xy_px_s = velocity
        track.detector_score_snr.add(candidate.normalized_score_snr)
        track.selection_score.add(
            candidate.normalized_score_snr
            if candidate.ranking_score is None
            else candidate.ranking_score
        )
        track.peak_contrast_snr.add(candidate.peak_contrast_snr)
        track.peak_to_neighbor_ratio.add(candidate.peak_to_neighbor_ratio)
        track.supporting_frame_count.add(candidate.supporting_frame_count)
        track.support_weight.add(candidate.support_weight)
        if self.config.measurement_model == "position_velocity":
            track.measurement_speed_px_s.add(math.hypot(*velocity))
        track.mahalanobis_distance_squared.add(mahalanobis_distance_squared)
        track.position_residual_px.add(position_residual_px)
        track.velocity_residual_px_s.add(velocity_residual_px_s)
        track.last_detector_score_snr = candidate.normalized_score_snr
        track.last_candidate_index = candidate.candidate_index

    def _new_track(
        self, candidate: CandidateRecord, batch: CandidateBatch
    ) -> _MutableTrack:
        initial_mean = _measurement(candidate)
        if self.config.measurement_model == "position_only":
            # Unknown velocity is a broad prior, NOT a measured zero velocity.
            initial_mean[2:] = 0
        track = _MutableTrack(
            track_id=self._next_track_id,
            mean=initial_mean,
            covariance=self._initial_covariance(),
            birth_timestamp_ns=batch.reference_timestamp_ns,
            state_timestamp_ns=batch.reference_timestamp_ns,
            last_measurement_timestamp_ns=batch.reference_timestamp_ns,
            age_windows=1,
            associated_update_count=1,
            independent_confirmation_hits=1,
            missed_windows=0,
            lifecycle_state="tentative",
            confirmation_timestamp_ns=None,
            last_detector_score_snr=candidate.normalized_score_snr,
            last_candidate_index=candidate.candidate_index,
            last_credited_frame_indices=batch.frame_indices,
        )
        self._observe_candidate(track, candidate)
        self._next_track_id += 1
        return track

    def _reset_reason(self, batch: CandidateBatch) -> str | None:
        if (
            self._segment_index is not None
            and batch.segment_index != self._segment_index
        ):
            return "coordinate_segment_changed"
        if self._last_timestamp_ns is None:
            return None
        if batch.reference_timestamp_ns <= self._last_timestamp_ns:
            return "reference_timestamp_not_strictly_increasing"
        assert self.config.maximum_timestamp_gap_s is not None
        if (
            batch.reference_timestamp_ns - self._last_timestamp_ns
        ) / 1e9 > self.config.maximum_timestamp_gap_s:
            return "reference_timestamp_gap"
        return None

    def _admit_births(self, batch, unmatched, assigned_tracks, deleted):
        """Least-occupied reference-grid cells first, never evict confirmed tracks.

        At capacity only an unobserved tentative track in a more populated cell
        can be replaced. Newborns/current measurements are protected this frame.
        Scores order proposals WITHIN a cell, not globally across the image.
        """
        slots = max(0, self.config.max_active_tracks - len(self._tracks))
        if self.config.birth_policy == "input_order":
            admitted = unmatched[:slots]
            born = []
            for index in admitted:
                track = self._new_track(batch.candidates[index], batch)
                self._tracks[track.track_id] = track
                born.append(track.track_id)
            return born, {"policy": "input_order"}

        size = self.config.birth_cell_size_px

        def cell(x, y):
            return (math.floor(x / size), math.floor(y / size))

        track_cells = {i: cell(*t.mean[:2]) for i, t in self._tracks.items()}
        occupancy = Counter(track_cells.values())
        queues = {}
        for index in unmatched:
            c = batch.candidates[index]
            queues.setdefault(cell(c.x_px, c.y_px), []).append(index)
        for indices in queues.values():
            indices.sort(
                key=lambda i: (
                    -batch.candidates[i].normalized_score_snr,
                    batch.candidates[i].y_px,
                    batch.candidates[i].x_px,
                    i,
                )
            )
        cells = sorted(queues)
        # Deterministic rotating tie-break: no permanently preferred raster corner.
        offset = batch.frame_indices[-1] % len(cells) if cells else 0
        rank = {c: i for i, c in enumerate(cells[offset:] + cells[:offset])}
        replaceable = {
            i
            for i, t in self._tracks.items()
            if i not in assigned_tracks
            and t.confirmation_timestamp_ns is None
            and t.missed_windows > 0
        }
        born, rejected, evictions = [], [], []
        while queues:
            chosen = min(queues, key=lambda c: (occupancy[c], rank[c]))
            index = queues[chosen].pop(0)
            if not queues[chosen]:
                del queues[chosen]
            if len(self._tracks) >= self.config.max_active_tracks:
                eligible = [
                    i
                    for i in replaceable
                    if occupancy[track_cells[i]] > occupancy[chosen] + 1
                ]
                if not eligible:
                    rejected.append(batch.candidates[index].candidate_index)
                    continue
                victim = max(
                    eligible,
                    key=lambda i: (
                        occupancy[track_cells[i]],
                        self._tracks[i].missed_windows,
                        -self._tracks[i].independent_confirmation_hits,
                        -i,
                    ),
                )
                previous = self._tracks.pop(victim)
                occupancy[track_cells[victim]] -= 1
                replaceable.remove(victim)
                record = dict(
                    track_id=victim,
                    previous_state=previous.lifecycle_state,
                    reason="spatial_capacity_tentative_replacement",
                    replacement_candidate_index=batch.candidates[index].candidate_index,
                )
                deleted.append(record)
                evictions.append(record)
            track = self._new_track(batch.candidates[index], batch)
            self._tracks[track.track_id] = track
            occupancy[chosen] += 1
            born.append(track.track_id)
        return (
            born,
            dict(
                policy="spatial_fair",
                cell_size_px=size,
                rejected_candidate_indices=rejected,
                tentative_replacements=evictions,
                confirmed_tracks_evicted=0,
            ),
        )

    def update(self, batch: CandidateBatch, *, include_quality_evidence: bool = True) -> TrackBatch:
        if not isinstance(include_quality_evidence, bool):
            raise ValueError("include_quality_evidence must be boolean")
        started = time.perf_counter_ns()
        if not batch.frame_indices:
            raise TemporalTrackingError("candidate batch must contain frame indices")
        if tuple(sorted(set(batch.frame_indices))) != batch.frame_indices:
            raise TemporalTrackingError(
                "candidate batch frame indices must be strictly increasing"
            )
        reset_reason = self._reset_reason(batch)
        deleted: list[dict[str, Any]] = []
        if reset_reason is not None:
            deleted.extend(
                {
                    "track_id": track_id,
                    "previous_state": track.lifecycle_state,
                    "reason": reset_reason,
                }
                for track_id, track in sorted(self._tracks.items())
            )
            self._tracks.clear()
        self._segment_index = batch.segment_index
        self._last_timestamp_ns = batch.reference_timestamp_ns

        # All caches are local to one update with fixed model/noise settings.
        # Only byte-identical covariance and exact integer time deltas share
        # results; no binning, rounding, age-based approximation or fast math.
        prediction_cache = {}
        for track in self._tracks.values():
            self._predict(track, batch.reference_timestamp_ns, cache=prediction_cache)

        measurement_covariance = self._measurement_covariance()
        dimension = measurement_covariance.shape[0]
        observation_matrix = np.eye(4, dtype=np.float64)[:dimension]
        identity = np.eye(4, dtype=np.float64)
        association_options: list[tuple[float, int, int, float, float | None]] = []
        rejected_by_position = 0
        rejected_by_velocity = 0
        rejected_by_mahalanobis = 0
        candidate_measurements = np.array(
            [_measurement(candidate) for candidate in batch.candidates],
            dtype=np.float64,
        ).reshape((-1, 4))
        innovation_cache = {}
        log_volumes = {}
        assert self.config.maximum_position_residual_px is not None
        assert self.config.maximum_velocity_residual_px_s is not None
        assert self.config.mahalanobis_gate_squared is not None
        for track_id, track in sorted(self._tracks.items()):
            innovation_covariance = (
                track.covariance[:dimension, :dimension] + measurement_covariance
            )
            # Tracks with identical observation/miss histories often have
            # byte-identical covariance. Reuse only within this update, keyed
            # by the full covariance, never by age or an approximation.
            covariance_key = innovation_covariance.tobytes()
            cached = innovation_cache.get(covariance_key)
            if cached is None:
                try:
                    inverse_innovation = np.linalg.inv(innovation_covariance)
                except np.linalg.LinAlgError as exc:
                    raise TemporalTrackingError(
                        "Kalman innovation covariance is singular"
                    ) from exc
                log_volume = (float(np.linalg.slogdet(innovation_covariance)[1])
                              if self.config.association_cost == "gaussian_nll" else None)
                cached = inverse_innovation, log_volume
                innovation_cache[covariance_key] = cached
            inverse_innovation, log_volume = cached
            if log_volume is not None:
                log_volumes[track_id] = log_volume
            residuals = candidate_measurements[:, :dimension] - track.mean[:dimension]
            position_residuals = np.sqrt(np.sum(residuals[:, :2] ** 2, axis=1))
            velocity_residuals = np.sqrt(np.sum(residuals[:, 2:] ** 2, axis=1))
            position_pass = (
                position_residuals <= self.config.maximum_position_residual_px
            )
            velocity_pass = (
                velocity_residuals <= self.config.maximum_velocity_residual_px_s
            )
            rejected_by_position += int(np.count_nonzero(~position_pass))
            rejected_by_velocity += int(
                np.count_nonzero(position_pass & ~velocity_pass)
            )
            if not np.any(position_pass & velocity_pass):
                # No pair can survive the existing geometric gates. Covariance
                # validation above still happens; skip only unused likelihoods.
                continue
            mahalanobis_values = _mahalanobis_values(residuals, inverse_innovation)
            mahalanobis_pass = (
                mahalanobis_values <= self.config.mahalanobis_gate_squared
            )
            rejected_by_mahalanobis += int(
                np.count_nonzero(position_pass & velocity_pass & ~mahalanobis_pass)
            )
            accepted = np.flatnonzero(position_pass & velocity_pass & mahalanobis_pass)
            for candidate_index in accepted:
                association_options.append(
                    (
                        float(mahalanobis_values[candidate_index]),
                        track_id,
                        int(candidate_index),
                        float(position_residuals[candidate_index]),
                        float(velocity_residuals[candidate_index])
                        if dimension == 4
                        else None,
                    )
                )
        # Gaussian likelihood includes uncertainty volume: a diffuse prediction
        # must not win solely because its Mahalanobis distance is artificially low.
        # Opt-in, causal association regularizer, NOT calibrated existence odds.
        # Independent observations reduce the finite maturity penalty; missed
        # windows remove stale-track preference. No gates or measurements change.
        prior_penalties = {
            i: 2 * math.log1p(
                self.config.confirmation_independent_hits / t.independent_confirmation_hits
            ) + 2 * math.log1p(t.missed_windows)
            for i, t in self._tracks.items()
        } if self.config.association_prior == "hit_maturity" else {}

        appearance_penalties = {}
        appearance_weights = {}
        if self.config.association_appearance != "none":
            for option in association_options:
                t = self._tracks[option[1]]
                current = abs(float(batch.candidates[option[2]].raw_sum_score))
                previous = abs(t.last_raw_response or 0.0)
                # Broad log-amplitude consistency, not a preference for brightness.
                # No gate changes; stale/unconfirmed/missing evidence contributes zero.
                penalty = 0.0
                weight = 0.0
                # A short disappearance does not erase the last observed
                # appearance. This opt-in mode retains soft, progressively
                # discounted evidence only within the existing coast budget.
                # No position gate, confirmation count or measurement changes.
                coast_memory = self.config.association_appearance == "log_response_coast"
                appearance_available = (not t.missed_windows or (
                    coast_memory and t.missed_windows <= self.config.max_missed_windows))
                if (appearance_available and t.independent_confirmation_hits >= self.config.confirmation_independent_hits
                        and current > 0 and previous > 0 and math.isfinite(current) and math.isfinite(previous)):
                    weight = (1.0 / (1.0 + t.missed_windows / (self.config.max_missed_windows + 1))
                              if coast_memory else 1.0)
                    penalty = weight * (math.log(current) - math.log(previous)) ** 2
                appearance_penalties[option[1], option[2]] = penalty
                if coast_memory:
                    appearance_weights[option[1], option[2]] = weight

        def cost(option):
            return (option[0] + log_volumes.get(option[1], 0.0)
                    + prior_penalties.get(option[1], 0.0)
                    + appearance_penalties.get((option[1], option[2]), 0.0))

        def association_order(item):
            # Opt-in experiment: tentative competitors cannot claim a gated
            # observation before an established track. Gates and likelihood
            # ranking within each group are unchanged. This is not confidence
            # in physical identity and can be wrong at a close crossing.
            tentative = (
                self.config.association_cascade == "confirmed_first"
                and self._tracks[item[1]].confirmation_timestamp_ns is None
            )
            return (tentative, cost(item), item[1], item[2])

        association_options.sort(key=association_order)
        selected_pairs = (
            minimum_cost_pairs([(o[1], o[2], cost(o)) for o in association_options])
            if self.config.association_assignment == "global_min_cost"
            else None
        )
        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()
        associations: list[dict[str, Any]] = []
        correction_cache = {}
        for (
            mahalanobis,
            track_id,
            candidate_index,
            position_residual,
            velocity_residual,
        ) in association_options:
            if (
                selected_pairs is not None
                and (track_id, candidate_index) not in selected_pairs
            ):
                continue
            if track_id in assigned_tracks or candidate_index in assigned_candidates:
                continue
            track = self._tracks[track_id]
            candidate = batch.candidates[candidate_index]
            measurement = _measurement(candidate)[:dimension]
            innovation = measurement - track.mean[:dimension]
            correction_key = track.covariance.tobytes()
            correction = correction_cache.get(correction_key)
            if correction is None:
                innovation_covariance = (
                    track.covariance[:dimension, :dimension] + measurement_covariance
                )
                gain = np.linalg.solve(
                    innovation_covariance.T, track.covariance[:, :dimension].T
                ).T
                residual_transform = identity - gain @ observation_matrix
                covariance = (
                    residual_transform @ track.covariance @ residual_transform.T
                    + gain @ measurement_covariance @ gain.T
                )
                covariance = 0.5 * (covariance + covariance.T)
                correction = gain, covariance
                correction_cache[correction_key] = correction
            gain, covariance = correction
            track.mean = track.mean + gain @ innovation
            track.covariance = covariance.copy()
            track.last_measurement_timestamp_ns = batch.reference_timestamp_ns
            track.associated_update_count += 1
            track.missed_windows = 0
            evidence_credited = min(batch.frame_indices) > max(
                track.last_credited_frame_indices
            )
            if evidence_credited:
                track.independent_confirmation_hits += 1
                track.last_credited_frame_indices = batch.frame_indices
            previous_state = track.lifecycle_state
            if track.independent_confirmation_hits >= int(
                self.config.confirmation_independent_hits
            ):
                track.lifecycle_state = "confirmed"
                if track.confirmation_timestamp_ns is None:
                    track.confirmation_timestamp_ns = batch.reference_timestamp_ns
            else:
                track.lifecycle_state = "tentative"
            self._observe_candidate(
                track,
                candidate,
                mahalanobis_distance_squared=mahalanobis,
                position_residual_px=position_residual,
                velocity_residual_px_s=velocity_residual,
            )
            assigned_tracks.add(track_id)
            assigned_candidates.add(candidate_index)
            associations.append(
                {
                    "track_id": track_id,
                    "candidate_index": candidate.candidate_index,
                    "mahalanobis_distance_squared": mahalanobis,
                    "position_residual_px": position_residual,
                    "velocity_residual_px_s": velocity_residual,
                    "independent_confirmation_evidence_credited": evidence_credited,
                    "previous_lifecycle_state": previous_state,
                    "new_lifecycle_state": track.lifecycle_state,
                }
            )

        assert self.config.max_missed_windows is not None
        for track_id, track in list(sorted(self._tracks.items())):
            if track_id in assigned_tracks:
                continue
            previous_state = track.lifecycle_state
            track.missed_windows += 1
            if previous_state in {"confirmed", "coasted"}:
                track.lifecycle_state = "coasted"
            if track.missed_windows > self.config.max_missed_windows:
                track.lifecycle_state = "deleted"
                deleted.append(
                    {
                        "track_id": track_id,
                        "previous_state": previous_state,
                        "reason": "maximum_missed_windows_exceeded",
                    }
                )
                del self._tracks[track_id]

        unmatched_candidates = [
            index
            for index in range(len(batch.candidates))
            if index not in assigned_candidates
        ]
        born, admission = self._admit_births(
            batch, unmatched_candidates, assigned_tracks, deleted
        )
        dropped_birth_count = len(unmatched_candidates) - len(born)
        records = tuple(
            self._record(track, include_quality_evidence=include_quality_evidence)
            for _, track in sorted(self._tracks.items())
        )
        # Audit close likelihood alternatives, including competition for either
        # track or observation. This is diagnostic, NOT calibrated confidence or
        # an assertion that a greedy assignment resolves a crossing/occlusion.
        association_audit = []
        if self.config.association_cost == "gaussian_nll":
            by_track, by_candidate = {}, {}
            for option in association_options:
                by_track.setdefault(option[1], []).append(option)
                by_candidate.setdefault(option[2], []).append(option)
            candidate_offsets = {
                c.candidate_index: i for i, c in enumerate(batch.candidates)
            }
            for association in associations:
                tid = association["track_id"]
                ci = candidate_offsets[association["candidate_index"]]
                chosen_cost = (
                    association["mahalanobis_distance_squared"] + log_volumes[tid]
                )
                alternatives = [
                    o
                    for o in by_track[tid] + by_candidate[ci]
                    if (o[1], o[2]) != (tid, ci)
                ]
                # Keep the existing likelihood diagnostic likelihood-only.
                alternate = min(
                    alternatives, key=lambda o: o[0] + log_volumes[o[1]]
                ) if alternatives else None
                gap = (alternate[0] + log_volumes[alternate[1]] - chosen_cost
                       if alternate else None)
                association_audit.append(
                    dict(
                        track_id=tid,
                        candidate_index=association["candidate_index"],
                        gaussian_twice_nll_without_constant=chosen_cost,
                        alternate_track_id=alternate[1] if alternate else None,
                        alternate_candidate_index=batch.candidates[
                            alternate[2]
                        ].candidate_index
                        if alternate
                        else None,
                        alternate_minus_selected_cost=gap,
                        competing_alternative_within_likelihood_factor_three=(
                            gap is not None and gap <= 2 * math.log(3)
                        ),
                        **({"hit_maturity_penalty": prior_penalties[tid],
                            "regularized_assignment_cost": chosen_cost + prior_penalties[tid] + appearance_penalties.get((tid, ci), 0.0),
                            "prior_is_calibrated_probability": False}
                           if prior_penalties else {}),
                        **({"log_response_penalty": appearance_penalties[tid, ci],
                            "appearance_is_calibrated_probability": False}
                           if appearance_penalties else {}),
                        **({"appearance_coast_weight": appearance_weights[tid, ci]}
                           if appearance_weights else {}),
                    )
                )
        status_counts = {
            state: sum(record.lifecycle_state == state for record in records)
            for state in ("tentative", "confirmed", "coasted")
        }
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        return TrackBatch(
            tracks=records,
            frame_indices=batch.frame_indices,
            reference_timestamp_ns=batch.reference_timestamp_ns,
            segment_index=batch.segment_index,
            associations=tuple(associations),
            born_track_ids=tuple(born),
            deleted_tracks=tuple(deleted),
            reset_reason=reset_reason,
            metrics={
                "input_candidate_count": len(batch.candidates),
                "association_option_count": len(association_options),
                "associated_candidate_count": len(assigned_candidates),
                "unmatched_candidate_count": len(unmatched_candidates),
                "birth_count": len(born),
                "dropped_birth_count_at_active_track_cap": dropped_birth_count,
                "deleted_track_count": len(deleted),
                "active_track_count": len(records),
                "lifecycle_counts": status_counts,
                "rejected_pair_counts": {
                    "position_gate": rejected_by_position,
                    "velocity_gate": rejected_by_velocity,
                    "mahalanobis_gate": rejected_by_mahalanobis,
                },
                "measurement_noise_policy": "configured_fixed_diagonal_sigma",
                "measurement_noise_source": self.config.measurement_noise_source,
                "measurement_model": self.config.measurement_model,
                "detector_score_used_as_track_confidence": False,
                "confirmation_evidence_policy": self.config.evidence_policy,
                "max_active_tracks": self.config.max_active_tracks,
                "birth_admission": admission,
                "association_cost": self.config.association_cost,
                "association_assignment": self.config.association_assignment,
                **({"association_prior": self.config.association_prior}
                   if self.config.association_prior != "none" else {}),
                "association_audit": association_audit,
            },
            timings_ms={"total": total_ms},
        )
