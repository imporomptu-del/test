"""Deterministic RANSAC camera-motion fitting and transform composition."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Any, Mapping

import numpy as np

from .geometry import grid_coverage
from .types import MotionCorrespondences


_FLOAT64_EPSILON = np.finfo(np.float64).eps


class GlobalMotionError(RuntimeError):
    """Global motion cannot be estimated or composed as requested."""


@dataclass(frozen=True, slots=True)
class GlobalMotionConfig:
    model: str = "translation"
    ransac_iterations: int = 500
    ransac_reprojection_px: float = 1.0
    random_seed: int = 75
    minimum_correspondences: int = 30
    minimum_inliers: int = 30
    minimum_inlier_ratio: float = 0.60
    grid_rows: int = 6
    grid_cols: int = 8
    minimum_inlier_grid_coverage: float = 0.25
    maximum_median_reprojection_px: float = 0.35
    maximum_p90_reprojection_px: float = 0.75
    maximum_reprojection_px: float = 2.0
    maximum_translation_px: float = 120.0
    maximum_rotation_deg: float = 2.0
    minimum_scale: float = 0.98
    maximum_scale: float = 1.02
    minimum_noncollinearity_ratio: float = 0.01
    minimum_sample_separation_px: float = 32.0
    failure_policy: str = "reset_reference"
    maximum_reuse_pairs: int = 1
    coverage_policy: str = "full_grid"
    sparse_minimum_cells: int = 4
    sparse_minimum_points_per_cell: int = 4
    sparse_minimum_span_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.model not in {"translation", "similarity"}:
            raise ValueError("global_motion.model must be translation or similarity")
        if self.coverage_policy not in {"full_grid", "translation_consensus"}:
            raise ValueError("Unknown global_motion.coverage_policy")
        if (
            self.coverage_policy == "translation_consensus"
            and self.model != "translation"
        ):
            raise ValueError("translation_consensus requires the translation model")
        if not 0 < self.sparse_minimum_span_fraction <= 1:
            raise ValueError("sparse_minimum_span_fraction must be in (0, 1]")
        for name in ("sparse_minimum_cells", "sparse_minimum_points_per_cell"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError(f"{name} must be an integer >= 4")
        for name in (
            "ransac_iterations",
            "minimum_correspondences",
            "minimum_inliers",
            "grid_rows",
            "grid_cols",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"global_motion.{name} must be a positive integer")
        if (
            isinstance(self.maximum_reuse_pairs, bool)
            or not isinstance(self.maximum_reuse_pairs, int)
            or self.maximum_reuse_pairs < 0
        ):
            raise ValueError("global_motion.maximum_reuse_pairs must be non-negative")
        for name in (
            "ransac_reprojection_px",
            "maximum_median_reprojection_px",
            "maximum_p90_reprojection_px",
            "maximum_reprojection_px",
            "maximum_translation_px",
            "maximum_rotation_deg",
            "minimum_scale",
            "maximum_scale",
            "minimum_sample_separation_px",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"global_motion.{name} must be positive and finite")
        for name in (
            "minimum_inlier_ratio",
            "minimum_inlier_grid_coverage",
            "minimum_noncollinearity_ratio",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"global_motion.{name} must be in [0, 1]")
        if self.minimum_scale > self.maximum_scale:
            raise ValueError("global_motion scale limits are reversed")
        if self.failure_policy not in {"reset_reference", "reuse_previous"}:
            raise ValueError(
                "global_motion.failure_policy must be reset_reference or reuse_previous"
            )
        if self.failure_policy == "reuse_previous" and self.maximum_reuse_pairs == 0:
            raise ValueError("reuse_previous requires maximum_reuse_pairs > 0")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GlobalMotionConfig":
        data = dict(value or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown global_motion configuration keys: {unknown}")
        return cls(**data)


def _readonly_array(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GlobalMotionEstimate:
    """Pairwise transform mapping previous-frame pixels to current-frame pixels."""

    model: str
    previous_frame_index: int
    current_frame_index: int
    previous_to_current_matrix: np.ndarray | None
    inlier_mask: np.ndarray
    residuals_px: np.ndarray
    parameters: dict[str, float] | None
    metrics: dict[str, Any]
    quality_status: str
    rejection_reasons: tuple[str, ...]
    timing_ms: float

    def __post_init__(self) -> None:
        matrix = self.previous_to_current_matrix
        if matrix is not None:
            matrix = _readonly_array(matrix, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("previous_to_current_matrix must be finite 3x3")
            object.__setattr__(self, "previous_to_current_matrix", matrix)
        mask = _readonly_array(self.inlier_mask, dtype=bool).reshape(-1)
        residuals = _readonly_array(self.residuals_px, dtype=np.float64).reshape(-1)
        if len(mask) != len(residuals):
            raise ValueError("inlier_mask and residuals_px must have equal length")
        if self.quality_status not in {"accepted", "rejected"}:
            raise ValueError("quality_status must be accepted or rejected")
        if self.quality_status == "accepted" and matrix is None:
            raise ValueError("an accepted transform must have a matrix")
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "residuals_px", residuals)

    @property
    def accepted(self) -> bool:
        return self.quality_status == "accepted"

    def to_dict(self, *, include_inlier_indices: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "previous_frame_index": self.previous_frame_index,
            "current_frame_index": self.current_frame_index,
            "mapping": "previous_frame_pixels_to_current_frame_pixels",
            "previous_to_current_matrix": (
                self.previous_to_current_matrix.tolist()
                if self.previous_to_current_matrix is not None
                else None
            ),
            "parameters": self.parameters,
            "metrics": self.metrics,
            "quality_status": self.quality_status,
            "rejection_reasons": list(self.rejection_reasons),
            "timing_ms": self.timing_ms,
        }
        if include_inlier_indices:
            result["inlier_indices"] = np.flatnonzero(self.inlier_mask).tolist()
        return result


def _fit_translation(previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    if len(previous) < 1:
        return None
    delta = np.mean(current - previous, axis=0, dtype=np.float64)
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = delta[0]
    matrix[1, 2] = delta[1]
    return matrix


def _fit_similarity(previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
    if len(previous) < 2:
        return None
    rows = np.zeros((len(previous) * 2, 4), dtype=np.float64)
    values = np.zeros(len(previous) * 2, dtype=np.float64)
    x = previous[:, 0]
    y = previous[:, 1]
    rows[0::2, 0] = x
    rows[0::2, 1] = -y
    rows[0::2, 2] = 1
    rows[1::2, 0] = y
    rows[1::2, 1] = x
    rows[1::2, 3] = 1
    values[0::2] = current[:, 0]
    values[1::2] = current[:, 1]
    solution, _residuals, rank, _singular = np.linalg.lstsq(rows, values, rcond=None)
    if rank < 4 or not np.isfinite(solution).all():
        return None
    a, b, tx, ty = solution
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64)]
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:3]


def _residuals(
    matrix: np.ndarray, previous: np.ndarray, current: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(_transform_points(matrix, previous) - current, axis=1)


def _sample_sets(
    points: np.ndarray,
    *,
    sample_size: int,
    iterations: int,
    minimum_separation_px: float,
    seed: int,
) -> list[tuple[int, ...]]:
    count = len(points)
    possible = math.comb(count, sample_size)
    if possible <= iterations:
        candidates = list(itertools.combinations(range(count), sample_size))
        if sample_size == 2:
            candidates = [
                sample
                for sample in candidates
                if np.linalg.norm(points[sample[0]] - points[sample[1]])
                >= minimum_separation_px
            ]
    elif sample_size == 1:
        rng = np.random.default_rng(seed)
        candidates = [
            (int(item),) for item in rng.choice(count, iterations, replace=False)
        ]
    else:
        rng = np.random.default_rng(seed)
        chosen: set[tuple[int, ...]] = set()
        maximum_attempts = iterations * 20
        for _ in range(maximum_attempts):
            sample = tuple(
                sorted(
                    int(item) for item in rng.choice(count, sample_size, replace=False)
                )
            )
            if sample_size == 2:
                separation = np.linalg.norm(points[sample[0]] - points[sample[1]])
                if separation < minimum_separation_px:
                    continue
            chosen.add(sample)
            if len(chosen) >= iterations:
                break
        candidates = sorted(chosen)
    return candidates


def _parameters(matrix: np.ndarray) -> dict[str, float]:
    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    return {
        "translation_x_px": float(matrix[0, 2]),
        "translation_y_px": float(matrix[1, 2]),
        "translation_magnitude_px": float(np.hypot(matrix[0, 2], matrix[1, 2])),
        "rotation_deg": float(np.degrees(np.arctan2(b, a))),
        "scale": float(np.hypot(a, b)),
        "shear": 0.0,
        "perspective": 0.0,
    }


def _noncollinearity_ratio(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
    if eigenvalues[-1] <= _FLOAT64_EPSILON:
        return 0.0
    return float(max(0.0, eigenvalues[0] / eigenvalues[-1]))


def _empty_estimate(
    correspondences: MotionCorrespondences,
    config: GlobalMotionConfig,
    reason: str,
    started_ns: int,
) -> GlobalMotionEstimate:
    count = correspondences.count
    return GlobalMotionEstimate(
        model=config.model,
        previous_frame_index=correspondences.previous_frame_index,
        current_frame_index=correspondences.current_frame_index,
        previous_to_current_matrix=None,
        inlier_mask=np.zeros(count, dtype=bool),
        residuals_px=np.full(count, np.nan),
        parameters=None,
        metrics={
            "correspondence_count": count,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
        },
        quality_status="rejected",
        rejection_reasons=(reason,),
        timing_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
    )


def _batched_translation_samples(previous, current, samples, threshold):
    """Score the identical samples, preserving count/median/index tie breaking.

    Limit scratch memory to 64 hypotheses. Do not rewrite (previous + delta) -
    current as delta - (current - previous): that changes rounding. Only medians
    that cannot possibly win on the primary integer count are omitted. Final
    refinement and all acceptance gates remain in the reference implementation.
    """
    best_matrix = best_mask = best_score = None
    for start in range(0, len(samples), 64):
        block = samples[start:start + 64]
        indices = np.asarray([sample[0] for sample in block], dtype=np.intp)
        delta = current[indices] - previous[indices]
        errors = previous[None, :, :] + delta[:, None, :]
        errors -= current[None, :, :]
        np.square(errors, out=errors)
        residuals = np.sqrt(errors[:, :, 0] + errors[:, :, 1])
        masks = residuals <= threshold
        counts = np.count_nonzero(masks, axis=1)
        maximum = int(counts.max())
        if best_score is not None and maximum < best_score[0]:
            continue
        for row in np.flatnonzero(counts == maximum):
            sample = block[int(row)]
            mask = masks[row]
            median = float(np.median(residuals[row, mask])) if maximum else math.inf
            score = (maximum, -median, tuple(-item for item in sample))
            if best_score is None or score > best_score:
                # Use the original fitter even for the winning single sample.
                best_matrix = _fit_translation(previous[list(sample)], current[list(sample)])
                best_mask = mask.copy()
                best_score = score
    return best_matrix, best_mask, best_score


def fit_global_motion(
    correspondences: MotionCorrespondences,
    config: GlobalMotionConfig | Mapping[str, Any] | None = None,
    *,
    execution: str = "reference",
) -> GlobalMotionEstimate:
    """Fit and quality-gate a previous→current transform using deterministic RANSAC."""

    resolved = (
        config
        if isinstance(config, GlobalMotionConfig)
        else GlobalMotionConfig.from_mapping(config)
    )
    if execution not in {"reference", "translation_batched_exact_v1"}:
        raise ValueError("Unknown global-motion execution policy")
    if execution == "translation_batched_exact_v1" and resolved.model != "translation":
        raise ValueError("Batched translation execution cannot fit a similarity model")
    started_ns = time.perf_counter_ns()
    previous = np.asarray(correspondences.previous_points, dtype=np.float64)
    current = np.asarray(correspondences.current_points, dtype=np.float64)
    sample_size = 1 if resolved.model == "translation" else 2
    if len(previous) < max(resolved.minimum_correspondences, sample_size):
        return _empty_estimate(
            correspondences, resolved, "insufficient_correspondences", started_ns
        )

    fitter = _fit_translation if resolved.model == "translation" else _fit_similarity
    seed = (
        resolved.random_seed
        ^ (correspondences.previous_frame_index * 0x45D9F3B)
        ^ (correspondences.current_frame_index * 0x119DE1F3)
    ) & 0xFFFFFFFFFFFFFFFF
    samples = _sample_sets(
        previous,
        sample_size=sample_size,
        iterations=min(
            resolved.ransac_iterations, math.comb(len(previous), sample_size)
        ),
        minimum_separation_px=resolved.minimum_sample_separation_px,
        seed=seed,
    )
    best_matrix: np.ndarray | None = None
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float, tuple[int, ...]] | None = None
    if execution == "translation_batched_exact_v1":
        best_matrix, best_mask, best_score = _batched_translation_samples(
            previous, current, samples, resolved.ransac_reprojection_px
        )
    for sample in samples if execution == "reference" else ():
        matrix = fitter(previous[list(sample)], current[list(sample)])
        if matrix is None:
            continue
        residuals = _residuals(matrix, previous, current)
        mask = residuals <= resolved.ransac_reprojection_px
        inlier_count = int(np.count_nonzero(mask))
        median = float(np.median(residuals[mask])) if inlier_count else math.inf
        score = (inlier_count, -median, tuple(-item for item in sample))
        if best_score is None or score > best_score:
            best_score = score
            best_matrix = matrix
            best_mask = mask
    if best_matrix is None or best_mask is None or not np.any(best_mask):
        return _empty_estimate(
            correspondences, resolved, "ransac_found_no_model", started_ns
        )

    for _ in range(3):
        refined = fitter(previous[best_mask], current[best_mask])
        if refined is None:
            break
        best_matrix = refined
        refined_residuals = _residuals(best_matrix, previous, current)
        refined_mask = refined_residuals <= resolved.ransac_reprojection_px
        if np.array_equal(refined_mask, best_mask):
            best_mask = refined_mask
            break
        best_mask = refined_mask
    final_matrix = fitter(previous[best_mask], current[best_mask])
    if final_matrix is None:
        return _empty_estimate(
            correspondences, resolved, "inlier_refit_degenerate", started_ns
        )
    all_residuals = _residuals(final_matrix, previous, current)
    final_mask = all_residuals <= resolved.ransac_reprojection_px
    inlier_residuals = all_residuals[final_mask]
    inlier_points = previous[final_mask]
    inlier_count = len(inlier_residuals)
    inlier_ratio = inlier_count / len(previous)
    coverage = grid_coverage(
        inlier_points,
        correspondences.full_image_size,
        grid_rows=resolved.grid_rows,
        grid_cols=resolved.grid_cols,
    )
    noncollinearity = _noncollinearity_ratio(inlier_points)
    parameters = _parameters(final_matrix)
    median_error = float(np.median(inlier_residuals)) if inlier_count else math.inf
    p90_error = float(np.percentile(inlier_residuals, 90)) if inlier_count else math.inf
    maximum_error = float(np.max(inlier_residuals)) if inlier_count else math.inf
    sparse_support = None
    sparse_accepted = False
    phase2_reasons = correspondences.metrics.get("quality_rejection_reasons")
    needs_sparse_support = (
        coverage["fraction"] < resolved.minimum_inlier_grid_coverage
        or correspondences.metrics.get("usable_for_transform") is False
    )
    if resolved.coverage_policy == "translation_consensus" and needs_sparse_support:
        from .translation_support import sparse_translation_support

        sparse_support = sparse_translation_support(
            previous,
            current,
            correspondences.full_image_size,
            final_matrix[:2, 2],
            resolved,
        )
        sparse_accepted = sparse_support["passed"]
    reasons: list[str] = []
    if correspondences.metrics.get("usable_for_transform") is False:
        # Override ONLY an explicitly identified spatial-coverage rejection.
        # Missing/unknown quality reasons and insufficient-feature gates stay closed.
        if not (sparse_accepted and phase2_reasons == ["low_grid_coverage"]):
            reasons.append("correspondence_quality_gate")
    if inlier_count < resolved.minimum_inliers:
        reasons.append("insufficient_inliers")
    if inlier_ratio < resolved.minimum_inlier_ratio:
        reasons.append("low_inlier_ratio")
    if (
        coverage["fraction"] < resolved.minimum_inlier_grid_coverage
        and not sparse_accepted
    ):
        reasons.append("low_inlier_grid_coverage")
    if resolved.model == "similarity" and (
        noncollinearity < resolved.minimum_noncollinearity_ratio
    ):
        reasons.append("degenerate_inlier_geometry")
    if median_error > resolved.maximum_median_reprojection_px:
        reasons.append("high_median_reprojection_error")
    if p90_error > resolved.maximum_p90_reprojection_px:
        reasons.append("high_p90_reprojection_error")
    if maximum_error > resolved.maximum_reprojection_px:
        reasons.append("high_maximum_reprojection_error")
    if parameters["translation_magnitude_px"] > resolved.maximum_translation_px:
        reasons.append("translation_limit")
    if abs(parameters["rotation_deg"]) > resolved.maximum_rotation_deg:
        reasons.append("rotation_limit")
    if not resolved.minimum_scale <= parameters["scale"] <= resolved.maximum_scale:
        reasons.append("scale_limit")

    metrics: dict[str, Any] = {
        "correspondence_count": len(previous),
        "ransac_samples_evaluated": len(samples),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": median_error,
        "p90_reprojection_error_px": p90_error,
        "maximum_reprojection_error_px": maximum_error,
        "residual_background_motion_px": {
            "median": median_error,
            "p90": p90_error,
            "maximum": maximum_error,
        },
        "inlier_grid_coverage": coverage,
        "inlier_noncollinearity_ratio": noncollinearity,
    }
    if resolved.coverage_policy == "translation_consensus":
        metrics["coverage_acceptance_path"] = (
            "rejected"
            if reasons
            else "sparse_translation_consensus"
            if sparse_accepted
            else "full_grid"
        )
        metrics["sparse_translation_support"] = sparse_support
    return GlobalMotionEstimate(
        model=resolved.model,
        previous_frame_index=correspondences.previous_frame_index,
        current_frame_index=correspondences.current_frame_index,
        previous_to_current_matrix=final_matrix,
        inlier_mask=final_mask,
        residuals_px=all_residuals,
        parameters=parameters,
        metrics=metrics,
        quality_status="accepted" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
        timing_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
    )


@dataclass(frozen=True, slots=True)
class ComposedMotionState:
    """Transform mapping the current frame directly into its segment reference."""

    reference_frame_index: int
    current_frame_index: int
    segment_index: int
    reference_from_current_matrix: np.ndarray
    status: str
    window_reset: bool
    reused_pairs: int
    pair_parameter_delta: dict[str, float] | None

    def __post_init__(self) -> None:
        matrix = _readonly_array(self.reference_from_current_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("reference_from_current_matrix must be finite 3x3")
        object.__setattr__(self, "reference_from_current_matrix", matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": "current_frame_pixels_to_segment_reference_pixels",
            "reference_frame_index": self.reference_frame_index,
            "current_frame_index": self.current_frame_index,
            "segment_index": self.segment_index,
            "reference_from_current_matrix": self.reference_from_current_matrix.tolist(),
            "status": self.status,
            "window_reset": self.window_reset,
            "reused_pairs": self.reused_pairs,
            "pair_parameter_delta": self.pair_parameter_delta,
        }


class GlobalMotionTracker:
    """Compose accepted pair transforms and apply the configured failure policy."""

    def __init__(self, config: GlobalMotionConfig, initial_frame_index: int) -> None:
        if initial_frame_index < 0:
            raise ValueError("initial_frame_index must be non-negative")
        self.config = config
        self.reference_frame_index = initial_frame_index
        self.current_frame_index = initial_frame_index
        self.segment_index = 0
        self.reference_from_current = np.eye(3, dtype=np.float64)
        self.reused_pairs = 0
        self.last_parameters: dict[str, float] | None = None

    def _reset(self, current_frame_index: int) -> ComposedMotionState:
        self.reference_frame_index = current_frame_index
        self.current_frame_index = current_frame_index
        self.segment_index += 1
        self.reference_from_current = np.eye(3, dtype=np.float64)
        self.reused_pairs = 0
        self.last_parameters = None
        return ComposedMotionState(
            reference_frame_index=current_frame_index,
            current_frame_index=current_frame_index,
            segment_index=self.segment_index,
            reference_from_current_matrix=self.reference_from_current,
            status="reset_reference",
            window_reset=True,
            reused_pairs=0,
            pair_parameter_delta=None,
        )

    def reset(self, current_frame_index: int) -> ComposedMotionState:
        """Start a new reference segment after an external discontinuity."""

        if current_frame_index <= self.current_frame_index:
            raise GlobalMotionError("Reset frame must follow the current chain frame")
        return self._reset(current_frame_index)

    def update(
        self, estimate: GlobalMotionEstimate, *, force_reset: bool = False,
    ) -> ComposedMotionState:
        if estimate.previous_frame_index != self.current_frame_index:
            raise GlobalMotionError(
                "Transform chain is discontinuous: expected previous frame "
                f"{self.current_frame_index}, got {estimate.previous_frame_index}"
            )
        if force_reset:
            return self._reset(estimate.current_frame_index)
        if not estimate.accepted or estimate.previous_to_current_matrix is None:
            if (
                self.config.failure_policy == "reuse_previous"
                and self.reused_pairs < self.config.maximum_reuse_pairs
            ):
                self.current_frame_index = estimate.current_frame_index
                self.reused_pairs += 1
                return ComposedMotionState(
                    reference_frame_index=self.reference_frame_index,
                    current_frame_index=self.current_frame_index,
                    segment_index=self.segment_index,
                    reference_from_current_matrix=self.reference_from_current,
                    status="reused_previous",
                    window_reset=False,
                    reused_pairs=self.reused_pairs,
                    pair_parameter_delta=None,
                )
            return self._reset(estimate.current_frame_index)

        try:
            current_to_previous = np.linalg.inv(estimate.previous_to_current_matrix)
        except np.linalg.LinAlgError:
            return self._reset(estimate.current_frame_index)
        self.reference_from_current = self.reference_from_current @ current_to_previous
        self.current_frame_index = estimate.current_frame_index
        self.reused_pairs = 0
        parameter_delta: dict[str, float] | None = None
        if self.last_parameters is not None and estimate.parameters is not None:
            parameter_delta = {
                key: float(estimate.parameters[key] - self.last_parameters[key])
                for key in (
                    "translation_x_px",
                    "translation_y_px",
                    "rotation_deg",
                    "scale",
                )
            }
        self.last_parameters = estimate.parameters
        return ComposedMotionState(
            reference_frame_index=self.reference_frame_index,
            current_frame_index=self.current_frame_index,
            segment_index=self.segment_index,
            reference_from_current_matrix=self.reference_from_current,
            status="accepted",
            window_reset=False,
            reused_pairs=0,
            pair_parameter_delta=parameter_delta,
        )
