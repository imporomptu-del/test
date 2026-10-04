"""Causal, prior-measurement-only placement for the offline V42 experiment.

There is no media I/O or target/class decision here. The eight prior actual
measurements forecast a current sampling center in reference coordinates. No
field of the current row's ``tracks`` is accessed, including its length.

The current global transform *is* supplied by the saved whole-current-frame
motion estimator. This is target-position-blind placement, not an experiment
whose every input predates the current image. Historical images are warped
only for camera geometry, never translated to align a purported object.
"""
import math
from collections.abc import Mapping

import numpy as np

from accuracy_v41_geometry import _sample


FRAME_NS = 100_000_000
HISTORY = 8
RADIUS = 64
SIZE = 2 * RADIUS + 1
MIN_MEASUREMENTS = 5
MAX_TRANSFORM_CONDITION = 1e12


def _unavailable(reason, geometry=None):
    return dict(available=False, reasons=[reason], geometry=geometry or {},
                current129=None, history129=None, prior_centers_xy=None,
                predicted_offset_xy=None)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    return int(value)


def _xy(value, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,)
            or np.iscomplexobj(value)
            or any(isinstance(v, (bool, np.bool_))
                   or not isinstance(v, (int, float, np.number))
                   or not math.isfinite(float(v)) for v in value)):
        raise ValueError(name + " must have two finite real coordinates")
    return np.asarray(value, dtype=np.float64)


def _matrix(value):
    if np.iscomplexobj(value):
        raise ValueError("source_to_reference must be a real 3x3 matrix")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_to_reference must be a real 3x3 matrix") from exc
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("source_to_reference must be a finite real 3x3 matrix")
    return result


def _project(matrix, points):
    """Project 3xN homogeneous points; reject zero/horizon-crossing support."""
    mapped = matrix @ points
    denominator = mapped[2]
    if (not np.isfinite(mapped).all() or np.any(denominator == 0)
            or (np.min(denominator) < 0 < np.max(denominator))):
        return None
    result = mapped[:2] / denominator
    return result if np.isfinite(result).all() else None


def prepare_history(frames, rows, segment, track_id, origins=None):
    """Return small provenance plus (current, eight history) float64 129 crops.

    Inputs are exactly nine chronological native uint8 grayscale frames and
    corresponding journal rows. Native full-frame arrays are never copied.
    Optional origins describe each crop's integer/global source origin; each
    row transform and every measurement always use global source coordinates.

    Missing/coasted same-ID prior observations remain NaN entries in the
    returned (8,2) prior_centers_xy array. At least five actual prior points
    are required to fit [1,u,u^2], u = (prior_time-current_time)/(8*100 ms).
    The fit predicts at u=0, then inverse current global motion maps it to
    source coordinates. A fitted point's center uses floor(x+0.5), including
    negative coordinates. Predicted offset is relative to that integer center.

    Malformed values raise ValueError. Unsupported chronology, missing prior
    measurements, resets, ill-conditioned transforms, and projective horizons
    return available=False with an explicit reason; they are not negatives.
    """
    if len(frames) != HISTORY + 1 or len(rows) != HISTORY + 1:
        return _unavailable("exactly_nine_frames_and_rows_required")
    segment = _integer(segment, "segment")
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be a nonempty string")
    for frame in frames:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 2
                or frame.dtype != np.uint8 or not frame.size):
            raise ValueError("Every frame must be nonempty native uint8 grayscale")
    if origins is None:
        origins = np.zeros((HISTORY + 1, 2), dtype=np.float64)
    if len(origins) != HISTORY + 1:
        raise ValueError("Exactly nine origins are required")
    origins = np.stack([_xy(origin, "origin") for origin in origins])
    if not np.equal(origins, np.floor(origins)).all():
        raise ValueError("Native crop origins must have integer coordinates")

    indices, timestamps, segments, reset_flags, matrices = [], [], [], [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every row must be a mapping")
        indices.append(_integer(row.get("frame_index"), "frame_index"))
        timestamps.append(_integer(row.get("timestamp_ns"), "timestamp_ns"))
        segments.append(_integer(row.get("segment"), "segment"))
        motion = row.get("motion")
        if not isinstance(motion, Mapping) or type(motion.get("reset")) is not bool:
            raise ValueError("Every row must have boolean motion.reset")
        reset_flags.append(motion["reset"])
        matrices.append(_matrix(row.get("source_to_reference")))
    geometry = dict(
        placement="quadratic_reference_forecast_from_prior_actual_same_id_only",
        current_target_fields_accessed=False,
        current_global_transform_uses_current_whole_frame=True,
        current_frame_index=indices[-1], current_timestamp_ns=timestamps[-1],
        prior_frame_indices=indices[:-1], prior_timestamps_ns=timestamps[:-1],
        segment=segment, track_id=track_id, origins_xy=origins.tolist(),
        prior_measurements=[], measured_prior_count=0,
        required_measured_prior_count=MIN_MEASUREMENTS,
        history_sampling="background_camera_warp_only_no_object_transport",
        polynomial_basis="[1,u,u^2]; u=(prior_ns-current_ns)/(8*100000000)",
    )
    if any(b - a != 1 for a, b in zip(indices, indices[1:])):
        return _unavailable("noncontiguous_frame_indices", geometry)
    if any(b - a != FRAME_NS for a, b in zip(timestamps, timestamps[1:])):
        return _unavailable("noncontiguous_100ms_timestamps", geometry)
    if any(value != segment for value in segments):
        return _unavailable("segment_boundary", geometry)
    if any(reset_flags):
        return _unavailable("motion_reset_within_nine_frames", geometry)
    if any(np.linalg.cond(matrix) > MAX_TRANSFORM_CONDITION for matrix in matrices):
        return _unavailable("singular_or_ill_conditioned_transform", geometry)

    prior_source = np.full((HISTORY, 2), np.nan, dtype=np.float64)
    prior_reference = np.full_like(prior_source, np.nan)
    for index, row in enumerate(rows[:-1]):
        tracks = row.get("tracks")
        if not isinstance(tracks, (list, tuple)):
            raise ValueError("Prior tracks must be a sequence")
        matches = []
        for track in tracks:
            if not isinstance(track, Mapping):
                raise ValueError("Prior tracks must contain mappings")
            if track.get("track_id") == track_id:
                matches.append(track)
        if len(matches) > 1:
            raise ValueError("Duplicate same-ID prior track within frame")
        if not matches:
            continue
        track = matches[0]
        if type(track.get("predicted")) is not bool:
            raise ValueError("Same-ID prior track must have boolean predicted")
        actual = track.get("measurement_source_xy")
        if track["predicted"]:
            if actual is not None:
                raise ValueError("Predicted prior track cannot contain an actual measurement")
            continue
        if actual is None:
            raise ValueError("Actual prior track must contain measurement_source_xy")
        source = _xy(actual, "prior measurement_source_xy")
        reference = _project(matrices[index], np.array([*source, 1.])[:, None])
        if reference is None:
            return _unavailable("prior_measurement_projective_horizon", geometry)
        prior_source[index] = source
        prior_reference[index] = reference[:, 0]
        geometry["prior_measurements"].append(dict(
            frame_index=indices[index], timestamp_ns=timestamps[index],
            source_xy=source.tolist(), reference_xy=reference[:, 0].tolist(),
            source_to_reference=matrices[index].tolist()))
    available = np.isfinite(prior_reference).all(axis=1)
    geometry["measured_prior_count"] = int(available.sum())
    if available.sum() < MIN_MEASUREMENTS:
        return _unavailable("fewer_than_five_prior_actual_same_id_measurements", geometry)

    # Integer subtraction before float conversion retains nanosecond precision.
    u = np.asarray([(value - timestamps[-1]) / (HISTORY * FRAME_NS)
                    for value in timestamps[:-1]], dtype=np.float64)
    design = np.column_stack((np.ones(HISTORY), u, u*u))[available]
    # Fit offsets to avoid losing fractional source pixels to a large absolute
    # reference origin. This also preserves an exactly constant half-pixel
    # trajectory instead of nudging it below the round-half-up tie by SVD noise.
    reference_origin = prior_reference[available][0]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, prior_reference[available] - reference_origin, rcond=None)
    coefficients[0] += reference_origin
    geometry["forecast_fit_rank"] = int(rank)
    geometry["forecast_fit_singular_values"] = singular_values.tolist()
    if rank != 3 or not np.isfinite(coefficients).all():
        return _unavailable("quadratic_forecast_rank_or_finite_failure", geometry)
    residual = prior_reference[available] - design @ coefficients
    geometry["forecast_fit_rmse_reference_px"] = float(
        np.sqrt(np.mean(np.sum(residual*residual, axis=1))))
    geometry["forecast_coefficients_reference_xy"] = coefficients.tolist()
    current_inverse = np.linalg.inv(matrices[-1])
    forecast = _project(current_inverse, np.array([*coefficients[0], 1.])[:, None])
    if forecast is None:
        return _unavailable("forecast_projective_horizon", geometry)
    forecast = forecast[:, 0]
    # Float grids avoid integer conversion overflow for remote unsupported fits.
    center = np.floor(forecast + .5)
    offset = forecast - center
    geometry.update(predicted_reference_xy=coefficients[0].tolist(),
                    predicted_source_xy=forecast.tolist(),
                    current_center_xy=center.tolist(),
                    predicted_offset_xy=offset.tolist(),
                    current_source_to_reference=matrices[-1].tolist())

    yy, xx = np.mgrid[-RADIUS:RADIUS+1, -RADIUS:RADIUS+1]
    global_x, global_y = xx + center[0], yy + center[1]
    homogeneous_grid = np.stack((global_x.ravel(), global_y.ravel(),
                                 np.ones(SIZE * SIZE)))
    current = _sample(frames[-1], global_x-origins[-1, 0], global_y-origins[-1, 1])
    history = np.empty((HISTORY, SIZE, SIZE), dtype=np.float64)
    prior_centers = np.full((HISTORY, 2), np.nan, dtype=np.float64)
    geometry["current_to_prior_matrices"] = []
    for index in range(HISTORY):
        warp = np.linalg.solve(matrices[index], matrices[-1])
        mapped = _project(warp, homogeneous_grid)
        if mapped is None:
            return _unavailable("history_crop_projective_horizon", geometry)
        history[index] = _sample(frames[index],
                                mapped[0].reshape(SIZE, SIZE)-origins[index, 0],
                                mapped[1].reshape(SIZE, SIZE)-origins[index, 1])
        geometry["current_to_prior_matrices"].append(warp.tolist())
        if available[index]:
            point = _project(current_inverse,
                             np.array([*prior_reference[index], 1.])[:, None])
            if point is None:
                return _unavailable("prior_center_projective_horizon", geometry)
            prior_centers[index] = point[:, 0] - center + RADIUS
    geometry["current_supported_pixels"] = int(np.isfinite(current).sum())
    geometry["history_supported_pixels"] = np.isfinite(history).sum(axis=(1, 2)).tolist()
    geometry["prior_centers_xy"] = [value.tolist() if np.isfinite(value).all() else None
                                    for value in prior_centers]
    return dict(available=True, reasons=[], geometry=geometry, current129=current,
                history129=history, prior_centers_xy=prior_centers,
                predicted_offset_xy=offset)
