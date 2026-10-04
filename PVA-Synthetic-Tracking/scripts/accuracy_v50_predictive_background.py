"""Prior-value-only guard forecasts and separate current-frame measurements.

This experiment predicts brightness, not objects. It does not calibrate intervals,
change source evidence, or make production decisions. The caller supplies eight
already aligned prior patches; their registration/crop geometry is not certified
to predate the current image. There is deliberately no inner pseudo-time API.

Only the fixed V47 guard candidates are inspected for prior eligibility. Only the
union of complete triplets contributes to forecasts or current measurements.
The one-DN scale floor is a normalization convention, NOT a sensor-noise bound.
"""

from collections.abc import Mapping
import hashlib
import json

import numpy as np


PATCH_SHAPE = (129, 129)
HISTORY_SHAPE = (8, 129, 129)
SCALE_FLOOR_DN = 1.0
ARMS = ("median8_unit_scale", "median8_temporal_scale", "median3_temporal_scale")
WEIGHTS = (1, -2, 1)


def _array(value, shape, name):
    # Shape/type checking must not scan or convert out-of-support pixel values.
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    return array


def _centers(value):
    try:
        if len(value) != 8:
            raise ValueError("Exactly eight prior centers or None entries required")
    except TypeError as exc:
        raise ValueError("Exactly eight prior centers or None entries required") from exc
    output = []
    for point in value:
        if point is None:
            output.append(None)
            continue
        point = _array(point, (2,), "prior center")
        with np.errstate(over="ignore", invalid="ignore"):
            point = np.asarray(point, dtype=np.float64)
        if not np.isfinite(point).all():
            raise ValueError("A prior center must be finite; use None for missing")
        output.append(point.tolist())
    return output


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def forecast_fingerprint(result):
    """Hash all forecast content except its own digest, without current pixels."""
    if not isinstance(result, Mapping):
        raise ValueError("forecast result must be a mapping")
    return _hash({key: value for key, value in result.items()
                  if key != "forecast_sha256"})


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def forecast(history129, prior_centers_xy):
    """Return three frozen guard predictors with identical prior-only support.

    Missing centers are explicit None values. Nonfinite prior candidate pixels
    are excluded before triplet construction; no current image is accepted.
    Arrays in each arm are 1-D, ordered exactly like ``used_points_xy``. They are
    independent, read-only copies. Empty support or nonfinite forecast arithmetic
    is unavailable rather than a negative source decision.
    """
    history = _array(history129, HISTORY_SHAPE, "history129")
    centers = _centers(prior_centers_xy)
    candidates = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                  if 40 <= max(abs(x-64), abs(y-64)) <= 56]
    yy, xx = np.asarray([p[1] for p in candidates]), np.asarray([p[0] for p in candidates])
    # Eligibility necessarily reads the candidate prior values, not the full
    # patches. This is the only pixel gather from the supplied prior arrays.
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_values = np.asarray(history[:, yy, xx], dtype=np.float64)
    finite = np.isfinite(candidate_values).all(axis=0)
    eligible = []
    rejection_counts = dict(prior_foreground_footprint=0, nonfinite_prior_history=0)
    for index, (x, y) in enumerate(candidates):
        near = any(point is not None and
                   max(abs(x-point[0]), abs(y-point[1])) <= 12 for point in centers)
        rejection_counts["prior_foreground_footprint"] += int(near)
        rejection_counts["nonfinite_prior_history"] += int(not finite[index])
        if not near and finite[index]:
            eligible.append((x, y))
    eligible_set = set(eligible)
    stencils = []
    for axis, step in (("x", (8, 0)), ("y", (0, 8))):
        for x, y in candidates:
            points = [(x-step[0], y-step[1]), (x, y), (x+step[0], y+step[1])]
            if all(point in eligible_set for point in points):
                stencils.append(dict(axis=axis, center_xy=[x, y],
                    pixels_xy=[list(point) for point in points], weights=list(WEIGHTS)))
    used = sorted({tuple(point) for stencil in stencils
                   for point in stencil["pixels_xy"]}, key=lambda point: (point[1], point[0]))
    result = dict(schema_version=1, available=False, reasons=[],
        candidate_count=len(candidates), eligible_count=len(eligible), used_count=len(used),
        stencil_count=len(stencils), candidate_support_sha256=_hash(candidates),
        eligible_support_sha256=_hash(eligible), used_support_sha256=_hash(used),
        stencil_sha256=_hash(stencils), used_points_xy=[list(point) for point in used],
        stencils=stencils, arms={}, prior_rejection_counts_nonexclusive=rejection_counts,
        missing_prior_center_indices=[i for i, point in enumerate(centers) if point is None],
        metadata=dict(
            prior_values_only_given_supplied_alignment=True,
            current_argument_accepted=False,
            current_or_prior_core_values_accessed=False,
            prior_values_accessed_only_at_guard_candidates=True,
            used_support_is_union_of_complete_triplets=True,
            candidate_grid_step=8, candidate_radial_range=[40, 56],
            exclusion_chebyshev_radius=12, prior_count=8,
            adaptive_scale_formula="max(1 DN, median(abs(history - median(history))))",
            both_temporal_arms_share_identical_scale=True,
            scale_floor_dn=SCALE_FLOOR_DN,
            scale_floor_is_normalization_not_physical_error_bound=True,
            noise_calibration_performed=False, interval_coverage_claimed=False,
            supplied_geometry_may_use_current_whole_frame=True,
            inner_pseudo_time_calibration_performed=False,
            missing_prior_centers_do_not_establish_guard_purity=True,
            current_guard_purity_or_guard_to_core_transfer_certified=False,
            missing_or_saturated_footprint_nan_mask_is_caller_responsibility=True,
            finite_zero_and_255_not_reclassified_as_saturation=True,
            no_source_class_or_production_decision=True))
    if not used:
        result["reasons"] = ["no_prior_selected_guard_contrasts"]
    else:
        lookup = {point: i for i, point in enumerate(candidates)}
        values = candidate_values[:, [lookup[point] for point in used]]
        with np.errstate(over="ignore", invalid="ignore"):
            median8 = np.median(values, axis=0)
            median3 = np.median(values[-3:], axis=0)
            scale = np.maximum(SCALE_FLOOR_DN, np.median(np.abs(values-median8), axis=0))
        if not (np.isfinite(median8).all() and np.isfinite(median3).all()
                and np.isfinite(scale).all()):
            result["reasons"] = ["nonfinite_prior_forecast_arithmetic"]
        else:
            result["arms"] = {
                ARMS[0]: dict(prediction=_readonly(median8), scale=_readonly(np.ones(len(used)))),
                ARMS[1]: dict(prediction=_readonly(median8), scale=_readonly(scale)),
                ARMS[2]: dict(prediction=_readonly(median3), scale=_readonly(scale)),
            }
            result["available"] = True
    result["forecast_sha256"] = forecast_fingerprint(result)
    return result


def measure_current(current129, forecast_result):
    """Measure a current guard against an unchanged, hash-bound forecast.

    Any nonfinite used current point makes the entire packet unavailable. No
    current-dependent support deletion, model fitting, scale update, or use of
    the current core occurs. Residual signs are current minus prediction.
    Returned measurements are not classifications or calibrated intervals.
    """
    current = _array(current129, PATCH_SHAPE, "current129")
    digest = forecast_fingerprint(forecast_result)
    if forecast_result.get("forecast_sha256") != digest:
        raise ValueError("Forecast content no longer matches its frozen hash")
    result = dict(available=False, reasons=[], forecast_sha256=digest,
        used_support_sha256=forecast_result["used_support_sha256"],
        used_count=forecast_result["used_count"],
        current_nonfinite_used_point_count=None, arms={},
        metadata=dict(current_values_accessed_only_at_used_guard_points=True,
            current_core_values_accessed=False, current_dependent_support_deletion=False,
            forecast_fitted_or_modified_from_current=False,
            score_is_maximum_normalized_absolute_error_over_fixed_guard=True,
            scores_are_not_independent_pixel_samples=True,
            no_source_class_or_production_decision=True))
    if not forecast_result["available"]:
        result["reasons"] = ["prior_forecast_unavailable"]
        return result
    points = forecast_result["used_points_xy"]
    yy, xx = np.asarray([p[1] for p in points]), np.asarray([p[0] for p in points])
    # The sole current-value gather: not a scan, cast or copy of the whole patch.
    with np.errstate(over="ignore", invalid="ignore"):
        observed = np.asarray(current[yy, xx], dtype=np.float64)
    invalid = int((~np.isfinite(observed)).sum())
    result["current_nonfinite_used_point_count"] = invalid
    if invalid:
        result["reasons"] = ["nonfinite_current_on_fixed_used_guard_support"]
        return result
    arms = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for name in ARMS:
            prediction = forecast_result["arms"][name]["prediction"]
            scale = forecast_result["arms"][name]["scale"]
            residual = observed-prediction
            absolute = np.abs(residual)
            normalized = absolute/scale
            peak = float(absolute.max())
            # Scaled reductions avoid overflowing the sum/square of finite errors.
            mae = 0.0 if peak == 0 else float(peak*np.mean(absolute/peak))
            rmse = 0.0 if peak == 0 else float(peak*np.sqrt(np.mean((absolute/peak)**2)))
            maximum = float(normalized.max())
            if not (np.isfinite(residual).all() and np.isfinite(normalized).all()
                    and np.isfinite([mae, rmse, maximum]).all()):
                result["reasons"] = ["nonfinite_current_residual_or_summary"]
                return result
            arms[name] = dict(residuals=_readonly(residual),
                normalized_absolute_errors=_readonly(normalized), max_score=maximum,
                mae_dn=mae, rmse_dn=rmse)
    result["available"] = True
    result["arms"] = arms
    if forecast_fingerprint(forecast_result) != digest:
        raise AssertionError("Forecast changed during current measurement")
    return result
