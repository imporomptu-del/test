"""V53 fixed-gain, zero-slope median-offset current-guard cross-fitting.

An isolated shadow estimator, not a detection filter or a prior-only temporal
forecast. Training uses only the complementary guard fold's current samples.
No clipping, response-based selection, numerical tuning or prediction fallback.
"""

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


LOSSES = ("median_offset",)
SPLITS = ("left_right", "checkerboard")


def model_constants():
    return dict(loss="absolute_deviation", fixed_gain=1.0,
        fixed_x_slope=0.0, fixed_y_slope=0.0,
        minimum_finite_training_rows=4,
        guard_coordinate_min=8, guard_coordinate_max=120, guard_coordinate_step=8,
        guard_center=64, guard_chebyshev_radius_min=40, guard_chebyshev_radius_max=56,
        median_rule="correctly_rounded_exact_rational_midpoint_of_middle_residuals",
        objective_rule="correctly_rounded_exact_mean_of_finite_absolute_deviations",
        optimality_rule="zero_in_mean_L1_subgradient_interval_from_exact_order_counts",
        clipping_or_tuning=False, implicit_fallback=False,
        arithmetic_failure_discards_entire_fit=True)


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fingerprint(value, excluded=()):
    payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(json.dumps(_plain(payload), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_fingerprint(model):
    return _fingerprint(model, ("model_sha256",))


def crossfit_fingerprint(result):
    return _fingerprint(result, ("crossfit_sha256",))


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _vector(value, count, name):
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric vector of length {count}")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(array, dtype=np.float64, copy=True)


def _geometry(points_xy):
    array = np.asarray(points_xy)
    if array.ndim != 2 or array.shape[1:] != (2,) or array.dtype.kind not in "iuf":
        raise ValueError("points_xy must be a real numeric array of shape (N, 2)")
    with np.errstate(over="ignore", invalid="ignore"):
        xy = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(xy).all():
        raise ValueError("guard geometry must be finite")
    if np.any((xy < 8) | (xy > 120) | (xy % 8 != 0)):
        raise ValueError("guard geometry must use the 8..120, step-8 grid")
    radius = np.max(np.abs(xy-64), axis=1) if len(xy) else np.empty(0)
    if np.any((radius < 40) | (radius > 56)):
        raise ValueError("only the Chebyshev-radius 40..56 guard is accepted; core excluded")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("duplicate guard coordinates are not accepted")
    return xy


def split_ids(points_xy, split="left_right"):
    xy = _geometry(points_xy)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "left_right":
        ids = xy[:, 0] >= 64
    else:
        ids = (xy[:, 0].astype(np.int64)//8 + xy[:, 1].astype(np.int64)//8) % 2
    return _readonly(np.asarray(ids, dtype=np.int8))


def _midpoint(lower, upper):
    """Correctly rounded finite midpoint, including extreme/subnormal inputs."""
    return float((Fraction.from_float(float(lower))+Fraction.from_float(float(upper)))/2)


def fit(train_xy, slow, current, loss="median_offset"):
    """Fit the median training residual, preserving unknown indices explicitly.

    Missing/nonfinite input rows are recorded and excluded without imputation.
    Overflow from otherwise finite rows invalidates the entire fit; those rows
    are never silently removed. Candidate and order diagnostics are retained
    for audit even when later objective arithmetic fails. Prediction accepts
    only an available offset, never a diagnostic-only candidate.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    xy = _geometry(train_xy)
    s = _vector(slow, len(xy), "slow")
    c = _vector(current, len(xy), "current")
    used = np.isfinite(s) & np.isfinite(c)
    result = dict(schema_version=1, loss=loss, available=False, unavailable_reason=None,
        training_count=len(xy), training_used_count=int(used.sum()),
        training_used_mask=_readonly(used), training_unavailable_reasons=tuple(
            None if good else "nonfinite_training_slow_or_current" for good in used),
        training_input_sha256=_fingerprint(dict(points_xy=xy, slow=s, current=c)),
        offset_dn=None, candidate_offset_dn=None,
        median_interval_dn=_readonly(np.full(2, np.nan)), objective_mae_dn=None,
        subgradient_interval=_readonly(np.full(2, np.nan)), residual_order_counts=None,
        constants=model_constants())

    def finish(reason=None):
        result["unavailable_reason"] = reason
        result["model_sha256"] = model_fingerprint(result)
        return result

    count = int(used.sum())
    if count < 4:
        return finish("insufficient_finite_training_rows")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = c[used]-s[used]
    if not np.isfinite(residuals).all():
        return finish("nonfinite_training_residual_arithmetic")
    ordered = np.sort(residuals)
    lower, upper = float(ordered[(count-1)//2]), float(ordered[count//2])
    offset = _midpoint(lower, upper)
    if not math.isfinite(offset) or not lower <= offset <= upper:
        return finish("nonfinite_or_outside_median_interval")
    below = int(np.count_nonzero(residuals < offset))
    above = int(np.count_nonzero(residuals > offset))
    tied = int(np.count_nonzero(residuals == offset))
    subgradient = np.array([(below-above-tied)/count, (below-above+tied)/count])
    result.update(candidate_offset_dn=offset, median_interval_dn=_readonly([lower, upper]),
        subgradient_interval=_readonly(subgradient),
        residual_order_counts=dict(below=below, above=above, tied=tied))
    if below+above+tied != count or not subgradient[0] <= 0 <= subgradient[1]:
        return finish("median_subgradient_certificate_failed")
    with np.errstate(over="ignore", invalid="ignore"):
        deviations = np.abs(residuals-offset)
    if not np.isfinite(deviations).all():
        return finish("nonfinite_objective_arithmetic")
    # Exact sum divided before rounding avoids overflowing a sum whose mean
    # remains finite, and avoids losing subnormal terms through early division.
    objective = float(sum((Fraction.from_float(float(value)) for value in deviations), Fraction())/count)
    if not math.isfinite(objective):
        return finish("nonfinite_objective_arithmetic")
    result.update(available=True, offset_dn=offset, objective_mae_dn=objective)
    return finish()


def predict(model, test_xy, slow):
    """Predict S+offset without accepting current/response values or labels."""
    xy = _geometry(test_xy)
    s = _vector(slow, len(xy), "slow")
    if not isinstance(model, Mapping) or model.get("model_sha256") != model_fingerprint(model):
        raise ValueError("model fingerprint mismatch")
    values = np.full(len(xy), np.nan)
    available = np.zeros(len(xy), dtype=bool)
    reasons = [None]*len(xy)
    if not model["available"]:
        reasons = ["fit_unavailable:"+str(model["unavailable_reason"])]*len(xy)
    else:
        finite = np.isfinite(s)
        for i in np.flatnonzero(~finite):
            reasons[int(i)] = "nonfinite_prediction_slow"
        indices = np.flatnonzero(finite)
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = s[finite]+model["offset_dn"]
        valid = np.isfinite(predicted)
        values[indices[valid]] = predicted[valid]
        available[indices[valid]] = True
        for i in indices[~valid]:
            reasons[int(i)] = "nonfinite_prediction_arithmetic"
    return dict(values=_readonly(values), available=_readonly(available),
                unavailable_reasons=tuple(reasons), model_sha256=model["model_sha256"])


def crossfit(points_xy, slow, current, split="left_right"):
    """Fit on each complementary guard fold and retain every held-out index."""
    xy = _geometry(points_xy)
    s = _vector(slow, len(xy), "slow")
    c = _vector(current, len(xy), "current")
    ids = split_ids(xy, split)
    fits = {}
    values = np.full(len(xy), np.nan)
    available = np.zeros(len(xy), dtype=bool)
    reasons = [None]*len(xy)
    for fold in (0, 1):
        heldout = ids == fold
        trained = fit(xy[~heldout], s[~heldout], c[~heldout])
        forecast = predict(trained, xy[heldout], s[heldout])
        fits[str(fold)] = trained
        values[heldout], available[heldout] = forecast["values"], forecast["available"]
        for index, reason in zip(np.flatnonzero(heldout), forecast["unavailable_reasons"]):
            reasons[int(index)] = reason
    result = dict(schema_version=1, split=split, total_count=len(xy), fold_id=ids,
        fits={"median_offset": fits}, predictions={"median_offset": dict(
            values=_readonly(values), available=_readonly(available),
            unavailable_reasons=tuple(reasons), model_sha256=None)}, constants=model_constants(),
        metadata=dict(current_complementary_guard_values_used=True,
            heldout_current_argument_accepted_by_predict=False,
            fit_dictionary_key_names_heldout_fold=True,
            core_pixels_accepted=False, scoring_or_forecast_selection_performed=False,
            unavailable_indices_preserved_without_fallback=True,
            prior_only_or_online_camera_causality_certified=False,
            guard_purity_or_guard_to_core_transfer_certified=False,
            production_detection_modified=False))
    result["crossfit_sha256"] = crossfit_fingerprint(result)
    return result
