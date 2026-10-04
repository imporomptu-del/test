"""Frozen V52 current-guard gain-plus-plane fitting, isolated from detection.

This is spatial cross-fitting, not a prior-only temporal forecast. Current
values from a training guard half are allowed; held-out current values are not
arguments to prediction. Guard purity, registration causality and guard-to-core
transfer are not certified. There is deliberately no scoring or fallback here.
"""

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


HUBER_DELTA_DN = 2.0
MAX_IRLS_UPDATES = 100
KKT_TOLERANCE_DN = 1e-7
RANK_RTOL = 1e-12
CONDITION_LIMIT = 1e8
LOSSES = ("ols", "huber")
SPLITS = ("left_right", "checkerboard")


def model_constants():
    return dict(huber_delta_dn=HUBER_DELTA_DN,
        max_irls_updates=MAX_IRLS_UPDATES, kkt_tolerance_dn=KKT_TOLERANCE_DN,
        rank_rtol=RANK_RTOL, condition_limit=CONDITION_LIMIT,
        training_scale_floor_dn=1.0, coordinate_center=64.0,
        coordinate_scale=56.0, gain_lower_bound=0.0,
        minimum_finite_training_rows=4,
        kkt_gradient_scaling="mean_gradient / max(1, RMS(conditioned_design_column))",
        full_rank_required=True, implicit_ridge_or_fallback=False)


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


def _design(xy, slow, center, scale):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.column_stack(((slow-center)/scale, np.ones(len(slow)),
                                (xy[:, 0]-64)/56, (xy[:, 1]-64)/56))


def _rank_condition(design):
    if not np.isfinite(design).all():
        return None, None, "nonfinite_design_arithmetic"
    try:
        singular = np.linalg.svd(design, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, None, "svd_failed"
    if not len(singular) or not np.isfinite(singular).all():
        return None, None, "nonfinite_singular_values"
    rank = int(np.sum(singular > RANK_RTOL*singular[0]))
    condition = (float(singular[0]/singular[-1])
                 if singular[-1] > 0 else None)
    if rank != design.shape[1]:
        return rank, condition, "rank_deficient_design"
    if condition is None or not math.isfinite(condition) or condition > CONDITION_LIMIT:
        return rank, condition, "ill_conditioned_design"
    return rank, condition, None


def _weighted_solve(design, response, weights):
    """Exact one-bound active-set WLS; never silently truncate singular modes."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        root = np.sqrt(weights)
        weighted = design*root[:, None]
        target = response*root
    rank, condition, reason = _rank_condition(weighted)
    if reason is not None:
        return None, f"weighted_{reason}"
    if not np.isfinite(target).all():
        return None, "nonfinite_weighted_response"
    try:
        coefficients, _, solved_rank, _ = np.linalg.lstsq(weighted, target, rcond=RANK_RTOL)
        if solved_rank != 4:
            return None, "weighted_lstsq_rank_mismatch"
        if coefficients[0] < 0:
            # With a single lower bound, a negative unconstrained minimizer
            # puts the constrained optimum on the gain-zero boundary.
            reduced = weighted[:, 1:]
            _, _, reason = _rank_condition(reduced)
            if reason is not None:
                return None, f"active_plane_{reason}"
            plane, _, solved_rank, _ = np.linalg.lstsq(reduced, target, rcond=RANK_RTOL)
            if solved_rank != 3:
                return None, "active_plane_lstsq_rank_mismatch"
            coefficients = np.r_[0.0, plane]
    except np.linalg.LinAlgError:
        return None, "lstsq_failed"
    if not np.isfinite(coefficients).all():
        return None, "nonfinite_coefficient_arithmetic"
    return coefficients, None


def _optimality(design, response, coefficients, loss):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = design @ coefficients-response
        if loss == "huber":
            psi = np.clip(residual, -HUBER_DELTA_DN, HUBER_DELTA_DN)
            absolute = np.abs(residual)
            small = np.minimum(absolute, HUBER_DELTA_DN)
            objective = np.mean(.5*small**2 + HUBER_DELTA_DN*(absolute-small))
        else:
            psi = residual
            objective = np.mean(.5*residual**2)
        gradient = (design.T @ psi)/len(response)
        scaling = np.maximum(1.0, np.sqrt(np.mean(design**2, axis=0)))
        projected = gradient/scaling
        if coefficients[0] == 0:
            projected[0] = min(projected[0], 0.0)
        kkt = np.max(np.abs(projected))
    finite = (np.isfinite(residual).all() and np.isfinite(gradient).all()
              and np.isfinite(scaling).all() and math.isfinite(float(objective))
              and math.isfinite(float(kkt)))
    if not finite:
        return None, None, None, "nonfinite_optimality_arithmetic"
    return float(objective), float(kkt), residual, None


def fit(train_xy, slow, current, loss="huber"):
    """Fit C = gain*S+b0+bx*(x-64)/56+by*(y-64)/56 on train rows only.

    Nonfinite train S/current rows are explicitly unavailable and omitted from
    solving, not imputed. Geometry errors are rejected. Rank deficiency,
    conditioning failures and unconverged IRLS produce an unavailable model.
    The returned physical coefficient order is (gain, b0, bx, by).
    The last candidate is retained for audit only, even on nonconvergence;
    prediction never reads it and requires the separately available model.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    xy = _geometry(train_xy)
    slow = _vector(slow, len(xy), "slow")
    current = _vector(current, len(xy), "current")
    used = np.isfinite(slow) & np.isfinite(current)
    reasons = tuple(None if good else "nonfinite_training_slow_or_current" for good in used)
    result = dict(schema_version=1, loss=loss, available=False,
        unavailable_reason=None, training_count=len(xy), training_used_count=int(used.sum()),
        training_used_mask=_readonly(used), training_unavailable_reasons=reasons,
        center=None, scale=None, coefficients_conditioned=_readonly(np.full(4, np.nan)),
        last_candidate_coefficients_conditioned=_readonly(np.full(4, np.nan)),
        coefficients_physical=_readonly(np.full(4, np.nan)), rank=None,
        condition_number=None, iterations=0, converged=False, gain_bound_active=None,
        objective=None, kkt_residual_dn=None,
        training_input_sha256=_fingerprint(dict(points_xy=xy, slow=slow, current=current)),
        constants=model_constants())

    def finish(reason=None):
        result["unavailable_reason"] = reason
        result["model_sha256"] = model_fingerprint(result)
        return result

    if used.sum() < 4:
        return finish("insufficient_finite_training_rows")
    s, c, xy = slow[used], current[used], xy[used]
    with np.errstate(over="ignore", invalid="ignore"):
        center = float(np.median(s))
        scale = float(np.maximum(1.0, np.median(np.abs(s-center))))
    if not math.isfinite(center) or not math.isfinite(scale):
        return finish("nonfinite_training_scaling")
    result.update(center=center, scale=scale)
    design = _design(xy, s, center, scale)
    rank, condition, reason = _rank_condition(design)
    result.update(rank=rank, condition_number=condition)
    if reason is not None:
        return finish(reason)
    coefficients, reason = _weighted_solve(design, c, np.ones(len(c)))
    if reason is not None:
        return finish(reason)
    result["last_candidate_coefficients_conditioned"] = _readonly(coefficients)
    objective, kkt, residual, reason = _optimality(design, c, coefficients, loss)
    if reason is not None:
        return finish(reason)
    for iteration in range(MAX_IRLS_UPDATES+1):
        result.update(iterations=iteration, objective=objective, kkt_residual_dn=kkt)
        if kkt <= KKT_TOLERANCE_DN:
            break
        if loss == "ols":
            return finish("ols_kkt_check_failed")
        if iteration == MAX_IRLS_UPDATES:
            return finish("irls_nonconvergence")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            absolute = np.abs(residual)
            weights = np.ones(len(c))
            outside = absolute > HUBER_DELTA_DN
            weights[outside] = HUBER_DELTA_DN/absolute[outside]
        coefficients, reason = _weighted_solve(design, c, weights)
        if reason is not None:
            return finish(reason)
        result["last_candidate_coefficients_conditioned"] = _readonly(coefficients)
        result["iterations"] = iteration+1
        result["objective"] = None
        result["kkt_residual_dn"] = None
        objective, kkt, residual, reason = _optimality(design, c, coefficients, loss)
        if reason is not None:
            return finish(reason)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = coefficients[0]/scale
        physical = np.array([gain, coefficients[1]-gain*center,
                             coefficients[2], coefficients[3]])
    if not np.isfinite(physical).all() or gain < 0:
        return finish("nonfinite_physical_coefficient_arithmetic")
    result.update(available=True, converged=True,
                  coefficients_conditioned=_readonly(coefficients),
                  coefficients_physical=_readonly(physical), gain_bound_active=bool(gain == 0))
    return finish()


def predict(model, test_xy, slow):
    """Apply one fitted training model without accepting held-out responses."""
    xy = _geometry(test_xy)
    slow = _vector(slow, len(xy), "slow")
    if not isinstance(model, Mapping) or model.get("model_sha256") != model_fingerprint(model):
        raise ValueError("model fingerprint mismatch")
    values = np.full(len(xy), np.nan)
    available = np.zeros(len(xy), dtype=bool)
    reasons = [None]*len(xy)
    if not model["available"]:
        reasons = ["fit_unavailable:"+str(model["unavailable_reason"])]*len(xy)
    else:
        finite = np.isfinite(slow)
        for i in np.flatnonzero(~finite):
            reasons[int(i)] = "nonfinite_prediction_slow"
        indices = np.flatnonzero(finite)
        design = _design(xy[finite], slow[finite], model["center"], model["scale"])
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = design @ np.asarray(model["coefficients_conditioned"], dtype=np.float64)
        valid = np.isfinite(design).all(axis=1) & np.isfinite(predicted)
        values[indices[valid]] = predicted[valid]
        available[indices[valid]] = True
        for i in indices[~valid]:
            reasons[int(i)] = "nonfinite_prediction_arithmetic"
    return dict(values=_readonly(values), available=_readonly(available),
                unavailable_reasons=tuple(reasons), model_sha256=model["model_sha256"])


def crossfit(points_xy, slow, current, split="left_right"):
    """Predict every declared guard index from its complementary current fold.

    Current samples in fold k can affect predictions for fold 1-k, never their
    own fold k. Both fixed losses are retained; no response-based arm selection
    or error computation occurs. A missing held-out current does not suppress
    that point's prediction, although it will be unscorable downstream.
    """
    xy = _geometry(points_xy)
    s = _vector(slow, len(xy), "slow")
    c = _vector(current, len(xy), "current")
    ids = split_ids(xy, split)
    fits, predictions = {}, {}
    for loss in LOSSES:
        fits[loss] = {}
        values = np.full(len(xy), np.nan)
        available = np.zeros(len(xy), dtype=bool)
        reasons = [None]*len(xy)
        for fold in (0, 1):
            heldout = ids == fold
            trained = fit(xy[~heldout], s[~heldout], c[~heldout], loss)
            forecast = predict(trained, xy[heldout], s[heldout])
            fits[loss][str(fold)] = trained
            values[heldout], available[heldout] = forecast["values"], forecast["available"]
            for index, reason in zip(np.flatnonzero(heldout), forecast["unavailable_reasons"]):
                reasons[int(index)] = reason
        predictions[loss] = dict(values=_readonly(values), available=_readonly(available),
                                unavailable_reasons=tuple(reasons), model_sha256=None)
    result = dict(schema_version=1, split=split, total_count=len(xy),
        fold_id=ids, fits=fits, predictions=predictions, constants=model_constants(),
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
