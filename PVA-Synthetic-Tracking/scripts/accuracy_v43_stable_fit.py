"""Support-aware, bounded-perturbation engineering screen for least squares.

This is NOT a calibrated noise model, physical-motion test or production gate.
``sigma_dn`` is an absolute per-training-response perturbation bound, not a
standard deviation. The admitted held-out prediction perturbation is at most
``2 * sigma_dn``: the difference of two observations with that error bound.
The caller supplies mandatory elementwise absolute train/test DESIGN bounds.
Those bounds may cover propagated native +/-0.5 DN measurement quantization;
they do not magically cover unknown camera noise, registration, future scene
deformation, correspondence, anchor selection or support selection. Identities
of rows/columns/support and the geometry are held fixed during this screen.

Columns are scaled by their combined train/test L2 norm. This is merely a
change of units, applied equally to design bounds; it is not regularization.
Let A,B be the scaled train/test designs, U,V their elementwise absolute error
bounds, eta=||U||2 and q=smin(A)-eta. Elementwise |dA|<=U implies ||dA||2<=eta.
Thus q>0 establishes full column rank for every admitted perturbed A. A
separate machine-precision rank check rejects numerically deficient designs.

For beta=A+ y and residual r=y-A beta, an exact normal-equation identity gives
  ||d_beta_design|| <= eta*||beta||/q + eta*||r||/q**2.
The pseudoinverse identity for full-column-rank A and A'=A+dA gives
  ||A'+ - A+|| <= eta/q**2 + eta/(q*smin(A)).
The implementation combines these with exact response-only row-L1 leverage
||B A+||_1 and test-design bounds. The resulting conservative pixelwise bounds
cover simultaneous admitted response and design perturbations, including
nonlinear least-squares refitting; they are not a first-order covariance fit.

For a nonnegative last coefficient, both the unconstrained fit and the exact
zero-last boundary fit must pass. A robust coefficient-sign bound determines
whether the active set can change. If it can, both prediction intervals are
included. No unstable branch silently supplies a fallback score/prediction.
Unsupported, deficient or amplifying cases return NO operative coefficients
or predictions. Diagnostics are finite JSON-safe values; successful operative
arrays are returned separately for internal callers.
"""

import numpy as np


PREDICTION_BUDGET_MULTIPLIER = 2.0


def _numeric(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}-dimensional array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite on its declared common support")
    return array


def _bound(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric")
    try:
        array = np.broadcast_to(np.asarray(array, dtype=np.float64), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def _norm_columns(matrix):
    # np.linalg.norm squares directly; hypot.reduce avoids avoidable overflow
    # or underflow from otherwise finite representable columns.
    with np.errstate(over="ignore", under="ignore"):
        return np.hypot.reduce(matrix, axis=0)


def _vector_norm(vector):
    return float(np.hypot.reduce(np.ravel(vector), initial=0.0))


def _energy_list(norms):
    # Inputs of interest are native-DN/normalized designs. If squaring an
    # extreme representable unit choice overflows, report the norm separately
    # and use None instead of non-JSON-safe Infinity for that raw-unit energy.
    with np.errstate(over="ignore", under="ignore"):
        energy = norms*norms
    return [float(value) if np.isfinite(value) else None for value in energy]


def _failed(reasons, diagnostics):
    return {"available": False, "reasons": reasons, "diagnostics": diagnostics,
            "coefficients": None, "prediction": None, "prediction_bound": None}


def _branch(train, target, test, train_bound, test_bound, sigma_dn):
    n, p = train.shape
    m = test.shape[0]
    train_norm = _norm_columns(train)
    test_norm = _norm_columns(test)
    with np.errstate(over="ignore", under="ignore"):
        common_norm = np.hypot(train_norm, test_norm)
    if not all(np.isfinite(value).all() for value in (train_norm, test_norm, common_norm)):
        return _failed(["unrepresentable_column_scaling"],
                       {"training_rows": n, "test_rows": m, "columns": p,
                        "prediction_budget_dn": PREDICTION_BUDGET_MULTIPLIER*sigma_dn})
    diagnostics = {
        "training_rows": n, "test_rows": m, "columns": p,
        "train_column_l2_norm": train_norm.tolist(),
        "test_column_l2_norm": test_norm.tolist(),
        "common_column_l2_norm": common_norm.tolist(),
        "train_column_energy": _energy_list(train_norm),
        "test_column_energy": _energy_list(test_norm),
        "common_column_energy": _energy_list(common_norm),
        "scaled_train_column_energy": None, "scaled_test_column_energy": None,
        "train_singular_values": None, "test_singular_values": None,
        "common_singular_values": None, "machine_rank": None,
        "machine_rank_tolerance": None, "scaled_train_condition_number": None,
        "train_design_perturbation_spectral_bound": None,
        "robust_minimum_singular_value": None,
        "response_prediction_row_l1_leverage": None,
        "response_prediction_row_l2_leverage": None,
        "response_prediction_operator_l2_norm": None,
        "prediction_response_bound_dn": None, "prediction_design_bound_dn": None,
        "prediction_total_bound_dn": None, "maximum_prediction_bound_dn": None,
        "coefficient_perturbation_l2_bound_scaled": None,
        "residual_l2_norm_dn": None,
        "prediction_budget_dn": PREDICTION_BUDGET_MULTIPLIER*sigma_dn,
    }
    if p == 0:
        diagnostics.update(
            scaled_train_column_energy=[], scaled_test_column_energy=[],
            train_singular_values=[], test_singular_values=[], common_singular_values=[],
            machine_rank=0, machine_rank_tolerance=0.0,
            scaled_train_condition_number=None, train_design_perturbation_spectral_bound=0.0,
            response_prediction_row_l1_leverage=[0.0]*m,
            response_prediction_row_l2_leverage=[0.0]*m,
            response_prediction_operator_l2_norm=0.0,
            prediction_response_bound_dn=[0.0]*m, prediction_design_bound_dn=[0.0]*m,
            prediction_total_bound_dn=[0.0]*m, maximum_prediction_bound_dn=0.0,
            coefficient_perturbation_l2_bound_scaled=0.0,
            residual_l2_norm_dn=_vector_norm(target))
        return {"available": True, "reasons": [], "diagnostics": diagnostics,
                "coefficients": np.empty(0), "prediction": np.zeros(m),
                "scaled_coefficients": np.empty(0), "scales": np.empty(0)}
    if (common_norm == 0).any() or (train_norm == 0).any():
        return _failed(["unsupported_common_or_training_column"], diagnostics)
    a, b = train/common_norm, test/common_norm
    u, v = train_bound/common_norm, test_bound/common_norm
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        return _failed(["unrepresentable_scaled_design_uncertainty"], diagnostics)
    singular = np.linalg.svd(a, compute_uv=False)
    tolerance = float(np.finfo(float).eps * max(a.shape) * singular[0])
    rank = int((singular > tolerance).sum())
    with np.errstate(over="ignore", divide="ignore"):
        condition = singular[0]/singular[-1] if singular[-1] > 0 else np.inf
    diagnostics.update(
        scaled_train_column_energy=np.sum(a*a, axis=0).tolist(),
        scaled_test_column_energy=np.sum(b*b, axis=0).tolist(),
        train_singular_values=singular.tolist(),
        test_singular_values=np.linalg.svd(b, compute_uv=False).tolist(),
        common_singular_values=np.linalg.svd(np.vstack((a, b)), compute_uv=False).tolist(),
        machine_rank=rank, machine_rank_tolerance=tolerance,
        scaled_train_condition_number=float(condition) if np.isfinite(condition) else None)
    if rank != p:
        return _failed(["machine_rank_deficient"], diagnostics)
    with np.errstate(over="ignore", invalid="ignore"):
        eta = float(np.linalg.norm(u, ord=2))
    smin = float(singular[-1])
    q = smin-eta
    if not np.isfinite(eta) or not np.isfinite(q):
        return _failed(["unrepresentable_design_perturbation_norm"], diagnostics)
    diagnostics.update(train_design_perturbation_spectral_bound=eta,
                       robust_minimum_singular_value=q)
    if q <= 0:
        return _failed(["design_perturbation_can_destroy_rank"], diagnostics)
    beta = np.linalg.lstsq(a, target, rcond=None)[0]
    inverse = np.linalg.pinv(a, rcond=np.finfo(float).eps*max(a.shape))
    residual = target-a@beta
    residual_norm = _vector_norm(residual)
    beta_norm = _vector_norm(beta)
    h = b@inverse
    # inverse.T = Q R with orthonormal Q, so the nonzero singular values of
    # B inverse = (B R.T) Q.T equal those of the narrow matrix B R.T. Avoid a
    # large test-rows by train-rows SVD solely to report this diagnostic.
    _, inverse_r = np.linalg.qr(inverse.T, mode="reduced")
    prediction_operator_norm = float(np.linalg.norm(b@inverse_r.T, ord=2))
    row_l1 = np.sum(np.abs(h), axis=1)
    row_l2 = np.sqrt(np.sum(h*h, axis=1))
    row_b_norm = np.sqrt(np.sum(b*b, axis=1))
    row_v_norm = np.sqrt(np.sum(v*v, axis=1))
    response_norm = sigma_dn*np.sqrt(n)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_change_bound = eta/(q*q) + eta/(q*smin)
        beta_design_bound = eta*beta_norm/q + eta*residual_norm/(q*q)
        prediction_response = sigma_dn*row_l1 + row_b_norm*inverse_change_bound*response_norm + row_v_norm*response_norm/q
        prediction_design = (row_b_norm+row_v_norm)*beta_design_bound + row_v_norm*beta_norm
        total = prediction_response+prediction_design
        beta_total_bound = response_norm/q + beta_design_bound
    if not (np.isfinite(total).all() and np.isfinite(beta_total_bound)):
        return _failed(["unbounded_or_unrepresentable_prediction_sensitivity"], diagnostics)
    diagnostics.update(
        response_prediction_row_l1_leverage=row_l1.tolist(),
        response_prediction_row_l2_leverage=row_l2.tolist(),
        response_prediction_operator_l2_norm=prediction_operator_norm,
        prediction_response_bound_dn=prediction_response.tolist(),
        prediction_design_bound_dn=prediction_design.tolist(),
        prediction_total_bound_dn=total.tolist(), maximum_prediction_bound_dn=float(total.max()),
        coefficient_perturbation_l2_bound_scaled=float(beta_total_bound),
        residual_l2_norm_dn=residual_norm)
    if np.any(total > PREDICTION_BUDGET_MULTIPLIER*sigma_dn):
        return _failed(["prediction_perturbation_exceeds_engineering_budget"], diagnostics)
    coefficients, prediction = beta/common_norm, b@beta
    if not np.isfinite(coefficients).all() or not np.isfinite(prediction).all():
        return _failed(["unrepresentable_operative_fit"], diagnostics)
    return {"available": True, "reasons": [], "diagnostics": diagnostics,
            "coefficients": coefficients, "prediction": prediction,
            "scaled_coefficients": beta, "scales": common_norm}


def supported_fit(train, target, test, sigma_dn, nonnegative_last=False, *,
                  train_design_bound=None, test_design_bound=None):
    """Fit only when the predeclared perturbation screen is supported.

    Designs are two-dimensional and share columns; ``target`` is one-dimensional.
    Absolute design bounds broadcast to their design shapes (e.g. a per-column
    vector). Use zero ONLY when that design is intentionally treated as exact.
    Missing uncertainty for a nonempty design is unavailable, not zero error.
    Successful coefficient/prediction arrays are new and inputs are not edited.
    """
    train = _numeric(train, 2, "train")
    test = _numeric(test, 2, "test")
    target = _numeric(target, 1, "target")
    if train.shape[0] == 0 or test.shape[0] == 0 or train.shape[0] != len(target) or train.shape[1] != test.shape[1]:
        raise ValueError("nonempty train/test rows, matching columns and target length are required")
    if isinstance(sigma_dn, (bool, np.bool_)) or not isinstance(sigma_dn, (int, float, np.integer, np.floating)) or not np.isfinite(sigma_dn) or sigma_dn < 0:
        raise ValueError("sigma_dn must be a finite nonnegative absolute perturbation bound")
    if float(sigma_dn) > np.finfo(float).max/PREDICTION_BUDGET_MULTIPLIER:
        raise ValueError("sigma_dn must have a representable engineering budget")
    if not isinstance(nonnegative_last, (bool, np.bool_)):
        raise ValueError("nonnegative_last must be Boolean")
    if nonnegative_last and train.shape[1] == 0:
        raise ValueError("a last coefficient requires a nonempty design")
    sigma_dn = float(sigma_dn)
    diagnostics = {
        "screen": "bounded_measurement_perturbation_only_not_physical_confidence",
        "response_perturbation_bound_dn": sigma_dn,
        "prediction_budget_dn": PREDICTION_BUDGET_MULTIPLIER*sigma_dn,
        "uncertainty_excludes": ["unknown_camera_noise", "registration_error", "temporal_deformation",
                                 "correspondence_error", "geometry_anchor_or_support_identity_changes"],
        "nonnegative_last": bool(nonnegative_last), "free_fit": None, "boundary_fit": None,
        "constraint_boundary_may_change_under_perturbation": None,
        "constraint_active_at_observed_inputs": None,
        "prediction_total_bound_dn": None, "maximum_prediction_bound_dn": None,
    }
    if train.shape[1] and (train_design_bound is None or test_design_bound is None):
        return _failed(["design_uncertainty_not_supplied"], diagnostics)
    train_bound = _bound(0.0 if train_design_bound is None else train_design_bound, train.shape, "train_design_bound")
    test_bound = _bound(0.0 if test_design_bound is None else test_design_bound, test.shape, "test_design_bound")
    free = _branch(train, target, test, train_bound, test_bound, sigma_dn)
    diagnostics["free_fit"] = free["diagnostics"]
    if not free["available"]:
        return _failed([f"free_fit:{reason}" for reason in free["reasons"]], diagnostics)
    if not nonnegative_last:
        diagnostics.update(prediction_total_bound_dn=free["diagnostics"]["prediction_total_bound_dn"],
                           maximum_prediction_bound_dn=free["diagnostics"]["maximum_prediction_bound_dn"])
        return {"available": True, "reasons": [], "diagnostics": diagnostics,
                "coefficients": free["coefficients"], "prediction": free["prediction"],
                "prediction_bound": np.asarray(free["diagnostics"]["prediction_total_bound_dn"])}
    boundary = _branch(train[:, :-1], target, test[:, :-1], train_bound[:, :-1], test_bound[:, :-1], sigma_dn)
    diagnostics["boundary_fit"] = boundary["diagnostics"]
    if not boundary["available"]:
        return _failed([f"boundary_fit:{reason}" for reason in boundary["reasons"]], diagnostics)
    active = bool(free["scaled_coefficients"][-1] < 0)
    sign_margin = abs(float(free["scaled_coefficients"][-1]))
    sign_error = free["diagnostics"]["coefficient_perturbation_l2_bound_scaled"]
    may_change = bool(sign_margin <= sign_error)
    chosen = boundary if active else free
    chosen_bound = np.asarray(chosen["diagnostics"]["prediction_total_bound_dn"])
    if may_change:
        other = free if active else boundary
        other_bound = np.asarray(other["diagnostics"]["prediction_total_bound_dn"])
        chosen_bound = np.maximum(chosen_bound, np.abs(other["prediction"]-chosen["prediction"])+other_bound)
    diagnostics.update(
        constraint_boundary_may_change_under_perturbation=may_change,
        constraint_active_at_observed_inputs=active,
        prediction_total_bound_dn=chosen_bound.tolist(), maximum_prediction_bound_dn=float(chosen_bound.max()))
    if np.any(chosen_bound > PREDICTION_BUDGET_MULTIPLIER*sigma_dn):
        return _failed(["constraint_branch_union_exceeds_engineering_budget"], diagnostics)
    coefficients = np.r_[boundary["coefficients"], 0.0] if active else free["coefficients"]
    return {"available": True, "reasons": [], "diagnostics": diagnostics,
            "coefficients": coefficients, "prediction": chosen["prediction"],
            "prediction_bound": chosen_bound}
