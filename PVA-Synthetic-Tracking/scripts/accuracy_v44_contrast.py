"""Synthetic-stage source contrast under fixed-support deterministic uncertainty.

This estimates a SIGNED least-squares source coefficient, not motion, identity,
airborne class, probability, or a production acceptance decision. The exact
caller-supplied affine subspace is projected out first. Uncertain nuisance/source
designs and responses can then change simultaneously and refit. The full interval
is returned without an arbitrary absolute-background prediction budget.

Derivation: docs/accuracy_v44_plan.md. All geometry, rows, columns, affine span and
template membership are held fixed. No camera noise/model-error calibration and
no causal-placement verification is supplied by this vector-level function.
"""
import numpy as np


def _array(value, ndim, name):
    result = np.asarray(value)
    if result.ndim != ndim or result.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite on fixed common support")
    return result


def _bounds(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    try:
        result = np.broadcast_to(array.astype(float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to its input") from exc
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _norm(value):
    return float(np.hypot.reduce(np.ravel(value), initial=0.0))


def source_contrast(y, Z, m, P, *, response_bound, nuisance_bound, source_bound):
    """Return a conditional source-coefficient interval or explicit unknown.

    y,m are N-vectors, Z is NxK nuisance design and P is NxJ exact affine basis.
    Bounds are elementwise absolute errors, not standard deviations; they broadcast
    to the matching input. Nuisance coefficients are signed and unrestricted.
    The returned coefficient is in the units of caller-supplied m; no nonnegative
    constraint or clipping conceals a zero-crossing interval. Inputs are not edited.
    """
    y, Z, m, P = (_array(v, d, name) for v, d, name in
                  ((y, 1, "y"), (Z, 2, "Z"), (m, 1, "m"), (P, 2, "P")))
    n = len(y)
    if n == 0 or len(m) != n or len(Z) != n or len(P) != n:
        raise ValueError("nonempty matched row counts required")
    result = dict(available=False, reasons=[], estimate=None, error_bound=None,
                  interval=None, interval_excludes_zero=None, coefficient_sign=None,
                  motion_status="unknown", physical_class="unknown", diagnostics={})
    diag = result["diagnostics"]
    diag.update(rows=n, original_nuisance_columns=Z.shape[1], exact_affine_columns=P.shape[1],
                perturbation_contract="fixed-support deterministic simultaneous response/design refits",
                caller_template_temporal_or_placement_provenance_certified=False,
                calibrated_noise_or_confidence=False, numerical_roundoff_in_error_set=False,
                no_production_gate=True)

    def unknown(reason):
        result["reasons"] = [reason]
        return result

    if response_bound is None or source_bound is None or (Z.shape[1] and nuisance_bound is None):
        return unknown("missing_declared_uncertainty")
    ey = _bounds(response_bound, y.shape, "response_bound")
    uz = _bounds(0 if nuisance_bound is None else nuisance_bound, Z.shape, "nuisance_bound")
    um = _bounds(source_bound, m.shape, "source_bound")
    # Scaling P changes no span and avoids arbitrary caller basis units.
    if P.shape[1]:
        norms = np.hypot.reduce(P, axis=0)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            return unknown("unsupported_exact_affine_basis")
        pu, ps, _ = np.linalg.svd(P / norms, full_matrices=False)
        ptol = np.finfo(float).eps * max(P.shape) * ps[0]
        if len(ps) < P.shape[1] or np.count_nonzero(ps > ptol) != P.shape[1]:
            return unknown("rank_deficient_exact_affine_basis")
        basis = pu[:, :P.shape[1]]
    else:
        basis = np.empty((n, 0))

    def project(value):
        return value - basis @ (basis.T @ value)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = np.column_stack((Z, m))
        errors = np.column_stack((uz, um))
        projected = project(raw)
        target = project(y)
        raw_norms = np.hypot.reduce(raw, axis=0)
        projected_norms = np.hypot.reduce(projected, axis=0)
    if not all(np.isfinite(v).all() for v in (raw_norms, projected_norms, projected, target)):
        return unknown("unrepresentable_projection_or_column_norm")
    # Tiny nonzero departures from the affine span can carry a source when
    # nuisance coefficients are unrestricted. A tolerance is NOT proof of
    # redundancy: remove only literal zero or +/- an identical supplied basis
    # column, and only with exactly zero declared uncertainty.
    roundoff = np.finfo(float).eps * max(n, P.shape[1], 1) * raw_norms
    redundant = [j for j in range(Z.shape[1]) if np.all(errors[:, j] == 0)
                 and (np.all(raw[:, j] == 0) or any(
                     np.array_equal(raw[:, j], sign*P[:, k])
                     for k in range(P.shape[1]) for sign in (-1, 1)))]
    keep = [j for j in range(raw.shape[1]) if j not in redundant]
    diag["exact_nuisance_columns_removed_in_affine_span"] = redundant
    if projected_norms[-1] <= roundoff[-1]:
        return unknown("source_in_numerical_affine_span")
    if any(projected_norms[j] <= roundoff[j] for j in keep[:-1]):
        return unknown("near_affine_nuisance_not_certified_redundant")
    scales = projected_norms[keep]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a, u = projected[:, keep] / scales, errors[:, keep] / scales
    if not np.isfinite(u).all() or not np.isfinite(a).all():
        return unknown("unrepresentable_scaled_design")
    left, singular, right = np.linalg.svd(a, full_matrices=False)
    rank_tol = np.finfo(float).eps * max(a.shape) * singular[0]
    diag.update(retained_columns=keep, projected_column_scales=scales.tolist(),
                scaled_projected_singular_values=singular.tolist(),
                machine_rank_tolerance=float(rank_tol),
                raw_design_bound_spectral_norm=None, robust_minimum_singular_value=None)
    if len(singular) < a.shape[1] or np.count_nonzero(singular > rank_tol) != a.shape[1]:
        return unknown("projected_design_machine_rank_deficient")
    with np.errstate(over="ignore", invalid="ignore"):
        eta = float(np.linalg.norm(u, ord=2))
    q = float(singular[-1] - eta)
    if not np.isfinite(eta) or not np.isfinite(q):
        return unknown("unrepresentable_design_uncertainty_norm")
    diag.update(raw_design_bound_spectral_norm=eta, robust_minimum_singular_value=q)
    diag["scaled_error_column_l2_norm"] = np.hypot.reduce(u, axis=0).tolist()
    if q <= 0:
        return unknown("projected_design_robust_rank_not_certified")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = (right.T / singular) @ left.T
        theta = inverse @ target
        residual = target - a @ theta
        h = inverse[-1]
        # a z=h as a column, computed from SVD rather than normal inversion.
        z = right.T @ (right[:, -1] / (singular * singular))
        column_component = _norm(u.T @ np.abs(h)) / q
        orthogonal_component = _norm(u @ np.abs(z))
        dual_row = float(np.hypot(column_component, orthogonal_component))
        dual_global = float(eta/(q*q) + eta/(q*singular[-1]))
        dual_change = min(dual_row, dual_global)
        design_response_error = u @ np.abs(theta)
        weighted = float(np.abs(h) @ (ey + design_response_error))
        remainder = dual_change * (_norm(residual) + _norm(ey) + _norm(design_response_error))
        coefficient = float(theta[-1] / scales[-1])
        bound = float((weighted + remainder) / scales[-1])
        low, high = coefficient - bound, coefficient + bound
    checks = [column_component, orthogonal_component, dual_row, dual_global,
              weighted, remainder, coefficient, bound, low, high]
    if not np.isfinite(checks).all() or not all(np.isfinite(v).all() for v in (theta, residual, h, z)):
        return unknown("unrepresentable_contrast_sensitivity")
    diag.update(nominal_scaled_coefficients=theta.tolist(),
                residual_l2_norm=_norm(residual), source_column_scale=float(scales[-1]),
                nominal_dual_l1_norm=float(np.abs(h).sum()),
                nominal_dual_l2_norm=_norm(h), dual_change_global_l2_bound=dual_global,
                dual_change_selected_row_l2_bound=dual_row, dual_change_used_l2_bound=dual_change,
                nominal_weighted_error_in_coefficient_units=float(weighted/scales[-1]),
                nominal_response_error_in_coefficient_units=float(np.abs(h) @ ey/scales[-1]),
                nominal_nuisance_error_in_coefficient_units=float(
                    np.abs(h) @ (u[:, :-1] @ np.abs(theta[:-1]))/scales[-1]),
                nominal_source_error_in_coefficient_units=float(
                    np.abs(h) @ (u[:, -1] * abs(theta[-1]))/scales[-1]),
                changed_dual_error_in_coefficient_units=float(remainder/scales[-1]),
                affine_annihilation_residual_norm=_norm(h @ basis))
    result.update(available=True, estimate=coefficient, error_bound=bound, interval=[low, high],
                  interval_excludes_zero=bool(low > 0 or high < 0),
                  coefficient_sign="positive" if low > 0 else "negative" if high < 0 else "unresolved")
    return result
