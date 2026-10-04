"""Conditional nuisance-residualized numerator sign, not object classification.

Exact affine brightness P is removed first by Q=I-P P+. Let N=QZ (with fixed
projected-column scales), Pi=N N+, R=I-Pi, b=Qy, t=Qm, w=Rt and r=Rb.
The signed least-squares source coefficient has numerator s=t.T R b and
denominator ||w||². We bound the numerator, never divide by the uncertain
denominator or impose an arbitrary absolute-background prediction budget.

For admitted source, response and nuisance errors dm,e,E, the exact change is
  ds = w.T e + r.T dm + dm.T Q R Q e
       - (t+Qdm).T (Pi'-Pi) (b+Qe).
Both w and r annihilate P, so their weighted terms use the original unprojected
entrywise error bounds. No independence or averaging reduction is assumed.

If eta=||U||2 and q=smin(N)-eta>0, every admitted N'=N+QE has the same full
column rank. Since (I-Pi')Pi=-(I-Pi')QE N+, its norm is at most eta/smin(N).
For equal-rank orthogonal projectors, that cross-projector norm equals
||Pi'-Pi||2 (the largest sine of their principal angles). Thus the projector
gap is bounded by g=min(1,eta/smin(N)); with no nuisance columns the gap is0.

The projector bilinear term is bounded both by g||t+Qdm||||b+Qe|| and by a
blockwise expression. In the nominal Pi/R decomposition, diagonal blocks of
Pi'-Pi have norms at most g², and off-diagonal blocks at most g. Inflate each
nominal projected/residual vector norm by its admitted full L2 error norm, and
take the smaller of the two valid bounds. This is algebra, not threshold tuning.

A strict positive/negative numerator interval also excludes a zero perturbed
residualized source (which would make the numerator0). It conditionally
establishes only the fitted coefficient's sign, never motion, identity, airborne
class, statistical confidence or a detection/rejection policy. Affine span,
geometry, support, template/column membership and error sets are held fixed.
Floating-point roundoff is not a calibrated or admitted image-noise model.

Separately, a numerical sign-resolution guard can only withhold a sign. With
d=max(row/column dimensions), gamma=d*eps/(1-d*eps), kappaP the condition of
column-normalized P (0 when absent), and cZ=||raw Z/scales||F/smin(N) (0 when
absent), its margin is gamma*||raw m||*||raw y||*(1+kappaP)*(1+cZ). The raw
norms retain cancellation from large affine brightness. cZ includes both
raw-to-projected amplification and nuisance conditioning; the product accounts
for affine-basis error propagating through nuisance projection. This is an
operation-scale, backward-error-inspired decline guard, NOT a proved IEEE
roundoff enclosure. The analytic interval is reported separately; the decision
interval adds this nonnegative numerical margin, never camera-noise tuning.
"""
import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value)
    if value.ndim != ndim or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite on its fixed support")
    return value


def _bound(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric")
    try:
        value = np.broadcast_to(value.astype(float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to its input") from exc
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _norm(value):
    return float(np.hypot.reduce(np.ravel(value), initial=0.0))


def source_presence(y, Z, m, P, *, response_bound, nuisance_bound, source_bound):
    """Return available/numerator/error_bound/interval/sign or explicit unknown.

    y,m are N-vectors, Z is NxK uncertain nuisance and P is NxJ exact affine
    basis. Bounds are absolute entrywise errors, not standard deviations.
    Nuisance coefficients are unrestricted. No current-frame recentering,
    template discovery, learned threshold or amplitude clipping is performed.
    """
    y, Z, m, P = (_array(value, ndim, name) for value, ndim, name in
                  ((y, 1, "y"), (Z, 2, "Z"), (m, 1, "m"), (P, 2, "P")))
    n = len(y)
    if not n or len(Z) != n or len(m) != n or len(P) != n:
        raise ValueError("Nonempty matching row counts required")
    result = dict(available=False, reasons=[], numerator=None, error_bound=None,
                  analytic_error_bound=None, analytic_interval=None,
                  numerical_resolution_margin=None,
                  interval=None, interval_excludes_zero=None, coefficient_sign=None,
                  motion_status="unknown", physical_class="unknown", diagnostics={})
    diag = result["diagnostics"]
    diag.update(rows=n, original_nuisance_columns=Z.shape[1], exact_affine_columns=P.shape[1],
                perturbation_contract="fixed-support deterministic simultaneous response/source/nuisance refits",
                source_amplitude_denominator_not_divided=True,
                nuisance_coefficients_unrestricted=True, no_independence_assumption=True,
                numerical_roundoff_in_error_set=False, calibrated_noise_or_confidence=False,
                numerical_resolution_is_ieee_certified_enclosure=False,
                caller_template_temporal_or_placement_provenance_certified=False,
                no_production_gate=True)
    def unknown(reason):
        result["reasons"] = [reason]
        return result

    if response_bound is None or source_bound is None or (Z.shape[1] and nuisance_bound is None):
        return unknown("missing_declared_uncertainty")
    ey = _bound(response_bound, y.shape, "response_bound")
    uz = _bound(0. if nuisance_bound is None else nuisance_bound, Z.shape, "nuisance_bound")
    um = _bound(source_bound, m.shape, "source_bound")
    if P.shape[1]:
        norms = np.hypot.reduce(P, axis=0)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            return unknown("unsupported_exact_affine_basis")
        basis, singular, _ = np.linalg.svd(P/norms, full_matrices=False)
        tolerance = np.finfo(float).eps*max(P.shape)*singular[0]
        if len(singular) < P.shape[1] or int((singular > tolerance).sum()) != P.shape[1]:
            return unknown("rank_deficient_exact_affine_basis")
        affine = basis[:, :P.shape[1]]
        affine_condition = float(singular[0]/singular[-1])
    else:
        affine = np.empty((n, 0))
        affine_condition = 0.0
    def project(value): return value-affine@(affine.T@value)

    with np.errstate(over="ignore", invalid="ignore"):
        b, t, nuisance = project(y), project(m), project(Z)
        raw_norms = np.hypot.reduce(Z, axis=0)
        projected_norms = np.hypot.reduce(nuisance, axis=0)
        source_norm, response_norm = _norm(t), _norm(b)
        source_raw_norm = _norm(m)
        response_raw_norm = _norm(y)
    if not all(np.isfinite(value).all() for value in
               (b, t, nuisance, raw_norms, projected_norms, source_norm, response_norm,
                source_raw_norm, response_raw_norm, affine_condition)):
        return unknown("unrepresentable_projection_or_column_norm")
    redundant = [j for j in range(Z.shape[1]) if np.all(uz[:, j] == 0) and (
        np.all(Z[:, j] == 0) or any(np.array_equal(Z[:, j], sign*P[:, k])
                                  for k in range(P.shape[1]) for sign in (-1, 1)))]
    keep = [j for j in range(Z.shape[1]) if j not in redundant]
    diag.update(exact_nuisance_columns_removed_in_affine_span=redundant,
                retained_nuisance_columns=keep)
    roundoff_factor = np.finfo(float).eps*max(n, P.shape[1], 1)
    if source_norm <= roundoff_factor*source_raw_norm:
        return unknown("source_in_numerical_affine_span")
    if any(projected_norms[j] <= roundoff_factor*raw_norms[j] for j in keep):
        return unknown("near_affine_nuisance_not_certified_redundant")
    if keep:
        scales = projected_norms[keep]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            nominal = nuisance[:, keep]/scales
            bound = uz[:, keep]/scales
        if not np.isfinite(nominal).all() or not np.isfinite(bound).all():
            return unknown("unrepresentable_scaled_nuisance_design")
        left, singular, _ = np.linalg.svd(nominal, full_matrices=False)
        tolerance = np.finfo(float).eps*max(nominal.shape)*singular[0]
        diag.update(projected_nuisance_column_scales=scales.tolist(),
                    nuisance_singular_values=singular.tolist(),
                    nuisance_machine_rank_tolerance=float(tolerance),
                    nuisance_design_error_spectral_norm=None, nuisance_robust_minimum_singular_value=None)
        if len(singular) < len(keep) or int((singular > tolerance).sum()) != len(keep):
            return unknown("projected_nuisance_machine_rank_deficient")
        with np.errstate(over="ignore", invalid="ignore"):
            eta = float(np.linalg.norm(bound, ord=2))
        minimum = float(singular[-1])
        q = minimum-eta
        if not np.isfinite(eta) or not np.isfinite(q):
            return unknown("unrepresentable_nuisance_uncertainty_norm")
        diag.update(nuisance_design_error_spectral_norm=eta,
                    nuisance_robust_minimum_singular_value=q)
        if q <= 0:
            return unknown("projected_nuisance_robust_rank_not_certified")
        nuisance_basis = left[:, :len(keep)]
        gap = min(1.0, eta/minimum)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            nuisance_numerical_amplification = _norm(raw_norms[keep]/scales)/minimum
    else:
        nuisance_basis = np.empty((n, 0))
        gap = 0.0
        nuisance_numerical_amplification = 0.0
        diag.update(projected_nuisance_column_scales=[], nuisance_singular_values=[],
                    nuisance_machine_rank_tolerance=0., nuisance_design_error_spectral_norm=0.,
                    nuisance_robust_minimum_singular_value=None)
    source_nuisance = nuisance_basis@(nuisance_basis.T@t)
    response_nuisance = nuisance_basis@(nuisance_basis.T@b)
    w, r = t-source_nuisance, b-response_nuisance
    w_norm, r_norm = _norm(w), _norm(r)
    if not all(np.isfinite(value).all() for value in (w, r, w_norm, r_norm)):
        return unknown("unrepresentable_nuisance_residualization")
    if w_norm <= np.finfo(float).eps*max(n, len(keep), 1)*source_norm:
        return unknown("source_in_nominal_nuisance_span")
    with np.errstate(over="ignore", invalid="ignore"):
        source_error, response_error = _norm(um), _norm(ey)
        first = float(np.abs(w)@ey)
        second = float(np.abs(r)@um)
        cross = source_error*response_error
        source_in_nuisance_norm = _norm(source_nuisance)
        response_in_nuisance_norm = _norm(response_nuisance)
        if gap == 0:
            global_projector = block_projector = 0.0
        else:
            global_projector = gap*(source_norm+source_error)*(response_norm+response_error)
            block_projector = (
                gap*gap*((source_in_nuisance_norm+source_error)*(response_in_nuisance_norm+response_error)
                         +(w_norm+source_error)*(r_norm+response_error))
                +gap*((source_in_nuisance_norm+source_error)*(r_norm+response_error)
                      +(w_norm+source_error)*(response_in_nuisance_norm+response_error)))
        projector = min(global_projector, block_projector)
        analytic_total = first+second+cross+projector
        numerator = float(t@r)
        dimension = max(n, P.shape[1], len(keep), 1)
        eps_dimension = np.finfo(float).eps*dimension
        gamma = eps_dimension/(1-eps_dimension)
        numerical_margin = (gamma*source_raw_norm*response_raw_norm
                            *(1+affine_condition)*(1+nuisance_numerical_amplification))
        total = analytic_total+numerical_margin
        analytic_low, analytic_high = numerator-analytic_total, numerator+analytic_total
        low, high = numerator-total, numerator+total
    values = [source_error, response_error, first, second, cross, global_projector, block_projector,
              projector, total, numerator, low, high, source_in_nuisance_norm, response_in_nuisance_norm,
              analytic_total, analytic_low, analytic_high, numerical_margin, gamma,
              nuisance_numerical_amplification]
    if not np.isfinite(values).all():
        return unknown("unrepresentable_numerator_sensitivity")
    diag.update(nuisance_projector_gap_bound=gap,
                projector_gap_derivation="Equal-rank orthogonal projector gap equals one-sided cross-projector norm; at most eta/smin.",
                projected_source_l2_norm=source_norm, projected_response_l2_norm=response_norm,
                residualized_source_l2_norm=w_norm, nuisance_response_residual_l2_norm=r_norm,
                source_in_nuisance_l2_norm=source_in_nuisance_norm,
                response_in_nuisance_l2_norm=response_in_nuisance_norm,
                source_error_l2_bound=source_error, response_error_l2_bound=response_error,
                nominal_weighted_response_term=first, nominal_weighted_source_term=second,
                source_response_cross_term=cross, projector_global_product_bound=global_projector,
                projector_block_product_bound=block_projector, projector_product_bound_used=projector,
                projector_bound_selected="global" if global_projector <= block_projector else "blockwise",
                numerical_resolution_formula="gamma_d * norm(raw_m) * norm(raw_y) * (1 + condition(column_normalized_P)) * (1 + norm(raw_Z / projected_scales, Frobenius) / smin(normalized_QZ))",
                numerical_resolution_dimension=dimension, numerical_resolution_gamma=gamma,
                normalized_affine_condition=affine_condition,
                nuisance_numerical_condition_amplification=nuisance_numerical_amplification,
                raw_source_l2_norm=source_raw_norm, raw_response_l2_norm=response_raw_norm,
                numerical_margin_can_only_withhold_sign=True,
                analytic_interval_excludes_zero=bool(analytic_low > 0 or analytic_high < 0),
                sign_withheld_by_numerical_resolution=bool(
                    (analytic_low > 0 or analytic_high < 0) and not (low > 0 or high < 0)),
                residualized_source_affine_annihilation_norm=_norm(w@affine),
                residualized_response_affine_annihilation_norm=_norm(r@affine))
    result.update(available=True, numerator=numerator, error_bound=total, interval=[low, high],
                  analytic_error_bound=analytic_total, analytic_interval=[analytic_low, analytic_high],
                  numerical_resolution_margin=numerical_margin,
                  interval_excludes_zero=bool(low > 0 or high < 0),
                  coefficient_sign="positive" if low > 0 else "negative" if high < 0 else "unresolved")
    return result
