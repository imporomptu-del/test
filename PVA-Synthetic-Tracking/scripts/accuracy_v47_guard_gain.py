"""Exact rational necessary gain constraints from a prior-selected raw guard.

Only same-frame current pixels used by fixed integer second-difference stencils
are gathered. The 25-square core is never inspected here. Weights (1,-2,1)
annihilate an exact affine plane on each equally spaced horizontal/vertical
triplet. They do not fit a common plane or certify that such a plane exists.

For each contrast, declared errors imply |Y-g*B| <= Ey+g*EB, conditional on
g>=0 and a shared photometric model. Intersect (B+EB)g>=Y-Ey and
(B-EB)g<=Y+Ey using exact Fractions of the supplied binary floating values.
This is a necessary-constraint relaxation: overlap between stencils, arbitrary
error correlation and joint feasibility are not treated as independence.
The sparse grid can miss between-grid contamination. These straight horizontal
and vertical second differences also annihilate a bilinear x*y term, so their
consistency does not even establish an affine model over the guard itself.

Neither clean prior selection nor a nonempty interval certifies guard purity,
current target absence, true noise calibration, or gain transfer to the core.
The background must have been built from priors only (caller responsibility).
Missing/saturated source footprints must already be represented by NaN. Exact
0/255 values are ordinary finite DN here, not reinterpreted as saturation.
"""
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


WEIGHTS = (1, -2, 1)
PATCH_SHAPE = (129, 129)
HISTORY_SHAPE = (8, 129, 129)


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean is not a numerical constraint")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        # as_integer_ratio also avoids silently narrowing an extended-precision
        # NumPy floating scalar to binary64 before forming its exact rational.
        return Fraction(*value.as_integer_ratio())
    raise ValueError("Constraint values must be finite integers, floats, or Fractions")


def intersect_gain_halfspaces(halfspaces):
    """Intersect exact (coefficient, sense, rhs) constraints with g>=0.

    Returns Fraction lower/upper endpoints (upper=None means unbounded) and
    feasible. This low-level audit API intentionally returns exact rationals;
    estimate_guard_gain performs the JSON conversion and outward rounding.
    """
    lower, upper = Fraction(0), None
    count = 0
    for coefficient, sense, rhs in halfspaces:
        count += 1
        coefficient, rhs = _fraction(coefficient), _fraction(rhs)
        if sense not in (">=", "<="):
            raise ValueError("Halfspace sense must be >= or <=")
        if sense == "<=":
            coefficient, rhs = -coefficient, -rhs
        if coefficient == 0:
            if rhs > 0:
                return dict(feasible=False, lower=None, upper=None, processed_constraints=count)
        elif coefficient > 0:
            lower = max(lower, rhs/coefficient)
        else:
            endpoint = rhs/coefficient
            upper = endpoint if upper is None else min(upper, endpoint)
        if upper is not None and upper < lower:
            return dict(feasible=False, lower=None, upper=None, processed_constraints=count)
    return dict(feasible=True, lower=lower, upper=upper, processed_constraints=count)


def outward_float_interval(lower, upper):
    """Enclose finite exact endpoints with adjacent IEEE floats as necessary.

    Returns None if no finite floating interval can represent the enclosure.
    This conversion is directed; it is not a whole-image roundoff guarantee.
    """
    lower, upper = _fraction(lower), _fraction(upper)
    if lower > upper:
        raise ValueError("Interval lower endpoint exceeds upper")
    try:
        lo, hi = float(lower), float(upper)
    except OverflowError:
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if Fraction.from_float(lo) > lower:
        lo = math.nextafter(lo, -math.inf)
    if Fraction.from_float(hi) < upper:
        hi = math.nextafter(hi, math.inf)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [lo, hi]


def _array(value, shape, name):
    # Shape/type validation only: particularly do NOT scan current core values.
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    return array


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def estimate_guard_gain(current129, history129, background129, background_bound129,
                        prior_centers_xy, response_bound=.5):
    """Return a conditional outer gain interval, or explicit unknown.

    Current nonfinite values at any USED stencil point make the result unknown;
    no current-dependent support deletion, contrast choice or recentering occurs.
    Unused candidate points and all current core pixels are not inspected.
    """
    current = _array(current129, PATCH_SHAPE, "current129")
    history = _array(history129, HISTORY_SHAPE, "history129")
    background = _array(background129, PATCH_SHAPE, "background129")
    ub = _array(background_bound129, PATCH_SHAPE, "background_bound129")
    response_error = _fraction(response_bound)
    if response_error < 0:
        raise ValueError("response_bound must be nonnegative")
    if len(prior_centers_xy) != 8:
        raise ValueError("Exactly eight prior centers or None entries required")
    centers = []
    for point in prior_centers_xy:
        if point is None:
            centers.append(None)
            continue
        point = _array(point, (2,), "prior center")
        if not np.isfinite(point).all():
            raise ValueError("Missing prior center must be None")
        centers.append([float(v) for v in point])
    candidates = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                  if 40 <= max(abs(x-64), abs(y-64)) <= 56]
    eligible = []
    rejection_counts = dict(prior_foreground_footprint=0, nonfinite_prior_history=0,
                            nonfinite_background=0, invalid_background_bound=0)
    for x, y in candidates:
        rejected = False
        if any(point is not None and max(abs(x-point[0]), abs(y-point[1])) <= 12 for point in centers):
            rejection_counts["prior_foreground_footprint"] += 1
            rejected = True
        if not np.isfinite(history[:, y, x]).all():
            rejection_counts["nonfinite_prior_history"] += 1
            rejected = True
        if not np.isfinite(background[y, x]):
            rejection_counts["nonfinite_background"] += 1
            rejected = True
        if not np.isfinite(ub[y, x]) or ub[y, x] < 0:
            rejection_counts["invalid_background_bound"] += 1
            rejected = True
        if not rejected:
            eligible.append((x, y))
    eligible_set = set(eligible)
    stencils = []
    for axis, step in (("x", (8, 0)), ("y", (0, 8))):
        for x, y in candidates:
            points = [(x-step[0], y-step[1]), (x, y), (x+step[0], y+step[1])]
            if all(point in eligible_set for point in points):
                stencils.append(dict(axis=axis, center_xy=[x, y], pixels_xy=[list(p) for p in points],
                                     weights=list(WEIGHTS)))
    used = sorted({tuple(p) for s in stencils for p in s["pixels_xy"]}, key=lambda p:(p[1], p[0]))
    result = dict(available=False, reasons=[], gain_interval=None, exact_gain_interval=None,
                  candidate_count=len(candidates), eligible_count=len(eligible), used_count=len(used),
                  stencil_count=len(stencils), candidate_support_sha256=_hash(candidates),
                  eligible_support_sha256=_hash(eligible), used_support_sha256=_hash(used),
                  stencil_sha256=_hash(stencils), used_points_xy=[list(p) for p in used],
                  stencils=stencils, contrast_constraints=[], contrast_constraints_sha256=None,
                  prior_rejection_counts_nonexclusive=rejection_counts,
                  missing_prior_center_indices=[i for i,p in enumerate(centers) if p is None],
                  current_nonfinite_used_point_count=None, motion_status="unknown", physical_class="unknown",
                  provenance=dict(
                      support_selection_uses_prior_data_only=True,
                      current_values_accessed_only_at_used_stencil_points=True,
                      current_core_values_accessed=False, same_frame_guard_uses_current_pixels=True,
                      background_prior_only_construction_is_caller_assumption=True,
                      guard_to_core_shared_photometric_model_assumed=True,
                      guard_purity_or_current_target_absence_certified=False,
                      global_affine_fit_or_joint_model_feasibility_certified=False,
                      sparse_grid_can_miss_between_grid_contamination=True,
                      horizontal_vertical_contrasts_also_annihilate_bilinear_xy=True,
                      constraint_intersection_is_necessary_relaxation=True,
                      exact_binary_float_rational_arithmetic=True,
                      final_float_endpoints_outward_rounded=True,
                      noise_bound_is_declared_not_camera_calibrated=True,
                      no_independence_or_sample_count_reduction=True,
                      saturation_footprint_nan_mask_is_caller_responsibility=True,
                      finite_zero_and_255_are_not_automatically_saturation=True,
                      missing_prior_centers_do_not_establish_guard_purity=True,
                      response_bound_exact=str(response_error),
                      exclusion_chebyshev_radius=12, grid_step=8, radial_range=[40, 56],
                      unknown_spatial_gain_deformation_and_new_objects_not_covered=True,
                      no_physical_classification_or_production_gate=True))
    def unknown(reason):
        result["reasons"] = [reason]
        return result
    if not stencils:
        return unknown("no_prior_selected_guard_contrasts")
    # The only current-value gather in this function. Never read the full core.
    yy, xx = np.asarray([p[1] for p in used]), np.asarray([p[0] for p in used])
    observed = current[yy, xx]
    invalid = int((~np.isfinite(observed)).sum())
    result["current_nonfinite_used_point_count"] = invalid
    if invalid:
        return unknown("nonfinite_current_on_fixed_used_guard_support")
    current_values = {point:_fraction(value) for point, value in zip(used, observed)}
    constraints = []
    ey = sum(abs(w) for w in WEIGHTS)*response_error
    for stencil in stencils:
        points = [tuple(p) for p in stencil["pixels_xy"]]
        yc = sum(w*current_values[p] for w,p in zip(WEIGHTS, points))
        bc = sum(w*_fraction(background[y,x]) for w,(x,y) in zip(WEIGHTS, points))
        eb = sum(abs(w)*_fraction(ub[y,x]) for w,(x,y) in zip(WEIGHTS, points))
        result["contrast_constraints"].append(dict(response=str(yc), background=str(bc),
                                                     response_error=str(ey), background_error=str(eb)))
        constraints.extend(((bc+eb, ">=", yc-ey), (bc-eb, "<=", yc+ey)))
    result["contrast_constraints_sha256"] = _hash(result["contrast_constraints"])
    intersection = intersect_gain_halfspaces(constraints)
    if not intersection["feasible"]:
        return unknown("guard_gain_necessary_constraints_inconsistent")
    lower, upper = intersection["lower"], intersection["upper"]
    result["exact_gain_interval"] = [str(lower), None if upper is None else str(upper)]
    if upper is None:
        return unknown("guard_gain_outer_interval_unbounded")
    interval = outward_float_interval(lower, upper)
    if interval is None:
        return unknown("guard_gain_interval_not_finitely_representable")
    result.update(available=True, gain_interval=interval)
    return result
