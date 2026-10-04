"""Tighter conditional component bounds using nonnegative normalization boxes.

The nominal components, aligned-input +/-0.5 DN experiment, +/-1 DN residual
and highpass bounds, finite support, geometry, anchor identities and exact used
stamp membership are UNCHANGED from V43. This changes only outer-bound algebra.
It introduces neither independence averaging nor a relaxed noise/decision budget.

For a nonnegative box L<=v<=U, normalized coordinate v_i/||v|| increases with
v_i and decreases with every other coordinate. Hence its exact marginal extrema
on this box are L_i/hypot(L_i,||U_without_i||) and
U_i/hypot(U_i,||L_without_i||). Coordinates' extrema generally occur at different
vectors; they are NOT a jointly attainable normalized vector or a probability.
The box norm lower bound must exceed the unchanged acceptance threshold.

Coordinatewise median is monotone. Median endpoint arrays therefore enclose the
median of the fixed used stamps; the same normalization-box formula then bounds
its final unit template. No uncertain used stamp may be discarded. Bilinear
placement has nonnegative weights and propagates both endpoints monotonically.

These are mathematical, fixed-membership measurement-sensitivity bounds. They do
not cover unknown camera noise, registration/deformation, altered selection or
physical identity. Floating operations outward-round each prefix/suffix hypot,
the combining hypot, and endpoint division. This is a local numerical enclosure
guard, not a general directed-rounding backend for the complete image pipeline.
"""

import numpy as np

import accuracy_v42_localized as legacy
import accuracy_v43_bounds as prior


ALIGNED_VALUE_BOUND_DN = prior.ALIGNED_VALUE_BOUND_DN
RESIDUAL_STAMP_BOUND_DN = prior.RESIDUAL_STAMP_BOUND_DN
HIGHPASS_STAMP_BOUND_DN = prior.HIGHPASS_STAMP_BOUND_DN
MIN_NORMALIZATION_NORM = prior.MIN_NORMALIZATION_NORM
APERTURE = prior.APERTURE


def _rounded_hypot(first, second, upward):
    """Outward-round a norm primitive; adding exact zero stays exact."""
    first, second = np.asarray(first), np.asarray(second)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        value = np.hypot(first, second)
        rounded = np.nextafter(value, np.inf if upward else -np.inf)
    rounded = np.maximum(rounded, 0.0)
    return np.where(first == 0, second, np.where(second == 0, first, rounded))


def _prefix_norm_intervals(vector):
    values = np.ravel(vector)
    lower = np.zeros(len(values)+1)
    upper = np.zeros(len(values)+1)
    for index, value in enumerate(values):
        lower[index+1] = _rounded_hypot(lower[index], value, False)
        upper[index+1] = _rounded_hypot(upper[index], value, True)
    return lower, upper


def _other_coordinate_norms(vector):
    """Avoid subtracting a dominant squared coordinate from a total square."""
    vector = np.asarray(vector, dtype=float)
    prefix_lower, prefix_upper = _prefix_norm_intervals(vector)
    reverse_lower, reverse_upper = _prefix_norm_intervals(vector[::-1])
    suffix_lower, suffix_upper = reverse_lower[::-1], reverse_upper[::-1]
    return (_rounded_hypot(prefix_lower[:-1], suffix_lower[1:], False),
            _rounded_hypot(prefix_upper[:-1], suffix_upper[1:], True))


def normalization_interval(lower, upper, minimum_norm=MIN_NORMALIZATION_NORM):
    """Return (normalized_lower, normalized_upper, metadata), or None endpoints.

    Inputs have matching finite/NaN support and nonnegative ordered endpoints.
    Returned intervals preserve that support. Exact zero coordinates remain zero.
    A lower norm at/below the original acceptance threshold is unavailable, not
    permission to delete a nominally used stamp or fill missing pixels.
    """
    lo, hi = np.asarray(lower), np.asarray(upper)
    if lo.shape != hi.shape or lo.dtype.kind not in "iuf" or hi.dtype.kind not in "iuf":
        raise ValueError("Matching real numeric lower/upper arrays required")
    lo, hi = lo.astype(float), hi.astype(float)
    if np.isinf(lo).any() or np.isinf(hi).any():
        raise ValueError("Infinity is not an interval endpoint")
    support = np.isfinite(lo)
    if not np.array_equal(support, np.isfinite(hi)):
        raise ValueError("Interval endpoints must have identical finite support")
    if np.any(lo[support] < 0) or np.any(hi[support] < lo[support]):
        raise ValueError("Nonnegative ordered interval endpoints required")
    if (isinstance(minimum_norm, (bool, np.bool_))
            or not isinstance(minimum_norm, (int, float, np.integer, np.floating))
            or not np.isfinite(minimum_norm) or minimum_norm < 0):
        raise ValueError("Finite nonnegative minimum normalization norm required")
    low_values, high_values = lo[support], hi[support]
    low_norm = float(_prefix_norm_intervals(low_values)[0][-1])
    high_norm = float(_prefix_norm_intervals(high_values)[1][-1])
    metadata = {
        "available": False, "reason": None, "finite_support_count": int(support.sum()),
        "norm_lower_bound": low_norm if np.isfinite(low_norm) else None,
        "norm_upper_bound": high_norm if np.isfinite(high_norm) else None,
        "minimum_accepted_norm": float(minimum_norm),
        "method": "exact_marginal_extrema_of_nonnegative_normalization_box",
        "joint_attainability_claimed": False,
    }
    if not np.isfinite(low_norm) or not np.isfinite(high_norm):
        metadata["reason"] = "unrepresentable_normalization_box_norm"
        return None, None, metadata
    if low_norm <= minimum_norm:
        metadata["reason"] = "normalization_box_lower_norm_not_above_acceptance_threshold"
        return None, None, metadata
    others_upper = _other_coordinate_norms(high_values)[1]
    others_lower = _other_coordinate_norms(low_values)[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower_values = low_values/_rounded_hypot(low_values, others_upper, True)
        upper_values = high_values/_rounded_hypot(high_values, others_lower, False)
    if not np.isfinite(lower_values).all() or not np.isfinite(upper_values).all():
        metadata["reason"] = "unrepresentable_normalization_box_endpoint"
        return None, None, metadata
    # Each norm accumulation was already widened outwards. Widen endpoint
    # quotients as well; no whole-image directed interval backend is claimed.
    lower_values = np.maximum(0.0, np.nextafter(lower_values, -np.inf))
    upper_values = np.minimum(1.0, np.nextafter(upper_values, np.inf))
    zero = high_values == 0.0
    lower_values[zero] = 0.0
    upper_values[zero] = 0.0
    # If only this coordinate can be nonzero, normalization is exactly one.
    one = (low_values > 0) & (others_upper == 0)
    lower_values[one] = 1.0
    upper_values[one] = 1.0
    normalized_lower = np.full(lo.shape, np.nan)
    normalized_upper = np.full(lo.shape, np.nan)
    normalized_lower[support], normalized_upper[support] = lower_values, upper_values
    metadata.update(available=True, maximum_interval_width=float(np.max(upper_values-lower_values)))
    return normalized_lower, normalized_upper, metadata


def _combined_interval(record, nominal_stamp, entry_dn_bound):
    stamps, norms, indices = prior._normalization_record(record)
    combined = legacy._combine_stamps(list(stamps))
    if combined is None or nominal_stamp is None:
        return None, None, {"available": False, "reasons": ["nominal_combined_template_missing"],
                            "used_history_indices": indices}
    if not np.allclose(combined, nominal_stamp, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError("Used stamp provenance does not reconstruct unchanged nominal template")
    lows, highs, details = [], [], []
    for index, stamp, norm in zip(indices, stamps, norms):
        vector = stamp*norm
        error = np.where(np.isfinite(vector), entry_dn_bound*APERTURE, np.nan)
        lower = np.maximum(vector-error, 0.0)
        upper = vector+error
        lower[APERTURE == 0], upper[APERTURE == 0] = 0.0, 0.0
        low, high, detail = normalization_interval(lower, upper)
        details.append(dict(history_index=index, **detail))
        if low is None:
            return None, None, {"available": False, "reasons": ["used_stamp_normalization_box_unavailable"],
                                "used_history_indices": indices, "stamp_intervals": details,
                                "uncertain_used_stamp_omitted": False}
        lows.append(low); highs.append(high)
    # Identical fixed membership/support in all endpoint calculations. A
    # correlated perturbation need not attain these medians; monotonicity still
    # makes them valid enclosing endpoints, with no averaging of uncertainty.
    median_lower, lower_counts = legacy._median(np.stack(lows), legacy.MIN_USABLE_STAMPS)
    median_upper, upper_counts = legacy._median(np.stack(highs), legacy.MIN_USABLE_STAMPS)
    if not np.array_equal(lower_counts, upper_counts):
        raise ValueError("Normalization endpoints changed fixed median membership")
    median_lower[APERTURE == 0], median_upper[APERTURE == 0] = 0.0, 0.0
    low, high, final = normalization_interval(median_lower, median_upper)
    if low is not None:
        # Preserve the exact already-computed nominal array under harmless
        # normalization rounding differences; never change the nominal itself.
        low = np.minimum(low, nominal_stamp)
        high = np.maximum(high, nominal_stamp)
    return low, high, {
        "available": low is not None,
        "reasons": [] if low is not None else ["combined_median_normalization_box_unavailable"],
        "used_history_indices": indices, "stamp_intervals": details,
        "median_rule": "coordinatewise medians of fixed-member interval endpoints; no independence reduction",
        "final_normalization": final,
        "uncertain_used_stamp_omitted": False,
    }


def _placed_symmetric_bound(nominal, stamp_lower, stamp_upper, center):
    lower = legacy._place(stamp_lower, center)
    upper = legacy._place(stamp_upper, center)
    if not (np.array_equal(np.isfinite(lower), np.isfinite(nominal))
            and np.array_equal(np.isfinite(upper), np.isfinite(nominal))):
        raise ValueError("Placed intervals must preserve original component support")
    return np.maximum(abs(nominal-lower), abs(upper-nominal))


def component_bounds(history129, prior_centers_xy, predicted_offset_xy,
                     polarity, components, protected=True):
    """V43-compatible conditional component bounds with unchanged nominal data."""
    history = legacy._array(history129, legacy.HISTORY_SHAPE, "history129")
    centers, offset = legacy._coordinates(prior_centers_xy, predicted_offset_xy)
    if polarity not in ("bright", "dark") or type(protected) is not bool:
        raise ValueError("Known polarity and Boolean protected flag required")
    background = legacy._array(components["background"], legacy.PATCH_SHAPE, "background")
    fixed = np.asarray(components["fixed_templates"])
    if fixed.ndim != 3 or fixed.shape[1:] != legacy.PATCH_SHAPE or np.isinf(fixed).any():
        raise ValueError("fixed_templates must be Kx129x129 without infinity")
    background_bound = np.where(np.isfinite(background), ALIGNED_VALUE_BOUND_DN, np.nan)
    metadata = {
        "conditional_only": True, "bound_version": "v45_nonnegative_normalization_boxes",
        "nominal_components_unchanged": True,
        "aligned_value_error_bound_dn": ALIGNED_VALUE_BOUND_DN,
        "moving_residual_stamp_error_bound_dn": RESIDUAL_STAMP_BOUND_DN,
        "fixed_highpass_stamp_error_bound_dn": HIGHPASS_STAMP_BOUND_DN,
        "protected_fixed_components": protected,
        "fixed_geometry": True, "fixed_anchor_identities": True, "fixed_finite_support": True,
        "fixed_used_stamp_membership": True, "no_independence_assumption": True,
        "joint_interval_attainability_claimed": False,
        "numerical_roundoff_not_a_camera_noise_model": True,
        "rounding_scope": "outward-rounded prefix/suffix and denominator hypot plus endpoint divisions; not a general whole-image interval backend",
        "omitted_stamp_selection_uncertainty": "Not bounded: a discarded zero-energy stamp can become used under perturbation.",
        "interpretation": "Conditional deterministic outer component-array bounds; not total camera noise, motion/class confidence or a rejection policy.",
    }
    provenance = prior._protected_history(components.get("template_history"), components["metadata"]) if protected else prior._legacy_history(
        history, centers, background, 1.0 if polarity == "bright" else -1.0, components["metadata"])
    if len(provenance["fixed"]) != len(fixed):
        raise ValueError("Fixed template provenance must preserve every nominal template")
    reasons = []
    moving_bound = None
    moving_stamp = components["moving_stamp"]
    if components["moving_template"] is None or moving_stamp is None:
        reasons.append("nominal_moving_template_missing")
        metadata["moving"] = dict(available=False, reasons=["nominal_moving_template_missing"])
    else:
        prior._validate_placed(moving_stamp, 64.0+offset, components["moving_template"])
        low, high, detail = _combined_interval(provenance["moving"], moving_stamp, RESIDUAL_STAMP_BOUND_DN)
        metadata["moving"] = detail
        if low is None:
            reasons.append("moving_component_uncertainty_unavailable")
        else:
            moving_bound = _placed_symmetric_bound(components["moving_template"], low, high, 64.0+offset)
    fixed_bounds, fixed_metadata = [], []
    for index, (entry, nominal) in enumerate(zip(provenance["fixed"], fixed)):
        stamps, _, _ = prior._normalization_record(entry)
        stamp = legacy._combine_stamps(list(stamps))
        if stamp is None:
            raise ValueError("Nominal fixed template lacks reconstructible used stamps")
        center = legacy._array(entry["centre_xy"], (2,), "fixed center")
        if not np.isfinite(center).all():
            raise ValueError("Finite fixed center required")
        prior._validate_placed(stamp, center, nominal)
        low, high, detail = _combined_interval(entry, stamp, HIGHPASS_STAMP_BOUND_DN)
        fixed_metadata.append(dict(template_index=index, centre_xy=center.tolist(), **detail))
        if low is None:
            fixed_bounds.append(np.full((129,129), np.nan))
            reasons.append("fixed_component_uncertainty_unavailable:"+str(index))
        else:
            fixed_bounds.append(_placed_symmetric_bound(nominal, low, high, center))
    metadata["fixed"] = fixed_metadata
    return dict(available=not reasons, reasons=reasons, background_bound129=background_bound,
                moving_template_bound129=moving_bound,
                fixed_template_bounds=np.stack(fixed_bounds) if fixed_bounds else np.empty((0,129,129)),
                metadata=metadata)
