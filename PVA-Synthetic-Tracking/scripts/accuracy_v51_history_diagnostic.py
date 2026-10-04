"""Continuous, prior-value-only descriptors for eight fixed per-point samples.

The caller owns support and geometry. These descriptors do not certify support
purity, online camera causality, physical noise, future behavior or object class.
No response, labels, threshold, classifier or predictor selector is accepted.
"""

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


VECTOR_FIELDS = (
    "slow8", "fast3", "early5", "scale", "fast_slow_delta",
    "fast_slow_delta_normalized", "early_recent_delta",
    "early_recent_delta_normalized", "departure_envelope",
    "departure_envelope_normalized", "return_fraction",
    "recent_center_suffix_length",
)
MATRIX_FIELDS = (
    "recent_departures", "recent_departures_normalized",
    "recent_center_margins", "recent_center_margins_normalized",
)


def _plain(value):
    """Canonical JSON values, with every nonfinite numeric value explicit null."""
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


def diagnostic_fingerprint(result):
    """Bind all arrays, availability and metadata except the digest itself."""
    if not isinstance(result, Mapping):
        raise ValueError("Diagnostic result must be a mapping")
    payload = {key: value for key, value in result.items()
               if key != "diagnostic_sha256"}
    return hashlib.sha256(json.dumps(_plain(payload), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _readonly(array):
    output = np.array(array, copy=True)
    output.flags.writeable = False
    return output


def diagnose(history):
    """Describe eight oldest-to-newest samples at N fixed points, retaining N.

    Input is a real numeric array of shape (8, N), including N=0. Computation
    uses float64. A nonfinite input or nonfinite arithmetic makes only that
    column unavailable, with all numeric descriptors NaN. Nothing is imputed.
    Return fraction is undefined for zero recent departure envelope. The
    recent-center suffix is undefined when fast3 equals early5; their margins
    are still the mathematically defined zeros. Suffix comparisons are strict,
    so an exactly zero margin interrupts the suffix.
    """
    array = np.asarray(history)
    if array.ndim != 2 or array.shape[0] != 8 or array.dtype.kind not in "iuf":
        raise ValueError("history must be a real numeric array of shape (8, N)")
    count = array.shape[1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        values = np.asarray(array, dtype=np.float64)
    input_finite = np.isfinite(values).all(axis=0)
    numeric = {name: np.full(count, np.nan) for name in VECTOR_FIELDS}
    numeric.update({name: np.full((3, count), np.nan) for name in MATRIX_FIELDS})
    available = np.zeros(count, dtype=bool)
    return_defined = np.zeros(count, dtype=bool)
    center_defined = np.zeros(count, dtype=bool)
    reasons = [None if finite else "nonfinite_input_after_float64_conversion"
               for finite in input_finite]

    indices = np.flatnonzero(input_finite)
    if len(indices):
        h = values[:, indices]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            slow = np.median(h, axis=0)
            fast = np.median(h[-3:], axis=0)
            early = np.median(h[:5], axis=0)
            scale = np.maximum(1.0, np.median(np.abs(h-slow), axis=0))
            departures = h[-3:]-early
            envelope = np.max(np.abs(departures), axis=0)
            margins = np.abs(departures)-np.abs(h[-3:]-fast)
            fast_slow, early_recent = fast-slow, fast-early
            computed = dict(slow8=slow, fast3=fast, early5=early, scale=scale,
                fast_slow_delta=fast_slow,
                fast_slow_delta_normalized=fast_slow/scale,
                early_recent_delta=early_recent,
                early_recent_delta_normalized=early_recent/scale,
                recent_departures=departures,
                recent_departures_normalized=departures/scale,
                departure_envelope=envelope,
                departure_envelope_normalized=envelope/scale,
                recent_center_margins=margins,
                recent_center_margins_normalized=margins/scale)
        valid = np.ones(len(indices), dtype=bool)
        for output in computed.values():
            valid &= np.isfinite(output) if output.ndim == 1 else np.isfinite(output).all(axis=0)
        # Optional descriptors are computed only where their mathematical
        # definitions apply; deliberate undefined values are not failures.
        fractions = np.full(len(indices), np.nan)
        has_return = valid & (envelope > 0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            fractions[has_return] = 1.0-np.abs(departures[-1, has_return])/envelope[has_return]
        valid &= ~has_return | np.isfinite(fractions)
        has_return &= valid
        has_center = valid & (fast != early)
        suffixes = np.full(len(indices), np.nan)
        suffixes[has_center] = 0.0
        continuing = has_center.copy()
        for position in (2, 1, 0):
            continuing &= margins[position] > 0
            suffixes[continuing] += 1.0
        computed["return_fraction"] = fractions
        computed["recent_center_suffix_length"] = suffixes
        for name, output in computed.items():
            if output.ndim == 1:
                numeric[name][indices[valid]] = output[valid]
            else:
                numeric[name][:, indices[valid]] = output[:, valid]
        available[indices[valid]] = True
        return_defined[indices[has_return]] = True
        center_defined[indices[has_center]] = True
        for index in indices[~valid]:
            reasons[int(index)] = "nonfinite_descriptor_arithmetic"

    result = dict(schema_version=1, total_count=count,
        available_count=int(available.sum()), point_available=_readonly(available),
        return_fraction_defined=_readonly(return_defined),
        recent_center_defined=_readonly(center_defined),
        point_unavailable_reasons=tuple(reasons),
        **{name: _readonly(output) for name, output in numeric.items()},
        metadata=dict(prior_count=8, prior_order="oldest_to_newest",
            samples_are_fixed_points_supplied_by_caller=True,
            per_point_support_selected_or_dropped=False,
            current_argument_accepted=False, labels_argument_accepted=False,
            geometry_argument_accepted=False, geometry_causality_certified=False,
            full_camera_online_causality_certified=False,
            forecast_values_selected_or_modified=False,
            scale_formula="max(1 DN, median(abs(history - median(history))))",
            scale_floor_dn=1.0, scale_is_descriptor_not_physical_noise_bound=True,
            normalization_divides_dn_descriptors_by_v50_scale=True,
            early_and_recent_windows_disjoint=True,
            recent_center_suffix_uses_latest_three_strict_positive_margins=True,
            return_fraction_is_observed_magnitude_return_not_future_probability=True,
            equal_prior_prefix_cannot_identify_different_future_responses=True,
            support_purity_or_guard_to_core_transfer_certified=False,
            classification_threshold_or_model_selection_performed=False,
            intervals_or_coverage_guarantees_produced=False,
            source_object_or_production_decision=False,
            unavailable_indices_preserved_without_imputation=True,
            nonfinite_json_representation="null"))
    result["diagnostic_sha256"] = diagnostic_fingerprint(result)
    return result
