"""Conditional bounded-background-gain numerator evidence, not a detector gate.

The caller supplies a justified finite nonnegative gain interval. This helper
does NOT estimate or certify that interval, its guard pixels, or guard-to-core
transfer. Same-frame guard independence, camera/model errors and physical class
remain external assumptions. The uncertainty arrays are deterministic bounds,
not calibrated independent noise or statistical confidence.

For Q projecting off the exact supplied affine span and R(F') projecting off
the uncertain fixed-template span in Q coordinates, consider

    s'(g) = (Qm')^T R(F') Q(y' - g B').

For every ONE admissible simultaneous realization (m',F',y',B'), s'(g) is affine
in the scalar gain. Thus every numerator for glo<=g<=ghi is a convex combination
of endpoint numerators. At endpoint gk, |dy-gk*dB|<=ey+gk*uB. Two immutable V45
source_presence evaluations therefore bound the entire gain interval by the
hull of their intervals. No independence is used: their response/source/fixed
errors can be correlated. All fixed columns, support and uncertainties remain.
Both endpoint evaluations must be available. There is no favorable-endpoint or
best-gain selection, and no division by uncertain source energy.

This is a DIFFERENT estimand from unrestricted-background-gain least squares.
It certifies only a conditional bounded-gain partial-regression numerator sign,
not source amplitude, physical motion, identity, airborne class or rejection.
It covers a gain selected adaptively inside the justified interval because the
enclosure is simultaneous for every gain in that interval.

Arithmetic: all inputs are interpreted as their float64 values. Each centered
response is computed as the exact Fraction y_i-g*B_i, then rounded once to a
float64. Its exact absolute rounding error is added to the exact Fraction
ey_i+g*uB_i, and the sum is rounded OUTWARD to a float64 bound. Subnormal errors
are not silently rounded to zero; unrepresentable finite outputs are unknown.
This certifies only this centering/bound-assembly step relative to the supplied
float64 values, NOT the subsequent V45 SVD/projection or complete image pipeline
as formal IEEE interval arithmetic. V45's numerical-resolution caveats remain.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math

import numpy as np

from accuracy_v45_presence import source_presence


def _array(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(name + " must be a real array of the declared dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite on the unchanged common support")
    return array


def _bound(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(name + " must be real numeric")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.broadcast_to(array.astype(np.float64), shape)
    except ValueError as exc:
        raise ValueError(name + " must broadcast to its input") from exc
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return array


def _gain_interval(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (2,) or array.dtype.kind not in "iuf":
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64)
    if not np.isfinite(array).all() or array[0] < 0 or array[0] > array[1]:
        return None
    return array.tolist()


def _fraction(value):
    return Fraction.from_float(float(value))


def _outward_nonnegative(value):
    """Smallest float64 at or above a nonnegative Fraction, or None on overflow."""
    if value < 0:
        raise ValueError("Expected a nonnegative exact value")
    try:
        rounded = float(value)
    except OverflowError:
        return None
    if not math.isfinite(rounded):
        return None
    if _fraction(rounded) < value:
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = float(np.nextafter(rounded, np.inf))
    return rounded if math.isfinite(rounded) else None


def centered_response(y, background, gain, response_bound, background_bound):
    """Return exact-float-centered response/bound arrays, or explicit unknown.

    This public audit helper has no inference policy. It preserves every input
    row; it never shrinks support or approximates an unrepresentable endpoint.
    """
    y, background = _array(y, 1, "y"), _array(background, 1, "background")
    if not len(y) or background.shape != y.shape:
        raise ValueError("Nonempty matching response/background vectors required")
    if (isinstance(gain, (bool, np.bool_)) or not isinstance(gain, (int, float, np.integer, np.floating))
            or not math.isfinite(float(gain)) or gain < 0):
        raise ValueError("Finite nonnegative endpoint gain required")
    ey = _bound(response_bound, y.shape, "response_bound")
    ub = _bound(background_bound, y.shape, "background_bound")
    result = dict(available=False, reason=None, response=None, response_bound=None,
                  centering_roundoff_bound=None)
    gain_exact = _fraction(gain)
    centered, bounds, roundoff = (np.empty(y.shape) for _ in range(3))
    for i in range(len(y)):
        exact = _fraction(y[i]) - gain_exact*_fraction(background[i])
        try:
            rounded = float(exact)
        except OverflowError:
            result["reason"] = "unrepresentable_centered_response"
            return result
        if not math.isfinite(rounded):
            result["reason"] = "unrepresentable_centered_response"
            return result
        rounding_error = abs(exact-_fraction(rounded))
        exact_error = _fraction(ey[i]) + gain_exact*_fraction(ub[i]) + rounding_error
        outward, arithmetic = _outward_nonnegative(exact_error), _outward_nonnegative(rounding_error)
        if outward is None or arithmetic is None:
            result["reason"] = "unrepresentable_centered_response_bound"
            return result
        centered[i], bounds[i], roundoff[i] = rounded, outward, arithmetic
    result.update(available=True, response=centered, response_bound=bounds,
                  centering_roundoff_bound=roundoff)
    return result


def _validate_core_record(record):
    """Do not turn malformed/partial legacy numerical results into a hull."""
    if (type(record.get("available")) is not bool or record.get("motion_status") != "unknown"
            or record.get("physical_class") != "unknown"
            or record.get("diagnostics", {}).get("no_production_gate") is not True):
        raise ValueError("Immutable endpoint core returned inconsistent availability/class status")
    if not record["available"]:
        if any(record.get(key) is not None for key in
               ("numerator", "error_bound", "interval", "interval_excludes_zero", "coefficient_sign")):
            raise ValueError("Unavailable endpoint has operative numerical evidence")
        return
    interval = record.get("interval")
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))
                   or not math.isfinite(float(x)) for x in interval) or interval[0] > interval[1]):
        raise ValueError("Available endpoint has invalid finite interval")
    sign = "positive" if interval[0] > 0 else "negative" if interval[1] < 0 else "unresolved"
    if (record.get("coefficient_sign") != sign
            or record.get("interval_excludes_zero") is not (sign != "unresolved")
            or record.get("numerator") is None or not math.isfinite(float(record["numerator"]))):
        raise ValueError("Endpoint interval/sign/numerator disagree")


def bounded_background_presence(y, B, F, m, P, *, gain_interval, response_bound,
                                background_bound, fixed_bound, source_bound, gain_provenance):
    """Enclose numerator for ALL gains in the caller-justified interval.

    Missing/unsupported gains or missing declared uncertainty return unavailable
    without calling the legacy core. Malformed finite-support array contracts
    raise ValueError. Gain provenance is descriptive only; neither its label nor
    current response values select the interval, support, columns, or an endpoint.
    """
    try:
        provenance = json.loads(json.dumps(gain_provenance, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("gain_provenance must be JSON-safe descriptive metadata") from exc
    result = dict(available=False, reasons=[],
        quantity="bounded_gain_nuisance_residualized_source_numerator",
        gain_interval=None, gain_provenance=provenance, gain_provenance_certified=False,
        caller_gain_interval_justification_assumed_not_inferred_from_core=True,
        interval=None, interval_excludes_zero=None, coefficient_sign=None,
        numerator=None, error_bound=None, endpoint_nominal_numerators=None,
        endpoint_evaluations=[], motion_status="unknown", physical_class="unknown",
        is_motion_or_classification_gate=False, production_changed=False,
        diagnostics=dict(immutable_endpoint_core="accuracy_v45_presence.source_presence",
            no_single_nominal_numerator_or_amplitude_claim=True,
            gain_and_background_correlation_retained_by_endpoint_hull=True,
            fixed_columns_not_dropped_by_wrapper=True, common_support_not_changed=True,
            nuisance_coefficients_unrestricted=True, no_endpoint_selection=True,
            no_independence_assumption=True, calibrated_noise_or_confidence=False,
            not_the_unrestricted_background_gain_least_squares_estimand=True,
            centering_arithmetic="exact Fractions of float64 inputs; one rounded response and outward assembled response error",
            whole_pipeline_is_ieee_certified_enclosure=False, no_production_gate=True,
            numerical_resolution_caveats="Immutable V45 SVD/projection resolution guard is not a formal IEEE enclosure.",
            physical_guard_cleanliness_and_core_transfer_not_certified=True))
    interval = _gain_interval(gain_interval)
    if interval is None:
        result["reasons"] = ["missing_or_unsupported_nonnegative_gain_interval"]
        return result
    result["gain_interval"] = interval
    if response_bound is None or background_bound is None or source_bound is None:
        result["reasons"] = ["missing_declared_uncertainty"]
        return result
    y, B, F, m, P = (_array(value, ndim, name) for value, ndim, name in
                     ((y,1,"y"),(B,1,"B"),(F,2,"F"),(m,1,"m"),(P,2,"P")))
    n = len(y)
    if not n or B.shape != y.shape or m.shape != y.shape or len(F) != n or len(P) != n:
        raise ValueError("Nonempty matched row counts required")
    if F.shape[1] and fixed_bound is None:
        result["reasons"] = ["missing_declared_fixed_uncertainty"]
        return result
    ey, ub, uf, um = (_bound(value, shape, name) for value, shape, name in (
        (response_bound,y.shape,"response_bound"),(background_bound,B.shape,"background_bound"),
        (0. if fixed_bound is None else fixed_bound,F.shape,"fixed_bound"),(source_bound,m.shape,"source_bound")))
    result["diagnostics"].update(rows=n, fixed_columns_passed=F.shape[1],
                                 singleton_gain_interval=interval[0] == interval[1],
                                 endpoint_calls=2)
    for label, gain in zip(("lower_gain", "upper_gain"), interval):
        centered = centered_response(y, B, gain, ey, ub)
        endpoint = dict(label=label, gain=gain, available=False, reasons=[],
                        arithmetic=None, source_presence=None)
        if not centered["available"]:
            endpoint["reasons"] = [centered["reason"]]
        else:
            endpoint["arithmetic"] = dict(
                response_bound_min=float(centered["response_bound"].min()),
                response_bound_max=float(centered["response_bound"].max()),
                centering_roundoff_bound_max=float(centered["centering_roundoff_bound"].max()),
                centering_roundoff_bound_nonzero_rows=int(np.count_nonzero(centered["centering_roundoff_bound"])))
            record = source_presence(centered["response"], F, m, P,
                response_bound=centered["response_bound"], nuisance_bound=uf, source_bound=um)
            _validate_core_record(record)
            endpoint.update(available=record["available"], source_presence=deepcopy(record),
                            reasons=list(record["reasons"]))
        result["endpoint_evaluations"].append(endpoint)
    if not all(endpoint["available"] for endpoint in result["endpoint_evaluations"]):
        result["reasons"] = ["both_gain_endpoints_must_be_available"]
        result["reasons"] += [endpoint["label"]+":"+reason
                              for endpoint in result["endpoint_evaluations"] for reason in endpoint["reasons"]]
        return result
    records = [endpoint["source_presence"] for endpoint in result["endpoint_evaluations"]]
    low = min(record["interval"][0] for record in records)
    high = max(record["interval"][1] for record in records)
    sign = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
    result.update(available=True, interval=[low,high], coefficient_sign=sign,
                  interval_excludes_zero=sign != "unresolved",
                  endpoint_nominal_numerators=[record["numerator"] for record in records])
    return result
