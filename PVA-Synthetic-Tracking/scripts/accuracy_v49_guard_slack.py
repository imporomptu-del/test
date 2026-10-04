"""Exact diagnostic-only response-error slack for V47's guard relaxation.

Given recorded contrasts (Y, B, Ey, EB), compute, for g >= 0,

    min_g max(0, max_i (abs(Y_i - g*B_i) - Ey_i - g*EB_i)/4).

The fixed denominator is the L1 weight of the (1, -2, 1) stencil. Thus the
result is the smallest *additional uniform response-side DN error* necessary
for the recorded stencil inequalities to become mutually feasible. It is NOT
a noise estimate, a proposed production tolerance, or a sufficient error for
a joint pixel/affine model. Overlapping stencils need not admit common errors.
Zero slack does not establish guard cleanliness, gain identifiability, transfer
to the core, or source presence. No detector bounds are modified by this code.

The computation uses an exact rational upper convex envelope, with a constant
zero line. Sorting takes O(n log n) rational comparisons; envelope construction
is O(n). Arbitrarily large rational numerators still have nonconstant bit cost.
Floats are interpreted as their exact binary ratios, not exact decimal values.
This certifies arithmetic on the supplied contrasts only, not their upstream
formation, sensor calibration, or physical model. Display floats are rounded
and may underflow; overflow is represented by None, never JSON infinity. Only
the rational strings and optimality certificate are authoritative.
"""

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Integral


STENCIL_L1_WEIGHT = 4
_FIELDS = ("response", "background", "response_error", "background_error")


def _fraction(value):
    """Accept recorded rational strings and exact finite numeric ratios."""
    if isinstance(value, bool) or type(value).__name__ == "bool":
        raise ValueError("Boolean is not a numerical constraint")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        # V47 writes integers or numerator/denominator strings. Do not silently
        # accept decimal/exponent strings with different floating semantics.
        parts = value.split("/")
        if len(parts) not in (1, 2):
            raise ValueError("Expected an integer or rational string")
        try:
            if any(str(int(part)) != part for part in parts):
                raise ValueError("Expected a canonical integer or rational string")
            return Fraction(int(parts[0]), int(parts[1]) if len(parts) == 2 else 1)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Expected a finite rational string") from exc
    ratio = getattr(value, "as_integer_ratio", None)
    # NumPy floating scalars preserve their own precision through this method;
    # Python Decimal also has it but is intentionally not an accepted float.
    if ratio is not None and (isinstance(value, float) or
                              type(value).__module__.split(".")[0] == "numpy"):
        try:
            numerator, denominator = ratio()
            return Fraction(numerator, denominator)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("Constraint values must be finite") from exc
    raise ValueError("Expected a finite integer, float, Fraction, or rational string")


def _normalise(constraints):
    rows = []
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            raise ValueError(f"Constraint {index} must be a mapping")
        try:
            row = tuple(_fraction(constraint[key]) for key in _FIELDS)
        except KeyError as exc:
            raise ValueError(f"Constraint {index} is missing {exc.args[0]}") from exc
        if row[2] < 0 or row[3] < 0:
            raise ValueError("Response and background error bounds must be nonnegative")
        rows.append(row)
    return rows


def _lines(rows):
    # line = (slope, intercept, identifier, constraint index, residual sign)
    result = [(Fraction(0), Fraction(0), "slack_floor", None, 0)]
    for index, (y, b, ey, eb) in enumerate(rows):
        for sign in (1, -1):
            result.append(((-sign*b-eb)/STENCIL_L1_WEIGHT,
                           (sign*y-ey)/STENCIL_L1_WEIGHT,
                           f"contrast_{index}_{'plus' if sign == 1 else 'minus'}",
                           index, sign))
    return result


def _upper_envelope(lines):
    """Return increasing-slope upper-envelope lines and exact start abscissae.

    The first start is None (minus infinity). At equal slopes only the largest
    intercept is retained. Coincident lines can disappear from the envelope,
    but the final active set is reconstructed from *all* original lines.
    """
    unique = {}
    for line in lines:
        previous = unique.get(line[0])
        if previous is None or line[1] > previous[1]:
            unique[line[0]] = line
    hull, starts = [], []
    for slope in sorted(unique):
        line = unique[slope]
        start = None
        while hull:
            previous = hull[-1]
            start = (previous[1]-line[1])/(line[0]-previous[0])
            if len(hull) == 1 or start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = None
        hull.append(line)
        starts.append(start)
    return hull, starts


def _line_record(line, gain):
    slope, intercept, identifier, index, sign = line
    return dict(line_id=identifier, constraint_index=index, residual_sign=sign,
                slope_exact=str(slope), intercept_exact=str(intercept),
                value_exact=str(slope*gain+intercept))


def _display(value):
    try:
        rounded = float(value)
    except OverflowError:
        return None
    return rounded if math.isfinite(rounded) else None


def minimum_guard_response_slack(contrast_constraints):
    """Return the exact minimum, earliest nonnegative minimizing gain, witness.

    Input entries have V47's response/background/response_error/background_error
    keys. Negative errors, missing keys, Booleans, and nonfinite values raise
    ValueError. Empty input returns zero slack at gain zero, explicitly flagged
    as vacuous; it is not evidence that an empty guard is valid.
    """
    rows = _normalise(contrast_constraints)
    lines = _lines(rows)
    hull, starts = _upper_envelope(lines)
    position = 0
    while position+1 < len(hull) and starts[position+1] <= 0:
        position += 1
    gain = Fraction(0)
    if hull[position][0] < 0:
        # The zero floor guarantees a later nonnegative-slope segment. This
        # also handles every nonconstant input line decreasing indefinitely.
        position += 1
        while hull[position][0] < 0:
            position += 1
        gain = starts[position]
    slack = max(line[0]*gain+line[1] for line in lines)
    active = [line for line in lines if line[0]*gain+line[1] == slack]
    if slack == 0:
        witness = dict(kind="zero_slack_global_lower_bound", line_ids=["slack_floor"])
    elif gain == 0:
        supporting = next(line for line in active if line[0] >= 0)
        witness = dict(kind="nonnegative_boundary_slope", line_ids=[supporting[2]],
                       supporting_slope_exact=str(supporting[0]))
    else:
        lower = min(active, key=lambda line: line[0])
        upper = max(active, key=lambda line: line[0])
        if not lower[0] < 0 <= upper[0]:
            raise AssertionError("Earliest interior minimizer lacks straddling active slopes")
        lower_weight = upper[0]/(upper[0]-lower[0])
        upper_weight = -lower[0]/(upper[0]-lower[0])
        witness = dict(kind="straddling_active_slopes",
                       line_ids=[lower[2], upper[2]],
                       convex_weights_exact=[str(lower_weight), str(upper_weight)],
                       slopes_exact=[str(lower[0]), str(upper[0])],
                       weighted_slope_exact="0")
    result = dict(
        schema_version=1,
        quantity="minimum_additional_uniform_response_error_DN_for_guard_stencil_relaxation",
        stencil_l1_weight=STENCIL_L1_WEIGHT,
        constraint_count=len(rows), signed_constraint_count=2*len(rows),
        vacuous_empty_constraints=not rows,
        minimizer_selection="smallest_nonnegative_gain",
        gain_exact=str(gain), slack_exact=str(slack),
        gain_display=_display(gain), slack_display=_display(slack),
        display_floats_are_not_authoritative=True,
        upper_envelope_line_count=len(hull),
        active_signed_constraints=[_line_record(line, gain) for line in active if line[3] is not None],
        zero_floor_active=slack == 0, optimality_witness=witness,
        interpretation=dict(
            diagnostic_only=True, production_bounds_changed=False,
            calibrated_noise_estimate=False, proposed_threshold=False,
            sufficient_for_joint_pixel_or_affine_feasibility=False,
            necessary_stencil_relaxation_only=True,
            proves_gain_identifiability=False, proves_guard_cleanliness=False,
            proves_guard_to_core_transfer=False, proves_source_presence=False,
            certifies_upstream_contrast_roundoff=False,
            float_inputs_use_exact_binary_ratios=True))
    _verify_rows_certificate(rows, result)
    return result


def verify_guard_response_slack_certificate(contrast_constraints, result):
    """Verify exact objective feasibility and a global lower-bound certificate.

    This does not construct an envelope or search gain candidates. A successful
    check proves the stated value is the global minimum for supplied contrasts,
    not uniqueness or physical validity. The earliest-minimizer convention is
    implemented by the producer; it is not part of this certificate's promise.
    Invalid certificates raise ValueError. Returns True on success.
    """
    return _verify_rows_certificate(_normalise(contrast_constraints), result)


def _verify_rows_certificate(rows, result):
    lines = _lines(rows)
    try:
        gain, slack = _fraction(result["gain_exact"]), _fraction(result["slack_exact"])
        if gain < 0 or slack < 0 or result["stencil_l1_weight"] != STENCIL_L1_WEIGHT:
            raise ValueError("Invalid gain, slack, or stencil weight")
        values = {line[2]: line[0]*gain+line[1] for line in lines}
        if max(values.values()) != slack:
            raise ValueError("Reported slack is not the exact objective at reported gain")
        active = {line[2]: line for line in lines if values[line[2]] == slack}
        expected = [_line_record(line, gain) for line in lines
                    if line[3] is not None and line[2] in active]
        if result["active_signed_constraints"] != expected or result["zero_floor_active"] != (slack == 0):
            raise ValueError("Reported active set differs from exact active constraints")
        witness = result["optimality_witness"]
        selected = [active[identifier] for identifier in witness["line_ids"]]
        kind = witness["kind"]
        if kind == "zero_slack_global_lower_bound":
            valid = slack == 0 and witness["line_ids"] == ["slack_floor"]
        elif kind == "nonnegative_boundary_slope":
            valid = (gain == 0 and len(selected) == 1 and selected[0][0] >= 0 and
                     _fraction(witness["supporting_slope_exact"]) == selected[0][0])
        elif kind == "straddling_active_slopes":
            weights = [_fraction(value) for value in witness["convex_weights_exact"]]
            slopes = [_fraction(value) for value in witness["slopes_exact"]]
            valid = (gain > 0 and len(selected) == len(weights) == len(slopes) == 2 and
                     selected[0][0] <= 0 <= selected[1][0] and
                     slopes == [line[0] for line in selected] and
                     all(weight >= 0 for weight in weights) and sum(weights) == 1 and
                     sum(weight*line[0] for weight, line in zip(weights, selected)) == 0 and
                     sum(weight*line[1] for weight, line in zip(weights, selected)) == slack and
                     _fraction(witness["weighted_slope_exact"]) == 0)
        else:
            valid = False
        if not valid:
            raise ValueError("Optimality witness does not prove a global lower bound")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed or inactive optimality witness") from exc
    return True
