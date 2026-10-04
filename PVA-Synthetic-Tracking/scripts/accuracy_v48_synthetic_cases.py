"""Versioned V47 fixture repair; exact pre-score witnesses, never detector scores.

Only the correlated-error case's current full annulus is rendered differently.
The immutable V47 renderer remains the regression for an invalid error witness.
All arithmetic claims below concern supplied binary64 latent values, not a
physical sensor-noise guarantee or the unknown validity of a real image guard.
"""
from copy import deepcopy
from fractions import Fraction
import math
import sys

import numpy as np

import accuracy_v47_synthetic_cases as previous


CASE_IDS = previous.CASE_IDS
ADAPTER_KEYS = previous.ADAPTER_KEYS
TWIN_IDS = previous.TWIN_IDS
COUNTEREXAMPLE_IDS = previous.COUNTEREXAMPLE_IDS
CORRECTED_CASE_ID = "correlated_guard_error_extremes"
ERROR_RADIUS = Fraction(1, 2)


def round_inside_error_box(latent: Fraction, intended_error: Fraction) -> float:
    """Nearest intended observation, moved inward only if its rounding escaped.

The error box is fixed at exactly +/-1/2 DN, not inflated by a tolerance. The
exact intended error must lie in that box. One adjacent binary64 value suffices
after nearest rounding: if it also lies outside, no finite binary64 observation
exists in the interval. Overflow is handled as a boundary, never as infinity.
    """
    if not isinstance(latent, Fraction) or not isinstance(intended_error, Fraction):
        raise TypeError("latent and intended_error must be exact Fraction values")
    if abs(intended_error) > ERROR_RADIUS:
        raise ValueError("intended error exceeds the unchanged +/-1/2-DN box")
    lower, upper = latent-ERROR_RADIUS, latent+ERROR_RADIUS
    intended = latent+intended_error
    try:
        candidate = float(intended)
    except OverflowError:
        candidate = math.copysign(sys.float_info.max, 1 if intended >= 0 else -1)
    if not math.isfinite(candidate):
        candidate = math.copysign(sys.float_info.max, candidate)
    exact = Fraction(candidate)
    if exact < lower:
        candidate = math.nextafter(candidate, math.inf)
    elif exact > upper:
        candidate = math.nextafter(candidate, -math.inf)
    if not math.isfinite(candidate) or not lower <= Fraction(candidate) <= upper:
        raise ValueError("the exact error box contains no finite binary64 value")
    return candidate


def _geometry():
    yy, xx = np.indices((129, 129), dtype=np.float64)
    radius = np.maximum(abs(xx-64.), abs(yy-64.))
    annulus = (radius >= 40.) & (radius <= 56.)
    signs = np.where((np.floor(xx/8.)+np.floor(yy/8.)) % 2 == 0, 1, -1)
    return xx, yy, annulus, signs


def _correction_metadata():
    return dict(
        version="v48_exact_binary64_error_box_renderer",
        affected_case_id=CORRECTED_CASE_ID,
        affected_array="current129",
        affected_region="all integer pixels with 40<=max(abs(x-64),abs(y-64))<=56",
        nominal_background="the unchanged V47 binary64 step_sine background B",
        latent_product="exact Fraction(binary64(1.2))*Fraction(binary64(B[y,x]))",
        intended_current_error="-s/2, s=(-1)^(floor(x/8)+floor(y/8))",
        stored_current_error="exact Fraction(stored current)-latent product; within [-1/2,+1/2], with intended sign",
        rounding_policy="round exact intended endpoint to binary64; nextafter toward interval interior only if outside",
        error_radius_dn=.5,
        arbitrary_epsilon_used=False,
        current_core_changed=False,
        history_or_background_changed=False,
        geometry_or_polarity_changed=False,
        detector_or_gain_solver_used_for_rendering=False,
        guard_selection_used_for_rendering=False,
        inherited_error_formula_describes_intended_not_exact_stored_error=True,
        inherited_prior_error_semantics="unchanged binary64 addition of s/2; exact realized errors remain inside [-1/2,+1/2], with intended sign but possibly inward-rounded magnitude",
        original_v47_failure_preserved=True,
        sensor_noise_contract_validated=False,
    )


def scenario_manifest():
    """Add a declared rendering revision without rewriting the original cases."""
    result = deepcopy(previous.scenario_manifest())
    result["renderer_revision"] = 48
    result["v48_current_error_rendering"] = _correction_metadata()
    result["v48_pre_score_contract"] = dict(
        required=True, uses_scores=False,
        current_witness_domain="full fixed annulus, independent of guard eligibility",
        prior_witness_domain="full fixed annulus excluding Chebyshev distance <=12 from every prior center",
        bound_arithmetic="exact Fraction comparisons, no epsilon",
        all_other_cases_and_uncorrected_arrays_preserved=True,
    )
    return result


def build_cases():
    """Return the same twenty cases, repairing only one declared current band."""
    cases = previous.build_cases()
    corrected = next(c for c in cases if c["case_id"] == CORRECTED_CASE_ID)
    xx, yy, annulus, signs = _geometry()
    background = previous._background("step_sine", xx, yy)
    gain = Fraction(1.2)
    current = corrected["adapter_inputs"]["current129"]
    for y, x in zip(*np.where(annulus)):
        latent = gain*Fraction(float(background[y, x]))
        current[y, x] = round_inside_error_box(latent, -int(signs[y, x])*ERROR_RADIUS)
    corrected["generator_truth"]["current_error_rendering_v48"] = _correction_metadata()
    corrected["provenance"]["renderer_correction_v48"] = _correction_metadata()
    return cases


def _array_equal(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def pre_score_contract(cases):
    """Audit generated inputs before scoring; return deterministic JSON facts.

This is a construction witness, not a solver call. The wider prior exclusion is
fixed by all prior centers, not by observed current values or guard outcomes.
Its background witness deliberately excludes source footprints; the unchanged
source-bearing core is not relabeled as source-free background.
    """
    issues = []
    original = previous.build_cases()
    if [c.get("case_id") for c in cases] != list(CASE_IDS):
        return dict(passed=False, issues=["case membership/order differs from the frozen twenty-case declaration"],
                    uses_scores=False, synthetic_only=True, current_witness_violation_count=None,
                    prior_witness_violation_count=None)
    unchanged_cases = 0
    corrected = None
    old_corrected = None
    for old, case in zip(original, cases):
        if set(case) != set(old):
            issues.append(f"{old['case_id']}: case fields changed")
        if set(case["adapter_inputs"]) != set(ADAPTER_KEYS):
            issues.append(f"{old['case_id']}: adapter argument keys changed")
        if case["case_id"] == CORRECTED_CASE_ID:
            corrected, old_corrected = case, old
            stripped = deepcopy({k: v for k, v in case.items() if k != "adapter_inputs"})
            for group, key in (("generator_truth", "current_error_rendering_v48"),
                               ("provenance", "renderer_correction_v48")):
                if stripped[group].pop(key, None) != _correction_metadata():
                    issues.append(f"{old['case_id']}: missing or changed explicit correction metadata")
            if stripped != {k: v for k, v in old.items() if k != "adapter_inputs"}:
                issues.append(f"{old['case_id']}: inherited case metadata changed")
            keys = set(ADAPTER_KEYS)-{"current129"}
        else:
            if {k: v for k, v in case.items() if k != "adapter_inputs"} != {k: v for k, v in old.items() if k != "adapter_inputs"}:
                issues.append(f"{old['case_id']}: unchanged case metadata changed")
            keys = set(ADAPTER_KEYS)
            unchanged_cases += 1
        for key in sorted(keys):
            a, b = case["adapter_inputs"][key], old["adapter_inputs"][key]
            equal = a == b if key == "polarity" else _array_equal(a, b)
            if not equal:
                issues.append(f"{old['case_id']}: unchanged {key} changed")
    a = corrected["adapter_inputs"]
    old_a = old_corrected["adapter_inputs"]
    current, history = np.asarray(a["current129"]), np.asarray(a["history129"])
    if current.shape != (129, 129) or current.dtype != np.float64 or history.shape != (8, 129, 129):
        return dict(passed=False, issues=issues+["corrected arrays have invalid shape/dtype"], uses_scores=False,
                    synthetic_only=True, current_witness_violation_count=None, prior_witness_violation_count=None)
    xx, yy, annulus, signs = _geometry()
    if not _array_equal(current[~annulus], old_a["current129"][~annulus]):
        issues.append("current values outside the declared full annulus changed")
    if not np.isfinite(current[annulus]).all() or not np.isfinite(history).all():
        return dict(passed=False, issues=issues+["nonfinite correlated-case observations"], uses_scores=False,
                    synthetic_only=True, current_witness_violation_count=None, prior_witness_violation_count=None)
    background = previous._background("step_sine", xx, yy)
    source = previous._point(xx, yy, (64., 64.), 30., 1.)
    source_nonzero = int(np.count_nonzero(source[annulus]))
    if source_nonzero:
        issues.append("stored current source image is nonzero on the correction band")
    current_bad = current_sign_bad = intended_renderer_mismatch = original_bad = 0
    current_max_abs = Fraction(0)
    current_max_inward = Fraction(0)
    changed_count = 0
    gain = Fraction(1.2)
    for y, x in zip(*np.where(annulus)):
        latent = gain*Fraction(float(background[y, x]))
        intended = -int(signs[y, x])*ERROR_RADIUS
        error = Fraction(float(current[y, x]))-latent
        current_bad += abs(error) > ERROR_RADIUS
        current_sign_bad += error*intended < 0
        intended_renderer_mismatch += current[y, x] != round_inside_error_box(latent, intended)
        original_bad += abs(Fraction(float(old_a["current129"][y, x]))-latent) > ERROR_RADIUS
        changed_count += current[y, x] != old_a["current129"][y, x]
        current_max_abs = max(current_max_abs, abs(error))
        current_max_inward = max(current_max_inward, abs(intended-error))
    prior_domain = annulus.copy()
    for cx, cy in old_a["prior_centers_xy"]:
        prior_domain &= np.maximum(abs(xx-cx), abs(yy-cy)) > 12
    prior_bad = prior_sign_bad = prior_inexact_endpoint = prior_source_changes = 0
    prior_max_abs = Fraction(0)
    for i, center in enumerate(old_a["prior_centers_xy"]):
        clean_stored_prior = background+previous._point(xx, yy, center, 30., 1.)
        prior_source_changes += int(np.count_nonzero(clean_stored_prior[prior_domain] != background[prior_domain]))
        for y, x in zip(*np.where(prior_domain)):
            error = Fraction(float(history[i, y, x]))-Fraction(float(background[y, x]))
            intended = int(signs[y, x])*ERROR_RADIUS
            prior_bad += abs(error) > ERROR_RADIUS
            prior_sign_bad += error*intended < 0
            prior_inexact_endpoint += error != intended
            prior_max_abs = max(prior_max_abs, abs(error))
    checks = [(current_bad, "current observation outside the exact +/-1/2-DN witness"),
              (current_sign_bad, "current error sign differs from predeclared correlation"),
              (intended_renderer_mismatch, "current observation differs from deterministic inward rendering"),
              (prior_bad, "prior observation outside the exact +/-1/2-DN witness"),
              (prior_sign_bad, "prior error sign differs from predeclared correlation"),
              (prior_source_changes, "stored prior source changes background outside fixed footprint exclusions")]
    issues.extend(message for count, message in checks if count)
    return dict(
        passed=not issues, issues=issues, schema_version=1, uses_scores=False, synthetic_only=True,
        corrected_case_id=CORRECTED_CASE_ID, total_cases=20, unchanged_cases=unchanged_cases,
        full_annulus_pixel_count=int(annulus.sum()), changed_current_pixel_count=int(changed_count),
        current_witness_violation_count=int(current_bad), current_error_sign_violation_count=int(current_sign_bad),
        deterministic_renderer_mismatch_count=int(intended_renderer_mismatch),
        current_max_abs_error_exact=str(current_max_abs), current_max_inward_distance_exact=str(current_max_inward),
        current_source_stored_nonzero_pixel_count=source_nonzero,
        prior_witness_spatial_pixel_count=int(prior_domain.sum()), prior_witness_observation_count=int(prior_domain.sum())*8,
        prior_witness_violation_count=int(prior_bad), prior_error_sign_violation_count=int(prior_sign_bad),
        prior_error_not_exact_endpoint_count=int(prior_inexact_endpoint),
        prior_source_changes_stored_background_count=prior_source_changes, prior_max_abs_error_exact=str(prior_max_abs),
        preserved_original_current_full_annulus_violation_count=int(original_bad),
        exact_gain=str(gain), exact_error_radius=str(ERROR_RADIUS),
        observation_error_semantics="exact stored observation minus exact product of supplied binary64 latent values",
        guard_solver_or_current_selected_support_used=False,
        original_failure_not_repaired_in_place=True,
        physical_noise_or_guard_validity_certified=False,
    )
