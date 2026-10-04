"""Independent saved-synthetic accounting and exact guard-constraint audit.

No real media, journals, packets or remote access. This does not independently
rederive every endpoint numerical score: V45 math and the separately audited
V47 endpoint wrapper remain frozen dependencies. It independently reconstructs
guard membership, rational contrasts, scalar feasible intervals, and images.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_accuracy_v46_synthetic import (ARMS, ARRAY_KEYS, ADAPTER_KEYS, ROOT, audit_evidence,
    audit_geometry, array_fingerprint, expected_arrays, require, json_hash, sha, load, evidence_counts)
from accuracy_v48_synthetic_cases import CASE_IDS, TWIN_IDS, COUNTEREXAMPLE_IDS


BASE = ROOT/"results/tiny_target/accuracy_v48_20260926"
OLD = ROOT/"results/tiny_target/accuracy_v46_20260925/synthetic_01"
OLD_RECEIPT = "252954f7c0a28b390f2f91c952faede59878cedcd6a9a63d277150ad7da31b51"
PREVIOUS_ROOT = ROOT/"results/tiny_target/accuracy_v47_20260926"
PREVIOUS = PREVIOUS_ROOT/"synthetic_01"
PREVIOUS_RECEIPT = "fdd0d3538d9b085e4d823d7cc863a3afd98c855d4bfcb9c21984f808efaad653"
PREVIOUS_FAILURE = "8b74c1a177bea4b306accb8b9414fec2e7f6fcef983f9c384f1d9dc5a58bc25f"
PREVIOUS_MATH = "0a9f17e3c724d894ccceabed9400aa1ac6aa1f844cf6d39b6be654eb10208140"
CORRECTED_ID = "correlated_guard_error_extremes"
NEW_ARM = "bounded_background_guard_gain"
CURRENT_USE_DESCRIPTION = ("core values supply response and original finite support; "
                           "outer guard values supply conditional gain bounds only")


def f(value):
    return Fraction(*float(value).as_integer_ratio())


def independently_round_boundary(latent, sign):
    """Find the inside representable endpoint without the renderer's helper."""
    require(isinstance(latent, Fraction) and sign in (-1, 1), "Exact latent and signed boundary required")
    boundary = latent + Fraction(sign, 2)
    observed = float(boundary)
    if sign*(f(observed)-boundary) > 0:
        observed = math.nextafter(observed, -math.inf if sign == 1 else math.inf)
    require(math.isfinite(observed) and abs(f(observed)-latent) <= Fraction(1, 2),
            "No representable observation in declared error interval")
    neighbor = math.nextafter(observed, math.inf if sign == 1 else -math.inf)
    require(sign*(f(neighbor)-boundary) > 0, "Selected observation is not the inward boundary endpoint")
    return observed


def independent_interval(constraints):
    """Collect all exact lower/upper halfspace bounds without helper reuse."""
    lows, highs, impossible = [Fraction(0)], [], False
    for background, response, eb, ey in constraints:
        for a, b in ((background+eb, response-ey), (eb-background, -response-ey)):
            if a > 0:
                lows.append(b/a)
            elif a < 0:
                highs.append(b/a)
            else:
                impossible |= b > 0
    lower, upper = max(lows), min(highs) if highs else None
    feasible = not impossible and (upper is None or lower <= upper)
    return feasible, lower if feasible else None, upper if feasible else None


def _guard_layout(history, background, bound, centers):
    candidates = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                  if 40 <= max(abs(x-64), abs(y-64)) <= 56]
    eligible = []
    for x, y in candidates:
        distant = all(p is None or abs(x-p[0]) > 12 or abs(y-p[1]) > 12 for p in centers)
        if (distant and np.isfinite(history[:, y, x]).all() and np.isfinite(background[y, x])
                and np.isfinite(bound[y, x]) and bound[y, x] >= 0):
            eligible.append((x, y))
    point_set, stencils = set(eligible), []
    for axis in (0, 1):
        for x, y in candidates:
            points = [(x-8, y), (x, y), (x+8, y)] if axis == 0 else [(x, y-8), (x, y), (x, y+8)]
            if set(points) <= point_set:
                stencils.append(dict(axis="x" if axis == 0 else "y", center_xy=[x, y],
                                     pixels_xy=[list(p) for p in points], weights=[1, -2, 1]))
    used = sorted({tuple(p) for s in stencils for p in s["pixels_xy"]}, key=lambda p: (p[1], p[0]))
    return candidates, eligible, stencils, used


def audit_gain(inputs, background, bound, value):
    """Rebuild saved guard evidence from pixel values with exact fractions."""
    candidates, eligible, stencils, used = _guard_layout(inputs["history129"], background, bound, inputs["prior_centers_xy"])
    require(value["stencils"] == stencils and value["used_points_xy"] == [list(p) for p in used], "Guard membership changed")
    for key, count in (("candidate_count", len(candidates)), ("eligible_count", len(eligible)),
                       ("stencil_count", len(stencils)), ("used_count", len(used))):
        require(value[key] == count, "Guard count changed: "+key)
    def digest(v):
        return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    for key, content in (("candidate_support_sha256", candidates), ("eligible_support_sha256", eligible),
                         ("used_support_sha256", used), ("stencil_sha256", stencils)):
        require(value[key] == digest(content), "Guard support digest mismatch")
    require(value["motion_status"] == value["physical_class"] == "unknown", "Guard physical promotion")
    provenance = value["provenance"]
    for field in ("guard_purity_or_current_target_absence_certified", "global_affine_fit_or_joint_model_feasibility_certified",
                  "current_core_values_accessed"):
        require(provenance[field] is False, "Guard validity overclaim")
    for field in ("support_selection_uses_prior_data_only", "constraint_intersection_is_necessary_relaxation",
                  "same_frame_guard_uses_current_pixels", "no_independence_or_sample_count_reduction",
                  "sparse_grid_can_miss_between_grid_contamination", "horizontal_vertical_contrasts_also_annihilate_bilinear_xy"):
        require(provenance[field] is True, "Guard conditional scope changed")
    if not stencils:
        require(not value["available"] and value["reasons"] == ["no_prior_selected_guard_contrasts"], "Empty guard must remain unknown")
        return dict(stencils=0, contrasts=0, outcome="empty")
    nonfinite = sum(not np.isfinite(inputs["current129"][y, x]) for x, y in used)
    require(value["current_nonfinite_used_point_count"] == nonfinite, "Used current validity changed")
    if nonfinite:
        require(not value["available"] and value["reasons"] == ["nonfinite_current_on_fixed_used_guard_support"] and
                value["contrast_constraints"] == [], "Missing current guard support was silently removed")
        return dict(stencils=len(stencils), contrasts=0, outcome="missing_current")
    constraints, records = [], []
    for stencil in stencils:
        pixels = stencil["pixels_xy"]
        yv = sum(w*f(inputs["current129"][y, x]) for w, (x, y) in zip((1, -2, 1), pixels))
        bv = sum(w*f(background[y, x]) for w, (x, y) in zip((1, -2, 1), pixels))
        eb = sum(abs(w)*f(bound[y, x]) for w, (x, y) in zip((1, -2, 1), pixels))
        ey = Fraction(2)
        constraints.append((bv, yv, eb, ey))
        records.append(dict(response=str(yv), background=str(bv), response_error=str(ey), background_error=str(eb)))
    require(value["contrast_constraints"] == records and value["contrast_constraints_sha256"] == digest(records),
            "Guard contrast/error reconstruction mismatch")
    feasible, low, high = independent_interval(constraints)
    if not feasible:
        require(value["available"] is False and value["reasons"] == ["guard_gain_necessary_constraints_inconsistent"], "Inconsistent guard became usable")
        outcome = "inconsistent"
    elif high is None:
        require(value["available"] is False and value["reasons"] == ["guard_gain_outer_interval_unbounded"] and
                value["exact_gain_interval"] == [str(low), None], "Unbounded gain was capped")
        outcome = "unbounded"
    else:
        require(value["exact_gain_interval"] == [str(low), str(high)], "Rational endpoint mismatch")
        if value["available"]:
            a, b = value["gain_interval"]
            require(math.isfinite(a) and math.isfinite(b) and f(a) <= low <= high <= f(b), "Gain interval rounded inward")
            require(a >= 0 and value["reasons"] == [], "Invalid available gain interval")
            outcome = "finite"
        else:
            require(value["reasons"] == ["guard_gain_interval_not_finitely_representable"], "Finite exact gain unexpectedly unavailable")
            outcome = "unrepresentable"
    return dict(stencils=len(stencils), contrasts=len(constraints), outcome=outcome)


def audit_guard_images(cases, manifest):
    """Independently render full saved synthetic fields from declared truth."""
    require(tuple(c["case_id"] for c in cases) == CASE_IDS and manifest["case_count"] == 20, "Guard case membership changed")
    specs = {s["case_id"]: s for s in manifest["scenarios"]}
    x, y = np.meshgrid(np.arange(129, dtype=float), np.arange(129, dtype=float))
    r = np.maximum(abs(x-64), abs(y-64)); band = (r >= 40) & (r <= 56)
    true_priors = [[32.+4*i, 64.] for i in range(8)]
    def point(cx, cy, amplitude, sigma):
        return amplitude*np.exp(-.5*((x-cx)/sigma)**2-.5*((y-cy)/sigma)**2)
    maximum = 0.
    for c in cases:
        a, t, p = c["adapter_inputs"], c["generator_truth"], c["provenance"]
        parameters = specs[c["case_id"]]["generation_parameters"]
        require(p["generation_parameters"] == parameters and set(a) == set(ADAPTER_KEYS), "Truth or spec leaked into adapter")
        require(a["prior_centers_xy"] == true_priors and a["predicted_offset_xy"] == [0., 0.], "Prior-only geometry changed")
        require(t["effective_background_gain_core"] == parameters["gain_core"] and
                t["effective_background_gain_guard"] == parameters["gain_guard"] and
                t["current_source_peak_dn"] == parameters["current_source_peak_dn"] and
                t["prior_source_peak_dn"] == parameters["prior_source_peak_dn"], "Gain/source truth changed")
        for key in ("known_guard_model_violation", "known_core_model_violation", "gain_transfer_valid_in_declared_world",
                    "correlated_guard_error", "current_guard_nan_xy"):
            require(t[key] == parameters[key], "Model-validity truth changed")
        require(t["current_source_present"] == (parameters["current_source_peak_dn"] != 0) and
                t["guard_validity_or_transfer_certified_from_observations"] is False and
                t["physical_motion_and_class_certified"] is False, "Truth became a certified decision")
        if parameters["background_kind"] == "step_sine":
            background = 50+20*(x >= 64)+8*np.sin(y/12)
        else:
            background = 50+.03*(x-64)-.02*(y-64)+20*(x >= 64)*(r <= 24)
            if parameters["background_kind"] == "near_affine_guard_core_step":
                background += .1*np.sin(y/12)
        prior_light = point(112, 64, parameters["prior_guard_light_peak_dn"], 1)
        history = np.stack([background+point(cx, cy, parameters["prior_source_peak_dn"], parameters["prior_source_sigma"])+prior_light
                            for cx, cy in true_priors])
        current = np.where(r <= 12, parameters["gain_core"], parameters["gain_guard"])*background
        c0, cx, cy = parameters["current_affine_plane"]
        current += c0+cx*(x-64)+cy*(y-64)
        current += point(64, 64, parameters["current_source_peak_dn"], parameters["current_source_sigma"])
        current += point(112, 64, parameters["current_guard_light_peak_dn"], 1)
        current += point(112, 64, parameters["current_new_guard_object_peak_dn"], 1)
        if parameters["correlated_guard_error"]:
            error = .5*np.where(((x//8+y//8).astype(int) & 1) == 0, 1, -1)*band
            history += error
            current -= error
            gain_exact = f(parameters["gain_guard"])
            for iy, ix in zip(*np.nonzero(band)):
                sign = -1 if (int(ix)//8 + int(iy)//8) % 2 == 0 else 1
                current[iy, ix] = independently_round_boundary(gain_exact*f(background[iy, ix]), sign)
            require(np.array_equal(a["current129"][band], current[band]),
                    "Corrected annulus does not equal exact inward endpoint rendering")
        if parameters["current_guard_nan_xy"] is not None:
            nx, ny = map(int, parameters["current_guard_nan_xy"])
            current[ny, nx] = np.nan
        for key, expected in (("current129", current), ("history129", history)):
            actual = a[key]
            require(actual.shape == expected.shape and actual.dtype == np.float64 and
                    np.array_equal(np.isnan(actual), np.isnan(expected)), "Saved image type/support changed")
            error = float(np.nanmax(abs(actual-expected)))
            require(error <= 1e-12, "Independent guard-world rendering mismatch")
            maximum = max(maximum, error)
    by_id = {c["case_id"]: c for c in cases}
    for key in ADAPTER_KEYS:
        first, second = [by_id[c]["adapter_inputs"][key] for c in TWIN_IDS]
        require(np.array_equal(first, second, equal_nan=True) if isinstance(first, np.ndarray) else first == second,
                "Latent-world twins changed inputs")
    first, second = [by_id[c] for c in COUNTEREXAMPLE_IDS]
    require(np.array_equal(first["adapter_inputs"]["history129"], second["adapter_inputs"]["history129"]), "Current-gain counterexample history changed")
    require(first["generator_truth"]["effective_background_gain_guard"] != second["generator_truth"]["effective_background_gain_guard"],
            "Past-gain counterexample lost gain difference")
    return dict(cases=20, full_images_reconstructed=180, render_arithmetic_atol=1e-12, max_abs_render_difference=maximum,
                twin_inputs_identical=True, same_history_different_current_gain=True)


def validate_hull(value):
    from audit_accuracy_v45_accounting import validate_evidence
    require(value["motion_status"] == value["physical_class"] == "unknown" and
            value["is_motion_or_classification_gate"] is False and value["gain_provenance_certified"] is False,
            "Bounded-gain result promoted a conditional or physical claim")
    require(value["quantity"] == "bounded_gain_nuisance_residualized_source_numerator" and
            value["numerator"] is None and value["error_bound"] is None, "Hull mislabeled as a single amplitude/numerator")
    diagnostics = value["diagnostics"]
    require(diagnostics["whole_pipeline_is_ieee_certified_enclosure"] is False and
            diagnostics["physical_guard_cleanliness_and_core_transfer_not_certified"] is True and
            diagnostics["fixed_columns_not_dropped_by_wrapper"] is True and
            diagnostics["common_support_not_changed"] is True, "Numerical or model-validity limitation omitted")
    endpoints = value["endpoint_evaluations"]
    for endpoint in endpoints:
        record = endpoint["source_presence"]
        if record is not None:
            validate_evidence(record, "numerator")
            require(endpoint["available"] is record["available"], "Endpoint availability changed")
    if value["available"]:
        require(len(endpoints) == 2 and all(e["available"] for e in endpoints), "Unavailable endpoint omitted")
        require([e["gain"] for e in endpoints] == value["gain_interval"], "Endpoints do not bracket admitted gain")
        lower = min(e["source_presence"]["interval"][0] for e in endpoints)
        upper = max(e["source_presence"]["interval"][1] for e in endpoints)
        sign = "positive" if lower > 0 else "negative" if upper < 0 else "unresolved"
        require(value["interval"] == [lower, upper] and value["coefficient_sign"] == sign and
                value["interval_excludes_zero"] is (sign != "unresolved"), "Favorable endpoint selected instead of hull")
    else:
        require(value["reasons"] and value["interval"] is None and value["coefficient_sign"] is None and
                value["interval_excludes_zero"] is None, "Unavailable hull carries operative evidence")


def validate_context_override(wrapped, baseline):
    context = dict(wrapped["raw_adapter_result"]["prior_context"])
    previous = dict(baseline["prior_context"])
    old_use = previous.pop("current_values_used_for")
    require(context.pop("current_values_used_for") == CURRENT_USE_DESCRIPTION and context == previous,
            "Prior context changed beyond declared current guard-use metadata")
    require(wrapped["current_use_metadata_override"] == dict(
        path="raw_adapter_result.prior_context.current_values_used_for", **{"from":old_use, "to":CURRENT_USE_DESCRIPTION}),
        "Current-use metadata override missing or incorrectly disclosed")


def validate_preserved_arrays(case_id, actual, previous):
    """Only the single declared current annulus is eligible for any change."""
    require(set(actual) == set(previous) == set(ARRAY_KEYS), "Preserved array denominator changed")
    corrected = case_id == "guard_"+CORRECTED_ID
    for key in ARRAY_KEYS:
        if corrected and key == "current129":
            yy, xx = np.indices((129,129))
            radius = np.maximum(abs(xx-64), abs(yy-64))
            band = (radius >= 40) & (radius <= 56)
            require(actual[key].dtype == previous[key].dtype == np.float64 and
                    actual[key].shape == previous[key].shape == (129,129), "Corrected image type changed")
            require(np.array_equal(actual[key][~band], previous[key][~band], equal_nan=True),
                    "Renderer changed a current pixel outside the declared full annulus")
            require(not np.array_equal(actual[key][band], previous[key][band], equal_nan=True),
                    "Corrected annulus did not change the failing fixture")
        else:
            require(array_fingerprint(actual[key]) == array_fingerprint(previous[key]),
                    "Preserved V47 input changed: "+case_id+":"+key)


def audit_previous_boundary_failure():
    """Reproduce the retained bad fixture; a passing new run cannot erase it."""
    records = {r["case_id"]:r for r in load(PREVIOUS/"results.json")}
    record = records["guard_"+CORRECTED_ID]["arms"][NEW_ARM]
    gain = record["gain_calibration"]
    with np.load(PREVIOUS/"inputs"/("guard_"+CORRECTED_ID+".npz"), allow_pickle=False) as archive:
        current = archive["current129"]
    yy, xx = np.indices((129,129), dtype=np.float64)
    background = 50+20*(xx >= 64)+8*np.sin(yy/12)
    true_gain = f(1.2)
    excesses = [max(Fraction(0), abs(f(current[y,x])-true_gain*f(background[y,x]))-Fraction(1,2))
                for x,y in gain["used_points_xy"]]
    contrast_excesses = [max(Fraction(0), abs(Fraction(c["response"])-true_gain*Fraction(c["background"]))-
                          Fraction(c["response_error"])-true_gain*Fraction(c["background_error"]))
                         for c in gain["contrast_constraints"]]
    require(len(excesses) == 141 and sum(e > 0 for e in excesses) == 73 and
            len(contrast_excesses) == 184 and sum(e > 0 for e in contrast_excesses) == 59,
            "Frozen predecessor failure no longer reproduces")
    require(gain["available"] is False and gain["reasons"] == ["guard_gain_necessary_constraints_inconsistent"] and
            record["raw_adapter_result"]["available"] is False, "Invalid predecessor was promoted from unknown")
    return dict(retained_invalid_current_pixels=73, retained_used_pixels=141,
                retained_invalid_constraints=59, retained_constraints=184,
                maximum_current_excess_exact=str(max(excesses)), maximum_constraint_excess_exact=str(max(contrast_excesses)),
                retained_exact_guard_outcome="unknown", failed_predecessor_not_overwritten=True)


def audit_rendering_contract(cases, witness):
    """Recompute every witness fact independently, never calling the repair.

    Array hashes bind the complete per-pixel data, while exact rational maxima,
    counts and deterministic endpoint equality verify the pre-score report.
    """
    from accuracy_v47_synthetic_cases import build_cases as previous_cases
    previous = previous_cases()
    require([c["case_id"] for c in cases] == [c["case_id"] for c in previous], "Witness case membership changed")
    for current_case, old_case in zip(cases, previous):
        meta = deepcopy({k:v for k,v in current_case.items() if k != "adapter_inputs"})
        if current_case["case_id"] == CORRECTED_ID:
            correction = meta["generator_truth"].pop("current_error_rendering_v48")
            require(meta["provenance"].pop("renderer_correction_v48") == correction,
                    "Correction truth/provenance descriptions disagree")
            require(correction["affected_case_id"] == CORRECTED_ID and correction["affected_array"] == "current129" and
                    correction["error_radius_dn"] == .5 and correction["original_v47_failure_preserved"] is True and
                    correction["inherited_error_formula_describes_intended_not_exact_stored_error"] is True,
                    "Renderer correction not explicitly declared")
            for key in ("arbitrary_epsilon_used", "current_core_changed", "history_or_background_changed",
                        "geometry_or_polarity_changed", "detector_or_gain_solver_used_for_rendering",
                        "guard_selection_used_for_rendering", "sensor_noise_contract_validated"):
                require(correction[key] is False, "Renderer metadata claims undeclared calibration or changed model")
        require(meta == {k:v for k,v in old_case.items() if k != "adapter_inputs"}, "Inherited case metadata changed")
        validate_preserved_arrays("guard_"+current_case["case_id"], expected_arrays(current_case["adapter_inputs"]),
                                  expected_arrays(old_case["adapter_inputs"]))
    corrected = next(c for c in cases if c["case_id"] == CORRECTED_ID)["adapter_inputs"]
    old = next(c for c in previous if c["case_id"] == CORRECTED_ID)["adapter_inputs"]
    current, history = corrected["current129"], corrected["history129"]
    yy, xx = np.indices((129,129), dtype=np.float64)
    radius = np.maximum(abs(xx-64), abs(yy-64)); band = (radius >= 40) & (radius <= 56)
    background = 50+20*(xx >= 64)+8*np.sin(yy/12)
    gain = f(1.2); half = Fraction(1,2)
    current_bad = current_sign_bad = mismatch = original_bad = changed = 0
    current_max, inward_max = Fraction(0), Fraction(0)
    for y,x in zip(*np.nonzero(band)):
        prior_sign = 1 if (int(x)//8 + int(y)//8) % 2 == 0 else -1
        latent = gain*f(background[y,x]); intended = -prior_sign*half
        error = f(current[y,x])-latent
        current_bad += abs(error) > half
        current_sign_bad += error*intended < 0
        mismatch += current[y,x] != independently_round_boundary(latent, -prior_sign)
        original_bad += abs(f(old["current129"][y,x])-latent) > half
        changed += current[y,x] != old["current129"][y,x]
        current_max, inward_max = max(current_max, abs(error)), max(inward_max, abs(intended-error))
    source = 30*np.exp(-.5*((xx-64)**2+(yy-64)**2))
    source_nonzero = int(np.count_nonzero(source[band]))
    prior_domain = band.copy()
    for cx,cy in old["prior_centers_xy"]:
        prior_domain &= (abs(xx-cx) > 12) | (abs(yy-cy) > 12)
    prior_bad = prior_sign_bad = prior_endpoint_bad = prior_source_changed = 0
    prior_max = Fraction(0)
    for i,(cx,cy) in enumerate(old["prior_centers_xy"]):
        source = 30*np.exp(-.5*((xx-cx)**2+(yy-cy)**2))
        prior_source_changed += int(np.count_nonzero((background+source)[prior_domain] != background[prior_domain]))
        for y,x in zip(*np.nonzero(prior_domain)):
            intended = half if (int(x)//8+int(y)//8) % 2 == 0 else -half
            error = f(history[i,y,x])-f(background[y,x])
            prior_bad += abs(error) > half
            prior_sign_bad += error*intended < 0
            prior_endpoint_bad += error != intended
            prior_max = max(prior_max, abs(error))
    require(not any((current_bad,current_sign_bad,mismatch,prior_bad,prior_sign_bad,
                     source_nonzero,prior_source_changed)), "Independent exact error witness failed")
    expected = dict(passed=True, issues=[], schema_version=1, uses_scores=False, synthetic_only=True,
        corrected_case_id=CORRECTED_ID, total_cases=20, unchanged_cases=19,
        full_annulus_pixel_count=int(band.sum()), changed_current_pixel_count=int(changed),
        current_witness_violation_count=int(current_bad), current_error_sign_violation_count=int(current_sign_bad),
        deterministic_renderer_mismatch_count=int(mismatch), current_max_abs_error_exact=str(current_max),
        current_max_inward_distance_exact=str(inward_max), current_source_stored_nonzero_pixel_count=source_nonzero,
        prior_witness_spatial_pixel_count=int(prior_domain.sum()), prior_witness_observation_count=int(prior_domain.sum())*8,
        prior_witness_violation_count=int(prior_bad), prior_error_sign_violation_count=int(prior_sign_bad),
        prior_error_not_exact_endpoint_count=int(prior_endpoint_bad),
        prior_source_changes_stored_background_count=prior_source_changed, prior_max_abs_error_exact=str(prior_max),
        preserved_original_current_full_annulus_violation_count=int(original_bad), exact_gain=str(gain), exact_error_radius=str(half),
        observation_error_semantics="exact stored observation minus exact product of supplied binary64 latent values",
        guard_solver_or_current_selected_support_used=False, original_failure_not_repaired_in_place=True,
        physical_noise_or_guard_validity_certified=False)
    require({k:v for k,v in witness.items() if k != "created_at_utc"} == expected,
            "Pre-score exact rendering witness differs from independent reconstruction")
    return dict(independently_recomputed_witness=expected, exact_fraction_arithmetic=True,
                producer_repair_helper_or_witness_function_called=False,
                current_witness_uses_full_annulus_not_guard_selection=True,
                prior_source_footprints_excluded_by_all_prior_centers=True)


def verify_bindings(bindings, run):
    allowed_runs = (ROOT/"results/tiny_target/accuracy_v44_20260925/synthetic_01",
                    ROOT/"results/tiny_target/accuracy_v45_20260925/synthetic_01", OLD, PREVIOUS, run)
    predecessor_reports = (PREVIOUS_ROOT/"synthetic_independent_audit_01.json",
                           PREVIOUS_ROOT/"math_independent_audit_01.json")
    for name, digest in bindings.items():
        path = Path(name).resolve()
        require(str(path) == name and not Path(name).is_symlink(), "Noncanonical or symlink audit dependency")
        allowed = (path.parent == ROOT/"scripts" and path.suffix == ".py" or
                   path.parent == ROOT/"tests/unit" and path.suffix == ".py" or
                   path in [ROOT/"docs"/f"accuracy_v{v}_plan.md" for v in (44, 45, 46, 47, 48)] or
                   path in predecessor_reports or
                   any(p in path.parents for p in allowed_runs))
        require(allowed, "Audit binding outside synthetic-only scope")
        require(sha(path) == digest, "Frozen hash changed: "+name)


def audit_predecessor(receipt, freeze, run):
    """A failed predecessor remains failed; only its immutable math is reused."""
    pinned = {PREVIOUS/"completion_receipt.json": PREVIOUS_RECEIPT,
              PREVIOUS_ROOT/"synthetic_independent_audit_01.json": PREVIOUS_FAILURE,
              PREVIOUS_ROOT/"math_independent_audit_01.json": PREVIOUS_MATH}
    for path, digest in pinned.items():
        require(sha(path) == digest and freeze["files_sha256"].get(str(path)) == digest,
                "Predecessor evidence is not exactly pinned")
    require(freeze["predecessor_receipt_sha256"] == PREVIOUS_RECEIPT and
            freeze["predecessor_failure_audit_sha256"] == PREVIOUS_FAILURE and
            freeze["math_audit_sha256"] == PREVIOUS_MATH, "Predecessor declaration changed")
    previous = load(PREVIOUS/"completion_receipt.json")
    failure = load(PREVIOUS_ROOT/"synthetic_independent_audit_01.json")
    math_report = load(PREVIOUS_ROOT/"math_independent_audit_01.json")
    require(previous["completed"] and previous["synthetic_only"] and not previous["real_packet_data_read"],
            "Predecessor is not synthetic-only")
    require(failure["passed"] is False and failure["completed"] is False and
            failure["synthetic_completion_receipt_sha256"] == PREVIOUS_RECEIPT and
            any(i["code"] == "declared_valid_fixture_exceeds_exact_error_contract" for i in failure["issues"]),
            "Failed V47 audit was erased or promoted")
    require(math_report["passed"] is True and math_report["completed"] is True and math_report["issues"] == [],
            "Mathematical predecessor audit failed")
    verify_bindings(previous["files_sha256"], run)
    for name, digest in previous["files_sha256"].items():
        require(freeze["files_sha256"].get(name) == digest and receipt["files_sha256"].get(name) == digest,
                "V47 dependency changed or disappeared")
    for name, digest in math_report["audit_files_sha256"].items():
        require(previous["files_sha256"].get(name) == digest and freeze["files_sha256"].get(name) == digest,
                "Mathematical method no longer matches audited frozen V47 implementation")
    return dict(predecessor_completion_receipt_sha256=PREVIOUS_RECEIPT,
                predecessor_failure_audit_sha256=PREVIOUS_FAILURE, predecessor_failure_retained=True,
                inherited_math_audit_sha256=PREVIOUS_MATH, all_method_dependencies_unchanged=True,
                failed_predecessor_is_not_readiness_authorization=True)


def audit(run):
    from accuracy_v43_components import prepare_components
    from accuracy_v45_bounds import component_bounds
    from accuracy_v46_synthetic_cases import build_cases as build_old, scenario_manifest as old_manifest
    from accuracy_v48_synthetic_cases import build_cases, scenario_manifest
    from accuracy_v47_guard_gain import estimate_guard_gain
    from run_accuracy_v44_synthetic import causal_cases
    run = Path(run).resolve()
    require(run.parent == BASE, "Audit only a direct V48 synthetic child")
    receipt, old_receipt = load(run/"completion_receipt.json"), load(OLD/"completion_receipt.json")
    require(sha(OLD/"completion_receipt.json") == OLD_RECEIPT, "V46 receipt changed")
    require(receipt["completed"] is True and receipt["synthetic_only"] is True and receipt["real_packet_data_read"] is False,
            "Incomplete/non-synthetic run")
    verify_bindings(receipt["files_sha256"], run)
    verify_bindings(old_receipt["files_sha256"], run)
    freeze, saved, witness, start, metadata, records, summary = [load(run/name) for name in
        ("freeze.json", "inputs_complete.json", "rendering_contract.json", "score_start.json",
         "input_manifest.json", "results.json", "summary.json")]
    predecessor_audit = audit_predecessor(receipt, freeze, run)
    predecessor_audit["independent_retained_failure_reconstruction"] = audit_previous_boundary_failure()
    require(freeze["baseline_receipt_sha256"] == OLD_RECEIPT and freeze["old_arms"] == {a:list(v) for a,v in ARMS.items()}
            and freeze["new_arm"] == NEW_ARM, "Frozen arms/baseline changed")
    require(all(freeze["files_sha256"].get(p) == h for p,h in old_receipt["files_sha256"].items()), "V46 dependencies no longer pinned")
    expected_bindings = set(freeze["files_sha256"]) | set(saved["files_sha256"]) | {
        str(run/name) for name in ("freeze.json", "inputs_complete.json", "rendering_contract.json", "score_start.json",
                                  "input_manifest.json", "results.json", "summary.json")}
    require(set(receipt["files_sha256"]) == expected_bindings, "Completion file accounting changed")
    for bindings in (freeze["files_sha256"], saved["files_sha256"]):
        require(all(receipt["files_sha256"].get(p) == h for p,h in bindings.items()), "Nested hash binding changed")
    for key, filename in (("freeze_sha256", "freeze.json"), ("inputs_complete_sha256", "inputs_complete.json"),
                          ("input_manifest_sha256", "input_manifest.json"), ("rendering_contract_sha256", "rendering_contract.json")):
        require(start[key] == sha(run/filename), "Score-start provenance changed")
    times = [datetime.fromisoformat(item["created_at_utc"]) for item in (freeze, saved, witness, start, receipt)]
    require(times == sorted(times) and saved["scores_started"] is False, "Producer chronology changed")
    require(freeze["rendering_contract_sha256"] == json_hash({k:v for k,v in witness.items() if k != "created_at_utc"}),
            "Frozen pre-score rendering contract hash changed")
    old_stress, new_cases, manifest = build_old(), build_cases(), scenario_manifest()
    geometry = audit_geometry(old_stress, old_manifest())
    image_audit = audit_guard_images(new_cases, manifest)
    rendering_contract_audit = audit_rendering_contract(new_cases, witness)
    require(freeze["guard_scenario_manifest"] == manifest, "Guard manifest changed")
    cases = []
    for case in causal_cases():
        cases.append(dict(case_id="v46_baseline_"+case["case_id"], group="v46_baseline", reference_case_id=case["case_id"],
            adapter_inputs={k:case[k] for k in ADAPTER_KEYS}, generator_truth=None,
            provenance={"forecast_origin":"unchanged_legacy_caller_supplied_assumption"}))
    for group, cohort in (("v46_stress", old_stress), ("guard", new_cases)):
        for case in cohort:
            cases.append(dict(case_id=group+"_"+case["case_id"], group=group, reference_case_id=case["case_id"],
                adapter_inputs=case["adapter_inputs"], generator_truth=case["generator_truth"], provenance=case["provenance"]))
    require(len(cases) == len(records) == 54 and [c["case_id"] for c in cases] == freeze["case_ids"] == [r["case_id"] for r in records],
            "54-case membership/order changed")
    expected_metadata = [dict({k:v for k,v in c.items() if k != "adapter_inputs"}, polarity=c["adapter_inputs"]["polarity"]) for c in cases]
    require(metadata == expected_metadata and json_hash(metadata) == freeze["input_metadata_sha256"], "Input truth/provenance changed")
    fingerprints, paths = {}, set()
    for case in cases:
        path = run/"inputs"/(case["case_id"]+".npz"); paths.add(str(path))
        expected = expected_arrays(case["adapter_inputs"])
        with np.load(path, allow_pickle=False) as actual:
            require(set(actual.files) == set(ARRAY_KEYS), "Input array denominator changed")
            for key in ARRAY_KEYS:
                require(array_fingerprint(actual[key]) == array_fingerprint(expected[key]), "Saved synthetic input changed: "+case["case_id"])
                fingerprints[case["case_id"]+":"+key] = array_fingerprint(actual[key])
            with np.load(PREVIOUS/"inputs"/(case["case_id"]+".npz"), allow_pickle=False) as previous:
                validate_preserved_arrays(case["case_id"], dict(actual), dict(previous))
            if case["group"] != "guard":
                prefix = "baseline_" if case["group"] == "v46_baseline" else "stress_"
                with np.load(OLD/"inputs"/(prefix+case["reference_case_id"]+".npz"), allow_pickle=False) as old:
                    require(set(old.files) == set(actual.files), "V46 input member changed")
                    for key in ARRAY_KEYS:
                        require(array_fingerprint(actual[key]) == array_fingerprint(old[key]), "V46 baseline input bytes changed")
    require(fingerprints == freeze["input_memory_sha256"] and set(saved["files_sha256"]) == paths ==
            {str(p.resolve()) for p in (run/"inputs").iterdir()}, "Saved/pre-score input membership or hashes changed")
    old_results = {}
    for group, name in (("v46_baseline", "baseline_results.json"), ("v46_stress", "stress_results.json")):
        old_results[group] = {r["case_id"]:r["arms"] for r in load(OLD/name)}
    previous_records = {r["case_id"]:r for r in load(PREVIOUS/"results.json")}
    require(set(previous_records) == {c["case_id"] for c in cases}, "V47 case identity changed")
    gain_audits, details = [], []
    for case, record in zip(cases, records):
        if case["case_id"] != "guard_"+CORRECTED_ID:
            require(record == previous_records[case["case_id"]], "Unchanged V47 case output/metadata failed exact reproduction")
        require({k:v for k,v in record.items() if k != "arms"} == {k:v for k,v in case.items() if k != "adapter_inputs"},
                "Saved result metadata changed")
        require(set(record["arms"]) == set(ARMS)|{NEW_ARM}, "One of five calculations omitted")
        old_four = {arm:record["arms"][arm] for arm in ARMS}
        audit_evidence([dict(case_id=case["case_id"], arms=old_four)])
        if case["group"] in old_results:
            require(old_four == old_results[case["group"]][case["reference_case_id"]], "V46 four-arm baseline changed")
        wrapped = record["arms"][NEW_ARM]; raw = wrapped["raw_adapter_result"]
        baseline = record["arms"]["presence_box_bounds"]["raw_adapter_result"]
        for layer in (wrapped, raw):
            require(layer["motion_status"] == layer["physical_class"] == "unknown" and layer["is_motion_or_classification_gate"] is False,
                    "New arm physical promotion")
        for field in ("learned_design_sha256", "common_support_sha256", "common_support_count", "components",
                      "component_bounds", "uncertainty_excludes"):
            require(raw[field] == baseline[field], "New arm changed original design/support/bounds: "+field)
        validate_context_override(wrapped, baseline)
        for field in ("conditional_on", "ambiguity_reasons"):
            require(raw[field][:len(baseline[field])] == baseline[field], "Existing ambiguity or assumption removed")
        require("outer_guard_background_validity_not_certified" in raw["ambiguity_reasons"] and
                "guard_to_core_shared_photometric_model_not_certified" in raw["ambiguity_reasons"] and
                wrapped["old_unrestricted_background_estimand_preserved"] is False and
                wrapped["uncertainty_excludes_guard_contamination_and_spatial_transfer_failure"] is True,
                "Changed estimand or conditional guard limitation omitted")
        contrast, gain = raw["numerical_contrast"], wrapped["gain_calibration"]
        require(raw["available"] is (contrast is not None and contrast["available"]), "New availability semantics changed")
        if contrast is not None:
            validate_hull(contrast)
            require(gain is not None, "Core evidence lacks guard provenance")
        if gain is not None:
            a = case["adapter_inputs"]
            components = prepare_components(a["history129"], a["prior_centers_xy"], a["predicted_offset_xy"], a["polarity"])
            bounds = component_bounds(a["history129"], a["prior_centers_xy"], a["predicted_offset_xy"], a["polarity"], components, protected=True)
            require(bounds["available"], "Gain stage bypassed unavailable component bounds")
            check = audit_gain(a, components["background"], bounds["background_bound129"], gain)
            if (case["group"] == "guard" and gain["contrast_constraints"] and
                    case["generator_truth"]["shared_gain_affine_background_model_valid_in_declared_world"]):
                true_gain = f(case["generator_truth"]["effective_background_gain_guard"])
                for constraint in gain["contrast_constraints"]:
                    residual = abs(Fraction(constraint["response"])-true_gain*Fraction(constraint["background"]))
                    limit = Fraction(constraint["response_error"])+true_gain*Fraction(constraint["background_error"])
                    require(residual <= limit, "Declared valid gain violates exact guard error contract: "+case["case_id"])
                check["declared_model_valid_gain_satisfies_every_exact_constraint"] = True
            for replacement in (np.nan, 1e200):
                changed = a["current129"].copy(); changed[52:77, 52:77] = replacement
                repeated = estimate_guard_gain(changed, a["history129"], components["background"],
                    bounds["background_bound129"], a["prior_centers_xy"])
                require(repeated == gain, "Core mutation changed guard interval/membership/provenance")
            if gain["available"]:
                require(contrast["gain_interval"] == gain["gain_interval"], "Core used a different gain interval")
                if case["group"] == "guard" and case["generator_truth"]["shared_gain_affine_background_model_valid_in_declared_world"]:
                    true_gain = f(case["generator_truth"]["effective_background_gain_guard"])
                    require(Fraction(gain["exact_gain_interval"][0]) <= true_gain <= Fraction(gain["exact_gain_interval"][1]),
                            "Declared model-valid synthetic gain excluded by exact guard constraints: "+case["case_id"])
                    check["declared_model_valid_true_guard_gain_contained"] = True
            else:
                require(not raw["available"], "Unavailable gain manufactured usable source evidence")
            gain_audits.append(dict(case_id=case["case_id"], **check, core_mutations_checked=2))
        details.append(dict(case_id=case["case_id"], group=case["group"], available=raw["available"], reasons=raw["reasons"],
            gain_interval=None if gain is None else gain["gain_interval"], gain_reasons=None if gain is None else gain["reasons"],
            interval=None if contrast is None else contrast["interval"], coefficient_sign=None if contrast is None else contrast["coefficient_sign"],
            guard_model_validity_unverified=True, core_transfer_unverified=True))
    lookup = {r["case_id"]:r for r in records}
    require(lookup["guard_"+TWIN_IDS[0]]["arms"] == lookup["guard_"+TWIN_IDS[1]]["arms"], "Latent explanation changed scores")
    group_counts = {g:{arm:evidence_counts([r for r in records if r["group"] == g], arm) for arm in (*ARMS, NEW_ARM)}
                    for g in ("v46_baseline", "v46_stress", "guard")}
    require(group_counts == summary["counts_by_group"] and summary["cases"] == 54 and summary["calculations"] == 5,
            "Summary denominators/counts changed")
    for field in ("v46_inputs_and_four_arm_results_exact", "unchanged_prior_design_support_and_v45_component_bounds",
                  "unchanged_53_v47_inputs_metadata_and_five_outputs_exact", "numerical_method_unchanged_from_v47",
                  "original_failed_v47_run_preserved",
                  "all_unknowns_retained", "all_motion_and_physical_class_unknown", "guard_is_same_frame_not_prior_only",
                  "guard_consistency_is_only_necessary_relaxation", "guard_to_core_transfer_not_certified",
                  "physical_identity_not_certified", "no_positive_fraction_accuracy_threshold"):
        require(summary[field] is True, "Summary overclaim or failed invariant")
    verify_bindings(receipt["files_sha256"], run)
    return dict(schema="seaqr.accuracy-v48-independent-synthetic-audit.v1", completed=True, passed=True, issues=[],
        created_at_utc=datetime.now(timezone.utc).isoformat(), stress_completion_receipt_sha256=sha(run/"completion_receipt.json"),
        synthetic_completion_receipt_sha256=sha(run/"completion_receipt.json"),
        baseline_completion_receipt_sha256=OLD_RECEIPT, checked_files_sha256=receipt["files_sha256"],
        predecessor_audit=predecessor_audit, pre_score_rendering_contract_audit=rendering_contract_audit,
        counts=dict(completion_bound_files=len(receipt["files_sha256"]), cases=54, input_archives=54, input_arrays=len(fingerprints),
                    preserved_v46_cases=34, guard_cases=20, regenerated_guard_cases=1,
                    preserved_v47_inputs_and_all_five_outputs=53, calculations=5, case_arm_entries=270,
                    guard_results_independently_reconstructed=len(gain_audits), current_core_mutations_checked=2*len(gain_audits)),
        old_stress_geometry_audit=geometry, guard_image_audit=image_audit, guard_constraint_audits=gain_audits,
        counts_by_group=group_counts, new_arm_details=details,
        all_old_inputs_and_four_arm_outputs_exact=True, prior_design_support_bounds_unchanged=True,
        unchanged_v47_53_inputs_and_five_outputs_exact=True, corrected_case_change_scope_exact=True,
        independent_exact_error_witness_checked_after_run=True, hash_bound_error_witness_precedes_scores=True,
        exact_current_guard_use_metadata_override_verified=True,
        all_truth_metadata_retained_but_not_adapter_inputs=True, twin_observations_and_all_outputs_identical=True,
        all_motion_and_physical_class_unknown=True, guard_validity_and_transfer_unverified=True,
        synthetic_only=True, real_data_accessed=False, production_changed=False,
        producer_chronology=[t.isoformat() for t in times],
        limitations=["Producer chronology is hash-bound, not externally clock-attested",
            "Guard interval is a necessary-constraint outer relaxation, not joint affine-model feasibility",
            "Guard purity, physical motion and transfer of gain to the target core are not established by interval availability",
            "Numerator endpoints reuse immutable separately audited numerical code; this accounting audit is not full IEEE certification",
            "Unknowns and non-positive intervals are not negative detections; new quantities are not unrestricted-background amplitudes"])


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run, output = args.run.resolve(), args.output.resolve()
    require(run.parent == BASE and output.parent == BASE and not output.exists(), "Fresh audit outside frozen V48 run required")
    receipt = run/"completion_receipt.json"
    require(load(receipt)["completed"] is True, "Wait for completed run")
    own = {str(p):sha(p) for p in (Path(__file__).resolve(), ROOT/"tests/unit/test_accuracy_v48_synthetic_audit.py", receipt)}
    freeze_path = output.with_name(output.stem+"_freeze.json")
    write_new(freeze_path, dict(created_at_utc=datetime.now(timezone.utc).isoformat(), files_sha256=own, source_scores_recomputed=False))
    try:
        result = audit(run)
    except Exception as exc:
        result = dict(schema="seaqr.accuracy-v48-independent-synthetic-audit-failure.v1",
            completed=False, passed=False, issues=[dict(type=type(exc).__name__, message=str(exc))],
            synthetic_completion_receipt_sha256=sha(receipt), synthetic_only=True,
            real_data_accessed=False, production_changed=False)
        result.update(audit_files_sha256=own, audit_freeze_sha256=sha(freeze_path))
        write_new(output, result)
        raise
    require(all(sha(p) == h for p,h in own.items()), "Audit inputs changed during execution")
    result.update(audit_files_sha256=own, audit_freeze_sha256=sha(freeze_path))
    write_new(output, result)
    print(json.dumps({k:result[k] for k in ("passed", "issues", "counts", "stress_completion_receipt_sha256")}, indent=2))


if __name__ == "__main__":
    main()
