"""Independent synthetic endpoint, native-perturbation and numerator audit.

No real media, real-image cache, journal, remote host or production configuration
is accessed. Seeded refits are bug finding, not proof or camera calibration.
Decimal checks independently reconstruct exact marginal normalization endpoints
from the binary floating inputs. Numerator refits use a QR nuisance projection
on the original unprojected [P,Z], independently of the core's two SVD stages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import numpy as np

import accuracy_v42_localized as legacy
import accuracy_v43_components as components43
import accuracy_v45_bounds as bounds45
from accuracy_v45_presence import source_presence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/tiny_target/accuracy_v45_20260925"
ENDPOINT_SEED = 450017
NUMERATOR_SEED = 450451
RAW_SEED = 450731
REFITS_PER_CASE = 120
RAW_DRAWS = 40
ARRAY_KEYS = ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound")
INITIAL_ENDPOINT_FINDING = {
    "core_sha256": "f5fbc4df32653fe9d3efdcd944bc3dd267ddb447eb938454aebf199cde80df2a",
    "seed": ENDPOINT_SEED, "decimal_precision": 100,
    "dimensions": [2, 3, 17, 289, 1024], "trials_per_dimension": 4,
    "endpoint_comparisons": 10680, "strict_inward_endpoints": 3437,
    "maximum_inward_ulps": 11.288214229799543,
    "scope": "Pre-freeze single-nextafter implementation under-enclosed exact marginal endpoints by tiny accumulated floating roundoff; normalization algebra was correct.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _issue(issues, condition, name, **details):
    if not condition:
        issues.append(dict(check=name, **details))


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return not isinstance(value, float) or bool(np.isfinite(value))


def endpoint_boxes():
    rng = np.random.default_rng(ENDPOINT_SEED)
    cases = []
    for dimension in (2, 3, 17, 289, 1024):
        for trial in range(4):
            lower = 10.0**rng.uniform(-8, 8, dimension)
            upper = lower*(1+rng.uniform(0, 1, dimension))
            cases.append((f"decimal_{dimension}_{trial}", lower, upper))
    return cases


def decimal_endpoints(lower, upper):
    """Independent 100-digit formula on exact representations of binary inputs."""
    with localcontext() as context:
        context.prec = 100
        lo = [Decimal.from_float(float(x)) for x in np.asarray(lower).ravel()]
        hi = [Decimal.from_float(float(x)) for x in np.asarray(upper).ravel()]
        sl, su = sum(x*x for x in lo), sum(x*x for x in hi)
        return [(lo[i]/(lo[i]*lo[i]+su-hi[i]*hi[i]).sqrt(),
                 hi[i]/(hi[i]*hi[i]+sl-lo[i]*lo[i]).sqrt()) for i in range(len(lo))]


def audit_endpoints():
    issues, comparisons, max_inward_ulps = [], 0, 0.0
    for name, lower, upper in endpoint_boxes():
        low, high, metadata = bounds45.normalization_interval(lower, upper)
        _issue(issues, metadata["available"], "decimal_box_unavailable", case_id=name)
        if low is None:
            continue
        for index, (exact_low, exact_high) in enumerate(decimal_endpoints(lower, upper)):
            for side, actual, exact in (("lower", low[index], exact_low), ("upper", high[index], exact_high)):
                delta = Decimal.from_float(float(actual))-exact
                inward = delta > 0 if side == "lower" else delta < 0
                if inward:
                    ulps = float(abs(delta)/Decimal.from_float(abs(float(np.spacing(actual)))))
                    max_inward_ulps = max(max_inward_ulps, ulps)
                    issues.append(dict(check="strict_decimal_endpoint_under_enclosure", case_id=name,
                                       coordinate=index, side=side, inward_ulps=ulps))
                comparisons += 1
    corners = 0
    for dimension in (2, 3, 4):
        lower = np.linspace(.2, .8, dimension)
        upper = lower + np.linspace(.3, .1, dimension)
        low, high, _ = bounds45.normalization_interval(lower, upper)
        for choice in itertools.product((False, True), repeat=dimension):
            vector = np.where(choice, upper, lower)
            normalized = vector/np.linalg.norm(vector)
            _issue(issues, np.all(normalized >= low-1e-15) and np.all(normalized <= high+1e-15),
                   "normalization_corner_outside", dimension=dimension, choice=list(choice))
            corners += 1
    lower, upper = np.array([0., np.nan, .2, .4]), np.array([0., np.nan, .8, .9])
    low, high, _ = bounds45.normalization_interval(lower, upper)
    _issue(issues, np.array_equal(np.isnan(low), np.isnan(lower)) and low[0] == high[0] == 0,
           "normalization_nan_zero_support_changed")
    declined = bounds45.normalization_interval(np.zeros(3), np.ones(3))
    _issue(issues, declined[0] is None and declined[1] is None, "zero_lower_norm_not_declined")
    return dict(endpoint_comparisons=comparisons, exhaustive_corner_vectors=corners,
                strict_inward_endpoints=sum(i["check"] == "strict_decimal_endpoint_under_enclosure" for i in issues),
                maximum_inward_ulps=max_inward_ulps, issues=issues)


def numerator_cases():
    rng = np.random.default_rng(NUMERATOR_SEED)
    cases = []
    for n in (32, 64):
        x = np.linspace(-1, 1, n)
        plane = np.column_stack((np.ones(n), x))
        for nuisance_count in (0, 2):
            for coefficient in (-12., 0., 12.):
                for residual_scale in (0., 20.):
                    for error in (0., .005):
                        raw = rng.normal(size=(n, nuisance_count+1))
                        response = plane@[100., -25.] + raw@np.r_[rng.normal(size=nuisance_count)*3, coefficient]
                        complete_basis = np.linalg.qr(np.column_stack((plane, raw)), mode="reduced")[0]
                        residual = rng.normal(size=n)
                        residual -= complete_basis@(complete_basis.T@residual)
                        cases.append(dict(case_id=f"grid_{len(cases):02d}", kind="grid",
                            y=response+residual_scale*residual, Z=raw[:, :-1], m=raw[:, -1], P=plane.copy(),
                            response_bound=np.linspace(.1, .5, n), nuisance_bound=np.full((n, nuisance_count), error),
                            source_bound=np.full(n, error)))
    n = 32
    p = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    m, z = rng.normal(size=(2, n))
    specs = [
        ("source_equals_nuisance", z[:, None], z, p, .0, .0, False),
        ("source_nearly_nuisance", z[:, None], z+1e-8*m, p, .005, .005, None),
        ("uncertain_zero_nuisance", np.zeros((n, 1)), m, p, .005, .0, False),
        ("exact_zero_nuisance", np.zeros((n, 1)), m, p, .0, .0, True),
        ("exact_affine_nuisance", p[:, :1], m, p, .0, .0, True),
        ("uncertain_affine_nuisance", p[:, :1], m, p, .005, .0, False),
        ("duplicate_nuisance", np.column_stack((z, z)), m, p, .0, .0, False),
        ("source_erasing_error", z[:, None], m, p, .0, None, True),
    ]
    for name, nuisance, source, plane, nuisance_error, source_error, expected in specs:
        cases.append(dict(case_id=name, kind="structural", expected_available=expected,
            y=plane@[100., -25.]+12*source, Z=nuisance, m=source, P=plane,
            response_bound=np.full(n, .5), nuisance_bound=np.full(nuisance.shape, nuisance_error),
            source_bound=abs(source) if source_error is None else np.full(n, source_error)))
    source = np.array([.5, -.5, .5, -.5]); plane = np.ones((4, 1))
    cases.append(dict(case_id="tiny_source_bearing_nuisance", kind="structural", expected_available=False,
        y=100+30*source, Z=plane+np.ldexp(source, -50)[:, None], m=source, P=plane,
        response_bound=np.zeros(4), nuisance_bound=np.zeros((4, 1)), source_bound=np.zeros(4)))
    for scale in (1., 1e6, 1e12):
        cases.append(dict(case_id="zero_error_exact_affine_"+str(scale), kind="zero_error_null",
            expected_available=True, y=np.full(n, scale), Z=np.empty((n, 0)), m=m.copy(), P=p.copy(),
            response_bound=np.zeros(n), nuisance_bound=np.empty((n, 0)), source_bound=np.zeros(n)))
    return cases


def _qr_projection(plane, nuisance):
    """Full-space QR with exact redundant columns removed only by literal proof."""
    kept = []
    for column in nuisance.T:
        if np.all(column == 0) or any(np.array_equal(column, sign*p) for p in plane.T for sign in (-1, 1)):
            continue
        kept.append(column)
    full = np.column_stack((plane, np.column_stack(kept))) if kept else plane
    if not full.shape[1]:
        return np.empty((len(nuisance), 0))
    # Column scaling leaves the range unchanged and avoids caller-unit effects.
    return np.linalg.qr(full/np.linalg.norm(full, axis=0), mode="reduced")[0]


def direct_numerator(y, nuisance, source, plane):
    basis = _qr_projection(plane, nuisance)
    response_residual = y-basis@(basis.T@y)
    source_residual = source-basis@(basis.T@source)
    return float(source_residual@response_residual), source_residual, response_residual, basis


def audit_numerator(cases):
    records, issues = [], []
    refits, invariances, zero_error_checks, gap_checks, conservative_controls = 0, 0, 0, 0, 0
    for index, case in enumerate(cases):
        args = {key: case[key] for key in ARRAY_KEYS}
        original = {key: np.asarray(value).copy() for key, value in args.items()}
        result = source_presence(**args)
        local = []
        _issue(local, _finite(result), "nonfinite_result")
        _issue(local, result["motion_status"] == result["physical_class"] == "unknown", "physical_sign_promotion")
        for key in ARRAY_KEYS:
            _issue(local, np.array_equal(original[key], args[key]), "input_mutated", field=key)
        if case.get("expected_available") is not None:
            _issue(local, result["available"] == case["expected_available"], "structural_availability_changed")
        if not result["available"]:
            for key in ("numerator", "error_bound", "interval", "interval_excludes_zero", "coefficient_sign"):
                _issue(local, result[key] is None, "unknown_has_operative_value", field=key)
            records.append(dict(case_id=case["case_id"], result=result, refits=0, issues=local))
            issues.extend(dict(case_id=case["case_id"], **i) for i in local)
            continue
        nominal, w, r, basis = direct_numerator(case["y"], case["Z"], case["m"], case["P"])
        tolerance = 1e-9*max(1., abs(nominal), abs(result["numerator"]))
        # Extreme exact-affine controls intentionally expose arithmetic
        # cancellation. Their true numerator is zero by construction; the
        # separate non-IEEE numerical decline margin must contain that zero.
        if case["kind"] == "zero_error_null":
            tolerance = max(tolerance, result["numerical_resolution_margin"])
        _issue(local, abs(nominal-result["numerator"]) <= tolerance, "independent_qr_numerator_mismatch")
        _issue(local, result["interval"] == [result["numerator"]-result["error_bound"],
                                             result["numerator"]+result["error_bound"]], "interval_algebra_mismatch")
        if case["kind"] == "zero_error_null":
            _issue(local, not result["interval_excludes_zero"] and result["coefficient_sign"] == "unresolved",
                   "floating_residual_became_zero_error_source_sign")
            zero_error_checks += 1
        rng = np.random.default_rng(NUMERATOR_SEED+index+1)
        bound_y, bound_z, bound_m = (np.broadcast_to(case[key], case[other].shape) for key, other in
            (("response_bound", "y"), ("nuisance_bound", "Z"), ("source_bound", "m")))
        worst_excess = 0.0
        for draw in range(REFITS_PER_CASE):
            if draw < 2:
                sign = -1 if draw == 0 else 1
                e = sign*bound_y*np.sign(w)
                dm = sign*bound_m*np.sign(r)
                dz = sign*bound_z
            elif draw == 2 and np.all(abs(case["m"]) <= bound_m):
                e, dm, dz = np.zeros_like(bound_y), -case["m"], np.zeros_like(bound_z)
            elif draw % 2:
                e, dm, dz = (b*rng.choice((-1., 1.), size=b.shape) for b in (bound_y, bound_m, bound_z))
            else:
                e, dm, dz = (b*rng.uniform(-1, 1, size=b.shape) for b in (bound_y, bound_m, bound_z))
            value, wp, rp, perturbed_basis = direct_numerator(case["y"]+e, case["Z"]+dz, case["m"]+dm, case["P"])
            excess = abs(value-result["numerator"])-result["error_bound"]
            worst_excess = max(worst_excess, excess)
            _issue(local, excess <= 1e-8*max(1., abs(value), result["error_bound"]),
                   "simultaneous_refit_outside_numerator_interval", draw=draw, excess=excess)
            if result["interval_excludes_zero"]:
                _issue(local, float(wp@wp) > 0 and ((value > 0) == (result["coefficient_sign"] == "positive")),
                       "certified_sign_not_preserved", draw=draw)
            if draw < 8:
                change = perturbed_basis@perturbed_basis.T-basis@basis.T
                actual_gap = float(np.linalg.norm(change, ord=2))
                _issue(local, actual_gap <= result["diagnostics"]["nuisance_projector_gap_bound"]+1e-10,
                       "projector_gap_exceeds_certificate", draw=draw, gap=actual_gap)
                gap_checks += 1
            refits += 1
        # Conservative guard can depend on raw affine amplitude; compare the
        # mathematical numerator and analytic bound, not a roundoff-policy width.
        shifted = dict(args, y=case["y"]+case["P"]@np.arange(1, case["P"].shape[1]+1)*5.)
        changed = source_presence(**shifted)
        _issue(local, changed["available"], "affine_shift_lost_availability")
        if changed["available"]:
            shift_tolerance = 1e-7*max(1., abs(nominal))
            if case["kind"] == "zero_error_null":
                shift_tolerance = max(shift_tolerance, changed["numerical_resolution_margin"]+result["numerical_resolution_margin"])
            _issue(local, abs(changed["numerator"]-result["numerator"]) <= shift_tolerance,
                   "affine_shift_changed_numerator")
        invariances += 1
        if case["Z"].shape[1]:
            order = np.arange(case["Z"].shape[1])[::-1]
            factors = np.array([-1e-3, 1e3][:len(order)])
            transformed = dict(args, Z=case["Z"][:, order]*factors,
                               nuisance_bound=bound_z[:, order]*abs(factors))
            changed = source_presence(**transformed)
            if case["case_id"] == "exact_affine_nuisance":
                # The narrow literal redundancy proof deliberately cannot
                # certify an arbitrary rescaled exact affine combination.
                _issue(local, not changed["available"], "scaled_affine_proof_limit_changed")
                conservative_controls += 1
            else:
                _issue(local, changed["available"], "nuisance_units_lost_availability")
                if changed["available"]:
                    for key in ("numerator", "analytic_error_bound", "numerical_resolution_margin", "error_bound"):
                        _issue(local, np.isclose(changed[key], result[key], atol=1e-7, rtol=1e-8),
                               "nuisance_units_changed_result", field=key)
                invariances += 1
        for factor in (-.125, 7.):
            changed = source_presence(**dict(args, m=case["m"]*factor, source_bound=bound_m*abs(factor)))
            _issue(local, changed["available"], "source_unit_change_lost_availability")
            if changed["available"]:
                source_tolerance = max(1e-7, result["numerical_resolution_margin"]*abs(factor)) if case["kind"] == "zero_error_null" else 1e-7
                _issue(local, np.isclose(changed["numerator"], result["numerator"]*factor, atol=source_tolerance, rtol=1e-8),
                       "source_unit_numerator_mismatch", factor=factor)
                _issue(local, np.isclose(changed["error_bound"], result["error_bound"]*abs(factor), atol=1e-7, rtol=1e-8),
                       "source_unit_bound_mismatch", factor=factor)
            invariances += 1
        records.append(dict(case_id=case["case_id"], result=result, independent_qr_numerator=nominal,
                            refits=REFITS_PER_CASE, maximum_interval_excess=worst_excess, issues=local))
        issues.extend(dict(case_id=case["case_id"], **i) for i in local)
    return dict(cases=len(cases), records=records, simultaneous_refits=refits, invariance_checks=invariances,
                direct_projector_gap_checks=gap_checks, zero_error_null_checks=zero_error_checks,
                conservative_transformed_controls=conservative_controls, issues=issues)


def raw_cases():
    yy, xx = np.indices((129, 129))
    background = 50+20*(xx >= 64)+8*np.sin(yy/12)
    output = []
    for name, polarity, fractional, fixed in (
        ("bright_ordinary", "bright", False, False),
        ("bright_fractional_with_fixed", "bright", True, True),
        ("dark_with_fixed", "dark", False, True),
    ):
        sign = 1 if polarity == "bright" else -1
        centers = [[29+4*i+(.3 if fractional else 0), 64+(.2 if fractional else 0)] for i in range(8)]
        point = lambda x, y: np.exp(-((xx-x)**2+(yy-y)**2)/2)
        history = np.stack([background+sign*30*point(*position) for position in centers])
        if fixed:
            history += sign*25*point(78, 78)
        output.append(dict(case_id=name, history=history, centers=centers,
                           offset=[.25, -.25] if fractional else [0., 0.], polarity=polarity,
                           fixed_template_required=fixed))
    return output


def frozen_membership_components(history, centers, offset, polarity, original):
    """Recompute ONLY the nominally used samples/anchors; never discover peaks."""
    yy, xx = np.indices((129, 129))
    masked = history.copy()
    sign = 1. if polarity == "bright" else -1.
    for index, center in enumerate(centers):
        if center is not None:
            masked[index, np.maximum(abs(xx-center[0]), abs(yy-center[1])) <= 8] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(masked, axis=0)
    background[np.isfinite(masked).sum(axis=0) < legacy.MIN_BACKGROUND_OBSERVATIONS] = np.nan
    moving = []
    for index in original["template_history"]["moving"]["prior_history_indices"]:
        stamp = legacy._positive_unit_stamp(legacy._stamp(sign*(history[index]-background), centers[index]))
        if stamp is None:
            raise ValueError("A frozen used moving stamp became unavailable")
        moving.append(stamp)
    combined = legacy._combine_stamps(moving)
    if combined is None:
        raise ValueError("Frozen used moving median became unavailable")
    moving_template = legacy._place(combined, 64+np.asarray(offset))
    fixed = []
    for record in original["template_history"]["fixed"]:
        stamps = []
        for index in record["prior_history_indices"]:
            safe = history[index].copy()
            center = centers[index]
            if center is None:
                safe[:] = np.nan
            else:
                safe[np.maximum(abs(xx-center[0]), abs(yy-center[1])) <= 8] = np.nan
            stamp = legacy._positive_unit_stamp(legacy._stamp(sign*legacy._box_highpass(safe), record["anchor_xy"]))
            if stamp is None:
                raise ValueError("A frozen used fixed stamp became unavailable")
            stamps.append(stamp)
        combined = legacy._combine_stamps(stamps)
        if combined is None:
            raise ValueError("Frozen used fixed median became unavailable")
        fixed.append(legacy._place(combined, record["anchor_xy"]))
    return dict(background=background, moving_template=moving_template,
                fixed_templates=np.stack(fixed) if fixed else np.empty((0, 129, 129)))


def audit_raw_histories(cases):
    records, issues, array_checks = [], [], 0
    for case_index, case in enumerate(cases):
        original = components43.prepare_components(case["history"], case["centers"], case["offset"], case["polarity"])
        bounds = bounds45.component_bounds(case["history"], case["centers"], case["offset"], case["polarity"], original)
        local = []
        _issue(local, bounds["available"], "raw_fixture_bounds_unavailable", reasons=bounds["reasons"])
        _issue(local, not case["fixed_template_required"] or len(original["fixed_templates"]) > 0,
               "predeclared_fixed_control_has_no_fixed_template")
        nominal = frozen_membership_components(case["history"], case["centers"], case["offset"], case["polarity"], original)
        for key in ("background", "moving_template", "fixed_templates"):
            _issue(local, np.allclose(nominal[key], original[key], atol=1e-14, rtol=1e-13, equal_nan=True),
                   "frozen_membership_nominal_reconstruction_mismatch", component=key)
        maximum_excess = 0.0
        rng = np.random.default_rng(RAW_SEED+case_index)
        for draw in range(RAW_DRAWS if bounds["available"] else 0):
            if draw < 2:
                perturbation = np.full_like(case["history"], -.5 if draw == 0 else .5)
            elif draw % 2:
                perturbation = .5*rng.choice((-1., 1.), size=case["history"].shape)
            else:
                perturbation = rng.uniform(-.5, .5, size=case["history"].shape)
            changed = frozen_membership_components(case["history"]+perturbation, case["centers"], case["offset"], case["polarity"], original)
            for key, bound_key in (("background", "background_bound129"), ("moving_template", "moving_template_bound129"),
                                   ("fixed_templates", "fixed_template_bounds")):
                support = np.isfinite(original[key])
                _issue(local, np.array_equal(support, np.isfinite(changed[key])), "raw_perturbation_support_changed", component=key, draw=draw)
                differences = abs(changed[key][support]-original[key][support])-bounds[bound_key][support]
                excess = float(np.max(differences)) if differences.size else 0.
                maximum_excess = max(maximum_excess, excess)
                _issue(local, excess <= 2e-12, "raw_perturbed_component_outside_bound", component=key, draw=draw, excess=excess)
                array_checks += 1
        records.append(dict(case_id=case["case_id"], available=bounds["available"],
            moving_history_indices=original["template_history"]["moving"]["prior_history_indices"],
            fixed_anchor_count=len(original["fixed_templates"]),
            fixed_histories=[dict(anchor_xy=r["anchor_xy"], history_indices=r["prior_history_indices"]) for r in original["template_history"]["fixed"]],
            draws=RAW_DRAWS if bounds["available"] else 0, maximum_interval_excess=maximum_excess, issues=local))
        issues.extend(dict(case_id=case["case_id"], **i) for i in local)
    return dict(cases=len(cases), native_history_perturbations=sum(r["draws"] for r in records),
                component_array_checks=array_checks, records=records, issues=issues,
                membership_and_anchors_frozen=True, peaks_rediscovered_after_perturbation=False)


def _array_hashes(case, keys):
    return {key: dict(shape=list(np.asarray(case[key]).shape), dtype=str(np.asarray(case[key]).dtype),
                     sha256=hashlib.sha256(np.asarray(case[key]).tobytes()).hexdigest()) for key in keys}


def run(output):
    output = Path(output).resolve()
    if output.parent != BASE:
        raise ValueError("Audit must be written directly in the V45 synthetic result base")
    freeze_path = output.with_name(output.stem+"_freeze.json")
    if output.exists() or freeze_path.exists():
        raise FileExistsError("Fresh audit/freeze paths required; earlier evidence is preserved")
    dependencies = [Path(__file__).resolve(), ROOT/"tests/unit/test_accuracy_v45_audit.py",
                    ROOT/"scripts/accuracy_v45_bounds.py", ROOT/"scripts/accuracy_v45_presence.py",
                    ROOT/"scripts/accuracy_v42_localized.py", ROOT/"scripts/accuracy_v43_components.py",
                    ROOT/"scripts/accuracy_v43_bounds.py"]
    bindings = {str(path): sha(path) for path in dependencies}
    numerical, native = numerator_cases(), raw_cases()
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), files_sha256=bindings,
        endpoint_seed=ENDPOINT_SEED, numerator_seed=NUMERATOR_SEED, raw_seed=RAW_SEED,
        decimal_precision=100, refits_per_available_case=REFITS_PER_CASE, raw_perturbations_per_case=RAW_DRAWS,
        endpoint_boxes=[dict(case_id=name, lower_sha256=hashlib.sha256(lo.tobytes()).hexdigest(),
                            upper_sha256=hashlib.sha256(hi.tobytes()).hexdigest()) for name, lo, hi in endpoint_boxes()],
        numerator_cases=[dict(case_id=c["case_id"], kind=c["kind"], expected_available=c.get("expected_available"),
                             arrays=_array_hashes(c, ARRAY_KEYS)) for c in numerical],
        raw_cases=[dict(case_id=c["case_id"], arrays=_array_hashes(c, ("history",)), centers=c["centers"],
                        offset=c["offset"], polarity=c["polarity"], fixed_template_required=c["fixed_template_required"]) for c in native],
        synthetic_only=True, saved_before_persisted_audit=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    endpoints = audit_endpoints()
    numerator = audit_numerator(numerical)
    raw = audit_raw_histories(native)
    issues = [dict(section=section, **item) for section, values in
              (("endpoints", endpoints), ("numerator", numerator), ("raw_histories", raw)) for item in values["issues"]]
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("Mathematical audit dependency changed during run: "+path)
    report = dict(schema="seaqr.accuracy-v45-independent-uncertainty-audit.v1", completed=True, passed=not issues,
        created_at_utc=datetime.now(timezone.utc).isoformat(), frozen_manifest=str(freeze_path),
        frozen_manifest_sha256=sha(freeze_path), inputs_sha256=bindings,
        preserved_preflight_endpoint_finding=INITIAL_ENDPOINT_FINDING,
        endpoints=endpoints, numerator=numerator, raw_histories=raw, issues=issues,
        synthetic_only=True, real_data_accessed=False, production_changed=False,
        limitations=[
            "Decimal endpoints validate mathematical marginal extrema at 100-digit precision, not all floating operations in a general interval backend",
            "Sampled simultaneous errors and raw-image perturbations are bug-finding checks, not proofs of universal coverage",
            "Raw-history reconstruction freezes nominal used stamps, anchor identities, geometry and support; discarded-stamp membership changes are excluded",
            "Positive source numerator is only a conditional fitted coefficient sign, not motion, identity or airborne classification",
            "No oracle parameter, aligned DN bound, nominal component, V44 file or production policy is modified",
        ])
    if not _finite(report):
        raise ValueError("Nonfinite audit report")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), passed=report["passed"], issues=len(issues),
                         endpoint_checks=endpoints["endpoint_comparisons"], numerator_cases=numerator["cases"],
                         refits=numerator["simultaneous_refits"], raw_draws=raw["native_history_perturbations"])))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output)["passed"] else 1)
