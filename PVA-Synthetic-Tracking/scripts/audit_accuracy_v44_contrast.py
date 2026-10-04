"""Independent synthetic-only audit of a conditional source contrast interval.

No media, saved real-image patches, journals, or production configuration are
read. The 48-case grid and seven structural controls below are fixed independently
of outcome. Direct refits are bug-finding Monte Carlo checks, NOT a proof.

Selected-functional proof used independently here: a'=a+QE; h=a z with
z=(a.T a)^-1 e_last; h'=a'(a'.T a')^-1 e_last. For d=h'-h,
  a'.T d = -E.T h,
  (I-Proj_a') d = (I-Proj_a') QE z.
The orthogonal components imply
  ||d||^2 <= (||U.T |h|||/q)^2 + ||U |z|||^2,
where q=smin(a)-||U||2 >0, |E|<=U, and Q is an exact orthogonal affine
complement. The exact scalar change is h(e-E theta)+d(r+Qe-QEtheta).
Triangle bounds give the interval; correlated errors require no independence.

The audit uses a QR affine basis and QR coefficient/dual solves rather than the
core's SVD fit. Perturbed coefficients are separately refitted in the original
unprojected [P,Z,m] system. Geometry, selected rows/columns, exact affine span,
template membership and the declared deterministic error set remain fixed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from accuracy_v44_contrast import source_contrast


SEED = 440442
REFITS_PER_CASE = 200


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plane(n):
    height = n//8
    y, x = np.indices((height, 8))
    return np.column_stack((np.ones(n), ((x-3.5)/3.5).ravel(),
                            ((y-(height-1)/2)/max((height-1)/2, 1)).ravel()))


def cases():
    """Generate the full fixed grid, without consulting any evaluated result."""
    rng = np.random.default_rng(SEED)
    output = []
    for n in (32, 64):
        for nuisance_count in (0, 2):
            for source_coefficient in (-12., 0., 12.):
                for residual_scale in (0., 25.):
                    for design_error in (0., .005):
                        p = _plane(n)
                        raw = rng.normal(size=(n, nuisance_count+1))
                        coefficients = np.r_[rng.normal(size=nuisance_count)*3, source_coefficient]
                        basis = np.linalg.qr(np.column_stack((p, raw)), mode="reduced")[0]
                        residual = rng.normal(size=n)
                        residual -= basis@(basis.T@residual)
                        y = p@np.asarray([100., 25., -40.])+raw@coefficients+residual_scale*residual
                        spec = dict(kind="grid", rows=n, nuisance_count=nuisance_count,
                                    source_coefficient=source_coefficient, residual_scale=residual_scale,
                                    design_entry_bound=design_error)
                        output.append(dict(case_id=f"grid_{len(output):02d}", specification=spec,
                                           y=y, Z=raw[:, :-1], m=raw[:, -1], P=p,
                                           response_bound=np.linspace(.1, .5, n),
                                           nuisance_bound=np.full((n, nuisance_count), design_error),
                                           source_bound=np.full(n, design_error)))
    n = 32
    p = _plane(n)
    z, other = rng.normal(size=(n, 2)).T
    structural = [
        ("source_equals_nuisance", z[:, None], z, p, 0., "projected_design_machine_rank_deficient"),
        ("source_nearly_nuisance", z[:, None], z+1e-8*other, p, .005, "projected_design_robust_rank_not_certified"),
        ("source_in_affine", z[:, None], p[:, 0], p, 0., "source_in_numerical_affine_span"),
        ("uncertain_affine_nuisance", p[:, :1], other, p, .005, "near_affine_nuisance_not_certified_redundant"),
        ("exact_redundant_affine_nuisance", p[:, :1], other, p, 0., None),
        ("rank_deficient_affine_basis", z[:, None], other, np.column_stack((p, p[:, 1])), 0., "rank_deficient_exact_affine_basis"),
    ]
    for name, nuisance, source, exact, error, expected in structural:
        output.append(dict(case_id=name, specification=dict(kind="structural", expected_unknown_reason=expected),
                           y=p@np.asarray([100., 25., -40.])+2*z+12*source,
                           Z=nuisance, m=source, P=exact, response_bound=np.full(n, .5),
                           nuisance_bound=np.full(nuisance.shape, error), source_bound=np.full(n, error)))
    # Independently discovered preflight regression: a tiny source-bearing
    # component is not removable merely because the nuisance is near affine.
    # An unrestricted nuisance coefficient can amplify that tiny component.
    source = np.asarray([.5, -.5, .5, -.5])
    exact = np.ones((4, 1))
    output.append(dict(case_id="near_affine_nuisance_carries_source", specification=dict(
        kind="structural", expected_unknown_reason="near_affine_nuisance_not_certified_redundant"),
        y=100+30*source, Z=exact+np.ldexp(source, -50)[:, None], m=source, P=exact,
        response_bound=np.zeros(4), nuisance_bound=np.zeros((4, 1)), source_bound=np.zeros(4)))
    return output


def independent_nominal(case):
    y, z, m, p = (case[key] for key in ("y", "Z", "m", "P"))
    if p.shape[1]:
        if np.linalg.matrix_rank(p) != p.shape[1]:
            return dict(available=False, reason="rank_deficient_exact_affine_basis")
        basis = np.linalg.qr(p, mode="reduced")[0]
    else:
        basis = np.empty((len(y), 0))
    def project(value): return value-basis@(basis.T@value)
    raw = np.column_stack((z, m))
    errors = np.column_stack((np.broadcast_to(case["nuisance_bound"], z.shape),
                              np.broadcast_to(case["source_bound"], m.shape)))
    projected = project(raw)
    norms = np.linalg.norm(projected, axis=0)
    tolerance = np.finfo(float).eps*max(len(y), p.shape[1], 1)*np.linalg.norm(raw, axis=0)
    # A numerical tolerance cannot establish exact redundancy of an unrestricted
    # coefficient. Only literal zero or +/- an identical supplied affine column
    # is removed; every other near-affine column remains explicit unknown.
    removed = [j for j in range(z.shape[1]) if np.all(errors[:, j] == 0) and (
        np.all(z[:, j] == 0) or any(np.array_equal(z[:, j], sign*p[:, k])
                                  for k in range(p.shape[1]) for sign in (-1, 1)))]
    keep = [j for j in range(raw.shape[1]) if j not in removed]
    if norms[-1] <= tolerance[-1]: return dict(available=False, reason="source_in_numerical_affine_span")
    if any(norms[j] <= tolerance[j] for j in keep[:-1]):
        return dict(available=False, reason="near_affine_nuisance_not_certified_redundant")
    scales = norms[keep]
    a, u, target = projected[:, keep]/scales, errors[:, keep]/scales, project(y)
    singular = np.linalg.svd(a, compute_uv=False)
    rank_tolerance = np.finfo(float).eps*max(a.shape)*singular[0]
    if len(singular) < a.shape[1] or int((singular > rank_tolerance).sum()) != a.shape[1]:
        return dict(available=False, reason="projected_design_machine_rank_deficient")
    eta = float(np.linalg.svd(u, compute_uv=False)[0])
    q = float(singular[-1]-eta)
    if q <= 0: return dict(available=False, reason="projected_design_robust_rank_not_certified")
    orthogonal, triangular = np.linalg.qr(a, mode="reduced")
    inverse = np.linalg.solve(triangular, orthogonal.T)
    theta = inverse@target
    residual = target-a@theta
    h = inverse[-1]
    e_last = np.eye(a.shape[1])[-1]
    dual_z = np.linalg.solve(triangular, np.linalg.solve(triangular.T, e_last))
    row_change = math.hypot(np.linalg.norm(u.T@abs(h))/q, np.linalg.norm(u@abs(dual_z)))
    global_change = eta/q**2+eta/(q*float(singular[-1]))
    used = min(row_change, global_change)
    response_error = np.broadcast_to(case["response_bound"], y.shape)
    design_response = u@abs(theta)
    weighted = float(abs(h)@(response_error+design_response))
    changed = used*(np.linalg.norm(residual)+np.linalg.norm(response_error)+np.linalg.norm(design_response))
    estimate = float(theta[-1]/scales[-1])
    bound = float((weighted+changed)/scales[-1])
    diagnostics = dict(exact_nuisance_columns_removed_in_affine_span=removed,
                       retained_columns=keep, projected_column_scales=scales.tolist(),
                       scaled_projected_singular_values=singular.tolist(),
                       raw_design_bound_spectral_norm=eta, robust_minimum_singular_value=q,
                       nominal_scaled_coefficients=theta.tolist(), residual_l2_norm=float(np.linalg.norm(residual)),
                       source_column_scale=float(scales[-1]), nominal_dual_l1_norm=float(abs(h).sum()),
                       nominal_dual_l2_norm=float(np.linalg.norm(h)),
                       dual_change_global_l2_bound=global_change,
                       dual_change_selected_row_l2_bound=row_change, dual_change_used_l2_bound=used,
                       nominal_weighted_error_in_coefficient_units=weighted/scales[-1],
                       changed_dual_error_in_coefficient_units=changed/scales[-1])
    return dict(available=True, estimate=estimate, error_bound=bound, interval=[estimate-bound, estimate+bound],
                diagnostics=diagnostics, internal=dict(basis=basis, h=h, raw=raw, raw_errors=errors,
                                                      scales=scales, keep=keep, theta=theta, dual_change=used))


def call_core(case):
    return source_contrast(case["y"], case["Z"], case["m"], case["P"],
                           response_bound=case["response_bound"], nuisance_bound=case["nuisance_bound"],
                           source_bound=case["source_bound"])


def _compare(expected, observed, path, issues):
    if isinstance(expected, dict):
        for name, value in expected.items(): _compare(value, observed.get(name), path+"."+name, issues)
    elif isinstance(expected, (list, tuple)):
        if observed is None or len(expected) != len(observed):
            issues.append(dict(path=path, reason="length_mismatch"))
        else:
            for i, (first, second) in enumerate(zip(expected, observed)):
                _compare(first, second, path+f"[{i}]", issues)
    elif isinstance(expected, (float, np.floating)):
        if observed is None or not math.isclose(float(expected), float(observed), rel_tol=2e-8, abs_tol=2e-8):
            issues.append(dict(path=path, reason="numeric_mismatch", expected=float(expected), observed=observed))
    elif expected != observed:
        issues.append(dict(path=path, reason="exact_mismatch", expected=expected, observed=observed))


def refit_audit(case, nominal, actual, count=REFITS_PER_CASE, seed=SEED):
    issues = []
    if not nominal["available"]:
        _compare(False, actual["available"], "available", issues)
        _compare([nominal["reason"]], actual["reasons"], "reasons", issues)
        for key in ("estimate", "error_bound", "interval"):
            _compare(None, actual[key], key, issues)
        return dict(issues=issues, refits=0, maximum_contrast_error_fraction=None, maximum_dual_error_fraction=None)
    _compare({key:nominal[key] for key in ("available", "estimate", "error_bound", "interval", "diagnostics")},
             actual, "nominal", issues)
    if not actual["available"]:
        return dict(issues=issues, refits=0, maximum_contrast_error_fraction=None, maximum_dual_error_fraction=None)
    rng = np.random.default_rng(seed)
    data = nominal["internal"]
    raw, errors, basis = data["raw"], data["raw_errors"], data["basis"]
    ey = np.broadcast_to(case["response_bound"], case["y"].shape)
    max_contrast, max_dual = 0., 0.
    completed = 0
    for iteration in range(count):
        if iteration % 4 == 0:
            design_error = rng.choice([-1., 1.], raw.shape)*errors
            response_error = rng.choice([-1., 1.], len(ey))*ey
        elif iteration % 4 == 1:
            row_signs = rng.choice([-1., 1.], len(ey))
            design_error = row_signs[:, None]*errors
            response_error = row_signs*ey
        elif iteration % 4 == 2:
            common_sign = rng.choice([-1., 1.])
            design_error = common_sign*errors
            response_error = common_sign*ey
        else:
            # Response-only extremal signs attain the nominal response dual
            # bound for exact designs. No fit-derived case is omitted.
            design_error = np.zeros_like(errors)
            response_error = np.sign(data["h"])*ey
        changed_raw = raw+design_error
        changed_y = case["y"]+response_error
        full = np.column_stack((case["P"], changed_raw))
        estimate = float(np.linalg.lstsq(full, changed_y, rcond=None)[0][-1])
        coefficient_error = abs(estimate-actual["estimate"])
        numerical_slack = 2e-8*(1+abs(actual["estimate"]))
        if coefficient_error > actual["error_bound"]+numerical_slack:
            issues.append(dict(path=f"refit[{iteration}]", reason="coefficient_interval_violation",
                               estimate=estimate, observed_error=coefficient_error, bound=actual["error_bound"]))
        # Compare scaled duals with the SAME nominal scales, not re-normalized
        # perturbed columns. That is the declared simultaneous-refit identity.
        ap = changed_raw[:, data["keep"]]/data["scales"]
        ap -= basis@(basis.T@ap)
        hp = np.linalg.pinv(ap)[-1]
        dual_error = float(np.linalg.norm(hp-data["h"]))
        if dual_error > data["dual_change"]+2e-10:
            issues.append(dict(path=f"refit[{iteration}]", reason="selected_dual_bound_violation",
                               observed_error=dual_error, bound=data["dual_change"]))
        if actual["error_bound"] > 0: max_contrast = max(max_contrast, coefficient_error/actual["error_bound"])
        if data["dual_change"] > 0: max_dual = max(max_dual, dual_error/data["dual_change"])
        completed += 1
    return dict(issues=issues, refits=completed, maximum_contrast_error_fraction=max_contrast,
                maximum_dual_error_fraction=max_dual)


def invariance_audit(case, baseline):
    if not baseline["available"]: return dict(checks=0, issues=[], conservative_transformed_controls=[])
    issues = []
    checks = 0
    conservative = []
    if case["P"].shape[1]:
        changed = dict(case)
        changed["y"] = case["y"]+case["P"]@np.linspace(300., -250., case["P"].shape[1])
        result = call_core(changed)
        _compare({key:baseline[key] for key in ("available", "estimate", "error_bound", "interval")}, result, "affine_response_change", issues)
        checks += 1
    if case["Z"].shape[1]:
        order = np.arange(case["Z"].shape[1])[::-1]
        units = np.asarray([-1e-3, 1e3][:len(order)])
        changed = dict(case)
        changed["Z"] = case["Z"][:, order]*units
        changed["nuisance_bound"] = np.broadcast_to(case["nuisance_bound"], case["Z"].shape)[:, order]*abs(units)
        result = call_core(changed)
        if case["case_id"] == "exact_redundant_affine_nuisance":
            # This rescaling leaves the affine span unchanged mathematically,
            # but exceeds the implementation's deliberately narrow exact
            # redundancy proof. Record conservative loss of availability,
            # rather than claiming successful interval invariance.
            _compare(dict(available=False, reasons=["near_affine_nuisance_not_certified_redundant"],
                          estimate=None, error_bound=None, interval=None), result,
                     "scaled_exact_affine_nuisance_conservative_unknown", issues)
            conservative.append(dict(transformation="nuisance_units_permutation", outcome=result,
                                     interpretation="Mathematically same affine span; exact-equality proof no longer applies; conservatively unknown, not invariant availability."))
        else:
            _compare({key:baseline[key] for key in ("available", "estimate", "error_bound", "interval")}, result, "nuisance_units_permutation", issues)
            checks += 1
    for factor in (-.125, 7.):
        changed = dict(case)
        changed["m"] = case["m"]*factor
        changed["source_bound"] = case["source_bound"]*abs(factor)
        expected = dict(available=True, estimate=baseline["estimate"]/factor,
                        error_bound=baseline["error_bound"]/abs(factor),
                        interval=sorted(value/factor for value in baseline["interval"]))
        _compare(expected, call_core(changed), "source_units:"+str(factor), issues)
        checks += 1
    return dict(checks=checks, issues=issues, conservative_transformed_controls=conservative)


def run(output):
    output = Path(output).resolve()
    freeze = output.with_name(output.stem+"_freeze.json")
    if output.exists() or freeze.exists(): raise FileExistsError("New audit and freeze output paths required")
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root/"scripts/accuracy_v44_contrast.py",
             root/"tests/unit/test_accuracy_v44_contrast_audit.py", root/"docs/accuracy_v44_plan.md"]
    hashes = {str(path):digest(path) for path in files}
    generated = cases()
    manifest = []
    for case in generated:
        arrays = {key:hashlib.sha256(np.asarray(case[key]).tobytes()).hexdigest()
                  for key in ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound")}
        manifest.append(dict(case_id=case["case_id"], specification=case["specification"], input_arrays_sha256=arrays))
    output.parent.mkdir(parents=True, exist_ok=True)
    with freeze.open("x") as stream:
        json.dump(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
                       refits_per_available_case=REFITS_PER_CASE, files_sha256=hashes,
                       cases=manifest, synthetic_only=True, before_persisted_audit_evaluation=True),
                  stream, indent=2, allow_nan=False)
    records, issues = [], []
    total_refits = invariances = available = conservative_controls = 0
    for index, case in enumerate(generated):
        nominal, actual = independent_nominal(case), call_core(case)
        refits = refit_audit(case, nominal, actual, seed=SEED+index)
        invariance = invariance_audit(case, actual)
        available += int(actual["available"])
        total_refits += refits["refits"]
        invariances += invariance["checks"]
        conservative_controls += len(invariance["conservative_transformed_controls"])
        if case["specification"]["kind"] == "structural":
            expected_reason = case["specification"]["expected_unknown_reason"]
            _compare([] if expected_reason is None else [expected_reason], actual["reasons"], "predeclared_structural_reason", refits["issues"])
        combined_issues = refits["issues"]+invariance["issues"]
        issues.extend(dict(case_id=case["case_id"], **issue) for issue in combined_issues)
        records.append(dict(case_id=case["case_id"], specification=case["specification"], core=actual,
                            independent={key:value for key,value in nominal.items() if key != "internal"},
                            perturbation_checks=refits, invariance_checks=invariance))
    for path, value in hashes.items():
        if digest(path) != value: raise ValueError("Bound audit input changed: "+path)
    result = dict(schema="seaqr.accuracy-v44-independent-contrast-audit.v1", completed=True, passed=not issues,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), frozen_manifest=str(freeze),
                  frozen_manifest_sha256=digest(freeze), inputs_sha256=hashes,
                  counts=dict(cases=len(generated), grid_cases=48, structural_cases=7,
                              available_cases=available, unknown_cases=len(generated)-available,
                              simultaneous_refits=total_refits, invariance_checks=invariances,
                              conservative_transformed_controls=conservative_controls),
                  issues=issues, records=records, synthetic_only=True, real_data_accessed=False,
                  production_changed=False,
                  revision_note="Supersedes contrast_independent_audit_01.json, which bound the pre-fix core and lacked the source-bearing near-affine nuisance regression. Earlier artifacts are preserved.",
                  interpretation="Monte Carlo/corner refits find bugs but do not prove coverage; the explicit dual decomposition supplies the mathematical justification, conditional on fixed inputs/error sets.")
    with output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], counts=result["counts"], issues=len(issues))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.output)
    raise SystemExit(0 if result["passed"] else 1)
