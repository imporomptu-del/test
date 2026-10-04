"""Bounded, independent V43 numerical replay audit without raw-media access.

The frozen fit/evaluator are NOT invoked. Fits use independently assembled
arrays, QR inverses, explicit perturbation formulas, branch interval unions.
Legacy components use the independent V42 scalar/direct-mean implementation.
Protected components are produced by their frozen helper and independently
checked for finite foreground-free dependencies and scalar stamp reconstruction.
If core fits are reached, component error arrays come from the frozen bounds
helper: that helper's derivation was separately preflight reviewed, but this
audit does not claim an independent second implementation of those arrays.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import audit_accuracy_v42_numerical as old_audit
from accuracy_v43_bounds import component_bounds
from accuracy_v43_components import prepare_components as protected_components


CLIPS = ("0029", "0126", "0055", "0082")
FIXED_KEYS = (("0029", 298, 0, "bright:980"), ("0126", 140, 0, "bright:1001"),
              ("0029", 346, 0, "bright:2641"), ("0029", 347, 0, "bright:2641"))


def identity(record):
    return record["clip"], record["frame_index"], record["segment"], record["track_id"]


def choose_keys(cache):
    """Frozen named diagnostics + first geometry-available identity per clip."""
    result = set(FIXED_KEYS)
    for clip in CLIPS:
        available = sorted(identity(record) for record in cache["states"]
                           if record["clip"] == clip and record["archive"] is not None)
        if not available:
            raise ValueError("No available geometry for predeclared clip")
        result.add(available[0])
    keys = {identity(record) for record in cache["states"] if record["archive"] is not None}
    if not result <= keys:
        raise ValueError("Predeclared audit state has no cached geometry")
    return sorted(result)


def _qr_inverse(design):
    q, r = np.linalg.qr(design, mode="reduced")
    return np.linalg.solve(r, q.T)


def independent_branch(train, target, test, ua, ub, sigma):
    n, p = train.shape
    m = len(test)
    train_norm = np.sqrt(np.asarray([math.fsum(float(v)*float(v) for v in column) for column in train.T]))
    test_norm = np.sqrt(np.asarray([math.fsum(float(v)*float(v) for v in column) for column in test.T]))
    scales = np.sqrt(train_norm*train_norm+test_norm*test_norm)
    diagnostics = dict(training_rows=n, test_rows=m, columns=p,
                       train_column_l2_norm=train_norm.tolist(), test_column_l2_norm=test_norm.tolist(),
                       common_column_l2_norm=scales.tolist(), train_column_energy=(train_norm**2).tolist(),
                       test_column_energy=(test_norm**2).tolist(), common_column_energy=(scales**2).tolist(),
                       prediction_budget_dn=2*sigma)

    def failed(reason):
        return dict(available=False, reasons=[reason], diagnostics=diagnostics,
                    coefficients=None, prediction=None, prediction_bound=None)

    if p == 0:
        diagnostics.update(machine_rank=0, machine_rank_tolerance=0.,
                           scaled_train_column_energy=[], scaled_test_column_energy=[],
                           train_singular_values=[], test_singular_values=[], common_singular_values=[],
                           train_design_perturbation_spectral_bound=0.,
                           coefficient_perturbation_l2_bound_scaled=0.,
                           maximum_prediction_bound_dn=0., prediction_total_bound_dn=[0.]*m,
                           prediction_response_bound_dn=[0.]*m, prediction_design_bound_dn=[0.]*m,
                           residual_l2_norm_dn=float(np.linalg.norm(target)))
        return dict(available=True, reasons=[], diagnostics=diagnostics, coefficients=np.empty(0),
                    scaled_coefficients=np.empty(0), prediction=np.zeros(m), prediction_bound=np.zeros(m))
    if np.any(scales == 0) or np.any(train_norm == 0):
        return failed("unsupported_common_or_training_column")
    a, b, u, v = train/scales, test/scales, ua/scales, ub/scales
    singular = np.linalg.svd(a, compute_uv=False)
    tolerance = np.finfo(float).eps*max(a.shape)*singular[0]
    rank = int((singular > tolerance).sum())
    diagnostics.update(scaled_train_column_energy=np.sum(a*a, axis=0).tolist(),
                       scaled_test_column_energy=np.sum(b*b, axis=0).tolist(),
                       train_singular_values=singular.tolist(),
                       test_singular_values=np.linalg.svd(b, compute_uv=False).tolist(),
                       common_singular_values=np.linalg.svd(np.vstack((a, b)), compute_uv=False).tolist(),
                       machine_rank=rank, machine_rank_tolerance=float(tolerance))
    if rank != p:
        return failed("machine_rank_deficient")
    eta = float(np.linalg.svd(u, compute_uv=False)[0])
    minimum = float(singular[-1])
    q = minimum-eta
    diagnostics.update(train_design_perturbation_spectral_bound=eta,
                       robust_minimum_singular_value=q,
                       scaled_train_condition_number=float(singular[0]/singular[-1]))
    if q <= 0:
        return failed("design_perturbation_can_destroy_rank")
    inverse = _qr_inverse(a)
    beta = inverse @ target
    residual_norm, beta_norm = float(np.linalg.norm(target-a@beta)), float(np.linalg.norm(beta))
    leverage = b @ inverse
    l1 = np.asarray([math.fsum(abs(float(v)) for v in row) for row in leverage])
    bn, vn = np.linalg.norm(b, axis=1), np.linalg.norm(v, axis=1)
    response_l2 = sigma*math.sqrt(n)
    inverse_change = eta/q**2+eta/(q*minimum)
    design_beta_error = eta*beta_norm/q+eta*residual_norm/q**2
    response_error = sigma*l1+bn*inverse_change*response_l2+vn*response_l2/q
    design_error = (bn+vn)*design_beta_error+vn*beta_norm
    total = response_error+design_error
    diagnostics.update(response_prediction_row_l1_leverage=l1.tolist(),
                       response_prediction_row_l2_leverage=np.linalg.norm(leverage, axis=1).tolist(),
                       response_prediction_operator_l2_norm=float(np.linalg.svd(leverage, compute_uv=False)[0]),
                       residual_l2_norm_dn=residual_norm,
                       prediction_response_bound_dn=response_error.tolist(),
                       prediction_design_bound_dn=design_error.tolist(),
                       prediction_total_bound_dn=total.tolist(),
                       maximum_prediction_bound_dn=float(total.max()),
                       coefficient_perturbation_l2_bound_scaled=response_l2/q+design_beta_error)
    if np.any(total > 2*sigma):
        return failed("prediction_perturbation_exceeds_engineering_budget")
    return dict(available=True, reasons=[], diagnostics=diagnostics,
                coefficients=beta/scales, scaled_coefficients=beta,
                prediction=b@beta, prediction_bound=total)


def independent_fit(train, target, test, sigma, ua, ub, constrained=False):
    train, test, target = np.asarray(train), np.asarray(test), np.asarray(target)
    ua, ub = np.broadcast_to(ua, train.shape), np.broadcast_to(ub, test.shape)
    free = independent_branch(train, target, test, ua, ub, sigma)
    diagnostics = dict(response_perturbation_bound_dn=sigma, prediction_budget_dn=2*sigma,
                       nonnegative_last=constrained, free_fit=free["diagnostics"], boundary_fit=None)

    def failed(reasons):
        return dict(available=False, reasons=reasons, diagnostics=diagnostics,
                    coefficients=None, prediction=None, prediction_bound=None)

    if not free["available"]:
        return failed(["free_fit:"+reason for reason in free["reasons"]])
    if not constrained:
        diagnostics.update(prediction_total_bound_dn=free["prediction_bound"].tolist(),
                           maximum_prediction_bound_dn=float(free["prediction_bound"].max()))
        return dict(available=True, reasons=[], diagnostics=diagnostics,
                    coefficients=free["coefficients"], prediction=free["prediction"], prediction_bound=free["prediction_bound"])
    boundary = independent_branch(train[:, :-1], target, test[:, :-1], ua[:, :-1], ub[:, :-1], sigma)
    diagnostics["boundary_fit"] = boundary["diagnostics"]
    if not boundary["available"]:
        return failed(["boundary_fit:"+reason for reason in boundary["reasons"]])
    active = bool(free["scaled_coefficients"][-1] < 0)
    may_change = abs(float(free["scaled_coefficients"][-1])) <= free["diagnostics"]["coefficient_perturbation_l2_bound_scaled"]
    chosen, other = (boundary, free) if active else (free, boundary)
    error = chosen["prediction_bound"]
    if may_change:
        error = np.maximum(error, np.abs(other["prediction"]-chosen["prediction"])+other["prediction_bound"])
    diagnostics.update(constraint_active_at_observed_inputs=active,
                       constraint_boundary_may_change_under_perturbation=bool(may_change),
                       prediction_total_bound_dn=error.tolist(), maximum_prediction_bound_dn=float(error.max()))
    if np.any(error > 2*sigma):
        return failed(["constraint_branch_union_exceeds_engineering_budget"])
    return dict(available=True, reasons=[], diagnostics=diagnostics,
                coefficients=np.r_[boundary["coefficients"], 0.] if active else free["coefficients"],
                prediction=chosen["prediction"], prediction_bound=error)


def compact(result):
    return {key: value for key, value in result.items()
            if key not in ("coefficients", "prediction", "prediction_bound", "scaled_coefficients")}


def _safe_dependency(image, center, ax, ay):
    if center is None or min(ax, ay) < 12 or max(ax, ay) > 116:
        return False
    values = image[ay-12:ay+13, ax-12:ax+13]
    if not np.isfinite(values).all():
        return False
    return not any(max(abs(x-center[0]), abs(y-center[1])) <= 8
                   for y in range(ay-12, ay+13) for x in range(ax-12, ax+13))


def protected_dependency_audit(history, centers, offset, polarity, protected, independent_legacy):
    issues, template_count, stamp_count = [], 0, 0
    for key in ("background", "background_observation_counts", "moving_template", "moving_stamp"):
        first, second = protected[key], independent_legacy[key]
        if first is None or second is None:
            if first is not second:
                issues.append(dict(path="protected."+key, reason="none_disagreement"))
        elif not np.allclose(first, second, rtol=2e-8, atol=2e-6, equal_nan=True):
            issues.append(dict(path="protected."+key, reason="independent_array_mismatch"))
    raw_highpasses = [(1 if polarity == "bright" else -1)*old_audit.direct_highpass(image) for image in history]
    for index, record in enumerate(protected["template_history"]["fixed"]):
        template_count += 1
        ax, ay = record["anchor_xy"]
        stamps = []
        for position, frame in enumerate(record["prior_history_indices"]):
            stamp_count += 1
            if not _safe_dependency(history[frame], centers[frame], ax, ay):
                issues.append(dict(path=f"protected.fixed[{index}].frame[{frame}]", reason="unsafe_native_dependency"))
            raw = old_audit.scalar_stamp(raw_highpasses[frame], (ax, ay))
            stamp = old_audit.normalize_stamp(raw)
            if stamp is None:
                issues.append(dict(path=f"protected.fixed[{index}].frame[{frame}]", reason="independent_stamp_missing"))
                continue
            stamps.append(stamp)
            if not np.allclose(stamp, record["normalized_stamps"][position], rtol=2e-8, atol=2e-6, equal_nan=True):
                issues.append(dict(path=f"protected.fixed[{index}].frame[{frame}]", reason="scalar_stamp_mismatch"))
            weighted = np.maximum(raw, 0)*old_audit.APERTURE
            weighted[old_audit.APERTURE == 0] = 0
            energy = math.fsum(float(v)*float(v) for v in weighted.flat if math.isfinite(v))
            old_audit.compare(float(energy), float(record["raw_weighted_energy_dn2"][position]),
                              f"protected.fixed[{index}].energy[{frame}]", issues)
        merged = old_audit.combine(stamps)
        if merged is None:
            issues.append(dict(path=f"protected.fixed[{index}]", reason="independent_combination_missing"))
        elif not np.allclose(old_audit.place(merged, (ax, ay)), protected["fixed_templates"][index],
                             rtol=2e-8, atol=2e-6, equal_nan=True):
            issues.append(dict(path=f"protected.fixed[{index}]", reason="scalar_placed_template_mismatch"))
    coverage = np.zeros((41, 41), dtype=np.uint8)
    for yy, ay in enumerate(range(44, 85)):
        for xx, ax in enumerate(range(44, 85)):
            coverage[yy, xx] = sum(_safe_dependency(image, center, ax, ay)
                                   for image, center in zip(history, centers))
    expected = dict(minimum=int(coverage.min()), maximum=int(coverage.max()),
                    sufficient_seed_count=int((coverage >= 3).sum()),
                    insufficient_seed_count=int((coverage < 3).sum()),
                    counts_row_major_uint8_sha256=hashlib.sha256(coverage.tobytes()).hexdigest())
    old_audit.compare(expected, protected["metadata"]["fixed_seed_opportunity_coverage"], "protected.coverage", issues)
    return dict(issues=issues, fixed_templates_checked=template_count,
                fixed_normalized_stamps_checked=stamp_count, opportunity_seeds_checked=41*41)


def audit_arm(current, history, centers, offset, polarity, components, stored, protected):
    issues, fits = [], []
    metadata = components["metadata"]
    # Fixed-component schema has extra provenance fields; subset compare.
    old_audit.compare(metadata, stored["components"], "components", issues)
    if metadata["causal_component_reasons"]:
        old_audit.compare(dict(available=False, reasons=metadata["causal_component_reasons"]), stored, "arm", issues)
        return dict(issues=issues, fits=fits, stopped_at="component_unavailable")
    background, moving, fixed = components["background"], components["moving_template"][52:77, 52:77], components["fixed_templates"][:, 52:77, 52:77]
    common = np.isfinite(current[52:77, 52:77]) & np.isfinite(background[52:77, 52:77]) & np.isfinite(moving)
    for column in fixed: common &= np.isfinite(column)
    yy, xx = np.indices((25, 25)); parity = (xx+yy)%2
    counts = [int((common & (parity == i)).sum()) for i in (0, 1)]
    old_audit.compare(dict(common_support_count=int(common.sum()), fold_support_counts=counts,
                           common_support_sha256=hashlib.sha256(common.astype(np.uint8).tobytes()).hexdigest()),
                      stored, "common", issues)
    if int(common.sum()) < 64 or min(counts) < 32:
        old_audit.compare(dict(available=False, reasons=["insufficient_common_core_or_checkerboard_support"]), stored, "arm", issues)
        return dict(issues=issues, fits=fits, stopped_at="core_support")
    annulus = [(x, y) for y in range(34, 95) for x in range(34, 95)
               if max(abs(x-64), abs(y-64)) >= 16 and np.isfinite(current[y, x]) and np.isfinite(background[y, x])]
    core_coords = [(x+52, y+52) for y in range(25) for x in range(25) if common[y, x]]
    if len(annulus) < 64:
        old_audit.compare(dict(available=False, reasons=["insufficient_annulus_support"]), stored, "arm", issues)
        return dict(issues=issues, fits=fits, stopped_at="annulus_support")
    def design(coordinates):
        return np.asarray([[1., (x-64)/32, (y-64)/32, background[y, x]] for x, y in coordinates])
    a, b = design(annulus), design(core_coords)
    ua, ub = np.zeros_like(a), np.zeros_like(b); ua[:, -1] = .5; ub[:, -1] = .5
    annulus_fit = independent_fit(a, np.asarray([current[y, x] for x, y in annulus]), b, .5, ua, ub, True)
    old_audit.compare(compact(annulus_fit), stored["stability"]["annulus"], "annulus_fit", issues)
    fits.append(dict(stage="annulus", **compact(annulus_fit)))
    if not annulus_fit["available"]:
        old_audit.compare(dict(available=False, reasons=["annulus:"+r for r in annulus_fit["reasons"]]), stored, "arm", issues)
        return dict(issues=issues, fits=fits, stopped_at="annulus_fit")
    response = np.full((25, 25), np.nan)
    response[common] = (current[52:77, 52:77][common]-annulus_fit["prediction"])*(1 if polarity == "bright" else -1)
    sigma = .5+float(annulus_fit["prediction_bound"].max())
    old_audit.compare(sigma, stored["stability"]["core_response_bound_dn"], "core_response_bound_dn", issues)
    # Scope limitation: these propagated component-array bounds are supplied by
    # the frozen bounds helper; independent fit calculations below do not call
    # the stable-fit helper or localized evaluator.
    bounds = component_bounds(history, centers, offset, polarity, components, protected=protected)
    old_audit.compare(dict(available=bounds["available"], reasons=bounds["reasons"]),
                      stored["stability"]["component_bounds"], "component_bounds", issues)
    if not bounds["available"]:
        old_audit.compare(dict(available=False, reasons=["component_bounds:"+r for r in bounds["reasons"]]), stored, "arm", issues)
        return dict(issues=issues, fits=fits, stopped_at="component_bounds")
    moving_error, fixed_error = bounds["moving_template_bound129"][52:77, 52:77], bounds["fixed_template_bounds"][:, 52:77, 52:77]
    expected_reasons, sse = [], np.zeros(2)
    for fold in (0, 1):
        train, test = common & (parity == fold), common & (parity != fold)
        stationary = independent_fit(fixed[:, train].T, response[train], fixed[:, test].T, sigma,
                                     fixed_error[:, train].T, fixed_error[:, test].T)
        augmented = independent_fit(np.column_stack((fixed[:, train].T, moving[train])), response[train],
                                    np.column_stack((fixed[:, test].T, moving[test])), sigma,
                                    np.column_stack((fixed_error[:, train].T, moving_error[train])),
                                    np.column_stack((fixed_error[:, test].T, moving_error[test])), True)
        block = dict(training_parity=fold, stationary=compact(stationary), augmented=compact(augmented))
        old_audit.compare(block, stored["stability"]["core_fits"][fold], f"core_fits[{fold}]", issues)
        fits.append(dict(stage=f"core:{fold}:stationary", **compact(stationary)))
        fits.append(dict(stage=f"core:{fold}:augmented", **compact(augmented)))
        for label, fit in (("stationary", stationary), ("augmented", augmented)):
            expected_reasons.extend(f"core:{fold}:{label}:"+r for r in fit["reasons"])
        if stationary["available"] and augmented["available"]:
            for j, fit in enumerate((stationary, augmented)):
                errors = response[test]-fit["prediction"]
                sse[j] += math.fsum(float(e)*float(e) for e in errors)
    expected = dict(available=not expected_reasons, reasons=expected_reasons)
    if not expected_reasons:
        expected.update(mse_stationary=float(sse[0]/common.sum()), mse_augmented=float(sse[1]/common.sum()),
                        advantage_stationary_minus_augmented=float((sse[0]-sse[1])/common.sum()))
    else:
        expected.update(mse_stationary=None, mse_augmented=None, advantage_stationary_minus_augmented=None, folds=[])
    old_audit.compare(expected, stored, "arm", issues)
    return dict(issues=issues, fits=fits, stopped_at="completed_core_fits")


def run(replay, output):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Audit output must be new")
    receipt = json.loads((replay/"completion_receipt.json").read_text())
    if not receipt["completed"]: raise ValueError("Replay not complete")
    checked = {str(replay/"completion_receipt.json"): old_audit.digest(replay/"completion_receipt.json")}
    def bind(path):
        path = Path(path).resolve(); value = old_audit.digest(path)
        expected = receipt["files_sha256"].get(str(path))
        if expected is not None or path.parent == replay or path.parent == replay/"inputs":
            if value != expected: raise ValueError("Unbound/changed replay artifact: "+str(path))
        checked[str(path)] = value
    bind(replay/"cache_manifest.json")
    cache = json.loads((replay/"cache_manifest.json").read_text())
    # Choose before opening arm results; no availability or score search.
    chosen = choose_keys(cache)
    selection_sha = hashlib.sha256(json.dumps(chosen, separators=(",", ":")).encode()).hexdigest()
    bind(replay/"states.jsonl")
    records = {identity(row): row for row in (json.loads(line) for line in (replay/"states.jsonl").read_text().splitlines())}
    if len(records) != 1698: raise ValueError("Wrong frozen state union")
    for name in ("audit_accuracy_v43_numerical.py", "audit_accuracy_v42_numerical.py",
                 "accuracy_v43_components.py", "accuracy_v43_bounds.py", "accuracy_v42_localized.py"):
        bind(Path(__file__).resolve().parent/name)
    bind(Path(__file__).resolve().parents[1]/"tests/unit/test_accuracy_v43_numerical_audit.py")
    details, issues, counts = [], [], Counter(selected_states=len(chosen))
    for key in chosen:
        record = records[key]
        archive = record["archive"]
        path = (replay/archive["path"]).resolve()
        if path.parent != replay/"inputs": raise ValueError("Unexpected archive scope")
        bind(path)
        if old_audit.digest(path) != archive["sha256"]: raise ValueError("Archive hash mismatch")
        with np.load(path, allow_pickle=False) as packet:
            archive_issues = old_audit.archive_geometry_audit(packet, record["geometry"]["geometry"])
            issues.extend(dict(key=list(key), arm="archive_geometry", **issue) for issue in archive_issues)
            history, current, offset = packet["history129"], packet["current129"], packet["predicted_offset_xy"]
            centers = [None if not np.isfinite(p).all() else p.tolist() for p in packet["prior_centers_xy"]]
        polarity = key[-1].split(":")[0]
        legacy = old_audit.independent_components(history, centers, offset, polarity)
        protected = protected_components(history, centers, offset, polarity)
        dependency = protected_dependency_audit(history, centers, offset, polarity, protected, legacy)
        arm_details = {}
        for arm, components in (("stable_only", legacy), ("combined", protected)):
            detail = audit_arm(current, history, centers, offset, polarity, components, record["arms"][arm], arm == "combined")
            arm_details[arm] = detail
            counts["arms_checked"] += 1
            counts["stopped_at:"+detail["stopped_at"]] += 1
            counts["independent_fit_calls"] += len(detail["fits"])
            counts["free_branches_checked"] += len(detail["fits"])
            counts["boundary_branches_checked"] += sum(fit["diagnostics"]["boundary_fit"] is not None for fit in detail["fits"])
            issues.extend(dict(key=list(key), arm=arm, **issue) for issue in detail["issues"])
        counts["protected_fixed_templates_checked"] += dependency["fixed_templates_checked"]
        counts["protected_normalized_stamps_checked"] += dependency["fixed_normalized_stamps_checked"]
        counts["protected_opportunity_seeds_checked"] += dependency["opportunity_seeds_checked"]
        issues.extend(dict(key=list(key), arm="protected_dependency", **issue) for issue in dependency["issues"])
        details.append(dict(key=list(key), protected_dependency=dependency, arms=arm_details))
        print("Audited "+str(key)+" cumulative issues "+str(len(issues)), flush=True)
    for path, value in checked.items():
        if old_audit.digest(path) != value: raise ValueError("Audit input changed: "+path)
    result = dict(schema="seaqr.accuracy-v43-independent-numerical-audit.v1", completed=True, passed=not issues,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
                  selection=dict(keys=[list(key) for key in chosen], selection_sha256=selection_sha,
                                 rule="Four predeclared known diagnostics plus first geometry-available identity per clip, before arm results read"),
                  checked_files_sha256=checked, issues=issues, details=details, production_changed=False,
                  limitations=["No raw AVI decode or native warp verification", "Bounded selected states only",
                               "Protected producer reused; its dependencies/stamps independently checked",
                               "Core component-error arrays reuse frozen bounds helper if reached",
                               "No physical accuracy or calibrated uncertainty claim",
                               "Unsupported early stages prevent real-data checks of downstream branches"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], counts=result["counts"], issues=len(issues))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.replay, args.output)
    raise SystemExit(0 if result["passed"] else 1)
