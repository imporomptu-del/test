"""Independent JSON-only audit of one frozen V52 shadow cross-fit run.

No benchmark, model, runner, or real-reader implementation is imported. Every
input, fit, prediction, score, and summary group is checked. Reads are limited
to literal V50 JSON files, named V52 run artifacts, and named V52 source files.
Neither old maps nor receipt paths authorize arbitrary file or media access.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/tiny_target/accuracy_v52_20260926"
V50 = ROOT / "results/tiny_target/accuracy_v50_20260926/prediction_01"
V50_RECEIPT_SHA = "9213ed07fa7e8efd81f7c0032290dbe2f8c37bd06f6376b86d0b53a1af7dc2b1"
OLD_NAMES = ("freeze.json", "state_results.jsonl", "calibration_forecasts.jsonl",
             "calibration_measurements.jsonl", "evaluation_forecasts.jsonl",
             "evaluation_measurements.jsonl", "reference_context.json")
RUN_NAMES = ("freeze.json", "inputs.jsonl", "state_results.jsonl", "reference_context.json",
             "forecasts.jsonl", "forecasts_frozen.json", "scores.jsonl", "summary.json")
SOURCE_NAMES = (
    "docs/accuracy_v52_plan.md", "scripts/accuracy_v52_benchmark.py",
    "scripts/accuracy_v52_crossfit.py", "scripts/accuracy_v52_real_scope.py",
    "scripts/run_accuracy_v52_crossfit.py", "scripts/audit_accuracy_v52_crossfit.py",
    "tests/unit/test_accuracy_v52_benchmark.py", "tests/unit/test_accuracy_v52_crossfit.py",
    "tests/unit/test_accuracy_v52_real_scope.py", "tests/unit/test_accuracy_v52_runner.py",
    "tests/unit/test_accuracy_v52_audit.py",
)
SPLITS = ("left_right", "checkerboard")
LOSSES = ("ols", "huber")
ARMS = ("corrected", "median8", "median3")
FAMILIES = ("stable", "offset_plus8", "offset_minus8", "gain125", "gain075",
            "gain_zero", "gain_negative", "plane", "gain_plane", "recent_step",
            "pulse_ended", "long_pulse_ended", "localized_change", "sparse_bright",
            "sparse_dark", "stripe", "missing_prior", "missing_current_left",
            "missing_current_right", "all_current_missing")
POINTS = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
          if 40 <= max(abs(x - 64), abs(y - 64)) <= 56]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value, exclude=()):
    if exclude:
        value = {key: item for key, item in value.items() if key not in exclude}
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compare(expected, actual, context="root", rtol=1e-10, atol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), context + ": keys differ")
        for key in expected:
            compare(expected[key], actual[key], context + "." + str(key), rtol, atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(expected) == len(actual), context + ": length differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{context}[{index}]", rtol, atol)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol), context + ": numeric value differs")
    else:
        require(type(expected) is type(actual) and expected == actual, context + ": value/type differs")


def raw_file(path):
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            "Canonical regular file required: " + str(path))
    return path.read_bytes()


def checked_read(path, expected_hash, bindings):
    raw = raw_file(path)
    require(hashlib.sha256(raw).hexdigest() == expected_hash, "File hash differs: " + str(path))
    bindings[str(path)] = expected_hash
    return ([json.loads(line) for line in raw.splitlines()]
            if path.name.endswith(".jsonl") else json.loads(raw))


def constants():
    return dict(huber_delta_dn=2.0, max_irls_updates=100, kkt_tolerance_dn=1e-7,
        rank_rtol=1e-12, condition_limit=1e8, training_scale_floor_dn=1.0,
        coordinate_center=64.0, coordinate_scale=56.0, gain_lower_bound=0.0,
        minimum_finite_training_rows=4,
        kkt_gradient_scaling="mean_gradient / max(1, RMS(conditioned_design_column))",
        full_rank_required=True, implicit_ridge_or_fallback=False)


def specifications():
    return [dict(schema="accuracy_v52_generated_benchmark_v1",
        case_id=f"{family}_{background}_noise{level}_seed{seed}", family=family,
        background=background, noise_level=level, seed=seed, history_length=8,
        point_count=144, analytic_simulation_only=True, physical_sensor_model=False)
        for family in FAMILIES for background in ("constant", "planar", "textured")
        for level, seed in ((0, 71), (1, 991))]


def generated_input(spec):
    require(spec in specifications(), "Unknown generated specification")
    xy = np.asarray(POINTS)
    dx, dy = xy.T.astype(float) - 64
    base = np.full(144, 96.)
    if spec["background"] != "constant":
        base += .05 * dx + .03 * dy
    if spec["background"] == "textured":
        base += 12 * np.sin(dx / 12) + 9 * np.cos(dy / 15) + 6 * np.sin((dx + dy) / 17)
    prior = np.tile(base, (8, 1))
    family = spec["family"]
    truth = base.copy()
    signal = np.zeros(144)
    if family in ("offset_plus8", "offset_minus8"):
        truth += 8 if family == "offset_plus8" else -8
    elif family in ("gain125", "gain075"):
        truth *= 1.25 if family == "gain125" else .75
    elif family == "gain_zero":
        truth[:] = 96
    elif family == "gain_negative":
        truth = 192 - base
    elif family == "plane":
        truth += 4 * dx / 56 - 3 * dy / 56
    elif family == "gain_plane":
        truth = 1.15 * base + 8 + 4 * dx / 56 - 3 * dy / 56
    elif family in ("recent_step", "pulse_ended"):
        prior[-2:] += 8
        if family == "recent_step":
            truth += 8
    elif family == "long_pulse_ended":
        prior += 8
    elif family == "localized_change":
        signal[(dx >= 0) & (dy >= 0)] = 16
    elif family in ("sparse_bright", "sparse_dark"):
        signal[np.arange(144) % 17 == 0] = 40 if family == "sparse_bright" else -40
    elif family == "stripe":
        signal[abs(dx) <= 8] = 32
    noise = np.random.default_rng(spec["seed"]).uniform(-spec["noise_level"], spec["noise_level"], (9, 144))
    prior += noise[:8]
    current = truth + signal + noise[8]
    if family == "missing_prior":
        prior[0, :8] = np.nan
    elif family == "missing_current_left":
        current[dx < 0] = np.nan
    elif family == "missing_current_right":
        current[dx >= 0] = np.nan
    elif family == "all_current_missing":
        current[:] = np.nan
    return plain(dict(input_id=spec["case_id"], kind="synthetic", metadata=spec,
        points_xy=xy, median8=np.median(prior, axis=0), median3=np.median(prior[-3:], axis=0),
        current=current, clean_current_background=truth, contamination_mask=signal != 0))


def real_inputs(documents):
    """Join original rows by frozen state key, never following any old paths."""
    states = documents["state_results.jsonl"]
    keyed = {tuple(row["state_key"]): row for row in states}
    require(len(keyed) == len(states), "Duplicate original states")
    assignments = {tuple(row["state_key"]): row for row in documents["freeze.json"]["scope"]["assignments"]}
    require(len(assignments) == len(documents["freeze.json"]["scope"]["assignments"]), "Duplicate original assignments")
    require(keyed.keys() == assignments.keys(), "Original scope membership differs")
    require(all(row["source_scores_and_original_detections_unchanged"] is True for row in states),
            "Original source decisions were not preserved")
    rows = []
    packet_available = 0
    points_available = 0
    for partition in ("calibration", "evaluation"):
        forecasts = {tuple(row["state_key"]): row for row in documents[partition + "_forecasts.jsonl"]}
        measured = {tuple(row["state_key"]): row for row in documents[partition + "_measurements.jsonl"]}
        expected = {key for key, row in assignments.items() if row["partition"] == partition and row["archived"]}
        require(set(forecasts) == set(measured) == expected, "Original forecast/measurement membership differs")
        require(len(forecasts) == len(documents[partition + "_forecasts.jsonl"])
                and len(measured) == len(documents[partition + "_measurements.jsonl"]), "Duplicate original packets")
        for key in sorted(expected):
            f = forecasts[key]["forecast"]
            m = measured[key]["measurement"]
            require(f["forecast_sha256"] == digest(f, ("forecast_sha256",)), "Original forecast fingerprint differs")
            require(m["forecast_sha256"] == f["forecast_sha256"] and
                    m["used_support_sha256"] == f["used_support_sha256"], "Original packet binding differs")
            points = f["used_points_xy"]
            require(f["used_count"] == m["used_count"] == len(points), "Original point count differs")
            require(f["used_support_sha256"] == digest(points), "Original support fingerprint differs")
            slow = f["arms"]["median8_unit_scale"]["prediction"]
            fast = f["arms"]["median3_temporal_scale"]["prediction"]
            require(slow == f["arms"]["median8_temporal_scale"]["prediction"], "Original median8 copies differ")
            if m["available"]:
                reconstructions = [[p + r for p, r in zip(f["arms"][arm]["prediction"],
                    m["arms"][arm]["residuals"])] for arm in
                    ("median8_unit_scale", "median8_temporal_scale", "median3_temporal_scale")]
                require(reconstructions[0] == reconstructions[1] == reconstructions[2], "Original current reconstruction disagrees")
                current = reconstructions[0]
                require(len(current) == len(points) and all(math.isfinite(v) for v in current), "Invalid reconstructed current")
                packet_available += 1
                points_available += len(points)
            else:
                require(m["arms"] == {} and m["current_nonfinite_used_point_count"] > 0,
                        "Original unavailable packet has unexpected residuals")
                current = [None] * len(points)
            require(keyed[key]["partition"] == partition, "Original partition differs")
            require(keyed[key]["v50_status"] == ("background_measured" if m["available"] else "response_unavailable"),
                    "Original status/current availability differs")
            rows.append(dict(input_id="real_" + "_".join(map(str, key)), kind="real",
                metadata=dict(state_key=list(key), partition=partition, available=m["available"], reasons=m["reasons"]),
                points_xy=points, median8=slow, median3=fast, current=current))
    count = sum(len(row["current"]) for row in rows)
    counts = dict(states=len(states), status_counts=dict(Counter(row["v50_status"] for row in states)),
        scored_archived_packets=len(rows), available_current_packets=packet_available,
        unavailable_current_packets=len(rows) - packet_available, guard_point_opportunities=count,
        available_current_points=points_available, unavailable_current_packet_points=count - points_available)
    return rows, counts


def geometry(points):
    xy = np.asarray(points, dtype=float)
    require(xy.ndim == 2 and xy.shape[1] == 2, "Invalid guard point shape")
    require(np.isfinite(xy).all() and ((xy >= 8) & (xy <= 120) & (xy % 8 == 0)).all(), "Invalid guard grid")
    radius = np.max(abs(xy - 64), axis=1)
    require(((radius >= 40) & (radius <= 56)).all(), "Source/core or outside point in guard")
    require(len(set(map(tuple, xy))) == len(xy), "Duplicate guard points")
    return xy


def conditioning(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(sum(singular > 1e-12 * singular[0]))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return rank, condition


def numerical_optimum(design, response, loss):
    """Independent constrained IRLS replay used to verify unavailable outcomes.

    Available models are additionally checked directly against the convex loss
    KKT conditions using their recorded coefficients, not only this replay.
    """
    weights = np.ones(len(response))
    previous = {}
    for iteration in range(101):
        scaled = design * np.sqrt(weights[:, None])
        target = response * np.sqrt(weights)
        rank, condition = conditioning(scaled)
        if rank != 4:
            return dict(reason="weighted_rank_deficient_design", iterations=max(0, iteration - 1), **previous)
        if condition is None or condition > 1e8:
            return dict(reason="weighted_ill_conditioned_design", iterations=max(0, iteration - 1), **previous)
        coefficients = np.linalg.lstsq(scaled, target, rcond=1e-12)[0]
        if coefficients[0] < 0:
            coefficients = np.r_[0., np.linalg.lstsq(scaled[:, 1:], target, rcond=1e-12)[0]]
        objective, kkt, residual = optimality(design, response, coefficients, loss)
        previous = dict(objective=objective, kkt=kkt, coefficients=coefficients.copy())
        if kkt <= 1e-7:
            return dict(reason=None, iterations=iteration, objective=objective, kkt=kkt, coefficients=coefficients)
        if loss == "ols":
            return dict(reason="ols_kkt_check_failed", iterations=iteration, objective=objective, kkt=kkt, coefficients=coefficients)
        if iteration == 100:
            return dict(reason="irls_nonconvergence", iterations=iteration, objective=objective, kkt=kkt, coefficients=coefficients)
        weights = np.ones(len(response))
        outside = abs(residual) > 2
        weights[outside] = 2 / abs(residual[outside])


def optimality(design, response, coefficients, loss):
    residual = design @ coefficients - response
    if loss == "ols":
        objective = float(np.mean(.5 * residual ** 2))
        influence = residual
    else:
        magnitude = abs(residual)
        quadratic = np.minimum(magnitude, 2)
        objective = float(np.mean(.5 * quadratic ** 2 + 2 * (magnitude - quadratic)))
        influence = np.clip(residual, -2, 2)
    gradient = design.T @ influence / len(response)
    gradient /= np.maximum(1., np.sqrt(np.mean(design ** 2, axis=0)))
    if coefficients[0] == 0:
        gradient[0] = min(0., gradient[0])
    return objective, float(max(abs(gradient))), residual


def check_fit(fit, xy, slow, current, loss):
    """Validate one model using only its complementary training fold."""
    require(fit["model_sha256"] == digest(fit, ("model_sha256",)), "Model fingerprint differs")
    require(fit["training_input_sha256"] == digest(dict(points_xy=xy, slow=slow, current=current)), "Training input binding differs")
    compare(constants(), fit["constants"], "fit.constants")
    used = np.isfinite(slow) & np.isfinite(current)
    compare(plain(used), fit["training_used_mask"], "training_used_mask")
    compare([None if good else "nonfinite_training_slow_or_current" for good in used],
            fit["training_unavailable_reasons"], "training_reasons")
    require(fit["schema_version"] == 1 and fit["loss"] == loss and fit["training_count"] == len(xy)
            and fit["training_used_count"] == int(used.sum()), "Training model identity/count differs")
    if used.sum() < 4:
        require(fit["unavailable_reason"] == "insufficient_finite_training_rows", "Missing-row failure reason differs")
        require(fit["center"] is None and fit["scale"] is None and fit["rank"] is None
                and fit["condition_number"] is None and fit["iterations"] == 0, "Missing-row diagnostics differ")
    else:
        s = slow[used]
        center = float(np.median(s))
        scale = max(1., float(np.median(abs(s - center))))
        compare(center, fit["center"], "train_only_center")
        compare(scale, fit["scale"], "train_only_scale")
        design = np.column_stack(((s - center) / scale, np.ones(len(s)), (xy[used] - 64) / 56))
        rank, condition = conditioning(design)
        compare(rank, fit["rank"], "rank")
        compare(condition, fit["condition_number"], "condition_number")
        if rank < 4:
            require(fit["unavailable_reason"] == "rank_deficient_design" and fit["iterations"] == 0,
                    "Rank failure not preserved")
        elif condition is None or condition > 1e8:
            require(fit["unavailable_reason"] == "ill_conditioned_design" and fit["iterations"] == 0,
                    "Condition failure not preserved")
        else:
            solved = numerical_optimum(design, current[used], loss)
            require(fit["unavailable_reason"] == solved["reason"], "Independent numerical result differs")
            require(fit["available"] is (solved["reason"] is None), "Availability differs from independent solve status")
            require(fit["iterations"] == solved["iterations"], "Independent iteration count differs")
            if "objective" in solved:
                compare(solved["objective"], fit["objective"], "objective replay")
                compare(solved["kkt"], fit["kkt_residual_dn"], "KKT replay", atol=1e-10)
                candidate = np.asarray(fit["last_candidate_coefficients_conditioned"], dtype=float)
                require(candidate.shape == (4,) and np.isfinite(candidate).all() and candidate[0] >= 0,
                        "Invalid retained last candidate")
                compare(plain(solved["coefficients"]), fit["last_candidate_coefficients_conditioned"],
                        "Last successful replay candidate")
                candidate_objective, candidate_kkt, _ = optimality(design, current[used], candidate, loss)
                compare(candidate_objective, fit["objective"], "Last candidate objective")
                compare(candidate_kkt, fit["kkt_residual_dn"], "Last candidate KKT", atol=1e-10)
            else:
                require(fit["objective"] is None and fit["kkt_residual_dn"] is None,
                        "No successful iterate can have loss diagnostics")
                compare([None] * 4, fit["last_candidate_coefficients_conditioned"], "No successful replay candidate")
            if fit["available"]:
                coefficient = np.asarray(fit["coefficients_conditioned"], dtype=float)
                require(coefficient.shape == (4,) and np.isfinite(coefficient).all() and coefficient[0] >= 0,
                        "Invalid constrained coefficients")
                compare(fit["coefficients_conditioned"], fit["last_candidate_coefficients_conditioned"], "Available last candidate")
                objective, kkt, _ = optimality(design, current[used], coefficient, loss)
                require(kkt <= 1e-7, "Available fit violates fixed KKT tolerance")
                compare(objective, fit["objective"], "Recorded coefficient objective")
                compare(kkt, fit["kkt_residual_dn"], "Recorded coefficient KKT", atol=1e-10)
                gain = coefficient[0] / scale
                physical = [gain, coefficient[1] - gain * center, coefficient[2], coefficient[3]]
                compare(physical, fit["coefficients_physical"], "physical coefficients")
                require(fit["gain_bound_active"] is bool(gain == 0) and fit["converged"] is True,
                        "Gain-bound/convergence status differs")
                require(fit["unavailable_reason"] is None, "Available model has failure reason")
                return
    require(fit["available"] is False and fit["converged"] is False
            and fit["gain_bound_active"] is None, "Unavailable model promoted or repaired")
    compare([None] * 4, fit["coefficients_conditioned"], "Unavailable conditioned coefficients")
    compare([None] * 4, fit["coefficients_physical"], "Unavailable physical coefficients")
    if fit["unavailable_reason"] in ("insufficient_finite_training_rows", "rank_deficient_design", "ill_conditioned_design"):
        require(fit["objective"] is None and fit["kkt_residual_dn"] is None, "Unsolved model has loss diagnostics")
        compare([None] * 4, fit["last_candidate_coefficients_conditioned"], "Unsolved last candidate")


def check_forecast(input_row, forecast):
    require(forecast["crossfit_sha256"] == digest(forecast, ("crossfit_sha256",)), "Cross-fit fingerprint differs")
    split = forecast["split"]
    require(split in SPLITS, "Unexpected split")
    xy = geometry(input_row["points_xy"])
    slow = np.asarray(input_row["median8"], dtype=float)
    current = np.asarray(input_row["current"], dtype=float)
    require(slow.shape == current.shape == (len(xy),), "Input vector shape differs")
    fold_id = ((xy[:, 0] >= 64).astype(int) if split == "left_right"
               else ((xy[:, 0] // 8 + xy[:, 1] // 8) % 2).astype(int))
    compare(fold_id.tolist(), forecast["fold_id"], "fold assignment")
    require(forecast["total_count"] == len(xy) and forecast["schema_version"] == 1, "Forecast point count/schema differs")
    compare(constants(), forecast["constants"], "forecast constants")
    compare(dict(current_complementary_guard_values_used=True,
        heldout_current_argument_accepted_by_predict=False, fit_dictionary_key_names_heldout_fold=True,
        core_pixels_accepted=False, scoring_or_forecast_selection_performed=False,
        unavailable_indices_preserved_without_fallback=True, prior_only_or_online_camera_causality_certified=False,
        guard_purity_or_guard_to_core_transfer_certified=False, production_detection_modified=False),
        forecast["metadata"], "forecast metadata")
    require(set(forecast["fits"]) == set(LOSSES) == set(forecast["predictions"]), "Loss membership differs")
    for loss in LOSSES:
        require(set(forecast["fits"][loss]) == {"0", "1"}, "Fold membership differs")
        values = np.full(len(xy), np.nan)
        available = np.zeros(len(xy), dtype=bool)
        reasons = [None] * len(xy)
        for fold in (0, 1):
            heldout = fold_id == fold
            fit = forecast["fits"][loss][str(fold)]
            check_fit(fit, xy[~heldout], slow[~heldout], current[~heldout], loss)
            for index in np.flatnonzero(heldout):
                if not fit["available"]:
                    reasons[index] = "fit_unavailable:" + fit["unavailable_reason"]
                elif not math.isfinite(slow[index]):
                    reasons[index] = "nonfinite_prediction_slow"
                else:
                    design = np.r_[(slow[index] - fit["center"]) / fit["scale"], 1., (xy[index] - 64) / 56]
                    value = float(design @ np.asarray(fit["coefficients_conditioned"], dtype=float))
                    if math.isfinite(value):
                        values[index] = value
                        available[index] = True
                    else:
                        reasons[index] = "nonfinite_prediction_arithmetic"
        compare(plain(dict(values=values, available=available, unavailable_reasons=reasons, model_sha256=None)),
                forecast["predictions"][loss], "predictions")


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def error_metrics(errors):
    values = [abs(float(value)) for value in errors]
    count = len(values)
    total = math.fsum(values)
    return dict(count=count, absolute_sum_dn=total,
        squared_sum_dn2=math.fsum(value * value for value in values),
        mae_dn=total / count if count else None,
        median_absolute_error_dn=quantile(values, .5) if count else None,
        p90_absolute_error_dn=quantile(values, .9) if count else None,
        max_absolute_error_dn=max(values) if count else None)


def distribution(values):
    values = list(values)
    return dict(count=len(values), mean=math.fsum(values) / len(values) if values else None,
        median=quantile(values, .5) if values else None, p90=quantile(values, .9) if values else None,
        max=max(values) if values else None)


def expected_score(row, forecast):
    current = np.asarray(row["current"], dtype=float)
    slow, fast = (np.asarray(row[name], dtype=float) for name in ("median8", "median3"))
    n = len(current)
    result = dict(input_id=row["input_id"], kind=row["kind"], split=forecast["split"], metadata=row["metadata"],
        total_points=n, current_available_count=int(np.isfinite(current).sum()), baseline_all={}, arms={})
    for name, pred in (("median8", slow), ("median3", fast)):
        use = np.isfinite(current) & np.isfinite(pred)
        result["baseline_all"][name] = dict(metrics=error_metrics(current[use] - pred[use]),
            complete=bool(n and use.all()), scored_indices=np.flatnonzero(use).tolist())
    for loss in LOSSES:
        prediction = forecast["predictions"][loss]
        values = np.asarray(prediction["values"], dtype=float)
        available = np.asarray(prediction["available"], dtype=bool)
        use = available & np.isfinite(current) & np.isfinite(slow) & np.isfinite(fast)
        arms = {"corrected": values, "median8": slow, "median3": fast}
        result["arms"][loss] = entry = dict(prediction_available_count=int(available.sum()), scored_count=int(use.sum()),
            complete=bool(n and use.all()), scored_indices=np.flatnonzero(use).tolist(),
            prediction_unavailable_reasons=dict(Counter(reason for reason in prediction["unavailable_reasons"] if reason is not None)),
            metrics={arm: error_metrics(current[use] - value[use]) for arm, value in arms.items()},
            residuals={arm: (current[use] - value[use]).tolist() for arm, value in arms.items()},
            fold_fits={key: {field: fit[field] for field in ("available", "training_count", "training_used_count",
                "rank", "condition_number", "iterations", "kkt_residual_dn", "gain_bound_active")} |
                {"reason": fit["unavailable_reason"]} for key, fit in forecast["fits"][loss].items()})
        if row["kind"] == "synthetic":
            truth = np.asarray(row["clean_current_background"], dtype=float)
            clean = available & np.isfinite(truth) & np.isfinite(slow) & np.isfinite(fast)
            entry["clean_truth_scored_indices"] = np.flatnonzero(clean).tolist()
            entry["clean_truth_metrics"] = {arm: error_metrics(truth[clean] - value[clean]) for arm, value in arms.items()}
    return result


def combined(records):
    records = list(records)
    count = sum(record["count"] for record in records)
    valid = [record for record in records if record["count"]]
    return dict(point_count=count,
        conditional_point_mae_dn=math.fsum(record["absolute_sum_dn"] for record in records) / count if count else None,
        conditional_point_rmse_dn=math.sqrt(math.fsum(record["squared_sum_dn2"] for record in records) / count) if count else None,
        maximum_absolute_error_dn=max(record["max_absolute_error_dn"] for record in valid) if valid else None,
        packet_mae=distribution(record["mae_dn"] for record in valid),
        packet_p90_absolute_error=distribution(record["p90_absolute_error_dn"] for record in valid),
        packet_maximum_absolute_error=distribution(record["max_absolute_error_dn"] for record in valid))


def expected_aggregate(rows, states=None):
    result = dict(packet_records=len(rows), point_opportunities=sum(row["total_points"] for row in rows),
        current_available_points=sum(row["current_available_count"] for row in rows), baseline_all={}, arms={})
    for base in ("median8", "median3"):
        result["baseline_all"][base] = dict(complete_packets=sum(row["baseline_all"][base]["complete"] for row in rows),
            metrics=combined(row["baseline_all"][base]["metrics"] for row in rows))
    for loss in LOSSES:
        records = [row["arms"][loss] for row in rows]
        complete = [record for record in records if record["complete"]]
        result["arms"][loss] = entry = dict(
            prediction_available_points=sum(record["prediction_available_count"] for record in records),
            scored_points=sum(record["scored_count"] for record in records), complete_packets=len(complete),
            incomplete_packets=len(records) - len(complete),
            fold_unavailable_reasons=dict(Counter(fit["reason"] for record in records for fit in record["fold_fits"].values() if not fit["available"])),
            matched_metrics={arm: combined(record["metrics"][arm] for record in records) for arm in ARMS},
            shared_complete_packet_metrics={arm: combined(record["metrics"][arm] for record in complete) for arm in ARMS})
        if rows and rows[0]["kind"] == "synthetic":
            entry["clean_truth_metrics"] = {arm: combined(record["clean_truth_metrics"][arm] for record in records) for arm in ARMS}
    if states is not None:
        frame_keys = {(state["state_key"][0], state["state_key"][2], state["state_key"][1]) for state in states}
        frames = defaultdict(list)
        for row in rows:
            key = row["metadata"]["state_key"]
            frames[(key[0], key[2], key[1])].append(row)
        require(set(frames) <= frame_keys, "Archive frames absent from ledger")
        result.update(states=len(states), original_status_counts=dict(Counter(state["v50_status"] for state in states)),
            selected_response_frames=len(frame_keys), frames_with_scored_archives=len(frames),
            frames_without_scored_archives=len(frame_keys) - len(frames))
        for base in ("median8", "median3"):
            complete = [frame for frame in frames.values() if all(row["baseline_all"][base]["complete"] for row in frame)]
            result["baseline_all"][base].update(complete_frame_count=len(complete),
                complete_frame_mean_packet_mae=distribution(math.fsum(row["baseline_all"][base]["metrics"]["mae_dn"] for row in frame) / len(frame) for frame in complete))
        for loss in LOSSES:
            complete = [frame for frame in frames.values() if all(row["arms"][loss]["complete"] for row in frame)]
            result["arms"][loss].update(complete_frame_count=len(complete), incomplete_archived_frame_count=len(frames) - len(complete),
                complete_frame_mean_packet_mae={arm: distribution(math.fsum(row["arms"][loss]["metrics"][arm]["mae_dn"] for row in frame) / len(frame) for frame in complete) for arm in ARMS},
                complete_frame_maximum_absolute_error={arm: distribution(max(row["arms"][loss]["metrics"][arm]["max_absolute_error_dn"] for row in frame) for frame in complete) for arm in ARMS})
    return result


def expected_summary(rows, states, created):
    result = dict(completed=True, created_at_utc=created, splits={}, no_source_decisions=True,
        current_guard_estimation_not_prior_forecasting=True, no_physical_or_object_accuracy_claim=True)
    for split in SPLITS:
        synthetic = [row for row in rows if row["split"] == split and row["kind"] == "synthetic"]
        real = [row for row in rows if row["split"] == split and row["kind"] == "real"]
        value = dict(synthetic=expected_aggregate(synthetic), real=expected_aggregate(real, states),
            synthetic_groups={}, real_groups={}, nine_frame_bins={})
        for field in ("family", "background", "noise_level"):
            groups = defaultdict(list)
            for row in synthetic:
                groups[str(row["metadata"][field])].append(row)
            value["synthetic_groups"][field] = {key: expected_aggregate(group) for key, group in sorted(groups.items())}
        for partition in ("calibration", "embargo", "evaluation"):
            for clip in sorted({state["state_key"][0] for state in states}):
                subset = [state for state in states if state["partition"] == partition and state["state_key"][0] == clip]
                packets = [row for row in real if row["metadata"]["partition"] == partition and row["metadata"]["state_key"][0] == clip]
                value["real_groups"][partition + "_" + clip] = expected_aggregate(packets, subset)
                for bin_id in sorted({state["state_key"][1] // 9 for state in subset}):
                    value["nine_frame_bins"][f"{partition}_{clip}_{bin_id}"] = expected_aggregate(
                        [row for row in packets if row["metadata"]["state_key"][1] // 9 == bin_id],
                        [state for state in subset if state["state_key"][1] // 9 == bin_id])
        result["splits"][split] = value
    return result


def audit_run(run):
    run = Path(run).absolute()
    require(run.parent == OUTPUT and run.resolve() == run and run.is_dir(), "Canonical immediate V52 run child required")
    require({path.name for path in run.iterdir()} == set(RUN_NAMES) | {"completion_receipt.json"}, "Unexpected or missing run artifacts")
    bindings = {}
    receipt_path = run / "completion_receipt.json"
    receipt_hash = hashlib.sha256(raw_file(receipt_path)).hexdigest()
    receipt = checked_read(receipt_path, receipt_hash, bindings)
    require(receipt["completed"] is True and all(receipt[key] is False for key in
        ("source_decisions_changed", "production_changed", "media_accessed")), "Invalid completion declarations")
    source_paths = {str(ROOT / name) for name in SOURCE_NAMES}
    old_paths = {str(V50 / name) for name in OLD_NAMES} | {str(V50 / "completion_receipt.json")}
    artifact_paths = {str(run / name) for name in RUN_NAMES}
    require(set(receipt["files_sha256"]) == source_paths | old_paths | artifact_paths, "Completion receipt path allowlist differs")
    for name in SOURCE_NAMES:
        path = ROOT / name
        require(hashlib.sha256(raw_file(path)).hexdigest() == receipt["files_sha256"][str(path)], "Source changed: " + name)
        bindings[str(path)] = receipt["files_sha256"][str(path)]
    original_receipt = checked_read(V50 / "completion_receipt.json", V50_RECEIPT_SHA, bindings)
    require(original_receipt["completed"] is True, "Original V50 incomplete")
    documents = {name: checked_read(V50 / name, original_receipt["files_sha256"][str(V50 / name)], bindings) for name in OLD_NAMES}
    compare({path: bindings[path] for path in old_paths},
            {path: receipt["files_sha256"][path] for path in old_paths}, "completion literal input bindings")
    artifacts = {name: checked_read(run / name, receipt["files_sha256"][str(run / name)], bindings) for name in RUN_NAMES}
    freeze = artifacts["freeze.json"]
    compare({path: bindings[path] for path in source_paths}, freeze["source_files_sha256"], "freeze sources")
    compare({path: bindings[path] for path in old_paths}, freeze["input_files_sha256"], "freeze literal inputs")
    compare(specifications(), freeze["specifications"], "frozen generated cases")
    compare(constants(), freeze["model_constants"], "frozen model constants")
    compare(list(SPLITS), freeze["splits"], "frozen splits")
    require(freeze["before_actual_fitting"] is True and freeze["before_evaluation_scoring"] is True, "Freeze chronology flags differ")
    compare(documents["state_results.jsonl"], artifacts["state_results.jsonl"], "Original states unchanged", rtol=0, atol=0)
    compare(documents["reference_context.json"], artifacts["reference_context.json"], "Original reference context unchanged", rtol=0, atol=0)
    actual_inputs, counts = real_inputs(documents)
    require(counts["states"] == 1211 and counts["scored_archived_packets"] == 493
            and counts["guard_point_opportunities"] == 62867 and counts["available_current_points"] == 61220
            and counts["unavailable_current_packets"] == 14, "Frozen original denominators differ")
    compare(dict(background_measured=479, response_unavailable=14, history_unknown=702, embargo_not_scored=16),
            counts["status_counts"], "original state status denominators")
    compare(counts, freeze["real_counts"], "real accounting")
    inputs = [generated_input(spec) for spec in specifications()] + actual_inputs
    compare(inputs, artifacts["inputs.jsonl"], "all input rows", rtol=0, atol=1e-12)
    manifest = artifacts["forecasts_frozen.json"]
    require(manifest["completed"] is True and manifest["input_count"] == 613
            and manifest["forecast_count"] == 1226 and manifest["evaluation_scoring_started"] is False
            and manifest["current_training_pixels_used"] is True, "Forecast freeze contract differs")
    compare({str(run / name): bindings[str(run / name)] for name in ("inputs.jsonl", "forecasts.jsonl")},
            manifest["files_sha256"], "prediction freeze bindings")
    chronology = [freeze["created_at_utc"], manifest["created_at_utc"], artifacts["summary.json"]["created_at_utc"], receipt["created_at_utc"]]
    require([datetime.fromisoformat(value) for value in chronology] == sorted(datetime.fromisoformat(value) for value in chronology),
            "Saved freeze/scoring/completion chronology differs")
    forecasts = artifacts["forecasts.jsonl"]
    expected_membership = [(row["input_id"], split) for row in inputs for split in SPLITS]
    compare(expected_membership, [(row["input_id"], row["forecast"]["split"]) for row in forecasts], "forecast membership")
    # Analytic reconstruction above permits tiny differences in expression
    # ordering. Numerical/input fingerprints bind the exact saved observations.
    by_id = {row["input_id"]: row for row in artifacts["inputs.jsonl"]}
    scores = []
    for index, record in enumerate(forecasts, 1):
        row = by_id[record["input_id"]]
        check_forecast(row, record["forecast"])
        scores.append(expected_score(row, record["forecast"]))
        if index % 200 == 0 or index == len(forecasts):
            print(f"V52 audit: checked {index}/{len(forecasts)} cross-fits", flush=True)
    compare(scores, artifacts["scores.jsonl"], "all scored rows")
    compare(expected_summary(scores, artifacts["state_results.jsonl"], artifacts["summary.json"]["created_at_utc"]),
            artifacts["summary.json"], "entire summary")
    for path, expected_hash in bindings.items():
        require(hashlib.sha256(raw_file(Path(path))).hexdigest() == expected_hash, "Bound file changed during audit")
    fits = [fit for record in forecasts for loss in LOSSES for fit in record["forecast"]["fits"][loss].values()]
    return dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        verified_inputs=613, verified_synthetic_inputs=120, verified_real_inputs=493,
        verified_crossfits=1226, verified_fits=len(fits), verified_score_rows=len(scores),
        verified_states=1211, original_reference_context_unchanged=True,
        unavailable_fit_reasons=dict(Counter(fit["unavailable_reason"] for fit in fits if not fit["available"])),
        all_available_fit_kkt_and_objectives_checked=True, all_unavailable_fits_independently_replayed=True,
        all_predictions_and_masks_checked=True, entire_summary_recomputed=True,
        complementary_training_input_hashes_verified=True, no_core_or_media_access=True,
        independent_implementation_imports=True, no_physical_or_object_accuracy_claim=True,
        reduction_comparison_rtol=1e-10, reduction_comparison_atol=1e-9, input_comparison_atol=1e-12,
        files_sha256=bindings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.absolute()
    require(destination.parent == OUTPUT and destination.resolve() == destination
            and not destination.exists(), "Fresh immediate V52 audit artifact required")
    result = audit_run(args.run)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
