"""Independent audit of the V53 offset-only shadow experiment.

The immutable V52 *auditor* supplies analytic-input and literal-JSON integrity
helpers. No model, runner, benchmark producer or real-reader is imported here.
V53 offset certificates, predictions and single-arm evaluation are recomputed
independently. No mutable V52 loss configuration is reused or changed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INHERITED_SHA256 = {
    "scripts/accuracy_v52_benchmark.py": "702e04065dd2e67facdaf8cc616e8fe2e7e199d69a1748aea936f98a7177fe4b",
    "scripts/accuracy_v52_real_scope.py": "8433633a62ab76dc48f6e65f72d95406009b3d133469ed5169e8f7594feaeef0",
    "scripts/audit_accuracy_v52_crossfit.py": "a17fc7e3f52603146c4dbcb976e88247dfb082efa08f076723fe65c53653476f",
    "tests/unit/test_accuracy_v52_benchmark.py": "5ab538619835a82d4fe7bb155c9dc0f5e3f2c08b75befc9d8dc12c632bf88138",
    "tests/unit/test_accuracy_v52_real_scope.py": "46933c316eeba758c26b8122e2d41833c3349ab8a7549502293bc8a87176c368",
    "tests/unit/test_accuracy_v52_audit.py": "96ce1eee10d7d740c896edc777c738e416293b14b760a116f98915b4f63ad078",
}


def verify_inherited_sources(root):
    """Check literal frozen sources before importing/trusting inherited helpers."""
    for name, expected_hash in INHERITED_SHA256.items():
        path = root / name
        if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash):
            raise ValueError("Pinned inherited V52 source changed: " + name)


verify_inherited_sources(ROOT)

from audit_accuracy_v52_crossfit import (
    V50_RECEIPT_SHA, OLD_NAMES, RUN_NAMES, POINTS, require, plain, digest,
    compare, raw_file, checked_read, specifications, generated_input,
    real_inputs, geometry, error_metrics, distribution, combined,
)


OUTPUT = ROOT / "results/tiny_target/accuracy_v53_20260926"
V50 = ROOT / "results/tiny_target/accuracy_v50_20260926/prediction_01"
SOURCE_NAMES = (
    "docs/accuracy_v53_plan.md", "scripts/accuracy_v53_offset.py",
    "scripts/run_accuracy_v53_offset.py", "scripts/audit_accuracy_v53_offset.py",
    "tests/unit/test_accuracy_v53_offset.py", "tests/unit/test_accuracy_v53_runner.py",
    "tests/unit/test_accuracy_v53_audit.py", "scripts/accuracy_v52_benchmark.py",
    "scripts/accuracy_v52_real_scope.py", "scripts/audit_accuracy_v52_crossfit.py",
    "tests/unit/test_accuracy_v52_benchmark.py", "tests/unit/test_accuracy_v52_real_scope.py",
    "tests/unit/test_accuracy_v52_audit.py",
)
SPLITS = ("left_right", "checkerboard")
LOSSES = ("median_offset",)
ARMS = ("corrected", "median8", "median3")


def midpoint(lower, upper):
    """Correctly rounded exact median midpoint, including subnormal values."""
    return float((Fraction(float(lower)) + Fraction(float(upper))) / 2)


def constants():
    return dict(loss="absolute_deviation", fixed_gain=1.0,
        fixed_x_slope=0.0, fixed_y_slope=0.0, minimum_finite_training_rows=4,
        guard_coordinate_min=8, guard_coordinate_max=120, guard_coordinate_step=8,
        guard_center=64, guard_chebyshev_radius_min=40, guard_chebyshev_radius_max=56,
        median_rule="correctly_rounded_exact_rational_midpoint_of_middle_residuals",
        objective_rule="correctly_rounded_exact_mean_of_finite_absolute_deviations",
        optimality_rule="zero_in_mean_L1_subgradient_interval_from_exact_order_counts",
        clipping_or_tuning=False, implicit_fallback=False,
        arithmetic_failure_discards_entire_fit=True)


def reconstructed_fit(xy, slow, current):
    """Independently reconstruct the full median/order certificate, no model import."""
    xy = geometry(xy)
    slow = np.asarray(slow, dtype=float)
    current = np.asarray(current, dtype=float)
    require(slow.shape == current.shape == (len(xy),), "Training vector shape differs")
    used = [math.isfinite(s) and math.isfinite(c) for s, c in zip(slow, current)]
    n = sum(used)
    result = dict(schema_version=1, loss="median_offset", available=False, unavailable_reason=None,
        training_count=len(xy), training_used_count=n, training_used_mask=used,
        training_unavailable_reasons=[None if selected else "nonfinite_training_slow_or_current" for selected in used],
        training_input_sha256=digest(dict(points_xy=xy, slow=slow, current=current)),
        offset_dn=None, candidate_offset_dn=None, median_interval_dn=[None, None],
        objective_mae_dn=None, subgradient_interval=[None, None], residual_order_counts=None,
        constants=constants())

    def finished(reason=None):
        result["unavailable_reason"] = reason
        result["model_sha256"] = digest(result)
        return result

    if n < 4:
        return finished("insufficient_finite_training_rows")
    residuals = [float(c) - float(s) for s, c, selected in zip(slow, current, used) if selected]
    if not all(math.isfinite(value) for value in residuals):
        return finished("nonfinite_training_residual_arithmetic")
    ordered = sorted(residuals)
    lower, upper = ordered[(n - 1) // 2], ordered[n // 2]
    candidate = midpoint(lower, upper)
    if not math.isfinite(candidate) or not lower <= candidate <= upper:
        return finished("nonfinite_or_outside_median_interval")
    below, above, tied = (sum(test(value, candidate) for value in residuals) for test in
                          (lambda v, c: v < c, lambda v, c: v > c, lambda v, c: v == c))
    interval = [(below - above - tied) / n, (below - above + tied) / n]
    result.update(candidate_offset_dn=candidate, median_interval_dn=[lower, upper],
        subgradient_interval=interval, residual_order_counts=dict(below=below, above=above, tied=tied))
    if below + above + tied != n or not interval[0] <= 0 <= interval[1]:
        return finished("median_subgradient_certificate_failed")
    deviations = [abs(value - candidate) for value in residuals]
    if not all(math.isfinite(value) for value in deviations):
        return finished("nonfinite_objective_arithmetic")
    objective = float(sum(map(Fraction, deviations), Fraction()) / n)
    if not math.isfinite(objective):
        return finished("nonfinite_objective_arithmetic")
    result.update(available=True, offset_dn=candidate, objective_mae_dn=objective)
    return finished()


def check_fit(fit, xy, slow, current):
    require(fit["model_sha256"] == digest(fit, ("model_sha256",)), "Model fingerprint differs")
    compare(reconstructed_fit(xy, slow, current), fit, "Exact independent median fit", rtol=0, atol=0)


def check_forecast(input_row, forecast):
    require(set(forecast) == {"schema_version", "split", "total_count", "fold_id", "fits", "predictions",
                             "constants", "metadata", "crossfit_sha256"}, "Forecast top-level schema differs")
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
    compare(len(xy), forecast["total_count"], "forecast count")
    compare(1, forecast["schema_version"], "forecast schema")
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
        values = [None] * len(xy)
        available = [False] * len(xy)
        reasons = [None] * len(xy)
        for fold in (0, 1):
            heldout = fold_id == fold
            fit = forecast["fits"][loss][str(fold)]
            check_fit(fit, xy[~heldout], slow[~heldout], current[~heldout])
            for index in np.flatnonzero(heldout):
                if not fit["available"]:
                    reasons[index] = "fit_unavailable:" + fit["unavailable_reason"]
                elif not math.isfinite(slow[index]):
                    reasons[index] = "nonfinite_prediction_slow"
                else:
                    value = float(slow[index]) + fit["offset_dn"]
                    if math.isfinite(value):
                        values[index] = value
                        available[index] = True
                    else:
                        reasons[index] = "nonfinite_prediction_arithmetic"
        compare(dict(values=values, available=available, unavailable_reasons=reasons, model_sha256=None),
                forecast["predictions"][loss], "predictions", rtol=0, atol=0)


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
                "offset_dn", "median_interval_dn", "objective_mae_dn", "subgradient_interval")} |
                {"reason": fit["unavailable_reason"]} for key, fit in forecast["fits"][loss].items()})
        if row["kind"] == "synthetic":
            truth = np.asarray(row["clean_current_background"], dtype=float)
            clean = available & np.isfinite(truth) & np.isfinite(slow) & np.isfinite(fast)
            entry["clean_truth_scored_indices"] = np.flatnonzero(clean).tolist()
            entry["clean_truth_metrics"] = {arm: error_metrics(truth[clean] - value[clean]) for arm, value in arms.items()}
    return result


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
        state_keys = {tuple(state["state_key"]) for state in states}
        require(len(state_keys) == len(states), "Duplicate state ledger key")
        packet_keys = [tuple(row["metadata"]["state_key"]) for row in rows]
        require(len(packet_keys) == len(set(packet_keys)), "Duplicate packet key")
        expected_packets = {tuple(state["state_key"]) for state in states if state["v50_status"] in
                            ("background_measured", "response_unavailable")}
        require(set(packet_keys) == expected_packets, "Missing or extra archived state key")
        frame_keys = {(key[0], key[2], key[1]) for key in state_keys}
        frames = defaultdict(list)
        for row in rows:
            key = row["metadata"]["state_key"]
            frames[(key[0], key[2], key[1])].append(row)
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


def check_input_rows(saved, synthetic_inputs, actual_inputs):
    """Allow only analytic-expression rounding; inherited real values are exact."""
    boundary = len(synthetic_inputs)
    require(len(saved) == boundary + len(actual_inputs), "Input row count differs")
    compare(synthetic_inputs, saved[:boundary], "analytic input rows", rtol=0, atol=1e-12)
    compare(actual_inputs, saved[boundary:], "exact inherited real input rows", rtol=0, atol=0)


def audit_run(run):
    verify_inherited_sources(ROOT)
    run = Path(run).absolute()
    require(run.parent == OUTPUT and run.resolve() == run and run.is_dir(), "Canonical immediate V53 run child required")
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
    synthetic_inputs = [generated_input(spec) for spec in specifications()]
    inputs = synthetic_inputs + actual_inputs
    check_input_rows(artifacts["inputs.jsonl"], synthetic_inputs, actual_inputs)
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
    by_id = {row["input_id"]: row for row in artifacts["inputs.jsonl"]}
    scores = []
    for index, record in enumerate(forecasts, 1):
        row = by_id[record["input_id"]]
        check_forecast(row, record["forecast"])
        scores.append(expected_score(row, record["forecast"]))
        if index % 200 == 0 or index == len(forecasts):
            print(f"V53 audit: checked {index}/{len(forecasts)} cross-fits", flush=True)
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
        all_offset_median_intervals_l1_certificates_and_objectives_checked=True,
        all_unavailable_fits_independently_reconstructed=True,
        all_predictions_and_masks_checked=True, entire_summary_recomputed=True,
        complementary_training_input_hashes_verified=True, no_core_or_media_access=True,
        no_producer_implementation_imports=True, immutable_v52_auditor_helpers_reused=True,
        no_physical_or_object_accuracy_claim=True, reduction_comparison_rtol=1e-10,
        reduction_comparison_atol=1e-9, input_comparison_atol=1e-12, files_sha256=bindings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.absolute()
    require(destination.parent == OUTPUT and destination.resolve() == destination
            and not destination.exists(), "Fresh immediate V53 audit artifact required")
    result = audit_run(args.run)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
