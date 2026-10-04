"""Audit the completed V44 synthetic-only matrix without decoding real media.

This checks frozen bindings, saved inputs against their generators, original
metadata, denominators and evidence semantics. It does not independently derive
the contrast interval or certify image-based motion or physical classification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from accuracy_v44_synthetic_cases import build_cases, scenario_manifest
from run_accuracy_v44_synthetic import causal_cases


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/tiny_target/accuracy_v44_20260925"
ARRAY_KEYS = ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound")
ORACLE_IDS = (
    "ordinary_moving", "background_only", "uniform_brightness_change", "plane_brightness_change",
    "fixed_flicker_separate", "moving_with_separate_fixed_flicker", "fixed_source_exact_overlap",
    "raw_normalization_unsupported_metadata", "uncertainty_contains_zero", "short_history",
    "unknown_nuisance_support", "slow_motion", "hovering", "curved_accelerating_motion",
    "identifiability_moving_world", "identifiability_fixed_emitter_world",
)
CAUSAL_IDS = (
    "ordinary_prior_learned", "shared_affine_change", "gain_and_affine_change", "current_source_absent",
    "current_source_six_pixels_off_forecast", "insufficient_prior_centers",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_json(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, path+"."+key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float):
        require(np.isfinite(value), "nonfinite JSON number: "+path)


def validate_contrast(value, name):
    """Check semantics only, not a fresh derivation of its numeric error bound."""
    finite_json(value, name)
    require(type(value["available"]) is bool, name+": availability not boolean")
    require(value["motion_status"] == "unknown" and value["physical_class"] == "unknown",
            name+": source coefficient promoted to physical identity")
    require(value["diagnostics"]["no_production_gate"] is True, name+": gate scope changed")
    require(value["diagnostics"]["caller_template_temporal_or_placement_provenance_certified"] is False,
            name+": vector solver claims temporal provenance")
    require(value["diagnostics"]["calibrated_noise_or_confidence"] is False,
            name+": deterministic interval promoted to calibrated confidence")
    if not value["available"]:
        require(bool(value["reasons"]), name+": unavailable without a reason")
        for field in ("estimate", "error_bound", "interval", "interval_excludes_zero", "coefficient_sign"):
            require(value[field] is None, name+": unavailable result retains operative "+field)
        return
    require(value["reasons"] == [], name+": available result retains failure reasons")
    estimate, bound = value["estimate"], value["error_bound"]
    require(isinstance(estimate, (int, float)) and isinstance(bound, (int, float)) and bound >= 0,
            name+": invalid estimate or bound")
    require(value["interval"] == [estimate-bound, estimate+bound], name+": interval algebra mismatch")
    low, high = value["interval"]
    require(value["interval_excludes_zero"] is bool(low > 0 or high < 0), name+": zero inclusion mismatch")
    expected_sign = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
    require(value["coefficient_sign"] == expected_sign, name+": coefficient sign mismatch")


def compare_archive(path, expected):
    with np.load(path, allow_pickle=False) as actual:
        require(set(actual.files) == set(expected), str(path)+": archive array membership changed")
        for name, generated in expected.items():
            array = np.asarray(generated)
            value = actual[name]
            require(value.dtype == array.dtype and value.shape == array.shape,
                    str(path)+":"+name+": dtype/shape mismatch")
            require(np.array_equal(value, array, equal_nan=True), str(path)+":"+name+": generated values changed")


def audit(run):
    run = Path(run).resolve()
    require(run.parent == BASE, "Audit is restricted to a direct V44 synthetic result directory")
    receipt_path = run/"completion_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt["completed"] is True, "Synthetic run not completed")
    require(receipt["production_changed"] is False and receipt["real_media_read"] is False,
            "Receipt scope changed")
    bindings = receipt["files_sha256"]
    for path, expected in bindings.items():
        resolved = Path(path).resolve()
        allowed = ((ROOT/"scripts") in resolved.parents or (ROOT/"tests/unit") in resolved.parents
                   or resolved == ROOT/"docs/accuracy_v44_plan.md" or run in resolved.parents)
        require(allowed, "Unexpected receipt input outside synthetic/code scope: "+path)
        require(sha(path) == expected, "Completion binding mismatch: "+path)
    freeze = read_json(run/"freeze.json")
    inputs_complete = read_json(run/"inputs_complete.json")
    score_start = read_json(run/"score_start.json")
    records = read_json(run/"oracle_results.json")
    probes = read_json(run/"causal_results.json")
    summary = read_json(run/"summary.json")
    for name, value in (("freeze", freeze), ("inputs_complete", inputs_complete), ("score_start", score_start),
                        ("oracle_results", records), ("causal_results", probes), ("summary", summary)):
        finite_json(value, name)
    expected_run_json = {str(run/name) for name in (
        "freeze.json", "inputs_complete.json", "score_start.json", "oracle_results.json", "causal_results.json", "summary.json")}
    all_expected_bindings = set(freeze["dependency_sha256"]) | set(inputs_complete["inputs_sha256"]) | expected_run_json
    require(set(bindings) == all_expected_bindings, "Receipt adds or omits frozen dependencies/inputs/outputs")
    for collection in (freeze["dependency_sha256"], inputs_complete["inputs_sha256"]):
        require(all(bindings.get(p) == digest for p, digest in collection.items()), "Nested binding mismatch")
    require(freeze["synthetic_only"] is True, "Freeze not synthetic-only")
    require(freeze["causal_offset_is_predeclared_not_estimated_from_current"] is True, "Offset provenance changed")
    require(score_start["inputs_complete_sha256"] == sha(run/"inputs_complete.json"), "Inputs-complete binding mismatch")
    require(score_start["freeze_sha256"] == sha(run/"freeze.json"), "Freeze binding mismatch")
    require(score_start["all_inputs_rehashed"] is True and inputs_complete["scores_started"] is False,
            "Pre-score input declaration changed")
    moments = [datetime.fromisoformat(value["created_at_utc"]) for value in (freeze, inputs_complete, score_start, receipt)]
    require(moments == sorted(moments), "Producer chronology inconsistent")

    generated = build_cases()
    generated_probes = causal_cases()
    require([c["case_id"] for c in generated] == list(ORACLE_IDS), "Oracle generator denominator changed")
    require([c["case_id"] for c in generated_probes] == list(CAUSAL_IDS), "Causal generator denominator changed")
    require([c["case_id"] for c in records] == list(ORACLE_IDS), "Oracle result denominator/order changed")
    require([c["case_id"] for c in probes] == list(CAUSAL_IDS), "Causal result denominator/order changed")
    require(freeze["oracle_manifest"] == scenario_manifest(), "Predeclared oracle manifest changed")
    require(freeze["causal_cases"] == list(CAUSAL_IDS), "Predeclared causal cases changed")
    expected_archives = {str(run/"inputs"/(name+".npz")) for name in ORACLE_IDS}
    expected_archives |= {str(run/"inputs"/("causal_"+name+".npz")) for name in CAUSAL_IDS}
    require(set(inputs_complete["inputs_sha256"]) == expected_archives, "Saved synthetic input denominator changed")
    require({str(p.resolve()) for p in (run/"inputs").iterdir()} == expected_archives,
            "Unaccounted file in frozen inputs directory")

    oracle_details = []
    for case, record in zip(generated, records):
        name = case["case_id"]
        expected_arrays = {key: case[key] for key in ARRAY_KEYS}
        expected_arrays.update({key: case["synthetic_observations"][key] for key in ("history_images", "current_image")})
        compare_archive(run/"inputs"/(name+".npz"), expected_arrays)
        for field in ("truth", "integration", "temporal_provenance"):
            require(record[field] == case[field], name+": original "+field+" changed")
        require(record["temporal_metadata_used_by_numerical_solver"] is False, name+": solver used metadata")
        require(record["motion_status"] == "unknown" and record["physical_class"] == "unknown", name+": physical promotion")
        require(record["truth"]["source_template_origin"] == "oracle_solver_unit_fixture", name+": oracle origin changed")
        validate_contrast(record["contrast"], name)
        value = record["contrast"]
        oracle_details.append(dict(case_id=name, numerical_role=case["truth"]["numerical_role"],
            true_source_amplitude=case["truth"]["source_amplitude"], available=value["available"],
            interval=value["interval"], coefficient_sign=value["coefficient_sign"], reasons=value["reasons"],
            external_integration_unknown_reasons=case["integration"]["unknown_reasons"],
            motion_status=record["motion_status"], physical_class=record["physical_class"]))

    causal_details = []
    learned_signatures = []
    supports = []
    for case, record in zip(generated_probes, probes):
        name = case["case_id"]
        centers = np.asarray([[np.nan, np.nan] if c is None else c for c in case["prior_centers_xy"]])
        expected_arrays = {key: case[key] for key in ("current129", "history129", "predicted_offset_xy")}
        expected_arrays["prior_centers_xy"] = centers
        compare_archive(run/"inputs"/("causal_"+name+".npz"), expected_arrays)
        value = record["result"]
        require(value["synthetic_only"] is True and value["production_changed"] is False, name+": causal scope changed")
        require(value["is_motion_or_classification_gate"] is False, name+": causal coefficient became gate")
        require(value["motion_status"] == "unknown" and value["physical_class"] == "unknown", name+": causal physical promotion")
        actual_centers = sum(c is not None for c in case["prior_centers_xy"])
        require(value["prior_context"]["actual_prior_centers"] == actual_centers, name+": center denominator changed")
        require(value["prior_context"]["templates_use_prior_images_only"] is True, name+": prior provenance changed")
        require(value["prior_context"]["supplied_forecast_offset_independently_verified"] is False,
                name+": causal placement certification invented")
        require("physical_motion_and_class_not_certified_by_source_contrast" in value["ambiguity_reasons"],
                name+": physical ambiguity erased")
        contrast = value["numerical_contrast"]
        if contrast is None:
            require(value["available"] is False and bool(value["reasons"]), name+": missing contrast treated as available")
        else:
            validate_contrast(contrast, name)
            require(value["available"] == contrast["available"], name+": availability disagrees with contrast")
        if name == "insufficient_prior_centers":
            require(actual_centers == 4 and value["reasons"] == ["insufficient_actual_prior_centers"],
                    name+": missing-history case changed")
            require(contrast is None and value["components"] is None, name+": insufficient history was scored")
        else:
            learned_signatures.append(value["learned_design_sha256"])
            supports.append((value["common_support_count"], value["common_support_sha256"]))
        causal_details.append(dict(case_id=name, actual_prior_centers=actual_centers,
            available=value["available"], interval=None if contrast is None else contrast["interval"],
            coefficient_sign=None if contrast is None else contrast["coefficient_sign"], reasons=value["reasons"],
            ambiguity_reasons=value["ambiguity_reasons"], motion_status=value["motion_status"], physical_class=value["physical_class"]))
    require(all(s == learned_signatures[0] and s is not None for s in learned_signatures),
            "Current-only causal variants changed learned template hashes")
    require(all(s == supports[0] for s in supports), "Finite causal current variants changed common support")

    twins = generated[-2:]
    for key in ARRAY_KEYS:
        require(twins[0][key].tobytes() == twins[1][key].tobytes(), "Twin core input mismatch: "+key)
    for key in ("history_images", "current_image"):
        require(twins[0]["synthetic_observations"][key].tobytes() == twins[1]["synthetic_observations"][key].tobytes(),
                "Twin image observations mismatch: "+key)
    require(records[-2]["contrast"] == records[-1]["contrast"], "Identifiability twins changed numerical evidence")
    require(records[-2]["integration"] == records[-1]["integration"], "Twin integration flags changed")
    require(records[-2]["truth"]["physical_interpretation"] != records[-1]["truth"]["physical_interpretation"],
            "Twin physical alternative disappeared")
    base = records[0]["contrast"]
    for record in records:
        if record["case_id"] in ("raw_normalization_unsupported_metadata", "short_history", "unknown_nuisance_support",
                                  "slow_motion", "hovering", "curved_accelerating_motion"):
            require(record["contrast"] == base, record["case_id"]+": external metadata altered identical-vector scores")

    available = sum(r["contrast"]["available"] for r in records)
    excluded = sum(r["contrast"]["interval_excludes_zero"] is True for r in records)
    counts = dict(states=len(records), interval_available=available, interval_excludes_zero=excluded,
                  interval_includes_zero=available-excluded, unavailable=len(records)-available)
    expected_summary = dict(oracle_vector_counts=counts, observationally_identical_outputs_equal=True,
        causal_probe_count=len(probes), causal_intervals_available=sum(r["result"]["available"] for r in probes),
        causal_intervals_excluding_zero=sum((r["result"]["numerical_contrast"] or {}).get("interval_excludes_zero") is True for r in probes),
        synthetic_only=True, production_changed=False, real_media_read=False,
        no_physical_motion_or_airborne_decision=True, no_accuracy_gain_claimed=True)
    require(summary == expected_summary, "Saved summary disagrees with all-case accounting")

    # Check immutable inputs again after generator execution. New audit files
    # live outside the frozen run and are additional, not replacement bindings.
    for path, digest in bindings.items():
        require(sha(path) == digest, "Frozen binding changed during accounting audit: "+path)
    checked = dict(bindings)
    for path in (receipt_path, Path(__file__).resolve(), ROOT/"tests/unit/test_accuracy_v44_accounting_audit.py"):
        checked[str(path)] = sha(path)
    report = dict(schema="seaqr.accuracy-v44-independent-accounting-audit.v1",
        completed=True, passed=True, issues=[], created_at_utc=datetime.now(timezone.utc).isoformat(),
        run_directory=str(run), checked_files_sha256=checked,
        counts=dict(completion_bound_files=len(bindings), checked_files=len(checked),
                    dependency_files=len(freeze["dependency_sha256"]), synthetic_input_archives=len(expected_archives),
                    oracle_cases=len(records), causal_cases=len(probes),
                    oracle_cases_with_external_unknown_metadata=sum(bool(c["integration"]["unknown_reasons"]) for c in generated)),
        recomputed_summary=expected_summary, oracle_cases=oracle_details, causal_cases=causal_details,
        saved_arrays_equal_frozen_generators=True, all_truth_and_integration_metadata_unchanged=True,
        identifiability_twin_arrays_and_numerical_outputs_identical=True,
        metadata_only_changes_do_not_change_numerical_outputs=True,
        current_only_causal_variants_preserve_learned_design_hashes_and_support=True,
        producer_declared_chronology=[moment.isoformat() for moment in moments],
        production_changed=False, real_media_read=False,
        limitations=[
            "Producer-declared timestamps plus hash bindings are not independent clock attestation",
            "Saved inputs are compared to the frozen generators, not an independent physical scene acquisition",
            "This accounting audit does not independently derive or rerun numerical contrast intervals",
            "Oracle vector templates are not learned detections; external metadata unknowns are not image-inferred failures",
            "Numerical coefficient signs do not certify motion or airborne class",
            "All six causal probes remain physically unresolved; the missing-history case remains in its denominator",
        ])
    finite_json(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == BASE, "Audit output must be outside frozen run, directly in the V44 base")
    require(not output.exists(), "Audit output already exists; use a new artifact")
    report = audit(args.run)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(output), passed=report["passed"], counts=report["counts"]), indent=2))


if __name__ == "__main__":
    main()
