"""Read-only independent accounting of the completed synthetic V45 matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/tiny_target/accuracy_v45_20260925"
OLD = ROOT/"results/tiny_target/accuracy_v44_20260925/synthetic_01"
PINNED_V44_RECEIPT = "782949f426b9538922e580175ec91241cf7569cffc4cc8568a81109d1b44d76c"
ARMS = {"amplitude_old_bounds": ("amplitude", "v43"), "presence_old_bounds": ("numerator", "v43"),
        "amplitude_box_bounds": ("amplitude", "v45"), "presence_box_bounds": ("numerator", "v45")}
ORACLE_IDS = ("ordinary_moving", "background_only", "uniform_brightness_change", "plane_brightness_change",
    "fixed_flicker_separate", "moving_with_separate_fixed_flicker", "fixed_source_exact_overlap",
    "raw_normalization_unsupported_metadata", "uncertainty_contains_zero", "short_history", "unknown_nuisance_support",
    "slow_motion", "hovering", "curved_accelerating_motion", "identifiability_moving_world", "identifiability_fixed_emitter_world")
CAUSAL_IDS = ("ordinary_prior_learned", "shared_affine_change", "gain_and_affine_change", "current_source_absent",
             "current_source_six_pixels_off_forecast", "insufficient_prior_centers")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return not isinstance(value, float) or bool(np.isfinite(value))


def validate_evidence(value, quantity):
    require(quantity in ("amplitude", "numerator"), "Unknown evidence quantity")
    require(finite_json(value), "Nonfinite JSON evidence")
    require(value["motion_status"] == value["physical_class"] == "unknown", "Numerical evidence promoted to a physical label")
    require(type(value["available"]) is bool, "Availability must be Boolean")
    operative = "estimate" if quantity == "amplitude" else "numerator"
    fields = [operative, "error_bound", "interval", "interval_excludes_zero", "coefficient_sign"]
    if quantity == "numerator":
        fields += ["analytic_error_bound", "analytic_interval", "numerical_resolution_margin"]
    if not value["available"]:
        require(bool(value["reasons"]), "Unavailable evidence lacks a reason")
        require(all(value[k] is None for k in fields), "Unavailable evidence contains an operative score")
        return
    require(value["reasons"] == [] and value["error_bound"] >= 0, "Available evidence has failure reason or negative bound")
    center = value[operative]
    require(value["interval"] == [center-value["error_bound"], center+value["error_bound"]], "Interval algebra mismatch")
    low, high = value["interval"]
    require(value["interval_excludes_zero"] is bool(low > 0 or high < 0), "Zero-inclusion flag mismatch")
    expected_sign = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
    require(value["coefficient_sign"] == expected_sign, "Evidence sign mismatch")
    if quantity == "numerator":
        a, margin = value["analytic_error_bound"], value["numerical_resolution_margin"]
        require(a >= 0 and margin >= 0 and value["error_bound"] == a+margin, "Numerical margin hidden or incorrectly combined")
        require(value["analytic_interval"] == [center-a, center+a], "Analytic numerator interval changed")
        require(value["diagnostics"]["numerical_resolution_is_ieee_certified_enclosure"] is False,
                "Heuristic numerical margin promoted to a proved IEEE enclosure")


def counts(values):
    result = dict(states=0, available=0, unavailable=0, excludes_zero=0, includes_zero=0, positive=0, negative=0, unresolved=0)
    for value in values:
        result["states"] += 1
        if value is None or not value["available"]:
            result["unavailable"] += 1
        else:
            result["available"] += 1
            result["excludes_zero" if value["interval_excludes_zero"] else "includes_zero"] += 1
            result[value["coefficient_sign"]] += 1
    return result


def compare_archives(new, old, name):
    fingerprints = {}
    with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), "Archive membership differs: "+name)
        for key in a.files:
            x, y = a[key], b[key]
            require(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True),
                    "V44 input changed: "+name+":"+key)
            h = hashlib.sha256()
            h.update(str((x.shape, x.dtype.str)).encode("ascii"))
            h.update(x.tobytes(order="C"))
            fingerprints[name+":"+key] = h.hexdigest()
    return fingerprints


def verify_bindings(bindings, run):
    for name, digest in bindings.items():
        path = Path(name).resolve()
        safe = (ROOT/"scripts" in path.parents or ROOT/"tests/unit" in path.parents
                or path in (ROOT/"docs/accuracy_v44_plan.md", ROOT/"docs/accuracy_v45_plan.md")
                or OLD in path.parents or run in path.parents)
        require(safe, "Binding outside synthetic/code scope: "+name)
        require(sha(path) == digest, "Frozen hash mismatch: "+name)


def audit(run):
    run = Path(run).resolve()
    require(run.parent == BASE, "Audit must address a direct V45 result directory")
    receipt = load(run/"completion_receipt.json")
    baseline_receipt = load(OLD/"completion_receipt.json")
    require(sha(OLD/"completion_receipt.json") == PINNED_V44_RECEIPT, "V44 receipt changed")
    require(receipt["completed"] is True and baseline_receipt["completed"] is True, "Incomplete run")
    verify_bindings(receipt["files_sha256"], run)
    verify_bindings(baseline_receipt["files_sha256"], run)
    freeze, inputs, start = (load(run/name) for name in ("freeze.json", "inputs_complete.json", "score_start.json"))
    records, causal, summary = (load(run/name) for name in ("oracle_results.json", "causal_results.json", "summary.json"))
    old_oracle, old_causal = (load(OLD/name) for name in ("oracle_results.json", "causal_results.json"))
    for value in (receipt, baseline_receipt, freeze, inputs, start, records, causal, summary):
        require(finite_json(value), "Nonfinite run JSON")
    require(freeze["baseline_receipt_sha256"] == PINNED_V44_RECEIPT, "Baseline binding differs")
    require(freeze["arms"] == {k:list(v) for k, v in ARMS.items()}, "Predeclared arms changed")
    expected_bindings = set(freeze["dependency_sha256"]) | set(inputs["inputs_sha256"]) | {
        str(run/name) for name in ("freeze.json", "inputs_complete.json", "score_start.json", "oracle_results.json", "causal_results.json", "summary.json")}
    require(set(receipt["files_sha256"]) == expected_bindings, "Completion receipt binding denominator changed")
    for nested in (freeze["dependency_sha256"], inputs["inputs_sha256"]):
        require(all(receipt["files_sha256"].get(name) == digest for name, digest in nested.items()), "Nested hash binding changed")
    require(start["freeze_sha256"] == sha(run/"freeze.json") and start["inputs_complete_sha256"] == sha(run/"inputs_complete.json"),
            "Score-start input binding mismatch")
    require(inputs["scores_started"] is False and freeze["all_22_inputs_array_equal_to_v44"] is True, "Pre-score input declaration changed")
    times = [datetime.fromisoformat(v["created_at_utc"]) for v in (freeze, inputs, start, receipt)]
    require(times == sorted(times), "Producer chronology out of order")
    require([r["case_id"] for r in records] == [r["case_id"] for r in old_oracle] == list(ORACLE_IDS), "Oracle denominator/order changed")
    require([r["case_id"] for r in causal] == [r["case_id"] for r in old_causal] == list(CAUSAL_IDS), "Causal denominator/order changed")
    require(freeze["oracle_manifest"] == load(OLD/"freeze.json")["oracle_manifest"], "Oracle assumptions changed")
    require(freeze["causal_cases"] == list(CAUSAL_IDS), "Causal manifest changed")
    names = list(ORACLE_IDS)+["causal_"+name for name in CAUSAL_IDS]
    paths = {str(run/"inputs"/(name+".npz")) for name in names}
    require(set(inputs["inputs_sha256"]) == paths and {str(p.resolve()) for p in (run/"inputs").iterdir()} == paths,
            "Input archive denominator or files changed")
    fingerprints = {}
    for name in names:
        fingerprints.update(compare_archives(run/"inputs"/(name+".npz"), OLD/"inputs"/(name+".npz"), name))
    require(fingerprints == freeze["in_memory_input_sha256"], "Saved arrays differ from pre-score memory fingerprints")

    oracle_details = []
    for record, old in zip(records, old_oracle):
        require(set(record["arms"]) == set(ARMS), "Oracle arm omitted")
        require(record["arms"]["amplitude_old_bounds"] == old["contrast"], "V44 oracle baseline changed")
        require(record["motion_status"] == record["physical_class"] == "unknown", "Oracle physical promotion")
        require(record["temporal_metadata_used_by_numerical_solver"] is False, "Oracle solver used external metadata")
        for key in ("truth", "integration", "temporal_provenance"):
            require(record[key] == old[key], "Original oracle "+key+" changed")
        require(record["arms"]["amplitude_old_bounds"] == record["arms"]["amplitude_box_bounds"], "Oracle amplitude bound arms differ")
        require(record["arms"]["presence_old_bounds"] == record["arms"]["presence_box_bounds"], "Oracle numerator bound arms differ")
        for name, value in record["arms"].items():
            validate_evidence(value, ARMS[name][0])
        oracle_details.append(dict(case_id=record["case_id"],
            amplitude_available=record["arms"]["amplitude_old_bounds"]["available"],
            numerator_available=record["arms"]["presence_old_bounds"]["available"],
            numerator_interval=record["arms"]["presence_old_bounds"]["interval"],
            numerator_sign=record["arms"]["presence_old_bounds"]["coefficient_sign"],
            external_unknown_reasons=record["integration"]["unknown_reasons"]))
    require(records[-2]["arms"] == records[-1]["arms"], "Twin evidence differs")
    for suffix in ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound", "history_images", "current_image"):
        require(fingerprints["identifiability_moving_world:"+suffix] == fingerprints["identifiability_fixed_emitter_world:"+suffix],
                "Twin saved input differs: "+suffix)

    causal_details = []
    identical_fields = ("learned_design_sha256", "common_support_sha256", "common_support_count", "ambiguity_reasons",
                        "prior_context", "components", "conditional_on", "uncertainty_excludes")
    for record, old in zip(causal, old_causal):
        require(set(record["arms"]) == set(ARMS), "Causal arm omitted")
        baseline = record["arms"]["amplitude_old_bounds"]["raw_adapter_result"]
        require(baseline == old["result"], "V44 causal baseline changed")
        detail = dict(case_id=record["case_id"], arms={}, ambiguity_reasons=baseline["ambiguity_reasons"])
        for name, (quantity, version) in ARMS.items():
            wrapped = record["arms"][name]
            require(wrapped["arm"] == name and wrapped["quantity"] == quantity and wrapped["bound_version"] == version,
                    "Arm quantity/version mislabeled")
            require(wrapped["legacy_numerical_contrast_field_contains"] == quantity, "Numerator mislabeled as amplitude")
            value = wrapped["raw_adapter_result"]
            for layer in (wrapped, value):
                require(layer["motion_status"] == layer["physical_class"] == "unknown" and layer["is_motion_or_classification_gate"] is False,
                        "Causal physical classification promoted")
            for key in identical_fields:
                require(value[key] == baseline[key], "Nominal causal evidence changed: "+key)
            score = value["numerical_contrast"]
            require(value["available"] is (score is not None and score["available"]), "Adapter availability changed score semantics")
            if score is not None:
                validate_evidence(score, quantity)
                metadata = value["component_bounds"]["metadata"]
                require(metadata["aligned_value_error_bound_dn"] == .5 and metadata["moving_residual_stamp_error_bound_dn"] == 1.
                        and metadata["fixed_highpass_stamp_error_bound_dn"] == 1., "Declared input error bounds changed")
                require(metadata["moving"]["used_history_indices"] == baseline["component_bounds"]["metadata"]["moving"]["used_history_indices"],
                        "Used moving stamp omitted or added")
                require([(v["centre_xy"], v["used_history_indices"]) for v in metadata["fixed"]]
                        == [(v["centre_xy"], v["used_history_indices"]) for v in baseline["component_bounds"]["metadata"]["fixed"]],
                        "Fixed membership/anchor changed")
                if version == "v45":
                    require(metadata["nominal_components_unchanged"] is True and metadata["moving"]["uncertain_used_stamp_omitted"] is False,
                            "Box bound changed nominal template or dropped a stamp")
            else:
                require(value["reasons"] == ["insufficient_actual_prior_centers"], "Unexplained unavailable causal case")
            detail["arms"][name] = dict(quantity=quantity, available=value["available"],
                interval=None if score is None else score["interval"],
                coefficient_sign=None if score is None else score["coefficient_sign"], reasons=value["reasons"])
        require(record["arms"]["amplitude_old_bounds"]["raw_adapter_result"]["component_bounds"] ==
                record["arms"]["presence_old_bounds"]["raw_adapter_result"]["component_bounds"], "Solver changes altered old bounds")
        require(record["arms"]["amplitude_box_bounds"]["raw_adapter_result"]["component_bounds"] ==
                record["arms"]["presence_box_bounds"]["raw_adapter_result"]["component_bounds"], "Solver changes altered new bounds")
        causal_details.append(detail)

    oracle_counts = {name: counts(r["arms"][name] for r in records) for name in ARMS}
    causal_counts = {name: counts(r["arms"][name]["raw_adapter_result"]["numerical_contrast"] for r in causal) for name in ARMS}
    require(summary["oracle_counts_by_arm"] == oracle_counts and summary["causal_counts_by_arm"] == causal_counts, "Summary counts differ")
    for key in ("oracle_box_arms_duplicate_supplied_bounds", "baseline_exactly_reproduced", "nominal_design_support_and_ambiguity_identical_across_arms",
                "observational_twins_identical", "numerator_and_amplitude_are_different_quantities", "synthetic_only", "no_motion_or_airborne_classification", "no_real_accuracy_gain_claimed"):
        require(summary[key] is True, "Summary scope mismatch: "+key)
    require(summary["production_changed"] is False and summary["real_media_read"] is False, "Summary scope changed")
    changes = [d for d in oracle_details if d["amplitude_available"] != d["numerator_available"]]
    require([d["case_id"] for d in changes] == ["uncertainty_contains_zero"] and changes[0]["numerator_sign"] == "unresolved",
            "Unexpected oracle availability difference")
    new_positive = {name: [d["case_id"] for d in causal_details if d["arms"][name]["coefficient_sign"] == "positive"] for name in ARMS}
    require(new_positive["amplitude_old_bounds"] == new_positive["presence_old_bounds"] == [], "Old-bound result changed")
    require(new_positive["amplitude_box_bounds"] == new_positive["presence_box_bounds"] == list(CAUSAL_IDS[:3]),
            "Learned-bound positive controls changed")
    verify_bindings(receipt["files_sha256"], run)
    verify_bindings(baseline_receipt["files_sha256"], run)
    checked = dict(baseline_receipt["files_sha256"])
    checked.update(receipt["files_sha256"])
    for path in (run/"completion_receipt.json", OLD/"completion_receipt.json", Path(__file__).resolve(),
                 ROOT/"tests/unit/test_accuracy_v45_accounting.py"):
        checked[str(path)] = sha(path)
    return dict(schema="seaqr.accuracy-v45-independent-accounting-audit.v1", completed=True, passed=True, issues=[],
        created_at_utc=datetime.now(timezone.utc).isoformat(), checked_files_sha256=checked,
        counts=dict(v45_completion_bindings=len(receipt["files_sha256"]), v44_completion_bindings=len(baseline_receipt["files_sha256"]),
                    distinct_checked_files=len(checked), input_archives=22, input_arrays=len(fingerprints),
                    oracle_cases=16, causal_cases=6, arms=4, saved_case_arm_entries=88),
        oracle_counts_by_arm=oracle_counts, causal_counts_by_arm=causal_counts,
        all_input_arrays_equal_to_v44=True, all_v44_baseline_results_exact=True,
        nominal_design_support_ambiguity_membership_and_dn_assumptions_unchanged=True,
        twin_arrays_and_four_arm_outputs_equal=True, all_motion_and_class_unknown=True,
        oracle_availability_changes=changes, oracle_cases=oracle_details, causal_cases=causal_details,
        positive_causal_case_ids_by_arm=new_positive, producer_chronology=[t.isoformat() for t in times],
        synthetic_only=True, real_data_accessed=False, production_changed=False,
        conclusions=[
            "Tighter learned-template bounds resolve the same three source-present causal controls for BOTH amplitude and numerator; numerator-only with old bounds resolves none",
            "Absent and off-forecast causal controls retain zero-containing intervals; missing history remains explicitly unavailable",
            "Source-erasing oracle uncertainty gains numerator availability but still includes zero; it is not an added detection",
            "All fixed-alternative warnings and physical unknowns remain; no measured false-alarm, recall, motion or airborne-class gain is demonstrated",
        ], limitations=[
            "Producer UTC chronology and hash binding are not independent clock attestation",
            "This accounting audit validates saved inputs/results/semantics, not a separate derivation or rerun of every numerical score",
            "Oracle bound-arm pairs are duplicates, not independent evidence; numerator and amplitude interval widths have different meanings",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == BASE, "Accounting output must remain directly in V45 base, outside frozen run")
    frozen = output.with_name(output.stem+"_freeze.json")
    require(not output.exists() and not frozen.exists(), "Fresh audit and freeze paths required")
    files = (Path(__file__).resolve(), ROOT/"tests/unit/test_accuracy_v45_accounting.py",
             args.run.resolve()/"completion_receipt.json", OLD/"completion_receipt.json")
    bindings = {str(path):sha(path) for path in files}
    with frozen.open("x") as stream:
        json.dump(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), files_sha256=bindings,
                       run=str(args.run.resolve()), saved_before_audit=True, synthetic_only=True), stream, indent=2)
        stream.write("\n")
    report = audit(args.run)
    require(all(sha(path) == digest for path, digest in bindings.items()), "Audit binding changed during execution")
    report.update(audit_freeze=str(frozen), audit_freeze_sha256=sha(frozen))
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), passed=report["passed"], counts=report["counts"])))


if __name__ == "__main__":
    main()
