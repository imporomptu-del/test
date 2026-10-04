"""Post-score V43 diagnostics; never selects a policy or reads source imagery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DIAGNOSTICS = (
    ("0029", 298, 0, "bright:980"),
    ("0126", 140, 0, "bright:1001"),
    ("0029", 346, 0, "bright:2641"),
    ("0029", 347, 0, "bright:2641"),
)
ARMS = ("baseline", "stable_only", "protected_only", "combined")


def key(row):
    return row["clip"], row["frame_index"], row["segment"], row["track_id"]


def compact_arm(arm):
    if arm is None:
        return None
    result = {name: arm.get(name) for name in (
        "available", "ambiguous", "reasons", "ambiguity_reasons",
        "common_support_count", "common_support_sha256", "mse_stationary",
        "mse_augmented", "advantage_stationary_minus_augmented",
    )}
    result["fixed_anchor_count"] = arm["components"]["fixed_anchor_count"]
    annulus = (arm.get("stability") or {}).get("annulus")
    free = ((annulus or {}).get("diagnostics") or {}).get("free_fit") or {}
    result["annulus_free_fit"] = {name: free.get(name) for name in (
        "train_singular_values", "train_design_perturbation_spectral_bound",
        "robust_minimum_singular_value", "maximum_prediction_bound_dn",
        "prediction_budget_dn",
    )} if annulus else None
    return result


def verify_bindings(receipt, hashes):
    """Reject partial or changed run inputs; the full chain is audited separately."""
    if receipt.get("completed") is not True:
        raise ValueError("run is not completed")
    for path, digest in hashes.items():
        if receipt.get("files_sha256", {}).get(path) != digest:
            raise ValueError("run input missing or mismatched in completion receipt: " + path)


def describe(states, references):
    indexed = {key(row): row for row in states}
    if len(indexed) != len(states):
        raise ValueError("duplicate state key")
    diagnostics = []
    for wanted in DIAGNOSTICS:
        row = indexed[wanted]
        baseline, protected = row["arms"]["baseline"], row["arms"]["protected_only"]
        support = baseline.get("common_support_sha256")
        diagnostics.append({
            "key": wanted, "reference_samples": row["reference_samples"],
            "both_available_same_support": bool(
                baseline["available"] and protected["available"] and support
                and support == protected.get("common_support_sha256")),
            "arms": {name: compact_arm(row["arms"][name]) for name in ARMS},
        })
    extrema = {}
    for name in ARMS:
        available = [row for row in states if (row["arms"][name] or {}).get("available")]
        extrema[name] = {"score_available": len(available), "maxima": {}}
        for metric in ("mse_stationary", "mse_augmented"):
            row = max(available, key=lambda r: r["arms"][name][metric]) if available else None
            extrema[name]["maxima"][metric] = None if row is None else {
                "key": key(row), "value": row["arms"][name][metric],
                "ambiguity_reasons": row["arms"][name]["ambiguity_reasons"],
                "selected_after_scoring_for_description_only": True,
            }
    lost = []
    for sample in references["samples"]:
        original = sample["original"]
        stage = "strict_qualified_measurement" if original["panel"] == "grid" else "baseline_qualified"
        assignment = original["stages"][stage]
        if not assignment[0]:
            continue
        alternative = next(a for a in sample["measured_alternatives"] if a["identity"] == assignment[1])
        baseline, protected = (alternative["arms"][a] or {} for a in ("baseline", "protected_only"))
        if baseline.get("available") and not protected.get("available"):
            lost.append({"sample_index": sample["sample_index"], "panel": original["panel"],
                         "clip": original["clip_id"], "frame_index": original["frame_index"],
                         "identity": assignment[1], "reasons": protected.get("reasons")})
    return {
        "schema": "seaqr.accuracy-v43-descriptive-report.v1",
        "predeclared_exposed_diagnostics": diagnostics,
        "all_state_score_extrema": extrema,
        "original_strict_assignment_lost_score_protected_only": lost,
        "no_best_arm_or_identity_selection": True,
        "production_changed": False, "accuracy_gain_claimed": False,
        "limitations": ["Per-frame states are not independent objects or encounters",
                        "No scores are detector negatives; unavailable means unknown",
                        "Smaller image error and absence of warnings do not establish motion or airborne class",
                        "Protected-only retains the old unsupported numerical solver",
                        "Different-support MSEs are not like-for-like comparisons"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.run / "completion_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    files = [args.run / name for name in ("states.jsonl", "reference_evidence.json")]
    # The independent accounting audit separately checks the complete hash chain.
    payloads = [path.read_bytes() for path in files]
    inputs = {str(path.resolve()): hashlib.sha256(data).hexdigest() for path, data in zip(files, payloads)}
    verify_bindings(receipt, inputs)
    inputs[str(receipt_path.resolve())] = hashlib.sha256(receipt_bytes).hexdigest()
    report = describe([json.loads(line) for line in payloads[0].splitlines() if line], json.loads(payloads[1]))
    report["read_files_sha256"] = inputs
    report["read_input_hashes_match_completed_receipt"] = True
    report["reporting_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "states": sum(1 for line in payloads[0].splitlines() if line),
                      "diagnostics": len(report["predeclared_exposed_diagnostics"])}))


if __name__ == "__main__":
    main()
