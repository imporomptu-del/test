"""Freeze all V46 controls plus 20 new guard cases before five-arm scoring."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from accuracy_v45_causal_probe import ARMS, evaluate_causal_probe
from accuracy_v46_synthetic_cases import build_cases as stress_cases
from accuracy_v48_synthetic_cases import build_cases as guard_cases, scenario_manifest, pre_score_contract
from accuracy_v47_probe import ARM, CURRENT_USE_DESCRIPTION, evaluate_probe
from run_accuracy_v44_synthetic import causal_cases
from run_accuracy_v46_stress import memory_hashes, check_nominal_and_physical
from run_accuracy_v46_shadow import validate_evidence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v48_20260926"
BASELINE = ROOT/"results/tiny_target/accuracy_v46_20260925/synthetic_01"
BASELINE_SHA = "252954f7c0a28b390f2f91c952faede59878cedcd6a9a63d277150ad7da31b51"
PREDECESSOR = ROOT/"results/tiny_target/accuracy_v47_20260926"
PREDECESSOR_SHA = "fdd0d3538d9b085e4d823d7cc863a3afd98c855d4bfcb9c21984f808efaad653"
FAILURE_SHA = "8b74c1a177bea4b306accb8b9414fec2e7f6fcef983f9c384f1d9dc5a58bc25f"
MATH_SHA = "0a9f17e3c724d894ccceabed9400aa1ac6aa1f844cf6d39b6be654eb10208140"
REPAIRED_CASE = "guard_correlated_guard_error_extremes"
KEYS = ("current129", "history129", "prior_centers_xy", "predicted_offset_xy", "polarity")
ARRAY_KEYS = KEYS[:-1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def require_hashes(bindings):
    changed = [p for p,h in bindings.items() if sha(p) != h]
    if changed:
        raise ValueError("Frozen file changed: "+repr(changed))


def source_dependencies():
    return sorted(set([ROOT/"docs/accuracy_v48_plan.md"]
        +list((ROOT/"scripts").glob("*accuracy_v48*.py"))
        +list((ROOT/"tests/unit").glob("test_accuracy_v48*.py"))))


def baseline_bindings():
    path = BASELINE/"completion_receipt.json"
    if sha(path) != BASELINE_SHA:
        raise ValueError("Pinned V46 synthetic receipt changed")
    receipt = json.loads(path.read_text())
    if not receipt["completed"] or not receipt["synthetic_only"] or receipt["real_packet_data_read"]:
        raise ValueError("Baseline is not completed synthetic-only evidence")
    require_hashes(receipt["files_sha256"])
    bound = dict(receipt["files_sha256"], **{str(path):BASELINE_SHA})
    predecessor_path = PREDECESSOR/"synthetic_01/completion_receipt.json"
    if sha(predecessor_path) != PREDECESSOR_SHA:
        raise ValueError("Pinned V47 predecessor changed")
    predecessor = json.loads(predecessor_path.read_text())
    if predecessor.get("completed") is not True or predecessor.get("synthetic_only") is not True:
        raise ValueError("Predecessor is not completed synthetic evidence")
    for name,digest in predecessor["files_sha256"].items():
        if name in bound and bound[name] != digest:
            raise ValueError("Conflicting predecessor dependency")
        bound[name] = digest
    bound[str(predecessor_path)] = PREDECESSOR_SHA
    bound[str(PREDECESSOR/"synthetic_independent_audit_01.json")] = FAILURE_SHA
    bound[str(PREDECESSOR/"math_independent_audit_01.json")] = MATH_SHA
    require_hashes(bound)
    failure = json.loads((PREDECESSOR/"synthetic_independent_audit_01.json").read_text())
    if (failure.get("passed") is not False or failure.get("completed") is not False
            or failure.get("synthetic_completion_receipt_sha256") != PREDECESSOR_SHA):
        raise ValueError("Original failed assessment not preserved")
    return bound


def collect_cases():
    result = []
    for c in causal_cases():
        result.append(dict(case_id="v46_baseline_"+c["case_id"], group="v46_baseline",
            reference_case_id=c["case_id"], adapter_inputs={k:c[k] for k in KEYS},
            generator_truth=None, provenance={"forecast_origin":"unchanged_legacy_caller_supplied_assumption"}))
    for group,cases in (("v46_stress",stress_cases()), ("guard",guard_cases())):
        for c in cases:
            result.append(dict(case_id=group+"_"+c["case_id"], group=group,
                reference_case_id=c["case_id"], adapter_inputs=c["adapter_inputs"],
                generator_truth=c["generator_truth"], provenance=c["provenance"]))
    if Counter(c["group"] for c in result) != {"v46_baseline":6,"v46_stress":28,"guard":20}:
        raise ValueError("All 54 predeclared cases required")
    if len({c["case_id"] for c in result}) != 54:
        raise ValueError("Duplicate case identity")
    return result


def input_material(cases):
    arrays, metadata = {}, []
    for c in cases:
        a = c["adapter_inputs"]
        if set(a) != set(KEYS):
            raise ValueError("Only observation/geometry keys may reach adapter")
        arrays[c["case_id"]] = dict(current129=np.asarray(a["current129"]),
            history129=np.asarray(a["history129"]),
            prior_centers_xy=np.asarray([[np.nan,np.nan] if p is None else p for p in a["prior_centers_xy"]]),
            predicted_offset_xy=np.asarray(a["predicted_offset_xy"]))
        metadata.append(dict({k:c[k] for k in c if k != "adapter_inputs"}, polarity=a["polarity"]))
    return arrays,metadata


def require_memory(cases, hashes, meta_hash):
    arrays,metadata = input_material(cases)
    if memory_hashes(arrays) != hashes or jhash(metadata) != meta_hash:
        raise ValueError("In-memory arrays/geometry/truth changed after freeze")


def require_baseline_inputs(cases, arrays, bindings):
    for c in cases:
        if c["group"] == "guard":
            continue
        prefix = "baseline_" if c["group"] == "v46_baseline" else "stress_"
        path = BASELINE/"inputs"/(prefix+c["reference_case_id"]+".npz")
        if str(path) not in bindings or sha(path) != bindings[str(path)]:
            raise ValueError("Baseline input not bound")
        with np.load(path, allow_pickle=False) as archive:
            values = arrays[c["case_id"]]
            if set(archive.files) != set(values):
                raise ValueError("Baseline input members changed")
            if memory_hashes({"input":dict(archive)}) != memory_hashes({"input":values}):
                raise ValueError("Baseline input arrays changed")


def require_predecessor_inputs(cases, arrays, metadata):
    previous_meta = json.loads((PREDECESSOR/"synthetic_01/input_manifest.json").read_text())
    previous_by_id = {c["case_id"]:c for c in previous_meta}
    if set(previous_by_id) != set(arrays):
        raise ValueError("V47 case membership changed")
    for c,meta in zip(cases,metadata):
        with np.load(PREDECESSOR/"synthetic_01/inputs"/(c["case_id"]+".npz"), allow_pickle=False) as old:
            if set(old.files) != set(ARRAY_KEYS):
                raise ValueError("Predecessor archive members changed")
            for key in ARRAY_KEYS:
                if c["case_id"] == REPAIRED_CASE and key == "current129":
                    continue
                if memory_hashes({"input":{key:old[key]}}) != memory_hashes({"input":{key:arrays[c["case_id"]][key]}}):
                    raise ValueError("Unchanged predecessor array differs")
        if c["case_id"] != REPAIRED_CASE and meta != previous_by_id[c["case_id"]]:
            raise ValueError("Unchanged predecessor metadata differs")


def evidence_counts(records, arm):
    counts = Counter(states=0,available=0,unavailable=0,positive=0,negative=0,unresolved=0)
    reasons = Counter()
    for r in records:
        value = r["arms"][arm]["raw_adapter_result"]
        counts["states"] += 1
        if not value["available"]:
            counts["unavailable"] += 1
            reasons.update(value["reasons"])
        else:
            counts["available"] += 1
            counts[value["numerical_contrast"]["coefficient_sign"]] += 1
    return dict(counts=dict(counts),unavailable_reasons=dict(reasons))


def validate_record(record):
    if set(record["arms"]) != set(ARMS)|{ARM}:
        raise ValueError("All five distinct calculations required")
    old = {"case_id":record["case_id"],"arms":{a:record["arms"][a] for a in ARMS}}
    check_nominal_and_physical([old])
    baseline = record["arms"]["presence_box_bounds"]["raw_adapter_result"]
    wrapped = record["arms"][ARM]
    validate_evidence(wrapped)
    new = wrapped["raw_adapter_result"]
    for field in ("learned_design_sha256","common_support_sha256","common_support_count",
                  "components","component_bounds","uncertainty_excludes"):
        if new[field] != baseline[field]:
            raise ValueError("New arm changed prior design/support: "+field)
    context = dict(new["prior_context"])
    if context.pop("current_values_used_for") != CURRENT_USE_DESCRIPTION:
        raise ValueError("New current-data use is not disclosed")
    old_context = dict(baseline["prior_context"])
    old_use = old_context.pop("current_values_used_for")
    if context != old_context or wrapped["current_use_metadata_override"] != {
            "path":"raw_adapter_result.prior_context.current_values_used_for",
            "from":old_use,"to":CURRENT_USE_DESCRIPTION}:
        raise ValueError("Prior context changed beyond declared current-use metadata")
    for field in ("conditional_on","ambiguity_reasons"):
        if new[field][:len(baseline[field])] != baseline[field]:
            raise ValueError("Original assumptions/ambiguity removed")
    if not wrapped["uncertainty_excludes_guard_contamination_and_spatial_transfer_failure"]:
        raise ValueError("Guard validity limitation omitted")
    if wrapped["old_unrestricted_background_estimand_preserved"]:
        raise ValueError("Changed estimand mislabeled as unchanged")


def summarize(records):
    if len(records) != 54:
        raise ValueError("Incomplete scientific result")
    for r in records:
        validate_record(r)
    baseline = {}
    for group,file in (("v46_baseline","baseline_results.json"),("v46_stress","stress_results.json")):
        baseline[group] = {r["case_id"]:r["arms"] for r in json.loads((BASELINE/file).read_text())}
    for r in records:
        if r["group"] in baseline and {a:r["arms"][a] for a in ARMS} != baseline[r["group"]][r["reference_case_id"]]:
            raise ValueError("V46 four-arm baseline failed exact reproduction")
    previous = {r["case_id"]:r for r in json.loads((PREDECESSOR/"synthetic_01/results.json").read_text())}
    for r in records:
        if r["case_id"] != REPAIRED_CASE and r != previous[r["case_id"]]:
            raise ValueError("One of 53 unchanged V47 five-arm results changed")
    groups = {g:[r for r in records if r["group"]==g] for g in ("v46_baseline","v46_stress","guard")}
    return dict(completed=True,synthetic_only=True,cases=54,calculations=5,
        counts_by_group={g:{a:evidence_counts(rows,a) for a in (*ARMS,ARM)} for g,rows in groups.items()},
        unchanged_53_v47_inputs_metadata_and_five_outputs_exact=True,
        numerical_method_unchanged_from_v47=True,
        original_failed_v47_run_preserved=True,
        v46_inputs_and_four_arm_results_exact=True,unchanged_prior_design_support_and_v45_component_bounds=True,
        all_unknowns_retained=True,all_motion_and_physical_class_unknown=True,
        guard_is_same_frame_not_prior_only=True,guard_consistency_is_only_necessary_relaxation=True,
        guard_to_core_transfer_not_certified=True,physical_identity_not_certified=True,
        no_positive_fraction_accuracy_threshold=True,production_changed=False,real_packet_data_read=False)


def run(output):
    output = Path(output).resolve()
    if output.parent != OUTPUT_ROOT:
        raise ValueError("Output must be a fresh direct V48 child")
    if output.exists():
        raise FileExistsError("Never reuse frozen scientific output")
    bound = baseline_bindings()
    bound.update({str(p):sha(p) for p in source_dependencies()})
    cases = collect_cases(); arrays,metadata = input_material(cases)
    require_baseline_inputs(cases,arrays,bound)
    require_predecessor_inputs(cases,arrays,metadata)
    families = {c["case_id"]:c["family"] for c in scenario_manifest()["scenarios"]}
    witness_cases = [dict(case_id=c["reference_case_id"], family=families[c["reference_case_id"]], adapter_inputs=c["adapter_inputs"],
        generator_truth=c["generator_truth"], provenance=c["provenance"])
        for c in cases if c["group"]=="guard"]
    contract = pre_score_contract(witness_cases)
    if contract.get("passed") is not True or contract.get("issues") != []:
        raise ValueError("Rendering contract failed before any scores")
    hashes,meta_hash = memory_hashes(arrays),jhash(metadata)
    for values in arrays.values():
        for value in values.values():
            value.flags.writeable = False
    output.mkdir(parents=True);(output/"inputs").mkdir()
    write_json(output/"freeze.json",dict(created_at_utc=now(),files_sha256=bound,
        case_ids=[c["case_id"] for c in cases],guard_scenario_manifest=scenario_manifest(),
        input_memory_sha256=hashes,input_metadata_sha256=meta_hash,
        old_arms={a:list(v) for a,v in ARMS.items()},new_arm=ARM,baseline_receipt_sha256=BASELINE_SHA,
        predecessor_receipt_sha256=PREDECESSOR_SHA,predecessor_failure_audit_sha256=FAILURE_SHA,
        math_audit_sha256=MATH_SHA,rendering_contract_sha256=jhash(contract),
        runtime=dict(python=platform.python_version(),numpy=np.__version__),synthetic_only=True))
    write_json(output/"input_manifest.json",metadata)
    saved = {}
    for name,values in arrays.items():
        path = output/"inputs"/(name+".npz")
        np.savez_compressed(path,**values); saved[str(path)]=sha(path)
    write_json(output/"inputs_complete.json",dict(created_at_utc=now(),files_sha256=saved,scores_started=False))
    require_hashes(bound);require_hashes(saved);require_memory(cases,hashes,meta_hash)
    if contract != pre_score_contract(witness_cases):
        raise ValueError("Rendering witness changed after archive freeze")
    write_json(output/"rendering_contract.json",dict(created_at_utc=now(),**contract))
    write_json(output/"score_start.json",dict(created_at_utc=now(),freeze_sha256=sha(output/"freeze.json"),
        inputs_complete_sha256=sha(output/"inputs_complete.json"),input_manifest_sha256=sha(output/"input_manifest.json"),
        rendering_contract_sha256=sha(output/"rendering_contract.json")))
    records=[]
    for index,c in enumerate(cases):
        args = c["adapter_inputs"]
        record={k:c[k] for k in c if k != "adapter_inputs"}
        record["arms"] = {a:evaluate_causal_probe(**args,arm=a) for a in ARMS}
        record["arms"][ARM] = evaluate_probe(**args)
        validate_record(record);records.append(record)
        if (index+1)%6 == 0:
            print(f"V48 synthetic {index+1}/54; all five calculations retained",flush=True)
    require_memory(cases,hashes,meta_hash)
    summary=summarize(records)
    write_json(output/"results.json",records);write_json(output/"summary.json",summary)
    require_hashes(bound);require_hashes(saved)
    bound.update(saved);bound.update({str(p):sha(p) for p in output.glob("*.json")})
    write_json(output/"completion_receipt.json",dict(completed=True,created_at_utc=now(),files_sha256=bound,
        synthetic_only=True,production_changed=False,real_packet_data_read=False))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        parser.error("--execute required")
    print(json.dumps(run(args.output),indent=2))
