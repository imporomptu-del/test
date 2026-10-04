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
from accuracy_v47_synthetic_cases import build_cases as guard_cases, scenario_manifest
from accuracy_v47_probe import ARM, CURRENT_USE_DESCRIPTION, evaluate_probe
from run_accuracy_v44_synthetic import causal_cases
from run_accuracy_v46_stress import memory_hashes, check_nominal_and_physical
from run_accuracy_v46_shadow import validate_evidence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v47_20260926"
BASELINE = ROOT/"results/tiny_target/accuracy_v46_20260925/synthetic_01"
BASELINE_SHA = "252954f7c0a28b390f2f91c952faede59878cedcd6a9a63d277150ad7da31b51"
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
    return sorted(set([ROOT/"docs/accuracy_v47_plan.md"]
        +list((ROOT/"scripts").glob("*accuracy_v47*.py"))
        +list((ROOT/"tests/unit").glob("test_accuracy_v47*.py"))))


def baseline_bindings():
    path = BASELINE/"completion_receipt.json"
    if sha(path) != BASELINE_SHA:
        raise ValueError("Pinned V46 synthetic receipt changed")
    receipt = json.loads(path.read_text())
    if not receipt["completed"] or not receipt["synthetic_only"] or receipt["real_packet_data_read"]:
        raise ValueError("Baseline is not completed synthetic-only evidence")
    require_hashes(receipt["files_sha256"])
    return dict(receipt["files_sha256"], **{str(path):BASELINE_SHA})


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
    groups = {g:[r for r in records if r["group"]==g] for g in ("v46_baseline","v46_stress","guard")}
    return dict(completed=True,synthetic_only=True,cases=54,calculations=5,
        counts_by_group={g:{a:evidence_counts(rows,a) for a in (*ARMS,ARM)} for g,rows in groups.items()},
        v46_inputs_and_four_arm_results_exact=True,unchanged_prior_design_support_and_v45_component_bounds=True,
        all_unknowns_retained=True,all_motion_and_physical_class_unknown=True,
        guard_is_same_frame_not_prior_only=True,guard_consistency_is_only_necessary_relaxation=True,
        guard_to_core_transfer_not_certified=True,physical_identity_not_certified=True,
        no_positive_fraction_accuracy_threshold=True,production_changed=False,real_packet_data_read=False)


def run(output):
    output = Path(output).resolve()
    if output.parent != OUTPUT_ROOT:
        raise ValueError("Output must be a fresh direct V47 child")
    if output.exists():
        raise FileExistsError("Never reuse frozen scientific output")
    bound = baseline_bindings()
    bound.update({str(p):sha(p) for p in source_dependencies()})
    cases = collect_cases(); arrays,metadata = input_material(cases)
    require_baseline_inputs(cases,arrays,bound)
    hashes,meta_hash = memory_hashes(arrays),jhash(metadata)
    for values in arrays.values():
        for value in values.values():
            value.flags.writeable = False
    output.mkdir(parents=True);(output/"inputs").mkdir()
    write_json(output/"freeze.json",dict(created_at_utc=now(),files_sha256=bound,
        case_ids=[c["case_id"] for c in cases],guard_scenario_manifest=scenario_manifest(),
        input_memory_sha256=hashes,input_metadata_sha256=meta_hash,
        old_arms={a:list(v) for a,v in ARMS.items()},new_arm=ARM,baseline_receipt_sha256=BASELINE_SHA,
        runtime=dict(python=platform.python_version(),numpy=np.__version__),synthetic_only=True))
    write_json(output/"input_manifest.json",metadata)
    saved = {}
    for name,values in arrays.items():
        path = output/"inputs"/(name+".npz")
        np.savez_compressed(path,**values); saved[str(path)]=sha(path)
    write_json(output/"inputs_complete.json",dict(created_at_utc=now(),files_sha256=saved,scores_started=False))
    require_hashes(bound);require_hashes(saved);require_memory(cases,hashes,meta_hash)
    write_json(output/"score_start.json",dict(created_at_utc=now(),freeze_sha256=sha(output/"freeze.json"),
        inputs_complete_sha256=sha(output/"inputs_complete.json"),input_manifest_sha256=sha(output/"input_manifest.json")))
    records=[]
    for index,c in enumerate(cases):
        args = c["adapter_inputs"]
        record={k:c[k] for k in c if k != "adapter_inputs"}
        record["arms"] = {a:evaluate_causal_probe(**args,arm=a) for a in ARMS}
        record["arms"][ARM] = evaluate_probe(**args)
        validate_record(record);records.append(record)
        if (index+1)%6 == 0:
            print(f"V47 synthetic {index+1}/54; all five calculations retained",flush=True)
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
