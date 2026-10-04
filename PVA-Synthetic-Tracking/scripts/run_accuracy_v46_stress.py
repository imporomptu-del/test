"""Frozen synthetic geometry/shape stress with all four immutable V45 arms."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from accuracy_v45_causal_probe import ARMS, evaluate_causal_probe
from accuracy_v46_synthetic_cases import build_cases, scenario_manifest
from run_accuracy_v44_synthetic import causal_cases


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v46_20260925"
BASELINE = ROOT/"results/tiny_target/accuracy_v45_20260925/synthetic_01"
BASELINE_RECEIPT_SHA256 = "a01c91b7b223986e4bb90f6eb6a1237ff6799c6d7d057a78e44a632b1ab9a5de"
ARRAY_KEYS = ("current129", "history129", "prior_centers_xy", "predicted_offset_xy")
ADAPTER_KEYS = (*ARRAY_KEYS, "polarity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_hashes(bindings):
    changed = [p for p,h in bindings.items() if sha(p) != h]
    if changed:
        raise ValueError("Frozen inputs or code changed: " + repr(changed))


def baseline_bindings():
    receipt_path = BASELINE/"completion_receipt.json"
    if sha(receipt_path) != BASELINE_RECEIPT_SHA256:
        raise ValueError("V45 pinned baseline receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if receipt["completed"] is not True:
        raise ValueError("V45 baseline incomplete")
    require_hashes(receipt["files_sha256"])
    return dict(receipt["files_sha256"], **{str(receipt_path):sha(receipt_path)})


def dependencies():
    return sorted(set([ROOT/"docs/accuracy_v46_plan.md"]
        + list((ROOT/"scripts").glob("*accuracy_v46*.py"))
        + list((ROOT/"tests/unit").glob("test_accuracy_v46*.py"))))


def inputs_and_metadata(baseline, stress):
    arrays, metadata = {}, {"baseline":[], "stress":[]}
    for kind, cases in (("baseline",baseline), ("stress",stress)):
        for c in cases:
            a = {k:c[k] for k in ADAPTER_KEYS} if kind == "baseline" else c["adapter_inputs"]
            if set(a) != set(ADAPTER_KEYS):
                raise ValueError("Only declared array/geometry inputs may reach the adapter")
            name = kind+"_"+c["case_id"]
            if name in arrays:
                raise ValueError("Duplicate input archive")
            arrays[name] = dict(current129=np.asarray(a["current129"]),
                history129=np.asarray(a["history129"]),
                prior_centers_xy=np.asarray([[np.nan,np.nan] if p is None else p for p in a["prior_centers_xy"]]),
                predicted_offset_xy=np.asarray(a["predicted_offset_xy"]))
            item = dict(case_id=c["case_id"],polarity=a["polarity"])
            if kind == "baseline":
                item["forecast_origin"] = "unchanged_legacy_caller_supplied_assumption"
            else:
                item.update({key:c[key] for key in ("family","generator_truth","provenance")})
            metadata[kind].append(item)
    return arrays, metadata


def memory_hashes(arrays):
    result = {}
    for name, values in arrays.items():
        for key,value in values.items():
            value = np.asarray(value)
            h = hashlib.sha256(str((value.shape,value.dtype.str)).encode("ascii"))
            h.update(value.tobytes(order="C"))
            result[name+":"+key] = h.hexdigest()
    return result


def require_baseline_arrays(arrays, bindings):
    for name, values in arrays.items():
        if not name.startswith("baseline_"):
            continue
        path = BASELINE/"inputs"/("causal_"+name.removeprefix("baseline_")+".npz")
        if str(path) not in bindings or sha(path) != bindings[str(path)]:
            raise ValueError("Baseline input is not bound by pinned V45 receipt")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(values):
                raise ValueError("Baseline array keys changed")
            for key, value in values.items():
                if (archive[key].dtype != value.dtype
                        or not np.array_equal(archive[key], value, equal_nan=True)):
                    raise ValueError("Baseline array changed: "+name+":"+key)


def require_memory(baseline, stress, hashes, meta_hash):
    arrays, metadata = inputs_and_metadata(baseline, stress)
    if hashes != memory_hashes(arrays) or meta_hash != json_hash(metadata):
        raise ValueError("In-memory inputs/geometry/truth changed after freeze")


def evidence_counts(records, arm):
    count = Counter(states=0, unavailable=0, available=0, positive=0, negative=0, unresolved=0)
    reasons = Counter()
    for record in records:
        value = record["arms"][arm]["raw_adapter_result"]
        count["states"] += 1
        if not value["available"]:
            count["unavailable"] += 1
            reasons.update(value["reasons"])
        else:
            count["available"] += 1
            count[value["numerical_contrast"]["coefficient_sign"]] += 1
    return dict(counts=dict(count),unavailable_reasons=dict(reasons))


def check_nominal_and_physical(records):
    fields = ("learned_design_sha256","common_support_sha256","common_support_count",
              "components","prior_context","ambiguity_reasons","conditional_on","uncertainty_excludes")
    for record in records:
        if set(record["arms"]) != set(ARMS):
            raise ValueError("Missing mathematical arm")
        baseline = record["arms"]["amplitude_old_bounds"]["raw_adapter_result"]
        for arm,wrapped in record["arms"].items():
            result = wrapped["raw_adapter_result"]
            for value in (wrapped,result):
                if value["motion_status"] != "unknown" or value["physical_class"] != "unknown" or value["is_motion_or_classification_gate"]:
                    raise ValueError("Numerical evidence became a physical decision")
            if any(result[k] != baseline[k] for k in fields):
                raise ValueError("Arm changed nominal design/support/ambiguity")
            numerical = result["numerical_contrast"]
            if numerical is not None:
                if numerical["motion_status"] != "unknown" or numerical["physical_class"] != "unknown":
                    raise ValueError("Numerical solver promoted physical class")
                if numerical["available"]:
                    low,high = numerical["interval"]
                    sign = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
                    if sign != numerical["coefficient_sign"]:
                        raise ValueError("Interval/sign inconsistent")
    json.dumps(records,allow_nan=False)


def summarize(baseline, stress, old):
    if baseline != old:
        raise ValueError("V45 six-case baseline failed exact reproduction")
    if len(baseline) != 6 or len(stress) != 28:
        raise ValueError("Incomplete frozen 34-case matrix")
    check_nominal_and_physical(baseline+stress)
    groups = {}
    for record in stress:
        groups.setdefault(record["family"],[]).append(record)
    prior_groups = {}
    for record in stress:
        group = record["provenance"]["prior_equivalence_group"]
        prior_groups.setdefault(group,[]).append(record)
    for records in prior_groups.values():
        for arm in ARMS:
            first=records[0]["arms"][arm]["raw_adapter_result"]
            for record in records[1:]:
                value=record["arms"][arm]["raw_adapter_result"]
                for key in ("learned_design_sha256","common_support_sha256","common_support_count","components"):
                    if value[key] != first[key]:
                        raise ValueError("Current-only variation changed learned design/support")
    twins=[r for r in stress if r["case_id"].startswith("observational_twin_")]
    if len(twins) != 2 or twins[0]["arms"] != twins[1]["arms"]:
        raise ValueError("Observational twins differ numerically")
    return dict(completed=True,synthetic_only=True,baseline_cases=6,stress_cases=28,arms=4,
        baseline_counts={a:evidence_counts(baseline,a) for a in ARMS},
        stress_counts={a:evidence_counts(stress,a) for a in ARMS},
        stress_family_counts={g:{a:evidence_counts(rows,a) for a in ARMS} for g,rows in groups.items()},
        invariants=dict(v45_baseline_exact=True,all_cases_and_arms_retained=True,
            nominal_design_support_and_ambiguity_unchanged_across_arms=True,
            current_only_groups_preserve_learned_design=True,observational_twins_identical=True,
            all_motion_and_physical_class_unknown=True),
        independent_audit_required_before_real_packets=True,
        geometry_shape_association_mismatch_not_covered_by_declared_dn_bound=True,
        stress_numerical_signs_are_not_recall_or_physical_class=True,
        production_changed=False,real_packet_data_read=False)


def run(output):
    output=Path(output).resolve()
    if output.parent != OUTPUT_ROOT:
        raise ValueError("Output must be a fresh direct V46 child directory")
    if output.exists():
        raise FileExistsError("Frozen output cannot be reused")
    bound=baseline_bindings()
    bound.update({str(p):sha(p) for p in dependencies()})
    baseline,stress=causal_cases(),build_cases()
    manifest=scenario_manifest()
    if (len(baseline)!=6 or len(stress)!=28 or len({c["case_id"] for c in stress})!=28
            or [c["case_id"] for c in stress] != [c["case_id"] for c in manifest["scenarios"]]):
        raise ValueError("Wrong predeclared stress case membership")
    arrays,metadata=inputs_and_metadata(baseline,stress)
    if len(arrays)!=34:
        raise ValueError("All34 inputs required before scoring")
    require_baseline_arrays(arrays, bound)
    hashes=memory_hashes(arrays); meta_hash=json_hash(metadata)
    for values in arrays.values():
        for value in values.values():
            value.flags.writeable=False
    output.mkdir(parents=True); (output/"inputs").mkdir()
    write_json(output/"freeze.json",dict(created_at_utc=now(),dependency_sha256=bound,
        scenario_manifest=manifest,input_memory_sha256=hashes,input_metadata_sha256=meta_hash,
        arms={a:list(v) for a,v in ARMS.items()},v45_receipt_sha256=BASELINE_RECEIPT_SHA256,
        synthetic_only=True,runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write_json(output/"input_manifest.json",metadata)
    saved={}
    for name,values in arrays.items():
        path=output/"inputs"/(name+".npz")
        np.savez_compressed(path,**values);saved[str(path)]=sha(path)
    write_json(output/"inputs_complete.json",dict(created_at_utc=now(),inputs_sha256=saved,scores_started=False))
    require_hashes(bound);require_hashes(saved)
    require_memory(baseline, stress, hashes, meta_hash)
    write_json(output/"score_start.json",dict(created_at_utc=now(),freeze_sha256=sha(output/"freeze.json"),
        inputs_complete_sha256=sha(output/"inputs_complete.json"),input_manifest_sha256=sha(output/"input_manifest.json")))
    baseline_results=[]
    for c in baseline:
        args={k:c[k] for k in ADAPTER_KEYS}
        baseline_results.append(dict(case_id=c["case_id"],arms={a:evaluate_causal_probe(**args,arm=a) for a in ARMS}))
    stress_results=[]
    for index,c in enumerate(stress):
        args=c["adapter_inputs"]
        record={key:c[key] for key in ("case_id","family","generator_truth","provenance")}
        record["arms"]={a:evaluate_causal_probe(**args,arm=a) for a in ARMS}
        stress_results.append(record)
        if index%7==6:
            print(f"Scored synthetic stress {index+1}/28; all four frozen arms",flush=True)
    require_memory(baseline, stress, hashes, meta_hash)
    old=json.loads((BASELINE/"causal_results.json").read_text())
    summary=summarize(baseline_results,stress_results,old)
    write_json(output/"baseline_results.json",baseline_results)
    write_json(output/"stress_results.json",stress_results)
    write_json(output/"summary.json",summary)
    require_hashes(bound);require_hashes(saved)
    bound=dict(bound,**saved)
    bound.update({str(p):sha(p) for p in output.glob("*.json")})
    write_json(output/"completion_receipt.json",dict(completed=True,created_at_utc=now(),files_sha256=bound,
        synthetic_only=True,production_changed=False,real_packet_data_read=False))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        parser.error("--execute required to freeze and run")
    print(json.dumps(run(args.output),indent=2))
