"""Conditional V47 diagnostic on the unchanged V46 two-clip packet ledger."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

import run_accuracy_v46_shadow as old
from accuracy_v47_probe import ARM, evaluate_probe
from run_accuracy_v47_synthetic import (source_dependencies, sha, now, write_json,
    require_hashes, validate_record)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT/"results/tiny_target/accuracy_v47_20260926"
V46 = ROOT/"results/tiny_target/accuracy_v46_20260925/shadow_01"
V46_RECEIPT_SHA = "ebe3fdaca3a69b3ea48151f40ad407cf117021d9eb5d1b8ad49e4e4e06711b96"
ALL_ARMS = (*old.ARMS,ARM)


class Progress(old.Progress):
    def _run(self):
        while not self.stop.wait(30):
            print("V47 shadow: "+self.message,flush=True)


def read_json(path):
    return old.read_json(path)


def require_readiness():
    path=old._regular_exact(OUTPUT_ROOT/"shadow_readiness_01.json")
    record=read_json(path)
    if (any(record.get(k) is not True for k in ("completed","audited","diagnostic_shadow_allowed"))
            or record.get("production_changed") is not False
            or record.get("permitted_clips") != ["0029","0126"]):
        raise ValueError("V47 synthetic safety readiness not authorized")
    bound={str(path):sha(path)}
    paths={"synthetic_receipt":OUTPUT_ROOT/"synthetic_01/completion_receipt.json",
           "math_audit":OUTPUT_ROOT/"math_independent_audit_01.json",
           "synthetic_audit":OUTPUT_ROOT/"synthetic_independent_audit_01.json"}
    for key,p in paths.items():
        if record.get(key+"_path") != str(p):
            raise ValueError("Unexpected readiness evidence path")
        old._regular_exact(p)
        bound[str(p)]=record[key+"_sha256"]
    require_hashes(bound)
    receipt=read_json(paths["synthetic_receipt"])
    if receipt.get("completed") is not True or receipt.get("synthetic_only") is not True:
        raise ValueError("Synthetic experiment incomplete")
    for key in ("math_audit","synthetic_audit"):
        report=read_json(paths[key])
        if report.get("completed") is not True or report.get("passed") is not True or report.get("issues") != []:
            raise ValueError("Independent audit did not pass")
    if read_json(paths["synthetic_audit"]).get("synthetic_completion_receipt_sha256") != bound[str(paths["synthetic_receipt"])]:
        raise ValueError("Synthetic audit does not bind the approved run")
    dependencies=receipt.get("files_sha256")
    if not isinstance(dependencies,dict) or not dependencies:
        raise ValueError("Synthetic receipt has no bound evidence")
    audit_bindings=read_json(paths["math_audit"]).get("audit_files_sha256")
    if not isinstance(audit_bindings,dict) or not audit_bindings:
        raise ValueError("Math audit has no bound implementation")
    for name,digest in audit_bindings.items():
        if name not in dependencies or dependencies[name]!=digest:
            raise ValueError("Math audit did not check the frozen implementation")
    synthetic_dirs=[ROOT/f"results/tiny_target/accuracy_v{v}_{d}/synthetic_01" for v,d in
                    ((44,"20260925"),(45,"20260925"),(46,"20260925"),(47,"20260926"))]
    for name,digest in dependencies.items():
        p=Path(name)
        permitted=any(folder in p.parents for folder in synthetic_dirs) or any(
            p.parent==ROOT/folder and p.suffix==suffix
            for folder,suffix in (("scripts",".py"),("tests/unit",".py"),("docs",".md")))
        if not permitted:
            raise ValueError("Synthetic evidence path outside allowlist")
        old._regular_exact(p);old.merge_bindings(bound,{name:digest})
    require_hashes(bound)
    return bound


def load_baseline():
    receipt_path=old._regular_exact(V46/"completion_receipt.json")
    if sha(receipt_path)!=V46_RECEIPT_SHA:
        raise ValueError("V46 shadow receipt changed")
    receipt=read_json(receipt_path)
    if receipt.get("completed") is not True:
        raise ValueError("V46 real baseline incomplete")
    bound={str(receipt_path):V46_RECEIPT_SHA}
    names=("states.jsonl","selected_ledger.json","reference_evidence.json","summary.json")
    for name in names:
        p=old._regular_exact(V46/name);bound[str(p)]=receipt["files_sha256"][str(p)]
    require_hashes(bound)
    states,samples,panels,compact,cache_receipt=old.load_scope_metadata()
    old.merge_bindings(bound,compact)
    ledger=read_json(V46/"selected_ledger.json")
    if ledger != dict(states=states,reference_samples=samples,inherited_reference_panels=panels):
        raise ValueError("V46 selected ledger differs from original compact metadata")
    rows=[json.loads(line) for line in (V46/"states.jsonl").read_text().splitlines()]
    if len(rows)!=len(states) or len({old.state_key(r) for r in rows})!=len(rows):
        raise ValueError("V46 baseline denominator/order invalid")
    for r,s in zip(rows,states):
        if {k:r[k] for k in s} != s or set(r["arms"]) != set(old.ARMS):
            raise ValueError("V46 original metadata or four arms differ")
        for value in r["arms"].values():
            if s["archive"] is None:
                if value is not None:raise ValueError("V46 unknown fabricated evidence")
            else:old.validate_evidence(value)
    references=read_json(V46/"reference_evidence.json")
    if references != old.reference_report(samples,panels,rows):
        raise ValueError("V46 references no longer preserve originals")
    return states,rows,references,read_json(V46/"summary.json"),bound,cache_receipt,receipt["files_sha256"]


def real_origin_copy(value):
    result=deepcopy(value)
    if result["synthetic_only"] is not True or result["raw_adapter_result"]["synthetic_only"] is not True:
        raise ValueError("Unexpected experiment origin schema")
    result["synthetic_only"]=False;result["raw_adapter_result"]["synthetic_only"]=False
    result.update(input_origin="unchanged_v46_8bit_avi_derived_cached_packet",
        origin_only_overrides=[{"path":"synthetic_only","from":True,"to":False},
                               {"path":"raw_adapter_result.synthetic_only","from":True,"to":False}],
        mathematical_implementation="accuracy_v47_probe.evaluate_probe")
    return result


def evaluate_packet(packet,state):
    before=old.packet_fingerprint(packet)
    centers=[p.tolist() if np.isfinite(p).all() else None for p in packet["prior_centers_xy"]]
    value=evaluate_probe(packet["current129"],packet["history129"],centers,
                         packet["predicted_offset_xy"],state["track_id"].split(":")[0])
    if old.packet_fingerprint(packet)!=before:
        raise ValueError("New method mutated a frozen packet")
    value=real_origin_copy(value);old.validate_evidence(value)
    return value


def add_references(original,records):
    result=deepcopy(original);lookup={old.state_key(r):r for r in records}
    for sample in result["samples"]:
        for alternative in sample["measured_alternatives"]:
            row=lookup[tuple(alternative["state_key"])]
            alternative["arm_evidence"][ARM]=old.evidence_summary(row["arms"][ARM])
        identity=sample["original_strict_assigned_identity"]
        if identity is not None:
            selected=[a for a in sample["measured_alternatives"] if a["identity"]==identity]
            if len(selected)!=1:raise ValueError("Original strict assignment lost")
            sample["original_strict_assigned_evidence"][ARM]=selected[0]["arm_evidence"][ARM]
    stripped=deepcopy(result)
    for sample in stripped["samples"]:
        for alternative in sample["measured_alternatives"]:
            del alternative["arm_evidence"][ARM]
        if sample["original_strict_assigned_evidence"] is not None:
            # In-memory generated fixtures may alias the matching alternative.
            # JSON-loaded records do not; preserve either representation.
            sample["original_strict_assigned_evidence"].pop(ARM,None)
    if stripped!=original:raise ValueError("Original reference evidence was overwritten")
    result["new_method_guard_and_spatial_transfer_assumptions_not_certified"]=True
    return result


def summarize(records,baseline,old_summary,references):
    if len(records)!=len(baseline):raise ValueError("State denominator changed")
    for r,b in zip(records,baseline):
        stripped=deepcopy(r);del stripped["arms"][ARM]
        if stripped!=b:raise ValueError("Original four-arm result or state was modified")
        if set(r["arms"])!=set(ALL_ARMS):raise ValueError("Missing calculation")
    counts={}
    for arm in ALL_ARMS:
        c=Counter(states=len(records),geometry_available=0,available=0,unavailable=0,positive=0,negative=0,unresolved=0)
        for row in records:
            c["geometry_available"]+=int(row["archive"] is not None)
            value=old.evidence_summary(row["arms"][arm])
            if value["available"]:
                c["available"]+=1;c[value["coefficient_sign"]]+=1
            else:c["unavailable"]+=1
        counts[arm]=dict(c)
    if {a:counts[a] for a in old.ARMS}!=old_summary["arms"]:
        raise ValueError("Old four-arm aggregate changed")
    return dict(completed=True,states=len(records),eligible_packets=sum(r["archive"] is not None for r in records),
        reference_samples=len(references["samples"]),arms=counts,production_changed=False,synthetic_only=False,
        old_four_arm_results_and_original_assignments_exact=True,original_misses_and_alternatives_preserved=True,
        detections_added=0,detections_removed=0,no_new_labels=True,no_filter_applied=True,
        prior_component_design_support_and_bounds_unchanged=True,
        new_quantity_is_conditional_bounded_gain_not_old_unrestricted_fit=True,
        current_outer_guard_is_not_prior_only=True,guard_validity_and_core_transfer_not_certified=True,
        current_global_transform_uses_current_whole_frame=True,
        no_airborne_accuracy_false_alarm_or_generalization_claim=True,
        packet_only_no_media_decode_or_journal_read=True)


def run(output,*,execute=False):
    if not execute:raise ValueError("Explicit execution and audited readiness required")
    output=Path(output).resolve()
    if output.parent!=OUTPUT_ROOT or output.name=="synthetic_01":
        raise ValueError("Fresh dedicated V47 real-output child required")
    if output.exists():raise FileExistsError("Never overwrite a frozen run")
    with Progress() as progress:
        progress.message="V47 synthetic readiness and unchanged V46 ledger"
        bound=require_readiness()
        states,baseline,old_refs,old_summary,metadata,cache_receipt,prior_receipt=load_baseline()
        old.merge_bindings(bound,metadata)
        old.merge_bindings(bound,{str(p):sha(p) for p in source_dependencies()})
        packets={}
        for state in states:
            if state["archive"] is not None:
                p=old.packet_path(state,old.CACHE);digest=state["archive"]["sha256"]
                if cache_receipt.get(str(p))!=digest or prior_receipt.get(str(p))!=digest:
                    raise ValueError("Packet no longer agrees with both frozen receipts")
                packets[str(p)]=digest
        if len(packets)!=sum(v[1] for v in old.EXPECTED_COUNTS.values()):
            raise ValueError("Wrong packet denominator")
        # First real packet access: only after independently audited readiness.
        progress.message="V47 hashing exact allowlisted packets"
        require_hashes(packets);require_hashes(bound)
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"selected_ledger.json",dict(states=states,
            baseline_reference_evidence_sha256=sha(V46/"reference_evidence.json")))
        write_json(output/"freeze.json",dict(created_at_utc=now(),files_sha256=bound,packet_sha256=packets,
            selected_ledger_sha256=sha(output/"selected_ledger.json"),allowed_clips=list(old.CLIPS),
            old_four_arms_reused_exactly=list(old.ARMS),new_arm=ARM,all_packets_hashed_before_scores=True,
            production_changed=False,synthetic_only=False))
        write_json(output/"score_start.json",dict(created_at_utc=now(),freeze_sha256=sha(output/"freeze.json")))
        records=[];current=None
        try:
            with (output/"states.jsonl").open("x") as stream:
                for i,state in enumerate(states):
                    current=old.state_key(state);progress.message=f"V47 state {i+1}/{len(states)}; {current}"
                    row=deepcopy(baseline[i]);row["arms"][ARM]=None
                    if state["archive"] is not None:
                        path=old.packet_path(state,old.CACHE)
                        row["arms"][ARM]=evaluate_packet(old.load_packet(path,state),state)
                        validate_record(dict(case_id=str(current),arms=row["arms"]))
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();records.append(row)
            refs=add_references(old_refs,records);summary=summarize(records,baseline,old_summary,refs)
            write_json(output/"reference_evidence.json",refs);write_json(output/"summary.json",summary)
            require_hashes(bound);require_hashes(packets)
            bound.update(packets)
            for name in ("selected_ledger.json","freeze.json","score_start.json","states.jsonl","reference_evidence.json","summary.json"):
                bound[str(output/name)]=sha(output/name)
            write_json(output/"completion_receipt.json",dict(completed=True,created_at_utc=now(),files_sha256=bound,
                production_changed=False,synthetic_only=False,packet_only_no_media_decode_or_journal_read=True))
        except Exception as exc:
            write_json(output/"failure.json",dict(completed=False,state_key=current,completed_records=len(records),
                exception_type=type(exc).__name__,message=str(exc),partial_records_are_not_complete=True))
            raise
        print(f"V47 shadow complete: {len(records)} unchanged states; {len(packets)} packets",flush=True)
        return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.output,execute=args.execute),indent=2))
