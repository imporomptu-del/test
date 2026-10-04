"""Frozen, no-media four-clip replay with separate stage-specific coverage."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from accuracy_v39_continuity import CausalMeasuredContinuity, ContinuityConfig
from evaluate_accuracy_v36 import ROOT, COUNTS, REFERENCES, read, sha, write, verified_inputs, inside, load_reference
from score_phase20_accuracy import assign_one_to_one

BASE = ROOT/"results/tiny_target/accuracy_v39_20260925"
V38 = ROOT/"results/tiny_target/accuracy_v38_20260925"
V36 = ROOT/"results/tiny_target/accuracy_v36_20260924"
STAGES = ("candidate", "actual_measurement", "baseline_qualified", "with_degraded", "v36_shadow")
AUDIT = V38/"global_multilag_saved_evidence_audit_01.json"
AUDIT_SHA = "ec5f682215f5561939673fe7ad6dd7810128aa0a0cb301624e92bd4c05c0bd5d"
CODE = ("scripts/accuracy_v39_continuity.py", "scripts/evaluate_accuracy_v39_continuity.py",
        "scripts/evaluate_accuracy_v36.py", "scripts/score_phase20_accuracy.py",
        "tiny_target/visible_regression.py", "tests/unit/test_accuracy_v39_continuity.py",
        "tests/unit/test_accuracy_v39_continuity_runner.py", "docs/accuracy_v39_plan.md")


def bind(files, path, expected=None):
    path = str(Path(path).resolve())
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Changed input: "+path)
    if files.setdefault(path,digest) != digest:
        raise ValueError("Input changed during binding")
    return digest


def recheck(files):
    for path,digest in files.items():
        if sha(path) != digest:raise ValueError("Changed bound input: "+path)


def samples_for(cid):
    samples = []
    for kind in ("dense","pilot"):
        labels,packet = read(REFERENCES[kind+"_labels"]),read(REFERENCES[kind+"_packet"])
        windows = {w["id"]:w for w in packet["windows"]}
        for event in labels["positive_windows"]:
            if windows[event["window_id"]]["clip_id"] == cid:
                for sample in event["visible_samples"]:
                    samples.append(dict(kind=kind,window=event["window_id"],frame=sample["frame_index"],
                        xy=sample["xy"],radius=sample["uncertainty_px"]+2.,polarity=event["polarity"]))
    if cid in ("0029","0126"):
        _,events = load_reference(REFERENCES["anchors_"+cid])
        for event in events:
            for sample in event["anchors"]:
                if sample["required"]:
                    samples.append(dict(kind="anchor",window=event["event_id"],frame=sample["frame_index"],
                        xy=sample["xy"],radius=sample["uncertainty_px"]+2.,polarity=event["polarity"]))
    if cid == "0029":
        reference=read(V38/"compact_light_reference_v1.json")
        for sample in reference["frames"]:
            if sample["visibility"] == "visible":
                samples.append(dict(kind="compact_light",window="class_unknown_compact_light_v1",frame=sample["frame_index"],
                    xy=sample["source_xy"],radius=sample["position_uncertainty_radius_px"]+2.,polarity="bright"))
    keys=[(s["kind"],s["window"],s["frame"]) for s in samples]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate reference sample")
    return samples


def match_stages(samples, row, decisions, shadow):
    """One-to-one matching separately per reference family; never mix families."""
    qualified_keys={key for key,value in decisions.items() if value["baseline_qualified"]}
    degraded_keys={key for key,value in decisions.items() if value["renderable"]}
    shadow_keys={(t["segment"],t["track_id"]) for t in shadow["tracks"] if t["accepted"]}
    if not qualified_keys <= degraded_keys or not shadow_keys <= qualified_keys:
        raise ValueError("A candidate lost baseline output or a shadow decision added output")
    stages = {name:[] for name in STAGES}
    stages["candidate"] = [dict(id=f'candidate:{i}',xy=c["source_xy"],polarity=c["polarity"])
                           for i,c in enumerate(row["candidates"])]
    for track in row["tracks"]:
        if not track["measured"]:continue
        key=(track["segment"],track["track_id"])
        point=dict(id=f'{key[0]}/{key[1]}',xy=track["measurement_source_xy"],polarity=key[1].split(":")[0])
        stages["actual_measurement"].append(point)
        for name,keys in (("baseline_qualified",qualified_keys),("with_degraded",degraded_keys),("v36_shadow",shadow_keys)):
            if key in keys:stages[name].append(point)
    evidence=[]
    for kind in sorted({s["kind"] for s in samples}):
        current=[s for s in samples if s["kind"]==kind]
        matching={name:assign_one_to_one(current,observations) for name,observations in stages.items()}
        for i,sample in enumerate(current):
            record={**sample,"stages":{}}
            for name,observations in stages.items():
                matches,neighbors=matching[name]
                assigned=observations[matches[i]] if i in matches else None
                record["stages"][name]=dict(hit=assigned is not None,
                    assigned_id=assigned["id"] if assigned else None,
                    all_gated_ids=[observations[j]["id"] for j in neighbors[i]],
                    ambiguous=len(neighbors[i])>1 or any(sum(j in n for n in neighbors)>1 for j in neighbors[i]),
                    distance_px=math.dist(assigned["xy"],sample["xy"]) if assigned else None)
            if record["stages"]["baseline_qualified"]["hit"] and not record["stages"]["with_degraded"]["hit"]:
                raise ValueError("Candidate loses an existing reference hit")
            evidence.append(record)
    return evidence


def summarize_evidence(evidence):
    result={}
    for kind in ("dense","pilot","anchor","compact_light"):
        chosen=[s for s in evidence if s["kind"]==kind]
        result[kind]=dict(samples=len(chosen),hits={name:sum(s["stages"][name]["hit"] for s in chosen) for name in STAGES},
            recovered_degraded_frames=[dict(window=s["window"],frame=s["frame"]) for s in chosen
                if not s["stages"]["baseline_qualified"]["hit"] and s["stages"]["with_degraded"]["hit"]],
            changed_proximity_assignments=[dict(window=s["window"],frame=s["frame"],
                before=s["stages"]["baseline_qualified"]["assigned_id"],after=s["stages"]["with_degraded"]["assigned_id"])
                for s in chosen if s["stages"]["baseline_qualified"]["hit"] and
                s["stages"]["baseline_qualified"]["assigned_id"]!=s["stages"]["with_degraded"]["assigned_id"]],
            airborne_truth=False,physical_identity_inferred=False)
    return result


def analyze(cid, spec, output):
    wanted=defaultdict(list)
    for sample in samples_for(cid):wanted[sample["frame"]].append(sample)
    controls=read(REFERENCES["controls"])["controls"] if cid=="0126" else []
    controls_counts=[dict(label=c["label"],baseline_measured=0,added_degraded_measured=0) for c in controls]
    policy=CausalMeasuredContinuity()
    totals=Counter();evidence=[];added_ids=set();frame_count=0
    with (Path(spec["path"])/"frames.jsonl").open() as source, \
         (V36/"full_context_01"/(cid+"_decisions.jsonl")).open() as shadow_stream, \
         (output/(cid+"_decisions.jsonl")).open("x") as destination:
        for line in source:
            row=json.loads(line);shadow_line=shadow_stream.readline()
            if not shadow_line:raise ValueError("Truncated frozen shadow decisions")
            shadow=json.loads(shadow_line)
            if (row["frame_index"]!=frame_count or row["timestamp_ns"]!=frame_count*100_000_000
                    or shadow["frame_index"]!=frame_count or shadow["segment"]!=row["segment"]):
                raise ValueError("Misaligned frame evidence")
            decisions=policy.update(row)
            rendered=[]
            for track in row["tracks"]:
                key=(track["segment"],track["track_id"]);d=decisions[key]
                totals["actual_measured_states"]+=int(track["measured"])
                totals["baseline_qualified_measured"]+=int(track["qualified_moving"] and track["measured"])
                totals["baseline_qualified_predictions"]+=int(track["qualified_moving"] and not track["measured"])
                totals["added_degraded_measured"]+=int(d["added_degraded_measurement"])
                if d["added_degraded_measurement"]:added_ids.add(key)
                if d["renderable"]:rendered.append(dict(segment=key[0],track_id=key[1],
                    source_xy=track["source_xy"],**d))
                if track["measured"]:
                    for control,counts in zip(controls,controls_counts):
                        first,last=control["frames_inclusive"]
                        if first<=frame_count<=last and inside(track["measurement_source_xy"],control["crop_xywh"]):
                            counts["baseline_measured"]+=int(track["qualified_moving"])
                            counts["added_degraded_measured"]+=int(d["added_degraded_measurement"])
            destination.write(json.dumps(dict(frame_index=frame_count,timestamp_ns=row["timestamp_ns"],segment=row["segment"],
                output_states=rendered,excluded_state_count=len(decisions)-len(rendered)),allow_nan=False)+"\n")
            if frame_count in wanted:evidence.extend(match_stages(wanted[frame_count],row,decisions,shadow))
            frame_count+=1
        if shadow_stream.readline():raise ValueError("Unexpected extra shadow frames")
    if frame_count!=spec["frames"] or len(evidence)!=sum(map(len,wanted.values())):
        raise ValueError("Changed source/reference denominator")
    totals["renderable_measured"]=totals["baseline_qualified_measured"]+totals["added_degraded_measured"]
    return dict(clip=cid,frames=frame_count,counts=dict(totals),distinct_ids_with_added_degraded_states=len(added_ids),
        provisional_controls=controls_counts,references=summarize_evidence(evidence),reference_evidence=evidence,
        predictions_added=0,baseline_states_removed=0,detector_or_association_changed=False)


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Fresh output required")
    files={};bind(files,AUDIT,AUDIT_SHA);audit=read(AUDIT)
    bind(files,BASE/"validation_coverage_plan_v1.json","cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc")
    if audit.get("verified") is not True:raise ValueError("Verified V38 required")
    for path,digest in audit["checked_files_sha256"].items():
        if Path(path).suffix.lower() not in (".avi",".mp4",".raw16"):
            bind(files,path,digest)
    inputs=verified_inputs()
    for spec in inputs.values():
        for path,digest in spec["files_sha256"].items():bind(files,path,digest)
    for path in REFERENCES.values():bind(files,path)
    for cid in COUNTS:bind(files,V36/"full_context_01"/(cid+"_decisions.jsonl"))
    for name in CODE:bind(files,ROOT/name)
    output.mkdir(parents=True)
    with (output/"unit.log").open("x") as log:
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests/unit","-p","test_accuracy_v39_continuity*.py","-v"],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in CODE:
        destination=output/"implementation"/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination);bind(files,destination,files[str(ROOT/name)])
    bind(files,output/"unit.log");recheck(files)
    freeze=dict(schema="seaqr.accuracy-v39-continuity-freeze.v1",created_ns=time.time_ns(),pre_replay=True,
        inputs=inputs,inputs_sha256=files.copy(),config=asdict(ContinuityConfig()),source_media_opened=False,
        defaults_changed=False,degraded_is_confirmed=False,scope="exposed four-clip development; no airborne truth")
    write(output/"freeze.json",freeze);bind(files,output/"freeze.json")
    clips={}
    for cid,spec in inputs.items():
        print("Frozen continuity replay "+cid,flush=True);clips[cid]=analyze(cid,spec,output)
    overall={kind:dict(samples=sum(c["references"][kind]["samples"] for c in clips.values()),
        hits={name:sum(c["references"][kind]["hits"][name] for c in clips.values()) for name in STAGES})
        for kind in ("dense","pilot","anchor","compact_light")}
    expected={"dense":(285,284),"pilot":(28,28),"anchor":(24,24),"compact_light":(8,6)}
    for kind,(samples,hits) in expected.items():
        if (overall[kind]["samples"],overall[kind]["hits"]["baseline_qualified"])!=(samples,hits):
            raise ValueError("Original denominator/baseline score changed: "+kind)
    outputs={cid+"_decisions.jsonl":bind(files,output/(cid+"_decisions.jsonl")) for cid in COUNTS}
    recheck(files)
    summary=dict(schema="seaqr.accuracy-v39-continuity-summary.v1",completed=True,freeze_sha256=sha(output/"freeze.json"),
        clips=clips,references=overall,outputs_sha256=outputs,defaults_changed=False,media_decoded=False,
        source_video_files_opened=False,airborne_accuracy_established=False,noise_reduction_claimed=False,
        degraded_states_promoted_to_confirmed=False,classifier_promoted=False,raw16_accessed=False,sealed_holdouts_accessed=False,
        decision="development-only pending independent workload/reference review",known_example_exposed_before_parameter_choice=True)
    write(output/"summary.json",summary);bind(files,output/"summary.json");recheck(files)
    write(output/"completion_receipt.json",dict(completed=True,all_inputs_and_outputs_rehashed=True,
        checked_files_sha256=files,source_videos_opened=False,classifier_promoted=False))
    print(json.dumps(overall),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute-replay",action="store_true");a=p.parse_args()
    if not a.execute_replay:p.error("Review/freeze candidate before explicit --execute-replay")
    run(a.output)
