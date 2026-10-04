"""Frozen bounded V38 global-camera source diagnostics; no gate or tuning."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from diagnose_accuracy_v37_temporal import (ROOT, PARENT, CONTEXT, FULL, SOURCES,
    COUNTS, read, sha, bind, recheck, write, verify_parent, validate_row)
from accuracy_v38_source_pairs import CausalSourceBuffer, LAGS
from accuracy_v38_evidence import SourceEvidenceDiagnostic, METRICS

BASE = ROOT / "results/tiny_target/accuracy_v38_20260925"
REFERENCE = BASE / "compact_light_reference_v1.json"
REFERENCE_REVIEW = BASE / "compact_light_reference_v1_root_review.json"
AUDIT37 = PARENT.parent / "accuracy_v37_20260924/temporal_saved_evidence_audit_01.json"
AUDIT37_SHA = "9076bb4f8b31f9eb50c320c17178f2e3cce339a05008974d0d315f715d7fbf78"
IMPLEMENTATION = (
    "scripts/diagnose_accuracy_v38_global_multilag.py", "scripts/accuracy_v38_source_pairs.py",
    "scripts/accuracy_v38_evidence.py", "scripts/accuracy_v36_context.py", "scripts/accuracy_v37_temporal.py",
    "scripts/diagnose_accuracy_v37_temporal.py", "docs/accuracy_v38_plan.md",
    "tests/unit/test_accuracy_v38_source_pairs.py", "tests/unit/test_accuracy_v38_evidence.py",
    "tests/unit/test_accuracy_v38_extraction.py",
    "scripts/review_accuracy_v38_reference.py",
    "docs/accuracy_v38_reference_review.md", "docs/accuracy_v38_reference_review_root.md",
)


def verify_reference(files):
    review = read(REFERENCE_REVIEW)
    if (review.get("approved_use")!="provisional_class_unknown_image_feature_regression"
            or review.get("before_v38_scoring") is not True
            or review.get("airborne_truth") is not False):
        raise ValueError("Second source-only reference review is required before scoring")
    bind(files, REFERENCE_REVIEW)
    bind(files, REFERENCE, review["annotation_sha256"])
    for path,digest in review["inputs_sha256"].items():
        bind(files,path,digest)
    reference = read(REFERENCE)
    for path,digest in reference["inputs_sha256"].items():
        bind(files,path,digest)
    return reference


def bind_parents():
    selection, files, parent = verify_parent(PARENT / "full_context_independent_audit_01.json")
    bind(files, AUDIT37, AUDIT37_SHA)
    audit = read(AUDIT37)
    if (audit.get("verified") is not True or audit.get("selected_observations") != 358
            or audit.get("complete_summary_independently_reconstructed") is not True
            or audit.get("source_video_files_opened") is not False):
        raise ValueError("Complete saved-evidence V37 audit required")
    for path, digest in audit["checked_files_sha256"].items():
        bind(files, path, digest)
    for cid, source in SOURCES.items():
        bind(files, source, parent["inputs"][cid]["source_sha256"])
    return selection, files, parent


def add_reference(selection, reference, journal, decisions):
    """Independent source positions select existing measured states, not vice versa.

    This single additional feature has at most one reference per frame. All
    gated states are retained for diagnostic availability; a nearest-distance
    assignment is recorded explicitly, not physical-identity ground truth.
    """
    if reference["clip"] != "0029" or reference["physical_class"] != "unknown":
        raise ValueError("Only the separately reviewed class-unknown 0029 feature is permitted")
    frames = reference["frames"]
    if [r["frame_index"] for r in frames] != list(range(12,25)):
        raise ValueError("Exactly the preselected 13 review frames required")
    visible = [r for r in frames if r["visibility"] == "visible"]
    if any(r["visibility"] not in ("visible", "ambiguous", "not_visible") for r in frames):
        raise ValueError("Invalid separate reference visibility")
    result = copy.deepcopy(selection)
    known = {o["key"]:o for o in result["observations"]}
    groups = []
    for sample in visible:
        frame = sample["frame_index"]
        xy = np.asarray(sample["source_xy"], dtype=float)
        uncertainty = sample["position_uncertainty_radius_px"]
        if (xy.shape!=(2,) or not np.isfinite(xy).all() or type(uncertainty) not in (int,float)
                or not np.isfinite(uncertainty) or uncertainty<=0):
            raise ValueError("Finite reviewed source location and uncertainty required")
        radius = uncertainty+2.0
        matches = []
        accepted = {f'{t["segment"]}/{t["track_id"]}' for t in decisions[frame]["tracks"] if t["accepted"]}
        for track in journal[frame]["tracks"]:
            if not (track["measured"] and track["qualified_moving"] and track["track_id"].startswith("bright:")):
                continue
            distance = float(np.linalg.norm(np.asarray(track["measurement_source_xy"])-xy))
            if distance>radius:
                continue
            identity = f'{track["segment"]}/{track["track_id"]}'
            key = f'0029/{frame}/{identity}'
            item = dict(key=key,clip="0029",frame=frame,identity=identity,
                measurement_source_xy=track["measurement_source_xy"],polarity="bright",provisional_control_indices=[])
            if key in known:
                if any(known[key][field]!=value for field,value in item.items() if field!="provisional_control_indices"):
                    raise ValueError("Reference observation conflicts with original selection")
            else:
                known[key] = item
            matches.append((distance,identity,key))
        matches.sort()
        groups.append(dict(kind="compact_light",clip="0029",window="class_unknown_compact_light_v1",frame=frame,
            keys=[k for _,_,k in matches],baseline_assigned_id=matches[0][1] if matches else None,
            source_reference_xy=xy.tolist(),uncertainty_px=uncertainty,matching_radius_px=radius,
            v36_retained_keys=[k for _,identity,k in matches if identity in accepted]))
    result["observations"] = sorted(known.values(),key=lambda o:(o["clip"],o["frame"],o["identity"]))
    result["groups"].extend(groups)
    result["additional_reference"] = dict(clip="0029",reviewed_frames=13,physical_class="unknown",
        visibility_counts=dict(Counter(r["visibility"] for r in frames)),source_selected_from_v36_review=True,
        independently_held_out=False,authoritative_airborne_truth=False,matching_extra_radius_px=2.0)
    return result


def save_array(arrays, prefix, name, value):
    key = prefix+"_"+name
    if key in arrays:
        raise ValueError("Duplicate array key")
    value = np.ascontiguousarray(value)
    arrays[key] = value.copy()
    return dict(array_key=key,shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),finite_pixels=int(np.isfinite(value).sum()))


def extract_clip(cid, source, rows, observations, original, arrays, records):
    selected = defaultdict(list)
    for observation in observations:
        if observation["clip"]==cid:
            selected[observation["frame"]].append(observation)
    buffer, evidence = CausalSourceBuffer(max_lag=8), SourceEvidenceDiagnostic()
    cap = cv2.VideoCapture(str(source))
    try:
        if (not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)!=10
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=COUNTS[cid]):
            raise ValueError("Unexpected source decoder metadata")
        last = max(selected)
        for frame in range(last+1):
            ok,bgr = cap.read()
            if not ok or bgr.shape!=(3190,4784,3) or bgr.dtype!=np.uint8:
                raise ValueError("Sequential source decode failed")
            row = rows[frame]
            validate_row(row,frame)
            buffer.update(row,cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
            for observation in selected.get(frame,()):
                prefix = f'observation_{len(records):04d}'
                pair = buffer.extract(observation["measurement_source_xy"],identity=observation["identity"])
                current = pair.pop("current25")
                current_check = observation["key"] in original
                if current_check and (not np.isfinite(current).all() or
                        hashlib.sha256(current.astype(np.uint8).tobytes()).hexdigest()!=original[observation["key"]]):
                    raise ValueError("Current native pixels changed from frozen V36 source")
                if pair["identity"]!=observation["identity"]:
                    raise ValueError("Source-pair identity differs from selected current measurement")
                result = {**copy.deepcopy(observation),**{k:v for k,v in pair.items() if k!="lags"},
                    "source_sha256":sha_cache[cid],"v36_native_patch_crosschecked":current_check,
                    "current25_array":save_array(arrays,prefix,"current25",current),"lags":[]}
                for lag in pair["lags"]:
                    prior = lag.pop("prior27")
                    item = dict(lag)
                    item["prior27_array"] = save_array(arrays,prefix,f'prior27_lag{lag["lag"]}',prior) if prior is not None else None
                    item["evidence"] = evidence.measure(current,prior,polarity=observation["polarity"],
                        current_xy=pair["current_xy"],previous_xy=lag["previous_point_current_grid_xy"]) if lag["available"] else None
                    result["lags"].append(item)
                records.append(result)
        return dict(first_frame=0,last_frame=last,decoded_frames=last+1,source_frames=COUNTS[cid],
            nominal_fps=10,max_buffer_frames=9,source_shape_hw=[3190,4784],backend=cap.getBackendName())
    finally:
        cap.release()


def flag(record,lag_index,kind):
    lag = record["lags"][lag_index]
    if kind=="geometry":
        return lag["available"]
    if not lag["evidence"]:
        return False
    evidence = lag["evidence"]
    if kind=="nominal_source_supported":
        return evidence["probes"][4]["source_supported"]
    if kind=="all_nine_source_supported":
        return evidence["source_supported_probes"]==9
    if kind=="nominal_joint_available":
        return evidence["probes"][4]["conditional_pair"]["available"]
    if kind=="all_nine_joint_available":
        return all(p["conditional_pair"]["available"] for p in evidence["probes"])
    return bool(lag["evidence"] and lag["evidence"][kind])


def scope_stats(records):
    result = dict(observations=len(records),per_lag={})
    for index,lag in enumerate(LAGS):
        values = [r["lags"][index] for r in records]
        result["per_lag"][str(lag)] = dict(
            geometry_available=sum(v["available"] for v in values),
            previous_actual_measurement_available=sum(v["previous_actual_measurement_available"] for v in values),
            nominal_contrast_available=sum(flag(r,index,"nominal_contrast_available") for r in records),
            all_nine_envelope_available=sum(flag(r,index,"envelope_available") for r in records),
            nominal_source_supported=sum(flag(r,index,"nominal_source_supported") for r in records),
            all_nine_source_supported=sum(flag(r,index,"all_nine_source_supported") for r in records),
            nominal_joint_available=sum(flag(r,index,"nominal_joint_available") for r in records),
            all_nine_joint_available=sum(flag(r,index,"all_nine_joint_available") for r in records),
            geometry_unknown_reasons=dict(Counter(reason for v in values for reason in v["reasons"])),
            probe_unknown_reasons=dict(Counter(reason for v in values if v["evidence"] for p in v["evidence"]["probes"] for reason in p["reasons"])),
            joint_unknown_reasons=dict(Counter(reason for v in values if v["evidence"] for p in v["evidence"]["probes"] for reason in p["conditional_pair"]["reasons"])),
            envelope_distributions={})
        for metric in METRICS:
            summaries = [v["evidence"]["envelope"][metric] for v in values if v["evidence"] and v["evidence"]["envelope_available"]]
            result["per_lag"][str(lag)]["envelope_distributions"][metric] = {
                field:dict(count=len(summaries),minimum=min(s[field] for s in summaries),
                           median=float(np.median([s[field] for s in summaries])),maximum=max(s[field] for s in summaries))
                for field in ("minimum","median","maximum","span")} if summaries else {"count":0}
    for kind in ("geometry","nominal_contrast_available","envelope_available"):
        result["all_four_lags_"+kind] = sum(all(flag(r,i,kind) for i in range(4)) for r in records)
    return result


def summarize(selection,records):
    by_key = {r["key"]:r for r in records}
    if len(by_key)!=len(records) or set(by_key)!={o["key"] for o in selection["observations"]}:
        raise ValueError("Missing/duplicate source diagnostics")
    scopes = defaultdict(set)
    groups = []
    for group in selection["groups"]:
        scopes[group["kind"]+"/"+group["window"]].update(group["keys"])
        item = copy.deepcopy(group)
        item["baseline_has_match"] = bool(group["keys"])
        item["diagnostic_available_keys"] = {
            kind:{str(lag):[k for k in group["keys"] if flag(by_key[k],i,kind)] for i,lag in enumerate(LAGS)}
            for kind in ("geometry","nominal_contrast_available","envelope_available")}
        item["all_four_lag_envelope_keys"] = [k for k in group["keys"] if all(flag(by_key[k],i,"envelope_available") for i in range(4))]
        groups.append(item)
    for i,c in enumerate(selection["controls"]):
        scopes[f'control/{i}/{c["label"]}'].update(r["key"] for r in records if i in r["provisional_control_indices"])
    scopes["all"].update(by_key)
    for cid in SOURCES:
        scopes["clip/"+cid].update(r["key"] for r in records if r["clip"]==cid)
    reference = {}
    for kind in ("dense","pilot","anchor","compact_light"):
        matched = [g for g in groups if g["kind"]==kind]
        reference[kind] = dict(samples=len(matched),baseline_matched_samples=sum(g["baseline_has_match"] for g in matched),
            per_lag={str(lag):{metric:sum(bool(g["diagnostic_available_keys"][metric][str(lag)]) for g in matched)
                for metric in ("geometry","nominal_contrast_available","envelope_available")} for lag in LAGS},
            all_four_lag_envelope_samples=sum(bool(g["all_four_lag_envelope_keys"]) for g in matched),
            detection_retention_claimed=False)
        if kind=="compact_light":
            reference[kind]["v36_matched_samples"] = sum(bool(g["v36_retained_keys"]) for g in matched)
    return dict(selected_observations=len(records),reference_provenance=reference,groups=groups,
        scopes={name:scope_stats([by_key[k] for k in sorted(keys)]) for name,keys in sorted(scopes.items())},
        diagnostic_only=True,detector_rerun=False,classifier_promoted=False,output_gate_applied=False,
        airborne_accuracy_established=False,thresholds_tuned=False,raw16_sources_accessed=False,sealed_holdouts_accessed=False,
        limitations=["Availability and sensitivity features are not detections or class labels",
            "Global geometry quality and the one-pixel envelope are not calibrated confidence",
            "Per-lag/probe features share source pixels and are not independent votes",
            "Unit photometric gain does not correct multiplicative exposure changes",
            "Missing prior point association can leave subtraction ghosts; zero residual is not absence",
            "New compact-light case was selected from development filter removals, not held-out truth"])


sha_cache = {}


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh exclusive output directory required")
    selection,files,parent = bind_parents()
    reference = verify_reference(files)
    rows,decisions = {},{}
    for cid in SOURCES:
        journal = Path(parent["inputs"][cid]["path"])/"frames.jsonl"
        with journal.open() as stream:
            rows[cid] = [json.loads(line) for line in stream]
        with (FULL/(cid+"_decisions.jsonl")).open() as stream:
            decisions[cid] = [json.loads(line) for line in stream]
        if len(rows[cid])!=COUNTS[cid] or len(decisions[cid])!=COUNTS[cid]:
            raise ValueError("Truncated original frame evidence")
        sha_cache[cid] = parent["inputs"][cid]["source_sha256"]
    selection = add_reference(selection,reference,rows["0029"],decisions["0029"])
    implementation = {name:bind(files,ROOT/name) for name in IMPLEMENTATION}
    output.mkdir(parents=True)
    with (output/"unit.log").open("x") as log:
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests/unit","-p","test_accuracy_v38*.py","-v"],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name,digest in implementation.items():
        target = output/"implementation"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        bind(files,target,digest)
    write(output/"selection.json",selection)
    bind(files,output/"selection.json")
    bind(files,output/"unit.log")
    recheck(files)
    freeze = dict(schema="seaqr.accuracy-v38-global-multilag-freeze.v1",created_ns=time.time_ns(),pre_extraction=True,
        inputs_sha256=files.copy(),implementation_sha256=implementation,source_videos={cid:dict(path=str(SOURCES[cid]),sha256=sha_cache[cid]) for cid in SOURCES},
        original_selection_count=358,selected_observations=len(selection["observations"]),lags=list(LAGS),
        sampling="float64 exact bilinear; global source transform; positive-weight finite support",
        shifts_xy=[[dx,dy] for dx in (-1,0,1) for dy in (-1,0,1)],source_frame_buffer_limit=9,
        local_registration_required=False,prior_association_required=False,photometric_gain=1.,diagnostic_only=True,
        classifier_promoted=False,output_gate_applied=False)
    write(output/"freeze.json",freeze)
    bind(files,output/"freeze.json")
    recheck(files)
    original = {r["key"]:r["patch_sha256"] for r in read(CONTEXT/"observations.json")}
    arrays,records,decoders = {},[],{}
    for cid,source in SOURCES.items():
        print("Frozen V38 source extraction "+cid,flush=True)
        decoders[cid] = extract_clip(cid,source,rows[cid],selection["observations"],original,arrays,records)
    if sum(r["v36_native_patch_crosschecked"] for r in records)!=358:
        raise ValueError("All 358 original source patches must be preserved")
    np.savez_compressed(output/"source_patches.npz",**arrays)
    write(output/"observations.json",records)
    summary = summarize(selection,records)
    summary.update(schema="seaqr.accuracy-v38-global-multilag-summary.v1",completed=True,freeze_sha256=sha(output/"freeze.json"),
        capture_records=decoders,original_native_patches_crosschecked=358,
        outputs_sha256={name:bind(files,output/name) for name in ("observations.json","source_patches.npz")})
    recheck(files)
    write(output/"summary.json",summary)
    print(json.dumps(summary["reference_provenance"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute-extraction",action="store_true")
    args=parser.parse_args()
    if not args.execute_extraction:
        parser.error("Explicit --execute-extraction required after code/tests/reference review")
    cv2.setNumThreads(2)
    run(args.output)
