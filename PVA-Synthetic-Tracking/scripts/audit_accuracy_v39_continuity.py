"""Independent saved-journal V39 continuity audit; no policy/scorer imports.

Reconstructs eligibility from at most three original rows rather than a policy
cache. Validates reference gates and maximum-cardinality assignments with an
independent breadth-first augmenting-path matcher. Reads no source media.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/tiny_target/accuracy_v39_20260925"
COUNTS={"0029":687,"0126":674,"0055":689,"0082":691}
STAGES=("candidate","actual_measurement","baseline_qualified","with_degraded","v36_shadow")
KINDS=("dense","pilot","anchor","compact_light")
REFERENCE_PATHS={
    "dense_labels":ROOT/"results/tiny_target/phase20/encounter_accuracy_v2_20260914/annotations.json",
    "dense_packet":ROOT/"results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json",
    "pilot_labels":ROOT/"results/tiny_target/phase20/accuracy_baseline_v1_20260914/annotations.json",
    "pilot_packet":ROOT/"results/tiny_target/phase20/accuracy_baseline_v1_20260914/source_review/packet.json",
    "anchor_0029":ROOT/"results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json",
    "anchor_0126":ROOT/"results/tiny_target/phase19/chunk0126_avi_20260913/visual_review/visual_annotations.json",
    "compact_light":ROOT/"results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1.json",
    "controls":ROOT/"results/tiny_target/phase20/clutter_review_20260913/background_controls.json",
}


def need(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read(path):
    with Path(path).open() as stream:return json.load(stream)


def same(actual,expected,message):
    need(actual==expected,message)


def position(xy):
    return isinstance(xy,list) and len(xy)==2 and all(type(v) in (float,int) and math.isfinite(v) for v in xy)


def track_map(row):
    tracks={}
    for t in row["tracks"]:
        key=(t["segment"],t["track_id"])
        need(key not in tracks and t["segment"]==row["segment"],"Duplicate or cross-segment original identity")
        need(type(t["measured"]) is bool and type(t["qualified_moving"]) is bool,"Explicit original measurement flags required")
        need(position(t["source_xy"]),"Original state lacks finite source coordinate")
        need(position(t["measurement_source_xy"]) if t["measured"] else t["measurement_source_xy"] is None,
             "Original measured/predicted coordinate mismatch")
        tracks[key]=t
    return tracks


def failed_quality_only(track,timestamp):
    q=track.get("motion_quality") or {}
    confirmation=track.get("confirmation_timestamp_ns")
    excursion=track.get("excursion_px")
    return (track["measured"] and not track["qualified_moving"]
        and q.get("ready") is True and q.get("passed") is False
        and type(confirmation) is int and 0<=confirmation<=timestamp
        and type(excursion) in (float,int) and math.isfinite(excursion) and excursion>=12.)


def direct_anchor(history,key):
    """Find a strict actual anchor by inspecting original rows, no state cache."""
    current=history[-1];now=current["row"];track=current["tracks"].get(key)
    if track is None or not failed_quality_only(track,now["timestamp_ns"]):return None
    for i in range(len(history)-2,-1,-1):
        prior=history[i];anchor=prior["tracks"].get(key)
        if anchor is None or not(anchor["qualified_moving"] and anchor["measured"]):continue
        first=prior["row"]
        if not(0<now["frame_index"]-first["frame_index"]<=2 and 0<now["timestamp_ns"]-first["timestamp_ns"]<=200_000_000):continue
        intact=True
        for item in history[i+1:]:
            r=item["row"];t=item["tracks"].get(key)
            if (r["segment"]!=first["segment"] or r["motion"]["reset"] or t is None
                    or not t["measured"] or not(t["qualified_moving"] or failed_quality_only(t,r["timestamp_ns"]))):
                intact=False;break
        if intact:return(first["frame_index"],first["timestamp_ns"])
    return None


def maximum_matching_size(adjacency):
    """Independent iterative alternating-path matching; does not use tie claims."""
    left_owner={};right_owner={}
    for start in range(len(adjacency)):
        if start in left_owner:continue
        queue=deque([start]);seen_left={start};parent_right={};free=None
        while queue and free is None:
            u=queue.popleft()
            for v in adjacency[u]:
                if v in parent_right:continue
                parent_right[v]=u
                if v not in right_owner:free=v;break
                owner=right_owner[v]
                if owner not in seen_left:seen_left.add(owner);queue.append(owner)
        while free is not None:
            u=parent_right[free];old=left_owner.get(u)
            left_owner[u]=free;right_owner[free]=u;free=old
    return len(left_owner)


def references(cid,source_sha):
    result=[]
    for kind in ("dense","pilot"):
        labels=read(REFERENCE_PATHS[kind+"_labels"]);packet=read(REFERENCE_PATHS[kind+"_packet"])
        same(labels["policy"]["extra_localization_tolerance_px"],2.,"Matching tolerance changed")
        windows={w["id"]:w for w in packet["windows"]}
        for event in labels["positive_windows"]:
            if windows[event["window_id"]]["clip_id"]!=cid:continue
            for s in event["visible_samples"]:
                result.append(dict(kind=kind,window=event["window_id"],frame=s["frame_index"],
                    xy=s["xy"],radius=s["uncertainty_px"]+2.,polarity=event["polarity"]))
    if cid in ("0029","0126"):
        original=read(REFERENCE_PATHS["anchor_"+cid]);same(original["source_sha256"],source_sha,"Anchor source mismatch")
        if cid=="0029":
            same(original["schema"],"manual_visual_reference_v1","Unknown anchor schema")
            for event in original["events"]:
                for s in event["anchors"]:
                    if s["visibility"]=="visible":
                        result.append(dict(kind="anchor",window=event["event_id"],frame=s["frame_index"],
                            xy=[s["x"],s["y"]],radius=s["position_uncertainty_px"]+2.,polarity="bright"))
        else:
            same(original["source"],"chunk_0126.avi","Unknown anchor schema")
            for s in original["anchors"]:
                if s["visibility"]=="Visually identifiable":
                    result.append(dict(kind="anchor",window="chunk0126_moving_point",frame=s["frame_index"],
                        xy=s["approximate_xy"],radius=s["approximate_position_uncertainty_px"]+2.,polarity="bright"))
    if cid=="0029":
        for s in read(REFERENCE_PATHS["compact_light"])["frames"]:
            if s["visibility"]=="visible":
                result.append(dict(kind="compact_light",window="class_unknown_compact_light_v1",frame=s["frame_index"],
                    xy=s["source_xy"],radius=s["position_uncertainty_radius_px"]+2.,polarity="bright"))
    keys=[(s["kind"],s["window"],s["frame"]) for s in result]
    same(len(set(keys)),len(keys),"Duplicate independent reference")
    return result


def stage_observations(row,added,shadow):
    result={stage:[] for stage in STAGES}
    result["candidate"]=[dict(id="candidate:"+str(i),xy=c["source_xy"],polarity=c["polarity"])
                         for i,c in enumerate(row["candidates"])]
    accepted={(s["segment"],s["track_id"]) for s in shadow["tracks"] if s["accepted"]}
    qualified={(t["segment"],t["track_id"]) for t in row["tracks"] if t["qualified_moving"]}
    need(accepted<=qualified,"V36 shadow creates a baseline-unqualified identity")
    for t in row["tracks"]:
        if not t["measured"]:continue
        key=(t["segment"],t["track_id"])
        item=dict(id=str(key[0])+"/"+key[1],xy=t["measurement_source_xy"],polarity=key[1].split(":")[0])
        result["actual_measurement"].append(item)
        if key in qualified:result["baseline_qualified"].append(item)
        if key in qualified or key in added:result["with_degraded"].append(item)
        if key in accepted:result["v36_shadow"].append(item)
    return result


def validate_reference_frame(samples,saved,observations):
    """Check exact neighbors/ambiguity, legal assignment and optimal cardinality."""
    for kind in KINDS:
        current=[s for s in samples if s["kind"]==kind]
        records=[saved[(s["kind"],s["window"],s["frame"])] for s in current]
        for s,r in zip(current,records):same({k:r[k] for k in s},s,"Reference sample changed")
        for stage in STAGES:
            points=observations[stage]
            adjacency=[sorted([i for i,p in enumerate(points) if p["polarity"]==s["polarity"]
                               and math.dist(p["xy"],s["xy"])<=s["radius"]],
                              key=lambda i:(math.dist(points[i]["xy"],s["xy"]),i)) for s in current]
            id_lookup={p["id"]:i for i,p in enumerate(points)}
            same(len(id_lookup),len(points),"Duplicate stage identity")
            assigned=set()
            for i,(s,r) in enumerate(zip(current,records)):
                claim=r["stages"][stage]
                same(claim["all_gated_ids"],[points[j]["id"] for j in adjacency[i]],"Saved stage neighbors changed")
                ambiguous=len(adjacency[i])>1 or any(sum(j in n for n in adjacency)>1 for j in adjacency[i])
                same(claim["ambiguous"],ambiguous,"Ambiguity flag changed")
                identity=claim["assigned_id"]
                same(claim["hit"],identity is not None,"Hit lacks assigned observation")
                if identity is None:same(claim["distance_px"],None,"Miss has a distance");continue
                need(identity in id_lookup and id_lookup[identity] in adjacency[i] and identity not in assigned,
                     "Invalid or multiply assigned measured observation")
                assigned.add(identity)
                same(claim["distance_px"],math.dist(points[id_lookup[identity]]["xy"],s["xy"]),"Assigned distance changed")
            same(len(assigned),maximum_matching_size(adjacency),"Assignment is not maximum-cardinality")
        for r in records:
            need(not r["stages"]["baseline_qualified"]["hit"] or r["stages"]["with_degraded"]["hit"],"Known baseline sample lost")


def summarize_references(evidence):
    output={}
    for kind in KINDS:
        rows=[r for r in evidence if r["kind"]==kind]
        recovered=[];changed=[]
        for r in rows:
            before=r["stages"]["baseline_qualified"];after=r["stages"]["with_degraded"]
            if not before["hit"] and after["hit"]:recovered.append(dict(window=r["window"],frame=r["frame"]))
            if before["hit"] and before["assigned_id"]!=after["assigned_id"]:
                changed.append(dict(window=r["window"],frame=r["frame"],before=before["assigned_id"],after=after["assigned_id"]))
        output[kind]=dict(samples=len(rows),hits={stage:sum(r["stages"][stage]["hit"] for r in rows) for stage in STAGES},
            recovered_degraded_frames=recovered,changed_proximity_assignments=changed,
            airborne_truth=False,physical_identity_inferred=False)
    return output


def audit_clip(cid,spec,summary,experiment,bound):
    original=Path(spec["path"])/"frames.jsonl"
    decisions=experiment/(cid+"_decisions.jsonl")
    shadow_path=ROOT/"results/tiny_target/accuracy_v36_20260924/full_context_01"/(cid+"_decisions.jsonl")
    for p in (original,decisions,shadow_path):need(str(p.resolve()) in bound,"Unbound input journal")
    samples=references(cid,spec["source_sha256"]);wanted=defaultdict(list)
    for s in samples:wanted[s["frame"]].append(s)
    evidence=summary["reference_evidence"]
    saved={(r["kind"],r["window"],r["frame"]):r for r in evidence}
    same(len(saved),len(evidence),"Duplicate saved reference evidence")
    same(set(saved),{(s["kind"],s["window"],s["frame"]) for s in samples},"Reference denominator changed")
    controls=read(REFERENCE_PATHS["controls"])["controls"] if cid=="0126" else []
    if cid=="0126":same(len(controls),7,"Fixed control inventory changed")
    control_counts=[dict(label=c["label"],baseline_measured=0,added_degraded_measured=0) for c in controls]
    counts=Counter();identities=set();history=[];age_frames=Counter();per_frame=[];checked_samples=0
    with original.open() as source,decisions.open() as decision_stream,shadow_path.open() as shadow_stream:
        for frame,line in enumerate(source):
            row=json.loads(line);decision_line=decision_stream.readline();shadow_line=shadow_stream.readline()
            need(decision_line and shadow_line,"Truncated decision journal")
            output=json.loads(decision_line);shadow=json.loads(shadow_line)
            for record in (row,output,shadow):
                same(record["frame_index"],frame,"Noncontiguous journal")
                same(record["timestamp_ns"],frame*100_000_000,"Unexpected original time grid")
                same(record["segment"],row["segment"],"Segment misalignment")
            tracks=track_map(row);history=(history+[dict(row=row,tracks=tracks)])[-3:]
            added={key:anchor for key in tracks if (anchor:=direct_anchor(history,key)) is not None}
            baseline={key for key,t in tracks.items() if t["qualified_moving"]}
            actual_outputs={(o["segment"],o["track_id"]):o for o in output["output_states"]}
            same(len(actual_outputs),len(output["output_states"]),"Duplicate output identity")
            same(set(actual_outputs),baseline|set(added),"Output set does not equal exact baseline plus bounded measured continuation")
            same(output["excluded_state_count"],len(tracks)-len(actual_outputs),"Excluded count changed")
            for key,o in actual_outputs.items():
                t=tracks[key];degraded=key in added;qualified=key in baseline
                for field in ("measured","measurement_source_xy","source_xy"):
                    same(o[field],t[field],"Original output field not preserved: "+field)
                expected_status=("quality_degraded_measured" if degraded else
                    "baseline_qualified_measured" if t["measured"] else "baseline_qualified_prediction")
                same(o["status"],expected_status,"State tier changed")
                same(o["reason"],"bounded_recent_qualified_measurement" if degraded else "unchanged_baseline_qualification","State reason changed")
                for field,value in (("baseline_qualified",qualified),("renderable",True),("added_degraded_measurement",degraded),
                                    ("confirmed_output",qualified),("physical_class","unknown"),("airborne_confirmed",False)):
                    same(o[field],value,"Output meaning changed: "+field)
                anchor=added[key] if degraded else (frame,row["timestamp_ns"]) if t["measured"] else None
                for field,value in (("anchor_frame",anchor[0] if anchor else None),
                    ("anchor_timestamp_ns",anchor[1] if anchor else None),
                    ("anchor_age_frames",frame-anchor[0] if anchor else None),
                    ("anchor_age_ns",row["timestamp_ns"]-anchor[1] if anchor else None)):
                    same(o[field],value,"Causal nonrefreshing anchor changed")
                if degraded:age_frames[str(o["anchor_age_frames"])]+=1
            measured=sum(t["measured"] for t in tracks.values())
            baseline_measured=sum(tracks[k]["measured"] for k in baseline)
            baseline_predictions=len(baseline)-baseline_measured
            counts["actual_measured_states"]+=measured
            counts["baseline_qualified_measured"]+=baseline_measured
            counts["baseline_qualified_predictions"]+=baseline_predictions
            counts["added_degraded_measured"]+=len(added);identities.update(added)
            per_frame.append(dict(frame=frame,baseline_measured=baseline_measured,baseline_predictions=baseline_predictions,added_measured=len(added)))
            for c,total in zip(controls,control_counts):
                first,last=c["frames_inclusive"];x,y,w,h=c["crop_xywh"]
                if not first<=frame<=last:continue
                for key,t in tracks.items():
                    if not t["measured"]:continue
                    px,py=t["measurement_source_xy"]
                    if x<=px<x+w and y<=py<y+h:
                        total["baseline_measured"]+=int(key in baseline)
                        total["added_degraded_measured"]+=int(key in added)
            if frame in wanted:
                validate_reference_frame(wanted[frame],saved,stage_observations(row,added,shadow))
                checked_samples+=len(wanted[frame])
        need(not decision_stream.readline() and not shadow_stream.readline(),"Extra saved decision frame")
    frames=len(per_frame);same(frames,COUNTS[cid],"Original frame count changed")
    same(frames,spec["frames"],"Frozen frame count changed");same(summary["frames"],frames,"Summary frame count changed")
    same(checked_samples,len(samples),"Reference frame outside original journal")
    counts["renderable_measured"]=counts["baseline_qualified_measured"]+counts["added_degraded_measured"]
    same(dict(counts),summary["counts"],"Per-clip state counts changed")
    same(len(identities),summary["distinct_ids_with_added_degraded_states"],"Added identity count changed")
    same(control_counts,summary["provisional_controls"],"Control workload changed")
    independent_references=summarize_references(evidence)
    same(independent_references,summary["references"],"Reference arithmetic changed")
    same(summary["predictions_added"],0,"Prediction was added")
    same(summary["baseline_states_removed"],0,"Baseline was removed")
    same(summary["detector_or_association_changed"],False,"Association scope changed")
    return dict(frames=frames,counts=dict(counts),distinct_added_segment_ids=len(identities),
        added_anchor_age_frames=dict(age_frames),maximum_added_in_one_frame=max(p["added_measured"] for p in per_frame),
        frames_with_added_measurements=sum(p["added_measured"]>0 for p in per_frame),
        added_measured_states_per_source_frame=counts["added_degraded_measured"]/frames,
        references=independent_references,provisional_controls=control_counts)


def synthetic_self_test():
    same(maximum_matching_size([[0,1],[0],[1,2]]),3,"Alternating-path matcher test failed")
    same(maximum_matching_size([[0],[0],[]]),1,"Competing reference matcher test failed")
    def item(f,qualified=False,measured=True,reset=False):
        t=dict(track_id="bright:1",segment=0,qualified_moving=qualified,measured=measured,
            confirmation_timestamp_ns=0,excursion_px=20.,motion_quality=dict(ready=True,passed=qualified))
        return dict(row=dict(frame_index=f,timestamp_ns=f*100_000_000,segment=0,motion=dict(reset=reset)),tracks={(0,"bright:1"):t})
    key=(0,"bright:1")
    same(direct_anchor([item(0,True),item(1),item(2)],key),(0,0),"Two-frame direct continuation failed")
    same(direct_anchor([item(1),item(2),item(3)],key),None,"Degraded self-refresh permitted")
    same(direct_anchor([item(0,True),item(1,measured=False),item(2)],key),None,"Coast lineage permitted")
    same(direct_anchor([item(0,True),item(1,reset=True)],key),None,"Reset lineage permitted")
    return dict(independent_matching=True,direct_nonrefreshing_history=True,coast_and_reset_break=True)


def run(experiment,output,expected_receipt_sha256):
    experiment=Path(experiment).resolve();output=Path(output).resolve()
    need(experiment.parent==BASE and output.parent==BASE and output.suffix==".json","Audit paths must stay in bounded V39 directory")
    if output.exists():raise FileExistsError("Fresh independent audit result required")
    receipt_path=experiment/"completion_receipt.json"
    same(sha(receipt_path),expected_receipt_sha256,"Completion receipt pin changed")
    receipt=read(receipt_path)
    need(receipt["completed"] is True and receipt["all_inputs_and_outputs_rehashed"] is True,"Incomplete replay receipt")
    need(receipt["source_videos_opened"] is False and receipt["classifier_promoted"] is False,"Replay scope violation")
    bound=receipt["checked_files_sha256"].copy()
    need(all(Path(p).suffix.lower() not in (".avi",".mp4",".raw16",".mkv") for p in bound),"Receipt unexpectedly binds source media")
    for p,digest in bound.items():same(sha(p),digest,"Bound input/output changed: "+p)
    for p in (experiment/"freeze.json",experiment/"summary.json",*REFERENCE_PATHS.values()):
        need(str(p.resolve()) in bound,"Missing bound audit input: "+str(p))
    frozen=read(experiment/"freeze.json");summary=read(experiment/"summary.json")
    same(frozen["schema"],"seaqr.accuracy-v39-continuity-freeze.v1","Unexpected freeze schema")
    same(summary["schema"],"seaqr.accuracy-v39-continuity-summary.v1","Unexpected summary schema")
    need(frozen["pre_replay"] and summary["completed"],"Freeze/completion state invalid")
    same(summary["freeze_sha256"],sha(experiment/"freeze.json"),"Freeze binding invalid")
    same(frozen["config"],dict(maximum_gap_frames=2,maximum_gap_ns=200_000_000,minimum_excursion_px=12.),"Candidate configuration changed")
    same(set(frozen["inputs"]),set(COUNTS),"Allowed clip set changed")
    for p,digest in frozen["inputs_sha256"].items():same(bound.get(p),digest,"Completion omitted a frozen input")
    tests=synthetic_self_test();clips={}
    for cid,spec in frozen["inputs"].items():
        print("Independently auditing saved journal "+cid,flush=True)
        clips[cid]=audit_clip(cid,spec,summary["clips"][cid],experiment,bound)
    overall={kind:dict(samples=sum(c["references"][kind]["samples"] for c in clips.values()),
        hits={stage:sum(c["references"][kind]["hits"][stage] for c in clips.values()) for stage in STAGES}) for kind in KINDS}
    same(overall,summary["references"],"Overall reference arithmetic changed")
    same({k:(v["samples"],v["hits"]["baseline_qualified"]) for k,v in overall.items()},
        dict(dense=(285,284),pilot=(28,28),anchor=(24,24),compact_light=(8,6)),"Original denominator/baseline changed")
    for cid,digest in summary["outputs_sha256"].items():same(sha(experiment/cid),digest,"Decision output binding changed")
    for p,digest in bound.items():same(sha(p),digest,"Bound input/output changed during audit: "+p)
    same(sha(receipt_path),expected_receipt_sha256,"Completion receipt changed during audit")
    bound[str(receipt_path)]=expected_receipt_sha256;bound[str(Path(__file__).resolve())]=sha(__file__)
    result=dict(schema="seaqr.accuracy-v39-independent-continuity-audit.v1",verified=True,
        experiment=str(experiment),completion_receipt_sha256=expected_receipt_sha256,
        freeze_sha256=sha(experiment/"freeze.json"),auditor_sha256=sha(__file__),
        checked_files_sha256=bound,all_bound_files_rehashed_after_analysis=True,
        original_journal_frames=sum(c["frames"] for c in clips.values()),clips=clips,references=overall,
        every_baseline_qualified_state_preserved_exactly=True,
        every_added_state_is_actual_same_identity_with_unbroken_bounded_strict_anchor=True,
        independent_three_row_history_reconstruction=True,producer_policy_or_scorer_imported=False,
        independent_maximum_cardinality_matching=True,all_gate_neighbors_and_assignments_verified=True,
        assignment_tie_rule_reexecuted=False,physical_identity_inferred=False,
        source_media_opened=False,raw16_accessed=False,sealed_holdouts_accessed=False,remote_accessed=False,
        airborne_accuracy_established=False,false_positive_rate_estimated=False,classifier_promoted=False,
        interpretation="Additional class-unknown measured output workload, not new detector hits or airborne recall",
        synthetic_self_tests=tests)
    with output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),verified=True,added_measured_states={k:v["counts"]["added_degraded_measured"] for k,v in clips.items()})),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-completion-sha256",required=True)
    args=parser.parse_args();run(args.experiment,args.output,args.expected_completion_sha256)
