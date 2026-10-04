#!/usr/bin/env python3
"""Independent complete saved-proposal tracker audit; never replay detection.

Only JSON/JSONL and one source-pinned pure retention function are read. Internal
state determinism is an original-runtime hash attestation, not state recomputed
by this metadata checker. A failed scientific guard remains a valid result.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re

SCHEMA="seaqr.tracker-maturity-shadow.summary.v1"
CLIPS=("0170","0240")
OLD="/tmp/seaqr_tracker_baseline_20261001_6DvV6F"
TRACE="/tmp/seaqr_feature_residual_trace_20260930_1lWybU"
OLD_FREEZE_SHA="c13c8bf0e40cbaecfa16c82d7f9f5fe30f6980bd9fcb2eac6bf06d3685e233ea"
BASE_SHA="6a26937ef03821db05a96f680d899234436e49a864c9d287628872e471c20c73"
JETSON_SHA="2a359b366ec487095456138eb8e7e108678e0d162bac38be82b207b9e3e0f891"
CANDIDATE_SHA="e4908f953b10c0385b3c66cd30d1bb4c9354eb2d25f8907a9fa3b9869d65b19c"
CLASS_SHA="68ae5acf6f23e998dcd55e035eef66301904603ac12b2aefbc40c6ea89960b3c"
SCORER_SHA="d8a4504b0ba739f1319b34f2be48de8b60292d9c2e74cebad3ac222f2ad2d50b"
REGRESSION_SHA="10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"
SAFETY_SHA="a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
METHOD_SHA="571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325"
OLD_RESULTS={"0170":"1b5f6e4914efb63f002ceea94e90a1d352aebe3ef996213f05763c01309edb96",
             "0240":"55f3fdd6511caa936846de4802cc3dfa539200de726167642e9fe92c5dbcf764"}
OLD_STATES={"0170":"5fedda90efa88f9f80f43386ea9042027478bfe53ab4bb94c78cad1b3eb5fe79",
            "0240":"622583ccaa09ea0356e35d119cd558cfa1ba61327f89d99d6311052ac4b3b75f"}
ARCHIVE_FILES={"0170":{
    "run/frames.jsonl":"56687056477c79f5cb9aa8c338c260438966c706fcd393ff29afa5536029eca5",
    "run/launch.json":"b4bfe0263c7f6aa216722b70bd909d211c95e26e0e2fc87f75c82bf0e3577a7b",
    "execution_receipt.json":"7e19411fb7622b02fa4900ec20e8d7ce44208c10da5c5cc6f431187601208a53"},
    "0240":{
    "run/frames.jsonl":"f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92",
    "run/launch.json":"c84c212c1d40200dc783924366a2bec422a053f34bd23ede3951e5bb9b6af869",
    "execution_receipt.json":"a0062c34858f827860a106b75e4a29ad3c0a5e7e09cea3e12fb492ac3766ed28"}}
FILES={"check_tracker_maturity_shadow_v1.py","test_tracker_maturity_shadow_v1.py",
    "tracker_maturity_candidate_v1.py","test_tracker_maturity_candidate_v1.py",
    "batch_tracker_maturity_shadow_v1.py","test_batch_tracker_maturity_shadow_v1.py",
    "batch_discovery_pair.py","compare_discovery_feature_supply.py","discovery_pair_regression_20260929.json"}
WINDOWS={"full":(0,672),"burst":(50,105)}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):digest.update(block)
    return digest.hexdigest()


def decode(text):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"Duplicate JSON key");result[key]=value
        return result
    def number(text):
        value=float(text);require(math.isfinite(value),"Nonfinite JSON");return value
    return json.loads(text,object_pairs_hook=pairs,parse_float=number,
        parse_constant=lambda _:require(False,"Nonfinite JSON"))


def regular(path):
    path=Path(path).absolute()
    require(path.resolve()==path and path.is_file() and not path.is_symlink(),"Regular metadata file required: "+str(path))
    return path


def normalized(value):
    # Byte-identical subset of the frozen baseline serializer for JSON values.
    if isinstance(value,dict):return ["dict",[(normalized(k),normalized(v)) for k,v in sorted(value.items(),key=lambda item:repr(item[0]))]]
    if isinstance(value,list):return ["list",[normalized(v) for v in value]]
    if type(value) is float:return ["float",value.hex()]
    require(type(value) in (int,str,bool) or value is None,"Non-JSON observable value")
    return value


def value_sha(value):
    return hashlib.sha256(json.dumps(normalized(value),separators=(",",":"),allow_nan=False).encode()).hexdigest()


def first_difference(a,b,path=()):
    if type(a) is not type(b):return dict(path=list(path),kind="type",expected_type=type(a).__name__,actual_type=type(b).__name__,expected=a,actual=b)
    if isinstance(a,dict):
        if a.keys()!=b.keys():return dict(path=list(path),kind="keys",missing=sorted(a.keys()-b.keys()),extra=sorted(b.keys()-a.keys()))
        for key in sorted(a):
            found=first_difference(a[key],b[key],path+(key,))
            if found is not None:return found
    elif isinstance(a,list):
        if len(a)!=len(b):return dict(path=list(path),kind="length",expected=len(a),actual=len(b))
        for i,(x,y) in enumerate(zip(a,b)):
            found=first_difference(x,y,path+(i,))
            if found is not None:return found
    elif type(a) is float:
        if a.hex()!=b.hex():return dict(path=list(path),kind="float_bits",expected=a,actual=b,expected_hex=a.hex(),actual_hex=b.hex(),absolute_difference=abs(a-b))
    elif a!=b:return dict(path=list(path),kind="value",expected=a,actual=b)
    return None


def protocol():
    return dict(policy="maturity_first_eligible_v1",clips=list(CLIPS),frames_per_clip=673,nominal_timestamp_step_ns=100000000,
        candidate_sha256=CANDIDATE_SHA,candidate_class_source_sha256=CLASS_SHA,original_baseline_workspace=OLD,
        original_baseline_freeze_sha256=OLD_FREEZE_SHA,original_baseline_results=OLD_RESULTS.copy(),original_baseline_states=OLD_STATES.copy(),
        baseline_exact_archive_and_old_state_required=True,candidate_copies=2,first_learning_divergence_recorded_before_update=True,
        target=dict(clip="0240",first=430,last_inclusive=464,samples=35,radius_native_px=8,polarity="dark",actual=True,
            qualified=True,coherent_all_samples=True,ambiguous_samples_must_be_zero=True,baseline_derived_not_independent_truth=True),
        workload_windows={name:list(window) for name,window in WINDOWS.items()},scientific_failure_completes=True,
        no_threshold_search=True,source_media_opened=False,detector_replayed=False,production_promotion=False)


def validate_freeze(freeze):
    require(set(freeze)=={"schema","protocol","files"} and freeze["schema"]=="tracker_maturity_shadow.v1"
        and first_difference(protocol(),freeze["protocol"]) is None and set(freeze["files"])==FILES,
        "Frozen candidate protocol/inventory differs")
    protected={"tracker_maturity_candidate_v1.py":CANDIDATE_SHA,"compare_discovery_feature_supply.py":SCORER_SHA,
        "discovery_pair_regression_20260929.json":REGRESSION_SHA,"batch_discovery_pair.py":SAFETY_SHA}
    require(all(freeze["files"][key]==expected for key,expected in protected.items())
        and all(isinstance(h,str) and re.fullmatch(r"[a-f0-9]{64}",h) for h in freeze["files"].values()),"Protected sources differ")


class Workload:
    """Recompute counts from saved public records, not predicted class labels."""
    def __init__(self):
        self.sums={name:Counter() for name in WINDOWS};self.identities={};self.alive=set()

    def add(self,row):
        frame=row["frame_index"];tracks=row["tracks"];metrics=row["tracking_metrics"]
        values=Counter(frames=1,ready_frames=int(row["coverage"]["detection_ready"]),candidates=len(row["candidates"]),active_tracks=len(tracks))
        hits=Counter();occupancy={p:Counter() for p in ("bright","dark")};self.alive=set()
        for record in tracks:
            key=f"{row['segment']}/{record['track_id']}"
            require(key not in self.alive and record["segment"]==row["segment"],"Duplicate/mismatched track identity")
            self.alive.add(key);count=record["independent_hits"]
            require(type(count) is int and count>=1 and type(record["measured"]) is bool
                and type(record["qualified_moving"]) is bool,"Invalid track evidence")
            prior=self.identities.setdefault(key,dict(first_frame=frame,last_frame=frame,maximum_hits=0))
            prior.update(last_frame=frame,maximum_hits=max(prior["maximum_hits"],count));hits[str(count)]+=1
            values["qualified_measured" if record["measured"] else "qualified_predicted"]+=int(record["qualified_moving"])
            polarity=record["track_id"].split(":")[0];require(polarity in occupancy,"Unknown polarity")
            x,y=record["reference_xy"];occupancy[polarity][f"{math.floor(x/256)},{math.floor(y/256)}"]+=1
        for polarity in ("bright","dark"):
            item=metrics.get(polarity,{})
            require(sum(occupancy[polarity].values())<=256,"Active capacity exceeded")
            if item:require(item["max_active_tracks"]==256 and item["active_track_count"]==sum(occupancy[polarity].values()),"Active count differs")
            for source,dest in (("birth_count","births"),("deleted_track_count","deletions"),("dropped_birth_count_at_active_track_cap","dropped_births")):
                values[dest]+=item.get(source,0)
            values.update({"lifecycle_"+k:v for k,v in item.get("lifecycle_counts",{}).items()})
            admission=item.get("birth_admission",{})
            require(admission.get("confirmed_tracks_evicted",0)==0,"Confirmed track evicted")
            values["replacements"]+=len(admission.get("tentative_replacements",[]))
        for name,(first,last) in WINDOWS.items():
            if first<=frame<=last:self.sums[name].update(values)
        return dict(counts=dict(values),independent_hit_histogram=dict(sorted(hits.items())),
            cell_occupancy_by_polarity={p:dict(sorted(c.items())) for p,c in occupancy.items()})

    def finish(self):
        result={}
        for name,(first,last) in WINDOWS.items():
            counts=dict(self.sums[name]);cohort={k:r for k,r in self.identities.items() if first<=r["first_frame"]<=last}
            result[name]=dict(counts=counts,mean_active_tracks=counts.get("active_tracks",0)/counts["frames"] if counts.get("frames") else None,
                first_seen_identity_count=len(cohort),first_seen_maximum_independent_hits_histogram=dict(sorted(Counter(str(r["maximum_hits"]) for r in cohort.values()).items())),
                never_reached_four_hits_by_end_of_clip=sum(r["maximum_hits"]<4 for r in cohort.values()),
                still_active_without_four_hits_at_end_of_clip=sum(r["maximum_hits"]<4 and k in self.alive for k,r in cohort.items()),
                cohort_followed_until_end_of_clip=True)
        return result


def inspect_rows(archive,candidate,audit,old_states,expected_frames=673):
    collectors={arm:Workload() for arm in ("baseline","candidate")}
    target={arm:[] for arm in collectors};first_learning=first_output=None;count=0
    sentinel=object()
    for index,items in enumerate(itertools.zip_longest(archive,candidate,audit,old_states,fillvalue=sentinel)):
        require(index<expected_frames and all(x is not sentinel for x in items),"Missing/extra shadow frames")
        old,new,check,state=items
        require(all(type(x["frame_index"]) is int and x["frame_index"]==index for x in items)
            and old["timestamp_ns"]==new["timestamp_ns"]==index*100000000
            and old["coverage"]["full_shape_hw"]==[3190,4784],"Frame order/time/native shape differs")
        require(set(new)==set(old)|{"shadow"},"Candidate journal fields differ")
        require(first_difference({k:v for k,v in old.items() if k not in ("tracks","tracking_metrics")},
            {k:v for k,v in new.items() if k not in ("tracks","tracking_metrics","shadow")}) is None,
            "Saved proposals, geometry, motion, coverage or archived timings changed")
        outputs=[{k:r[k] for k in ("tracks","tracking_metrics")} for r in (old,new)]
        hashes=list(map(value_sha,outputs))
        require(check["output_sha256"]==[hashes[0],hashes[1],hashes[1]] and state["output_sha256"]==[hashes[0]]*2
            and state["archive_output_sha256"]==hashes[0] and state["differences"]=={},"Archived/candidate observable hashes differ")
        internal,learning=check["internal_state_sha256"],check["learning_centers_sha256"]
        require(len(internal)==len(learning)==3 and all(isinstance(h,str) and re.fullmatch(r"[a-f0-9]{64}",h) for h in internal+learning)
            and internal[1]==internal[2] and state["internal_state_sha256"]==[internal[0]]*2
            and learning[1]==learning[2] and state["learning_centers_sha256"]==[learning[0]]*2
            and state["archive_derived_learning_sha256"]==learning[0],"Baseline state/learning or candidate determinism hash mismatch")
        if first_learning is None and learning[0]!=learning[1]:
            first_learning=dict(frame_index=index,measured_before_frame_update=True,baseline_sha256=learning[0],candidate_sha256=learning[1],
                consequence="Saved proposals remain fixed; this is not a causal full-pipeline result.")
        if first_output is None and hashes[0]!=hashes[1]:first_output=dict(frame_index=index,detail=first_difference(*outputs))
        diverged=first_learning is not None
        require(check["past_or_at_learning_divergence"] is diverged and new["shadow"]==dict(saved_proposals=True,
            candidate_policy="maturity_first_eligible_v1",detector_coverage_motion_and_timings_are_archived_not_rerun=True,
            past_or_at_learning_divergence=diverged),"Pre-update learning divergence annotation differs")
        for arm,r in (("baseline",old),("candidate",new)):
            measured=collectors[arm].add(r)
            require(first_difference(check["workload"][arm],measured) is None,"Saved frame workload differs: "+arm)
            if 430<=index<=464:target[arm].append(dict(frame_index=index,segment=r["segment"],tracks=r["tracks"],detection_ready=r["coverage"]["detection_ready"]))
        count+=1
    require(count==expected_frames,"Incomplete shadow")
    return dict(frames=count,first_learning_divergence=first_learning,first_output_divergence=first_output,
        workload={name:c.finish() for name,c in collectors.items()}),target


def target_guard(scorer,rows,references):
    baseline=scorer.retention(rows["baseline"],references,radius=8.,polarity="dark")
    candidate=scorer.retention(rows["candidate"],references,radius=8.,polarity="dark")
    require(baseline["preservation_guard_passed"] and baseline["ambiguous_frames"]==0
        and baseline["any_identity_matched_frames"]==35,"Frozen baseline target does not reproduce")
    details=[]
    for old,new in zip(baseline["details"],candidate["details"]):
        require(old["frame_index"]==new["frame_index"],"Target frame inventory differs")
        details.append(dict(frame_index=old["frame_index"],baseline=old,candidate=new,
            newly_lost=bool(old["matched_identities"]) and not new["matched_identities"],
            recovered=not old["matched_identities"] and bool(new["matched_identities"])))
    return dict(baseline=baseline,candidate=candidate,per_sample=details,
        newly_lost_frames=[r["frame_index"] for r in details if r["newly_lost"]],recovered_frames=[r["frame_index"] for r in details if r["recovered"]],
        scientific_guard_passed=candidate["preservation_guard_passed"] and candidate["ambiguous_frames"]==0 and candidate["any_identity_matched_frames"]==35)


def compare_target(saved,actual):
    """Guard outcomes/coordinates exact; expose platform hypot rounding separately."""
    differences=[]
    def clean(value,path=()):
        if isinstance(value,dict):return {k:clean(v,path+(k,)) for k,v in value.items() if k!="distance_native_px"}
        if isinstance(value,list):return [clean(v,path+(i,)) for i,v in enumerate(value)]
        return value
    require(first_difference(clean(saved),clean(actual)) is None,"Target identities/coordinates/guard differ from receipt")
    def walk(a,b):
        if isinstance(a,dict):
            require(isinstance(b,dict) and a.keys()==b.keys(),"Target detail fields differ")
            for k in a:
                if k=="distance_native_px":
                    require(type(a[k]) in (int,float) and math.isfinite(a[k]) and 0<=a[k]<=8
                        and type(b[k]) in (int,float) and math.isfinite(b[k]) and 0<=b[k]<=8,"Invalid native matching distance")
                    require(abs(a[k]-b[k])<=max(math.ulp(float(a[k])),math.ulp(float(b[k]))),
                            "Target matching distance differs beyond one rounding ULP")
                    differences.append(abs(a[k]-b[k]))
                else:walk(a[k],b[k])
        elif isinstance(a,list):
            require(isinstance(b,list) and len(a)==len(b),"Target detail lengths differ")
            for x,y in zip(a,b):walk(x,y)
    walk(saved,actual)
    return dict(decisions_and_coordinates_exact=True,distance_values_exact=all(x==0 for x in differences),
        maximum_distance_arithmetic_difference_px=max(differences,default=None),scientific_gate_tolerance_relaxed=False)


class Evidence:
    def __init__(self):self.hashes={}
    def bind(self,path,expected=None):
        path=regular(path);digest=sha(path)
        require(expected is None or digest==expected,"Evidence hash differs: "+str(path))
        self.hashes[str(path)]=digest;return path
    def read(self,path,expected=None):
        path=self.bind(path,expected);value=decode(path.read_text())
        require(sha(path)==self.hashes[str(path)],"Evidence changed during read")
        return value
    def rows(self,path):
        with regular(path).open() as stream:
            for line in stream:
                require(len(line)<=32*1024*1024,"Oversize metadata row")
                yield decode(line)
    def verify(self):
        require(all(sha(regular(p))==h for p,h in self.hashes.items()),"Input changed during summary")


def validate_runtime(actual,reference):
    require(all(first_difference(reference.get(k),actual.get(k)) is None for k in
        ("blas","affinity","numpy","opencv","opencv_threads","thread_environment","clock_ticks")),"Original numerical runtime metadata differs")


def child_common(row,workspace,digest,clip,old,expected_inputs):
    require(row.get("schema")=="seaqr.tracker-maturity-shadow.v1"+(".preflight" if clip is None else ".run")
        and row.get("workspace")==workspace and row.get("freeze_sha256")==digest and row.get("clip")==clip
        and row.get("passed") is True and row.get("error") is None and row.get("inputs_unchanged_after_check") is True
        and row.get("clock_controls_unchanged") is True and row.get("clock_policy_before")==row.get("clock_policy_after"),"Child identity/integrity differs")
    require(all(row.get(k) is False for k in ("source_media_opened","detector_replayed","raw16_or_holdouts_accessed",
        "production_promotion","scientific_improvement_claimed")) and row.get("saved_proposals_shadow_only") is True,"Child scope differs")
    require(row.get("inputs_sha256")==expected_inputs,"Child input bindings differ")
    reference=old[clip or "0170"]
    for key in ("runtime_before",) if clip is None else ("runtime_before","runtime_after"):
        validate_runtime(row[key],reference[key])


def summarize(directory,bundle,baseline_directory,archive_directory,freeze_sha256):
    roots=[Path(p).absolute() for p in (directory,bundle,baseline_directory,archive_directory)]
    require(all(p.resolve()==p and p.is_dir() and not p.is_symlink() for p in roots),"Regular input directories required")
    directory,bundle,baseline_directory,archive_directory=roots
    require(isinstance(freeze_sha256,str) and re.fullmatch(r"[a-f0-9]{64}",freeze_sha256),"Caller freeze required")
    evidence=Evidence();summary_path=evidence.bind(Path(__file__).resolve())
    freeze=evidence.read(bundle/"freeze.json",freeze_sha256);validate_freeze(freeze)
    evidence.bind(directory/"freeze.json",freeze_sha256)
    for name,digest in freeze["files"].items():evidence.bind(bundle/name,digest)
    scorer_path=bundle/"compare_discovery_feature_supply.py"
    spec=importlib.util.spec_from_file_location("maturity_frozen_pure_retention",scorer_path)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    batch=evidence.read(directory/"batch_status.json")
    require(batch.get("schema")=="seaqr.tracker-maturity-shadow.batch.v1" and batch.get("complete") is True
        and batch.get("execution_passed") is True and batch.get("bundle_and_child_artifacts_unchanged") is True
        and batch.get("error") is None and batch.get("current") is None and batch.get("not_run")==[]
        and batch.get("freeze_sha256")==freeze_sha256 and batch.get("source_media_accessed") is False
        and batch.get("production_changed") is False and batch.get("shadow_only") is True
        and batch.get("scientific_improvement_claimed") is False and batch.get("automatic_retries")==0,"Complete unchanged shadow batch required")
    require(batch.get("execution")==dict(workers=1,children=3,start_below_celsius=65,stop_at_celsius=75,batch_deadline_seconds=3900,automatic_retries=0)
        and 0<=batch["elapsed_seconds"]<3900,"Execution bounds differ")
    phases=batch.get("phases",[])
    require([p["name"] for p in phases]==["preflight","run_0170","run_0240"] and len({p["pid"] for p in phases})==3
        and all(type(p["pid"]) is int and p["pid"]>0 and p["returncode"]==0 for p in phases),"Three fresh completed child processes required")
    command=phases[0]["command"];require(len(command)==9 and re.fullmatch(r"/tmp/seaqr_tracker_maturity_20261001_[A-Za-z0-9]{6}",command[5]),"Invalid execution workspace")
    workspace=command[5]
    for phase,args,limit in zip(phases,(["--preflight"],["--clip","0170"],["--clip","0240"]),(300,1800,1800)):
        require(phase["command"]==["/usr/bin/python3","-I","-u",workspace+"/check_tracker_maturity_shadow_v1.py","--workspace",workspace,
            "--freeze-sha256",freeze_sha256,*args] and 0<=phase["elapsed_seconds"]<limit,"Recorded child command/deadline differs")
    remote_inputs={workspace+"/freeze.json":freeze_sha256,**{workspace+"/"+n:h for n,h in freeze["files"].items()},
        OLD+"/freeze.json":OLD_FREEZE_SHA,OLD+"/check_tracker_capacity_shadow.py":BASE_SHA,OLD+"/check_tracker_capacity_jetson.py":JETSON_SHA}
    evidence.bind(baseline_directory/"freeze.json",OLD_FREEZE_SHA)
    old={}
    for clip in CLIPS:
        old[clip]=evidence.read(baseline_directory/clip/"result.json",OLD_RESULTS[clip])
        evidence.bind(baseline_directory/clip/(clip+"_state_hashes.jsonl"),OLD_STATES[clip])
        value=old[clip];replay=value["replay"]
        require(value["schema"]=="seaqr.tracker-baseline-jetson.v1.run" and value["passed"] is True and value["error"] is None
            and value["workspace"]==OLD and value["freeze_sha256"]==OLD_FREEZE_SHA and value["clip"]==clip
            and value["inputs_unchanged_after_check"] is True and value["clock_controls_unchanged"] is True
            and replay["passed"] is True and replay["first_difference"] is None
            and all(replay[k]==673 for k in ("attempted_frames","exact_archive_frames","dual_state_exact_frames","derived_learning_exact_frames"))
            and replay["state_hashes_sha256"]==OLD_STATES[clip],"Pinned original baseline proof incomplete")
        remote_inputs.update({OLD+"/"+clip+"/result.json":OLD_RESULTS[clip],OLD+"/"+clip+"/"+clip+"_state_hashes.jsonl":OLD_STATES[clip]})
        remote_inputs.update({p:h for p,h in value["inputs_sha256"].items() if Path(OLD) not in Path(p).parents})
        for relative,h in ARCHIVE_FILES[clip].items():
            evidence.bind(archive_directory/clip/relative,h)
            require(remote_inputs[TRACE+"/"+clip+"/"+relative]==h,"Original archive provenance differs")
    pre=evidence.read(directory/"preflight.json");child_common(pre,workspace,freeze_sha256,None,old,remote_inputs)
    require(pre.get("frames_processed")==0,"Preflight processed frames")
    artifacts={workspace+"/preflight.json":evidence.hashes[str(directory/"preflight.json")]}
    intake=evidence.read(bundle/"discovery_pair_regression_20260929.json",REGRESSION_SHA);positive=intake["positive_pass"]
    references=positive["baseline_measurements"]
    require(intake["scope"]["allowed_clip_ids"]==list(CLIPS) and positive["clip_id"]=="0240"
        and [r["frame_index"] for r in references]==list(range(430,465)),"Frozen reference inventory differs")
    results={}
    for clip in CLIPS:
        path=directory/clip/"result.json";row=evidence.read(path)
        child_common(row,workspace,freeze_sha256,clip,old,{**remote_inputs,**artifacts})
        require(row["preflight_sha256"]==artifacts[workspace+"/preflight.json"]
            and (clip!="0240" or row["prior_clip_sha256"]==artifacts[workspace+"/0170/result.json"]),"Sequential child binding differs")
        replay=row["replay"]
        require(replay["passed"] is True and replay["shadow_only"] is True and replay["no_tolerance_relaxation"] is True
            and all(replay[k]==673 for k in ("attempted_frames","exact_archive_frames","exact_previous_state_frames","deterministic_candidate_frames","baseline_learning_exact_frames"))
            and replay["separate_adapter_owners_verified"] is True and len(replay["adapter_runs"])==3,"Incomplete parity/determinism receipt")
        for counters in replay["adapter_runs"]:
            require(all(counters[k]==0 for k in ("geometry_calls","geometry_fallbacks","batch_fallbacks","innovation_fallbacks"))
                and counters["batch_track_rows"]==counters["innovation_tracks"],"Unexpected native fallback")
        require(len(replay["candidate_bindings"])==2,"Two candidate bindings required")
        for binding in replay["candidate_bindings"]:
            require(binding["sha256"]==CANDIDATE_SHA and binding["class_source_sha256"]==CLASS_SHA
                and binding["unchanged_transformed_source_sha256"]==METHOD_SHA and binding["changed_global_bindings"]==["_VictimIndex_v28"]
                and binding["eligibility_changed"] is False and binding["configuration_changed"] is False
                and binding["original_scope_unchanged"] is True,"Candidate implementation/protection binding differs")
        expected_paths={workspace+"/"+clip+"/"+name for name in ("candidate_frames.jsonl","shadow_audit.jsonl")}
        require(set(replay["artifacts_sha256"])==expected_paths,"Unexpected shadow artifact inventory")
        for name in ("candidate_frames.jsonl","shadow_audit.jsonl"):
            remote=workspace+"/"+clip+"/"+name;evidence.bind(directory/clip/name,replay["artifacts_sha256"][remote])
        checked,target=inspect_rows(evidence.rows(archive_directory/clip/"run/frames.jsonl"),evidence.rows(directory/clip/"candidate_frames.jsonl"),
            evidence.rows(directory/clip/"shadow_audit.jsonl"),evidence.rows(baseline_directory/clip/(clip+"_state_hashes.jsonl")))
        for key in ("first_learning_divergence","first_output_divergence","workload"):
            require(first_difference(replay[key],checked[key]) is None,"Recomputed receipt differs: "+clip+" "+key)
        checked["target"]=target_guard(scorer,target,references) if clip=="0240" else None
        checked["target_receipt_comparison"]=compare_target(replay["target"],checked["target"]) if clip=="0240" else None
        require(replay["scientific_guard_passed"]==(checked["target"]["scientific_guard_passed"] if clip=="0240" else None)
            and (clip!="0170" or replay["target"] is None),"Scientific target guard differs")
        checked["workload_delta"]={name:{k:checked["workload"]["candidate"][name]["counts"].get(k,0)-checked["workload"]["baseline"][name]["counts"].get(k,0)
            for k in sorted(set(checked["workload"]["candidate"][name]["counts"])|set(checked["workload"]["baseline"][name]["counts"]))} for name in WINDOWS}
        checked["metadata_integrity_passed"]=True;results[clip]=checked
        artifacts.update(replay["artifacts_sha256"]);artifacts[workspace+"/"+clip+"/result.json"]=evidence.hashes[str(path)]
    require(batch["child_artifacts_sha256"]==artifacts,"Batch child artifact bindings differ")
    telemetry_path=evidence.bind(directory/"telemetry.jsonl");peaks={};samples=0
    for sample in evidence.rows(telemetry_path):
        require(sample["phase"] in [p["name"] for p in phases] and sample["temperatures_c"],"Invalid telemetry")
        for zone,value in sample["temperatures_c"].items():
            require(type(value) in (int,float) and math.isfinite(value) and value<75,"Invalid/excess temperature")
            peaks[zone]=max(peaks.get(zone,value),value)
        samples+=1
    require(samples>0,"Missing thermal telemetry");evidence.verify()
    scientific=results["0240"]["target"]["scientific_guard_passed"]
    return dict(schema=SCHEMA,completed=True,metadata_only=True,integrity_passed=True,source_media_opened=False,
        detector_replayed=False,numerical_tracker_runs=0,production_promotion=False,freeze_sha256=freeze_sha256,
        execution_workspace=workspace,summary_source_sha256=evidence.hashes[str(summary_path)],input_sha256=evidence.hashes,
        frames_checked=1346,clips=results,target_scientific_guard_passed=scientific,
        outcome="target_guard_passed_shadow_only" if scientific else "target_guard_failed_no_promotion",
        elapsed_seconds_not_pipeline_throughput=batch["elapsed_seconds"],thermal=dict(samples=samples,observed_peak_c=max(peaks.values()),per_zone_peak_c=peaks),
        limitations=["Saved proposals are fixed: learning-protection feedback after the first pre-update divergence was not causally rerun.",
            "Original archive observable hashes are independently recomputed; internal states and second candidate execution are checked through original-runtime saved hash attestations.",
            "The35-frame guard is baseline-derived, not independent object truth, broad recall, airborne classification or false-positive rate.",
            "Track, birth, replacement and qualified-output counts describe workload. Lower counts can also hide targets; greater maturity can preserve clutter.",
            "This three-tracker metadata workload does not measure pipeline speed. No threshold tuning, retry or production promotion follows from this summary.",
            "Recorded thermal samples do not establish continuous peak temperature between samples."])


def save_summary(directory,bundle,baseline_directory,archive_directory,freeze_sha256,output):
    output=Path(output).absolute()
    require(output.suffix==".json" and output.parent.resolve()==output.parent and output.parent.is_dir()
        and not output.exists() and not output.is_symlink(),"Fresh JSON output required; no overwrite")
    value=summarize(directory,bundle,baseline_directory,archive_directory,freeze_sha256)
    with output.open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("directory","bundle","baseline-directory","archive-directory","output"):parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--freeze-sha256",required=True);args=parser.parse_args()
    value=save_summary(args.directory,args.bundle,args.baseline_directory,args.archive_directory,args.freeze_sha256,args.output)
    print(json.dumps({k:value[k] for k in ("completed","frames_checked","target_scientific_guard_passed","outcome")}))
