"""Freeze synthetic study then uniformly score all V38 saved current patches.

No new media access, localization replacement, family-selection policy, output
filter or accuracy claim. References group results only after inference.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from accuracy_v39_shape_study import SourceShapeDiagnostic, FAMILIES, run_study
from evaluate_accuracy_v36 import ROOT, read, write, sha
from diagnose_accuracy_v37_temporal import bind, recheck

BASE=ROOT/"results/tiny_target/accuracy_v39_20260925"
V38=ROOT/"results/tiny_target/accuracy_v38_20260925"
AUDIT=V38/"global_multilag_saved_evidence_audit_01.json"
AUDIT_SHA="ec5f682215f5561939673fe7ad6dd7810128aa0a0cb301624e92bd4c05c0bd5d"
PARENT=V38/"global_multilag_01"
CODE=("scripts/evaluate_accuracy_v39_shapes.py", "scripts/accuracy_v39_shape_study.py",
      "tests/unit/test_accuracy_v39_shape_study.py", "docs/accuracy_v39_plan.md")


def compare(actual,expected):
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError("Baseline feature keys changed")
        for key in expected:compare(actual[key],expected[key])
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError("Baseline feature dimensions changed")
        for a,b in zip(actual,expected):compare(a,b)
    elif type(expected) is float:
        if not np.isclose(actual,expected,rtol=1e-12,atol=1e-12):raise ValueError("Frozen baseline feature changed")
    elif type(actual) is not type(expected) or actual!=expected:raise ValueError("Baseline feature value changed")


def scope_summary(records):
    summary=dict(observations=len(records),available=sum(r["diagnostic"]["available"] for r in records),families={})
    for family in FAMILIES:
        fits=[r["diagnostic"]["families"][family]["best"] for r in records if r["diagnostic"]["available"]]
        values=[f["gain_over_best_edge_fraction"] for f in fits if f["gain_over_best_edge_fraction"] is not None]
        summary["families"][family]=dict(fitted_states=len(fits),
            search_boundary_winners=sum(f["search_boundary_winner"] for f in fits),
            compact_coefficient_zero=sum(f["compact_coefficient_on_zero_boundary"] for f in fits),
            conditional_gain_informative_states=len(values),
            median_conditional_gain=float(np.median(values)) if values else None,
            median_fitted_center_offset_px=float(np.median([np.linalg.norm(f["compact"]["center_offset_xy"]) for f in fits])) if fits else None,
            median_compact_amplitude_dn=float(np.median([f["compact_amplitude_dn"] for f in fits])) if fits else None)
    return summary


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Fresh study output required")
    files={};bind(files,AUDIT,AUDIT_SHA)
    bind(files,BASE/"validation_coverage_plan_v1.json","cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc")
    audit=read(AUDIT)
    if not audit["verified"] or audit["selected_observations"]!=364:raise ValueError("Verified exact V38 source evidence required")
    for path,digest in audit["checked_files_sha256"].items():
        if Path(path).suffix.lower() not in (".avi",".mp4",".raw16"):bind(files,path,digest)
    for name in CODE:bind(files,ROOT/name)
    output.mkdir(parents=True)
    with (output/"unit.log").open("x") as log:
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests/unit","-p","test_accuracy_v39_shape_study.py","-v"],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in CODE:
        destination=output/"implementation"/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination);bind(files,destination,files[str(ROOT/name)])
    write(output/"synthetic_study.json",run_study())
    bind(files,output/"synthetic_study.json");bind(files,output/"unit.log");recheck(files)
    freeze=dict(schema="seaqr.accuracy-v39-source-shape-freeze.v1",created_ns=time.time_ns(),pre_real_patch_scoring=True,
        inputs_sha256=files.copy(),all_selected_current_patches=364,families=list(FAMILIES),
        media_opened=False,reference_coordinates_are_model_inputs=False,classifier_promoted=False)
    write(output/"freeze.json",freeze);bind(files,output/"freeze.json")
    parent=read(PARENT/"observations.json");selection=read(PARENT/"selection.json")
    if [r["key"] for r in parent]!=[o["key"] for o in selection["observations"]]:raise ValueError("Changed selection order")
    records=[];used=set()
    with np.load(PARENT/"source_patches.npz",allow_pickle=False) as archive:
        for i,record in enumerate(parent):
            desc=record["current25_array"];key=desc["array_key"]
            if key in used:raise ValueError("Repeated current source patch")
            used.add(key);patch=archive[key]
            if (patch.shape!=(25,25) or patch.dtype!=np.float64
                    or hashlib.sha256(patch.tobytes()).hexdigest()!=desc["sha256"]):
                raise ValueError("Changed saved current source pixels")
            diagnostic=SourceShapeDiagnostic(record["current_xy"]).measure(patch,record["polarity"])
            compare(diagnostic["baseline"],record["lags"][0]["evidence"]["current_source"]["features"])
            records.append(dict(key=record["key"],clip=record["clip"],frame=record["frame"],identity=record["identity"],
                original_measurement_source_xy=record["measurement_source_xy"],
                source_patch_sha256=desc["sha256"],diagnostic=diagnostic))
            if (i+1)%50==0:print(f"Scored {i+1}/{len(parent)} saved current patches",flush=True)
    scopes=defaultdict(set);by_key={r["key"]:r for r in records}
    for group in selection["groups"]:scopes[group["kind"]+"/"+group["window"]].update(group["keys"])
    for observation in selection["observations"]:
        scopes["all"].add(observation["key"])
        scopes["clip/"+observation["clip"]].add(observation["key"])
        for index in observation["provisional_control_indices"]:scopes["control/"+str(index)].add(observation["key"])
    write(output/"observations.json",records);bind(files,output/"observations.json");recheck(files)
    reference_groups=[dict(kind=g["kind"],window=g["window"],frame=g["frame"],
        baseline_selected_keys=g["keys"],
        available_shape_keys=[key for key in g["keys"] if by_key[key]["diagnostic"]["available"]])
        for g in selection["groups"]]
    references={kind:dict(samples=sum(g["kind"]==kind for g in reference_groups),
        baseline_matched_samples=sum(g["kind"]==kind and bool(g["baseline_selected_keys"]) for g in reference_groups),
        shape_available_samples=sum(g["kind"]==kind and bool(g["available_shape_keys"]) for g in reference_groups),
        missing_samples_recovered=False,airborne_truth=False)
        for kind in ("dense","pilot","anchor","compact_light")}
    summary=dict(schema="seaqr.accuracy-v39-source-shape-summary.v1",completed=True,
        selected_observations=len(records),unchanged_baseline_features_crosschecked=len(records),
        references=references,reference_groups=reference_groups,
        scopes={name:scope_summary([by_key[key] for key in sorted(keys)]) for name,keys in sorted(scopes.items())},
        freeze_sha256=sha(output/"freeze.json"),observations_sha256=sha(output/"observations.json"),
        family_selection_applied=False,classifier_promoted=False,measurement_coordinates_replaced=False,
        real_video_frames_decoded=0,airborne_accuracy_established=False,raw16_accessed=False,sealed_holdouts_accessed=False,
        limitations=["Same pixels used for center/shape search and scoring; gains are not calibrated",
            "Finite center banks cannot repair arbitrary offsets; boundary and alternative fits remain explicit",
            "No new classification, no fewer reported responses, no recovered detector/qualification outputs",
            "The model has no truth coordinates, history, clip IDs or class labels as inputs"])
    write(output/"summary.json",summary)
    bind(files,output/"summary.json");recheck(files)
    write(output/"completion_receipt.json",dict(completed=True,all_inputs_and_outputs_rehashed=True,
        checked_files_sha256=files,source_videos_opened=False,classifier_promoted=False))
    print(json.dumps(dict(completed=True,selected_observations=len(records))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute-study",action="store_true");args=p.parse_args()
    if not args.execute_study:p.error("Review before explicit --execute-study")
    run(args.output)
