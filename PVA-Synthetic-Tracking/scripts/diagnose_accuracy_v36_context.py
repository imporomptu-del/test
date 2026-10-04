"""Frozen, bounded native-source patch diagnostic over v34 observations.

This neither runs nor changes the detector/tracker. Selection is recorded before
any source pixels are evaluated; feature code cannot see selection labels.
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

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from accuracy_v36_context import PointEdgeDiagnostic

SHADOW = ROOT / "results/tiny_target/accuracy_v36_20260924/shadow_01"
SOURCES = {
    "0029": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0029.avi",
    "0126": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0126.avi",
}
IMPLEMENTATION = (
    "scripts/accuracy_v36_context.py", "scripts/diagnose_accuracy_v36_context.py",
    "tests/unit/test_accuracy_v36_context.py", "tests/unit/test_accuracy_v36_context_runner.py",
    "docs/accuracy_v36_context_plan.md",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)


def key(clip, frame, identity):
    return f"{clip}/{frame}/{identity}"


def select():
    frozen = read(SHADOW / "freeze.json")
    controls = read(frozen["references"]["controls"])["controls"]
    observations, groups, inputs = {}, [], {}
    for cid in SOURCES:
        result_path = SHADOW / (cid + "_results.json")
        inputs[str(result_path)] = sha(result_path)
        result = read(result_path)["arms"]["baseline"]
        for kind in ("dense", "pilot"):
            for window in result["references"][kind]["positive_windows"]:
                for evidence in window["evidence"]:
                    frame = evidence["frame_index"]
                    groups.append(dict(kind=kind, clip=cid, window=window["window_id"], frame=frame,
                        keys=[key(cid, frame, i) for i in evidence["all_gated_same_polarity_ids"]],
                        baseline_assigned_id=evidence["assigned_track_id"]))
        for anchor in result["required_anchor_evidence"]:
            groups.append(dict(kind="anchor", clip=cid, window=anchor["event_id"], frame=anchor["frame"],
                keys=[key(cid, anchor["frame"], i) for i in anchor["ids"]], baseline_assigned_id=None))
        wanted = {k for g in groups if g["clip"] == cid for k in g["keys"]}
        journal = Path(frozen["inputs"][cid]["path"]) / "frames.jsonl"
        inputs[str(journal)] = sha(journal)
        if inputs[str(journal)] != frozen["inputs"][cid]["files_sha256"][str(journal)]:
            raise ValueError("Changed parent journal")
        seen = 0
        with journal.open() as source:
            for line in source:
                row = json.loads(line)
                if row["frame_index"] != seen:
                    raise ValueError("Noncontiguous parent journal")
                frame = seen
                seen += 1
                for track in row["tracks"]:
                    if not (track["qualified_moving"] and track["measured"]):
                        continue
                    identity = f'{track["segment"]}/{track["track_id"]}'
                    oid = key(cid, frame, identity)
                    xy = track["measurement_source_xy"]
                    nuisance = []
                    if cid == "0126":
                        for j, control in enumerate(controls):
                            x, y, w, h = control["crop_xywh"]
                            first, last = control["frames_inclusive"]
                            if first <= frame <= last and x <= xy[0] < x+w and y <= xy[1] < y+h:
                                nuisance.append(j)
                    if oid not in wanted and not nuisance:
                        continue
                    if oid in observations:
                        raise ValueError("Duplicate observation")
                    observations[oid] = dict(key=oid, clip=cid, frame=frame, identity=identity,
                        measurement_source_xy=xy, polarity=track["track_id"].split(":")[0],
                        provisional_control_indices=nuisance)
        if seen != frozen["inputs"][cid]["frames"] or not wanted <= observations.keys():
            raise ValueError("Missing selected reference observation")
    for kind, denominator, baseline_hits in (("dense",285,284),("pilot",28,28),("anchor",24,24)):
        chosen = [g for g in groups if g["kind"] == kind]
        if len(chosen) != denominator or sum(bool(g["keys"]) for g in chosen) != baseline_hits:
            raise ValueError("Changed reference denominator or baseline hits")
    if sum(len(o["provisional_control_indices"]) for o in observations.values()) != 70:
        raise ValueError("Changed provisional control workload")
    inputs[str(SHADOW/"freeze.json")] = sha(SHADOW/"freeze.json")
    inputs[str(SHADOW/"summary.json")] = sha(SHADOW/"summary.json")
    for name, path in frozen["references"].items():
        inputs[path] = sha(path)
        if inputs[path] != frozen["reference_sha256"][name]:
            raise ValueError("Changed frozen reference")
    return dict(observations=list(observations.values()), groups=groups, controls=controls), inputs, frozen


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), minimum=float(values.min()), p05=float(np.quantile(values,.05)),
        median=float(np.median(values)), p95=float(np.quantile(values,.95)), maximum=float(values.max()))


def summarize(selection, results):
    by_key = {r["key"]:r for r in results}
    feature_names = ("point_minus_edge_fraction", "point_gain_fraction", "edge_gain_fraction",
        "point_gain_after_edge_fraction", "point_amplitude_dn", "point_after_edge_amplitude_dn", "residual_rms_dn")
    groups_out, feature_groups = [], defaultdict(set)
    for group in selection["groups"]:
        passed = [k for k in group["keys"] if by_key[k]["zero_margin_ablation_passed"]]
        groups_out.append(dict(**group, passing_keys=passed, baseline_hit=bool(group["keys"]),
            diagnostic_hit=bool(passed), new_miss=bool(group["keys"]) and not passed))
        feature_groups[group["kind"]+"/"+group["window"]].update(group["keys"])
    controls = []
    for j, control in enumerate(selection["controls"]):
        chosen = [r for r in results if j in r["provisional_control_indices"]]
        controls.append(dict(**control, selected_measured_states=len(chosen),
            passing_measured_states=sum(r["zero_margin_ablation_passed"] for r in chosen)))
        feature_groups[f'control/{j}/{control["label"]}'].update(r["key"] for r in chosen)
    per_kind = {}
    for kind in ("dense","pilot","anchor"):
        chosen = [g for g in groups_out if g["kind"] == kind]
        per_kind[kind] = dict(samples=len(chosen), baseline_hits=sum(g["baseline_hit"] for g in chosen),
            diagnostic_hits=sum(g["diagnostic_hit"] for g in chosen),
            newly_missed_samples=[dict(clip=g["clip"],window=g["window"],frame=g["frame"]) for g in chosen if g["new_miss"]],
            baseline_ambiguous_samples=sum(len(g["keys"])>1 for g in chosen),
            no_longer_preserves_all_baseline_ids=sum(bool(g["keys"]) and set(g["keys"])!=set(g["passing_keys"]) for g in chosen))
    stats = {name:{feature:distribution([by_key[k]["features"][feature] for k in keys])
        for feature in feature_names} for name,keys in sorted(feature_groups.items())}
    for name, keys in feature_groups.items():
        stats[name]["availability"] = dict(observations=len(keys),
            informative=sum(by_key[k]["features"]["informative"] for k in keys),
            conditional_informative=sum(by_key[k]["features"]["conditional_informative"] for k in keys),
            distributions_include_uninformative_zero_coded_gains=True)
    return dict(reference_retention=per_kind, provisional_controls=controls, groups=groups_out,
        feature_distributions=stats, selected_observations=len(results),
        uninformative_patches=sum(not r["features"]["informative"] for r in results),
        promoted=False, detector_rerun=False, thresholds_tuned=False,
        airborne_precision=None, airborne_recall=None, false_alarms_per_minute=None,
        limitations=["Only selected development observations, no full-output gate replay",
            "Known points have unknown physical class; nuisance controls are provisional and selected",
            "Overlapping references and adjacent frames are not independent samples",
            "Unequal template searches and correlated pixels: no calibrated confidence",
            "Native source pixels differ from warped temporally filtered detector domain",
            "Single-frame evidence alone does not prove independent object motion"])


def run(output, audit_path):
    if output.exists():
        raise FileExistsError("Fresh diagnostic output directory required")
    # Audit file existence is not verification. Its accepted field names are
    # checked explicitly once supplied; failed or incomplete audits fail closed.
    audit = read(audit_path)
    if (audit.get("schema") != "seaqr.accuracy-v36-independent-audit.v1"
            or audit.get("verified") is not True
            or audit.get("experiment_freeze_sha256") != sha(SHADOW/"freeze.json")
            or audit.get("experiment") != str(SHADOW)
            or audit.get("auditor_sha256") != sha(ROOT/"scripts/audit_accuracy_v36.py")):
        raise ValueError("Completed independent shadow audit required")
    selection, inputs, shadow_freeze = select()
    for path, digest in inputs.items():
        if audit["checked_files_sha256"].get(path) != digest:
            raise ValueError("Input not covered by independent audit: " + path)
    for path, digest in audit["checked_files_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Changed independently audited input: " + path)
        inputs[path] = digest
    inputs[str(audit_path.resolve())] = sha(audit_path)
    source_digests = {c:sha(p) for c,p in SOURCES.items()}
    if any(d != shadow_freeze["inputs"][c]["source_sha256"] for c,d in source_digests.items()):
        raise ValueError("Source-video digest mismatch")
    output.mkdir(parents=True)
    with (output/"unit.log").open("x") as log:
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests/unit",
            "-p","test_accuracy_v36_context*.py","-v"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(output/"selection.json", selection)
    freeze = dict(pre_extraction=True, created_ns=time.time_ns(), source_videos={
        c:dict(path=str(SOURCES[c]),sha256=d) for c,d in source_digests.items()}, inputs_sha256=inputs,
        implementation_sha256={p:sha(ROOT/p) for p in IMPLEMENTATION},
        selection_sha256=sha(output/"selection.json"), unit_log_sha256=sha(output/"unit.log"),
        opencv_version=cv2.__version__, numpy_version=np.__version__, raw16_accessed=False,
        opencv_build_information_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        patch_size=25, native_pixels=True, rounding="floor(source_coordinate+0.5)",
        classifier_promoted=False, zero_margin_ablation="informative and point_minus_edge_fraction > 0")
    write(output/"freeze.json", freeze)
    for p in IMPLEMENTATION:
        target = output/"implementation"/p
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/p,target)
    diagnostic = PointEdgeDiagnostic()
    results, patches, capture_records = [], {}, {}
    for cid, source in SOURCES.items():
        selected = defaultdict(list)
        for observation in selection["observations"]:
            if observation["clip"] == cid:
                selected[observation["frame"]].append(observation)
        cap = cv2.VideoCapture(str(source))
        try:
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-10)>1e-6:
                raise ValueError("Source decoder/FPS unavailable or unexpected")
            if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (4784,3190):
                raise ValueError("Unexpected source dimensions")
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != shadow_freeze["inputs"][cid]["frames"]:
                raise ValueError("Unexpected reported source frame count")
            capture_records[cid] = dict(backend=cap.getBackendName(),fps=cap.get(cv2.CAP_PROP_FPS),
                width=4784,height=3190,decoded_through_frame=max(selected),
                reported_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            print(f"Sequential native-source decode {cid}: frames0..{max(selected)}",flush=True)
            for frame in range(max(selected)+1):
                ok, bgr = cap.read()
                if not ok or bgr.shape != (3190,4784,3):
                    raise ValueError("Unexpected decode failure or source shape")
                if not selected.get(frame):
                    continue
                gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                for observation in selected[frame]:
                    xy = np.asarray(observation["measurement_source_xy"],dtype=np.float64)
                    x,y = np.floor(xy+.5).astype(int)
                    left,top = int(x)-12,int(y)-12
                    if left<0 or top<0 or left+25>4784 or top+25>3190:
                        raise ValueError("Truncated selected patch")
                    patch = np.ascontiguousarray(gray[top:top+25,left:left+25])
                    features = diagnostic.measure(patch,observation["polarity"])
                    patch_key = f"patch_{len(results):04d}"
                    patches[patch_key] = patch.copy()
                    results.append(dict(**observation,source_sha256=source_digests[cid],
                        integer_center_xy=[int(x),int(y)],fractional_center_offset_xy=(xy-[x,y]).tolist(),
                        crop_xywh=[left,top,25,25],patch_array_key=patch_key,
                        patch_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),
                        patch_min_dn=int(patch.min()),patch_max_dn=int(patch.max()),
                        saturated_low_pixels=int(np.count_nonzero(patch==0)),
                        saturated_high_pixels=int(np.count_nonzero(patch==255)),
                        features=features,zero_margin_ablation_passed=bool(features["informative"] and features["point_minus_edge_fraction"]>0)))
        finally:
            cap.release()
    if len(results)!=len(selection["observations"]) or len({r["key"] for r in results})!=len(results):
        raise ValueError("Missing or duplicated selected observations")
    np.savez_compressed(output/"native_patches.npz",**patches)
    write(output/"observations.json",results)
    summary = summarize(selection,results)
    if (any(sha(p)!=d for p,d in inputs.items())
            or any(sha(ROOT/p)!=d for p,d in freeze["implementation_sha256"].items())
            or any(sha(output/"implementation"/p)!=d for p,d in freeze["implementation_sha256"].items())
            or sha(output/"unit.log")!=freeze["unit_log_sha256"]
            or any(sha(SOURCES[c])!=d for c,d in source_digests.items())
            or sha(output/"selection.json")!=freeze["selection_sha256"]):
        raise ValueError("Evidence or implementation changed during analysis")
    summary.update(completed=True,freeze_sha256=sha(output/"freeze.json"),
        outputs_sha256={n:sha(output/n) for n in ("native_patches.npz","observations.json")},
        capture_records=capture_records)
    write(output/"summary.json",summary)
    print(json.dumps(dict(selected_observations=summary["selected_observations"],
        reference_retention=summary["reference_retention"],control_before=70,
        control_after=sum(c["passing_measured_states"] for c in summary["provisional_controls"])),indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--shadow-audit",type=Path,required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve(),args.shadow_audit.resolve())
