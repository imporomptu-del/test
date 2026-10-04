"""Experimental side-by-side output preview, never an accuracy/FPS claim."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from render_visible_v34_demo import AUDIT, SOURCES, Encoder, annotate, selected, sha, text


def canvas(frame, row, accepted_keys, clip, crop=None):
    native = crop is not None
    h,w = frame.shape[:2]
    crop = crop or (0,0,w,h)
    x,y,cw,ch = crop
    if min(x,y)<0 or x+cw>w or y+ch>h:
        raise ValueError("Invalid comparison crop")
    if native:
        left = frame[y:y+ch,x:x+cw].copy()
    else:
        left = cv2.resize(frame,(1200,round(h*1200/w)),interpolation=cv2.INTER_AREA)
    right = left.copy()
    before = selected(row,crop)
    after = [(t,xy) for t,xy in before if (t["segment"],t["track_id"]) in accepted_keys]
    annotate(left,before,crop,native=native)
    annotate(right,after,crop,native=native)
    ph,pw = left.shape[:2]
    header,footer,gap = 140,82,24
    height = header+ph+footer
    height += height%2
    result = np.full((height,2*pw+gap,3),(27,24,20),dtype=np.uint8)
    result[header:header+ph,:pw] = left
    result[header:header+ph,pw+gap:] = right
    index = row["frame_index"]
    text(result,f"SEAQR ACCURACY EXPERIMENT / chunk {clip} / source {index/10:.1f}s / frame {index}",18,29,.75)
    text(result,"Same source pixels and original track states. Right side changes OUTPUT ELIGIBILITY ONLY.",18,58,.65)
    text(result,"Green circle = measured | Amber square = predicted | Neither is a verified airborne label.",18,86,.62)
    text(result,"BASELINE v34",18,125,.70)
    text(result,"EXPERIMENTAL POINT / EDGE FILTER",pw+gap+18,125,.70,max_width=pw-36)
    a = sum(t["measured"] for t,_ in before)
    b = sum(t["measured"] for t,_ in after)
    text(result,f"In view: {a} measured / {len(before)-a} predicted",18,header+ph+28,.63)
    text(result,f"In view: {b} measured / {len(after)-b} predicted",pw+gap+18,header+ph+28,.63)
    mode = "Native 1:1 post-hoc comparison crop" if native else "Full-frame downscaled overview; use native crop for tiny targets"
    text(result,mode+". 10 FPS SOURCE playback, not processing speed.",18,header+ph+54,.55)
    text(result,"Unpromoted prototype: can reject genuine points on edges; not a detector rerun or general accuracy proof.",18,header+ph+75,.53)
    return result


def run(experiment,output,clip):
    if output.exists():
        raise FileExistsError("Fresh preview directory required")
    # Full-context receipts are checked against the exact source/journal below.
    summary_path = experiment/"summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("schema")!="seaqr.accuracy-v36-full-context-summary.v1"
            or summary.get("completed") is not True
            or summary.get("freeze_sha256")!=sha(experiment/"freeze.json")):
        raise ValueError("Completed full-context experiment required")
    source = SOURCES[clip]
    baseline = AUDIT/"evidence/run"/("full_repeat0_"+clip)
    launch = json.loads((baseline/"launch.json").read_text())
    manifest = json.loads((AUDIT/"evidence/export_manifest_v34_01.json").read_text())
    journal = baseline/"frames.jsonl"
    if sha(source)!=launch["source_sha256"] or sha(journal)!=manifest["files"][f"run/full_repeat0_{clip}/frames.jsonl"]:
        raise ValueError("Changed audited source or parent journal")
    decisions = experiment/(clip+"_decisions.jsonl")
    if sha(decisions)!=summary["outputs_sha256"][decisions.name]:
        raise ValueError("Changed full-context decisions")
    full_freeze = json.loads((experiment/"freeze.json").read_text())
    if full_freeze["source_videos"][clip]["sha256"]!=sha(source):
        raise ValueError("Context source identity mismatch")
    inputs = {str(p):sha(p) for p in (source,journal,decisions,summary_path,
        experiment/"freeze.json",Path(__file__),Path(__file__).with_name("render_visible_v34_demo.py"))}
    # No selected IDs or events enter overview output. Native crop is explicitly
    # presentation-only and uses the previously reviewed v34 chunk126 window.
    output.mkdir(parents=True)
    shutil.copy2(__file__,output/"renderer.py")
    with (output/"freeze.json").open("x") as stream:
        json.dump(dict(inputs_sha256=inputs,clip=clip,detector_rerun=False,promoted=False),stream,indent=2)
    cap = cv2.VideoCapture(str(source))
    encoders = {}
    total = 0
    probes = {}
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)!=10:
            raise ValueError("Unexpected source decoder")
        with journal.open() as parent,decisions.open() as gate:
            for frame_index,parent_line in enumerate(parent):
                row = json.loads(parent_line)
                raw_gate = gate.readline()
                if not raw_gate:
                    raise ValueError("Incomplete decision journal")
                decision = json.loads(raw_gate)
                if (row["frame_index"]!=frame_index or decision["frame_index"]!=frame_index
                        or row["timestamp_ns"]!=decision["timestamp_ns"]
                        or row["segment"]!=decision["segment"]):
                    raise ValueError("Frame-alignment mismatch")
                accepted_keys = decision_keys(decision)
                baseline_keys = {(t["segment"],t["track_id"]) for t in row["tracks"] if t["qualified_moving"]}
                logged_keys = {(t["segment"],t["track_id"]) for t in decision["tracks"]}
                if logged_keys!=baseline_keys or not accepted_keys<=baseline_keys:
                    raise ValueError("Preview omitted or invented baseline tracks")
                ok,frame = cap.read()
                if not ok or frame.shape!=(3190,4784,3):
                    raise ValueError("Unexpected source decode")
                views = {"full_comparison":canvas(frame,row,accepted_keys,clip)}
                if clip=="0126" and 90<=frame_index<=229:
                    views["native_target_comparison_09s_23s"] = canvas(frame,row,accepted_keys,clip,(2550,2450,850,650))
                for name,img in views.items():
                    if name not in encoders:
                        encoders[name] = Encoder(output/(clip+"_"+name+".mp4"),img.shape,10)
                    encoders[name].send(img)
                    if frame_index in (150,350,600) and name=="full_comparison":
                        if not cv2.imwrite(str(output/f"frame_{frame_index:04d}.png"),img):
                            raise RuntimeError("Unable to write comparison still")
                total += 1
            if gate.readline() or cap.read()[0]:
                raise ValueError("Unexpected extra source/decision frames")
        if total!=summary["clips"][clip]["frames"]:
            raise ValueError("Preview frame count differs from full-context receipt")
        for name,encoder in encoders.items():
            expected = total if name=="full_comparison" else 140
            probes[name] = encoder.finish(expected)
        if any(sha(p)!=d for p,d in inputs.items()):
            raise ValueError("Preview inputs changed during rendering")
        with (output/"manifest.json").open("x") as stream:
            json.dump(dict(completed=True,clip=clip,frames=total,probes=probes,
                output_sha256={p.name:sha(p) for p in output.glob("*.mp4")},
                detector_rerun=False,promoted=False,manual_track_selection=False,
                nominal_playback_fps=10,processing_fps=None),stream,indent=2)
    finally:
        cap.release()
        for encoder in encoders.values():
            encoder.stop()


def decision_keys(row):
    """No manual ID selection: every accepted logged state is drawn in view."""
    keys,accepted = set(),set()
    for track in row["tracks"]:
        key = (track["segment"],track["track_id"])
        if key in keys or type(track.get("accepted")) is not bool:
            raise ValueError("Duplicate or malformed preview decision")
        keys.add(key)
        if track["accepted"]:
            accepted.add(key)
    return accepted


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--clip",choices=tuple(SOURCES),default="0126")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.experiment.resolve(),args.output.resolve(),args.clip)
