"""Bounded native source review of V36 removals; not truth or a gate change."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from diagnose_accuracy_v37_temporal import ROOT, PARENT, FULL, bind, read, recheck, sha, write, verify_parent
from render_visible_v34_demo import SOURCES

INVENTORY = ROOT / "docs/accuracy_v37_review_inventory.md"
EXPECTED_IDS = {
    "0029": ("bright:138", "bright:2079"), "0126": ("bright:201", "bright:6301"),
    "0055": ("bright:3", "bright:1312"), "0082": ("bright:51", "bright:2738"),
}
EXPECTED_CROPS = (
    (12,24,2619,2998), (283,295,2336,2970), (330,342,2360,2970), (644,656,2586,2985),
    (6,18,2769,1625), (393,405,2938,1697), (528,540,2860,1624), (660,672,2794,1524),
    (6,18,2274,2998), (226,238,1841,2939), (391,403,1668,2904), (610,622,1340,2784),
    (8,20,3109,2899), (639,651,2139,2968), (652,664,2159,2963), (684,690,2157,2951),
)


def selection():
    windows, rows_by_clip = [], {}
    for cid, expected in EXPECTED_IDS.items():
        with (FULL / (cid+"_decisions.jsonl")).open() as stream:
            rows = [json.loads(line) for line in stream]
        if [r["frame_index"] for r in rows] != list(range(len(rows))):
            raise ValueError("Noncontiguous decision journal")
        rows_by_clip[cid] = rows
        rejected, measured = defaultdict(list), defaultdict(list)
        for row in rows:
            for track in row["tracks"]:
                if track["measured"]:
                    identity = (track["segment"], track["track_id"])
                    item = (row["frame_index"], track)
                    measured[identity].append(item)
                    if not track["accepted"]:
                        rejected[identity].append(item)
        eligible = {k:v for k,v in rejected.items() if len(v) >= 5}
        earliest = min(eligible, key=lambda k:(eligible[k][0][0], k))
        longest = min(eligible, key=lambda k:(-(eligible[k][-1][0]-eligible[k][0][0]),
                                             -len(eligible[k]), eligible[k][0][0], k))
        if (earliest[1], longest[1]) != expected or earliest == longest:
            raise ValueError("Selected identities differ from pre-view inventory")
        for role, identity in (("earliest", earliest), ("longest", longest)):
            values = rejected[identity]
            indices = (0,) if role == "earliest" else (0, len(values)//2, len(values)-1)
            for index in indices:
                anchor = values[index][0]
                start, stop = max(0, anchor-6), min(len(rows)-1, anchor+6)
                positions = np.asarray([t["measurement_source_xy"] for f,t in measured[identity]
                                        if start <= f <= stop])
                center = np.floor((positions.min(axis=0)+positions.max(axis=0))/2 + .5).astype(int)
                origin = np.clip(center-(128,96), (0,0), (4784-256,3190-192))
                if not np.all((positions >= origin) & (positions < origin+(256,192))):
                    raise ValueError("Selected measured position outside bounded review crop")
                windows.append(dict(clip=cid, role=role, segment=identity[0], track_id=identity[1],
                    rejected_anchor_frame=anchor, start=start, stop=stop,
                    crop_xywh=[int(origin[0]),int(origin[1]),256,192],
                    physical_class="unknown", visual_label="not_yet_reviewed"))
    geometry = tuple((w["start"],w["stop"],*w["crop_xywh"][:2]) for w in windows)
    if geometry != EXPECTED_CROPS or sum(w["stop"]-w["start"]+1 for w in windows) != 202:
        raise ValueError("Review bounds differ from pre-view inventory")
    return windows, rows_by_clip


def sheet(frames, window, marked=False):
    # Every source crop is shown at native 1:1; headers occupy separate pixels.
    gap, cell_w, cell_h = 8, 256, 216
    canvas = np.full((70+4*cell_h+3*gap, 4*cell_w+3*gap, 3), 24, np.uint8)
    title = f'{window["clip"]} {window["track_id"]} | {window["role"]} | native 1:1'
    cv2.putText(canvas,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.57,(240,240,240),1,cv2.LINE_AA)
    subtitle = "DIAGNOSTIC ONLY: green retained / red removed / amber predicted" if marked else "UNMARKED SOURCE: physical class unknown; no negative truth labels"
    cv2.putText(canvas,subtitle,(8,48),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
    for i,(frame,crop,track) in enumerate(frames):
        row,col = divmod(i,4)
        x,y = col*(cell_w+gap),70+row*(cell_h+gap)
        image = crop.copy()
        if marked and track is not None:
            xy = track["measurement_source_xy"] if track["measured"] else track["source_xy"]
            px,py = np.floor(np.asarray(xy)-window["crop_xywh"][:2]+.5).astype(int)
            color = ((80,255,80) if track["accepted"] else (80,80,255)) if track["measured"] else (0,190,255)
            if track["measured"]:
                cv2.circle(image,(int(px),int(py)),9,color,1,cv2.LINE_AA)
            else:
                cv2.rectangle(image,(int(px-8),int(py-8)),(int(px+8),int(py+8)),color,1)
        canvas[y:y+192,x:x+256] = image
        cv2.putText(canvas,f'frame {frame} | {frame/10:.1f}s',(x+4,y+210),
                    cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
    return canvas


def run(output):
    if output.exists():
        raise FileExistsError("Exclusive fresh review output required")
    _, files, _ = verify_parent(PARENT / "full_context_independent_audit_01.json")
    freeze = read(FULL / "freeze.json")
    for cid in EXPECTED_IDS:
        source = SOURCES[cid]
        if freeze["source_videos"][cid]["path"] != str(source):
            raise ValueError("Review source path mismatch")
        bind(files, source, freeze["source_videos"][cid]["sha256"])
    bind(files, Path(__file__))
    bind(files, Path(__file__).with_name("diagnose_accuracy_v37_temporal.py"))
    bind(files, Path(__file__).with_name("render_visible_v34_demo.py"))
    bind(files, INVENTORY)
    windows, rows = selection()
    output.mkdir(parents=True)
    shutil.copy2(__file__,output / "renderer.py")
    shutil.copy2(INVENTORY,output / "pre_view_inventory.md")
    bind(files,output / "renderer.py",sha(Path(__file__)))
    bind(files,output / "pre_view_inventory.md",sha(INVENTORY))
    write(output / "freeze.json",dict(schema="seaqr.accuracy-v37-rejected-review-freeze.v1",
        diagnostic_only=True,inputs_sha256=files, windows=windows,
        native_crop_frames=202, pre_view_selection=True, no_new_truth_labels=True,
        detector_rerun=False, nominal_source_fps=10))
    exported, arrays, counts = [], {}, {}
    for cid in EXPECTED_IDS:
        active = [(i,w) for i,w in enumerate(windows) if w["clip"]==cid]
        extracted = {i:[] for i,w in active}
        cap = cv2.VideoCapture(str(SOURCES[cid]))
        try:
            if (not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)!=10
                    or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows[cid])):
                raise ValueError("Review source decoder mismatch")
            for f in range(max(w["stop"] for _,w in active)+1):
                ok, frame = cap.read()
                if not ok or frame.shape != (3190,4784,3) or frame.dtype!=np.uint8:
                    raise ValueError("Review sequential decode mismatch")
                for i,w in active:
                    if w["start"] <= f <= w["stop"]:
                        x,y,width,height = w["crop_xywh"]
                        crop = frame[y:y+height,x:x+width].copy()
                        matches = [t for t in rows[cid][f]["tracks"]
                                   if t["segment"]==w["segment"] and t["track_id"]==w["track_id"]]
                        if len(matches)>1:
                            raise ValueError("Duplicate review identity")
                        extracted[i].append((f,crop,matches[0] if matches else None))
            counts[cid] = f+1
        finally:
            cap.release()
        for i,w in active:
            data = extracted[i]
            if [f for f,_,_ in data] != list(range(w["start"],w["stop"]+1)):
                raise ValueError("Missing selected review frame")
            key = f'{i:02d}_{cid}_{w["start"]:04d}_{w["stop"]:04d}'
            arrays[key] = np.stack([crop for _,crop,_ in data])
            for marked in (False,True):
                name = key + ("_marked.png" if marked else "_unmarked.png")
                if not cv2.imwrite(str(output/name),sheet(data,w,marked)):
                    raise RuntimeError("Review PNG write failed")
                exported.append(dict(name=name,sha256=sha(output/name),marked=marked,window_index=i))
        print(f"Review crops complete: {cid}",flush=True)
    np.savez_compressed(output / "native_bgr_crops.npz",**arrays)
    recheck(files)
    write(output / "manifest.json",dict(schema="seaqr.accuracy-v37-rejected-review-manifest.v1",
        diagnostic_only=True,completed=True,windows=windows,
        renderer_sha256=sha(output / "renderer.py"),inventory_sha256=sha(output / "pre_view_inventory.md"),
        native_crop_frames=sum(len(a) for a in arrays.values()),decoded_frames=counts,
        exported=exported, native_bgr_crops_sha256=sha(output / "native_bgr_crops.npz"),
        freeze_sha256=sha(output / "freeze.json"),no_new_truth_labels=True,
        detector_rerun=False,promoted=False,review_is_not_accuracy_estimation=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output.resolve())
