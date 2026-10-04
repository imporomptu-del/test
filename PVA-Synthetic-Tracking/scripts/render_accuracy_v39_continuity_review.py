"""Native crop comparison of unchanged baseline and explicit degraded states.

Uses only the previously reviewed 13-frame 0029 saved crop, no source-video
decode or new review labels. Biased illustrative regression, not accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/tiny_target/accuracy_v39_20260925"
REPLAY=BASE/"continuity_01"
REVIEW=ROOT.parent/"outputs/seaqr_accuracy_v37_review_20260924"
ARCHIVE=REVIEW/"native_bgr_crops.npz"
ORIGIN=(2619,2998)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def paint(source, states, *, candidate):
    canvas=cv2.resize(source,(512,384),interpolation=cv2.INTER_NEAREST)
    for state in states:
        if not candidate and not state["baseline_qualified"]:continue
        xy=state["measurement_source_xy"] if state["measured"] else state["source_xy"]
        x,y=xy[0]-ORIGIN[0],xy[1]-ORIGIN[1]
        if not (0<=x<256 and 0<=y<192):continue
        center=(round(2*x),round(2*y))
        color=(0,190,255) if state["added_degraded_measurement"] else (30,240,80) if state["measured"] else (180,180,180)
        cv2.circle(canvas,center,10,color,1,cv2.LINE_AA)
        name=state["track_id"].split(":")[1]+(" D" if state["added_degraded_measurement"] else " M" if state["measured"] else " P")
        tx,ty=min(center[0]+12,460),max(center[1]-6,12)
        cv2.putText(canvas,name,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(canvas,name,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return canvas


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Fresh review output directory required")
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg is None:raise RuntimeError("ffmpeg required for playable H.264 review")
    summary=json.loads((REPLAY/"summary.json").read_text())
    if not summary["completed"] or summary["degraded_states_promoted_to_confirmed"]:raise ValueError("Completed explicitly degraded replay required")
    if sha(REPLAY/"0029_decisions.jsonl")!=summary["outputs_sha256"]["0029_decisions.jsonl"]:raise ValueError("Changed replay ledger")
    reference=json.loads((ROOT/"results/tiny_target/accuracy_v38_20260925/compact_light_reference_v1.json").read_text())
    if sha(ARCHIVE)!=reference["inputs_sha256"][str(ARCHIVE)]:raise ValueError("Changed prior native crop archive")
    with np.load(ARCHIVE,allow_pickle=False) as archive:source=archive["00_0029_0012_0024"]
    if source.shape!=(13,192,256,3) or source.dtype!=np.uint8:raise ValueError("Unexpected native crop")
    states={}
    with (REPLAY/"0029_decisions.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line)
            if 12<=row["frame_index"]<=24:states[row["frame_index"]]=row["output_states"]
            if row["frame_index"]>=24:break
    if set(states)!=set(range(12,25)):raise ValueError("Incomplete review scope")
    output.mkdir(parents=True)
    width,height=1568,496
    video=output/"chunk0029_continuity_1p2_to_2p4s_slow.mp4"
    command=[ffmpeg,"-v","error","-nostdin","-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}",
        "-r","4","-i","pipe:0","-an","-c:v","libx264","-crf","16","-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=[]
    for frame in range(12,25):
        panel=np.full((height,width,3),22,np.uint8)
        raw=cv2.resize(source[frame-12],(512,384),interpolation=cv2.INTER_NEAREST)
        images=(raw,paint(source[frame-12],states[frame],candidate=False),paint(source[frame-12],states[frame],candidate=True))
        for i,(label,img) in enumerate(zip(("Unmarked source","Unchanged baseline","Baseline + DEGRADED measurements"),images)):
            x=8+i*520;panel[48:432,x:x+512]=img
            cv2.putText(panel,label,(x,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(panel,f'chunk0029 | frame {frame} | source time {frame/10:.1f}s | 2x nearest pixels | playback 4 fps (source nominal 10 fps)',
                    (8,454),cv2.FONT_HERSHEY_SIMPLEX,.54,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(panel,'GREEN M: baseline measured | AMBER D: degraded measured, NOT confirmed | GRAY P: prediction only | airborne class UNKNOWN',
                    (8,480),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
        frames.append(panel)
        if frame==16:
            if not cv2.imwrite(str(output/"frame0016_comparison.png"),panel):raise RuntimeError("Preview write failed")
    try:
        for _ in range(3):
            for panel in frames:process.stdin.write(panel.tobytes())
        process.stdin.close();error=process.stderr.read();code=process.wait()
        if code:raise RuntimeError(error.decode())
    except BaseException:
        process.kill();process.wait();raise
    metadata=dict(schema="seaqr.accuracy-v39-continuity-illustration.v1",source_clip="0029",frames_inclusive=[12,24],
        crop_xywh=[2619,2998,256,192],native_source_member="00_0029_0012_0024",nearest_neighbor_scale=2,
        nominal_source_fps=10,playback_fps=4,repetitions=3,encoded_frames=39,review_selected_from_known_failure=True,
        new_source_video_decode=False,airborne_truth=False,cleaner_output_claimed=False,
        coordinate_source="unchanged original measured/predicted state; not review annotation positions",
        inputs_sha256={str(ARCHIVE):sha(ARCHIVE),str(REPLAY/"0029_decisions.jsonl"):sha(REPLAY/"0029_decisions.jsonl"),
            str(REPLAY/"summary.json"):sha(REPLAY/"summary.json"),str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        outputs_sha256={p.name:sha(p) for p in (video,output/"frame0016_comparison.png")})
    with (output/"render_receipt.json").open("x") as stream:json.dump(metadata,stream,indent=2,allow_nan=False)
    print(video)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);run(p.parse_args().output)
