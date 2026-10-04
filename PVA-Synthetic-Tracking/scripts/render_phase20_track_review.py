"""Focused post-hoc review, not the detector's complete output.

Track IDs come from separate post-run anchor diagnostics. Detection was already
run full-frame without this selection. Other tracks are hidden here; prefix
review does not imply that the whole source was processed.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--event", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--crop", type=int, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--diagnostics",
        type=Path,
        help="Explicit prefix anchor diagnostics instead of full-clip regression",
    )
    a = p.parse_args()
    launch = json.loads((a.run / "launch.json").read_text())
    report = json.loads((a.run / "report.json").read_text())
    diagnostic_path = a.diagnostics or a.run / "regression.json"
    regression = json.loads(diagnostic_path.read_text())
    if regression["source_sha256"] != launch["source_sha256"]:
        raise ValueError("Diagnostic/source mismatch")
    if not report["completed"] or a.end >= report["frames"]:
        raise ValueError("Review interval must lie within a completed run")
    events = (
        regression["sparse_anchor_results"] if a.diagnostics else regression["events"]
    )
    event = next(e for e in events if e["event_id"] == a.event)
    track_id = event["dominant_track_id"]
    if track_id is None:
        raise ValueError("No matched automatic track to render")
    if a.output.exists():
        raise FileExistsError(a.output)
    x, y, w, h = a.crop
    if min(x, y, a.start) < 0 or min(w, h) <= 0 or a.end < a.start:
        raise ValueError("Invalid interval/crop")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tiny_target.visible_baseline import sha256

    if sha256(a.source) != launch["source_sha256"]:
        raise ValueError("Wrong source")
    selected = {}
    with (a.run / "frames.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if a.start <= row["frame_index"] <= a.end:
                selected[row["frame_index"]] = next(
                    (t for t in row["tracks"] if t["track_id"] == track_id), None
                )
    cap = cv2.VideoCapture(str(a.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    outw = w + (w % 2)
    outh = h + 84 + ((h + 84) % 2)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{outw}x{outh}",
        "-framerate",
        str(launch["fps"]),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(a.output),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for idx in range(a.start, a.end + 1):
            if idx not in selected:
                raise ValueError("Missing journal frame")
            ok, frame = cap.read()
            if not ok or y + h > frame.shape[0] or x + w > frame.shape[1]:
                raise ValueError("Source/crop mismatch")
            canvas = np.full((outh, outw, 3), 24, np.uint8)
            canvas[84 : 84 + h, :w] = frame[y : y + h, x : x + w]
            t = selected[idx]
            status = "No live state for selected ID"
            if t:
                observed = t["measured"]
                xy = t["measurement_source_xy"] if observed else t["source_xy"]
                px, py = round(xy[0] - x), round(xy[1] - y) + 84
                color = (80, 230, 100) if observed else (0, 180, 255)
                if 0 <= px < w and 84 <= py < h + 84:
                    if observed:
                        cv2.circle(canvas, (px, py), 12, color, 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(
                            canvas, (px - 12, py - 12), (px + 12, py + 12), color, 1
                        )
                status = f'{track_id} | {"MEASURED" if observed else "PREDICTED ONLY"} | {t["lifecycle"]} | qualified={t["qualified_moving"]}'
            for j, text in enumerate(
                [
                    f'{a.event} | source {idx/launch["fps"]:.1f}s | {status}',
                    "Focused post-hoc track selection; other tracks hidden. NOT overall accuracy.",
                    "Green circle = measurement; amber square = prediction. Native crop.",
                ]
            ):
                font_scale = 0.43
                text_width = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0][0]
                font_scale *= min(1.0, (outw - 16) / max(text_width, 1))
                cv2.putText(
                    canvas,
                    text,
                    (8, 20 + j * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    a.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                run=str(a.run),
                source_sha256=launch["source_sha256"],
                event=a.event,
                full_clip_processed=report["full_clip"],
                diagnostic_path=str(diagnostic_path),
                diagnostic_sha256=sha256(diagnostic_path),
                selected_automatic_track_id=track_id,
                selection="post-hoc from scorer; other automatic tracks hidden",
                frames_inclusive=[a.start, a.end],
                crop_xywh=a.crop,
                source_pixels_resized=False,
                encoding="H264 CRF12 viewing copy, no contrast enhancement",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
