"""Post-run side-by-side source crops; actual qualified track states only."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from score_phase20_accuracy import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--after-label", default="Learning protection experiment")
    p.add_argument("--episode", choices=("126", "029_A", "029_B"), default="126")
    p.add_argument("--first", type=int)
    p.add_argument("--last", type=int)
    a = p.parse_args()
    source = a.root / "source_review"
    packet = json.loads((source / "packet.json").read_text())
    ep = next(e for e in packet["episodes"] if e["id"] == a.episode)
    source_spec = next(s for s in packet["plan"]["sources"] if s["clip_id"] == ep["clip_id"])
    first = a.first if a.first is not None else ep["first"]
    last = a.last if a.last is not None else (218 if a.episode == "126" else ep["last"])
    if not ep["first"] <= first <= last <= ep["last"] or ep["size_wh"] != [128, 128]:
        raise ValueError("Review interval must be inside a native 128-pixel source episode")
    origins = {r["frame_index"]: r["crop_xywh"] for r in ep["frames"]}
    archive = source / ep["native_archive"]
    if digest(archive) != ep["native_archive_sha256"]:
        raise ValueError("Source archive changed")
    journals = []
    for path in (a.before, a.after):
        launch = json.loads((path / "launch.json").read_text())
        report = json.loads((path / "report.json").read_text())
        if (
            not report["completed"]
            or report["frames"] <= last
            or launch["source_sha256"] != source_spec["sha256"]
            or report["source_sha256"] != source_spec["sha256"]
        ):
            raise ValueError("Completed matching source run required")
        selected = {}
        for row in map(json.loads, (path / "frames.jsonl").open()):
            if first <= row["frame_index"] <= last:
                selected[row["frame_index"]] = row
        if len(selected) != last - first + 1:
            raise ValueError("Missing journal frame")
        journals.append(selected)
    if a.output.exists():
        raise FileExistsError(a.output)
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
        "1024x624",
        "-framerate",
        "10",
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
        "10",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(a.output),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        with np.load(archive, allow_pickle=False) as patches:
            for f in range(first, last + 1):
                canvas = np.full((624, 1024, 3), 24, np.uint8)
                ox, oy, _, _ = origins[f]
                for col, (label, rows) in enumerate(
                    zip(("Frozen baseline", a.after_label), journals)
                ):
                    left = col * 512
                    shown = cv2.resize(
                        patches[str(f)], (512, 512), interpolation=cv2.INTER_NEAREST
                    )
                    canvas[48:560, left : left + 512] = shown
                    cv2.putText(
                        canvas,
                        f"{label} | f{f} ({f/10:.1f}s)",
                        (left + 8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (235, 235, 235),
                        1,
                    )
                    for t in rows[f]["tracks"]:
                        if not t["qualified_moving"]:
                            continue
                        xy = (
                            t["measurement_source_xy"]
                            if t["measured"]
                            else t["source_xy"]
                        )
                        x, y = round((xy[0] - ox) * 4), round((xy[1] - oy) * 4)
                        if not 0 <= x < 512 or not 0 <= y < 512:
                            continue
                        center = (left + x, 48 + y)
                        color = (70, 235, 100) if t["measured"] else (0, 185, 255)
                        if t["measured"]:
                            cv2.circle(canvas, center, 23, color, 1)
                        else:
                            cv2.rectangle(
                                canvas,
                                (center[0] - 23, center[1] - 23),
                                (center[0] + 23, center[1] + 23),
                                color,
                                1,
                            )
                        cv2.putText(
                            canvas,
                            t["track_id"],
                            (
                                max(left + 2, min(left + 380, center[0] + 25)),
                                max(62, min(550, center[1] - 8)),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            color,
                            1,
                        )
                cv2.putText(
                    canvas,
                    "Green circle: qualified measurement. Amber square: prediction only. Neither proves airborne class.",
                    (8, 582),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                )
                cv2.putText(
                    canvas,
                    "4x nearest source pixels; post-hoc moving review crop. No annotation/crop hints entered either detector.",
                    (8, 606),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                )
                encoder.stdin.write(canvas.tobytes())
                if f == first:
                    cv2.imwrite(str(a.output.with_suffix(".png")), canvas)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    with a.output.with_suffix(".json").open("x") as f:
        json.dump(
            dict(
                before=str(a.before.resolve()),
                after=str(a.after.resolve()),
                frames_inclusive=[first, last],
                source_sha256=source_spec["sha256"],
                source_archive_sha256=digest(archive),
                before_journal_sha256=digest(a.before / "frames.jsonl"),
                after_journal_sha256=digest(a.after / "frames.jsonl"),
                video_sha256=digest(a.output),
                view="Post-hoc moving crop, all qualified tracks whose position is inside the crop; unqualified states and outside tracks hidden. Not whole-frame accuracy.",
                renderer_sha256=digest(__file__),
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
