"""Reproducible bounded diagnostic sample; not a false-positive prevalence study."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.visible_baseline import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Sample qualified proposals without pretending a regression reference exists",
    )
    args = parser.parse_args()
    launch = json.loads((args.run / "launch.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    regression = (
        None
        if args.unlabeled
        else json.loads((args.run / "regression.json").read_text())
    )
    if not report["completed"] or not report["full_clip"]:
        raise ValueError("Completed full-clip run required")
    if sha256(args.source) != launch["source_sha256"]:
        raise ValueError("Wrong source")
    known = (
        set()
        if regression is None
        else {
            t
            for e in regression["events"]
            for a in e["anchors"]
            for t in a["qualified_measured_track_ids"]
        }
    )
    summaries = {t["track_id"]: t for t in report["qualified_tracks"]}
    histories = defaultdict(list)
    with (args.run / "frames.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            for t in row["tracks"]:
                if t["track_id"] in summaries and t["measured"]:
                    histories[t["track_id"]].append((row["frame_index"], t))
    rng = random.Random(20260913)
    selected = []
    for polarity in ("bright", "dark"):
        pool = sorted(t for t in histories if t.startswith(polarity) and t not in known)
        chosen = rng.sample(pool, min(4, len(pool)))
        for key in (
            lambda t: np.median([r[1]["measurement_score"] for r in histories[t]]),
            lambda t: len(histories[t]),
        ):
            remaining = [t for t in pool if t not in chosen]
            if remaining:
                chosen.append(max(remaining, key=key))
        selected.extend(chosen)
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    needed = defaultdict(list)
    for idx, tid in enumerate(selected):
        anchor = summaries[tid]["first_qualified_frame"]
        point = next((t for i, t in histories[tid] if i == anchor), None)
        if point is None:
            raise ValueError("First qualification must be observation-supported")
        cx, cy = point["measurement_source_xy"]
        x = min(max(0, round(cx) - 96), launch["source_probe"]["width"] - 192)
        y = min(max(0, round(cy) - 96), launch["source_probe"]["height"] - 192)
        frames = [
            max(0, min(report["frames"] - 1, anchor + d)) for d in (-6, -3, 0, 3, 6)
        ]
        entry = dict(
            index=idx,
            track_id=tid,
            first_qualified_frame=anchor,
            crop_xywh=[x, y, 192, 192],
            frames=frames,
            measurement_history=[dict(frame_index=i, **t) for i, t in histories[tid]],
            review_label="unreviewed",
        )
        entries.append(entry)
        for column, frame_index in enumerate(frames):
            needed[frame_index].append((idx, column))
    sheets = [
        np.full((4 * 244, 5 * 192, 3), 24, np.uint8)
        for _ in range((len(entries) + 3) // 4)
    ]
    cap = cv2.VideoCapture(str(args.source))
    try:
        for frame_index in sorted(needed):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Cannot decode selected frame")
            for idx, column in needed[frame_index]:
                e = entries[idx]
                x, y, w, h = e["crop_xywh"]
                sheet = sheets[idx // 4]
                top = (idx % 4) * 244
                left = column * 192
                patch = frame[y : y + h, x : x + w].copy()
                obs = next(
                    (t for i, t in histories[e["track_id"]] if i == frame_index), None
                )
                if obs:
                    px, py = obs["measurement_source_xy"]
                    cv2.circle(
                        patch, (round(px - x), round(py - y)), 12, (70, 230, 100), 1
                    )
                sheet[top + 52 : top + 244, left : left + 192] = patch
                cv2.putText(
                    sheet,
                    f'{e["track_id"]} f{frame_index}',
                    (left + 4, top + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (240, 240, 240),
                    1,
                )
                cv2.putText(
                    sheet,
                    "measured" if obs else "no measurement",
                    (left + 4, top + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (240, 240, 240),
                    1,
                )
    finally:
        cap.release()
    for idx, sheet in enumerate(sheets):
        if not cv2.imwrite(str(args.output / f"sheet_{idx+1}.png"), sheet):
            raise RuntimeError("Could not save sheet")
    result = dict(
        source=str(args.source.resolve()),
        source_sha256=launch["source_sha256"],
        source_run=str(args.run.resolve()),
        report_sha256=sha256(args.run / "report.json"),
        source_is_unlabeled=args.unlabeled,
        selection=(
            "Per polarity: seeded four random qualified proposals, one strongest remaining, one longest remaining; no reference exclusions; diagnostic, not representative prevalence"
            if args.unlabeled
            else "Per polarity: seeded four random unmatched tracks, one strongest remaining, one longest remaining; diagnostic, not representative prevalence"
        ),
        seed=20260913,
        crop_display="Native pixels, fixed source coordinates, no contrast enhancement; green ring indicates selected track's measured location",
        excluded_known_reference_ids=sorted(known),
        entries=entries,
    )
    (args.output / "sample.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(output=str(args.output), sampled=len(entries))))


if __name__ == "__main__":
    main()
