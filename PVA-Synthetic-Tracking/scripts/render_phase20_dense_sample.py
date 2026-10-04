"""Render every frame of selected diagnostic intervals, preserving native pixels."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.visible_baseline import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--track-ids", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sample = json.loads(a.sample.read_text())
    if sha256(sample["source"]) != sample["source_sha256"]:
        raise ValueError("Source hash mismatch")
    entries = {e["track_id"]: e for e in sample["entries"]}
    selected = [entries[t] for t in a.track_ids]
    a.output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(sample["source"])
    results = []
    try:
        for e in selected:
            first, last = min(e["frames"]), max(e["frames"])
            x, y, w, h = e["crop_xywh"]
            count = last - first + 1
            canvas = np.full((((count + 4) // 5) * (h + 40), 5 * w, 3), 24, np.uint8)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            measurements = {t["frame_index"]: t for t in e["measurement_history"]}
            for index, f in enumerate(range(first, last + 1)):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Incomplete selected interval")
                left, top = (index % 5) * w, (index // 5) * (h + 40)
                patch = frame[y : y + h, x : x + w].copy()
                obs = measurements.get(f)
                if obs:
                    px, py = obs["measurement_source_xy"]
                    cv2.circle(
                        patch, (round(px - x), round(py - y)), 12, (70, 230, 100), 1
                    )
                canvas[top + 40 : top + 40 + h, left : left + w] = patch
                cv2.putText(
                    canvas,
                    f'{e["track_id"]} f{f}',
                    (left + 4, top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (240, 240, 240),
                    1,
                )
                cv2.putText(
                    canvas,
                    "measurement" if obs else "no measurement",
                    (left + 4, top + 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (240, 240, 240),
                    1,
                )
            filename = e["track_id"].replace(":", "_") + ".png"
            if not cv2.imwrite(str(a.output / filename), canvas):
                raise RuntimeError("Cannot write sheet")
            results.append(
                dict(
                    track_id=e["track_id"],
                    frames_inclusive=[first, last],
                    crop_xywh=e["crop_xywh"],
                    sheet=filename,
                )
            )
    finally:
        cap.release()
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                source=sample["source"],
                source_sha256=sample["source_sha256"],
                entries=results,
                display="All frames in interval; native pixels, no contrast enhancement; post-hoc diagnostic crop",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
