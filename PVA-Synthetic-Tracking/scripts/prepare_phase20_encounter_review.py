"""Every-frame source-only review of reported encounters and longer light fields.

Historical manual anchors position the REVIEW CROPS only. Interpolated crop
centers are never annotations. No inference report, candidate, or track is read.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_phase20_accuracy_review import digest, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def interpolate_crop_center(anchors, index):
    """Piecewise linear/extrapolated viewing aid, explicitly NOT ground truth."""
    after = next((i for i, a in enumerate(anchors) if a[0] > index), len(anchors))
    i = max(0, min(after - 1, len(anchors) - 2))
    a, b = anchors[i : i + 2]
    t = (index - a[0]) / (b[0] - a[0])
    return [round(a[j] + t * (b[j] - a[j])) for j in (1, 2)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old_plan = json.loads(
        (ROOT / "configs/evaluation/phase20_accuracy_review_v1.json").read_text()
    )
    sources = validate_plan(old_plan)
    refs = [
        ROOT
        / "results/tiny_target/phase19/chunk0029_visual_review_20260913/visual_annotations.json",
        ROOT
        / "results/tiny_target/phase19/chunk0126_avi_20260913/visual_review/visual_annotations.json",
    ]
    ref29, ref126 = [json.loads(path.read_text()) for path in refs]
    episodes = []
    for ev, first, last, name in zip(
        ref29["events"], (40, 220), (109, 389), ("029_A", "029_B")
    ):
        episodes.append(
            dict(
                id=name,
                clip_id="0029",
                first=first,
                last=last,
                role="known_moving_feature_class_unknown",
                anchors=[[v["frame_index"], v["x"], v["y"]] for v in ev["anchors"]],
                size_wh=[128, 128],
                scale=2,
            )
        )
    episodes.append(
        dict(
            id="126",
            clip_id="0126",
            first=80,
            last=229,
            role="known_moving_feature_class_unknown",
            anchors=[
                [v["frame_index"], *v["approximate_xy"]] for v in ref126["anchors"]
            ],
            size_wh=[128, 128],
            scale=2,
        )
    )
    for cid in ("0055", "0082"):
        episodes.append(
            dict(
                id=cid + "_lights",
                clip_id=cid,
                first=450,
                last=649,
                role="source_selected_background_review_not_prelabelled_negative",
                fixed_crop=[3460, 2562, 256, 192],
                size_wh=[256, 192],
                scale=1,
            )
        )
    for s in sources.values():
        if digest(s["path"]) != s["sha256"]:
            raise ValueError("Source identity mismatch")
    a.output.mkdir(parents=True, exist_ok=False)
    plan = dict(
        schema="seaqr.extended-source-review-plan.v1",
        sources=list(sources.values()),
        episodes=episodes,
        target_class="airborne_only",
        physical_class_of_known_movers="unknown",
        reference_sha256={str(v): digest(v) for v in refs},
        scope="Every frame across reported encounter intervals plus temporal padding; not a claim of physical onset/disappearance or exhaustive full-frame truth",
        crop_interpolation_is_truth=False,
        renderer_sha256=digest(__file__),
    )
    with (a.output / "plan.json").open("x") as f:
        json.dump(plan, f, indent=2)
    records = []
    for ep in episodes:
        s = sources[ep["clip_id"]]
        cap = cv2.VideoCapture(s["path"])
        patches = {}
        per_frame = []
        sheets = []
        sw, sh = ep["size_wh"]
        scale = ep["scale"]
        dw, dh = sw * scale, sh * scale
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ep["first"])
            for i, f in enumerate(range(ep["first"], ep["last"] + 1)):
                ok, frame = cap.read()
                if (
                    not ok
                    or list(frame.shape[:2]) != s["shape_hw"]
                    or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != f + 1
                ):
                    raise ValueError("Frame/shape mismatch")
                if "fixed_crop" in ep:
                    x, y, _, _ = ep["fixed_crop"]
                else:
                    cx, cy = interpolate_crop_center(ep["anchors"], f)
                    x = max(0, min(cx - sw // 2, frame.shape[1] - sw))
                    y = max(0, min(cy - sh // 2, frame.shape[0] - sh))
                patch = frame[y : y + sh, x : x + sw].copy()
                patches[str(f)] = patch
                per_frame.append(dict(frame_index=f, crop_xywh=[x, y, sw, sh]))
                if i % 16 == 0:
                    canvas = np.full((4 * (dh + 26), 4 * dw, 3), 24, np.uint8)
                    sheet_first = f
                left, top = (i % 4) * dw, ((i % 16) // 4) * (dh + 26)
                shown = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_NEAREST)
                canvas[top + 26 : top + 26 + dh, left : left + dw] = shown
                cv2.putText(
                    canvas,
                    f'{ep["id"]} f{f} {scale}x nearest',
                    (left + 4, top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (240, 240, 240),
                    1,
                )
                if i % 16 == 15 or f == ep["last"]:
                    name = f'{ep["id"]}_{sheet_first:04d}_{f:04d}.png'
                    if not cv2.imwrite(str(a.output / name), canvas):
                        raise RuntimeError("Cannot write sheet")
                    sheets.append(
                        dict(
                            path=name,
                            first=sheet_first,
                            last=f,
                            sha256=digest(a.output / name),
                        )
                    )
        finally:
            cap.release()
        np.savez_compressed(a.output / (ep["id"] + "_native.npz"), **patches)
        records.append(
            dict(
                **ep,
                frames=per_frame,
                sheets=sheets,
                native_archive=ep["id"] + "_native.npz",
                native_archive_sha256=digest(a.output / (ep["id"] + "_native.npz")),
            )
        )
    with (a.output / "packet.json").open("x") as f:
        json.dump(
            dict(
                plan=plan,
                plan_sha256=digest(a.output / "plan.json"),
                episodes=records,
                detector_outputs_read=False,
                visibility_labels_assigned=False,
                enhancement=False,
            ),
            f,
            indent=2,
        )
    print(
        json.dumps(
            dict(
                output=str(a.output),
                frames=sum(len(e["frames"]) for e in records),
                sheets=sum(len(e["sheets"]) for e in records),
            )
        )
    )


if __name__ == "__main__":
    main()
