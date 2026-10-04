"""Render bounded source-only evidence. Never read a detector report/journal."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

# Fail closed before stat/hash/decode: only these explicitly scoped development
# sources are permitted. Do not discover media or consult the holdout manifest.
ALLOWED = {
    "0029": "0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359",
    "0126": "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344",
    "0055": "c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f",
    "0082": "465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_plan(plan):
    if plan.get("schema") != "seaqr.source-only-review-plan.v1":
        raise ValueError("Unsupported review plan")
    sources = {}
    for s in plan["sources"]:
        cid = s["clip_id"]
        path = Path(s["path"])
        if (
            cid not in ALLOWED
            or s["sha256"] != ALLOWED[cid]
            or path.name != f"chunk_{cid}.avi"
            or cid in sources
        ):
            raise ValueError("Source outside explicit development allowlist")
        # Do not follow a differently named media symlink before hashing.
        if path.is_symlink() or path.resolve().name != path.name:
            raise ValueError("Source symlink rejected")
        sources[cid] = s
    ids = set()
    for w in plan["windows"]:
        s = sources[w["clip_id"]]
        x, y, width, height = w["crop_xywh"]
        h, width_source = s["shape_hw"]
        if (
            w["id"] in ids
            or not w["id"].replace("_", "").isalnum()
            or not 0 <= w["first"] <= w["last"] < s["frames"]
            or w["last"] - w["first"] >= 60
            or not (0 <= x < x + width <= width_source)
            or not (0 <= y < y + height <= h)
        ):
            raise ValueError("Invalid bounded review window")
        ids.add(w["id"])
    return sources


def prepare(plan_path, output):
    plan = json.loads(plan_path.read_text())
    sources = validate_plan(plan)
    for s in sources.values():
        if digest(s["path"]) != s["sha256"]:
            raise ValueError("Source hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for cid, s in sources.items():
        cap = cv2.VideoCapture(s["path"])
        try:
            if (
                not cap.isOpened()
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != s["frames"]
                or abs(cap.get(cv2.CAP_PROP_FPS) - s["fps"]) > 1e-6
            ):
                raise ValueError("Unexpected video metadata")
            for w in (w for w in plan["windows"] if w["clip_id"] == cid):
                x, y, width, height = w["crop_xywh"]
                count = w["last"] - w["first"] + 1
                canvas = np.full(
                    (((count + 3) // 4) * (height + 28), 4 * width, 3), 24, np.uint8,
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, w["first"])
                pixel_hashes = []
                for i, index in enumerate(range(w["first"], w["last"] + 1)):
                    ok, frame = cap.read()
                    if (
                        not ok
                        or list(frame.shape[:2]) != s["shape_hw"]
                        or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1
                    ):
                        raise ValueError("Source frame/shape mismatch")
                    patch = frame[y : y + height, x : x + width]
                    pixel_hashes.append(hashlib.sha256(patch.tobytes()).hexdigest())
                    left, top = (i % 4) * width, (i // 4) * (height + 28)
                    canvas[top + 28 : top + 28 + height, left : left + width] = patch
                    cv2.putText(
                        canvas,
                        f"{cid} f{index} ({index/s['fps']:.1f}s) native",
                        (left + 4, top + 19),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (240, 240, 240),
                        1,
                    )
                name = w["id"] + ".png"
                if not cv2.imwrite(str(output / name), canvas):
                    raise RuntimeError("Cannot write source review sheet")
                records.append(
                    dict(
                        **w,
                        sheet=name,
                        sheet_sha256=digest(output / name),
                        source_patch_bgr_sha256=pixel_hashes,
                    )
                )
        finally:
            cap.release()
    result = dict(
        schema="seaqr.source-only-review-packet.v1",
        plan=plan,
        plan_sha256=digest(plan_path),
        renderer_sha256=digest(__file__),
        detector_outputs_accessed=False,
        native_pixels=True,
        every_frame=True,
        enhancement=False,
        labels_assigned=False,
        windows=records,
    )
    with (output / "packet.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(output=str(output), windows=len(records))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.output)
