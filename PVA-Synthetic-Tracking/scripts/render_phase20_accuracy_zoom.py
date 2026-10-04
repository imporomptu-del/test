"""Nearest-neighbor inspection of an existing source-only review sheet."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_phase20_accuracy_review import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--crop-xywh", type=int, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    packet = json.loads(a.packet.read_text())
    w = next(w for w in packet["windows"] if w["id"] == a.window)
    source = a.packet.parent / w["sheet"]
    if digest(source) != w["sheet_sha256"]:
        raise ValueError("Sheet mismatch")
    _, _, sw, sh = w["crop_xywh"]
    x, y, width, height = a.crop_xywh
    if not (0 <= x < x + width <= sw and 0 <= y < y + height <= sh):
        raise ValueError("Inspection outside reviewed native crop")
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise ValueError("Output exists; preserving")
    sheet = cv2.imread(str(source))
    count = w["last"] - w["first"] + 1
    canvas = np.full(
        (((count + 3) // 4) * (height * 4 + 28), width * 16, 3), 24, np.uint8
    )
    for i in range(count):
        left, top = (i % 4) * sw, (i // 4) * (sh + 28)
        patch = sheet[top + 28 + y : top + 28 + y + height, left + x : left + x + width]
        enlarged = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        left, top = (i % 4) * width * 4, (i // 4) * (height * 4 + 28)
        canvas[top + 28 : top + 28 + height * 4, left : left + width * 4] = enlarged
        cv2.putText(
            canvas,
            f'f{w["first"]+i} 4x nearest',
            (left + 3, top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (240, 240, 240),
            1,
        )
    if not cv2.imwrite(str(a.output), canvas):
        raise RuntimeError("Cannot write zoom")
    with a.output.with_suffix(".json").open("x") as f:
        json.dump(
            dict(
                packet_sha256=digest(a.packet),
                window=a.window,
                relative_crop_xywh=a.crop_xywh,
                scale=4,
                interpolation="nearest",
                contrast_enhancement=False,
                output_sha256=digest(a.output),
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
