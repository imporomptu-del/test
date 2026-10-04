"""Source-only assisted annotation proposals, requiring subsequent visual review.

Visibility is assigned before localization. This is not detector ground truth:
every proposal needs visual acceptance. No inference outputs are read.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from score_phase20_accuracy import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    packet = json.loads((a.source_review / "packet.json").read_text())
    visible = {
        "029_A": [40, 47, 50, 53, 59, 62, 65, 72, 75, 78, 81, 84, 87, 90],
        "029_B": list(range(220, 352)),
        "126": list(range(80, 219)),
    }
    a.output.mkdir(parents=True, exist_ok=False)
    output = []
    for ep in packet["episodes"]:
        if ep["id"] not in visible:
            continue
        archive = a.source_review / ep["native_archive"]
        if digest(archive) != ep["native_archive_sha256"]:
            raise ValueError("Source archive changed")
        origins = {r["frame_index"]: r["crop_xywh"] for r in ep["frames"]}
        entries, sheets = [], []
        with np.load(archive, allow_pickle=False) as patches:
            for i, f in enumerate(visible[ep["id"]]):
                im = patches[str(f)]
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)
                # Broad manually identified central neighborhood; source only.
                # Weighted centroid of the connected bright component containing
                # its maximum. This estimate still needs visual acceptance.
                roi = gray[44:85, 44:85]
                yy, xx = np.unravel_index(np.argmax(roi), roi.shape)
                base = float(np.median(roi))
                threshold = base + 0.45 * (float(roi.max()) - base)
                _, components = cv2.connectedComponents(
                    (roi > threshold).astype(np.uint8)
                )
                component = components == components[yy, xx]
                weights = np.maximum(roi - base, 0) * component
                ys, xs = np.indices(roi.shape)
                xy = [
                    float((weights * xs).sum() / weights.sum()) + 44,
                    float((weights * ys).sum() / weights.sum()) + 44,
                ]
                origin = origins[f]
                entries.append(
                    dict(
                        frame_index=f,
                        xy=[round(origin[j] + xy[j], 2) for j in (0, 1)],
                        patch_xy=xy,
                        uncertainty_px=5.0,
                        visually_accepted=False,
                        source_crop_xywh=origin,
                    )
                )
                if i % 16 == 0:
                    canvas = np.full((4 * 282, 4 * 256, 3), 24, np.uint8)
                    first = f
                shown = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
                center = tuple(round(v * 2) for v in xy)
                # Ring leaves the source point unobscured.
                cv2.circle(shown, center, 13, (0, 190, 190), 1)
                left, top = (i % 4) * 256, ((i % 16) // 4) * 282
                canvas[top + 26 : top + 282, left : left + 256] = shown
                cv2.putText(
                    canvas,
                    f'{ep["id"]} f{f} proposal',
                    (left + 4, top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (240, 240, 240),
                    1,
                )
                if i % 16 == 15 or i == len(visible[ep["id"]]) - 1:
                    name = f'{ep["id"]}_{first:04d}_{f:04d}_proposals.png'
                    if not cv2.imwrite(str(a.output / name), canvas):
                        raise RuntimeError("Write failed")
                    sheets.append(dict(path=name, sha256=digest(a.output / name)))
        output.append(
            dict(
                id=ep["id"],
                samples=entries,
                sheets=sheets,
                unknown_frames=sorted(
                    set(range(ep["first"], ep["last"] + 1)) - set(visible[ep["id"]])
                ),
            )
        )
    with (a.output / "proposals.json").open("x") as f:
        json.dump(
            dict(
                source_packet_sha256=digest(a.source_review / "packet.json"),
                localizer_sha256=digest(__file__),
                episodes=output,
                visibility_assigned_before_localization=True,
                detector_outputs_read=False,
                accepted_as_truth=False,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
