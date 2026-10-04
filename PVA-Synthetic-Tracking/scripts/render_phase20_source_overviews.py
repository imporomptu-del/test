"""Source-only scene overview; downsampling must not be used to label absence."""
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
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    freeze = json.loads(a.freeze.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source in freeze["sources"]:
        if sha256(source["local_path"]) != source["sha256"]:
            raise ValueError("Source mismatch")
        cap = cv2.VideoCapture(source["local_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = [min(count - 1, round(t * fps)) for t in (0, 20, 40, 60)]
        panels = []
        for frame_index in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Decode failed")
            width = 1196
            resized = cv2.resize(
                frame,
                (width, round(frame.shape[0] * width / frame.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.full((resized.shape[0] + 45, width, 3), 24, np.uint8)
            panel[45:] = resized
            cv2.putText(
                panel,
                f'{source["clip_id"]} f{frame_index} ({frame_index/fps:.1f}s) | 25% OVERVIEW: NOT suitable for tiny-target absence labels',
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (240, 240, 240),
                1,
            )
            panels.append(panel)
        cap.release()
        canvas = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        dest = a.output / f'chunk{source["clip_id"]}_overview.jpg'
        if not cv2.imwrite(str(dest), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError("Image write failed")
        manifest.append(
            dict(
                **source,
                source_frame_indices=frames,
                fps=fps,
                total_frames=count,
                overview=str(dest),
            )
        )
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                source_only=True,
                detector_outputs_accessed=False,
                purpose="Scene context, not exhaustive object/absence annotation",
                clips=manifest,
            ),
            indent=2,
        )
    )
    print(json.dumps(dict(clips=len(manifest), output=str(a.output))))


if __name__ == "__main__":
    main()
