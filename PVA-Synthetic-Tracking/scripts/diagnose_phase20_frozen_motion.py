"""Separate fixed-configuration motion diagnostics; never modify evaluation runs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.config import load_config
from tiny_target.motion import (
    PvaMotionConfig,
    PvaPyrLkMotionEstimator,
    GlobalMotionConfig,
    fit_global_motion,
)
from tiny_target.types import Frame, TimestampSource
from tiny_target.visible_baseline import sha256


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--motion-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        sha256(a.source)
        != "c11a00c5360fe076ef5adb9ec30e74dfffbcbdce1fc40e2dc371e346677febfc"
    ):
        raise ValueError("Diagnostic is scoped only to the preselected chunk027")
    if (
        sha256(a.motion_config)
        != "473b19b76f9a25035bf7b5d7f02b899144f3e3cd8369706712a012df350de4fe"
    ):
        raise ValueError("Do not tune the frozen motion configuration")
    raw = load_config(a.motion_config).raw
    motion_config = PvaMotionConfig.from_mapping(raw.get("motion"))
    global_config = GlobalMotionConfig.from_mapping(raw.get("global_motion"))
    estimator = PvaPyrLkMotionEstimator(motion_config)
    cap = cv2.VideoCapture(str(a.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pairs = []
    for current in (1, 100, 300, 600):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current - 1)
        frames = []
        for index in (current - 1, current):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Source pair missing")
            frames.append(
                Frame(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    round(index / fps * 1e9),
                    index,
                    "visible-baseline",
                    8,
                    TimestampSource.CONTAINER_RATE,
                )
            )
        correspondence = estimator.estimate(*frames)
        estimate = fit_global_motion(correspondence, global_config)
        pairs.append(
            dict(
                current_frame=current,
                correspondences=correspondence.to_dict(include_points=False),
                global_motion=estimate.to_dict(include_inlier_indices=False),
            )
        )
    cap.release()
    result = dict(
        scope="Four post-run diagnostic pairs, unchanged settings, not a rerun/replacement of the frozen full-clip test",
        source_sha256=sha256(a.source),
        motion_config_sha256=sha256(a.motion_config),
        motion_config=asdict(motion_config),
        global_config=asdict(global_config),
        pairs=pairs,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
