"""Bounded no-tuning motion-only transfer check on two preselected non-holdout clips."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.config import load_config
from tiny_target.motion import (
    GlobalMotionConfig,
    PvaMotionConfig,
    PvaPyrLkMotionEstimator,
    fit_global_motion,
)
from tiny_target.types import Frame, TimestampSource
from tiny_target.visible_baseline import sha256
from tiny_target.visible_coverage import finite_json

SOURCES = {
    "0055": "c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f",
    "0082": "465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configs = [
        load_config(ROOT / path).raw
        for path in (
            "configs/tiny_target_phase12_cfar_test.yaml",
            "configs/evaluation/phase20_motion_v8.json",
        )
    ]
    assert configs[0]["motion"] == configs[1]["motion"]
    estimator = PvaPyrLkMotionEstimator(
        PvaMotionConfig.from_mapping(configs[1]["motion"])
    )
    results = []
    for clip, digest in SOURCES.items():
        source = a.source_root / f"chunk_{clip}.avi"
        if sha256(source) != digest:
            raise ValueError("Preselected source hash mismatch")
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        pairs = []
        for current in (1, 100, 300, 600):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current - 1)
            frames = []
            for i in (current - 1, current):
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Missing diagnostic frame")
                frames.append(
                    Frame(
                        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                        round(i / fps * 1e9),
                        i,
                        "visible-baseline",
                        8,
                        TimestampSource.CONTAINER_RATE,
                    )
                )
            matches = estimator.estimate(*frames)
            old, new = [
                fit_global_motion(
                    matches, GlobalMotionConfig.from_mapping(c["global_motion"])
                )
                for c in configs
            ]
            pairs.append(
                dict(
                    current_frame=current,
                    correspondences=matches.to_dict(include_points=False),
                    old=old.to_dict(include_inlier_indices=False),
                    new=new.to_dict(include_inlier_indices=False),
                )
            )
        cap.release()
        results.append(
            dict(
                clip=clip,
                source_sha256=digest,
                pairs=pairs,
                old_accepted=sum(
                    p["old"]["quality_status"] == "accepted" for p in pairs
                ),
                new_accepted=sum(
                    p["new"]["quality_status"] == "accepted" for p in pairs
                ),
            )
        )
    result = finite_json(
        dict(
            scope="Additional preselected four-pair motion-only checks, after candidate freeze; not full-clip detection coverage, accuracy, or final holdout evaluation",
            no_tuning=True,
            runs=results,
            script_sha256=sha256(Path(__file__)),
            config_sha256=sha256(ROOT / "configs/evaluation/phase20_motion_v8.json"),
            global_motion_sha256=sha256(ROOT / "tiny_target/motion/global_motion.py"),
            sparse_validator_sha256=sha256(
                ROOT / "tiny_target/motion/translation_support.py"
            ),
        )
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            [
                dict(
                    clip=r["clip"],
                    old_accepted=r["old_accepted"],
                    new_accepted=r["new_accepted"],
                )
                for r in results
            ],
            indent=2,
        )
    )
