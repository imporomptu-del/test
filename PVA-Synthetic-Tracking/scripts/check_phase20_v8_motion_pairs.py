"""Four-pair controlled comparison using exactly the same PVA correspondences."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.config import load_config
from tiny_target.motion import (
    PvaMotionConfig,
    PvaPyrLkMotionEstimator,
    GlobalMotionConfig,
    fit_global_motion,
)
from tiny_target.types import Frame, TimestampSource
from tiny_target.visible_baseline import sha256
from tiny_target.visible_coverage import finite_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        sha256(a.source)
        != "c11a00c5360fe076ef5adb9ec30e74dfffbcbdce1fc40e2dc371e346677febfc"
    ):
        raise ValueError("Only the development clip027 is in scope")
    configs = [
        load_config(ROOT / name).raw
        for name in (
            "configs/tiny_target_phase12_cfar_test.yaml",
            "configs/evaluation/phase20_motion_v8.json",
        )
    ]
    if configs[0]["motion"] != configs[1]["motion"]:
        raise ValueError("PVA feature/flow settings must match")
    estimator = PvaPyrLkMotionEstimator(
        PvaMotionConfig.from_mapping(configs[0]["motion"])
    )
    cap = cv2.VideoCapture(str(a.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []
    for current in (1, 100, 300, 600):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current - 1)
        pair = []
        for i in (current - 1, current):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Missing frame")
            pair.append(
                Frame(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    round(i / fps * 1e9),
                    i,
                    "visible-baseline",
                    8,
                    TimestampSource.CONTAINER_RATE,
                )
            )
        matches = estimator.estimate(*pair)
        estimates = [
            fit_global_motion(
                matches, GlobalMotionConfig.from_mapping(c["global_motion"])
            )
            for c in configs
        ]
        results.append(
            dict(
                current_frame=current,
                correspondences=matches.to_dict(),
                old=estimates[0].to_dict(include_inlier_indices=False),
                new=estimates[1].to_dict(include_inlier_indices=False),
            )
        )
    cap.release()
    result = finite_json(
        dict(
            source_sha256=sha256(a.source),
            pairs=results,
            all_old_rejected=all(
                r["old"]["quality_status"] == "rejected" for r in results
            ),
            all_new_accepted=all(
                r["new"]["quality_status"] == "accepted" for r in results
            ),
        )
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
    if not result["all_new_accepted"]:
        raise SystemExit(2)
