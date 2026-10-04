"""Declare two full hardware runs without changing the frozen V8c candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256

PARENT = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
PARENT_SHA = "32e48486425f99d28bca97a7f7316c4a6dd91757d57cc019fd9257e479128546"
SOURCES = {
    "0055": ("c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f", 689),
    "0082": ("465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117", 691),
}


def prepare(output):
    if sha256(PARENT / "freeze.json") != PARENT_SHA:
        raise ValueError("Parent freeze changed")
    parent = json.loads((PARENT / "freeze.json").read_text())
    for name, digest in parent["files_sha256"].items():
        if any(sha256(p) != digest for p in (ROOT / name, PARENT / "snapshot" / name)):
            raise ValueError(f"Frozen implementation changed: {name}")
    sources = []
    for clip, (digest, frames) in SOURCES.items():
        path = (
            ROOT.parent
            / "outputs/v7_frozen_evaluation_20260913/sources"
            / f"chunk_{clip}.avi"
        )
        if sha256(path) != digest:
            raise ValueError(f"Source hash mismatch: {clip}")
        cap = cv2.VideoCapture(str(path))
        try:
            probe = dict(
                frames=round(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                container_fps=cap.get(cv2.CAP_PROP_FPS),
            )
        finally:
            cap.release()
        if probe != dict(frames=frames, width=4784, height=3190, container_fps=10.0):
            raise ValueError(f"Unexpected source probe: {clip}: {probe}")
        sources.append(
            dict(
                clip_id=clip,
                path=str(path),
                sha256=digest,
                probe=probe,
                remote_path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
                labels="Unlabeled; presence/absence not independently established",
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PARENT / "freeze.json", output / "parent_freeze.json")
    for name in parent["files_sha256"]:
        target = output / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PARENT / "snapshot" / name, target)
    manifest = dict(
        schema="seaqr.phase20.v8c-full-transfer-freeze.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_freeze_sha256=PARENT_SHA,
        files_sha256=parent["files_sha256"],
        sources=sources,
        planned_runs=[
            dict(clip=c, backend="pva", full_clip=True, expected_frames=n)
            for c, (_, n) in SOURCES.items()
        ],
        workers=1,
        per_clip_timeout_seconds=2400,
        no_tuning=True,
        prior_exposure="Prior complete CPU V7 runs and four PVA frame-pair checks per clip; not an untouched final holdout",
        review_plan="Per clip, use existing review_phase20_clutter.py --unlabeled: seed 20260913, four random plus strongest and longest remaining per polarity, up to 12 proposals; inspect all contact sheets. Do not infer object counts or false-positive rates.",
        checks=[
            "Complete contiguous full clip",
            "Exact frozen package and configurations",
            "Actual PVA execution without CPU fallback",
            "Availability/reset/gap accounting",
            "Tracking capacity accounting",
            "Bounded unlabeled visual review",
        ],
        sealed_holdout_accessed=False,
        precision=None,
        recall=None,
        generalization_proven=False,
        tooling_sha256={
            Path(__file__).name: sha256(Path(__file__)),
            "review_phase20_clutter.py": sha256(
                ROOT / "scripts/review_phase20_clutter.py"
            ),
        },
    )
    with (output / "freeze.json").open("x") as f:
        json.dump(manifest, f, indent=2)
    print(
        json.dumps(
            dict(
                output=str(output.resolve()),
                sources=list(SOURCES),
                frozen_files=len(parent["files_sha256"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
