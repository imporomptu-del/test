"""Freeze V7 and an explicitly preselected, non-sealed transfer-test protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
GUARD = {
    "tiny_target/visible_baseline.py": "d35113ccb9b08bfe2688965a338b3eeea6911fea0d423dfdca3850f20fa234a1",
    "tiny_target/tracking/kalman.py": "082e4e9fa525ddede7f524357a24da6cf6944b6764c3febf31ecd5fb73bc5c11",
    "tiny_target/visible_quality.py": "bb7935fe76d892559d47cc398e6b7f94305cd38714a55ec2654531d1e2e411c7",
    "configs/evaluation/phase20_visible_v7.json": "6682f2952105450316b435a0260db4b384d90ef80d8b420977594a7506a32175",
    "configs/evaluation/phase20_visible_v7_pva.json": "e8a35ef886434756f4c38ca3ea7a40a87616faf312f98ab119acdc6ec876177d",
    "configs/tiny_target_phase12_cfar_test.yaml": "473b19b76f9a25035bf7b5d7f02b899144f3e3cd8369706712a012df350de4fe",
}
SOURCES = {
    "0027": (
        "/Users/romanmaksymiuk/Documents/SEAQR/outputs/jetson_review_clips_20260913/chunk_0027.avi",
        "c11a00c5360fe076ef5adb9ec30e74dfffbcbdce1fc40e2dc371e346677febfc",
    ),
    "0055": (
        "/Users/romanmaksymiuk/Documents/SEAQR/outputs/v7_frozen_evaluation_20260913/sources/chunk_0055.avi",
        "c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f",
    ),
    "0082": (
        "/Users/romanmaksymiuk/Documents/SEAQR/outputs/v7_frozen_evaluation_20260913/sources/chunk_0082.avi",
        "465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117",
    ),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def verify(directory):
    manifest = json.loads((directory / "freeze.json").read_text())
    for name, expected in manifest["implementation_sha256"].items():
        if (
            digest(ROOT / name) != expected
            or digest(directory / "snapshot" / name) != expected
        ):
            raise ValueError(f"Frozen implementation changed: {name}")
    for source in manifest["sources"]:
        if digest(source["local_path"]) != source["sha256"]:
            raise ValueError("Source changed")
    return manifest


def freeze(directory):
    for name, expected in GUARD.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Not the previously validated V7: {name}")
    sources = []
    for clip, (path, expected) in SOURCES.items():
        if digest(path) != expected:
            raise ValueError(f"Unexpected source bytes: {clip}")
        sources.append(
            dict(
                clip_id=clip,
                local_path=path,
                sha256=expected,
                remote_path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
                labels="unlabeled; presence/absence not established",
            )
        )
    directory.mkdir(parents=True, exist_ok=False)
    files = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "tiny_target").rglob("*.py")
    )
    files += [name for name in GUARD if name.startswith("configs/")]
    hashes = {name: digest(ROOT / name) for name in files}
    for name in files:
        target = directory / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = dict(
        schema="seaqr.phase20.frozen-transfer-evaluation.v1",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        selection="027 previously suggested by user, now explicitly unconfirmed; 055/082 later non-sealed recordings, preselected without inspecting detection outcomes. 090 excluded before media access because prior evaluation use was found.",
        interpretation="Not used to choose V7; same-camera transfer test, not independent-camera or final sealed-holdout evidence.",
        sources=sources,
        implementation_sha256=hashes,
        runs=[
            dict(clip_id=c, backend="cpu_translation", full_clip=True) for c in SOURCES
        ]
        + [dict(clip_id="0027", backend="pva", full_clip=True)],
        no_tuning=True,
        no_replacement_of_failed_clips=True,
        annotation_policy="Source-only review may nominate references; proposal-guided references must be identified as such. No exhaustive truth assumed. Never feed either into detection.",
        review_sample=dict(
            seed=20260913,
            policy="Per polarity: four random qualified IDs plus strongest and longest remaining; no clean-output selection",
            count_max_per_run=12,
        ),
        metrics=[
            "completion",
            "coverage/warmup/resets",
            "PVA errors",
            "capacity losses",
            "qualified proposal workload",
            "native-resolution review samples",
        ],
        unavailable_without_independent_truth=[
            "recall",
            "precision",
            "false positives/minute",
            "identity switches of actual physical objects",
        ],
        sealed_holdout_access=False,
    )
    (directory / "freeze.json").write_text(json.dumps(manifest, indent=2))
    return verify(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = verify(args.output) if args.verify else freeze(args.output)
    print(
        json.dumps(
            dict(
                frozen_files=len(value["implementation_sha256"]),
                clips=[s["clip_id"] for s in value["sources"]],
                manifest=str(args.output / "freeze.json"),
                verified=True,
            )
        )
    )
