"""Freeze a bounded development validation; never open any holdout media."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256

SOURCES = {
    "0027": "c11a00c5360fe076ef5adb9ec30e74dfffbcbdce1fc40e2dc371e346677febfc",
    "0029": "0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359",
    "0126": "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344",
}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cpu-reference",
        type=Path,
        help="Reuse CPU regressions only if the sole package difference is the PVA-only sparse validator",
    )
    a = p.parse_args()
    media = ROOT.parent / "outputs/jetson_review_clips_20260913"
    sources = []
    for clip, digest in SOURCES.items():
        path = media / f"chunk_{clip}.avi"
        if sha256(path) != digest:
            raise ValueError(f"Source mismatch: {clip}")
        sources.append(dict(clip_id=clip, path=str(path), sha256=digest))
    old = json.loads((ROOT / "configs/tiny_target_phase12_cfar_test.yaml").read_text())
    new = json.loads((ROOT / "configs/evaluation/phase20_motion_v8.json").read_text())
    assert old["motion"] == new["motion"]
    assert old["stabilization"] == new["stabilization"]
    assert all(new["global_motion"][k] == v for k, v in old["global_motion"].items())
    if a.cpu_reference:
        prior = json.loads((a.cpu_reference / "freeze.json").read_text())
        package_names = {
            str(p.relative_to(ROOT)) for p in (ROOT / "tiny_target").rglob("*.py")
        }
        assert package_names == {
            n for n in prior["files_sha256"] if n.startswith("tiny_target/")
        }
        differences = {
            n for n in package_names if sha256(ROOT / n) != prior["files_sha256"][n]
        }
        assert differences == {"tiny_target/motion/translation_support.py"}, differences
    a.output.mkdir(parents=True, exist_ok=False)
    files = list(sorted((ROOT / "tiny_target").rglob("*.py"))) + [
        ROOT / "configs/evaluation/phase20_visible_v7.json",
        ROOT / "configs/evaluation/phase20_visible_v7_pva.json",
        ROOT / "configs/evaluation/phase20_motion_v8.json",
        ROOT / "configs/tiny_target_phase12_cfar_test.yaml",
        ROOT / "scripts/check_phase20_v8_motion_pairs.py",
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = a.output / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(path)
    result = dict(
        schema="seaqr.phase20.v8-development-freeze.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        files_sha256=hashes,
        sources=sources,
        changes="Opt-in sparse translation consensus and explicit detection availability; V7 detector/tracker controls and old motion limits unchanged.",
        planned_runs=[
            dict(clip="0027", backend="pva", full_clip=True),
            dict(clip="0126", backend="pva", max_frames=230, full_clip=False),
            dict(clip="0029", backend="cpu_translation", full_clip=True),
            dict(clip="0126", backend="cpu_translation", full_clip=True),
        ],
        preflight="Four fixed PVA frame pairs on 027 (ending 1,100,300,600), compare old/new gates on identical correspondences before full runs.",
        gates=[
            "027 usable coverage restored with no PVA runtime failures",
            "all 24 CPU confident anchors retained",
            "all 6 PVA 126 confident anchors in prefix retained",
            "synthetic bad-motion rejection and recovery tests pass",
        ],
        limits="Development regression, not fresh generalization or precision/recall; 027 unlabeled; no holdout access; do not overwrite failed attempts.",
    )
    if a.cpu_reference:
        result["cpu_regression_reference"] = dict(
            root=str(a.cpu_reference.resolve()),
            freeze_sha256=sha256(a.cpu_reference / "freeze.json"),
            package_differences=["tiny_target/motion/translation_support.py"],
            rationale="The sole package difference is used only by PVA global motion; CPU translation, detector, tracker, availability and all their dependencies are identical. No CPU result is promoted to hardware evidence.",
        )
        for run in result["planned_runs"]:
            if run["backend"] == "cpu_translation":
                run["reference_run"] = str(
                    (a.cpu_reference / f'cpu_chunk{run["clip"]}').resolve()
                )
    with (a.output / "freeze.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            dict(
                output=str(a.output.resolve()), files=len(hashes), sources=list(SOURCES)
            )
        )
    )
