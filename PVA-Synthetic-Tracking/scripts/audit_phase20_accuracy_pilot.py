"""Verify unchanged detector, source-review provenance and original regressions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_regression import score
from score_phase20_accuracy import digest


def audit(root):
    packet_path = root / "source_review/packet.json"
    packet = json.loads(packet_path.read_text())
    freeze = json.loads((root / "scoring_freeze.json").read_text())
    if (
        digest(root / "annotations.json") != freeze["labels_sha256"]
        or digest(packet_path) != freeze["packet_sha256"]
    ):
        raise ValueError("Review labels changed after scoring freeze")
    for w in packet["windows"]:
        if digest(packet_path.parent / w["sheet"]) != w["sheet_sha256"]:
            raise ValueError("Source review sheet changed")
    if (
        digest(
            root / "initial_review_implementation/prepare_phase20_accuracy_review.py"
        )
        != packet["renderer_sha256"]
    ):
        raise ValueError("Initial renderer provenance lost")
    for name in ("029_A_zoom", "029_B_zoom", "126_motion_zoom"):
        meta = json.loads((packet_path.parent / f"{name}.json").read_text())
        if (
            meta["packet_sha256"] != digest(packet_path)
            or digest(packet_path.parent / f"{name}.png") != meta["output_sha256"]
        ):
            raise ValueError("Zoom provenance mismatch")
    parent_path = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
    parent = json.loads((parent_path / "freeze.json").read_text())
    for name, value in parent["files_sha256"].items():
        if (
            digest(ROOT / name) != value
            or digest(parent_path / "snapshot" / name) != value
        ):
            raise ValueError(f"Frozen detector changed: {name}")
    phase19 = ROOT / "configs/evaluation/phase19_dense_screen_v1.json"
    if (
        digest(phase19)
        != "e9eb5d86e64beb8bcaf3ffb77967120e1745b16838eff9722aa49657e940a8ed"
    ):
        raise ValueError("Phase 19 frozen configuration changed")
    development = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    regressions = []
    for c in development["clips"]:
        spec = next(
            s
            for s in freeze["runs"]
            if "chunk" + s["clip_id"] == c["clip_id"]
            and s["backend"] == "cpu_translation"
        )
        result = score(spec["run"], ROOT / c["annotations"])
        regressions.append(
            dict(
                clip_id=c["clip_id"],
                reference_sha256=digest(ROOT / c["annotations"]),
                result=result,
            )
        )
    events = [e for r in regressions for e in r["result"]["events"]]
    retained = sum(e["required_anchor_count"] for e in events) == 24 and all(
        e["dominant_track_anchor_hits"] == e["required_anchor_count"] for e in events
    )
    if not retained:
        raise ValueError("Original confident-anchor regression failed")
    initial = json.loads((root / "summary.json").read_text())
    verified = json.loads((root / "verified_summary.json").read_text())
    initial.pop("scorer_sha256")
    verified_scorer = verified.pop("scorer_sha256")
    if initial != verified or verified_scorer != digest(
        ROOT / "scripts/score_phase20_accuracy.py"
    ):
        raise ValueError("Scoring changed during final formatting/verification")
    result = dict(
        schema="seaqr.accuracy-pilot-audit.v1",
        frozen_detector_files_unchanged=len(parent["files_sha256"]),
        original_24_confident_cpu_anchors_retained=retained,
        original_regressions=regressions,
        phase19_frozen_configuration_unchanged=True,
        source_sheet_integrity_pass=True,
        initial_and_verified_scores_equivalent=True,
        annotations_sha256=digest(root / "annotations.json"),
        verified_summary_sha256=digest(root / "verified_summary.json"),
        review_plan_sha256=packet["plan_sha256"],
        no_new_inference_jobs=True,
        current_evaluation_code_sha256={
            n: digest(ROOT / n)
            for n in (
                "scripts/prepare_phase20_accuracy_review.py",
                "scripts/render_phase20_accuracy_zoom.py",
                "scripts/score_phase20_accuracy.py",
                "scripts/audit_phase20_accuracy_pilot.py",
                "tests/unit/test_accuracy_pilot.py",
            )
        },
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("original_regressions", "current_evaluation_code_sha256")
            },
            indent=2,
        )
    )
