"""Integrity, test and acceptance record for the two isolated V3 experiments."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig
from tiny_target.visible_regression import score
from score_phase20_accuracy import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variance-root", type=Path, required=True)
    p.add_argument("--assignment-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    variance = json.loads((a.variance_root / "verified_summary.json").read_text())
    reference = ROOT / "results/tiny_target/phase20/encounter_accuracy_v2_20260914"
    if (
        digest(reference / "annotations.json")
        != "56b98b77b8acdfabc564978b80203ca0e9a698bffbca9c8d44e46ff886e67fed"
    ):
        raise ValueError("Frozen labels changed")
    frozen = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
    manifest = json.loads((frozen / "freeze.json").read_text())
    for name, sha in manifest["files_sha256"].items():
        if "phase18" in name:
            raise ValueError("Sealed split access forbidden")
        if digest(frozen / "snapshot" / name) != sha:
            raise ValueError("Frozen implementation changed")
    cfg = VisibleConfig()
    if (
        cfg.learning_exclusion_radius_px != 0
        or cfg.tracking_association_assignment != "greedy"
        or cfg.tracking_association_cascade != "none"
    ):
        raise ValueError("Experimental policy promoted unexpectedly")
    dev = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    anchors = []
    for cid in ("0029", "0126"):
        ref = next(c for c in dev["clips"] if c["clip_id"] == "chunk" + cid)
        baseline = score(
            ROOT
            / "results/tiny_target/phase20/v8_motion_fix_20260913"
            / ("cpu_chunk" + cid),
            ROOT / ref["annotations"],
        )
        current = json.loads(
            (a.variance_root / ("cpu_" + cid) / "evaluation.json").read_text()
        )["original_anchor_regression"]
        for before, after in zip(baseline["events"], current["events"]):
            anchors.append(
                dict(
                    clip_id=cid,
                    event_id=before["event_id"],
                    required=before["required_anchor_count"],
                    baseline_dominant_hits=before["dominant_track_anchor_hits"],
                    variance_only_dominant_hits=after["dominant_track_anchor_hits"],
                )
            )
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stdout + proc.stderr)
    count = re.search(r"Ran (\d+) tests", proc.stderr)
    if not count:
        raise ValueError("Missing test result")
    assignment = json.loads((a.assignment_root / "audit.json").read_text())
    result = dict(
        tests_passed=int(count.group(1)),
        test_stdout=proc.stdout,
        test_stderr=proc.stderr,
        frozen_v8c_snapshot_files_intact=len(manifest["files_sha256"]),
        frozen_labels_sha256=variance["labels_sha256"],
        variance_only_no_new_visible_misses=all(
            not c["newly_missed_frames"]
            for r in variance["runs"]
            for c in r["comparison"]
        ),
        variance_only_cpu_anchor_identity=anchors,
        assignment_not_promoted=True,
        variance_only_not_promoted=True,
        variance_only_fresh_0055_0082_nuisance_runs_completed=False,
        assignment_reviewed_light_response_frames=[
            r["experiment"]["light_field_review"]["measured_qualified_response_frames"]
            for r in assignment["runs"]
            if r["experiment"]["light_field_review"]
        ],
        verified_airborne_accuracy=False,
        note="Assignment nuisance checks reuse frozen detections; they do not validate the new variance policy. Full PVA126 and fresh nuisance validation remain pending. Experiments were not combined.",
        artifacts_sha256={
            str(path): digest(path)
            for path in (
                a.variance_root / "verified_summary.json",
                a.assignment_root / "audit.json",
                a.assignment_root / "default_parity.json",
            )
        },
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("test_stdout", "test_stderr")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
