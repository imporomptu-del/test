"""Close-out integrity/tests audit; never reads media or the sealed split."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

from score_phase20_accuracy import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    freeze = json.loads((root / "scoring_freeze.json").read_text())
    assert digest(root / "annotations.json") == freeze["labels_sha256"]
    assert digest(root / "scoring_packet.json") == freeze["packet_sha256"]
    packet = json.loads((root / "source_review/packet.json").read_text())
    proposals = json.loads((root / "annotation_proposals/proposals.json").read_text())
    implementation = root / "review_implementation"
    assert (
        digest(implementation / "prepare_phase20_encounter_review.py")
        == packet["plan"]["renderer_sha256"]
    )
    assert (
        digest(implementation / "localize_phase20_review_samples.py")
        == proposals["localizer_sha256"]
    )
    acceptance = json.loads((root / "review_acceptance.json").read_text())
    assert (
        digest(root / "annotation_proposals/proposals.json")
        == acceptance["proposal_sha256"]
    )
    for ep in packet["episodes"]:
        assert (
            digest(root / "source_review" / ep["native_archive"])
            == ep["native_archive_sha256"]
        )
        for s in ep["sheets"]:
            assert digest(root / "source_review" / s["path"]) == s["sha256"]
    parent = ROOT / "results/tiny_target/phase20/v8c_motion_fix_20260913"
    previous = json.loads((parent / "freeze.json").read_text())
    differences = []
    for name, sha in previous["files_sha256"].items():
        if "phase18" in name:
            raise ValueError("Sealed-split access forbidden")
        assert digest(parent / "snapshot" / name) == sha
        if digest(ROOT / name) != sha:
            differences.append(name)
    phase19 = {
        "configs/evaluation/phase19_dense_screen_v1.json": "e9eb5d86e64beb8bcaf3ffb77967120e1745b16838eff9722aa49657e940a8ed",
        "tiny_target/dense_screen.py": "86a811c08c80ee2089f9ead363e51150392dd804191b5798240a05dfa0ed7628",
        "configs/tiny_target_phase12_cfar_test.yaml": "473b19b76f9a25035bf7b5d7f02b899144f3e3cd8369706712a012df350de4fe",
    }
    for name, sha in phase19.items():
        assert digest(ROOT / name) == sha
    # Formatting may change hashes but must not change tested runtime semantics.
    snap = root / "learning_guard_experiment/cpu_0029/implementation"
    for name in ("visible_baseline.py", "tracking/kalman.py", "visible_quality.py"):
        assert ast.dump(
            ast.parse((ROOT / "tiny_target" / name).read_text())
        ) == ast.dump(ast.parse((snap / name).read_text())), name
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    tests = re.search(r"Ran (\d+) tests", proc.stderr)
    if proc.returncode or tests is None:
        raise RuntimeError(proc.stdout + proc.stderr)
    guard = json.loads((root / "learning_guard_experiment/summary.json").read_text())
    events = [
        e for r in guard["runs"] for e in r["original_anchor_regression"]["events"]
    ]
    assert sum(e["required_anchor_count"] for e in events) == 24
    original_pass = all(
        e["dominant_track_anchor_hits"] == e["required_anchor_count"] for e in events
    )
    result = dict(
        tests_passed=int(tests.group(1)),
        test_stdout=proc.stdout,
        test_stderr=proc.stderr,
        frozen_v8c_snapshot_files_intact=len(previous["files_sha256"]),
        intentional_current_differences=differences,
        frozen_phase19_files_intact=True,
        source_review_and_annotation_integrity=True,
        formatted_runtime_ast_matches_tested_snapshot=True,
        original_24_cpu_anchor_checks_pass=original_pass,
        current_code_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "tiny_target/visible_baseline.py",
                ROOT / "tiny_target/tracking/kalman.py",
                ROOT / "scripts/replay_phase20_tracking.py",
            ]
        },
        no_sealed_split_or_media_read=True,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("test_stdout", "test_stderr", "current_code_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
