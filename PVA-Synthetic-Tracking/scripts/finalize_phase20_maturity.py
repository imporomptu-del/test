"""Final surviving-candidate evidence; rejected faint-recovery trials stay rejected."""
import argparse
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run_phase20_maturity import write
from score_phase20_accuracy import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fresh = json.loads((a.root / "audit.json").read_text())
    replay = json.loads((a.root / "current_replay/audit.json").read_text())
    if sorted(Path(r["run"]).name for r in fresh["runs"]) != ["cpu_0055", "cpu_0082", "pva_0126"]:
        raise ValueError("Both full additional clips and the fresh PVA prefix are required")
    cfg = json.loads((a.root / "config.json").read_text())
    if cfg["learning_exclusion_radius_px"] or cfg["learning_protection_geometry"] != "circle" or cfg["tracking_association_prior"] != "hit_maturity":
        raise ValueError("Unexpected candidate configuration")
    if any(r["reviewed_regression_gate_pass"] is False for audit in (fresh, replay) for r in audit["runs"]):
        raise ValueError("Surviving candidate failed a frozen regression")
    old = ROOT / "results/tiny_target/phase20/maturity_v5_20260914"
    exact = {}
    for name in ("cpu_0029", "cpu_0126", "pva_0126"):
        n = 0
        with (old / name / "frames.jsonl").open() as x, (a.root / "current_replay" / name / "frames.jsonl").open() as y:
            for left, right in itertools.zip_longest(x, y):
                if left is None or right is None:
                    raise ValueError("Changed replay length")
                left, right = json.loads(left), json.loads(right)
                left.pop("timings_ms"); right.pop("timings_ms")
                if left != right:
                    raise ValueError(f"Disabled shape-learning option changed replay at {name}:{n}")
                n += 1
        exact[name] = n
    actual = json.loads((a.root / "pva_exact_replay_match.json").read_text())
    roundoff = json.loads((a.root / "pva_roundoff_diagnostic.json").read_text())
    for path, sha in roundoff["journals_sha256"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Numerically compared journal changed")
    # Exact comparison remains failed and is reported as such. This additional
    # cross-platform numerical check is not a change to detection/scoring gates.
    if not (roundoff["all_discrete_fields_identical"] and
            roundoff["exact_candidates_mapping_and_coverage"] and
            roundoff["maximum_absolute_numeric_difference"] < 1e-9):
        raise ValueError("Fresh PVA difference exceeds numerical-only diagnostic")
    anchors = json.loads((a.root / "pva_prefix_anchors.json").read_text())
    if not all(x["all_required_anchors_same_measured_id"] for x in anchors["sparse_anchor_results"]):
        raise ValueError("PVA anchor identity regression")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"],
                           cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"Ran (\d+) tests", tests.stderr)
    if tests.returncode or not match:
        raise RuntimeError(tests.stdout + tests.stderr)
    artifacts = [a.root / n for n in ("audit.json", "current_replay/audit.json", "freeze.json", "config.json", "pva_config.json",
        "pva_transfer_manifest.json", "pva_prefix_anchors.json", "pva_exact_replay_match.json", "pva_roundoff_diagnostic.json", "default_parity.json", "visual_review_notes.md")]
    for audit in (fresh, replay):
        for run in audit["runs"]:
            for name, sha in run["artifacts_sha256"].items():
                path = Path(run["run"]) / name
                if digest(path) != sha: raise ValueError("Audited result changed")
                artifacts.append(path)
    for cid in ("0055", "0082"):
        sample = a.root / ("review_" + cid) / "sample.json"
        if Path(json.loads(sample.read_text())["source_run"]).name != "cpu_" + cid:
            raise ValueError("Wrong source review")
        artifacts.append(sample)
        artifacts.extend(sorted(sample.parent.glob("sheet_*.png")))
    record = dict(tests_passed=int(match.group(1)), test_stdout=tests.stdout, test_stderr=tests.stderr,
        exact_repeat_replay_frames=exact, fresh_pva_compared_frames=actual["frames"],
        exact_fresh_pva_replay_match=actual["exact_candidate_tracking_coverage_match"],
        cross_platform_numerical_diagnostic=roundoff,
        fresh_video_frames=sum(r["frames"] for r in fresh["runs"]), actual_pva_pairs=sum(r["actual_pva_pairs"] for r in fresh["runs"]),
        all_frozen_reference_regressions_pass=True,
        remaining_126_visible_misses=[80, 81, 82, 83, 216],
        faint_signal_recovery_accepted=False,
        rejected_learning_experiments=["combined_v5_20260914", "shape_learning_v6_20260914"],
        additional_full_cpu_clips=["0055", "0082"],
        airborne_accuracy_verified=False, production_promoted=False,
        full_clip_pva_validation_completed=False, sealed_holdout_media_accessed=False,
        artifacts_sha256={str(p.resolve()): digest(p) for p in artifacts})
    tools_dir = a.root / "verification_tools"
    tools_dir.mkdir(exist_ok=False)
    scripts = [ROOT / "scripts" / n for n in ("run_phase20_maturity.py", "run_phase20_v5_full.py", "replay_phase20_tracking.py",
        "audit_phase20_v5.py", "audit_phase20_encounter_results.py", "score_phase20_accuracy.py", "finalize_phase20_maturity.py",
        "prepare_phase20_v5_pva.py", "compare_phase20_replay.py", "diagnose_phase20_replay_roundoff.py", "summarize_phase20_pva_prefix.py",
        "render_phase20_guard_comparison.py", "review_phase20_clutter.py")]
    scripts.extend(sorted((ROOT / "tests/unit").glob("*.py")))
    record["verification_tools_sha256"] = {}
    for path in scripts:
        dest = tools_dir / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        record["verification_tools_sha256"][str(path.relative_to(ROOT))] = digest(dest)
    write(a.output, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("test_stdout", "test_stderr", "artifacts_sha256", "verification_tools_sha256")}, indent=2))


if __name__ == "__main__":
    main()
