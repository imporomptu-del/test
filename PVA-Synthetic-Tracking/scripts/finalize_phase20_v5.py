"""Final bounded-experiment integrity and test record (no production promotion)."""
import argparse
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
    audit = json.loads((a.root / "audit.json").read_text())
    runs = audit["runs"]
    if sorted(Path(r["run"]).name for r in runs) != sorted(
        ["cpu_0029", "cpu_0126", "pva_0126", "cpu_0055", "cpu_0082"]):
        raise ValueError("All five fresh runs are required")
    freeze = json.loads((a.root / "freeze.json").read_text())
    for name, sha in freeze["code_sha256"].items():
        if not name.startswith("tiny_target/"):
            raise ValueError("Unexpected runtime snapshot path")
        snapshot_file = a.root / "cpu_0029/implementation" / name.removeprefix("tiny_target/")
        if digest(snapshot_file) != sha:
            raise ValueError("Frozen experiment snapshot changed")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"],
                           cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"Ran (\d+) tests", tests.stderr)
    if tests.returncode or not match:
        raise RuntimeError(tests.stdout + tests.stderr)
    artifacts = [a.root / n for n in ("audit.json", "freeze.json", "config.json",
                 "pva_config.json", "pva_transfer_manifest.json", "pva_prefix_anchors.json")]
    for r in runs:
        path = Path(r["run"])
        for name, sha in r["artifacts_sha256"].items():
            if digest(path / name) != sha:
                raise ValueError("Audited run changed")
        if "replay" in r["execution_mode"]:
            raise ValueError("Fresh source execution required here")
    for cid in ("0055", "0082"):
        sample = a.root / ("review_" + cid) / "sample.json"
        review = json.loads(sample.read_text())
        artifacts.append(sample)
        if Path(review["source_run"]).name != "cpu_" + cid:
            raise ValueError("Wrong review source")
        artifacts.extend(sorted(sample.parent.glob("sheet_*.png")))
    anchor = json.loads((a.root / "pva_prefix_anchors.json").read_text())
    required_anchors_ok = all(e["all_required_anchors_same_measured_id"] for e in anchor["sparse_anchor_results"])
    record = dict(tests_passed=int(match.group(1)), test_stdout=tests.stdout, test_stderr=tests.stderr,
        fresh_source_frames=sum(r["frames"] for r in runs),
        actual_pva_motion_pairs=sum(r["actual_pva_pairs"] for r in runs),
        reviewed_regression_gate_pass=all(r["reviewed_regression_gate_pass"] is not False for r in runs) and required_anchors_ok,
        remaining_cpu_visible_misses={r["clip_id"]: [f for c in r["comparison_to_original"] for f in c["remaining_misses"]]
            for r in runs if Path(r["run"]).name.startswith("cpu_") and r["comparison_to_original"]},
        unit_tests_do_not_establish_airborne_accuracy=True, production_promoted=False,
        full_clip_pva_validation_completed=False, airborne_accuracy_verified=False,
        runtime_frozen=True, sealed_holdout_media_accessed=False,
        reviewed_extra_clips=["0055", "0082"],
        artifacts_sha256={str(p.resolve()): digest(p) for p in artifacts})
    snapshot = a.root / "verification_tools"
    snapshot.mkdir(exist_ok=False)
    tool_paths = [ROOT / "scripts" / n for n in (
        "run_phase20_maturity.py", "run_phase20_v5_full.py", "replay_phase20_tracking.py",
        "audit_phase20_v5.py", "audit_phase20_encounter_results.py",
        "score_phase20_accuracy.py", "finalize_phase20_v5.py",
        "prepare_phase20_v5_pva.py", "summarize_phase20_pva_prefix.py",
        "render_phase20_guard_comparison.py", "review_phase20_clutter.py")]
    tool_paths.extend(sorted((ROOT / "tests/unit").glob("*.py")))
    record["verification_tools_sha256"] = {}
    for tool in tool_paths:
        relative = tool.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tool, target)
        record["verification_tools_sha256"][str(relative)] = digest(target)
    write(a.output, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("test_stdout", "test_stderr", "artifacts_sha256")}, indent=2))


if __name__ == "__main__":
    main()
