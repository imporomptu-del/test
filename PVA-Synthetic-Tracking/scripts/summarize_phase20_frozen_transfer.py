"""Report the predeclared frozen runs without treating unlabeled clips as empty."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256
from freeze_phase20_v7_evaluation import verify


def summarize(path, clip, backend, freeze):
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    source = next(s for s in freeze["sources"] if s["clip_id"] == clip)
    config_name = (
        "configs/evaluation/phase20_visible_v7"
        + ("_pva" if backend == "pva" else "")
        + ".json"
    )
    if not (launch["source_sha256"] == report["source_sha256"] == source["sha256"]):
        raise ValueError("Source mismatch")
    if launch["config_sha256"] != freeze["implementation_sha256"][config_name]:
        raise ValueError("Configuration was changed")
    for key, value in launch["code_sha256"].items():
        if value != freeze["implementation_sha256"]["tiny_target/" + key]:
            raise ValueError("Implementation was changed")
    if (
        backend == "pva"
        and launch["motion_config_sha256"]
        != freeze["implementation_sha256"]["configs/tiny_target_phase12_cfar_test.yaml"]
    ):
        raise ValueError("Motion configuration was changed")
    if (
        not report["completed"]
        or not report["full_clip"]
        or report["frames"] != launch["expected_frames"]
    ):
        raise ValueError(
            "Complete full-clip run required; failures must be reported separately"
        )
    counts = Counter()
    longest = current = 0
    with (path / "frames.jsonl").open() as f:
        for i, row in enumerate(map(json.loads, f)):
            if row["frame_index"] != i:
                raise ValueError("Non-contiguous journal")
            counts["frames"] += 1
            warmup = row["coverage"]["warmup"]
            counts["warmup_frames"] += warmup
            counts["detection_ready_frames"] += not warmup
            current = current + 1 if not warmup else 0
            longest = max(longest, current)
            counts["motion_resets"] += row["motion"]["reset"]
            counts["pva_runtime_errors"] += row["motion"].get("pva_failure", False)
            if "accepted" in row["motion"]:
                counts["motion_fits_attempted"] += 1
                counts["motion_fits_accepted"] += row["motion"]["accepted"]
            counts["candidate_count"] += len(row["candidates"])
    if counts["frames"] != report["frames"]:
        raise ValueError("Frame count mismatch")
    return dict(
        clip_id=clip,
        backend=backend,
        run=str(path.resolve()),
        source_sha256=source["sha256"],
        config_sha256=launch["config_sha256"],
        completed_full_clip=True,
        counts=dict(counts),
        detection_ready_fraction=counts["detection_ready_frames"] / counts["frames"],
        longest_detection_ready_run_frames=longest,
        usable_detection_coverage=counts["detection_ready_frames"] > 0,
        qualified_proposal_count=report["qualified_track_count"],
        cap_drops=report["counts"],
        processed_fps=report["processed_fps"],
        performance_note="Exploratory run, not a controlled real-time benchmark; CPU runs concurrent",
        precision=None,
        recall=None,
        actual_object_id_switches=None,
        false_positives_per_minute=None,
        verdict=(
            "unusable_no_detection_ready_frames"
            if counts["detection_ready_frames"] == 0
            else "unlabeled_review_required"
        ),
        report_sha256=sha256(path / "report.json"),
        journal_sha256=sha256(path / "frames.jsonl"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frozen = verify(a.root)
    runs = []
    for run in frozen["runs"]:
        prefix = "pva" if run["backend"] == "pva" else "cpu"
        runs.append(
            summarize(
                a.root / f'{prefix}_chunk{run["clip_id"]}',
                run["clip_id"],
                run["backend"],
                frozen,
            )
        )
    result = dict(
        schema="seaqr.phase20.frozen-transfer-results.v1",
        freeze_sha256=sha256(a.root / "freeze.json"),
        all_predeclared_runs_present=True,
        frozen_implementation_verified=True,
        generalization_proven=False,
        runs=runs,
        caveat="No independent positive/negative truth; no-detection runs are not evidence of empty scenes. No tuning during evaluation.",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
