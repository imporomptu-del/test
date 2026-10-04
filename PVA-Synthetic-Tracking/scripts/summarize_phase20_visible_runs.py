"""Compact audit of explicitly supplied, completed visible-baseline runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.visible_baseline import sha256


def summarize(directory):
    launch = json.loads((directory / "launch.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    regression = json.loads((directory / "regression.json").read_text())
    if not report["completed"] or not report["full_clip"]:
        raise ValueError(f"Not a complete full-clip run: {directory}")
    if report["frames"] != launch["expected_frames"]:
        raise ValueError("Frame-count mismatch")
    if len({v["source_sha256"] for v in (launch, report, regression)}) != 1:
        raise ValueError("Source-hash mismatch")
    for name, digest in launch["code_sha256"].items():
        if sha256(directory / "implementation" / name) != digest:
            raise ValueError("Implementation snapshot differs from launch")
    return dict(
        run=str(directory.resolve()),
        source=launch["source"],
        source_sha256=launch["source_sha256"],
        config_sha256=launch["config_sha256"],
        code_sha256=launch["code_sha256"],
        configuration=launch["configuration"],
        frames=report["frames"],
        execution_mode=report.get("execution_mode", "full_media_decode"),
        performance_note=report.get(
            "performance_note",
            "End-to-end exploratory runtime; not a controlled benchmark",
        ),
        source_dimensions_wh=[
            launch["source_probe"]["width"],
            launch["source_probe"]["height"],
        ],
        timestamp_basis=launch["timestamp_basis"],
        annotations_supplied_to_detector=launch["annotations_supplied_to_detector"],
        all_events_pass=regression["all_events_pass"],
        all_confident_anchors_matched_by_one_id=all(
            e["dominant_track_anchor_hits"] == e["required_anchor_count"]
            for e in regression["events"]
        ),
        events=[
            {k: v for k, v in e.items() if k != "anchors"} for e in regression["events"]
        ],
        qualified_track_count=regression["qualified_track_count"],
        unmatched_qualified_track_count=regression["unmatched_qualified_track_count"],
        false_tracks_per_minute=None,
        counts=report["counts"],
        elapsed_seconds=report["elapsed_seconds"],
        processed_fps=report["processed_fps"],
        timings_ms=report["timings_ms"],
        report_sha256=sha256(directory / "report.json"),
        regression_sha256=sha256(directory / "regression.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        schema="seaqr.phase20.development-run-audit.v1",
        scope="Reviewed development clips, not untouched validation",
        limitations=[
            "Sparse anchors do not establish continuous identity or exhaustive recall.",
            "Unmatched tracks are unlabeled proposals, not measured false positives.",
            "CPU translation backend, not PVA hardware validation.",
            "Concurrent local exploratory runs, not controlled runtime benchmarks.",
            "Native 8-bit AVI branch; RAW16 and independent synthetic branch unvalidated here.",
        ],
        runs=[summarize(d) for d in args.runs],
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"runs": len(result["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
