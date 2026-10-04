"""Audit completed runs' capacity accounting; no image or truth inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.visible_baseline import sha256


def audit(path):
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    if not report["completed"] or report["source_sha256"] != launch["source_sha256"]:
        raise ValueError("Incomplete run or source mismatch")
    totals = Counter()
    per_polarity_peak = Counter()
    count = 0
    with (path / "frames.jsonl").open() as f:
        for row in map(json.loads, f):
            if row["frame_index"] != count:
                raise ValueError("Non-contiguous journal")
            count += 1
            for polarity in ("bright", "dark"):
                m = row["tracking_metrics"][polarity]
                if m["active_track_count"] > m["max_active_tracks"]:
                    raise ValueError("Track cap violated")
                if (
                    m["unmatched_candidate_count"]
                    != m["birth_count"] + m["dropped_birth_count_at_active_track_cap"]
                ):
                    raise ValueError("Birth accounting mismatch")
                admission = m.get("birth_admission", {})
                replacements = admission.get("tentative_replacements", [])
                if any(e["previous_state"] != "tentative" for e in replacements):
                    raise ValueError("Non-tentative track replaced at capacity")
                rejected = admission.get("rejected_candidate_indices")
                if (
                    rejected is not None
                    and len(rejected) != m["dropped_birth_count_at_active_track_cap"]
                ):
                    raise ValueError("Rejection provenance mismatch")
                per_polarity_peak[polarity] = max(
                    per_polarity_peak[polarity], m["active_track_count"]
                )
                totals["births"] += m["birth_count"]
                totals["dropped_birth_attempts"] += m[
                    "dropped_birth_count_at_active_track_cap"
                ]
                totals["tentative_replacements"] += len(replacements)
                totals["associations"] += m["associated_candidate_count"]
                totals["close_likelihood_alternative_flags"] += sum(
                    a["competing_alternative_within_likelihood_factor_three"]
                    for a in m.get("association_audit", [])
                )
    if count != report["frames"]:
        raise ValueError("Report/journal count mismatch")
    return dict(
        run=str(path.resolve()),
        frames=count,
        full_clip=report["full_clip"],
        execution_mode=report.get("execution_mode", "source_decode_and_pipeline"),
        source_sha256=launch["source_sha256"],
        journal_sha256=sha256(path / "frames.jsonl"),
        report_sha256=sha256(path / "report.json"),
        counts=dict(totals),
        peak_active_per_polarity=dict(per_polarity_peak),
        capacity_and_accounting_checks_passed=True,
        interpretation="Repeated workload events, not missed-object/false-positive counts; ambiguity flags are diagnostic only.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        schema="seaqr.tracking-capacity-audit.v1", runs=[audit(p) for p in args.runs]
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
