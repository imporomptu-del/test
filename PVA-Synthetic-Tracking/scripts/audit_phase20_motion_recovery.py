"""Compare actual availability at the intermediate candidate's rejected frames."""
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads((d / "report.json").read_text()) for d in (a.old, a.new)]
    if (
        not all(r["completed"] and r["full_clip"] for r in reports)
        or reports[0]["source_sha256"] != reports[1]["source_sha256"]
    ):
        raise ValueError("Matched complete full clips required")
    rejected = {}
    with (a.old / "frames.jsonl").open() as f:
        for row in map(json.loads, f):
            if row["motion"]["reset"]:
                rejected[row["frame_index"]] = row["motion"].get(
                    "rejection_reasons", []
                )
    evidence = []
    count = 0
    with (a.new / "frames.jsonl").open() as f:
        for row in map(json.loads, f):
            if row["frame_index"] != count:
                raise ValueError("Noncontiguous new journal")
            count += 1
            if row["frame_index"] in rejected:
                support = (
                    row["motion"]
                    .get("motion_fit", {})
                    .get("metrics", {})
                    .get("sparse_translation_support")
                    or {}
                )
                evidence.append(
                    dict(
                        frame=row["frame_index"],
                        old_reasons=rejected[row["frame_index"]],
                        new_accepted=row["motion"].get("accepted", False),
                        new_reset=row["motion"]["reset"],
                        new_detection_ready=row["coverage"]["detection_ready"],
                        consensus_cells=support.get("consensus_cells"),
                        excluded_cell_ids=support.get("excluded_cell_ids"),
                        new_reasons=row["motion"].get("rejection_reasons", []),
                    )
                )
    if any(r["frames"] != count for r in reports) or len(evidence) != len(rejected):
        raise ValueError("Frame count mismatch")
    result = dict(
        source_sha256=reports[0]["source_sha256"],
        old=str(a.old.resolve()),
        new=str(a.new.resolve()),
        formerly_rejected_frames=len(rejected),
        now_motion_accepted=sum(e["new_accepted"] for e in evidence),
        now_detection_ready=sum(e["new_detection_ready"] for e in evidence),
        evidence=evidence,
        caveat="Actual same-source development runs; availability recovery is not target recall or independent camera-motion ground truth",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))
