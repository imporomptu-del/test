"""Hardware execution diagnostics; prefix runs never become full-clip passes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiny_target.visible_regression import load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run / "report.json").read_text())
    launch = json.loads((a.run / "launch.json").read_text())
    digest, events = load_reference(a.annotations)
    if (
        not report["completed"]
        or launch["source_sha256"] != digest
        or report["source_sha256"] != digest
    ):
        raise ValueError("Incomplete run or source mismatch")
    if report["configuration"]["motion_backend"] != "pva":
        raise ValueError("Not a PVA run")
    evidence = []
    counts = Counter()
    with (a.run / "frames.jsonl").open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row["frame_index"] != i:
                raise ValueError("Non-contiguous prefix")
            counts["frames"] += 1
            counts["pva_errors"] += row["motion"].get("pva_failure", False)
            counts["accepted_motion_fits"] += row["motion"].get("accepted", False)
            counts["motion_resets"] += row["motion"]["reset"]
            for event in events:
                for anchor in event["anchors"]:
                    if anchor["frame_index"] != i:
                        continue
                    matched = [
                        t["track_id"]
                        for t in row["tracks"]
                        if t["track_id"].startswith(event["polarity"] + ":")
                        and t["measured"]
                        and t["qualified_moving"]
                        and math.dist(t["measurement_source_xy"], anchor["xy"])
                        <= anchor["uncertainty_px"] + 2
                    ]
                    evidence.append(
                        dict(
                            event_id=event["event_id"],
                            frame_index=i,
                            required=anchor["required"],
                            qualified_measured_track_ids=matched,
                        )
                    )
    if counts["frames"] != report["frames"]:
        raise ValueError("Prefix/report count mismatch")
    anchor_results = []
    for event in events:
        required = [a for a in event["anchors"] if a["required"]]
        seen = [
            a for a in evidence if a["event_id"] == event["event_id"] and a["required"]
        ]
        frequencies = Counter(
            t for a in seen for t in set(a["qualified_measured_track_ids"])
        )
        dominant, hits = frequencies.most_common(1)[0] if frequencies else (None, 0)
        anchor_results.append(
            dict(
                event_id=event["event_id"],
                required_anchor_count=len(required),
                required_anchors_in_prefix=len(seen),
                dominant_track_id=dominant,
                dominant_track_anchor_hits=hits,
                all_required_anchors_same_measured_id=(
                    len(required) >= 2 and hits == len(required)
                ),
            )
        )
    result = dict(
        scope=(
            "Tracking replay of inherited PVA detections; no hardware executed"
            if report.get("source_media_decoded_in_this_run") is False
            else "Bounded PVA prefix execution, not full-clip validation or real-time readiness"
        ),
        execution_mode=report.get("execution_mode", "source_decode_and_pipeline"),
        motion_counts_inherited=report.get("source_media_decoded_in_this_run") is False,
        performance_note=report.get("performance_note", "End-to-end prefix throughput"),
        source_sha256=digest,
        counts=dict(counts),
        full_clip=report["full_clip"],
        full_clip_regression_pass=None,
        processed_fps=report["processed_fps"],
        anchor_evidence_in_processed_prefix=evidence,
        sparse_anchor_results=anchor_results,
        required_anchors_outside_prefix=[
            a["frame_index"]
            for e in events
            for a in e["anchors"]
            if a["required"] and a["frame_index"] >= counts["frames"]
        ],
        timings_ms=report["timings_ms"],
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
