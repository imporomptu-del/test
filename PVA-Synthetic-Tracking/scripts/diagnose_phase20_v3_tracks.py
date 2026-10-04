"""Post-run local evidence at frozen-reference misses, never detector inputs."""
import argparse
import json
import math
from pathlib import Path
from score_phase20_accuracy import digest


def inspect(path, wanted):
    findings = []
    for row in map(json.loads, (path / "frames.jsonl").open()):
        f = row["frame_index"]
        if f not in wanted:
            continue
        ref = wanted[f]
        candidates = [
            c for c in row["candidates"] if math.dist(c["source_xy"], ref) <= 7
        ]
        tracks = [
            t
            for t in row["tracks"]
            if math.dist(
                t["measurement_source_xy"] if t["measured"] else t["source_xy"], ref
            )
            <= 20
        ]
        identifiers = {t["track_id"] for t in tracks}
        associations = []
        for polarity in ("bright", "dark"):
            for pair in row["tracking_metrics"][polarity].get("association_audit", []):
                if f"{polarity}:{pair['track_id']}" in identifiers:
                    associations.append(dict(polarity=polarity, **pair))
        findings.append(
            dict(
                frame_index=f,
                reference_xy=ref,
                nearby_candidates=candidates,
                nearby_tracks=tracks,
                selected_association_diagnostics=associations,
                predictions_count_as_measurements=False,
            )
        )
    if len(findings) != len(wanted):
        raise ValueError("Missing diagnostic frame")
    return dict(
        run=str(path.resolve()),
        journal_sha256=digest(path / "frames.jsonl"),
        evidence=findings,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--variance-root", type=Path, required=True)
    p.add_argument("--assignment-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    baseline = json.loads((a.reference_root / "baseline_summary.json").read_text())
    variance = json.loads((a.variance_root / "verified_summary.json").read_text())
    assignment = json.loads((a.assignment_root / "summary.json").read_text())
    if (
        digest(a.reference_root / "annotations.json") != variance["labels_sha256"]
        or assignment["labels_sha256"] != variance["labels_sha256"]
    ):
        raise ValueError("Reference changed")
    results = []
    for old in baseline["runs"][:3]:
        cid, backend = old["clip_id"], old["provenance"]["backend"]
        vr = next(
            r
            for r in variance["runs"]
            if r["clip_id"] == cid and r["backend"] == backend
        )
        ar = next(
            r
            for r in assignment["runs"]
            if r["clip_id"] == cid and r["backend"] == backend
        )
        wanted = {}
        for before, v, g in zip(
            old["positive_windows"],
            vr["evaluation"]["positive_windows"],
            ar["positive_windows"],
        ):
            failed = (
                set(before["missed_visible_frames"])
                | set(v["missed_visible_frames"])
                | set(g["missed_visible_frames"])
            )
            # Include the preceding visible sample for handoff context.
            for e in before["evidence"]:
                if e["frame_index"] in failed or e["frame_index"] + 1 in failed:
                    wanted[e["frame_index"]] = e["reference_xy"]
        paths = [
            Path(old["provenance"]["run"]),
            a.variance_root / (("pva_" if backend == "pva" else "cpu_") + cid),
            Path(ar["run"]),
        ]
        results.append(
            dict(
                clip_id=cid,
                backend=backend,
                variants={
                    name: inspect(path, wanted)
                    for name, path in zip(
                        ("baseline", "variance_only", "global_assignment"), paths
                    )
                },
            )
        )
    with a.output.open("x") as f:
        json.dump(
            dict(
                labels_sha256=variance["labels_sha256"],
                scope="Post-run failure diagnosis using frozen references; not pipeline input or new labels",
                runs=results,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
