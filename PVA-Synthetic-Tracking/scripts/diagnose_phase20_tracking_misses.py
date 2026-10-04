"""Post-run, label-boundary-preserving diagnostics for known tracking failures.

Only completed journals are read. This does not tune, replay or relabel a run.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite a diagnostic")
    audit_path = args.root / "full_development_audit.json"
    audit = json.loads(audit_path.read_text())
    observations = []
    for run in audit["runs"]:
        cid = run["clip_id"]
        if cid not in {"0126", "0029"}:
            continue
        journal = args.root / "pipeline" / ("full_" + cid) / "frames.jsonl"
        if sha256(journal) != run["scored"]["artifacts_sha256"]["frames.jsonl"]:
            raise ValueError("Scored journal changed")
        wanted = {}
        for label_set, scores in (("dense", run["scored"]["evaluation"]), ("pilot", run["pilot"])):
            for window in scores["positive_windows"]:
                for evidence in window["evidence"]:
                    if not evidence["qualified_measured_hit"] or evidence["association_ambiguous"]:
                        wanted.setdefault(evidence["frame_index"], []).append(
                            dict(label_set=label_set, window_id=window["window_id"], evidence=evidence))
        with journal.open() as handle:
            for row in map(json.loads, handle):
                for item in wanted.get(row["frame_index"], []):
                    evidence = item["evidence"]
                    xy = evidence["reference_xy"]
                    radius = evidence["match_radius_px"]
                    candidates, tracks = [], []
                    for candidate in row["candidates"]:
                        distance = math.dist(candidate["source_xy"], xy)
                        if distance <= 2 * radius:
                            candidates.append(dict(distance_px=distance,
                                inside_scoring_gate=distance <= radius,
                                **{k: v for k, v in candidate.items() if k != "shape"}))
                    for track in row["tracks"]:
                        point = track["measurement_source_xy"] if track["measured"] else track["source_xy"]
                        distance = math.dist(point, xy)
                        if distance <= 2 * radius:
                            tracks.append(dict(distance_px=distance,
                                inside_scoring_gate=distance <= radius,
                                **{k: v for k, v in track.items() if k != "learning_shape_reference_xy"}))
                    observations.append(dict(clip_id=cid, frame_index=row["frame_index"],
                        **item, nearby_candidates=candidates, nearby_tracks=tracks))
    result = dict(schema="seaqr.postrun-tracking-failure-diagnostic.v1",
        audit_sha256=sha256(audit_path), observer_sha256=sha256(__file__),
        labels_used_during_processing=False, detector_or_tracking_policy_changed=False,
        predictions_count_as_measurements=False, media_read=False,
        observation_radius="twice the frozen scoring radius for diagnosis only; scoring is unchanged",
        observations=observations)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(observations=len(observations),
        frames=[(v["clip_id"], v["frame_index"], v["label_set"]) for v in observations]), indent=2))


if __name__ == "__main__":
    main()
