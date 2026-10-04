"""Post-run sparse-anchor scoring. Never imported by the detection runner."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def load_reference(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") == "manual_visual_reference_v1":
        events = value["events"]
        result = []
        for event in events:
            anchors = []
            for a in event["anchors"]:
                anchors.append(
                    dict(
                        frame_index=a["frame_index"],
                        xy=[a["x"], a["y"]],
                        uncertainty_px=a["position_uncertainty_px"],
                        required=a["visibility"] == "visible",
                    )
                )
            result.append(
                dict(event_id=event["event_id"], polarity="bright", anchors=anchors)
            )
    elif (
        value.get("source") == "chunk_0126.avi"
        and "approximate_xy" in value["anchors"][0]
    ):
        result = [
            dict(
                event_id="chunk0126_moving_point",
                polarity="bright",
                anchors=[
                    dict(
                        frame_index=a["frame_index"],
                        xy=a["approximate_xy"],
                        uncertainty_px=a["approximate_position_uncertainty_px"],
                        required=a["visibility"] == "Visually identifiable",
                    )
                    for a in value["anchors"]
                ],
            )
        ]
    else:
        raise ValueError("Unsupported annotation schema; no guessed field mapping")
    return value["source_sha256"], result


def score(run_directory, reference_path):
    directory = Path(run_directory)
    report = json.loads((directory / "report.json").read_text())
    launch = json.loads((directory / "launch.json").read_text())
    digest, events = load_reference(reference_path)
    if report["source_sha256"] != digest or launch["source_sha256"] != digest:
        raise ValueError("Annotation source hash does not match detection source")
    if (
        not report["completed"]
        or not report["full_clip"]
        or report["frames"] != launch["expected_frames"]
    ):
        raise ValueError("Regression requires a completed full-clip run")
    lookup = {}
    for e in events:
        for a in e["anchors"]:
            lookup.setdefault(a["frame_index"], []).append((e, a))
    evidence = {e["event_id"]: [] for e in events}
    qualified_ids = {t["track_id"] for t in report["qualified_tracks"]}
    seen = set()
    with (directory / "frames.jsonl").open() as f:
        for line in f:
            frame = json.loads(line)
            idx = frame["frame_index"]
            if idx in seen:
                raise ValueError("Duplicate journal frame")
            seen.add(idx)
            for e, a in lookup.get(idx, []):
                radius = a["uncertainty_px"] + 2.0
                candidates = [
                    p
                    for p in frame["candidates"]
                    if p["polarity"] == e["polarity"]
                    and math.dist(p["source_xy"], a["xy"]) <= radius
                ]
                matched = []
                predictions = []
                for t in frame["tracks"]:
                    if not t["track_id"].startswith(e["polarity"] + ":"):
                        continue
                    if (
                        t["measured"]
                        and t["qualified_moving"]
                        and math.dist(t["measurement_source_xy"], a["xy"]) <= radius
                    ):
                        matched.append(t["track_id"])
                    if (
                        not t["measured"]
                        and t["qualified_moving"]
                        and math.dist(t["source_xy"], a["xy"]) <= radius
                    ):
                        predictions.append(t["track_id"])
                evidence[e["event_id"]].append(
                    dict(
                        **a,
                        match_radius_px=radius,
                        candidate_hit=bool(candidates),
                        qualified_measured_track_ids=matched,
                        nearby_prediction_only_ids=predictions,
                        segment=frame["segment"],
                        warmup=frame["coverage"]["warmup"]
                    )
                )
    if seen != set(range(report["frames"])):
        raise ValueError("Journal is incomplete or non-contiguous")
    results = []
    associated_ids = set()
    for event_id, anchors in evidence.items():
        required = [a for a in anchors if a["required"]]
        if len(required) < 2:
            raise ValueError("At least two confident anchors required per event")
        counts = Counter(
            t for a in required for t in set(a["qualified_measured_track_ids"])
        )
        for a in anchors:
            associated_ids.update(a["qualified_measured_track_ids"])
        dominant, count = counts.most_common(1)[0] if counts else (None, 0)
        fraction = count / len(required)
        results.append(
            dict(
                event_id=event_id,
                required_anchor_count=len(required),
                candidate_anchor_hits=sum(a["candidate_hit"] for a in required),
                qualified_measured_anchor_hits=sum(
                    bool(a["qualified_measured_track_ids"]) for a in required
                ),
                dominant_track_id=dominant,
                dominant_track_anchor_hits=count,
                dominant_track_anchor_fraction=fraction,
                sparse_anchor_regression_pass=count >= 2 and fraction >= 0.8,
                anchors=anchors,
            )
        )
    return dict(
        schema="seaqr.visible-sparse-anchor-regression.v1",
        source_sha256=digest,
        reference=str(Path(reference_path).resolve()),
        scope="development only; sparse manual anchors, not exhaustive event detection or identity-continuity ground truth",
        policy=dict(
            minimum_dominant_track_confident_anchor_fraction=0.8,
            minimum_confident_anchor_hits=2,
            extra_localization_tolerance_px=2.0,
            prediction_only_matches_do_not_count=True,
            low_confidence_anchors_are_diagnostic_only=True,
        ),
        events=results,
        all_events_pass=all(e["sparse_anchor_regression_pass"] for e in results),
        qualified_track_count=len(qualified_ids),
        unmatched_qualified_track_count=len(qualified_ids - associated_ids),
        false_tracks_per_minute=None,
        false_alarm_note="Unmatched tracks are unlabeled review workload, NOT measured false positives; no verified-negative manifest supplied.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = score(args.run, args.annotations)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "all_events_pass",
                    "qualified_track_count",
                    "unmatched_qualified_track_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
