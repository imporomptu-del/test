"""Freeze reviewed source-only labels and adapter packet for the existing scorer."""
import argparse
import json
from pathlib import Path

from score_phase20_accuracy import digest, inside, validate_labels


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--existing-runs", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    acceptance = json.loads((root / "review_acceptance.json").read_text())
    prop_path = root / "annotation_proposals/proposals.json"
    proposals = json.loads(prop_path.read_text())
    source_path = root / "source_review/packet.json"
    source = json.loads(source_path.read_text())
    if (
        digest(prop_path) != acceptance["proposal_sha256"]
        or digest(source_path) != proposals["source_packet_sha256"]
        or acceptance["all_proposal_sheets_reviewed"] is not True
    ):
        raise ValueError("Review provenance mismatch")
    proposed = {e["id"]: e for e in proposals["episodes"]}
    labels = dict(
        schema="seaqr.bounded-visual-accuracy-labels.v1",
        policy=dict(extra_localization_tolerance_px=2.0),
        scoring_scope="Known moving-feature regression, physical class unknown; not airborne accuracy",
        positive_windows=[],
        negative_windows=[],
        unresolved_windows=[],
        review_acceptance_sha256=digest(root / "review_acceptance.json"),
    )
    windows = []
    for ep in source["episodes"]:
        rects = [v["crop_xywh"] for v in ep["frames"]]
        x, y = min(r[0] for r in rects), min(r[1] for r in rects)
        # Union is only an adapter for existing scorer coordinate validation.
        # Positive labels are ALSO checked against the actual per-frame crop.
        union = [
            x,
            y,
            max(r[0] + r[2] for r in rects) - x,
            max(r[1] + r[3] for r in rects) - y,
        ]
        windows.append(
            dict(
                id=ep["id"],
                clip_id=ep["clip_id"],
                first=ep["first"],
                last=ep["last"],
                crop_xywh=union,
                bounding_union_not_exhaustively_reviewed=True,
            )
        )
        if ep["id"] in proposed:
            q = proposed[ep["id"]]
            if ep["id"] not in acceptance["accepted_episodes"]:
                raise ValueError("Unaccepted episode")
            origins = {r["frame_index"]: r["crop_xywh"] for r in ep["frames"]}
            for sample in q["samples"]:
                if sample["source_crop_xywh"] != origins[
                    sample["frame_index"]
                ] or not inside(sample["xy"], sample["source_crop_xywh"]):
                    raise ValueError("Coordinate outside reviewed source")
            for sheet in q["sheets"]:
                if (
                    digest(root / "annotation_proposals" / sheet["path"])
                    != sheet["sha256"]
                ):
                    raise ValueError("Reviewed annotation sheet changed")
            labels["positive_windows"].append(
                dict(
                    window_id=ep["id"],
                    event_id=ep["id"],
                    polarity="bright",
                    motion_confirmed=True,
                    airborne_target_verified=False,
                    visible_samples=[
                        dict(
                            frame_index=s["frame_index"],
                            xy=s["xy"],
                            uncertainty_px=s["uncertainty_px"],
                        )
                        for s in q["samples"]
                    ],
                    unknown_frames=q["unknown_frames"],
                )
            )
        else:
            labels["unresolved_windows"].append(
                dict(window_id=ep["id"], **acceptance["background_review"][ep["id"]])
            )
    if (
        sum(len(e["visible_samples"]) for e in labels["positive_windows"])
        != acceptance["accepted_sample_count"]
    ):
        raise ValueError("Accepted count mismatch")
    packet = dict(
        plan=source["plan"], windows=windows, source_packet_sha256=digest(source_path)
    )
    validate_labels(labels, packet)
    write(root / "annotations.json", labels)
    write(root / "scoring_packet.json", packet)
    runs = json.loads(a.existing_runs.read_text())["runs"]
    write(
        root / "scoring_freeze.json",
        dict(
            runs=runs,
            labels_sha256=digest(root / "annotations.json"),
            packet_sha256=digest(root / "scoring_packet.json"),
        ),
    )


if __name__ == "__main__":
    main()
