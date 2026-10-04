"""Audit source review, preserve uncertainty, and compare tracking experiments."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_regression import score
from score_phase20_accuracy import digest, score_rows


def inspect_run(path, windows, reviewed):
    wanted = {e["frame_index"]: e for w in windows for e in w["evidence"]}
    predicted_only = []
    light_rows = []
    for row in map(json.loads, (path / "frames.jsonl").open()):
        f = row["frame_index"]
        if f in wanted and not wanted[f]["qualified_measured_hit"]:
            e = wanted[f]
            near = [
                t["track_id"]
                for t in row["tracks"]
                if t["qualified_moving"]
                and not t["measured"]
                and math.dist(t["source_xy"], e["reference_xy"]) <= e["match_radius_px"]
            ]
            predicted_only.append(
                dict(
                    frame_index=f, nearby_predictions=near, counts_as_measurement=False
                )
            )
        if reviewed is not None and reviewed["first"] <= f <= reviewed["last"]:
            x, y, w, h = reviewed["fixed_crop"]
            tracks = [
                t
                for t in row["tracks"]
                if t["measured"]
                and t["qualified_moving"]
                and x <= t["measurement_source_xy"][0] < x + w
                and y <= t["measurement_source_xy"][1] < y + h
            ]
            # A manually reviewed persistent bright patch, not an absence ROI.
            landmark = [
                t["track_id"]
                for t in tracks
                if math.dist(t["measurement_source_xy"], [3551, 2617]) <= 9
            ]
            light_rows.append(
                dict(
                    frame_index=f,
                    roi_response_ids=[t["track_id"] for t in tracks],
                    landmark_response_ids=landmark,
                )
            )
    return dict(
        missed_visible_frames_prediction_diagnostic=predicted_only,
        light_field_review=dict(
            frames=len(light_rows),
            roi_seconds=len(light_rows) / 10,
            measured_qualified_response_frames=sum(
                len(e["roi_response_ids"]) for e in light_rows
            ),
            distinct_response_ids=sorted(
                {i for e in light_rows for i in e["roi_response_ids"]}
            ),
            persistent_landmark_response_frames=sum(
                len(e["landmark_response_ids"]) for e in light_rows
            ),
            false_alarm_count=None,
            physical_class_adjudication="unresolved",
            evidence=light_rows,
        )
        if reviewed is not None
        else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment-root", type=Path)
    a = p.parse_args()
    root = a.root
    packet = json.loads((root / "source_review/packet.json").read_text())
    verified = 0
    for ep in packet["episodes"]:
        for item in ep["sheets"]:
            if digest(root / "source_review" / item["path"]) != item["sha256"]:
                raise ValueError("Review image changed")
            verified += 1
        if (
            digest(root / "source_review" / ep["native_archive"])
            != ep["native_archive_sha256"]
        ):
            raise ValueError("Native source archive changed")
    baseline = json.loads((root / "baseline_summary.json").read_text())
    exp_root = a.experiment_root or root / "confirmed_first_experiment"
    exp = json.loads((exp_root / "summary.json").read_text())
    if digest(root / "annotations.json") != exp["labels_sha256"]:
        raise ValueError("Frozen labels changed")
    pilot = ROOT / "results/tiny_target/phase20/accuracy_baseline_v1_20260914"
    old_labels = json.loads((pilot / "annotations.json").read_text())
    old_packet = json.loads((pilot / "source_review/packet.json").read_text())
    dev = json.loads(
        (ROOT / "configs/evaluation/phase20_visible_development.json").read_text()
    )
    rows = []
    for old, new in zip(baseline["runs"], exp["runs"]):
        cid = old["clip_id"]
        if cid != new["clip_id"] or old["provenance"]["backend"] != new["backend"]:
            raise ValueError("Baseline/experiment run mapping mismatch")
        reviewed = next(
            (ep for ep in packet["episodes"] if ep["id"] == cid + "_lights"), None
        )
        path = Path(new["run"])
        if digest(path / "frames.jsonl") != new["journal_sha256"]:
            raise ValueError("Experiment journal changed")
        launch = json.loads((path / "launch.json").read_text())
        for name, sha in launch["code_sha256"].items():
            if digest(path / "implementation" / name) != sha:
                raise ValueError("Replay implementation changed")
        if (
            digest(Path(launch["parent_run"]) / "frames.jsonl")
            != launch["parent_journal_sha256"]
        ):
            raise ValueError("Parent journal changed")
        with (path / "frames.jsonl").open() as f:
            pilot_score = score_rows(
                map(json.loads, f), old_labels, old_packet, cid, 10
            )
        reference = next(
            (c for c in dev["clips"] if c["clip_id"] == "chunk" + cid), None
        )
        legacy = (
            score(path, ROOT / reference["annotations"])
            if reference and json.loads((path / "report.json").read_text())["full_clip"]
            else None
        )
        comparisons = []
        for ow, nw in zip(old["positive_windows"], new["positive_windows"]):
            comparisons.append(
                dict(
                    window_id=ow["window_id"],
                    visible_samples=ow["visible_samples"],
                    baseline_hits=ow["qualified_measured_hits"],
                    experiment_hits=nw["qualified_measured_hits"],
                    baseline_ambiguity_frames=ow["ambiguity_frames"],
                    experiment_ambiguity_frames=nw["ambiguity_frames"],
                    baseline_ids=ow["observed_track_ids"],
                    experiment_ids=nw["observed_track_ids"],
                    baseline_misses=ow["missed_visible_frames"],
                    experiment_misses=nw["missed_visible_frames"],
                    newly_missed_frames=sorted(
                        set(nw["missed_visible_frames"])
                        - set(ow["missed_visible_frames"])
                    ),
                )
            )
        rows.append(
            dict(
                clip_id=cid,
                backend=new["backend"],
                comparison=comparisons,
                baseline=inspect_run(
                    Path(old["provenance"]["run"]), old["positive_windows"], reviewed
                ),
                experiment=inspect_run(path, new["positive_windows"], reviewed),
                pilot=pilot_score,
                original_anchor_regression=legacy,
            )
        )
    result = dict(
        source_sheets_integrity_verified=verified,
        unique_reviewed_frames=790,
        confident_visible_reference_frames=285,
        unknown_visibility_frames=105,
        background_review_frames=400,
        verified_airborne_encounters=0,
        verified_negative_intervals=0,
        runs=rows,
        experimental_policy_default="off",
        generalization_proven=False,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [{k: r[k] for k in ("clip_id", "backend", "comparison")} for r in rows],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
