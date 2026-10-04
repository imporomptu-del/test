#!/usr/bin/env python3
"""Aggregate completed Phase 19 reports without promoting them to truth labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


REPORT_PATTERN = re.compile(r"chunk_(\d{4})_dense_screen\.json$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--max-followups", type=int, default=64)
    parser.add_argument("--max-followups-per-clip", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_followups <= 0 or args.max_followups_per_clip <= 0:
        raise ValueError("follow-up limits must be positive")
    split = json.loads(args.split.read_text())
    discovery = set(str(item) for item in split["discovery"])
    holdout = set(str(item) for item in split["untouched_holdout"])
    if discovery & holdout or len(discovery) != 80 or len(holdout) != 20:
        raise ValueError("invalid frozen Phase 18 split")
    status_path = args.batch_dir / "phase19_dense_discovery_status.json"
    status = json.loads(status_path.read_text())
    if not args.allow_partial and status.get("state") != "complete":
        raise RuntimeError("Phase 19 batch is not complete; use --allow-partial to inspect")

    clips = []
    all_tracks = []
    report_ids = set()
    for path in sorted((args.batch_dir / "reports").glob("*.json")):
        match = REPORT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        clip_id = match.group(1)
        if clip_id in holdout:
            raise RuntimeError(f"sealed holdout report exists: {clip_id}")
        if clip_id not in discovery:
            raise RuntimeError(f"report is outside the discovery split: {clip_id}")
        report = json.loads(path.read_text())
        if report.get("schema_version") != "seaqr.tiny-target.dense-screen.v1":
            raise ValueError(f"wrong dense-screen schema in {path}")
        report_ids.add(clip_id)
        screening = report["screening"]
        synthetic = screening["synthetic_tracking"]
        motion = report["source"]["pva_stabilization"]["metrics"]
        clip = {
            "clip_id": clip_id,
            "report_path": str(path.resolve()),
            "frames_seen": int(screening["frames_seen"]),
            "frames_screened_after_background_warmup": int(
                screening["frames_screened_after_background_warmup"]
            ),
            "accepted_global_transforms": int(
                motion["accepted_global_transforms"]
            ),
            "rejected_global_transforms": int(
                motion["rejected_global_transforms"]
            ),
            "synthetic_window_count": int(synthetic["window_count"]),
            "intermediate_candidate_count": int(
                synthetic["candidate_count_before_clip_pool"]
            ),
            "qualified_track_pool_count": int(
                synthetic["qualified_track_pool_count"]
            ),
            "shortlist_count": int(synthetic["shortlist_count"]),
            "processed_frames_per_second": report["performance"][
                "processed_frames_per_second"
            ],
        }
        clips.append(clip)
        for track in synthetic["track_pool"]:
            all_tracks.append(
                {
                    "clip_id": clip_id,
                    "report_path": str(path.resolve()),
                    **track,
                }
            )

    missing = sorted(discovery - report_ids)
    if missing and not args.allow_partial:
        raise RuntimeError(f"Phase 19 is missing {len(missing)} discovery reports")
    ranked = sorted(
        all_tracks,
        key=lambda item: (
            -int(item["hit_count"]),
            -int(item["independent_nonoverlapping_hit_count"]),
            float(item["fit_rmse_px"]),
            -float(item["median_selection_score"]),
            item["clip_id"],
            int(item["track_id"]),
        ),
    )
    per_clip: dict[str, int] = {}
    followups = []
    for track in ranked:
        clip_id = track["clip_id"]
        if per_clip.get(clip_id, 0) >= args.max_followups_per_clip:
            continue
        followups.append(track)
        per_clip[clip_id] = per_clip.get(clip_id, 0) + 1
        if len(followups) >= args.max_followups:
            break

    summary = {
        "schema_version": "seaqr.tiny-target.phase19-discovery-summary.v1",
        "batch_state": status.get("state"),
        "discovery_report_count": len(clips),
        "expected_discovery_report_count": len(discovery),
        "missing_discovery_clip_ids": missing,
        "sealed_holdout_clip_count": len(holdout),
        "sealed_holdout_report_count": 0,
        "totals": {
            "frames_seen": sum(item["frames_seen"] for item in clips),
            "frames_screened_after_background_warmup": sum(
                item["frames_screened_after_background_warmup"] for item in clips
            ),
            "synthetic_windows": sum(
                item["synthetic_window_count"] for item in clips
            ),
            "intermediate_candidates": sum(
                item["intermediate_candidate_count"] for item in clips
            ),
            "qualified_track_pool_entries": sum(
                item["qualified_track_pool_count"] for item in clips
            ),
        },
        "clips": clips,
        "followup_policy": {
            "maximum_total": args.max_followups,
            "maximum_per_clip": args.max_followups_per_clip,
            "ranking": (
                "hit_count, independent non-overlapping hits, lower fit RMSE, "
                "median local-CFAR score"
            ),
        },
        "followup_shortlist": followups,
        "interpretation": {
            "real_object_count": None,
            "false_alarm_count": None,
            "warning": (
                "Every track is unlabeled review workload. Counts are not real-object "
                "detections, false alarms, precision, or recall."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip_id",
                "track_id",
                "hit_count",
                "independent_nonoverlapping_hit_count",
                "fit_rmse_px",
                "fitted_vx_px_s",
                "fitted_vy_px_s",
                "median_selection_score",
                "followup_first_frame",
                "followup_last_frame",
                "review_label",
                "review_notes",
            ]
        )
        for track in followups:
            writer.writerow(
                [
                    track["clip_id"],
                    track["track_id"],
                    track["hit_count"],
                    track["independent_nonoverlapping_hit_count"],
                    track["fit_rmse_px"],
                    track["fitted_velocity_xy_px_s"][0],
                    track["fitted_velocity_xy_px_s"][1],
                    track["median_selection_score"],
                    track["followup_frame_range"][0],
                    track["followup_frame_range"][1],
                    "unreviewed",
                    "",
                ]
            )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
