"""Replay tracking against a completed frozen full-clip candidate journal.

No media decode or detection rerun; all parent candidate/coverage data survives.
Only explicitly enumerated tracking settings may differ from the parent run.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny_target.visible_baseline import VisibleConfig, VisibleTracks, sha256

TRACKING_FIELDS = {
    "max_active_tracks_per_polarity",
    "position_sigma_px",
    "initial_velocity_sigma_px_s",
    "acceleration_sigma_px_s2",
    "position_gate_px",
    "mahalanobis_gate_squared",
    "confirmation_hits",
    "coast_seconds",
    "minimum_moving_excursion_px",
    "tracking_peak_nms_radius_px",
    "motion_quality_enabled",
    "motion_quality_window_hits",
    "motion_quality_minimum_hits",
    "motion_quality_maximum_rmse_px",
    "tracking_association_cost",
    "tracking_association_cascade",
    "tracking_association_assignment",
    "tracking_association_prior",
    "tracking_association_appearance",
    "tracking_birth_policy",
    "tracking_birth_cell_size_px",
}


def validate_configuration(parent, proposed):
    old, new = asdict(VisibleConfig(**parent)), asdict(proposed)
    changed = {k for k in new if new[k] != old[k]}
    forbidden = changed - TRACKING_FIELDS
    if forbidden:
        raise ValueError(
            f"Tracking replay cannot change detection/motion settings: {sorted(forbidden)}"
        )
    if changed and VisibleConfig(**parent).learning_protection_enabled:
        raise ValueError("Tracking-only replay cannot change a closed-loop learning run; decode and rerun detection")
    return sorted(changed)


def validate_parent(launch, report, allow_prefix=False):
    if not report["completed"]:
        raise ValueError("Parent must be completed")
    if report["full_clip"]:
        if report["frames"] != launch["expected_frames"]:
            raise ValueError("Full-clip parent frame count mismatch")
    elif not (
        allow_prefix
        and launch.get("max_frames") == report["frames"]
        and 0 < report["frames"] < launch["expected_frames"]
    ):
        raise ValueError(
            "Parent must be full-clip, or explicitly allowed completed prefix"
        )


def run(parent, config_path, output, allow_prefix=False):
    cfg = VisibleConfig(**json.loads(config_path.read_text()))
    launch = json.loads((parent / "launch.json").read_text())
    report = json.loads((parent / "report.json").read_text())
    validate_parent(launch, report, allow_prefix)
    if report["source_sha256"] != launch["source_sha256"]:
        raise ValueError("Parent source hashes differ")
    changed = validate_configuration(launch["configuration"], cfg)
    output.mkdir(parents=True, exist_ok=False)
    files = ("visible_baseline.py", "tracking/kalman.py", "visible_quality.py")
    code = {p: sha256(ROOT / "tiny_target" / p) for p in files}
    package = {str(p.relative_to(ROOT / "tiny_target")): sha256(p)
               for p in sorted((ROOT / "tiny_target").rglob("*.py"))}
    new_launch = {
        **launch,
        "configuration": asdict(cfg),
        "config_sha256": sha256(config_path),
        "code_sha256": code,
        "parent_detection_package_sha256": launch.get("package_sha256"),
        "package_sha256": package,
        "complete_replay_runtime_snapshot": True,
        "execution_mode": (
            "tracking_replay_of_frozen_full_clip_detections"
            if report["full_clip"]
            else "tracking_replay_of_frozen_prefix_detections"
        ),
        "parent_run": str(parent.resolve()),
        "parent_journal_sha256": sha256(parent / "frames.jsonl"),
        "parent_report_sha256": sha256(parent / "report.json"),
        "replay_script_sha256": sha256(Path(__file__)),
        "changed_tracking_fields": changed,
        "source_media_decoded_in_this_run": False,
    }
    (output / "launch.json").write_text(json.dumps(new_launch, indent=2))
    for p in package:
        target = output / "implementation" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "tiny_target" / p, target)
    tracker = VisibleTracks(cfg, launch["fps"])
    totals = defaultdict(float)
    elapsed_tracking = []
    count = 0
    start = time.perf_counter()
    try:
        with (parent / "frames.jsonl").open() as src, (output / "frames.jsonl").open(
            "x"
        ) as out:
            for line in src:
                row = json.loads(line)
                if row["frame_index"] != count:
                    raise ValueError("Parent journal non-contiguous or duplicated")
                before = time.perf_counter()
                tracks, metrics = tracker.update(
                    row["candidates"],
                    row["frame_index"],
                    row["timestamp_ns"],
                    row["segment"],
                    np.array(row["source_to_reference"]),
                    row["coverage"]["full_shape_hw"],
                )
                ms = 1000 * (time.perf_counter() - before)
                elapsed_tracking.append(ms)
                row["tracks"], row["tracking_metrics"] = tracks, metrics
                row["timings_ms"] = {"tracking": ms}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                for k in ("dropped_at_tile_cap", "dropped_at_frame_cap"):
                    totals[k] += row["coverage"][k]
                totals["candidate_count"] += len(row["candidates"])
                totals["dropped_track_births"] += sum(
                    m.get("dropped_birth_count_at_active_track_cap", 0)
                    for m in metrics.values()
                )
                totals["dropped_at_tracking_resolution_nms"] += metrics[
                    "resolution_nms"
                ]["dropped_candidate_count"]
                totals["motion_resets"] += row["motion"]["reset"]
                totals["warmup_frames"] += row["coverage"]["warmup"]
                count += 1
                if count % 100 == 0:
                    print(
                        json.dumps(
                            dict(
                                replayed_frames=count,
                                qualified=len(tracker.ever_qualified),
                            )
                        ),
                        flush=True,
                    )
            if count != report["frames"]:
                raise ValueError("Parent journal incomplete")
        duration = time.perf_counter() - start
        result = dict(
            schema="seaqr.visible-tracking-replay.v1",
            completed=True,
            full_clip=report["full_clip"],
            execution_mode=new_launch["execution_mode"],
            source_media_decoded_in_this_run=False,
            frames=count,
            source_sha256=launch["source_sha256"],
            configuration=asdict(cfg),
            counts=dict(totals),
            qualified_tracks=list(tracker.summary.values()),
            qualified_track_count=len(tracker.ever_qualified),
            elapsed_seconds=duration,
            processed_fps=count / duration,
            timings_ms={
                "tracking": dict(
                    mean=float(np.mean(elapsed_tracking)),
                    median=float(np.median(elapsed_tracking)),
                    p95=float(np.percentile(elapsed_tracking, 95)),
                )
            },
            performance_note="Tracking replay only; NOT end-to-end throughput.",
            interpretation="Unlabeled automatic proposals, not verified objects/false positives.",
            faint_target_synthetic_branch_enabled=False,
        )
        (output / "report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(frames=count, qualified=result["qualified_track_count"])))
        return result
    except BaseException as exc:
        (output / "failure.json").write_text(
            json.dumps(dict(completed=False, frames=count, error=repr(exc)), indent=2)
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-prefix",
        action="store_true",
        help="Replay a completed bounded prefix; never mark it full-clip",
    )
    a = p.parse_args()
    run(a.parent, a.config, a.output, a.allow_prefix)
