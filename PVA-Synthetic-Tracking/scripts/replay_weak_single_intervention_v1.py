"""One historical state intervention, not a new weak-selection policy.

After a passed hash-bound full historical replay, apply only the already audited
chunk0126 frame20 bright:211 correction. No later weak observations are supplied.
This tests conditional computational sufficiency, not physical identity/accuracy.
"""
from copy import deepcopy
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

PARENT_HELPER_SHA = "8a8ff3fa357be07999cb9a93c6cf3099be968534f903d49482638b7e5c35be82"
EVENT_FRAME = 20
EVENT_TRACK = "bright:211"
FRAME_COUNT = 41


def helper():
    spec = importlib.util.find_spec("replay_weak_divergence_v1")
    if spec is None or spec.origin is None:
        raise ValueError("Pinned parent replay helper missing")
    import hashlib
    path = Path(spec.origin)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_HELPER_SHA:
        raise ValueError("Parent replay helper pin differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parent(path, expected_sha, receipt_sha, geometry_sha, batch_sha, h):
    path = Path(path)
    h.require(path.is_file() and not path.is_symlink() and h.sha(path) == expected_sha, "Parent replay binding differs")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(268435457)
    h.require(len(raw) <= 268435456, "Parent replay exceeds diagnostic bound")
    result = h.loads(raw)
    h.require(result.get("schema") == "seaqr.weak-divergence-replay.v1" and result.get("passed") is True
        and result.get("error") is None and result.get("completed_frames") == FRAME_COUNT
        and len(result.get("frames", [])) == FRAME_COUNT and result.get("script_sha256") == PARENT_HELPER_SHA
        and result.get("input_receipt_sha256") == receipt_sha
        and result.get("media_accessed") is False and result.get("weak_selection_reevaluated") is False
        and result.get("production_changed") is False, "Passed same-prefix historical replay required")
    adapter = result.get("adapter", {})
    h.require(adapter.get("geometry_library_sha256") == geometry_sha and adapter.get("batch_library_sha256") == batch_sha,
              "Parent native backend differs")
    h.require(h.sha(path) == expected_sha, "Parent replay changed while loading")
    return result


def assignments(records):
    return [dict(track_id=r["track_id"], measurement_source_xy=r["measurement_source_xy"], qualified=r["qualified_moving"])
            for r in records if r["measured"]]


def qualified_coordinates(records):
    return sorted([r["track_id"].split(":")[0], *r["measurement_source_xy"]]
                  for r in records if r["measured"] and r["qualified_moving"])


def changes(records, other):
    current, reference = assignments(records), assignments(other)
    return dict(exact_assignment_changed=[(r["track_id"], r["measurement_source_xy"]) for r in current] !=
                [(r["track_id"], r["measurement_source_xy"]) for r in reference],
        qualified_actual_coordinate_multiset_changed=qualified_coordinates(records) != qualified_coordinates(other),
        actual_measurement_count=len(current), reference_actual_measurement_count=len(reference),
        qualified_actual_measurement_count=len(qualified_coordinates(records)),
        reference_qualified_actual_measurement_count=len(qualified_coordinates(other)))


def intervene_rows(clean, shadow, reference, method, parent_frames, h):
    from tiny_target.visible_baseline import VisibleConfig, VisibleTracks, map_point
    from tiny_target.visible_quality import suppress_nearby
    from tiny_target.tracking.kalman import KalmanTrackManager
    from weak_continuation_shadow_v1 import joseph_weak_update
    h.require(len(clean) == len(shadow) == len(parent_frames) == FRAME_COUNT, "Complete fixed prefix required")
    cfg = VisibleConfig(**reference["configuration"])
    trackers = {name: VisibleTracks(cfg, reference["fps"]) for name in ("baseline", "single")}
    frames, histories, captured = [], {name: {} for name in trackers}, {}
    active = None
    applied = 0

    def update(manager, batch, *, include_quality_evidence=True):
        polarity = next(p for p, m in trackers[active].managers.items() if m is manager)
        result = method(manager, batch, include_quality_evidence=include_quality_evidence)
        captured[polarity] = dict(associations=h.plain(result.associations), born_track_ids=h.plain(result.born_track_ids),
            deleted_tracks=h.plain(result.deleted_tracks), birth_admission=deepcopy(result.metrics["birth_admission"]),
            next_track_id=manager._next_track_id)
        return result

    with patch.object(KalmanTrackManager, "update", update):
        for frame, (b, s, parent) in enumerate(zip(clean, shadow, parent_frames)):
            h.require(b["frame_index"] == s["frame_index"] == parent["frame_index"] == frame
                      and b["timestamp_ns"] == s["timestamp_ns"] == frame*100000000
                      and b["segment"] == s["segment"] == parent["segment"], "Prefix coordinate sequence differs")
            proposals = deepcopy(s["strong_proposals"])
            h.compare([{k: v for k, v in p.items() if k != "source_xy"} for p in b["candidates"]],
                      [{k: v for k, v in p.items() if k != "source_xy"} for p in proposals], f"frame{frame}.input")
            matrix, shape = np.asarray(b["source_to_reference"], np.float64), tuple(b["coverage"]["full_shape_hw"])
            kept, _ = suppress_nearby(proposals, cfg.tracking_peak_nms_radius_px)
            selected = {p: [q for q in kept if q["polarity"] == p] for p in ("bright", "dark")}
            frame_output = dict(frame_index=frame, timestamp_ns=b["timestamp_ns"], arms={}, intervention=None)
            for active in ("baseline", "single"):
                tracker = trackers[active]
                captured = {}
                records, metrics = tracker.update(deepcopy(proposals), frame, b["timestamp_ns"], b["segment"], matrix, shape)
                if active == "single" and frame == EVENT_FRAME:
                    archived = [r for r in s["records"] if r["track_id"] == EVENT_TRACK]
                    h.require(len(archived) == 1 and archived[0]["weak_evidence"]["applied"] is True, "Fixed historical intervention absent")
                    note = archived[0]["weak_evidence"]
                    current = next((r for r in records if r["track_id"] == EVENT_TRACK), None)
                    h.require(current is not None and not current["measured"] and current["qualified_moving"],
                              "Single-intervention owner absent, strongly matched, or unqualified")
                    polarity, raw_id = EVENT_TRACK.split(":")
                    manager = tracker.managers[polarity]
                    track = manager._tracks[int(raw_id)]
                    h.require(note["identity"] == f"{b['segment']}/{EVENT_TRACK}" and
                              track.last_measurement_timestamp_ns == note["strong_anchor_timestamp_ns"], "Fixed strong anchor differs")
                    prior = next((p for p in parent["arms"]["shadow"]["managers"][polarity]["prior"]
                                  if p["track_id"] == int(raw_id)), None)
                    h.require(prior is not None and prior["birth_timestamp_ns"] == track.birth_timestamp_ns
                              and prior["associated_update_count"] == track.associated_update_count
                              and prior["independent_confirmation_hits"] == track.independent_confirmation_hits,
                              "Intervention computational owner history differs")
                    h.compare(track.mean.tolist(), note["mean_before"], "single.weak_mean_before", True)
                    h.compare(track.covariance.tolist(), note["covariance_before"], "single.weak_covariance_before", True)
                    mean, covariance = joseph_weak_update(track.mean, track.covariance, note["measurement_reference_xy"], manager._measurement_covariance())
                    h.compare(mean.tolist(), note["mean_after"], "single.weak_mean_after", True)
                    h.compare(covariance.tolist(), note["covariance_after"], "single.weak_covariance_after", True)
                    track.mean, track.covariance = mean, covariance
                    current.update(reference_xy=mean[:2].tolist(), source_xy=map_point(np.linalg.inv(matrix), *mean[:2]),
                                   velocity_reference_xy_px_s=mean[2:].tolist())
                    applied += 1
                    frame_output["intervention"] = deepcopy(note)
                if active == "baseline":
                    h.compare_records(records, b["tracks"], f"frame{frame}.baseline.records")
                    h.compare(metrics, b["tracking_metrics"], f"frame{frame}.baseline.metrics", True)
                    h.compare_records(records, parent["arms"]["baseline"]["records"], f"frame{frame}.parent_baseline.records")
                elif frame < EVENT_FRAME:
                    h.compare_records(records, b["tracks"], f"frame{frame}.preintervention.records")
                    h.compare(metrics, b["tracking_metrics"], f"frame{frame}.preintervention.metrics", True)
                for polarity, info in captured.items():
                    manager = tracker.managers[polarity]
                    for association in info["associations"]:
                        tid = f"{polarity}:{association['track_id']}"
                        observation = selected[polarity][association["candidate_index"]]
                        histories[active].setdefault(tid, []).append(dict(frame=frame, x=observation["x"], y=observation["y"], kind="strong_association"))
                    for tid in info["born_track_ids"]:
                        observation = selected[polarity][manager._tracks[tid].last_candidate_index]
                        histories[active].setdefault(f"{polarity}:{tid}", []).append(dict(frame=frame, x=observation["x"], y=observation["y"], kind="strong_birth"))
                frame_output["arms"][active] = dict(managers=captured, strong_assignments=assignments(records),
                    qualified_actual_coordinates=qualified_coordinates(records),
                    comparison_to_original_baseline=changes(records, b["tracks"]),
                    comparison_to_all_weak_shadow=changes(records, s["records"]),
                    selected_quality=(h.visible_state(tracker) if frame in (EVENT_FRAME, 21, 32, 34) else None))
            frames.append(frame_output)
    h.require(applied == 1, "Exactly one historical correction required")
    return dict(frames=frames, strong_histories=histories, weak_corrections_applied=applied,
        later_weak_corrections_applied=0,
        first_assignment_difference_from_baseline=next((r["frame_index"] for r in frames if r["arms"]["single"]["comparison_to_original_baseline"]["exact_assignment_changed"]), None),
        first_qualified_coordinate_difference_from_baseline=next((r["frame_index"] for r in frames if r["arms"]["single"]["comparison_to_original_baseline"]["qualified_actual_coordinate_multiset_changed"]), None))


def run(directory, receipt_sha256, parent_replay, parent_sha256, geometry, geometry_sha256, batch_library, batch_sha256, output):
    h = helper()
    output = Path(output)
    h.require(not output.exists() and not output.is_symlink(), "Fresh single-intervention output required")
    report = dict(schema="seaqr.weak-single-intervention.v1", passed=False, error=None,
        script_sha256=h.sha(__file__), parent_replay_sha256=parent_sha256, input_receipt_sha256=receipt_sha256,
        fixed_event=dict(clip="0126", frame=EVENT_FRAME, track_id=EVENT_TRACK), media_accessed=False,
        production_changed=False, weak_selection_reevaluated=False, physical_identity_established=False,
        interpretation="One historical state intervention tests computational sufficiency only; not necessity, deployment policy or accuracy")
    try:
        parent = load_parent(parent_replay, parent_sha256, receipt_sha256, geometry_sha256, batch_sha256, h)
        _, reference, clean, shadow = h.read_inputs(directory, receipt_sha256)
        report["runtime_hashes"] = h.verify_runtime(reference)
        with h.native_backend(geometry, geometry_sha256, batch_library, batch_sha256) as (method, optimized, _):
            h.require(optimized.transformed_sha256 == parent["adapter"]["transformed_sha256"], "Parent transformed tracker differs")
            report["replay"] = intervene_rows(clean, shadow, reference, method, parent["frames"], h)
        h.read_inputs(directory, receipt_sha256)
        h.verify_runtime(reference)
        h.require(h.sha(parent_replay) == parent_sha256, "Parent replay changed")
        report["passed"] = True
    except BaseException as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc), detail=getattr(exc, "detail", None))
        raise
    finally:
        opener = gzip.open if output.suffix == ".gz" else open
        with opener(output, "xt") as stream:
            json.dump(report, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "parent-replay", "geometry", "batch-library", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("receipt-sha256", "parent-sha256", "geometry-sha256", "batch-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    run(args.directory, args.receipt_sha256, args.parent_replay, args.parent_sha256,
        args.geometry, args.geometry_sha256, args.batch_library, args.batch_sha256, args.output)
