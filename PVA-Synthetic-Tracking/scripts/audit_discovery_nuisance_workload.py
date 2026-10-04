"""Read-only descriptive workload audit of two hash-pinned 8-bit journals.

This does not decode pixels, replay/refit a detector, classify unreviewed motion,
or select thresholds. Spatial quarters describe records; they are not masks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930/evidence"
CLIPS = ("0170", "0240")
JOURNAL_SHA = {
    "0170": "56687056477c79f5cb9aa8c338c260438966c706fcd393ff29afa5536029eca5",
    "0240": "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92",
}
SOURCE_SHA = {
    "0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
    "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585",
}
WINDOWS = {"whole_clip": (0, 672), "preceding_9_49": (9, 49),
           "burst_50_105": (50, 105), "following_106_161": (106, 161),
           "positive_window_430_464": (430, 464)}
HEIGHT, WIDTH = 3190, 4784


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    out = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            out.update(block)
    return out.hexdigest()


def decode(text):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_journal(path, expected_sha):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink() and path.suffix == ".jsonl", "regular absolute journal path required")
    require(digest(path) == expected_sha, "journal hash differs")
    with path.open() as stream:
        rows = [decode(line) for line in stream]
    require(digest(path) == expected_sha, "journal changed while being read")
    require(len(rows) == 673, "expected complete 673-frame journal")
    for index, row in enumerate(rows):
        require(type(row["frame_index"]) is int and row["frame_index"] == index
                and row["timestamp_ns"] == index * 100_000_000, "frame/timestamp continuity differs")
        require(row["coverage"]["full_shape_hw"] == [HEIGHT, WIDTH], "native image dimensions differ")
        require(type(row["coverage"]["detection_ready"]) is bool, "readiness must be boolean")
        require(isinstance(row["candidates"], list) and isinstance(row["tracks"], list), "record collections differ")
        keys = [(t["segment"], t["track_id"]) for t in row["tracks"]]
        require(len(keys) == len(set(keys)), "duplicate track identity in frame")
    return rows


def median(values):
    return statistics.median(values) if values else None


def y_quarters(points):
    bins = Counter()
    for x, y in points:
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            bins["outside_image"] += 1
        else:
            bins[str(min(3, int(y * 4 / HEIGHT)))] += 1
    return {key: bins[key] for key in ("0", "1", "2", "3", "outside_image")}


def analyze(rows):
    # Lifetime and most-recent-eight ACTUAL measurement histories; no fitting.
    histories = defaultdict(lambda: deque(maxlen=8))
    identities = {}
    measured_span = {}
    per_frame = []
    for row in rows:
        index = row["frame_index"]
        for track in row["tracks"]:
            key = (track["segment"], track["track_id"])
            record = identities.setdefault(key, {"first_frame": index, "last_frame": index,
                                                "max_hits": 0, "ever_qualified": False})
            record["last_frame"] = index
            record["max_hits"] = max(record["max_hits"], track["hits"])
            record["ever_qualified"] |= track["qualified_moving"]
            if track["measured"]:
                histories[key].append(index)
                measured_span[(index, key)] = histories[key][-1] - histories[key][0]
        ready = row["coverage"]["detection_ready"]
        metrics = [row["tracking_metrics"][p] for p in ("bright", "dark")] if ready else []
        per_frame.append({
            "frame_index": index, "detection_ready": ready,
            "unavailable_reason": row["coverage"].get("unavailable_reason"),
            "candidate_records": len(row["candidates"]),
            "qualified_measured_records": sum(t["qualified_moving"] and t["measured"] for t in row["tracks"]),
            "qualified_predicted_records": sum(t["qualified_moving"] and not t["measured"] for t in row["tracks"]),
            **{name: sum(m[key] for m in metrics) for name, key in (
                ("active_tracks", "active_track_count"), ("births", "birth_count"),
                ("deleted_tracks", "deleted_track_count"), ("dropped_births", "dropped_birth_count_at_active_track_cap"))},
            "tentative_tracks": sum(m["lifecycle_counts"]["tentative"] for m in metrics),
            "coasted_tracks": sum(m["lifecycle_counts"]["coasted"] for m in metrics),
        })
    windows = {}
    for name, (start, end) in WINDOWS.items():
        selected = [row for row in rows if start <= row["frame_index"] <= end]
        ready_rows = [row for row in selected if row["coverage"]["detection_ready"]]
        candidates = [c for row in ready_rows for c in row["candidates"]]
        tracks = [(row["frame_index"], t) for row in ready_rows for t in row["tracks"]]
        qualified = [(f, t) for f, t in tracks if t["qualified_moving"]]
        measured = [(f, t) for f, t in qualified if t["measured"]]
        metrics = [row["tracking_metrics"][p] for row in ready_rows for p in ("bright", "dark")]
        series = [r for r in per_frame if start <= r["frame_index"] <= end]
        births = [v for v in identities.values() if start <= v["first_frame"] <= end]
        count = len(ready_rows)
        active_total = sum(m["active_track_count"] for m in metrics)
        tentative_total = sum(m["lifecycle_counts"]["tentative"] for m in metrics)
        windows[name] = {
            "frame_range_inclusive": [start, end], "frames": len(selected), "ready_frames": count,
            "unavailable_frames": len(selected)-count,
            "unavailable_reasons": dict(Counter(r["coverage"].get("unavailable_reason") for r in selected
                                                  if not r["coverage"]["detection_ready"])),
            "candidate_records": len(candidates),
            "candidate_records_per_ready_frame": len(candidates)/count if count else None,
            "candidate_polarity_counts": dict(Counter(c["polarity"] for c in candidates)),
            "candidate_y_quarters": y_quarters(c["source_xy"] for c in candidates),
            "candidate_score_median": median([c["score"] for c in candidates]),
            "candidate_noise_floor_fraction": sum(c["noise_sigma_dn"] == .5 for c in candidates)/len(candidates) if candidates else None,
            "candidate_without_bounded_shape_records": sum(not c.get("shape") for c in candidates),
            "qualified_measured_records": len(measured),
            "qualified_predicted_records": len(qualified)-len(measured),
            "qualified_identities": len({(t["segment"], t["track_id"]) for _, t in qualified}),
            "qualified_measured_y_quarters": y_quarters(t["measurement_source_xy"] for _, t in measured),
            "qualified_measured_lifetime_hits_median": median([t["hits"] for _, t in measured]),
            "qualified_measured_last_eight_measurements_span_frames_median": median([
                measured_span[(f, (t["segment"], t["track_id"]))] for f, t in measured]),
            "qualified_measured_last_eight_measurements_span_frames_max": max([
                measured_span[(f, (t["segment"], t["track_id"]))] for f, t in measured], default=None),
            "active_tracks_mean_per_ready_frame": active_total/count if count else None,
            "tentative_tracks_mean_per_ready_frame": tentative_total/count if count else None,
            "tentative_fraction_of_active_records": tentative_total/active_total if active_total else None,
            "coasted_tracks_mean_per_ready_frame": sum(m["lifecycle_counts"]["coasted"] for m in metrics)/count if count else None,
            "per_polarity_cap_saturated_frame_polarity_count": sum(m["active_track_count"] == m["max_active_tracks"] for m in metrics),
            "max_active_tracks_both_polarities": max((s["active_tracks"] for s in series), default=0),
            **{name: sum(m[key] for m in metrics) for name, key in (
                ("births", "birth_count"), ("deleted_tracks", "deleted_track_count"),
                ("dropped_births", "dropped_birth_count_at_active_track_cap"),
                ("associated_candidates", "associated_candidate_count"))},
            "spatial_fair_tentative_replacements": sum(len(m["birth_admission"].get("tentative_replacements", [])) for m in metrics),
            "tile_cap_drops": sum(r["coverage"]["dropped_at_tile_cap"] for r in ready_rows),
            "frame_cap_drops": sum(r["coverage"]["dropped_at_frame_cap"] for r in ready_rows),
            "first_seen_identity_cohort": {
                "count": len(births),
                "eventual_max_hits_histogram_observed_until_clip_end": dict(sorted(Counter(str(v["max_hits"]) for v in births).items(), key=lambda item: int(item[0]))),
                "ever_qualified_in_observed_clip": sum(v["ever_qualified"] for v in births),
                "observed_at_clip_final_frame_right_censored": sum(v["last_frame"] == 672 for v in births),
                "warning": "First-seen identities include coordinate resets. Future-to-window lifecycle description only; not an online rule or labeled objects.",
            },
        }
    return {"windows": windows, "per_frame": per_frame}


def build(journals):
    results, inputs = {}, {}
    for clip in CLIPS:
        rows = load_journal(journals[clip], JOURNAL_SHA[clip])
        inputs[clip] = {"journal": str(journals[clip]), "journal_sha256": JOURNAL_SHA[clip],
                        "source_media_sha256_from_frozen_provenance_not_reopened": SOURCE_SHA[clip]}
        results[clip] = analyze(rows)
    return {
        "schema": "seaqr.discovery-nuisance-workload-audit.v2",
        "script_sha256": digest(Path(__file__).resolve()), "inputs": inputs,
        "population": "All673 causal output frames perclip; aggregates of candidates and tracks use detection-ready frames only.",
        "coordinate_basis": {
            "candidates": "candidate.source_xy: inverse-warped raw detector measurement in native source coordinates; not reference x/y",
            "qualified_measured": "track.measurement_source_xy: raw measurement in native source coordinates; not filtered track.source_xy",
        },
        "supersedes": {
            "artifact": "nuisance_workload_audit.json",
            "sha256": "79a6a15f9eb0cf09de2d3c6abd19a8c5d6e95c53877cdd60113c5d696fc58287",
            "reason": "Corrected native spatial distributions: v1 used candidate reference coordinates and filtered track positions. Non-spatial counts are unchanged. Original artifact retained.",
        },
        "spatial_quarters": "Native y intervals [0,797.5),[797.5,1595),[1595,2392.5),[2392.5,3190); descriptive only, no image mask.",
        "counts_are_objects_or_false_positives": False,
        "interpretation_limits": [
            "Candidates, identities and M/P states are unlabeled workload, not object counts or precision.",
            "No pixels opened: scene-wide photometric change, edge origin, physical class and genuine motion cannot be established.",
            "Unbounded/missing shape is not a nuisance label. Strong or persistent tracks are not proven objects.",
            "Journal replay was previously verified against the original candidate for all non-timing fields; this audit only validates the exact pinned journals.",
            "No algorithm changes, refits, threshold sweeps or overlay suppression were performed.",
        ], "clips": results,
    }


def save_new(path, result):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent
            and path.suffix == ".json" and not path.exists(), "new absolute JSON output required")
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-0170", type=Path, default=EVIDENCE/"0170/run/frames.jsonl")
    parser.add_argument("--journal-0240", type=Path, default=EVIDENCE/"0240/run/frames.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing existing output")
    result = build({"0170": args.journal_0170, "0240": args.journal_0240})
    save_new(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
