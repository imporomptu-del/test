"""Independently audit full saved journals against observation-output exports.

Only JSON metadata is read. This deliberately imports neither the projection nor
its replay implementation. An audit report is created exclusively, only on pass.
It establishes output semantics, not detector accuracy or physical object class.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def decode(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def integer(value):
    return type(value) is int and value >= 0


def coordinate(value):
    require(isinstance(value, list) and len(value) == 2,
            "Source coordinate must be a JSON two-number list")
    for component in value:
        require(type(component) in (int, float), "Coordinate component must be numeric, not boolean")
        try:
            valid = math.isfinite(component)
        except OverflowError:
            valid = False
        require(valid, "Coordinate component must be finite")
    return list(value)


def same(actual, expected, label):
    """Exact recursive contract equality; booleans cannot equal numeric fields."""
    require(type(actual) is type(expected), label + ": value type differs")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), label + ": keys differ")
        for key in expected:
            same(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label + ": list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f"{label}[{index}]")
    else:
        require(actual == expected, label + ": value differs")


def audit(journal, channels, summary, output):
    paths = {name: Path(value).resolve() for name, value in
             (("journal", journal), ("channels", channels), ("summary", summary))}
    output = Path(output).resolve()
    require(len(set(paths.values())) == 3, "Three distinct audit inputs required")
    require(output not in paths.values() and not output.exists(), "Fresh exclusive audit output required")
    initial_hashes = {name: sha(path) for name, path in paths.items()}
    saved = decode(paths["summary"].read_bytes())
    require(isinstance(saved, dict), "Replay summary must be an object")
    same(saved.get("schema"), "seaqr.visible-output-replay.v1", "summary.schema")
    same(saved.get("completed"), True, "summary.completed")
    require(integer(saved.get("frames")) and saved["frames"] > 0, "Positive full frame count required")
    stream = saved.get("stream_id")
    require(isinstance(stream, str) and bool(stream.strip()), "Explicit stream identity required")
    same(saved.get("input_journal"), str(paths["journal"]), "summary.input_journal")
    same(saved.get("input_sha256"), initial_hashes["journal"], "summary.input_sha256")
    same(saved.get("channels_sha256"), initial_hashes["channels"], "summary.channels_sha256")
    for key, value in (
            ("detector_or_tracker_rerun", False), ("qualification_changed", False),
            ("all_qualified_measured_observations_preserved", True),
            ("all_qualified_prediction_context_preserved", True),
            ("predictions_in_observation_alerts", 0)):
        same(saved.get(key), value, "summary." + key)

    counts = {key: 0 for key in (
        "track_context_states", "track_context_frames", "observation_alerts_states",
        "observation_alerts_frames", "retained_prediction_states",
        "alerts_without_current_observation", "prediction_age_unknown_states")}
    observed_ids, context_ids = set(), set()
    # Independent provenance ledger. Every actual observation, even before
    # qualification, refreshes its key. Keys absent from a row are discarded.
    provenance = {}
    prior_timestamp = prior_segment = None
    journal_digest, channel_digest = hashlib.sha256(), hashlib.sha256()
    frame_count = 0
    track_count = actual_count = 0
    with paths["journal"].open("rb") as source, paths["channels"].open("rb") as export:
        for raw in source:
            journal_digest.update(raw)
            row = decode(raw)
            require(isinstance(row, dict), "Journal row must be an object")
            frame, timestamp, segment = (row.get(k) for k in ("frame_index", "timestamp_ns", "segment"))
            require(integer(frame) and frame == frame_count and frame < saved["frames"],
                    "Full journal frame inventory must start at zero and be contiguous")
            require(integer(timestamp) and (prior_timestamp is None or timestamp > prior_timestamp),
                    "Strictly increasing nonnegative timestamp required")
            require(integer(segment) and (prior_segment is None or segment >= prior_segment),
                    "Nondecreasing nonnegative segment required")
            tracks = row.get("tracks")
            require(isinstance(tracks, list), "Journal tracks must be a list")
            if prior_segment != segment:
                provenance.clear()
            row_keys = set()
            expected_context, expected_alerts = [], []
            for track in tracks:
                require(isinstance(track, dict), "Journal track must be an object")
                tid = track.get("track_id")
                require(isinstance(tid, str) and bool(tid.strip()) and "/" not in tid,
                        "Unambiguous nonempty track identity required")
                require(integer(track.get("segment")) and track["segment"] == segment,
                        "Track segment must match frame segment")
                key = (segment, tid)
                require(key not in row_keys, "Duplicate track identity in journal frame")
                row_keys.add(key)
                measured, qualified = (track.get(k) for k in ("measured", "qualified_moving"))
                require(type(measured) is bool and type(qualified) is bool,
                        "Explicit measured/qualified booleans required")
                filtered = coordinate(track.get("source_xy"))
                require("measurement_source_xy" in track, "Explicit current measurement field required")
                actual = coordinate(track["measurement_source_xy"]) if measured else None
                if not measured:
                    require(track["measurement_source_xy"] is None,
                            "Predicted record cannot contain a current measurement")
                else:
                    provenance[key] = {"frame": frame, "timestamp": timestamp}
                    actual_count += 1
                track_count += 1
                if not qualified:
                    continue
                origin = provenance.get(key)
                item = {
                    "identity": f"{stream}/{segment}/{tid}", "track_id": tid,
                    "segment": segment, "measured": measured, "qualified_moving": True,
                    "source_xy": actual if measured else filtered,
                    "coordinate_kind": "measurement" if measured else "prediction",
                    "last_measurement_timestamp_ns": None if origin is None else origin["timestamp"],
                    "last_measurement_age_ns": None if origin is None else timestamp-origin["timestamp"],
                    "last_measurement_age_frames": None if origin is None else frame-origin["frame"],
                    "physical_class": "unknown", "airborne_confirmed": False,
                }
                expected_context.append(item)
                context_ids.add(item["identity"])
                if measured:
                    expected_alerts.append({**item, "kind": "qualified_motion_observation"})
                    observed_ids.add(item["identity"])
                else:
                    counts["retained_prediction_states"] += 1
                    counts["prediction_age_unknown_states"] += origin is None
            provenance = {key: origin for key, origin in provenance.items() if key in row_keys}
            expected = {
                "schema": "seaqr.visible-observation-output.v1", "stream_id": stream,
                "frame_index": frame, "timestamp_ns": timestamp, "segment": segment,
                "observation_alerts": expected_alerts, "track_context": expected_context,
            }
            exported_raw = export.readline()
            require(bool(exported_raw), "Export ended before complete journal")
            channel_digest.update(exported_raw)
            same(decode(exported_raw), expected, f"channels.frame{frame}")
            for channel, items in (("track_context", expected_context), ("observation_alerts", expected_alerts)):
                counts[channel + "_states"] += len(items)
                counts[channel + "_frames"] += bool(items)
            frame_count += 1
            prior_timestamp, prior_segment = timestamp, segment
        require(export.read(1) == b"", "Export has extra frames or bytes")

    require(frame_count == saved["frames"], "Journal ended before declared full frame count")
    same(journal_digest.hexdigest(), initial_hashes["journal"], "journal streaming hash")
    same(channel_digest.hexdigest(), initial_hashes["channels"], "channels streaming hash")
    same(saved.get("counts"), counts, "summary.counts")
    same(saved.get("unique_identity_counts"), {
        "observation_alerts": len(observed_ids), "track_context": len(context_ids)},
        "summary.unique_identity_counts")
    final_hashes = {name: sha(path) for name, path in paths.items()}
    same(final_hashes, initial_hashes, "audit inputs unchanged")
    result = {
        "schema": "seaqr.visible-output-independent-audit.v1", "passed": True,
        "stream_id": stream, "frames": frame_count, "journal_track_states_checked": track_count,
        "journal_actual_observations_checked": actual_count, "counts": counts,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256_before": initial_hashes, "input_sha256_after": final_hashes,
        "audit_code_sha256": sha(__file__),
        "exact_qualified_measured_alert_subset": True, "exact_qualified_context_subset": True,
        "exact_coordinates_and_identity": True, "measurement_ages_independently_derived": True,
        "full_journal_inventory_checked": True, "inputs_unchanged": True,
        "projection_or_replay_imported": False, "media_decoded": False,
        "interpretation": "Output-contract audit only; physical class unknown; not accuracy or false-alarm validation",
    }
    with output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("journal", "channels", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.journal, args.channels, args.summary, args.output)
    print(json.dumps({"passed": result["passed"], "frames": result["frames"]}))


if __name__ == "__main__":
    main()
