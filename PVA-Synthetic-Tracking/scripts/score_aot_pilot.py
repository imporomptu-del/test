#!/usr/bin/env python3
"""Frozen descriptive AOT point-observation scoring; no detector/media imports.

Freeze uses labels and source code only. Score requires the reviewed freeze and
a complete, provenance-checked run. No parameter search or automatic tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "seaqr.aot.point-scoring.v1"
FRAME_COUNT, WIDTH, HEIGHT, INTERVAL_NS = 300, 2448, 2048, 100_000_000
MANIFEST_SHA256 = "425b8220417e9853a8fbf272a1119d4e1989678984c84f295c9aeb2507b72d8d"
VIDEO_SHA256 = "869c37637b68de5eb2c65a6140caebcea58f01833b653a1f2991fec3b16e4d6f"
CONFIG_SHA256 = "7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f"
BASELINE_SHA256 = "d059ef60c9436bbf942a458586d9b546b8f1806db080f8fd20b22b768bf96df2"
MOTION_SHA256 = "fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1"
RUNTIME_IDENTITIES = {
    "config_sha256": CONFIG_SHA256, "motion_config_sha256": MOTION_SHA256,
    "v29_freeze_sha256": "60b79d450672d131b517e9ed5a33fdeb40a6a2a9b29c6c0f584a9a8e93cc0dbc",
    "reuse_method_sha256": "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733",
    "tracking_transformed_sha256": "571641e429f6604123ab98eba974d4e9623abcad2402be147913b7622d860325",
    "vpi_version": "3.2.4",
}
LIBRARIES = {
    "/tmp/seaqr_visible_front_v26_retry_24sEU7/build_03/candidate.so": "fd689b653175eb9baf3e84259ccccd431ba3ec8aa6c6fa4378a596eb03f8a027",
    "/tmp/seaqr_phase20_decode_v10_Iz9RSF/libseaqr_shapes.so": "ec3ef02040df013536a5610d78bcde9763c1bcade0ec8a43e6525a6b3f0bb7ca",
    "/tmp/seaqr_visible_speed_v20_XUf1LR/build/libtracking_geometry_v20.so": "1d81a0369a78ca462fce16e9655e9c81811d4544e071c013682f5aaf58821782",
    "/tmp/seaqr_tracking_v27_pmUXGZ/build_01/libtracking_batch_v27.so": "bdabc75a633da7cb72c0565a3d2b87dcb6662b17b12ba96ccd7a7c586bebb644",
    "/tmp/seaqr_visible_speed_v17_EER6lm/build/liblearning_mask_v17.so": "1eb3e13dc035644162e3dd8056c2ef5cc7ef40de33b25a82efaab491a3dd8c70",
}
ADAPTERS = {
    "motion_reuse_v12": "038e45d83c46909958fcbdf5b94d791f7a69733ef765851c7d0fdf77c19a48f8",
    "raw16_speed_v8_common": "09c89333193888e97bae1d1f90bed4ca3e33ea5a2a895c0bdc245dd4a84b997d",
    "visible_stage_v24": "d19fc73d4528e3783eb0f0b019a2c328d6ce21eee7df837cfdada56cc66c8f2a",
    "visible_overlap_v23": "b611ea55a68a67a727c2754751b478e8d3d6a4ff0b731af8a83c08612312426f",
    "frame_lookahead_v23": "8193584fda014cff16945d92fbe19ca89266fc4b713dae05ebf4672a41052ad8",
    "stage_control_v24": "47dbf736a65ca4c19c4010d6d6703cc7434eba0de39e98f23975341e04bc4328",
    "visible_front_v26": "59013c83604f636aa2239ccab5fcc362bacd2e1c741fb282d5bd732fb92d8790",
    "learning_mask_v17": "eb9f3869a5e54d883dde30136e9fc04d0ef4b9729e9b77e5404ae7cde622b5c8",
    "tracking_geometry_v20": "2021152f598082b88cabac762f12a7bb886def17a85b85cec57d1c0b31ad4e52",
    "tracking_batch_v27": "6ff824041e254bb7ada457263708e5210cb503f62d5691ee3af012564c7a4793",
    "tracking_stage_v28": "92b4e5a7b2be8556de434f1a216530d896e428e6c62e4879789ec7a9cc360975",
    "profile_visible_interaction_v30": "190ca65f9455138ba5b766238821873df07850c0a3b9436a0bd57612559931e0",
}
STAGES = ("all_measured", "qualified_measured", "raw_candidates_diagnostic")
GATES = {"primary_box": 0.0, "sensitivity_box_plus_3px": 3.0}
WINDOWS = {"all_300": (0, 299), "fixed_eligible_8_299": (8, 299)}
STRATA = ("all_labeled", "tiny_box_area_le_100", "known_range_le_700m")
ROOT = Path(__file__).resolve().parents[1]
FREEZE_FILES = ("scripts/score_aot_pilot.py", "tests/test_aot_scoring.py", "docs/aot_pilot_scoring_v1.md")
HARNESS_FILE = "scripts/run_aot_frozen_baseline.py"
POLICY = {
    "frame_count": FRAME_COUNT, "native_shape_hw": [HEIGHT, WIDTH],
    "gates_padding_px": GATES, "box_edges": "inclusive; LTWH; no rescaling",
    "windows_inclusive": {k: list(v) for k, v in WINDOWS.items()},
    "stages": list(STAGES), "strata": list(STRATA),
    "assignment": "maximum-cardinality one-to-one; deterministic distance-ordered augmenting paths",
    "assignment_scope": "all publisher labels first; strata applied after assignment",
    "polarity": "no GT polarity inferred; both detector polarities compete for each label",
    "measurement_coordinate": "measurement_source_xy only; never source_xy filtered/predicted track state",
    "coasts": "diagnostic counts only; cannot match labels",
    "coverage": "motion-invalid, runtime-warmup and unavailable frames remain in both fixed denominators",
    "negative_definition": "publisher-empty-label frames only; far/unknown-range labels are not negative",
    "claim": "custom descriptive development pilot; not official AOT AFDR/EDR, classification, or deployment accuracy",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=nonfinite)


def load_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing/linked metadata: {path}")
    require(path.stat().st_size <= 8 * 1024 * 1024, "metadata size cap exceeded")
    return strict_json(path.read_text())


def write_new(path, obj):
    data = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(data)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def point(value):
    require(isinstance(value, list) and len(value) == 2 and all(finite_number(v) for v in value),
            "finite two-number source coordinates required")
    return tuple(value)


def nonnegative_int(value):
    return type(value) is int and value >= 0


def validate_manifest(manifest):
    require(manifest.get("part") == "part1" and isinstance(manifest.get("flight_id"), str)
            and re.fullmatch(r"[a-f0-9]{32}", manifest["flight_id"]), "invalid AOT sequence identity")
    require(manifest.get("bb_convention") == "left,top,width,height", "unexpected GT box convention")
    frames = manifest.get("frames")
    require(isinstance(frames, list) and len(frames) == FRAME_COUNT, "exactly 300 GT frames required")
    output, previous_frame, previous_time = [], None, None
    for index, row in enumerate(frames):
        source, timestamp = row.get("source_frame"), row.get("timestamp_ns")
        require(nonnegative_int(source) and (previous_frame is None or source == previous_frame + 1),
                "missing/reordered source frame")
        require(isinstance(timestamp, str) and re.fullmatch(r"[0-9]{19}", timestamp)
                and (previous_time is None or int(timestamp) > previous_time), "invalid exact source time")
        name = f"{timestamp}{manifest['flight_id']}.png"
        require(row.get("img_name") == name, "source image/time identity mismatch")
        entities = row.get("entities")
        require(isinstance(entities, list) and len(entities) > 0, "explicit labeled or empty entity required")
        objects, seen, empties = [], set(), 0
        for entity in entities:
            require(isinstance(entity, dict) and entity.get("flight_id") == manifest["flight_id"]
                    and entity.get("img_name") == name and type(entity.get("time")) is int
                    and entity["time"] == int(timestamp) and entity.get("blob", {}).get("frame") == source,
                    "entity identity disagrees with source frame")
            require(("bb" in entity) == ("id" in entity), "partial object annotation")
            if "bb" not in entity:
                empties += 1
                continue
            oid, box = entity["id"], entity["bb"]
            require(isinstance(oid, str) and oid.strip() and oid not in seen, "invalid/duplicate object ID")
            require(isinstance(box, list) and len(box) == 4 and all(finite_number(v) for v in box)
                    and min(box[2:]) > 0, "invalid LTWH box")
            distance = entity["blob"].get("range_distance_m")
            require(distance is None or finite_number(distance) and distance >= 0, "invalid optional range")
            horizon = entity.get("labels", {}).get("is_above_horizon")
            require(horizon is None or type(horizon) is int and horizon in (-1, 0, 1), "invalid horizon")
            seen.add(oid)
            objects.append({"object_id": oid, "box_ltwh": box[:], "area_px2": box[2] * box[3],
                            "range_distance_m": distance, "is_above_horizon": horizon})
        require(not (objects and empties) and (objects or empties == 1), "mixed/duplicate empty frame record")
        require(type(row.get("airborne_label_count")) is int
                and row["airborne_label_count"] == len(objects), "label count mismatch")
        output.append({"frame_index": index, "source_frame": source, "source_timestamp_ns": timestamp,
                       "img_name": name, "objects": sorted(objects, key=lambda o: o["object_id"])})
        previous_frame, previous_time = source, int(timestamp)
    return output


def validate_rows(rows):
    require(isinstance(rows, list) and len(rows) == FRAME_COUNT, "exactly 300 completed journal rows required")
    previous_segment = 0
    for index, row in enumerate(rows):
        require(type(row.get("frame_index")) is int and row["frame_index"] == index,
                "missing/duplicate/reordered journal frame")
        require(type(row.get("timestamp_ns")) is int and row["timestamp_ns"] == index * INTERVAL_NS,
                "unexpected nominal 10 Hz journal timestamp")
        segment = row.get("segment")
        require(nonnegative_int(segment) and segment >= previous_segment, "invalid/decreasing segment")
        previous_segment = segment
        coverage, motion = row.get("coverage"), row.get("motion")
        require(isinstance(coverage, dict) and isinstance(motion, dict), "explicit coverage and motion required")
        require(coverage.get("full_shape_hw") == [HEIGHT, WIDTH]
                and coverage.get("configured_crop") is None and coverage.get("native_pixel_sampling") is True,
                "native uncropped source coverage required")
        require(all(type(coverage.get(k)) is bool for k in ("warmup", "detection_ready"))
                and nonnegative_int(coverage.get("searchable_pixels"))
                and coverage["searchable_pixels"] <= HEIGHT * WIDTH, "invalid availability fields")
        require(coverage["detection_ready"] == (not coverage["warmup"] and coverage["searchable_pixels"] > 0),
                "inconsistent detection readiness")
        require(motion.get("backend") == "pva" and type(motion.get("reset")) is bool
                and type(motion.get("pva_failure")) is bool and isinstance(motion.get("status"), str),
                "explicit PVA motion diagnostics required")
        if "accepted" in motion:
            require(type(motion["accepted"]) is bool, "motion accepted must be boolean")
        tracks, candidates = row.get("tracks"), row.get("candidates")
        require(isinstance(tracks, list) and isinstance(candidates, list), "explicit tracks/candidates lists required")
        require(len(tracks) <= 512 and len(candidates) <= 512, "frozen observation cap exceeded")
        identities = set()
        for track in tracks:
            tid = track.get("track_id")
            require(isinstance(tid, str) and re.fullmatch(r"(?:bright|dark):[0-9]+", tid), "invalid polarity/track ID")
            require(type(track.get("segment")) is int and track["segment"] == segment and tid not in identities,
                    "duplicate/cross-segment track state")
            identities.add(tid)
            require(type(track.get("measured")) is bool and type(track.get("qualified_moving")) is bool,
                    "explicit measurement/qualification booleans required")
            point(track.get("source_xy"))  # validate but never use filtered state for a measured hit
            require("measurement_source_xy" in track, "explicit current measurement or null required")
            if track["measured"]:
                point(track["measurement_source_xy"])
            else:
                require(track["measurement_source_xy"] is None, "coast cannot contain a current measurement")
        for candidate in candidates:
            point(candidate.get("source_xy"))
            require(candidate.get("polarity") in ("bright", "dark"), "invalid proposal polarity")


def stage_observations(row):
    measured = [{"identity": f"{t['segment']}/{t['track_id']}",
                 "segment": t["segment"], "polarity": t["track_id"].split(":")[0],
                 "track_id": t["track_id"], "xy": point(t["measurement_source_xy"]),
                 "qualified": t["qualified_moving"]}
                for t in row["tracks"] if t["measured"]]
    measured.sort(key=lambda o: o["identity"])
    candidates = [{"identity": f"candidate:{i:04d}", "xy": point(c["source_xy"]),
                   "polarity": c["polarity"]} for i, c in enumerate(row["candidates"])]
    return {"all_measured": measured, "qualified_measured": [o for o in measured if o["qualified"]],
            "raw_candidates_diagnostic": candidates}


def inside(xy, box, padding):
    left, top, width, height = box
    return left - padding <= xy[0] <= left + width + padding and top - padding <= xy[1] <= top + height + padding


def assign(objects, observations, padding):
    """Maximum cardinality, not greedy nearest-only or a temporal identity oracle."""
    neighbors = []
    for obj in objects:
        left, top, width, height = obj["box_ltwh"]
        center = left + width / 2, top + height / 2
        neighbors.append(sorted((j for j, obs in enumerate(observations) if inside(obs["xy"], obj["box_ltwh"], padding)),
                                key=lambda j: (math.dist(center, observations[j]["xy"]), observations[j]["identity"])))
    owner = {}

    def augment(index, visited):
        for j in neighbors[index]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or augment(owner[j], visited):
                owner[j] = index
                return True
        return False

    for i in range(len(objects)):
        augment(i, set())
    return {i: j for j, i in owner.items()}, neighbors


def object_strata(obj):
    return {"all_labeled": True, "tiny_box_area_le_100": obj["area_px2"] <= 100,
            "known_range_le_700m": obj["range_distance_m"] is not None and obj["range_distance_m"] <= 700}


def coverage_summary(rows, gt):
    counts, statuses, reasons = Counter(), Counter(), Counter()
    for row, truth in zip(rows, gt):
        c, m = row["coverage"], row["motion"]
        issue = m["pva_failure"] or m["reset"] or m.get("accepted") is False or m["status"] not in ("accepted", "initial_reference")
        flags = {"runtime_warmup": c["warmup"], "not_detection_ready": not c["detection_ready"],
                 "no_searchable_support": c["searchable_pixels"] == 0, "motion_reset": m["reset"],
                 "pva_failure": m["pva_failure"], "motion_fit_rejected": m.get("accepted") is False,
                 "motion_issue": issue}
        for name, flag in flags.items():
            counts[name + "_frames"] += bool(flag)
            counts[name + "_labeled_annotations"] += len(truth["objects"]) * bool(flag)
        statuses.update([m["status"]])
        reasons.update([str(c.get("unavailable_reason"))])
    return {**dict(counts), "motion_status_counts": dict(statuses), "unavailable_reason_counts": dict(reasons),
            "frames_excluded_for_motion_or_availability": 0}


def summarize_history(history, track_stage):
    hits = [h for h in history if h["matched_identity"] is not None]
    ambiguous = any(h["ambiguous_gate"] for h in history)
    contiguous = all(b["frame_index"] == a["frame_index"] + 1 for a, b in zip(history, history[1:]))
    first = hits[0] if hits and not hits[0]["ambiguous_gate"] else None
    first_hit = None if first is None else {
        "frame_index": first["frame_index"], "source_frame": first["source_frame"],
        "source_timestamp_ns": first["source_timestamp_ns"], "identity": first["matched_identity"],
        "nominal_seconds_after_first_label_in_window": (first["frame_index"] - history[0]["frame_index"]) / 10,
        "source_ns_after_first_label_in_window": str(int(first["source_timestamp_ns"]) - int(history[0]["source_timestamp_ns"])),
        "not_physical_onset_latency": True,
    }
    fragments = None
    if track_stage and contiguous and not ambiguous:
        fragments, previous = 0, None
        for entry in history:
            current = entry["matched_identity"]
            if current is not None and current != previous:
                fragments += 1
            previous = current
    return {"labeled_annotation_frames": len(history), "matched_annotation_frames": len(hits),
            "matched_identities": sorted({h["matched_identity"] for h in hits}) if track_stage else None,
            "first_unambiguous_hit": first_hit,
            "first_hit_unavailable_reason": None if first else "no_match" if not hits else "ambiguous_first_match",
            "identity_fragments": fragments,
            "identity_fragmentation_additional_fragments": max(0, fragments - 1) if fragments is not None else None,
            "identity_fragmentation_available": fragments is not None,
            "ambiguous_gate_present": ambiguous, "continuous_gt_in_window": contiguous, "history": history}


def score_stage(rows, truth, stage, padding):
    totals, negative_ids, histories, details = Counter(), set(), {}, []
    strata = {name: {"labeled_annotations": 0, "matched_annotations": 0} for name in STRATA}
    for row, gt in zip(rows, truth):
        observations = stage_observations(row)[stage]
        objects = gt["objects"]
        matches, neighbors = assign(objects, observations, padding)
        gated = {j for edge in neighbors for j in edge}
        degrees = Counter(j for edge in neighbors for j in edge)
        unmatched = set(range(len(observations))) - set(matches.values())
        counts = {
            "observations": len(observations), "matched_annotations": len(matches),
            "unmatched_observations": len(unmatched),
            "unmatched_inside_any_gt_gate": len(unmatched & gated),
            "unmatched_outside_all_gt_gates": len(unmatched - gated),
            "source_points_outside_image": sum(not (0 <= o["xy"][0] < WIDTH and 0 <= o["xy"][1] < HEIGHT) for o in observations),
        }
        totals.update(counts)
        if not objects:
            totals["publisher_empty_frames"] += 1
            totals["observations_on_publisher_empty_frames"] += len(observations)
            totals["publisher_empty_frames_with_observations"] += bool(observations)
            negative_ids.update(o["identity"] for o in observations)
        assignments = []
        for i, obj in enumerate(objects):
            for name, member in object_strata(obj).items():
                if member:
                    strata[name]["labeled_annotations"] += 1
                    strata[name]["matched_annotations"] += i in matches
            chosen = observations[matches[i]] if i in matches else None
            ambiguous = len(neighbors[i]) > 1 or any(degrees[j] > 1 for j in neighbors[i])
            entry = {k: gt[k] for k in ("frame_index", "source_frame", "source_timestamp_ns")}
            entry.update(matched_identity=None if chosen is None else chosen["identity"],
                         measured_source_xy=None if chosen is None else list(chosen["xy"]),
                         gated_identities=[observations[j]["identity"] for j in neighbors[i]], ambiguous_gate=ambiguous)
            histories.setdefault(obj["object_id"], []).append(entry)
            assignments.append({"object_id": obj["object_id"], **entry})
        details.append({"frame_index": row["frame_index"], **counts, "object_assignments": assignments,
                        "unmatched_identities": [observations[j]["identity"] for j in sorted(unmatched)]})
    for values in strata.values():
        values["missed_annotations"] = values["labeled_annotations"] - values["matched_annotations"]
        values["annotation_match_fraction"] = values["matched_annotations"] / values["labeled_annotations"] if values["labeled_annotations"] else None
    negative_frames = totals["publisher_empty_frames"]
    negative = {key: totals[key] for key in ("publisher_empty_frames", "observations_on_publisher_empty_frames", "publisher_empty_frames_with_observations")}
    negative.update(nominal_exposure_seconds=negative_frames / 10,
                    observations_per_publisher_empty_frame=totals["observations_on_publisher_empty_frames"] / negative_frames if negative_frames else None,
                    unique_segment_polarity_track_identities=len(negative_ids) if stage != "raw_candidates_diagnostic" else None,
                    no_hourly_false_alarm_or_generalization_claim=True)
    return {"counts": dict(totals), "strata": strata, "publisher_empty_context": negative,
            "per_object": {oid: summarize_history(hist, stage != "raw_candidates_diagnostic") for oid, hist in histories.items()},
            "frames": details}


def score_rows(manifest, rows):
    truth = validate_manifest(manifest)
    validate_rows(rows)
    windows = {}
    for name, (first, last) in WINDOWS.items():
        selected, gt = rows[first:last + 1], truth[first:last + 1]
        coasts = [t for row in selected for t in row["tracks"] if not t["measured"]]
        windows[name] = {
            "input_frame_range_inclusive": [first, last], "frame_count": len(selected),
            "labeled_frames": sum(bool(g["objects"]) for g in gt),
            "publisher_empty_frames": sum(not g["objects"] for g in gt),
            "coverage_context": coverage_summary(selected, gt),
            "coasts_diagnostic_only": {"states": len(coasts), "qualified_states": sum(t["qualified_moving"] for t in coasts),
                "unique_segment_polarity_track_identities": len({(t["segment"], t["track_id"]) for t in coasts}), "counted_as_hits": 0},
            "gates": {gate: {stage: score_stage(selected, gt, stage, padding) for stage in STAGES}
                      for gate, padding in GATES.items()},
        }
    return {"schema": SCHEMA, "policy": POLICY, "part": manifest["part"], "flight_id": manifest["flight_id"],
            "physical_class_inferred": False, "official_aot_metrics": False, "independent_validation": False,
            "deployment_accuracy_claim": False, "windows": windows}


def read_journal(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024 * 1024,
            "missing/linked/oversize frame journal")
    rows = []
    with path.open() as stream:
        while True:
            line = stream.readline(2 * 1024 * 1024 + 1)
            if not line:
                break
            require(line.strip() and len(line.encode()) <= 2 * 1024 * 1024 and len(rows) < FRAME_COUNT,
                    "blank/oversize/excess journal row")
            rows.append(strict_json(line))
    validate_rows(rows)
    return rows


def validate_run_metadata(launch, report):
    require(launch.get("source_sha256") == report.get("source_sha256") == VIDEO_SHA256,
            "run source is not the frozen lossless AOT video")
    require(launch.get("config_sha256") == CONFIG_SHA256
            and launch.get("code_sha256", {}).get("visible_baseline.py") == BASELINE_SHA256,
            "frozen detector/configuration identity changed")
    require(launch.get("motion_config_sha256") == MOTION_SHA256, "motion configuration identity changed")
    require(launch.get("annotations_supplied_to_detector") is False, "detector must not receive annotations")
    require(report.get("completed") is True and type(report.get("frames")) is int and report["frames"] == FRAME_COUNT
            and report.get("full_clip") is True and launch.get("max_frames", "missing") is None
            and launch.get("expected_frames") == FRAME_COUNT and launch.get("fps") == 10,
            "run incomplete or wrong cadence/frame count")
    probe = launch.get("source_probe", {})
    require(probe.get("width") == WIDTH and probe.get("height") == HEIGHT
            and probe.get("codec") == "ffv1" and probe.get("pixel_format") in ("gray", "gray8")
            and str(probe.get("frame_rate")) in ("10", "10/1") and probe.get("declared_frame_count") == FRAME_COUNT,
            "not the native lossless 300-frame 10 Hz input")
    cfg = launch.get("configuration")
    require(isinstance(cfg, dict) and report.get("configuration") == cfg, "launch/report configuration mismatch")
    expected = {"input_bit_depth": 8, "warmup_frames": 8, "motion_backend": "pva",
                "state_update_backend": "cuda_resident", "spatial_filter_backend": "cuda_median5",
                "stabilization_execution": "cuda_cubic_resident", "frame_decode_execution": "prefetch_one"}
    require(all(cfg.get(key) == value for key, value in expected.items()), "not the frozen native PVA/GPU execution")
    stats = report.get("frame_decode", {})
    require(stats.get("decoded_frames") == stats.get("consumed_frames") == FRAME_COUNT
            and stats.get("dropped_frames") == 0 and stats.get("worker_joined") is True
            and stats.get("capture_released") is True, "decode completion/order receipt failed")


def validate_execution(receipt, preflight, hashes):
    """Independent receipt checks; never import or execute the runtime harness."""
    schema = "seaqr.aot.frozen-baseline.v1"
    require(receipt.get("schema") == schema and receipt.get("passed") is True and receipt.get("error") is None,
            "successful frozen combined-stack execution receipt required")
    require(receipt.get("processed_frames") == receipt.get("pixel_hashes_verified") == FRAME_COUNT,
            "execution/pixel validation incomplete")
    for key in ("algorithm_changed", "detector_configuration_changed", "annotations_supplied_to_detector",
                "raw16_accessed", "private_camera_media_accessed", "clocks_changed"):
        require(receipt.get(key) is False, f"execution scope changed: {key}")
    require(receipt.get("remote_clocks_unchanged") is True
            and isinstance(receipt.get("clock_policy_before"), dict) and receipt["clock_policy_before"]
            and receipt.get("clock_policy_after") == receipt["clock_policy_before"], "remote clock policy changed")
    require(all(receipt.get(key) == value for key, value in RUNTIME_IDENTITIES.items()), "runtime identity changed")
    require(receipt.get("libraries") == LIBRARIES and isinstance(receipt.get("adapters"), dict)
            and {key: value.get("sha256") for key, value in receipt["adapters"].items()} == ADAPTERS,
            "native libraries/adapters differ from frozen stack")
    for key in ("journal", "launch", "report", "preflight", "scoring_freeze", "script"):
        require(receipt.get(key + "_sha256") == hashes[key], f"receipt {key} hash mismatch")
    inputs = receipt.get("input_sha256", {})
    require(inputs.get("frozen_image_manifest.json") == MANIFEST_SHA256
            and inputs.get("pilot_gray8_ffv1_10fps.avi") == VIDEO_SHA256
            and inputs.get("scoring_freeze.json") == hashes["scoring_freeze"], "execution input/freeze mismatch")
    require(preflight.get("schema") == schema + ".preflight" and preflight.get("passed") is True
            and preflight.get("detector_run") is False and preflight.get("script_sha256") == hashes["script"]
            and preflight.get("input_sha256") == inputs and preflight.get("workspace") == receipt.get("workspace")
            and preflight.get("decode", {}).get("passed") is True
            and preflight["decode"].get("pixel_hashes_verified") == FRAME_COUNT, "preflight linkage failed")
    for key in ("libraries", "adapters", "config_sha256", "motion_config_sha256", "v29_freeze_sha256", "reuse_method_sha256", "vpi_version"):
        require(preflight.get(key) == receipt.get(key), "dependencies changed after preflight")
    fronts = receipt.get("gpu_fronts")
    require(isinstance(fronts, list) and len(fronts) == 1 and fronts[0].get("closed") is True
            and all(fronts[0].get(k) == FRAME_COUNT for k in ("calls", "device_calls", "finish_calls"))
            and fronts[0].get("host_calls") == 0 and receipt.get("native_mask_calls") == 0,
            "GPU lifecycle/fallback gate failed")
    tracking = receipt.get("tracking", {})
    require(all(tracking.get(k) == 0 for k in ("geometry_calls", "geometry_fallbacks", "batch_fallbacks", "innovation_fallbacks"))
            and nonnegative_int(tracking.get("batch_track_rows"))
            and tracking.get("innovation_tracks") == tracking["batch_track_rows"]
            and nonnegative_int(tracking.get("batch_calls"))
            and tracking.get("exercised") is (tracking["batch_calls"] > 0), "tracking fallback/accounting failed")
    motion, attempts = receipt.get("motion_instances"), receipt.get("motion_attempts")
    require(isinstance(motion, list) and len(motion) == 1 and motion[0].get("failed") is False
            and motion[0].get("closed") is True and receipt.get("cleanup_errors") == [], "PVA cleanup failed")
    require(isinstance(attempts, list) and [row.get("frame") for row in attempts] == list(range(1, FRAME_COUNT))
            and all(row.get("error") is None or row.get("expected_unavailable") is True for row in attempts),
            "missing/unexpected PVA motion attempts")
    execution = receipt.get("execution", {})
    require(execution.get("policy") == execution.get("mode") == "reference"
            and [row.get("frame") for row in execution.get("frames", [])] == list(range(FRAME_COUNT)),
            "combined execution frame/ownership policy changed")
    before, after = receipt.get("runtime_before", {}), receipt.get("runtime_after", {})
    keys = ("blas", "affinity", "numpy", "opencv", "thread_environment", "clock_ticks")
    require(all(key in before and before[key] == after.get(key) for key in keys)
            and before.get("numpy") == "1.26.1" and before.get("opencv") == "4.10.0"
            and len(before.get("blas", [])) == 1 and before["blas"][0].get("threads") == 12
            and all(value is None for value in before.get("thread_environment", {}).values())
            and after.get("opencv_threads") == 2, "frozen numerical runtime changed")


def make_freeze(manifest_path):
    require(digest(manifest_path) == MANIFEST_SHA256, "not the selected frozen pilot manifest")
    truth = validate_manifest(load_json(manifest_path))
    return {"schema": SCHEMA + ".freeze", "created_utc": datetime.now(timezone.utc).isoformat(),
            "detector_outputs_read": False, "manifest_sha256": MANIFEST_SHA256,
            "video_sha256": VIDEO_SHA256, "policy": POLICY,
            "files_sha256": {name: digest(ROOT / name) for name in FREEZE_FILES},
            "runtime_harness_sha256": digest(ROOT / HARNESS_FILE),
            "label_only_census": {"frames": len(truth), "labeled_frames": sum(bool(g["objects"]) for g in truth),
                                   "publisher_empty_frames": sum(not g["objects"] for g in truth)},
            "evaluation_requires_root_review": True}


def verify_freeze(frozen, manifest_path):
    require(frozen.get("schema") == SCHEMA + ".freeze" and frozen.get("detector_outputs_read") is False,
            "missing pre-output scoring freeze")
    require(frozen.get("manifest_sha256") == digest(manifest_path) == MANIFEST_SHA256
            and frozen.get("video_sha256") == VIDEO_SHA256 and frozen.get("policy") == POLICY,
            "frozen source or policy changed")
    require(frozen.get("files_sha256") == {name: digest(ROOT / name) for name in FREEZE_FILES},
            "scorer/tests/policy changed after freeze")
    require(frozen.get("runtime_harness_sha256") == digest(ROOT / HARNESS_FILE), "runtime harness changed after freeze")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execution-receipt", type=Path, help="Defaults to run directory's sibling execution_receipt.json")
    parser.add_argument("--preflight", type=Path, help="Defaults to run directory's sibling preflight.json")
    parser.add_argument("--review-approved", action="store_true", help="Root reviewed the frozen policy before evaluation")
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output required")
    if args.phase == "freeze":
        require(args.freeze is None and args.run_dir is None and args.execution_receipt is None and args.preflight is None,
                "freeze never reads detector output")
        write_new(args.output, make_freeze(args.manifest))
        return
    require(args.review_approved and args.freeze is not None and args.run_dir is not None,
            "root-reviewed freeze and completed run directory required")
    frozen = load_json(args.freeze)
    verify_freeze(frozen, args.manifest)
    launch_path, report_path, journal_path = (args.run_dir / name for name in ("launch.json", "report.json", "frames.jsonl"))
    require(not (args.run_dir / "failure.json").exists(), "failed execution cannot be scored")
    receipt_path = args.execution_receipt or args.run_dir.parent / "execution_receipt.json"
    preflight_path = args.preflight or args.run_dir.parent / "preflight.json"
    hashes = {"manifest": digest(args.manifest), "scoring_freeze": digest(args.freeze),
              "launch": digest(launch_path), "report": digest(report_path), "journal": digest(journal_path),
              "preflight": digest(preflight_path), "script": frozen["runtime_harness_sha256"],
              "execution_receipt": digest(receipt_path)}
    validate_execution(load_json(receipt_path), load_json(preflight_path), hashes)
    validate_run_metadata(load_json(launch_path), load_json(report_path))
    result = score_rows(load_json(args.manifest), read_journal(journal_path))
    for name, path in (("manifest", args.manifest), ("scoring_freeze", args.freeze), ("launch", launch_path),
                       ("report", report_path), ("journal", journal_path), ("preflight", preflight_path),
                       ("execution_receipt", receipt_path)):
        require(digest(path) == hashes[name], f"scoring input changed while reading: {name}")
    verify_freeze(frozen, args.manifest)
    result["inputs_sha256"] = hashes
    write_new(args.output, result)


if __name__ == "__main__":
    main()
