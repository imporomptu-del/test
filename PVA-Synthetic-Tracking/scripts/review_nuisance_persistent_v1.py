"""Frozen source-only review of persistent qualified-measurement identities.

Ranks archived metadata, not visual appearance. No detector or GPU execution.
All fixed source crops retain decoded native BGR pixels and identity contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "outputs/seaqr_discovery_pair_20260928/sources/chunk_0240.avi"
JOURNAL = ROOT.parent / "outputs/seaqr_feature_residual_trace_20260930/evidence/0240/run/frames.jsonl"
SOURCE_SHA = "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"
JOURNAL_SHA = "f669520c0af65315b5a1f106bf60a5b0ebffe7b8539d3dd973d9b2f2bdb7ee92"
SCHEMA = "seaqr.nuisance-persistent-review.v1"
FIRST, LAST, FULL_FRAMES, WIDTH, HEIGHT, CROP = 50, 105, 673, 4784, 3190, 257
COUNT = 8
OLD_PACKET = ROOT.parent / "outputs/seaqr_nuisance_origin_20261001"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def floating(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=floating,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or linked metadata")
    return decode(path.read_text())


def write(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "existing output; no overwrite")
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(path, digest):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            "regular canonical input path required")
    require(sha(path) == digest, "input SHA256 mismatch")


def coordinates(value):
    require(isinstance(value, list) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value), "finite native source point required")
    return [float(v) for v in value]


def track_key(track):
    require(type(track.get("segment")) is int and track["segment"] >= 0, "invalid track segment")
    name = track.get("track_id")
    require(isinstance(name, str), "invalid track ID")
    matched = re.fullmatch(r"(bright|dark):([1-9][0-9]*|0)", name)
    require(matched is not None, "track identity lacks numeric polarity-qualified ID")
    return track["segment"], matched[1], int(matched[2])


def identity(key):
    segment, polarity, number = key
    return f"{segment}/{polarity}:{number}"


def fixed_roi(points):
    xy = np.asarray([coordinates(p) for p in points], np.float64)
    require(len(xy) > 0, "ROI needs qualified raw measurements")
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    # Python round: nearest integer, ties-to-even; fixed once for the whole clip window.
    center = [round(float(v)) for v in ((lower + upper) / 2)]
    roi = [center[0] - CROP // 2, center[1] - CROP // 2, CROP, CROP]
    x, y, w, h = roi
    available = x >= 0 and y >= 0 and x + w <= WIDTH and y + h <= HEIGHT
    out = sum(not (x <= p[0] < x + w and y <= p[1] < y + h) for p in xy)
    return dict(roi_source_xywh=roi, roi_center_source_xy=center, reviewable=available,
        unreviewable_reason=None if available else "fixed_roi_exceeds_source_edge_no_clamp_or_padding",
        raw_measurement_min_xy=lower.tolist(), raw_measurement_max_xy=upper.tolist(),
        raw_measurement_span_xy=(upper - lower).tolist(), span_exceeds_crop=bool(np.any(upper - lower > CROP)),
        raw_measurements_outside_fixed_roi=out)


def select(window):
    """Pure selection from all56 archived rows. Does not access media."""
    require([r.get("frame_index") for r in window] == list(range(FIRST, LAST + 1)), "complete ordered review window required")
    cohort, states = {}, {}
    qualified_measured = qualified_predicted = qualified_coasted = 0
    for row in window:
        f = row["frame_index"]
        seen = set()
        for track in row["tracks"]:
            key = track_key(track)
            require(key not in seen, "duplicate identity within frame")
            seen.add(key)
            require(type(track.get("measured")) is bool and type(track.get("qualified_moving")) is bool,
                    "measurement/qualification flags must be boolean")
            measured, qualified = track["measured"], track["qualified_moving"]
            filtered = coordinates(track["source_xy"])
            raw = coordinates(track["measurement_source_xy"]) if measured else None
            require(measured or track.get("measurement_source_xy") is None, "prediction carries a raw measurement")
            state = dict(frame_index=f, present=True, measured=measured, qualified_moving=qualified,
                lifecycle=track["lifecycle"], raw_measurement_source_xy=raw, filtered_track_source_xy=filtered,
                coordinate_note="raw measurement when present; filtered tracker position is separate and may be predicted")
            states[(f, key)] = state
            qualified_predicted += bool(qualified and not measured)
            qualified_coasted += bool(qualified and track["lifecycle"] == "coasted")
            if qualified and measured:
                qualified_measured += 1
                entry = cohort.setdefault(key, dict(key=key, first=f, last=f, measured=[]))
                entry["last"] = f
                entry["measured"].append(state)
    ordered = sorted(cohort.values(), key=lambda e: (-len(e["measured"]), e["first"], *e["key"]))
    ranking = []
    for rank, e in enumerate(ordered, 1):
        key = e["key"]
        history = [states[(f, key)] for f in range(FIRST, LAST + 1) if (f, key) in states]
        ranking.append(dict(rank=rank, identity=identity(key), segment=key[0], polarity=key[1], numeric_track_id=key[2],
            qualified_measured_states=len(e["measured"]), first_qualified_measured_frame=e["first"],
            last_qualified_measured_frame=e["last"],
            qualified_predicted_states=sum(s["qualified_moving"] and not s["measured"] for s in history),
            qualified_coasted_states=sum(s["qualified_moving"] and s["lifecycle"] == "coasted" for s in history),
            raw_qualified_measurements=[dict(frame_index=s["frame_index"], source_xy=s["raw_measurement_source_xy"]) for s in e["measured"]],
            raw_measurements=[dict(frame_index=s["frame_index"], source_xy=s["raw_measurement_source_xy"]) for s in history if s["measured"]],
            physical_class="unknown", selected=rank <= COUNT))
    chosen = []
    for ranked, entry in zip(ranking[:COUNT], ordered[:COUNT]):
        key = entry["key"]
        item = dict(ranked, **fixed_roi([s["source_xy"] for s in ranked["raw_measurements"]]))
        item["frames"] = [states.get((f, key), dict(frame_index=f, present=False, measured=None,
            qualified_moving=None, lifecycle=None, raw_measurement_source_xy=None, filtered_track_source_xy=None))
            for f in range(FIRST, LAST + 1)]
        first, last = entry["first"], entry["last"]
        item["contact_frames"] = [round(first + (last - first) * i / 7) for i in range(8)]
        item["contact_sampling"] = "Eight evenly spaced temporal slots from first through last qualified raw measurement, nearest frame ties-to-even; repeats retained for short spans."
        item["contact_has_repeated_frames"] = len(set(item["contact_frames"])) < 8
        item["all_measured_states_outside_fixed_roi"] = sum(s["measured"] is True
            and not inside(s["raw_measurement_source_xy"], item["roi_source_xywh"]) for s in item["frames"])
        item["qualified_measurements_outside_fixed_roi"] = sum(not inside(s["source_xy"], item["roi_source_xywh"])
            for s in ranked["raw_qualified_measurements"])
        chosen.append(item)
    selected_count = sum(s["qualified_measured_states"] for s in chosen)
    workload = dict(qualified_measured_states=qualified_measured, qualified_predicted_states=qualified_predicted,
        qualified_coasted_states=qualified_coasted, measured_identity_count=len(ranking), selected_identity_count=len(chosen),
        selected_qualified_measured_states=selected_count,
        selected_qualified_measured_fraction=selected_count / qualified_measured if qualified_measured else None,
        selected_qualified_predicted_states=sum(s["qualified_predicted_states"] for s in chosen),
        counts_are_objects_or_false_positives=False)
    return ranking, chosen, workload


def inside(point, roi):
    x, y, w, h = roi
    return x <= point[0] < x + w and y <= point[1] < y + h


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and plan.get("clip") == "0240"
            and plan.get("frame_range_inclusive") == [FIRST, LAST] and plan.get("crop_size") == CROP,
            "persistent review scope differs")
    require(plan.get("input_paths") == dict(source=str(SOURCE), journal=str(JOURNAL))
            and plan.get("input_sha256") == dict(source=SOURCE_SHA, journal=JOURNAL_SHA), "fixed approved input paths/hashes differ")
    require(plan.get("workload", {}).get("qualified_measured_states") == 587, "frozen measured workload differs")
    selected, ranking = plan["selected"], plan["ranking"]
    require(len(selected) == COUNT and len({s["identity"] for s in selected}) == COUNT,
            "exactly eight distinct selected identities required")
    require([s["identity"] for s in selected] == [r["identity"] for r in ranking[:COUNT]], "selection differs from top8ranking")
    for s in selected:
        require([r["frame_index"] for r in s["frames"]] == list(range(FIRST, LAST + 1)), "selected timeline differs")
        expected = fixed_roi([r["source_xy"] for r in s["raw_measurements"]])
        require(all(s[k] == v for k, v in expected.items()), "fixed ROI differs from raw measurement envelope")
        require(len(s["contact_frames"]) == 8 and s["contact_frames"][0] == s["first_qualified_measured_frame"]
                and s["contact_frames"][-1] == s["last_qualified_measured_frame"], "contact endpoints differ")
    return plan


def build_plan(source=SOURCE, journal=JOURNAL):
    require(Path(source) == SOURCE and Path(journal) == JOURNAL, "unapproved input path; only fixed0240 allowed")
    verify(source, SOURCE_SHA)
    verify(journal, JOURNAL_SHA)
    window, count = [], 0
    with Path(journal).open() as stream:
        for count, line in enumerate(stream, 1):
            row = decode(line)
            require(type(row.get("frame_index")) is int and row["frame_index"] == count - 1
                    and row.get("timestamp_ns") == (count - 1) * 100000000, "journal continuity/cadence differs")
            if FIRST <= row["frame_index"] <= LAST:
                window.append(row)
    require(count == FULL_FRAMES, "complete frozen journal required")
    ranking, selected, workload = select(window)
    plan = dict(schema=SCHEMA, clip="0240", frame_range_inclusive=[FIRST, LAST], crop_size=CROP,
        nominal_fps=10, source_shape_hw=[HEIGHT, WIDTH], input_paths=dict(source=str(source), journal=str(journal)),
        input_sha256=dict(source=SOURCE_SHA, journal=JOURNAL_SHA), ranking=ranking, selected=selected, workload=workload,
        ranking_policy="All identities with qualified_moving and measured in50..105; count descending, first qualifying measurement frame, segment, polarity lexical, numeric track ID ascending. Top8 without appearance replacement.",
        coordinate_policy="Fixed257x257 source ROI centered at round((min+max)/2) of ALL raw measurement_source_xy states of the selected identity over the window, including unqualified measurements; no filtered positions, recentering, clamping or padding. Ranking remains qualified-measured-only.",
        separate_controls=dict(existing_packet=str(OLD_PACKET), no_additional_decode=True,
            episodes=["known_target_435_dark", "known_target_450_dark", "burst_060_bright", "burst_060_dark", "burst_075_bright", "burst_075_dark"]),
        source_decode_limit_inclusive=LAST, detector_run=False, gpu_run=False, raw16_accessed=False,
        sealed_holdouts_accessed=False, independent_negative_truth=False, production_changed=False,
        limitations=["Persistent qualified states are review workload, not confirmed objects or false positives.",
            "Ranking is detector-led and development-only; no appearance-based sample replacement.",
            "Coasted/predicted states are counted separately and never anchor the ROI.",
            "Lossless PNGs preserve decoded native BGR pixels, not the original compressed MJPEG bitstream.",
            "Nominal10Hz playback is not physical capture cadence or processing FPS."])
    verify(source, SOURCE_SHA)
    verify(journal, JOURNAL_SHA)
    return validate_plan(plan)


def crop_native(frame, roi):
    require(isinstance(frame, np.ndarray) and frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8,
            "native decoded BGR frame required")
    require(isinstance(roi, list) and len(roi) == 4 and all(type(v) is int for v in roi), "integer fixed ROI required")
    x, y, w, h = roi
    require(w == h == CROP and x >= 0 and y >= 0 and x + w <= WIDTH and y + h <= HEIGHT,
            "source edge exceeded; no clamping or padding")
    return frame[y:y + h, x:x + w].copy()


def slug(item):
    return f"rank{item['rank']:02d}_seg{item['segment']}_{item['polarity']}_{item['numeric_track_id']}"


def contact_sheet(item, directory):
    import cv2
    cell, gap, header, label = CROP, 5, 78, 25
    canvas = np.full((header + 2 * (cell + label + gap), 4 * (cell + gap), 3), 22, np.uint8)
    cv2.putText(canvas, item["identity"] + " | unmarked native source | identity/class unknown", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (235, 235, 235), 1)
    cv2.putText(canvas, "Fixed257x257 ROI; shared0..255 identity contrast; M=measured, P=prediction only", (8, 47), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1)
    cv2.putText(canvas, "Image pixels are unmarked; state captions refer to this selected tracker identity only.", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1)
    for index, f in enumerate(item["contact_frames"]):
        pixels = cv2.imread(str(Path(directory) / f"frame_{f:03d}.png"), cv2.IMREAD_COLOR)
        require(pixels is not None and pixels.shape == (CROP, CROP, 3), "missing lossless native crop")
        state = item["frames"][f - FIRST]
        tag = "not active" if not state["present"] else ("M" if state["measured"] else "P") + (" qualified" if state["qualified_moving"] else " unqualified")
        x, y = (index % 4) * (cell + gap), header + (index // 4) * (cell + label + gap)
        cv2.putText(canvas, f"frame{f} | {tag}", (x + 3, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1)
        canvas[y + label:y + label + cell, x:x + cell] = pixels
    return canvas


def render(plan_path, output):
    import cv2
    plan_path, output = Path(plan_path), Path(output)
    require(not output.exists() and not output.is_symlink(), "render output exists; no overwrite")
    plan_digest, own_sha = sha(plan_path), sha(Path(__file__))
    plan = validate_plan(read(plan_path))
    require(plan == build_plan(), "plan differs from frozen deterministic selection")
    output.mkdir()
    selected = [s for s in plan["selected"] if s["reviewable"]]
    for item in selected:
        (output / slug(item)).mkdir()
    require(sha(plan_path) == plan_digest and sha(Path(__file__)) == own_sha, "plan or renderer changed before decoding")
    cap, crops, count = cv2.VideoCapture(str(SOURCE)), [], 0
    try:
        require(cap.isOpened(), "verified0240 source did not open")
        for f in range(LAST + 1):
            ok, pixels = cap.read()
            require(ok and pixels is not None and pixels.shape == (HEIGHT, WIDTH, 3) and pixels.dtype == np.uint8,
                    "source decode failed or native geometry differs")
            count += 1
            if f < FIRST:
                continue
            for item in selected:
                cropped = crop_native(pixels, item["roi_source_xywh"])
                path = output / slug(item) / f"frame_{f:03d}.png"
                require(not path.exists() and cv2.imwrite(str(path), cropped), "native PNG write failed")
                require(np.array_equal(cv2.imread(str(path), cv2.IMREAD_COLOR), cropped), "saved native PNG differs")
                crops.append(dict(identity=item["identity"], frame_index=f, path=str(path.relative_to(output)), sha256=sha(path)))
    finally:
        cap.release()
    require(count == LAST + 1 and len(crops) == len(selected) * (LAST - FIRST + 1), "bounded decode/crop count differs")
    sheets = []
    for item in selected:
        directory = output / slug(item)
        path = directory / "contact_native.png"
        require(cv2.imwrite(str(path), contact_sheet(item, directory)), "contact sheet write failed")
        sheets.append(dict(identity=item["identity"], path=str(path.relative_to(output)), sha256=sha(path)))
    verify(SOURCE, SOURCE_SHA)
    verify(JOURNAL, JOURNAL_SHA)
    require(sha(plan_path) == plan_digest and sha(Path(__file__)) == own_sha, "plan or renderer changed during execution")
    receipt = dict(schema=SCHEMA + ".render", passed=True, plan_sha256=plan_digest, renderer_sha256=own_sha,
        source_sha256=SOURCE_SHA, decoded_source_frames=count, last_source_frame_decoded=LAST,
        crop_frames_per_reviewable_identity=LAST - FIRST + 1, crops=crops, contact_sheets=sheets,
        video_encoded=False, no_video_reason="Bounded source review uses lossless native PNG sequences and contact sheets.",
        unavailable_identities=[dict(identity=s["identity"], reason=s["unreviewable_reason"]) for s in plan["selected"] if not s["reviewable"]],
        detector_run=False, gpu_run=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        contrasts="Decoded native BGR0..255 preserved; no per-patch stretch, resize, overlay or source-edge padding in PNGs.",
        physical_class_inferred=False, counts_are_objects_or_false_positives=False)
    write(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-output", type=Path)
    action.add_argument("--render-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_output:
        require(args.output is None, "render output only with render plan")
        require(not args.plan_output.exists() and not args.plan_output.is_symlink(), "plan exists; no overwrite")
        write(args.plan_output, build_plan())
    else:
        require(args.output is not None, "fresh output directory required")
        render(args.render_plan, args.output)


if __name__ == "__main__":
    main()
