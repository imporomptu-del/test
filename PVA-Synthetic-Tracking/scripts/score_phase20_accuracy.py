"""Post-run, bounded accuracy scoring. No media decoding or detector imports.

Missing labels stay unknown; predictions never count as measured detections.
This pilot cannot establish whole-clip precision, recall, or onset latency.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(xy, rect):
    x, y, w, h = rect
    return x <= xy[0] < x + w and y <= xy[1] < y + h


def validate_labels(labels, packet):
    if labels.get("schema") != "seaqr.bounded-visual-accuracy-labels.v1":
        raise ValueError("Unsupported labels")
    if labels["policy"]["extra_localization_tolerance_px"] != 2.0:
        raise ValueError("Matching policy changed")
    windows = {w["id"]: w for w in packet["windows"]}
    if len(windows) != len(packet["windows"]):
        raise ValueError("Duplicate packet window")
    accounted = set()
    for entry in labels["positive_windows"]:
        w = windows[entry["window_id"]]
        if entry["polarity"] not in {"bright", "dark"} or not entry["motion_confirmed"]:
            raise ValueError("Positive must have reviewed motion and polarity")
        samples = entry["visible_samples"]
        visible = [s["frame_index"] for s in samples]
        unknown = entry["unknown_frames"]
        if (
            not samples
            or len(set(visible + unknown)) != len(visible) + len(unknown)
            or set(visible + unknown) != set(range(w["first"], w["last"] + 1))
        ):
            raise ValueError(
                "Every reviewed frame must be visible or explicitly unknown"
            )
        for s in samples:
            if (
                len(s["xy"]) != 2
                or not all(math.isfinite(v) for v in s["xy"])
                or not inside(s["xy"], w["crop_xywh"])
                or not math.isfinite(s["uncertainty_px"])
                or not 0 < s["uncertainty_px"] <= 8
            ):
                raise ValueError("Invalid source-coordinate annotation")
    for kind in ("positive_windows", "negative_windows", "unresolved_windows"):
        for entry in labels[kind]:
            wid = entry["window_id"]
            if wid not in windows or wid in accounted:
                raise ValueError("Duplicate/conflicting window classification")
            accounted.add(wid)
            if kind == "negative_windows":
                if (
                    not all(
                        entry.get(k) is True
                        for k in (
                            "every_frame_reviewed",
                            "native_pixels_reviewed",
                            "exhaustive_roi",
                            "target_definition_adjudicated",
                        )
                    )
                    or entry.get("status") != "verified_no_target"
                ):
                    raise ValueError(
                        "Negative requires explicit exhaustive adjudication"
                    )
    if accounted != set(windows):
        raise ValueError("Missing review disposition")
    # Overlapping positive/negative claims would invalidate absence scoring.
    for n in labels["negative_windows"]:
        nw = windows[n["window_id"]]
        for p in labels["positive_windows"]:
            pw = windows[p["window_id"]]
            if nw["clip_id"] == pw["clip_id"]:
                for s in p["visible_samples"]:
                    if nw["first"] <= s["frame_index"] <= nw["last"] and inside(
                        s["xy"], nw["crop_xywh"]
                    ):
                        raise ValueError("Positive annotation inside negative scope")
    return windows


def assign_one_to_one(samples, observations):
    """Maximum-cardinality gated matching; deterministic, distance-ordered edges.

    Not a minimum-total-distance or physical-identity oracle. Ambiguous gates
    are reported, and cannot support an identity-continuity claim.
    """
    neighbors = []
    for s in samples:
        neighbors.append(
            sorted(
                (
                    i
                    for i, o in enumerate(observations)
                    if o["polarity"] == s["polarity"]
                    and math.dist(s["xy"], o["xy"]) <= s["radius"]
                ),
                key=lambda i: (math.dist(s["xy"], observations[i]["xy"]), i),
            )
        )
    owner = {}

    def augment(index, seen):
        for j in neighbors[index]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = index
                return True
        return False

    for i in range(len(samples)):
        augment(i, set())
    return {i: j for j, i in owner.items()}, neighbors


def score_rows(rows, labels, packet, cid, fps):
    windows = validate_labels(labels, packet)
    positives = [
        p
        for p in labels["positive_windows"]
        if windows[p["window_id"]]["clip_id"] == cid
    ]
    negatives = [
        n
        for n in labels["negative_windows"]
        if windows[n["window_id"]]["clip_id"] == cid
    ]
    wanted = {}
    evidence = {p["window_id"]: [] for p in positives}
    for p in positives:
        for s in p["visible_samples"]:
            wanted.setdefault(s["frame_index"], []).append(
                dict(
                    **s,
                    window_id=p["window_id"],
                    polarity=p["polarity"],
                    radius=s["uncertainty_px"] + 2.0,
                )
            )
    negative_evidence = {n["window_id"]: [] for n in negatives}
    seen = set()
    for row in rows:
        f = row["frame_index"]
        if f in seen:
            raise ValueError("Duplicate frame")
        seen.add(f)
        tracks = []
        track_keys = set()
        for t in row["tracks"]:
            key = f'{t["segment"]}/{t["track_id"]}'
            if key in track_keys:
                raise ValueError("Duplicate track identity in frame")
            track_keys.add(key)
            if t["measured"] and t["qualified_moving"]:
                xy = t["measurement_source_xy"]
                if xy is None or not all(math.isfinite(v) for v in xy):
                    raise ValueError("Invalid measured coordinate")
                tracks.append(dict(id=key, xy=xy, polarity=t["track_id"].split(":")[0]))
        samples = wanted.get(f, [])
        candidates = [
            dict(xy=c["source_xy"], polarity=c["polarity"]) for c in row["candidates"]
        ]
        cm, _ = assign_one_to_one(samples, candidates)
        tm, edges = assign_one_to_one(samples, tracks)
        used = set(tm.values())
        for i, s in enumerate(samples):
            chosen = tracks[tm[i]] if i in tm else None
            shared = any(sum(j in e for e in edges) > 1 for j in edges[i])
            nearby_other = [
                t["id"]
                for j, t in enumerate(tracks)
                if j not in used and math.dist(t["xy"], s["xy"]) <= s["radius"]
            ]
            evidence[s["window_id"]].append(
                dict(
                    frame_index=f,
                    reference_xy=s["xy"],
                    match_radius_px=s["radius"],
                    candidate_hit=i in cm,
                    qualified_measured_hit=chosen is not None,
                    assigned_track_id=chosen["id"] if chosen else None,
                    all_gated_same_polarity_ids=[tracks[j]["id"] for j in edges[i]],
                    association_ambiguous=shared or len(edges[i]) > 1,
                    extra_nearby_qualified_response_ids=nearby_other,
                    localization_error_px=math.dist(chosen["xy"], s["xy"])
                    if chosen
                    else None,
                    detection_ready=row["coverage"].get(
                        "detection_ready", not row["coverage"]["warmup"]
                    ),
                    segment=row["segment"],
                )
            )
        for n in negatives:
            w = windows[n["window_id"]]
            if w["first"] <= f <= w["last"]:
                negative_evidence[w["id"]].append(
                    dict(
                        frame_index=f,
                        ids=[
                            t["id"] for t in tracks if inside(t["xy"], w["crop_xywh"])
                        ],
                        ready=row["coverage"].get(
                            "detection_ready", not row["coverage"]["warmup"]
                        ),
                    )
                )
    if not set(wanted) <= seen:
        raise ValueError(
            "Labeled frames outside completed run; never silently drop labels"
        )
    output = []
    for p in positives:
        ev = sorted(evidence[p["window_id"]], key=lambda e: e["frame_index"])
        counts = Counter(
            e["assigned_track_id"] for e in ev if e["qualified_measured_hit"]
        )
        ids = [e["assigned_track_id"] for e in ev]
        misses = [e["frame_index"] for e in ev if not e["qualified_measured_hit"]]
        dense = not p["unknown_frames"]
        all_unique = all(
            e["qualified_measured_hit"] and not e["association_ambiguous"] for e in ev
        )
        output.append(
            dict(
                window_id=p["window_id"],
                event_id=p["event_id"],
                visible_samples=len(ev),
                unknown_frames=len(p["unknown_frames"]),
                candidate_hits=sum(e["candidate_hit"] for e in ev),
                qualified_measured_hits=len(ev) - len(misses),
                missed_visible_frames=misses,
                observed_track_ids=sorted(counts),
                dominant_id_visible_fraction=max(counts.values(), default=0) / len(ev),
                continuous_same_id_in_reviewed_window=(len(set(ids)) == 1)
                if dense and all_unique
                else None,
                ambiguity_frames=sum(e["association_ambiguous"] for e in ev),
                extra_nearby_response_frames=sum(
                    bool(e["extra_nearby_qualified_response_ids"]) for e in ev
                ),
                unavailable_visible_frames=sum(not e["detection_ready"] for e in ev),
                onset_to_detection_seconds=None,
                latency_note="Event already present at selected window start; onset unknown. No detection-latency estimate.",
                evidence=ev,
            )
        )
    negative_results = []
    for n in negatives:
        w = windows[n["window_id"]]
        ev = negative_evidence[w["id"]]
        if len(ev) != w["last"] - w["first"] + 1:
            raise ValueError("Incomplete negative interval")
        negative_results.append(
            dict(
                window_id=w["id"],
                crop_xywh=w["crop_xywh"],
                exposure_roi_seconds=len(ev) / fps,
                measured_qualified_track_frames=sum(len(e["ids"]) for e in ev),
                distinct_track_ids=sorted({t for e in ev for t in e["ids"]}),
                unavailable_frames=sum(not e["ready"] for e in ev),
                scope="Explicitly reviewed ROI only; not extrapolated to full-frame false alarms/minute",
            )
        )
    return dict(
        clip_id=cid,
        positive_windows=output,
        verified_negative_windows=negative_results,
        unresolved_windows=[
            e["window_id"]
            for e in labels["unresolved_windows"]
            if windows[e["window_id"]]["clip_id"] == cid
        ],
        precision=None,
        recall=None,
        false_alarms_per_full_frame_minute=None,
        scoring_scope=labels.get("scoring_scope", "moving-feature regression only"),
        airborne_precision=None,
        airborne_recall=None,
        verified_airborne_windows=sum(
            p.get("airborne_target_verified") is True for p in positives
        ),
        caveat="Conditional visible-sample checks, not exhaustive recall. Nearby extra responses are not proven duplicates or false positives.",
    )


def read_verified_run(spec, source):
    path, freeze_path = Path(spec["run"]), Path(spec["freeze"])
    freeze = json.loads(freeze_path.read_text())
    launch = json.loads((path / "launch.json").read_text())
    report = json.loads((path / "report.json").read_text())
    package = {
        k.removeprefix("tiny_target/"): v
        for k, v in freeze["files_sha256"].items()
        if k.startswith("tiny_target/")
    }
    if launch["package_sha256"] != package:
        raise ValueError("Run package differs from frozen implementation")
    for name, value in package.items():
        if digest(path / "implementation" / name) != value:
            raise ValueError("Implementation snapshot mismatch")
    config_name = (
        "configs/evaluation/phase20_visible_v7"
        + ("_pva" if spec["backend"] == "pva" else "")
        + ".json"
    )
    if launch["config_sha256"] != freeze["files_sha256"][config_name]:
        raise ValueError("Visible configuration mismatch")
    if (
        spec["backend"] == "pva"
        and launch["motion_config_sha256"]
        != freeze["files_sha256"]["configs/evaluation/phase20_motion_v8.json"]
    ):
        raise ValueError("Motion configuration mismatch")
    count = source["frames"] if spec["full_clip"] else spec["max_frames"]
    if (
        not report["completed"]
        or report["frames"] != count
        or report["full_clip"] != spec["full_clip"]
        or launch["expected_frames"] != source["frames"]
        or launch["max_frames"] != spec.get("max_frames")
        or launch["source_sha256"] != source["sha256"]
        or report["source_sha256"] != source["sha256"]
        or launch["annotations_supplied_to_detector"]
        or launch["configuration"]["motion_backend"] != spec["backend"]
        or abs(launch["fps"] - source["fps"]) > 1e-6
    ):
        raise ValueError("Run/source/backend/completion scope mismatch")

    def rows():
        seen = 0
        with (path / "frames.jsonl").open() as f:
            for row in map(json.loads, f):
                if row["frame_index"] != seen:
                    raise ValueError("Non-contiguous journal")
                if spec["backend"] == "pva" and seen:
                    backends = row["motion"]["motion_backends"]
                    if backends.get("cpu_fallback") is not False or any(
                        backends.get(k) != "PVA"
                        for k in ("gaussian_pyramid", "harris", "optical_flow_pyrlk")
                    ):
                        raise ValueError("PVA hardware evidence missing")
                seen += 1
                yield row
        if seen != count:
            raise ValueError("Incomplete journal")

    provenance = dict(
        **spec,
        source_sha256=source["sha256"],
        freeze_sha256=digest(freeze_path),
        artifacts_sha256={
            n: digest(path / n) for n in ("launch.json", "report.json", "frames.jsonl")
        },
        processed_frames=count,
    )
    return rows(), provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "labels", "runs", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    packet, labels, specs = [
        json.loads(v.read_text()) for v in (a.packet, a.labels, a.runs)
    ]
    if (
        digest(a.packet) != specs["packet_sha256"]
        or digest(a.labels) != specs["labels_sha256"]
    ):
        raise ValueError("Labels/packet changed after scoring freeze")
    sources = {s["clip_id"]: s for s in packet["plan"]["sources"]}
    results = []
    for spec in specs["runs"]:
        s = sources[spec["clip_id"]]
        rows, provenance = read_verified_run(spec, s)
        result = score_rows(rows, labels, packet, s["clip_id"], s["fps"])
        results.append(dict(**result, provenance=provenance))
    result = dict(
        schema="seaqr.bounded-accuracy-baseline.v1",
        runs=results,
        labels_sha256=digest(a.labels),
        packet_sha256=digest(a.packet),
        runs_manifest_sha256=digest(a.runs),
        scorer_sha256=digest(__file__),
        status="Development pilot; not a complete accuracy baseline or permission to optimize",
        generalization_proven=False,
        precision=None,
        recall=None,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [
                dict(
                    clip=r["clip_id"],
                    backend=r["provenance"]["backend"],
                    windows=[
                        {
                            k: w[k]
                            for k in (
                                "window_id",
                                "visible_samples",
                                "candidate_hits",
                                "qualified_measured_hits",
                                "missed_visible_frames",
                                "continuous_same_id_in_reviewed_window",
                            )
                        }
                        for w in r["positive_windows"]
                    ],
                )
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
