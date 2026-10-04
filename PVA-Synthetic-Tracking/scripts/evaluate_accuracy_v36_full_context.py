"""Full-development-clip source-context shadow gate; never reruns tracking.

Only the four frozen 8-bit sources are supported. Labels are used for scoring
after gate decisions, never supplied to the feature function or causal gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_accuracy_v36 import (AUDIT, REFERENCES, compare_samples, inside,
                                  load_reference, read, score_rows, sha,
                                  verified_inputs, write)

EXPERIMENT = ROOT / "results/tiny_target/accuracy_v36_20260924"
CONTEXT = EXPERIMENT / "context_01"
SHADOW = EXPERIMENT / "shadow_01"
COUNTS = {"0029": 687, "0126": 674, "0055": 689, "0082": 691}
SOURCES = {
    "0029": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0029.avi",
    "0126": ROOT.parent / "outputs/jetson_review_clips_20260913/chunk_0126.avi",
    "0055": ROOT.parent / "outputs/v7_frozen_evaluation_20260913/sources/chunk_0055.avi",
    "0082": ROOT.parent / "outputs/v7_frozen_evaluation_20260913/sources/chunk_0082.avi",
}
ARMS = ("baseline", "point_context")
IMPLEMENTATION = (
    "scripts/accuracy_v36_context.py",
    "scripts/accuracy_v36_context_gate.py",
    "scripts/evaluate_accuracy_v36_full_context.py",
    "scripts/evaluate_accuracy_v36.py",
    "scripts/accuracy_v36_policy.py",
    "scripts/score_phase20_accuracy.py",
    "tiny_target/visible_regression.py",
    "tests/unit/test_accuracy_v36_context.py",
    "tests/unit/test_accuracy_v36_context_runner.py",
    "tests/unit/test_accuracy_v36_context_gate.py",
    "tests/unit/test_accuracy_v36_full_context.py",
    "docs/accuracy_v36_full_context_plan.md",
)
CAVEAT = (
    "Output-only shadow eligibility on four development clips; original "
    "detection, association, state and learning feedback are unchanged. "
    "Unlabeled workload and selected provisional controls are not false-positive "
    "rates. Known moving image features have unknown physical class."
)


def bind(files, path, expected=None):
    path = str(Path(path).resolve())
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("SHA256 mismatch: " + path)
    previous = files.setdefault(path, digest)
    if previous != digest:
        raise ValueError("Input changed during preflight: " + path)
    return digest


def recheck(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Bound file changed: " + path)


def verify_context_audit(audit_path):
    """Fail closed unless an independent audit covers the exact context run."""
    audit_path = Path(audit_path).resolve()
    audit = read(audit_path)
    freeze = read(CONTEXT / "freeze.json")
    summary = read(CONTEXT / "summary.json")
    if (audit.get("schema") != "seaqr.accuracy-v36-context-independent-audit.v1"
            or audit.get("verified") is not True
            or audit.get("experiment") != str(CONTEXT)
            or audit.get("context_freeze_sha256") != sha(CONTEXT / "freeze.json")
            or audit.get("complete_summary_verified") is not True
            or audit.get("zero_margin_any_alternative_decisions_match") is not True
            or summary.get("completed") is not True
            or summary.get("freeze_sha256") != sha(CONTEXT / "freeze.json")
            or freeze.get("pre_extraction") is not True
            or freeze.get("classifier_promoted") is not False):
        raise ValueError("Completed independently audited context_01 required")
    checked = audit.get("checked_files_sha256")
    if not isinstance(checked, dict) or not checked:
        raise ValueError("Independent context audit lacks bound files")
    required = {str(CONTEXT / name) for name in (
        "freeze.json", "summary.json", "selection.json", "observations.json",
        "native_patches.npz", "unit.log")}
    required.update(freeze["inputs_sha256"])
    for name, digest in freeze["implementation_sha256"].items():
        required.update((str(ROOT / name), str(CONTEXT / "implementation" / name)))
        if checked.get(str(ROOT / name)) != digest:
            raise ValueError("Context implementation not independently bound: " + name)
    if not required <= checked.keys():
        raise ValueError("Independent context audit omits required evidence")
    if audit.get("auditor_sha256") != sha(ROOT / "scripts/audit_accuracy_v36_context.py"):
        raise ValueError("Context auditor source changed")
    files = {}
    for path, digest in checked.items():
        bind(files, path, digest)
    bind(files, audit_path)
    bind(files, CONTEXT / "selection.json", freeze["selection_sha256"])
    bind(files, CONTEXT / "unit.log", freeze["unit_log_sha256"])
    for name, digest in summary["outputs_sha256"].items():
        bind(files, CONTEXT / name, digest)
    for path, digest in freeze["inputs_sha256"].items():
        bind(files, path, digest)
    return files, freeze


def validate_decisions(row, decisions):
    """Verify the gate cannot add tracks, alter measurements, or use the future."""
    if not isinstance(decisions, dict):
        raise ValueError("Gate must return an identity-keyed dictionary")
    tracks = {(t["segment"], t["track_id"]): t for t in row["tracks"]}
    if len(tracks) != len(row["tracks"]) or not decisions.keys() <= tracks.keys():
        raise ValueError("Duplicate or unknown gate identity")
    relevant = {key for key, t in tracks.items() if t["qualified_moving"]}
    if not relevant <= decisions.keys():
        raise ValueError("Missing baseline-qualified gate decision")
    for key, decision in decisions.items():
        if (not isinstance(decision, dict)
                or not {"accepted", "reason", "measurement_frame", "features"} <= decision.keys()
                or type(decision.get("accepted")) is not bool
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"]):
            raise ValueError("Malformed gate decision")
        track = tracks[key]
        if decision["accepted"] and not track["qualified_moving"]:
            raise ValueError("Gate added an unqualified output")
        frame = decision.get("measurement_frame")
        if track["qualified_moving"] and track["measured"] and frame != row["frame_index"]:
            raise ValueError("Qualified measurement requires current-frame provenance")
        if frame is not None:
            if type(frame) is not int or not 0 <= frame <= row["frame_index"]:
                raise ValueError("Invalid or future measurement provenance")
            if not track["measured"] and frame >= row["frame_index"]:
                raise ValueError("Prediction cannot borrow a current measurement")
        features = decision.get("features")
        if features is not None and (not isinstance(features, dict) or frame is None):
            raise ValueError("Features require measurement provenance")
        json.dumps(decision, allow_nan=False)
    return [tracks[key] for key in tracks if key in relevant]


def reference_spec(cid):
    references = {name: (read(REFERENCES[name + "_labels"]),
                         read(REFERENCES[name + "_packet"]))
                  for name in ("dense", "pilot")}
    frames = set()
    for labels, packet in references.values():
        windows = {window["id"]: window for window in packet["windows"]}
        for window in labels["positive_windows"]:
            if windows[window["window_id"]]["clip_id"] == cid:
                frames.update(s["frame_index"] for s in window["visible_samples"])
    anchors = []
    if cid in ("0029", "0126"):
        _, events = load_reference(REFERENCES["anchors_" + cid])
        anchors = [dict(event_id=e["event_id"], polarity=e["polarity"], **anchor)
                   for e in events for anchor in e["anchors"] if anchor["required"]]
        frames.update(anchor["frame_index"] for anchor in anchors)
    controls = read(REFERENCES["controls"])["controls"] if cid == "0126" else []
    return references, frames, anchors, controls


def new_stats(controls):
    return {arm: dict(measured=0, predicted=0, ids=set(), per_frame=[],
                     controls=[dict(measured=0, predicted=0, ids=set())
                               for _ in controls]) for arm in ARMS}


def accumulate(row, tracks, stats, controls):
    measured = sum(t["measured"] for t in tracks)
    stats["measured"] += measured
    stats["predicted"] += len(tracks) - measured
    stats["ids"].update((t["segment"], t["track_id"]) for t in tracks)
    stats["per_frame"].append(dict(frame_index=row["frame_index"], measured=measured,
                                   predicted=len(tracks) - measured))
    for control, counts in zip(controls, stats["controls"]):
        first, last = control["frames_inclusive"]
        if not first <= row["frame_index"] <= last:
            continue
        local = [t for t in tracks if inside(t["measurement_source_xy"] if t["measured"]
                                            else t["source_xy"], control["crop_xywh"])]
        m = sum(t["measured"] for t in local)
        counts["measured"] += m
        counts["predicted"] += len(local) - m
        counts["ids"].update((t["segment"], t["track_id"]) for t in local)


def sparse_row(row, tracks):
    return dict(frame_index=row["frame_index"], segment=row["segment"],
                coverage=row["coverage"], candidates=[dict(source_xy=p["source_xy"],
                polarity=p["polarity"]) for p in row["candidates"]], tracks=[
                    {k: t[k] for k in ("track_id", "segment", "measured",
                                      "qualified_moving", "measurement_source_xy")}
                    for t in tracks])


def matched_anchors(row, tracks, anchors):
    return [dict(event_id=anchor["event_id"], frame=anchor["frame_index"], ids=[
        f'{t["segment"]}/{t["track_id"]}' for t in tracks if t["measured"]
        and t["track_id"].split(":")[0] == anchor["polarity"]
        and math.dist(t["measurement_source_xy"], anchor["xy"]) <= anchor["uncertainty_px"] + 2])
        for anchor in anchors if anchor["frame_index"] == row["frame_index"]]


def summarize_clip(cid, frames, stats, sparse_rows, anchors_evidence,
                   availability, decision_sha256):
    references, _, _, controls = reference_spec(cid)
    scores = {arm: {name: score_rows(sparse_rows[arm], labels, packet, cid, 10)
                    for name, (labels, packet) in references.items()} for arm in ARMS}
    result = {}
    for arm in ARMS:
        data = stats[arm]
        control_rows = [dict(frames_inclusive=control["frames_inclusive"],
            crop_xywh=control["crop_xywh"], label=control["label"],
            measured_states=counts["measured"], predicted_states=counts["predicted"],
            distinct_segment_track_ids=len(counts["ids"]))
            for control, counts in zip(controls, data["controls"])]
        comparisons = {name: compare_samples(scores["baseline"][name], scores[arm][name])
                       for name in references}
        old_anchors = anchors_evidence["baseline"]
        new_anchors = anchors_evidence[arm]
        if [(a["event_id"], a["frame"]) for a in old_anchors] != [
                (a["event_id"], a["frame"]) for a in new_anchors]:
            raise ValueError("Changed required anchor denominator")
        lost_anchors = [dict(event_id=a["event_id"], frame=a["frame"])
                        for a, b in zip(old_anchors, new_anchors) if a["ids"] and not b["ids"]]
        intersections = {}
        for event in sorted({a["event_id"] for a in new_anchors}):
            ids = [set(a["ids"]) for a in new_anchors if a["event_id"] == event]
            intersections[event] = sorted(set.intersection(*ids))
        result[arm] = dict(qualified_measured_states=data["measured"],
            qualified_predicted_states=data["predicted"], distinct_segment_track_ids=len(data["ids"]),
            qualified_states_per_frame=(data["measured"] + data["predicted"]) / frames,
            maximum_qualified_states_in_frame=max(x["measured"] + x["predicted"] for x in data["per_frame"]),
            provisional_controls=control_rows,
            provisional_control_measured_states=sum(x["measured_states"] for x in control_rows),
            provisional_control_predicted_states=sum(x["predicted_states"] for x in control_rows),
            references=scores[arm], retention=comparisons, required_anchor_evidence=new_anchors,
            lost_required_anchors=lost_anchors, common_id_across_required_anchors=intersections,
            known_reference_retention_passed=(all(x["no_new_misses"] and not x["changed_assignments"]
                for x in comparisons.values()) and not lost_anchors and all(intersections.values())))
    return dict(clip=cid, frames=frames, arms=result, availability=availability,
                decisions_sha256=decision_sha256, airborne_precision=None, airborne_recall=None,
                false_alarms_per_minute=None, detector_rerun=False, feedback_changed=False)


def check_bounded_observation(expected, row, gray, decisions):
    """Link a previously audited saved patch to this sequential source decode."""
    identity = expected["identity"]
    segment, track_id = identity.split("/", 1)
    key = (int(segment), track_id)
    tracks = [t for t in row["tracks"] if (t["segment"], t["track_id"]) == key]
    if (len(tracks) != 1 or not tracks[0]["measured"] or not tracks[0]["qualified_moving"]
            or tracks[0]["measurement_source_xy"] != expected["measurement_source_xy"]):
        raise ValueError("Bounded observation identity or measurement changed")
    xy = expected["measurement_source_xy"]
    x, y = (math.floor(v + .5) for v in xy)
    patch = np.ascontiguousarray(gray[y-12:y+13, x-12:x+13])
    if (patch.shape != (25, 25) or patch.dtype != np.uint8
            or hashlib.sha256(patch.tobytes()).hexdigest() != expected["patch_sha256"]):
        raise ValueError("Sequential source decode differs from audited bounded patch")
    decision = decisions[key]
    expected_acceptance = (not expected["features"]["informative"]
                           or expected["zero_margin_ablation_passed"])
    if decision["accepted"] != expected_acceptance:
        raise ValueError("Bounded gate decision or conservative unknown policy changed")
    actual = decision["features"]
    if not isinstance(actual, dict) or actual.keys() != expected["features"].keys():
        raise ValueError("Bounded feature inventory changed")
    for name, value in expected["features"].items():
        observed = actual[name]
        if isinstance(value, bool):
            equal = type(observed) is bool and observed == value
        else:
            # Same validation tolerances as the independent direct-LS audit;
            # they are numerical checks, never eligibility thresholds.
            equal = np.allclose(observed, value, rtol=2e-10, atol=2e-8)
        if not equal:
            raise ValueError("Bounded feature value changed: " + name)


def analyze_clip(cid, spec, output, gate_factory, capture_factory=None,
                 known_observations=()):
    """Full sequential decode with no skip/seek; injectable factories aid tests."""
    if cid not in COUNTS or spec["frames"] != COUNTS[cid]:
        raise ValueError("Only frozen four-clip full runs are supported")
    _, required, anchors, controls = reference_spec(cid)
    stats = new_stats(controls)
    sparse = {arm: [] for arm in ARMS}
    anchor_evidence = {arm: [] for arm in ARMS}
    availability = Counter()
    reason_counts = {"measured": Counter(), "predicted": Counter()}
    known = {}
    for observation in known_observations:
        if observation["clip"] != cid:
            raise ValueError("Wrong clip in bounded crosscheck")
        known.setdefault(observation["frame"], []).append(observation)
    known_checked = 0
    gate = gate_factory()
    capture = (capture_factory or cv2.VideoCapture)(str(SOURCES[cid]))
    frame_count = 0
    decisions_path = output / (cid + "_decisions.jsonl")
    workload_path = output / (cid + "_workload.jsonl")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if (not capture.isOpened() or not math.isfinite(fps) or abs(fps - 10) > 1e-6
                or (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (4784, 3190)
                or capture.get(cv2.CAP_PROP_FRAME_COUNT) != spec["frames"]):
            raise ValueError("Unexpected source decoder metadata")
        decoder = dict(backend=capture.getBackendName(), fps=10, width=4784, height=3190,
                       reported_frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        with (Path(spec["path"]) / "frames.jsonl").open() as source, decisions_path.open("x") as log:
            for line in source:
                row = json.loads(line)
                if row["frame_index"] != frame_count or row["timestamp_ns"] != frame_count * 100000000:
                    raise ValueError("Noncontiguous frozen journal")
                if frame_count >= spec["frames"]:
                    raise ValueError("Extra journal frame")
                ok, bgr = capture.read()
                if not ok or bgr is None or bgr.shape != (3190, 4784, 3) or bgr.dtype != np.uint8:
                    raise ValueError("Unexpected source frame/decode failure")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                original = json.dumps(row, sort_keys=True, allow_nan=False)
                decisions = gate.update(row, gray)
                if original != json.dumps(row, sort_keys=True, allow_nan=False):
                    raise ValueError("Shadow gate mutated the original journal")
                relevant = validate_decisions(row, decisions)
                for expected in known.get(frame_count, []):
                    check_bounded_observation(expected, row, gray, decisions)
                    known_checked += 1
                compact = []
                for track in relevant:
                    decision = decisions[track["segment"], track["track_id"]]
                    state = "measured" if track["measured"] else "predicted"
                    reason_counts[state][decision["reason"]] += 1
                    features = decision["features"]
                    if not track["measured"]:
                        status = ("inherited_decision" if decision["measurement_frame"] is not None
                                  else "missing_history")
                    else:
                        status = "missing" if features is None else (
                            "informative" if features["informative"] else "uninformative")
                    availability[state + "_" + status] += 1
                    compact.append(dict(track_id=track["track_id"], segment=track["segment"],
                        measured=track["measured"], source_xy=track["source_xy"],
                        measurement_source_xy=track["measurement_source_xy"], **decision))
                log.write(json.dumps(dict(frame_index=frame_count, timestamp_ns=row["timestamp_ns"],
                    segment=row["segment"], tracks=compact), allow_nan=False) + "\n")
                for arm in ARMS:
                    accepted = relevant if arm == "baseline" else [t for t in relevant
                        if decisions[t["segment"], t["track_id"]]["accepted"]]
                    accumulate(row, accepted, stats[arm], controls)
                    if frame_count in required:
                        sparse[arm].append(sparse_row(row, accepted))
                    anchor_evidence[arm].extend(matched_anchors(row, accepted, anchors))
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"{cid}: {frame_count}/{spec['frames']} frames", flush=True)
            if frame_count != spec["frames"]:
                raise ValueError("Incomplete frozen journal")
            if known_checked != len(known_observations):
                raise ValueError("Incomplete bounded-source crosscheck")
            more, extra = capture.read()
            if more or extra is not None:
                raise ValueError("Video has extra frames after frozen journal EOF")
    finally:
        capture.release()
    with workload_path.open("x") as workload:
        for index in range(frame_count):
            workload.write(json.dumps(dict(frame_index=index,
                arms={arm: stats[arm]["per_frame"][index] for arm in ARMS})) + "\n")
    availability_out = dict(state_evidence_counts=dict(availability),
                            reasons={kind: dict(counts) for kind, counts in reason_counts.items()})
    result = summarize_clip(cid, frame_count, stats, sparse, anchor_evidence,
                            availability_out, sha(decisions_path))
    result.update(source_sha256=spec["source_sha256"], decoder=decoder,
                  decoded_frames=frame_count, eof_checked=True, workload_sha256=sha(workload_path),
                  bounded_context_observations_crosschecked=known_checked)
    # The complete scored baseline must be identical to the audited shadow run.
    prior = read(SHADOW / (cid + "_results.json"))["arms"]["baseline"]
    if result["arms"]["baseline"] != prior:
        raise ValueError("Full-context baseline differs from audited shadow baseline")
    return result


def global_reference_totals(results):
    totals = {}
    for arm in ARMS:
        totals[arm] = {}
        for name in ("dense", "pilot"):
            windows = [w for result in results.values()
                       for w in result["arms"][arm]["references"][name]["positive_windows"]]
            totals[arm][name] = dict(samples=sum(w["visible_samples"] for w in windows),
                hits=sum(w["qualified_measured_hits"] for w in windows))
        anchors = [a for result in results.values()
                   for a in result["arms"][arm]["required_anchor_evidence"]]
        totals[arm]["anchors"] = dict(samples=len(anchors), hits=sum(bool(a["ids"]) for a in anchors))
    if totals["baseline"] != {"dense": {"samples": 285, "hits": 284},
                               "pilot": {"samples": 28, "hits": 28},
                               "anchors": {"samples": 24, "hits": 24}}:
        raise ValueError("Frozen baseline reference totals changed")
    return totals


def run(output, context_audit):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh full-context output directory required")
    # Read-only evidence verification happens before any source-video access.
    files, context_freeze = verify_context_audit(context_audit)
    inputs = verified_inputs()
    if {cid: spec["frames"] for cid, spec in inputs.items()} != COUNTS:
        raise ValueError("Frozen four-clip scope changed")
    for cid, spec in inputs.items():
        for path, digest in spec["files_sha256"].items():
            bind(files, path, digest)
        bind(files, SHADOW / (cid + "_results.json"))
    for path in REFERENCES.values():
        bind(files, path)
    source_digests = {cid: sha(path) for cid, path in SOURCES.items()}
    if any(source_digests[cid] != inputs[cid]["source_sha256"] for cid in COUNTS):
        raise ValueError("Source-video identity mismatch")
    feature_path = "scripts/accuracy_v36_context.py"
    if sha(ROOT / feature_path) != context_freeze["implementation_sha256"][feature_path]:
        raise ValueError("Frozen diagnostic feature implementation changed")
    output.mkdir(parents=True)
    implementation = {path: sha(ROOT / path) for path in IMPLEMENTATION}
    with (output / "unit.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/unit",
            "-p", "test_accuracy_v36*.py", "-v"], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=True)
    freeze = dict(schema="seaqr.accuracy-v36-full-context-freeze.v1", pre_extraction=True,
        created_ns=time.time_ns(), inputs=inputs, inputs_sha256=files,
        source_videos={cid: dict(path=str(SOURCES[cid]), sha256=digest)
                       for cid, digest in source_digests.items()},
        implementation_sha256=implementation, unit_log_sha256=sha(output / "unit.log"),
        context_audit_sha256=sha(context_audit), policy="informative point-minus-edge > 0; unknown conservatively keeps baseline; coasts inherit latest measured decision",
        frames=2741, clips=list(COUNTS), arms=list(ARMS), patch_size=25,
        opencv_version=cv2.__version__, numpy_version=np.__version__,
        opencv_build_information_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        detector_rerun=False, feedback_changed=False, raw16_accessed=False,
        remote_state_accessed=False, classifier_promoted=False, thresholds_tuned=False)
    write(output / "freeze.json", freeze)
    snapshot_hashes = {}
    bind(snapshot_hashes, output / "freeze.json")
    for path, digest in implementation.items():
        target = output / "implementation" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
        bind(snapshot_hashes, target, digest)
        if sha(ROOT / path) != digest:
            raise ValueError("Implementation changed before decode")
    # Import after tests and the freeze; no parent trial modules execute media.
    from accuracy_v36_context_gate import CausalPointContextGate
    results = {}
    known_observations = read(CONTEXT / "observations.json")
    if len(known_observations) != 358:
        raise ValueError("Changed bounded observation count")
    for cid in COUNTS:
        print(f"Full native source shadow: {cid}, {COUNTS[cid]} frames", flush=True)
        selected_known = [o for o in known_observations if o["clip"] == cid]
        result = analyze_clip(cid, inputs[cid], output, CausalPointContextGate,
                              known_observations=selected_known)
        results[cid] = result
        write(output / (cid + "_results.json"), result)
    if sum(r["frames"] for r in results.values()) != 2741:
        raise ValueError("Incomplete full-context experiment")
    totals = global_reference_totals(results)
    retention = all(r["arms"]["point_context"]["known_reference_retention_passed"]
                    for r in results.values())
    before = results["0126"]["arms"]["baseline"]["provisional_control_measured_states"]
    after = results["0126"]["arms"]["point_context"]["provisional_control_measured_states"]
    if before != 70:
        raise ValueError("Baseline control workload changed")
    recheck(files)
    recheck(snapshot_hashes)
    for path, digest in implementation.items():
        if sha(ROOT / path) != digest:
            raise ValueError("Implementation changed during decode")
    for cid, digest in source_digests.items():
        if sha(SOURCES[cid]) != digest:
            raise ValueError("Source video changed during decode")
    if sha(output / "unit.log") != freeze["unit_log_sha256"]:
        raise ValueError("Unit log changed")
    output_hashes = {cid + suffix: sha(output / (cid + suffix)) for cid in COUNTS
                     for suffix in ("_decisions.jsonl", "_results.json", "_workload.jsonl")}
    summary = dict(schema="seaqr.accuracy-v36-full-context-summary.v1", completed=True,
        freeze_sha256=sha(output / "freeze.json"), outputs_sha256=output_hashes,
        frames=2741, reference_totals=totals,
        clips={cid: {key: value for key, value in result.items() if key != "arms"}
               | {"arms": {arm: {k: v for k, v in values.items()
                        if k not in ("references", "required_anchor_evidence")}
                           for arm, values in result["arms"].items()}}
               for cid, result in results.items()},
        known_reference_retention_passed=retention,
        provisional_control_measured_before=before, provisional_control_measured_after=after,
        eligible_for_further_study=retention and after < before, promoted=False,
        detector_rerun=False, feedback_changed=False, defaults_changed=False,
        thresholds_tuned=False, raw16_accessed=False, remote_state_accessed=False,
        airborne_precision=None, airborne_recall=None, false_alarms_per_minute=None,
        accuracy_established=False, performance_benchmark=False, caveat=CAVEAT)
    write(output / "summary.json", summary)
    print(json.dumps(dict(completed=True, frames=2741, reference_totals=totals,
        known_reference_retention_passed=retention, control_before=before,
        control_after=after, promoted=False), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-audit", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    run(args.output, args.context_audit.resolve())
