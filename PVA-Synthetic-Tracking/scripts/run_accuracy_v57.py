"""Metadata-only frozen V57 shadow replay. Never opens camera media."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil

from accuracy_v57_persistence import CausalEdgePersistence, PersistenceConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results/tiny_target/"
V36 = RESULTS + "accuracy_v36_20260924/"
TRACE = RESULTS + "accuracy_v55_20260926/trace_01/trace.json"
GRID = RESULTS + "accuracy_v40_20260925/coverage_workload_01/visible_reference_evidence.json"
CONTROLS = "configs/evaluation/accuracy_v56_diagnostic_probes.json"
GUARD = RESULTS + "accuracy_v56_20260926/compact_01/frame216_track_continuity.json"
CONFIG = "configs/evaluation/accuracy_v57_persistence.json"
COUNTS = {"0029": 687, "0126": 674, "0055": 689, "0082": 691}
PANELS = {"dense": 285, "pilot": 28, "anchor": 24, "compact_light": 8, "grid": 11}
IMPLEMENTATION = [
    "scripts/accuracy_v57_persistence.py", "scripts/run_accuracy_v57.py",
    "scripts/audit_accuracy_v57.py", "scripts/stress_accuracy_v57.py",
    "scripts/accuracy_v36_context.py", "tests/unit/test_accuracy_v36_context.py",
    "docs/accuracy_v57_plan.md", CONFIG,
    "tests/unit/test_accuracy_v57_persistence.py", "tests/unit/test_accuracy_v57_runner.py",
    "tests/unit/test_accuracy_v57_audit.py", "tests/unit/test_accuracy_v57_stress.py",
]
ARMS = ("baseline", "v36", "v57")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "Duplicate JSON key: " + k)
        result[k] = v
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON: " + value)


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite JSON number: " + value)
    return result


def decode(text):
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant,
                      parse_float=finite_float)


def read(path):
    return decode(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "Blank journal row")
            yield decode(line)


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def same(a, b, message):
    require(encoded(a) == encoded(b), message)


def sha(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), "Regular file required")
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity(track):
    return f'{track["segment"]}/{track["track_id"]}'


def join_features(row, saved):
    for k in ("frame_index", "timestamp_ns", "segment"):
        same(row[k], saved[k], "Saved feature frame mismatch: " + k)
    qualified = [t for t in row["tracks"] if t["qualified_moving"]]
    require(len(qualified) == len(saved["tracks"]), "Saved qualified inventory mismatch")
    evidence = {}
    for t, d in zip(qualified, saved["tracks"]):
        for k in ("track_id", "segment", "measured", "source_xy", "measurement_source_xy"):
            same(t[k], d[k], "Saved feature identity/coordinate mismatch: " + k)
        require(type(d["accepted"]) is bool, "Saved verdict must be boolean")
        if t["measured"]:
            same(d["measurement_frame"], row["frame_index"], "Noncurrent measured feature")
            key = (t["segment"], t["track_id"])
            require(key not in evidence, "Duplicate feature identity")
            evidence[key] = {k: d[k] for k in ("features", "reason")}
        else:
            require(d["features"] is None, "Prediction supplied current features")
            origin = d["measurement_frame"]
            require(origin is None or type(origin) is int and 0 <= origin < row["frame_index"],
                    "Invalid prediction feature provenance")
    return qualified, evidence


def stage(value):
    if isinstance(value, list):
        return dict(hit=value[0], assigned_id=value[1], all_gated_ids=value[2])
    return {k: value[k] for k in ("hit", "assigned_id", "all_gated_ids")}


def reference_key(sample):
    return tuple(sample[k] for k in ("panel", "clip_id", "window_id", "frame_index"))


def reference_samples(trace, grid):
    result = []
    for entry in trace["reference_summary"]["records"]:
        source = entry["original"]
        item = {k: source[k] for k in ("panel", "clip_id", "window_id", "frame_index",
                                       "source_xy", "position_uncertainty_px", "polarity")}
        stages = source["stages"]
        strict_name = "baseline_qualified" if "baseline_qualified" in stages else "strict_qualified_measurement"
        item["actual"] = stage(stages["actual_measurement"])
        item["baseline"] = stage(stages[strict_name])
        same(item["baseline"]["assigned_id"], entry["original_strict_assigned_identity"],
             "Changed saved reference assignment")
        result.append(item)
    by_key = {reference_key(x): x for x in result}
    require(len(by_key) == len(result), "Duplicate reference sample")
    for source in grid:
        item = dict(panel="grid", **{k: source[k] for k in ("clip_id", "window_id", "frame_index",
                    "source_xy", "position_uncertainty_px", "polarity")})
        item["actual"] = stage(source["stages"]["actual_measurement"])
        item["baseline"] = stage(source["stages"]["strict_qualified_measurement"])
        key = reference_key(item)
        if key in by_key:
            same(item, by_key[key], "Overlapping grid references changed")
        else:
            result.append(item)
            by_key[key] = item
    require(Counter(x["panel"] for x in result) == PANELS, "Changed panel denominators")
    return result


def score_reference(sample, row, verdicts):
    """Never reassign a reference after applying the candidate filter."""
    gate = sample["position_uncertainty_px"] + 2
    actual = [t for t in row["tracks"] if t["measured"]
              and t["track_id"].split(":")[0] == sample["polarity"]
              and math.dist(t["measurement_source_xy"], sample["source_xy"]) <= gate]
    actual_ids = {identity(t) for t in actual}
    baseline_ids = {identity(t) for t in actual if t["qualified_moving"]}
    for name, ids in (("actual", actual_ids), ("baseline", baseline_ids)):
        saved = sample[name]
        require(ids == set(saved["all_gated_ids"]), "Reference gated inventory changed")
        require(saved["hit"] == (saved["assigned_id"] is not None), "Malformed reference hit")
        require(saved["assigned_id"] is None or saved["assigned_id"] in ids,
                "Reference assignment outside original gate")
    assigned = sample["baseline"]["assigned_id"]
    arms = {}
    for arm, decisions in verdicts.items():
        accepted = baseline_ids if arm == "baseline" else {
            i for i in baseline_ids if decisions[i]["accepted"]}
        arms[arm] = dict(original_assignment_retained=assigned in accepted,
                         any_qualified_alternative_retained=bool(accepted),
                         retained_ids=sorted(accepted),
                         lost_original_assignment=assigned is not None and assigned not in accepted)
    return dict(sample=sample, gate_px=gate, original_assignment=assigned, arms=arms,
                physical_class="unknown", airborne_truth=False)


def inside(xy, crop):
    x, y, w, h = crop
    return x <= xy[0] < x + w and y <= xy[1] < y + h


def empty_stats():
    return {a: dict(measured=0, predicted=0, ids=set()) for a in ARMS}


def accumulate(stats, qualified, old, new, predicate=lambda _: True):
    for t, d0, d1 in zip(qualified, old, new):
        if not predicate(t):
            continue
        for arm, accepted in (("baseline", True), ("v36", d0["accepted"]), ("v57", d1["accepted"])):
            if accepted:
                stats[arm]["measured" if t["measured"] else "predicted"] += 1
                stats[arm]["ids"].add(identity(t))


def finished_stats(stats):
    return {arm: dict(measured=s["measured"], predicted=s["predicted"],
                      distinct_identities=len(s["ids"])) for arm, s in stats.items()}


def replay_clip(journal, features, output, expected_frames, config, refs, controls, guard=None):
    gate = CausalEdgePersistence(config)
    stats, control_stats = empty_stats(), [empty_stats() for _ in controls]
    tiers, reasons = Counter(), Counter()
    by_frame = defaultdict(list)
    for ref in refs:
        by_frame[ref["frame_index"]].append(ref)
    scores, continuity, frames = [], None, 0
    with Path(output).open("x") as stream:
        for row, saved in zip_longest(lines(journal), lines(features)):
            require(row is not None and saved is not None, "Mismatched journal lengths")
            qualified, evidence = join_features(row, saved)
            decisions = gate.update(row, evidence)
            require(len(decisions) == len(qualified), "Candidate changed output inventory")
            logged = {k: row[k] for k in ("frame_index", "timestamp_ns", "segment")}
            logged["tracks"] = decisions
            stream.write(encoded(logged) + "\n")
            accumulate(stats, qualified, saved["tracks"], decisions)
            for control, counts in zip(controls, control_stats):
                a, b = control["frames_inclusive"]
                if a <= row["frame_index"] <= b:
                    accumulate(counts, qualified, saved["tracks"], decisions,
                        lambda t: inside(t["measurement_source_xy"] if t["measured"] else t["source_xy"],
                                         control["crop_xywh"]))
            for d in decisions:
                tiers[d["tier"]] += 1
                reasons[d["reason"]] += 1
            verdicts = {"baseline": {}, "v36": {identity(d): d for d in saved["tracks"]},
                        "v57": {identity(d): d for d in decisions}}
            for sample in by_frame[row["frame_index"]]:
                scores.append(score_reference(sample, row, verdicts))
            if guard is not None and row["frame_index"] == guard["frame_index"]:
                track = next(t for t in row["tracks"] if identity(t) == identity(guard["track"]))
                for k in ("track_id", "segment", "measured", "qualified_moving", "source_xy",
                          "measurement_source_xy", "hits", "learning_shape_reference_xy"):
                    same(track[k], guard["track"][k], "Continuity guard changed: " + k)
                continuity = dict(frame_index=row["frame_index"], original_track=track,
                                  shadow=verdicts["v57"][identity(track)],
                                  counted_as_actual_reference_hit=False)
            frames += 1
    require(frames == expected_frames, "Incomplete clip")
    require(len(scores) == len(refs), "Missing reference frames")
    if guard is not None:
        require(continuity is not None, "Missing continuity guard")
    controls_out = []
    for c, stats_c in zip(controls, control_stats):
        measured = finished_stats(stats_c)
        for arm in ("baseline", "v36"):
            for kind in ("measured", "predicted"):
                require(measured[arm][kind] == c[arm + "_" + kind], "Original control counts changed")
        controls_out.append(dict(scope=c, workload=measured, verified_airborne_negative=False))
    return dict(frames=frames, workload=finished_stats(stats), tiers=dict(tiers), reasons=dict(reasons),
                controls=controls_out, references=scores, continuity=continuity)


def preflight(test_log):
    """Verify exact metadata files, never recurse through historical media bindings."""
    test_path = Path(test_log)
    require(not test_path.is_absolute() and ".." not in test_path.parts and test_path.suffix == ".log",
            "Repository-relative test log required")
    inputs = {}
    def bind(relative, expected=None):
        actual = sha(ROOT / relative)
        require(expected is None or actual == expected, "Prior receipt mismatch: " + relative)
        inputs[relative] = actual
    audit_path = V36 + "full_context_independent_audit_01.json"
    trace_receipt_path = RESULTS + "accuracy_v55_20260926/trace_01/completion_receipt.json"
    trace_receipt = read(ROOT / trace_receipt_path)
    require(trace_receipt["completed"] is True, "Completed V55 parent receipt required")
    audit = read(ROOT / audit_path)
    require(audit["verified"] is True and audit["every_baseline_qualified_state_verified_once"] is True,
            "Audited parent features required")
    bind(audit_path, trace_receipt["files_sha256"][str(ROOT / audit_path)])
    clips = {}
    for cid, count in COUNTS.items():
        journal = RESULTS + f"visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_{cid}/frames.jsonl"
        features = V36 + f"full_context_01/{cid}_decisions.jsonl"
        for relative in (journal, features):
            bind(relative, audit["checked_files_sha256"][str(ROOT / relative)])
        clips[cid] = dict(journal=journal, features=features, frames=count)
    for relative, receipt, field in (
        (TRACE, RESULTS + "accuracy_v55_20260926/trace_01/completion_receipt.json", "files_sha256"),
        (GRID, RESULTS + "accuracy_v40_20260925/coverage_workload_01/completion_receipt.json", "inputs_outputs_sha256")):
        data = read(ROOT / receipt)
        require(data["completed"] is True, "Incomplete parent reference receipt")
        bind(receipt)
        bind(relative, data[field][str(ROOT / relative)])
    verification_path = RESULTS + "accuracy_v56_20260926/verification_01.json"
    verification = read(ROOT / verification_path)
    require(verification["completed"] is True, "Completed V56 continuity verification required")
    bind(verification_path)
    bind(GUARD, verification["files_sha256"]["compact_01/frame216_track_continuity.json"])
    parent_freeze_path = RESULTS + "accuracy_v56_20260926/compact_01/freeze.json"
    parent_audit_path = RESULTS + "accuracy_v56_20260926/compact_01/independent_audit.json"
    bind(parent_audit_path, verification["files_sha256"]["compact_01/independent_audit.json"])
    parent_audit = read(ROOT / parent_audit_path)
    require(parent_audit["passed"] is True, "Passed V56 parent audit required")
    bind(parent_freeze_path, parent_audit["files_sha256"]["freeze.json"])
    bind(CONTROLS, read(ROOT / parent_freeze_path)["files_sha256"]["probes.json"])
    control_rows = read(ROOT / CONTROLS)["provisional_controls"]
    require(len(control_rows) == 7 and len({encoded(c) for c in control_rows}) == 7,
            "Seven original control scopes required")
    require(all(c["clip"] == "0126" and c["verified_airborne_negative"] is False
                and 0 <= c["frames_inclusive"][0] <= c["frames_inclusive"][1] < COUNTS["0126"]
                for c in control_rows), "Changed original control scope")
    for key, value in (("baseline_measured", 70), ("baseline_predicted", 64),
                       ("v36_measured", 16), ("v36_predicted", 9)):
        require(sum(c[key] for c in control_rows) == value, "Changed original control totals")
    for relative in (CONFIG, test_log):
        bind(relative)
    log_text = (ROOT / test_log).read_text()
    require("OK" in log_text and "FAILED" not in log_text and "ERROR" not in log_text,
            "Successful generated unittest log required")
    return inputs, clips


def recheck(freeze):
    for mapping in (freeze["inputs"], freeze["implementation"]):
        for path, expected in mapping.items():
            require(sha(ROOT / path) == expected, "Frozen file changed: " + path)


def run(output, test_log):
    from stress_accuracy_v57 import run_stress
    output = Path(output).resolve()
    require(not output.exists(), "Fresh output directory required; no overwrite")
    settings = read(ROOT / CONFIG)
    same(settings["config"], asdict(PersistenceConfig()), "Only the fixed V57 policy is supported")
    same(settings["reference_panels"], PANELS, "Changed reference panels")
    require(settings["production_changed"] is False and settings["promotion_allowed"] is False,
            "Nonproduction shadow only")
    inputs, clips = preflight(test_log)
    implementation = {p: sha(ROOT / p) for p in IMPLEMENTATION}
    freeze = dict(schema="seaqr.accuracy_v57.freeze.v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  pre_scoring=True, config=settings["config"], clips=clips, inputs=inputs,
                  implementation=implementation, production_changed=False, promotion_allowed=False,
                  reference_trace=TRACE, grid_reference=GRID, controls_manifest=CONTROLS,
                  continuity_guard=GUARD, test_log=test_log)
    output.mkdir(parents=True)
    for p, expected in implementation.items():
        target = output / "implementation" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, target)
        require(sha(target) == expected, "Implementation copy changed")
    write(output / "freeze.json", freeze)
    recheck(freeze)
    stress = run_stress()
    require(stress["promotion_allowed"] is False, "Stress cannot authorize promotion")
    write(output / "stress.json", stress)
    refs = reference_samples(read(ROOT / TRACE), read(ROOT / GRID))
    controls = read(ROOT / CONTROLS)["provisional_controls"]
    config = PersistenceConfig(**freeze["config"])
    results = {}
    for cid, spec in clips.items():
        print("Replaying frozen metadata:", cid, flush=True)
        results[cid] = replay_clip(ROOT / spec["journal"], ROOT / spec["features"],
            output / f"{cid}_decisions.jsonl", spec["frames"], config,
            [r for r in refs if r["clip_id"] == cid], [c for c in controls if c["clip"] == cid],
            read(ROOT / GUARD) if cid == "0126" else None)
    panel_scores = {}
    for panel, count in PANELS.items():
        samples = [r for result in results.values() for r in result["references"]
                   if r["sample"]["panel"] == panel]
        require(len(samples) == count, "Incomplete panel")
        panel_scores[panel] = dict(samples=count, arms={arm: {
            key: sum(r["arms"][arm][key] for r in samples) for key in
            ("original_assignment_retained", "any_qualified_alternative_retained", "lost_original_assignment")}
            for arm in ARMS})
    recheck(freeze)
    for p, expected in implementation.items():
        require(sha(output / "implementation" / p) == expected, "Frozen snapshot changed")
    summary = dict(schema="seaqr.accuracy_v57.summary.v1", completed=True, frames=sum(COUNTS.values()),
        freeze_sha256=sha(output / "freeze.json"), clips=results, panels=panel_scores,
        production_changed=False, promotion_allowed=False, classifier_promoted=False, parameter_sweep=False,
        known_counterexample_rejected=stress["known_counterexample_rejected"],
        known_counterexample_blocks_promotion=True, airborne_accuracy_established=False,
        video_decoded=False, raw16_accessed=False, holdouts_accessed=False, remote_accessed=False,
        detector_rerun=False, feedback_changed=False, performance_benchmark=False,
        references_overlap_not_independent=True, workload_is_false_positive_rate=False,
        independent_audit_required=True, input_and_code_hashes_rechecked=True)
    write(output / "summary.json", summary)
    files = ["freeze.json", "stress.json", "summary.json"] + [c + "_decisions.jsonl" for c in clips]
    write(output / "completion_receipt.json", dict(completed=True, production_changed=False,
        promotion_allowed=False, outputs_sha256={p: sha(output / p) for p in files},
        all_frozen_inputs_and_code_rehashed=True))
    print("Completed nonpromoted shadow:", output, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-log", required=True, help="Repository-relative successful generated unittest log")
    args = parser.parse_args()
    run(args.output, args.test_log)
