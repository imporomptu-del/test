"""Bounded original-runtime saved-proposal tracker shadow; no media or detector.

Integrity success is distinct from scientific success. Both candidate copies
must agree exactly; the baseline must still match the original archive and the
completed Jetson replay state hashes with zero tolerance. A failed target guard
is a completed scientific rejection, never an invitation to retry or retune.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

SCHEMA = "seaqr.tracker-maturity-shadow.v1"
FREEZE_SCHEMA = "tracker_maturity_shadow.v1"
PATTERN = r"/tmp/seaqr_tracker_maturity_20261001_[A-Za-z0-9]{6}"
OLD = Path("/tmp/seaqr_tracker_baseline_20261001_6DvV6F")
OLD_FREEZE_SHA = "c13c8bf0e40cbaecfa16c82d7f9f5fe30f6980bd9fcb2eac6bf06d3685e233ea"
BASE_SHA = "6a26937ef03821db05a96f680d899234436e49a864c9d287628872e471c20c73"
JETSON_SHA = "2a359b366ec487095456138eb8e7e108678e0d162bac38be82b207b9e3e0f891"
CANDIDATE_SHA = "e4908f953b10c0385b3c66cd30d1bb4c9354eb2d25f8907a9fa3b9869d65b19c"
CLASS_SHA = "68ae5acf6f23e998dcd55e035eef66301904603ac12b2aefbc40c6ea89960b3c"
SCORER_SHA = "d8a4504b0ba739f1319b34f2be48de8b60292d9c2e74cebad3ac222f2ad2d50b"
REGRESSION_SHA = "10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
OLD_RESULTS = {"0170": "1b5f6e4914efb63f002ceea94e90a1d352aebe3ef996213f05763c01309edb96",
               "0240": "55f3fdd6511caa936846de4802cc3dfa539200de726167642e9fe92c5dbcf764"}
OLD_STATES = {"0170": "5fedda90efa88f9f80f43386ea9042027478bfe53ab4bb94c78cad1b3eb5fe79",
              "0240": "622583ccaa09ea0356e35d119cd558cfa1ba61327f89d99d6311052ac4b3b75f"}
FILES = {"check_tracker_maturity_shadow_v1.py", "test_tracker_maturity_shadow_v1.py",
         "tracker_maturity_candidate_v1.py", "test_tracker_maturity_candidate_v1.py",
         "batch_tracker_maturity_shadow_v1.py", "test_batch_tracker_maturity_shadow_v1.py",
         "batch_discovery_pair.py", "compare_discovery_feature_supply.py",
         "discovery_pair_regression_20260929.json"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def module(name, path, expected):
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            "module hash differs: " + str(path))
    binding = (str(path.resolve()), expected)
    absent = object()
    previous = sys.modules.get(name, absent)
    if previous is not absent:
        require(getattr(previous, "__seaqr_source_binding__", None) == binding,
                "existing module binding differs: " + name)
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    # inspect.getsource(class) resolves class.__module__ through sys.modules.
    # Register before execution, like normal import, without replacing an
    # existing module of different provenance or retaining a failed import.
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
        require(sys.modules.get(name) is value, "module replaced its registration")
        require(sha(path) == expected, "module changed during import")
        value.__seaqr_source_binding__ = binding
    except BaseException:
        if previous is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return value


def freeze_spec():
    return dict(policy="maturity_first_eligible_v1", clips=["0170", "0240"],
        frames_per_clip=673, nominal_timestamp_step_ns=100000000,
        candidate_sha256=CANDIDATE_SHA, candidate_class_source_sha256=CLASS_SHA,
        original_baseline_workspace=str(OLD), original_baseline_freeze_sha256=OLD_FREEZE_SHA,
        original_baseline_results=OLD_RESULTS.copy(), original_baseline_states=OLD_STATES.copy(),
        baseline_exact_archive_and_old_state_required=True, candidate_copies=2,
        first_learning_divergence_recorded_before_update=True,
        target=dict(clip="0240", first=430, last_inclusive=464, samples=35,
                    radius_native_px=8, polarity="dark", actual=True, qualified=True,
                    coherent_all_samples=True, ambiguous_samples_must_be_zero=True,
                    baseline_derived_not_independent_truth=True),
        workload_windows=dict(full=[0, 672], burst=[50, 105]),
        scientific_failure_completes=True, no_threshold_search=True,
        source_media_opened=False, detector_replayed=False, production_promotion=False)


def workspace_guard(workspace):
    workspace = Path(workspace)
    require(re.fullmatch(PATTERN, str(workspace)) and workspace.is_dir()
            and workspace.resolve() == workspace and not workspace.is_symlink()
            and os.geteuid() != 0, "fixed unprivileged workspace required")
    return workspace


def load(workspace, digest):
    """Metadata/code only; callable by the supervisor before numerical imports."""
    workspace = Path(workspace)
    path = workspace / "freeze.json"
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            and path.is_file() and not path.is_symlink() and sha(path) == digest,
            "caller-bound freeze differs")
    base = module("maturity_baseline_tools", OLD / "check_tracker_capacity_shadow.py", BASE_SHA)
    freeze = base.read(path)
    require(freeze.get("schema") == FREEZE_SCHEMA and freeze.get("protocol") == freeze_spec()
            and set(freeze.get("files", {})) == FILES, "frozen experiment scope differs")
    hashes = {}
    base.bind(path, digest, hashes)
    for name, expected in freeze["files"].items():
        require(Path(name).name == name and re.fullmatch(r"[0-9a-f]{64}", expected), "unsafe bundle file")
        base.bind(workspace / name, expected, hashes)
    protected = {"tracker_maturity_candidate_v1.py": CANDIDATE_SHA,
                 "compare_discovery_feature_supply.py": SCORER_SHA,
                 "discovery_pair_regression_20260929.json": REGRESSION_SHA,
                 "batch_discovery_pair.py": SAFETY_SHA}
    require(all(freeze["files"][name] == expected for name, expected in protected.items())
            and sha(Path(__file__).resolve()) == freeze["files"]["check_tracker_maturity_shadow_v1.py"],
            "protected or executed source differs")
    for name, expected in (("check_tracker_capacity_shadow.py", BASE_SHA),
                           ("check_tracker_capacity_jetson.py", JETSON_SHA), ("freeze.json", OLD_FREEZE_SHA)):
        base.bind(OLD / name, expected, hashes)
    for clip in base.CLIPS:
        result_path = OLD / clip / "result.json"
        states_path = OLD / clip / (clip + "_state_hashes.jsonl")
        base.bind(result_path, OLD_RESULTS[clip], hashes)
        base.bind(states_path, OLD_STATES[clip], hashes)
        value = base.read(result_path)
        replay = value["replay"]
        require(value["schema"] == "seaqr.tracker-baseline-jetson.v1.run"
                and value["passed"] is True and value["error"] is None
                and value["workspace"] == str(OLD) and value["freeze_sha256"] == OLD_FREEZE_SHA
                and value["clip"] == clip and value["inputs_unchanged_after_check"] is True
                and value["clock_controls_unchanged"] is True and replay["passed"] is True
                and all(replay[k] == 673 for k in ("attempted_frames", "exact_archive_frames",
                    "dual_state_exact_frames", "derived_learning_exact_frames"))
                and replay["first_difference"] is None and replay["state_hashes_file"] == str(states_path)
                and replay["state_hashes_sha256"] == OLD_STATES[clip], "original baseline was not complete/exact")
    base.verify_unchanged(hashes)
    return base, freeze, hashes


def reference(base, workspace):
    intake = base.read(workspace / "discovery_pair_regression_20260929.json")
    positive = intake["positive_pass"]
    contract = positive["retention_contract"]
    require(intake["scope"]["allowed_clip_ids"] == ["0170", "0240"]
            and (positive["clip_id"], positive["first_frame"], positive["last_frame_inclusive"], positive["reference_frame_count"])
            == ("0240", 430, 464, 35)
            and [r["frame_index"] for r in positive["baseline_measurements"]] == list(range(430, 465))
            and contract["radius_native_px"] == 8 and contract["required_polarity"] == "dark"
            and contract["require_measured"] is True and contract["require_qualified"] is True
            and contract["primary"] == "one_coherent_candidate_identity_covers_all_reference_frames",
            "frozen positive reference differs")
    return positive["baseline_measurements"]


class Workload:
    """Development workload, not physical-object/false-positive classification."""
    def __init__(self):
        self.sums = {"full": Counter(), "burst": Counter()}
        self.identities = {}
        self.alive = set()

    def add(self, row, output, tracker):
        frame = row["frame_index"]
        tracks, metrics = output["tracks"], output["tracking_metrics"]
        values = Counter(frames=1, ready_frames=int(row["coverage"]["detection_ready"]),
                         candidates=len(row["candidates"]), active_tracks=len(tracks))
        hit_histogram, occupancy = Counter(), {}
        self.alive = set()
        for record in tracks:
            key = f"{row['segment']}/{record['track_id']}"
            self.alive.add(key)
            hits = record["independent_hits"]
            previous = self.identities.setdefault(key, dict(first_frame=frame, last_frame=frame, maximum_hits=0))
            previous.update(last_frame=frame, maximum_hits=max(previous["maximum_hits"], hits))
            hit_histogram[str(hits)] += 1
            values["qualified_measured" if record["measured"] else "qualified_predicted"] += int(record["qualified_moving"])
        for polarity, manager in tracker.managers.items():
            require(len(manager._tracks) <= manager.config.max_active_tracks == 256
                    and manager.config.confirmation_independent_hits == 4 and manager.config.max_missed_windows == 7,
                    "frozen capacity/lifecycle differs")
            cell_counts = Counter((math.floor(t.mean[0] / 256), math.floor(t.mean[1] / 256)) for t in manager._tracks.values())
            occupancy[polarity] = {f"{x},{y}": n for (x, y), n in sorted(cell_counts.items())}
            item = metrics.get(polarity, {})
            for source, destination in (("birth_count", "births"), ("deleted_track_count", "deletions"),
                    ("dropped_birth_count_at_active_track_cap", "dropped_births")):
                values[destination] += item.get(source, 0)
            values.update({"lifecycle_" + k: v for k, v in item.get("lifecycle_counts", {}).items()})
            admission = item.get("birth_admission", {})
            require(admission.get("confirmed_tracks_evicted", 0) == 0, "confirmed track evicted")
            values["replacements"] += len(admission.get("tentative_replacements", []))
        for window in ("full", "burst"):
            if window == "full" or 50 <= frame <= 105:
                self.sums[window].update(values)
        return dict(counts=dict(values), independent_hit_histogram=dict(sorted(hit_histogram.items())),
                    cell_occupancy_by_polarity=occupancy)

    def finish(self):
        result = {}
        for name, (first, last) in {"full": (0, 672), "burst": (50, 105)}.items():
            counts = dict(self.sums[name])
            cohort = {key: row for key, row in self.identities.items() if first <= row["first_frame"] <= last}
            result[name] = dict(counts=counts,
                mean_active_tracks=counts.get("active_tracks", 0) / counts["frames"] if counts.get("frames") else None,
                first_seen_identity_count=len(cohort),
                first_seen_maximum_independent_hits_histogram=dict(sorted(Counter(str(r["maximum_hits"]) for r in cohort.values()).items())),
                never_reached_four_hits_by_end_of_clip=sum(r["maximum_hits"] < 4 for r in cohort.values()),
                still_active_without_four_hits_at_end_of_clip=sum(r["maximum_hits"] < 4 and k in self.alive for k, r in cohort.items()),
                cohort_followed_until_end_of_clip=True)
        return result


def target_comparison(scorer, baseline_rows, candidate_rows, references):
    baseline = scorer.retention(baseline_rows, references, radius=8.0, polarity="dark")
    candidate = scorer.retention(candidate_rows, references, radius=8.0, polarity="dark")
    require(baseline["preservation_guard_passed"] and baseline["ambiguous_frames"] == 0
            and baseline["any_identity_matched_frames"] == 35, "original target reference no longer matches baseline")
    details = []
    for old, new in zip(baseline["details"], candidate["details"]):
        require(old["frame_index"] == new["frame_index"], "target inventory differs")
        details.append(dict(frame_index=old["frame_index"], baseline=old, candidate=new,
                            newly_lost=bool(old["matched_identities"]) and not new["matched_identities"],
                            recovered=not old["matched_identities"] and bool(new["matched_identities"])))
    return dict(baseline=baseline, candidate=candidate, per_sample=details,
                newly_lost_frames=[r["frame_index"] for r in details if r["newly_lost"]],
                recovered_frames=[r["frame_index"] for r in details if r["recovered"]],
                scientific_guard_passed=candidate["preservation_guard_passed"]
                    and candidate["ambiguous_frames"] == 0 and candidate["any_identity_matched_frames"] == 35)


def replay_rows(base, rows, previous_states, trackers, methods, manager_class, audit_stream,
                candidate_stream, *, expected_frames=673, workload_factory=Workload):
    import numpy as np
    require(len(trackers) == len(methods) == 3, "one baseline and two candidate copies required")
    stats = [workload_factory(), workload_factory()]
    target_rows = [[], []]
    result = dict(attempted_frames=0, exact_archive_frames=0, exact_previous_state_frames=0,
                  deterministic_candidate_frames=0, baseline_learning_exact_frames=0,
                  first_learning_divergence=None, first_output_divergence=None,
                  passed=False, shadow_only=True, no_tolerance_relaxation=True)
    sentinel = object()
    for index, (row, old) in enumerate(itertools.zip_longest(rows, previous_states, fillvalue=sentinel)):
        require(row is not sentinel and old is not sentinel, "journal/state count differs")
        require(index < expected_frames and type(row["frame_index"]) is int
                and row["frame_index"] == old["frame_index"] == index
                and row["timestamp_ns"] == index*100000000
                and row["coverage"]["full_shape_hw"] == [3190, 4784], "frame/timestamp/native shape differs")
        result["attempted_frames"] += 1
        outputs, states, learning = [], [], []
        for tracker, method in zip(trackers, methods):
            learning.append(base.value_sha(tracker.learning_centers(row["timestamp_ns"], row["segment"])))
            with patch.object(manager_class, "update", method):
                tracks, metrics = tracker.update(row["candidates"], index, row["timestamp_ns"], row["segment"],
                                                  np.array(row["source_to_reference"]), row["coverage"]["full_shape_hw"])
            outputs.append(base.json_value(dict(tracks=tracks, tracking_metrics=metrics)))
            states.append(base.value_sha(base.snapshot(tracker)))
        archive = dict(tracks=row["tracks"], tracking_metrics=row["tracking_metrics"])
        require(base.first_difference(archive, outputs[0]) is None, "baseline archived observable mismatch at " + str(index))
        output_hashes = [base.value_sha(o) for o in outputs]
        require(old["differences"] == {} and old["output_sha256"] == [output_hashes[0]] * 2
                and old["archive_output_sha256"] == output_hashes[0]
                and old["internal_state_sha256"] == [states[0]] * 2,
                "baseline previous state/observable mismatch at " + str(index))
        require(old["learning_centers_sha256"] == [learning[0]] * 2
                and old["archive_derived_learning_sha256"] == learning[0],
                "baseline previous learning mismatch at " + str(index))
        require(base.first_difference(outputs[1], outputs[2]) is None
                and states[1] == states[2] and learning[1] == learning[2],
                "candidate nondeterminism at " + str(index))
        if learning[0] != learning[1] and result["first_learning_divergence"] is None:
            result["first_learning_divergence"] = dict(frame_index=index, measured_before_frame_update=True,
                baseline_sha256=learning[0], candidate_sha256=learning[1],
                consequence="Saved proposals remain fixed; this is not a causal full-pipeline result.")
        if output_hashes[0] != output_hashes[1] and result["first_output_divergence"] is None:
            result["first_output_divergence"] = dict(frame_index=index,
                detail=base.first_difference(outputs[0], outputs[1]))
        frame_workload = {name: collector.add(row, output, tracker) for name, collector, output, tracker
                          in zip(("baseline", "candidate"), stats, outputs, trackers)}
        audit = dict(frame_index=index, output_sha256=output_hashes, internal_state_sha256=states,
                     learning_centers_sha256=learning, workload=frame_workload,
                     past_or_at_learning_divergence=result["first_learning_divergence"] is not None)
        audit_stream.write(json.dumps(audit, allow_nan=False) + "\n")
        candidate_row = dict(row, **outputs[1], shadow=dict(
            saved_proposals=True, candidate_policy="maturity_first_eligible_v1",
            detector_coverage_motion_and_timings_are_archived_not_rerun=True,
            past_or_at_learning_divergence=result["first_learning_divergence"] is not None))
        candidate_stream.write(json.dumps(candidate_row, allow_nan=False) + "\n")
        if 430 <= index <= 464:
            for target, output in zip(target_rows, outputs):
                target.append(dict(frame_index=index, segment=row["segment"], tracks=output["tracks"],
                                   detection_ready=row["coverage"]["detection_ready"]))
        for key in ("exact_archive_frames", "exact_previous_state_frames", "deterministic_candidate_frames", "baseline_learning_exact_frames"):
            result[key] += 1
        if index and index % 50 == 0:
            audit_stream.flush()
            candidate_stream.flush()
            print(json.dumps(dict(completed_frames=index+1)), flush=True)
    require(result["attempted_frames"] == expected_frames, "incomplete replay")
    result.update(passed=True, workload={name: collector.finish() for name, collector in zip(("baseline", "candidate"), stats)})
    return result, target_rows


def run_clip(base, candidate, scorer, workspace, clip, launch, libraries, references):
    from tiny_target.visible_baseline import VisibleConfig, VisibleTracks
    from tiny_target.tracking.kalman import KalmanTrackManager
    cfg = VisibleConfig(**launch["configuration"])
    trackers = [VisibleTracks(cfg, launch["fps"]) for _ in range(3)]
    adapters = [base.make_adapter(libraries) for _ in range(3)]
    for components in ([a[1] for a in adapters], [a[2] for a in adapters], [a[2].geometry for a in adapters]):
        require(len({id(value) for value in components}) == 3, "adapters must own separate mutable/native instances")
    methods, bindings = [adapters[0][0]], []
    for method, _, _ in adapters[1:]:
        changed, binding = candidate.make_candidate_method(method, expected_module_sha256=CANDIDATE_SHA)
        require(binding["class_source_sha256"] == CLASS_SHA, "new victim class source differs")
        methods.append(changed)
        bindings.append(binding)
    output = workspace / clip
    candidate_path, audit_path = output / "candidate_frames.jsonl", output / "shadow_audit.jsonl"
    journal = base.EVIDENCE / clip / "run/frames.jsonl"
    old_states = OLD / clip / (clip + "_state_hashes.jsonl")
    with journal.open() as rows, old_states.open() as previous, audit_path.open("x") as audit, candidate_path.open("x") as candidate_stream:
        result, selected = replay_rows(base, (base.decode(line) for line in rows), (base.decode(line) for line in previous),
            trackers, methods, KalmanTrackManager, audit, candidate_stream)
    result["target"] = target_comparison(scorer, selected[0], selected[1], references) if clip == "0240" else None
    result["scientific_guard_passed"] = result["target"]["scientific_guard_passed"] if result["target"] else None
    result["candidate_bindings"] = bindings
    result["separate_adapter_owners_verified"] = True
    result["adapter_runs"] = []
    for _, geometry, optimized in adapters:
        counters = dict(geometry_calls=geometry.calls, geometry_fallbacks=geometry.fallbacks,
                        batch_fallbacks=optimized.geometry.fallbacks, innovation_fallbacks=optimized.innovation_fallbacks,
                        batch_track_rows=optimized.geometry.track_rows, innovation_tracks=optimized.innovation_tracks)
        require(all(counters[k] == 0 for k in ("geometry_calls", "geometry_fallbacks", "batch_fallbacks", "innovation_fallbacks"))
                and counters["batch_track_rows"] == counters["innovation_tracks"], "unexpected native fallback/accounting")
        result["adapter_runs"].append(counters)
    result["artifacts_sha256"] = {str(p): sha(p) for p in (candidate_path, audit_path)}
    return result


def validate_child_receipt(workspace, digest, phase):
    base, freeze, hashes = load(workspace, digest)
    workspace = Path(workspace)
    require(phase in ("preflight", "run_0170", "run_0240"), "unknown child phase")
    preflight = phase == "preflight"
    clip = None if preflight else phase[-4:]
    path = workspace / "preflight.json" if preflight else workspace / clip / "result.json"
    receipt_sha = sha(base.regular(path))
    value = base.read(path)
    require(value["schema"] == SCHEMA + (".preflight" if preflight else ".run")
            and value["workspace"] == str(workspace) and value["freeze_sha256"] == digest
            and value["clip"] == clip and value["passed"] is True and value["error"] is None
            and value["inputs_unchanged_after_check"] is True and value["clock_controls_unchanged"] is True
            and value["clock_policy_before"] == value["clock_policy_after"], "child integrity failed")
    expected_inputs = dict(hashes)
    old_runtime = {}
    for original_clip in ("0170", "0240"):
        old_result = base.read(OLD / original_clip / "result.json")
        old_runtime[original_clip] = old_result
        expected_inputs.update({p: h for p, h in old_result["inputs_sha256"].items()
                                if OLD not in Path(p).parents})
    require(all(value[k] is False for k in ("source_media_opened", "detector_replayed",
            "raw16_or_holdouts_accessed", "production_promotion", "scientific_improvement_claimed"))
            and value["saved_proposals_shadow_only"] is True, "child scope differs")
    runtime_reference = old_runtime[clip or "0170"]
    for name in ("runtime_before",) if preflight else ("runtime_before", "runtime_after"):
        require(all(value[name].get(k) == runtime_reference[name].get(k) for k in
            ("blas", "affinity", "numpy", "opencv", "opencv_threads", "thread_environment", "clock_ticks")),
            "child original runtime metadata differs")
    artifacts = {str(path): receipt_sha}
    if not preflight:
        pre = workspace / "preflight.json"
        require(value["preflight_sha256"] == sha(pre), "child preflight binding differs")
        previous = validate_child_receipt(workspace, digest, "preflight")
        if clip == "0240":
            previous.update(validate_child_receipt(workspace, digest, "run_0170"))
            require(value["prior_clip_sha256"] == sha(workspace / "0170/result.json"), "prior clip binding differs")
        artifacts.update(previous)
        expected_inputs.update(previous)
        replay = value["replay"]
        require(replay["passed"] is True and all(replay[k] == 673 for k in (
            "attempted_frames", "exact_archive_frames", "exact_previous_state_frames",
            "deterministic_candidate_frames", "baseline_learning_exact_frames")), "incomplete shadow parity")
        require(replay["separate_adapter_owners_verified"] is True and len(replay["adapter_runs"]) == 3,
                "separate adapter proof missing")
        for counters in replay["adapter_runs"]:
            require(all(counters[k] == 0 for k in ("geometry_calls", "geometry_fallbacks", "batch_fallbacks", "innovation_fallbacks"))
                    and counters["batch_track_rows"] == counters["innovation_tracks"], "child native fallback/accounting")
        expected_paths = {str(workspace / clip / name) for name in ("candidate_frames.jsonl", "shadow_audit.jsonl")}
        require(set(replay["artifacts_sha256"]) == expected_paths and len(replay["candidate_bindings"]) == 2,
                "shadow artifact inventory differs")
        for binding in replay["candidate_bindings"]:
            require(binding["sha256"] == CANDIDATE_SHA and binding["class_source_sha256"] == CLASS_SHA
                    and binding["unchanged_transformed_source_sha256"] == base.METHOD_SHA
                    and binding["eligibility_changed"] is False and binding["configuration_changed"] is False,
                    "candidate class/helper binding differs")
        for artifact, expected in replay["artifacts_sha256"].items():
            base.bind(Path(artifact), expected, artifacts)
            with Path(artifact).open() as stream:
                count = 0
                for index, line in enumerate(stream):
                    require(base.decode(line)["frame_index"] == index, "shadow artifact discontinuity")
                    count += 1
            require(count == 673, "shadow artifact frame count differs")
    require(value["inputs_sha256"] == expected_inputs, "child input binding differs")
    base.verify_unchanged(expected_inputs)
    base.verify_unchanged(artifacts)
    return artifacts


def child(workspace, digest, *, preflight=False, clip=None):
    workspace = workspace_guard(workspace)
    require(preflight != (clip in ("0170", "0240")), "exactly one child mode required")
    output = workspace if preflight else workspace / clip
    receipt_path = output / ("preflight.json" if preflight else "result.json")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "refusing existing receipt")
    if not preflight:
        require(not output.exists() and not output.is_symlink(), "refusing existing clip directory")
        output.mkdir()
    result = dict(schema=SCHEMA + (".preflight" if preflight else ".run"), passed=False, error=None,
        workspace=str(workspace), freeze_sha256=digest, clip=clip, source_media_opened=False,
        detector_replayed=False, raw16_or_holdouts_accessed=False, production_promotion=False,
        saved_proposals_shadow_only=True, scientific_improvement_claimed=False,
        note="Integrity and scientific guards are separate. Maturity may preserve clutter as well as targets; workload is not a false-positive rate.")
    base, jetson, hashes = None, None, {}
    began = time.monotonic()
    try:
        base, freeze, hashes = load(workspace, digest)
        jetson = module("maturity_frozen_jetson", OLD / "check_tracker_capacity_jetson.py", JETSON_SHA)
        base.EVIDENCE = jetson.TRACE
        launches, receipts, libraries, profiler = jetson.dependencies(base, hashes)
        before = profiler.runtime_info()
        for reference_receipt in receipts.values():
            jetson.check_runtime(before, reference_receipt["runtime_before"])
        result.update(clock_policy_before=jetson.clock_policy(), runtime_before=before)
        references = reference(base, workspace)
        if preflight:
            result.update(passed=True, frames_processed=0)
        else:
            validated = validate_child_receipt(workspace, digest, "preflight")
            hashes.update(validated)
            result["preflight_sha256"] = sha(workspace / "preflight.json")
            if clip == "0240":
                hashes.update(validate_child_receipt(workspace, digest, "run_0170"))
                result["prior_clip_sha256"] = sha(workspace / "0170/result.json")
            candidate = module("frozen_maturity_candidate", workspace / "tracker_maturity_candidate_v1.py", CANDIDATE_SHA)
            scorer = module("frozen_maturity_scorer", workspace / "compare_discovery_feature_supply.py", SCORER_SHA)
            import cv2
            cv2.setNumThreads(launches[clip]["configuration"]["opencv_threads"])
            result["replay"] = run_clip(base, candidate, scorer, workspace, clip, launches[clip], libraries, references)
            result["runtime_after"] = profiler.runtime_info()
            jetson.check_runtime(result["runtime_after"], receipts[clip]["runtime_after"], after=True)
            result["passed"] = result["replay"]["passed"]
    except BaseException as exc:
        result["error"] = repr(exc)
        result["passed"] = False
    finally:
        result["inputs_sha256"] = hashes.copy()
        if base is not None:
            try:
                base.verify_unchanged(hashes)
                if result.get("replay"):
                    base.verify_unchanged(result["replay"]["artifacts_sha256"])
                result["inputs_unchanged_after_check"] = True
            except BaseException as exc:
                result.update(passed=False, inputs_unchanged_after_check=False, postcheck_error=repr(exc))
        if jetson is not None and "clock_policy_before" in result:
            try:
                result["clock_policy_after"] = jetson.clock_policy()
                result["clock_controls_unchanged"] = result["clock_policy_after"] == result["clock_policy_before"]
                require(result["clock_controls_unchanged"], "clock controls changed")
            except BaseException as exc:
                result.update(passed=False, clock_postcheck_error=repr(exc))
        result["elapsed_seconds_not_pipeline_throughput"] = time.monotonic() - began
        with receipt_path.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(passed=result["passed"], clip=clip, preflight=preflight, receipt=str(receipt_path),
                         sha256=sha(receipt_path), error=result["error"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--clip", choices=("0170", "0240"))
    args = parser.parse_args()
    return 0 if child(args.workspace, args.freeze_sha256, preflight=args.preflight, clip=args.clip)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
