#!/usr/bin/env python3
"""Conditional, fixed four-clip historical regression; no labels enter inference.

Successful hash-bound two-clip development evidence is required before source
access. Original baseline assertion code is reused in a private namespace with
only the approved single-source identity and full frame count overridden.

Prepared only, not hardware-executed or qualified for launch. Development of
this conditional extension stopped when the candidate's pair readiness failed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from types import FunctionType, SimpleNamespace

SCHEMA = "seaqr.feature-selection.legacy.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_feature_selection_legacy_20260929_[A-Za-z0-9]{6}"
SELECTION_SHA = "db5a92d69fb7fc503a3ef2d56236460d1bb32656111ad3fd170203b9ab53bc4d"
PAIR_WORKSPACE = "/tmp/seaqr_feature_selection_20260929_q4iI5B"
PAIR_FREEZE_SHA = "9ae62028ca25fe063241e62697cb3578a708078a1b03969061b4736f0c9c6b7e"
PAIR_LOCAL_EVIDENCE = "/Users/romanmaksymiuk/Documents/SEAQR/outputs/seaqr_feature_selection_20260929/evidence"
COUNTS = {"0029": 687, "0126": 674, "0055": 689, "0082": 691}
SOURCES = {
    "0029": "0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359",
    "0126": "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344",
    "0055": "c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f",
    "0082": "465ba7b00393e655534f78d9be07aa965d6d404c1761b4d0e0841c1766bce117",
}
PAIR_FILES = {"comparison": "pair_comparison.json", "freeze": "pair_freeze.json",
              "0170": "pair_0170_execution_receipt.json", "0240": "pair_0240_execution_receipt.json"}
SCOPED_FUNCTIONS = ("source_spec", "check_frame", "validate_probe", "validate_output", "execute_baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular nonlinked input required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024 * 1024,
            "bounded regular JSON metadata required")
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=invalid, parse_float=number)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_spec(clip):
    require(clip in COUNTS, "source outside fixed historical four")
    return dict(path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
        sha256=SOURCES[clip], frames=COUNTS[clip], width=4784, height=3190,
        fps=10, codec="mjpeg", pixel_format="yuvj420p")


def workspace_guard(workspace, clip, mode):
    workspace = Path(workspace)
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside historical experiment workspace")
    require(workspace.is_dir() and not workspace.is_symlink() and os.geteuid() != 0,
            "real unprivileged workspace required")
    source_spec(clip)
    require(mode in ("preflight", "run"), "invalid mode")
    directory = workspace / clip
    require(not directory.is_symlink() and (not directory.exists() or directory.is_dir()), "invalid clip directory")
    for name in ("run", "execution_receipt.json") + (("preflight.json",) if mode == "preflight" else ()):
        path = directory / name
        require(not path.exists() and not path.is_symlink(), "existing output; no overwrite")
    return workspace


def load_selection(workspace):
    path = workspace / "run_discovery_feature_selection.py"
    require(sha(path) == SELECTION_SHA, "selection implementation identity differs")
    spec = importlib.util.spec_from_file_location("frozen_legacy_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_freeze(freeze, hashes, selection):
    require(freeze.get("schema") == "feature_selection_legacy.v1"
            and freeze.get("candidate") == selection.CANDIDATE
            and freeze.get("baseline_workspace") == str(selection.BASELINE_WORKSPACE)
            and freeze.get("pair_workspace") == PAIR_WORKSPACE, "historical freeze scope differs")
    require(freeze.get("sources") == {c: source_spec(c) for c in COUNTS}, "historical source allowlist differs")
    require(all(type(row[k]) is int for row in freeze["sources"].values()
                for k in ("frames", "width", "height", "fps")), "invalid source numerical types")
    require(all(type(freeze["candidate"][k]) is int for k in
                ("harris_gain", "max_features", "max_features_per_cell", "grid_rows", "grid_cols")),
            "invalid candidate integer types")
    files = freeze.get("files", {})
    mandatory = {"run_feature_selection_legacy.py", "run_discovery_feature_selection.py", *PAIR_FILES.values()}
    require(isinstance(files, dict) and mandatory <= files.keys(), "missing frozen evidence/code")
    require(all(isinstance(name, str) and Path(name).name == name and name not in ("", ".", "..", "freeze.json")
                and isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest)
                for name, digest in files.items()), "invalid frozen file names/hashes")
    require(files == hashes and files["run_discovery_feature_selection.py"] == SELECTION_SHA,
            "transferred bundle differs from freeze")
    gate = freeze.get("pair_gate", {})
    require(set(gate) == {"comparison_sha256", "freeze_sha256", "receipts_sha256"}
            and gate["freeze_sha256"] == PAIR_FREEZE_SHA
            and set(gate["receipts_sha256"]) == {"0170", "0240"}, "missing exact pair gate provenance")
    require(files[PAIR_FILES["comparison"]] == gate["comparison_sha256"]
            and files[PAIR_FILES["freeze"]] == gate["freeze_sha256"]
            and all(files[PAIR_FILES[c]] == gate["receipts_sha256"][c] for c in ("0170", "0240")),
            "pair gate evidence hashes disagree")


def validate_pair_gate(comparison, pair_freeze, receipts, provenance, selection):
    require(comparison.get("schema") == "seaqr.feature-selection.comparison.v1"
            and comparison.get("production_changed") is False
            and comparison.get("media_read") is False
            and set(comparison.get("clips", {})) == {"0170", "0240"}, "wrong pair comparison scope")
    assessment = comparison.get("assessment", {})
    require(assessment.get("declared_development_gates_passed") is True
            and assessment.get("promotion_allowed") is False,
            "pair development gate failed; historical sources remain blocked")
    expected_gates = {"both_clips_at_least_95pct_ready", "no_pva_errors", "all_35_coherent_actual_qualified_dark"}
    require(set(assessment.get("gates", {})) == expected_gates
            and all(value is True for value in assessment["gates"].values()), "incomplete pair decision")
    require(pair_freeze.get("schema") == "feature_selection.v1" and pair_freeze.get("candidate") == selection.CANDIDATE
            and pair_freeze.get("baseline_workspace") == str(selection.BASELINE_WORKSPACE)
            and pair_freeze.get("sources") == {c: selection.source_spec(c) for c in ("0170", "0240")}
            and pair_freeze.get("files", {}).get("run_discovery_feature_selection.py") == SELECTION_SHA,
            "pair freeze is not the fixed candidate")
    for clip in ("0170", "0240"):
        row, receipt = comparison["clips"][clip]["selection"], receipts[clip]
        counts = row["counts"]
        require(type(counts.get("frames")) is int and counts["frames"] == 673
                and type(counts.get("ready_frames")) is int and 640 <= counts["ready_frames"] <= 673
                and row.get("ready_fraction") == counts["ready_frames"] / 673
                and type(counts.get("pva_runtime_errors")) is int and counts["pva_runtime_errors"] == 0,
                "recomputed pair readiness/backend gate failed")
        require(receipt.get("schema") == selection.SCHEMA and receipt.get("passed") is True
                and receipt.get("error") is None and receipt.get("workspace") == PAIR_WORKSPACE
                and receipt.get("clip") == clip and receipt.get("source") == selection.source_spec(clip)
                and receipt.get("candidate") == selection.CANDIDATE
                and receipt.get("processed_frames") == receipt.get("decoded_frames_verified") == 673,
                "pair receipt incomplete or wrong candidate")
        require(receipt.get("input_sha256", {}).get("freeze_sha256") == PAIR_FREEZE_SHA
                and receipt["input_sha256"].get("files") == pair_freeze["files"], "pair receipt freeze binding differs")
        for key in ("detector_configuration_changed", "tracker_configuration_changed", "global_motion_gates_changed",
                    "production_promotion", "annotations_supplied_to_detector", "raw16_accessed", "sealed_holdouts_accessed",
                    "harris_score_precision_changed"):
            require(receipt.get(key) is False, "pair receipt scope differs: " + key)
        require(receipt.get("feature_algorithm_changed") is True, "missing pair feature-change provenance")
        expected_path = f"{PAIR_LOCAL_EVIDENCE}/{clip}/execution_receipt.json"
        require(comparison.get("input_sha256", {}).get(expected_path) == provenance["receipts_sha256"][clip],
                "pair comparison does not bind supplied receipt")
    positive = comparison["clips"]["0240"]["selection"]["positive_pass"]
    require(positive.get("reference_frames") == positive.get("best_coherent_identity_frames") == 35
            and positive.get("preservation_guard_passed") is True and positive.get("radius_native_px") == 8
            and positive.get("baseline_derived_not_independent_truth") is True
            and isinstance(positive.get("complete_coherent_identities"), list)
            and len(positive["complete_coherent_identities"]) >= 1,
            "pair coherent35frame retention gate failed")
    return dict(passed=True, comparison_sha256=provenance["comparison_sha256"],
        pair_freeze_sha256=PAIR_FREEZE_SHA, pair_receipts_sha256=provenance["receipts_sha256"],
        readiness_threshold=.95, required_coherent_reference_frames=35, pva_errors_allowed=0,
        independent_recall=False, production_promotion=False)


def inputs(workspace, clip, selection):
    freeze = read(workspace / "freeze.json")
    validate_freeze(freeze, freeze.get("files", {}), selection)
    hashes = {name: sha(workspace / name) for name in freeze["files"]}
    validate_freeze(freeze, hashes, selection)
    require(sha(Path(__file__)) == hashes["run_feature_selection_legacy.py"], "executed historical runner differs")
    # Gate before loading any legacy source bytes, decoder or device dependency.
    pair = {key: read(workspace / name) for key, name in PAIR_FILES.items()}
    decision = validate_pair_gate(pair["comparison"], pair["freeze"],
        {c: pair[c] for c in ("0170", "0240")}, freeze["pair_gate"], selection)
    baseline = selection.load_baseline()
    pinned = {baseline.HELPER: baseline.HELPER_SHA, baseline.REFERENCE: baseline.REFERENCE_SHA,
              baseline.REFERENCE_RUNTIME: baseline.REFERENCE_RUNTIME_SHA,
              Path(source_spec(clip)["path"]): SOURCES[clip]}
    for path, digest in pinned.items():
        require(sha(path) == digest, "pinned historical input differs: " + str(path))
    return baseline, read(baseline.REFERENCE), read(baseline.REFERENCE_RUNTIME), dict(
        files=hashes, freeze_sha256=sha(workspace / "freeze.json"), source_sha256=SOURCES[clip],
        pair_gate=decision, original_baseline_helper_sha256=selection.BASELINE_HELPER_SHA,
        baseline_harness_sha256=baseline.HELPER_SHA, reference_launch_sha256=baseline.REFERENCE_SHA,
        reference_runtime_sha256=baseline.REFERENCE_RUNTIME_SHA)


def scoped_baseline(baseline, clip):
    """Retain original assertion code objects; never mutate old module globals."""
    require(clip in COUNTS and baseline.FRAMES == 673 and baseline.WIDTH == 4784 and baseline.HEIGHT == 3190
            and set(baseline.SOURCE_HASHES) == {"0170", "0240"}, "unexpected original harness scope")
    namespace = dict(vars(baseline))
    namespace.update(FRAMES=COUNTS[clip], SOURCE_HASHES={clip: SOURCES[clip]})
    for name in SCOPED_FUNCTIONS:
        old = getattr(baseline, name)
        require(isinstance(old, FunctionType) and old.__closure__ is None, "unexpected scoped function closure")
        namespace[name] = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__)
    require(namespace["source_spec"](clip) == source_spec(clip), "scoped source contract differs")
    audit = dict(original_baseline_frames=673, effective_frames=COUNTS[clip],
        global_overrides={"FRAMES": COUNTS[clip], "SOURCE_HASHES": {clip: SOURCES[clip]}},
        functions=list(SCOPED_FUNCTIONS), all_original_code_objects_retained=True,
        original_module_globals_modified=False, native_size_wh=[4784, 3190])
    return SimpleNamespace(**{name: namespace[name] for name in SCOPED_FUNCTIONS}), audit


def validate_preflight(pre, clip, workspace, hashes, identities, selection):
    require(pre.get("schema") == SCHEMA + ".preflight" and pre.get("passed") is True
            and pre.get("workspace") == str(workspace) and pre.get("clip") == clip
            and pre.get("source") == source_spec(clip) and pre.get("candidate") == selection.CANDIDATE
            and pre.get("input_sha256") == hashes and pre.get("probe_passed") is True
            and pre.get("detector_run") is False and pre.get("pair_gate") == hashes["pair_gate"],
            "historical preflight identity/completion differs")
    require(pre.get("conversion", {}).get("passed") is True and pre.get("cpu_parity", {}).get("passed") is True
            and [r.get("name") for r in pre.get("cpu_parity", {}).get("cases", [])] == list(selection.PARITY_NAMES)
            and all(r.get("passed") is True for r in pre["cpu_parity"]["cases"])
            and [r.get("name") for r in pre.get("controls", [])] == list(selection.CONTROL_NAMES)
            and all(r.get("passed") is True and r.get("closed") is True for r in pre["controls"]),
            "historical generated controls incomplete")
    require(all(pre.get(k) == v for k, v in identities.items()), "historical dependencies differ from preflight")


def preflight(workspace, clip):
    workspace = workspace_guard(workspace, clip, "preflight")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA + ".preflight", passed=False, workspace=str(workspace), clip=clip,
                   source=source_spec(clip), detector_run=False, full_pixel_predecode=False, controls=[])
    try:
        selection = load_selection(workspace)
        receipt["candidate"] = dict(selection.CANDIDATE)
        baseline, reference, runtime, hashes = inputs(workspace, clip, selection)
        scoped, audit = scoped_baseline(baseline, clip)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        from tiny_target.frame_source import probe_video
        import cv2
        probe = probe_video(source_spec(clip)["path"])
        scoped.validate_probe(probe)
        motion, global_config = selection.configurations(helper)
        receipt.update(input_sha256=hashes, pair_gate=hashes["pair_gate"], scoped_harness=audit,
            runtime_before=before, clock_policy=clocks, probe=probe.to_dict(), probe_passed=True, **identities)
        cv2.setNumThreads(2)
        receipt["cpu_parity"] = selection.generated_cpu_parity(motion)
        receipt["conversion"] = selection.conversion_check()
        receipt["feature_adapter"] = {}
        with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            selection.run_controls(candidate, motion, global_config, receipt["controls"])
        after = info()
        helper.runtime_check(after, runtime, after=True)
        require(helper.clock_policy_snapshot() == clocks, "clock controls changed")
        require(inputs(workspace, clip, selection)[3] == hashes
                and helper.dependencies(reference)[1] == identities, "historical inputs/dependencies changed")
        receipt.update(passed=True, runtime_after=after, global_configuration=asdict(global_config),
                       clocks_changed=False, generated_pair_calls=3)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "preflight.json", receipt)
    return receipt


def run(workspace, clip):
    workspace = workspace_guard(workspace, clip, "run")
    directory = workspace / clip
    directory.mkdir(exist_ok=True)
    receipt = dict(schema=SCHEMA, passed=False, error=None, processed_frames=0, workspace=str(workspace),
        clip=clip, source=source_spec(clip), algorithm_changed=True, feature_algorithm_changed=True,
        feature_quota_algorithm_changed=True, exact_cpu_execution_changed=True, harris_score_precision_changed=False,
        detector_configuration_changed=False, tracker_configuration_changed=False, global_motion_gates_changed=False,
        production_promotion=False, annotations_supplied_to_detector=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        airborne_class_verified=False, physical_class="unknown", clocks_changed=None, remote_clocks_unchanged=None,
        timestamp_basis="consumer index / nominal10Hz container; physical acquisition cadence unverified",
        purpose="Historical regression only; references are scored separately, never supplied to inference.",
        launch_config_semantics="Unchanged frozen file/package hashes; feature_adapter declares actual in-memory gain,capacity,384/8quota and exact CPU execution.",
        timing_instrumentation="Native frame/count/order and correspondence checks; no pixel hashing in inference. Current clocks, uncontrolled timing.")
    try:
        selection = load_selection(workspace)
        receipt["candidate"] = dict(selection.CANDIDATE)
        baseline, reference, runtime, hashes = inputs(workspace, clip, selection)
        scoped, audit = scoped_baseline(baseline, clip)
        helper = baseline.load_helper()
        modules, identities = helper.dependencies(reference)
        # Require ALL four generated preflights before any real full run.
        for allowed in COUNTS:
            other_hashes = dict(hashes, source_sha256=SOURCES[allowed])
            validate_preflight(read(workspace / allowed / "preflight.json"), allowed,
                               workspace, other_hashes, identities, selection)
        pre = read(directory / "preflight.json")
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        require(clocks == pre.get("clock_policy"), "clock controls differ from preflight")
        motion, global_config = selection.configurations(helper)
        receipt.update(input_sha256=hashes, pair_gate=hashes["pair_gate"], scoped_harness=audit,
            preflight_sha256=sha(directory / "preflight.json"), runtime_before=before, clock_policy_before=clocks,
            feature_adapter={}, global_configuration=asdict(global_config),
            tracking_transformed_sha256=helper.TRACKING_METHOD_SHA, **identities)
        with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            changed_modules = dict(modules, motion_reuse_v12=SimpleNamespace(ReuseMotionV12=candidate))
            report = scoped.execute_baseline(helper, changed_modules, Path(source_spec(clip)["path"]), directory / "run", receipt)
        scoped.validate_output(report, read(directory / "run/launch.json"), reference, clip, receipt["decoded_frames_verified"])
        require(receipt["feature_adapter"]["estimator_instances"] == 1
                and receipt["feature_adapter"]["successful_pair_backend_checks"] ==
                sum(row["error"] is None for row in receipt["motion_attempts"]), "unverified candidate pair path")
        selection.validate_adapter_roundtrip(receipt["feature_adapter"], pre["feature_adapter"])
        after, clock_after = info(), helper.clock_policy_snapshot()
        helper.runtime_check(after, runtime, after=True)
        receipt.update(runtime_after=after, clock_policy_after=clock_after,
            clocks_changed=clock_after != clocks, remote_clocks_unchanged=clock_after == clocks)
        require(clock_after == clocks, "clock controls changed")
        require(inputs(workspace, clip, selection)[3] == hashes
                and helper.dependencies(reference)[1] == identities, "historical inputs/dependencies changed")
        receipt.update(passed=True, processed_frames=report["frames"], journal_sha256=sha(directory / "run/frames.jsonl"),
            report_sha256=sha(directory / "run/report.json"), launch_sha256=sha(directory / "run/launch.json"),
            availability=report["availability"], detection_status=report["detection_status"])
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        write(directory / "execution_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--clip", choices=tuple(COUNTS), required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    (preflight if arguments.preflight else run)(arguments.workspace, arguments.clip)
