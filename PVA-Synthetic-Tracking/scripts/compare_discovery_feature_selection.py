"""Three-arm, metadata-only development comparison on exactly0170 and0240.

Readiness, baseline-derived target retention and review workload are distinct.
No media, new detector/model run, independent recall/false-positive claim, or
production promotion is authorized by this postprocessor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT.parent / "outputs"
ROOTS = dict(baseline=OUTPUTS / "seaqr_discovery_pair_20260928/evidence",
             supply=OUTPUTS / "seaqr_feature_supply_20260929/evidence",
             selection=OUTPUTS / "seaqr_feature_selection_20260929/evidence")
REGRESSION = ROOT / "configs/evaluation/discovery_pair_regression_20260929.json"
REGRESSION_SHA = "10d7ec68ba04b05784d4e19a56b0a6dbb561a9e44c7517513b7c1a3be5c881c3"
SUPPLY_COMPARISON = ROOTS["supply"].parent / "comparison.json"
SUPPLY_COMPARISON_SHA = "bdc047479820ac9a065fd03a40a32ddbfd21d101fc04f6e4a4e0c288937c9c48"
SELECTION_BUNDLE = ROOTS["selection"].parent / "bundle"
SELECTION_FREEZE_SHA = "9ae62028ca25fe063241e62697cb3578a708078a1b03969061b4736f0c9c6b7e"
SELECTION_RUNNER_SHA = "db5a92d69fb7fc503a3ef2d56236460d1bb32656111ad3fd170203b9ab53bc4d"
HELPERS = {"comparison": (ROOT / "scripts/compare_discovery_feature_supply.py",
    "d8a4504b0ba739f1319b34f2be48de8b60292d9c2e74cebad3ac222f2ad2d50b"),
    "registration": (ROOT / "scripts/summarize_feature_supply_registration.py",
    "df313dbc9d6ee8e62a40552f4f868efab92f9f9fd0fd0c79a967c16c65063fb2")}
CLIPS = ("0170", "0240")
FILES = dict(execution_receipt="execution_receipt.json", preflight="preflight.json",
             journal="run/frames.jsonl", report="run/report.json", launch="run/launch.json")
SELECTION = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5,
    feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8, grid_rows=6, grid_cols=8)
OVERRIDES = dict(harris_capacity_policy="complete_grid", feature_cpu_policy="batched_exact_v1",
                 max_features=384, max_features_per_cell=8)
PARITY_NAMES = ("native_gray8", "masked_and_excluded_gray8", "empty", "all_ineligible", "native_proxy_boundaries")
CONTROL_NAMES = ("low_contrast_static", "low_contrast_translated", "flat_static")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink()
            and path.suffix in (".json", ".jsonl", ".py"), "only literal regular metadata/code paths")
    return path


def sha(path):
    value = hashlib.sha256()
    with regular(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def decode(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    def finite(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number
    return json.loads(text, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    return decode(regular(path).read_text())


def bind(path, hashes, expected=None):
    digest = sha(path)
    require(expected is None or digest == expected, "frozen metadata identity differs: " + str(path))
    hashes[str(path)] = digest
    return digest


def helpers(hashes):
    result = {}
    for name, (path, digest) in HELPERS.items():
        bind(path, hashes, digest)
        spec = importlib.util.spec_from_file_location("selection_comparison_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result["comparison"], result["registration"]


def load_selection_bundle(selection_root, registration, hashes):
    bind(SELECTION_BUNDLE / "freeze.json", hashes, SELECTION_FREEZE_SHA)
    freeze = read(SELECTION_BUNDLE / "freeze.json")
    require(freeze.get("schema") == "feature_selection.v1" and freeze.get("candidate") == SELECTION
            and freeze.get("sources") == {clip: registration.source_spec(clip) for clip in CLIPS}
            and freeze.get("baseline_workspace") == "/tmp/seaqr_discovery_pair_20260928_ZGLHH7",
            "selection freeze candidate/source scope differs")
    files = freeze.get("files", {})
    require(set(files) == {"batch_discovery_pair.py", "batch_feature_selection.py", "feature_selection_plan.json",
                          "run_discovery_feature_selection.py", "test_discovery_feature_selection.py"}
            and files["run_discovery_feature_selection.py"] == SELECTION_RUNNER_SHA, "selection frozen bundle inventory differs")
    for name, digest in files.items():
        bind(SELECTION_BUNDLE / name, hashes, digest)
    bind(selection_root / "freeze.json", hashes, SELECTION_FREEZE_SHA)
    require(read(selection_root / "freeze.json") == freeze, "copied evidence freeze differs")
    return freeze


def validate_bundle_binding(receipt, preflight, freeze):
    require(freeze is not None, "selection bundle validation required")
    for value in (receipt, preflight):
        require(value["input_sha256"].get("freeze_sha256") == SELECTION_FREEZE_SHA
                and value["input_sha256"].get("files") == freeze["files"], "execution/preflight is not bound to frozen bundle")


def validate_scope(baseline_root, supply_root, selection_root, regression, output):
    roots = dict(zip(ROOTS, map(lambda p: Path(p).absolute(), (baseline_root, supply_root, selection_root))))
    require(roots == ROOTS, "only the exact three evidence roots are authorized")
    regression, output = Path(regression).absolute(), Path(output).absolute()
    require(regression == REGRESSION, "only frozen discovery intake allowed")
    require(output.parent == ROOTS["selection"].parent and output.suffix == ".json"
            and output.name not in {"freeze.json", "batch_status.json"}, "output outside selection root")
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    for path in (*roots.values(), output.parent):
        require(path.is_dir() and path.resolve() == path and not path.is_symlink(), "missing/linked evidence root")
    return roots, regression, output


def validate_intake(intake, registration):
    require(intake["schema"] == "seaqr.discovery-pair.regression-intake.v1"
            and intake["scope"]["allowed_clip_ids"] == list(CLIPS)
            and [r["clip_id"] for r in intake["recordings"]] == list(CLIPS), "intake pair scope differs")
    for row in intake["recordings"]:
        require(row["frame_count"] == 673 and row["native_shape_hw"] == [3190, 4784]
                and row["source_sha256"] == registration.SOURCE_SHA[row["clip_id"]], "intake source differs")
    positive = intake["positive_pass"]
    require((positive["clip_id"], positive["first_frame"], positive["last_frame_inclusive"], positive["reference_frame_count"])
            == ("0240", 430, 464, 35), "positive interval differs")
    require([r["frame_index"] for r in positive["baseline_measurements"]] == list(range(430, 465)), "positive inventory differs")
    contract = positive["retention_contract"]
    require(contract["radius_native_px"] == 8 and contract["required_polarity"] == "dark"
            and contract["require_measured"] is True and contract["require_qualified"] is True
            and contract["primary"] == "one_coherent_candidate_identity_covers_all_reference_frames",
            "target retention contract differs")
    return {r["clip_id"]: r for r in intake["recordings"]}, positive


def validate_cpu_parity(parity, numpy_version):
    for key in ("passed", "same_candidate_quota_in_both_paths", "exact_reference_comparison",
                "source_pixels_unchanged", "points_and_scores_unchanged"):
        require(parity.get(key) is True, "missing exact CPU parity: " + key)
    require(parity.get("score_precision_changed") is False and parity.get("numpy_version") == numpy_version
            and isinstance(numpy_version, str) and bool(numpy_version), "parity precision/runtime differs")
    require([r.get("name") for r in parity.get("cases", [])] == list(PARITY_NAMES), "parity case inventory differs")
    for key in ("max_features", "max_features_per_cell", "grid_rows", "grid_cols"):
        require(type(parity.get(key)) is int and parity[key] == SELECTION[key], "parity quota/grid differs")
    for row in parity["cases"]:
        require(row.get("passed") is True and row.get("generated_only") is True
                and type(row.get("selected_count")) is int and 0 <= row["selected_count"] <= 384
                and type(row.get("maximum_selected_per_cell")) is int and 0 <= row["maximum_selected_per_cell"] <= 8,
                "parity case failed/exceeded quota")
        identity_keys = ("eligible_sha256", "selected_indices_sha256")
        if row["name"] == "native_proxy_boundaries":
            require(row.get("proxy_size_wh") == [2392, 1595] and row.get("native_source_shape_hw") == [3190, 4784]
                    and row.get("source_pixels_accessed") is False and row.get("selection_and_coverage_only") is True,
                    "native-proxy parity scope differs")
        else:
            identity_keys += ("source_pixel_sha256",)
        for key in identity_keys:
            require(isinstance(row.get(key), str) and len(row[key]) == 64
                    and all(c in "0123456789abcdef" for c in row[key]), "parity case missing identity")


def validate_selection(receipt, preflight, reference_original, reference_global, registration):
    require(receipt.get("schema") == "seaqr.discovery-feature-selection.v1"
            and receipt.get("candidate") == preflight.get("candidate") == SELECTION, "wrong selection candidate")
    for key in ("algorithm_changed", "feature_algorithm_changed", "feature_quota_algorithm_changed", "exact_cpu_execution_changed"):
        require(receipt.get(key) is True, "selection change declaration missing: " + key)
    for key in ("detector_configuration_changed", "tracker_configuration_changed", "global_motion_gates_changed",
                "production_promotion", "annotations_supplied_to_detector", "raw16_accessed", "sealed_holdouts_accessed",
                "harris_score_precision_changed"):
        require(receipt.get(key) is False, "selection scope differs: " + key)
    audit = receipt["feature_adapter"]
    original, effective = audit["original_motion_configuration"], audit["effective_motion_configuration"]
    require(original == reference_original and effective == dict(reference_original, **OVERRIDES), "effective selection config differs")
    changes = {k: dict(before=original[k], after=effective[k]) for k in original if original[k] != effective[k]}
    require(set(changes) == set(OVERRIDES) and audit["configuration_changes"] == changes, "not exactly four configuration changes")
    require(original["max_features"] == 1000 and original["max_features_per_cell"] is None
            and original["feature_cpu_policy"] == "reference" and original["harris_capacity_policy"] == "legacy_default"
            and effective["feature_image_scale"] == .5 and (effective["grid_rows"], effective["grid_cols"]) == (6, 8),
            "baseline quota/grid identity differs")
    for key in ("max_features", "max_features_per_cell", "grid_rows", "grid_cols"):
        require(type(effective[key]) is int and type(receipt["candidate"][key]) is int, "noninteger selection declaration")
    transform = audit["source_transformation"]
    require(transform["original_method_sha256"] == registration.METHOD_SHA
            and transform["transformed_method_sha256"] == registration.GAIN_METHOD_SHA
            and transform["exact_original_recovered"] is True, "gain transformation differs")
    for key in ("source_transformation", "effective_motion_configuration", "change_classification"):
        require(audit[key] == preflight["feature_adapter"][key], "execution differs from preflight: " + key)
    require(receipt["global_configuration"] == preflight["global_configuration"] == reference_global, "global gates changed")
    require(preflight.get("conversion", {}).get("passed") is True
            and preflight.get("conversion", {}).get("no_input_clipping") is True
            and preflight.get("generated_pair_calls") == 3
            and [r.get("name") for r in preflight.get("controls", [])] == list(CONTROL_NAMES)
            and all(r.get("passed") is True and r.get("closed") is True for r in preflight["controls"]),
            "generated PVA controls/conversion incomplete")
    validate_cpu_parity(preflight.get("cpu_parity", {}), preflight["runtime_after"]["numpy"])
    require(audit["estimator_instances"] == 1 and audit["successful_pair_backend_checks"] ==
            sum(a["error"] is None for a in receipt["motion_attempts"]), "unverified selection pair path")


def validate_lifecycle(receipt):
    require(receipt.get("cleanup_errors") == [] and receipt.get("native_mask_calls") == 0, "cleanup/native host path differs")
    require(receipt.get("clocks_changed") is False and receipt.get("remote_clocks_unchanged") is True
            and receipt.get("clock_policy_before") == receipt.get("clock_policy_after"), "clock policy changed")
    motion, fronts = receipt["motion_instances"], receipt["gpu_fronts"]
    require(len(motion) == len(fronts) == 1 and motion[0]["failed"] is False and motion[0]["closed"] is True
            and motion[0]["hits"] + motion[0]["misses"] == 672, "motion lifecycle differs")
    require(fronts[0]["calls"] == fronts[0]["device_calls"] == fronts[0]["finish_calls"] == 673
            and fronts[0]["host_calls"] == 0 and fronts[0]["closed"] is True, "GPU front lifecycle differs")
    tracking = receipt["tracking"]
    require(all(tracking[k] == 0 for k in ("geometry_fallbacks", "batch_fallbacks", "innovation_fallbacks"))
            and tracking["batch_track_rows"] == tracking["innovation_tracks"], "tracking fallback/accounting differs")


def load_arm(root, arm, clip, expected, hashes, comparison, registration, original=None, global_config=None, selection_freeze=None):
    require(arm in ROOTS and clip in CLIPS, "unapproved arm/source")
    require(expected is None and arm == "selection" or isinstance(expected, dict) and set(expected) == set(FILES),
            "complete historical artifact pins required")
    paths = {key: root / clip / suffix for key, suffix in FILES.items()}
    for key, path in paths.items():
        bind(path, hashes, expected.get(key) if expected is not None else None)
    receipt, pre, report, launch = (read(paths[k]) for k in ("execution_receipt", "preflight", "report", "launch"))
    for key in ("preflight", "journal", "report", "launch"):
        require(receipt[key + "_sha256"] == hashes[str(paths[key])], "receipt artifact binding differs: " + key)
    require(receipt.get("passed") is True and receipt.get("error") is None
            and receipt.get("source") == registration.source_spec(clip) and receipt.get("clip") == clip
            and receipt.get("processed_frames") == receipt.get("decoded_frames_verified") == 673,
            "failed/incomplete/wrong-source execution")
    require(pre.get("schema") == receipt["schema"] + ".preflight" and pre.get("passed") is True
            and pre.get("detector_run") is False and pre.get("probe_passed") is True
            and pre.get("source") == receipt["source"] and pre.get("clip") == clip
            and pre.get("workspace") == receipt["workspace"] and pre.get("input_sha256") == receipt["input_sha256"],
            "preflight identity/completion differs")
    for key in ("adapters", "libraries", "config_sha256", "motion_config_sha256", "v29_freeze_sha256", "vpi_version"):
        require(pre[key] == receipt[key], "preflight execution dependency differs: " + key)
    if arm == "selection":
        validate_bundle_binding(receipt, pre, selection_freeze)
        validate_selection(receipt, pre, original, global_config, registration)
    else:
        comparison.validate_receipt(receipt, "baseline" if arm == "baseline" else "candidate", clip, registration.SOURCE_SHA[clip])
        registration.validate_artifacts(receipt, pre, report, launch, "baseline" if arm == "baseline" else "candidate", clip)
    validate_lifecycle(receipt)
    require(report.get("schema") == "seaqr.visible-baseline.v1" and report.get("completed") is True
            and report.get("full_clip") is True and report.get("frames") == 673
            and report.get("source_sha256") == launch.get("source_sha256") == registration.SOURCE_SHA[clip]
            and launch.get("source") == receipt["source"]["path"]
            and report.get("configuration") == launch.get("configuration"), "report/launch differs")
    require(receipt["config_sha256"] == launch["config_sha256"] == registration.CONFIG_SHA
            and receipt["motion_config_sha256"] == launch["motion_config_sha256"] == registration.MOTION_SHA, "original file configuration differs")
    probe = launch["source_probe"]
    require(probe == pre["probe"] and (probe["width"], probe["height"], probe["declared_frame_count"]) == (4784, 3190, 673)
            and probe["codec"] == "mjpeg" and probe["pixel_format"] == "yuvj420p"
            and registration.Fraction(probe["frame_rate"]) == 10, "native source probe differs")
    decode_info = report["frame_decode"]
    require(decode_info["decoded_frames"] == decode_info["consumed_frames"] == 673 and decode_info["dropped_frames"] == 0
            and decode_info["worker_joined"] is True and decode_info["capture_released"] is True, "decode lifecycle differs")
    trimmed = []
    with paths["journal"].open() as stream:
        for line in stream:
            row = decode(line)
            trimmed.append({k: row[k] for k in ("frame_index", "timestamp_ns", "coverage", "motion", "timings_ms")})
    registration.validate_rows(trimmed, receipt["motion_attempts"])
    full, selected = comparison.collect(paths["journal"])
    validate_target_rows(selected)
    require(full["counts"]["ready_frames"] == report["availability"]["counts"]["detection_ready_frames"], "report readiness differs")
    require(full["counts"]["motion_resets"] == report["counts"]["motion_resets"], "report reset count differs")
    if arm == "selection":
        for row in trimmed[1:]:
            corr = row["motion"].get("correspondence_metrics")
            if corr is not None:
                require(0 < corr["selected_count"] <= 384 and corr["minimum_accepted_features"] == 30
                        and corr["minimum_grid_coverage"] == .2, "observed selection budget/gates differ")
                cap = corr["harris_output"]
                require(cap["capacity_policy"] == "complete_grid" and cap["capacity"] == 60300
                        and cap["capacity_exhausted"] is False, "observed Harris capacity differs")
    pairs = trimmed[1:]
    fitted = [row for row in pairs if registration.get(row, "motion.motion_fit.parameters") is not None]
    accepted = [row for row in pairs if row["motion"].get("accepted") is True]
    diagnostic = dict(pair_count=672, fit_accepted_pairs=len(accepted), fitted_pairs=len(fitted),
        feature_supply=registration.metrics(pairs, "motion.correspondence_metrics", registration.FEATURE_FIELDS),
        all_fitted=registration.metrics(fitted, "motion.motion_fit.metrics", registration.FIT_FIELDS),
        accepted_fits=registration.metrics(accepted, "motion.motion_fit.metrics", registration.FIT_FIELDS),
        pva_work_ms=registration.timing_summary(trimmed)["pva_work_ms"],
        saved_residual_summaries_are_inlier_only=True, independent_geometric_accuracy=False)
    failures = [attempt for attempt in receipt["motion_attempts"] if attempt["error"] is not None]
    full["motion_exception_classification"] = dict(total_failure_flags=len(failures),
        expected_unavailable=sum(attempt["expected_unavailable"] is True for attempt in failures),
        unexpected_runtime_errors=sum(attempt["expected_unavailable"] is not True for attempt in failures),
        legacy_counter_name="counts.pva_runtime_errors counts all motion.pva_failure flags, including expected unavailability")
    return dict(receipt=receipt, report=report, launch=launch, full=full, selected=selected,
                journal=paths["journal"], registration=diagnostic)


def validate_target_rows(rows):
    require([r["frame_index"] for r in rows] == list(range(430, 465)), "target row inventory differs")
    for row in rows:
        require(type(row["segment"]) is int and type(row["detection_ready"]) is bool, "invalid target segment/readiness")
        for track in row["tracks"]:
            require(type(track.get("segment")) is int and track["segment"] == row["segment"]
                    and type(track.get("qualified_moving")) is bool and type(track.get("measured")) is bool
                    and isinstance(track.get("track_id"), str) and track["track_id"].split(":")[0] in ("bright", "dark"),
                    "invalid target track identity/measurement flags")
            if track["measured"] and track["qualified_moving"]:
                xy = track["measurement_source_xy"]
                require(isinstance(xy, list) and len(xy) == 2
                        and all(type(v) in (int, float) and math.isfinite(v) for v in xy), "invalid actual target coordinates")


def workload(result):
    counts = result["counts"]
    measured, predicted = counts["qualified_measured_states"], counts["qualified_predicted_states"]
    ready = counts["ready_frames"]
    result["workload"] = dict(measured_states=measured, predicted_states=predicted, total_states=measured + predicted,
        ready_frame_denominator=ready, measured_per_ready=measured / ready if ready else None,
        predicted_per_ready=predicted / ready if ready else None,
        total_per_ready=(measured + predicted) / ready if ready else None,
        per_ready_is_total_state_count_divided_by_ready_frames=True,
        states_are_not_distinct_objects_or_false_positives=True)
    return result


def decision(clips):
    retention = clips["0240"]["selection"]["positive_pass"]
    gates = dict(both_clips_at_least_95pct_ready=all(clips[c]["selection"]["ready_fraction"] >= .95 for c in CLIPS),
        no_motion_failure_flags_conservative=all(clips[c]["selection"]["counts"]["pva_runtime_errors"] == 0 for c in CLIPS),
        all_35_coherent_actual_qualified_dark=retention["reference_frames"] == 35
            and retention["best_coherent_identity_frames"] == 35 and retention["preservation_guard_passed"] is True)
    return dict(gates=gates, declared_development_gates_passed=all(gates.values()), promotion_allowed=False,
                independent_recall_available=False, false_positive_rate_available=False)


def run(baseline_root, supply_root, selection_root, regression, output):
    roots, regression, output = validate_scope(baseline_root, supply_root, selection_root, regression, output)
    hashes = {}
    bind(Path(__file__).resolve(), hashes)
    comparison, registration = helpers(hashes)
    selection_freeze = load_selection_bundle(roots["selection"], registration, hashes)
    bind(regression, hashes, REGRESSION_SHA)
    intake = read(regression)
    records, positive = validate_intake(intake, registration)
    bind(SUPPLY_COMPARISON, hashes, SUPPLY_COMPARISON_SHA)
    prior = read(SUPPLY_COMPARISON)
    require(prior["schema"] == "seaqr.feature-supply.comparison.v1", "wrong prior supply comparison")
    results = {}
    for clip in CLIPS:
        baseline_pins = {k: records[clip]["artifacts"][k]["sha256"] for k in FILES}
        supply_pins = {}
        for key, suffix in FILES.items():
            relative = "../outputs/seaqr_feature_supply_20260929/evidence/" + clip + "/" + suffix
            supply_pins[key] = prior["input_sha256"][relative]
        baseline = load_arm(roots["baseline"], "baseline", clip, baseline_pins, hashes, comparison, registration)
        supply = load_arm(roots["supply"], "supply", clip, supply_pins, hashes, comparison, registration)
        selection = load_arm(roots["selection"], "selection", clip, None, hashes, comparison, registration,
            supply["receipt"]["feature_adapter"]["original_motion_configuration"], supply["receipt"]["global_configuration"], selection_freeze)
        require(selection["receipt"]["input_sha256"]["baseline_inputs"] == baseline["receipt"]["input_sha256"],
                "selection baseline dependency inputs differ")
        results[clip] = {}
        for arm, data in (("baseline", baseline), ("supply", supply), ("selection", selection)):
            for key in ("configuration", "code_sha256", "package_sha256"):
                require(data["launch"][key] == baseline["launch"][key], "unchanged launch differs: " + key)
            for key in ("adapters", "libraries", "v29_freeze_sha256", "tracking_transformed_sha256"):
                require(data["receipt"][key] == baseline["receipt"][key], "unchanged execution dependency differs: " + key)
            full = workload(data["full"])
            full.update(processed_fps=data["report"]["processed_fps"], elapsed_seconds=data["report"]["elapsed_seconds"],
                stage_timings_ms=data["report"]["timings_ms"], timing_is_uncontrolled_cross_run_diagnostic=True,
                timing_semantics=data["report"]["timing_semantics"], timing_instrumentation=data["receipt"]["timing_instrumentation"],
                registration=data["registration"])
            for key in ("processed_fps", "elapsed_seconds"):
                require(type(full[key]) in (int, float) and math.isfinite(full[key]) and full[key] > 0, "invalid throughput")
            if clip == "0240":
                full["positive_pass"] = comparison.retention(data["selected"], positive["baseline_measurements"], radius=8, polarity="dark")
                burst, _ = comparison.collect(data["journal"], 50, 105)
                full["burst_50_105"] = workload(burst)
            results[clip][arm] = full
        require(results[clip]["baseline"]["counts"] == prior["clips"][clip]["baseline"]["counts"]
                and results[clip]["supply"]["counts"] == prior["clips"][clip]["candidate"]["counts"], "old-arm summary parity differs")
    result = dict(schema="seaqr.feature-selection.comparison.v1", input_sha256=hashes, clips=results,
        assessment=decision(results), production_changed=False, new_model_fits=False, media_read=False,
        physical_airborne_class_from_user_review_not_algorithm=True, nuisance_scopes_are_verified_negative=False,
        interpretation=["Three arms share source bytes and detector/tracker/global-motion settings; the feature algorithms differ.",
            "The35 reference positions are frozen baseline actual measurements, not independent per-frame truth or recall.",
            "Qualified state totals and per-ready ratios measure review workload, not distinct objects or false positives.",
            "Legacy pva_runtime_errors counts every motion-failure flag; expected unavailability is classified separately from unexpected runtime errors.",
            "Stages overlap decode and nested work; do not sum durations. Throughput is an uncontrolled cross-run observation.",
            "Passing these development gates does not authorize production promotion or replace older-reference regression."])
    require(all(sha(path) == digest for path, digest in hashes.items()), "comparison inputs changed")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "supply-root", "selection-root", "regression", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args()))["assessment"]))
