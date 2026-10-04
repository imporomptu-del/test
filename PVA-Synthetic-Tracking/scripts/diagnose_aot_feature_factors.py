#!/usr/bin/env python3
"""Fixed 5-arm x 21-case AOT feature experiment, never a production promotion.

The legacy arm bridges capacity only. The complete-capacity factorial separates
feature image scale from Harris-only S16 gain. No detector, fallback, retry,
threshold tuning, clock write, or change to any previous experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from unittest.mock import patch

SCHEMA = "seaqr.aot.feature-factors.v1"
PLAN_SCHEMA = "seaqr.aot.feature-factors-plan.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_aot_factors_20260927_[A-Za-z0-9]{6}"
INPUT_WORKSPACE = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
HELPER_SHA = "bef64132f13112435d006642413fa706bd0e70f3a19e4001ab0d382518e6c2e4"
HARNESS_SHA = "ccbe0ddca4b15b7cbb05fcc3a1b7f05dcb099e90e34c296b16147aeac842c093"
METHOD_SHA = "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733"
BASELINE_JOURNAL_SHA = "f1b23ea70b758bea4fe5079ae897e7aa50fc338d293daa91fa6c0125b78d1211"
REFERENCE_RESULT_SHA = "b415844215b32b692fe9d55a80e64674f6bbdadfbd8913c525dba62c5a75c844"
CURRENT_INDICES = (1, 43, 86, 128, 171, 213, 256, 299)
CONTROL_NAMES = ("high_contrast_static", "high_contrast_translated", "low_contrast_static",
                 "low_contrast_translated", "flat_static")
CONTROL_SEED = 20260927
WIDTH, HEIGHT, PAIR_COUNT, MAX_HARRIS = 2448, 2048, 105, 78899
ARMS = (
    dict(id="half_gain1_legacy", feature_image_scale=0.5, harris_gain=1,
         harris_capacity_policy="legacy_default", harris_capacity=8192),
    dict(id="half_gain1_complete", feature_image_scale=0.5, harris_gain=1,
         harris_capacity_policy="complete_grid", harris_capacity=19866),
    dict(id="half_gain16_complete", feature_image_scale=0.5, harris_gain=16,
         harris_capacity_policy="complete_grid", harris_capacity=19866),
    dict(id="full_gain1_complete", feature_image_scale=1.0, harris_gain=1,
         harris_capacity_policy="complete_grid", harris_capacity=78899),
    dict(id="full_gain16_complete", feature_image_scale=1.0, harris_gain=16,
         harris_capacity_policy="complete_grid", harris_capacity=78899),
)
GAIN_ANCHOR = '        conversion = {"offset": -32768.0} if uses_u16 else {}\n'
GAIN_REPLACEMENT = ('        conversion = {"offset": -32768.0} if uses_u16 '
                    'else {"scale": 16.0, "offset": 0.0}\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scope_path(workspace):
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside exact factor workspace scope")
    return Path(workspace)


def workspace_guard(workspace):
    workspace = scope_path(workspace)
    require(os.geteuid() != 0, "never run feature factors as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked factor workspace")
    for name in ("result.json", "failure.json"):
        require(not (workspace / name).exists() and not (workspace / name).is_symlink(),
                "existing result/failure; refusing overwrite")
    require(INPUT_WORKSPACE.is_dir() and not INPUT_WORKSPACE.is_symlink()
            and (INPUT_WORKSPACE / "input").is_dir()
            and not (INPUT_WORKSPACE / "input").is_symlink(), "missing/linked pilot input")
    return workspace


def case_ids():
    return [f"aot_{i:03d}_{kind}" for i in CURRENT_INDICES for kind in ("adjacent", "stationary")] + list(CONTROL_NAMES)


def planned_calls():
    return [(case, arm["id"]) for case in case_ids() for arm in ARMS]


def validate_manifest(manifest, hashes):
    fixed = dict(schema=PLAN_SCHEMA, input_workspace=str(INPUT_WORKSPACE),
        baseline_harness_sha256=HARNESS_SHA, helper_sha256=HELPER_SHA,
        baseline_journal_sha256=BASELINE_JOURNAL_SHA,
        reference_diagnostic_result_sha256=REFERENCE_RESULT_SHA,
        current_indices=list(CURRENT_INDICES), control_names=list(CONTROL_NAMES),
        control_seed=CONTROL_SEED, include_stationary_counterfactuals=True,
        motion_pair_calls=PAIR_COUNT, arms=list(ARMS))
    require(all(manifest.get(k) == v for k, v in fixed.items()), "fixed factorial plan differs")
    require(type(manifest["control_seed"]) is int and type(manifest["motion_pair_calls"]) is int
            and manifest["include_stationary_counterfactuals"] is True
            and all(type(i) is int for i in manifest["current_indices"])
            and all(type(arm["harris_gain"]) is int and type(arm["harris_capacity"]) is int
                    and type(arm["feature_image_scale"]) in (int, float) for arm in manifest["arms"]),
            "invalid fixed plan numeric/bool types")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        require(isinstance(hashes.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", hashes[key])
                and manifest.get(key) == hashes[key], f"factor plan hash differs: {key}")


def artifact_hashes(workspace):
    paths = dict(script_sha256=workspace / "diagnose_aot_feature_factors.py",
                 tests_sha256=workspace / "test_aot_feature_factors.py", plan_sha256=workspace / "plan.md")
    require(Path(__file__).resolve() == paths["script_sha256"], "script must execute from factor workspace")
    for path in paths.values():
        require(path.is_file() and not path.is_symlink(), f"missing/linked artifact: {path}")
    return {key: sha(path) for key, path in paths.items()}


def load_helper(workspace):
    path = workspace / "diagnose_aot_features.py"
    require(path.is_file() and not path.is_symlink() and sha(path) == HELPER_SHA,
            "original diagnostic helper identity differs")
    spec = importlib.util.spec_from_file_location("_frozen_aot_feature_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def verify_references(workspace):
    paths = {"helper_sha256": (workspace / "diagnose_aot_features.py", HELPER_SHA),
             "baseline_journal_sha256": (INPUT_WORKSPACE / "run/frames.jsonl", BASELINE_JOURNAL_SHA),
             "reference_diagnostic_result_sha256": (workspace / "reference_diagnostic_result.json", REFERENCE_RESULT_SHA)}
    result = {}
    for key, (path, digest) in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing/linked reference: {key}")
        result[key] = sha(path)  # Identity only; do not parse old journal/annotations/results.
        require(result[key] == digest, f"original reference identity differs: {key}")
    return result


def resolve_config(baseline, arm):
    require(arm in ARMS, "undeclared factor arm")
    result = replace(baseline, feature_image_scale=arm["feature_image_scale"],
                     harris_capacity_policy=arm["harris_capacity_policy"])
    before, after = asdict(baseline), asdict(result)
    changes = {key: dict(before=before[key], after=after[key]) for key in before if before[key] != after[key]}
    require(set(changes) <= {"feature_image_scale", "harris_capacity_policy"}, "nonfactor motion config changed")
    return result, changes


def instrument_source(source, gain, helper):
    require(gain in (1, 16) and type(gain) is int, "undeclared Harris gain")
    require(hashlib.sha256(source.encode()).hexdigest() == METHOD_SHA, "original generated source differs")
    require(source.count(GAIN_ANCHOR) == 1, "Harris conversion anchor differs")
    transformed = helper.instrument_source(source)
    if gain == 16:
        transformed = transformed.replace(GAIN_ANCHOR, GAIN_REPLACEMENT)
    recovered = transformed.replace(GAIN_REPLACEMENT, GAIN_ANCHOR) if gain == 16 else transformed
    for _, stage, indent in helper.HOOKS:
        recovered = recovered.replace(" " * indent + f'self._aot_capture.record("{stage}", locals())\n', "")
    require(recovered == source, "factor transformation changed nonfactor estimator statements")
    return transformed


def raw_array(array, helper):
    import numpy as np
    array = np.asarray(array)
    require(array.size <= MAX_HARRIS * 2 and array.dtype.kind in "uifb", "unbounded/non-numerical factor array")
    return dict(dtype=str(array.dtype), shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(), values=helper.finite(array.tolist()))


def conversion_comparison(proxy_u8, s16, gain):
    import numpy as np
    require(type(gain) is int and gain in (1, 16), "undeclared Harris gain")
    require(proxy_u8.dtype == np.uint8 and s16.dtype == np.int16 and proxy_u8.shape == s16.shape,
            "factor conversion representation differs")
    expected = proxy_u8.astype(np.int32) * gain
    difference = s16.astype(np.int32) - expected
    return dict(gain=gain, exact_expected_equality=bool(np.all(difference == 0)),
        unequal_pixels=int(np.count_nonzero(difference)), maximum_absolute_error=int(np.abs(difference).max()),
        expected_range=[int(expected.min()), int(expected.max())],
        actual_range=[int(s16.min()), int(s16.max())], representable_without_clipping=bool(
            expected.min() >= -32768 and expected.max() <= 32767),
        s16_boundary_pixels=int(np.count_nonzero((s16 == -32768) | (s16 == 32767))))


def expected_backends(arm):
    return dict(intensity_conversion="CPU", motion_image_rescale="CUDA" if arm["feature_image_scale"] == 0.5 else "none",
                gaussian_pyramid="PVA", harris_input_conversion="CUDA", harris="PVA",
                optical_flow_pyrlk="PVA", cpu_fallback=False)


def native_center_points(points, motion_size):
    """Same pixel-center mapping as the frozen coordinate lift, for raw audits."""
    import numpy as np
    result = np.asarray(points, dtype=np.float64).copy()
    require(result.ndim == 2 and result.shape[1] == 2, "points must be Nx2")
    result[:, 0] = (result[:, 0] + 0.5) * WIDTH / motion_size[0] - 0.5
    result[:, 1] = (result[:, 1] + 0.5) * HEIGHT / motion_size[1] - 0.5
    return result


class Capture:
    def __init__(self, helper, arm):
        helper.Capture.__init__(self)
        self.helper, self.arm = helper, arm

    def copy_vpi(self, image):
        return self.helper.Capture.copy_vpi(image)

    def record(self, stage, state):
        import numpy as np
        if stage in ("pixels", "flow"):
            self.helper.Capture.record(self, stage, state)
            if stage == "pixels":
                require(all(self.data[stage][name]["native"]["sha256"]
                            == self.data[stage][name]["feature"]["sha256"] for name in ("previous", "current")),
                        "factor changed native U8 feature pixels")
            return
        try:
            require(stage not in self.stages, "duplicate factor capture stage")
            size = (round(WIDTH * self.arm["feature_image_scale"]),
                    round(HEIGHT * self.arm["feature_image_scale"]))
            if stage == "proxy":
                previous, current = (self.copy_vpi(state[name + "_motion"]) for name in ("previous", "current"))
                require(previous.dtype == current.dtype == np.uint8
                        and previous.shape == current.shape == (size[1], size[0]), "factor U8 proxy differs")
                self.previous_proxy = previous
                self.data[stage] = dict(previous=self.helper.image_summary(previous),
                                        current=self.helper.image_summary(current))
                if self.arm["feature_image_scale"] == 1.0:
                    require(all(self.data[stage][name]["sha256"]
                                == self.data["pixels"][name]["native"]["sha256"]
                                for name in ("previous", "current")), "native-scale proxy changed source pixels")
                expected = expected_backends(self.arm)
                require(state["rescale_backend"] == expected["motion_image_rescale"]
                        and state["pyramid_backend_name"] == "PVA"
                        and state["config"].optical_flow_backend == "PVA", "unexpected factor backend")
                self.data["backends_requested"] = expected
            elif stage == "s16":
                previous = self.copy_vpi(state["previous_s16"])
                comparison = conversion_comparison(self.previous_proxy, previous, self.arm["harris_gain"])
                self.data[stage] = dict(previous=self.helper.image_summary(previous), conversion=comparison)
                require(comparison["exact_expected_equality"] and comparison["representable_without_clipping"]
                        and comparison["s16_boundary_pixels"] == 0, "Harris-only gain mapping/clipping differs")
                self.previous_proxy = None
            elif stage == "harris":
                count, capacity = state["detected_count"], self.arm["harris_capacity"]
                expected_capacity = None if self.arm["harris_capacity_policy"] == "legacy_default" else capacity
                require(state["harris_capacity"] == expected_capacity and 0 <= count <= capacity,
                        "factor Harris capacity/allocation differs")
                points = self.copy_vpi(state["features"]) if count else np.empty((0, 2), np.float32)
                scores = self.copy_vpi(state["scores"]).reshape(-1) if count else np.empty(0, np.uint32)
                require(points.shape == (count, 2) and points.dtype == np.float32
                        and scores.shape == (count,) and scores.dtype == np.uint32, "raw Harris representation differs")
                roundtrip = scores.astype(np.float32).astype(np.float64) - scores.astype(np.float64)
                self.data[stage] = dict(raw_count=count, output_capacity=capacity,
                    capacity_policy=self.arm["harris_capacity_policy"], capacity_saturation_observed=count == capacity,
                    coordinates=raw_array(points, self.helper), scores=raw_array(scores, self.helper),
                    native_center_coordinates=raw_array(native_center_points(points, size), self.helper),
                    raw_score_minimum=int(scores.min()) if count else None,
                    raw_score_maximum=int(scores.max()) if count else None,
                    max_u32_score_count=int(np.count_nonzero(scores == np.iinfo(np.uint32).max)),
                    float32_rounding_changed_scores=int(np.count_nonzero(roundtrip)),
                    float32_rounding_maximum_absolute_error=float(np.abs(roundtrip).max()) if count else 0.0)
                # The original complete_grid exception executes immediately after
                # this hook, preserving the observed raw arrays on exhaustion.
            elif stage == "selection":
                indices = np.asarray(state["selected_indices"])
                self.data[stage] = dict(selected_count=len(indices),
                    eligible_mask=raw_array(state["eligible_mask"], self.helper),
                    selected_indices=raw_array(indices, self.helper),
                    coordinates=raw_array(state["detected_points"][indices], self.helper),
                    scores_float32=raw_array(state["detected_scores"][indices], self.helper),
                    exclusions=self.helper.finite(state["feature_exclusions"]))
            else:
                raise ValueError("unknown factor capture stage")
            self.stages.append(stage)
            self.data["last_production_timings_ms"] = dict(state["timings"])
        except BaseException as exc:
            self.error = repr(exc)
            raise RuntimeError(f"factor capture failed at {stage}: {exc}") from exc


def scientific_gates(row, expected=None):
    """Scientific outcomes never become permission to retry/tune/promote."""
    raw_count = row.get("capture", {}).get("harris", {}).get("raw_count")
    if row.get("case_id", row.get("id")) == "flat_static":
        return dict(kind="flat_negative", passed=raw_count == 0, raw_harris_zero=raw_count == 0,
                    motion_estimate_success=False)
    fit = row.get("global_fit") or {}
    accepted = fit.get("quality_status") == "accepted"
    if expected is None:
        return dict(kind="real_no_motion_truth", original_fit_accepted=accepted,
                    passed=None, motion_truth_verified=False)
    parameters = fit.get("parameters") or {}
    x, y = parameters.get("translation_x_px"), parameters.get("translation_y_px")
    vector_error = math.hypot(x - expected[0], y - expected[1]) if (
        type(x) in (int, float) and type(y) in (int, float)
        and math.isfinite(x) and math.isfinite(y)) else None
    errors = row.get("generated_control_error") or {}
    median, maximum = errors.get("median_error_px"), errors.get("maximum_error_px")
    criteria = dict(original_fit_accepted=accepted,
        translation_vector_error_at_most_0_1=vector_error is not None and vector_error <= 0.1,
        at_least_30_interior_points=type(errors.get("accepted_interior_points")) is int
            and errors["accepted_interior_points"] >= 30,
        interior_median_error_at_most_0_1=type(median) in (int, float) and math.isfinite(median) and 0 <= median <= 0.1,
        interior_maximum_error_at_most_0_5=type(maximum) in (int, float) and math.isfinite(maximum) and 0 <= maximum <= 0.5)
    return dict(kind="known_motion", expected_displacement_xy=list(expected),
                translation_vector_error_px=vector_error, criteria=criteria, passed=all(criteria.values()))


def evaluate_pair(helper, reuse, motion_config, global_config, previous_pixels, current_pixels,
                  row, expected=None):
    from tiny_target.types import Frame, TimestampSource
    from tiny_target.motion import PvaMotionError, fit_global_motion
    arm = row["arm"]
    capture, estimator = Capture(helper, arm), None
    previous = Frame(previous_pixels, row["previous_index"] * 100_000_000, row["previous_index"],
                     "visible-baseline", 8, TimestampSource.CONTAINER_RATE)
    current = Frame(current_pixels, row["current_index"] * 100_000_000, row["current_index"],
                    "visible-baseline", 8, TimestampSource.CONTAINER_RATE)
    before = [hashlib.sha256(frame.image.tobytes(order="C")).hexdigest() for frame in (previous, current)]
    row.update(previous_metadata=previous.metadata_dict(), current_metadata=current.metadata_dict(),
        capture=capture.data, native_pixel_sha256=dict(previous=before[0], current=before[1]),
        expected_unavailable=False, error=None, completed=False, global_fit=None)
    try:
        estimator = reuse.ReuseMotionV12(motion_config)
        estimator._aot_capture = capture
        try:
            correspondence = estimator.estimate(previous, current)
        except PvaMotionError as exc:
            expected_error = any(text in str(exc) for text in ("zero features", "No finite in-bounds"))
            row.update(error=repr(exc), expected_unavailable=expected_error and capture.error is None,
                       outcome="motion_unavailable")
            require(row["expected_unavailable"] and not estimator.failed,
                    "unexpected backend/capture/capacity failure")
        else:
            require(correspondence.backends == expected_backends(arm), "unexpected factor motion fallback")
            row.update(outcome="correspondences_returned", correspondence=helper.finite(correspondence.to_dict()),
                       global_fit=helper.finite(fit_global_motion(correspondence, global_config).to_dict()))
            if expected is not None:
                row["generated_control_error"] = helper.control_error(correspondence, expected)
                error = row["generated_control_error"]
                error.update(excluded_from_interior=error["all_accepted_points"] - error["accepted_interior_points"],
                    interior_retained_fraction=error["accepted_interior_points"] / error["all_accepted_points"]
                    if error["all_accepted_points"] else 0.0,
                    attrition_note="Post-hoc interior mask includes observed current-point bounds; inspect attrition alongside survivor error.")
        require(capture.error is None and capture.stages[:4] == ["pixels", "proxy", "s16", "harris"],
                "required factor observations missing")
        required = ["pixels", "proxy", "s16", "harris"]
        if capture.data["harris"]["raw_count"]:
            required.append("selection")
        if not row["expected_unavailable"]:
            required.append("flow")
        require(capture.stages == required and not estimator.failed
                and estimator.hits == 0 and estimator.misses == 1, "factor stage/lifecycle differs")
        require(before == [hashlib.sha256(frame.image.tobytes(order="C")).hexdigest()
                           for frame in (previous, current)], "factor mutated native pixels")
        row.update(completed=True, scientific_gates=scientific_gates(row, expected))
    except BaseException as exc:
        row.update(completed=False, unexpected_error=repr(exc))
        raise
    finally:
        row.update(captured_stages=capture.stages, capture_error=capture.error)
        capture.previous_proxy = None
        if estimator is not None:
            try:
                estimator.close()
                require(estimator.closed, "factor estimator did not close")
            except BaseException as exc:
                row.update(completed=False, cleanup_error=repr(exc))
                raise
            finally:
                row["lifecycle"] = dict(closed=estimator.closed, failed=estimator.failed,
                    hits=estimator.hits, misses=estimator.misses, resets=estimator.resets)
    return row


def comparisons(rows, helper):
    require([(row["case_id"], row["arm"]["id"]) for row in rows] == planned_calls(),
            "factor case/arm matrix differs")
    counterfactuals = {arm["id"]: helper.compare_counterfactuals(
        [row for row in rows if row["arm"]["id"] == arm["id"]]) for arm in ARMS}
    gain_inputs, capacity = [], []
    for case in case_ids():
        cases = [row for row in rows if row["case_id"] == case]
        for first, second in ((cases[1], cases[2]), (cases[3], cases[4])):
            fields = {f"{stage}_{name}": first["capture"][stage][name]["sha256"]
                      == second["capture"][stage][name]["sha256"]
                      for stage in ("proxy",) for name in ("previous", "current")}
            fields["native_previous"] = first["native_pixel_sha256"]["previous"] == second["native_pixel_sha256"]["previous"]
            fields["native_current"] = first["native_pixel_sha256"]["current"] == second["native_pixel_sha256"]["current"]
            gain_inputs.append(dict(case_id=case, scale=first["arm"]["feature_image_scale"], equal=fields))
            require(all(fields.values()), "Harris gain arm unexpectedly changed native/U8 proxy input")
        a, b = (row["capture"]["harris"] for row in cases[:2])
        count = a["raw_count"]
        capacity.append(dict(case_id=case, legacy_raw_count=count, complete_raw_count=b["raw_count"],
            legacy_capacity_reached=a["capacity_saturation_observed"],
            exact_legacy_prefix_coordinates=a["coordinates"]["values"] == b["coordinates"]["values"][:count],
            exact_legacy_prefix_scores=a["scores"]["values"] == b["scores"]["values"][:count]))
    return dict(real_stationary=counterfactuals, gain_input_invariance=gain_inputs, capacity_bridge=capacity)


def cases(helper, frames, source_rows):
    for current in CURRENT_INDICES:
        for stationary in (False, True):
            case = f"aot_{current:03d}_" + ("stationary" if stationary else "adjacent")
            row = dict(case_id=case, kind="aot_stationary_counterfactual" if stationary else "aot_adjacent",
                synthetic=stationary, previous_index=current - 1, current_index=current,
                source_previous={key: source_rows[current - 1][key] for key in ("source_frame", "timestamp_ns", "img_name")},
                source_current={key: source_rows[current][key] for key in ("source_frame", "timestamp_ns", "img_name")},
                current_pixels_from_index=current - 1 if stationary else current,
                acquisition_metadata_note="Synthetic rows replace current pixels only, not frame identity/timing metadata.")
            yield row, frames[current - 1], frames[current - 1 if stationary else current], (0, 0) if stationary else None
    for name, previous, current, expected in helper.generated_controls():
        yield dict(case_id=name, kind="generated_control", synthetic=True, previous_index=0, current_index=1,
            generator=dict(seed=CONTROL_SEED, block_size_px=32, pattern_levels=16,
                high_mapping="16 + 14 * pattern", low_mapping="120 + pattern", translation_xy_px=list(expected),
                wrapping=False, border_fill=128, native_resolution=[WIDTH, HEIGHT], native_dtype="uint8")), previous, current, expected


def run(workspace):
    workspace = workspace_guard(workspace)
    helper = load_helper(workspace)
    manifest = helper.read(workspace / "manifest.json")
    artifact_before = artifact_hashes(workspace)
    validate_manifest(manifest, artifact_before)
    references_before = verify_references(workspace)
    manifest_sha = sha(workspace / "manifest.json")
    receipt = dict(schema=SCHEMA, passed=False, error=None, workspace=str(workspace),
        input_workspace=str(INPUT_WORKSPACE), manifest_sha256=manifest_sha,
        artifact_sha256_before=artifact_before, references_sha256_before=references_before,
        planned_motion_pair_calls=PAIR_COUNT, completed_motion_pair_calls=0, rows=[], arms=list(ARMS),
        current_indices=list(CURRENT_INDICES), control_names=list(CONTROL_NAMES), control_seed=CONTROL_SEED,
        production_algorithm_changed=False, diagnostic_algorithm_changed=True,
        allowed_changes=["feature_image_scale", "harris_capacity_policy", "Harris-only S16 gain"],
        detector_run=False, production_promotion=False, auto_selection=False, retries=0,
        annotations_used=False, raw16_accessed=False, private_camera_media_accessed=False, clock_writes=False,
        interpretation=dict(
            scope="Original baseline/reference artifacts preserved. Feature experiment only; original motion-quality gates retained.",
            pairing="Fresh pair-local preparation cache, unchanged process-global VPI state. Not full-clip temporal equivalence.",
            timing="Nominal10Hz; original acquisition metadata retained separately. Diagnostic host readbacks prohibit speed comparison.",
            amplitude="Gain16 changes Harris response selectivity, not source information/SNR or U8 pyramid/flow intensities. Exact no-input-clipping does not establish internal Harris arithmetic precision.",
            scale="Fixed NMS/LK/pyramid settings have different native-pixel footprints at different image scales.",
            capacity="Legacy8192 limit is censored scientific evidence; explicit complete capacity exhaustion is an integrity stop.",
            gates="Known-motion truth gates and flat-negative gates are observations, not execution-pass gates. Real adjacent AOT has no independent camera-motion truth.",
            output="Bounded105 rows; raw coordinates/scores fully retained up to78899 each, so result JSON may exceed32MiB. Metadata inputs remain bounded32MiB."))
    harness = runtime_info = cv2 = None
    original_threads = None
    with (workspace / "result.json").open("x", encoding="utf-8") as stream:
        try:
            harness = helper.load_harness()
            receipt["baseline_harness_sha256_before"] = helper.sha(INPUT_WORKSPACE / "run_aot_frozen_baseline.py")
            require(receipt["baseline_harness_sha256_before"] == HARNESS_SHA, "baseline harness identity differs")
            values, input_hashes = harness.inputs(INPUT_WORKSPACE)
            modules, identities = harness.dependencies(values["reference_launch_v34.json"])
            preflight = harness.read(INPUT_WORKSPACE / "preflight.json")
            harness.validate_preflight(preflight, input_hashes, identities, HARNESS_SHA, INPUT_WORKSPACE)
            receipt.update(input_sha256_before=input_hashes, frozen_dependencies=identities,
                baseline_preflight_sha256=sha(INPUT_WORKSPACE / "preflight.json"))
            runtime_info = modules["profile_visible_interaction_v30"].runtime_info
            receipt["runtime_before"] = runtime_info()
            harness.runtime_check(receipt["runtime_before"], values["reference_runtime.json"])
            receipt["clock_policy_before"] = harness.clock_policy_snapshot()
            import cv2
            from tiny_target.config import load_config
            from tiny_target.motion import PvaMotionConfig, GlobalMotionConfig
            original_threads = cv2.getNumThreads()
            receipt["opencv_thread_policy"] = dict(original=original_threads, diagnostic=None, restored=None,
                                                   process_local_only=True)
            cv2.setNumThreads(2)
            receipt["opencv_thread_policy"]["diagnostic"] = cv2.getNumThreads()
            harness.runtime_check(runtime_info(), values["reference_runtime.json"], after=True)
            raw = load_config(harness.MOTION_CONFIG).raw
            baseline = PvaMotionConfig.from_mapping(raw.get("motion"))
            global_config = GlobalMotionConfig.from_mapping(raw.get("global_motion"))
            require(baseline.feature_image_scale == 0.5 and baseline.minimum_accepted_features == 30
                    and baseline.minimum_grid_coverage == 0.2 and baseline.harris_capacity_policy == "legacy_default"
                    and baseline.flow_status_policy == "legacy_default"
                    and global_config.minimum_correspondences == global_config.minimum_inliers == 30,
                    "original feature/fit gates differ")
            configurations = {arm["id"]: resolve_config(baseline, arm) for arm in ARMS}
            receipt.update(original_motion_configuration=asdict(baseline),
                global_configuration=asdict(global_config),
                arm_motion_configurations={key: dict(configuration=asdict(value[0]), changes=value[1])
                                           for key, value in configurations.items()})
            reuse = modules["motion_reuse_v12"]
            original_source = reuse.generated_method()
            methods, source_info = {}, {}
            for gain in (1, 16):
                transformed = instrument_source(original_source, gain, helper)
                namespace = dict(vars(reuse.pva))
                exec(compile(transformed, str(workspace / "diagnose_aot_feature_factors.py")
                             + f":gain{gain}", "exec"), namespace)
                methods[gain] = namespace["_estimate_v12"]
                source_info[str(gain)] = dict(original_method_sha256=METHOD_SHA,
                    transformed_method_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                    exact_original_recovered_after_reversing_gain_and_hooks=True,
                    instrumented_method_source=transformed)
            receipt["source_instrumentation"] = source_info
            frames, decoded = helper.decode_selected(harness, values["download_validation.json"]["images"])
            receipt["decode"] = decoded
            for case, previous, current, expected in cases(helper, frames, values["frozen_image_manifest.json"]["frames"]):
                for arm in ARMS:
                    row = dict(case, id=case["case_id"] + "__" + arm["id"], arm=dict(arm))
                    receipt["rows"].append(row)
                    with patch.object(reuse, "_ESTIMATE", methods[arm["harris_gain"]]):
                        evaluate_pair(helper, reuse, configurations[arm["id"]][0], global_config,
                                      previous, current, row, expected)
                    receipt["completed_motion_pair_calls"] += 1
                    print(json.dumps(dict(case=row["case_id"], arm=arm["id"],
                        completed=receipt["completed_motion_pair_calls"], raw_harris=row["capture"]["harris"]["raw_count"],
                        outcome=row["outcome"], scientific_pass=row["scientific_gates"]["passed"])), flush=True)
            require(receipt["completed_motion_pair_calls"] == PAIR_COUNT, "incomplete factor calls")
            receipt["comparisons"] = comparisons(receipt["rows"], helper)
            receipt["scientific_summary"] = {arm["id"]: [dict(case_id=row["case_id"], gates=row["scientific_gates"])
                for row in receipt["rows"] if row["arm"]["id"] == arm["id"]] for arm in ARMS}
            receipt["runtime_after"] = runtime_info()
            harness.runtime_check(receipt["runtime_after"], values["reference_runtime.json"], after=True)
            require(harness.dependencies(values["reference_launch_v34.json"])[1] == identities,
                    "frozen dependencies changed")
            receipt["frozen_dependencies_reverified"] = True
            receipt["input_sha256_after"] = harness.inputs(INPUT_WORKSPACE)[1]
            require(receipt["input_sha256_after"] == input_hashes, "pilot inputs changed")
            require(sha(INPUT_WORKSPACE / "preflight.json") == receipt["baseline_preflight_sha256"], "preflight changed")
            receipt["baseline_harness_sha256_after"] = sha(INPUT_WORKSPACE / "run_aot_frozen_baseline.py")
            require(receipt["baseline_harness_sha256_after"] == HARNESS_SHA, "baseline harness changed")
            receipt["artifact_sha256_after"] = artifact_hashes(workspace)
            receipt["references_sha256_after"] = verify_references(workspace)
            require(receipt["artifact_sha256_after"] == artifact_before
                    and receipt["references_sha256_after"] == references_before
                    and sha(workspace / "manifest.json") == manifest_sha, "factor artifacts/references changed")
            receipt["clock_policy_after"] = harness.clock_policy_snapshot()
            receipt["remote_clocks_unchanged"] = receipt["clock_policy_after"] == receipt["clock_policy_before"]
            require(receipt["remote_clocks_unchanged"], "clock controls changed")
            receipt["passed"] = True
        except BaseException as exc:
            receipt.update(passed=False, error=repr(exc))
            raise
        finally:
            try:
                if cv2 is not None and original_threads is not None:
                    cv2.setNumThreads(original_threads)
                    receipt["opencv_thread_policy"]["restored"] = cv2.getNumThreads()
                    require(cv2.getNumThreads() == original_threads, "OpenCV thread restoration failed")
                    receipt["runtime_after_thread_restore"] = runtime_info()
                    harness.runtime_check(receipt["runtime_after_thread_restore"], values["reference_runtime.json"])
                if harness is not None and "clock_policy_before" in receipt:
                    receipt["clock_policy_final"] = harness.clock_policy_snapshot()
                    require(receipt["clock_policy_final"] == receipt["clock_policy_before"], "clock controls changed at cleanup")
            except BaseException as exc:
                receipt.update(passed=False, cleanup_error=repr(exc))
            json.dump(helper.finite(receipt), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
    require(receipt["passed"], "factor integrity/cleanup gates failed")
    print(json.dumps(dict(passed=True, completed_motion_pair_calls=PAIR_COUNT,
                          result=str(workspace / "result.json"))), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    run(parser.parse_args().workspace)


if __name__ == "__main__":
    main()
