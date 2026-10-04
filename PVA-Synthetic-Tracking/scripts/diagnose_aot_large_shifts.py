#!/usr/bin/env python3
"""Predeclared 93-call larger natural-texture integer-shift diagnostic, not a detector."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
from unittest.mock import patch

SCHEMA = "seaqr.aot.large-shifts.v1"
PLAN_SCHEMA = "seaqr.aot.large-shifts-plan.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_aot_large_shifts_20260928_[A-Za-z0-9]{6}"
INPUT_WORKSPACE = Path("/tmp/seaqr_aot_pilot_20260927_aeZ0yA")
REFERENCE_FACTOR_RESULT = Path("/tmp/seaqr_aot_factors_20260927_5FZmd7/result.json")
REFERENCE_FACTOR_SHA = "aaeb0cb9898199548afe9c8303d27717c1694c596dec2b775df6355d24584f56"
FACTOR_SHA = "f0234ad5e1f4ff5686b979adea8086ab00783ef82a83c6fe6751166f42719c61"
HELPER_SHA = "bef64132f13112435d006642413fa706bd0e70f3a19e4001ab0d382518e6c2e4"
HARNESS_SHA = "ccbe0ddca4b15b7cbb05fcc3a1b7f05dcb099e90e34c296b16147aeac842c093"
BASELINE_JOURNAL_SHA = "f1b23ea70b758bea4fe5079ae897e7aa50fc338d293daa91fa6c0125b78d1211"
PREVIOUS_INDICES = (0, 42, 85, 127, 170, 212, 255, 298)
SHIFTS = ((0, 0), (1, 0), (-1, 0), (16, 0), (-16, 0), (32, 0), (-32, 0), (48, 0), (-48, 0), (32, -16), (-32, 16))
CONTROL_NAMES = ("high_contrast_static", "high_contrast_translated", "low_contrast_static",
                 "low_contrast_translated", "flat_static")
CONTROL_SEED, WIDTH, HEIGHT, PAIR_COUNT = 20260927, 2448, 2048, 93
ARM = dict(id="half_gain16_complete", feature_image_scale=0.5, harris_gain=16,
           harris_capacity_policy="complete_grid", harris_capacity=19866)
OUTPUT_NAMES = ("result.json", "result.json.gz", "compression_receipt.json", "failure.json")


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
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)), "outside exact natural-shift scope")
    return Path(workspace)


def workspace_guard(workspace):
    workspace = scope_path(workspace)
    require(os.geteuid() != 0, "never run natural shifts as root")
    require(workspace.is_dir() and not workspace.is_symlink(), "missing/linked shift workspace")
    for name in OUTPUT_NAMES:
        require(not (workspace / name).exists() and not (workspace / name).is_symlink(),
                "existing result/failure/compression output; refusing overwrite")
    require(INPUT_WORKSPACE.is_dir() and not INPUT_WORKSPACE.is_symlink()
            and (INPUT_WORKSPACE / "input").is_dir() and not (INPUT_WORKSPACE / "input").is_symlink(),
            "missing/linked fixed pilot input")
    return workspace


def case_id(index, shift):
    return f"aot_prev{index:03d}_dx{shift[0]:+d}_dy{shift[1]:+d}"


def case_ids():
    return [case_id(index, shift) for index in PREVIOUS_INDICES for shift in SHIFTS] + list(CONTROL_NAMES)


def validate_manifest(manifest, hashes):
    expected = dict(schema=PLAN_SCHEMA, input_workspace=str(INPUT_WORKSPACE),
        helper_sha256=HELPER_SHA, factor_helper_sha256=FACTOR_SHA, baseline_harness_sha256=HARNESS_SHA,
        baseline_journal_sha256=BASELINE_JOURNAL_SHA,
        reference_factor_result_sha256=REFERENCE_FACTOR_SHA, previous_indices=list(PREVIOUS_INDICES),
        shifts=[list(shift) for shift in SHIFTS], control_names=list(CONTROL_NAMES),
        control_seed=CONTROL_SEED, arm=ARM, motion_pair_calls=PAIR_COUNT)
    require(all(manifest.get(k) == v for k, v in expected.items()), "fixed natural-shift plan differs")
    require(type(manifest["control_seed"]) is int and type(manifest["motion_pair_calls"]) is int
            and all(type(i) is int for i in manifest["previous_indices"])
            and all(type(v) is int for shift in manifest["shifts"] for v in shift)
            and type(manifest["arm"]["harris_gain"]) is int
            and type(manifest["arm"]["harris_capacity"]) is int,
            "invalid fixed plan integer types")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        require(isinstance(hashes.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", hashes[key])
                and manifest.get(key) == hashes[key], f"natural-shift artifact hash differs: {key}")


def artifact_hashes(workspace):
    paths = dict(script_sha256=workspace / "diagnose_aot_large_shifts.py",
                 tests_sha256=workspace / "test_aot_large_shifts.py", plan_sha256=workspace / "plan.md")
    require(Path(__file__).resolve() == paths["script_sha256"], "script must run in guarded workspace")
    for path in paths.values():
        require(path.is_file() and not path.is_symlink(), f"missing/linked artifact: {path}")
    return {key: sha(path) for key, path in paths.items()}


def references(workspace):
    paths = {"helper_sha256": (workspace / "diagnose_aot_features.py", HELPER_SHA),
             "factor_helper_sha256": (workspace / "diagnose_aot_feature_factors.py", FACTOR_SHA),
             "reference_factor_result_sha256": (REFERENCE_FACTOR_RESULT, REFERENCE_FACTOR_SHA),
             "baseline_journal_sha256": (INPUT_WORKSPACE / "run/frames.jsonl", BASELINE_JOURNAL_SHA),
             "baseline_harness_sha256": (INPUT_WORKSPACE / "run_aot_frozen_baseline.py", HARNESS_SHA)}
    result = {}
    for key, (path, expected) in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing/linked reference: {key}")
        result[key] = sha(path)
        require(result[key] == expected, f"frozen reference differs: {key}")
    return result


def load_module(path, expected, name):
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, "frozen helper differs before import")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_raw(array):
    import numpy as np
    array = np.asarray(array)
    points = array.dtype.kind == "f" and array.dtype.itemsize == 4 and array.ndim == 2 and array.shape[1] == 2
    flags = array.dtype.kind == "u" and array.dtype.itemsize == 1 and array.ndim == 1
    require((points or flags) and array.shape[0] <= 1000,
            "invalid/bounded raw flow representation")
    data = array.tobytes(order="C")
    return dict(dtype=array.dtype.str, shape=list(array.shape), order="C", byte_count=len(data),
                sha256=hashlib.sha256(data).hexdigest(), base64=base64.b64encode(data).decode("ascii"))


def decode_raw(record):
    """Bounded lossless audit decoder, including nonfinite payloads and endianness."""
    import numpy as np
    require(isinstance(record, dict) and record.get("order") == "C", "invalid raw flow metadata")
    require(record.get("dtype") in ("<f4", ">f4", "|u1"), "invalid raw flow dtype")
    dtype, shape = np.dtype(record["dtype"]), record.get("shape")
    require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), "invalid raw flow shape")
    require((dtype.kind == "f" and len(shape) == 2 and shape[1] == 2
             or dtype.kind == "u" and len(shape) == 1) and shape[0] <= 1000, "unbounded raw flow shape")
    count = shape[0] * (2 if dtype.kind == "f" else 1) * dtype.itemsize
    encoded = record.get("base64")
    require(type(record.get("byte_count")) is int and record["byte_count"] == count
            and isinstance(encoded, str) and len(encoded) == 4 * ((count + 2) // 3), "raw flow byte count differs")
    require(isinstance(record.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", record["sha256"]), "invalid raw flow hash")
    data = base64.b64decode(encoded, validate=True)
    require(len(data) == count and hashlib.sha256(data).hexdigest() == record["sha256"], "raw flow bytes/hash differ")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def preflow_support(points, expected):
    """Support depends on selected previous points and known truth, never flow q."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) <= 1000,
            "selected previous points must be bounded Nx2")
    require(len(expected) == 2 and all(type(v) is int for v in expected), "expected shift must be integer xy")
    target = points + np.asarray(expected, dtype=np.float64)
    margin = 128
    mask = (np.isfinite(points).all(axis=1) & (points[:, 0] >= margin) & (points[:, 0] < WIDTH - margin)
            & (points[:, 1] >= margin) & (points[:, 1] < HEIGHT - margin)
            & (target[:, 0] >= margin) & (target[:, 0] < WIDTH - margin)
            & (target[:, 1] >= margin) & (target[:, 1] < HEIGHT - margin))
    return dict(selected_count=len(points), fixed_support_count=int(mask.sum()),
                selected_indices=np.flatnonzero(mask).tolist(), mask=mask.tolist(),
                expected_shift_xy=list(expected), margin_px=margin,
                denominator_basis="selected native previous p and expected p+shift; independent of observed q")


def make_capture_class(factor, helper):
    class ShiftCapture(factor.Capture):
        def __init__(self, helper_arg, arm):
            super().__init__(helper_arg, arm)
            self.expected_shift = ShiftCapture.active_shift

        def record(self, stage, state):
            super().record(stage, state)
            try:
                if stage == "selection":
                    selected = state["detected_points"][state["selected_indices"]]
                    native = factor.native_center_points(selected, state["motion_size"])
                    self.data["preflow_support"] = preflow_support(native, self.expected_shift)
                elif stage == "flow":
                    self.data["raw_flow_bytes"] = {name: encode_raw(state[name]) if state[name] is not None else None
                        for name in ("current_motion_points", "forward_status_array",
                                     "backward_motion_points", "backward_status_array")}
                    require(all(value is None or value["sha256"] == self.data["flow"][name]["sha256"]
                                for name, value in self.data["raw_flow_bytes"].items()), "flow byte/hash capture differs")
            except BaseException as exc:
                self.error = repr(exc)
                raise RuntimeError(f"natural-shift capture failed at {stage}: {exc}") from exc

    ShiftCapture.active_shift = None
    return ShiftCapture


def attach_attrition(row):
    """Count accepted survivors against fixed pre-flow support, without a new gate."""
    import numpy as np
    capture = row["capture"]
    support = capture.get("preflow_support")
    if support is None:
        row["fixed_support_attrition"] = dict(available=False, reason="selection_not_reached")
        return
    selected = capture["selection"]["coordinates"]["values"]
    accepted = (row.get("correspondence") or {}).get("correspondences", [])
    # Raw Harris coordinates are integer-valued; half-scale center mapping is exact.
    native_selected = [(2 * point[0] + 0.5, 2 * point[1] + 0.5) for point in selected]
    require(len(set(native_selected)) == len(native_selected), "duplicate selected native points")
    accepted_previous = {tuple(item["previous_xy"]) for item in accepted}
    require(accepted_previous <= set(native_selected), "accepted point not in original selection")
    survivors = sum(native_selected[index] in accepted_previous for index in support["selected_indices"])
    count = support["fixed_support_count"]
    row["fixed_support_attrition"] = dict(available=True, selected_count=support["selected_count"],
        preflow_support_count=count, accepted_support_count=survivors, lost_support_count=count - survivors,
        support_survival_fraction=survivors / count if count else None,
        all_accepted_points=len(accepted), observed_q_interior_points=(row.get("generated_control_error") or {}).get("accepted_interior_points"),
        additional_gate=False, note="Fixed pre-flow support versus flow acceptance; independent of observed-q interior crop.")
    audit = row["fixed_support_attrition"]
    raw = capture.get("raw_flow_bytes")
    if raw is None:
        audit.update(disjoint_stages={"flow_not_observed": count}, finite_forward_points=0,
                     direct_truth_error_px=[None] * count, missing_or_rejected_truth_observations=count)
        require(not accepted, "accepted points without raw flow evidence")
        return
    values = {key: decode_raw(record) if record is not None else None for key, record in raw.items()}
    p = np.asarray(native_selected, dtype=np.float32).reshape(-1, 2)
    q_proxy, flags = values["current_motion_points"], values["forward_status_array"]
    b_proxy, back_flags = values["backward_motion_points"], values["backward_status_array"]
    n = len(p)
    require(q_proxy is not None and b_proxy is not None and flags is not None and back_flags is not None
            and q_proxy.shape == b_proxy.shape == (n, 2) and flags.shape == back_flags.shape == (n,)
            and q_proxy.dtype.kind == b_proxy.dtype.kind == "f"
            and flags.dtype.kind == back_flags.dtype.kind == "u", "incomplete/malformed frozen forward-backward evidence")
    with np.errstate(invalid="ignore", over="ignore"):
        q = (q_proxy + np.float32(0.5)) * np.float32(2) - np.float32(0.5)
        b = (b_proxy + np.float32(0.5)) * np.float32(2) - np.float32(0.5)
        displacement = np.linalg.norm(q - p, axis=1)
        fb_error = np.linalg.norm(b - p, axis=1).astype(np.float32)
    tests = [
        ("forward_nonfinite", np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)),
        ("forward_status", flags == 0),
        ("out_of_bounds", (q[:, 0] >= 0) & (q[:, 0] < WIDTH) & (q[:, 1] >= 0) & (q[:, 1] < HEIGHT)),
        ("excessive_displacement", displacement <= 120.0),
        ("backward_nonfinite", np.isfinite(b).all(axis=1) & np.isfinite(fb_error)),
        ("backward_status", back_flags == 0),
        ("forward_backward_error", fb_error <= 3.0),
    ]
    alive = np.asarray(support["mask"], dtype=bool).copy()
    final_all = np.ones(n, dtype=bool)
    stages = {}
    for name, keep in tests:
        stages[name] = int(np.count_nonzero(alive & ~keep))
        alive &= keep
        final_all &= keep
    stages["accepted"] = int(alive.sum())
    require(sum(stages.values()) == count and stages["accepted"] == survivors, "fixed support stages do not partition cohort")
    require({native_selected[i] for i in np.flatnonzero(final_all)} == accepted_previous,
            "raw-byte acceptance audit differs from unchanged estimator output")
    cohort = np.asarray(support["mask"], dtype=bool)
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.linalg.norm(q.astype(np.float64) - p.astype(np.float64)
                                - np.asarray(row["expected_shift_xy"], dtype=np.float64), axis=1)
    finite = np.isfinite(errors) & np.isfinite(q).all(axis=1)
    observed = errors[cohort & finite]
    def histogram(array):
        unique, counts = np.unique(array[cohort], return_counts=True)
        return {str(int(key)): int(value) for key, value in zip(unique, counts)}
    audit.update(disjoint_stage_order=[name for name, _ in tests] + ["accepted"], disjoint_stages=stages,
        finite_forward_points=int(np.count_nonzero(cohort & np.isfinite(q).all(axis=1))),
        nonfinite_forward_points=int(np.count_nonzero(cohort & ~np.isfinite(q).all(axis=1))),
        forward_status_values=histogram(flags), backward_status_values=histogram(back_flags),
        status_readback_caveat="Status arrays are final post-bidirectional readbacks under legacy sharing policy; categories reflect the original acceptance filter, not causal stage timing.",
        direct_truth_error_px=[float(errors[i]) if alive[i] and finite[i] else None for i in support["selected_indices"]],
        valid_accepted_truth_observations=int(np.count_nonzero(alive & finite)),
        missing_or_rejected_truth_observations=count - int(np.count_nonzero(alive & finite)),
        raw_finite_coordinate_error_px=[float(errors[i]) if finite[i] else None for i in support["selected_indices"]],
        raw_finite_coordinate_observations=len(observed),
        raw_finite_coordinate_median_px=float(np.median(observed)) if len(observed) else None,
        raw_finite_coordinate_maximum_px=float(observed.max()) if len(observed) else None,
        raw_finite_coordinate_caveat="Includes failed status/filter observations; numerical zero here is not valid or successful flow.",
        raw_finite_coordinate_within_0_1_count=int(np.count_nonzero(cohort & finite & (errors <= 0.1))),
        accepted_truth_within_0_1_count=int(np.count_nonzero(alive & finite & (errors <= 0.1))),
        accepted_truth_within_0_5_count=int(np.count_nonzero(alive & finite & (errors <= 0.5))))


def generate_cases(helper, frames, source_rows):
    for index in PREVIOUS_INDICES:
        previous = frames[index]
        for shift in SHIFTS:
            current = helper.translate_no_wrap(previous, shift[0], shift[1], 128)
            yield dict(case_id=case_id(index, shift), kind="aot_known_shift", synthetic=True,
                previous_index=index, current_index=index + 1, source_previous_index=index,
                source_previous={key: source_rows[index][key] for key in ("source_frame", "timestamp_ns", "img_name")},
                expected_shift_xy=list(shift), current_pixel_origin="integer-shifted previous image, NOT actual next AOT image",
                translation=dict(wrapping=False, fill=128, resampler="none", native_integer_copy=True)), previous, current, shift
    for name, previous, current, shift in helper.generated_controls():
        yield dict(case_id=name, kind="generated_control", synthetic=True, previous_index=0, current_index=1,
            expected_shift_xy=list(shift), generator=dict(seed=CONTROL_SEED, unchanged_helper_sha256=HELPER_SHA,
                wrapping=False, border_fill=128, native_resolution=[WIDTH, HEIGHT], native_dtype="uint8")), previous, current, shift


def invariance(rows, complete=True):
    expected = case_ids() if complete else case_ids()[:len(rows)]
    require([row["case_id"] for row in rows] == expected, "fixed93 larger-shift inventory differs")
    result = []
    paths = {"native": ("pixels", "previous", "native"), "proxy": ("proxy", "previous"),
             "s16": ("s16", "previous"), "raw_harris_coordinates": ("harris", "coordinates"),
             "raw_harris_scores": ("harris", "scores"), "selected_indices": ("selection", "selected_indices")}
    for index in PREVIOUS_INDICES:
        cases = [row for row in rows if row.get("source_previous_index") == index]
        if not complete and not cases:
            continue
        require(len(cases) == len(SHIFTS), "missing natural-shift comparison row")
        checks = {}
        for label, path in paths.items():
            identities = []
            for row in cases:
                value = row["capture"]
                for key in path:
                    value = value.get(key) if value is not None else None
                identities.append(None if value is None else (value["dtype"], value["shape"], value["sha256"]))
            checks[label] = all(identity == identities[0] for identity in identities)
        require(all(checks.values()), "previous-image feature/state invariance differs across shifts")
        result.append(dict(source_previous_index=index, equal=checks,
            current_native_and_proxy=[dict(case_id=row["case_id"], expected_shift_xy=row["expected_shift_xy"],
                native_sha256=row["native_pixel_sha256"]["current"],
                proxy_sha256=row["capture"]["proxy"]["current"]["sha256"]) for row in cases]))
    return result


def compress_result(workspace):
    raw, compressed, receipt_path = (workspace / name for name in OUTPUT_NAMES[:3])
    require(raw.is_file() and not raw.is_symlink(), "raw result missing/linked")
    for path in (compressed, receipt_path):
        require(not path.exists() and not path.is_symlink(), "existing compression output; refusing overwrite")
    digest = sha(raw)
    with raw.open("rb") as source, compressed.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as zipped:
            shutil.copyfileobj(source, zipped, length=1024 * 1024)
    restored = hashlib.sha256()
    with gzip.open(compressed, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            restored.update(block)
    require(restored.hexdigest() == digest == sha(raw), "gzip verification or raw identity failed")
    receipt = dict(schema=SCHEMA + ".compression", passed=True, raw_sha256=digest,
        gzip_sha256=sha(compressed), raw_bytes=raw.stat().st_size, gzip_bytes=compressed.stat().st_size,
        decompressed_sha256=restored.hexdigest(), raw_preserved=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def run(workspace):
    workspace = workspace_guard(workspace)
    reference_before = references(workspace)
    factor = load_module(workspace / "diagnose_aot_feature_factors.py", FACTOR_SHA, "_frozen_natural_shift_factor")
    helper = load_module(workspace / "diagnose_aot_features.py", HELPER_SHA, "_frozen_natural_shift_helper")
    require(factor.ARMS[2] == ARM, "sole inherited factor arm differs")
    manifest = helper.read(workspace / "manifest.json")
    artifact_before = artifact_hashes(workspace)
    validate_manifest(manifest, artifact_before)
    manifest_sha = sha(workspace / "manifest.json")
    receipt = dict(schema=SCHEMA, passed=False, error=None, workspace=str(workspace),
        input_workspace=str(INPUT_WORKSPACE), artifact_sha256_before=artifact_before,
        references_sha256_before=reference_before, manifest_sha256=manifest_sha,
        planned_motion_pair_calls=PAIR_COUNT, completed_motion_pair_calls=0, rows=[], arm=dict(ARM),
        previous_indices=list(PREVIOUS_INDICES), shifts=[list(s) for s in SHIFTS],
        control_names=list(CONTROL_NAMES), control_seed=CONTROL_SEED,
        detector_run=False, production_promotion=False, auto_selection=False, retries=0,
        production_algorithm_changed=False, inherited_factor_algorithm_changed=False,
        annotations_used=False, raw16_accessed=False, private_camera_media_accessed=False, clock_writes=False,
        interpretation=dict(
            source="Known integer copies of approved previous AOT images, not actual next frames; no host resampling.",
            state="Fresh pair-local cache; unchanged global VPI cache and legacy flow-status policy; one worker.",
            scope="Passed means scope/integrity/93 completed calls, not scientific truth gates or airborne detection.",
            conditional_truth="Inherited truth metric conditions on accepted flow and observed-q interior bounds; fixed pre-flow cohort is reported separately, without new gates.",
            status_readbacks="Both status arrays are final post-bidirectional readbacks under legacy sharing policy, not pristine forward-pass status snapshots.",
            limitation="Copies move noise/artifacts with scene; not independent-noise, exposure-change, occlusion, parallax or real stable-camera validation.",
            timing="Nominal100ms pair timestamps; diagnostic host readbacks/hashes prohibit speed comparison."))
    harness = runtime_info = cv2 = None
    original_threads = None
    with (workspace / "result.json").open("x", encoding="utf-8") as result_file:
        try:
            harness = helper.load_harness()
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
            motion_config, differences = factor.resolve_config(baseline, ARM)
            require(motion_config.minimum_accepted_features == 30 and motion_config.minimum_grid_coverage == 0.2
                    and motion_config.max_displacement_px == 120 and motion_config.max_forward_backward_error_px == 3
                    and motion_config.forward_backward_check and motion_config.flow_status_policy == "legacy_default"
                    and global_config.minimum_correspondences == global_config.minimum_inliers == 30,
                    "frozen inherited motion/quality policy differs")
            receipt.update(original_motion_configuration=asdict(baseline), motion_configuration=asdict(motion_config),
                           motion_config_differences=differences, global_configuration=asdict(global_config))
            reuse = modules["motion_reuse_v12"]
            original = reuse.generated_method()
            transformed = factor.instrument_source(original, 16, helper)
            namespace = dict(vars(reuse.pva))
            exec(compile(transformed, str(workspace / "diagnose_aot_large_shifts.py") + ":inherited-gain16", "exec"), namespace)
            receipt["source_instrumentation"] = dict(original_method_sha256=factor.METHOD_SHA,
                transformed_method_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                exact_original_recovered_after_reversing_gain_and_hooks=True,
                instrumented_method_source=transformed)
            frames, decoded = helper.decode_selected(harness, values["download_validation.json"]["images"])
            receipt["decode"] = decoded
            capture_class = make_capture_class(factor, helper)
            with patch.object(factor, "Capture", capture_class), patch.object(reuse, "_ESTIMATE", namespace["_estimate_v12"]):
                for case, previous, current, expected in generate_cases(helper, frames, values["frozen_image_manifest.json"]["frames"]):
                    row = dict(case, id=case["case_id"], arm=dict(ARM))
                    receipt["rows"].append(row)
                    capture_class.active_shift = tuple(expected)
                    factor.evaluate_pair(helper, reuse, motion_config, global_config, previous, current, row, tuple(expected))
                    attach_attrition(row)
                    receipt["completed_motion_pair_calls"] += 1
                    if row["kind"] == "aot_known_shift" and receipt["completed_motion_pair_calls"] % len(SHIFTS) == 0:
                        receipt["previous_image_invariance"] = invariance(receipt["rows"], complete=False)
                    print(json.dumps(dict(case_id=row["case_id"], completed=receipt["completed_motion_pair_calls"],
                        raw_harris=row["capture"]["harris"]["raw_count"],
                        fit_accepted=(row.get("global_fit") or {}).get("quality_status") == "accepted",
                        scientific_pass=row["scientific_gates"]["passed"],
                        fixed_support_count=row["fixed_support_attrition"].get("preflow_support_count"),
                        accepted_support_count=row["fixed_support_attrition"].get("accepted_support_count"))), flush=True)
            require(receipt["completed_motion_pair_calls"] == PAIR_COUNT, "incomplete/excess natural-shift calls")
            receipt["previous_image_invariance"] = invariance(receipt["rows"])
            receipt["scientific_summary"] = [dict(case_id=row["case_id"], gates=row["scientific_gates"],
                fixed_support={key: value for key, value in row["fixed_support_attrition"].items()
                               if key not in ("direct_truth_error_px", "raw_finite_coordinate_error_px")}) for row in receipt["rows"]]
            receipt["runtime_after"] = runtime_info()
            harness.runtime_check(receipt["runtime_after"], values["reference_runtime.json"], after=True)
            require(harness.dependencies(values["reference_launch_v34.json"])[1] == identities, "frozen dependencies changed")
            receipt["frozen_dependencies_reverified"] = True
            receipt["input_sha256_after"] = harness.inputs(INPUT_WORKSPACE)[1]
            require(receipt["input_sha256_after"] == input_hashes, "pilot inputs changed")
            require(sha(INPUT_WORKSPACE / "preflight.json") == receipt["baseline_preflight_sha256"], "preflight changed")
            receipt["artifact_sha256_after"] = artifact_hashes(workspace)
            receipt["references_sha256_after"] = references(workspace)
            require(receipt["artifact_sha256_after"] == artifact_before
                    and receipt["references_sha256_after"] == reference_before
                    and sha(workspace / "manifest.json") == manifest_sha, "experiment artifacts/references changed")
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
            json.dump(helper.finite(receipt), result_file, indent=2, allow_nan=False)
            result_file.write("\n")
            result_file.flush()
    require(receipt["passed"], "natural-shift integrity/cleanup gates failed")
    compression = compress_result(workspace)
    print(json.dumps(dict(passed=True, completed_motion_pair_calls=PAIR_COUNT,
                          result=str(workspace / "result.json"), compression=compression)), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    run(parser.parse_args().workspace)


if __name__ == "__main__":
    main()
