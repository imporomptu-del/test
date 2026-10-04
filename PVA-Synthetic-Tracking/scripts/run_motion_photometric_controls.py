#!/usr/bin/env python3
"""Generated-only PVA photometric diagnosis; no media, detector or retuning.

The fixed analytic translations are truth for this synthetic corpus only. The
NCC checker's expectations do not become PVA pass/fail criteria: one corner
cannot satisfy the unchanged 30-feature global-fit gate, and four-pixel motion
is outside that checker's search but not necessarily outside PVA's capability.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

SCHEMA = "seaqr.motion-photometric-controls.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_motion_photometric_20260930_[A-Za-z0-9]{6}"
ORIGINAL_WORKSPACE = Path("/tmp/seaqr_feature_selection_20260929_q4iI5B")
ORIGINAL_FREEZE_SHA = "9ae62028ca25fe063241e62697cb3578a708078a1b03969061b4736f0c9c6b7e"
SELECTION_SHA = "db5a92d69fb7fc503a3ef2d56236460d1bb32656111ad3fd170203b9ab53bc4d"
GENERATOR_SHA = "b1d915604327cca4a5a7bf83a3eef9eca197d43ecd9cc8e2cfc63a1d61aeb97c"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5,
                 feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
                 grid_rows=6, grid_cols=8)
FILES = {"run_motion_photometric_controls.py", "batch_motion_photometric_controls.py",
         "validate_motion_patch_controls.py", "batch_discovery_pair.py",
         "test_motion_photometric_controls.py", "test_motion_patch_controls.py"}
EXECUTION = dict(workers=1, phases=1, phase_deadline_seconds=900, batch_deadline_seconds=3600,
                 start_below_celsius=65, stop_at_celsius=75, automatic_retries=0)
SHAPE = (512, 640)
INTERIOR_MARGIN = 128
UNAVAILABLE_MESSAGES = {"PVA Harris returned zero features; motion is unavailable",
                        "No finite in-bounds Harris features survived selection"}
LIMITATIONS = [
    "Generated analytic truth is not evidence of real-video registration accuracy or detection recall.",
    "Only accepted PVA correspondences are scored; raw rejected endpoints are not captured.",
    "Truth interior selection uses previous points and known translated endpoints, never measured endpoints.",
    "A fixed native128px margin excludes edges; this is not an exact accounting of all pyramid receptive fields.",
    "Single-corner/degenerate availability and global-fit rejection are not automatically truth errors.",
    "NCC qualification expectations are retained as corpus metadata, not applied to PVA.",
    "The 0.25px descriptive error band comes from the frozen generated corpus, not a new production gate.",
    "Fresh pair-local estimators do not test long-running reuse state, throughput, or the real-camera footprint.",
    "Fresh estimator instances do not prove inter-case backend independence under preserved legacy_default flow-status/cache policy; ordered photometric comparisons are suggestive, not sole-cause proof.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024 * 1024,
            "missing/linked/oversize metadata: " + str(path))
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number
    with path.open() as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_float=finite,
                         parse_constant=lambda value: require(False, "nonfinite JSON constant"))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pinned(path, digest):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest,
            "pinned file differs: " + str(path))


def load_module(path, name, digest):
    pinned(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_freeze(value):
    require(set(value) == {"schema", "files", "generated_only", "candidate", "original_workspace",
                         "original_freeze_sha256", "source_shape_hw", "case_count", "execution"},
            "unexpected freeze fields")
    require(value["schema"] == "motion_photometric_controls.v1" and value["generated_only"] is True
            and value["candidate"] == CANDIDATE and value["original_workspace"] == str(ORIGINAL_WORKSPACE)
            and value["original_freeze_sha256"] == ORIGINAL_FREEZE_SHA
            and value["source_shape_hw"] == list(SHAPE) and type(value["case_count"]) is int
            and value["case_count"] == 20 and value["execution"] == EXECUTION, "freeze contract differs")
    require(isinstance(value["files"], dict) and set(value["files"]) == FILES
            and all(isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d) for d in value["files"].values()),
            "fixed generated-only bundle differs")
    require(value["files"]["validate_motion_patch_controls.py"] == GENERATOR_SHA
            and value["files"]["batch_discovery_pair.py"] == SAFETY_SHA, "generator/safety pin differs")


def bundle_inputs(workspace, freeze_path, freeze_sha256):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)) and workspace.is_dir()
            and not workspace.is_symlink() and os.geteuid() != 0, "unprivileged real scoped workspace required")
    require(freeze_path == workspace / "freeze.json" and isinstance(freeze_sha256, str)
            and re.fullmatch(r"[a-f0-9]{64}", freeze_sha256), "explicit workspace freeze and hash required")
    pinned(freeze_path, freeze_sha256)
    freeze = read(freeze_path)
    validate_freeze(freeze)
    for name, digest in freeze["files"].items():
        pinned(workspace / name, digest)
    pinned(Path(__file__), freeze["files"]["run_motion_photometric_controls.py"])
    return dict(freeze_sha256=freeze_sha256, files=dict(freeze["files"]))


def generated_inputs(workspace, freeze_path, freeze_sha256):
    """Only explicitly pinned metadata/code. Never call legacy media inputs()."""
    hashes = bundle_inputs(workspace, freeze_path, freeze_sha256)
    pinned(ORIGINAL_WORKSPACE / "freeze.json", ORIGINAL_FREEZE_SHA)
    selection = load_module(ORIGINAL_WORKSPACE / "run_discovery_feature_selection.py",
                            "photometric_frozen_selection", SELECTION_SHA)
    require(selection.CANDIDATE == CANDIDATE, "candidate identity differs")
    baseline = selection.load_baseline()
    pinned(baseline.REFERENCE, baseline.REFERENCE_SHA)
    pinned(baseline.REFERENCE_RUNTIME, baseline.REFERENCE_RUNTIME_SHA)
    reference, runtime = read(baseline.REFERENCE), read(baseline.REFERENCE_RUNTIME)
    helper = baseline.load_helper()
    generator = load_module(Path(workspace) / "validate_motion_patch_controls.py",
                            "photometric_analytic_generator", GENERATOR_SHA)
    hashes.update(original_freeze_sha256=ORIGINAL_FREEZE_SHA, selection_sha256=SELECTION_SHA,
                  baseline_sha256=selection.BASELINE_HELPER_SHA, helper_sha256=baseline.HELPER_SHA,
                  reference_sha256=baseline.REFERENCE_SHA, reference_runtime_sha256=baseline.REFERENCE_RUNTIME_SHA)
    return selection, helper, generator, reference, runtime, hashes


def descriptor(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    data = array.tobytes()
    return dict(dtype=array.dtype.str, shape=list(array.shape), data_base64=base64.b64encode(data).decode("ascii"),
                sha256=hashlib.sha256(data).hexdigest())


def statistics(values):
    import numpy as np
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "nonfinite truth errors")
    return dict(count=len(values), median_px=float(np.median(values)) if len(values) else None,
                p90_px=float(np.percentile(values, 90)) if len(values) else None,
                maximum_px=float(values.max()) if len(values) else None,
                above_0_25px=int((values > .25).sum()))


def truth_metrics(correspondence, fit, truth, shape=SHAPE):
    import numpy as np
    p, q = np.asarray(correspondence.previous_points, np.float64), np.asarray(correspondence.current_points, np.float64)
    truth = np.asarray(truth, np.float64)
    require(p.ndim == 2 and p.shape[1] == 2 and p.shape == q.shape
            and np.isfinite(p).all() and np.isfinite(q).all() and truth.shape == (2,)
            and np.isfinite(truth).all(), "invalid accepted correspondences/truth")
    limit = np.array([shape[1], shape[0]], np.float64) - INTERIOR_MARGIN
    expected = p + truth
    interior = ((p >= INTERIOR_MARGIN) & (p < limit) & (expected >= INTERIOR_MARGIN) & (expected < limit)).all(axis=1)
    errors = np.linalg.norm(q - expected, axis=1)
    mask = np.asarray(fit.inlier_mask, bool)
    require(mask.shape == errors.shape, "fit mask differs from accepted points")
    parameters = fit.parameters or {}
    vector = [parameters.get("translation_x_px"), parameters.get("translation_y_px")]
    valid_vector = all(type(v) in (int, float) and math.isfinite(v) for v in vector)
    return dict(expected_displacement_xy=truth.tolist(), accepted_count=len(p), interior_margin_native_px=INTERIOR_MARGIN,
        accepted_interior_count=int(interior.sum()), excluded_count=int((~interior).sum()),
        selection_uses_measured_endpoint=False, interior_indices=np.flatnonzero(interior).tolist(),
        all_accepted_interior=statistics(errors[interior]),
        original_fit_inlier_interior=statistics(errors[interior & mask]),
        original_fit_outlier_interior=statistics(errors[interior & ~mask]),
        accepted_truth_errors_px=descriptor(errors),
        original_fit_vector_available=valid_vector, original_fit_accepted=bool(fit.accepted),
        original_fit_translation_error_px=float(np.linalg.norm(np.array(vector)-truth)) if valid_vector else None,
        original_fit_rejected_is_not_automatically_truth_failure=True)


def expected_unavailable(exc, estimator, pva_error_type):
    return (isinstance(exc, pva_error_type) and str(exc) in UNAVAILABLE_MESSAGES
            and estimator is not None and getattr(estimator, "failed", True) is False)


@contextmanager
def allow_empty_diagnostic_result(selection, audit):
    """Allow a genuine zero-accepted result, without changing PVA or fit gates.

    The original media runner's postcondition required positive accepted count.
    Degenerate generated cases need to retain normal empty results. Nonempty
    results still go through the exact original verifier. The zero-only branch
    repeats every other original invariant and adds empty-array/count checks.
    """
    import numpy as np
    original = selection.verify_correspondence
    audit.update(diagnostic_postcondition_changed=True, compute_or_quality_gates_changed=False,
                 scope="twenty generated cases only; existing preflight controls use original verifier",
                 change="genuine zero accepted LK result may be recorded as unavailable", empty_results=0)
    def verify(correspondence, shape):
        if correspondence.count != 0:
            return original(correspondence, shape)
        height, width = shape
        require(correspondence.backends == selection.EXPECTED_BACKENDS, "unexpected motion backend/fallback")
        require(tuple(correspondence.full_image_size) == (width, height)
                and tuple(correspondence.motion_image_size) == (round(width*.5), round(height*.5)),
                "native/proxy dimensions changed")
        metrics = correspondence.metrics
        capacity = metrics.get("harris_output", {})
        require(capacity.get("capacity_policy") == "complete_grid"
                and capacity.get("capacity") == selection.capacity_for_shape(shape)
                and capacity.get("capacity_exhausted") is False
                and 0 < metrics.get("detected_count", 0) < capacity["capacity"], "Harris capacity differs/exhausted")
        selected = metrics.get("selected_count", 0)
        require(metrics.get("minimum_accepted_features") == 30 and metrics.get("minimum_grid_coverage") == .2
                and type(selected) is int and 0 < selected <= 384, "feature count/gates changed")
        require(metrics.get("accepted_count") == 0 and metrics.get("rejected_count") == selected
                and metrics.get("usable_for_transform") is False
                and "insufficient_accepted_features" in metrics.get("quality_rejection_reasons", []),
                "empty result metrics inconsistent")
        for name, expected in (("previous_points", (0,2)), ("current_points", (0,2)),
                               ("harris_scores", (0,)), ("forward_backward_error_px", (0,))):
            require(np.asarray(getattr(correspondence, name)).shape == expected, "empty result arrays inconsistent")
        audit["empty_results"] += 1
    with patch.object(selection, "verify_correspondence", verify):
        yield


def summarize_cases(rows):
    comparisons = []
    for changed in rows:
        case = changed["case"]
        if case["current_gain"] == 1.0 and case["current_offset_dn"] == 0.0:
            continue
        controls = [row for row in rows if row["case"]["family"] == case["family"]
                    and row["case"]["truth_displacement_xy"] == case["truth_displacement_xy"]
                    and row["case"]["current_gain"] == 1.0 and row["case"]["current_offset_dn"] == 0.0
                    and row["case"]["expectation"] == case["expectation"]]
        require(len(controls) == 1, "photometric counterpart must be unique")
        control = controls[0]
        def compact(row):
            truth = row.get("truth", {})
            return dict(case_id=row["case"]["case_id"], status=row["status"],
                accepted_count=truth.get("accepted_count"), interior_count=truth.get("accepted_interior_count"),
                truth_error=truth.get("all_accepted_interior"),
                original_fit_accepted=truth.get("original_fit_accepted"),
                fit_translation_error_px=truth.get("original_fit_translation_error_px"))
        comparisons.append(dict(control=compact(control), photometric=compact(changed),
            feature_populations_may_differ=True, interpretation="paired images/family/truth, not paired feature identities"))
    degenerate = [row for row in rows if row["case"]["family"] in {"flat", "edge", "periodic"}]
    accepted_degenerate = [row["case"]["case_id"] for row in degenerate
                           if row.get("truth", {}).get("original_fit_accepted") is True]
    wrong_accepted = [row["case"]["case_id"] for row in rows
                     if row.get("truth", {}).get("original_fit_accepted") is True
                     and row["truth"]["original_fit_translation_error_px"] is not None
                     and row["truth"]["original_fit_translation_error_px"] > .25]
    return dict(status_counts={name: sum(row["status"] == name for row in rows)
                for name in ("measured", "unavailable", "runtime_error", "not_run")},
        photometric_counterparts=comparisons, degenerate_case_count=len(degenerate),
        degenerate_global_fit_accepted_case_ids=accepted_degenerate,
        degenerate_acceptance_is_observability_warning_not_measured_false_positive_rate=True,
        accepted_global_fit_error_above_0_25px_case_ids=wrong_accepted,
        accepted_fit_error_band_is_synthetic_diagnostic_not_production_gate=True)


def run_cases(candidate, motion, global_config, generator, rows, frame_type, timestamp_source,
              pva_error_type, fit_function):
    """One estimator and one unmodified original fit per analytic pair."""
    inventory = generator.control_inventory(SHAPE)
    require(len(inventory["cases"]) == 20 and inventory["total_pairs"] == 20, "twenty fixed cases required")
    rows.extend(dict(case=case, status="not_run", closed=None, runtime_error=None) for case in inventory["cases"])
    for index, pair in enumerate(generator.generated_controls(SHAPE)):
        require(index < len(rows) and pair["case"] == rows[index]["case"], "generator order/content changed")
        row, estimator = rows[index], None
        began = time.perf_counter()
        row.update(status="running", previous_pixel_sha256=pair["previous_pixel_sha256"],
                   current_pixel_sha256=pair["current_pixel_sha256"], quantization=pair["quantization"])
        try:
            p = frame_type(pair["previous_gray"].copy(), 0, 0, "generated-photometric:"+pair["case"]["case_id"], 8, timestamp_source)
            q = frame_type(pair["current_gray"].copy(), 100_000_000, 1, "generated-photometric:"+pair["case"]["case_id"], 8, timestamp_source)
            row["frame_pixel_sha256"] = dict(previous=p.pixel_sha256(), current=q.pixel_sha256())
            estimator = candidate(motion)
            correspondence = estimator.estimate(p, q)
            row.update(correspondence=dict(previous_points=descriptor(correspondence.previous_points),
                current_points=descriptor(correspondence.current_points),
                harris_scores=descriptor(correspondence.harris_scores),
                forward_backward_error_px=descriptor(correspondence.forward_backward_error_px),
                metrics=correspondence.metrics, backends=correspondence.backends,
                full_image_size=list(correspondence.full_image_size), motion_image_size=list(correspondence.motion_image_size)))
            fit = fit_function(correspondence, global_config)
            row.update(status="measured", original_fit=fit.to_dict(),
                inlier_mask=descriptor(fit.inlier_mask), original_residuals_px=descriptor(fit.residuals_px),
                truth=truth_metrics(correspondence, fit, pair["case"]["truth_displacement_xy"]))
            if len(correspondence.previous_points) == 0:
                require(not fit.accepted, "zero-correspondence fit cannot be accepted")
                row.update(status="unavailable", unavailable_reason="zero accepted LK correspondences")
            require(row["frame_pixel_sha256"] == dict(previous=p.pixel_sha256(), current=q.pixel_sha256()),
                    "generated input Frame mutated")
        except Exception as exc:
            if expected_unavailable(exc, estimator, pva_error_type):
                row.update(status="unavailable", unavailable_reason=str(exc), runtime_error=None)
            else:
                row.update(status="runtime_error", runtime_error=repr(exc),
                    error_category="backend_or_estimator_error" if isinstance(exc, pva_error_type) else "contract_or_runtime_error")
        finally:
            if estimator is not None:
                try:
                    estimator.close()
                    row["closed"] = estimator.closed is True
                    require(row["closed"], "estimator failed to close")
                except Exception as exc:
                    row.update(status="runtime_error", cleanup_error=repr(exc), runtime_error=repr(exc), error_category="cleanup_error")
            row["elapsed_seconds"] = time.perf_counter()-began
        require(generator.array_sha(pair["previous_gray"]) == pair["previous_pixel_sha256"]
                and generator.array_sha(pair["current_gray"]) == pair["current_pixel_sha256"], "analytic source mutated")
        print(json.dumps(dict(case=row["case"]["case_id"], status=row["status"])), flush=True)
        require(row["status"] != "runtime_error", "generated case runtime failure; remaining cases not run: " + row["case"]["case_id"])
    require(all(row["status"] != "not_run" for row in rows), "generator ended before twenty cases")


def run(workspace, freeze_path, freeze_sha256):
    workspace = Path(workspace)
    selection, helper, generator, reference, runtime, hashes = generated_inputs(workspace, freeze_path, freeze_sha256)
    output = workspace / "generated_pva_controls.json"
    require(not output.exists() and not output.is_symlink(), "existing output; no overwrite")
    # Reserve the output before touching hardware; an interrupted run stays visible.
    with output.open("x") as stream:
        stream.write("")
    receipt = dict(schema=SCHEMA, completed=False, passed_integrity=False, execution_passed=False,
        generated_only=True, source_media_accessed=False, detector_run=False, production_promotion=False,
        scientific_accuracy_gate_applied=False, case_count=20, input_sha256=hashes, candidate=CANDIDATE,
        controls=[], cases=[], limitations=LIMITATIONS, error=None)
    began = time.perf_counter()
    try:
        modules, identities = helper.dependencies(reference)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        import cv2
        from tiny_target.types import Frame, TimestampSource
        from tiny_target.motion import PvaMotionError, fit_global_motion
        motion, global_config = selection.configurations(helper)
        receipt.update(runtime_before=before, clock_policy_before=clocks, identities=identities,
                       original_motion_configuration=asdict(motion), global_configuration=asdict(global_config),
                       inventory=generator.control_inventory(SHAPE), feature_adapter={},
                       pyramid_dimensions=selection.validate_pyramid_dimensions(SHAPE, motion))
        cv2.setNumThreads(2)
        receipt["cpu_parity"] = selection.generated_cpu_parity(motion)
        receipt["conversion"] = selection.conversion_check()
        with selection.candidate_adapter(modules["motion_reuse_v12"], motion, receipt["feature_adapter"]) as candidate:
            selection.run_controls(candidate, motion, global_config, receipt["controls"])
            receipt["preflight_passed"] = True
            receipt["diagnostic_postcondition"] = {}
            with allow_empty_diagnostic_result(selection, receipt["diagnostic_postcondition"]):
                run_cases(candidate, motion, global_config, generator, receipt["cases"], Frame,
                          TimestampSource.CONTAINER_RATE, PvaMotionError, fit_global_motion)
        after = info()
        helper.runtime_check(after, runtime, after=True)
        require(helper.clock_policy_snapshot() == clocks, "clock policy changed")
        require(generated_inputs(workspace, freeze_path, freeze_sha256)[-1] == hashes
                and helper.dependencies(reference)[1] == identities, "input/dependency identities changed")
        summary = summarize_cases(receipt["cases"])
        statuses = summary["status_counts"]
        receipt.update(completed=True, passed_integrity=True, runtime_after=after, clocks_changed=False,
                       status_counts=statuses, summary=summary, execution_passed=statuses["runtime_error"] == 0)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter()-began
        # This process owns the exclusively reserved evidence file only.
        require(output.is_file() and not output.is_symlink() and output.stat().st_size == 0,
                "reserved output changed")
        with output.open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    options = parser.parse_args()
    result = run(options.workspace, options.freeze, options.freeze_sha256)
    print(json.dumps(dict(completed=result["completed"], execution_passed=result["execution_passed"],
                          status_counts=result["status_counts"])), flush=True)
    raise SystemExit(0 if result["execution_passed"] else 1)
