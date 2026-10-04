#!/usr/bin/env python3
"""One fresh-process static generated pair, optionally observed without edits.

No media, detector, PVA warm-up/preflight pair, changed status initializer,
threshold, pyramid setting, or global fit. Host readbacks can perturb execution;
separate uninstrumented-process non-timing parity is mandatory before inference.
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
import sys
import time

SCHEMA = "seaqr.static-pva-texture.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_static_pva_texture_20261001_[A-Za-z0-9]{6}"
PHOTOMETRIC_WORKSPACE = Path("/tmp/seaqr_motion_photometric_20260930_6RkevU")
PHOTOMETRIC_SHA = "569220932c8132944120ffc6026f59e01c47cb81cfd11d9c2875f890566f32f1"
PHOTOMETRIC_FREEZE_SHA = "aa4edcfaee0f78c3b87ce9101a740576641029da1a5a7c62e42b73538d2c93ee"
PRIOR_RESULT_SHA = "181b0e47e7a0a6332970c2e29014489cbf7ce4d64f8be27217566861478a580f"
METHOD_SHA = "bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b"
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
SHAPE = (512, 640)
CASES, MODES = ("bridge", "texture"), ("base", "trace")
FILES = {"probe_static_pva_texture.py", "batch_static_pva_texture.py", "test_static_pva_texture.py",
         "test_batch_static_pva_texture.py", "batch_discovery_pair.py"}
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5,
                 feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
                 grid_rows=6, grid_cols=8)
EXECUTION = dict(workers=1, phases=4, phase_deadline_seconds=900, batch_deadline_seconds=3600,
                 start_below_celsius=65, stop_at_celsius=75, automatic_retries=0)
ANCHORS = {
    "pyramids": '        timings["gaussian_pyramids_submit"] = (submitted - started) / 1_000_000',
    "before_forward": '        forward_options = {}',
    "after_forward": '        backward_points_vpi = None',
    "after_backward": '            flow_key = "backward_pyrlk_" + config.optical_flow_backend.lower()',
    "readback": '    selected_count = len(selected_points)',
    "final_filter": '    coverage = grid_coverage(',
}
LIMITATIONS = [
    "sys.settrace, synchronization-complete CPU locks and copies can perturb execution and allocation timing.",
    "Separate fresh-process base/trace non-timing parity is required; this child does not establish that parity alone.",
    "Legacy initial forward status is implicit inside OpticalFlowPyrLK and has no exposed accessor; it is not assumed zero.",
    "Original forward_status readback is post-backward under the shared status-buffer policy.",
    "Pyramid statistics describe realized images, not a proof of why VPI marks a point lost.",
    "Fresh processes isolate prior-pair state; each focal pair still uses the original legacy status policy.",
    "Generated static truth, not real-camera registration, detector accuracy, or production throughput.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024*1024, "Invalid metadata file")
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON number")
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda value: require(False, "Nonfinite JSON constant"))


def pinned(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, "Pinned file changed: "+str(path))


def validate_freeze(value):
    expected = dict(schema="static_pva_texture.v1", cases=list(CASES), modes=list(MODES), shape_hw=list(SHAPE),
        photometric_workspace=str(PHOTOMETRIC_WORKSPACE), photometric_runner_sha256=PHOTOMETRIC_SHA,
        candidate=CANDIDATE, no_preflight_pva_calls=True, execution=EXECUTION)
    require(set(value) == set(expected) | {"files"} and all(value.get(k) == v for k, v in expected.items()),
            "Static diagnostic freeze scope differs")
    require(value["no_preflight_pva_calls"] is True and isinstance(value["files"], dict)
            and set(value["files"]) == FILES
            and all(isinstance(x, str) and re.fullmatch(r"[a-f0-9]{64}", x) for x in value["files"].values()),
            "Static diagnostic bundle differs")
    require(value["files"]["batch_discovery_pair.py"] == SAFETY_SHA, "Safety helper differs")


def bundle_inputs(workspace, freeze_path, freeze_sha256):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)) and workspace.is_dir()
            and not workspace.is_symlink() and os.geteuid() != 0, "Unprivileged fresh scoped workspace required")
    require(freeze_path == workspace/"freeze.json" and isinstance(freeze_sha256, str)
            and re.fullmatch(r"[a-f0-9]{64}", freeze_sha256), "Explicit scoped freeze/hash required")
    pinned(freeze_path, freeze_sha256)
    value = read(freeze_path)
    validate_freeze(value)
    for name, digest in value["files"].items():
        pinned(workspace/name, digest)
    pinned(Path(__file__), value["files"]["probe_static_pva_texture.py"])
    return dict(freeze_sha256=freeze_sha256, files=value["files"])


def load_runtime():
    path = PHOTOMETRIC_WORKSPACE/"run_motion_photometric_controls.py"
    pinned(path, PHOTOMETRIC_SHA)
    spec = importlib.util.spec_from_file_location("static_frozen_photometric", path)
    phot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phot)
    selection, helper, generator, reference, runtime, hashes = phot.generated_inputs(
        PHOTOMETRIC_WORKSPACE, PHOTOMETRIC_WORKSPACE/"freeze.json", PHOTOMETRIC_FREEZE_SHA)
    prior_path = PHOTOMETRIC_WORKSPACE/"generated_pva_controls.json"
    pinned(prior_path, PRIOR_RESULT_SHA)
    prior = read(prior_path)
    require(prior.get("completed") is True and prior.get("passed_integrity") is True
            and prior.get("execution_passed") is True and prior["input_sha256"] == hashes,
            "Original generated evidence no longer matches")
    return phot, selection, helper, generator, reference, runtime, hashes, prior


def focal_pair(case, selection, generator, prior):
    import numpy as np
    require(case in CASES, "Only the fixed static pair cases are allowed")
    if case == "texture":
        pair = next(generator.generated_controls(SHAPE))
        require(pair["case"]["case_id"] == "texture__static" and pair["case"]["truth_displacement_xy"] == [0., 0.],
                "Original texture case differs")
        previous, current = pair["previous_gray"], pair["current_gray"]
        old = next(row for row in prior["cases"] if row["case"]["case_id"] == "texture__static")
        require(pair["previous_pixel_sha256"] == old["previous_pixel_sha256"]
                and pair["current_pixel_sha256"] == old["current_pixel_sha256"], "Original texture pixels differ")
    else:
        name, previous, current, truth = next(iter(selection.generated_controls()))
        require(name == "low_contrast_static" and list(truth) == [0, 0], "Original bridge case differs")
        old = next(row for row in prior["controls"] if row["name"] == name)
        require(old["passed"] is True and old["closed"] is True
                and hashlib.sha256(previous.tobytes()).hexdigest() == old["previous_pixel_sha256"]
                and hashlib.sha256(current.tobytes()).hexdigest() == old["current_pixel_sha256"], "Original bridge pixels differ")
    require(previous.shape == current.shape == SHAPE and previous.dtype == current.dtype == np.uint8
            and np.array_equal(previous, current), "Focal case must be identical native U8 frames")
    return previous, current, dict(case=case, native_shape_hw=list(SHAPE), truth_displacement_xy=[0., 0.],
        previous_pixel_sha256=generator.array_sha(previous), current_pixel_sha256=generator.array_sha(current),
        prior_result_sha256=PRIOR_RESULT_SHA)


def descriptor(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    require(a.dtype.kind in "uifb" and a.size <= 512*640, "Unbounded/non-numerical capture")
    raw = a.tobytes()
    return dict(dtype=a.dtype.str, shape=list(a.shape), data_base64=base64.b64encode(raw).decode(),
                sha256=hashlib.sha256(raw).hexdigest())


def image_record(value):
    import numpy as np
    a = np.asarray(value)
    require(a.ndim == 2 and a.dtype == np.uint8 and 0 < a.size <= 512*640, "Unexpected pyramid level representation")
    f = a.astype(np.float64)
    gradients = {}
    for name, axis in (("x", 1), ("y", 0)):
        d = np.diff(f, axis=axis)
        gradients[name] = dict(samples=int(d.size), nonzero=int(np.count_nonzero(d)),
            mean_absolute=float(np.abs(d).mean()) if d.size else None,
            rms=float(np.sqrt(np.mean(d*d))) if d.size else None)
    return dict(array=descriptor(a), minimum=int(a.min()), maximum=int(a.max()),
                mean=float(f.mean()), std=float(f.std()), unique_values=int(len(np.unique(a))), gradients=gradients)


def copy_vpi(value):
    import numpy as np
    with value.rlock_cpu() as data:
        return np.array(data, copy=True)


def copy_pyramid(value):
    import numpy as np
    # Installed VPI3.2.4 API verified separately: one readonly lock yields a
    # list of NumPy level views. Copies must complete before releasing it.
    with value.rlock_cpu() as data:
        require(isinstance(data, list) and len(data) == 4, "Expected four locked pyramid level views")
        arrays = [np.array(level, copy=True) for level in data]
    expected = [(256, 320), (128, 160), (64, 80), (32, 40)]
    require([a.shape for a in arrays] == expected, "Prepared pyramid dimensions differ")
    return [image_record(a) for a in arrays]


def vpi_identity(value):
    return None if value is None else dict(vpi_id=int(value.id))


def anchor_lines(source):
    require(hashlib.sha256(source.encode()).hexdigest() == METHOD_SHA, "Gain16 estimator source differs")
    lines = source.splitlines()
    result = {}
    for stage, anchor in ANCHORS.items():
        require(lines.count(anchor) == 1, "Missing/duplicate frozen observation anchor: "+stage)
        result[lines.index(anchor)+1] = stage
    require(len(result) == len(ANCHORS), "Observation anchors overlap")
    return result


class Capture:
    def __init__(self):
        self.stages, self.data, self.error = [], {}, None
        self.method_calls = 0

    def record(self, stage, state):
        import numpy as np
        require(stage not in self.stages, "Duplicate logical capture stage")
        if stage == "pyramids":
            require(state["pyramid_backend_name"] == "PVA" and state["rescale_backend"] == "CUDA",
                    "Prepared motion backend differs")
            data = {name: copy_pyramid(state[name+"_pyramid"]) for name in ("previous", "current")}
            data["proxy"] = {name: image_record(copy_vpi(state[name+"_motion"])) for name in ("previous", "current")}
        elif stage == "before_forward":
            data = {name: descriptor(state[name]) for name in ("selected_points", "selected_scores", "selected_indices")}
            data.update(initial_forward_status=None,
                initial_forward_status_observation="unavailable: implicit constructor-owned status under legacy_default; no accessor",
                status_initializer_changed=False, selected_count=len(state["selected_points"]))
        elif stage in ("after_forward", "after_backward"):
            names = ["tracked_points", "forward_status"]
            if stage == "after_backward":
                names += ["backward_points_vpi", "backward_status_vpi"]
            data = {name: descriptor(copy_vpi(state[name])) for name in names}
            data["identities"] = {name: vpi_identity(state[name]) for name in names}
            if stage == "after_backward":
                data["status_same_python_object"] = state["forward_status"] is state["backward_status_vpi"]
                data["points_same_python_object"] = state["tracked_points"] is state["backward_points_vpi"]
                data["backward_constructor_input_status_was_forward_status"] = state["backward_initial_status"] is state["forward_status"]
        elif stage == "readback":
            data = {name: descriptor(state[name]) if state[name] is not None else None for name in (
                "current_motion_points", "forward_status_array", "backward_motion_points", "backward_status_array")}
        elif stage == "final_filter":
            data = {name: descriptor(state[name]) if state[name] is not None else None for name in (
                "previous_full_points", "current_full_points", "backward_full_points", "accepted_mask", "fb_error",
                "accepted_previous", "accepted_current", "accepted_scores", "accepted_fb_error")}
            data["rejection_counts"] = dict(state["rejection_counts"])
            data["selected_indices_of_rejected_points"] = descriptor(np.flatnonzero(~state["accepted_mask"]))
            data["selected_indices_of_accepted_points"] = descriptor(np.flatnonzero(state["accepted_mask"]))
        else:
            raise ValueError("Undeclared trace stage")
        self.stages.append(stage)
        self.data[stage] = data

    def summary(self, unavailable=False):
        expected = ["pyramids"] if unavailable else list(ANCHORS)
        require(self.stages == expected and self.method_calls == 1 and self.error is None,
                "Incomplete/invalid focal observation")
        if unavailable:
            return dict(stages=self.stages, method_calls=self.method_calls, data=self.data,
                        error=None, expected_feature_unavailable=True, partial=True)
        before, after = self.data["after_forward"], self.data["after_backward"]
        return dict(stages=self.stages, method_calls=self.method_calls, data=self.data, error=self.error,
            forward_status_bytes_changed_by_backward=before["forward_status"] != after["forward_status"],
            forward_point_bytes_changed_by_backward=before["tracked_points"] != after["tracked_points"])


@contextmanager
def observe(method, source, capture):
    """Observe exactly one existing code object; never replace estimator code."""
    lines = anchor_lines(source)
    expected = compile(source, method.__code__.co_filename, "exec")
    code = next(item for item in expected.co_consts if hasattr(item, "co_name") and item.co_name == "_estimate_v12")
    require(method.__code__ == code and sys.gettrace() is None, "Unexpected estimator code or existing tracer")
    def tracer(frame, event, arg):
        if frame.f_code is not method.__code__:
            return None
        if event == "call":
            capture.method_calls += 1
            require(capture.method_calls == 1, "Multiple focal estimate calls")
        elif event == "line" and frame.f_lineno in lines:
            stage = lines[frame.f_lineno]
            if stage not in capture.stages:
                try:
                    capture.record(stage, frame.f_locals)
                except BaseException as exc:
                    capture.error = repr(exc)
                    raise
        return tracer
    try:
        sys.settrace(tracer)
        yield
    finally:
        sys.settrace(None)


def result_record(correspondence, fit):
    arrays = {name: descriptor(getattr(correspondence, name)) for name in (
        "previous_points", "current_points", "harris_scores", "forward_backward_error_px")}
    fit_dict = fit.to_dict()
    require("timing_ms" in fit_dict, "Original fit timing schema differs")
    fit_timing = fit_dict.pop("timing_ms")
    value = dict(correspondence=dict(arrays=arrays, metrics=correspondence.metrics, backends=correspondence.backends,
        full_image_size=list(correspondence.full_image_size), motion_image_size=list(correspondence.motion_image_size)),
        fit=fit_dict, inlier_mask=descriptor(fit.inlier_mask), residuals_px=descriptor(fit.residuals_px))
    return value, dict(correspondence_timings_ms=correspondence.timings_ms, fit_timing_ms=fit_timing)


def focal_result(estimator, previous, current, fit_function, global_config, phot,
                 pva_error_type, receipt, method=None, source=None, capture=None):
    """Exactly one focal estimate; only declared feature-unavailable errors survive."""
    receipt["focal_estimate_calls"] += 1
    try:
        if capture is None:
            correspondence = estimator.estimate(previous, current)
        else:
            with observe(method, source, capture):
                correspondence = estimator.estimate(previous, current)
    except Exception as exc:
        if not phot.expected_unavailable(exc, estimator, pva_error_type):
            raise
        if capture is not None:
            receipt["capture"] = capture.summary(unavailable=True)
        receipt.update(scientific_fit_accepted=False, accepted_points=None,
                       scientific_status="unavailable", unavailable_reason=str(exc))
        return dict(status="unavailable", reason=str(exc), correspondence=None, fit=None), {}
    if capture is not None:
        receipt["capture"] = capture.summary()
    receipt["global_fits"] += 1
    fit = fit_function(correspondence, global_config)
    if correspondence.count == 0:
        require(not fit.accepted, "Zero accepted correspondences cannot produce an accepted fit")
    result, timings = result_record(correspondence, fit)
    receipt.update(scientific_fit_accepted=bool(fit.accepted), accepted_points=correspondence.count,
                   scientific_status="measured" if correspondence.count else "unavailable",
                   unavailable_reason=None if correspondence.count else "zero accepted LK correspondences")
    return result, timings


def run(workspace, freeze_path, freeze_sha256, case, trace=False):
    workspace = Path(workspace)
    require(case in CASES and type(trace) is bool, "Invalid focal case/mode")
    inputs = bundle_inputs(workspace, freeze_path, freeze_sha256)
    mode = "trace" if trace else "base"
    output = workspace/(case+"_"+mode+".json")
    require(not output.exists() and not output.is_symlink(), "Existing focal evidence; no overwrite")
    # Exclusive reservation belongs to this process and is finalized even on error.
    with output.open("x"):
        pass
    receipt = dict(schema=SCHEMA, case=case, mode=mode, completed=False, passed_integrity=False,
        input_sha256=inputs, generated_only=True, source_media_accessed=False, detector_run=False,
        global_fits=0, focal_estimate_calls=0, preflight_pva_calls=0, production_changes=False,
        capture=None, canonical_nontiming=None, canonical_nontiming_sha256=None, error=None,
        limitations=LIMITATIONS, cross_process_parity_established_by_this_child=False)
    began, estimator, capture = time.perf_counter(), None, Capture() if trace else None
    try:
        phot, selection, helper, generator, reference, runtime, prior_inputs, prior = load_runtime()
        inputs.update(photometric_runner_sha256=PHOTOMETRIC_SHA, prior_photometric_inputs=prior_inputs,
                      prior_generated_result_sha256=PRIOR_RESULT_SHA)
        modules, identities = helper.dependencies(reference)
        info = modules["profile_visible_interaction_v30"].runtime_info
        before, clocks = info(), helper.clock_policy_snapshot()
        helper.runtime_check(before, runtime)
        import cv2
        from tiny_target.types import Frame, TimestampSource
        from tiny_target.motion import fit_global_motion
        motion, global_config = selection.configurations(helper)
        require(motion.flow_status_policy == "legacy_default" and motion.forward_backward_check is True,
                "Frozen bidirectional legacy status policy required")
        previous, current, input_pair = focal_pair(case, selection, generator, prior)
        p = Frame(previous.copy(), 0, 0, "generated-static-texture:"+case, 8, TimestampSource.CONTAINER_RATE)
        q = Frame(current.copy(), 100_000_000, 1, "generated-static-texture:"+case, 8, TimestampSource.CONTAINER_RATE)
        frame_hashes = dict(previous=p.pixel_sha256(), current=q.pixel_sha256())
        receipt.update(runtime_before=before, clock_policy_before=clocks, identities=identities, input_pair=input_pair,
            frame_pixel_sha256=frame_hashes, pyramid_dimensions=selection.validate_pyramid_dimensions(SHAPE, motion),
            original_motion_configuration=asdict(motion), global_configuration=asdict(global_config),
            feature_adapter={}, diagnostic_postcondition={})
        cv2.setNumThreads(2)
        reuse = modules["motion_reuse_v12"]
        source = selection.transform_source(reuse.generated_method())
        lines = anchor_lines(source)
        receipt["observation_contract"] = dict(method_sha256=METHOD_SHA, line_stages={str(k):v for k,v in lines.items()},
            sys_settrace=trace, estimator_source_changed=False, initial_status_changed=False,
            extra_pva_compute_calls=0, additional_cpu_readbacks=trace)
        with selection.candidate_adapter(reuse, motion, receipt["feature_adapter"]) as candidate:
            with phot.allow_empty_diagnostic_result(selection, receipt["diagnostic_postcondition"]):
                estimator = candidate(motion)
                result, timings = focal_result(estimator, p, q, fit_global_motion, global_config, phot,
                    reuse.pva.PvaMotionError, receipt, reuse._ESTIMATE, source, capture)
                canonical = dict(input_pair=input_pair, frame_pixel_sha256=frame_hashes,
                    effective_motion_configuration=receipt["feature_adapter"]["effective_motion_configuration"],
                    global_configuration=asdict(global_config), result=result)
                receipt.update(canonical_nontiming=canonical, canonical_nontiming_sha256=canonical_sha(canonical), timings=timings)
                require(frame_hashes == dict(previous=p.pixel_sha256(), current=q.pixel_sha256()), "Focal pixels mutated")
                estimator.close()
                require(estimator.closed is True and estimator.failed is False, "Focal lifecycle failed")
                receipt["closed"] = True
        after = info()
        helper.runtime_check(after, runtime, after=True)
        require(helper.clock_policy_snapshot() == clocks, "Clock policy changed")
        require(bundle_inputs(workspace, freeze_path, freeze_sha256) == {k:inputs[k] for k in ("freeze_sha256", "files")},
                "Frozen child bundle changed")
        require(load_runtime()[-2] == prior_inputs and helper.dependencies(reference)[1] == identities, "Dependencies changed")
        receipt.update(completed=True, passed_integrity=True, runtime_after=after, clocks_changed=False)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        if estimator is not None and not estimator.closed:
            try:
                estimator.close()
                receipt["closed"] = estimator.closed is True
            except BaseException as exc:
                receipt.update(completed=False, passed_integrity=False, cleanup_error=repr(exc))
                if receipt["error"] is None:
                    receipt["error"] = repr(exc)
        if capture is not None and receipt["capture"] is None:
            receipt["capture"] = dict(stages=capture.stages, data=capture.data, error=capture.error,
                                      method_calls=capture.method_calls, partial=True)
        receipt["elapsed_seconds"] = time.perf_counter()-began
        require(output.is_file() and not output.is_symlink() and output.stat().st_size == 0, "Reserved focal output changed")
        with output.open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    value = run(args.workspace, args.freeze, args.freeze_sha256, args.case, args.trace)
    print(json.dumps(dict(case=value["case"], mode=value["mode"], passed_integrity=value["passed_integrity"],
                          canonical_nontiming_sha256=value["canonical_nontiming_sha256"])), flush=True)
    raise SystemExit(0 if value["passed_integrity"] else 1)
