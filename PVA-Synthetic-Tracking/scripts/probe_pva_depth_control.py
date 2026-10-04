#!/usr/bin/env python3
"""Generated-only two-level causal diagnostic; never a production configuration.

Exactly one pair in a fresh process. All helpers and four-level references are
immutable and hash-bound. The only algorithm-setting difference is 4 -> 2
pyramid levels; extra readbacks still require independent base/trace parity.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

SCHEMA = "seaqr.pva-depth-control.v1"
WORKSPACE_PATTERN = r"/tmp/seaqr_pva_depth_control_20261001_[A-Za-z0-9]{6}"
REFERENCE_WORKSPACE = Path("/tmp/seaqr_static_pva_texture_20261001_SKYCZ8")
REFERENCE_PROBE_SHA = "48cff34b8248fdcfbc447eda32774ff91f33f88b78bb62c6986b65c50d634907"
REFERENCE_BATCH_SHA = "2f74bac6432a93d62e558eb59798baf033892f81f4c3d114f741397b0c7471c7"
REFERENCE_FREEZE_SHA = "39cc9e77ad24afb9e155137db81ce7728893492b29159397ae5b738d5d7bcda3"
REFERENCE_RESULTS = {
    "bridge_base.json": "ef6b895f76a116b6fe588106bd6e2abe1352ae9cc7d6191918ea3c1a478ab856",
    "bridge_trace.json": "96442e51ef54c6732fdd9bb47cc7f8b9b64e3b661f4c39c546ef125fd22e9dba",
    "texture_base.json": "7f70f206a75264ad1ad85b3d0d5d54fc2f947f168f23e6f24164e8de6b818c09",
    "texture_trace.json": "4e6488a6c86552547f9f908aac5d9c16982e74a353a7f52cc6513b0ff1584d3b",
}
CASES, MODES, SHAPE = ("bridge", "texture"), ("base", "trace"), (512, 640)
FILES = {"probe_pva_depth_control.py", "batch_pva_depth_control.py", "test_pva_depth_control.py",
         "test_batch_pva_depth_control.py", "batch_discovery_pair.py"}
SAFETY_SHA = "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=.5,
    feature_cpu_policy="batched_exact_v1", max_features=384, max_features_per_cell=8,
    grid_rows=6, grid_cols=8, pyramid_levels=2)
EXECUTION = dict(workers=1, phases=4, phase_deadline_seconds=900, batch_deadline_seconds=3600,
    start_below_celsius=65, stop_at_celsius=75, automatic_retries=0)
PROTOCOL = dict(reference_pyramid_levels=4, diagnostic_pyramid_levels=2,
    only_configuration_change="pyramid_levels", feature_seed_and_pyramid_prefix_exact=True,
    new_base_trace_exact_parity_required=True, availability_restoration="texture immediate-forward valid count > 0",
    stronger_recovery="original global fit accepted and all-accepted native static maximum error <= 0.25px",
    bridge_required="original zero-motion fit accepted", production_promotion=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def import_pinned(path, digest, name):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Pinned helper differs: "+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def old_helper():
    return import_pinned(REFERENCE_WORKSPACE/"probe_static_pva_texture.py", REFERENCE_PROBE_SHA, "depth_reference_probe")


def freeze_fields():
    return dict(schema="pva_depth_control.v1", cases=list(CASES), modes=list(MODES), shape_hw=list(SHAPE),
        reference_workspace=str(REFERENCE_WORKSPACE), reference_probe_sha256=REFERENCE_PROBE_SHA,
        reference_batch_sha256=REFERENCE_BATCH_SHA, reference_freeze_sha256=REFERENCE_FREEZE_SHA,
        reference_results=dict(REFERENCE_RESULTS), candidate=dict(CANDIDATE), protocol=dict(PROTOCOL),
        no_preflight_pva_calls=True, execution=dict(EXECUTION))


def validate_freeze(value):
    expected = freeze_fields()
    require(isinstance(value, dict) and set(value) == set(expected) | {"files"}
        and json.dumps({k:value[k] for k in expected}, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False), "Depth diagnostic freeze scope differs")
    require(isinstance(value["files"], dict) and set(value["files"]) == FILES
        and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in value["files"].values())
        and value["files"]["batch_discovery_pair.py"] == SAFETY_SHA, "Depth bundle file inventory differs")


def bundle(workspace, freeze_path, digest, old):
    workspace, freeze_path = Path(workspace), Path(freeze_path)
    require(re.fullmatch(WORKSPACE_PATTERN, str(workspace)) and workspace.is_dir()
        and not workspace.is_symlink() and os.geteuid() != 0, "Unprivileged scoped depth workspace required")
    require(freeze_path == workspace/"freeze.json", "Scoped freeze required")
    old.pinned(freeze_path, digest)
    value = old.read(freeze_path)
    validate_freeze(value)
    for name, expected in value["files"].items():
        old.pinned(workspace/name, expected)
    old.pinned(Path(__file__), value["files"]["probe_pva_depth_control.py"])
    return value


def references(old):
    old.pinned(REFERENCE_WORKSPACE/"freeze.json", REFERENCE_FREEZE_SHA)
    original_freeze = old.read(REFERENCE_WORKSPACE/"freeze.json")
    old.validate_freeze(original_freeze)
    for name, digest in original_freeze["files"].items():
        old.pinned(REFERENCE_WORKSPACE/name, digest)
    rows = {}
    for name, digest in REFERENCE_RESULTS.items():
        old.pinned(REFERENCE_WORKSPACE/name, digest)
        row = old.read(REFERENCE_WORKSPACE/name)
        case, mode = name[:-5].split("_")
        require(row.get("schema") == old.SCHEMA and row.get("completed") is True
            and row.get("passed_integrity") is True and row.get("case") == case and row.get("mode") == mode
            and row["input_sha256"]["freeze_sha256"] == REFERENCE_FREEZE_SHA
            and old.canonical_sha(row["canonical_nontiming"]) == row["canonical_nontiming_sha256"],
            "Original four-level result contract differs")
        rows[case+"_"+mode] = row
    for case in CASES:
        require(rows[case+"_base"]["canonical_nontiming"] == rows[case+"_trace"]["canonical_nontiming"],
                "Original four-level parity differs")
    return rows


def require_depth_difference(original, changed):
    # Configuration tuples become lists in saved receipts. Compare the exact
    # JSON representations, not Python's tuple/list implementation detail.
    original=json.loads(json.dumps(original,allow_nan=False))
    changed=json.loads(json.dumps(changed,allow_nan=False))
    require(original.get("pyramid_levels") == 4 and type(original["pyramid_levels"]) is int
        and changed.get("pyramid_levels") == 2 and type(changed["pyramid_levels"]) is int
        and set(original) == set(changed)
        and {k for k in original if original[k] != changed[k]} == {"pyramid_levels"},
        "Only pyramid_levels 4 -> 2 may differ")


def depth_config(original):
    changed = replace(original, pyramid_levels=2)
    require_depth_difference(asdict(original), asdict(changed))
    require(changed.feature_image_scale == .5 and changed.pyramid_scale == .5
        and changed.flow_status_policy == "legacy_default" and changed.forward_backward_check is True,
        "Frozen feature scale/pyramid scale/status policy differs")
    return changed


def dimensions(config):
    require(config.pyramid_levels == 2 and config.feature_image_scale == config.pyramid_scale == .5,
            "Explicit two-level dimensions required")
    sizes = [[320, 256], [160, 128]]
    require(min(sizes[-1]) >= 32, "PVA smallest pyramid dimension below 32")
    return dict(native_shape_hw=list(SHAPE), level_size_wh=sizes, pyramid_levels=2,
                original_pyramid_levels=4, minimum_level_size_wh=[32,32], passed=True)


def capture_type(old):
    class DepthCapture(old.Capture):
        def record(self, stage, state):
            if stage != "pyramids":
                return super().record(stage, state)
            import numpy as np
            require(stage not in self.stages and state["pyramid_backend_name"] == "PVA"
                and state["rescale_backend"] == "CUDA", "Depth capture backend/stage differs")
            data = {}
            for name in ("previous", "current"):
                with state[name+"_pyramid"].rlock_cpu() as values:
                    require(isinstance(values, list) and len(values) == 2, "Expected exactly two pyramid views")
                    arrays = [np.array(value, copy=True) for value in values]
                require([a.shape for a in arrays] == [(256,320),(128,160)], "Realized two-level dimensions differ")
                data[name] = [old.image_record(value) for value in arrays]
            data["proxy"] = {name:old.image_record(old.copy_vpi(state[name+"_motion"])) for name in ("previous","current")}
            self.stages.append(stage)
            self.data[stage] = data
    return DepthCapture


def reference_parity(row, reference, trace):
    new, previous = row["canonical_nontiming"], reference["canonical_nontiming"]
    for name in ("input_pair", "frame_pixel_sha256", "global_configuration"):
        require(new[name] == previous[name], "Depth comparison input differs: "+name)
    require_depth_difference(previous["effective_motion_configuration"], new["effective_motion_configuration"])
    if not trace:
        return dict(inputs_equal=True, only_depth_changed=True, trace_required=True,
                    feature_and_prefix_equal=None, complete=False)
    new_data, old_data = row["capture"]["data"], reference["capture"]["data"]
    require("pyramids" in new_data, "Prepared pyramids not observed")
    for name in ("previous", "current"):
        a,b = new_data["pyramids"][name],old_data["pyramids"][name]
        require(len(a) == 2 and len(b) == 4 and [x["array"] for x in a] == [x["array"] for x in b[:2]],
                "Realized two-level prefix differs: "+name)
        require(new_data["pyramids"]["proxy"][name]["array"] == old_data["pyramids"]["proxy"][name]["array"],
                "Motion proxy differs: "+name)
    if "before_forward" not in new_data:
        return dict(inputs_equal=True, only_depth_changed=True, pyramid_prefix_equal=True,
                    feature_and_prefix_equal=False, complete=False, reason="Feature availability prevented seed observation")
    for key in ("selected_points", "selected_scores", "selected_indices"):
        require(new_data["before_forward"][key] == old_data["before_forward"][key], "Feature seed differs: "+key)
    return dict(inputs_equal=True, only_depth_changed=True, pyramid_prefix_equal=True,
                feature_and_prefix_equal=True, complete=True)


def decode(record):
    import numpy as np
    require(isinstance(record, dict) and set(record) == {"dtype","shape","data_base64","sha256"}, "Invalid captured array")
    dtype=np.dtype(record["dtype"]);shape=record["shape"]
    require(dtype.kind in "uifb" and isinstance(shape,list) and len(shape)<=2
            and all(type(x) is int and x>=0 for x in shape) and np.prod(shape)<=512*640, "Invalid captured array shape/type")
    raw=base64.b64decode(record["data_base64"],validate=True)
    require(hashlib.sha256(raw).hexdigest()==record["sha256"] and len(raw)==int(np.prod(shape))*dtype.itemsize,
            "Captured array integrity differs")
    return np.frombuffer(raw,dtype=dtype).reshape(shape)


def scientific_comparison(new_rows, old_rows, exact_new_parity):
    """No conclusion before exact base/trace parity and all controlled identities."""
    import numpy as np
    require(type(exact_new_parity) is bool, "Explicit parity required")
    if not exact_new_parity:
        return dict(interpretable=False, reason="New base/trace non-timing parity failed", production_promotion=False)
    comparisons={}
    for case in CASES:
        require(new_rows[case+"_base"]["canonical_nontiming"]==new_rows[case+"_trace"]["canonical_nontiming"],
                "Claimed new parity does not match child canonical data")
        row,previous=new_rows[case+"_trace"],old_rows[case+"_trace"]
        check=reference_parity(row,previous,True)
        if not check["complete"]:
            return dict(interpretable=False, reason="Feature seed comparison unavailable", production_promotion=False)
        def observation(item):
            data=item["capture"]["data"]
            require("after_forward" in data, "Immediate-forward capture missing")
            status=decode(data["after_forward"]["forward_status"]).reshape(-1)
            selected=decode(data["before_forward"]["selected_points"])
            require(len(status)==len(selected), "Selected/status count differs")
            result=item["canonical_nontiming"]["result"]
            arrays=result["correspondence"]["arrays"]
            p,q=decode(arrays["previous_points"]),decode(arrays["current_points"])
            errors=np.linalg.norm(q.astype(np.float64)-p.astype(np.float64),axis=1)
            maximum=float(errors.max()) if len(errors) and np.isfinite(errors).all() else None
            model=result["fit"]
            return dict(selected=len(selected), immediate_forward_valid=int(np.count_nonzero(status==0)),
                accepted=len(p), all_accepted_native_static_max_error_px=maximum,
                original_global_fit_accepted=model["quality_status"]=="accepted",
                original_global_parameters=model["parameters"], static_error_population="all final accepted LK correspondences")
        comparisons[case]=dict(depth4=observation(previous),depth2=observation(row),controlled_identity=check)
    bridge=comparisons["bridge"]["depth2"];texture=comparisons["texture"]["depth2"]
    require(comparisons["texture"]["depth4"]["immediate_forward_valid"]==0, "Original texture no longer has zero forward availability")
    parameters=bridge["original_global_parameters"] or {}
    bridge_ok=(bridge["original_global_fit_accepted"] and bridge["all_accepted_native_static_max_error_px"]==0
               and parameters.get("translation_x_px")==0 and parameters.get("translation_y_px")==0)
    availability=bridge_ok and texture["immediate_forward_valid"]>0
    strong=(availability and texture["original_global_fit_accepted"]
            and texture["all_accepted_native_static_max_error_px"] is not None
            and texture["all_accepted_native_static_max_error_px"]<=.25)
    return dict(interpretable=True, cases=comparisons, bridge_zero_motion_retained=bridge_ok,
        texture_immediate_forward_availability_restored=texture["immediate_forward_valid"]>0,
        coarse_level_hypothesis_availability_support=availability, stronger_original_global_recovery=strong,
        production_promotion=False, limitations=["Static generated pairs only, not real-motion or photometric accuracy.",
            "Nonzero forward availability alone is not successful registration.",
            "Lost unchanged endpoints are not valid zero-motion measurements.",
            "Depth intervention does not expose the internal PVA reason for its status flags."])


def run(workspace, freeze_path, digest, case, trace=False):
    workspace=Path(workspace)
    require(case in CASES and type(trace) is bool, "Fixed case/mode required")
    old=old_helper();frozen=bundle(workspace,freeze_path,digest,old)
    mode="trace" if trace else "base";output=workspace/(case+"_"+mode+".json")
    require(not output.exists() and not output.is_symlink(), "Existing evidence; no overwrite")
    with output.open("x"):pass
    receipt=dict(schema=SCHEMA,case=case,mode=mode,completed=False,passed_integrity=False,error=None,
        input_sha256=dict(freeze_sha256=digest,files=frozen["files"],reference_results=REFERENCE_RESULTS,
            reference_freeze_sha256=REFERENCE_FREEZE_SHA,reference_probe_sha256=REFERENCE_PROBE_SHA),
        protocol=PROTOCOL,generated_only=True,source_media_accessed=False,detector_run=False,
        production_changes=False,pyramid_depth_changed=True,other_motion_settings_changed=False,
        focal_estimate_calls=0,global_fits=0,preflight_pva_calls=0,capture=None,
        canonical_nontiming=None,canonical_nontiming_sha256=None)
    estimator=None;capture=capture_type(old)() if trace else None;began=time.perf_counter()
    try:
        prior=references(old)
        phot,selection,helper,generator,reference,runtime,prior_inputs,photo_result=old.load_runtime()
        receipt["input_sha256"]["prior_photometric_inputs"]=prior_inputs
        modules,identities=helper.dependencies(reference)
        info=modules["profile_visible_interaction_v30"].runtime_info
        before,clocks=info(),helper.clock_policy_snapshot();helper.runtime_check(before,runtime)
        import cv2
        from tiny_target.types import Frame,TimestampSource
        from tiny_target.motion import fit_global_motion
        original,global_config=selection.configurations(helper);motion=depth_config(original)
        previous,current,input_pair=old.focal_pair(case,selection,generator,photo_result)
        p=Frame(previous.copy(),0,0,"generated-static-texture:"+case,8,TimestampSource.CONTAINER_RATE)
        q=Frame(current.copy(),100_000_000,1,"generated-static-texture:"+case,8,TimestampSource.CONTAINER_RATE)
        frame_hashes=dict(previous=p.pixel_sha256(),current=q.pixel_sha256())
        receipt.update(runtime_before=before,clock_policy_before=clocks,identities=identities,
            input_pair=input_pair,frame_pixel_sha256=frame_hashes,pyramid_dimensions=dimensions(motion),
            original_motion_configuration=asdict(original),diagnostic_motion_configuration=asdict(motion),
            global_configuration=asdict(global_config),feature_adapter={},diagnostic_postcondition={})
        cv2.setNumThreads(2);reuse=modules["motion_reuse_v12"]
        source=selection.transform_source(reuse.generated_method());old.anchor_lines(source)
        receipt["observation_contract"]=dict(method_sha256=old.METHOD_SHA,estimator_source_changed=False,
            initial_status_changed=False,sys_settrace=trace,additional_cpu_readbacks=trace,extra_pva_compute_calls=0)
        with selection.candidate_adapter(reuse,motion,receipt["feature_adapter"]) as candidate:
            require_depth_difference(prior[case+"_"+mode]["canonical_nontiming"]["effective_motion_configuration"],
                                     receipt["feature_adapter"]["effective_motion_configuration"])
            with phot.allow_empty_diagnostic_result(selection,receipt["diagnostic_postcondition"]):
                estimator=candidate(motion)
                result,timings=old.focal_result(estimator,p,q,fit_global_motion,global_config,phot,
                    reuse.pva.PvaMotionError,receipt,reuse._ESTIMATE,source,capture)
                canonical=dict(input_pair=input_pair,frame_pixel_sha256=frame_hashes,
                    effective_motion_configuration=receipt["feature_adapter"]["effective_motion_configuration"],
                    global_configuration=asdict(global_config),result=result)
                canonical=json.loads(json.dumps(canonical,allow_nan=False))
                receipt.update(canonical_nontiming=canonical,canonical_nontiming_sha256=old.canonical_sha(canonical),timings=timings)
                receipt["reference_identity"]=reference_parity(receipt,prior[case+"_"+mode],trace)
                require(frame_hashes==dict(previous=p.pixel_sha256(),current=q.pixel_sha256()),"Generated Frame pixels mutated")
                estimator.close();require(estimator.closed is True and estimator.failed is False,"Focal lifecycle failed")
                receipt["closed"]=True
        after=info();helper.runtime_check(after,runtime,after=True)
        require(helper.clock_policy_snapshot()==clocks,"Clock policy changed")
        require(bundle(workspace,freeze_path,digest,old)==frozen,"Depth bundle changed")
        references(old)
        require(old.load_runtime()[-2]==prior_inputs and helper.dependencies(reference)[1]==identities,"Frozen dependencies changed")
        receipt.update(completed=True,passed_integrity=True,runtime_after=after,clocks_changed=False)
    except BaseException as exc:
        receipt["error"]=repr(exc)
        raise
    finally:
        if estimator is not None and not estimator.closed:
            try:
                estimator.close();receipt["closed"]=estimator.closed is True
            except BaseException as exc:
                receipt.update(completed=False,passed_integrity=False,cleanup_error=repr(exc))
                if receipt["error"] is None:receipt["error"]=repr(exc)
        if capture is not None and receipt["capture"] is None:
            receipt["capture"]=dict(stages=capture.stages,data=capture.data,error=capture.error,method_calls=capture.method_calls,partial=True)
        receipt["elapsed_seconds"]=time.perf_counter()-began
        require(output.is_file() and not output.is_symlink() and output.stat().st_size==0,"Reserved output changed")
        with output.open("w") as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True);parser.add_argument("--case",choices=CASES,required=True)
    parser.add_argument("--trace",action="store_true")
    args=parser.parse_args();row=run(args.workspace,args.freeze,args.freeze_sha256,args.case,args.trace)
    print(json.dumps({key:row[key] for key in ("case","mode","completed","passed_integrity","canonical_nontiming_sha256")}))
    raise SystemExit(0 if row["passed_integrity"] else 1)
