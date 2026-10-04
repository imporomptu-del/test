#!/usr/bin/env python3
"""One frozen generated pair/depth in a fresh process; no production promotion.

Scientific unavailable/inaccurate outcomes are retained. Only integrity,
backend, instrumentation and cleanup failures abort the supervised batch.
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
import time

SCHEMA="seaqr.pva-robustness.v1"
WORKSPACE_PATTERN=r"/tmp/seaqr_pva_robustness_20261001_[A-Za-z0-9]{6}"
SHAPE=(512,640)
DEPTHS=(4,2)
MODES=("base","trace")
DEPTH_HELPER=Path("/tmp/seaqr_pva_depth_control_20261001_HzshKd/probe_pva_depth_control.py")
DEPTH_HELPER_SHA="48345393d7f2e8c760169946cdd6e88a8e018152fd403891d2fe26730de2ed0f"
STATIC_HELPER_SHA="48cff34b8248fdcfbc447eda32774ff91f33f88b78bb62c6986b65c50d634907"
PHOTOMETRIC_HELPER_SHA="569220932c8132944120ffc6026f59e01c47cb81cfd11d9c2875f890566f32f1"
GENERATOR_SHA="b1d915604327cca4a5a7bf83a3eef9eca197d43ecd9cc8e2cfc63a1d61aeb97c"
SAFETY_SHA="a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
SOURCES={"probe_pva_robustness.py","batch_pva_robustness.py","summarize_pva_robustness.py",
         "test_pva_robustness.py","test_batch_pva_robustness.py","test_summarize_pva_robustness.py",
         "batch_discovery_pair.py"}
SAFETY=dict(start_c=65,stop_c=75,phase_seconds=900,batch_seconds=3600,workers=1,automatic_retries=0)
GATES=dict(fit_translation_error_px=.25,point_error_px=.25,truth_interior_margin_px=128)
CONDITIONS=(
    ("static",(0.,0.),1.,0.),("shift_half",(.5,-.75),1.,0.),
    ("shift_two",(2.,-1.),1.,0.),("shift_fractional",(-1.25,.5),1.,0.),
    ("static_gain08_offset12",(0.,0.),.8,12.),("shift_half_gain08_offset12",(.5,-.75),.8,12.),
    ("static_gain12_offsetm12",(0.,0.),1.2,-12.),("shift_half_gain12_offsetm12",(.5,-.75),1.2,-12.))


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024*1024,"Invalid metadata file")
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"Duplicate JSON key");value[key]=item
        return value
    def number(text):
        value=float(text);require(math.isfinite(value),"Nonfinite JSON number");return value
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_float=number,
                      parse_constant=lambda _:require(False,"Nonfinite JSON constant"))


def pinned(path,digest):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path)==digest,"Pinned file differs: "+str(path))


def inventory():
    """Stdlib-only declaration used by the supervisor and freeze generator."""
    result=[]
    def case(identifier,family,truth,gain,offset,group,kind,source,ncc,reference):
        return dict(id=identifier,family=family,truth_displacement_xy=list(truth),gain=gain,offset_dn=offset,
            scientific_class=group,shape_hw=list(SHAPE),input_kind=kind,source_case_id=source,
            inherited_ncc_expectation=ncc,photometric_reference_id=reference)
    for family in ("texture","corner"):
        for name,truth,gain,offset in CONDITIONS:
            identifier=family+"__"+name
            reference=family+"__"+("static" if truth==(0.,0.) else "shift_half") if gain!=1. else None
            result.append(case(identifier,family,truth,gain,offset,"global_positive" if family=="texture" else "sparse_corner",
                "analytic",identifier,"qualified_with_error_at_most_0.25px",reference))
    for identifier,family,truth,group in (("flat","flat",(.5,-.75),"degenerate"),
        ("straight_edge","edge",(.5,-.75),"degenerate"),("periodic_ambiguity","periodic",(.5,-.75),"degenerate"),
        ("out_of_range_shift4","texture",(4.,0.),"global_positive")):
        result.append(case(identifier,family,truth,1.,0.,group,"analytic",identifier,"abstain",None))
    for name,source,truth in (("static","low_contrast_static",(0.,0.)),("shift_4_m2","low_contrast_translated",(4.,-2.))):
        for suffix,gain,offset in (("",1.,0.),("_gain08_offset12",.8,12.),("_gain12_offsetm12",1.2,-12.)):
            identifier="bridge__"+name+suffix
            result.append(case(identifier,"bridge",truth,gain,offset,"global_positive","bridge",source,None,
                               "bridge__"+name if suffix else None))
    return result


def freeze_spec():
    return dict(schema="pva_robustness.v1",scope="generated_only_no_production_change",cases=inventory(),
        shape_hw=list(SHAPE),depths=list(DEPTHS),modes=list(MODES),safety=dict(SAFETY),gates=dict(GATES),
        helpers=dict(depth_path=str(DEPTH_HELPER),depth_sha256=DEPTH_HELPER_SHA,static_sha256=STATIC_HELPER_SHA,
            photometric_sha256=PHOTOMETRIC_HELPER_SHA,generator_sha256=GENERATOR_SHA),
        protocol=dict(fresh_processes=104,focal_estimate_calls_per_child=1,preflight_pva_calls=0,
            status_policy="legacy_default",only_depth_configuration_difference="pyramid_levels",
            base_trace_exact_parity_required=True,cross_depth_input_seed_prefix_identity_required=True,
            bridge_photometry="current U8 to float64 gain+offset, np.rint once, no clipping; previous unchanged",
            truth_cohort="selected previous native p and p+known truth inside fixed128px support; never measured q",
            global_positive_cases=15,global_positive_guard="each original global fit accepted and translation error <= 0.25px",
            negative_cases=3,negative_guard="no original global fit accepted",
            sparse_corner_cases=8,sparse_corner_global_fit_required=False,
            nondegenerate_point_guard="each of23cases has >=1 interior accepted point and maximum error <=0.25px including original RANSAC outliers",
            accuracy_gates_evaluated_per_depth=True,
            overall_ready_requires="all integrity, point, global-positive and negative-case guards pass; no promotion",
            science_failure_is_execution_failure=False,production_promotion=False,
            ncc_expectations_are_not_pva_gates=True))


def validate_freeze(value):
    expected=freeze_spec()
    require(isinstance(value,dict) and set(value)==set(expected)|{"source_sha256"}
        and canonical_sha({k:value[k] for k in expected})==canonical_sha(expected),"Frozen robustness protocol differs")
    require(isinstance(value["source_sha256"],dict) and set(value["source_sha256"])==SOURCES
        and all(isinstance(x,str) and re.fullmatch(r"[0-9a-f]{64}",x) for x in value["source_sha256"].values())
        and value["source_sha256"]["batch_discovery_pair.py"]==SAFETY_SHA,"Frozen source inventory differs")


def bundle(workspace,freeze_path,digest):
    workspace,freeze_path=Path(workspace),Path(freeze_path)
    require(re.fullmatch(WORKSPACE_PATTERN,str(workspace)) and workspace.is_dir()
        and not workspace.is_symlink() and os.geteuid()!=0,"Unprivileged scoped robustness workspace required")
    require(freeze_path==workspace/"freeze.json" and isinstance(digest,str) and re.fullmatch(r"[a-f0-9]{64}",digest),"Explicit scoped freeze/hash required")
    pinned(freeze_path,digest);value=read(freeze_path);validate_freeze(value)
    for name,expected in value["source_sha256"].items():pinned(workspace/name,expected)
    pinned(Path(__file__),value["source_sha256"]["probe_pva_robustness.py"])
    return value


def load_helpers():
    pinned(DEPTH_HELPER,DEPTH_HELPER_SHA)
    spec=importlib.util.spec_from_file_location("robustness_depth_helper",DEPTH_HELPER)
    depth=importlib.util.module_from_spec(spec);spec.loader.exec_module(depth)
    require(depth.REFERENCE_PROBE_SHA==STATIC_HELPER_SHA,"Static helper pin differs")
    old=depth.old_helper()
    require(old.PHOTOMETRIC_SHA==PHOTOMETRIC_HELPER_SHA,"Photometric helper pin differs")
    return depth,old,old.load_runtime()


def bridge_photometry(current,gain,offset):
    import numpy as np
    require(current.shape==SHAPE and current.dtype==np.uint8
        and (gain,offset) in ((1.,0.),(.8,12.),(1.2,-12.)),"Undeclared bridge photometry")
    intended=current.astype(np.float64)*gain+offset
    require(np.isfinite(intended).all() and intended.min()>=0 and intended.max()<=255,"Bridge photometry would clip")
    result=np.rint(intended).astype(np.uint8)
    return result,dict(current_maximum_abs_rounding_error_dn=float(np.max(np.abs(intended-result))),
        clipped_pixels=0,gain=gain,offset_dn=offset,applied_once_to_original_current_u8=True)


def pair_for_case(case,selection,generator,prior):
    import numpy as np
    require(case in inventory(),"Undeclared case")
    if case["input_kind"]=="analytic":
        pair=next(pair for pair in generator.generated_controls(SHAPE) if pair["case"]["case_id"]==case["id"])
        meta=pair["case"]
        require(meta["truth_displacement_xy"]==case["truth_displacement_xy"] and meta["current_gain"]==case["gain"]
            and meta["current_offset_dn"]==case["offset_dn"] and meta["family"]==case["family"]
            and meta["expectation"]==case["inherited_ncc_expectation"],"Original analytic case changed")
        previous,current=pair["previous_gray"],pair["current_gray"]
        old=next(row for row in prior["cases"] if row["case"]["case_id"]==case["id"])
        require(pair["previous_pixel_sha256"]==old["previous_pixel_sha256"]
            and pair["current_pixel_sha256"]==old["current_pixel_sha256"],"Original analytic pixels changed")
        quantization=pair["quantization"]
        original_current_sha=generator.array_sha(current)
    else:
        name,previous,original_current,truth=next(item for item in selection.generated_controls() if item[0]==case["source_case_id"])
        require(list(truth)==case["truth_displacement_xy"],"Original bridge truth differs")
        old=next(row for row in prior["controls"] if row["name"]==name)
        require(old["passed"] is True and old["closed"] is True
            and hashlib.sha256(previous.tobytes()).hexdigest()==old["previous_pixel_sha256"]
            and hashlib.sha256(original_current.tobytes()).hexdigest()==old["current_pixel_sha256"],"Original bridge pixels differ")
        original_current_sha=generator.array_sha(original_current)
        current,quantization=bridge_photometry(original_current,case["gain"],case["offset_dn"])
    require(previous.shape==current.shape==SHAPE and previous.dtype==current.dtype==np.uint8,"Generated shape/dtype differs")
    return previous,current,dict(native_shape_hw=list(SHAPE),previous_pixel_sha256=generator.array_sha(previous),
        current_pixel_sha256=generator.array_sha(current),original_current_pixel_sha256=original_current_sha,
        quantization=quantization,source_media_accessed=False)


def support_mask(p,truth):
    import numpy as np
    p=np.asarray(p,np.float64);truth=np.asarray(truth,np.float64)
    require(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() and truth.shape==(2,) and np.isfinite(truth).all(),"Invalid native truth geometry")
    low=GATES["truth_interior_margin_px"];high=np.array([SHAPE[1],SHAPE[0]],np.float64)-low
    return ((p>=low)&(p<high)&(p+truth>=low)&(p+truth<high)).all(axis=1)


def error_statistics(values):
    import numpy as np
    a=np.asarray(values,np.float64)
    require(a.ndim==1 and np.isfinite(a).all() and (a>=0).all(),"Nonfinite or negative truth errors")
    return dict(count=len(a),median_px=float(np.median(a)) if len(a) else None,
        p90_px=float(np.quantile(a,.9)) if len(a) else None,maximum_px=float(a.max()) if len(a) else None,
        above_0_25px=int(np.count_nonzero(a>GATES["point_error_px"])))


def accepted_truth(result,case,decoder):
    import numpy as np
    if result["correspondence"] is None:
        return dict(measured=False,accepted_count=None,truth_interior_count=None,all_accepted=None,truth_interior=None,
            original_fit_ran=False,original_fit_accepted=None,candidate_translation_error_px=None,accepted_fit_error_px=None,
            point_guard_passed=False,global_guard_passed=False,unavailable_reason=result["reason"])
    correspondence=result["correspondence"];arrays=correspondence["arrays"]
    p,q=decoder(arrays["previous_points"]),decoder(arrays["current_points"])
    require(p.shape==q.shape and p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() and np.isfinite(q).all(),"Invalid accepted endpoints")
    truth=np.array(case["truth_displacement_xy"],np.float64)
    interior=support_mask(p,truth);errors=np.linalg.norm(q.astype(np.float64)-p.astype(np.float64)-truth,axis=1)
    model=result["fit"];parameters=model["parameters"] or {}
    vector=[parameters.get("translation_x_px"),parameters.get("translation_y_px")]
    candidate_available=all(type(x) in (int,float) and math.isfinite(x) for x in vector)
    candidate_error=float(np.linalg.norm(np.array(vector)-truth)) if candidate_available else None
    fit_accepted=model["quality_status"]=="accepted"
    require(not fit_accepted or candidate_available,"Accepted original fit lacks finite translation")
    inner=error_statistics(errors[interior]);all_points=error_statistics(errors)
    mask=decoder(result["inlier_mask"])
    require(mask.dtype==np.bool_ and mask.shape==(len(p),),"Original inlier mask differs")
    ransac_ran=model["metrics"].get("ransac_samples_evaluated",0)>0
    return dict(measured=bool(len(p)),accepted_count=len(p),truth_interior_count=int(interior.sum()),
        all_accepted=all_points,truth_interior=inner,original_fit_ran=True,original_fit_accepted=fit_accepted,
        original_ransac_executed=ransac_ran,
        original_inlier_interior=error_statistics(errors[interior&mask]) if ransac_ran else None,
        original_outlier_interior=error_statistics(errors[interior&~mask]) if ransac_ran else None,
        candidate_translation_available=candidate_available,candidate_translation_error_px=candidate_error,
        rejected_candidate_error_is_descriptive_only=not fit_accepted,
        accepted_fit_error_px=candidate_error if fit_accepted else None,
        accepted_wrong_fit=fit_accepted and candidate_error>GATES["fit_translation_error_px"],
        point_guard_passed=inner["count"]>0 and inner["maximum_px"]<=GATES["point_error_px"],
        global_guard_passed=fit_accepted and candidate_error<=GATES["fit_translation_error_px"],
        original_global_gates_changed=False,truth_population="accepted p and known p+truth in fixed128px interior; no observed-q selection")


def trace_truth(capture,case,decoder,descriptor):
    import numpy as np
    data=capture["data"]
    if "final_filter" not in data:
        return dict(available=False,reason="Feature unavailability prevented full selected/flow capture",selected_count=None,
                    fixed_support_count=None,missing_points_are_not_zero_error=True)
    p=decoder(data["final_filter"]["previous_full_points"]).astype(np.float64)
    expected=np.array(case["truth_displacement_xy"],np.float64)
    cohort=support_mask(p,expected)
    selected=decoder(data["before_forward"]["selected_points"])
    require(selected.shape==p.shape and np.array_equal((selected.astype(np.float64)+.5)*2-.5,p),"Native lift differs from selected points")
    q_motion=decoder(data["after_forward"]["tracked_points"])
    q=(q_motion.astype(np.float64)+.5)*2-.5
    flags=decoder(data["after_forward"]["forward_status"]).reshape(-1)
    require(q.shape==p.shape and flags.shape==(len(p),) and flags.dtype==np.uint8,"Immediate-forward schema differs")
    valid_status=flags==0;finite=np.isfinite(q).all(axis=1)
    usable=valid_status&finite;errors=np.full(len(p),np.nan,np.float64)
    errors[usable]=np.linalg.norm(q[usable]-p[usable]-expected,axis=1)
    accepted=decoder(data["final_filter"]["accepted_mask"])
    require(accepted.dtype==np.bool_ and accepted.shape==(len(p),),"Accepted selected-mask differs")
    accepted_indices=decoder(data["final_filter"]["selected_indices_of_accepted_points"])
    require(np.array_equal(accepted_indices,np.flatnonzero(accepted)),"Accepted selected-index mapping differs")
    return dict(available=True,selected_count=len(p),fixed_support_count=int(cohort.sum()),
        fixed_support_selected_indices=descriptor(np.flatnonzero(cohort)),
        immediate_forward_status_valid_count=int(valid_status.sum()),
        immediate_forward_lost_count=int((~valid_status).sum()),
        immediate_forward_nonfinite_status_valid_count=int((valid_status&~finite).sum()),
        immediate_forward_measurable_count=int(usable.sum()),
        fixed_support_immediate_forward_measurable_count=int((cohort&usable).sum()),
        fixed_support_immediate_forward_missing_count=int((cohort&~usable).sum()),
        immediate_forward_all_measurable=error_statistics(errors[usable]),
        immediate_forward_fixed_support=error_statistics(errors[cohort&usable]),
        immediate_forward_error_px_with_nan_for_missing=descriptor(errors),
        final_accepted_count=int(accepted.sum()),fixed_support_final_accepted_count=int((cohort&accepted).sum()),
        fixed_support_final_lost_count=int((cohort&~accepted).sum()),
        original_rejection_counts=data["final_filter"]["rejection_counts"],
        missing_points_are_not_zero_error=True,selection_uses_measured_endpoint=False,
        final_forward_status_is_shared_post_backward=True)


def configuration_for_depth(original,requested,depth_helper):
    require(type(requested) is int and requested in DEPTHS,"Only declared integer depths allowed")
    require(original.pyramid_levels==4 and original.feature_image_scale==original.pyramid_scale==.5
        and original.flow_status_policy=="legacy_default" and original.forward_backward_check is True,
        "Original depth/status policy differs")
    return original if requested==4 else depth_helper.depth_config(original)


def run(workspace,freeze_path,digest,case_id,requested_depth,trace=False):
    workspace=Path(workspace)
    cases={row["id"]:row for row in inventory()}
    require(case_id in cases and type(requested_depth) is int and requested_depth in DEPTHS and type(trace) is bool,
            "Undeclared case/depth/mode")
    frozen=bundle(workspace,freeze_path,digest);case=cases[case_id]
    mode="trace" if trace else "base";output=workspace/(case_id+"_depth"+str(requested_depth)+"_"+mode+".json")
    require(not output.exists() and not output.is_symlink(),"Existing focal evidence; no overwrite")
    with output.open("x"):pass
    receipt=dict(schema=SCHEMA,case=case_id,case_metadata=case,depth=requested_depth,mode=mode,
        completed=False,passed_integrity=False,error=None,generated_only=True,source_media_accessed=False,
        detector_run=False,production_changes=False,production_promotion=False,
        input_sha256=dict(freeze_sha256=digest,source_sha256=frozen["source_sha256"],helpers=frozen["helpers"]),
        focal_estimate_calls=0,global_fits=0,preflight_pva_calls=0,capture=None,trace_truth=None,
        canonical_nontiming=None,canonical_nontiming_sha256=None,closed=None,
        limitations=["Scientific failures/unavailable measurements are not execution failures.",
            "Single fresh pair per child; no production throughput or repeated-trial reliability estimate.",
            "Legacy implicit initial forward status is not observable or assumed zero.",
            "Original forward-status readback is shared post-backward; immediate-forward capture is separate.",
            "Generated-only known shifts/photometry, not real-scene or detector accuracy.",
            "Inherited NCC query expectation is provenance, not a PVA gate; shift4 is within PVA range.",
            "Readback noninterference requires matching uninstrumented fresh-process canonical output."])
    estimator=None;capture=None;began=time.perf_counter()
    try:
        depth,old,loaded=load_helpers()
        phot,selection,helper,generator,reference,runtime,prior_inputs,prior=loaded
        require(generator is not None and phot.GENERATOR_SHA==GENERATOR_SHA,"Analytic generator pin differs")
        receipt["input_sha256"].update(prior_photometric_inputs=prior_inputs,prior_generated_result_sha256=old.PRIOR_RESULT_SHA)
        modules,identities=helper.dependencies(reference)
        info=modules["profile_visible_interaction_v30"].runtime_info
        before,clocks=info(),helper.clock_policy_snapshot();helper.runtime_check(before,runtime)
        import cv2
        from tiny_target.types import Frame,TimestampSource
        from tiny_target.motion import fit_global_motion
        original,global_config=selection.configurations(helper)
        motion=configuration_for_depth(original,requested_depth,depth)
        previous,current,input_pair=pair_for_case(case,selection,generator,prior)
        p=Frame(previous.copy(),0,0,"generated-pva-robustness:"+case_id,8,TimestampSource.CONTAINER_RATE)
        q=Frame(current.copy(),100_000_000,1,"generated-pva-robustness:"+case_id,8,TimestampSource.CONTAINER_RATE)
        frame_hashes=dict(previous=p.pixel_sha256(),current=q.pixel_sha256())
        receipt.update(runtime_before=before,clock_policy_before=clocks,identities=identities,input_pair=input_pair,
            frame_pixel_sha256=frame_hashes,original_motion_configuration=asdict(original),
            requested_motion_configuration=asdict(motion),global_configuration=asdict(global_config),
            pyramid_dimensions=selection.validate_pyramid_dimensions(SHAPE,motion) if requested_depth==4 else depth.dimensions(motion),
            feature_adapter={},diagnostic_postcondition={})
        cv2.setNumThreads(2);reuse=modules["motion_reuse_v12"]
        source=selection.transform_source(reuse.generated_method());old.anchor_lines(source)
        receipt["observation_contract"]=dict(method_sha256=old.METHOD_SHA,estimator_source_changed=False,
            sys_settrace=trace,additional_cpu_readbacks=trace,initial_status_changed=False,
            extra_pva_compute_calls=0,only_depth_configuration_difference="pyramid_levels" if requested_depth==2 else None)
        if trace:capture=old.Capture() if requested_depth==4 else depth.capture_type(old)()
        original_effective=asdict(selection.candidate_config(original)[0])
        with selection.candidate_adapter(reuse,motion,receipt["feature_adapter"]) as candidate:
            actual=receipt["feature_adapter"]["effective_motion_configuration"]
            if requested_depth==2:depth.require_depth_difference(original_effective,actual)
            else:require(canonical_sha(actual)==canonical_sha(original_effective),"Original effective four-level configuration changed")
            with phot.allow_empty_diagnostic_result(selection,receipt["diagnostic_postcondition"]):
                estimator=candidate(motion)
                result,timings=old.focal_result(estimator,p,q,fit_global_motion,global_config,phot,
                    reuse.pva.PvaMotionError,receipt,reuse._ESTIMATE,source,capture)
                truth=accepted_truth(result,case,depth.decode)
                canonical=dict(case_metadata=case,input_pair=input_pair,frame_pixel_sha256=frame_hashes,
                    effective_motion_configuration=actual,global_configuration=asdict(global_config),
                    result=result,accepted_truth=truth)
                canonical=json.loads(json.dumps(canonical,allow_nan=False))
                receipt.update(canonical_nontiming=canonical,canonical_nontiming_sha256=canonical_sha(canonical),timings=timings)
                if trace:receipt["trace_truth"]=trace_truth(receipt["capture"],case,depth.decode,old.descriptor)
                require(frame_hashes==dict(previous=p.pixel_sha256(),current=q.pixel_sha256()),"Generated Frame mutated")
                require(input_pair["previous_pixel_sha256"]==generator.array_sha(previous)
                    and input_pair["current_pixel_sha256"]==generator.array_sha(current),"Original generated images mutated")
                estimator.close();require(estimator.closed is True and estimator.failed is False,"Focal lifecycle failed")
                receipt["closed"]=True
        after=info();helper.runtime_check(after,runtime,after=True)
        require(helper.clock_policy_snapshot()==clocks,"Clock policy changed")
        require(bundle(workspace,freeze_path,digest)==frozen,"Frozen robustness bundle changed")
        require(load_helpers()[2][-2]==prior_inputs and helper.dependencies(reference)[1]==identities,"Pinned runtime dependencies changed")
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
        require(output.is_file() and not output.is_symlink() and output.stat().st_size==0,"Reserved focal output changed")
        with output.open("w") as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True);parser.add_argument("--case",choices=[x["id"] for x in inventory()],required=True)
    parser.add_argument("--depth",type=int,choices=DEPTHS,required=True);parser.add_argument("--trace",action="store_true")
    args=parser.parse_args();row=run(args.workspace,args.freeze,args.freeze_sha256,args.case,args.depth,args.trace)
    print(json.dumps({key:row[key] for key in ("case","depth","mode","completed","passed_integrity","canonical_nontiming_sha256")}))
    raise SystemExit(0 if row["passed_integrity"] else 1)
