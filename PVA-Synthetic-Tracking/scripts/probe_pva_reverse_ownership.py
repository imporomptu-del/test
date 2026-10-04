#!/usr/bin/env python3
"""Generated reverse-status ownership contrast, without resetting lost flags.

Plain is unwrapped. Shared/copied perform identical allocation, copy and readback
work; only the actual second constructor's status-object choice differs. CPU
active-byte equality does not expose all internal VPI/device state.
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

SCHEMA="seaqr.pva-reverse-ownership.v1"
WORKSPACE_PATTERN=r"/tmp/seaqr_pva_reverse_20261001_[A-Za-z0-9]{6}"
REFERENCE=Path("/tmp/seaqr_pva_robustness_20261001_sDW19P")
REFERENCE_SHA="9ccc3bb0c0d664a12421c348f06bfa810e1ea9f32a19590f514101e2a8446e79"
REFERENCE_FREEZE_SHA="6cad8a44b9aebf8ac0c335d436727a4a008bc683f82174a2bd69ee2495a322ad"
SAFETY_SHA="a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf"
SOURCES={"probe_pva_reverse_ownership.py","batch_pva_reverse_ownership.py","summarize_pva_reverse_ownership.py",
         "test_pva_reverse_ownership.py","test_batch_pva_reverse_ownership.py","test_summarize_pva_reverse_ownership.py",
         "batch_discovery_pair.py"}
ROLES=(("plain","base"),("shared","base"),("shared","trace"),("copied","base"),("copied","trace"))
SHAPE=(512,640)
SAFETY=dict(start_c=65,stop_c=75,phase_seconds=900,batch_seconds=3600,workers=1,automatic_retries=0)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024*1024,"Invalid metadata file")
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
    path=Path(path);require(path.is_file() and not path.is_symlink() and sha(path)==digest,"Pinned file differs: "+str(path))


def inventory():
    cases=[]
    for identifier,family,truth,group,kind,source,ncc in (
        ("texture__shift_two","texture",[2.,-1.],"global_positive","analytic","texture__shift_two","qualified_with_error_at_most_0.25px"),
        ("texture__shift_fractional","texture",[-1.25,.5],"global_positive","analytic","texture__shift_fractional","qualified_with_error_at_most_0.25px"),
        ("texture__shift_half","texture",[.5,-.75],"global_positive","analytic","texture__shift_half","qualified_with_error_at_most_0.25px"),
        ("out_of_range_shift4","texture",[4.,0.],"global_positive","analytic","out_of_range_shift4","abstain"),
        ("bridge__shift_4_m2","bridge",[4.,-2.],"global_positive","bridge","low_contrast_translated",None),
        ("flat","flat",[.5,-.75],"degenerate","analytic","flat","abstain")):
        cases.append(dict(id=identifier,family=family,truth_displacement_xy=truth,gain=1.,offset_dn=0.,
            scientific_class=group,shape_hw=list(SHAPE),input_kind=kind,source_case_id=source,
            inherited_ncc_expectation=ncc,photometric_reference_id=None))
    return cases


def freeze_spec():
    return dict(schema="pva_reverse_ownership.v1",scope="generated_only_no_production_change",cases=inventory(),
        depth=2,roles=[dict(arm=arm,mode=mode) for arm,mode in ROLES],shape_hw=list(SHAPE),safety=dict(SAFETY),
        reference=dict(workspace=str(REFERENCE),probe_sha256=REFERENCE_SHA,freeze_sha256=REFERENCE_FREEZE_SHA),
        gates=dict(fit_translation_error_px=.25,point_error_px=.25,truth_interior_margin_px=128),
        protocol=dict(fresh_processes=30,focal_estimate_calls_per_child=1,preflight_pva_calls=0,
            original_estimator_source_unchanged=True,forward_status_initialization_unchanged=True,
            constructor_stream_context_unchanged=True,all_motion_quality_gates_unchanged=True,
            shared_copied_difference="actual reverse kptstatus identity only; both allocate/write/read/retain same-sized copy",
            status_copy="preserve every active U8 byte, size and capacity; no zero reset; inactive capacity bytes not claimed equal",
            matched_shared_must_equal_fresh_plain_canonical=True,shared_copied_each_base_trace_exact=True,
            before_reverse_input_equality_required=True,
            recover_cases=["texture__shift_two","texture__shift_fractional"],
            preserve_cases=["texture__shift_half","out_of_range_shift4","bridge__shift_4_m2"],
            negative_case="flat",positive_guard="original fit accepted, translation error<=.25, >=1 fixed-interior accepted point and max error<=.25",
            unsuccessful_branch="stop without retries or tuning",production_promotion=False))


def validate_freeze(value):
    expected=freeze_spec()
    require(isinstance(value,dict) and set(value)==set(expected)|{"source_sha256"}
        and canonical_sha({k:value[k] for k in expected})==canonical_sha(expected),"Reverse ownership protocol differs")
    require(isinstance(value["source_sha256"],dict) and set(value["source_sha256"])==SOURCES
        and all(isinstance(x,str) and re.fullmatch(r"[a-f0-9]{64}",x) for x in value["source_sha256"].values())
        and value["source_sha256"]["batch_discovery_pair.py"]==SAFETY_SHA,"Reverse source inventory differs")


def bundle(workspace,freeze_path,digest):
    workspace,freeze_path=Path(workspace),Path(freeze_path)
    require(re.fullmatch(WORKSPACE_PATTERN,str(workspace)) and workspace.is_dir()
        and not workspace.is_symlink() and os.geteuid()!=0,"Unprivileged scoped reverse workspace required")
    require(freeze_path==workspace/"freeze.json" and isinstance(digest,str) and re.fullmatch(r"[a-f0-9]{64}",digest),"Explicit scoped freeze/hash required")
    pinned(freeze_path,digest);value=read(freeze_path);validate_freeze(value)
    for name,expected in value["source_sha256"].items():pinned(workspace/name,expected)
    pinned(Path(__file__),value["source_sha256"]["probe_pva_reverse_ownership.py"])
    return value


def load_reference():
    path=REFERENCE/"probe_pva_robustness.py";pinned(path,REFERENCE_SHA)
    spec=importlib.util.spec_from_file_location("reverse_frozen_robustness",path)
    robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
    pinned(REFERENCE/"freeze.json",REFERENCE_FREEZE_SHA)
    original=read(REFERENCE/"freeze.json");robust.validate_freeze(original)
    for name,digest in original["source_sha256"].items():pinned(REFERENCE/name,digest)
    require(all(case in robust.inventory() for case in inventory()),"Frozen case metadata differs")
    return robust,robust.load_helpers(),original


class OwnershipFacade:
    """Delegate VPI unchanged except actual second-constructor status argument."""
    def __init__(self,vpi,arm,old,audit):
        require(arm in ("shared","copied"),"Plain must not install a facade")
        self._real,self.arm,self.old,self.audit=vpi,arm,old,audit
        self.source_status=self.clone=self.points=None
        self.retained=[]
        audit.update(enabled=True,arm=arm,constructor_calls=0,forward=None,reverse=None,after=None,
            clone_retained_until_estimator_close=False,extra_pva_pair_calls=0,
            inactive_capacity_bytes_observed=False,device_internal_state_equality_claimed=False)

    def __getattr__(self,name):return getattr(self._real,name)

    def array_record(self,value):
        data=self.old.copy_vpi(value)
        return dict(array=self.old.descriptor(data),metadata=dict(id=int(value.id),size=int(value.size),
            capacity=int(value.capacity),type=str(value.type)))

    def pyramid_record(self,value):
        import numpy as np
        with value.rlock_cpu() as values:
            require(isinstance(values,list) and len(values)==2,"Expected two-level pyramid")
            arrays=[np.array(x,copy=True) for x in values]
        require([x.shape for x in arrays]==[(256,320),(128,160)],"Pyramid dimensions differ")
        return [self.old.image_record(x) for x in arrays]

    def OpticalFlowPyrLK(self,*args,**kwargs):
        import numpy as np
        self.audit["constructor_calls"]+=1;ordinal=self.audit["constructor_calls"]
        require(ordinal in (1,2) and len(args)==2 and kwargs.get("backend")==self._real.Backend.PVA,
                "Unexpected native flow constructor call")
        if ordinal==1:
            require(set(kwargs)=={"backend"},"Forward constructor initializer changed")
            self.audit["forward"]=dict(keyword_names=sorted(kwargs),explicit_kptstatus=False,
                initial_forward_status_unobservable=True,pyramid=self.pyramid_record(args[0]),keypoints=self.array_record(args[1]))
            return self._real.OpticalFlowPyrLK(*args,**kwargs)
        require(set(kwargs)=={"backend","kptstatus"},"Reverse constructor keywords changed")
        self.source_status=kwargs["kptstatus"];self.points=args[1]
        source=self.array_record(self.source_status);points=self.array_record(self.points)
        flags=self.old.copy_vpi(self.source_status)
        require(flags.dtype==np.uint8 and flags.ndim==1 and self.source_status.type==self._real.Type.U8
            and len(flags)==int(self.source_status.size)==int(self.points.size)
            and 0<len(flags)<=384 and int(self.source_status.capacity)>=len(flags),"Reverse status size/type differs")
        # Both matched arms perform this exact initialization. All active zeros
        # are overwritten with the original flags, including every nonzero byte.
        self.clone=self._real.Array.zeros(int(self.source_status.capacity),self.source_status.type)
        self.clone.size=int(self.source_status.size);self.retained.append(self.clone)
        with self.clone.rwlock_cpu() as values:
            values=np.asarray(values)
            require(values.dtype==np.uint8 and values.shape==flags.shape,"Status copy view differs")
            values[:]=flags
        clone=self.array_record(self.clone)
        require(source["array"]==clone["array"]==self.array_record(self.source_status)["array"],"Active copied status bytes differ")
        require(source["metadata"]["id"]!=clone["metadata"]["id"]
            and all(source["metadata"][key]==clone["metadata"][key] for key in ("size","capacity","type")),
            "Copy ownership/size/capacity/type differs")
        chosen=self.source_status if self.arm=="shared" else self.clone
        require(points==self.array_record(self.points),"Keypoint bytes or identity changed during status copy")
        self.audit["reverse"]=dict(keyword_names=sorted(kwargs),backend=str(kwargs["backend"]),
            pyramid=self.pyramid_record(args[0]),keypoints_before=points,source_status_before=source,clone_status_before=clone,
            active_status_bytes_equal=True,size_capacity_type_equal=True,chosen_status="source" if self.arm=="shared" else "clone",
            actual_argument_is_forward_status=chosen is self.source_status,actual_argument_is_clone=chosen is self.clone,
            actual_argument_metadata=dict(source["metadata"] if self.arm=="shared" else clone["metadata"]),cloned_in_both_arms=True,
            nonzero_flags_preserved=True,constructor_context_changed=False,coordinates_argument_unchanged=True,
            allocation_context="unchanged ambient context; no stream context entered")
        supplied=dict(kwargs);supplied["kptstatus"]=chosen
        return self._real.OpticalFlowPyrLK(*args,**supplied)

    def finish(self,correspondence_returned):
        if not correspondence_returned:
            require(self.audit["constructor_calls"]==0,"Expected feature unavailability after unexpected constructor activity")
            self.audit["expected_no_flow"]=True
            return
        require(self.audit["constructor_calls"]==2 and len(self.retained)==1 and self.retained[0] is self.clone,
                "Reverse clone/call lifecycle differs")
        self.audit["after"]=dict(source_status=self.array_record(self.source_status),clone_status=self.array_record(self.clone),
            keypoints=self.array_record(self.points),clone_retained=True)

    def closed(self):
        self.audit["clone_retained_until_estimator_close"]=self.clone is None or (len(self.retained)==1 and self.retained[0] is self.clone)


def capture_type(depth,old,audit):
    class Capture(depth.capture_type(old)):
        def record(self,stage,state):
            super().record(stage,state)
            if stage=="after_backward":
                row=self.data[stage]
                row["estimator_local_backward_initial_status_is_forward_status"]=row.pop("backward_constructor_input_status_was_forward_status")
                require(audit.get("reverse") is not None,"Actual reverse argument audit missing")
                row["actual_backward_argument_is_forward_status"]=audit["reverse"]["actual_argument_is_forward_status"]
                row["actual_backward_argument_is_clone"]=audit["reverse"]["actual_argument_is_clone"]
    return Capture


def run(workspace,freeze_path,digest,case_id,arm,trace=False):
    workspace=Path(workspace);mode="trace" if trace else "base"
    cases={x["id"]:x for x in inventory()}
    require(case_id in cases and type(trace) is bool and (arm,mode) in ROLES,"Undeclared case/arm/mode")
    frozen=bundle(workspace,freeze_path,digest);case=cases[case_id]
    output=workspace/(case_id+"_"+arm+"_"+mode+".json")
    require(not output.exists() and not output.is_symlink(),"Existing focal evidence; no overwrite")
    with output.open("x"):pass
    receipt=dict(schema=SCHEMA,case=case_id,case_metadata=case,depth=2,arm=arm,mode=mode,completed=False,passed_integrity=False,error=None,
        input_sha256=dict(freeze_sha256=digest,source_sha256=frozen["source_sha256"],reference=frozen["reference"]),
        generated_only=True,source_media_accessed=False,detector_run=False,production_changes=False,production_promotion=False,
        focal_estimate_calls=0,global_fits=0,preflight_pva_calls=0,capture=None,trace_truth=None,closed=None,
        canonical_nontiming=None,canonical_nontiming_sha256=None,
        constructor_audit=dict(enabled=False,arm=arm,constructor_calls=0,forward=None,reverse=None,after=None),
        limitations=["Active CPU bytes, not inactive capacity bytes or internal device state, are verified equal.",
            "Matched allocation/readback control must match fresh plain output before ownership inference.",
            "Shared/copy constructor context and forward initializer remain legacy defaults.",
            "Copy does not reset lost flags, alter point coordinates, or lower motion-quality gates.",
            "Synthetic diagnostic only; no production promotion, no reliability distribution."])
    estimator=facade=capture=real_vpi=None;began=time.perf_counter()
    try:
        robust,(depth,old,loaded),prior_freeze=load_reference()
        phot,selection,helper,generator,reference,runtime,prior_inputs,prior=loaded
        receipt["input_sha256"].update(reference_source_sha256=prior_freeze["source_sha256"],prior_photometric_inputs=prior_inputs,
                                      prior_generated_result_sha256=old.PRIOR_RESULT_SHA)
        modules,identities=helper.dependencies(reference)
        info=modules["profile_visible_interaction_v30"].runtime_info
        before,clocks=info(),helper.clock_policy_snapshot();helper.runtime_check(before,runtime)
        import cv2
        from tiny_target.types import Frame,TimestampSource
        from tiny_target.motion import fit_global_motion
        original,global_config=selection.configurations(helper);motion=robust.configuration_for_depth(original,2,depth)
        previous,current,input_pair=robust.pair_for_case(case,selection,generator,prior)
        p=Frame(previous.copy(),0,0,"generated-pva-robustness:"+case_id,8,TimestampSource.CONTAINER_RATE)
        q=Frame(current.copy(),100_000_000,1,"generated-pva-robustness:"+case_id,8,TimestampSource.CONTAINER_RATE)
        frame_hashes=dict(previous=p.pixel_sha256(),current=q.pixel_sha256())
        receipt.update(runtime_before=before,clock_policy_before=clocks,identities=identities,input_pair=input_pair,
            frame_pixel_sha256=frame_hashes,original_motion_configuration=asdict(original),requested_motion_configuration=asdict(motion),
            global_configuration=asdict(global_config),pyramid_dimensions=depth.dimensions(motion),feature_adapter={},diagnostic_postcondition={})
        cv2.setNumThreads(2);reuse=modules["motion_reuse_v12"]
        source=selection.transform_source(reuse.generated_method());old.anchor_lines(source)
        receipt["observation_contract"]=dict(method_sha256=old.METHOD_SHA,estimator_source_changed=False,sys_settrace=trace,
            forward_initializer_changed=False,constructor_context_changed=False,reverse_status_object_substitution=arm=="copied",
            matched_constructor_readbacks_and_copy=arm!="plain",extra_pva_compute_calls=0)
        if trace:capture=capture_type(depth,old,receipt["constructor_audit"])()
        with selection.candidate_adapter(reuse,motion,receipt["feature_adapter"]) as candidate:
            depth.require_depth_difference(asdict(selection.candidate_config(original)[0]),receipt["feature_adapter"]["effective_motion_configuration"])
            with phot.allow_empty_diagnostic_result(selection,receipt["diagnostic_postcondition"]):
                estimator=candidate(motion);real_vpi=estimator._vpi
                if arm!="plain":
                    facade=OwnershipFacade(real_vpi,arm,old,receipt["constructor_audit"]);estimator._vpi=facade
                result,timings=old.focal_result(estimator,p,q,fit_global_motion,global_config,phot,reuse.pva.PvaMotionError,
                    receipt,reuse._ESTIMATE,source,capture)
                if facade is not None:facade.finish(result["correspondence"] is not None)
                canonical=dict(case_metadata=case,input_pair=input_pair,frame_pixel_sha256=frame_hashes,
                    effective_motion_configuration=receipt["feature_adapter"]["effective_motion_configuration"],
                    global_configuration=asdict(global_config),result=result,accepted_truth=robust.accepted_truth(result,case,depth.decode))
                canonical=json.loads(json.dumps(canonical,allow_nan=False))
                receipt.update(canonical_nontiming=canonical,canonical_nontiming_sha256=canonical_sha(canonical),timings=timings)
                if trace:receipt["trace_truth"]=robust.trace_truth(receipt["capture"],case,depth.decode,old.descriptor)
                require(frame_hashes==dict(previous=p.pixel_sha256(),current=q.pixel_sha256()),"Generated Frame pixels mutated")
                require(input_pair["previous_pixel_sha256"]==generator.array_sha(previous)
                    and input_pair["current_pixel_sha256"]==generator.array_sha(current),"Generated source pixels mutated")
                estimator.close();require(estimator.closed is True and estimator.failed is False,"Focal lifecycle failed")
                receipt["closed"]=True
                if facade is not None:facade.closed()
                estimator._vpi=real_vpi
        after=info();helper.runtime_check(after,runtime,after=True)
        require(helper.clock_policy_snapshot()==clocks,"Clock policy changed")
        require(bundle(workspace,freeze_path,digest)==frozen,"Reverse bundle changed")
        require(load_reference()[1][2][-2]==prior_inputs and helper.dependencies(reference)[1]==identities,"Pinned dependencies changed")
        receipt.update(completed=True,passed_integrity=True,runtime_after=after,clocks_changed=False)
    except BaseException as exc:
        receipt["error"]=repr(exc)
        raise
    finally:
        if estimator is not None:
            if not estimator.closed:
                try:
                    estimator.close();receipt["closed"]=estimator.closed is True
                    if facade is not None:facade.closed()
                except BaseException as exc:
                    receipt.update(completed=False,passed_integrity=False,cleanup_error=repr(exc))
                    if receipt["error"] is None:receipt["error"]=repr(exc)
            if real_vpi is not None:estimator._vpi=real_vpi
        if capture is not None and receipt["capture"] is None:
            receipt["capture"]=dict(stages=capture.stages,data=capture.data,error=capture.error,method_calls=capture.method_calls,partial=True)
        receipt["elapsed_seconds"]=time.perf_counter()-began
        require(output.is_file() and not output.is_symlink() and output.stat().st_size==0,"Reserved output changed")
        with output.open("w") as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True);parser.add_argument("--case",choices=[x["id"] for x in inventory()],required=True)
    parser.add_argument("--arm",choices=("plain","shared","copied"),required=True);parser.add_argument("--trace",action="store_true")
    args=parser.parse_args();row=run(args.workspace,args.freeze,args.freeze_sha256,args.case,args.arm,args.trace)
    print(json.dumps({k:row[k] for k in ("case","arm","mode","completed","passed_integrity","canonical_nontiming_sha256")}))
    raise SystemExit(0 if row["passed_integrity"] else 1)
