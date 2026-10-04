"""Generated arrays and mocked VPI ownership/lifecycle only; no accelerator/media."""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=imported("reverse_worker_test",ROOT/"scripts/probe_pva_reverse_ownership.py")
robust=imported("reverse_robust_test",ROOT/"scripts/probe_pva_robustness.py")
old=imported("reverse_static_test",ROOT/"scripts/probe_static_pva_texture.py")
depth=imported("reverse_depth_test",ROOT/"scripts/probe_pva_depth_control.py")
generator=imported("reverse_generator_test",ROOT/"scripts/validate_motion_patch_controls.py")

def frozen():
    value=m.freeze_spec();value["source_sha256"]={name:m.SAFETY_SHA if name=="batch_discovery_pair.py" else "a"*64 for name in m.SOURCES}
    return value

class Array:
    next_id=1
    def __init__(self,values,kind="U8",capacity=None,events=None):
        self.values=np.array(values,copy=True);self.type=kind
        self.capacity=len(values) if capacity is None else capacity
        self._size=len(values);self.id=Array.next_id;Array.next_id+=1
        self.events=[] if events is None else events;self.read_locks=0;self.write_locks=0
        shape=(self.capacity,)+self.values.shape[1:]
        self.storage=np.zeros(shape,dtype=self.values.dtype);self.storage[:self._size]=self.values
    @property
    def size(self):return self._size
    @size.setter
    def size(self,value):
        assert 0<=value<=self.capacity
        self._size=value;self.events.append(("size",value))
    @contextmanager
    def rlock_cpu(self):
        self.read_locks+=1;self.events.append(("rlock",self.type,self.size));yield self.storage[:self.size]
    @contextmanager
    def rwlock_cpu(self):
        self.write_locks+=1;self.events.append(("rwlock",self.type,self.size));yield self.storage[:self.size]

class Pyramid:
    def __init__(self,events=None):
        self.values=[np.full((256,320),124,np.uint8),np.full((128,160),125,np.uint8)]
        self.events=[] if events is None else events
    @contextmanager
    def rlock_cpu(self):
        self.events.append(("pyramid_lock",));yield self.values

class VPI:
    def __init__(self):
        self.events=[];self.calls=[];self.allocations=[]
        self.Backend=SimpleNamespace(PVA="PVA");self.Type=SimpleNamespace(U8="U8")
        self.Array=SimpleNamespace(zeros=self.zeros);self.sentinel=object()
    def zeros(self,capacity,kind):
        self.events.append(("allocate",capacity,kind))
        value=Array(np.zeros(capacity,np.uint8),kind,events=self.events)
        self.allocations.append(value);return value
    def OpticalFlowPyrLK(self,*args,**kwargs):
        self.events.append(("constructor",len(self.calls)+1,tuple(sorted(kwargs))))
        self.calls.append((args,kwargs));return (args,kwargs)

def facade_fixture(arm):
    vpi=VPI();audit={};facade=m.OwnershipFacade(vpi,arm,old,audit)
    p=Array(np.array([[1.25,2.5],[3.,4.],[5.,6.],[7.,8.]],np.float32),"VEC2F",9,vpi.events)
    flags=Array(np.array([0,1,7,255],np.uint8),capacity=9,events=vpi.events)
    pyramid=Pyramid(vpi.events)
    return vpi,audit,facade,p,flags,pyramid

class ScopeAndInventory(unittest.TestCase):
    def test_stdlib_exact_6_cases_30_roles_and_old_inventory_identity(self):
        with patch.dict(sys.modules,{"numpy":None}):cases=m.inventory();spec=m.freeze_spec()
        self.assertEqual(len(cases),6);self.assertEqual(len(cases)*len(m.ROLES),30)
        self.assertEqual([x["id"] for x in cases],list(spec["protocol"]["recover_cases"])+list(spec["protocol"]["preserve_cases"])+["flat"])
        self.assertTrue(all(case in robust.inventory() for case in cases))
        self.assertEqual(spec["depth"],2);self.assertEqual(spec["protocol"]["preflight_pva_calls"],0)
        self.assertEqual(m.ROLES,(("plain","base"),("shared","base"),("shared","trace"),("copied","base"),("copied","trace")))
    def test_freeze_guard_exact_types_configuration_and_sources(self):
        m.validate_freeze(frozen())
        for key,value in (("depth",4),("depth",True),("cases",m.inventory()[:-1]),("roles",[]),("scope","camera")):
            bad=frozen();bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.validate_freeze(bad)
        bad=frozen();bad["gates"]["point_error_px"]=.5
        with self.assertRaises(ValueError):m.validate_freeze(bad)
        bad=frozen();bad["source_sha256"]["unapproved.py"]="a"*64
        with self.assertRaises(ValueError):m.validate_freeze(bad)
        bad=frozen();bad["source_sha256"]["batch_discovery_pair.py"]="a"*64
        with self.assertRaises(ValueError):m.validate_freeze(bad)
    def test_reference_pin_matches_immutable_local_worker(self):
        self.assertEqual(m.sha(ROOT/"scripts/probe_pva_robustness.py"),m.REFERENCE_SHA)
    def test_invalid_scope_and_role_before_reference_import(self):
        with patch.object(m,"load_reference") as loader:
            for case,arm,trace in (("camera","plain",False),("flat","plain",True),("flat","copied",1),("flat","other",False)):
                with self.assertRaises(ValueError):m.run("/tmp","/tmp/freeze.json","a"*64,case,arm,trace)
            with self.assertRaises(ValueError):m.run("/tmp","/tmp/freeze.json","a"*64,"flat","plain")
            loader.assert_not_called()
    def test_strict_json_no_nonfinite_duplicate_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"record.json"
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e309}'):
                path.write_text(text)
                with self.assertRaises(ValueError):m.read(path)
            path.write_text('{}');m.pinned(path,m.sha(path))
            link=Path(directory)/"link";link.symlink_to(path)
            with self.assertRaises(ValueError):m.read(link)

class StatusOwnership(unittest.TestCase):
    def test_both_arms_preserve_every_active_flag_and_capacity(self):
        schedules=[];locks=[]
        for arm in ("shared","copied"):
            vpi,audit,facade,p,flags,pyr=facade_fixture(arm)
            original_flags=flags.storage.copy();original_p=p.storage.copy()
            facade.OpticalFlowPyrLK(pyr,p,backend=vpi.Backend.PVA)
            self.assertEqual(vpi.calls[0][1],dict(backend="PVA"))
            facade.OpticalFlowPyrLK(pyr,p,kptstatus=flags,backend=vpi.Backend.PVA)
            self.assertIs(vpi.calls[1][0][0],pyr);self.assertIs(vpi.calls[1][0][1],p)
            chosen=flags if arm=="shared" else facade.clone
            self.assertIs(vpi.calls[1][1]["kptstatus"],chosen)
            self.assertEqual(facade.clone.capacity,9);self.assertEqual(facade.clone.size,4)
            self.assertNotEqual(facade.clone.id,flags.id)
            np.testing.assert_array_equal(facade.clone.storage[:4],[0,1,7,255])
            np.testing.assert_array_equal(flags.storage,original_flags);np.testing.assert_array_equal(p.storage,original_p)
            self.assertEqual(len(vpi.allocations),1);self.assertIs(facade.retained[0],facade.clone)
            self.assertIs(facade.sentinel,vpi.sentinel)
            facade.finish(True);facade.closed()
            self.assertTrue(audit["clone_retained_until_estimator_close"])
            self.assertFalse(audit["device_internal_state_equality_claimed"])
            self.assertFalse(audit["inactive_capacity_bytes_observed"])
            schedules.append(vpi.events)
            locks.append((flags.read_locks,flags.write_locks,facade.clone.read_locks,facade.clone.write_locks,p.read_locks,p.write_locks))
        self.assertEqual(schedules[0],schedules[1])
        self.assertEqual(locks[0],locks[1]);self.assertEqual(locks[0],(4,0,2,1,4,0))
    def test_actual_backward_mutation_does_not_write_unused_status(self):
        for arm in ("shared","copied"):
            vpi,audit,facade,p,flags,pyr=facade_fixture(arm)
            facade.OpticalFlowPyrLK(pyr,p,backend="PVA")
            facade.OpticalFlowPyrLK(pyr,p,kptstatus=flags,backend="PVA")
            chosen=vpi.calls[1][1]["kptstatus"];chosen.storage[0]=9
            facade.finish(True)
            used="source_status" if arm=="shared" else "clone_status"
            unused="clone_status" if arm=="shared" else "source_status"
            self.assertEqual(depth.decode(audit["after"][used]["array"])[0],9)
            self.assertEqual(depth.decode(audit["after"][unused]["array"])[0],0)
            self.assertEqual(audit["reverse"]["actual_argument_is_clone"],arm=="copied")
    def test_no_forward_initialization_or_extra_constructor(self):
        vpi,_,facade,p,flags,pyr=facade_fixture("copied")
        with self.assertRaisesRegex(ValueError,"Forward"):facade.OpticalFlowPyrLK(pyr,p,backend="PVA",kptstatus=flags)
        vpi,_,facade,p,flags,pyr=facade_fixture("shared")
        facade.OpticalFlowPyrLK(pyr,p,backend="PVA");facade.OpticalFlowPyrLK(pyr,p,backend="PVA",kptstatus=flags)
        with self.assertRaisesRegex(ValueError,"Unexpected"):facade.OpticalFlowPyrLK(pyr,p,backend="PVA",kptstatus=flags)
        with self.assertRaisesRegex(ValueError,"Plain"):m.OwnershipFacade(vpi,"plain",old,{})
    def test_malformed_status_or_clone_fails_before_reverse_constructor(self):
        for mutation in ("dtype","type","size","capacity","clone"):
            vpi,_,facade,p,flags,pyr=facade_fixture("copied")
            facade.OpticalFlowPyrLK(pyr,p,backend="PVA")
            if mutation=="dtype":flags.storage=flags.storage.astype(np.int16)
            elif mutation=="type":flags.type="S16"
            elif mutation=="size":flags._size=3
            elif mutation=="capacity":flags.capacity=3
            else:vpi.Array.zeros=lambda capacity,kind:Array(np.zeros(capacity,np.int16),kind)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):facade.OpticalFlowPyrLK(pyr,p,backend="PVA",kptstatus=flags)
            self.assertEqual(len(vpi.calls),1)
    def test_empty_feature_path_no_constructor_and_partial_call_fail(self):
        _,audit,facade,p,_,pyr=facade_fixture("shared")
        facade.finish(False);facade.closed();self.assertTrue(audit["expected_no_flow"])
        self.assertTrue(audit["clone_retained_until_estimator_close"])
        facade.OpticalFlowPyrLK(pyr,p,backend="PVA")
        with self.assertRaises(ValueError):facade.finish(False)
        with self.assertRaises(ValueError):facade.finish(True)
    def test_capture_distinguishes_local_from_actual_constructor_argument(self):
        class Base:
            def __init__(self):self.data={}
            def record(self,stage,state):self.data[stage]=dict(backward_constructor_input_status_was_forward_status=True)
        fake_depth=SimpleNamespace(capture_type=lambda old:Base)
        audit=dict(reverse=dict(actual_argument_is_forward_status=False,actual_argument_is_clone=True))
        capture=m.capture_type(fake_depth,None,audit)();capture.record("after_backward",{})
        row=capture.data["after_backward"]
        self.assertNotIn("backward_constructor_input_status_was_forward_status",row)
        self.assertTrue(row["estimator_local_backward_initial_status_is_forward_status"])
        self.assertFalse(row["actual_backward_argument_is_forward_status"]);self.assertTrue(row["actual_backward_argument_is_clone"])
        with self.assertRaises(ValueError):m.capture_type(fake_depth,None,{})().record("after_backward",{})

@dataclass
class Config:
    pyramid_levels:int=4
    pyramid_scale:float=.5
    feature_image_scale:float=.5
    flow_status_policy:str="legacy_default"
    forward_backward_check:bool=True

class Lifecycle(unittest.TestCase):
    def run_fixture(self,directory,arm="plain",estimate_error=None,close_error=None,wrong_fit=False):
        vpi=VPI();p=np.array([[200,200],[300,300]],np.float32);q=p+[2,-1]
        corr=SimpleNamespace(count=2,previous_points=p,current_points=q.astype(np.float32),harris_scores=np.ones(2,np.float32),
            forward_backward_error_px=np.zeros(2,np.float32),metrics={"accepted_count":2},backends={},
            full_image_size=(640,512),motion_image_size=(320,256),timings_ms={})
        model=SimpleNamespace(accepted=wrong_fit,inlier_mask=np.ones(2,bool),residuals_px=np.zeros(2,np.float32),
            to_dict=lambda:dict(timing_ms=1.,quality_status="accepted" if wrong_fit else "rejected",
                parameters=dict(translation_x_px=20.,translation_y_px=0.) if wrong_fit else None,metrics={}))
        estimator=SimpleNamespace(closed=False,failed=False,_vpi=vpi)
        def estimate(previous,current):
            if estimate_error:raise estimate_error
            points=Array(np.array([[100,100],[150,150]],np.float32),"VEC2F",4)
            status=Array(np.array([0,1],np.uint8),capacity=4)
            estimator._vpi.OpticalFlowPyrLK(Pyramid(),points,backend="PVA")
            estimator._vpi.OpticalFlowPyrLK(Pyramid(),points,kptstatus=status,backend="PVA")
            return corr
        estimator.estimate=Mock(side_effect=estimate)
        def close():
            if close_error:raise close_error
            if arm!="plain" and not estimate_error:
                self.assertIsInstance(estimator._vpi,m.OwnershipFacade)
                self.assertIs(estimator._vpi.retained[0],estimator._vpi.clone)
            estimator.closed=True
        estimator.close=Mock(side_effect=close)
        @contextmanager
        def adapter(reuse,config,audit):
            audit["effective_motion_configuration"]=asdict(config);yield lambda config:estimator
        @contextmanager
        def empty(selection,audit):yield
        selection=SimpleNamespace(configurations=lambda helper:(Config(),Config()),candidate_config=lambda config:(config,{}),
            transform_source=lambda source:source,candidate_adapter=adapter)
        phot=SimpleNamespace(allow_empty_diagnostic_result=empty,expected_unavailable=lambda exc,est,kind:str(exc)=="zero features")
        reuse=SimpleNamespace(generated_method=lambda:"source",_ESTIMATE=Mock(),pva=SimpleNamespace(PvaMotionError=RuntimeError))
        modules={"motion_reuse_v12":reuse,"profile_visible_interaction_v30":SimpleNamespace(runtime_info=lambda:{})}
        helper=SimpleNamespace(dependencies=lambda ref:(modules,{}),runtime_check=Mock(),clock_policy_snapshot=lambda:{"same":True})
        class Frame:
            def __init__(self,image,*args):self.image=image
            def pixel_sha256(self):return generator.array_sha(self.image)
        image=np.zeros(m.SHAPE,np.uint8);pixel_sha=generator.array_sha(image)
        pair=(image,image.copy(),dict(previous_pixel_sha256=pixel_sha,current_pixel_sha256=pixel_sha))
        fitfn=Mock(return_value=model);loaded=(depth,old,(phot,selection,helper,generator,{},{},{},{}))
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(m,"bundle",side_effect=lambda *args:frozen()))
                stack.enter_context(patch.object(m,"load_reference",return_value=(robust,loaded,{"source_sha256":{}})))
                stack.enter_context(patch.object(robust,"pair_for_case",return_value=pair))
                stack.enter_context(patch.object(old,"anchor_lines",return_value={}))
                stack.enter_context(patch.dict(sys.modules,{"cv2":SimpleNamespace(setNumThreads=Mock()),
                    "tiny_target.types":SimpleNamespace(Frame=Frame,TimestampSource=SimpleNamespace(CONTAINER_RATE="rate")),
                    "tiny_target.motion":SimpleNamespace(fit_global_motion=fitfn)}))
                value=m.run(directory,"unused","a"*64,"texture__shift_two",arm)
        finally:
            self.assertIs(estimator._vpi,vpi)
        return value,estimator,fitfn
    def test_scientific_failure_retained_plain_shared_canonical_exact(self):
        rows=[]
        for arm in ("plain","shared","copied"):
            with tempfile.TemporaryDirectory() as directory:
                row,estimator,fitfn=self.run_fixture(directory,arm)
                self.assertTrue(row["passed_integrity"]);self.assertEqual(row["focal_estimate_calls"],1)
                self.assertEqual(row["global_fits"],1);self.assertEqual(row["preflight_pva_calls"],0)
                self.assertFalse(row["canonical_nontiming"]["accepted_truth"]["global_guard_passed"])
                self.assertEqual(estimator.close.call_count,1);self.assertEqual(fitfn.call_count,1)
                self.assertEqual(row["canonical_nontiming_sha256"],m.canonical_sha(row["canonical_nontiming"]))
                rows.append(row)
        self.assertEqual(rows[0]["canonical_nontiming"],rows[1]["canonical_nontiming"])
        self.assertEqual(rows[1]["canonical_nontiming"],rows[2]["canonical_nontiming"])
    def test_wrong_fit_not_runtime_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            row,_,_=self.run_fixture(directory,wrong_fit=True)
            self.assertTrue(row["passed_integrity"])
            self.assertTrue(row["canonical_nontiming"]["accepted_truth"]["accepted_wrong_fit"])
    def test_no_features_is_unavailable_not_zero_truth_or_fabricated_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            row,estimator,fitfn=self.run_fixture(directory,"copied",estimate_error=RuntimeError("zero features"))
            self.assertTrue(row["passed_integrity"]);fitfn.assert_not_called()
            self.assertTrue(row["constructor_audit"]["expected_no_flow"])
            self.assertIsNone(row["canonical_nontiming"]["result"]["correspondence"])
            self.assertFalse(row["canonical_nontiming"]["accepted_truth"]["original_fit_ran"])
    def test_runtime_cleanup_faults_abort_and_retain_receipts(self):
        for arguments in (dict(estimate_error=RuntimeError("backend fault")),dict(close_error=RuntimeError("close fault"))):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):self.run_fixture(directory,"copied",**arguments)
                row=m.read(Path(directory)/"texture__shift_two_copied_base.json")
                self.assertFalse(row["passed_integrity"]);self.assertFalse(row["completed"])
                self.assertIsNotNone(row["error"])
    def test_no_overwrite_before_reference_import(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"bundle",return_value=frozen()),patch.object(m,"load_reference") as loader:
            path=Path(directory)/"flat_plain_base.json";path.write_text("preserved")
            with self.assertRaisesRegex(ValueError,"overwrite"):m.run(directory,"unused","a"*64,"flat","plain")
            self.assertEqual(path.read_text(),"preserved");loader.assert_not_called()

if __name__=="__main__":unittest.main()
