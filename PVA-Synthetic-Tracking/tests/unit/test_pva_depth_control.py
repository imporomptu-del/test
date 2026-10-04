"""Pure/generated depth-intervention contracts. No VPI or camera input."""
from __future__ import annotations
import copy
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
m=module("depth_probe_test",ROOT/"scripts/probe_pva_depth_control.py")
old=m.import_pinned(ROOT/"scripts/probe_static_pva_texture.py",m.REFERENCE_PROBE_SHA,"depth_test_old")

@dataclass
class Config:
    pyramid_levels:int=4
    pyramid_scale:float=.5
    feature_image_scale:float=.5
    flow_status_policy:str="legacy_default"
    forward_backward_check:bool=True
    window_size:int=11
    exclusions:tuple=()

def frozen():
    value=m.freeze_fields();value["files"]={name:m.SAFETY_SHA if name=="batch_discovery_pair.py" else "a"*64 for name in m.FILES};return value

def row(depth=4,valid=0,accepted=0,good=False,error=0.):
    selected=np.array([[20,20],[30,30]],np.float32)
    arrays=dict(previous_points=old.descriptor(selected[:accepted]*2),
        current_points=old.descriptor(selected[:accepted]*2+np.array([error,0],np.float32)),
        harris_scores=old.descriptor(np.arange(accepted,dtype=np.float32)),
        forward_backward_error_px=old.descriptor(np.zeros(accepted,np.float32)))
    status=np.ones(2,np.uint8);status[:valid]=0
    level=lambda i:dict(array=old.descriptor(np.full((2,3),i,np.uint8)))
    data={"pyramids":{name:[level(i) for i in range(depth)] for name in ("previous","current")},
        "before_forward":dict(selected_points=old.descriptor(selected),selected_scores=old.descriptor(np.array([99,100],np.float32)),selected_indices=old.descriptor(np.array([4,9],np.int64))),
        "after_forward":dict(forward_status=old.descriptor(status))}
    data["pyramids"]["proxy"]={name:level(0) for name in ("previous","current")}
    return dict(canonical_nontiming=dict(input_pair={"same":True},frame_pixel_sha256={"same":True},
        global_configuration={"min":30},effective_motion_configuration=dict(pyramid_levels=depth,other="unchanged"),
        result=dict(correspondence=dict(arrays=arrays),fit=dict(quality_status="accepted" if good else "rejected",
            parameters=dict(translation_x_px=0.,translation_y_px=0.) if good else None))),capture=dict(data=data))

def inventory(depth,texture_valid=0,texture_accepted=0,texture_good=False,texture_error=0.):
    rows={}
    for mode in m.MODES:
        rows["bridge_"+mode]=row(depth,2,2,True)
        rows["texture_"+mode]=row(depth,texture_valid,texture_accepted,texture_good,texture_error)
    return rows

class Contracts(unittest.TestCase):
    def test_freeze_exact_scope(self):
        m.validate_freeze(frozen())
        for key,value in (("shape_hw",[480,640]),("cases",["bridge"]),("no_preflight_pva_calls",False),
            ("candidate",dict(m.CANDIDATE,pyramid_levels=3)),("reference_results",{}),("protocol",{})):
            f=frozen();f[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_freeze(f)
        f=frozen();f["files"]["extra.py"]="a"*64
        with self.assertRaises(ValueError):m.validate_freeze(f)
        f=frozen();f["execution"]["workers"]=True
        with self.assertRaises(ValueError):m.validate_freeze(f)

    def test_constants_and_pinned_import(self):
        self.assertEqual(len(m.REFERENCE_RESULTS),4)
        self.assertTrue(all(len(v)==64 for v in m.REFERENCE_RESULTS.values()))
        self.assertEqual(old.METHOD_SHA,"bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b")
        with self.assertRaises(ValueError):m.import_pinned(ROOT/"scripts/probe_static_pva_texture.py","0"*64,"bad")

    def test_depth_only_no_mutation(self):
        original=Config();changed=m.depth_config(original)
        self.assertEqual(original.pyramid_levels,4);self.assertEqual(changed.pyramid_levels,2)
        self.assertEqual({k for k in asdict(original) if asdict(original)[k]!=asdict(changed)[k]},{"pyramid_levels"})
        bad=asdict(changed);bad["window_size"]=7
        with self.assertRaises(ValueError):m.require_depth_difference(asdict(original),bad)
        with self.assertRaises(ValueError):m.depth_config(Config(pyramid_levels=3))
        with self.assertRaises(ValueError):m.depth_config(Config(flow_status_policy="explicit"))
        m.require_depth_difference(dict(pyramid_levels=4,regions=[]),dict(pyramid_levels=2,regions=()))

    def test_two_level_dimensions_not_four_level_receipt(self):
        value=m.dimensions(m.depth_config(Config()))
        self.assertEqual(value["level_size_wh"],[[320,256],[160,128]])
        self.assertEqual(value["pyramid_levels"],2);self.assertEqual(value["original_pyramid_levels"],4)
        with self.assertRaises(ValueError):m.dimensions(Config())

    def test_scope_and_no_overwrite(self):
        with self.assertRaises(ValueError):m.bundle("/tmp","/tmp/freeze.json","a"*64,old)
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"old_helper",return_value=old),patch.object(m,"bundle",return_value=frozen()):
            output=Path(directory)/"texture_base.json";output.write_text("preserve")
            with patch.object(m,"references") as refs:
                with self.assertRaisesRegex(ValueError,"overwrite"):m.run(directory,"unused","a"*64,"texture")
                refs.assert_not_called()
            self.assertEqual(output.read_text(),"preserve")

    def test_failure_retained_before_any_focal_call(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"old_helper",return_value=old),patch.object(m,"bundle",return_value=frozen()),patch.object(m,"references",side_effect=ValueError("reference changed")):
            with self.assertRaisesRegex(ValueError,"reference changed"):m.run(directory,"unused","a"*64,"texture")
            output=old.read(Path(directory)/"texture_base.json")
            self.assertFalse(output["passed_integrity"]);self.assertEqual(output["focal_estimate_calls"],0)
            self.assertEqual(output["preflight_pva_calls"],0)

class PrefixAndScience(unittest.TestCase):
    def test_prefix_seeds_and_two_directions_exact(self):
        four,two=row(4),row(2)
        self.assertTrue(m.reference_parity(two,four,True)["complete"])
        self.assertFalse(m.reference_parity(two,four,False)["complete"])
        for direction in ("previous","current"):
            bad=copy.deepcopy(two);bad["capture"]["data"]["pyramids"][direction][1]["array"]=old.descriptor(np.zeros((2,3),np.uint8))
            with self.assertRaisesRegex(ValueError,"prefix"):m.reference_parity(bad,four,True)
        for name in ("selected_points","selected_scores","selected_indices"):
            bad=copy.deepcopy(two);bad["capture"]["data"]["before_forward"][name]=old.descriptor(np.zeros(1,np.float32))
            with self.assertRaisesRegex(ValueError,"seed"):m.reference_parity(bad,four,True)

    def test_frame_global_and_config_changes_fail(self):
        for name in ("input_pair","frame_pixel_sha256","global_configuration"):
            bad=row(2);bad["canonical_nontiming"][name]={"changed":True}
            with self.assertRaises(ValueError):m.reference_parity(bad,row(4),True)
        bad=row(2);bad["canonical_nontiming"]["effective_motion_configuration"]["other"]="changed"
        with self.assertRaises(ValueError):m.reference_parity(bad,row(4),True)

    def test_partial_capture_never_complete(self):
        bad=row(2);del bad["capture"]["data"]["before_forward"]
        result=m.reference_parity(bad,row(4),True)
        self.assertFalse(result["complete"]);self.assertFalse(result["feature_and_prefix_equal"])

    def test_parity_required_before_observation(self):
        result=m.scientific_comparison({}, {},False)
        self.assertFalse(result["interpretable"])
        new=inventory(2);new["texture_base"]["canonical_nontiming"]["bad"]=True
        with self.assertRaisesRegex(ValueError,"parity"):m.scientific_comparison(new,inventory(4),True)

    def test_availability_not_global_recovery(self):
        result=m.scientific_comparison(inventory(2,1,1,False),inventory(4),True)
        self.assertTrue(result["coarse_level_hypothesis_availability_support"])
        self.assertFalse(result["stronger_original_global_recovery"])
        self.assertFalse(result["production_promotion"])

    def test_stronger_requires_original_fit_and_error_bound(self):
        for error,expected in ((0.,True),(.25,True),(.5,False)):
            result=m.scientific_comparison(inventory(2,2,2,True,error),inventory(4),True)
            self.assertEqual(result["stronger_original_global_recovery"],expected)
        result=m.scientific_comparison(inventory(2),inventory(4),True)
        self.assertFalse(result["texture_immediate_forward_availability_restored"])

    def test_bridge_required_and_prior_texture_zero(self):
        new=inventory(2,2,2,True)
        for mode in m.MODES:new["bridge_"+mode]=row(2,2,2,False)
        result=m.scientific_comparison(new,inventory(4),True)
        self.assertFalse(result["bridge_zero_motion_retained"]);self.assertFalse(result["coarse_level_hypothesis_availability_support"])
        with self.assertRaises(ValueError):m.scientific_comparison(inventory(2,2,2,True),inventory(4,1),True)

    def test_array_decode_exact_and_invalid(self):
        a=np.array([0x80000000,0x7fc12345],np.uint32).view(np.float32)
        self.assertEqual(m.decode(old.descriptor(a)).tobytes(),a.tobytes())
        bad=old.descriptor(a);bad["sha256"]="0"*64
        with self.assertRaises(ValueError):m.decode(bad)
        bad=old.descriptor(a);bad["shape"]=[True]
        with self.assertRaises(ValueError):m.decode(bad)

    def test_two_level_readonly_capture_copies_inside_lock(self):
        class Locked:
            def __init__(self,data):self.data=data
            @contextmanager
            def rlock_cpu(self):
                yield self.data
                for a in self.data if isinstance(self.data,list) else [self.data]:a[:]=255
        state=dict(pyramid_backend_name="PVA",rescale_backend="CUDA")
        for name in ("previous","current"):
            state[name+"_pyramid"]=Locked([np.zeros((256,320),np.uint8),np.ones((128,160),np.uint8)])
            state[name+"_motion"]=Locked(np.zeros((256,320),np.uint8))
        capture=m.capture_type(old)();capture.record("pyramids",state)
        self.assertEqual([x["minimum"] for x in capture.data["pyramids"]["previous"]],[0,1])
        self.assertEqual(capture.stages,["pyramids"])
        state["previous_pyramid"]=Locked([np.zeros((256,320),np.uint8)]*4)
        with self.assertRaises(ValueError):m.capture_type(old)().record("pyramids",state)

if __name__=="__main__":unittest.main()
