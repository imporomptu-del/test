"""Generated-only contracts and mocked lifecycle; never imports VPI or media."""
from __future__ import annotations
from contextlib import contextmanager,ExitStack
from dataclasses import dataclass,asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
m=imported("robustness_worker_test",ROOT/"scripts/probe_pva_robustness.py")
old=imported("robustness_old_probe_test",ROOT/"scripts/probe_static_pva_texture.py")
depth=imported("robustness_depth_test",ROOT/"scripts/probe_pva_depth_control.py")
generator=imported("robustness_generator_test",ROOT/"scripts/validate_motion_patch_controls.py")

def frozen():
    value=m.freeze_spec();value["source_sha256"]={name:m.SAFETY_SHA if name=="batch_discovery_pair.py" else "a"*64 for name in m.SOURCES};return value

def result(p,q,accepted=False,vector=None,ransac=True):
    p,q=np.asarray(p,np.float32).reshape(-1,2),np.asarray(q,np.float32).reshape(-1,2)
    parameters=None if vector is None else dict(translation_x_px=vector[0],translation_y_px=vector[1])
    return dict(correspondence=dict(arrays=dict(previous_points=old.descriptor(p),current_points=old.descriptor(q),
        harris_scores=old.descriptor(np.arange(len(p),dtype=np.float32)),forward_backward_error_px=old.descriptor(np.zeros(len(p),np.float32)))),
        fit=dict(quality_status="accepted" if accepted else "rejected",parameters=parameters,
            metrics=dict(ransac_samples_evaluated=len(p)) if ransac else {}),
        inlier_mask=old.descriptor(np.array([i%2==0 for i in range(len(p))],bool)),residuals_px=old.descriptor(np.zeros(len(p),np.float32)))

@dataclass
class Config:
    pyramid_levels:int=4
    pyramid_scale:float=.5
    feature_image_scale:float=.5
    flow_status_policy:str="legacy_default"
    forward_backward_check:bool=True


class InventoryAndGuards(unittest.TestCase):
    def test_exact_26_and_104_phases_stdlib_only(self):
        with patch.dict(sys.modules,{"numpy":None}):cases=m.inventory();spec=m.freeze_spec()
        self.assertEqual(len(cases),26);self.assertEqual(len({x["id"] for x in cases}),26)
        self.assertEqual({key:sum(x["scientific_class"]==key for x in cases) for key in ("global_positive","sparse_corner","degenerate")},
                         dict(global_positive=15,sparse_corner=8,degenerate=3))
        self.assertEqual(len(cases)*len(m.DEPTHS)*len(m.MODES),104)
        self.assertEqual(spec["protocol"]["preflight_pva_calls"],0)
        self.assertEqual([x["id"] for x in cases[:20]],[x["case_id"] for x in generator.control_inventory(m.SHAPE)["cases"]])
        shift=next(x for x in cases if x["id"]=="out_of_range_shift4")
        self.assertEqual(shift["scientific_class"],"global_positive");self.assertEqual(shift["inherited_ncc_expectation"],"abstain")

    def test_freeze_protocol_and_source_inventory_exact(self):
        m.validate_freeze(frozen())
        for key,value in (("depths",[2]),("modes",["trace"]),("cases",m.inventory()[:-1]),("scope","media")):
            bad=frozen();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate_freeze(bad)
        bad=frozen();bad["gates"]["point_error_px"]=.5
        with self.assertRaises(ValueError):m.validate_freeze(bad)
        bad=frozen();bad["source_sha256"]["other.py"]="a"*64
        with self.assertRaises(ValueError):m.validate_freeze(bad)

    def test_pinned_helper_identities(self):
        self.assertEqual(m.sha(ROOT/"scripts/probe_pva_depth_control.py"),m.DEPTH_HELPER_SHA)
        self.assertEqual(m.sha(ROOT/"scripts/probe_static_pva_texture.py"),m.STATIC_HELPER_SHA)
        self.assertEqual(m.sha(ROOT/"scripts/run_motion_photometric_controls.py"),m.PHOTOMETRIC_HELPER_SHA)
        self.assertEqual(m.sha(ROOT/"scripts/validate_motion_patch_controls.py"),m.GENERATOR_SHA)

    def test_scope_invalid_case_and_bool_depth_before_helper_import(self):
        with patch.object(m,"load_helpers") as loader:
            for case,d in (("private-camera",2),("texture__static",True),("texture__static",3)):
                with self.assertRaises(ValueError):m.run("/tmp","/tmp/freeze.json","a"*64,case,d)
            with self.assertRaises(ValueError):m.run("/tmp","/tmp/freeze.json","a"*64,"texture__static",2)
            loader.assert_not_called()

    def test_strict_json_and_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"x.json"
            for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e309}'):
                path.write_text(text)
                with self.assertRaises(ValueError):m.read(path)
            path.write_text('{}');m.pinned(path,m.sha(path))
            link=Path(directory)/"link";link.symlink_to(path)
            with self.assertRaises(ValueError):m.pinned(link,m.sha(path))

    def test_depth_only_change(self):
        original=Config()
        self.assertIs(m.configuration_for_depth(original,4,depth),original)
        changed=m.configuration_for_depth(original,2,depth)
        self.assertEqual({k for k in asdict(original) if asdict(original)[k]!=asdict(changed)[k]},{"pyramid_levels"})
        self.assertEqual(original.pyramid_levels,4)
        with self.assertRaises(ValueError):m.configuration_for_depth(original,True,depth)
        with self.assertRaises(ValueError):m.configuration_for_depth(Config(flow_status_policy="explicit"),4,depth)

class PixelsAndTruth(unittest.TestCase):
    def test_bridge_once_rint_no_clipping_no_mutation(self):
        source=np.tile(np.arange(120,136,dtype=np.uint8),(512,40));before=source.copy()
        for gain,offset in ((1.,0.),(.8,12.),(1.2,-12.)):
            image,stats=m.bridge_photometry(source,gain,offset)
            np.testing.assert_array_equal(image,np.rint(source.astype(np.float64)*gain+offset).astype(np.uint8))
            self.assertEqual(stats["clipped_pixels"],0);self.assertTrue(stats["applied_once_to_original_current_u8"])
        np.testing.assert_array_equal(source,before)
        with self.assertRaisesRegex(ValueError,"clip"):m.bridge_photometry(np.zeros(m.SHAPE,np.uint8),1.2,-12.)
        with self.assertRaises(ValueError):m.bridge_photometry(source,.9,0.)

    def test_bridge_pair_previous_fixed_and_original_hash_guard(self):
        image=np.full(m.SHAPE,125,np.uint8);current=image.copy();current[:,0]=128
        selection=SimpleNamespace(generated_controls=lambda:[("low_contrast_translated",image,current,(4,-2))])
        prior=dict(controls=[dict(name="low_contrast_translated",passed=True,closed=True,
            previous_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),current_pixel_sha256=hashlib.sha256(current.tobytes()).hexdigest())])
        case=next(x for x in m.inventory() if x["id"]=="bridge__shift_4_m2_gain08_offset12")
        p,q,info=m.pair_for_case(case,selection,generator,prior)
        np.testing.assert_array_equal(p,image);np.testing.assert_array_equal(q,np.rint(current.astype(float)*.8+12).astype(np.uint8))
        self.assertEqual(info["original_current_pixel_sha256"],generator.array_sha(current))
        prior["controls"][0]["current_pixel_sha256"]="0"*64
        with self.assertRaises(ValueError):m.pair_for_case(case,selection,generator,prior)

    def test_analytic_original_pair_bytes_unchanged(self):
        pair=next(generator.generated_controls(m.SHAPE));case=m.inventory()[0]
        prior=dict(cases=[dict(case=pair["case"],previous_pixel_sha256=pair["previous_pixel_sha256"],current_pixel_sha256=pair["current_pixel_sha256"])])
        p,q,info=m.pair_for_case(case,None,generator,prior)
        np.testing.assert_array_equal(p,pair["previous_gray"]);np.testing.assert_array_equal(q,pair["current_gray"])
        self.assertEqual(info["current_pixel_sha256"],pair["current_pixel_sha256"])

    def test_fixed_interior_known_truth_not_measured_q(self):
        points=np.array([[128,128],[511,383],[512,128],[127.9,200]],float)
        np.testing.assert_array_equal(m.support_mask(points,[0,0]),[True,True,False,False])
        np.testing.assert_array_equal(m.support_mask(points,[1,0]),[True,False,False,False])

    def test_missing_empty_not_zero_error_or_rejected_fit(self):
        case=m.inventory()[0]
        value=m.accepted_truth(dict(correspondence=None,fit=None,reason="zero features"),case,depth.decode)
        self.assertFalse(value["original_fit_ran"]);self.assertIsNone(value["original_fit_accepted"])
        self.assertIsNone(value["truth_interior"]);self.assertFalse(value["point_guard_passed"])
        value=m.accepted_truth(result([],[],False,None,False),case,depth.decode)
        self.assertTrue(value["original_fit_ran"]);self.assertEqual(value["accepted_count"],0)
        self.assertIsNone(value["truth_interior"]["maximum_px"]);self.assertFalse(value["point_guard_passed"])

    def test_point_guard_includes_ransac_outliers(self):
        p=np.array([[200,200],[300,300]],np.float32);q=p.copy();q[1,0]+=2
        value=m.accepted_truth(result(p,q,True,[0,0]),m.inventory()[0],depth.decode)
        self.assertEqual(value["original_inlier_interior"]["maximum_px"],0.)
        self.assertEqual(value["original_outlier_interior"]["maximum_px"],2.)
        self.assertFalse(value["point_guard_passed"]);self.assertTrue(value["global_guard_passed"])

    def test_rejected_finite_model_error_descriptive_only(self):
        p=np.array([[200,200]],np.float32)
        value=m.accepted_truth(result(p,p,False,[2,0]),m.inventory()[0],depth.decode)
        self.assertEqual(value["candidate_translation_error_px"],2.)
        self.assertIsNone(value["accepted_fit_error_px"]);self.assertFalse(value["global_guard_passed"])
        value=m.accepted_truth(result(p,p,True,[2,0]),m.inventory()[0],depth.decode)
        self.assertTrue(value["accepted_wrong_fit"]);self.assertFalse(value["global_guard_passed"])

    def test_original_ransac_not_run_mask_not_outlier_labels(self):
        p=np.array([[200,200]],np.float32)
        value=m.accepted_truth(result(p,p,False,None,False),m.inventory()[0],depth.decode)
        self.assertFalse(value["original_ransac_executed"]);self.assertIsNone(value["original_outlier_interior"])

    def test_threshold_inclusive_and_nonfinite_rejected(self):
        p=np.array([[200,200]],np.float32)
        value=m.accepted_truth(result(p,p+[.25,0],True,[.25,0]),m.inventory()[0],depth.decode)
        self.assertTrue(value["point_guard_passed"]);self.assertTrue(value["global_guard_passed"])
        with self.assertRaises(ValueError):m.error_statistics([float("nan")])
        with self.assertRaises(ValueError):m.error_statistics([-1])

    def test_selected_truth_lost_unchanged_is_missing_not_zero(self):
        p=np.array([[200,200],[250,250],[300,300],[400,300]],np.float32);motion=(p+.5)/2-.5
        q=motion.copy();q[2]=np.nan;q[3]+=[1,0]
        mask=np.array([True,False,False,False])
        capture=dict(data=dict(before_forward=dict(selected_points=old.descriptor(motion)),
            after_forward=dict(tracked_points=old.descriptor(q),forward_status=old.descriptor(np.array([0,1,0,0],np.uint8))),
            final_filter=dict(previous_full_points=old.descriptor(p),accepted_mask=old.descriptor(mask),
                selected_indices_of_accepted_points=old.descriptor(np.array([0],np.int64)),rejection_counts={"lost":3})))
        value=m.trace_truth(capture,m.inventory()[0],depth.decode,old.descriptor)
        self.assertEqual(value["selected_count"],4);self.assertEqual(value["fixed_support_count"],4)
        self.assertEqual(value["immediate_forward_status_valid_count"],3)
        self.assertEqual(value["immediate_forward_nonfinite_status_valid_count"],1)
        self.assertEqual(value["fixed_support_immediate_forward_missing_count"],2)
        self.assertEqual(value["immediate_forward_fixed_support"]["maximum_px"],2.)
        errors=depth.decode(value["immediate_forward_error_px_with_nan_for_missing"])
        self.assertTrue(np.isnan(errors[1:3]).all());self.assertEqual(errors[0],0.)
        self.assertEqual(value["fixed_support_final_lost_count"],3)
        capture["data"]["final_filter"]["selected_indices_of_accepted_points"]=old.descriptor(np.array([1]))
        with self.assertRaises(ValueError):m.trace_truth(capture,m.inventory()[0],depth.decode,old.descriptor)

    def test_partial_trace_explicitly_unavailable(self):
        value=m.trace_truth(dict(data={"pyramids":{}}),m.inventory()[0],depth.decode,old.descriptor)
        self.assertFalse(value["available"]);self.assertIsNone(value["selected_count"])

class Lifecycle(unittest.TestCase):
    def run_fixture(self,directory,fit_accepted=False,fit_vector=(0.,0.),estimate_error=None,close_error=None):
        p=np.array([[200,200],[300,300]],np.float32)
        corr=SimpleNamespace(count=2,previous_points=p,current_points=p.copy(),harris_scores=np.ones(2,np.float32),
            forward_backward_error_px=np.zeros(2,np.float32),metrics={"accepted_count":2},backends={},
            full_image_size=(640,512),motion_image_size=(320,256),timings_ms={})
        model=SimpleNamespace(accepted=fit_accepted,inlier_mask=np.ones(2,bool),residuals_px=np.zeros(2,np.float32),
            to_dict=lambda:dict(timing_ms=1.,quality_status="accepted" if fit_accepted else "rejected",
                parameters=dict(translation_x_px=fit_vector[0],translation_y_px=fit_vector[1]),metrics={"ransac_samples_evaluated":2}))
        estimator=SimpleNamespace(closed=False,failed=False,estimate=Mock(return_value=corr,side_effect=estimate_error))
        def close():
            if close_error:raise close_error
            estimator.closed=True
        estimator.close=Mock(side_effect=close)
        @contextmanager
        def adapter(reuse,config,audit):
            audit["effective_motion_configuration"]=asdict(config)
            yield lambda config:estimator
        @contextmanager
        def empty(selection,audit):yield
        selection=SimpleNamespace(configurations=lambda helper:(Config(),Config()),candidate_config=lambda config:(config,{}),
            transform_source=lambda source:source,candidate_adapter=adapter)
        phot=SimpleNamespace(GENERATOR_SHA=m.GENERATOR_SHA,allow_empty_diagnostic_result=empty,
                             expected_unavailable=lambda exc,est,kind:str(exc)=="zero features")
        reuse=SimpleNamespace(generated_method=lambda:"source",_ESTIMATE=Mock(),pva=SimpleNamespace(PvaMotionError=RuntimeError))
        modules={"motion_reuse_v12":reuse,"profile_visible_interaction_v30":SimpleNamespace(runtime_info=lambda:{})}
        helper=SimpleNamespace(dependencies=lambda ref:(modules,{"same":True}),runtime_check=Mock(),clock_policy_snapshot=lambda:{"same":True})
        class Frame:
            def __init__(self,image,*args):self.image=image
            def pixel_sha256(self):return generator.array_sha(self.image)
        image=np.zeros(m.SHAPE,np.uint8);pixel_sha=generator.array_sha(image)
        pair=(image,image.copy(),dict(previous_pixel_sha256=pixel_sha,current_pixel_sha256=pixel_sha))
        fitfn=Mock(return_value=model)
        loaded=(depth,old,(phot,selection,helper,generator,{}, {},{},{}))
        with ExitStack() as stack:
            stack.enter_context(patch.object(m,"bundle",side_effect=lambda *args:frozen()))
            stack.enter_context(patch.object(m,"load_helpers",return_value=loaded))
            stack.enter_context(patch.object(m,"pair_for_case",return_value=pair))
            stack.enter_context(patch.object(old,"anchor_lines",return_value={}))
            stack.enter_context(patch.dict(sys.modules,{"cv2":SimpleNamespace(setNumThreads=Mock()),
                "tiny_target.types":SimpleNamespace(Frame=Frame,TimestampSource=SimpleNamespace(CONTAINER_RATE="rate")),
                "tiny_target.motion":SimpleNamespace(fit_global_motion=fitfn)}))
            value=m.run(directory,"unused","a"*64,"texture__static",2)
        return value,estimator,fitfn

    def test_scientific_rejected_fit_does_not_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            value,estimator,fitfn=self.run_fixture(directory)
            self.assertTrue(value["passed_integrity"]);self.assertFalse(value["canonical_nontiming"]["accepted_truth"]["global_guard_passed"])
            self.assertEqual(estimator.estimate.call_count,1);self.assertEqual(fitfn.call_count,1)
            self.assertEqual(estimator.close.call_count,1);self.assertEqual(value["preflight_pva_calls"],0)
            self.assertEqual(value["canonical_nontiming_sha256"],m.canonical_sha(value["canonical_nontiming"]))

    def test_scientific_wrong_accepted_fit_does_not_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            value,_,_=self.run_fixture(directory,fit_accepted=True,fit_vector=(2.,0.))
            self.assertTrue(value["passed_integrity"]);self.assertTrue(value["canonical_nontiming"]["accepted_truth"]["accepted_wrong_fit"])

    def test_expected_feature_unavailability_retained_without_fabricated_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            value,estimator,fitfn=self.run_fixture(directory,estimate_error=RuntimeError("zero features"))
            self.assertTrue(value["passed_integrity"]);fitfn.assert_not_called()
            self.assertFalse(value["canonical_nontiming"]["accepted_truth"]["original_fit_ran"])
            self.assertIsNone(value["canonical_nontiming"]["result"]["correspondence"])

    def test_runtime_and_cleanup_failures_retained(self):
        for args in (dict(estimate_error=RuntimeError("backend fault")),dict(close_error=RuntimeError("cleanup fault"))):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):self.run_fixture(directory,**args)
                value=m.read(Path(directory)/"texture__static_depth2_base.json")
                self.assertFalse(value["passed_integrity"]);self.assertFalse(value["completed"])

    def test_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"bundle",return_value=frozen()),patch.object(m,"load_helpers") as loader:
            path=Path(directory)/"texture__static_depth2_base.json";path.write_text("preserve")
            with self.assertRaisesRegex(ValueError,"overwrite"):m.run(directory,"unused","a"*64,"texture__static",2)
            self.assertEqual(path.read_text(),"preserve");loader.assert_not_called()


class OptionalSavedGeneratedCaptureRegression(unittest.TestCase):
    def test_prior_depth_capture_coordinate_and_dtype_contracts_if_available(self):
        # Optional immutable GENERATED capture fixtures, never source media.
        # The portable synthetic tests above do not depend on these user files.
        fixtures=(
            ("seaqr_static_pva_texture_20261001","bridge","96442e51ef54c6732fdd9bb47cc7f8b9b64e3b661f4c39c546ef125fd22e9dba"),
            ("seaqr_static_pva_texture_20261001","texture","4e6488a6c86552547f9f908aac5d9c16982e74a353a7f52cc6513b0ff1584d3b"),
            ("seaqr_pva_depth_control_20261001","bridge","3fb0b2419d40c948033ee667d99762ae0c838c6f95e74c530eee13d82d1cc3fa"),
            ("seaqr_pva_depth_control_20261001","texture","f21a10b75327ce86b63d492ba3bd3df745e40657d3d094d40f0933b54116d7b9"))
        found=0
        for folder,family,digest in fixtures:
            path=ROOT.parent/"outputs"/folder/"jetson"/(family+"_trace.json")
            if not path.exists():continue
            found+=1
            with self.subTest(fixture=folder+"/"+family):
                m.pinned(path,digest);row=m.read(path)
                self.assertTrue(row["completed"]);self.assertTrue(row["passed_integrity"])
                case=next(x for x in m.inventory() if x["id"]==family+"__static")
                measured=m.trace_truth(row["capture"],case,depth.decode,old.descriptor)
                result=row["canonical_nontiming"]["result"]
                accepted=m.accepted_truth(result,case,depth.decode)
                metrics=result["correspondence"]["metrics"]
                self.assertTrue(measured["available"])
                self.assertEqual(measured["selected_count"],metrics["selected_count"])
                self.assertEqual(measured["final_accepted_count"],metrics["accepted_count"])
                self.assertEqual(measured["final_accepted_count"],accepted["accepted_count"])
                self.assertEqual(measured["fixed_support_final_accepted_count"],accepted["truth_interior_count"])
                self.assertEqual(measured["immediate_forward_status_valid_count"]+measured["immediate_forward_lost_count"],metrics["selected_count"])
                self.assertEqual(measured["fixed_support_immediate_forward_measurable_count"]+
                    measured["fixed_support_immediate_forward_missing_count"],measured["fixed_support_count"])
                errors=depth.decode(measured["immediate_forward_error_px_with_nan_for_missing"])
                self.assertEqual(np.isfinite(errors).sum(),measured["immediate_forward_measurable_count"])
        if not found:self.skipTest("Optional hash-pinned prior generated PVA captures not available")

if __name__=="__main__":unittest.main()
