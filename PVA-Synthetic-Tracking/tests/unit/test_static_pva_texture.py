"""Generated/mocked static diagnostic tests; no VPI, camera media, or network."""
from __future__ import annotations

import base64
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

PATH = Path(__file__).resolve().parents[2]/"scripts/probe_static_pva_texture.py"
SPEC = importlib.util.spec_from_file_location("static_probe_test", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def freeze():
    return dict(schema="static_pva_texture.v1", cases=list(m.CASES), modes=list(m.MODES),
        shape_hw=list(m.SHAPE), photometric_workspace=str(m.PHOTOMETRIC_WORKSPACE),
        photometric_runner_sha256=m.PHOTOMETRIC_SHA, candidate=dict(m.CANDIDATE),
        no_preflight_pva_calls=True, execution=dict(m.EXECUTION),
        files={name: m.SAFETY_SHA if name == "batch_discovery_pair.py" else "a"*64 for name in m.FILES})


def decoded(record):
    raw = base64.b64decode(record["data_base64"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    return np.frombuffer(raw, dtype=record["dtype"]).reshape(record["shape"])


class Locked:
    def __init__(self, data, identity=7, mutate_after=False):
        self.data, self.id, self.mutate_after = data, identity, mutate_after
        self.locks, self.active = 0, False

    @contextmanager
    def rlock_cpu(self):
        self.active = True
        self.locks += 1
        try:
            yield self.data
        finally:
            self.active = False
            if self.mutate_after:
                for value in self.data if isinstance(self.data, list) else [self.data]:
                    value[...] = 255


def correspondence(count=2):
    return SimpleNamespace(count=count,
        previous_points=np.zeros((count, 2), np.float32), current_points=np.ones((count, 2), np.float32),
        harris_scores=np.arange(count, dtype=np.float32), forward_backward_error_px=np.zeros(count, np.float32),
        metrics={"accepted_count": count}, backends={"optical_flow": "PVA"},
        full_image_size=(640, 512), motion_image_size=(320, 256), timings_ms={"total": 23.})


def fit(count=2):
    return SimpleNamespace(accepted=False, inlier_mask=np.zeros(count, bool), residuals_px=np.zeros(count, np.float32),
        to_dict=lambda: dict(timing_ms=45., model="translation", metrics={"points": count}, accepted=False))


class Guards(unittest.TestCase):
    def test_fixed_freeze_and_changes(self):
        m.validate_freeze(freeze())
        for key, value in (("cases", ["texture"]), ("shape_hw", [480,640]),
                           ("no_preflight_pva_calls", False), ("candidate", {}), ("execution", {})):
            bad = freeze(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_freeze(bad)
        bad = freeze(); bad["files"]["unexpected.py"] = "a"*64
        with self.assertRaises(ValueError): m.validate_freeze(bad)
        bad = freeze(); bad["files"]["batch_discovery_pair.py"] = "a"*64
        with self.assertRaises(ValueError): m.validate_freeze(bad)

    def test_scope_denies_before_import(self):
        with patch.object(m, "load_runtime") as loader:
            for path in ("/tmp", "/", "/tmp/seaqr_static_pva_texture_20261001_bad"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    m.run(path, Path(path)/"freeze.json", "a"*64, "texture")
            loader.assert_not_called()

    def test_read_strict_and_pinning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metadata.json"
            for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e309}'):
                path.write_text(content)
                with self.assertRaises(ValueError): m.read(path)
            path.write_text('{"x":1}')
            self.assertEqual(m.read(path), {"x":1})
            m.pinned(path, m.sha(path))
            with self.assertRaises(ValueError): m.pinned(path, "0"*64)
            link = Path(directory)/"link"; link.symlink_to(path)
            with self.assertRaises(ValueError): m.read(link)

    def test_exclusive_output_before_runtime_and_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, "bundle_inputs", return_value={"files":{},"freeze_sha256":"a"*64}):
            path = Path(directory)/"texture_base.json"
            with patch.object(m, "load_runtime", side_effect=RuntimeError("sentinel import failure")) as loader:
                with self.assertRaisesRegex(RuntimeError, "sentinel"): m.run(directory, "unused", "a"*64, "texture")
                self.assertEqual(loader.call_count, 1)
            failed = m.read(path)
            self.assertFalse(failed["completed"]); self.assertFalse(failed["passed_integrity"])
            self.assertEqual(failed["focal_estimate_calls"], 0)
            original = path.read_bytes()
            with patch.object(m, "load_runtime") as loader:
                with self.assertRaisesRegex(ValueError, "overwrite"): m.run(directory, "unused", "a"*64, "texture")
                loader.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


class ArraysAndCapture(unittest.TestCase):
    def test_exact_bytes_and_nonfinite_payloads(self):
        values = np.array([0x80000000, 0x7fc12345, 0x7f800000, 0xff800000], np.uint32).view(np.float32)
        record = m.descriptor(values)
        self.assertEqual(decoded(record).tobytes(), values.tobytes())
        u32 = np.array([2**32-1, 2**24+1], np.uint32)
        np.testing.assert_array_equal(decoded(m.descriptor(u32)), u32)
        with self.assertRaises(ValueError): m.descriptor(np.array([object()], object))
        with self.assertRaises(ValueError): m.descriptor(np.zeros(512*640+1, np.uint8))

    def test_image_stats_signed_gradient(self):
        a = np.array([[255,0],[255,0]], np.uint8)
        value = m.image_record(a)
        self.assertEqual(value["gradients"]["x"]["mean_absolute"], 255.)
        self.assertEqual(value["gradients"]["y"]["rms"], 0.)
        self.assertEqual(value["unique_values"], 2)
        with self.assertRaises(ValueError): m.image_record(a.astype(np.int16))

    def test_lock_copies_before_release_and_pyramid_sizes(self):
        value = Locked(np.array([1,2], np.uint8), mutate_after=True)
        np.testing.assert_array_equal(m.copy_vpi(value), [1,2])
        self.assertFalse(value.active); self.assertEqual(value.locks,1)
        shapes = [(256,320),(128,160),(64,80),(32,40)]
        value = Locked([np.full(s, i, np.uint8) for i,s in enumerate(shapes)], mutate_after=True)
        rows = m.copy_pyramid(value)
        self.assertEqual([row["minimum"] for row in rows], [0,1,2,3])
        self.assertEqual(value.locks,1); self.assertFalse(value.active)
        with self.assertRaises(ValueError): m.copy_pyramid(Locked([np.zeros((3,3), np.uint8)]*4))
        with self.assertRaises(ValueError): m.copy_pyramid(Locked(tuple(np.zeros(s,np.uint8) for s in shapes)))

    def test_forward_snapshot_is_before_shared_mutation(self):
        cap = m.Capture()
        points = Locked(np.array([[1.,2.]],np.float32), 11)
        status = Locked(np.array([0],np.uint8), 12)
        state = dict(tracked_points=points, forward_status=status)
        cap.record("after_forward", state)
        points.data[:] = [5.,6.]; status.data[:] = 1
        state.update(backward_points_vpi=points, backward_status_vpi=status, backward_initial_status=status)
        cap.record("after_backward", state)
        np.testing.assert_array_equal(decoded(cap.data["after_forward"]["tracked_points"]), [[1,2]])
        self.assertEqual(decoded(cap.data["after_forward"]["forward_status"])[0], 0)
        self.assertEqual(decoded(cap.data["after_backward"]["forward_status"])[0], 1)
        self.assertTrue(cap.data["after_backward"]["backward_constructor_input_status_was_forward_status"])
        cap.stages=list(m.ANCHORS); cap.method_calls=1
        result=cap.summary()
        self.assertTrue(result["forward_status_bytes_changed_by_backward"])
        self.assertTrue(result["forward_point_bytes_changed_by_backward"])

    def test_initial_status_unavailable_and_filter_indices(self):
        cap=m.Capture()
        cap.record("before_forward", dict(selected_points=np.zeros((3,2),np.float32),
            selected_scores=np.ones(3,np.float32),selected_indices=np.array([9,3,5])))
        self.assertIsNone(cap.data["before_forward"]["initial_forward_status"])
        state={key:np.zeros((3,2),np.float32) for key in ("previous_full_points","current_full_points","backward_full_points")}
        state.update(accepted_mask=np.array([True,False,True]),fb_error=np.zeros(3),
            accepted_previous=np.zeros((2,2)),accepted_current=np.zeros((2,2)),accepted_scores=np.ones(2),
            accepted_fb_error=np.zeros(2),rejection_counts={"forward_status":1})
        cap.record("final_filter",state)
        np.testing.assert_array_equal(decoded(cap.data["final_filter"]["selected_indices_of_rejected_points"]), [1])
        np.testing.assert_array_equal(decoded(cap.data["final_filter"]["selected_indices_of_accepted_points"]), [0,2])

    def test_partial_capture_only_declared_unavailable_prefix(self):
        cap=m.Capture();cap.method_calls=1;cap.stages=["pyramids"]
        self.assertTrue(cap.summary(unavailable=True)["partial"])
        with self.assertRaises(ValueError):cap.summary()
        cap.error="failure"
        with self.assertRaises(ValueError):cap.summary(unavailable=True)


class ObserverAndFocalResult(unittest.TestCase):
    SOURCE="def _estimate_v12(value):\n    first = value + 1\n    return first\n"

    def observer_setup(self):
        namespace={};exec(compile(self.SOURCE,"generated-only-fixture","exec"),namespace)
        cap=m.Capture()
        def record(stage,state):
            cap.stages.append(stage);cap.data[stage]=state["first"]
        cap.record=record
        return namespace["_estimate_v12"],cap

    def test_source_identity_missing_duplicate_anchors(self):
        digest=hashlib.sha256(self.SOURCE.encode()).hexdigest()
        with patch.object(m,"METHOD_SHA",digest),patch.object(m,"ANCHORS",{"final":"    return first"}):
            self.assertEqual(m.anchor_lines(self.SOURCE),{3:"final"})
            with self.assertRaises(ValueError):m.anchor_lines(self.SOURCE+"\n")
        for anchor in ("missing", ""):
            source=self.SOURCE+"\n\n" if not anchor else self.SOURCE
            with patch.object(m,"METHOD_SHA",hashlib.sha256(source.encode()).hexdigest()),patch.object(m,"ANCHORS",{"final":anchor}):
                with self.assertRaises(ValueError):m.anchor_lines(source)

    def test_observer_one_unchanged_code_object_and_restore(self):
        method,cap=self.observer_setup();original=method.__code__
        with patch.object(m,"METHOD_SHA",hashlib.sha256(self.SOURCE.encode()).hexdigest()),patch.object(m,"ANCHORS",{"final":"    return first"}):
            with m.observe(method,self.SOURCE,cap):self.assertEqual(method(3),4)
        self.assertIs(method.__code__,original);self.assertIsNone(sys.gettrace())
        self.assertEqual(cap.method_calls,1);self.assertEqual(cap.data,{"final":4})

    def test_observer_error_restoration_and_multiple_calls(self):
        method,cap=self.observer_setup()
        with patch.object(m,"METHOD_SHA",hashlib.sha256(self.SOURCE.encode()).hexdigest()),patch.object(m,"ANCHORS",{"final":"    return first"}):
            with self.assertRaisesRegex(ValueError,"Multiple"),m.observe(method,self.SOURCE,cap):
                method(3);method(4)
            self.assertIsNone(sys.gettrace())
            method,cap=self.observer_setup();cap.record=Mock(side_effect=ValueError("capture failed"))
            with self.assertRaisesRegex(ValueError,"capture failed"),m.observe(method,self.SOURCE,cap):method(3)
            self.assertIn("capture failed",cap.error);self.assertIsNone(sys.gettrace())

    def test_existing_tracer_and_wrong_code_rejected(self):
        method,cap=self.observer_setup()
        with patch.object(m,"METHOD_SHA",hashlib.sha256(self.SOURCE.encode()).hexdigest()),patch.object(m,"ANCHORS",{"final":"    return first"}):
            with patch.object(m.sys,"gettrace",return_value=object()):
                with self.assertRaises(ValueError),m.observe(method,self.SOURCE,cap):pass
            with self.assertRaises(ValueError),m.observe(lambda value:value,self.SOURCE,cap):pass

    def test_exact_result_excludes_only_declared_timings(self):
        corr,model=correspondence(),fit()
        first,times=m.result_record(corr,model)
        corr.timings_ms={"total":999.}; model.to_dict=lambda:dict(timing_ms=100.,model="translation",metrics={"points":2},accepted=False)
        second,_=m.result_record(corr,model)
        self.assertEqual(m.canonical_sha(first),m.canonical_sha(second))
        self.assertEqual(times["fit_timing_ms"],45.)
        corr.current_points[0,0]=3
        third,_=m.result_record(corr,model)
        self.assertNotEqual(m.canonical_sha(first),m.canonical_sha(third))

    def test_one_estimate_and_original_fit_even_zero_accepted(self):
        corr=correspondence(0);est=SimpleNamespace(estimate=Mock(return_value=corr))
        fitting=Mock(return_value=fit(0));receipt=dict(focal_estimate_calls=0,global_fits=0)
        result,_=m.focal_result(est,None,None,fitting,{},SimpleNamespace(),RuntimeError,receipt)
        self.assertEqual(receipt["focal_estimate_calls"],1);self.assertEqual(receipt["global_fits"],1)
        est.estimate.assert_called_once_with(None,None);fitting.assert_called_once_with(corr,{})
        self.assertEqual(receipt["scientific_status"],"unavailable");self.assertFalse(receipt["scientific_fit_accepted"])
        self.assertEqual(result["correspondence"]["arrays"]["previous_points"]["shape"],[0,2])

    def test_expected_feature_unavailability_not_runtime_failure(self):
        est=SimpleNamespace(estimate=Mock(side_effect=RuntimeError("zero features")))
        phot=SimpleNamespace(expected_unavailable=lambda exc,est,kind:str(exc)=="zero features")
        receipt=dict(focal_estimate_calls=0,global_fits=0);fitting=Mock()
        result,_=m.focal_result(est,None,None,fitting,{},phot,RuntimeError,receipt)
        self.assertIsNone(result["correspondence"]);self.assertEqual(receipt["global_fits"],0);fitting.assert_not_called()
        est.estimate.side_effect=RuntimeError("PVA_ERROR_INVALID_ARGUMENT")
        with self.assertRaisesRegex(RuntimeError,"PVA_ERROR"):
            m.focal_result(est,None,None,fitting,{},phot,RuntimeError,receipt)

    def test_focal_pair_pins_same_static_pixels(self):
        a=np.full(m.SHAPE,127,np.uint8);digest=hashlib.sha256(a.tobytes()).hexdigest()
        selection=SimpleNamespace(generated_controls=lambda:iter([("low_contrast_static",a,a.copy(),[0,0])]))
        generator=SimpleNamespace(array_sha=lambda value:hashlib.sha256(value.tobytes()).hexdigest())
        prior={"controls":[dict(name="low_contrast_static",passed=True,closed=True,previous_pixel_sha256=digest,current_pixel_sha256=digest)]}
        p,q,meta=m.focal_pair("bridge",selection,generator,prior)
        self.assertEqual(meta["previous_pixel_sha256"],digest);np.testing.assert_array_equal(p,q)
        prior["controls"][0]["passed"]=False
        with self.assertRaises(ValueError):m.focal_pair("bridge",selection,generator,prior)
        with self.assertRaises(ValueError):m.focal_pair("unapproved",selection,generator,prior)


class MockedLifecycle(unittest.TestCase):
    def run_mock(self, directory, estimate_error=None, close_error=None):
        corr=correspondence(0); model=fit(0)
        estimator=SimpleNamespace(closed=False,failed=False,estimate=Mock(return_value=corr,side_effect=estimate_error))
        def close():
            if close_error:raise close_error
            estimator.closed=True
        estimator.close=Mock(side_effect=close)
        motion=SimpleNamespace(flow_status_policy="legacy_default",forward_backward_check=True)
        global_config=SimpleNamespace(minimum_points=30)
        @contextmanager
        def adapter(reuse,config,audit):
            audit["effective_motion_configuration"]={"unchanged":True}
            yield lambda config:estimator
        @contextmanager
        def empty(selection,audit):yield
        selection=SimpleNamespace(configurations=Mock(return_value=(motion,global_config)),
            validate_pyramid_dimensions=Mock(return_value={}),transform_source=lambda source:source,
            candidate_adapter=adapter)
        phot=SimpleNamespace(allow_empty_diagnostic_result=empty,expected_unavailable=lambda exc,est,kind:False)
        reuse=SimpleNamespace(generated_method=lambda:"source",_ESTIMATE=Mock(),pva=SimpleNamespace(PvaMotionError=RuntimeError))
        modules={"motion_reuse_v12":reuse,"profile_visible_interaction_v30":SimpleNamespace(runtime_info=lambda:{})}
        helper=SimpleNamespace(dependencies=Mock(return_value=(modules,{"pinned":True})),
            runtime_check=Mock(),clock_policy_snapshot=Mock(return_value={"fixed":True}))
        class Frame:
            def __init__(self,image,*args):self.image=image
            def pixel_sha256(self):return hashlib.sha256(self.image.tobytes()).hexdigest()
        fitting=Mock(return_value=model)
        loaded=(phot,selection,helper,None,{}, {},{}, {})
        mocks={"cv2":SimpleNamespace(setNumThreads=Mock()),
               "tiny_target.types":SimpleNamespace(Frame=Frame,TimestampSource=SimpleNamespace(CONTAINER_RATE="rate")),
               "tiny_target.motion":SimpleNamespace(fit_global_motion=fitting)}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,mocks))
            stack.enter_context(patch.object(m,"bundle_inputs",side_effect=lambda *args:{"files":{},"freeze_sha256":"a"*64}))
            stack.enter_context(patch.object(m,"load_runtime",return_value=loaded))
            stack.enter_context(patch.object(m,"focal_pair",return_value=(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8),{})))
            stack.enter_context(patch.object(m,"asdict",side_effect=lambda value:dict(vars(value))))
            stack.enter_context(patch.object(m,"anchor_lines",return_value={1:"pyramids"}))
            try:
                result=m.run(directory,"unused","a"*64,"texture")
            except Exception:
                result=m.read(Path(directory)/"texture_base.json")
                raise
        return result,estimator,fitting,helper

    def test_single_call_success_cleanup_and_no_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            result,estimator,fitting,helper=self.run_mock(directory)
            self.assertTrue(result["completed"]);self.assertTrue(result["passed_integrity"])
            self.assertEqual(result["focal_estimate_calls"],1);self.assertEqual(result["global_fits"],1)
            self.assertEqual(result["preflight_pva_calls"],0)
            self.assertEqual(estimator.estimate.call_count,1);self.assertEqual(estimator.close.call_count,1)
            self.assertEqual(fitting.call_count,1);self.assertEqual(helper.runtime_check.call_count,2)
            self.assertEqual(result["canonical_nontiming_sha256"],m.canonical_sha(result["canonical_nontiming"]))

    def test_estimate_failure_closed_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"backend sentinel"):
                self.run_mock(directory,estimate_error=RuntimeError("backend sentinel"))
            value=m.read(Path(directory)/"texture_base.json")
            self.assertFalse(value["passed_integrity"]);self.assertTrue(value["closed"])
            self.assertEqual(value["focal_estimate_calls"],1);self.assertEqual(value["global_fits"],0)

    def test_cleanup_failure_never_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"close sentinel"):
                self.run_mock(directory,close_error=RuntimeError("close sentinel"))
            value=m.read(Path(directory)/"texture_base.json")
            self.assertFalse(value["passed_integrity"]);self.assertFalse(value["completed"])
            self.assertIn("close sentinel",value["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
