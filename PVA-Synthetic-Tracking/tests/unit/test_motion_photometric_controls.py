"""Generated-only PVA-wrapper tests; never load VPI, camera media or remote code."""
import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("run_motion_photometric_controls")
G = load("validate_motion_patch_controls")


def freeze():
    return dict(schema="motion_photometric_controls.v1", files={name:
        M.GENERATOR_SHA if name == "validate_motion_patch_controls.py" else
        M.SAFETY_SHA if name == "batch_discovery_pair.py" else "a"*64 for name in M.FILES},
        generated_only=True, candidate=copy.deepcopy(M.CANDIDATE), original_workspace=str(M.ORIGINAL_WORKSPACE),
        original_freeze_sha256=M.ORIGINAL_FREEZE_SHA, source_shape_hw=[512,640], case_count=20,
        execution=copy.deepcopy(M.EXECUTION))


class FakePvaError(RuntimeError):
    pass


class FakeFrame:
    def __init__(self, image, *_):
        self.image = image
    def pixel_sha256(self):
        return G.array_sha(self.image)


class FakeFit:
    def __init__(self, count, vector=(0.,0.), accepted=True):
        self.inlier_mask = np.ones(count, bool)
        self.residuals_px = np.zeros(count)
        self.accepted = accepted
        self.parameters = dict(translation_x_px=vector[0], translation_y_px=vector[1]) if accepted else None
    def to_dict(self):
        return dict(quality_status="accepted" if self.accepted else "rejected", parameters=self.parameters,
                    rejection_reasons=[] if self.accepted else ["insufficient_correspondences"])


def fake_candidate(error_index=None, cleanup_error_index=None, empty_index=None):
    inventory = G.control_inventory()["cases"]
    instances = []
    class Candidate:
        def __init__(self, config):
            self.index = len(instances)
            self.closed = False
            self.failed = False
            instances.append(self)
        def estimate(self, pframe, qframe):
            if self.index == error_index:
                self.failed = True
                raise FakePvaError("device runtime fault")
            case = inventory[self.index]
            if case["family"] == "flat":
                raise FakePvaError("PVA Harris returned zero features; motion is unavailable")
            count = 1 if case["family"] == "corner" else 35
            if self.index == empty_index:
                count = 0
            p = np.array([[200 + i, 210] for i in range(count)], np.float32).reshape(-1,2)
            return SimpleNamespace(previous_points=p, current_points=p+np.array(case["truth_displacement_xy"],np.float32),
                harris_scores=np.ones(count,np.float32), forward_backward_error_px=np.zeros(count,np.float32),
                metrics=dict(accepted_count=count), backends={}, full_image_size=(640,512), motion_image_size=(320,256))
        def close(self):
            if self.index == cleanup_error_index:
                raise RuntimeError("cleanup fault")
            self.closed = True
    return Candidate, instances


def fake_fit(corr, config):
    vector = (corr.current_points-corr.previous_points).mean(axis=0) if len(corr.previous_points) else np.zeros(2)
    return FakeFit(len(corr.previous_points), [float(v) for v in vector], len(corr.previous_points) >= 30)


class PhotometricControlsTests(unittest.TestCase):
    def test_exact_freeze_and_pins(self):
        M.validate_freeze(freeze())
        for mutator in (
            lambda value: value.update(source_shape_hw=[256,320]),
            lambda value: value.update(case_count=True),
            lambda value: value.update(real_media="anything"),
            lambda value: value["files"].update({"camera.avi":"a"*64}),
            lambda value: value["files"].update({"validate_motion_patch_controls.py":"a"*64}),
            lambda value: value["candidate"].update(max_features=500),
            lambda value: value["execution"].update(workers=2),
        ):
            data=freeze(); mutator(data)
            with self.assertRaises(ValueError):
                M.validate_freeze(data)

    def test_strict_json_no_duplicate_nonfinite_or_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"metadata.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e9999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    M.read(path)

    def test_array_bytes_keep_nonfinite_fit_residuals(self):
        values=np.array([0., np.nan, np.inf],np.float64)
        record=M.descriptor(values)
        raw=base64.b64decode(record["data_base64"])
        self.assertEqual(raw,values.tobytes())
        self.assertEqual(record["sha256"],hashlib.sha256(raw).hexdigest())
        json.dumps(record,allow_nan=False)

    def test_truth_interior_does_not_hide_measured_endpoint_errors(self):
        p=np.array([[200.,200.],[127.,200.],[511.,200.],[500.,200.],[200.,383.]])
        q=p+np.array([2.,0.]); q[0]=[20.,200.]
        corr=SimpleNamespace(previous_points=p,current_points=q)
        result=M.truth_metrics(corr,FakeFit(5,(2.,0.)),[2.,0.])
        self.assertEqual(result["interior_indices"],[0,3,4])
        self.assertEqual(result["accepted_interior_count"],3)
        self.assertEqual(result["all_accepted_interior"]["maximum_px"],182.)
        self.assertEqual(result["all_accepted_interior"]["above_0_25px"],1)
        self.assertFalse(result["selection_uses_measured_endpoint"])

    def test_unavailable_rejected_fit_and_empty_truth_are_not_zero_error(self):
        corr=SimpleNamespace(previous_points=np.array([[20.,20.]]),current_points=np.array([[20.,20.]]))
        result=M.truth_metrics(corr,FakeFit(1,accepted=False),[0.,0.])
        self.assertEqual(result["accepted_interior_count"],0)
        self.assertIsNone(result["all_accepted_interior"]["median_px"])
        self.assertIsNone(result["original_fit_translation_error_px"])
        self.assertFalse(result["original_fit_accepted"])

    def test_nonfinite_correspondence_is_rejected(self):
        corr=SimpleNamespace(previous_points=np.array([[200.,200.]]),current_points=np.array([[np.nan,200.]]))
        with self.assertRaises(ValueError):
            M.truth_metrics(corr,FakeFit(1),[0.,0.])

    def test_expected_availability_requires_exact_message_and_healthy_estimator(self):
        exc=FakePvaError("PVA Harris returned zero features; motion is unavailable")
        self.assertTrue(M.expected_unavailable(exc,SimpleNamespace(failed=False),FakePvaError))
        self.assertFalse(M.expected_unavailable(exc,SimpleNamespace(failed=True),FakePvaError))
        self.assertFalse(M.expected_unavailable(exc,None,FakePvaError))
        self.assertFalse(M.expected_unavailable(RuntimeError(str(exc)),SimpleNamespace(failed=False),FakePvaError))
        self.assertFalse(M.expected_unavailable(FakePvaError("other zero features"),SimpleNamespace(failed=False),FakePvaError))

    def run_fake(self, candidate, rows):
        with redirect_stdout(io.StringIO()):
            M.run_cases(candidate,None,None,G,rows,FakeFrame,"generated",FakePvaError,fake_fit)

    def test_twenty_fresh_estimators_and_low_feature_global_rejections_retained(self):
        candidate,instances=fake_candidate(); rows=[]
        self.run_fake(candidate,rows)
        self.assertEqual(len(rows),20)
        self.assertEqual(len(instances),20)
        self.assertTrue(all(item.closed for item in instances))
        self.assertEqual([row["case"] for row in rows],G.control_inventory()["cases"])
        self.assertEqual(sum(row["status"]=="unavailable" for row in rows),1)
        self.assertEqual(sum(row["status"]=="measured" for row in rows),19)
        for row in rows[8:16]:
            self.assertEqual(row["status"],"measured")
            self.assertFalse(row["truth"]["original_fit_accepted"])
            self.assertEqual(row["truth"]["all_accepted_interior"]["maximum_px"],0.)
        summary=M.summarize_cases(rows)
        self.assertEqual(len(summary["photometric_counterparts"]),8)
        self.assertEqual(summary["degenerate_case_count"],3)
        self.assertEqual(summary["degenerate_global_fit_accepted_case_ids"],["straight_edge","periodic_ambiguity"])
        self.assertEqual(summary["accepted_global_fit_error_above_0_25px_case_ids"],[])
        json.dumps(rows,allow_nan=False)

    def test_runtime_error_closes_and_aborts_preserving_unrun_rows(self):
        candidate,instances=fake_candidate(error_index=2); rows=[]
        with self.assertRaisesRegex(ValueError,"remaining cases not run"):
            self.run_fake(candidate,rows)
        self.assertEqual(len(rows),20)
        self.assertEqual(len(instances),3)
        self.assertTrue(all(item.closed for item in instances))
        self.assertEqual(rows[2]["status"],"runtime_error")
        self.assertEqual(sum(row["status"]=="not_run" for row in rows),17)

    def test_cleanup_failure_aborts_without_next_estimator(self):
        candidate,instances=fake_candidate(cleanup_error_index=1); rows=[]
        with self.assertRaisesRegex(ValueError,"remaining cases not run"):
            self.run_fake(candidate,rows)
        self.assertEqual(len(instances),2)
        self.assertEqual(rows[1]["error_category"],"cleanup_error")
        self.assertEqual(sum(row["status"]=="not_run" for row in rows),18)

    def test_zero_accepted_result_is_retained_unavailable_not_runtime_failure(self):
        candidate,instances=fake_candidate(empty_index=1); rows=[]
        self.run_fake(candidate,rows)
        self.assertEqual(len(instances),20)
        row=rows[1]
        self.assertEqual(row["status"],"unavailable")
        self.assertEqual(row["unavailable_reason"],"zero accepted LK correspondences")
        self.assertEqual(row["truth"]["accepted_count"],0)
        self.assertIsNone(row["truth"]["all_accepted_interior"]["maximum_px"])
        self.assertIsNone(row["runtime_error"])
        self.assertFalse(row["truth"]["original_fit_accepted"])

    def empty_verifier_fixture(self):
        calls=[]
        def original(correspondence,shape):
            calls.append(correspondence.count)
        selection=SimpleNamespace(verify_correspondence=original,EXPECTED_BACKENDS={"flow":"PVA"},
                                  capacity_for_shape=lambda shape: 120)
        corr=SimpleNamespace(count=0,backends={"flow":"PVA"},full_image_size=(640,512),motion_image_size=(320,256),
            previous_points=np.zeros((0,2)),current_points=np.zeros((0,2)),harris_scores=np.zeros(0),
            forward_backward_error_px=np.zeros(0),metrics=dict(harris_output=dict(capacity_policy="complete_grid",
                capacity=120,capacity_exhausted=False),detected_count=70,selected_count=35,
                minimum_accepted_features=30,minimum_grid_coverage=.2,accepted_count=0,rejected_count=35,
                usable_for_transform=False,quality_rejection_reasons=["insufficient_accepted_features","low_grid_coverage"]))
        return selection,corr,calls

    def test_empty_verifier_preserves_nonempty_original_and_restores(self):
        selection,corr,calls=self.empty_verifier_fixture()
        original=selection.verify_correspondence; audit={}
        with M.allow_empty_diagnostic_result(selection,audit):
            selection.verify_correspondence(corr,M.SHAPE)
            nonempty=copy.copy(corr); nonempty.count=4
            selection.verify_correspondence(nonempty,M.SHAPE)
        self.assertIs(selection.verify_correspondence,original)
        self.assertEqual(calls,[4])
        self.assertEqual(audit["empty_results"],1)
        self.assertFalse(audit["compute_or_quality_gates_changed"])

    def test_empty_verifier_fails_closed_on_other_invariant(self):
        for field, bad in (("backends",{}),("motion_image_size",(640,512)),("previous_points",np.zeros((1,2)))):
            selection,corr,_=self.empty_verifier_fixture(); setattr(corr,field,bad)
            original=selection.verify_correspondence
            with self.assertRaises(ValueError):
                with M.allow_empty_diagnostic_result(selection,{}):
                    selection.verify_correspondence(corr,M.SHAPE)
            self.assertIs(selection.verify_correspondence,original)
        for field,bad in (("selected_count",400),("minimum_accepted_features",1),
                          ("accepted_count",1),("rejected_count",0),("usable_for_transform",True)):
            selection,corr,_=self.empty_verifier_fixture(); corr.metrics[field]=bad
            with self.assertRaises(ValueError):
                with M.allow_empty_diagnostic_result(selection,{}):
                    selection.verify_correspondence(corr,M.SHAPE)

    def test_no_calls_to_media_inputs_or_decoder_or_rerun_fits(self):
        source=(ROOT/"scripts/run_motion_photometric_controls.py").read_text()
        tree=ast.parse(source)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        attrs=[node.func.attr for node in calls if isinstance(node.func,ast.Attribute)]
        self.assertNotIn("inputs",attrs)
        self.assertNotIn("VideoCapture",attrs)
        self.assertNotIn("probe_video",source)
        self.assertEqual(sum(isinstance(node.func,ast.Name) and node.func.id=="fit_function" for node in calls),1)

    def test_supervisor_scoped_child_and_fixed_policy(self):
        source=(ROOT/"scripts/batch_motion_photometric_controls.py").read_text()
        tree=ast.parse(source)
        popens=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and node.func.attr=="Popen"]
        self.assertEqual(len(popens),1)
        self.assertTrue(any(item.arg=="start_new_session" and item.value.value is True for item in popens[0].keywords))
        self.assertIn("safety.stop_owned(child)",source)
        self.assertNotIn("killall",source)
        self.assertNotIn("pkill",source)
        self.assertEqual(M.EXECUTION,dict(workers=1,phases=1,phase_deadline_seconds=900,batch_deadline_seconds=3600,
                                        start_below_celsius=65,stop_at_celsius=75,automatic_retries=0))


if __name__ == "__main__":
    unittest.main()
