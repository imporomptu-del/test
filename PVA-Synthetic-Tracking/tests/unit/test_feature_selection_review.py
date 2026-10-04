"""Generated metadata/pixels only; no real sources or candidate outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("feature_selection_review", Path(__file__).parents[2] / "scripts/render_feature_selection_comparison.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def metadata(arm):
    schema = "seaqr.discovery-pair.baseline.v1" if arm == "baseline" else "seaqr.discovery-feature-selection.v1"
    source = dict(path=r.SOURCE_REMOTE, sha256=r.SOURCE_SHA, frames=673, width=4784, height=3190, fps=10, codec="mjpeg", pixel_format="yuvj420p")
    receipt = dict(schema=schema, passed=True, error=None, clip="0240", source=source, processed_frames=673,
        decoded_frames_verified=673, workspace="generated", input_sha256={"generated": "0"*64}, algorithm_changed=arm=="candidate",
        detector_configuration_changed=False, annotations_supplied_to_detector=False, raw16_accessed=False, sealed_holdouts_accessed=False)
    pre = dict(schema=schema+".preflight", passed=True, clip="0240", source=source, detector_run=False, probe_passed=True,
        workspace="generated", input_sha256=receipt["input_sha256"])
    if arm == "candidate":
        original=dict(harris_capacity_policy="legacy_default",feature_image_scale=.5,feature_cpu_policy="reference",max_features=1000,max_features_per_cell=None,grid_rows=6,grid_cols=8)
        effective=dict(original,harris_capacity_policy="complete_grid",feature_cpu_policy="batched_exact_v1",max_features=384,max_features_per_cell=8)
        adapter=dict(candidate=r.CANDIDATE,estimator_instances=1,original_motion_configuration=original,effective_motion_configuration=effective,
            source_transformation=dict(exact_original_recovered=True,only_statement_change="U8 to Harris S16 scale16 offset0",heavy_capture_hooks=False))
        receipt.update(candidate=r.CANDIDATE,feature_algorithm_changed=True,tracker_configuration_changed=False,global_motion_gates_changed=False,
            production_promotion=False,feature_adapter=adapter)
        pre.update(candidate=r.CANDIDATE,feature_adapter=adapter,conversion=dict(passed=True),
            cpu_parity=dict(passed=True,exact_reference_comparison=True,same_candidate_quota_in_both_paths=True,score_precision_changed=False,max_features=384,max_features_per_cell=8,grid_rows=6,grid_cols=8,
                cases=[dict(name=name,passed=True) for name in ("native_gray8","masked_and_excluded_gray8","empty","all_ineligible","native_proxy_boundaries")]),
            controls=[dict(name=name,passed=True,closed=True) for name in ("low_contrast_static","low_contrast_translated","flat_static")])
    launch=dict(source_sha256=r.SOURCE_SHA,source=r.SOURCE_REMOTE,expected_frames=673,max_frames=None,fps=10,annotations_supplied_to_detector=False,
        source_probe=dict(codec="mjpeg",pixel_format="yuvj420p",width=4784,height=3190,declared_frame_count=673,frame_rate="10"),
        configuration={},package_sha256={"generated.py":"1"*64})
    report=dict(completed=True,full_clip=True,frames=673,source_sha256=r.SOURCE_SHA,configuration={},
        frame_decode=dict(decoded_frames=673,consumed_frames=673,dropped_frames=0,worker_joined=True,capture_released=True,
            maximum_observed_frames_ahead=1,contract=dict(execution="prefetch_one")),availability=dict(counts=dict(detection_ready_frames=673)))
    return receipt,pre,launch,report


def row(index, ready=True, tracks=None):
    return dict(frame_index=index,timestamp_ns=index*100_000_000,segment=0,tracks=tracks or [],candidates=[],
        coverage=dict(full_shape_hw=[3190,4784],native_pixel_sampling=True,configured_crop=None,warmup=not ready,
            detection_ready=ready,searchable_pixels=100 if ready else 0,unavailable_reason=None if ready else "warmup"),
        motion=dict(status="accepted",reset=False))


def track(tid="dark:1", point=(2800,2800), measured=True):
    return dict(track_id=tid,segment=0,measured=measured,qualified_moving=True,source_xy=[2810.,2810.],measurement_source_xy=list(point) if measured else None)


class ContractTests(unittest.TestCase):
    def test_both_declared_arms_valid(self):
        for arm in ("baseline","candidate"):r.validate_contract(*metadata(arm),arm)

    def test_baseline_receipt_cannot_impersonate_candidate(self):
        with self.assertRaises(ValueError):r.validate_contract(*metadata("baseline"),"candidate")

    def test_candidate_gain_and_policy_declared(self):
        for change in (dict(harris_gain=1,harris_capacity_policy="complete_grid",feature_image_scale=.5),dict(r.CANDIDATE,feature_image_scale=1)):
            data=copy.deepcopy(metadata("candidate"));data[0]["candidate"]=change
            with self.assertRaises(ValueError):r.validate_contract(*data,"candidate")

    def test_failed_incomplete_wrong_source_scope_rejected(self):
        for key,value in (("passed",False),("processed_frames",672),("production_promotion",True),("tracker_configuration_changed",True),("error","failed")):
            data=copy.deepcopy(metadata("candidate"));data[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_contract(*data,"candidate")
        data=copy.deepcopy(metadata("candidate"));data[0]["source"]["sha256"]="0"*64
        with self.assertRaises(ValueError):r.validate_contract(*data,"candidate")

    def test_failed_preflight_controls_not_silently_replaced(self):
        for which in ("passed","conversion","controls"):
            data=copy.deepcopy(metadata("candidate"))
            if which=="passed":data[1][which]=False
            elif which=="conversion":data[1][which]["passed"]=False
            else:data[1][which][0]["passed"]=False
            with self.assertRaises(ValueError):r.validate_contract(*data,"candidate")

    def test_changed_effective_configuration_rejected(self):
        data=copy.deepcopy(metadata("candidate"));data[0]["feature_adapter"]["effective_motion_configuration"]["new_gate"]=1
        with self.assertRaises(ValueError):r.validate_contract(*data,"candidate")

    def test_exact_cpu_parity_required_for_new_execution(self):
        for change in (lambda p:p.update(passed=False),
                       lambda p:p.update(max_features_per_cell=9),
                       lambda p:p.update(score_precision_changed=True),
                       lambda p:p["cases"].pop(),
                       lambda p:p["cases"][-1].update(passed=False)):
            data=copy.deepcopy(metadata("candidate"))
            change(data[1]["cpu_parity"])
            with self.assertRaises(ValueError):r.validate_contract(*data,"candidate")

    def test_report_decode_lifecycle_rejected(self):
        for key,value in (("decoded_frames",672),("dropped_frames",1),("worker_joined",False),("capture_released",False)):
            data=copy.deepcopy(metadata("baseline"));data[3]["frame_decode"][key]=value
            with self.assertRaises(ValueError):r.validate_contract(*data,"baseline")

    def test_bound_file_and_receipt_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"run").mkdir()
            rec,pre,launch,report=metadata("baseline")
            paths={"preflight":root/"preflight.json","launch":root/"run/launch.json","report":root/"run/report.json","journal":root/"run/frames.jsonl"}
            for key,value in (("preflight",pre),("launch",launch),("report",report),("journal",{})):
                r.base.write_json(paths[key],value);rec[key+"_sha256"]=r.sha(paths[key])
            r.base.write_json(root/"execution_receipt.json",rec);digest=r.sha(root/"execution_receipt.json")
            self.assertEqual(r.load_arm(root,digest,"baseline")["receipt"],rec)
            with self.assertRaises(ValueError):r.load_arm(root,"0"*64,"baseline")
            with paths["journal"].open("a") as f:f.write("\n")
            with self.assertRaises(ValueError):r.load_arm(root,digest,"baseline")


class TimelineTests(unittest.TestCase):
    def test_ready_zero_candidates_distinct_from_unavailable(self):
        self.assertEqual(r.availability(row(50))["label"],"READY")
        self.assertEqual(r.availability(row(50,False))["label"],"UNAVAILABLE: warmup")

    def test_inconsistent_or_non_native_availability_rejected(self):
        for key,value in (("detection_ready",False),("unavailable_reason","warmup"),("native_pixel_sampling",False),("configured_crop",[0,0,1,1]),("warmup",0)):
            data=row(50);data["coverage"][key]=value
            with self.assertRaises(ValueError):r.availability(data)

    def test_full673_rows_before_window_preserve_coast_age(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"journal.jsonl"
            with path.open("x") as f:
                for i in range(673):
                    tracks=[track(measured=i==49)] if 49<=i<=50 else []
                    f.write(json.dumps(row(i,tracks=tracks))+"\n")
            result=r.collect(path,"baseline",metadata("baseline")[3])
            self.assertEqual(len(result),91)
            self.assertEqual(result[50]["channels"]["track_context"][0]["last_measurement_age_ns"],100_000_000)
            self.assertEqual(sorted(result),list(range(50,106))+list(range(430,465)))

    def test_missing_extra_rows_and_report_readiness_rejected(self):
        for n in (672,674):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"journal.jsonl"
                with path.open("x") as f:
                    for i in range(n):f.write(json.dumps(row(i))+"\n")
                with self.assertRaises(ValueError):r.collect(path,"baseline",metadata("baseline")[3])


class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.image=np.full((3190,4784,3),115,np.uint8)

    def data(self,index,tracks,ready=True,stream="generated"):
        value=row(index,ready,tracks)
        return dict(channels=r.base.ObservationOutput(stream).update(value),status=r.availability(value))

    def test_fixed_inventory_and_native_crop_raw_exact(self):
        b=self.data(430,[track()]);c=self.data(430,[track("dark:changed"),track("bright:2",(2900,2800))],False)
        original=copy.deepcopy((b,c))
        canvas,shown=r.canvas_for(self.image,b,c,r.WINDOWS[1])
        np.testing.assert_array_equal(canvas[r.HEADER:r.HEADER+384,:640],self.image[2592:2976,2592:3232])
        self.assertEqual(shown["baseline"]["current_measurements_in_view"],1)
        self.assertEqual(shown["candidate"]["current_measurements_in_view"],2)
        self.assertEqual(shown["candidate"]["label"],"UNAVAILABLE: warmup")
        self.assertEqual((b,c),original)
        self.assertTrue(np.all(self.image==115))
        self.assertEqual([v["last"]-v["first"]+1 for v in r.WINDOWS],[56,35])

    def test_prediction_not_measurement_offscreen_not_clamped(self):
        b=self.data(430,[track(measured=False),track("dark:off",(-1,50))]);c=self.data(430,[])
        _,shown=r.canvas_for(self.image,b,c,r.WINDOWS[1])
        self.assertEqual(shown["baseline"]["current_measurements_in_view"],0)
        self.assertEqual(shown["baseline"]["predictions_in_view"],1)
        self.assertEqual(shown["baseline"]["qualified_states_offscreen"],1)

    def test_window_and_arm_frame_mismatch_rejected(self):
        b=self.data(430,[]);c=self.data(431,[])
        with self.assertRaises(ValueError):r.canvas_for(self.image,b,c,r.WINDOWS[1])
        with self.assertRaises(ValueError):r.canvas_for(self.image,b,b,dict(r.WINDOWS[1],crop=[0,0,640,384]))

    def test_full_field_overview_raw_exact_global_resize(self):
        b=self.data(50,[]);c=self.data(50,[])
        canvas,_=r.canvas_for(self.image,b,c,r.WINDOWS[0])
        self.assertEqual(canvas.shape[1],2912)
        np.testing.assert_array_equal(canvas[r.HEADER:r.HEADER+640,:960],np.full((640,960,3),115,np.uint8))


if __name__=="__main__":unittest.main()
