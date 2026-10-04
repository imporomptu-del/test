"""Generated metadata fixtures only: no media, VPI, estimator or remote calls."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("summarize_pva_robustness")
W = load("probe_pva_robustness")


def desc(value):
    value = np.ascontiguousarray(value)
    raw = value.tobytes()
    return dict(dtype=value.dtype.str, shape=list(value.shape), data_base64=base64.b64encode(raw).decode(),
                sha256=hashlib.sha256(raw).hexdigest())


def record(case=None, depth=2, mode="trace", selected=35, slots=None, error=0., fit_accepted=True):
    case = copy.deepcopy(case or M.inventory()[0])
    if case["scientific_class"] == "sparse_corner":
        selected, fit_accepted = 4, False
    points = np.array([[80+(i%7)*10, 80+(i//7)*10] for i in range(selected)], np.float32)
    p = (points+.5)*2-.5
    q = p + np.asarray(case["truth_displacement_xy"], np.float32) + np.array([error, 0], np.float32)
    scores = np.arange(selected, dtype=np.float32)+1
    indices = np.arange(selected, dtype=np.int64)+41
    slots = np.arange(selected) if slots is None else np.array(slots, np.int64)
    mask = np.zeros(selected, bool); mask[slots] = True
    slots = np.flatnonzero(mask)
    count = len(slots)
    fb = np.zeros(selected, np.float32)
    model_available = count >= 30
    matrix = np.eye(3); matrix[:2,2] = np.asarray(case["truth_displacement_xy"])+[error,0]
    fit_mask = np.ones(count, bool) if model_available else np.zeros(count, bool)
    residuals = np.zeros(count, np.float64) if model_available else np.full(count, np.nan, np.float64)
    fit = dict(model="translation", mapping="previous_frame_pixels_to_current_frame_pixels",
        previous_frame_index=0, current_frame_index=1, previous_to_current_matrix=matrix.tolist() if model_available else None,
        parameters=dict(translation_x_px=float(matrix[0,2]), translation_y_px=float(matrix[1,2])) if model_available else None,
        quality_status="accepted" if fit_accepted and model_available else "rejected",
        rejection_reasons=[] if fit_accepted and model_available else ["insufficient_correspondences"],
        metrics=dict(correspondence_count=count, inlier_count=int(fit_mask.sum()), inlier_ratio=float(fit_mask.mean()) if count else 0),
        inlier_indices=np.flatnonzero(fit_mask).tolist())
    corr = dict(arrays=dict(previous_points=desc(p[mask]), current_points=desc(q[mask]),
        harris_scores=desc(scores[mask]), forward_backward_error_px=desc(fb[mask])),
        full_image_size=[640,512], motion_image_size=[320,256], metrics=dict(accepted_count=count), backends={})
    result = dict(correspondence=corr, fit=fit, inlier_mask=desc(fit_mask), residuals_px=desc(residuals))
    motion = dict(pyramid_levels=depth, pyramid_scale=.5, feature_image_scale=.5, flow_status_policy="legacy_default",
        forward_backward_check=True, max_features=384, max_features_per_cell=8, minimum_accepted_features=30,
        minimum_grid_coverage=.2, window_size=11)
    pixel = hashlib.sha256(case["id"].encode()).hexdigest()
    canonical = dict(case_metadata=case, effective_motion_configuration=motion, global_configuration=M.EXPECTED_GLOBAL,
        input_pair=dict(native_shape_hw=[512,640], source_media_accessed=False, previous_pixel_sha256="a"*64,
            current_pixel_sha256=pixel, original_current_pixel_sha256=pixel,
            quantization=dict(current_maximum_abs_rounding_error_dn=.5, clipped_pixels=0)),
        frame_pixel_sha256=dict(previous="b"*64, current=pixel), result=result, accepted_truth={})
    levels = [dict(array=desc(np.full(shape, 120, np.uint8))) for shape in ((256,320),(128,160),(64,80),(32,40))[:depth]]
    data = dict(pyramids=dict(previous=levels, current=levels,
        proxy={side:dict(array=desc(np.full((256,320),120,np.uint8))) for side in ("previous","current")}),
        before_forward=dict(selected_count=selected, selected_points=desc(points), selected_scores=desc(scores), selected_indices=desc(indices)),
        after_forward=dict(forward_status=desc(np.zeros(selected,np.uint8))), after_backward={}, readback={},
        final_filter=dict(previous_full_points=desc(p), current_full_points=desc(q), accepted_mask=desc(mask),
            selected_indices_of_accepted_points=desc(slots), selected_indices_of_rejected_points=desc(np.flatnonzero(~mask)),
            accepted_previous=desc(p[mask]), accepted_current=desc(q[mask]), accepted_scores=desc(scores[mask]),
            accepted_fb_error=desc(fb[mask]), fb_error=desc(fb)))
    capture = dict(method_calls=1, error=None, data=data, stages=["pyramids","before_forward","after_forward","after_backward","readback","final_filter"])
    if case["scientific_class"] == "degenerate":
        canonical["result"] = dict(status="unavailable", reason="no features", correspondence=None, fit=None)
        capture = dict(method_calls=1, error=None, data=dict(pyramids=data["pyramids"]), stages=["pyramids"],
                       partial=True, expected_feature_unavailable=True)
    return dict(schema="seaqr.pva-robustness.v1", case=case["id"], case_metadata=case, depth=depth, mode=mode,
        completed=True, passed_integrity=True, error=None, generated_only=True, source_media_accessed=False,
        detector_run=False, production_changes=False, production_promotion=False, preflight_pva_calls=0,
        focal_estimate_calls=1, global_fits=0 if case["scientific_class"]=="degenerate" else 1,
        closed=True, clocks_changed=False, capture=capture if mode=="trace" else None,
        canonical_nontiming=canonical, canonical_nontiming_sha256=M.canonical_sha(canonical),
        observation_contract=dict(method_sha256="bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b",
            estimator_source_changed=False, initial_status_changed=False, extra_pva_compute_calls=0,
            sys_settrace=mode=="trace", additional_cpu_readbacks=mode=="trace"))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(root):
    directory, bundle = root/"evidence", root/"bundle"
    directory.mkdir(); bundle.mkdir()
    frozen = W.freeze_spec()
    for name in M.SOURCES:
        original = ROOT/"scripts"/name
        if name in {"summarize_pva_robustness.py", "batch_discovery_pair.py"}:
            (bundle/name).write_bytes(original.read_bytes())
        else:
            (bundle/name).write_text("synthetic source fixture: "+name)
    frozen["source_sha256"] = {name:M.sha(bundle/name) for name in M.SOURCES}
    write(directory/"freeze.json", frozen); digest=M.sha(directory/"freeze.json")
    batch=dict(schema="seaqr.pva-robustness.batch.v1", complete=True, execution_passed=True,
        generated_only=True, camera_media_accessed=False, clock_writes=False, error=None, not_run=[], current=None,
        freeze_sha256=digest, source_sha256=frozen["source_sha256"], elapsed_seconds=104., phases=[], parity_passed=True,
        execution=dict(workers=1, phases=104, phase_deadline_seconds=900, batch_deadline_seconds=3600,
                       start_below_celsius=65, stop_at_celsius=75, automatic_retries=0))
    parity=dict(schema="seaqr.pva-robustness.batch.v1.parity", cases=[], passed=True)
    workspace="/tmp/seaqr_pva_robustness_20261001_Ab1234"
    for case in M.inventory():
        for depth in M.DEPTHS:
            for mode in M.MODES:
                row=record(case,depth,mode)
                row["input_sha256"] = dict(freeze_sha256=digest, source_sha256=frozen["source_sha256"], helpers=frozen["helpers"])
                name=M.phase_name(case["id"],depth,mode)
                write(directory/(name+".json"),row)
                command=["/usr/bin/python3","-I",workspace+"/probe_pva_robustness.py","--workspace",workspace,
                    "--freeze",workspace+"/freeze.json","--freeze-sha256",digest,"--case",case["id"],"--depth",str(depth)]
                if mode=="trace":command.append("--trace")
                batch["phases"].append(dict(name=name,case=case["id"],depth=depth,mode=mode,pid=1000+len(batch["phases"]),
                    returncode=0,elapsed_seconds=1.,result_sha256=M.sha(directory/(name+".json")),command=command))
            parity["cases"].append(dict(case=case["id"],depth=depth,passed=True,exact_nontiming_parity=True,
                baseline_sha256=row["canonical_nontiming_sha256"],trace_sha256=row["canonical_nontiming_sha256"]))
    write(directory/"parity.json",parity);batch["parity_sha256"]=M.sha(directory/"parity.json")
    write(directory/"batch_status.json",batch)
    return directory,bundle,digest


class Numerical(unittest.TestCase):
    def test_independent_inventory_exact(self):
        self.assertEqual(M.inventory(),W.inventory())
        self.assertEqual([sum(c["scientific_class"]==kind for c in M.inventory()) for kind in
            ("global_positive","sparse_corner","degenerate")],[15,8,3])

    def test_native_nonzero_signed_truth_and_factor_two_error(self):
        case=M.inventory()[1]; row=record(case,error=.5)
        analyzed,_=M.accepted_analysis(row["canonical_nontiming"],case)
        self.assertEqual(analyzed["all_accepted"]["maximum_px"],.5)
        self.assertEqual(analyzed["original_fit"]["translation_error_px"],.5)
        self.assertFalse(analyzed["point_guard_passed"])
        self.assertFalse(M.gates(case,analyzed)["global_positive_passed"])

    def test_native_pixel_center_lift_and_mask_order(self):
        row=record(slots=[0,4,7]); case=row["case_metadata"]
        analyzed,a=M.accepted_analysis(row["canonical_nontiming"],case)
        available,internal=M.capture_analysis(row,case,a)
        self.assertEqual(available["final_accepted_selected_slots"],[0,4,7])
        self.assertEqual(available["truth_interior_missing_count"],32)
        bad=copy.deepcopy(row);data=bad["capture"]["data"]
        data["final_filter"]["previous_full_points"]=desc(M.array(data["before_forward"]["selected_points"])*2)
        with self.assertRaisesRegex(ValueError,"native"):
            M.capture_analysis(bad,case,a)

    def test_truth_mask_boundaries_do_not_use_measured_endpoint(self):
        p=np.array([[128,128],[127.999,200],[511,200],[500,200],[200,383],[200,384]])
        np.testing.assert_array_equal(M.truth_mask(p,[2,0]),[True,False,False,True,True,False])
        row=record(error=100.)
        value,_=M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])
        self.assertEqual(value["truth_interior"]["maximum_px"],100.)

    def test_empty_not_zero_and_no_fit_not_rejected(self):
        row=record(slots=[])
        value,a=M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])
        self.assertEqual(value["accepted_count"],0);self.assertIsNone(value["all_accepted"]["maximum_px"])
        self.assertFalse(value["point_guard_passed"]);self.assertTrue(value["original_fit"]["fit_ran"])
        self.assertFalse(value["original_fit"]["accepted"])
        row=record(M.inventory()[16]);value,a=M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])
        self.assertIsNone(value["accepted_count"]);self.assertIsNone(value["original_fit"]["accepted"])
        self.assertFalse(value["original_fit"]["fit_ran"])
        availability,internal=M.capture_analysis(row,row["case_metadata"],a)
        self.assertIsNone(availability["selected_count"]);self.assertIsNone(internal)

    def test_rejected_finite_model_is_not_success_or_unknown(self):
        row=record(error=.5,fit_accepted=False)
        value,_=M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])
        self.assertTrue(value["original_fit"]["model_available"])
        self.assertEqual(value["original_fit"]["translation_error_px"],.5)
        self.assertFalse(M.gates(row["case_metadata"],value)["global_positive_passed"])

    def test_matrix_parameter_residual_and_mask_tampering(self):
        for mutation in (
            lambda c:c["result"]["fit"]["parameters"].update(translation_x_px=1.),
            lambda c:c["result"]["fit"]["previous_to_current_matrix"][0].__setitem__(0,2.),
            lambda c:c["result"].update(residuals_px=desc(np.ones(35,np.float64))),
            lambda c:c["result"].update(inlier_mask=desc(np.zeros(35,bool))),
            lambda c:c["result"]["correspondence"]["arrays"].update(current_points=desc(np.full((35,2),np.nan,np.float32))),
        ):
            row=record();mutation(row["canonical_nontiming"])
            with self.assertRaises(ValueError):M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])

    def test_common_seed_join_retains_lost_recovered_and_duplicate_coordinates(self):
        population=[]
        for slots in ([0,4,7],[4,7,8]):
            row=record(slots=slots);_,a=M.accepted_analysis(row["canonical_nontiming"],row["case_metadata"])
            _,i=M.capture_analysis(row,row["case_metadata"],a);population.append(i)
        for p in population:
            p["points"] = p["points"].copy(); p["points"][1] = p["points"][0]
        comparison=M.matched_populations(*population)
        self.assertEqual(comparison["common_accepted_selected_slots"],[4,7])
        self.assertEqual(comparison["left_only_selected_slots"],[0]);self.assertEqual(comparison["right_only_selected_slots"],[8])
        self.assertEqual(comparison["common_truth_interior_count"],2)

    def test_cross_depth_seed_prefix_identity_and_partial_negatives(self):
        four,two=record(depth=4),record(depth=2)
        self.assertTrue(M.cross_depth_identity(four,two)["selected_seed_identity_exact"])
        for mutate in (
            lambda r:r["capture"]["data"]["before_forward"].update(selected_indices=desc(np.arange(35,dtype=np.int64))),
            lambda r:r["capture"]["data"]["pyramids"]["current"][1].update(array=desc(np.zeros((128,160),np.uint8))),
            lambda r:r["canonical_nontiming"]["effective_motion_configuration"].update(window_size=9),
        ):
            changed=copy.deepcopy(two);mutate(changed)
            with self.assertRaises(ValueError):M.cross_depth_identity(four,changed)
        result=M.cross_depth_identity(record(M.inventory()[16],4),record(M.inventory()[16],2))
        self.assertTrue(result["selection_unavailable_at_both_depths"])
        self.assertIsNone(result["selected_seed_identity_exact"])

    def test_actual_saved_static_trace_regression_no_media(self):
        paths=[ROOT.parent/"outputs"/folder/"jetson"/filename for folder in
            ("seaqr_static_pva_texture_20261001","seaqr_pva_depth_control_20261001")
            for filename in ("bridge_trace.json","texture_trace.json")]
        for path in paths:
            if not path.is_file():self.skipTest("Archived metadata not bundled")
            with self.subTest(path=path):
                row=M.read(path);case=copy.deepcopy(M.inventory()[0])
                analysis,accepted=M.accepted_analysis(row["canonical_nontiming"],case)
                available,_=M.capture_analysis(row,case,accepted)
                self.assertEqual(available["truth_interior_selected_count"],123 if path.name.startswith("bridge") else 110)

    def test_strict_json_array_tamper_nonfinite_and_shape(self):
        value=desc(np.array([np.nan],np.float64));self.assertTrue(np.isnan(M.array(value)[0]))
        for key,bad in (("sha256","0"*64),("shape",[True]),("dtype","O")):
            changed=copy.deepcopy(value);changed[key]=bad
            with self.assertRaises(ValueError):M.array(changed)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json"
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e9999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):M.read(path)


class Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve()
        cls.directory,cls.bundle,cls.digest=fixture(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full104_receipts_and52_parity_all_gates(self):
        value=M.summarize(self.directory,self.digest,self.bundle)
        self.assertEqual(value["children_completed"],104);self.assertEqual(value["base_trace_pairs_checked"],52)
        self.assertTrue(value["ready_for_bounded_real_video_validation"])
        self.assertEqual([value["scientific_gates"]["2"][key]["passed"] for key in
            ("global_positive","degenerate_rejection","point_guard")],[15,3,23])
        self.assertFalse(value["production_promotion"]);json.dumps(value,allow_nan=False)

    def test_lifecycle_helper_fit_count_and_scope_rejected(self):
        frozen=M.read(self.directory/"freeze.json");case=M.inventory()[0]
        row=M.read(self.directory/(M.phase_name(case["id"],4,"base")+".json"))
        for key,bad in (("error","oops"),("closed",False),("clocks_changed",True),("global_fits",0),
                        ("source_media_accessed",True),("production_promotion",True)):
            value=copy.deepcopy(row);value[key]=bad
            with self.assertRaises(ValueError):M.validate_child(value,case,4,"base",self.digest,frozen)
        value=copy.deepcopy(row);value["input_sha256"]["helpers"]={}
        with self.assertRaises(ValueError):M.validate_child(value,case,4,"base",self.digest,frozen)

    def test_scientific_failure_retained_after_all104_children(self):
        case=M.inventory()[0]
        names=[M.phase_name(case["id"],2,mode)+".json" for mode in M.MODES]+["batch_status.json","parity.json"]
        originals={name:(self.directory/name).read_bytes() for name in names}
        try:
            for options in (dict(error=.5),dict(slots=[])):
                batch=json.loads(originals["batch_status.json"])
                parity=json.loads(originals["parity.json"])
                for mode in M.MODES:
                    name=M.phase_name(case["id"],2,mode)+".json"
                    old=json.loads(originals[name]);row=record(case,2,mode,**options)
                    row["input_sha256"]=old["input_sha256"]
                    write(self.directory/name,row)
                    entry=next(p for p in batch["phases"] if p["name"]+".json"==name)
                    entry["result_sha256"]=M.sha(self.directory/name)
                pair=next(p for p in parity["cases"] if p["case"]==case["id"] and p["depth"]==2)
                pair.update(baseline_sha256=row["canonical_nontiming_sha256"],trace_sha256=row["canonical_nontiming_sha256"])
                write(self.directory/"parity.json",parity);batch["parity_sha256"]=M.sha(self.directory/"parity.json")
                write(self.directory/"batch_status.json",batch)
                result=M.summarize(self.directory,self.digest,self.bundle)
                self.assertEqual(result["children_completed"],104)
                self.assertTrue(result["all_base_trace_parity_passed"])
                self.assertFalse(result["ready_for_bounded_real_video_validation"])
                self.assertEqual(result["scientific_gates"]["2"]["global_positive"]["passed"],14)
                self.assertEqual(result["scientific_gates"]["2"]["point_guard"]["passed"],22)
                if options.get("slots")==[]:
                    self.assertEqual(result["cases"][0]["cross_depth_populations"]["left_only_accepted_count"],35)
                    self.assertIsNone(result["cases"][0]["depths"]["2"]["all_accepted"]["maximum_px"])
        finally:
            for name,raw in originals.items():(self.directory/name).write_bytes(raw)

    def test_hash_inventory_parity_and_command_tamper_fail_closed(self):
        path=self.directory/"batch_status.json";original=path.read_bytes()
        try:
            for mutation in (
                lambda b:b["phases"].pop(),
                lambda b:b["phases"][1].update(pid=b["phases"][0]["pid"]),
                lambda b:b["phases"][0].update(result_sha256="0"*64),
                lambda b:b["phases"][0]["command"].__setitem__(12,"2"),
                lambda b:b.update(parity_passed=False),
            ):
                batch=json.loads(original);mutation(batch);write(path,batch)
                with self.assertRaises(ValueError):M.summarize(self.directory,self.digest,self.bundle)
        finally:path.write_bytes(original)

    def test_input_changed_during_read_is_rejected(self):
        original=M.sha;target=self.directory/"batch_status.json";calls=0
        def changed(path):
            nonlocal calls
            if Path(path)==target:
                calls+=1
                if calls>1:return "0"*64
            return original(path)
        with patch.object(M,"sha",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"changed during"):M.summarize(self.directory,self.digest,self.bundle)

    def test_fresh_output_serialization_no_overwrite(self):
        path=self.root/"summary.json"
        value=M.save_summary(self.directory,self.digest,self.bundle,path)
        self.assertEqual(M.read(path),value)
        with self.assertRaisesRegex(ValueError,"overwrite"):M.save_summary(self.directory,self.digest,self.bundle,path)


if __name__ == "__main__":
    unittest.main()
