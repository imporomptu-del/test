"""Generated metadata only; no AVI, hardware, fitting or candidate outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / "scripts/summarize_feature_supply_registration.py"
SPEC = importlib.util.spec_from_file_location("registration_summary_tests", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def pair(index, accepted=True, translation=(0.0, 0.0), fitted=True, failure=False):
    metrics = dict(correspondence_count=60, inlier_count=54, inlier_ratio=.9,
        inlier_grid_coverage=dict(occupied_cells=12, fraction=.25), median_reprojection_error_px=.05,
        p90_reprojection_error_px=.1, maximum_reprojection_error_px=.3,
        coverage_acceptance_path="dense" if accepted else "rejected")
    tx, ty = translation
    fit = dict(model="translation", previous_frame_index=index-1, current_frame_index=index,
        mapping="previous_frame_pixels_to_current_frame_pixels", parameters=dict(translation_x_px=tx, translation_y_px=ty),
        previous_to_current_matrix=[[1, 0, tx], [0, 1, ty], [0, 0, 1]],
        quality_status="accepted" if accepted else "rejected",
        rejection_reasons=[] if accepted else ["low_inlier_grid_coverage"], metrics=metrics, timing_ms=1.5)
    if not fitted:
        fit.update(parameters=None, previous_to_current_matrix=None)
        fit["metrics"] = dict(correspondence_count=60, inlier_count=0, inlier_ratio=0)
    corr = dict(detected_count=120, selected_count=100, accepted_count=60, rejected_count=40,
        track_survival_ratio=.6, grid_coverage=dict(occupied_cells=12, fraction=.25),
        median_displacement_px=.05, p95_displacement_px=.2, median_forward_backward_error_px=.02,
        usable_for_transform=False, rejections={"forward_status":40},
        feature_exclusions_before_tracking={"saturated_neighborhood":3},
        harris_output=dict(capacity=60300, capacity_exhausted=False))
    motion = dict(backend="pva", cpu_warp_fallback=False, stabilization_execution="cuda_cubic_resident",
        status="accepted" if accepted else "reset_reference", reset=not accepted, pva_failure=False,
        accepted=accepted, rejection_reasons=fit["rejection_reasons"], motion_fit=fit,
        correspondence_metrics=corr, motion_backends=dict(runner.BACKENDS),
        pva_timings_ms=dict(harris_pva=2, total_including_metrics=10),
        warp_timings_ms=dict(exact_cuda_total=3))
    if failure:
        motion = dict(backend="pva", cpu_warp_fallback=False, stabilization_execution="cuda_cubic_resident",
                      status="reset_reference", reset=True, pva_failure=True, error="zero features",
                      rejection_reasons=["pva_runtime_error"])
    if index == 0:
        motion = dict(backend="pva", cpu_warp_fallback=False, stabilization_execution="cuda_cubic_resident",
                      status="initial_reference", reset=False, pva_failure=False, rejection_reasons=[])
    return dict(frame_index=index, timestamp_ns=index*100_000_000, motion=motion,
        coverage=dict(full_shape_hw=[3190,4784], native_pixel_sampling=True, configured_crop=None,
                      warmup=index == 0, searchable_pixels=0 if index == 0 else 100, detection_ready=index > 0),
        timings_ms=dict(decode=5, motion_and_warp=15, detection=2, tracking=1),
        candidates=["unneeded"], tracks=["unneeded"])


def rows():
    return [pair(i) for i in range(runner.FRAMES)]


def attempts(values):
    return [dict(frame=r["frame_index"], error="PvaMotionError('zero features')" if r["motion"]["pva_failure"] else None,
                 expected_unavailable=r["motion"]["pva_failure"]) for r in values[1:]]


def artifacts(arm="baseline", clip="0170"):
    source = runner.source_spec(clip)
    schema = "seaqr.discovery-pair.baseline.v1" if arm == "baseline" else "seaqr.discovery-feature-supply.v1"
    receipt = dict(schema=schema, passed=True, clip=clip, workspace="/generated", source=source,
        processed_frames=runner.FRAMES, decoded_frames_verified=runner.FRAMES,
        algorithm_changed=arm == "candidate", detector_configuration_changed=False,
        annotations_supplied_to_detector=False, raw16_accessed=False, sealed_holdouts_accessed=False,
        input_sha256={"generated":"fixture"}, config_sha256=runner.CONFIG_SHA, motion_config_sha256=runner.MOTION_SHA,
        adapters={"fixed":"identity"}, libraries={"fixed":"identity"}, v29_freeze_sha256="frozen",
        tracking_transformed_sha256="tracking", timing_instrumentation="generated")
    if arm == "candidate":
        original = dict(feature_image_scale=.5, harris_capacity_policy="legacy_default", quality_gate=30)
        receipt.update(candidate=dict(runner.CANDIDATE), feature_algorithm_changed=True,
            global_motion_gates_changed=False, tracker_configuration_changed=False, production_promotion=False,
            feature_adapter=dict(original_motion_configuration=original,
                effective_motion_configuration=dict(original,harris_capacity_policy="complete_grid"),
                source_transformation=dict(original_method_sha256=runner.METHOD_SHA,
                    transformed_method_sha256=runner.GAIN_METHOD_SHA, exact_original_recovered=True)))
    pre = dict(schema=schema+".preflight", passed=True, clip=clip, source=source,
               workspace=receipt["workspace"], input_sha256=receipt["input_sha256"])
    config = dict(motion_backend="pva", detector="unchanged")
    report = dict(schema="seaqr.visible-baseline.v1", completed=True, full_clip=True, frames=runner.FRAMES,
        source_sha256=source["sha256"], configuration=config, elapsed_seconds=100, processed_fps=runner.FRAMES/100,
        frame_decode=dict(decoded_frames=runner.FRAMES,consumed_frames=runner.FRAMES,dropped_frames=0,
                          worker_joined=True,capture_released=True),
        availability=dict(counts=dict(detection_ready_frames=runner.FRAMES-1)), timing_semantics="overlap")
    launch = dict(source=source["path"],source_sha256=source["sha256"],configuration=config,
        config_sha256=runner.CONFIG_SHA,motion_config_sha256=runner.MOTION_SHA,
        code_sha256={"fixed":"identity"},package_sha256={"fixed":"identity"},
        source_probe=dict(width=4784,height=3190,declared_frame_count=runner.FRAMES,
                          codec="mjpeg",pixel_format="yuvj420p",frame_rate="10"))
    return receipt, pre, report, launch


def write_bundle(root, arm, clip):
    base = root / clip
    (base / "run").mkdir(parents=True)
    receipt, pre, report, launch = artifacts(arm,clip)
    values = rows()
    receipt["motion_attempts"] = attempts(values)
    for name, value in (("preflight.json",pre),("run/report.json",report),("run/launch.json",launch)):
        (base/name).write_text(json.dumps(value))
    (base/"run/frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in values))
    for key,name in (("preflight","preflight.json"),("report","run/report.json"),
                     ("launch","run/launch.json"),("journal","run/frames.jsonl")):
        receipt[key+"_sha256"] = runner.sha(base/name)
    (base/"execution_receipt.json").write_text(json.dumps(receipt))
    return base


class SummaryTests(unittest.TestCase):
    def test_quantiles_missing_and_signed_values(self):
        value = runner.summary([None,-2,0,2,4])
        self.assertEqual((value["total"],value["count"],value["missing"]),(5,4,1))
        self.assertEqual(value["mean"],1)
        self.assertEqual(value["median"],1)
        self.assertAlmostEqual(value["p90"],3.4)
        self.assertIsNone(runner.summary([None])["median"])
        for invalid in (True,float("nan"),float("inf"),"1"):
            with self.assertRaises(ValueError): runner.summary([invalid])

    def test_all_transitions_and_native_vector_difference(self):
        b,c = rows(),rows()
        b[2]=pair(2,accepted=False)
        c[3]=pair(3,accepted=False)
        b[4]=pair(4,accepted=False,fitted=False)
        c[4]=pair(4,accepted=False,fitted=False)
        c[1]=pair(1,translation=(3,4))
        result=runner.compare_rows(b,c)
        self.assertEqual(result["transition_counts"],dict(both_accepted=669,baseline_only=1,candidate_only=1,neither_accepted=1))
        disagreement=result["translation_disagreement"]
        self.assertEqual(disagreement["points"]["maximum"],5)
        self.assertEqual(disagreement["worst_10"][0]["frame_index"],1)
        self.assertEqual(result["newly_accepted"]["frame_indices"],[2])
        self.assertEqual(result["lost_acceptance"]["frame_indices"],[3])
        self.assertEqual(result["paired_fit_metric_difference"]["metrics"]["inlier_ratio"]["median"],0)

    def test_no_joint_acceptance_is_unavailable_not_zero(self):
        b,c=rows(),[pair(i,accepted=False,fitted=False) for i in range(runner.FRAMES)]
        result=runner.compare_rows(b,c)
        self.assertEqual(result["translation_disagreement"]["points"]["count"],0)
        self.assertIsNone(result["translation_disagreement"]["points"]["median"])
        self.assertEqual(result["joint_accepted"]["candidate"]["frames"],0)

    def test_original_fit_acceptance_not_correspondence_flag(self):
        values=rows()
        values[1]=pair(1,accepted=False,fitted=False)
        r,p,report,launch=artifacts()
        data=dict(receipt=r,report=report,rows=values)
        result=runner.arm_summary(data)
        self.assertEqual(result["accepted_pairs"],671)
        self.assertEqual(result["fits"]["all_fitted"]["frames"],671)
        self.assertEqual(result["unavailable_fit_parameters"],1)
        self.assertEqual(result["selected_total"],67200)
        self.assertEqual(result["point_weighted_survival"],.6)
        self.assertEqual(result["flow_rejection_totals"],{"forward_status":26880})
        self.assertEqual(result["timing"]["all_frames"]["outer_stages_ms"]["motion_and_warp"]["mean"],15)
        self.assertEqual(result["timing"]["first_pair"]["frames"],1)
        self.assertEqual(result["timing"]["later_pairs"]["frames"],671)


class ValidationTests(unittest.TestCase):
    def test_complete_inventory_and_pair_mapping(self):
        values=rows()
        runner.validate_rows(values,attempts(values))
        for change in (lambda x:x.pop(),lambda x:x[5].update(frame_index=6),
                       lambda x:x[5].update(timestamp_ns=0),
                       lambda x:x[5]["motion"]["motion_fit"].update(previous_frame_index=0),
                       lambda x:x[5]["coverage"].update(detection_ready=False),
                       lambda x:x[5]["motion"]["motion_backends"].update(cpu_fallback=True)):
            bad=copy.deepcopy(values);change(bad)
            with self.assertRaises(ValueError):runner.validate_rows(bad,attempts(values))

    def test_fit_acceptance_and_translation_must_be_consistent(self):
        values=rows()
        for change in (lambda m:m.update(accepted=False),lambda m:m["motion_fit"].update(quality_status="rejected"),
                       lambda m:m["motion_fit"].update(model="affine"),
                       lambda m:m["motion_fit"]["previous_to_current_matrix"][0].__setitem__(2,9),
                       lambda m:m["motion_fit"].update(parameters=None),
                       lambda m:m["correspondence_metrics"].update(accepted_count=59)):
            bad=copy.deepcopy(values);change(bad[1]["motion"])
            with self.assertRaises(ValueError):runner.validate_rows(bad,attempts(values))

    def test_expected_unavailable_retained_not_zero_fit(self):
        values=rows();values[1]=pair(1,accepted=False,failure=True)
        seen=attempts(values)
        runner.validate_rows(values,seen)
        seen[0]["error"]=None
        with self.assertRaises(ValueError):runner.validate_rows(values,seen)
        seen=attempts(values);seen[0]["expected_unavailable"]=False
        with self.assertRaises(ValueError):runner.validate_rows(values,seen)

    def test_failed_receipt_no_recovery_and_candidate_exact(self):
        for arm in ("baseline","candidate"):
            originals=artifacts(arm)
            runner.validate_artifacts(*originals,arm,"0170")
            for index,key,value in ((0,"passed",False),(0,"clip","0240"),(1,"passed",False),
                                    (2,"frames",672),(2,"full_clip",False),(3,"motion_config_sha256","wrong")):
                bad=copy.deepcopy(originals);bad[index][key]=value
                with self.assertRaises(ValueError):runner.validate_artifacts(*bad,arm,"0170")
        bad=artifacts("candidate")
        bad[0]["feature_adapter"]["effective_motion_configuration"]["quality_gate"]=29
        with self.assertRaises(ValueError):runner.validate_artifacts(*bad,"candidate","0170")

    def test_parse_nonfinite_rejected(self):
        for text in ('{"x":NaN}','{"x":Infinity}','{"x":-Infinity}'):
            with self.assertRaises(ValueError):runner.parse(text)

    def test_load_hashes_and_discards_unneeded_journal_arrays(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp)
            base=write_bundle(root,"baseline","0170")
            hashes={}
            result=runner.load_arm(root,"baseline","0170",hashes)
            self.assertEqual(len(hashes),5)
            self.assertEqual(len(result["rows"]),673)
            self.assertNotIn("tracks",result["rows"][0])
            self.assertNotIn("candidates",result["rows"][0])
            (base/"run/report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                runner.load_arm(root,"baseline","0170",{})

    def test_fixed_scope_exclusive_output_and_complete_run(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp)
            baseline,candidate=root/"baseline",root/"candidate"
            for clip in runner.CLIPS:
                write_bundle(baseline,"baseline",clip)
                write_bundle(candidate,"candidate",clip)
            output=root/"registration.json"
            with patch.object(runner,"BASELINE_ROOT",baseline),patch.object(runner,"CANDIDATE_ROOT",candidate), \
                    patch.object(runner,"OUTPUT_ROOT",root):
                with self.assertRaises(ValueError):runner.validate_scope(candidate,baseline,output)
                result=runner.run(baseline,candidate,output)
                self.assertTrue(result["passed_integrity"])
                self.assertEqual(list(result["clips"]),list(runner.CLIPS))
                self.assertFalse(result["new_acceptance_gate"])
                self.assertFalse(result["independent_geometry_truth"])
                self.assertFalse(result["media_read"])
                self.assertEqual(len(result["input_sha256"]),21)
                with self.assertRaises(ValueError):runner.run(baseline,candidate,output)
                linked=root/"linked.json";linked.symlink_to(root/"missing")
                with self.assertRaises(ValueError):runner.validate_scope(baseline,candidate,linked)


if __name__ == "__main__":
    unittest.main()
