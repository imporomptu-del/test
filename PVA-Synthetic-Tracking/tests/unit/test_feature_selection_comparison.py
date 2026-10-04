"""Generated metadata and frozen pure helpers only; no media or hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.unit import test_feature_supply_registration as fixture

PATH = Path(__file__).resolve().parents[2] / "scripts/compare_discovery_feature_selection.py"
SPEC = importlib.util.spec_from_file_location("selection_comparison_tests", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
comparison, registration = runner.helpers({})


def parity():
    rows=[]
    for name in runner.PARITY_NAMES:
        row=dict(name=name,passed=True,generated_only=True,selected_count=0 if name in ("empty","all_ineligible") else 384,
            maximum_selected_per_cell=0 if name in ("empty","all_ineligible") else 8,
            eligible_sha256="a"*64,selected_indices_sha256="b"*64)
        if name=="native_proxy_boundaries":
            row.update(proxy_size_wh=[2392,1595],native_source_shape_hw=[3190,4784],
                       source_pixels_accessed=False,selection_and_coverage_only=True)
        else:row["source_pixel_sha256"]="c"*64
        rows.append(row)
    return dict(passed=True,same_candidate_quota_in_both_paths=True,exact_reference_comparison=True,
        source_pixels_unchanged=True,points_and_scores_unchanged=True,score_precision_changed=False,
        numpy_version="1.26.1",max_features=384,max_features_per_cell=8,grid_rows=6,grid_cols=8,cases=rows)


def artifact_fixture(arm,clip="0170"):
    receipt,pre,report,launch=fixture.artifacts("baseline" if arm=="baseline" else "candidate",clip)
    receipt.update(error=None,cleanup_errors=[],native_mask_calls=0,clocks_changed=False,remote_clocks_unchanged=True,
        clock_policy_before={"fixed":"policy"},clock_policy_after={"fixed":"policy"},vpi_version="3.2.4",
        motion_instances=[dict(hits=671,misses=1,failed=False,closed=True)],
        gpu_fronts=[dict(calls=673,device_calls=673,finish_calls=673,host_calls=0,closed=True)],
        tracking=dict(geometry_fallbacks=0,batch_fallbacks=0,innovation_fallbacks=0,batch_track_rows=1,innovation_tracks=1))
    original=dict(harris_capacity_policy="legacy_default",feature_cpu_policy="reference",max_features=1000,
        max_features_per_cell=None,feature_image_scale=.5,grid_rows=6,grid_cols=8,quality_gate=30)
    global_cfg=dict(minimum_correspondences=30,minimum_inliers=30)
    if arm!="baseline":
        audit=receipt["feature_adapter"]
        effective=dict(original,**(runner.OVERRIDES if arm=="selection" else {"harris_capacity_policy":"complete_grid"}))
        audit.update(original_motion_configuration=original,effective_motion_configuration=effective,
            configuration_changes={k:dict(before=original[k],after=effective[k]) for k in original if original[k]!=effective[k]},
            change_classification={"generated":"declaration"},estimator_instances=1,successful_pair_backend_checks=672)
        receipt["global_configuration"]=global_cfg
        pre.update(feature_adapter=copy.deepcopy(audit),global_configuration=global_cfg,
            conversion=dict(passed=True,no_input_clipping=True),generated_pair_calls=3,
            controls=[dict(name=n,passed=True,closed=True) for n in runner.CONTROL_NAMES])
    if arm=="selection":
        receipt.update(schema="seaqr.discovery-feature-selection.v1",candidate=dict(runner.SELECTION),
            feature_quota_algorithm_changed=True,exact_cpu_execution_changed=True,harris_score_precision_changed=False)
        pre.update(schema=receipt["schema"]+".preflight",candidate=dict(runner.SELECTION),cpu_parity=parity())
        receipt["input_sha256"].update(freeze_sha256=runner.SELECTION_FREEZE_SHA,files={"generated.py":"a"*64})
    pre.update(detector_run=False,probe_passed=True,probe=launch["source_probe"],runtime_after={"numpy":"1.26.1"})
    for key in ("adapters","libraries","config_sha256","motion_config_sha256","v29_freeze_sha256","vpi_version"):
        pre[key]=receipt[key]
    report.update(counts=dict(motion_resets=0),timings_ms={"motion_and_warp":dict(mean=15,median=15,p95=15)})
    return receipt,pre,report,launch,original,global_cfg


def generated_rows():
    values=fixture.rows()
    for row in values:
        row.update(segment=0,candidates=[],tracks=[],tracking_metrics={"bright":{},"dark":{}})
        row["coverage"].update(dropped_at_tile_cap=0,dropped_at_frame_cap=0)
        if row["frame_index"]:
            corr=row["motion"]["correspondence_metrics"]
            corr.update(minimum_accepted_features=30,minimum_grid_coverage=.2)
            corr["harris_output"]["capacity_policy"]="complete_grid"
        if 430 <= row["frame_index"] <= 464:
            row["tracks"]=[dict(track_id="dark:fixture",segment=0,qualified_moving=True,measured=True,measurement_source_xy=[100,100])]
    return values


def bundle(root,arm,clip):
    base=root/clip;(base/"run").mkdir(parents=True)
    receipt,pre,report,launch,original,global_cfg=artifact_fixture(arm,clip)
    values=generated_rows();receipt["motion_attempts"]=fixture.attempts(values)
    for name,value in (("preflight.json",pre),("run/report.json",report),("run/launch.json",launch)):
        (base/name).write_text(json.dumps(value))
    (base/"run/frames.jsonl").write_text("".join(json.dumps(row)+"\n" for row in values))
    for key in ("preflight","report","launch","journal"):
        receipt[key+"_sha256"]=runner.sha(base/runner.FILES[key])
    (base/"execution_receipt.json").write_text(json.dumps(receipt))
    pins={k:runner.sha(base/f) for k,f in runner.FILES.items()}
    return pins,original,global_cfg


class ValidationTests(unittest.TestCase):
    def test_bundle_and_evidence_freeze_and_every_transferred_file_bound(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp);package=root/"bundle";evidence=root/"evidence"
            package.mkdir();evidence.mkdir()
            names=("batch_discovery_pair.py","batch_feature_selection.py","feature_selection_plan.json",
                   "run_discovery_feature_selection.py","test_discovery_feature_selection.py")
            for name in names:(package/name).write_text("generated fixture")
            frozen=dict(schema="feature_selection.v1",candidate=dict(runner.SELECTION),
                sources={clip:registration.source_spec(clip) for clip in runner.CLIPS},
                baseline_workspace="/tmp/seaqr_discovery_pair_20260928_ZGLHH7",
                files={name:runner.sha(package/name) for name in names})
            content=json.dumps(frozen)
            (package/"freeze.json").write_text(content);(evidence/"freeze.json").write_text(content)
            digest=runner.sha(package/"freeze.json")
            with patch.object(runner,"SELECTION_BUNDLE",package),patch.object(runner,"SELECTION_FREEZE_SHA",digest), \
                    patch.object(runner,"SELECTION_RUNNER_SHA",frozen["files"]["run_discovery_feature_selection.py"]):
                hashes={}
                self.assertEqual(runner.load_selection_bundle(evidence,registration,hashes),frozen)
                self.assertEqual(len(hashes),7)
                (package/"feature_selection_plan.json").write_text("changed")
                with self.assertRaises(ValueError):runner.load_selection_bundle(evidence,registration,{})
                (package/"feature_selection_plan.json").write_text("generated fixture")
                (evidence/"freeze.json").write_text(content+"\n")
                with self.assertRaises(ValueError):runner.load_selection_bundle(evidence,registration,{})

    def test_selection_bundle_bindings_and_strict_json(self):
        receipt,pre,*_=artifact_fixture("selection")
        freeze=dict(files=receipt["input_sha256"]["files"])
        runner.validate_bundle_binding(receipt,pre,freeze)
        for change in (lambda r,p:r["input_sha256"].update(freeze_sha256="wrong"),
                       lambda r,p:p["input_sha256"].update(files={})):
            r,p=copy.deepcopy((receipt,pre));change(r,p)
            with self.assertRaises(ValueError):runner.validate_bundle_binding(r,p,freeze)
        with self.assertRaises(ValueError):runner.validate_bundle_binding(receipt,pre,None)
        for text in ('{"x":1,"x":2}','{"x":1e309}','{"x":NaN}'):
            with self.assertRaises(ValueError):runner.decode(text)

    def test_target_track_segments_and_flags_not_coerced(self):
        selected=[dict(frame_index=r["frame_index"],segment=r["segment"],detection_ready=r["coverage"]["detection_ready"],tracks=r["tracks"])
                  for r in generated_rows()[430:465]]
        runner.validate_target_rows(selected)
        for change in (lambda x:x[0]["tracks"][0].update(segment=1),
                       lambda x:x[0]["tracks"][0].update(measured=1),
                       lambda x:x[0]["tracks"][0].update(qualified_moving="true"),
                       lambda x:x[0]["tracks"][0].update(measurement_source_xy=[True,100])):
            bad=copy.deepcopy(selected);change(bad)
            with self.assertRaises(ValueError):runner.validate_target_rows(bad)

    def test_exact_cpu_parity_all_five_cases(self):
        value=parity();runner.validate_cpu_parity(value,"1.26.1")
        for change in (lambda x:x.update(passed=False),lambda x:x.update(score_precision_changed=True),
                       lambda x:x.update(numpy_version="different"),lambda x:x.update(max_features=385),
                       lambda x:x["cases"].pop(),lambda x:x["cases"][-1].update(proxy_size_wh=[320,256]),
                       lambda x:x["cases"][-1].update(source_pixels_accessed=True),
                       lambda x:x["cases"][0].update(selected_count=385),
                       lambda x:x["cases"][0].update(maximum_selected_per_cell=9),
                       lambda x:x["cases"][0].update(selected_indices_sha256="bad")):
            bad=copy.deepcopy(value);change(bad)
            with self.assertRaises(ValueError):runner.validate_cpu_parity(bad,"1.26.1")

    def test_exact_four_overrides_and_preflight_identity(self):
        receipt,pre,report,launch,original,global_cfg=artifact_fixture("selection")
        receipt["motion_attempts"]=fixture.attempts(fixture.rows())
        runner.validate_selection(receipt,pre,original,global_cfg,registration)
        for change in (lambda r,p:r.update(global_motion_gates_changed=True),
                       lambda r,p:r["feature_adapter"]["effective_motion_configuration"].update(quality_gate=29),
                       lambda r,p:r["feature_adapter"]["configuration_changes"].pop("max_features"),
                       lambda r,p:p["feature_adapter"]["source_transformation"].update(exact_original_recovered=False),
                       lambda r,p:p["controls"][0].update(closed=False),
                       lambda r,p:p["cpu_parity"]["cases"].pop(),
                       lambda r,p:r.update(feature_quota_algorithm_changed=False)):
            r,p=copy.deepcopy((receipt,pre));change(r,p)
            with self.assertRaises(ValueError):runner.validate_selection(r,p,original,global_cfg,registration)

    def test_lifecycle_rejects_all_fallback_and_cleanup_paths(self):
        receipt=artifact_fixture("selection")[0];runner.validate_lifecycle(receipt)
        for change in (lambda r:r.update(cleanup_errors=["bad"]),lambda r:r.update(native_mask_calls=1),
                       lambda r:r.update(clocks_changed=True),lambda r:r["motion_instances"][0].update(failed=True),
                       lambda r:r["gpu_fronts"][0].update(host_calls=1),
                       lambda r:r["tracking"].update(batch_fallbacks=1),
                       lambda r:r["tracking"].update(innovation_tracks=0)):
            bad=copy.deepcopy(receipt);change(bad)
            with self.assertRaises(ValueError):runner.validate_lifecycle(bad)

    def test_scope_forbids_media_and_historical_pin_omission(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            media=Path(temp)/"test.avi";media.write_bytes(b"generated")
            with self.assertRaises(ValueError):runner.sha(media)
            with self.assertRaises(ValueError):runner.load_arm(Path(temp),"baseline","0170",None,{},comparison,registration)
            with self.assertRaises(ValueError):runner.load_arm(Path(temp),"selection","0126",None,{},comparison,registration)

    def test_failed_receipt_and_hash_mismatch_are_not_recovered(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp);pins,_,_=bundle(root,"baseline","0170")
            badpins=dict(pins,journal="a"*64)
            with self.assertRaisesRegex(ValueError,"identity"):
                runner.load_arm(root,"baseline","0170",badpins,{},comparison,registration)
            p=root/"0170/execution_receipt.json";r=json.loads(p.read_text());r["passed"]=False;p.write_text(json.dumps(r))
            pins["execution_receipt"]=runner.sha(p)
            with self.assertRaisesRegex(ValueError,"failed/incomplete"):
                runner.load_arm(root,"baseline","0170",pins,{},comparison,registration)


class SummaryTests(unittest.TestCase):
    def test_workload_null_and_distinct_denominators(self):
        value=dict(counts=dict(qualified_measured_states=10,qualified_predicted_states=6,ready_frames=2))
        result=runner.workload(value)["workload"]
        self.assertEqual(result["total_states"],16);self.assertEqual(result["total_per_ready"],8)
        value["counts"]["ready_frames"]=0
        self.assertIsNone(runner.workload(value)["workload"]["total_per_ready"])

    def test_coherent_retention_requires_actual_qualified_dark_ready(self):
        references=[dict(frame_index=i,measurement_source_xy=[100,100]) for i in range(430,465)]
        selected=[dict(frame_index=i,segment=0,detection_ready=True,
            tracks=[dict(track_id="dark:new",qualified_moving=True,measured=True,measurement_source_xy=[108,100])])
                  for i in range(430,465)]
        self.assertTrue(comparison.retention(selected,references)["preservation_guard_passed"])
        for change in (lambda x:x[0].update(detection_ready=False),lambda x:x[0]["tracks"][0].update(measured=False),
                       lambda x:x[0]["tracks"][0].update(qualified_moving=False),
                       lambda x:x[0]["tracks"][0].update(track_id="bright:new"),
                       lambda x:x[0]["tracks"][0].update(track_id="dark:split"),
                       lambda x:x[0]["tracks"][0].update(measurement_source_xy=[108.01,100])):
            bad=copy.deepcopy(selected);change(bad)
            self.assertFalse(comparison.retention(bad,references)["preservation_guard_passed"])

    def test_decision_all_three_requirements_and_never_promotion(self):
        retention=dict(reference_frames=35,best_coherent_identity_frames=35,preservation_guard_passed=True)
        clips={c:dict(selection=dict(ready_fraction=.95,counts=dict(pva_runtime_errors=0),positive_pass=retention)) for c in runner.CLIPS}
        self.assertTrue(runner.decision(clips)["declared_development_gates_passed"])
        self.assertFalse(runner.decision(clips)["promotion_allowed"])
        for change in (lambda x:x["0170"]["selection"].update(ready_fraction=.949),
                       lambda x:x["0240"]["selection"]["counts"].update(pva_runtime_errors=1),
                       lambda x:x["0240"]["selection"]["positive_pass"].update(best_coherent_identity_frames=34)):
            bad=copy.deepcopy(clips);change(bad)
            self.assertFalse(runner.decision(bad)["declared_development_gates_passed"])

    def test_load_all_three_generated_arms_and_old_collect_parity(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            for arm in runner.ROOTS:
                root=Path(temp)/arm;pins,original,global_cfg=bundle(root,arm,"0240")
                data=runner.load_arm(root,arm,"0240",None if arm=="selection" else pins,{},comparison,registration,original,global_cfg,
                    {"files":{"generated.py":"a"*64}} if arm=="selection" else None)
                old,selected=comparison.collect(root/"0240/run/frames.jsonl")
                self.assertEqual({k:data["full"][k] for k in old},old);self.assertEqual(data["selected"],selected)
                self.assertEqual(data["registration"]["feature_supply"]["selected_count"]["median"],100)
                self.assertEqual(len(data["selected"]),35)
                self.assertEqual(data["full"]["counts"]["qualified_measured_states"],35)


if __name__ == "__main__":
    unittest.main()
