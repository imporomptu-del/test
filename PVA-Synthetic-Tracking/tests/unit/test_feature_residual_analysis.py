"""Synthetic accepted-point arrays only; no source images, VPI or model fit."""
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

PATH = Path(__file__).resolve().parents[2] / "scripts/analyze_feature_residual_trace.py"
SPEC = importlib.util.spec_from_file_location("residual_trace_analysis_tests", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def points():
    p = np.array([[100+i*10,200] for i in range(12)],np.float32)
    q = p + np.array([[v,0] for v in (.125,.25,.5,.75,1,1.5,.125,.25,.5,.75,1,1.5)],np.float32)
    score = np.array([1,1,2,3,5,8]*2,np.float32)
    fb = np.arange(12,dtype=np.float32)/10
    residual = np.linalg.norm(q.astype(np.float64)-p,axis=1)
    return p,q,score,fb,residual<=1,residual,dict(translation_x_px=0,translation_y_px=0)


def record(array):
    array=np.ascontiguousarray(array)
    raw=array.tobytes()
    return dict(dtype=array.dtype.str,shape=list(array.shape),data_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())


def trace_row(index=5):
    p,q,s,f,m,r,t=points()
    matrix=np.eye(3,dtype=np.float64)
    metrics=dict(correspondence_count=len(p),inlier_count=int(m.sum()),inlier_ratio=float(m.mean()),
        median_reprojection_error_px=float(np.median(r[m])),p90_reprojection_error_px=float(np.percentile(r[m],90)),
        maximum_reprojection_error_px=float(r[m].max()))
    corr_metrics=dict(accepted_count=len(p),rejections={"forward_backward":26})
    fit=dict(model="translation",quality_status="rejected",rejection_reasons=["high_median_reprojection_error"],parameters=t,
        metrics=metrics,previous_to_current_matrix=record(matrix),inlier_mask=record(m),residuals_px=record(r))
    corr=dict(previous_points=record(p),current_points=record(q),harris_scores=record(s),forward_backward_error_px=record(f),
              metrics=corr_metrics,backends={"optical_flow_pyrlk":"PVA"})
    row=dict(previous_frame_index=index-1,current_frame_index=index,previous_timestamp_ns=(index-1)*100_000_000,
        current_timestamp_ns=index*100_000_000,full_image_size=[4784,3190],motion_image_size=[2392,1595],
        native_gray_pixel_sha256=dict(previous="a"*64,current="b"*64),correspondence=corr,fit=fit)
    saved={k:fit[k] for k in ("model","quality_status","rejection_reasons","parameters","metrics")}
    saved["previous_to_current_matrix"]=matrix.tolist()
    motion=dict(motion_fit=saved,correspondence_metrics=corr_metrics,motion_backends=corr["backends"])
    return row,motion


def metadata_bundle(root):
    bundle,evidence,original=root/"bundle",root/"evidence",root/"original"
    bundle.mkdir();evidence.mkdir();original.mkdir()
    artifacts={}
    for clip in analysis.CLIPS:
        folder=original/clip;(folder/"run").mkdir(parents=True)
        rows=[]
        for index in range(673):
            _,motion=trace_row(max(1,index))
            rows.append(dict(frame_index=index,timestamp_ns=index*100_000_000,motion=motion,coverage={},timings_ms={}))
        (folder/"run/frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        for name in ("preflight.json","run/report.json","run/launch.json"):(folder/name).write_text("{}")
        (folder/"execution_receipt.json").write_text(json.dumps(dict(passed=True,processed_frames=673,input_sha256={"clip":clip})))
        artifacts[clip]={k:analysis.sha(folder/f) for k,f in analysis.ARTIFACT_FILES.items()}
    sources={clip:dict(sha256=analysis.SOURCE_SHA[clip],frames=673,width=4784,height=3190,fps=10) for clip in analysis.CLIPS}
    pairs={"0170":list(range(1,57)),"0240":list(range(1,45))}
    groups={clip:dict(failed=[1],adjacent=[2],temporal=pairs[clip][2:],positive=[]) for clip in analysis.CLIPS}
    plan=dict(schema="seaqr.feature-residual-trace.plan.v1",sources=sources,candidate=analysis.CANDIDATE,
        original_freeze_sha256=analysis.ORIGINAL_FREEZE_SHA,original_workspace="/tmp/seaqr_feature_selection_20260929_q4iI5B",
        original_artifacts=artifacts,parity=dict(ignored_exact_paths=analysis.IGNORED_PATHS),
        capture_pair_counts={"0170":56,"0240":44},capture_pairs=pairs,capture_groups=groups)
    (bundle/"feature_residual_trace_plan.json").write_text(json.dumps(plan))
    names=("run_feature_residual_trace.py","batch_feature_residual_trace.py","batch_discovery_pair.py")
    for name in names:(bundle/name).write_text("generated fixture")
    files={name:analysis.sha(bundle/name) for name in (*names,"feature_residual_trace_plan.json")}
    freeze=dict(schema="feature_residual_trace.v1",candidate=analysis.CANDIDATE,sources=sources,files=files,
        original_freeze_sha256=analysis.ORIGINAL_FREEZE_SHA,original_workspace=plan["original_workspace"])
    for folder in (bundle,evidence):(folder/"freeze.json").write_text(json.dumps(freeze))
    digest=analysis.sha(bundle/"freeze.json")
    receipts={}
    for clip in analysis.CLIPS:
        folder=evidence/clip;(folder/"run").mkdir(parents=True)
        bound=dict(files=files,freeze_sha256=digest,plan_sha256=files["feature_residual_trace_plan.json"],
            original_artifacts=artifacts,original_workspace=plan["original_workspace"],original_freeze_sha256=analysis.ORIGINAL_FREEZE_SHA,
            original_runner_sha256=analysis.ORIGINAL_RUNNER_SHA,selection_inputs={"clip":clip})
        pre=dict(schema="seaqr.discovery-feature-residual-trace.v1.preflight",passed=True,clip=clip,detector_run=False,input_sha256=bound,
            passive_capture_check=dict(passed=True,generated_only=True,original_fit_calls=1,same_return_object=True,
                input_arrays_unchanged=True,exact_array_byte_serialization=True,native_gray_hashes=True))
        trace=dict(schema="seaqr.feature-residual-trace.v1.trace",clip=clip,source=sources[clip],candidate=analysis.CANDIDATE,
            input_sha256=bound,capture_pairs=pairs[clip],capture_groups=groups[clip],original_fit_calls=672,full_causal_history=True,
            captured_pairs=len(pairs[clip]),rows=[trace_row(i)[0] for i in pairs[clip]])
        (folder/"preflight.json").write_text(json.dumps(pre));(folder/"trace.json").write_text(json.dumps(trace))
        (folder/"run/frames.jsonl").write_text((original/clip/"run/frames.jsonl").read_text())
        (folder/"run/report.json").write_text(json.dumps(dict(completed=True,full_clip=True,frames=673,source_sha256=analysis.SOURCE_SHA[clip])))
        (folder/"run/launch.json").write_text(json.dumps(dict(source_sha256=analysis.SOURCE_SHA[clip])))
        parity=dict(schema="seaqr.feature-residual-trace.v1.parity",passed=True,rows_compared=673,mismatch_frames=[],
            excluded_paths=analysis.IGNORED_PATHS,original_journal_sha256=artifacts[clip]["journal"],
            diagnostic_journal_sha256=analysis.sha(folder/"run/frames.jsonl"))
        (folder/"parity.json").write_text(json.dumps(parity))
        receipt=dict(schema="seaqr.discovery-feature-residual-trace.v1",passed=True,error=None,processed_frames=673,
            clip=clip,source=sources[clip],candidate=analysis.CANDIDATE,input_sha256=bound,
            non_timing_journal_parity_passed=True,original_fit_calls=672,captured_pairs=len(pairs[clip]),full_causal_history=True)
        for flag in ("candidate_algorithm_changed_relative_to_original","global_motion_gates_changed","detector_configuration_changed",
                     "tracker_configuration_changed","extra_vpi_readbacks","estimator_method_source_changed"):receipt[flag]=False
        for key,name in ("preflight","preflight.json"),("trace","trace.json"),("parity","parity.json"),("journal","run/frames.jsonl"),("report","run/report.json"),("launch","run/launch.json"):
            receipt[key+"_sha256"]=analysis.sha(folder/name)
        receipts[clip]=receipt
    for clip,receipt in receipts.items():
        receipt["both_preflight_sha256"]={c:analysis.sha(evidence/c/"preflight.json") for c in analysis.CLIPS}
        (evidence/clip/"execution_receipt.json").write_text(json.dumps(receipt))
    return bundle,evidence,original,digest


class NumericalTests(unittest.TestCase):
    def test_missing_distribution_is_not_zero(self):
        result=analysis.distribution([np.nan,np.inf,1,3])
        self.assertEqual((result["finite"],result["unavailable"]),(2,2))
        self.assertEqual(result["median"],2)
        self.assertIsNone(analysis.distribution([np.nan])["median"])
        self.assertIsNone(analysis.distribution([])["mean"])

    def test_tied_ranks_and_descriptive_associations(self):
        np.testing.assert_array_equal(analysis.midranks([3,1,1,2]),[3,.5,.5,2])
        self.assertAlmostEqual(analysis.rank_association([1,1,3],[9,9,10])["spearman"],1)
        self.assertAlmostEqual(analysis.rank_association([1,2,3],[3,2,1])["spearman"],-1)
        self.assertIsNone(analysis.rank_association([1,1],[2,3])["spearman"])
        self.assertIsNone(analysis.rank_association([np.nan],[2])["spearman"])
        self.assertIsNone(analysis.rank_association([1,2],[2,3])["population_p_value"])

    def test_native_grid_boundaries_and_out_of_bounds(self):
        w,h=analysis.IMAGE_SIZE_WH
        np.testing.assert_array_equal(analysis.cell_ids([[0,0],[w/8,h/6],[w-1,h-1]]),[0,9,47])
        for p in ([[-1,0]],[[w,0]],[[0,h]],[[np.nan,2]]):
            with self.assertRaises(ValueError):analysis.cell_ids(p)

    def test_saved_fit_populations_and_fixed_thresholds(self):
        arrays=points();result=analysis.point_analysis(*arrays)
        self.assertEqual(result["points"],12)
        self.assertEqual(result["populations"]["original_ransac_inliers"]["points"],10)
        self.assertEqual(result["populations"]["original_ransac_outliers"]["points"],2)
        self.assertEqual(result["populations"]["original_ransac_inliers"]["saved_residual_norm_px"]["median"],.5)
        self.assertEqual(result["populations"]["all_lk_accepted"]["saved_residual_norm_px"]["median"],.625)
        self.assertEqual(result["spatial"]["descriptive_point_tail_concentration"]["1.0"]["point_count"],2)
        self.assertEqual(result["spatial"]["descriptive_point_tail_concentration"]["1.0"]["maximum_cell_share"],1)
        self.assertEqual(sum(row["points"] for row in result["harris_rank"]["bins"]),12)

    def test_no_fit_remains_unavailable_not_outlier_or_identity(self):
        p,q,s,f,m,r,t=points()
        result=analysis.point_analysis(p,q,s,f,np.zeros(12,bool),np.full(12,np.nan),None)
        self.assertEqual(result["populations"]["unavailable_original_residual"]["points"],12)
        self.assertEqual(result["populations"]["original_ransac_outliers"]["points"],0)
        self.assertIsNone(result["populations"]["all_lk_accepted"]["residual_vector"]["component_median_xy"])
        self.assertIsNone(result["associations"]["fb_vs_residual"]["spearman"])
        self.assertIsNone(result["spatial"]["descriptive_point_tail_concentration"]["1.0"]["maximum_cell_share"])

    def test_identity_residual_check_does_not_refit_bad_saved_model(self):
        p,q,s,f,m,r,t=points()
        with self.assertRaisesRegex(ValueError,"residual identity"):
            analysis.point_analysis(p,q,s,f,m,r,dict(translation_x_px=.25,translation_y_px=0))
        wrong=m.copy();wrong[0]=False
        with self.assertRaisesRegex(ValueError,"mask"):
            analysis.point_analysis(p,q,s,f,wrong,r,t)
        with self.assertRaisesRegex(ValueError,"undefined fit"):
            analysis.point_analysis(p,q,s,f,m,r,None)

    def test_rank_ties_preserve_empty_bins(self):
        p,q,s,f,m,r,t=points();s[:]=2**24
        result=analysis.point_analysis(p,q,s,f,m,r,t)
        self.assertEqual([b["points"] for b in result["harris_rank"]["bins"]],[0,0,12,0])
        self.assertTrue(result["harris_rank"]["original_score_precision_not_recovered"])

    def test_cell_vector_low_support_unavailable_and_source_immutable(self):
        p,q,s,f,m,r,t=points();p[0]=[4700,3000];q[0]=p[0]+[.125,0]
        before=[a.copy() for a in (p,q,s,f,m,r)]
        result=analysis.point_analysis(p,q,s,f,m,r,t)
        cell=result["spatial"]["cells"][47]
        self.assertEqual(cell["points"],1)
        self.assertIsNone(cell["displacement_median_xy"])
        self.assertIsNone(cell["residual_vector_median_xy"])
        for a,b in zip((p,q,s,f,m,r),before):np.testing.assert_array_equal(a,b)

    def test_empty_population_and_nonfinite_accepted_inputs(self):
        result=analysis.point_analysis(np.empty((0,2)),np.empty((0,2)),np.empty(0),np.empty(0),
                                      np.empty(0,bool),np.empty(0),None)
        self.assertEqual(result["points"],0)
        p,q,s,f,m,r,t=points();f[0]=np.nan
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            analysis.point_analysis(p,q,s,f,m,r,t)


class IntegrityTests(unittest.TestCase):
    def test_complete_loader_and_analysis_bind_all100_captures(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp);bundle,evidence,original,digest=metadata_bundle(root)
            with patch.object(analysis,"OUTPUT_ROOT",root),patch.object(analysis,"ORIGINAL_ROOT",original):
                result=analysis.run(evidence,bundle,digest,root/"analysis.json")
                self.assertTrue(result["passed_integrity"])
                self.assertEqual([len(result["clips"][c]["rows"]) for c in analysis.CLIPS],[56,44])
                self.assertEqual(result["plan_sha256"],analysis.sha(bundle/"feature_residual_trace_plan.json"))
                self.assertIn(str(evidence/"0170/trace.json"),result["input_sha256"])
                self.assertFalse(result["source_media_read"]);self.assertFalse(result["new_fit_performed"])
                with self.assertRaisesRegex(ValueError,"hash"):
                    analysis.verify_inputs(evidence,bundle,"a"*64)
                trace=evidence/"0170/trace.json";trace.write_text(trace.read_text()+"\n")
                with self.assertRaisesRegex(ValueError,"binding"):
                    analysis.verify_inputs(evidence,bundle,digest)

    def test_exact_array_bytes_including_nan_payload_and_negative_zero(self):
        raw=np.array([0x8000000000000000,0x7ff8000000000001],np.uint64).view(np.float64)
        saved=record(raw);actual=analysis.decode_array(saved,"<f8",(2,))
        self.assertEqual(actual.tobytes(),raw.tobytes())
        for change in (lambda r:r.update(sha256="a"*64),lambda r:r.update(dtype="<f4"),
                       lambda r:r.update(shape=[True]),lambda r:r.update(data_base64="not valid!")):
            bad=copy.deepcopy(saved);change(bad)
            with self.assertRaises(ValueError):analysis.decode_array(bad,"<f8",(2,))
        boolean=record(np.array([True],bool));boolean.update(data_base64=base64.b64encode(b"\x02").decode(),sha256=hashlib.sha256(b"\x02").hexdigest())
        with self.assertRaisesRegex(ValueError,"boolean"):analysis.decode_array(boolean,"|b1",(1,))

    def test_unpack_matches_exact_captured_arrays_and_rejects_shape(self):
        row,_=trace_row()
        arrays=analysis.unpack_row(row,"0170")
        for actual,expected in zip(arrays,points()[:6]):np.testing.assert_array_equal(actual,expected)
        for change in (lambda r:r.update(previous_frame_index=1),lambda r:r.update(motion_image_size=[320,256]),
                       lambda r:r["correspondence"]["previous_points"].update(shape=[385,2]),
                       lambda r:r["native_gray_pixel_sha256"].update(current="wrong")):
            bad=copy.deepcopy(row);change(bad)
            with self.assertRaises(ValueError):analysis.unpack_row(bad,"0170")

    def test_original_journal_and_saved_statistic_identity(self):
        row,motion=trace_row()
        result=analysis.analyze_row(row,"0170",motion,dict(failed=[5],adjacent=[],temporal=[5],positive=[]))
        self.assertEqual(result["groups"],["failed","temporal"])
        self.assertTrue(result["original_fit_statistics_reproduced"])
        self.assertFalse(result["raw_lk_filtered_coordinates_available"])
        self.assertEqual(result["original_correspondence_metrics"]["rejections"]["forward_backward"],26)
        for key in ("median_reprojection_error_px","p90_reprojection_error_px","maximum_reprojection_error_px"):
            bad,saved=copy.deepcopy((row,motion));bad["fit"]["metrics"][key]+=1;saved["motion_fit"]["metrics"][key]+=1
            with self.assertRaisesRegex(ValueError,"statistic"):analysis.analyze_row(bad,"0170",saved,{})
        bad=copy.deepcopy(motion);bad["motion_fit"]["quality_status"]="accepted"
        with self.assertRaisesRegex(ValueError,"journal"):analysis.analyze_row(row,"0170",bad,{})

    def test_group_overlap_and_missing_groups_are_explicit(self):
        row,motion=trace_row()
        result=analysis.analyze_row(row,"0170",motion,{})
        group=analysis.group_summary([result],[5])
        self.assertEqual(group["selected_pairs"],1)
        self.assertEqual(group["populations"]["original_ransac_outliers"]["point_total"],2)
        empty=analysis.group_summary([result],[])
        self.assertEqual(empty["selected_pairs"],0)
        self.assertIsNone(empty["populations"]["all_lk_accepted"]["per_pair_residual_medians"]["median"])
        with self.assertRaises(ValueError):analysis.group_summary([result],[5,5])

    def test_full673_nontiming_equality_exact_paths_only(self):
        rows=[dict(frame_index=i,timestamp_ns=i*100_000_000,motion={"pva_timings_ms":{"x":1},
                   "motion_fit":{"timing_ms":1,"metrics":{"kept":2}},"warp_timings_ms":{"x":1}},
                   timings_ms={"x":1},coverage={"detection_ms":1},tracks=[{"actual":True}]) for i in range(673)]
        changed=copy.deepcopy(rows)
        changed[5]["motion"]["motion_fit"]["timing_ms"]=99
        changed[6]["coverage"]["detection_ms"]=99
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            old,new=Path(temp)/"old.jsonl",Path(temp)/"new.jsonl"
            old.write_text("".join(json.dumps(r)+"\n" for r in rows))
            new.write_text("".join(json.dumps(r)+"\n" for r in changed))
            self.assertEqual(sorted(analysis.verify_full_parity(old,new,[5,6])),[5,6])
            changed[6]["motion"]["motion_fit"]["metrics"]["kept"]=3
            new.write_text("".join(json.dumps(r)+"\n" for r in changed))
            with self.assertRaisesRegex(ValueError,"non-timing"):analysis.verify_full_parity(old,new,[5,6])
            new.write_text("".join(json.dumps(r)+"\n" for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError,"inventory"):analysis.verify_full_parity(old,new,[5,6])

    def test_strict_json_metadata_only_and_output_nooverwrite(self):
        for text in ('{"x":1,"x":2}','{"x":1e309}','{"x":NaN}'):
            with self.assertRaises(ValueError):analysis.decode_json(text)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            root=Path(temp);media=root/"forbidden.avi";media.write_bytes(b"generated")
            with self.assertRaises(ValueError):analysis.sha(media)
            output=root/"analysis.json";output.write_text("existing")
            with patch.object(analysis,"OUTPUT_ROOT",root),patch.object(analysis,"verify_inputs") as verify:
                with self.assertRaisesRegex(ValueError,"fresh"):analysis.run(root,root,"a"*64,output)
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
