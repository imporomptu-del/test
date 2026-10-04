"""Synthetic metadata tests. No source pixels, tracker execution or remote calls."""
from contextlib import contextmanager
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


M=imported("maturity_summary",ROOT/"scripts/summarize_tracker_maturity_shadow_v1.py")
S=imported("maturity_pure_retention",ROOT/"scripts/compare_discovery_feature_supply.py")


def write(path,value):path.write_text(json.dumps(value,allow_nan=False))
def write_rows(path,values):path.write_text("".join(json.dumps(v,allow_nan=False)+"\n" for v in values))


def track(frame,identity="dark:1",hits=5):
    return dict(track_id=identity,segment=0,independent_hits=hits,measured=True,qualified_moving=hits>=4,
        reference_xy=[200.+frame,300.],measurement_source_xy=[200.+frame,300.],lifecycle="confirmed" if hits>=4 else "tentative")


def generated(count=673,lost=False):
    archive=[];candidate=[];audits=[];states=[]
    workloads={arm:M.Workload() for arm in ("baseline","candidate")}
    for i in range(count):
        records=[track(i)]
        def metrics(tracks):
            return {p:dict(active_track_count=sum(t["track_id"].startswith(p+":") for t in tracks),max_active_tracks=256,
                birth_count=int(i==0 and p=="dark"),deleted_track_count=0,dropped_birth_count_at_active_track_cap=0,
                lifecycle_counts={state:sum(t["track_id"].startswith(p+":") and t["lifecycle"]==state for t in tracks) for state in ("tentative","confirmed","coasted")},
                birth_admission=dict(confirmed_tracks_evicted=0,tentative_replacements=[])) for p in ("bright","dark")}
        row=dict(frame_index=i,timestamp_ns=i*100000000,segment=0,source_to_reference=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
            coverage=dict(full_shape_hw=[3190,4784],detection_ready=i>=5,warmup=i<5),candidates=[dict(x=3.,y=4.,score=5.)],
            tracks=records,tracking_metrics=metrics(records),motion=dict(unchanged=True),timings_ms=dict(archived=1.))
        new=copy.deepcopy(row)
        if 50<=i<=54:new["tracks"].append(track(i,"bright:2",1))
        if lost and i==440:new["tracks"]=[]
        new["tracking_metrics"]=metrics(new["tracks"])
        hashes=[M.value_sha({k:r[k] for k in ("tracks","tracking_metrics")}) for r in (row,new)]
        internal=["b"*64, "c"*64 if i>=50 else "b"*64];learning=["d"*64,"e"*64 if i>=52 else "d"*64]
        state=dict(frame_index=i,differences={},output_sha256=[hashes[0]]*2,archive_output_sha256=hashes[0],
            internal_state_sha256=[internal[0]]*2,learning_centers_sha256=[learning[0]]*2,archive_derived_learning_sha256=learning[0])
        audit=dict(frame_index=i,output_sha256=[hashes[0],hashes[1],hashes[1]],internal_state_sha256=[internal[0],internal[1],internal[1]],
            learning_centers_sha256=[learning[0],learning[1],learning[1]],past_or_at_learning_divergence=i>=52,
            workload={arm:workloads[arm].add(r) for arm,r in (("baseline",row),("candidate",new))})
        new["shadow"]=dict(saved_proposals=True,candidate_policy="maturity_first_eligible_v1",
            detector_coverage_motion_and_timings_are_archived_not_rerun=True,past_or_at_learning_divergence=i>=52)
        archive.append(row);candidate.append(new);audits.append(audit);states.append(state)
    return archive,candidate,audits,states


@contextmanager
def fixture(root,lost=False):
    root=Path(root).resolve();bundle=root/"bundle";directory=root/"jetson";baseline=root/"baseline";archive_root=root/"archive"
    for p in (bundle,directory,baseline,archive_root):p.mkdir()
    for name in M.FILES:
        actual=ROOT/"scripts"/name
        if name in {"tracker_maturity_candidate_v1.py","compare_discovery_feature_supply.py","batch_discovery_pair.py"}:
            (bundle/name).write_bytes(actual.read_bytes())
        else:(bundle/name).write_text("synthetic frozen source fixture")
    refs=[dict(frame_index=i,measurement_source_xy=[200.+i,300.]) for i in range(430,465)]
    regression=dict(scope=dict(allowed_clip_ids=list(M.CLIPS)),positive_pass=dict(clip_id="0240",baseline_measurements=refs))
    write(bundle/"discovery_pair_regression_20260929.json",regression)
    write(baseline/"freeze.json",dict(synthetic_old_freeze=True));old_freeze=M.sha(baseline/"freeze.json")
    original_files={};old_results={};old_states={};old_rows={}
    before=dict(blas=[dict(threads=12)],affinity=[0],numpy="1.26.1",opencv="4.10.0",opencv_threads=12,thread_environment={},clock_ticks=100)
    after=dict(before,opencv_threads=2)
    for clip in M.CLIPS:
        (baseline/clip).mkdir();(archive_root/clip/"run").mkdir(parents=True);(directory/clip).mkdir()
        rows=generated(lost=lost and clip=="0240");old_rows[clip]=rows
        write_rows(archive_root/clip/"run/frames.jsonl",rows[0]);write_rows(baseline/clip/(clip+"_state_hashes.jsonl"),rows[3])
        write(archive_root/clip/"run/launch.json",{});write(archive_root/clip/"execution_receipt.json",{})
        original_files[clip]={name:M.sha(archive_root/clip/name) for name in M.ARCHIVE_FILES[clip]}
        old_states[clip]=M.sha(baseline/clip/(clip+"_state_hashes.jsonl"))
    original_inputs={M.TRACE+"/"+clip+"/"+name:h for clip,files in original_files.items() for name,h in files.items()}
    for clip in M.CLIPS:
        replay=dict(passed=True,first_difference=None,state_hashes_sha256=old_states[clip],
            **{k:673 for k in ("attempted_frames","exact_archive_frames","dual_state_exact_frames","derived_learning_exact_frames")})
        result=dict(schema="seaqr.tracker-baseline-jetson.v1.run",passed=True,error=None,workspace=M.OLD,freeze_sha256=old_freeze,
            clip=clip,inputs_unchanged_after_check=True,clock_controls_unchanged=True,replay=replay,
            inputs_sha256=original_inputs,runtime_before=before,runtime_after=after)
        write(baseline/clip/"result.json",result);old_results[clip]=M.sha(baseline/clip/"result.json")
    with patch.multiple(M,OLD_FREEZE_SHA=old_freeze,OLD_RESULTS=old_results,OLD_STATES=old_states,ARCHIVE_FILES=original_files,
                        REGRESSION_SHA=M.sha(bundle/"discovery_pair_regression_20260929.json")):
        freeze=dict(schema="tracker_maturity_shadow.v1",protocol=M.protocol(),files={n:M.sha(bundle/n) for n in M.FILES})
        write(bundle/"freeze.json",freeze);write(directory/"freeze.json",freeze);digest=M.sha(bundle/"freeze.json")
        workspace="/tmp/seaqr_tracker_maturity_20261001_Ab1234"
        inputs={workspace+"/freeze.json":digest,**{workspace+"/"+n:h for n,h in freeze["files"].items()},
            M.OLD+"/freeze.json":old_freeze,M.OLD+"/check_tracker_capacity_shadow.py":M.BASE_SHA,
            M.OLD+"/check_tracker_capacity_jetson.py":M.JETSON_SHA,**original_inputs}
        for clip in M.CLIPS:
            inputs.update({M.OLD+"/"+clip+"/result.json":old_results[clip],M.OLD+"/"+clip+"/"+clip+"_state_hashes.jsonl":old_states[clip]})
        common=dict(passed=True,error=None,workspace=workspace,freeze_sha256=digest,inputs_unchanged_after_check=True,
            clock_controls_unchanged=True,clock_policy_before={"same":True},clock_policy_after={"same":True},
            source_media_opened=False,detector_replayed=False,raw16_or_holdouts_accessed=False,production_promotion=False,
            scientific_improvement_claimed=False,saved_proposals_shadow_only=True,runtime_before=before)
        pre=dict(common,schema="seaqr.tracker-maturity-shadow.v1.preflight",clip=None,frames_processed=0,inputs_sha256=inputs)
        write(directory/"preflight.json",pre);artifacts={workspace+"/preflight.json":M.sha(directory/"preflight.json")}
        for clip in M.CLIPS:
            rows=old_rows[clip];checked,target=M.inspect_rows(*rows)
            write_rows(directory/clip/"candidate_frames.jsonl",rows[1]);write_rows(directory/clip/"shadow_audit.jsonl",rows[2])
            filehash={workspace+"/"+clip+"/"+name:M.sha(directory/clip/name) for name in ("candidate_frames.jsonl","shadow_audit.jsonl")}
            binding=dict(sha256=M.CANDIDATE_SHA,class_source_sha256=M.CLASS_SHA,unchanged_transformed_source_sha256=M.METHOD_SHA,
                changed_global_bindings=["_VictimIndex_v28"],eligibility_changed=False,configuration_changed=False,original_scope_unchanged=True)
            counters=dict(geometry_calls=0,geometry_fallbacks=0,batch_fallbacks=0,innovation_fallbacks=0,batch_track_rows=1,innovation_tracks=1)
            target_result=M.target_guard(S,target,refs) if clip=="0240" else None
            replay=dict(passed=True,shadow_only=True,no_tolerance_relaxation=True,separate_adapter_owners_verified=True,
                adapter_runs=[counters]*3,candidate_bindings=[binding]*2,artifacts_sha256=filehash,
                first_learning_divergence=checked["first_learning_divergence"],first_output_divergence=checked["first_output_divergence"],
                workload=checked["workload"],target=target_result,scientific_guard_passed=target_result["scientific_guard_passed"] if target_result else None,
                **{k:673 for k in ("attempted_frames","exact_archive_frames","exact_previous_state_frames","deterministic_candidate_frames","baseline_learning_exact_frames")})
            result=dict(common,schema="seaqr.tracker-maturity-shadow.v1.run",clip=clip,runtime_after=after,replay=replay,
                preflight_sha256=artifacts[workspace+"/preflight.json"],inputs_sha256={**inputs,**artifacts})
            if clip=="0240":result["prior_clip_sha256"]=artifacts[workspace+"/0170/result.json"]
            write(directory/clip/"result.json",result);artifacts.update(filehash);artifacts[workspace+"/"+clip+"/result.json"]=M.sha(directory/clip/"result.json")
        phases=[]
        for i,(name,args) in enumerate((("preflight",["--preflight"]),("run_0170",["--clip","0170"]),("run_0240",["--clip","0240"]))):
            phases.append(dict(name=name,pid=100+i,returncode=0,elapsed_seconds=1.,command=["/usr/bin/python3","-I","-u",
                workspace+"/check_tracker_maturity_shadow_v1.py","--workspace",workspace,"--freeze-sha256",digest,*args]))
        batch=dict(schema="seaqr.tracker-maturity-shadow.batch.v1",complete=True,execution_passed=True,bundle_and_child_artifacts_unchanged=True,
            error=None,current=None,not_run=[],freeze_sha256=digest,source_media_accessed=False,production_changed=False,shadow_only=True,
            scientific_improvement_claimed=False,automatic_retries=0,elapsed_seconds=3.,phases=phases,child_artifacts_sha256=artifacts,
            execution=dict(workers=1,children=3,start_below_celsius=65,stop_at_celsius=75,batch_deadline_seconds=3900,automatic_retries=0))
        write(directory/"batch_status.json",batch)
        write_rows(directory/"telemetry.jsonl",[dict(phase=p["name"],temperatures_c={"cpu":49.}) for p in phases])
        yield (directory,bundle,baseline,archive_root,digest)


class Core(unittest.TestCase):
    def test_typed_float_serializer_matches_frozen_metadata_values(self):
        base=imported("maturity_frozen_baseline_json_only",ROOT/"scripts/check_tracker_capacity_shadow.py")
        for value in ({"a":-0.,"b":[True,1,1.,None]},generated(1)[0][0]):self.assertEqual(M.value_sha(value),base.value_sha(value))
        self.assertNotEqual(M.value_sha({"x":0.}),M.value_sha({"x":-0.}))
        self.assertIsNotNone(M.first_difference({"x":1},{"x":1.}))

    def test_all673_and_first_preupdate_learning_divergence(self):
        checked,_=M.inspect_rows(*generated())
        self.assertEqual(checked["frames"],673);self.assertEqual(checked["first_output_divergence"]["frame_index"],50)
        self.assertEqual(checked["first_learning_divergence"]["frame_index"],52)
        self.assertTrue(checked["first_learning_divergence"]["measured_before_frame_update"])
        self.assertEqual(checked["workload"]["candidate"]["burst"]["counts"]["frames"],56)
        self.assertEqual(checked["workload"]["candidate"]["full"]["never_reached_four_hits_by_end_of_clip"],1)

    def test_changed_candidates_matrix_coverage_timing_rejected(self):
        for field in ("candidates","source_to_reference","coverage","timings_ms"):
            values=generated(3);values[1][1][field]=[]
            with self.assertRaises(ValueError):M.inspect_rows(*values,expected_frames=3)

    def test_nondeterminism_learning_annotation_and_old_state_fail(self):
        for mutate in (
            lambda values:values[2][1]["internal_state_sha256"].__setitem__(2,"f"*64),
            lambda values:values[2][1]["learning_centers_sha256"].__setitem__(2,"f"*64),
            lambda values:values[2][1].update(past_or_at_learning_divergence=True),
            lambda values:values[3][1].update(archive_output_sha256="f"*64),
        ):
            values=generated(3);mutate(values)
            with self.assertRaises(ValueError):M.inspect_rows(*values,expected_frames=3)

    def test_missing_extra_and_discontinuous_frames_fail(self):
        values=generated(3);values[1].pop()
        with self.assertRaises(ValueError):M.inspect_rows(*values,expected_frames=3)
        with self.assertRaises(ValueError):M.inspect_rows(*generated(4),expected_frames=3)
        values=generated(3);values[2][1]["frame_index"]=2
        with self.assertRaises(ValueError):M.inspect_rows(*values,expected_frames=3)

    def test_workload_prediction_measured_and_right_censoring(self):
        values=generated(3);r=values[0][0];r["tracks"][0]["measured"]=False
        collector=M.Workload();frame=collector.add(r)
        self.assertEqual(frame["counts"]["qualified_predicted"],1)
        self.assertNotIn("qualified_measured",frame["counts"])
        r=copy.deepcopy(r);r["tracks"][0]["independent_hits"]=1;r["tracks"][0]["qualified_moving"]=False
        collector=M.Workload();collector.add(r)
        self.assertEqual(collector.finish()["full"]["still_active_without_four_hits_at_end_of_clip"],1)

    def test_target_loss_predictions_fragmentation_and_ambiguity(self):
        _,rows=M.inspect_rows(*generated());refs=[dict(frame_index=i,measurement_source_xy=[200.+i,300.]) for i in range(430,465)]
        self.assertTrue(M.target_guard(S,rows,refs)["scientific_guard_passed"])
        for mutate in (
            lambda r:r["candidate"][0]["tracks"][0].update(measured=False),
            lambda r:r["candidate"][0]["tracks"][0].update(track_id="dark:100"),
            lambda r:r["candidate"][0]["tracks"].append(dict(r["candidate"][0]["tracks"][0],track_id="dark:100")),
        ):
            changed=copy.deepcopy(rows);mutate(changed)
            self.assertFalse(M.target_guard(S,changed,refs)["scientific_guard_passed"])

    def test_distance_one_ulp_only_without_membership_tolerance(self):
        value=dict(details=[dict(distance_native_px=1.,measurement_source_xy=[1.,2.])],passed=True)
        one=copy.deepcopy(value);one["details"][0]["distance_native_px"]=math.nextafter(1.,2.)
        audit=M.compare_target(value,one);self.assertFalse(audit["distance_values_exact"])
        bad=copy.deepcopy(value);bad["details"][0]["distance_native_px"]=7.
        with self.assertRaises(ValueError):M.compare_target(value,bad)
        bad=copy.deepcopy(value);bad["passed"]=False
        with self.assertRaises(ValueError):M.compare_target(value,bad)
        bad=copy.deepcopy(value);bad["details"][0].pop("distance_native_px")
        with self.assertRaises(ValueError):M.compare_target(value,bad)


class Loading(unittest.TestCase):
    def test_full1346_complete_loader_and_serialization_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder,fixture(folder) as inputs:
            output=Path(folder).resolve()/"summary.json";result=M.save_summary(*inputs,output)
            self.assertEqual(result["frames_checked"],1346);self.assertTrue(result["integrity_passed"])
            self.assertTrue(result["target_scientific_guard_passed"]);self.assertFalse(result["production_promotion"])
            self.assertEqual(result["thermal"]["observed_peak_c"],49.)
            self.assertEqual(M.decode(output.read_text()),result)
            with self.assertRaisesRegex(ValueError,"overwrite"):M.save_summary(*inputs,output)

    def test_scientific_loss_is_complete_result_not_integrity_failure(self):
        with tempfile.TemporaryDirectory() as folder,fixture(folder,lost=True) as inputs:
            result=M.summarize(*inputs)
            self.assertTrue(result["completed"]);self.assertFalse(result["target_scientific_guard_passed"])
            self.assertEqual(result["clips"]["0240"]["target"]["newly_lost_frames"],[440])
            self.assertEqual(result["outcome"],"target_guard_failed_no_promotion")

    def test_hash_receipt_and_incomplete_batch_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder,fixture(folder) as inputs:
            directory=inputs[0];path=directory/"0170/candidate_frames.jsonl";old=path.read_bytes();path.write_bytes(old+b" ")
            with self.assertRaisesRegex(ValueError,"hash"):M.summarize(*inputs)
            path.write_bytes(old);batch=M.decode((directory/"batch_status.json").read_text());batch["complete"]=False
            write(directory/"batch_status.json",batch)
            with self.assertRaisesRegex(ValueError,"Complete"):M.summarize(*inputs)

    def test_failed_caller_freeze_before_scorer_import(self):
        with tempfile.TemporaryDirectory() as folder,fixture(folder) as inputs:
            with self.assertRaisesRegex(ValueError,"hash"):M.summarize(*inputs[:-1],"f"*64)

    def test_input_mutation_during_read_and_source_endcheck(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/"metadata.json";write(path,dict(a=1));evidence=M.Evidence();original=M.decode
            def mutate(text):
                result=original(text);path.write_text('{"a":2}');return result
            with patch.object(M,"decode",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"during read"):evidence.read(path)
            evidence=M.Evidence();evidence.bind(path);path.write_text('{"a":3}')
            with self.assertRaisesRegex(ValueError,"Input changed"):evidence.verify()

    def test_duplicate_nonfinite_and_symlink_metadata(self):
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
            with self.assertRaises(ValueError):M.decode(text)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/"file.json";write(path,{});link=path.with_name("linked.json");link.symlink_to(path)
            with self.assertRaises(ValueError):M.Evidence().read(link)


if __name__=="__main__":unittest.main()
