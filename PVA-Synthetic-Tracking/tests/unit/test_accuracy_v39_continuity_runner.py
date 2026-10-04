"""Synthetic stage scoring and temporary-only provenance tests; no real media."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import evaluate_accuracy_v39_continuity as runner
from accuracy_v39_continuity import CausalMeasuredContinuity


def track(tid="bright:1", xy=(20.,30.), qualified=True, measured=True):
    return dict(track_id=tid,segment=0,measured=measured,qualified_moving=qualified,
        source_xy=list(xy),measurement_source_xy=list(xy) if measured else None,
        confirmation_timestamp_ns=0,excursion_px=20.,
        motion_quality=dict(ready=True,passed=qualified,
            quadratic_fit_rmse_px=1. if qualified else 4.,maximum_rmse_px=3.))


def row(frame=0, tracks=(), candidates=None):
    if candidates is None:
        candidates=[dict(source_xy=t["measurement_source_xy"],polarity=t["track_id"].split(":")[0])
                    for t in tracks if t["measured"]]
    return dict(frame_index=frame,timestamp_ns=frame*100_000_000,segment=0,
        motion=dict(reset=False),tracks=list(tracks),candidates=candidates)


def sample(kind="dense", window="a", frame=0, xy=(20.,30.), radius=2., polarity="bright"):
    return dict(kind=kind,window=window,frame=frame,xy=list(xy),radius=radius,polarity=polarity)


def shadow(frame=0, tracks=()):
    return dict(frame_index=frame,timestamp_ns=frame*100_000_000,segment=0,
        tracks=[dict(segment=0,track_id=t["track_id"],accepted=True) for t in tracks])


def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl_write(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(value)+"\n" for value in values))


class StageTests(unittest.TestCase):
    def evaluate(self, samples, source, previous=None, accepted=()):
        policy=CausalMeasuredContinuity()
        if previous is not None:policy.update(previous)
        decisions=policy.update(source)
        return runner.match_stages(samples,source,decisions,shadow(source["frame_index"],accepted))

    def test_all_stages_use_source_measurement_and_radius_boundary(self):
        t=track(xy=(22.,30.));source=row(tracks=[t]);before=copy.deepcopy(source)
        t["source_xy"]=[900.,900.]
        evidence=self.evaluate([sample()],source,accepted=[t])[0]
        self.assertTrue(all(v["hit"] for v in evidence["stages"].values()))
        self.assertEqual(evidence["stages"]["actual_measurement"]["distance_px"],2.)
        self.assertEqual(source["tracks"][0]["measurement_source_xy"],before["tracks"][0]["measurement_source_xy"])

    def test_prediction_is_not_a_measured_hit_in_any_output_stage(self):
        t=track(measured=False);source=row(tracks=[t])
        evidence=self.evaluate([sample()],source,accepted=[t])[0]
        self.assertTrue(all(not v["hit"] for v in evidence["stages"].values()))

    def test_candidate_without_association_has_candidate_hit_only(self):
        source=row(candidates=[dict(source_xy=[20.,30.],polarity="bright")])
        stages=self.evaluate([sample()],source)[0]["stages"]
        self.assertTrue(stages["candidate"]["hit"])
        self.assertTrue(all(not stages[name]["hit"] for name in runner.STAGES if name!="candidate"))

    def test_added_degraded_measurement_is_not_baseline_or_shadow(self):
        previous=row(tracks=[track()]);source=row(1,[track(qualified=False)])
        stages=self.evaluate([sample(frame=1)],source,previous)[0]["stages"]
        self.assertEqual({k:v["hit"] for k,v in stages.items()},dict(candidate=True,
            actual_measurement=True,baseline_qualified=False,with_degraded=True,v36_shadow=False))

    def test_unqualified_birth_does_not_borrow_another_id_anchor(self):
        previous=row(tracks=[track()]);source=row(1,[track(tid="bright:2",qualified=False)])
        stages=self.evaluate([sample(frame=1)],source,previous)[0]["stages"]
        self.assertTrue(stages["actual_measurement"]["hit"])
        self.assertFalse(stages["with_degraded"]["hit"])

    def test_polarity_and_outside_radius_do_not_match(self):
        for t in (track(tid="dark:1"),track(xy=(22.00001,30.))):
            with self.subTest(track=t):
                self.assertTrue(all(not v["hit"] for v in self.evaluate([sample()],row(tracks=[t]))[0]["stages"].values()))

    def test_reference_families_are_separate_but_same_family_is_one_to_one(self):
        samples=[sample(),sample(window="b"),sample(kind="pilot"),sample(kind="anchor")]
        evidence=self.evaluate(samples,row(tracks=[track()]))
        summary=runner.summarize_evidence(evidence)
        self.assertEqual(summary["dense"]["samples"],2)
        self.assertEqual(summary["dense"]["hits"]["baseline_qualified"],1)
        self.assertEqual(summary["pilot"]["hits"]["baseline_qualified"],1)
        self.assertEqual(summary["anchor"]["hits"]["baseline_qualified"],1)
        self.assertTrue(all(e["stages"]["actual_measurement"]["ambiguous"] for e in evidence if e["kind"]=="dense"))

    def test_all_gated_alternatives_are_reported_without_identity_claim(self):
        source=row(tracks=[track(),track(tid="bright:2",xy=(21.,30.))])
        e=self.evaluate([sample()],source)[0]
        self.assertEqual(e["stages"]["baseline_qualified"]["all_gated_ids"],["0/bright:1","0/bright:2"])
        self.assertTrue(e["stages"]["baseline_qualified"]["ambiguous"])
        summary=runner.summarize_evidence([e])["dense"]
        self.assertFalse(summary["airborne_truth"]);self.assertFalse(summary["physical_identity_inferred"])

    def test_shadow_cannot_create_a_baseline_ineligible_identity(self):
        source=row(tracks=[track(qualified=False)])
        with self.assertRaises(ValueError):self.evaluate([sample()],source,accepted=source["tracks"])

    def test_summary_preserves_missing_and_empty_denominators(self):
        source=row();samples=[sample(),sample(kind="compact_light",window="c")]
        s=runner.summarize_evidence(self.evaluate(samples,source))
        self.assertEqual(s["dense"]["samples"],1);self.assertEqual(s["compact_light"]["samples"],1)
        self.assertEqual(s["pilot"]["samples"],0)
        self.assertTrue(all(value==0 for kind in s.values() for value in kind["hits"].values()))

    def test_summary_reports_changed_proximity_assignment_separately(self):
        source=row(tracks=[track()]);e=self.evaluate([sample()],source)[0]
        e["stages"]["with_degraded"]["assigned_id"]="0/bright:2"
        s=runner.summarize_evidence([e])["dense"]
        self.assertEqual(s["changed_proximity_assignments"],[dict(window="a",frame=0,before="0/bright:1",after="0/bright:2")])
        self.assertFalse(s["recovered_degraded_frames"])


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.references={}
        for kind in ("dense","pilot"):
            self.references[kind+"_labels"]=self.root/(kind+"_labels.json")
            self.references[kind+"_packet"]=self.root/(kind+"_packet.json")
            json_write(self.references[kind+"_packet"],dict(windows=[dict(id="included",clip_id="0029"),dict(id="other",clip_id="0126")]))
            json_write(self.references[kind+"_labels"],dict(positive_windows=[
                dict(window_id="included",polarity="bright",visible_samples=[dict(frame_index=1,xy=[4.,5.],uncertainty_px=3.)]),
                dict(window_id="other",polarity="dark",visible_samples=[dict(frame_index=9,xy=[7.,8.],uncertainty_px=2.)])]))
        self.references["anchors_0029"]=self.root/"anchors.json"
        json_write(self.root/"compact_light_reference_v1.json",dict(frames=[
            dict(frame_index=1,visibility="visible",source_xy=[4.,5.],position_uncertainty_radius_px=4.),
            dict(frame_index=2,visibility="ambiguous",source_xy=[4.,5.],position_uncertainty_radius_px=4.)]))

    def test_original_family_separation_required_anchors_and_unknown_visibility(self):
        events=[dict(event_id="a",polarity="bright",anchors=[
            dict(required=True,frame_index=1,xy=[4.,5.],uncertainty_px=2.),
            dict(required=False,frame_index=2,xy=[4.,5.],uncertainty_px=2.)])]
        with patch.object(runner,"REFERENCES",self.references),patch.object(runner,"V38",self.root),patch.object(runner,"load_reference",return_value=("source",events)):
            samples=runner.samples_for("0029")
        self.assertEqual([s["kind"] for s in samples],["dense","pilot","anchor","compact_light"])
        self.assertEqual([s["radius"] for s in samples],[5.,5.,4.,6.])
        self.assertTrue(all(s["frame"]==1 for s in samples))

    def test_duplicate_reference_sample_is_rejected(self):
        labels=json.loads(self.references["dense_labels"].read_text());entry=labels["positive_windows"][0]
        entry["visible_samples"].append(copy.deepcopy(entry["visible_samples"][0]))
        json_write(self.references["dense_labels"],labels)
        with patch.object(runner,"REFERENCES",self.references),patch.object(runner,"V38",self.root),patch.object(runner,"load_reference",return_value=("source",[])):
            with self.assertRaisesRegex(ValueError,"Duplicate reference"):runner.samples_for("0029")


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.source=self.root/"source";self.output=self.root/"output";self.output.mkdir()
        self.v36=self.root/"v36"

    def run_rows(self, rows, shadows=None, samples=(), frames=None, cid="0029", controls=None):
        if shadows is None:shadows=[shadow(r["frame_index"],[t for t in r["tracks"] if t["qualified_moving"]]) for r in rows]
        jsonl_write(self.source/"frames.jsonl",rows)
        jsonl_write(self.v36/"full_context_01"/(cid+"_decisions.jsonl"),shadows)
        control_path=self.root/"controls.json";json_write(control_path,dict(controls=[] if controls is None else controls))
        with patch.object(runner,"V36",self.v36),patch.object(runner,"samples_for",return_value=list(samples)),patch.object(runner,"REFERENCES",dict(controls=control_path)):
            return runner.analyze(cid,dict(path=str(self.source),frames=len(rows) if frames is None else frames),self.output)

    def test_degraded_expiry_predictions_and_denominators_remain_distinct(self):
        rows=[row(0,[track()]),row(1,[track(qualified=False)]),row(2,[track(qualified=False)]),
              row(3,[track(qualified=False)]),row(4,[track(measured=False)])]
        result=self.run_rows(rows,samples=[sample(frame=f) for f in range(5)])
        counts=result["counts"]
        self.assertEqual(counts["actual_measured_states"],4)
        self.assertEqual(counts["baseline_qualified_measured"],1)
        self.assertEqual(counts["baseline_qualified_predictions"],1)
        self.assertEqual(counts["added_degraded_measured"],2)
        self.assertEqual(counts["renderable_measured"],3)
        self.assertEqual(result["references"]["dense"]["samples"],5)
        self.assertEqual(result["references"]["dense"]["hits"]["with_degraded"],3)
        self.assertEqual(result["predictions_added"],0)
        written=[json.loads(line) for line in (self.output/"0029_decisions.jsonl").read_text().splitlines()]
        self.assertEqual(written[1]["output_states"][0]["anchor_frame"],0)
        self.assertFalse(written[1]["output_states"][0]["confirmed_output"])
        self.assertIsNone(written[4]["output_states"][0]["measurement_source_xy"])
        self.assertEqual(written[4]["output_states"][0]["source_xy"],[20.,30.])
        self.assertEqual(written[4]["output_states"][0]["status"],"baseline_qualified_prediction")

    def test_fixed_control_inclusive_time_halfopen_spatial_and_no_coast_counts(self):
        rows=[row(0,[track()]),row(1,[track(qualified=False)]),row(2,[track(qualified=False)]),row(3,[track(measured=False)])]
        controls=[dict(label="inside",frames_inclusive=[0,2],crop_xywh=[20.,30.,1.,1.]),
                  dict(label="outside",frames_inclusive=[0,3],crop_xywh=[19.,29.,1.,1.])]
        result=self.run_rows(rows,cid="0126",controls=controls)
        self.assertEqual(result["provisional_controls"],[dict(label="inside",baseline_measured=1,added_degraded_measured=2),dict(label="outside",baseline_measured=0,added_degraded_measured=0)])

    def test_reference_after_eof_is_denominator_failure(self):
        with self.assertRaisesRegex(ValueError,"denominator"):
            self.run_rows([row()],samples=[sample(frame=2)])

    def test_source_frame_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"denominator"):self.run_rows([row()],frames=2)

    def test_truncated_shadow_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Truncated"):self.run_rows([row()],shadows=[])

    def test_extra_shadow_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"extra"):self.run_rows([row()],shadows=[shadow(),shadow(1)])

    def test_wrong_frame_timestamp_and_shadow_segment_are_rejected(self):
        for change in ("frame","timestamp","segment"):
            with self.subTest(change=change):
                source=row();s=shadow()
                if change=="frame":source["frame_index"]=1
                elif change=="timestamp":source["timestamp_ns"]=1
                else:s["segment"]=1
                subdir=self.root/change;subdir.mkdir();self.output=subdir
                with self.assertRaisesRegex(ValueError,"Misaligned"):self.run_rows([source],shadows=[s])

    def test_measured_nan_is_not_silently_scored_as_missing(self):
        t=track();t["measurement_source_xy"]=[float("nan"),30.]
        with self.assertRaises(ValueError):self.run_rows([row(tracks=[t])],samples=[sample()])

    def test_existing_decision_log_is_not_overwritten(self):
        path=self.output/"0029_decisions.jsonl";path.write_text("preserve")
        with self.assertRaises(FileExistsError):self.run_rows([row()])
        self.assertEqual(path.read_text(),"preserve")


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.audit=self.root/"audit.json"
        self.checked=self.root/"checked.json";json_write(self.checked,{"old":True})
        self.journal=self.root/"original.jsonl";self.journal.write_text("synthetic original\n")
        self.v36=self.root/"v36";self.refs={"reference":self.root/"reference.json"};json_write(self.refs["reference"],{})
        self.code=("scripts/policy.py","tests/unit/test_runner.py","docs/plan.md")
        for name in self.code:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("frozen synthetic source\n")
        self.inputs={cid:dict(path=str(self.root/cid),frames=n,source_sha256="synthetic-source",
            files_sha256={str(self.journal):runner.sha(self.journal)}) for cid,n in {"0029":687,"0126":674,"0055":689,"0082":691}.items()}
        for cid in self.inputs:json_write(self.v36/"full_context_01"/(cid+"_decisions.jsonl"),{})
        json_write(self.audit,dict(verified=True,checked_files_sha256={str(self.checked):runner.sha(self.checked),str(self.root/"must-not-open.avi"):"not-read"}))
        self.audit_sha=runner.sha(self.audit);self.output=self.root/"output"
        self.coverage=self.root/"validation_coverage_plan_v1.json";json_write(self.coverage,dict(synthetic=True))
        self.coverage_bindings=[]

    def patches(self):
        original_bind=runner.bind
        def bind_fixture(files,path,expected=None):
            if Path(path).resolve()==self.coverage:
                self.assertEqual(expected,"cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc")
                self.coverage_bindings.append(str(path))
                return original_bind(files,path)
            return original_bind(files,path,expected)
        return (patch.multiple(runner,ROOT=self.root,BASE=self.root,V36=self.v36,AUDIT=self.audit,AUDIT_SHA=self.audit_sha,
            CODE=self.code,REFERENCES=self.refs,bind=bind_fixture),patch.object(runner,"verified_inputs",return_value=self.inputs))

    def result(self,cid):
        r={kind:dict(samples=0,hits={stage:0 for stage in runner.STAGES}) for kind in ("dense","pilot","anchor","compact_light")}
        if cid=="0029":
            for kind,(n,hits) in {"dense":(285,284),"pilot":(28,28),"anchor":(24,24),"compact_light":(8,6)}.items():
                r[kind]=dict(samples=n,hits={stage:hits for stage in runner.STAGES})
        return dict(references=r)

    def run_mock(self, callback=None):
        def analyze(cid,spec,output):
            frozen=json.loads((output/"freeze.json").read_text())
            self.assertTrue(frozen["pre_replay"]);self.assertFalse(frozen["degraded_is_confirmed"])
            for name in self.code:self.assertEqual((output/"implementation"/name).read_bytes(),(self.root/name).read_bytes())
            if callback:callback(cid,spec,output)
            jsonl_write(output/(cid+"_decisions.jsonl"),[dict(synthetic=True)])
            return self.result(cid)
        a,b=self.patches()
        with a,b,patch.object(runner.subprocess,"run") as tests,patch.object(runner,"analyze",side_effect=analyze) as analysis:
            runner.run(self.output)
            return tests,analysis

    def test_all_inputs_snapshots_and_tests_are_frozen_before_any_replay(self):
        tests,analysis=self.run_mock()
        tests.assert_called_once();self.assertTrue(tests.call_args.kwargs["check"])
        self.assertEqual(analysis.call_count,4)
        freeze=json.loads((self.output/"freeze.json").read_text());summary=json.loads((self.output/"summary.json").read_text())
        self.assertEqual(freeze["config"]["maximum_gap_frames"],2)
        self.assertEqual(freeze["config"]["maximum_gap_ns"],200_000_000)
        self.assertIn(str(self.checked),freeze["inputs_sha256"])
        self.assertIn(str(self.journal),freeze["inputs_sha256"])
        self.assertIn(str(self.output/"unit.log"),freeze["inputs_sha256"])
        self.assertEqual(self.coverage_bindings,[str(self.coverage)])
        self.assertIn(str(self.coverage),freeze["inputs_sha256"])
        self.assertEqual(summary["freeze_sha256"],runner.sha(self.output/"freeze.json"))
        self.assertFalse(summary["airborne_accuracy_established"])
        self.assertFalse(summary["source_video_files_opened"])
        self.assertEqual(summary["references"]["compact_light"]["samples"],8)
        completion=json.loads((self.output/"completion_receipt.json").read_text())
        self.assertTrue(completion["completed"])
        self.assertTrue(completion["all_inputs_and_outputs_rehashed"])
        self.assertEqual(completion["checked_files_sha256"][str(self.output/"summary.json")],runner.sha(self.output/"summary.json"))
        self.assertFalse(completion["source_videos_opened"])

    def test_existing_output_rejected_before_preflight(self):
        self.output.mkdir();marker=self.output/"marker";marker.write_text("keep")
        with patch.object(runner,"verified_inputs") as verify:
            with self.assertRaises(FileExistsError):runner.run(self.output)
            verify.assert_not_called()
        self.assertEqual(marker.read_text(),"keep")

    def test_audit_hash_change_rejected_before_output(self):
        json_write(self.audit,dict(verified=False,checked_files_sha256={}))
        a,b=self.patches()
        with a,b:
            with self.assertRaisesRegex(ValueError,"Changed input"):runner.run(self.output)
        self.assertFalse(self.output.exists())

    def test_unverified_receipt_rejected_even_with_matching_hash(self):
        json_write(self.audit,dict(verified=False,checked_files_sha256={}));self.audit_sha=runner.sha(self.audit)
        a,b=self.patches()
        with a,b:
            with self.assertRaisesRegex(ValueError,"Verified V38"):runner.run(self.output)
        self.assertFalse(self.output.exists())

    def test_checked_input_mutation_rejected_before_output(self):
        self.checked.write_text("changed")
        a,b=self.patches()
        with a,b:
            with self.assertRaisesRegex(ValueError,"Changed input"):runner.run(self.output)
        self.assertFalse(self.output.exists())

    def test_original_input_mutation_rejected_before_output(self):
        self.journal.write_text("changed")
        a,b=self.patches()
        with a,b:
            with self.assertRaisesRegex(ValueError,"Changed input"):runner.run(self.output)
        self.assertFalse(self.output.exists())

    def test_mutation_during_replay_prevents_completed_summary(self):
        def mutate(cid,spec,output):
            if cid=="0029":self.checked.write_text("changed during replay")
        with self.assertRaisesRegex(ValueError,"Changed bound input"):self.run_mock(mutate)
        self.assertTrue((self.output/"freeze.json").exists());self.assertFalse((self.output/"summary.json").exists())

    def test_test_failure_prevents_freeze_and_replay(self):
        a,b=self.patches()
        with a,b,patch.object(runner.subprocess,"run",side_effect=RuntimeError("tests failed")),patch.object(runner,"analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError,"tests failed"):runner.run(self.output)
            analyze.assert_not_called()
        self.assertFalse((self.output/"freeze.json").exists())

    def test_changed_reference_denominator_prevents_completion(self):
        original=self.result
        def wrong(cid):
            result=original(cid)
            if cid=="0029":result["references"]["dense"]["samples"]-=1
            return result
        with patch.object(self,"result",side_effect=wrong):
            with self.assertRaisesRegex(ValueError,"denominator"):self.run_mock()
        self.assertFalse((self.output/"summary.json").exists())


if __name__=="__main__":unittest.main()
