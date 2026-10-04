"""Generated metadata only: no producer/core imports or real experiment outcomes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().with_name("summarize_weak_continuation_shadow_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2]/"scripts/summarize_weak_continuation_shadow_v1.py"
spec = importlib.util.spec_from_file_location("shadow_summary",SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DEFAULT_FRAMES = sum(n for n,_ in m.SOURCES.values())
DEFAULT_SLOTS = m.SCHEDULED_SLOTS
GENERATED_SOURCES = {clip:(3,str(i)*64) for i,clip in enumerate(m.SOURCES,1)}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))


def read(path):
    return json.loads(path.read_text())


def record(tid=0,measured=True,qualified=True,xy=None,weak=False,shadow=False):
    result = dict(track_id=f"bright:{tid}",segment=0,measured=measured,qualified_moving=qualified,
                  measurement_source_xy=(xy or [10.0,20.0]) if measured else None)
    if shadow:
        note = dict(identity=f"0/bright:{tid}",applied=weak,is_ordinary_measurement=False,physical_identity_verified=False,
                    status="weak_kinematic_correction" if weak else "strong_measurement_priority" if measured else "missing_capture")
        if weak:
            note["observations"] = dict(coverage_known=True,coverage_unknown_reasons=[])
        result["weak_evidence"] = note
    return result


def rows():
    baseline,trace = [],[]
    for frame in range(3):
        b = [record(measured=frame!=1)]
        s = [record(tid=1 if frame==2 else 0,measured=frame!=1,xy=[12.,20.] if frame==2 else None,weak=frame==1,shadow=True)]
        header = dict(frame_index=frame,timestamp_ns=frame*100000000,segment=0)
        baseline.append(dict(header,tracks=b))
        trace.append(dict(header,capture_scheduled=frame==1,records=s,
            metrics=dict(weak_continuation=dict(frame_index=frame,decisions=[r["weak_evidence"] for r in s],applied_count=sum(r["weak_evidence"]["applied"] for r in s)))))
    return baseline,trace


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source_patch = patch.dict(m.SOURCES,GENERATED_SOURCES,clear=True)
        self.slot_patch = patch.object(m,"SCHEDULED_SLOTS",3)
        self.source_patch.start();self.slot_patch.start()
        plan = dict(schema="seaqr.weak-continuation-shadow.plan.v1",full_causal_replay=True,concurrent_workers=1,
                    clips={clip:dict(frames=count,source_sha256=digest,weak_windows_inclusive=[[1,1]]) for clip,(count,digest) in m.SOURCES.items()})
        write(self.root/"plan.json",plan)
        self.plan_sha = m.sha(self.root/"plan.json")
        write(self.root/"freeze.json",dict(schema="seaqr.weak-continuation-shadow.freeze.v1",pre_run=True,plan_sha256=self.plan_sha))
        self.freeze_sha = m.sha(self.root/"freeze.json")
        write(self.root/"batch_status.json",dict(schema="seaqr.weak-continuation-shadow.batch.v1",passed=True,error=None,
            concurrent_workers=1,production_changed=False,runs=[dict(clip=clip,arm=arm,returncode=0) for clip in m.SOURCES for arm in ("clean","shadow","audit")]))
        for clip in m.SOURCES:
            self.save_clip(clip,*rows())

    def tearDown(self):
        self.source_patch.stop();self.slot_patch.stop();self.temp.cleanup()

    def save_clip(self,clip,baseline,trace,receipt_changes=None):
        root = self.root/clip
        lines(root/"clean/frames.jsonl",baseline)
        lines(root/"shadow/shadow_trace.jsonl",trace)
        for arm in ("clean","shadow"):
            receipt = dict(schema="seaqr.weak-continuation-shadow.run.v1",passed=True,error=None,clip=clip,arm=arm,
                processed_frames=3,expected_frames=3,freeze_sha256=self.freeze_sha,plan_sha256=self.plan_sha,
                source_sha256=m.SOURCES[clip][1],production_changed=False,weak_learning_enabled=False,
                trace_sha256=m.sha(root/"shadow/shadow_trace.jsonl") if arm=="shadow" else None,
                diagnostic_cost_ms=dict(capture=10,shadow=50,snapshot_write=20))
            if receipt_changes and arm=="shadow":
                receipt.update(receipt_changes)
            write(root/(arm+".shadow.json"),receipt)
        required = ("clean/frames.jsonl","shadow/shadow_trace.jsonl","clean.shadow.json","shadow.shadow.json")
        write(root/"independent_audit.json",dict(schema="seaqr.weak-continuation-shadow.audit.v1",passed=True,clip=clip,frames=3,
            source_sha256=m.SOURCES[clip][1],freeze_sha256=self.freeze_sha,plan_sha256=self.plan_sha,
            baseline_journal_non_timing_exact=True,baseline_output_state_learning_digests_exact=True,native_state_guards_unchanged=True,
            production_changed=False,weak_learning_enabled=False,files_sha256={name:m.sha(root/name) for name in required}))

    def summarize(self):
        return m.summarize(self.root,self.freeze_sha,self.plan_sha)

    def test_completed_cohort_counts_partitions_lifetimes_and_null_truth(self):
        result = self.summarize()
        self.assertTrue(result["passed"])
        self.assertEqual((result["unique_source_frames"],result["decoded_frame_instances"],result["scheduled_shadow_frame_slots"]),(9,18,3))
        clip = result["clips"]["0029"]
        self.assertEqual(clip["state_totals"]["baseline"]["predicted_only"],1)
        self.assertEqual(clip["state_totals"]["shadow"]["weak_corrected_qualified"],1)
        self.assertEqual(clip["state_totals"]["shadow"]["predicted_only"],0)
        self.assertEqual(clip["qualified_identity_counts"],dict(baseline=1,shadow=2))
        life = clip["qualified_identities"]["baseline"]["0/bright:0"]
        self.assertEqual(life["journal_visible_lifetime_span_seconds"],.2)
        self.assertEqual(life["qualified_frame_count"],3)
        self.assertTrue(all(v is None for v in clip["truth_metrics"].values()))
        self.assertEqual(DEFAULT_FRAMES,2050)
        self.assertEqual(2*DEFAULT_FRAMES,4100)
        self.assertEqual(DEFAULT_SLOTS,182)

    def test_assignment_diagnostics_only_actual_ids_and_positions(self):
        diff = self.summarize()["clips"]["0029"]["exact_native_assignment_diagnostic_totals"]
        self.assertEqual(diff["frames_with_different_exact_assignments"],1)
        self.assertEqual(diff["baseline_only_native_ids"],1)
        self.assertEqual(diff["shadow_only_native_ids"],1)
        self.assertFalse(m.assignment_difference({}, {})["changed"])
        changed = m.assignment_difference({"a":[1.0,2.]},{"a":[1.0000000001,2.]})
        self.assertEqual(changed["same_native_id_different_actual_measurement"],1)

    def test_nested_snapshot_write_separate_not_fps(self):
        clip = self.summarize()["clips"]["0029"]
        self.assertEqual(clip["diagnostic_timing_ms"],dict(capture=10,shadow_excluding_snapshot_write=30,snapshot_write=20,shadow_inclusive_snapshot_write=50))
        self.assertTrue(clip["timing_is_not_pipeline_fps"])
        b,t=rows();self.save_clip("0029",b,t,dict(diagnostic_cost_ms=dict(capture=10,shadow=19,snapshot_write=20)))
        with self.assertRaisesRegex(ValueError,"nested snapshot"):
            self.summarize()

    def test_missing_partial_failed_audit_and_batch_fail_closed(self):
        path=self.root/"0029/independent_audit.json"
        original=read(path)
        for changes in ({"passed":False},{"frames":2},{"native_state_guards_unchanged":False}):
            write(path,dict(original,**changes))
            with self.assertRaises(ValueError):self.summarize()
        write(path,original)
        batch=read(self.root/"batch_status.json");batch["runs"].pop();write(self.root/"batch_status.json",batch)
        with self.assertRaisesRegex(ValueError,"nine completed"):
            self.summarize()

    def test_run_receipt_failed_even_when_bound_in_audit(self):
        self.save_clip("0029",*rows(),dict(passed=False,error="stopped"))
        with self.assertRaisesRegex(ValueError,"incomplete run"):
            self.summarize()

    def test_post_audit_journal_tamper_rejected(self):
        path=self.root/"0029/clean/frames.jsonl"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"changed bound"):
            self.summarize()

    def test_partial_extra_blank_rows_rejected_even_rebound(self):
        for change in (lambda b,t:(b[:-1],t[:-1]),lambda b,t:(b+[b[0]],t+[t[0]])):
            self.save_clip("0029",*change(*rows()))
            with self.assertRaises(ValueError):self.summarize()

    def test_weak_and_strong_never_conflated(self):
        b,t=rows();t[1]["records"][0]["measured"]=True;t[1]["records"][0]["measurement_source_xy"]=[10.,20.]
        self.save_clip("0029",b,t)
        with self.assertRaisesRegex(ValueError,"conflation"):
            self.summarize()

    def test_unmeasured_actual_coordinates_rejected(self):
        b,t=rows();b[1]["tracks"][0]["measurement_source_xy"]=[1,2]
        self.save_clip("0029",b,t)
        with self.assertRaisesRegex(ValueError,"unmeasured"):
            self.summarize()

    def test_decision_record_mismatch_rejected(self):
        b,t=rows();t[1]["metrics"]["weak_continuation"]["applied_count"]=0
        self.save_clip("0029",b,t)
        with self.assertRaisesRegex(ValueError,"decision/record"):
            self.summarize()

    def test_censored_available_ratios_and_invalid_status_reported(self):
        b,t=rows()
        note=t[1]["records"][0]["weak_evidence"]
        note.update(applied=False,status="capture_coverage_unknown",observations=dict(coverage_known=False,coverage_unknown_reasons=["unsupported"]))
        t[1]["metrics"]["weak_continuation"]["applied_count"]=0
        self.save_clip("0029",b,t)
        cov=self.summarize()["clips"]["0029"]["scheduled_capture_coverage"]
        self.assertEqual(cov["censored_fraction_of_observed_captures"]["value"],1)
        self.assertEqual(cov["available_capture_fraction"]["value"],1)
        note.update(status="invalid_capture_or_weak_covariance");note.pop("observations")
        self.save_clip("0029",b,t)
        result=self.summarize()
        self.assertTrue(result["invalid_or_unknown_decisions_present"])
        self.assertIsNone(result["clips"]["0029"]["scheduled_capture_coverage"]["censored_fraction_of_observed_captures"]["value"])

    def test_unscheduled_missing_capture_is_separate(self):
        b,t=rows()
        t[2]["records"]=[record(measured=False,shadow=True)]
        t[2]["metrics"]["weak_continuation"]["decisions"]=[t[2]["records"][0]["weak_evidence"]]
        self.save_clip("0029",b,t)
        result=self.summarize()["clips"]["0029"]
        self.assertEqual(result["all_frame_capture_coverage"]["missing_capture"],1)
        self.assertEqual(result["scheduled_capture_coverage"].get("missing_capture",0),0)

    def test_symlink_and_audit_missing_binding_rejected(self):
        path=self.root/"0029/independent_audit.json";audit=read(path)
        audit["files_sha256"].pop("shadow/shadow_trace.jsonl");write(path,audit)
        with self.assertRaisesRegex(ValueError,"lacks summary"):
            self.summarize()
        self.save_clip("0029",*rows())
        path=self.root/"0029/clean/frames.jsonl";data=path.read_text();path.unlink()
        target=self.root/"copied.jsonl";target.write_text(data);path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,"symlink"):
            self.summarize()

    def test_strict_json_and_exclusive_output_before_reads(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
            with self.assertRaises(ValueError):m.loads(text)
        output=self.root/"summary.json"
        m.write_summary(self.root,self.freeze_sha,self.plan_sha,output)
        with patch.object(m,"summarize",side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError,"fresh summary"):
                m.write_summary(self.root,self.freeze_sha,self.plan_sha,output)


if __name__ == "__main__":
    unittest.main()
