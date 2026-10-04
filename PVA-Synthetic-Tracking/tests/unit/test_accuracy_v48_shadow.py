"""Generated-only V48 readiness, reuse and reference-accounting checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import run_accuracy_v48_shadow as shadow
from accuracy_v47_probe import CURRENT_USE_DESCRIPTION
from test_accuracy_v46_shadow import generated_scope,generated_packet,evidence


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def old_value(arm):
    value=evidence(arm=arm);raw=value["raw_adapter_result"]
    raw.update(synthetic_only=False,conditional_on=["fixed_geometry"],uncertainty_excludes=["physical_identity"])
    raw["prior_context"]["current_values_used_for"]="response only"
    return value


def new_value(*args,**kwargs):
    value=old_value("presence_box_bounds");raw=value["raw_adapter_result"]
    raw["synthetic_only"]=True
    raw["conditional_on"].append("guard_background_validity")
    raw["ambiguity_reasons"].append("guard_transfer_unverified")
    old_use=raw["prior_context"]["current_values_used_for"]
    raw["prior_context"]["current_values_used_for"]=CURRENT_USE_DESCRIPTION
    value.update(arm=shadow.ARM,synthetic_only=True,production_changed=False,
        gain_calibration=None,uncertainty_excludes_guard_contamination_and_spatial_transfer_failure=True,
        old_unrestricted_background_estimand_preserved=False)
    value["current_use_metadata_override"]={"path":"raw_adapter_result.prior_context.current_values_used_for",
                                           "from":old_use,"to":CURRENT_USE_DESCRIPTION}
    return value


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.output=self.root/"v48";self.cache=self.root/"cache";self.v46=self.root/"v46"
        self.patches=[mock.patch.object(shadow,"ROOT",self.root),mock.patch.object(shadow,"OUTPUT_ROOT",self.output),
            mock.patch.object(shadow,"V46",self.v46),mock.patch.object(shadow.old,"CACHE",self.cache),
            mock.patch.object(shadow.old,"EXPECTED_COUNTS",{"0029":(2,1),"0126":(2,1)})]
        for p in self.patches:p.start()
        self.addCleanup(self.temp.cleanup)
        for p in self.patches:self.addCleanup(p.stop)

    def ready(self):
        code=self.root/"scripts/generated_model.py";code.parent.mkdir();code.write_text("# generated fixture\n")
        receipt=self.output/"synthetic_01/completion_receipt.json"
        digest=shadow.sha(code)
        dump(receipt,dict(completed=True,synthetic_only=True,files_sha256={str(code):digest}))
        math=self.output/"math_independent_audit_01.json"
        audit=self.output/"synthetic_independent_audit_01.json"
        dump(math,dict(completed=True,passed=True,issues=[],audit_files_sha256={str(code):digest}))
        dump(audit,dict(completed=True,passed=True,issues=[],audit_files_sha256={str(code):digest},synthetic_completion_receipt_sha256=shadow.sha(receipt)))
        readiness=dict(completed=True,audited=True,diagnostic_shadow_allowed=True,production_changed=False,
            permitted_clips=["0029","0126"])
        for key,path in (("synthetic_receipt",receipt),("math_audit",math),("synthetic_audit",audit)):
            readiness[key+"_path"]=str(path);readiness[key+"_sha256"]=shadow.sha(path)
        dump(self.output/"shadow_readiness_01.json",readiness)
        return readiness

    def baseline(self):
        manifest,inventory=generated_scope();states=manifest["states"];packets={}
        rows=[]
        for state in states:
            if state["archive"] is not None:
                p=self.cache/state["archive"]["path"];p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,**generated_packet());state["archive"]["sha256"]=shadow.sha(p)
                packets[str(p)]=shadow.sha(p)
            row=deepcopy(state);row["arms"]={a:None if state["archive"] is None else old_value(a) for a in shadow.old.ARMS}
            rows.append(row)
        samples=[dict(inventory_index=i,original=dict(zip(shadow.old.SAMPLE_COLUMNS,r))) for i,r in enumerate(inventory["samples"])]
        refs=shadow.old.reference_report(samples,inventory["panels"],rows)
        summary=shadow.old.summarize(rows,states,refs)
        dump(self.v46/"reference_evidence.json",refs)
        return states,rows,refs,summary,{},packets,packets.copy()

    def test_no_execute_or_missing_readiness_never_loads_baseline_or_packets(self):
        with mock.patch.object(shadow,"load_baseline") as baseline,mock.patch.object(shadow.old,"load_packet") as packet:
            with self.assertRaisesRegex(ValueError,"execution"):
                shadow.run(self.output/"shadow")
            with self.assertRaises(ValueError):shadow.run(self.output/"shadow",execute=True)
        baseline.assert_not_called();packet.assert_not_called()

    def test_readiness_rejects_changed_flags_audits_and_wrong_receipt_binding(self):
        record=self.ready();path=self.output/"shadow_readiness_01.json"
        shadow.require_readiness()
        for key in ("completed","audited","diagnostic_shadow_allowed"):
            dump(path,dict(record,**{key:False}))
            with self.assertRaisesRegex(ValueError,"readiness"):
                shadow.require_readiness()
        audit=self.output/"synthetic_independent_audit_01.json"
        dump(audit,dict(completed=True,passed=True,issues=[],synthetic_completion_receipt_sha256="0"*64))
        record["synthetic_audit_sha256"]=shadow.sha(audit);dump(path,record)
        with self.assertRaisesRegex(ValueError,"approved run"):
            shadow.require_readiness()

    def test_math_audit_must_match_frozen_code_not_just_say_pass(self):
        record=self.ready();path=self.output/"math_independent_audit_01.json"
        dump(path,dict(completed=True,passed=True,issues=[],audit_files_sha256={str(self.root/"scripts/generated_model.py"):"0"*64}))
        record["math_audit_sha256"]=shadow.sha(path);dump(self.output/"shadow_readiness_01.json",record)
        with self.assertRaisesRegex(ValueError,"frozen implementation"):
            shadow.require_readiness()

    def test_synthetic_audit_must_bind_its_frozen_source(self):
        record=self.ready();path=self.output/"synthetic_independent_audit_01.json"
        report=shadow.read_json(path)
        for bindings in ({}, {record["synthetic_receipt_path"]:record["synthetic_receipt_sha256"]},
                         {str(self.root/"scripts/generated_model.py"):"0"*64}):
            dump(path,dict(report,audit_files_sha256=bindings))
            record["synthetic_audit_sha256"]=shadow.sha(path)
            dump(self.output/"shadow_readiness_01.json",record)
            with self.assertRaisesRegex(ValueError,"bound implementation|frozen implementation"):
                shadow.require_readiness()

    def test_each_audit_must_complete_and_pass_with_no_issues(self):
        record=self.ready()
        for key in ("math_audit","synthetic_audit"):
            path=Path(record[key+"_path"]);original=shadow.read_json(path)
            for change in ({"completed":False},{"passed":False},{"issues":["unresolved"]}):
                dump(path,dict(original,**change));record[key+"_sha256"]=shadow.sha(path)
                dump(self.output/"shadow_readiness_01.json",record)
                with self.assertRaisesRegex(ValueError,"did not pass"):
                    shadow.require_readiness()
            dump(path,original);record[key+"_sha256"]=shadow.sha(path)
            dump(self.output/"shadow_readiness_01.json",record)

    def test_audits_can_bind_exact_receipt_without_a_self_hash_cycle(self):
        record=self.ready();receipt=record["synthetic_receipt_path"]
        for key in ("math_audit","synthetic_audit"):
            path=Path(record[key+"_path"]);report=shadow.read_json(path)
            report["audit_files_sha256"][receipt]=record["synthetic_receipt_sha256"]
            dump(path,report);record[key+"_sha256"]=shadow.sha(path)
        dump(self.output/"shadow_readiness_01.json",record)
        shadow.require_readiness()
        path=Path(record["synthetic_audit_path"]);report=shadow.read_json(path)
        report["audit_files_sha256"][receipt]="0"*64
        dump(path,report);record["synthetic_audit_sha256"]=shadow.sha(path)
        dump(self.output/"shadow_readiness_01.json",record)
        with self.assertRaisesRegex(ValueError,"approved receipt"):
            shadow.require_readiness()

    def test_forbidden_synthetic_dependency_is_rejected_before_hashing_its_bytes(self):
        record=self.ready();receipt=Path(record["synthetic_receipt_path"])
        forbidden=self.root/"journal_payload.npy"
        saved=shadow.read_json(receipt);saved["files_sha256"][str(forbidden)]="0"*64
        dump(receipt,saved);record["synthetic_receipt_sha256"]=shadow.sha(receipt)
        audit=Path(record["synthetic_audit_path"]);report=shadow.read_json(audit)
        report["synthetic_completion_receipt_sha256"]=record["synthetic_receipt_sha256"]
        dump(audit,report);record["synthetic_audit_sha256"]=shadow.sha(audit)
        dump(self.output/"shadow_readiness_01.json",record)
        with mock.patch.object(shadow,"sha",wraps=shadow.sha) as hashes:
            with self.assertRaisesRegex(ValueError,"allowlist"):
                shadow.require_readiness()
        self.assertNotIn(forbidden,[Path(call.args[0]) for call in hashes.call_args_list])

    def test_only_two_literal_predecessor_reports_are_permitted(self):
        record=self.ready();receipt=Path(record["synthetic_receipt_path"])
        previous=self.root/"results/tiny_target/accuracy_v47_20260926"
        saved=shadow.read_json(receipt)
        for name,passed in (("synthetic_independent_audit_01.json",False),("math_independent_audit_01.json",True)):
            path=previous/name;dump(path,dict(completed=True,passed=passed))
            saved["files_sha256"][str(path)]=shadow.sha(path)
        audit=Path(record["synthetic_audit_path"])
        def refresh():
            dump(receipt,saved);record["synthetic_receipt_sha256"]=shadow.sha(receipt)
            report=shadow.read_json(audit)
            report["synthetic_completion_receipt_sha256"]=record["synthetic_receipt_sha256"]
            dump(audit,report);record["synthetic_audit_sha256"]=shadow.sha(audit)
            dump(self.output/"shadow_readiness_01.json",record)
        refresh();shadow.require_readiness()
        saved["files_sha256"][str(previous/"unapproved_other_report.json")]="0"*64
        refresh()
        with self.assertRaisesRegex(ValueError,"allowlist"):
            shadow.require_readiness()

    def test_changed_bound_source_rejected_before_baseline_or_packet_reads(self):
        self.ready();code=self.root/"scripts/generated_model.py"
        code.write_text("# changed generated fixture\n")
        with mock.patch.object(shadow,"load_baseline") as baseline, \
             mock.patch.object(shadow.old,"load_packet") as packets:
            with self.assertRaisesRegex(ValueError,"changed"):
                shadow.run(self.output/"shadow",execute=True)
        baseline.assert_not_called();packets.assert_not_called()

    def test_baseline_reads_only_explicit_compact_files_not_recursive_receipt(self):
        states,rows,refs,summary,_,packets,_=self.baseline()
        samples=[dict(inventory_index=s["inventory_index"],original=s["original"]) for s in refs["samples"]]
        panels=generated_scope()[1]["panels"]
        dump(self.v46/"selected_ledger.json",dict(states=states,reference_samples=samples,inherited_reference_panels=panels))
        dump(self.v46/"summary.json",summary)
        (self.v46/"states.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        names=("states.jsonl","selected_ledger.json","reference_evidence.json","summary.json")
        files={str(self.v46/name):shadow.sha(self.v46/name) for name in names}
        forbidden=self.root/"never_read_journal_payload"
        files[str(forbidden)]="0"*64;files.update(packets)
        receipt=self.v46/"completion_receipt.json";dump(receipt,dict(completed=True,files_sha256=files))
        with mock.patch.object(shadow,"V46_RECEIPT_SHA",shadow.sha(receipt)), \
             mock.patch.object(shadow.old,"load_scope_metadata",return_value=(states,samples,panels,{},packets)), \
             mock.patch.object(shadow.old,"load_packet") as packet:
            actual=shadow.load_baseline()
        self.assertEqual(actual[:4],(states,rows,refs,summary));packet.assert_not_called()

    def test_output_scope_rejected_before_readiness(self):
        with mock.patch.object(shadow,"require_readiness") as readiness:
            for path in (self.root/"outside",self.output/"synthetic_01",self.output/"synthetic_01/child"):
                with self.assertRaisesRegex(ValueError,"dedicated"):
                    shadow.run(path,execute=True)
        readiness.assert_not_called()

    def test_origin_override_changes_no_numeric_or_guard_fields(self):
        source=new_value();source["gain_calibration"]={"unknown":"unaltered"};before=deepcopy(source)
        value=shadow.real_origin_copy(source)
        self.assertEqual(source,before)
        self.assertFalse(value["synthetic_only"]);self.assertFalse(value["raw_adapter_result"]["synthetic_only"])
        value["raw_adapter_result"]["synthetic_only"]=True
        self.assertEqual(value["raw_adapter_result"],source["raw_adapter_result"])
        self.assertEqual(value["gain_calibration"],source["gain_calibration"])

    def test_generated_full_run_preserves_four_arms_unknowns_and_miss(self):
        self.ready();baseline=self.baseline();out=self.output/"shadow_test"
        def checked(*args,**kwargs):
            self.assertEqual(len(args),5);self.assertFalse(kwargs)
            freeze=shadow.read_json(out/"freeze.json")
            self.assertEqual(len(freeze["packet_sha256"]),2)
            self.assertTrue((out/"score_start.json").is_file())
            return new_value()
        with mock.patch.object(shadow,"load_baseline",return_value=baseline), \
             mock.patch.object(shadow,"source_dependencies",return_value=[]), \
             mock.patch.object(shadow,"evaluate_probe",side_effect=checked) as evaluate:
            summary=shadow.run(out,execute=True)
        self.assertEqual(evaluate.call_count,2);self.assertEqual(summary["states"],4)
        self.assertEqual(summary["arms"][shadow.ARM]["unavailable"],4)
        self.assertEqual(summary["arms"][shadow.ARM]["negative"],0)
        refs=shadow.read_json(out/"reference_evidence.json")
        self.assertIsNone(refs["samples"][-1]["original_strict_assigned_evidence"])
        self.assertEqual(refs["samples"][-1]["original"]["frame_index"],216)
        receipt=shadow.read_json(out/"completion_receipt.json");shadow.require_hashes(receipt["files_sha256"])
        with self.assertRaises(FileExistsError):shadow.run(out,execute=True)

    def test_corrupted_packet_fails_before_scoring_or_output(self):
        self.ready();baseline=self.baseline();p=next(iter(baseline[-1]));Path(p).write_bytes(b"corrupted fixture")
        out=self.output/"bad"
        with mock.patch.object(shadow,"load_baseline",return_value=baseline), \
             mock.patch.object(shadow,"source_dependencies",return_value=[]), \
             mock.patch.object(shadow,"evaluate_probe") as evaluate:
            with self.assertRaisesRegex(ValueError,"changed"):
                shadow.run(out,execute=True)
        evaluate.assert_not_called();self.assertFalse(out.exists())

    def test_original_arm_or_metadata_mutation_is_rejected(self):
        _,baseline,refs,summary,*_=self.baseline();rows=deepcopy(baseline)
        for r in rows:r["arms"][shadow.ARM]=None
        rows[0]["qualified_moving"]=not rows[0]["qualified_moving"]
        with self.assertRaisesRegex(ValueError,"modified"):
            shadow.summarize(rows,baseline,summary,refs)

    def test_packet_wrapper_never_passes_truth_or_ref_to_adapter(self):
        state=generated_scope()[0]["states"][0]
        with mock.patch.object(shadow,"evaluate_probe",side_effect=new_value) as score:
            value=shadow.evaluate_packet(generated_packet(),state)
        self.assertEqual(len(score.call_args.args),5);self.assertFalse(score.call_args.kwargs)
        self.assertFalse(value["synthetic_only"])


if __name__=="__main__":unittest.main()
