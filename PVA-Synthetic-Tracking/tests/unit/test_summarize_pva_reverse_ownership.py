"""Generated metadata fixtures only; no accelerator, media or model fits."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


M=imported("reverse_summary",ROOT/"scripts/summarize_pva_reverse_ownership.py")
H=M.load_helper(ROOT/"scripts/summarize_pva_robustness.py")
W=imported("reverse_worker_metadata",ROOT/"scripts/probe_pva_reverse_ownership.py")
F=imported("old_generated_summary_fixtures",ROOT/"tests/unit/test_summarize_pva_robustness.py")
desc=F.desc


def packet(value,identity,kind="U8"):
    return dict(array=desc(value),metadata=dict(id=identity,size=len(value),capacity=384,type=kind))


def row(case=None,arm="shared",mode="trace",recovery=False):
    case=copy.deepcopy(case or M.inventory(H)[0])
    failed=case["id"] in M.FAILED and not (arm=="copied" and recovery)
    value=F.record(case,2,"trace",slots=list(range(20)) if failed else None)
    value.update(schema="seaqr.pva-reverse-ownership.v1",arm=arm,mode=mode)
    value["canonical_nontiming"]["effective_motion_configuration"]=copy.deepcopy(M.EXPECTED_MOTION)
    value["canonical_nontiming"]["global_configuration"]=copy.deepcopy(H.EXPECTED_GLOBAL)
    corr=value["canonical_nontiming"]["result"]["correspondence"]
    data=value["capture"]["data"]
    if corr is not None:
        final=data["final_filter"];p=H.array(final["previous_full_points"]);q=H.array(final["current_full_points"])
        mask=H.array(final["accepted_mask"]);count=len(p)
        forward=np.zeros(count,np.uint8);backward=(~mask).astype(np.uint8)
        qproxy=(q+.5)/2-.5
        pproxy=(p+.5)/2-.5
        after_forward=backward if arm!="copied" else forward
        data["after_forward"].update(tracked_points=desc(qproxy))
        data["after_backward"].update(forward_status=desc(after_forward),backward_status_vpi=desc(backward),
            backward_points_vpi=desc(pproxy),tracked_points=desc(qproxy),
            estimator_local_backward_initial_status_is_forward_status=True,
            actual_backward_argument_is_forward_status=arm=="shared",actual_backward_argument_is_clone=arm=="copied")
        final["backward_full_points"]=desc(p)
        # Attribution may change without any returned measurement changing.
        corr["metrics"]["rejections"]={"lost_forward_status":0 if arm=="copied" else int((~mask).sum()),
            "lost_backward_status":int((~mask).sum()) if arm=="copied" else 0}
        if arm=="plain":corr["metrics"]["rejections"]=dict(lost_forward_status=int((~mask).sum()),lost_backward_status=0)
        source=packet(forward,1);clone=packet(forward,2);points=packet(qproxy,3,"KEYPOINT_F32")
        audit=dict(enabled=True,arm=arm,constructor_calls=2,
            forward=dict(keyword_names=["backend"],explicit_kptstatus=False,pyramid=data["pyramids"]["previous"],
                keypoints=packet(pproxy,4,"KEYPOINT_F32")),
            reverse=dict(keyword_names=["backend","kptstatus"],backend="PVA",pyramid=data["pyramids"]["current"],
                keypoints_before=points,source_status_before=source,clone_status_before=clone,
                active_status_bytes_equal=True,size_capacity_type_equal=True,cloned_in_both_arms=True,
                nonzero_flags_preserved=True,constructor_context_changed=False,
                chosen_status="clone" if arm=="copied" else "source",
                actual_argument_is_forward_status=arm=="shared",actual_argument_is_clone=arm=="copied",
                actual_argument_metadata=clone["metadata"] if arm=="copied" else source["metadata"]),
            after=dict(source_status=packet(after_forward,1),clone_status=packet(backward if arm=="copied" else forward,2),
                keypoints=points,clone_retained=True),clone_retained_until_estimator_close=True)
    else:audit=dict(enabled=arm!="plain",arm=arm,constructor_calls=0,forward=None,reverse=None,after=None)
    if arm=="plain":audit=dict(enabled=False,arm=arm,constructor_calls=0,forward=None,reverse=None,after=None)
    value["constructor_audit"]=audit
    value["observation_contract"]=dict(method_sha256="bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b",
        estimator_source_changed=False,sys_settrace=mode=="trace",forward_initializer_changed=False,
        constructor_context_changed=False,reverse_status_object_substitution=arm=="copied",
        matched_constructor_readbacks_and_copy=arm!="plain",extra_pva_compute_calls=0)
    if mode=="base":value["capture"]=None
    value["canonical_nontiming_sha256"]=H.canonical_sha(value["canonical_nontiming"])
    return value


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


def fixture(root,recovery=False,mutate=None):
    directory,bundle=root/"evidence",root/"bundle"
    directory.mkdir();bundle.mkdir()
    for name in M.SOURCES:
        original=ROOT/("tests/unit" if name.startswith("test_") else "scripts")/name
        if name in {"summarize_pva_reverse_ownership.py","batch_discovery_pair.py"}:
            (bundle/name).write_bytes(original.read_bytes())
        else:(bundle/name).write_text("Generated source fixture: "+name)
    frozen=W.freeze_spec();frozen["source_sha256"]={name:M.sha(bundle/name) for name in M.SOURCES}
    write(directory/"freeze.json",frozen);digest=M.sha(directory/"freeze.json")
    batch=dict(schema="seaqr.pva-reverse-ownership.batch.v1",complete=True,execution_passed=True,
        generated_only=True,camera_media_accessed=False,clock_writes=False,error=None,not_run=[],current=None,
        freeze_sha256=digest,source_sha256=frozen["source_sha256"],elapsed_seconds=30.,phases=[],
        execution=dict(workers=1,phases=30,phase_deadline_seconds=900,batch_deadline_seconds=3600,
            start_below_celsius=65,stop_at_celsius=75,automatic_retries=0))
    parity=dict(schema="seaqr.pva-reverse-ownership.batch.v1.parity",cases=[],controls=[])
    workspace="/tmp/seaqr_pva_reverse_20261001_Ab1234"
    for case in M.inventory(H):
        records={}
        for arm,mode in M.ROLES:
            value=row(case,arm,mode,recovery)
            value["input_sha256"]=dict(freeze_sha256=digest,source_sha256=frozen["source_sha256"],
                reference=M.REFERENCE,reference_source_sha256=M.REFERENCE_SOURCES)
            if mutate:mutate(value)
            value["canonical_nontiming_sha256"]=H.canonical_sha(value["canonical_nontiming"])
            records[arm,mode]=value
            name=M.phase_name(case["id"],arm,mode);write(directory/(name+".json"),value)
            command=["/usr/bin/python3","-I",workspace+"/probe_pva_reverse_ownership.py","--workspace",workspace,
                "--freeze",workspace+"/freeze.json","--freeze-sha256",digest,"--case",case["id"],"--arm",arm]
            if mode=="trace":command.append("--trace")
            batch["phases"].append(dict(name=name,case=case["id"],arm=arm,mode=mode,pid=1000+len(batch["phases"]),
                returncode=0,elapsed_seconds=1.,result_sha256=M.sha(directory/(name+".json")),command=command))
        for arm in ("shared","copied"):
            base,trace=records[arm,"base"],records[arm,"trace"]
            same=base["canonical_nontiming"]==trace["canonical_nontiming"]
            parity["cases"].append(dict(case=case["id"],arm=arm,passed=same,exact_nontiming_parity=same,
                baseline_sha256=base["canonical_nontiming_sha256"],trace_sha256=trace["canonical_nontiming_sha256"],
                timings_excluded=True,trace_may_explain_baseline_only_if_passed=same))
        plain,shared=records["plain","base"],records["shared","base"]
        same=plain["canonical_nontiming"]==shared["canonical_nontiming"]
        parity["controls"].append(dict(case=case["id"],passed=same,exact_nontiming_parity=same,
            plain_sha256=plain["canonical_nontiming_sha256"],shared_sha256=shared["canonical_nontiming_sha256"]))
    parity["passed"]=all(p["passed"] for p in parity["cases"])
    parity["controls_passed"]=all(p["passed"] for p in parity["controls"])
    batch["parity_passed"]=parity["passed"];batch["control_parity_passed"]=parity["controls_passed"]
    write(directory/"parity.json",parity);batch["parity_sha256"]=M.sha(directory/"parity.json")
    write(directory/"batch_status.json",batch)
    return directory,bundle,digest


def summary(directory,bundle,digest):
    return M.summarize(directory,digest,bundle,ROOT/"scripts/summarize_pva_robustness.py")


class Numerical(unittest.TestCase):
    def test_independent_freeze_inventory(self):
        frozen=W.freeze_spec();frozen["source_sha256"]={name:"a"*64 for name in M.SOURCES}
        frozen["source_sha256"]["batch_discovery_pair.py"]=M.SAFETY_SHA
        M.validate_freeze(H,frozen)
        self.assertEqual(M.inventory(H),W.inventory())
        frozen["protocol"]["forward_status_initialization_unchanged"]=False
        with self.assertRaises(ValueError):M.validate_freeze(H,frozen)

    def test_constructor_contract_both_arms(self):
        a,b=row(),row(arm="copied")
        for value in (a,b):self.assertTrue(M.constructor_check(H,value)["independent_clone_identity"])
        self.assertTrue(M.matched_constructor_schedule(H,a,b)["compared"])
        self.assertTrue(M.preintervention_identity(H,a,b)["selected_seeds_exact"])

    def test_no_zero_reset_alias_lifetime_or_context_change(self):
        mutations=(lambda a:a["reverse"].update(constructor_context_changed=True),
            lambda a:a["reverse"]["clone_status_before"]["metadata"].update(id=1),
            lambda a:a["reverse"].update(actual_argument_is_clone=True),
            lambda a:a["reverse"]["clone_status_before"].update(array=desc(np.ones(35,np.uint8))),
            lambda a:a["after"].update(clone_retained=False))
        for mutate in mutations:
            value=row();mutate(value["constructor_audit"])
            with self.assertRaises(ValueError):M.constructor_check(H,value)

    def test_preintervention_point_status_pyramid_and_seed_tamper(self):
        mutations=(lambda d:d["after_forward"].update(forward_status=desc(np.ones(35,np.uint8))),
            lambda d:d["after_forward"].update(tracked_points=desc(np.zeros((35,2),np.float32))),
            lambda d:d["before_forward"].update(selected_indices=desc(np.arange(35,dtype=np.int64))),
            lambda d:d["pyramids"]["current"][1].update(array=desc(np.zeros((128,160),np.uint8))))
        for mutate in mutations:
            a,b=row(),row(arm="copied");mutate(b["capture"]["data"])
            with self.assertRaises(ValueError):M.preintervention_identity(H,a,b)

    def test_classification_only_not_measurement_recovery(self):
        result=M.changes(H,row(),row(arm="copied"))
        self.assertTrue(result["only_rejection_classification_changed"])
        self.assertFalse(result["scientific_measurements_changed"])
        self.assertTrue(result["reverse_outputs_exact"])
        self.assertEqual(M.mask_counterfactual(H,row())["recovered_count"],0)

    def test_counterfactual_holds_backward_fixed_and_rejects_nonfinite(self):
        value=row();data=value["capture"]["data"]
        # The altered current mask is explanatory metadata, not a refitted model.
        mask=H.array(data["final_filter"]["accepted_mask"]).copy();mask[0]=False
        data["final_filter"]["accepted_mask"]=desc(mask)
        flags=H.array(data["after_backward"]["forward_status"]).copy();flags[0]=1
        data["after_backward"]["forward_status"]=desc(flags)
        result=M.mask_counterfactual(H,value)
        self.assertEqual(result["recovered_selected_slots"],[0]);self.assertEqual(result["fits_performed"],0)
        back=H.array(data["final_filter"]["backward_full_points"]).copy();back[0]=np.nan
        data["final_filter"]["backward_full_points"]=desc(back)
        self.assertEqual(M.mask_counterfactual(H,value)["recovered_count"],0)

    def test_no_fit_unknown_and_empty_nonvacuous(self):
        flat=row(M.inventory(H)[-1]);analysis,_=H.accepted_analysis(flat["canonical_nontiming"],flat["case_metadata"])
        self.assertTrue(M.case_gates(flat["case_metadata"],analysis)["passed"])
        self.assertIsNone(analysis["original_fit"]["accepted"])
        self.assertIsNone(M.mask_counterfactual(H,flat)["recovered_count"])
        value=F.record(M.inventory(H)[0],slots=[]);analysis,_=H.accepted_analysis(value["canonical_nontiming"],value["case_metadata"])
        self.assertFalse(M.case_gates(value["case_metadata"],analysis)["passed"])

    def test_native_point_guard_all_accepted_not_just_inliers(self):
        value=F.record(M.inventory(H)[2],error=.5)
        analysis,_=H.accepted_analysis(value["canonical_nontiming"],value["case_metadata"])
        self.assertEqual(analysis["original_fit"]["translation_error_px"],.5)
        self.assertFalse(M.case_gates(value["case_metadata"],analysis)["passed"])

    def test_saved_generated_focal_counterfactual_no_recovery(self):
        for name in M.CASES[:-1]:
            path=ROOT.parent/"outputs/seaqr_pva_robustness_20261001/jetson"/(name+"_depth2_trace.json")
            if not path.is_file():self.skipTest("Prior generated metadata not bundled")
            value=H.read(path);result=M.mask_counterfactual(H,value)
            self.assertEqual(result["recovered_count"],0,name);self.assertEqual(result["removed_count"],0,name)


class Loading(unittest.TestCase):
    def test_full30_no_effect_classification_only_and_serialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();directory,bundle,digest=fixture(root)
            output=root/"summary.json"
            result=M.save_summary(directory,digest,bundle,ROOT/"scripts/summarize_pva_robustness.py",output)
            self.assertEqual(result["children_completed"],30);self.assertEqual(result["identity_failures"],[])
            self.assertEqual(result["decision"]["outcome"],"no_effect_stop")
            self.assertEqual(H.read(output),result)
            self.assertTrue(result["cases"][0]["changes"]["only_rejection_classification_changed"])
            with self.assertRaisesRegex(ValueError,"overwrite"):
                M.save_summary(directory,digest,bundle,ROOT/"scripts/summarize_pva_robustness.py",output)

    def test_both_recoveries_control_preservation_only_full_regression_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            result=summary(*fixture(Path(folder),recovery=True))
            self.assertEqual(result["identity_failures"],[])
            self.assertTrue(result["decision"]["ready_for_full_generated_regression"])
            self.assertFalse(result["production_promotion"])
            self.assertEqual(result["decision"]["changed_measurement_case_ids"],list(M.FAILED))

    def test_plain_control_or_trace_mismatch_prevents_interpretation(self):
        for arm,mode in (("plain","base"),("shared","trace")):
            def mutate(value):
                if value["case"]==M.CASES[0] and (value["arm"],value["mode"])==(arm,mode):
                    value["canonical_nontiming"]["accepted_truth"]["fixture_observer_difference"]=True
            with tempfile.TemporaryDirectory() as folder:
                result=summary(*fixture(Path(folder),recovery=True,mutate=mutate))
                self.assertEqual(result["decision"]["outcome"],"inconclusive_stop")
                self.assertIsNone(result["cases"][0]["changes"])

    def test_wrong_global_nonfinite_and_changed_original_gate_rejected(self):
        for mutate in (
            lambda value:value.update(closed=False),
            lambda value:value["canonical_nontiming"]["effective_motion_configuration"].update(max_displacement_px=121.),
            lambda value:value["canonical_nontiming"]["global_configuration"].update(maximum_median_reprojection_px=.5),
        ):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):summary(*fixture(Path(folder),mutate=mutate))

    def test_shared_positive_baseline_must_reproduce_even_if_copied_passes(self):
        def mutate(value):
            if value["case"]==M.POSITIVE_CONTROLS[0] and value["arm"]!="copied":
                fit=value["canonical_nontiming"]["result"]["fit"]
                fit["quality_status"]="rejected";fit["rejection_reasons"]=["fixture_baseline_failure"]
        with tempfile.TemporaryDirectory() as folder:
            result=summary(*fixture(Path(folder),recovery=True,mutate=mutate))
            self.assertEqual(result["decision"]["outcome"],"inconclusive_stop")
            self.assertEqual(result["decision"]["copied_failed_case_ids"],[])
            self.assertTrue(any("baseline control did not reproduce" in f["reason"] for f in result["identity_failures"]))

    def test_hash_missing_process_and_command_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            directory,bundle,digest=fixture(Path(folder));batch=H.read(directory/"batch_status.json")
            path=directory/(batch["phases"][0]["name"]+".json")
            original=path.read_bytes();path.write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError,"hash differs"):summary(directory,bundle,digest)
            path.write_bytes(original)
            batch["phases"][0]["command"][-1]="copied";write(directory/"batch_status.json",batch)
            with self.assertRaisesRegex(ValueError,"command"):summary(directory,bundle,digest)
            batch["phases"].pop();write(directory/"batch_status.json",batch)
            with self.assertRaisesRegex(ValueError,"30 fresh"):summary(directory,bundle,digest)

    def test_hash_mutation_during_read(self):
        with tempfile.TemporaryDirectory() as folder:
            directory,bundle,digest=fixture(Path(folder));original=H.read
            def changing(path):
                value=original(path)
                if Path(path).name=="batch_status.json":Path(path).write_bytes(Path(path).read_bytes()+b" ")
                return value
            with patch.object(M,"load_helper",return_value=H),patch.object(H,"read",side_effect=changing):
                with self.assertRaisesRegex(ValueError,"during read"):summary(directory,bundle,digest)

    def test_decision_requires_all_recoveries_and_controls(self):
        cases=[]
        for case in M.inventory(H):
            cases.append(dict(case=case,copied=dict(gates=dict(passed=True)),changes=dict(scientific_measurements_changed=True)))
        cases[2]["copied"]["gates"]["passed"]=False
        self.assertEqual(M.decision(cases,[])["outcome"],"insufficient_recovery_stop")
        self.assertEqual(M.decision(cases,["identity"])["outcome"],"inconclusive_stop")


if __name__=="__main__":unittest.main()
