"""Generated-cache safety tests; never open any experiment's real packets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import run_accuracy_v46_shadow as shadow


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def generated_packet():
    return {"current129": np.full((129,129), 40.), "history129": np.full((8,129,129),40.),
            "prior_centers_xy": np.asarray([[30.+4*i,64.] for i in range(8)]),
            "predicted_offset_xy": np.asarray([.1,-.2])}


def state(clip, frame, track, sample_indices=(), available=True):
    packet = generated_packet()
    geometry = dict(available=available, reasons=[] if available else [
        "fewer_than_five_prior_actual_same_id_measurements"], geometry=dict(
        current_target_fields_accessed=False, current_global_transform_uses_current_whole_frame=True,
        measured_prior_count=8 if available else 2, prior_frame_indices=list(range(frame-8,frame)),
        prior_centers_xy=packet["prior_centers_xy"].tolist(), predicted_offset_xy=packet["predicted_offset_xy"].tolist()))
    return dict(clip=clip,frame_index=frame,segment=0,track_id=track,actual_source_xy=[999.,998.],
                qualified_moving=available,grid_windows=[],reference_samples=list(sample_indices),
                save_audit_inputs=False,geometry=geometry,archive=None if not available else dict(
                    path=f'inputs/{clip}_{frame:04d}_0_{track.replace(":","_")}.npz',sha256="0"*64))


def stage(identity):
    return [identity is not None, identity, [] if identity is None else [identity], False, None]


def generated_scope():
    states = [state("0029",10,"bright:1",[0]), state("0029",11,"bright:2",available=False),
              state("0126",20,"dark:3",[1]), state("0126",21,"dark:4",available=False)]
    samples = []
    for clip,frame,identity in (("0029",10,"0/bright:1"),("0126",20,"0/dark:3"),("0126",216,None)):
        samples.append(["dense",clip,"generated_only",frame,[1234.,2345.],5.,"bright",0,
                        dict(candidate=stage(identity),actual_measurement=stage(identity),baseline_qualified=stage(identity))])
    return dict(states=states, completed_before_any_arm_score=True, all_geometry_exactly_reproduced=True,
                all_seven_v42_saved_inputs_exactly_reproduced=True), dict(
        sample_columns=shadow.SAMPLE_COLUMNS,samples=samples,
        panels={"dense":dict(original_strict_stage="baseline_qualified",samples=3)})


def evidence(*args, arm, **kwargs):
    raw = dict(synthetic_only=True,available=False,reasons=["synthetic_test_unknown"],
        numerical_contrast=None,motion_status="unknown",physical_class="unknown",
        is_motion_or_classification_gate=False,production_changed=False,ambiguity_reasons=["unknown_identity"],
        prior_context={"prior_only":True},components={"unchanged":True},component_bounds={"available":False},
        learned_design_sha256={"moving":"same"},common_support_sha256="same",common_support_count=625)
    return dict(arm=arm,quantity=shadow.ARMS[arm][0],bound_version=shadow.ARMS[arm][1],
                raw_adapter_result=raw,is_motion_or_classification_gate=False,
                motion_status="unknown",physical_class="unknown")


def numerical_evidence(interval=None, available=False):
    value=evidence(arm="presence_box_bounds")
    sign=None if not available else "positive" if interval[0]>0 else "negative" if interval[1]<0 else "unresolved"
    value["raw_adapter_result"].update(available=available, numerical_contrast=dict(
        available=available, interval=interval, coefficient_sign=sign,
        interval_excludes_zero=None if not available else sign!="unresolved",
        numerator=None if not available else sum(interval)/2, error_bound=None if not available else 1.,
        motion_status="unknown",physical_class="unknown",diagnostics={"no_production_gate":True}))
    return value


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cache = self.root/"cache"
        self.output_root = self.root/"v46"
        self.stress = self.output_root/"synthetic_01"
        self.inventory_path = self.root/"inventory.json"
        self.patches = [mock.patch.object(shadow,"ROOT",self.root),
                        mock.patch.object(shadow,"OUTPUT_ROOT",self.output_root),
                        mock.patch.object(shadow,"CACHE",self.cache),
                        mock.patch.object(shadow,"INVENTORY",self.inventory_path),
                        mock.patch.object(shadow,"EXPECTED_COUNTS",{"0029":(2,1),"0126":(2,1)}),
                        mock.patch.object(shadow,"EXPECTED_PANELS",{"dense":3})]
        for patch in self.patches:
            patch.start()
        self.addCleanup(self.temporary.cleanup)
        for patch in self.patches:
            self.addCleanup(patch.stop)

    def create_readiness(self):
        data = self.stress/"synthetic_only.txt"
        data.parent.mkdir(parents=True)
        data.write_text("generated test fixture, not real image data")
        receipt = self.stress/"completion_receipt.json"
        dump(receipt,dict(completed=True,files_sha256={str(data):shadow.sha(data)}))
        audit = self.output_root/"synthetic_independent_audit_01.json"
        dump(audit,dict(passed=True,issues=[],stress_completion_receipt_sha256=shadow.sha(receipt)))
        readiness = dict(completed=True,audited=True,real_shadow_readiness_passed=True,
            diagnostic_shadow_allowed=True,stress_run=str(self.stress),
            stress_completion_receipt_sha256=shadow.sha(receipt),synthetic_audit_path=str(audit),
            synthetic_audit_sha256=shadow.sha(audit),permitted_clips=["0029","0126"],production_changed=False)
        dump(self.output_root/"shadow_readiness_01.json",readiness)
        return readiness

    def create_cache(self):
        manifest, inventory = generated_scope()
        bound = {}
        for item in manifest["states"]:
            if item["archive"] is None:
                continue
            path = self.cache/item["archive"]["path"]
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**generated_packet())
            item["archive"]["sha256"] = shadow.sha(path)
            bound[str(path)] = shadow.sha(path)
        manifest_path = self.cache/"cache_manifest.json"
        dump(manifest_path,manifest)
        dump(self.inventory_path,inventory)
        bound[str(manifest_path)] = shadow.sha(manifest_path)
        bound[str(self.inventory_path)] = shadow.sha(self.inventory_path)
        # This nonexistent media path must never be opened or recursively hashed.
        bound[str(self.root/"never_open_video.avi")] = "f"*64
        receipt_path = self.cache/"completion_receipt.json"
        dump(receipt_path,dict(completed=True,files_sha256=bound))
        pins = dict(receipt=shadow.sha(receipt_path),manifest=shadow.sha(manifest_path),
                    inventory=shadow.sha(self.inventory_path))
        return manifest,inventory,pins

    def test_production_scope_constants_are_predeclared(self):
        # The test patches intentionally shrink generated fixtures, not production.
        source = Path(shadow.__file__).read_text()
        self.assertIn('"0029": (741, 350), "0126": (470, 159)',source)
        self.assertIn('"dense": 285, "pilot": 28, "anchor": 24, "compact_light": 8, "grid": 10',source)

    def test_execute_required_before_any_read(self):
        with mock.patch.object(shadow,"read_json") as read, mock.patch.object(shadow,"sha") as sha:
            with self.assertRaisesRegex(ValueError,"execute"):
                shadow.run(self.output_root/"shadow",self.stress)
        read.assert_not_called(); sha.assert_not_called()

    def test_missing_readiness_prevents_real_metadata_and_packet_access(self):
        with mock.patch.object(shadow,"load_scope_metadata") as scope, mock.patch.object(shadow,"load_packet") as packet:
            with self.assertRaises(ValueError):
                shadow.run(self.output_root/"shadow",self.stress,execute=True)
        scope.assert_not_called(); packet.assert_not_called()

    def test_every_readiness_flag_and_clean_audit_required(self):
        original = self.create_readiness()
        path = self.output_root/"shadow_readiness_01.json"
        for field in ("completed","audited","real_shadow_readiness_passed","diagnostic_shadow_allowed"):
            with self.subTest(field=field):
                dump(path,dict(original,**{field:False}))
                with self.assertRaisesRegex(ValueError,"readiness"):
                    shadow.require_stress_readiness(self.stress)
        dump(path,original)
        audit=self.output_root/"synthetic_independent_audit_01.json"
        dump(audit,dict(passed=True,issues=["not clean"]))
        original["synthetic_audit_sha256"] = shadow.sha(audit); dump(path,original)
        with self.assertRaisesRegex(ValueError,"not clean"):
            shadow.require_stress_readiness(self.stress)

    def test_stress_receipt_cannot_authorize_real_cache_or_journal(self):
        original=self.create_readiness()
        for forbidden in (self.cache/"inputs/0029_0010_0_bright_1.npz",self.root/"frames.jsonl"):
            receipt=self.stress/"completion_receipt.json"
            dump(receipt,dict(completed=True,files_sha256={str(forbidden):"0"*64}))
            original["stress_completion_receipt_sha256"]=shadow.sha(receipt)
            audit=self.output_root/"synthetic_independent_audit_01.json"
            dump(audit,dict(passed=True,issues=[],stress_completion_receipt_sha256=shadow.sha(receipt)))
            original["synthetic_audit_sha256"]=shadow.sha(audit)
            dump(self.output_root/"shadow_readiness_01.json",original)
            with self.assertRaisesRegex(ValueError,"allowlist"):
                shadow.require_stress_readiness(self.stress)

    def test_audit_must_bind_exact_stress_receipt(self):
        readiness=self.create_readiness()
        audit=self.output_root/"synthetic_independent_audit_01.json"
        dump(audit,dict(passed=True,issues=[],stress_completion_receipt_sha256="e"*64))
        readiness["synthetic_audit_sha256"]=shadow.sha(audit)
        dump(self.output_root/"shadow_readiness_01.json",readiness)
        with self.assertRaisesRegex(ValueError,"authorized stress receipt"):
            shadow.require_stress_readiness(self.stress)

    def test_literal_packet_allowlist_rejects_other_clip_traversal_absolute_and_alias(self):
        manifest,_,_=self.create_cache()
        original=manifest["states"][0]
        for bad in ("../inputs/0029_0010_0_bright_1.npz","/tmp/packet.npz",
                    "inputs/0126_0010_0_bright_1.npz","inputs/./0029_0010_0_bright_1.npz"):
            item=deepcopy(original); item["archive"]["path"]=bad
            with self.assertRaisesRegex(ValueError,"exactly"):
                shadow.packet_path(item,self.cache)
        item=deepcopy(original); item["clip"]="0055"
        with self.assertRaisesRegex(ValueError,"allowlist"):
            shadow.packet_path(item,self.cache)

    def test_packet_symlink_escape_rejected(self):
        item=state("0029",10,"bright:1")
        outside=self.root/"outside.npz"; np.savez(outside,**generated_packet())
        target=self.cache/item["archive"]["path"]; target.parent.mkdir(parents=True)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,"nonsymlink"):
            shadow.packet_path(item,self.cache)

    def test_dropped_duplicate_states_and_reference_reassignment_rejected(self):
        manifest,inventory=generated_scope()
        shadow.validate_scope(manifest,inventory)
        for altered in (dict(manifest,states=manifest["states"][:-1]),
                        dict(manifest,states=manifest["states"]+[manifest["states"][0]])):
            with self.assertRaises(ValueError):
                shadow.validate_scope(altered,inventory)
        changed=deepcopy(inventory)
        changed["samples"][0][8]["baseline_qualified"][1]="0/bright:999"
        with self.assertRaisesRegex(ValueError,"assignment"):
            shadow.validate_scope(manifest,changed)
        changed=deepcopy(inventory); changed["samples"][2][8]["actual_measurement"]=stage("0/bright:999")
        with self.assertRaises(ValueError):
            shadow.validate_scope(manifest,changed)

    def test_old_receipt_only_checks_explicit_selected_metadata(self):
        _,_,pins=self.create_cache()
        with mock.patch.object(shadow,"PINNED",pins):
            states,samples,_,_,_=shadow.load_scope_metadata()
        self.assertEqual(len(states),4); self.assertEqual(len(samples),3)
        self.assertFalse((self.root/"never_open_video.avi").exists())

    def test_packet_shape_support_and_geometry_validation(self):
        manifest,_,_=self.create_cache(); item=manifest["states"][0]
        path=shadow.packet_path(item,self.cache)
        packet=shadow.load_packet(path,item)
        self.assertFalse(packet["current129"].flags.writeable)
        changed=deepcopy(item); changed["geometry"]["geometry"]["predicted_offset_xy"]=[0.,0.]
        with self.assertRaisesRegex(ValueError,"geometry"):
            shadow.load_packet(path,changed)
        malformed=generated_packet(); malformed["current129"]=np.ones((25,25))
        np.savez_compressed(path,**malformed)
        item["archive"]["sha256"]=shadow.sha(path)
        with self.assertRaisesRegex(ValueError,"Malformed"):
            shadow.load_packet(path,item)

    def test_decoder_checks_same_bytes_it_will_decode(self):
        manifest,_,_=self.create_cache(); item=manifest["states"][0]
        path=shadow.packet_path(item,self.cache)
        packet=generated_packet(); packet["current129"][0,0]=77.
        np.savez_compressed(path,**packet)
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            shadow.load_packet(path,item)

    def test_origin_only_overrides_preserve_evidence_and_original(self):
        original=evidence(arm="presence_box_bounds")
        original["raw_adapter_result"]["numerical_contrast"]={"available":True,"interval":[1.,2.]}
        frozen=deepcopy(original)
        real=shadow.real_origin_copy(original)
        self.assertEqual(original,frozen)
        self.assertFalse(real["synthetic_only"])
        restored=deepcopy(real["raw_adapter_result"]); restored["synthetic_only"]=True
        self.assertEqual(restored,original["raw_adapter_result"])
        self.assertEqual(len(real["legacy_adapter_reuse"]["origin_only_overrides"]),1)

    def test_current_detection_truth_and_clip_are_not_adapter_arguments(self):
        packet=generated_packet(); item=state("0029",10,"bright:1")
        with mock.patch.object(shadow,"evaluate_causal_probe",side_effect=evidence) as evaluate:
            shadow.evaluate_packet(packet,item)
        self.assertEqual(evaluate.call_count,4)
        for call in evaluate.call_args_list:
            self.assertEqual(len(call.args),5)
            self.assertEqual(set(call.kwargs),{"arm"})
            self.assertIs(call.args[0],packet["current129"])
            self.assertEqual(call.args[4],"bright")

    def test_input_mutation_and_nominal_arm_change_fail_closed(self):
        def mutate(*args,**kwargs):
            args[0][0,0]+=1
            return evidence(*args,**kwargs)
        with mock.patch.object(shadow,"evaluate_causal_probe",side_effect=mutate):
            with self.assertRaisesRegex(ValueError,"mutated"):
                shadow.evaluate_packet(generated_packet(),state("0029",10,"bright:1"))
        def alter(*args,**kwargs):
            value=evidence(*args,**kwargs)
            value["raw_adapter_result"]["common_support_count"]=10 if kwargs["arm"]=="presence_box_bounds" else 625
            return value
        with mock.patch.object(shadow,"evaluate_causal_probe",side_effect=alter):
            with self.assertRaisesRegex(ValueError,"nominal"):
                shadow.evaluate_packet(generated_packet(),state("0029",10,"bright:1"))

    def test_complete_generated_run_freezes_all_packets_and_preserves_unknowns_miss(self):
        self.create_readiness(); _,_,pins=self.create_cache()
        output=self.output_root/"shadow_test"
        def checked(*args,**kwargs):
            self.assertTrue((output/"score_start.json").is_file())
            freeze=shadow.read_json(output/"freeze.json")
            self.assertEqual(len(freeze["packet_sha256"]),2)
            shadow.require_hashes(freeze["packet_sha256"])
            return evidence(*args,**kwargs)
        with mock.patch.object(shadow,"PINNED",pins), mock.patch.object(shadow,"dependency_paths",return_value=[]), \
             mock.patch.object(shadow,"evaluate_causal_probe",side_effect=checked) as evaluate:
            summary=shadow.run(output,self.stress,execute=True)
        self.assertEqual(evaluate.call_count,8)
        self.assertEqual(summary["states"],4); self.assertEqual(summary["packets"],2)
        self.assertEqual(summary["detections_removed"],0)
        for value in summary["arms"].values():
            self.assertEqual(value["unavailable"],4); self.assertEqual(value["negative"],0)
        refs=shadow.read_json(output/"reference_evidence.json")
        self.assertEqual(len(refs["samples"]),3)
        self.assertEqual(refs["samples"][2]["original"]["frame_index"],216)
        self.assertEqual(refs["samples"][2]["measured_alternatives"],[])
        self.assertIsNone(refs["samples"][2]["original_strict_assigned_evidence"])
        self.assertFalse(summary["production_changed"])
        receipt=shadow.read_json(output/"completion_receipt.json")
        shadow.require_hashes(receipt["files_sha256"])
        with self.assertRaises(FileExistsError):
            shadow.run(output,self.stress,execute=True)

    def test_tampered_packet_fails_before_any_score_or_output(self):
        self.create_readiness(); manifest,_,pins=self.create_cache()
        path=self.cache/manifest["states"][0]["archive"]["path"]
        path.write_bytes(b"generated corrupted packet")
        output=self.output_root/"bad"
        with mock.patch.object(shadow,"PINNED",pins), mock.patch.object(shadow,"dependency_paths",return_value=[]), \
             mock.patch.object(shadow,"evaluate_causal_probe") as evaluate:
            with self.assertRaisesRegex(ValueError,"changed"):
                shadow.run(output,self.stress,execute=True)
        evaluate.assert_not_called(); self.assertFalse(output.exists())

    def test_unexpected_solver_error_preserves_failure_not_complete_result(self):
        self.create_readiness(); _,_,pins=self.create_cache(); output=self.output_root/"failure"
        with mock.patch.object(shadow,"PINNED",pins), mock.patch.object(shadow,"dependency_paths",return_value=[]), \
             mock.patch.object(shadow,"evaluate_causal_probe",side_effect=ValueError("generated failure")):
            with self.assertRaisesRegex(ValueError,"generated failure"):
                shadow.run(output,self.stress,execute=True)
        self.assertFalse((output/"completion_receipt.json").exists())
        self.assertFalse(shadow.read_json(output/"failure.json")["completed"])

    def test_fresh_output_scope_guard_precedes_readiness(self):
        with mock.patch.object(shadow,"require_stress_readiness") as readiness:
            for output in (self.root/"outside", self.output_root/"synthetic_01",self.output_root,
                           self.output_root/"synthetic_01"/"nested_overwrite"):
                with self.assertRaisesRegex(ValueError,"dedicated"):
                    shadow.run(output,self.stress,execute=True)
        readiness.assert_not_called()

    def test_conflicting_hashes_cannot_rebind_a_changed_stress_dependency(self):
        bound={"/generated/code.py":"a"*64}
        shadow.merge_bindings(bound,{"/generated/code.py":"a"*64})
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            shadow.merge_bindings(bound,{"/generated/code.py":"b"*64})
        self.assertEqual(bound["/generated/code.py"],"a"*64)

    def test_available_without_numerical_evidence_is_not_counted_as_success(self):
        value=evidence(arm="presence_box_bounds")
        value["raw_adapter_result"]["available"]=True
        with self.assertRaisesRegex(ValueError,"operative"):
            shadow.evidence_summary(value)

    def test_wrapper_adapter_and_numerical_physical_flags_fail_closed(self):
        for layer in ("wrapper","adapter","numerical"):
            for field,bad in (("motion_status","moving"),("physical_class","airborne"),
                              ("is_motion_or_classification_gate",True),("production_changed",True)):
                with self.subTest(layer=layer,field=field):
                    value=numerical_evidence([1.,2.],True)
                    target=value if layer=="wrapper" else value["raw_adapter_result"]
                    if layer=="numerical": target=target["numerical_contrast"]
                    target[field]=bad
                    with self.assertRaises(ValueError): shadow.evidence_summary(value)
        value=numerical_evidence([1.,2.],True)
        value["raw_adapter_result"]["numerical_contrast"]["diagnostics"]["no_production_gate"]=False
        with self.assertRaises(ValueError): shadow.evidence_summary(value)

    def test_intervals_require_finite_ordered_real_endpoints(self):
        for bad in ([float("nan"),2.],[1.,float("inf")],[2.,1.],["1",2.],[True,2.],[1.],None):
            with self.subTest(interval=bad):
                value=numerical_evidence([1.,2.],True)
                value["raw_adapter_result"]["numerical_contrast"]["interval"]=bad
                with self.assertRaisesRegex(ValueError,"interval"):
                    shadow.evidence_summary(value)

    def test_interval_sign_and_zero_exclusion_are_consistent_at_boundaries(self):
        for interval,sign in (([1.,2.],"positive"),([-2.,-1.],"negative"),
                              ([-1.,1.],"unresolved"),([0.,1.],"unresolved"),
                              ([-1.,0.],"unresolved"),([0.,0.],"unresolved")):
            value=numerical_evidence(interval,True)
            self.assertEqual(shadow.evidence_summary(value)["coefficient_sign"],sign)
            for field,bad in (("coefficient_sign","wrong"),("interval_excludes_zero",sign=="unresolved")):
                altered=deepcopy(value); altered["raw_adapter_result"]["numerical_contrast"][field]=bad
                with self.assertRaisesRegex(ValueError,"disagree"):
                    shadow.evidence_summary(altered)

    def test_unavailable_numerical_record_is_unknown_not_an_operative_interval(self):
        value=numerical_evidence()
        self.assertFalse(shadow.evidence_summary(value)["available"])
        for field,bad in (("interval",[0.,0.]),("coefficient_sign","unresolved"),
                          ("interval_excludes_zero",False),("numerator",0.),("error_bound",1.)):
            altered=deepcopy(value); altered["raw_adapter_result"]["numerical_contrast"][field]=bad
            with self.assertRaisesRegex(ValueError,"Unavailable"):
                shadow.evidence_summary(altered)
        altered=deepcopy(value); altered["raw_adapter_result"]["available"]=True
        with self.assertRaisesRegex(ValueError,"availability disagrees"):
            shadow.evidence_summary(altered)

    def test_packet_evaluation_checks_wrapper_and_numeric_flags_before_emission(self):
        def wrong(*args,**kwargs):
            value=evidence(*args,**kwargs); value["physical_class"]="airborne"
            return value
        with mock.patch.object(shadow,"evaluate_causal_probe",side_effect=wrong):
            with self.assertRaisesRegex(ValueError,"wrapper"):
                shadow.evaluate_packet(generated_packet(),state("0029",10,"bright:1"))


if __name__ == "__main__":
    unittest.main()
