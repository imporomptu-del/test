"""Scoped depth supervision, callback restoration and fail-fast tests."""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
m=module("depth_batch_test",ROOT/"scripts/batch_pva_depth_control.py")
old=m.imported(ROOT/"scripts/batch_static_pva_texture.py",m.REFERENCE_BATCH_SHA,"depth_batch_original_test")

class Batch(unittest.TestCase):
    def test_four_fresh_commands(self):
        for case,mode in m.PHASES:
            args=m.command(Path("/tmp/scoped"),"a"*64,case,mode)
            self.assertEqual(args[1],"-I");self.assertTrue(args[2].endswith("probe_pva_depth_control.py"))
            self.assertEqual("--trace" in args,mode=="trace");self.assertNotIn("--preflight",args)
            self.assertEqual(args[args.index("--case")+1],case)
        with self.assertRaises(ValueError):m.command(Path("/tmp"),"a"*64,"media","trace")

    def test_new_child_depth_schema_required(self):
        canonical=dict(effective_motion_configuration=dict(pyramid_levels=2))
        row=dict(schema="seaqr.pva-depth-control.v1",completed=True,passed_integrity=True,case="texture",mode="trace",
            input_sha256=dict(freeze_sha256="a"*64),pyramid_depth_changed=True,other_motion_settings_changed=False,
            canonical_nontiming=canonical,canonical_nontiming_sha256=old.canonical_sha(canonical))
        m.validate_result(row,"texture","trace","a"*64,old)
        for key,value in (("schema","seaqr.static-pva-texture.v1"),("passed_integrity",False),("other_motion_settings_changed",True)):
            bad=dict(row);bad[key]=value
            with self.assertRaises(ValueError):m.validate_result(bad,"texture","trace","a"*64,old)
        row["canonical_nontiming"]["effective_motion_configuration"]["pyramid_levels"]=4
        row["canonical_nontiming_sha256"]=old.canonical_sha(row["canonical_nontiming"])
        with self.assertRaises(ValueError):m.validate_result(row,"texture","trace","a"*64,old)

    def test_scope_before_import(self):
        with patch.object(m,"imported") as imported:
            with self.assertRaises(ValueError):m.load("/tmp","/tmp/freeze.json","a"*64)
            imported.assert_not_called()

    def test_original_safety_policy_boundaries(self):
        self.assertEqual(old.PHASES,m.PHASES)
        old.guard({"CPU":64.9},0,0,0,starting=True)
        for args in (({"CPU":65},0,0,0,True),({"CPU":75},0,0,0,False),({"CPU":40},900,0,0,False),({"CPU":40},3600,0,3590,False)):
            with self.assertRaises(ValueError):old.guard(*args)

    def test_callbacks_restored_after_runtime_failure_no_comparison(self):
        supervisor=SimpleNamespace(bundle="original_bundle",command="original_command",validate_result="original_check",SCHEMA="original_schema")
        probe=SimpleNamespace(bundle=Mock(),scientific_comparison=Mock())
        def fail(*args):
            self.assertEqual(supervisor.SCHEMA,m.SCHEMA)
            self.assertIs(supervisor.command,m.command)
            raise RuntimeError("child failed")
        supervisor.run=Mock(side_effect=fail)
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"load",return_value=(supervisor,probe,old,{})):
            with self.assertRaisesRegex(RuntimeError,"child failed"):m.run(directory,"unused","a"*64)
            probe.scientific_comparison.assert_not_called()
        self.assertEqual(supervisor.bundle,"original_bundle");self.assertEqual(supervisor.SCHEMA,"original_schema")

    def test_inherited_engine_first_child_failure_stops_owned_no_next_case(self):
        safety=SimpleNamespace(temperatures=lambda:{"CPU":40.},stop_owned=Mock())
        loader=SimpleNamespace(exec_module=lambda value:None)
        spec=SimpleNamespace(loader=loader)
        child=SimpleNamespace(pid=123,returncode=1,poll=lambda:1)
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            stack.enter_context(patch.object(old,"bundle",return_value={"files":{}}))
            stack.enter_context(patch.object(old.importlib.util,"spec_from_file_location",return_value=spec))
            stack.enter_context(patch.object(old.importlib.util,"module_from_spec",return_value=safety))
            popen=stack.enter_context(patch.object(old.subprocess,"Popen",return_value=child))
            stack.enter_context(patch.object(old.signal,"signal",return_value=None))
            with self.assertRaisesRegex(ValueError,"Child failed"):old.run(directory,"unused","a"*64)
            self.assertEqual(popen.call_count,1);safety.stop_owned.assert_called_once_with(child)
            status=json.loads((Path(directory)/"batch_status.json").read_text())
            self.assertFalse(status["complete"]);self.assertEqual(len(status["phases"]),1)

    def test_existing_comparison_prevents_supervisor(self):
        supervisor=SimpleNamespace(run=Mock())
        with tempfile.TemporaryDirectory() as directory,patch.object(m,"load",return_value=(supervisor,None,old,{})):
            (Path(directory)/"depth_comparison.json").write_text("preserved")
            with self.assertRaises(ValueError):m.run(directory,"unused","a"*64)
            supervisor.run.assert_not_called()

    def comparison_fixture(self,directory,tamper_before=False,tamper_during=False):
        root=Path(directory);digest="a"*64
        canonical=dict(effective_motion_configuration=dict(pyramid_levels=2))
        phases=[]
        for case,mode in m.PHASES:
            name=case+"_"+mode;path=root/(name+".json")
            value=dict(schema="seaqr.pva-depth-control.v1",completed=True,passed_integrity=True,case=case,mode=mode,
                input_sha256=dict(freeze_sha256=digest),pyramid_depth_changed=True,other_motion_settings_changed=False,
                canonical_nontiming=canonical,canonical_nontiming_sha256=old.canonical_sha(canonical))
            path.write_text(json.dumps(value))
            phases.append(dict(name=name,case=case,mode=mode,returncode=0,result_sha256=m.sha(path)))
        (root/"parity.json").write_text('{"passed":true}')
        status=dict(complete=True,execution_passed=True,parity_passed=True,phases=phases,parity_sha256=m.sha(root/"parity.json"))
        (root/"batch_status.json").write_text(json.dumps(status))
        supervisor=SimpleNamespace(bundle=None,command=None,validate_result=None,SCHEMA="original",run=Mock(return_value=status))
        def pinned(path,digest):
            if m.sha(path)!=digest:raise ValueError("phase bytes changed")
        helper=SimpleNamespace(read=old.read,pinned=pinned,canonical_sha=old.canonical_sha)
        def science(*args):
            if tamper_during:(root/"texture_trace.json").write_text('{}')
            return dict(interpretable=True)
        probe=SimpleNamespace(bundle=Mock(return_value={}),scientific_comparison=Mock(side_effect=science),
            references=Mock(return_value={}),PROTOCOL={},REFERENCE_RESULTS={},REFERENCE_FREEZE_SHA="b"*64)
        if tamper_before:(root/"texture_trace.json").write_text('{}')
        return supervisor,probe,helper,{}

    def test_all_phase_hashes_revalidated_before_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded=self.comparison_fixture(directory,tamper_before=True)
            with patch.object(m,"load",return_value=loaded),self.assertRaisesRegex(ValueError,"phase bytes changed"):
                m.run(directory,"unused","a"*64)
            loaded[1].scientific_comparison.assert_not_called()
            failure=old.read(Path(directory)/"depth_comparison_failure.json")
            self.assertTrue(failure["hardware_phases_complete"]);self.assertFalse(failure["scientific_comparison_complete"])
            self.assertFalse((Path(directory)/"depth_comparison.json").exists())

    def test_postread_mutation_refused_before_exclusive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded=self.comparison_fixture(directory,tamper_during=True)
            with patch.object(m,"load",return_value=loaded),self.assertRaisesRegex(ValueError,"phase bytes changed"):
                m.run(directory,"unused","a"*64)
            self.assertFalse((Path(directory)/"depth_comparison.json").exists())

    def test_bound_successful_comparison_write(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded=self.comparison_fixture(directory)
            with patch.object(m,"load",return_value=loaded):result=m.run(directory,"unused","a"*64)
            path=Path(directory)/"depth_comparison.json"
            self.assertEqual(result["depth_comparison_sha256"],m.sha(path))
            comparison=old.read(path)
            self.assertEqual(len(comparison["input_sha256"]),6)
            self.assertFalse((Path(directory)/"depth_comparison_failure.json").exists())

if __name__=="__main__":unittest.main()
