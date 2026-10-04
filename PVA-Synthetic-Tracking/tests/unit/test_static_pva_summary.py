import base64
import hashlib
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("static_pva_summary", Path(__file__).resolve().parents[2]/"scripts/summarize_static_pva_texture.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def descriptor(value):
    raw=value.tobytes()
    return dict(dtype=value.dtype.str,shape=list(value.shape),sha256=hashlib.sha256(raw).hexdigest(),
                data_base64=base64.b64encode(raw).decode())


def fixture():
    """Independent generated metadata only, no worker import or PVA execution."""
    levels=[dict(array=descriptor(np.full(shape,index,np.uint8))) for index,shape in
            enumerate(((256,320),(128,160),(64,80),(32,40)))]
    flags=descriptor(np.array([0,1],np.uint8))
    capture=dict(partial=False,forward_status_bytes_changed_by_backward=True,
        forward_point_bytes_changed_by_backward=False,data={
            "before_forward":{"selected_count":2},
            "after_forward":{"forward_status":descriptor(np.array([0,0],np.uint8))},
            "after_backward":{"forward_status":flags,"backward_status_vpi":flags,
                "identities":{"forward_status":{"vpi_id":7},"backward_status_vpi":{"vpi_id":7}},
                "status_same_python_object":True,"points_same_python_object":False},
            "final_filter":{"rejection_counts":{"forward_status":1}},
            "pyramids":{"previous":copy.deepcopy(levels),"current":copy.deepcopy(levels)}})
    frozen=dict(schema="static_pva_texture.v1",cases=["bridge","texture"],modes=["base","trace"],
                files={"probe_static_pva_texture.py":"a"*64})
    results={}
    for case in ("bridge","texture"):
        for mode in ("base","trace"):
            canonical=dict(result=dict(correspondence=dict(metrics={"accepted_count":1}),
                                       fit=dict(quality_status="rejected")))
            results[case+"_"+mode]=dict(schema="seaqr.static-pva-texture.v1",completed=True,
                passed_integrity=True,case=case,mode=mode,generated_only=True,source_media_accessed=False,
                production_changes=False,detector_run=False,input_sha256={"files":copy.deepcopy(frozen["files"])},
                accepted_points=1,scientific_fit_accepted=False,canonical_nontiming=canonical,
                canonical_nontiming_sha256=digest(canonical),capture=copy.deepcopy(capture) if mode=="trace" else None)
    batch=dict(schema="seaqr.static-pva-texture.batch.v1",complete=True,execution_passed=True,
        generated_only=True,camera_media_accessed=False,clock_writes=False,files=copy.deepcopy(frozen["files"]),
        phases=[dict(name=name,pid=100+i) for i,name in enumerate(results)])
    parity=dict(schema="seaqr.static-pva-texture.batch.v1.parity",cases=[dict(case=c,passed=True) for c in ("bridge","texture")])
    return dict(freeze=frozen,batch=batch,parity=parity,results=results)


def save_fixture(directory, data):
    def write(name,value):
        path=directory/name
        path.write_text(json.dumps(value,allow_nan=False))
        return module.sha(path)
    freeze_sha=write("freeze.json",data["freeze"])
    data["batch"]["freeze_sha256"]=freeze_sha
    for phase in data["batch"]["phases"]:
        result=data["results"][phase["name"]]
        result["input_sha256"]["freeze_sha256"]=freeze_sha
        phase["result_sha256"]=write(phase["name"]+".json",result)
    data["batch"]["parity_sha256"]=write("parity.json",data["parity"])
    write("batch_status.json",data["batch"])


class SummaryTests(unittest.TestCase):
    def record(self, value):
        raw = value.tobytes()
        return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=hashlib.sha256(raw).hexdigest(), data_base64=base64.b64encode(raw).decode())

    def test_array_exact(self):
        value = np.array([[0., -0.], [np.nan, np.inf]], dtype=np.float32)
        self.assertEqual(module.array(self.record(value)).tobytes(), value.tobytes())

    def test_status_distribution(self):
        self.assertEqual(module.status(self.record(np.array([0,1,1,2],np.uint8))), {"0":1,"1":2,"2":1})

    def test_capture_hash_changed(self):
        record = self.record(np.array([0,1],np.uint8))
        record["sha256"] = "f"*64
        with self.assertRaises(ValueError):
            module.array(record)

    def test_wrong_status_type(self):
        with self.assertRaises(ValueError):
            module.status(self.record(np.array([0,1],np.float32)))

    def test_full_generated_fixture_validates_and_summarizes_both_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);save_fixture(directory,fixture())
            result=module.summarize(directory)
        self.assertEqual([row["case"] for row in result["cases"]],["bridge","texture"])
        self.assertEqual(len(result["input_sha256"]),7)
        for row in result["cases"]:
            self.assertTrue(row["trace_interpretation_allowed"])
            observed=row["observations"]
            self.assertEqual(observed["forward_status_immediately_after_forward"],{"0":2})
            self.assertEqual(observed["forward_status_after_backward"],{"0":1,"1":1})
            self.assertEqual(len(observed["pyramids"]),4)
            self.assertTrue(all(level["identical_previous_current"] for level in observed["pyramids"]))

    def test_mismatched_trace_is_retained_but_never_interpreted(self):
        data=fixture();trace=data["results"]["texture_trace"]
        trace["canonical_nontiming"]["result"]["correspondence"]["metrics"]["accepted_count"]=2
        trace["canonical_nontiming_sha256"]=digest(trace["canonical_nontiming"])
        trace["capture"]={"invalid_capture_must_not_be_read":True}
        data["parity"]["cases"][1]["passed"]=False
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);save_fixture(directory,data)
            result=module.summarize(directory)
        row=result["cases"][1]
        self.assertFalse(row["exact_base_trace_parity"])
        self.assertFalse(row["trace_interpretation_allowed"])
        self.assertNotIn("observations",row)
        self.assertEqual(row["baseline_result"]["metrics"]["accepted_count"],1)

    def test_expected_unavailability_keeps_null_fit_and_partial_uninterpreted(self):
        data=fixture()
        for mode in ("base","trace"):
            result=data["results"]["texture_"+mode]
            result["canonical_nontiming"]={"result":{"status":"unavailable","reason":"zero features",
                                                       "correspondence":None,"fit":None}}
            result["canonical_nontiming_sha256"]=digest(result["canonical_nontiming"])
            result["accepted_points"]=0;result["scientific_fit_accepted"]=False
        data["results"]["texture_trace"]["capture"]={"partial":True,"data":{"pyramids":"uninterpreted"}}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);save_fixture(directory,data)
            result=module.summarize(directory)
        row=result["cases"][1]
        self.assertTrue(row["exact_base_trace_parity"])
        self.assertFalse(row["trace_interpretation_allowed"])
        self.assertEqual(row["baseline_unavailable_reason"],"zero features")
        self.assertNotIn("observations",row)

    def test_rehashed_invalid_batch_or_child_is_rejected(self):
        mutations=(
            lambda d:d["batch"].update(schema="other"),
            lambda d:d["batch"].update(execution_passed=False),
            lambda d:d["batch"].update(camera_media_accessed=True),
            lambda d:d["batch"]["phases"].reverse(),
            lambda d:d["batch"]["phases"][1].update(pid=100),
            lambda d:d["results"]["texture_trace"].update(passed_integrity=False),
            lambda d:d["results"]["texture_trace"].update(case="bridge"),
            lambda d:d["results"]["texture_trace"].update(mode="base"),
            lambda d:d["results"]["texture_trace"].update(source_media_accessed=True),
            lambda d:d["results"]["texture_trace"]["input_sha256"].update(files={}),
            lambda d:d["parity"].update(schema="wrong"),
            lambda d:d["parity"]["cases"][0].update(passed=False),
            lambda d:d["results"]["texture_trace"].update(canonical_nontiming_sha256="b"*64),
        )
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as temp:
                data=fixture();mutate(data);directory=Path(temp);save_fixture(directory,data)
                with self.assertRaises(ValueError):module.summarize(directory)

    def test_status_count_and_pyramid_representation_fail_closed(self):
        mutations=(
            lambda c:c["data"]["after_forward"].update(forward_status=descriptor(np.zeros(1,np.uint8))),
            lambda c:c["data"]["pyramids"]["previous"].pop(),
            lambda c:c["data"]["pyramids"]["previous"][0].update(array=descriptor(np.zeros((320,256),np.uint8))),
            lambda c:c["data"]["pyramids"]["previous"][0].update(array=descriptor(np.zeros((256,320),np.uint16))),
        )
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as temp:
                data=fixture();mutate(data["results"]["bridge_trace"]["capture"])
                directory=Path(temp);save_fixture(directory,data)
                with self.assertRaises(ValueError):module.summarize(directory)

    def test_changed_child_bytes_fail_binding_before_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);save_fixture(directory,fixture())
            path=directory/"bridge_trace.json";path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"Child evidence differs"):
                module.summarize(directory)

    def test_postread_file_mutation_fails_end_recheck(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);save_fixture(directory,fixture());original=module.array;changed=False
            def mutate_after_read(record):
                nonlocal changed
                result=original(record)
                if not changed:
                    path=directory/"texture_base.json";path.write_text(path.read_text()+"\n")
                    changed=True
                return result
            with patch.object(module,"array",side_effect=mutate_after_read):
                with self.assertRaisesRegex(ValueError,"Evidence changed during summary"):
                    module.summarize(directory)


if __name__ == "__main__":
    unittest.main()
