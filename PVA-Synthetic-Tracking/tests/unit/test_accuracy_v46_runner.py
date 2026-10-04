import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import run_accuracy_v46_stress as runner


class V46RunnerTests(unittest.TestCase):
    def test_exact_matrix_and_truth_remain_separate(self):
        arrays,metadata=runner.inputs_and_metadata(runner.causal_cases(),runner.build_cases())
        self.assertEqual(len(arrays),34)
        self.assertEqual(len(metadata["baseline"]),6)
        self.assertEqual(len(metadata["stress"]),28)
        for values in arrays.values():
            self.assertEqual(set(values),set(runner.ARRAY_KEYS))
            self.assertEqual(values["history129"].shape,(8,129,129))
            self.assertEqual(values["prior_centers_xy"].shape,(8,2))
        self.assertIn("generator_truth",metadata["stress"][0])
        json.dumps(metadata,allow_nan=False)

    def test_undeclared_adapter_inputs_are_rejected(self):
        cases=runner.build_cases()
        cases[0]["adapter_inputs"]["true_current_position"]=[64,64]
        with self.assertRaisesRegex(ValueError,"declared"):
            runner.inputs_and_metadata(runner.causal_cases(),cases)

    def test_memory_and_metadata_fingerprints_detect_different_changes(self):
        baseline,stress=runner.causal_cases(),runner.build_cases()
        arrays,metadata=runner.inputs_and_metadata(baseline,stress)
        original=runner.memory_hashes(arrays);original_meta=runner.json_hash(metadata)
        stress[0]["adapter_inputs"]["predicted_offset_xy"][0]=.1
        changed,other=runner.inputs_and_metadata(baseline,stress)
        self.assertNotEqual(original,runner.memory_hashes(changed))
        self.assertEqual(original_meta,runner.json_hash(other))
        stress[0]["generator_truth"]["current_peak_dn"]=31
        _,other=runner.inputs_and_metadata(baseline,stress)
        self.assertNotEqual(original_meta,runner.json_hash(other))

    def test_wrong_output_scope_is_rejected_before_any_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"wrong"
            with self.assertRaisesRegex(ValueError,"direct V46"):
                runner.run(output)
            self.assertFalse(output.exists())

    def test_exact_baseline_arrays_and_frozen_memory_are_checked(self):
        baseline,stress=runner.causal_cases(),runner.build_cases()
        arrays,metadata=runner.inputs_and_metadata(baseline,stress)
        bindings=runner.baseline_bindings()
        runner.require_baseline_arrays(arrays,bindings)
        hashes=runner.memory_hashes(arrays); meta_hash=runner.json_hash(metadata)
        runner.require_memory(baseline,stress,hashes,meta_hash)
        baseline[0]["current129"][0,0]+=1
        with self.assertRaisesRegex(ValueError,"Baseline array changed"):
            runner.require_baseline_arrays(arrays,bindings)
        with self.assertRaisesRegex(ValueError,"after freeze"):
            runner.require_memory(baseline,stress,hashes,meta_hash)

    def test_unavailable_records_are_not_counted_as_negative(self):
        def record(available,sign=None):
            result=dict(available=available,reasons=[] if available else ["unsupported"],
                        numerical_contrast=None if not available else dict(coefficient_sign=sign))
            return dict(arms={"test":dict(raw_adapter_result=result)})
        values=[record(False),record(True,"unresolved"),record(True,"positive")]
        result=runner.evidence_counts(values,"test")
        self.assertEqual(result["counts"],dict(states=3,unavailable=1,available=2,positive=1,negative=0,unresolved=1))
        self.assertEqual(result["unavailable_reasons"],{"unsupported":1})

    def test_complete_temp_smoke_freezes_all_cases_before_first_score(self):
        # No asserted detection score or case selection. Check plumbing and the
        # frozen baseline only; persisted scientific run happens separately.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();output=root/"smoke"
            actual=runner.evaluate_causal_probe
            calls=[]
            def evaluate(**kwargs):
                self.assertEqual(set(kwargs),set(runner.ADAPTER_KEYS)|{"arm"})
                self.assertTrue((output/"score_start.json").is_file())
                inputs=json.loads((output/"inputs_complete.json").read_text())
                self.assertEqual(len(inputs["inputs_sha256"]),34)
                self.assertFalse(kwargs["current129"].flags.writeable)
                self.assertFalse(kwargs["history129"].flags.writeable)
                calls.append(kwargs["arm"])
                return actual(**kwargs)
            with mock.patch.object(runner,"OUTPUT_ROOT",root),mock.patch.object(runner,"evaluate_causal_probe",side_effect=evaluate):
                result=runner.run(output)
                self.assertEqual(len(calls),136)
                self.assertTrue(all(result["invariants"].values()))
                self.assertEqual(result["stress_cases"],28)
                self.assertFalse(result["production_changed"])
                self.assertFalse(result["real_packet_data_read"])
                receipt=json.loads((output/"completion_receipt.json").read_text())
                runner.require_hashes(receipt["files_sha256"])
                with self.assertRaises(FileExistsError):
                    runner.run(output)
                stress=json.loads((output/"stress_results.json").read_text())
                baseline=json.loads((output/"baseline_results.json").read_text())
                changed=copy.deepcopy(stress)
                changed[0]["arms"]["presence_box_bounds"]["raw_adapter_result"]["motion_status"]="moving"
                with self.assertRaisesRegex(ValueError,"physical"):
                    runner.summarize(baseline,changed,baseline)


if __name__=="__main__":unittest.main()
