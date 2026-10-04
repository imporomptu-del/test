import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import run_accuracy_v47_synthetic as runner


class V47RunnerTests(unittest.TestCase):
    def test_frozen_case_counts_and_separate_truth(self):
        cases=runner.collect_cases(); arrays,meta=runner.input_material(cases)
        self.assertEqual(len(cases),54); self.assertEqual(len(arrays),54)
        self.assertEqual(len(meta),54)
        for c in cases:
            self.assertEqual(set(c["adapter_inputs"]),set(runner.KEYS))
        runner.require_baseline_inputs(cases,arrays,runner.baseline_bindings())

    def test_metadata_or_array_mutation_invalidates_freeze(self):
        cases=runner.collect_cases();arrays,meta=runner.input_material(cases)
        h,m=runner.memory_hashes(arrays),runner.jhash(meta)
        runner.require_memory(cases,h,m)
        cases[-1]["generator_truth"]["current_source_present"]=False
        with self.assertRaisesRegex(ValueError,"after freeze"):
            runner.require_memory(cases,h,m)
        cases=runner.collect_cases();cases[0]["adapter_inputs"]["current129"][0,0]+=1
        with self.assertRaisesRegex(ValueError,"after freeze"):
            runner.require_memory(cases,h,m)

    def test_undeclared_truth_input_or_wrong_output_scope_rejected(self):
        cases=runner.collect_cases();cases[-1]["adapter_inputs"]["true_gain"]=1.2
        with self.assertRaisesRegex(ValueError,"Only observation"):
            runner.input_material(cases)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"direct V47"):
                runner.run(Path(tmp)/"wrong")

    def test_full_generated_matrix_freezes_before_any_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();output=root/"synthetic_test"
            actual=runner.evaluate_probe;calls=[]
            def checked(**args):
                self.assertEqual(set(args),set(runner.KEYS))
                self.assertTrue((output/"score_start.json").is_file())
                self.assertEqual(len(json.loads((output/"inputs_complete.json").read_text())["files_sha256"]),54)
                self.assertFalse(args["current129"].flags.writeable)
                self.assertFalse(args["history129"].flags.writeable)
                calls.append(1)
                return actual(**args)
            with mock.patch.object(runner,"OUTPUT_ROOT",root), \
                 mock.patch.object(runner,"evaluate_probe",side_effect=checked):
                result=runner.run(output)
                self.assertEqual(len(calls),54)
                self.assertEqual(result["calculations"],5)
                self.assertTrue(result["v46_inputs_and_four_arm_results_exact"])
                self.assertFalse(result["production_changed"])
                self.assertFalse(result["real_packet_data_read"])
                receipt=json.loads((output/"completion_receipt.json").read_text())
                runner.require_hashes(receipt["files_sha256"])
                with self.assertRaises(FileExistsError):runner.run(output)
                rows=json.loads((output/"results.json").read_text())
                changed=copy.deepcopy(rows[0]);changed["arms"][runner.ARM]["physical_class"]="airborne"
                with self.assertRaisesRegex(ValueError,"production/classification"):
                    runner.validate_record(changed)
                changed=copy.deepcopy(rows[0]);changed["arms"][runner.ARM]["raw_adapter_result"]["common_support_count"]-=1
                with self.assertRaisesRegex(ValueError,"prior design/support"):
                    runner.validate_record(changed)


if __name__=="__main__":unittest.main()
