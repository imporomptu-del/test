import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_accuracy_v44_synthetic as runner


class V44RunnerTests(unittest.TestCase):
    def test_causal_variants_keep_history_and_forecast_fixed(self):
        cases = runner.causal_cases()
        self.assertEqual(len(cases), 6)
        first = cases[0]
        for case in cases:
            np.testing.assert_array_equal(case["history129"], first["history129"])
            self.assertEqual(case["predicted_offset_xy"], [0., 0.])
        self.assertEqual(sum(c is not None for c in cases[-1]["prior_centers_xy"]), 4)
        self.assertFalse(np.array_equal(cases[-2]["current129"], first["current129"]))

    def test_scope_guard_rejects_unrelated_output_before_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"not_v44"
            with self.assertRaises(ValueError):
                runner.run(output)
            self.assertFalse(output.exists())

    def test_all_inputs_frozen_before_mocked_scores_and_existing_run_rejected(self):
        contrast = dict(available=True, estimate=1., error_bound=2., interval=[-1., 3.],
                        interval_excludes_zero=False, motion_status="unknown", physical_class="unknown")
        probe = dict(available=True, numerical_contrast=contrast)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            output = root/"run"
            def score(**kwargs):
                self.assertTrue((output/"freeze.json").exists())
                self.assertTrue((output/"score_start.json").exists())
                manifest = json.loads((output/"inputs_complete.json").read_text())
                self.assertEqual(len(manifest["inputs_sha256"]), 22)
                runner.require_hashes(manifest["inputs_sha256"])
                self.assertEqual(set(kwargs), set(runner.ARRAY_KEYS))
                return dict(contrast)
            with patch.object(runner, "OUTPUT_ROOT", root), patch.object(runner, "dependency_paths", return_value=[]), \
                 patch.object(runner, "source_contrast", side_effect=score) as solver, \
                 patch.object(runner, "evaluate_causal_probe", return_value=probe) as causal:
                result = runner.run(output)
                self.assertEqual(solver.call_count, 16)
                self.assertEqual(causal.call_count, 6)
                self.assertEqual(result["oracle_vector_counts"]["states"], 16)
                self.assertEqual(result["oracle_vector_counts"]["interval_includes_zero"], 16)
                self.assertFalse(result["production_changed"])
                receipt = json.loads((output/"completion_receipt.json").read_text())
                runner.require_hashes(receipt["files_sha256"])
                self.assertTrue(receipt["completed"])
                with self.assertRaises(FileExistsError):
                    runner.run(output)

    def test_identical_inputs_cannot_have_different_interpretation_dependent_scores(self):
        records = [dict(case_id="identifiability_a", contrast=dict(available=False)),
                   dict(case_id="identifiability_b", contrast=dict(available=True, interval_excludes_zero=True))]
        with self.assertRaisesRegex(ValueError, "identical inputs"):
            runner.summarize(records, [])

    def test_changed_bound_file_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"data"
            path.write_text("before")
            binding = {str(path): runner.sha(path)}
            path.write_text("after")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.require_hashes(binding)


if __name__ == "__main__":
    unittest.main()
