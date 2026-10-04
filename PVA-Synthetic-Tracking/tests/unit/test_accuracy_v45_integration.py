"""Synthetic-only paired integration; no detector performance targets."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v44_causal_probe as frozen
import accuracy_v45_causal_probe as probe
import run_accuracy_v45_synthetic as runner


class V45IntegrationTests(unittest.TestCase):
    def test_invalid_arm_is_rejected_before_input_access(self):
        with self.assertRaisesRegex(ValueError, "arm"):
            probe.evaluate_causal_probe(None, None, None, None, None, arm="winner")

    def test_dependency_injection_is_local_and_thread_safe(self):
        case = runner.causal_cases()[0]
        args = {k:v for k,v in case.items() if k != "case_id"}
        original_solver = frozen._source_contrast
        original_bounds = frozen.component_bounds
        def evidence(tag):
            return dict(available=True, reasons=[], marker=tag,
                        motion_status="unknown", physical_class="unknown")
        with mock.patch.object(probe, "source_contrast", return_value=evidence("amplitude")), \
             mock.patch.object(probe, "source_presence", return_value=evidence("numerator")):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda arm:probe.evaluate_causal_probe(**args,arm=arm), probe.ARMS))
        self.assertIs(frozen._source_contrast, original_solver)
        self.assertIs(frozen.component_bounds, original_bounds)
        for result in results:
            self.assertEqual(result["raw_adapter_result"]["numerical_contrast"]["marker"],result["quantity"])
            self.assertFalse(result["is_motion_or_classification_gate"])

    def test_changed_current_cannot_change_learned_design_in_any_arm(self):
        case = runner.causal_cases()[0]
        first = {k:v for k,v in case.items() if k != "case_id"}
        second = dict(first, current129=np.full((129,129),153.))
        captured = []
        def evidence(y, Z, m, P, **errors):
            captured.append((y.copy(), Z.copy(), m.copy(), P.copy(), errors))
            return dict(available=True,reasons=[])
        with mock.patch.object(probe,"source_contrast",side_effect=evidence), \
             mock.patch.object(probe,"source_presence",side_effect=evidence):
            for arm in probe.ARMS:
                a=probe.evaluate_causal_probe(**first,arm=arm)["raw_adapter_result"]
                b=probe.evaluate_causal_probe(**second,arm=arm)["raw_adapter_result"]
                self.assertEqual(a["learned_design_sha256"],b["learned_design_sha256"])
                self.assertEqual(a["common_support_sha256"],b["common_support_sha256"])
                self.assertEqual(a["ambiguity_reasons"],b["ambiguity_reasons"])
                x,y=captured[-2:]
                self.assertFalse(np.array_equal(x[0],y[0]))
                for left,right in zip(x[1:4],y[1:4]):
                    np.testing.assert_array_equal(left,right)
                for key in x[4]:
                    np.testing.assert_array_equal(x[4][key],y[4][key])

    def test_insufficient_history_never_calls_any_solver_or_bounds(self):
        case = runner.causal_cases()[-1]
        with mock.patch.object(probe,"source_contrast") as a, \
             mock.patch.object(probe,"source_presence") as b, \
             mock.patch.object(probe,"box_bounds") as c, mock.patch.object(probe,"old_bounds") as d:
            for arm in probe.ARMS:
                value=probe.evaluate_causal_probe(**{k:v for k,v in case.items() if k != "case_id"},arm=arm)
                self.assertFalse(value["raw_adapter_result"]["available"])
            for method in (a,b,c,d):
                method.assert_not_called()

    def test_input_archives_and_baseline_are_exact_and_pinned(self):
        arrays=runner.input_arrays(runner.build_cases(),runner.causal_cases())
        bindings, baseline=runner.checked_baseline(arrays)
        self.assertEqual(len(arrays),22)
        self.assertEqual(len(baseline["oracle_results"]),16)
        self.assertEqual(len(baseline["causal_results"]),6)
        runner.require_hashes(bindings)
        arrays["ordinary_moving"]["y"][0]+=1
        with self.assertRaisesRegex(ValueError,"input changed"):
            runner.checked_baseline(arrays)

    def test_fresh_output_scope_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"wrong_scope"
            with self.assertRaisesRegex(ValueError,"dedicated"):
                runner.run(output)
            self.assertFalse(output.exists())

    def test_complete_matrix_binds_all_inputs_before_scores_and_reproduces_baseline(self):
        # Integration smoke in a temporary directory; no score target, selected
        # case, threshold, artifact promotion or persisted experiment outcome.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve(); output=root/"smoke"
            actual=runner.source_presence
            def checked_score(**kwargs):
                self.assertTrue((output/"score_start.json").exists())
                inputs=json.loads((output/"inputs_complete.json").read_text())
                self.assertEqual(len(inputs["inputs_sha256"]),22)
                runner.require_hashes(inputs["inputs_sha256"])
                self.assertFalse(kwargs["y"].flags.writeable)
                return actual(**kwargs)
            with mock.patch.object(runner,"OUTPUT_ROOT",root), \
                 mock.patch.object(runner,"source_presence",side_effect=checked_score):
                summary=runner.run(output)
                self.assertTrue(summary["baseline_exactly_reproduced"])
                self.assertTrue(summary["nominal_design_support_and_ambiguity_identical_across_arms"])
                self.assertTrue(summary["observational_twins_identical"])
                self.assertFalse(summary["production_changed"])
                for counts in summary["oracle_counts_by_arm"].values():
                    self.assertEqual(counts["states"],16)
                for counts in summary["causal_counts_by_arm"].values():
                    self.assertEqual(counts["states"],6)
                receipt=json.loads((output/"completion_receipt.json").read_text())
                runner.require_hashes(receipt["files_sha256"])
                with self.assertRaises(FileExistsError):
                    runner.run(output)

    def test_in_memory_fingerprints_detect_geometry_and_value_changes(self):
        cases,probes=runner.build_cases(),runner.causal_cases()
        original=runner.memory_fingerprints(runner.input_arrays(cases,probes))
        probes[0]["predicted_offset_xy"][0]=.1
        changed=runner.memory_fingerprints(runner.input_arrays(cases,probes))
        self.assertNotEqual(original,changed)
        probes[0]["predicted_offset_xy"][0]=0.
        cases[0]["y"][0]+=1
        self.assertNotEqual(original,runner.memory_fingerprints(runner.input_arrays(cases,probes)))

    def test_evidence_accounting_does_not_count_unknown_as_negative(self):
        counts=runner.count_evidence([
            None,dict(available=False),
            dict(available=True,interval_excludes_zero=False,coefficient_sign="unresolved"),
            dict(available=True,interval_excludes_zero=True,coefficient_sign="positive"),
            dict(available=True,interval_excludes_zero=True,coefficient_sign="negative")])
        self.assertEqual(counts["states"],5)
        self.assertEqual(counts["unavailable"],2)
        self.assertEqual(counts["negative"],1)
        self.assertEqual(counts["unresolved"],1)


if __name__ == "__main__":
    unittest.main()
