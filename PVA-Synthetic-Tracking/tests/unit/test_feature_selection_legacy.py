"""Minimal generated guard tests for an unexecuted conditional harness.

These do not qualify a hardware run. No source video is opened or decoded.
"""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = module("legacy_test_runner", ROOT / "scripts/run_feature_selection_legacy.py")
selection = module("legacy_test_selection", ROOT / "scripts/run_discovery_feature_selection.py")
baseline = module("legacy_test_baseline", ROOT / "scripts/run_discovery_pair_baseline.py")


def gate_fixture():
    provenance = dict(comparison_sha256="a" * 64, freeze_sha256=r.PAIR_FREEZE_SHA,
                      receipts_sha256={"0170": "b" * 64, "0240": "c" * 64})
    pair_freeze = dict(schema="feature_selection.v1", candidate=selection.CANDIDATE,
        baseline_workspace=str(selection.BASELINE_WORKSPACE),
        sources={c: selection.source_spec(c) for c in ("0170", "0240")},
        files={"run_discovery_feature_selection.py": r.SELECTION_SHA})
    comparison = dict(schema="seaqr.feature-selection.comparison.v1", production_changed=False,
        media_read=False, input_sha256={}, clips={}, assessment=dict(declared_development_gates_passed=True,
            promotion_allowed=False, gates=dict(both_clips_at_least_95pct_ready=True,
                no_pva_errors=True, all_35_coherent_actual_qualified_dark=True)))
    receipts = {}
    for clip in ("0170", "0240"):
        comparison["clips"][clip] = dict(selection=dict(ready_fraction=640 / 673,
            counts=dict(frames=673, ready_frames=640, pva_runtime_errors=0)))
        comparison["input_sha256"][f"{r.PAIR_LOCAL_EVIDENCE}/{clip}/execution_receipt.json"] = provenance["receipts_sha256"][clip]
        receipts[clip] = dict(schema=selection.SCHEMA, passed=True, error=None, workspace=r.PAIR_WORKSPACE,
            clip=clip, source=selection.source_spec(clip), candidate=selection.CANDIDATE,
            processed_frames=673, decoded_frames_verified=673,
            input_sha256=dict(freeze_sha256=r.PAIR_FREEZE_SHA, files=pair_freeze["files"]),
            detector_configuration_changed=False, tracker_configuration_changed=False,
            global_motion_gates_changed=False, production_promotion=False,
            annotations_supplied_to_detector=False, raw16_accessed=False, sealed_holdouts_accessed=False,
            harris_score_precision_changed=False, feature_algorithm_changed=True)
    comparison["clips"]["0240"]["selection"]["positive_pass"] = dict(reference_frames=35,
        best_coherent_identity_frames=35, preservation_guard_passed=True, radius_native_px=8,
        baseline_derived_not_independent_truth=True, complete_coherent_identities=["0/dark:generated"])
    return comparison, pair_freeze, receipts, provenance


class ConditionalGateTests(unittest.TestCase):
    def test_minimum_full_ready_integer_and_exact_provenance_pass_generated_only(self):
        result = r.validate_pair_gate(*gate_fixture(), selection)
        self.assertTrue(result["passed"])
        self.assertFalse(result["production_promotion"])
        self.assertFalse(result["independent_recall"])

    def test_readiness_fail_cannot_be_overridden_by_true_summary_boolean(self):
        for ready in (0, 531, 639):
            values = gate_fixture()
            row = values[0]["clips"]["0170"]["selection"]
            row["counts"]["ready_frames"], row["ready_fraction"] = ready, ready / 673
            with self.subTest(ready=ready), self.assertRaisesRegex(ValueError, "readiness"):
                r.validate_pair_gate(*values, selection)

    def test_gate_failure_provenance_and_promotion_rejected(self):
        mutations = (
            lambda v: v[0]["assessment"].update(declared_development_gates_passed=False),
            lambda v: v[0]["assessment"].update(promotion_allowed=True),
            lambda v: v[0]["input_sha256"].clear(),
            lambda v: v[2]["0170"]["input_sha256"].update(freeze_sha256="d" * 64),
            lambda v: v[2]["0170"].update(global_motion_gates_changed=True),
            lambda v: v[0]["clips"]["0240"]["selection"]["positive_pass"].update(best_coherent_identity_frames=34),
            lambda v: v[0]["clips"]["0170"]["selection"]["counts"].update(pva_runtime_errors=1),
        )
        for mutate in mutations:
            values = copy.deepcopy(gate_fixture())
            mutate(values)
            with self.assertRaises(ValueError):
                r.validate_pair_gate(*values, selection)

    def test_failed_input_gate_preserves_failed_receipt_without_inference(self):
        for mode in ("preflight", "run"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fake_selection = SimpleNamespace(CANDIDATE=selection.CANDIDATE, candidate_adapter=Mock())
                with patch.object(r, "workspace_guard", return_value=root), \
                        patch.object(r, "load_selection", return_value=fake_selection), \
                        patch.object(r, "inputs", side_effect=ValueError("pair development gate failed")):
                    with self.assertRaisesRegex(ValueError, "gate failed"):
                        getattr(r, mode)(root, "0029")
                fake_selection.candidate_adapter.assert_not_called()
                name = "preflight.json" if mode == "preflight" else "execution_receipt.json"
                receipt = r.read(root / "0029" / name)
                self.assertFalse(receipt["passed"])
                self.assertFalse((root / "0029/run").exists())


class ScopedAssertionsTests(unittest.TestCase):
    def test_four_sources_match_metadata_scorer(self):
        scorer = module("legacy_test_scorer", ROOT / "scripts/score_feature_selection_references.py")
        self.assertEqual(r.COUNTS, scorer.COUNTS)
        self.assertEqual(r.SOURCES, scorer.SOURCES)
        for clip in ("0170", "0240", "0001", "0127", "../../escape"):
            with self.assertRaises(ValueError):
                r.source_spec(clip)

    def test_private_namespace_retains_code_and_never_expands_original(self):
        original_hashes = dict(baseline.SOURCE_HASHES)
        for clip, count in r.COUNTS.items():
            scoped, audit = r.scoped_baseline(baseline, clip)
            self.assertEqual(scoped.source_spec(clip), r.source_spec(clip))
            for other in set(r.COUNTS) - {clip}:
                with self.assertRaises(ValueError):
                    scoped.source_spec(other)
            for name in r.SCOPED_FUNCTIONS:
                self.assertIs(getattr(scoped, name).__code__, getattr(baseline, name).__code__)
                self.assertIsNot(getattr(scoped, name).__globals__, getattr(baseline, name).__globals__)
            self.assertEqual(audit["effective_frames"], count)
            self.assertFalse(audit["original_module_globals_modified"])
        self.assertEqual(baseline.FRAMES, 673)
        self.assertEqual(baseline.SOURCE_HASHES, original_hashes)
        with self.assertRaises(ValueError):
            baseline.source_spec("0029")

    def test_native_frame_and_probe_count_assertions_retained(self):
        for clip, count in r.COUNTS.items():
            scoped, _ = r.scoped_baseline(baseline, clip)
            frame = SimpleNamespace(index=count - 1, gray=SimpleNamespace(shape=(3190, 4784), dtype="uint8"))
            scoped.check_frame(frame, count - 1)
            with self.assertRaises(ValueError):
                scoped.check_frame(SimpleNamespace(index=count, gray=frame.gray), count)
            frame.gray.shape = (1595, 2392)
            with self.assertRaises(ValueError):
                scoped.check_frame(frame, count - 1)
            probe = SimpleNamespace(codec="mjpeg", pixel_format="yuvj420p", width=4784, height=3190,
                                    frame_rate=Fraction(10), declared_frame_count=count)
            scoped.validate_probe(probe)
            probe.declared_frame_count = 673
            with self.assertRaises(ValueError):
                scoped.validate_probe(probe)


if __name__ == "__main__":
    unittest.main()
