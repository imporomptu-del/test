import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from report_accuracy_v43 import DIAGNOSTICS, compact_arm, describe, verify_bindings


class V43ReportingTests(unittest.TestCase):
    def fixture(self):
        arm = dict(available=True, ambiguous=False, reasons=[], ambiguity_reasons=[],
                   common_support_sha256="abc", common_support_count=20,
                   mse_stationary=10., mse_augmented=9., components={"fixed_anchor_count": 2})
        states = [dict(clip=c, frame_index=f, segment=s, track_id=t, reference_samples=[],
                       arms={"baseline": copy.deepcopy(arm), "protected_only": copy.deepcopy(arm),
                             "stable_only": None, "combined": None}) for c, f, s, t in DIAGNOSTICS]
        return states

    def test_unavailable_extrema_remain_none_and_inputs_unchanged(self):
        states = self.fixture()
        before = copy.deepcopy(states)
        result = describe(states, {"samples": []})
        self.assertEqual(states, before)
        self.assertIsNone(result["all_state_score_extrema"]["stable_only"]["maxima"]["mse_augmented"])
        self.assertIsNone(compact_arm(None))

    def test_same_support_requires_nonempty_matching_hash_and_both_scores(self):
        states = self.fixture()
        states[0]["arms"]["protected_only"]["common_support_sha256"] = "other"
        states[1]["arms"]["protected_only"]["available"] = False
        states[2]["arms"]["baseline"]["common_support_sha256"] = None
        states[2]["arms"]["protected_only"]["common_support_sha256"] = None
        result = describe(states, {"samples": []})
        self.assertEqual([d["both_available_same_support"] for d in result["predeclared_exposed_diagnostics"]],
                         [False, False, False, True])

    def test_loss_uses_original_assignment_not_favorable_alternative(self):
        references = {"samples": [{"sample_index": 17,
            "original": dict(panel="grid", clip_id="0029", frame_index=85,
                             stages={"strict_qualified_measurement": [True, "old"]}),
            "measured_alternatives": [dict(identity="old", arms=dict(baseline={"available": True},
                                        protected_only={"available": False, "reasons": ["unsupported"]})),
                                      dict(identity="other", arms=dict(baseline=None, protected_only={"available": True}))]}]}
        result = describe(self.fixture(), references)
        self.assertEqual(result["original_strict_assignment_lost_score_protected_only"],
                         [dict(sample_index=17, panel="grid", clip="0029", frame_index=85,
                               identity="old", reasons=["unsupported"])])

    def test_duplicate_state_rejected(self):
        states = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            describe(states + states[:1], {"samples": []})

    def test_incomplete_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "not completed"):
            verify_bindings({"completed": False, "files_sha256": {"a": "hash"}}, {"a": "hash"})

    def test_missing_or_changed_binding_rejected(self):
        for bindings in ({}, {"a": "other"}):
            with self.assertRaisesRegex(ValueError, "missing or mismatched"):
                verify_bindings({"completed": True, "files_sha256": bindings}, {"a": "hash"})

    def test_completed_matching_binding_passes(self):
        verify_bindings({"completed": True, "files_sha256": {"a": "hash"}}, {"a": "hash"})


if __name__ == "__main__":
    unittest.main()
