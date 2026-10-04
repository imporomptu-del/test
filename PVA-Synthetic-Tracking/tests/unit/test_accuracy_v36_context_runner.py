"""Synthetic summary accounting for frozen, measured-state selections only.

No files or media are opened by these tests. Fixtures represent the compact
selection contract, not source footage, class labels, or independent samples.
"""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/diagnose_accuracy_v36_context.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v36_context_runner_under_test", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
with mock.patch.object(sys, "path", [str(MODULE_PATH.parent), *sys.path]):
    SPEC.loader.exec_module(runner)


def observation(key, *, passed=True, memberships=(), informative=True, point_gain=None):
    gain = (0.8 if passed else 0.2) if point_gain is None else point_gain
    edge_gain = 0.3
    if not informative:
        gain = edge_gain = 0.0
    return {
        "key": key,
        "clip": "synthetic",
        "frame": 0,
        "identity": "0/bright:synthetic",
        "measurement_source_xy": [20.0, 30.0],
        "polarity": "bright",
        "provisional_control_indices": list(memberships),
        "zero_margin_ablation_passed": passed and informative,
        "features": {
            "informative": informative,
            "conditional_informative": informative,
            "point_minus_edge_fraction": gain - edge_gain,
            "point_gain_fraction": gain,
            "edge_gain_fraction": edge_gain,
            "point_gain_after_edge_fraction": 0.6 if informative else 0.0,
            "point_amplitude_dn": 3.0 if informative else 0.0,
            "point_after_edge_amplitude_dn": 2.0 if informative else 0.0,
            "residual_rms_dn": 4.0 if informative else 0.0,
        },
    }


def group(kind="dense", keys=(), *, frame=0, window="synthetic_window", assigned=None):
    return {
        "kind": kind,
        "clip": "synthetic",
        "window": window,
        "frame": frame,
        "keys": list(keys),
        "baseline_assigned_id": assigned,
    }


def control(index):
    return {
        "label": f"provisional_nuisance_{index}",
        "track_id": f"historical:case{index}",
        "frames_inclusive": [10 * index, 10 * index + 5],
        "crop_xywh": [100 * index, 20, 25, 25],
    }


def selection(groups=(), controls=(), results=()):
    return {
        "groups": copy.deepcopy(list(groups)),
        "controls": copy.deepcopy(list(controls)),
        "observations": [
            {k: copy.deepcopy(v) for k, v in result.items()
             if k not in ("features", "zero_margin_ablation_passed")}
            for result in results
        ],
    }


class AccuracyV36ContextRunnerTests(unittest.TestCase):
    def test_baseline_miss_keeps_denominator_without_becoming_new_miss(self):
        results = [observation("preserved"), observation("lost", passed=False)]
        chosen = selection([
            group(keys=["preserved"], frame=1),
            group(keys=[], frame=2),
            group(keys=["lost"], frame=3),
        ], results=results)
        summary = runner.summarize(chosen, results)
        counts = summary["reference_retention"]["dense"]
        self.assertEqual(counts["samples"], 3)
        self.assertEqual(counts["baseline_hits"], 2)
        self.assertEqual(counts["diagnostic_hits"], 1)
        self.assertEqual(counts["newly_missed_samples"], [
            {"clip": "synthetic", "window": "synthetic_window", "frame": 3}
        ])
        missing = summary["groups"][1]
        self.assertFalse(missing["baseline_hit"])
        self.assertFalse(missing["diagnostic_hit"])
        self.assertFalse(missing["new_miss"])

    def test_passing_gated_alternative_counts_even_if_assigned_track_fails(self):
        results = [observation("assigned", passed=False), observation("alternative")]
        chosen = selection([group(keys=["assigned", "alternative"], assigned="assigned")], results=results)
        summary = runner.summarize(chosen, results)
        counts = summary["reference_retention"]["dense"]
        self.assertEqual(counts["diagnostic_hits"], 1)
        self.assertEqual(counts["newly_missed_samples"], [])
        self.assertEqual(counts["baseline_ambiguous_samples"], 1)
        self.assertEqual(counts["no_longer_preserves_all_baseline_ids"], 1)
        self.assertEqual(summary["groups"][0]["passing_keys"], ["alternative"])
        self.assertEqual(summary["groups"][0]["baseline_assigned_id"], "assigned")

    def test_all_gated_alternatives_lost_is_one_newly_missed_sample(self):
        results = [observation("a", passed=False), observation("b", passed=False)]
        chosen = selection([group(keys=["a", "b"], frame=8)], results=results)
        summary = runner.summarize(chosen, results)
        counts = summary["reference_retention"]["dense"]
        self.assertEqual(counts["samples"], 1)
        self.assertEqual(counts["baseline_hits"], 1)
        self.assertEqual(counts["diagnostic_hits"], 0)
        self.assertEqual(len(counts["newly_missed_samples"]), 1)
        self.assertTrue(summary["groups"][0]["new_miss"])

    def test_all_alternatives_preserved_does_not_claim_unique_identity(self):
        results = [observation("a"), observation("b")]
        summary = runner.summarize(selection([group(keys=["a", "b"])], results=results), results)
        counts = summary["reference_retention"]["dense"]
        self.assertEqual(counts["diagnostic_hits"], 1)
        self.assertEqual(counts["baseline_ambiguous_samples"], 1)
        self.assertEqual(counts["no_longer_preserves_all_baseline_ids"], 0)
        self.assertEqual(summary["groups"][0]["passing_keys"], ["a", "b"])

    def test_overlapping_dense_pilot_anchor_references_remain_separate(self):
        results = [observation("shared")]
        groups = [group(kind, ["shared"], frame=6, window="same_encounter")
                  for kind in ("dense", "pilot", "anchor")]
        summary = runner.summarize(selection(groups, results=results), results)
        self.assertEqual(summary["selected_observations"], 1)
        self.assertEqual(set(summary["reference_retention"]), {"dense", "pilot", "anchor"})
        for kind in ("dense", "pilot", "anchor"):
            counts = summary["reference_retention"][kind]
            self.assertEqual(counts["samples"], 1)
            self.assertEqual(counts["baseline_hits"], 1)
            self.assertEqual(counts["diagnostic_hits"], 1)
            self.assertEqual(summary["feature_distributions"][kind + "/same_encounter"]
                             ["point_gain_fraction"]["count"], 1)

    def test_stricter_pilot_loss_not_hidden_by_passing_dense_group(self):
        results = [observation("dense_match"), observation("pilot_match", passed=False)]
        groups = [group("dense", ["dense_match", "pilot_match"], frame=4),
                  group("pilot", ["pilot_match"], frame=4),
                  group("anchor", ["dense_match"], frame=4)]
        summary = runner.summarize(selection(groups, results=results), results)
        self.assertEqual(summary["reference_retention"]["dense"]["diagnostic_hits"], 1)
        self.assertEqual(summary["reference_retention"]["pilot"]["diagnostic_hits"], 0)
        self.assertEqual(summary["reference_retention"]["anchor"]["diagnostic_hits"], 1)
        self.assertEqual(len(summary["reference_retention"]["pilot"]["newly_missed_samples"]), 1)

    def test_selected_measured_control_states_keep_all_fixed_window_memberships(self):
        controls = [control(0), control(1), control(2)]
        results = [observation("shared", memberships=[0, 1]),
                   observation("second", memberships=[1], passed=False),
                   observation("not_a_control")]
        chosen = selection(controls=controls, results=results)
        summary = runner.summarize(chosen, results)
        self.assertEqual(summary["selected_observations"], 3)
        self.assertEqual(len(summary["provisional_controls"]), 3)
        self.assertEqual([c["selected_measured_states"] for c in summary["provisional_controls"]], [1, 2, 0])
        self.assertEqual([c["passing_measured_states"] for c in summary["provisional_controls"]], [1, 1, 0])
        for original, reported in zip(controls, summary["provisional_controls"]):
            self.assertEqual({k: reported[k] for k in original}, original)
        self.assertEqual(summary["feature_distributions"]["control/2/provisional_nuisance_2"]
                         ["point_gain_fraction"], {"count": 0})

    def test_empty_control_does_not_eliminate_reference_sample_or_fabricate_exposure(self):
        results = [observation("reference")]
        chosen = selection([group(keys=["reference"])], [control(0)], results)
        summary = runner.summarize(chosen, results)
        self.assertEqual(summary["reference_retention"]["dense"]["samples"], 1)
        self.assertEqual(summary["reference_retention"]["dense"]["diagnostic_hits"], 1)
        self.assertEqual(summary["provisional_controls"][0]["selected_measured_states"], 0)
        self.assertEqual(summary["provisional_controls"][0]["passing_measured_states"], 0)
        self.assertIsNone(summary["false_alarms_per_minute"])

    def test_shared_reference_and_nuisance_membership_is_not_silently_removed(self):
        results = [observation("shared", memberships=[0])]
        chosen = selection([group(keys=["shared"])], [control(0)], results)
        summary = runner.summarize(chosen, results)
        self.assertEqual(summary["selected_observations"], 1)
        self.assertEqual(summary["reference_retention"]["dense"]["diagnostic_hits"], 1)
        self.assertEqual(summary["provisional_controls"][0]["selected_measured_states"], 1)

    def test_feature_distributions_deduplicate_same_selected_state_within_group(self):
        results = [observation("shared", point_gain=0.75)]
        chosen = selection([group(keys=["shared"], frame=1), group(keys=["shared"], frame=2)], results=results)
        summary = runner.summarize(chosen, results)
        self.assertEqual(summary["reference_retention"]["dense"]["samples"], 2)
        stats = summary["feature_distributions"]["dense/synthetic_window"]["point_gain_fraction"]
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["minimum"], 0.75)
        self.assertEqual(stats["median"], 0.75)
        self.assertEqual(stats["maximum"], 0.75)

    def test_uninformative_patch_stays_in_reference_denominator(self):
        results = [observation("flat", informative=False)]
        summary = runner.summarize(selection([group(keys=["flat"])], results=results), results)
        self.assertEqual(summary["uninformative_patches"], 1)
        counts = summary["reference_retention"]["dense"]
        self.assertEqual(counts["samples"], 1)
        self.assertEqual(counts["baseline_hits"], 1)
        self.assertEqual(counts["diagnostic_hits"], 0)
        self.assertEqual(len(counts["newly_missed_samples"]), 1)

    def test_uninformative_zero_coding_is_explicit_in_availability_not_dropped(self):
        results = [observation("flat", informative=False),
                   observation("edge_only", passed=False), observation("point")]
        results[1]["features"]["conditional_informative"] = False
        results[1]["features"]["point_gain_after_edge_fraction"] = 0.0
        results[1]["features"]["point_after_edge_amplitude_dn"] = 0.0
        groups = [group(keys=[item["key"]], frame=index)
                  for index, item in enumerate(results)]
        summary = runner.summarize(selection(groups, results=results), results)
        stats = summary["feature_distributions"]["dense/synthetic_window"]
        self.assertEqual(stats["availability"], {
            "observations": 3, "informative": 2, "conditional_informative": 1,
            "distributions_include_uninformative_zero_coded_gains": True,
        })
        self.assertEqual(stats["point_gain_after_edge_fraction"]["count"], 3)
        self.assertEqual(stats["point_gain_after_edge_fraction"]["median"], 0.0)
        self.assertEqual(summary["reference_retention"]["dense"]["samples"], 3)

    def test_unknown_operational_accuracy_remains_unknown_and_unpromoted(self):
        results = [observation("retained"), observation("nuisance", memberships=[0], passed=False)]
        chosen = selection([group(keys=["retained"])], [control(0)], results)
        summary = runner.summarize(chosen, results)
        for name in ("airborne_precision", "airborne_recall", "false_alarms_per_minute"):
            self.assertIsNone(summary[name])
        for name in ("promoted", "detector_rerun", "thresholds_tuned"):
            self.assertIs(summary[name], False)
        self.assertTrue(summary["limitations"])
        self.assertNotIn("false_positive_count", summary)

    def test_empty_reference_kind_is_explicit_not_omitted(self):
        summary = runner.summarize(selection(controls=[control(0)]), [])
        for kind in ("dense", "pilot", "anchor"):
            self.assertEqual(summary["reference_retention"][kind]["samples"], 0)
            self.assertEqual(summary["reference_retention"][kind]["baseline_hits"], 0)
            self.assertEqual(summary["reference_retention"][kind]["diagnostic_hits"], 0)
            self.assertEqual(summary["reference_retention"][kind]["newly_missed_samples"], [])
        self.assertEqual(len(summary["provisional_controls"]), 1)

    def test_summary_does_not_mutate_selection_or_features(self):
        results = [observation("a", memberships=[0]), observation("b", passed=False)]
        chosen = selection([group(keys=["a", "b"])], [control(0)], results)
        before_selection = copy.deepcopy(chosen)
        before_results = copy.deepcopy(results)
        first = runner.summarize(chosen, results)
        second = runner.summarize(chosen, results)
        self.assertEqual(chosen, before_selection)
        self.assertEqual(results, before_results)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
