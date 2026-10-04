import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v44_causal_probe as v44
from accuracy_v45_causal_probe import evaluate_causal_probe
import accuracy_v47_probe as probe
from run_accuracy_v44_synthetic import causal_cases


def inputs(case):
    return {key:case[key] for key in ("current129", "history129", "prior_centers_xy",
                                    "predicted_offset_xy", "polarity")}


class GuardProbeTests(unittest.TestCase):
    def test_nominal_design_support_and_prior_context_unchanged(self):
        args = inputs(causal_cases()[0])
        old = evaluate_causal_probe(**args, arm="presence_box_bounds")["raw_adapter_result"]
        new = probe.evaluate_probe(**args)
        raw = new["raw_adapter_result"]
        for field in ("learned_design_sha256", "common_support_sha256", "common_support_count",
                      "components", "component_bounds", "uncertainty_excludes"):
            self.assertEqual(raw[field], old[field], field)
        context = dict(raw["prior_context"])
        self.assertEqual(context["current_values_used_for"], probe.CURRENT_USE_DESCRIPTION)
        context["current_values_used_for"] = new["current_use_metadata_override"]["from"]
        self.assertEqual(context, old["prior_context"])
        self.assertEqual(raw["conditional_on"][:len(old["conditional_on"])], old["conditional_on"])
        self.assertEqual(raw["ambiguity_reasons"][:len(old["ambiguity_reasons"])], old["ambiguity_reasons"])
        self.assertFalse(new["old_unrestricted_background_estimand_preserved"])
        self.assertEqual(new["physical_class"], "unknown")
        self.assertEqual(raw["motion_status"], "unknown")

    def test_no_global_dependency_mutation_and_no_input_mutation(self):
        old_bounds, old_contrast = v44.component_bounds, v44._source_contrast
        args = inputs(causal_cases()[0]); before = copy.deepcopy(args)
        probe.evaluate_probe(**args)
        self.assertIs(v44.component_bounds, old_bounds)
        self.assertIs(v44._source_contrast, old_contrast)
        for key in args:
            np.testing.assert_array_equal(args[key], before[key])

    def test_short_history_stays_unknown_before_guard_access(self):
        args = inputs(causal_cases()[-1])
        with mock.patch.object(probe, "estimate_guard_gain") as guard:
            value = probe.evaluate_probe(**args)
        guard.assert_not_called()
        self.assertFalse(value["raw_adapter_result"]["available"])
        self.assertIsNone(value["gain_calibration"])

    def test_unavailable_gain_is_not_replaced_with_identity_gain(self):
        args = inputs(causal_cases()[0])
        gain = dict(available=False, reasons=["unbounded_gain"], gain_interval=None)
        actual = probe.bounded_background_presence
        with mock.patch.object(probe, "estimate_guard_gain", return_value=gain), \
             mock.patch.object(probe, "bounded_background_presence", wraps=actual) as score:
            value = probe.evaluate_probe(**args)
        self.assertEqual(score.call_count, 1)
        self.assertIsNone(score.call_args.kwargs["gain_interval"])
        self.assertFalse(value["raw_adapter_result"]["available"])
        self.assertEqual(value["gain_calibration"], gain)

    def test_background_and_all_fixed_columns_forwarded_without_selection(self):
        args = inputs(causal_cases()[0])
        captured = []
        actual = probe.bounded_background_presence
        def inspect(y, background, fixed, moving, plane, **kwargs):
            captured.append((y, background, fixed, moving, plane, kwargs))
            return actual(y, background, fixed, moving, plane, **kwargs)
        with mock.patch.object(probe, "bounded_background_presence", side_effect=inspect):
            value = probe.evaluate_probe(**args)
        self.assertEqual(len(captured), 1)
        y, background, fixed, moving, plane, kwargs = captured[0]
        raw = value["raw_adapter_result"]
        self.assertEqual(y.size, raw["common_support_count"])
        self.assertEqual(fixed.shape, (y.size, raw["components"]["fixed_anchor_count"]))
        self.assertEqual(plane.shape, (y.size, 3))
        self.assertEqual(background.shape, moving.shape)
        self.assertEqual(kwargs["fixed_bound"].shape, fixed.shape)
        self.assertFalse(kwargs["gain_provenance"]["background_validity_and_core_transfer_certified"])

    def test_current_core_values_never_change_guard_or_learned_design(self):
        args = inputs(causal_cases()[0]); other = copy.deepcopy(args)
        other["current129"][52:77,52:77] += np.arange(625).reshape(25,25)
        first, second = probe.evaluate_probe(**args), probe.evaluate_probe(**other)
        self.assertEqual(first["gain_calibration"], second["gain_calibration"])
        for key in ("learned_design_sha256", "common_support_sha256", "components"):
            self.assertEqual(first["raw_adapter_result"][key], second["raw_adapter_result"][key])


if __name__ == "__main__":
    unittest.main()
