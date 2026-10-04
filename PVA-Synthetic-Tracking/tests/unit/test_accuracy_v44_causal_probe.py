"""Synthetic adapter provenance; numerical solver tested independently."""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v44_causal_probe as probe


def synthetic_fixture(polarity="bright"):
    y, x = np.indices((129, 129))
    background = 50+20*(x >= 64)+8*np.sin(y/12)
    centers = [[29+4*i, 64] for i in range(8)]
    sign = 1 if polarity == "bright" else -1
    def point(cx, cy):
        return 30*np.exp(-((x-cx)**2+(y-cy)**2)/2)
    return (background+sign*point(64, 64),
            np.stack([background+sign*point(*position) for position in centers]), centers, [0, 0], polarity)


def mock_contrast(*args, **kwargs):
    return dict(available=True, estimate=1.0, error_bound=.2, interval=[.8, 1.2],
                interval_excludes_zero=True, reasons=[], diagnostics={"test_stub": True})


class CausalProbeTests(unittest.TestCase):
    def test_replacing_finite_current_changes_response_only_not_prior_design(self):
        args = list(synthetic_fixture())
        calls = []
        def capture(*a, **kw):
            calls.append(([value.copy() for value in a], {k: v.copy() if isinstance(v, np.ndarray) else v for k,v in kw.items()}))
            return mock_contrast()
        with mock.patch.object(probe, "_source_contrast", side_effect=capture):
            first = probe.evaluate_causal_probe(*args)
            args[0] = np.full((129, 129), 153.0)
            second = probe.evaluate_causal_probe(*args)
        self.assertEqual(len(calls), 2)
        self.assertEqual(first["learned_design_sha256"], second["learned_design_sha256"])
        self.assertEqual(first["common_support_sha256"], second["common_support_sha256"])
        self.assertEqual(first["components"], second["components"])
        self.assertFalse(np.array_equal(calls[0][0][0], calls[1][0][0]))
        for a, b in zip(calls[0][0][1:], calls[1][0][1:]):
            np.testing.assert_array_equal(a, b)
        for key in ("response_bound", "nuisance_bound", "source_bound"):
            np.testing.assert_array_equal(calls[0][1][key], calls[1][1][key])

    def test_fewer_than_five_actual_priors_is_unknown_without_learning(self):
        args = list(synthetic_fixture()); args[2][:4] = [None]*4
        with mock.patch.object(probe, "prepare_components") as prepare:
            result = probe.evaluate_causal_probe(*args)
        prepare.assert_not_called()
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"], ["insufficient_actual_prior_centers"])

    def test_missing_template_stays_unknown_without_contrast_solver(self):
        args = list(synthetic_fixture()); args[1][:] = 100.
        with mock.patch.object(probe, "_source_contrast") as solve:
            result = probe.evaluate_causal_probe(*args)
        solve.assert_not_called()
        self.assertFalse(result["available"])
        self.assertIn("insufficient_causal_foreground_template", result["reasons"])

    def test_overlay_and_incomplete_fixed_coverage_do_not_erase_motion_ambiguity(self):
        args = list(synthetic_fixture())
        y,x = np.indices((129,129)); fixed=25*np.exp(-((x-64)**2+(y-64)**2)/2)
        args[0] += fixed; args[1] += fixed
        with mock.patch.object(probe,"_source_contrast",side_effect=mock_contrast):
            result = probe.evaluate_causal_probe(*args)
        self.assertTrue(result["available"])
        self.assertEqual(result["motion_status"], "unknown")
        self.assertEqual(result["physical_class"], "unknown")
        self.assertTrue(any("fixed" in reason for reason in result["ambiguity_reasons"]))
        self.assertFalse(result["is_motion_or_classification_gate"])

    def test_absent_current_feature_never_becomes_motion_or_class_claim(self):
        args = list(synthetic_fixture()); y,x=np.indices((129,129))
        args[0] = 50+20*(x>=64)+8*np.sin(y/12)
        with mock.patch.object(probe,"_source_contrast",side_effect=mock_contrast):
            result = probe.evaluate_causal_probe(*args)
        self.assertEqual(result["motion_status"], "unknown")
        self.assertEqual(result["physical_class"], "unknown")
        self.assertFalse(result["prior_context"]["supplied_forecast_offset_independently_verified"])
        json.dumps(result,allow_nan=False)

    def test_dark_template_sign_and_nonnegative_design_bounds(self):
        captured = []
        def capture(*args, **kwargs):
            captured.append((args,kwargs)); return mock_contrast()
        with mock.patch.object(probe,"_source_contrast",side_effect=capture):
            result=probe.evaluate_causal_probe(*synthetic_fixture("dark"))
        self.assertTrue(result["available"])
        self.assertTrue(np.all(captured[0][0][2] <= 0))
        self.assertTrue(np.any(captured[0][0][2] < 0))
        self.assertTrue(np.all(captured[0][1]["source_bound"] >= 0))

    def test_missing_bounds_do_not_shrink_support(self):
        args=synthetic_fixture()
        original=probe.component_bounds
        def bad_bounds(*a,**kw):
            value=original(*a,**kw)
            value["moving_template_bound129"][64,64]=np.nan
            return value
        with mock.patch.object(probe,"component_bounds",side_effect=bad_bounds),mock.patch.object(probe,"_source_contrast") as solve:
            result=probe.evaluate_causal_probe(*args)
        solve.assert_not_called()
        self.assertFalse(result["available"])
        self.assertEqual(result["common_support_count"],625)
        self.assertIn("component_bounds_missing_on_original_common_support",result["reasons"])

    def test_insufficient_current_support_does_not_change_prior_design_or_call_solver(self):
        args=list(synthetic_fixture());args[0][52:77,52:77]=np.nan
        with mock.patch.object(probe,"_source_contrast") as solve:
            result=probe.evaluate_causal_probe(*args)
        solve.assert_not_called()
        self.assertIsNotNone(result["learned_design_sha256"])
        self.assertEqual(result["common_support_count"],0)
        self.assertEqual(result["reasons"],["insufficient_original_common_core_support"])

    def test_unavailable_contrast_record_is_retained_without_amending_motion_class(self):
        unavailable=dict(available=False,estimate=None,error_bound=None,interval=None,
                         interval_excludes_zero=None,reasons=["uncertain"],diagnostics={})
        with mock.patch.object(probe,"_source_contrast",return_value=unavailable):
            result=probe.evaluate_causal_probe(*synthetic_fixture())
        self.assertIs(result["numerical_contrast"],unavailable)
        self.assertFalse(result["available"])
        self.assertEqual(result["reasons"],["source_contrast:uncertain"])
        self.assertEqual(result["motion_status"],"unknown")

    def test_inputs_remain_unmodified(self):
        args=synthetic_fixture();current=args[0].copy();history=args[1].copy()
        with mock.patch.object(probe,"_source_contrast",side_effect=mock_contrast):
            probe.evaluate_causal_probe(*args)
        np.testing.assert_array_equal(current,args[0]);np.testing.assert_array_equal(history,args[1])

    def test_actual_solver_ordinary_gain_absence_and_off_forecast_remain_inconclusive(self):
        args = synthetic_fixture()
        y,x = np.indices((129,129))
        background = 50+20*(x>=64)+8*np.sin(y/12)
        scenarios = {
            "ordinary": args[0],
            "gain_plane": 1.2*args[0]+7+.01*(x-64)-.02*(y-64),
            "absent": background,
            "off_forecast": background+30*np.exp(-((x-70)**2+(y-64)**2)/2),
        }
        results = {name:probe.evaluate_causal_probe(current,*args[1:])
                   for name,current in scenarios.items()}
        for name,result in results.items():
            with self.subTest(name=name):
                self.assertTrue(result["available"])
                contrast=result["numerical_contrast"]
                self.assertFalse(contrast["interval_excludes_zero"])
                self.assertLessEqual(contrast["interval"][0],0)
                self.assertGreaterEqual(contrast["interval"][1],0)
                self.assertEqual(result["motion_status"],"unknown")
                self.assertEqual(result["physical_class"],"unknown")
                self.assertEqual(result["learned_design_sha256"],results["ordinary"]["learned_design_sha256"])
                json.dumps(result,allow_nan=False)
        ordinary=results["ordinary"]["numerical_contrast"]
        self.assertGreater(ordinary["estimate"],0)
        self.assertAlmostEqual(results["gain_plane"]["numerical_contrast"]["estimate"],1.2*ordinary["estimate"],places=9)
        self.assertAlmostEqual(results["absent"]["numerical_contrast"]["estimate"],0,places=9)

    def test_current_nan_changes_validity_support_only_never_learned_templates(self):
        args=list(synthetic_fixture())
        captured=[]
        def capture(*values,**kwargs):
            captured.append(values);return mock_contrast()
        with mock.patch.object(probe,"_source_contrast",side_effect=capture):
            original=probe.evaluate_causal_probe(*args)
            args[0]=args[0].copy();args[0][64,64]=np.nan
            masked=probe.evaluate_causal_probe(*args)
        self.assertEqual(original["common_support_count"],625)
        self.assertEqual(masked["common_support_count"],624)
        self.assertEqual(original["learned_design_sha256"],masked["learned_design_sha256"])
        self.assertEqual(len(captured[0][0]),625)
        self.assertEqual(len(captured[1][0]),624)


if __name__=="__main__":
    unittest.main()
