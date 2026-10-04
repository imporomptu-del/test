from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v43_stable_fit import supported_fit
from audit_accuracy_v42_numerical import compare
from audit_accuracy_v43_numerical import _safe_dependency, choose_keys, compact, independent_fit


class NumericalAuditTests(unittest.TestCase):
    def test_independent_qr_response_design_bounds_and_constraint_branches(self):
        x = np.linspace(-1, 1, 37)
        a = np.column_stack((np.ones(len(x)), x))
        b = np.asarray([[1., -.2], [1., 0], [1., .2]])
        for constrained in (False, True):
            for coefficient in (-20., -.01, 0., .01, 20.):
                target = 4+coefficient*x+.4*np.sin(x*5)
                actual = supported_fit(a, target, b, .5, constrained,
                                       train_design_bound=1e-5, test_design_bound=2e-5)
                independent = independent_fit(a, target, b, .5, 1e-5, 2e-5, constrained)
                self.assertEqual(compare(compact(independent), compact(actual)), [])
                self.assertTrue(independent["available"])
                np.testing.assert_allclose(independent["prediction"], actual["prediction"], atol=1e-10)

    def test_rank_deficiency_and_design_uncertainty_failure_reasons(self):
        examples = [(np.ones((8, 2)), np.ones((4, 2)), 0),
                    (np.column_stack((np.ones(8), np.ones(8)+1e-6*np.arange(8))), np.ones((4, 2)), .001)]
        for a, b, error in examples:
            actual = supported_fit(a, np.arange(len(a), dtype=float), b, .5,
                                   train_design_bound=error, test_design_bound=error)
            independent = independent_fit(a, np.arange(len(a), dtype=float), b, .5, error, error)
            self.assertFalse(independent["available"])
            self.assertEqual(compare(compact(independent), compact(actual)), [])

    def test_prediction_budget_failure_includes_large_residual_sensitivity(self):
        a = np.column_stack((np.ones(40), np.linspace(-1, 1, 40)))
        b = np.asarray([[1., 0]])
        y = 1e6*np.sin(np.arange(40))
        actual = supported_fit(a, y, b, .5, train_design_bound=.01, test_design_bound=.01)
        independent = independent_fit(a, y, b, .5, .01, .01)
        self.assertFalse(independent["available"])
        self.assertEqual(compare(compact(independent), compact(actual)), [])

    def test_empty_dictionary_reconstructs_exact_zero(self):
        a, b = np.empty((5, 0)), np.empty((2, 0))
        actual = supported_fit(a, np.arange(5.), b, .5)
        independent = independent_fit(a, np.arange(5.), b, .5, 0, 0)
        self.assertEqual(compare(compact(independent), compact(actual)), [])
        np.testing.assert_array_equal(independent["prediction_bound"], [0, 0])

    def test_native_dependency_requires_known_center_finite_full_footprint(self):
        image = np.ones((129, 129))
        self.assertFalse(_safe_dependency(image, [58, 64], 78, 64))
        self.assertTrue(_safe_dependency(image, [58.5, 64], 79, 64))
        self.assertFalse(_safe_dependency(image, None, 79, 64))
        image[64, 91] = np.nan
        self.assertFalse(_safe_dependency(image, [58.5, 64], 79, 64))

    def test_selection_is_geometry_only_not_successful_score(self):
        states = [dict(clip=c, frame_index=f, segment=s, track_id=t, archive={"path":"x"})
                  for c,f,s,t in (("0029",298,0,"bright:980"),("0126",140,0,"bright:1001"),
                                   ("0029",346,0,"bright:2641"),("0029",347,0,"bright:2641"))]
        for clip in ("0029", "0126", "0055", "0082"):
            states.extend([dict(clip=clip, frame_index=8, segment=0, track_id="bright:0", archive=None),
                           dict(clip=clip, frame_index=9, segment=0, track_id="bright:1", archive={})])
        chosen = choose_keys({"states":states})
        self.assertEqual(len(chosen), 8)
        self.assertTrue(all((clip,9,0,"bright:1") in chosen for clip in ("0029","0126","0055","0082")))


if __name__ == "__main__": unittest.main()
