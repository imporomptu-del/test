import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from audit_accuracy_v42_numerical import (archive_geometry_audit, compare, forecast_audit, independent_components,
                                         independent_localized, scalar_sample, solve)
from accuracy_v42_history import prepare_history
from accuracy_v42_localized import evaluate_localized, prepare_components


class NumericalAuditTests(unittest.TestCase):
    def fixture(self, polarity="bright"):
        y, x = np.indices((129, 129))
        background = 70+.03*x+.07*y+6*np.sin(x/9)*np.sin(y/13)
        centers = [[29+4*i, 64] for i in range(8)]
        def point(cx, cy):
            return 35*np.exp(-((x-cx)**2+(y-cy)**2)/2)
        sign = 1 if polarity == "bright" else -1
        history = np.stack([background+sign*point(*center) for center in centers])
        return background+sign*point(64, 64), history, centers, [0., 0.], polarity

    def test_independent_component_assembly_matches_synthetic(self):
        args = self.fixture()
        actual, independent = prepare_components(*args[1:]), independent_components(*args[1:])
        self.assertEqual(compare(independent["metadata"], actual["metadata"]), [])
        for name in ("background", "moving_template", "fixed_templates", "moving_stamp", "background_observation_counts"):
            np.testing.assert_allclose(independent[name], actual[name], atol=2e-8, equal_nan=True)

    def test_independent_available_checkerboard_fit_bright_and_dark(self):
        for polarity in ("bright", "dark"):
            args = self.fixture(polarity)
            independent = independent_localized(*args)
            actual = evaluate_localized(*args)
            self.assertTrue(independent["available"])
            self.assertEqual(compare(independent, actual), [])

    def test_missing_and_subpixel_stamps(self):
        current, history, centers, offset, polarity = self.fixture()
        centers[2] = None
        centers[5] = None
        centers[7] = [57.25, 64.5]
        history[:, 80, 85] = np.nan
        current[64, 64] = np.nan
        args = current, history, centers, [.2, -.3], polarity
        self.assertEqual(compare(independent_localized(*args), evaluate_localized(*args)), [])

    def test_unknown_foreground_is_also_audited(self):
        current, history, centers, offset, polarity = self.fixture()
        args = current, history, [None]*8, offset, polarity
        independently = independent_localized(*args)
        self.assertFalse(independently["available"])
        self.assertEqual(compare(independently, evaluate_localized(*args)), [])

    def test_fixed_flicker_component_and_nonnegative_constraint(self):
        current, history, centers, offset, polarity = self.fixture()
        y, x = np.indices((129, 129))
        first = np.exp(-((x-58)**2+(y-58)**2)/2)
        second = np.exp(-((x-73)**2+(y-67)**2)/2)
        for i in range(8):
            history[i] += (15+i)*first+(40-i)*second
        current += 60*first+10*second
        args = current, history, centers, offset, polarity
        actual = evaluate_localized(*args)
        self.assertGreaterEqual(actual["components"]["fixed_anchor_count"], 2)
        self.assertEqual(compare(independent_localized(*args), actual), [])

    def test_scalar_bilinear_positive_weight_contract(self):
        image = np.arange(25, dtype=float).reshape(5, 5)
        image[2, 3] = np.nan
        self.assertEqual(scalar_sample(image, 2, 2), 12)
        self.assertTrue(np.isnan(scalar_sample(image, 2.5, 2)))
        self.assertTrue(np.isnan(scalar_sample(image, -.1, 2)))

    def test_qr_solve_and_rank_deficient_fallback(self):
        design = np.asarray([[1, 0], [1, 1], [1, 2]], dtype=float)
        coefficient, rank = solve(design, np.asarray([1, 3, 5]))
        self.assertEqual(rank, 2)
        np.testing.assert_allclose(coefficient, [1, 2])
        coefficient, rank = solve(np.ones((3, 2)), np.ones(3))
        self.assertEqual(rank, 1)
        np.testing.assert_allclose(coefficient, [.5, .5])

    def test_forecast_from_recorded_prior_coordinates_independently_checked(self):
        frames = [np.full((180, 180), 80, dtype=np.uint8)]*9
        rows = []
        for index in range(9):
            transform = np.eye(3); transform[0, 2] = -index
            row = dict(frame_index=30+index, timestamp_ns=3_000_000_000+index*100_000_000,
                       segment=0, motion={"reset": False}, source_to_reference=transform.tolist())
            if index < 8:
                row["tracks"] = [dict(track_id="bright:0", predicted=False,
                                      measurement_source_xy=[80+index+.1*(index-8)**2, 80+.2*(index-8)**2])]
            rows.append(row)
        geometry = prepare_history(frames, rows, 0, "bright:0")["geometry"]
        self.assertTrue(forecast_audit(geometry)["passed"])
        broken = copy.deepcopy(geometry)
        broken["predicted_source_xy"][0] += 1
        self.assertFalse(forecast_audit(broken)["passed"])

    def test_comparison_detects_changed_scores_and_support(self):
        expected = dict(score=1.0, support=10, available=True)
        self.assertEqual(compare(expected, dict(expected, extra=1)), [])
        self.assertEqual(len(compare(expected, dict(score=1.1, support=9, available=False))), 3)

    def test_archive_must_match_separately_audited_geometry(self):
        inputs = dict(current129=np.ones((129, 129)), history129=np.ones((8, 129, 129)),
                      prior_centers_xy=np.full((8, 2), np.nan), predicted_offset_xy=np.asarray([.1, -.2]))
        geometry = dict(prior_centers_xy=[None]*8, predicted_offset_xy=[.1, -.2],
                        current_supported_pixels=129*129, history_supported_pixels=[129*129]*8)
        self.assertEqual(archive_geometry_audit(inputs, geometry), [])
        inputs["predicted_offset_xy"][0] = .3
        self.assertTrue(archive_geometry_audit(inputs, geometry))
        inputs["prior_centers_xy"][0, 0] = 1
        with self.assertRaises(ValueError):
            archive_geometry_audit(inputs, geometry)


if __name__ == "__main__":
    unittest.main()
