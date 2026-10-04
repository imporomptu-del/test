import copy
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v42_localized import prepare_components as prepare_v42
from accuracy_v43_components import _full_stamp_eligible, _raw_peak_masks, prepare_components


class ComponentsTests(unittest.TestCase):
    def fixture(self, centers=None, polarity="bright"):
        y, x = np.indices((129, 129))
        centers = [[48+2*i, 64] for i in range(8)] if centers is None else centers
        background = 70+.02*x+.03*y+3*np.sin(x/14)*np.sin(y/11)
        sign = 1 if polarity == "bright" else -1
        history = np.stack([background+sign*self.point(*center) if center is not None else background.copy()
                            for center in centers])
        return history, centers, [.1, -.2], polarity

    def point(self, cx, cy, value=35):
        y, x = np.indices((129, 129))
        return value*np.exp(-((x-cx)**2+(y-cy)**2)/2)

    def test_moving_source_cannot_create_repeat_supported_fixed_anchor(self):
        args = self.fixture()
        old, new = prepare_v42(*args), prepare_components(*args)
        self.assertGreater(old["metadata"]["persistent_anchor_count_before_cap"], 0)
        self.assertEqual(new["metadata"]["fixed_anchor_count"], 0)
        self.assertTrue(new["metadata"]["unresolved_removed_fixed_alternative"])
        self.assertTrue(new["metadata"]["fixed_alternative_coverage_incomplete"])

    def test_independently_flickering_fixed_source_away_from_foreground_retained(self):
        args = list(self.fixture(centers=[[29+4*i, 64] for i in range(8)]))
        for i in range(8):
            args[0][i] += self.point(80, 46, 20+3*i)
        result = prepare_components(*args)
        fixed = result["template_history"]["fixed"]
        self.assertTrue(any(record["anchor_xy"] == [80, 46] for record in fixed))
        record = next(record for record in fixed if record["anchor_xy"] == [80, 46])
        self.assertEqual(record["prior_history_indices"], list(range(8)))
        self.assertEqual(record["normalized_stamps"].shape, (8, 17, 17))
        self.assertTrue((record["raw_weighted_energy_dn2"] > 0).all())

    def test_foreground_pixel_poison_cannot_change_retained_fixed_template(self):
        history, centers, offset, polarity = self.fixture(centers=[[29+4*i, 64] for i in range(8)])
        for i in range(8):
            history[i] += self.point(80, 46, 20+3*i)
        before = prepare_components(history, centers, offset, polarity)
        poisoned = history.copy()
        y, x = np.indices((129, 129))
        for i, center in enumerate(centers):
            mask = np.maximum(np.abs(x-center[0]), np.abs(y-center[1])) <= 8
            poisoned[i, mask] += 50*np.sin(x[mask])+40
        after = prepare_components(poisoned, centers, offset, polarity)
        def pick(result):
            index = next(i for i, record in enumerate(result["template_history"]["fixed"])
                         if record["anchor_xy"] == [80, 46])
            return result["fixed_templates"][index], result["template_history"]["fixed"][index]
        a, ar = pick(before); b, br = pick(after)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ar["normalized_stamps"], br["normalized_stamps"])
        self.assertEqual(ar["prior_history_indices"], br["prior_history_indices"])

    def test_same_background_and_moving_templates_for_bright_dark_ablation(self):
        for polarity in ("bright", "dark"):
            args = self.fixture(polarity=polarity)
            original, result = prepare_v42(*args), prepare_components(*args)
            for key in ("background", "background_observation_counts", "moving_stamp", "moving_template"):
                if original[key] is None:
                    self.assertIsNone(result[key])
                else:
                    np.testing.assert_array_equal(original[key], result[key])
            self.assertEqual(original["metadata"]["usable_moving_stamp_indices"],
                             result["template_history"]["moving"]["prior_history_indices"])

    def test_full_dependency_not_just_peak_center_is_excluded(self):
        # Center at58; anchor78 is outside foreground radius8 but its native
        # footprint begins at66, touching the foreground boundary at66.
        args = list(self.fixture(centers=[[58, 64]]*8))
        for i in range(8): args[0][i] += self.point(78, 64, 60)
        result = prepare_components(*args)
        self.assertFalse(any(record["anchor_xy"] == [78, 64] for record in result["template_history"]["fixed"]))
        self.assertTrue(result["metadata"]["unresolved_removed_fixed_alternative"])

    def test_fractional_foreground_boundary_and_dark_fixed_anchor(self):
        args = list(self.fixture(centers=[[58.5, 64]]*8, polarity="dark"))
        for i in range(8): args[0][i] -= self.point(79, 64, 60)
        result = prepare_components(*args)
        # Native foreground ends at x66. Anchor79's footprint starts at67,
        # unlike anchor78 whose x66 boundary is contaminated.
        self.assertTrue(any(record["anchor_xy"] == [79, 64] for record in result["template_history"]["fixed"]))

    def test_four_anchor_cap_is_not_expanded(self):
        args = list(self.fixture(centers=[[15, 15]]*8))
        for cx, cy in ((47, 47), (60, 47), (77, 47), (47, 77), (60, 77), (77, 77)):
            args[0] += self.point(cx, cy, 40)
        result = prepare_components(*args)
        self.assertGreater(result["metadata"]["persistent_anchor_count_before_cap"], 4)
        self.assertLessEqual(result["metadata"]["fixed_anchor_count"], 4)
        self.assertTrue(result["metadata"]["anchor_dictionary_truncated"])

    def test_missing_coasted_positions_are_not_unmasked_fixed_learning_frames(self):
        history, centers, offset, polarity = self.fixture()
        centers = [None]*8
        for i in range(8): history[i] += self.point(80, 46, 60)
        result = prepare_components(history, centers, offset, polarity)
        self.assertEqual(result["metadata"]["fixed_anchor_count"], 0)
        self.assertEqual(result["metadata"]["missing_actual_position_history_indices"], list(range(8)))
        self.assertEqual(result["metadata"]["fixed_seed_opportunity_coverage"]["maximum"], 0)
        self.assertTrue(result["metadata"]["fixed_alternative_coverage_incomplete"])

    def test_crossing_and_hover_remain_unknown_not_absence(self):
        for centers in ([[64, 64]]*8, [[42+4*i, 64] for i in range(8)]):
            args = list(self.fixture(centers=centers))
            args[0] += self.point(64, 64, 30)
            result = prepare_components(*args)
            self.assertTrue(result["metadata"]["fixed_alternative_coverage_incomplete"])
            self.assertNotIn("fixed_explanation_absent", result["metadata"])
            self.assertNotIn("reject", result)

    def test_original_maxima_not_promoted_when_unsafe_bright_neighbor_removed(self):
        maps = np.zeros((8, 129, 129))
        maps[:, 64, 64] = 5
        maps[:, 64, 65] = 6
        masks = _raw_peak_masks(maps)
        self.assertFalse(masks[:, 64, 64].any())
        self.assertTrue(masks[:, 64, 65].all())
        eligible = np.ones_like(masks)
        eligible[:, 64, 65] = False
        self.assertFalse((masks & eligible)[:, 64, 64].any())

    def test_nan_saturation_and_border_disqualify_whole_fixed_stamp(self):
        highpass = np.ones((129, 129))
        eligible = _full_stamp_eligible(highpass)
        self.assertFalse(eligible[:8].any())
        self.assertFalse(eligible[:, :8].any())
        self.assertTrue(eligible[64, 64])
        highpass[64, 72] = np.nan
        self.assertFalse(_full_stamp_eligible(highpass)[64, 64])
        args = list(self.fixture(centers=[[29+4*i, 64] for i in range(8)]))
        args[0] += self.point(80, 46, 50)
        args[0][:, 46, 92] = np.nan  # dependency edge anchor80+12
        result = prepare_components(*args)
        self.assertFalse(any(record["anchor_xy"] == [80, 46] for record in result["template_history"]["fixed"]))

    def test_current_independent_signature_and_inputs_unmodified(self):
        self.assertEqual(list(inspect.signature(prepare_components).parameters),
                         ["history129", "prior_centers_xy", "predicted_offset_xy", "polarity"])
        args = self.fixture()
        original = args[0].copy(), copy.deepcopy(args[1]), list(args[2])
        result = prepare_components(*args)
        np.testing.assert_array_equal(args[0], original[0])
        self.assertEqual(args[1], original[1])
        self.assertEqual(args[2], original[2])
        json.dumps(result["metadata"], allow_nan=False)

    def test_raw_energy_matches_saved_normalized_history(self):
        result = prepare_components(*self.fixture(centers=[[29+4*i, 64] for i in range(8)]))
        record = result["template_history"]["moving"]
        np.testing.assert_allclose(record["raw_weighted_l2_norm_dn"]**2, record["raw_weighted_energy_dn2"])
        for stamp in record["normalized_stamps"]:
            self.assertAlmostEqual(float(np.nansum(stamp*stamp)), 1.0)
        self.assertEqual(len(record["prior_history_indices"]), len(record["raw_weighted_energy_dn2"]))


if __name__ == "__main__":
    unittest.main()
