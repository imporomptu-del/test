from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import accuracy_v50_calibration as calibration


class V50CalibrationTests(unittest.TestCase):
    def test_nine_units_produce_rank_nine_without_clamping(self):
        result = calibration.empirical_quantile([9, 2, 8, 3, 7, 4, 6, 5, 1])
        self.assertTrue(result["available"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["rank"], 9)
        self.assertEqual(result["q"], 9.0)
        self.assertEqual(result["total_units"], 9)
        self.assertEqual(result["finite_units"], 9)
        self.assertEqual(result["missing_units"], 0)

    def test_eight_units_are_insufficient_and_rank_is_not_clamped(self):
        result = calibration.empirical_quantile(list(range(8)))
        self.assertFalse(result["available"])
        self.assertEqual(result["rank"], 9)
        self.assertIsNone(result["q"])
        self.assertEqual(result["reasons"], ["requested_rank_exceeds_finite_units"])

    def test_more_units_use_integer_ceiling_rank_not_maximum(self):
        result = calibration.empirical_quantile(list(range(1, 21)))
        self.assertEqual(result["rank"], 19)
        self.assertEqual(result["q"], 19.0)

    def test_ties_and_zero_are_not_deduplicated_or_excluded(self):
        result = calibration.empirical_quantile([0] * 9)
        self.assertTrue(result["available"])
        self.assertEqual(result["finite_units"], 9)
        self.assertEqual(result["q"], 0.0)
        self.assertTrue(0.0 <= result["q"])
        self.assertFalse(np.nextafter(0.0, 1.0) <= result["q"])

    def test_quantile_comparison_is_inclusive_at_float_threshold(self):
        result = calibration.empirical_quantile([0.5] * 9)
        self.assertTrue(0.5 <= result["q"])
        self.assertFalse(np.nextafter(0.5, 1.0) <= result["q"])

    def test_missing_units_remain_counted_and_rank_uses_finite_units(self):
        result = calibration.empirical_quantile([None, *range(9), None])
        self.assertTrue(result["available"])
        self.assertEqual(result["total_units"], 11)
        self.assertEqual(result["finite_units"], 9)
        self.assertEqual(result["missing_units"], 2)
        self.assertEqual(result["rank"], 9)
        self.assertEqual(result["q"], 8.0)

    def test_missing_unit_can_make_rank_unavailable(self):
        result = calibration.empirical_quantile([None, *range(8)])
        self.assertFalse(result["available"])
        self.assertEqual(result["total_units"], 9)
        self.assertEqual(result["finite_units"], 8)
        self.assertEqual(result["missing_units"], 1)
        self.assertEqual(result["rank"], 9)
        self.assertIsNone(result["q"])

    def test_empty_and_all_missing_are_unavailable(self):
        for scores in ([], [None, None]):
            with self.subTest(scores=scores):
                result = calibration.empirical_quantile(scores)
                self.assertFalse(result["available"])
                self.assertIsNone(result["q"])
                self.assertEqual(result["finite_units"], 0)
                self.assertEqual(result["missing_units"], len(scores))
                self.assertEqual(result["rank"], 1)

    def test_quantile_rejects_invalid_score_even_after_missing(self):
        for invalid in (True, False, -1, -0.1, float("nan"), float("inf"),
                        -float("inf"), "2", complex(1), 10**400, np.bool_(True)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    calibration.empirical_quantile([None, invalid, *range(9)])

    def test_real_numpy_values_are_allowed(self):
        result = calibration.empirical_quantile([np.float64(1.5)] * 9)
        self.assertEqual(result["q"], 1.5)
        self.assertEqual(calibration.frame_score([np.int64(2), np.float32(1.5)]), 2.0)

    def test_target_ratio_validated(self):
        for numerator, denominator in ((0, 10), (-1, 10), (11, 10), (9, 0),
                                       (9, -1), (True, 10), (9, False),
                                       (9.0, 10), (9, 10.0)):
            with self.subTest(numerator=numerator, denominator=denominator):
                with self.assertRaises(ValueError):
                    calibration.empirical_quantile([1] * 9, numerator, denominator)

    def test_target_one_is_not_silently_relaxed(self):
        result = calibration.empirical_quantile([1] * 9, 1, 1)
        self.assertEqual(result["rank"], 10)
        self.assertFalse(result["available"])
        self.assertIsNone(result["q"])

    def test_custom_ratio_has_exact_integer_rank(self):
        result = calibration.empirical_quantile([1, 2, 3], 1, 2)
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["q"], 2.0)

    def test_frame_score_is_all_packet_maximum_never_mean_or_quantile(self):
        self.assertEqual(calibration.frame_score([1] * 99 + [100]), 100.0)
        self.assertEqual(calibration.frame_score([100] + [1] * 99), 100.0)
        self.assertEqual(calibration.frame_score([0]), 0.0)

    def test_any_missing_packet_invalidates_entire_frame_unit(self):
        for scores in ([None], [100, None, 1], [None, 100], [1, None]):
            with self.subTest(scores=scores):
                self.assertIsNone(calibration.frame_score(scores))

    def test_empty_frame_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            calibration.frame_score([])

    def test_frame_score_validates_all_packets_even_after_missing(self):
        for invalid in (True, -1, float("nan"), float("inf"), "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    calibration.frame_score([None, invalid])

    def test_greedy_anchors_are_sorted_unique_and_earliest(self):
        self.assertEqual(calibration.greedyanchors([28, 19, 10, 18, 1, 1, 9, 20]),
                         [1, 10, 19, 28])
        self.assertEqual(calibration.greedyanchors([11, 3, 12, 21, 20]), [3, 12, 21])

    def test_greedy_anchors_allow_exact_gap_and_empty_input(self):
        self.assertEqual(calibration.greedyanchors([0, 8, 9, 17, 18]), [0, 9, 18])
        self.assertEqual(calibration.greedyanchors([]), [])
        self.assertEqual(calibration.greedyanchors([3, 1, 2, 2], gap=1), [1, 2, 3])

    def test_frame_metadata_rejects_bool_negative_or_non_integer(self):
        for invalid in (True, False, -1, 1.5, 1.0, float("nan"), "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    calibration.greedyanchors([0, invalid])

    def test_gap_is_positive_integer(self):
        for gap in (0, -1, True, False, 9.0, float("inf"), "9"):
            with self.subTest(gap=gap):
                with self.assertRaises(ValueError):
                    calibration.greedyanchors([], gap=gap)
                with self.assertRaises(ValueError):
                    calibration.select_calibration_frames(
                        [], calibration.ALL_CALIBRATION_FRAMES, gap=gap)

    def test_two_named_policies_are_explicit_and_distinct(self):
        frames = [19, 10, 1, 2, 10, 18]
        self.assertEqual(calibration.POLICIES,
                         ("disjoint_anchor_frames", "all_calibration_frames"))
        self.assertEqual(calibration.select_calibration_frames(
            frames, calibration.DISJOINT_ANCHOR_FRAMES), [1, 10, 19])
        self.assertEqual(calibration.select_calibration_frames(
            frames, calibration.ALL_CALIBRATION_FRAMES), [1, 2, 10, 18, 19])
        with self.assertRaises(ValueError):
            calibration.select_calibration_frames(frames, "best_scored_frames")

    def test_missing_anchor_is_not_replaced_by_adjacent_scored_frame(self):
        frames = list(range(0, 81))
        packet_scores = {frame: [1.0] for frame in frames}
        packet_scores[9] = [None, 1.0]
        selected = calibration.select_calibration_frames(
            frames, calibration.DISJOINT_ANCHOR_FRAMES)
        self.assertEqual(selected, list(range(0, 81, 9)))
        scores = [calibration.frame_score(packet_scores[frame]) for frame in selected]
        result = calibration.empirical_quantile(scores)
        self.assertEqual(result["total_units"], 9)
        self.assertEqual(result["finite_units"], 8)
        self.assertEqual(result["missing_units"], 1)
        self.assertFalse(result["available"])
        self.assertNotIn(10, selected)

    def test_helpers_do_not_mutate_inputs(self):
        frames = [9, 0, 8, 0]
        scores = [2, None, 1]
        calibration.greedyanchors(frames)
        calibration.select_calibration_frames(frames, calibration.ALL_CALIBRATION_FRAMES)
        calibration.empirical_quantile(scores)
        calibration.frame_score(scores)
        self.assertEqual(frames, [9, 0, 8, 0])
        self.assertEqual(scores, [2, None, 1])


if __name__ == "__main__":
    unittest.main()
