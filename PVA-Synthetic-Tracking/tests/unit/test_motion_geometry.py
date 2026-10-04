from __future__ import annotations

import unittest

import numpy as np

from tiny_target.motion.geometry import (
    correspondence_acceptance_mask,
    grid_coverage,
    lift_points_to_full_resolution,
    lower_points_to_motion_resolution,
    select_spatially_distributed,
)


class MotionGeometryTests(unittest.TestCase):
    def test_pixel_center_coordinate_lift_round_trips_for_odd_sizes(self) -> None:
        points = np.array([[0.0, 0.0], [12.25, 9.75], [2389.0, 1592.0]], np.float32)
        full_size = (4783, 3187)
        motion_size = (2391, 1593)
        lifted = lift_points_to_full_resolution(points, motion_size, full_size)
        restored = lower_points_to_motion_resolution(lifted, full_size, motion_size)
        np.testing.assert_allclose(restored, points, atol=2e-4)
        self.assertAlmostEqual(lifted[0, 0], 0.5 * 4783 / 2391 - 0.5)

    def test_feature_selection_is_score_ranked_and_cell_limited(self) -> None:
        points = np.array(
            [[1, 1], [2, 2], [7, 1], [8, 2], [1, 7], [8, 8]], np.float32
        )
        scores = np.array([1, 9, 5, 4, 3, 2], np.float32)
        selected = select_spatially_distributed(
            points,
            scores,
            (10, 10),
            grid_rows=2,
            grid_cols=2,
            max_features=4,
            max_per_cell=1,
        )
        self.assertEqual(selected.tolist(), [1, 2, 4, 5])
        self.assertEqual(
            grid_coverage(points[selected], (10, 10), grid_rows=2, grid_cols=2),
            {"occupied_cells": 4, "total_cells": 4, "fraction": 1.0},
        )

    def test_feature_selection_honors_external_eligibility(self) -> None:
        selected = select_spatially_distributed(
            np.array([[1, 1], [2, 2], [8, 8]], np.float32),
            np.array([10, 9, 1], np.float32),
            (10, 10),
            grid_rows=2,
            grid_cols=2,
            max_features=3,
            eligible_mask=np.array([False, True, True]),
        )
        self.assertEqual(selected.tolist(), [1, 2])

    def test_correspondence_filter_accounts_for_each_failure_kind(self) -> None:
        previous = np.array(
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], np.float32
        )
        current = np.array(
            [[2, 1], [3, 2], [np.nan, 3], [30, 4], [15, 5], [7, 6]], np.float32
        )
        forward_status = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        backward = previous.copy()
        backward[-1] += 4
        backward_status = np.zeros(6, np.uint8)
        mask, rejected, fb_error = correspondence_acceptance_mask(
            previous,
            current,
            forward_status,
            image_size=(20, 20),
            max_displacement_px=5,
            backward_points=backward,
            backward_status=backward_status,
            max_forward_backward_error_px=2,
        )
        self.assertEqual(mask.tolist(), [True, False, False, False, False, False])
        self.assertEqual(
            rejected,
            {
                "nonfinite": 1,
                "forward_status": 1,
                "out_of_bounds": 1,
                "excessive_displacement": 1,
                "forward_backward": 1,
            },
        )
        self.assertAlmostEqual(float(fb_error[-1]), np.sqrt(32), places=5)


if __name__ == "__main__":
    unittest.main()
