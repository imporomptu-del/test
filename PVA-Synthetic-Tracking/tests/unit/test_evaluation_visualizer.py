from __future__ import annotations

import unittest

import numpy as np

from tiny_target.evaluation_visualizer import (
    latest_completed_batch,
    map_reference_points_to_source,
)


class EvaluationVisualizerTests(unittest.TestCase):
    def test_reference_points_map_back_to_source(self) -> None:
        source_to_reference = np.array(
            [[1, 0, 5], [0, 1, -3], [0, 0, 1]], np.float64
        )
        mapped = map_reference_points_to_source(
            np.array([[15, 17], [25, 27]], np.float64), source_to_reference
        )
        np.testing.assert_allclose(mapped, [[10, 20], [20, 30]])

    def test_latest_batch_uses_only_completed_windows(self) -> None:
        batches = [
            {"candidate_batch": {"frame_indices": [7, 8, 9, 10]}},
            {"candidate_batch": {"frame_indices": [8, 9, 10, 11]}},
        ]
        self.assertIsNone(latest_completed_batch(9, batches))
        self.assertIs(latest_completed_batch(10, batches), batches[0])
        self.assertIs(latest_completed_batch(12, batches), batches[1])


if __name__ == "__main__":
    unittest.main()
