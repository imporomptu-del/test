"""BLAS and batched-matmul graphs must never be substituted for each other."""
import unittest
from unittest.mock import patch
import numpy as np
from test_kalman_tracking import config, candidate, batch
from tiny_target.tracking import KalmanTrackManager
from tiny_target.tracking.quadratic import execution_plan, mahalanobis_values


class ExactQuadraticTests(unittest.TestCase):
    def test_installed_graph_matches_original_over_shapes_scales_and_layouts(self):
        rng = np.random.default_rng(808)
        for dimension in (2, 4):
            for count in (0, 1, 2, 3, 4, 8, 9, 17, 32, 65, 128, 257, 512):
                for index in range(50):
                    residual = rng.normal(size=(count, dimension))*10.**rng.uniform(-8, 8)
                    a = rng.normal(size=(dimension, dimension))
                    inverse = np.linalg.inv(a @ a.T + np.eye(dimension)*.05)
                    if index % 3 == 1:
                        residual = np.asfortranarray(residual)
                    elif index % 3 == 2:
                        inverse = np.asfortranarray(inverse)
                    expected = np.einsum("ni,ij,nj->n", residual, inverse, residual, optimize=True)
                    np.testing.assert_array_equal(mahalanobis_values(residual, inverse), expected)

    def test_unknown_plan_and_unsupported_dimensions_keep_reference(self):
        rng = np.random.default_rng(29)
        for dimension in (2, 3, 4):
            residual = rng.normal(size=(10, dimension))
            inverse = np.eye(dimension)
            expected = np.einsum("ni,ij,nj->n", residual, inverse, residual, optimize=True)
            path = execution_plan(10, dimension)[1]
            with patch("tiny_target.tracking.quadratic.execution_plan", return_value=("reference", path)):
                np.testing.assert_array_equal(mahalanobis_values(residual, inverse), expected)
        self.assertEqual(execution_plan(10, 3)[0], "reference")

    def test_geometrically_impossible_pairs_skip_only_likelihood(self):
        manager = KalmanTrackManager(config())
        manager.update(batch(0, (0,), (candidate(0, 10, 10),)))
        with patch("tiny_target.tracking.kalman._mahalanobis_values", wraps=mahalanobis_values) as likelihood:
            result = manager.update(batch(100_000_000, (1,), (candidate(0, 1000, 1000),)))
        self.assertEqual(likelihood.call_count, 0)
        self.assertEqual(result.metrics["rejected_pair_counts"],
            dict(position_gate=1, velocity_gate=0, mahalanobis_gate=0))
        self.assertFalse(result.associations)


if __name__ == "__main__":
    unittest.main()
