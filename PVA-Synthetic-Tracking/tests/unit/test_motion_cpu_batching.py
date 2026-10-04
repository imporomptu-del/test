from dataclasses import replace
import unittest

import numpy as np

from tiny_target.motion.geometry import grid_coverage, select_spatially_distributed
from tiny_target.motion.pva_pyrlk import (
    PvaMotionConfig, _feature_eligibility, _unsaturated_neighborhoods,
)
from tiny_target.types import Frame, TimestampSource


def frame(image, mask=None, bits=16):
    return Frame(image=image, valid_mask=mask, bit_depth=bits, timestamp_ns=0,
                 frame_index=0, source_id="generated_cpu_parity",
                 timestamp_source=TimestampSource.MANIFEST)


class MotionCpuBatchingTests(unittest.TestCase):
    def test_opt_in_config_and_positional_contract(self):
        self.assertEqual(PvaMotionConfig(.5).feature_cpu_policy, "reference")
        PvaMotionConfig(feature_cpu_policy="batched_exact_v1")
        with self.assertRaises(ValueError):
            PvaMotionConfig(feature_cpu_policy="approximate")

    def test_saturation_edges_radius_thresholds_and_chunk_boundaries(self):
        rng = np.random.default_rng(52941)
        for shape in ((1, 1), (2, 5), (17, 21)):
            for dtype, limit in ((np.uint8, 255), (np.uint16, 65535), (np.float32, 500)):
                image = rng.integers(0, limit + 1, shape).astype(dtype)
                image.flat[0] = limit
                if dtype == np.float32 and image.size > 1:
                    image.flat[1] = np.nan
                yy, xx = np.indices(shape)
                centers = np.column_stack((xx.ravel(), yy.ravel()))
                for radius in (0, 1, 2, 8, 9, 32):
                    for level in (0., limit * .995, float(limit), limit + .5):
                        expected = _unsaturated_neighborhoods(image, centers, radius, level)
                        actual = _unsaturated_neighborhoods(image, centers, radius, level,
                            execution="batched_exact_v1", batch_points=7)
                        np.testing.assert_array_equal(actual, expected)

    def test_eligibility_randomized_reasons_and_source_immutability(self):
        rng = np.random.default_rng(29171)
        for case in range(64):
            height, width = (int(v) for v in rng.integers(3, 100, 2))
            bits = (8, 10, 12, 16)[case % 4]
            image = rng.integers(0, 1 << bits, (height, width), dtype=np.uint16)
            mask = None if case % 3 == 0 else rng.random(image.shape) > .2
            source = frame(image, mask, bits)
            size = (max(1, width // 2), max(1, height // 2))
            points = rng.uniform(-10, max(size) + 10, (250, 2)).astype(np.float32)
            points[:4] = [[np.nan, 0], [0, np.inf], [-np.inf, 0], [0, 0]]
            cfg = PvaMotionConfig(feature_border_px=case % 5,
                saturation_fraction=(.1, .995, 1.)[case % 3],
                saturated_neighborhood_radius_px=(0, 1, 2, 8, 9)[case % 5],
                exclusion_regions_xyxy=((1., 1., 7., 13.),))
            before = source.pixel_sha256()
            expected, reasons = _feature_eligibility(points, source, size, cfg)
            actual, actual_reasons = _feature_eligibility(points, source, size,
                replace(cfg, feature_cpu_policy="batched_exact_v1"))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_reasons, reasons)
            self.assertEqual(before, source.pixel_sha256())

    def compare_selection(self, points, scores, size, rows, cols, maximum, quota, mask):
        options = dict(grid_rows=rows, grid_cols=cols, max_features=maximum,
                       max_per_cell=quota, eligible_mask=mask)
        expected = select_spatially_distributed(points, scores, size, **options)
        actual = select_spatially_distributed(points, scores, size,
                                             execution="batched_exact_v1", **options)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(grid_coverage(points, size, grid_rows=rows, grid_cols=cols),
            grid_coverage(points, size, grid_rows=rows, grid_cols=cols,
                          execution="batched_exact_v1"))

    def test_stable_quotas_boundaries_nonfinite_and_randomized_selection(self):
        rng = np.random.default_rng(67213)
        for size, rows, cols in (((2392, 1595), 6, 8), ((997, 613), 7, 11),
                                 ((17, 19), 29, 23), ((1, 1), 1, 1)):
            width, height = size
            boundary = []
            for y in np.linspace(0, height, rows + 1).astype(np.float32):
                for x in np.linspace(0, width, cols + 1).astype(np.float32):
                    boundary.extend([[np.nextafter(x, np.float32(-np.inf)), y],
                        [x, y], [np.nextafter(x, np.float32(np.inf)), y],
                        [x, np.nextafter(y, np.float32(-np.inf))],
                        [x, np.nextafter(y, np.float32(np.inf))]])
            points = np.vstack((boundary,
                rng.uniform([-1, -1], [width + 1, height + 1], (3000, 2)),
                [[np.nan, 1], [1, np.inf], [-np.inf, 1]])).astype(np.float32)
            scores = rng.integers(-2, 5, len(points)).astype(np.float32)
            scores[::101] = np.nan
            scores[::103] = np.inf
            for quota in (None, 1, 5, 21):
                for maximum in (1, 16, 1000):
                    mask = rng.random(len(points)) > .1
                    self.compare_selection(points, scores, size, rows, cols, maximum, quota, mask)

    def test_empty_and_no_eligible_points(self):
        for points in (np.empty((0, 2), np.float32), np.array([[np.nan, 0], [-1, -1]], np.float32)):
            self.compare_selection(points, np.ones(len(points)), (13, 9), 6, 8, 1000,
                                   None, np.zeros(len(points), bool))

    def test_unusual_parameters_keep_reference_path(self):
        points = np.array([[1, 1], [2, 2], [7, 7]], np.float32)
        self.compare_selection(points, np.array([3, 2, 1]), (10, 10),
                               2, 2, 1.5, 1.5, None)


if __name__ == "__main__":
    unittest.main()
