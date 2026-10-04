"""Exact noise arithmetic and sparse access semantics across adversarial inputs."""
import unittest
import warnings
import numpy as np
from tiny_target.visible_noise import tile_noise_statistics
from tiny_target.visible_resident import SparseSpatial, sample_layout


def scalar_noise_reference(samples, support, layout, stride, floor):
    stats = np.empty((len(layout), 2), np.float32)
    sigmas = []
    for j, (ys, xs, offset, count) in enumerate(layout):
        sample = samples[offset:offset + count][support[ys, xs][::stride, ::stride].ravel()]
        center = float(np.median(sample)) if sample.size else 0.0
        sigma = max(floor, 1.4826 * float(np.median(np.abs(sample - center))) if sample.size else 0.0)
        stats[j] = center, sigma
        sigmas.append(sigma)
    return stats, float(np.median(sigmas))


class DetectionBatchingTests(unittest.TestCase):
    def test_noise_exact_odd_even_empty_masked_edge_tiles_and_dtypes(self):
        rng = np.random.default_rng(623014)
        for case in range(240):
            shape = (int(rng.integers(1, 180)), int(rng.integers(1, 220)))
            tile, stride = (8, 17, 32, 64)[case % 4], (1, 2, 3, 4, 7)[case % 5]
            layout, ids = sample_layout(shape, tile, stride)
            samples = rng.normal(size=len(ids)).astype((np.float32, np.float64)[case % 2])
            samples *= 10. ** ((case % 17) - 8)
            mask = rng.random(shape) > (case % 7) / 6
            floor = (.5, 1e-8, 2.25)[case % 3]
            if case % 13 == 0:
                samples[:] = 3
            before_samples, before_mask = samples.copy(), mask.copy()
            old = scalar_noise_reference(samples, mask, layout, stride, floor)
            new = tile_noise_statistics(samples, mask, layout, stride, floor)
            np.testing.assert_array_equal(old[0], new[0])
            self.assertEqual(old[1], new[1])
            np.testing.assert_array_equal(samples, before_samples)
            np.testing.assert_array_equal(mask, before_mask)

    def test_noise_native_layout_and_nonfinite_reference_semantics(self):
        rng = np.random.default_rng(53020)
        for shape in ((3190, 4784), (17, 17)):
            layout, ids = sample_layout(shape, 256, 4)
            samples = rng.normal(0, 3, len(ids)).astype(np.float32)
            support = np.ones(shape, bool)
            support[:6] = False; support[-6:] = False
            support[:, :6] = False; support[:, -6:] = False
            old = scalar_noise_reference(samples, support, layout, 4, .5)
            new = tile_noise_statistics(samples, support, layout, 4, .5)
            np.testing.assert_array_equal(old[0], new[0]); self.assertEqual(old[1], new[1])
        layout, ids = sample_layout((8, 16), 8, 1)
        mask = np.ones((8, 16), bool)
        for special in (np.nan, np.inf, -np.inf):
            values = np.full(len(ids), special, np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                old = scalar_noise_reference(values, mask, layout, 1, .5)
                new = tile_noise_statistics(values, mask, layout, 1, .5)
            np.testing.assert_array_equal(old[0], new[0]); self.assertEqual(old[1], new[1])

    def test_sparse_first_pixel_last_duplicate_window_semantics_and_no_alias(self):
        # Deliberately inconsistent overlapping patches preserve the old lookup
        # rules: first supplied pixel wins, but the last duplicate window wins.
        rng = np.random.default_rng(461)
        for case in range(80):
            shape = (45, 61)
            seeds = rng.integers([0, 0], [61, 45], (45, 2), dtype=np.int32)
            seeds = np.concatenate((seeds, [[20, 20], [20, 20], [8, 8]]))
            patches = rng.normal(size=(len(seeds), 17, 17)).astype(np.float32)
            if case % 2:
                patches = patches[:, ::-1, :]
            expected, windows = {}, {}
            for (x, y), patch in zip(seeds, patches):
                if x < 8 or y < 8 or x + 8 >= shape[1] or y + 8 >= shape[0]:
                    continue
                windows[y - 8, x - 8] = patch
                for dy in range(17):
                    for dx in range(17):
                        expected.setdefault((int(y) - 8 + dy) * shape[1] + int(x) - 8 + dx, patch[dy, dx])
            sparse = SparseSpatial(shape, seeds, patches)
            keys = np.asarray(sorted(expected), np.int64)
            np.testing.assert_array_equal(sparse.keys, keys)
            np.testing.assert_array_equal(sparse.values, [expected[k] for k in keys])
            yy, xx = np.divmod(keys, shape[1])
            np.testing.assert_array_equal(sparse[yy, xx], sparse.values)
            for (y, x), patch in windows.items():
                np.testing.assert_array_equal(sparse[y:y + 17, x:x + 17], patch)
            self.assertFalse(np.shares_memory(sparse.values, patches))


if __name__ == "__main__":
    unittest.main()
