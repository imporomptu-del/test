"""Execution shortcuts must preserve masks, state, association and telemetry."""
import copy
import math
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from test_kalman_tracking import batch, candidate, config
from tiny_target.tracking import KalmanTrackManager
from tiny_target.tracking.kalman import TemporalTrackingError
from tiny_target.visible_learning import shape_learning_mask
from tiny_target.visible_resident import PEAK_DTYPE, decode_peak_cells


def dense_learning_reference(support, regions, margin):
    protected = np.zeros(support.shape, np.uint8)
    h, w = support.shape
    for region in regions:
        points = np.rint(np.asarray(region["support_reference_xy"], np.float64))
        inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        x, y = points[inside].astype(np.int64).T
        protected[y, x] = 1
    radius = math.ceil(margin)
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    disk = (x*x + y*y <= margin*margin).astype(np.uint8)
    protected = cv2.dilate(protected, disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return support & ~protected.astype(bool)


class HostEfficiencyTests(unittest.TestCase):
    def test_sparse_and_dense_dispatch_match_dilation_exactly(self):
        rng = np.random.default_rng(51062)
        for index in range(160):
            shape = ((int(rng.integers(1, 90)), int(rng.integers(1, 100)))
                     if index < 159 else (3190, 4784))
            support = rng.random(shape) > .05
            count = int(rng.integers(0, 600))
            points = rng.uniform([-20, -20], [shape[1]+20, shape[0]+20], (count, 2))
            points = np.concatenate((points, [[-.5, -.5], [.5, .5], [1.5, 1.5], [1e100, 1e100]]))
            regions = [dict(support_reference_xy=points.tolist())]
            margin = (1., 2., 2.5, 3.1, 8., 16.)[index % 6]
            before = support.copy()
            result = shape_learning_mask(support, regions, margin)
            np.testing.assert_array_equal(result, dense_learning_reference(support, regions, margin))
            np.testing.assert_array_equal(support, before)
            self.assertFalse(np.shares_memory(result, support))

    def test_empty_outside_duplicate_and_integer_masks(self):
        for support in (np.ones((20, 30), bool), np.full((20, 30), 2, np.uint8)):
            for regions in ([], [dict(support_reference_xy=[[-1, 0], [1e100, 1e100]])],
                            [dict(support_reference_xy=[[10, 10], [10, 10]])]):
                np.testing.assert_array_equal(shape_learning_mask(support, regions, 2),
                    dense_learning_reference(support, regions, 2))

    def test_omitted_quality_does_not_change_tracking_or_later_full_evidence(self):
        rng = np.random.default_rng(6013)
        a = KalmanTrackManager(config(max_active_tracks=80))
        b = KalmanTrackManager(config(max_active_tracks=80))
        for frame in range(50):
            observations = tuple(candidate(i, int(x), int(y)) for i, (x, y) in
                enumerate(rng.integers(0, 80, (30, 2))))
            current = batch(frame*100_000_000, (frame,), observations, segment=frame//25)
            full = a.update(current)
            small = b.update(current, include_quality_evidence=frame == 49)
            left, right = full.to_dict(), small.to_dict()
            left.pop("timings_ms"); right.pop("timings_ms")
            if frame != 49:
                for first, second in zip(left["tracks"], right["tracks"]):
                    self.assertEqual(second["quality_evidence"],
                        {"omitted": True, "reason": "caller_does_not_consume_quality_summary"})
                    first.pop("quality_evidence"); second.pop("quality_evidence")
            self.assertEqual(left, right)
            for key in a._tracks:
                self.assertEqual(a._quality_evidence(a._tracks[key]), b._quality_evidence(b._tracks[key]))

    def test_quality_request_is_explicit_boolean(self):
        manager = KalmanTrackManager(config())
        with self.assertRaises(ValueError):
            manager.update(batch(0, (0,)), include_quality_evidence=0)

    def test_covariance_reuse_is_exact_bounded_and_does_not_alias(self):
        manager = KalmanTrackManager(config(measurement_model="position_only",
            association_cost="gaussian_nll", max_active_tracks=12))
        observations = tuple(candidate(i, 30*i, 50) for i in range(8))
        manager.update(batch(0, (0,), observations))
        for frame, expected_unique in ((1, 1), (2, 2)):
            if frame == 2:
                manager._tracks[0].covariance[0, 0] += .25
            with patch("numpy.linalg.inv", wraps=np.linalg.inv) as inv, \
                 patch("numpy.linalg.slogdet", wraps=np.linalg.slogdet) as det, \
                 patch("numpy.linalg.solve", wraps=np.linalg.solve) as solve:
                result = manager.update(batch(frame*100_000_000, (frame,), observations))
            self.assertEqual(len(result.associations), 8)
            self.assertEqual(inv.call_count, expected_unique)
            self.assertEqual(det.call_count, expected_unique)
            self.assertEqual(solve.call_count, expected_unique)
            states = [t.covariance for t in manager._tracks.values()]
            self.assertTrue(all(not np.shares_memory(a, b)
                for i, a in enumerate(states) for b in states[i+1:]))
        manager.update(batch(300_000_000, (3,), observations, segment=1))
        with patch("numpy.linalg.inv", wraps=np.linalg.inv) as inv:
            manager.update(batch(400_000_000, (4,), observations, segment=1))
        self.assertEqual(inv.call_count, 1)

    def test_prediction_cache_keeps_distinct_dt_and_covariance(self):
        manager = KalmanTrackManager(config())
        manager.update(batch(0, (0,), (candidate(0, 30, 50),)))
        track = manager._tracks[0]
        cache = {}
        for dt_ns in (100_000_000, 200_000_001, 100_000_000):
            for change in (0., .25):
                current = copy.deepcopy(track)
                current.covariance[0, 0] += change
                expected = manager._predicted_state(current, dt_ns)
                actual = manager._predicted_state(current, dt_ns, cache=cache)
                for first, second in zip(expected, actual):
                    np.testing.assert_array_equal(first, second)
                actual[1][:] = 0  # A consumer cannot corrupt the cached result.
                again = manager._predicted_state(current, dt_ns, cache=cache)
                np.testing.assert_array_equal(expected[1], again[1])

    def test_singular_innovation_still_fails_explicitly(self):
        manager = KalmanTrackManager(config())
        manager.update(batch(0, (0,), (candidate(0, 30, 50),)))
        with patch.object(manager, "_predicted_state", return_value=(np.zeros(4), np.zeros((4, 4)))), \
             patch.object(manager, "_measurement_covariance", return_value=np.zeros((4, 4))):
            with self.assertRaisesRegex(TemporalTrackingError, "singular"):
                manager.update(batch(100_000_000, (1,), (candidate(0, 30, 50),)))

    def test_peak_conversion_preserves_values_order_and_empty_cells(self):
        rng = np.random.default_rng(8421)
        for shape in ((0, 12), (8, 0), (3, 7), (520, 12)):
            peaks = np.zeros(shape, PEAK_DTYPE)
            for name in ("x", "y"):
                peaks[name] = rng.integers(-1, 5000, shape)
            for name in ("score", "response", "noise"):
                peaks[name] = rng.uniform(-100, 100, shape).astype(np.float32)
            peaks["x"][rng.random(shape) < .8] = -1
            expected = []
            for j, cell in enumerate(peaks):
                selected = [dict(x=int(p["x"]), y=int(p["y"]), polarity="bright" if j % 2 == 0 else "dark",
                    score=float(p["score"]), response_dn=float(p["response"]), noise_sigma_dn=float(p["noise"]))
                    for p in cell if p["x"] >= 0]
                if selected:
                    expected.append(selected)
            self.assertEqual(decode_peak_cells(peaks), expected)
