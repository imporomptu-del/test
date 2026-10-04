import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from build_tracking_batch_v27 import build
from build_tracking_geometry_v20 import build as build_scalar
from check_tracking_stage_v28 import generated
from replay_tracking_v27 import digest
from tracking_stage_v28 import predict_all, innovation_batch, VictimIndex, record_values, TrackingStageV28
from tiny_target.tracking.kalman import KalmanTrackManager, TemporalTrackingError


class TrackingStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='seaqr_v28_test_')
        root = Path(cls.temp.name)
        build(root / 'batch')
        cls.library = root / 'batch/libtracking_batch_v27.so'
        cls.geometry = build_scalar(root / 'scalar')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generated_output_and_complete_private_state(self):
        result = generated(self.library, self.geometry)
        self.assertEqual(len(result['scenarios']), 36)
        self.assertTrue(all(r['exact'] and r['frames'] == 40 for r in result['scenarios']))
        self.assertGreater(result['innovation_batches'], 0)
        self.assertEqual(result['innovation_fallbacks'], 0)

    def test_prediction_identical_and_independently_owned(self):
        rng = np.random.default_rng(281)
        tracks = {i: SimpleNamespace(mean=rng.normal(size=4), covariance=np.eye(4),
                  state_timestamp_ns=(i % 3)*10000000, age_windows=i) for i in range(12)}
        manager = SimpleNamespace(config=SimpleNamespace(acceleration_process_sigma_px_s2=1.3), _tracks=tracks)
        expected = copy.deepcopy(manager)
        cache = {}
        for t in expected._tracks.values():
            t.mean, t.covariance = KalmanTrackManager._predicted_state(expected, t, 123456789, cache=cache)
            t.state_timestamp_ns = 123456789
            t.age_windows += 1
        predict_all(manager, 123456789)
        for i in tracks:
            self.assertEqual(digest(vars(tracks[i])), digest(vars(expected._tracks[i])))
        self.assertFalse(np.shares_memory(tracks[0].covariance, tracks[3].covariance))
        tracks[0].covariance[0, 0] = 99
        self.assertNotEqual(tracks[0].covariance[0, 0], tracks[3].covariance[0, 0])

    def test_prediction_backward_error(self):
        manager = SimpleNamespace(_tracks={0: SimpleNamespace(state_timestamp_ns=2)})
        with self.assertRaisesRegex(TemporalTrackingError, 'backward'):
            predict_all(manager, 1)

    def test_stacked_inverse_and_logdet_exact(self):
        rng = np.random.default_rng(282)
        for dimension in (2, 4):
            for gaussian in (False, True):
                tracks = {}
                for i in range(100):
                    a = rng.normal(size=(4, 4))
                    covariance = a @ a.T + np.eye(4)*10**(-i % 10)
                    tracks[i] = SimpleNamespace(covariance=covariance)
                tracks[100] = SimpleNamespace(covariance=tracks[0].covariance.copy())
                noise = np.eye(dimension)*.25
                result = innovation_batch(tracks, noise, gaussian)
                for i, track in tracks.items():
                    a = track.covariance[:dimension, :dimension] + noise
                    self.assertEqual(result[i][0].tobytes(), np.linalg.inv(a).tobytes())
                    self.assertEqual(result[i][1], float(np.linalg.slogdet(a)[1]) if gaussian else None)
                saved = result[0][0].tobytes()
                innovation_batch(tracks, noise*3, gaussian)
                self.assertEqual(saved, result[0][0].tobytes())

    def test_ill_conditioned_and_signed_zero_inverse(self):
        for dimension in (2, 4):
            for value in (1e-250, 1e-100, np.nextafter(1., 2.), 1e100, 1e250):
                a = np.diag([value, 1., 2., 3.])
                a[0, 1] = -0.
                result = innovation_batch({0: SimpleNamespace(covariance=a)}, np.zeros((dimension, dimension)), True)
                self.assertEqual(result[0][0].tobytes(), np.linalg.inv(a[:dimension, :dimension]).tobytes())
                self.assertEqual(result[0][1], float(np.linalg.slogdet(a[:dimension, :dimension])[1]))

    def test_innovation_fallback_and_empty(self):
        noise = np.eye(2)
        self.assertEqual(innovation_batch({}, noise, True), {})
        for a in (np.eye(4).astype(np.float32), np.eye(4).T, np.full((4, 4), np.nan), -np.eye(4)):
            self.assertIsNone(innovation_batch({0: SimpleNamespace(covariance=a)}, noise, True))
        self.assertIsNone(innovation_batch({i: SimpleNamespace(covariance=np.eye(4)) for i in range(513)}, noise, True))

    def test_victim_index_exact_with_changing_occupancies(self):
        rng = np.random.default_rng(283)
        for repeat in range(50):
            tracks = {i: SimpleNamespace(missed_windows=int(rng.integers(1, 6)),
                      independent_confirmation_hits=int(rng.integers(1, 5))) for i in range(200)}
            cells = {i: (int(rng.integers(0, 12)), 0) for i in tracks}
            occupancy = Counter(cells.values())
            replaceable = {i for i in tracks if i % 3}
            index = VictimIndex(tracks, cells, replaceable)
            for _ in range(50):
                chosen = (int(rng.integers(0, 12)), 0)
                eligible = [i for i in replaceable if occupancy[cells[i]] > occupancy[chosen] + 1]
                expected = max(eligible, key=lambda i: (occupancy[cells[i]], tracks[i].missed_windows,
                    -tracks[i].independent_confirmation_hits, -i)) if eligible else None
                self.assertEqual(index.take(occupancy, occupancy[chosen] + 1), expected)
                if expected is not None:
                    replaceable.remove(expected)
                    occupancy[cells[expected]] -= 1
                    occupancy[chosen] += 1

    def test_record_packing_bits_and_fallback(self):
        mean = np.array([-0., np.inf, -np.inf, np.nextafter(0., 1.)])
        covariance = np.arange(16, dtype=np.float64).reshape(4, 4)
        for a, b in ((mean, covariance), (mean[::-1], covariance.T), (mean.astype(np.float32), covariance.astype(np.float32))):
            expected = tuple(float(v) for v in a), tuple(tuple(float(v) for v in row) for row in b)
            self.assertEqual(digest(record_values(a, b)), digest(expected))

    def test_unknown_method_rejected(self):
        with self.assertRaisesRegex(ValueError, 'original v20'):
            TrackingStageV28(self.library).adapt(KalmanTrackManager.update)


if __name__ == '__main__':
    unittest.main()
