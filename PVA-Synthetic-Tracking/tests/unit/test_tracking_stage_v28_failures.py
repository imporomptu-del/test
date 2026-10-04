"""Supplemental whole-update failure/fallback checks; not timing inputs."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import test_tracking_stage_v28 as fixtures
import test_kalman_tracking as data
from replay_tracking_v27 import digest
from tracking_geometry_v20 import GeometryV20
from tracking_stage_v28 import TrackingStageV28
from tiny_target.tracking.kalman import KalmanTrackManager


class TrackingStageFailureTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.TrackingStageTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.TrackingStageTests.tearDownClass.__func__)

    def compare(self, reference, batch, **options):
        candidate = copy.deepcopy(reference)
        stage = TrackingStageV28(self.library)
        method = stage.adapt(GeometryV20(self.geometry).adapter(KalmanTrackManager.update))
        outputs = []
        for manager, update in ((reference, KalmanTrackManager.update), (candidate, method)):
            try:
                out = update(manager, batch, **options).to_dict()
                out.pop('timings_ms')
                outputs.append(('ok', digest(out)))
            except Exception as exc:
                outputs.append((type(exc).__name__, str(exc)))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(digest(vars(reference)), digest(vars(candidate)))
        return outputs[0], stage

    def seeded(self):
        manager = KalmanTrackManager(data.config(measurement_model='position_only', association_cost='gaussian_nll'))
        manager.update(data.batch(0, (0,), (data.candidate(0, 1., 2.), data.candidate(1, 90., 80.))))
        return manager

    def test_singular_update_keeps_exception_and_partial_state(self):
        manager = self.seeded()
        manager._tracks[1].covariance = -np.eye(4)
        result, stage = self.compare(manager, data.batch(1, (1,)))
        self.assertEqual(result[0], 'TemporalTrackingError')
        self.assertIn('singular', result[1])
        self.assertEqual(stage.innovation_fallbacks, 1)

    def test_nonfinite_and_strided_update_fallback(self):
        for covariance in (np.full((4, 4), np.nan), np.eye(4).T):
            manager = self.seeded()
            manager._tracks[1].covariance = covariance
            with np.errstate(all='ignore'):
                self.compare(manager, data.batch(100000000, (1,), (data.candidate(0, 1., 2.),)))

    def test_invalid_batch_and_argument_errors_keep_state(self):
        for frames, options in (((), {}), ((2, 1), {}), ((1, 1), {}), ((1,), {'include_quality_evidence': 1})):
            result, _ = self.compare(self.seeded(), data.batch(100000000, frames), **options)
            self.assertNotEqual(result[0], 'ok')


if __name__ == '__main__':
    unittest.main()
