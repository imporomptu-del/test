import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v42_history import prepare_history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.image = (np.arange(200*220).reshape(200, 220) % 200 + 20).astype(np.uint8)
        self.frames = [self.image.copy() for _ in range(9)]
        self.rows = [dict(frame_index=20+i, timestamp_ns=2_000_000_000+i*100_000_000,
                          segment=3, motion={"reset": False},
                          source_to_reference=np.eye(3).tolist(),
                          tracks=[dict(track_id="bright:1", predicted=False,
                                       measurement_source_xy=[100+i, 100],
                                       source_xy=[9999, 9999], velocity_xy=[999, 999])])
                     for i in range(9)]

    def call(self, **kwargs):
        args = dict(frames=self.frames, rows=self.rows, segment=3, track_id="bright:1")
        args.update(kwargs)
        return prepare_history(**args)

    def test_constant_speed_native_grid_and_prior_centers(self):
        result = self.call()
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["geometry"]["predicted_source_xy"], [108, 100], atol=1e-10)
        np.testing.assert_equal(result["current129"], self.image[36:165, 44:173])
        np.testing.assert_equal(result["history129"], np.repeat(result["current129"][None], 8, axis=0))
        np.testing.assert_allclose(result["prior_centers_xy"], [[56+i, 64] for i in range(8)], atol=1e-10)
        self.assertEqual(result["geometry"]["forecast_fit_rank"], 3)
        self.assertLess(result["geometry"]["forecast_fit_rmse_reference_px"], 1e-10)

    def test_current_tracks_poison_cannot_change_any_output(self):
        class Explodes:
            def __iter__(self): raise AssertionError("current tracks read")
            def __len__(self): raise AssertionError("current tracks read")
            def __getitem__(self, key): raise AssertionError("current tracks read")
        baseline = self.call()
        for poison in [Explodes(), None, [], [{"track_id": "bright:1", "measurement_source_xy": [np.nan, np.inf],
                                             "source_xy": [-99999, 99999], "velocity_xy": [1e99, -1e99]}]]:
            self.rows[-1]["tracks"] = poison
            result = self.call()
            self.assertEqual(result["geometry"], baseline["geometry"])
            for key in ("current129", "history129", "prior_centers_xy", "predicted_offset_xy"):
                np.testing.assert_equal(result[key], baseline[key])

    def test_missing_current_track_field_is_accepted(self):
        del self.rows[-1]["tracks"]
        self.assertTrue(self.call()["available"])

    def test_quadratic_turn_uses_only_prior_points(self):
        for i, row in enumerate(self.rows[:-1]):
            t = i-8
            row["tracks"][0]["measurement_source_xy"] = [100 + .5*t*t, 100+2*t]
        self.rows[-1]["tracks"][0]["measurement_source_xy"] = [5000, -5000]
        result = self.call()
        np.testing.assert_allclose(result["geometry"]["predicted_source_xy"], [100, 100], atol=1e-10)
        self.assertGreater(result["prior_centers_xy"][0, 0], result["prior_centers_xy"][7, 0])

    def test_camera_shift_not_object_motion(self):
        for i, row in enumerate(self.rows):
            transform = np.eye(3)
            transform[:2, 2] = [-i, 2*i]
            row["source_to_reference"] = transform.tolist()
            row["tracks"][0]["measurement_source_xy"] = [100+i, 100-2*i]
        result = self.call()
        np.testing.assert_allclose(result["geometry"]["predicted_source_xy"], [108, 84], atol=1e-10)
        np.testing.assert_allclose(result["prior_centers_xy"], [[64, 64]]*8, atol=1e-10)
        np.testing.assert_equal(result["history129"][0], self.image[36:165, 36:165])
        np.testing.assert_equal(result["current129"], self.image[20:149, 44:173])

    def test_camera_plus_independent_object_direction(self):
        for i, row in enumerate(self.rows):
            transform = np.eye(3)
            transform[0, 2] = -i
            row["source_to_reference"] = transform.tolist()
            row["tracks"][0]["measurement_source_xy"] = [100+3*i, 100]
        result = self.call()
        np.testing.assert_allclose(result["geometry"]["predicted_source_xy"], [124, 100], atol=1e-10)
        np.testing.assert_allclose(result["prior_centers_xy"][:, 0], [48+2*i for i in range(8)], atol=1e-10)
        np.testing.assert_equal(result["history129"][0], self.image[36:165, 52:181])

    def test_missing_and_coast_do_not_substitute_filtered_predictions(self):
        self.rows[0]["tracks"] = []
        self.rows[2]["tracks"][0].update(predicted=True, measurement_source_xy=None, source_xy=[100000, 100000])
        self.rows[4]["tracks"] = [dict(track_id="bright:other", predicted=False,
                                       measurement_source_xy=[-100, -100])]
        result = self.call()
        self.assertTrue(result["available"])
        self.assertEqual(result["geometry"]["measured_prior_count"], 5)
        self.assertTrue(np.isnan(result["prior_centers_xy"][[0, 2, 4]]).all())
        np.testing.assert_allclose(result["geometry"]["predicted_source_xy"], [108, 100], atol=1e-10)
        self.rows[1]["tracks"] = []
        result = self.call()
        self.assertFalse(result["available"])
        self.assertIn("fewer_than_five", result["reasons"][0])

    def test_large_epoch_timestamps_preserve_small_differences(self):
        baseline = self.call()
        for row in self.rows: row["timestamp_ns"] += 1_700_000_000_000_000_013
        result = self.call()
        np.testing.assert_equal(result["current129"], baseline["current129"])
        np.testing.assert_equal(result["geometry"]["predicted_source_xy"], baseline["geometry"]["predicted_source_xy"])

    def test_no_future_or_missing_chronology_substitution(self):
        for key, bad in [("frame_index", 40), ("timestamp_ns", 5_000_000_000),
                         ("frame_index", 20), ("timestamp_ns", 2_100_000_000)]:
            rows = copy.deepcopy(self.rows)
            rows[3][key] = bad
            with self.subTest(key=key, bad=bad):
                self.assertFalse(self.call(rows=rows)["available"])
        for frames, rows in [(self.frames[:8], self.rows[:8]), (self.frames, self.rows[:8]),
                             (self.frames*2, self.rows*2)]:
            self.assertFalse(self.call(frames=frames, rows=rows)["available"])

    def test_reset_and_segment_boundaries_are_unknown(self):
        for i in range(9):
            rows = copy.deepcopy(self.rows)
            rows[i]["motion"]["reset"] = True
            self.assertEqual(self.call(rows=rows)["reasons"], ["motion_reset_within_nine_frames"])
        self.rows[2]["segment"] = 2
        self.assertEqual(self.call()["reasons"], ["segment_boundary"])

    def test_crop_origins_do_not_change_global_geometry(self):
        baseline = self.call()
        origins = [[20+i, 10+i] for i in range(9)]
        frames = [frame[y:y+175, x:x+190] for frame, (x, y) in zip(self.frames, origins)]
        result = self.call(frames=frames, origins=origins)
        for key in ("current129", "history129", "prior_centers_xy", "predicted_offset_xy"):
            np.testing.assert_equal(result[key], baseline[key])

    def test_round_half_up_center_and_fractional_offset(self):
        for row in self.rows[:-1]: row["tracks"][0]["measurement_source_xy"] = [100.49, 99.51]
        result = self.call()
        self.assertEqual(result["geometry"]["current_center_xy"], [100, 100])
        np.testing.assert_allclose(result["predicted_offset_xy"], [.49, -.49], atol=1e-10)

    def test_constant_exact_half_pixel_ties_round_toward_positive_infinity(self):
        for row in self.rows[:-1]: row["tracks"][0]["measurement_source_xy"] = [100.5, 99.5]
        result = self.call()
        self.assertEqual(result["geometry"]["current_center_xy"], [101, 100])
        np.testing.assert_equal(result["predicted_offset_xy"], [-.5, -.5])
        for row in self.rows[:-1]: row["tracks"][0]["measurement_source_xy"] = [-.5, -1.5]
        result = self.call()
        self.assertEqual(result["geometry"]["current_center_xy"], [0, -1])

    def test_border_and_saturation_are_unknown(self):
        for row in self.rows[:-1]: row["tracks"][0]["measurement_source_xy"] = [2, 3]
        result = self.call()
        self.assertTrue(np.isnan(result["current129"][:61]).all())
        self.assertTrue(np.isnan(result["current129"][:, :62]).all())
        self.frames[-1][3, 2] = 0
        self.frames[0][3, 2] = 255
        result = self.call()
        self.assertTrue(np.isnan(result["current129"][64, 64]))
        self.assertTrue(np.isnan(result["history129"][0, 64, 64]))

    def test_fractional_prior_warp_and_saturated_positive_weight_corner(self):
        # Keep all prior reference points fixed at (100.25,100.5).
        for row in self.rows[:-1]: row["tracks"][0]["measurement_source_xy"] = [100.25, 100.5]
        self.rows[-1]["source_to_reference"] = [[1, 0, .25], [0, 1, .5], [0, 0, 1]]
        result = self.call()
        expected = (self.image[36:165, 36:165].astype(float)*.375
                    + self.image[36:165, 37:166]*.125
                    + self.image[37:166, 36:165]*.375
                    + self.image[37:166, 37:166]*.125)
        np.testing.assert_allclose(result["history129"][0], expected)
        self.frames[0][100, 101] = 255
        self.assertTrue(np.isnan(self.call()["history129"][0, 64, 64]))

    def test_zero_weight_saturated_neighbor_not_support_requirement(self):
        self.frames[0][100, 109] = 255
        result = self.call()
        self.assertTrue(np.isfinite(result["history129"][0, 64, 64]))
        self.assertTrue(np.isnan(result["history129"][0, 64, 65]))

    def test_singular_and_projective_horizon_are_unknown(self):
        self.rows[2]["source_to_reference"] = np.zeros((3, 3)).tolist()
        self.assertEqual(self.call()["reasons"], ["singular_or_ill_conditioned_transform"])
        self.setUp()
        # Prior source-to-reference has inverse denominator 1-x/108.
        self.rows[0]["source_to_reference"] = [[1, 0, 0], [0, 1, 0], [1/108, 0, 1]]
        self.assertFalse(self.call()["available"])

    def test_malformed_inputs_raise(self):
        cases = []
        rows = copy.deepcopy(self.rows); rows[0]["tracks"] *= 2; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["tracks"][0]["measurement_source_xy"] = [True, 0]; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["tracks"][0]["predicted"] = True; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["tracks"][0]["measurement_source_xy"] = None; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["timestamp_ns"] = True; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["motion"]["reset"] = None; cases.append(dict(rows=rows))
        rows = copy.deepcopy(self.rows); rows[0]["source_to_reference"] = np.eye(2).tolist(); cases.append(dict(rows=rows))
        cases += [dict(frames=[self.image.astype(float)]*9), dict(segment=True),
                  dict(track_id=1), dict(origins=[[.5, 0]]*9), dict(origins=[[0, 0]]*8)]
        for index, kwargs in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError): self.call(**kwargs)

    def test_inputs_are_not_mutated(self):
        rows = copy.deepcopy(self.rows)
        images = [frame.copy() for frame in self.frames]
        self.call()
        self.assertEqual(rows, self.rows)
        for before, after in zip(images, self.frames): np.testing.assert_equal(before, after)


if __name__ == "__main__":
    unittest.main()
