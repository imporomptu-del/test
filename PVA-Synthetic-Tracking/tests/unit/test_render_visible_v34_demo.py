"""Presentation overlay semantics; generated pixels only, no input videos."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import render_visible_v34_demo as d


def track(name='bright:1', measured=True, qualified=True, measurement=(60,60), prediction=(90,90)):
    return dict(track_id=name, qualified_moving=qualified, measured=measured,
        measurement_source_xy=list(measurement) if measured else None, source_xy=list(prediction))


class DemoTests(unittest.TestCase):
    def test_coordinates_use_measurement_or_prediction(self):
        measured, coast = track(), track(name='bright:2', measured=False)
        records = d.selected(dict(tracks=[measured, coast]), (0,0,100,100))
        self.assertEqual(records[0][1], [60,60])
        self.assertEqual(records[1][1], [90,90])

    def test_qualified_filter_and_crop_boundary(self):
        tracks = [track(), track(qualified=False), track(measurement=(100,50)),
            track(measurement=(-1,50)), track(measurement=(0,0))]
        self.assertEqual(len(d.selected(dict(tracks=tracks), (0,0,100,100))), 2)

    def test_corrupt_measured_position_is_not_replaced_by_prediction(self):
        t = track()
        for xy in (None, [float('nan'), 2]):
            t['measurement_source_xy'] = xy
            with self.assertRaises(ValueError):
                d.selected(dict(tracks=[t]), (0,0,100,100))

    def test_native_left_source_is_exact_and_inputs_unchanged(self):
        frame = np.full((400,600,3), 115, np.uint8)
        row = dict(frame_index=150, motion=dict(status='accepted'), tracks=[track()])
        before = copy.deepcopy(row)
        canvas = d.canvas_for(frame, row, '0126', 10, 5.157, (0,0,300,300))
        np.testing.assert_array_equal(canvas[d.HEADER:d.HEADER+300, :300], frame[:300,:300])
        np.testing.assert_array_equal(frame, np.full((400,600,3),115,np.uint8))
        self.assertEqual(row, before)
        self.assertTrue(np.any(canvas[d.HEADER:d.HEADER+300,324:624] != frame[:300,:300]))

    def test_full_overview_keeps_entire_source_shape(self):
        frame = np.full((3190,4784,3), 115, np.uint8)
        out = d.canvas_for(frame, dict(frame_index=0, motion={}, tracks=[]), '0082', 10, 9.49)
        self.assertEqual(out.shape, (1284,1600,3))
        h = round(3190*1600/4784)
        np.testing.assert_array_equal(out[d.HEADER:d.HEADER+h], np.full((h,1600,3),115,np.uint8))

    def test_review_windows_stay_inside_full_frames(self):
        for clip, specs in d.ZOOMS.items():
            for name, first, last, (x,y,w,h), poster in specs:
                self.assertLessEqual(first, poster)
                self.assertLessEqual(poster, last)
                self.assertGreaterEqual(first, 0)
                self.assertLess(last, 674 if clip == '0126' else 687)
                self.assertLessEqual(x+w, 4784)
                self.assertLessEqual(y+h, 3190)


if __name__ == '__main__':
    unittest.main()
