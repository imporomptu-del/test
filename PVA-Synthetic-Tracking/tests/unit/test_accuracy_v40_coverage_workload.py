import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v40_coverage_workload import inside, summarize_window


def track(tid="bright:1", measured=True, qualified=True, xy=(10, 20)):
    return dict(segment=0, track_id=tid, measured=measured,
                qualified_moving=qualified, measurement_source_xy=list(xy) if measured else None,
                source_xy=[99, 99] if measured else list(xy))


class CoverageWorkloadTest(unittest.TestCase):
    def setUp(self):
        self.window = dict(window_id="synthetic", frame_start=3, frame_end_inclusive=4,
                           crop_xywh=[10, 20, 4, 6])
        self.rows = [dict(frame_index=i, segment=0, candidates=[], tracks=[]) for i in (3, 4)]

    def test_half_open_edges(self):
        self.assertTrue(inside([10, 20], self.window["crop_xywh"]))
        for xy in ([14, 20], [10, 26], [9.999, 20], [10, 19.999]):
            self.assertFalse(inside(xy, self.window["crop_xywh"]))

    def test_measurements_not_filtered_positions(self):
        self.rows[0]["tracks"] = [track()]
        r = summarize_window(self.window, self.rows)
        self.assertEqual(r["counts"]["qualified_measured_states"], 1)
        self.assertEqual(r["frames"][0]["actual_measurements"][0]["source_xy"], [10, 20])

    def test_predictions_are_separate(self):
        self.rows[0]["tracks"] = [track(measured=False)]
        r = summarize_window(self.window, self.rows)
        self.assertEqual(r["counts"]["qualified_prediction_states"], 1)
        self.assertEqual(r["counts"]["qualified_measured_states"], 0)

    def test_candidate_and_unqualified_are_separate(self):
        self.rows[0]["candidates"] = [dict(source_xy=[11, 21], polarity="dark")]
        self.rows[0]["tracks"] = [track(qualified=False)]
        r = summarize_window(self.window, self.rows)
        self.assertEqual(r["counts"]["candidate_states"], 1)
        self.assertEqual(r["counts"]["actual_measured_states"], 1)
        self.assertEqual(r["counts"]["qualified_measured_states"], 0)
        self.assertIsNone(r["false_positive_count"])

    def test_same_identity_is_not_two_objects(self):
        for r in self.rows:
            r["tracks"] = [track()]
        r = summarize_window(self.window, self.rows)
        self.assertEqual(r["counts"]["qualified_measured_states"], 2)
        self.assertEqual(r["distinct_qualified_measured_ids"], 1)

    def test_inputs_unchanged(self):
        original = copy.deepcopy(self.rows)
        summarize_window(self.window, self.rows)
        self.assertEqual(self.rows, original)

    def test_missing_and_duplicate_frames_rejected(self):
        for rows in (self.rows[:1], [self.rows[0], self.rows[0]], self.rows[::-1]):
            with self.assertRaises(ValueError):
                summarize_window(self.window, rows)

    def test_malformed_tracks_rejected(self):
        for ts in ([track(), track()], [dict(track(), measured=1)],
                   [dict(track(measured=False), measurement_source_xy=[10, 20])],
                   [dict(track(), measurement_source_xy=[float("nan"), 20])]):
            self.rows[0]["tracks"] = ts
            with self.assertRaises(ValueError):
                summarize_window(self.window, self.rows)


if __name__ == "__main__":
    unittest.main()
