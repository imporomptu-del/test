"""Generated journal/pixel fixtures only; never opens archived or real media."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import render_discovery_pair_review as r


def track(tid="bright:1", point=(300, 300), *, measured=True, qualified=True):
    return dict(track_id=tid, segment=0, measured=measured, qualified_moving=qualified,
        source_xy=[900., 800.], measurement_source_xy=list(point) if measured else None)


def row(index, tracks=None):
    return dict(frame_index=index, timestamp_ns=index * 100_000_000, segment=0, tracks=tracks or [])


def channels(rows):
    policy = r.ObservationOutput("0170")
    return [policy.update(v) for v in rows]


class SelectionTests(unittest.TestCase):
    def selection(self, rows, frames=60):
        return r.select_windows(channels(rows), frames, Fraction(10), 1200, 1000)

    def test_rank_net_displacement_not_sample_count_and_anchor_actual(self):
        rows = [row(i) for i in range(60)]
        rows[0] = row(0, [track("a", (300, 300)), track("b", (500, 500))])
        rows[1] = row(1, [track("a", (310, 300)), track("b", (530, 500))])
        out = self.selection(rows)
        winner = out["windows"][0]
        self.assertEqual(winner["identity"], "0170/0/b")
        self.assertEqual(winner["anchor_xy"], [500, 500])
        self.assertEqual(winner["crop_xywh"], [308, 308, 384, 384])
        self.assertEqual((winner["first_frame"], winner["last_frame_inclusive"]), (0, 39))
        self.assertEqual(out["candidate_identity_bin_count"], 2)
        self.assertEqual(out["omitted_identity_bin_count"], 1)

    def test_coasts_and_unqualified_cannot_anchor(self):
        rows = [row(i, [track(measured=False), track("b", qualified=False)]) for i in range(60)]
        self.assertEqual(self.selection(rows)["windows"], [])

    def test_no_duplicate_identity_and_no_cross_bin_replacement(self):
        rows = [row(i) for i in range(60)]
        rows[0] = row(0, [track("a")])
        rows[10] = row(10, [track("a")])
        rows[20] = row(20, [track("b")])
        result = self.selection(rows)
        self.assertEqual([v["bin"] for v in result["windows"]], [0, 2])
        self.assertEqual(result["bins"][1]["candidates"][0]["omission_reason"], "identity_selected_in_earlier_bin")

    def test_fallback_earliest_then_identity(self):
        rows = [row(i) for i in range(60)]
        rows[2] = row(2, [track("z"), track("a")])
        rows[3] = row(3, [track("b")])
        self.assertEqual(self.selection(rows)["windows"][0]["identity"], "0170/0/a")

    def test_displacement_tie_first_frame_then_identity(self):
        rows = [row(i) for i in range(60)]
        rows[1] = row(1, [track("z"), track("a")])
        rows[2] = row(2, [track("b")])
        rows[3] = row(3, [track("z", (305, 300)), track("a", (305, 300)), track("b", (305, 300))])
        self.assertEqual(self.selection(rows)["windows"][0]["identity"], "0170/0/a")

    def test_two_samples_outrank_one_even_zero_displacement(self):
        rows = [row(i) for i in range(60)]
        rows[0] = row(0, [track("single")])
        rows[1] = row(1, [track("repeat")])
        rows[2] = row(2, [track("repeat")])
        self.assertEqual(self.selection(rows)["windows"][0]["identity"], "0170/0/repeat")

    def test_uneven_bins_and_last_window_truncation(self):
        rows = [row(i, [track(str(i))]) for i in range(61)]
        result = self.selection(rows, 61)
        self.assertEqual([b["first_frame"] for b in result["bins"]], [0, 11, 21, 31, 41, 51])
        self.assertEqual([v["anchor_frame"] for v in result["windows"]], [0, 11, 21, 31, 41, 51])
        self.assertEqual(result["windows"][-1]["last_frame_inclusive"], 60)

    def test_offscreen_anchor_retained_not_clamped(self):
        rows = [row(i) for i in range(60)]
        rows[0] = row(0, [track("off", (-1, 50))])
        result = self.selection(rows)
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["bins"][0]["candidates"][0]["first_xy"], [-1, 50])
        self.assertEqual(result["bins"][0]["candidates"][0]["omission_reason"], "offscreen_first_measurement")

    def test_crop_shift_preserves_extent(self):
        self.assertEqual(r.crop_at([1.2, 2.3], 1200, 1000), [0, 0, 384, 384])
        self.assertEqual(r.crop_at([1199.9, 999.9], 1200, 1000), [816, 616, 384, 384])


class JournalTests(unittest.TestCase):
    def read(self, values, frames=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text("".join(json.dumps(v) + "\n" for v in values))
            return list(r.journal_rows(path, frames, Fraction(10), "0170"))

    def test_complete_empty_frames(self):
        self.assertEqual(len(self.read([row(0), row(1)])), 2)

    def test_missing_extra_reordered_duplicate(self):
        for rows in ([row(0)], [row(0), row(1), row(2)], [row(1), row(0)], [row(0), row(0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read(rows)

    def test_bad_timestamp(self):
        bad = row(1)
        bad["timestamp_ns"] += 1
        with self.assertRaises(ValueError):
            self.read([row(0), bad])

    def test_nonfinite_measurement_and_false_coast_measurement(self):
        for t in (track(point=(float("nan"), 1)), dict(track(measured=False), measurement_source_xy=[1, 2])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.read([row(0, [t]), row(1)])

    def test_duplicate_json_and_exponent(self):
        for raw in ('{"x":1,"x":2}', '{"x":1e999}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                r.decode(raw)

    def test_policy_preserves_measurements_and_prediction_age(self):
        result = self.read([row(0, [track()]), row(1, [track(measured=False)])])
        self.assertEqual(result[0][1]["observation_alerts"][0]["source_xy"], [300, 300])
        self.assertEqual(result[1][1]["observation_alerts"], [])
        context = result[1][1]["track_context"][0]
        self.assertEqual(context["source_xy"], [900, 800])
        self.assertEqual(context["last_measurement_age_ns"], 100_000_000)


class CanvasTests(unittest.TestCase):
    def test_native_raw_left_exact_no_mutation_all_nearby(self):
        image = np.full((1000, 1200, 3), 117, np.uint8)
        c = channels([row(0, [track("a"), track("b", (350, 350)), track("c", measured=False)])])[0]
        before = copy.deepcopy(c)
        canvas = r.canvas_for(image, c, "0170", Fraction(10), [150, 150, 384, 384])
        np.testing.assert_array_equal(canvas[r.HEADER:r.HEADER + 384, :384], image[150:534, 150:534])
        self.assertTrue(np.any(canvas[r.HEADER:r.HEADER + 384, 400:784] != 117))
        self.assertEqual(len(r.records_inside(c["observation_alerts"], [150, 150, 384, 384])), 2)
        self.assertEqual(c, before)
        self.assertTrue(np.all(image == 117))

    def test_overview_raw_left_exact_global_resize(self):
        import cv2
        image = np.arange(1000 * 1200 * 3, dtype=np.uint8).reshape(1000, 1200, 3)
        c = channels([row(0, [track()])])[0]
        canvas = r.canvas_for(image, c, "0240", Fraction(10))
        expected = cv2.resize(image, (960, 800), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(canvas[r.HEADER:r.HEADER + 800, :960], expected)

    def test_offscreen_records_do_not_create_edge_markers(self):
        image = np.full((1000, 1200, 3), 117, np.uint8)
        c = channels([row(0, [track("a", (-1, 50)), track("b", (1200, 50))])])[0]
        panel = image.copy()
        r.annotate(panel, c["track_context"], [0, 0, 1200, 1000])
        np.testing.assert_array_equal(panel, image)


if __name__ == "__main__":
    unittest.main()
