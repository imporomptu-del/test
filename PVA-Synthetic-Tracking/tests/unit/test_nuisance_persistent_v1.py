import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("review_nuisance_persistent_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/review_nuisance_persistent_v1.py"
spec = importlib.util.spec_from_file_location("persistent_review", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def track(number=1, polarity="dark", segment=0, xy=(500, 500), qualified=True, measured=True):
    return dict(track_id=f"{polarity}:{number}", segment=segment, qualified_moving=qualified, measured=measured,
        source_xy=[4000.0, 2500.0], measurement_source_xy=list(xy) if measured else None,
        lifecycle="confirmed" if measured else "coasted")


def window():
    rows = [dict(frame_index=f, tracks=[]) for f in range(50, 106)]
    for i in (0, 1):
        rows[i]["tracks"] = [track(number=n, xy=(500 + 40 * n, 500)) for n in range(1, 9)]
    return rows


class PersistentTests(unittest.TestCase):
    def test_top8_rank_count_then_time_numeric_id(self):
        rows = window()
        rows[2]["tracks"] = [track(8), track(9), track(10)]
        ranking, selected, totals = m.select(rows)
        self.assertEqual(ranking[0]["numeric_track_id"], 8)
        self.assertEqual([r["numeric_track_id"] for r in ranking[-2:]], [9, 10])
        self.assertEqual(len(selected), 8)
        self.assertEqual(totals["qualified_measured_states"], 19)
        self.assertEqual(totals["selected_qualified_measured_states"], 17)

    def test_ties_segment_polarity_numeric_not_string(self):
        rows = window()
        for row in rows:
            row["tracks"] = []
        rows[0]["tracks"] = [track(10), track(2), track(2, "bright"), track(1, segment=1)]
        ranking, _, _ = m.select(rows)
        self.assertEqual([r["identity"] for r in ranking], ["0/bright:2", "0/dark:2", "0/dark:10", "1/dark:1"])

    def test_first_qualified_measurement_breaks_tie(self):
        rows = window()
        for row in rows:
            row["tracks"] = []
        rows[0]["tracks"] = [track(10)]
        rows[1]["tracks"] = [track(2)]
        ranking, _, _ = m.select(rows)
        self.assertEqual([r["numeric_track_id"] for r in ranking], [10, 2])

    def test_roi_uses_all_raw_measurements_not_filtered_positions(self):
        rows = window()
        rows[2]["tracks"] = [track(1, xy=(740, 500), qualified=False)]
        _, selected, _ = m.select(rows)
        item = selected[0]
        self.assertEqual(item["roi_center_source_xy"], [640, 500])
        self.assertEqual(item["qualified_measured_states"], 2)
        self.assertEqual(len(item["raw_measurements"]), 3)
        self.assertEqual(item["frames"][2]["raw_measurement_source_xy"], [740.0, 500.0])
        self.assertEqual(item["frames"][2]["filtered_track_source_xy"], [4000.0, 2500.0])

    def test_predictions_separate_not_roi_anchors(self):
        rows = window()
        rows[2]["tracks"] = [track(1, measured=False)]
        _, selected, totals = m.select(rows)
        item = selected[0]
        self.assertEqual(item["roi_center_source_xy"], [540, 500])
        self.assertEqual(item["qualified_predicted_states"], 1)
        self.assertEqual(totals["qualified_coasted_states"], 1)
        self.assertIsNone(item["frames"][2]["raw_measurement_source_xy"])
        self.assertFalse(item["frames"][2]["measured"])

    def test_no_edge_clamp_or_replacement(self):
        rows = window()
        rows[0]["tracks"][0]["measurement_source_xy"] = [50, 50]
        rows[1]["tracks"][0]["measurement_source_xy"] = [60, 60]
        ranking, selected, _ = m.select(rows)
        self.assertFalse(selected[0]["reviewable"])
        self.assertEqual(selected[0]["roi_source_xywh"], [-73, -73, 257, 257])
        self.assertEqual(selected[0]["identity"], ranking[0]["identity"])

    def test_long_span_keeps_fixed_roi_and_reports_outside(self):
        roi = m.fixed_roi([[500, 500], [900, 500]])
        self.assertEqual(roi["roi_source_xywh"], [572, 372, 257, 257])
        self.assertTrue(roi["reviewable"])
        self.assertTrue(roi["span_exceeds_crop"])
        self.assertEqual(roi["raw_measurements_outside_fixed_roi"], 2)

    def test_fixed_rounding_and_contact_endpoints(self):
        roi = m.fixed_roi([[500, 500], [501, 501]])
        self.assertEqual(roi["roi_center_source_xy"], [500, 500])
        _, selected, _ = m.select(window())
        self.assertEqual(len(selected[0]["contact_frames"]), 8)
        self.assertEqual(selected[0]["contact_frames"][0], 50)
        self.assertEqual(selected[0]["contact_frames"][-1], 51)
        self.assertTrue(selected[0]["contact_has_repeated_frames"])

    def test_coverage_is_state_fraction_not_object_count(self):
        rows = window()
        rows[2]["tracks"] = [track(9)]
        ranking, selected, workload = m.select(rows)
        self.assertEqual(workload["selected_qualified_measured_fraction"], 16 / 17)
        self.assertFalse(workload["counts_are_objects_or_false_positives"])
        self.assertTrue(all(r["physical_class"] == "unknown" for r in ranking))

    def test_duplicate_identity_and_missing_frame_rejected(self):
        rows = window()
        rows[0]["tracks"].append(copy.deepcopy(rows[0]["tracks"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            m.select(rows)
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            m.select(window()[:-1])

    def test_invalid_numeric_id_and_nonfinite_measurement(self):
        for name in ("dark:02", "other:2", "dark:-2"):
            with self.assertRaises(ValueError):
                m.track_key(dict(track(), track_id=name))
        rows = window()
        rows[0]["tracks"][0]["measurement_source_xy"] = [float("nan"), 5]
        with self.assertRaisesRegex(ValueError, "finite native"):
            m.select(rows)

    def test_native_crop_preserves_bgr_without_resize(self):
        frame = np.broadcast_to(np.array([3, 7, 11], np.uint8), (m.HEIGHT, m.WIDTH, 3))
        pixels = m.crop_native(frame, [500, 500, 257, 257])
        self.assertEqual(pixels.shape, (257, 257, 3))
        np.testing.assert_array_equal(pixels, frame[500:757, 500:757])
        with self.assertRaisesRegex(ValueError, "no clamping"):
            m.crop_native(frame, [-1, 500, 257, 257])

    def test_source_mismatch_and_unapproved_path_before_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "wrong.avi"
            path.write_bytes(b"generated wrong bytes")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                m.verify(path, m.SOURCE_SHA)
        with patch.object(m, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "unapproved"):
                m.build_plan(source=Path("/not-approved/chunk_0240.avi"))
            verify.assert_not_called()

    def test_no_overwrite_json_and_renderer(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "plan.json"
            m.write(file, {"original": True})
            with self.assertRaisesRegex(ValueError, "no overwrite"):
                m.write(file, {})
            self.assertEqual(json.loads(file.read_text()), {"original": True})
            with patch.object(m, "build_plan") as builder, patch("cv2.VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    m.render(file, Path(temp))
                builder.assert_not_called()
                decoder.assert_not_called()

    def test_plan_change_during_validation_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            file, output = Path(temp) / "plan.json", Path(temp) / "new-output"
            plan = {"selected": []}
            m.write(file, plan)
            def changed_plan():
                file.write_text(json.dumps(dict(plan, changed=True)))
                return plan
            with patch.object(m, "validate_plan", return_value=plan), patch.object(m, "build_plan", side_effect=changed_plan), patch("cv2.VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "changed before decoding"):
                    m.render(file, output)
                decoder.assert_not_called()

    def test_strict_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                m.decode(text)


if __name__ == "__main__":
    unittest.main()
