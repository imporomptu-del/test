import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_nuisance_context_v1.py"
spec = importlib.util.spec_from_file_location("context_runner", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def rows():
    return [dict(frame_index=f, segment=0, motion=dict(accepted=True, reset=False),
                 source_to_reference=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) for f in range(465)]


def data():
    selected = []
    for identity, count in zip(m.SELECTED, (55, 52, 52, 32, 23, 18)):
        states = [dict(frame_index=f, measured=True, qualified_moving=f < 50+count,
                       raw_measurement_source_xy=[1000+f, 2000]) for f in range(50, 106)]
        selected.append(dict(identity=identity, polarity=identity.split("/")[1].split(":")[0], reviewable=True, frames=states))
    references = [dict(frame_index=f, measurement_source_xy=[2000+f, 2800]) for f in range(430, 465)]
    return dict(selected=selected), references, rows()


class RunnerTests(unittest.TestCase):
    def test_scope_and_count(self):
        plan, refs, history = data()
        out = m.build_requests(plan, refs, history)
        self.assertEqual(len(out), 267)
        self.assertEqual(sum(r["identity"] == "known_0240_pass" for r in out), 35)
        self.assertEqual(out, sorted(out, key=lambda r: (r["frame_index"], r["identity"])))
        self.assertTrue(all(r["prior_frame_index"] == r["frame_index"]-4 for r in out))

    def test_positive_first_four_history_unknown(self):
        out = m.build_requests(*data())
        positive = [r for r in out if r["identity"] == "known_0240_pass"]
        self.assertEqual([r["frame_index"] for r in positive if r["temporal_unavailable_reason"]], [430, 431, 432, 433])
        self.assertEqual(positive[4]["previous_source_xy"], [2430, 2800])

    def test_prior_prediction_not_measurement(self):
        plan, refs, history = data()
        plan["selected"][0]["frames"][0]["measured"] = False
        # Preserve frozen current count by promoting one other raw state.
        plan["selected"][0]["frames"][-1]["qualified_moving"] = True
        out = m.build_requests(plan, refs, history)
        r = next(r for r in out if r["identity"] == "0/dark:571" and r["frame_index"] == 54)
        self.assertIsNone(r["previous_source_xy"])
        self.assertIsNotNone(r["temporal_unavailable_reason"])

    def test_unqualified_actual_history_is_allowed(self):
        history = rows()
        r = m.request("id", "group", "dark", 60, [10, 20], [8, 20], history)
        self.assertIsNone(r["temporal_unavailable_reason"])

    def test_reset_barrier_anywhere_in_interval(self):
        for index in range(56, 61):
            history = rows()
            history[index]["motion"]["reset"] = True
            self.assertEqual(m.pair_reason(history, 60, [1, 2]), "reset_barrier")

    def test_segment_change_and_geometry_rejection(self):
        history = rows()
        history[57]["segment"] = 2
        self.assertEqual(m.pair_reason(history, 60, [1, 2]), "segment_change")
        history = rows()
        history[58]["motion"]["accepted"] = False
        self.assertEqual(m.pair_reason(history, 60, [1, 2]), "geometry_not_accepted")

    def test_unknown_geometry_and_reset_fail_closed(self):
        history = rows()
        del history[59]["motion"]["accepted"]
        self.assertEqual(m.pair_reason(history, 60, [1, 2]), "geometry_not_accepted")
        history = rows()
        del history[59]["motion"]["reset"]
        self.assertEqual(m.pair_reason(history, 60, [1, 2]), "reset_barrier")

    def test_noncontiguous_and_missing_prior(self):
        history = rows()
        history[57]["frame_index"] = 90
        self.assertEqual(m.pair_reason(history, 60, [1, 2]), "noncontiguous_history")
        self.assertEqual(m.pair_reason(history, 60, None), "prior_actual_measurement_unavailable_in_frozen_scope")

    def test_no_selection_replacement_or_missing_positive(self):
        plan, refs, history = data()
        plan["selected"][0]["identity"] = "0/dark:999"
        with self.assertRaisesRegex(ValueError, "selected identity"):
            m.build_requests(plan, refs, history)
        plan, refs, history = data()
        with self.assertRaisesRegex(ValueError, "complete frozen positive"):
            m.build_requests(plan, refs[:-1], history)

    def test_current_must_be_measured_and_qualified(self):
        plan, refs, history = data()
        plan["selected"][0]["frames"][0]["qualified_moving"] = False
        with self.assertRaisesRegex(ValueError, "request count"):
            m.build_requests(plan, refs, history)

    def test_strict_json(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                m.read_json(value)

    def test_write_no_overwrite_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "test.json"
            m.write_json(path, {"a": 1})
            m.verify(path, m.sha(path))
            with self.assertRaises(FileExistsError):
                m.write_json(path, {})
            with self.assertRaisesRegex(ValueError, "input changed"):
                m.verify(path, "0"*64)

    def test_coordinates_not_filtered_and_not_mutated(self):
        plan, refs, history = data()
        before = copy.deepcopy((plan, refs, history))
        out = m.build_requests(plan, refs, history)
        self.assertEqual((plan, refs, history), before)
        first = next(r for r in out if r["identity"] == "0/dark:571")
        self.assertEqual(first["current_source_xy"], [1050, 2000])

    def test_summary_unknowns_not_zeros(self):
        records = [dict(identity="a", review_group="unknown", spatial=None, pair=None,
                        temporal_status="missing"),
                   dict(identity="a", review_group="unknown", spatial=dict(A=2.0, interpretation_available=True),
                        pair=None, temporal_status="missing")]
        out = m.summarize(records)["a"]
        self.assertEqual(out["current_records"], 2)
        self.assertEqual(out["features"]["/spatial/A"]["n"], 1)
        self.assertEqual(out["features"]["/spatial/A"]["median"], 2)
        self.assertEqual(out["spatial_interpretation_available_count"], 1)

    def test_summary_separates_censored_or_degenerate(self):
        records = [dict(identity="a", review_group="unknown", spatial=dict(A=2.0, interpretation_available=True), pair=None, temporal_status="missing"),
                   dict(identity="a", review_group="unknown", spatial=dict(A=20.0, interpretation_available=False), pair=None, temporal_status="missing")]
        out = m.summarize(records)["a"]
        self.assertEqual(out["features"]["/spatial/A"]["n"], 2)
        self.assertEqual(out["interpretable_features"]["/spatial/A"]["median"], 2)
        self.assertEqual(out["noninterpretable_features"]["/spatial/A"]["median"], 20)

    def test_existing_output_never_decodes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cv2.VideoCapture") as decoder:
            with self.assertRaisesRegex(ValueError, "fresh output"):
                m.run(Path(tmp) / "absent.json", Path(tmp))
            decoder.assert_not_called()

    def test_changed_plan_never_decodes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cv2.VideoCapture") as decoder:
            plan = Path(tmp) / "plan.json"
            m.write_json(plan, {"different": True})
            with patch.object(m, "build_plan", return_value={"different": False}):
                with self.assertRaisesRegex(ValueError, "plan differs"):
                    m.run(plan, Path(tmp) / "result")
            decoder.assert_not_called()
            self.assertFalse((Path(tmp) / "result").exists())


if __name__ == "__main__":
    unittest.main()
