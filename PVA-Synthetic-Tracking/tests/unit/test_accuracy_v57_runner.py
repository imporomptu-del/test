"""Generated metadata only: no archived journals, labels or camera media."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_accuracy_v57 as runner
from accuracy_v57_persistence import PersistenceConfig


def fixture(frame=0, measured=True, qualified=True, margin=-0.5):
    track = dict(track_id="bright:1", segment=0, measured=measured,
                 qualified_moving=qualified, source_xy=[10., 10.],
                 measurement_source_xy=[10., 10.] if measured else None)
    row = dict(frame_index=frame, timestamp_ns=frame * 100000000, segment=0,
               motion=dict(reset=False), tracks=[track])
    saved = {k: row[k] for k in ("frame_index", "timestamp_ns", "segment")}
    saved["tracks"] = []
    if qualified:
        saved["tracks"] = [dict(
            **{k: track[k] for k in ("track_id", "segment", "measured", "source_xy", "measurement_source_xy")},
            accepted=False, reason="edge_preferred_or_tie" if measured else "coast_edge",
            measurement_frame=frame if measured else frame - 1,
            features=dict(informative=True, point_minus_edge_fraction=margin) if measured else None)]
    return row, saved


def sample(assigned="0/bright:1"):
    return dict(panel="generated", clip_id="fixture", window_id="w", frame_index=1,
                source_xy=[10., 10.], position_uncertainty_px=0, polarity="bright",
                actual=dict(hit=True, assigned_id="0/bright:1", all_gated_ids=["0/bright:1"]),
                baseline=dict(hit=assigned is not None, assigned_id=assigned,
                              all_gated_ids=["0/bright:1"] if assigned else []))


class ParseTests(unittest.TestCase):
    def test_duplicate_json_fails(self):
        with self.assertRaises(ValueError): runner.decode('{"a":1,"a":2}')

    def test_nonfinite_fails(self):
        for s in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            with self.subTest(s=s), self.assertRaises(ValueError): runner.decode(s)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            runner.write(p, {"original": True})
            with self.assertRaises(FileExistsError): runner.write(p, {})
            self.assertEqual(runner.read(p), {"original": True})

    def test_blank_jsonl_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_text('{}\n\n')
            with self.assertRaises(ValueError): list(runner.lines(p))

    def test_fresh_run_required_before_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Fresh output"):
                runner.run(d, "not_read.log")


class JoinTests(unittest.TestCase):
    def test_current_only(self):
        row, old = fixture()
        original = deepcopy((row, old))
        qualified, evidence = runner.join_features(row, old)
        self.assertEqual(set(evidence), {(0, "bright:1")})
        self.assertEqual((row, old), original)
        row, old = fixture(1, measured=False)
        self.assertEqual(runner.join_features(row, old)[1], {})

    def test_wrong_identity_coordinate_time_or_status_fails(self):
        for field, value in (("track_id", "bright:9"), ("source_xy", [10., 11.]),
                             ("measurement_source_xy", [10., 11.]), ("measured", False),
                             ("measurement_frame", 3), ("accepted", 1)):
            row, old = fixture()
            old["tracks"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): runner.join_features(row, old)
        row, old = fixture(); old["timestamp_ns"] = 1
        with self.assertRaises(ValueError): runner.join_features(row, old)

    def test_prediction_current_pixels_or_future_history_refused(self):
        row, old = fixture(1, measured=False)
        old["tracks"][0]["features"] = {}
        with self.assertRaises(ValueError): runner.join_features(row, old)
        old["tracks"][0]["features"] = None
        old["tracks"][0]["measurement_frame"] = 1
        with self.assertRaises(ValueError): runner.join_features(row, old)

    def test_missing_qualified_feature_fails(self):
        row, old = fixture(); old["tracks"].clear()
        with self.assertRaises(ValueError): runner.join_features(row, old)


class ReferenceTests(unittest.TestCase):
    def test_assignment_loss_not_hidden_by_alternative(self):
        row, _ = fixture(1)
        other = deepcopy(row["tracks"][0]); other["track_id"] = "bright:2"
        other["measurement_source_xy"] = [11., 10.]; row["tracks"].append(other)
        ref = sample()
        for name in ("actual", "baseline"):
            ref[name]["all_gated_ids"].append("0/bright:2")
        result = runner.score_reference(ref, row, dict(baseline={}, v57={
            "0/bright:1": {"accepted": False}, "0/bright:2": {"accepted": True}}))
        self.assertTrue(result["arms"]["v57"]["lost_original_assignment"])
        self.assertTrue(result["arms"]["v57"]["any_qualified_alternative_retained"])
        self.assertFalse(result["arms"]["v57"]["original_assignment_retained"])

    def test_prediction_is_not_reference_measurement(self):
        row, _ = fixture(1, measured=False)
        with self.assertRaisesRegex(ValueError, "inventory"):
            runner.score_reference(sample(), row, {"baseline": {}})

    def test_polarity_and_exact_gate(self):
        row, _ = fixture(1); row["tracks"][0]["measurement_source_xy"] = [12., 10.]
        result = runner.score_reference(sample(), row, {"baseline": {}})
        self.assertTrue(result["arms"]["baseline"]["original_assignment_retained"])
        for point in ([12.000001, 10.], [12., 12.]):
            row["tracks"][0]["measurement_source_xy"] = point
            with self.assertRaises(ValueError): runner.score_reference(sample(), row, {"baseline": {}})
        row["tracks"][0]["measurement_source_xy"] = [10., 10.]
        row["tracks"][0]["track_id"] = "dark:1"
        with self.assertRaises(ValueError): runner.score_reference(sample(), row, {"baseline": {}})

    def test_baseline_miss_remains_miss(self):
        row, _ = fixture(1, qualified=False)
        result = runner.score_reference(sample(None), row, dict(baseline={}, v57={}))
        self.assertFalse(result["arms"]["v57"]["any_qualified_alternative_retained"])
        self.assertFalse(result["arms"]["v57"]["lost_original_assignment"])

    def test_scope_half_open(self):
        self.assertTrue(runner.inside([10, 10], [10, 10, 2, 2]))
        self.assertFalse(runner.inside([12, 10], [10, 10, 2, 2]))
        self.assertFalse(runner.inside([10, 12], [10, 10, 2, 2]))


class ReplayTests(unittest.TestCase):
    def write_fixture(self, directory, pairs):
        a, b = Path(directory) / "journal.jsonl", Path(directory) / "features.jsonl"
        a.write_text("".join(json.dumps(p[0]) + "\n" for p in pairs))
        b.write_text("".join(json.dumps(p[1]) + "\n" for p in pairs))
        return a, b, Path(directory) / "decisions.jsonl"

    def test_complete_measured_and_prediction_workload(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_fixture(d, [fixture(0), fixture(1), fixture(2, measured=False)])
            control = dict(frames_inclusive=[0, 2], crop_xywh=[9, 9, 2, 2],
                           baseline_measured=2, baseline_predicted=1, v36_measured=0, v36_predicted=0)
            result = runner.replay_clip(*paths, 3, PersistenceConfig(), [sample()], [control])
            self.assertEqual(result["workload"]["v57"], dict(measured=1, predicted=0, distinct_identities=1))
            self.assertEqual(result["controls"][0]["workload"], result["workload"])
            self.assertEqual(result["references"][0]["arms"]["v57"]["lost_original_assignment"], True)
            self.assertEqual(len(list(runner.lines(paths[2]))), 3)

    def test_incomplete_and_unequal_streams_fail(self):
        for unequal in (False, True):
            with self.subTest(unequal=unequal), tempfile.TemporaryDirectory() as d:
                paths = self.write_fixture(d, [fixture(0)])
                if unequal: paths[1].write_text("")
                with self.assertRaises(ValueError):
                    runner.replay_clip(*paths, 2, PersistenceConfig(), [], [])

    def test_changed_control_counts_fail(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_fixture(d, [fixture(0)])
            control = dict(frames_inclusive=[0, 0], crop_xywh=[9, 9, 2, 2],
                           baseline_measured=2, baseline_predicted=0, v36_measured=0, v36_predicted=0)
            with self.assertRaisesRegex(ValueError, "control counts"):
                runner.replay_clip(*paths, 1, PersistenceConfig(), [], [control])

    def test_missing_reference_frame_fails(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_fixture(d, [fixture(0)])
            with self.assertRaisesRegex(ValueError, "Missing reference"):
                runner.replay_clip(*paths, 1, PersistenceConfig(), [sample()], [])


if __name__ == "__main__":
    unittest.main()
