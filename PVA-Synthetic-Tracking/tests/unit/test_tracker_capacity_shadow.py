import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_tracker_capacity_shadow.py"
spec = importlib.util.spec_from_file_location("capacity_shadow", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class FakeManager:
    def update(self):
        pass


class FakeTracker:
    def __init__(self, *, output_delta=0, state_delta=0):
        self.output_delta, self.state_delta = output_delta, state_delta
        self.managers = {}
        self.extents, self.qualified, self.summary = {}, set(), {}
        self.ever_qualified, self.previous_records = set(), []
        self.previous_timestamp_ns, self.quality = None, {}

    def learning_centers(self, timestamp, segment):
        return []

    def update(self, candidates, index, timestamp, segment, matrix, shape):
        self.extents["value"] = index+self.state_delta
        records = [{"value": index+self.output_delta}]
        self.previous_records, self.previous_timestamp_ns = records, timestamp
        return records, {"n": len(candidates)}


def row(index):
    return dict(frame_index=index, timestamp_ns=index*100_000_000, segment=0,
                source_to_reference=np.eye(3).tolist(), candidates=[],
                coverage=dict(full_shape_hw=[3190, 4784], detection_ready=False, warmup=True,
                              learning_protection=dict(causal_measured_track_centers=0)),
                tracks=[{"value": index}], tracking_metrics={"n": 0})


class CapacityShadowTests(unittest.TestCase):
    def replay(self, rows, trackers=None, count=3):
        stream = io.StringIO()
        method = FakeManager.update
        result = check.replay_rows(rows, trackers or [FakeTracker(), FakeTracker()], [method, method],
                                   FakeManager, FakeTracker(), stream, expected_frames=count)
        self.assertIs(FakeManager.update, method)
        return result, [json.loads(line) for line in stream.getvalue().splitlines()]

    def test_strict_diff_has_no_numeric_tolerance_or_signed_zero_exception(self):
        for a, b in ((1.0, np.nextafter(1.0, 2.0).item()), (0.0, -0.0), (1, 1.0), (True, 1)):
            self.assertIsNotNone(check.first_difference({"x": a}, {"x": b}))
        self.assertIsNone(check.first_difference({"a": [1.0]}, {"a": [1.0]}))

    def test_complete_baseline_and_dual_state_hashes(self):
        result, traces = self.replay([row(i) for i in range(3)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["exact_archive_frames"], 3)
        self.assertEqual(result["dual_state_exact_frames"], 3)
        self.assertEqual(result["derived_learning_exact_frames"], 3)
        self.assertEqual(len(traces), 3)
        self.assertEqual(traces[0]["internal_state_sha256"][0], traces[0]["internal_state_sha256"][1])

    def test_first_cold_frame_difference_stops_input_consumption(self):
        def stream():
            yield row(0)
            raise AssertionError("must not consume later frame after difference")
        result, traces = self.replay(stream(), [FakeTracker(output_delta=1), FakeTracker(output_delta=1)])
        self.assertFalse(result["passed"])
        self.assertEqual(result["attempted_frames"], 1)
        self.assertEqual(result["exact_archive_frames"], 0)
        self.assertEqual(result["first_difference"]["category"], "original_archive_observables")
        self.assertTrue(result["first_difference"]["compared_during_cold_start_or_warmup"])
        self.assertEqual(result["first_difference"]["detail"]["path"], ["tracks", 0, "value"])
        self.assertEqual(len(traces), 1)

    def test_internal_state_difference_blocks_even_when_outputs_match(self):
        result, traces = self.replay([row(0)], [FakeTracker(), FakeTracker(state_delta=1)], count=1)
        self.assertEqual(result["first_difference"]["category"], "dual_replay_internal_state")
        self.assertEqual(traces[0]["output_sha256"][0], traces[0]["output_sha256"][1])

    def test_missing_frame_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, "incomplete journal"):
            self.replay([row(0)])
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            self.replay([row(1)])

    def test_missing_archive_learning_count_is_not_fabricated(self):
        rows = [row(0)]
        rows[0]["coverage"].pop("learning_protection")
        result, traces = self.replay(rows, count=1)
        self.assertTrue(result["passed"])
        self.assertEqual(traces[0]["differences"], {})

    def test_archive_learning_count_difference_blocks(self):
        rows = [row(0)]
        rows[0]["coverage"]["learning_protection"]["causal_measured_track_centers"] = 1
        result, _ = self.replay(rows, count=1)
        self.assertEqual(result["first_difference"]["category"], "archive_learning_center_count")

    def test_real_visible_track_constructors_and_cold_empty_frames(self):
        from tiny_target.visible_baseline import VisibleConfig, VisibleTracks
        from tiny_target.tracking.kalman import KalmanTrackManager
        cfg = VisibleConfig(**check.read(check.CONFIG))
        reference = VisibleTracks(cfg, 10)
        rows = []
        for index in range(8):
            current = row(index)
            records, metrics = reference.update([], index, index*100_000_000, 0, np.eye(3), (3190, 4784))
            current["tracks"], current["tracking_metrics"] = check.json_value([records, metrics])
            rows.append(current)
        method = KalmanTrackManager.update
        result = check.replay_rows(rows, [VisibleTracks(cfg, 10), VisibleTracks(cfg, 10)], [method, method],
                                   KalmanTrackManager, VisibleTracks(cfg, 10), io.StringIO(), expected_frames=8)
        self.assertTrue(result["passed"])
        self.assertEqual(result["exact_archive_frames"], 8)

    def test_json_and_raw_array_state_integrity(self):
        for text in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                check.decode(text)
        self.assertNotEqual(check.value_sha(np.array([1], dtype="float32")),
                            check.value_sha(np.array([1], dtype="float64")))
        self.assertNotEqual(check.value_sha([1.0]), check.value_sha((1.0,)))

    def test_error_receipt_serializes_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/"out"
            with patch.object(check, "OUTPUT_ROOT", root), patch.object(check, "validate_inputs", side_effect=ValueError("test input")):
                result = check.run(root/"baseline_01")
                path = root/"baseline_01/baseline_result.json"
                self.assertEqual(json.loads(path.read_text()), result)
                self.assertFalse(result["passed"])
                self.assertFalse(result["candidate_implemented"])
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "existing output"):
                    check.run(root/"baseline_01")
                self.assertEqual(path.read_bytes(), original)
                with self.assertRaisesRegex(ValueError, "bounded output"):
                    check.output_guard(root/"wrong")

    def test_postcheck_changed_input_marks_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()/"out"
            with patch.object(check, "OUTPUT_ROOT", root), \
                 patch.object(check, "validate_inputs", return_value=({"file": "digest"}, {}, {})), \
                 patch.object(check.importlib, "import_module", side_effect=ValueError("test before run")), \
                 patch.object(check, "verify_unchanged", side_effect=ValueError("input changed")):
                result = check.run(root/"baseline_01")
            self.assertFalse(result["passed"])
            self.assertFalse(result["inputs_unchanged_after_check"])
            self.assertIn("input changed", result["postcheck_error"])


if __name__ == "__main__":
    unittest.main()
