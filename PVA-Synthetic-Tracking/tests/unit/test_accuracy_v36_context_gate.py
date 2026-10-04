"""Causal source-context gate contracts using only synthetic pixel arrays."""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/accuracy_v36_context_gate.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v36_context_gate_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
with mock.patch.object(sys, "path", [str(MODULE_PATH.parent), *sys.path]):
    SPEC.loader.exec_module(module)


def features(margin=0.4, informative=True):
    return {"informative": informative, "point_minus_edge_fraction": margin,
            "point_gain_fraction": 0.5 + margin / 2,
            "edge_gain_fraction": 0.5 - margin / 2}


class FakeDiagnostic:
    def __init__(self, responses=(), default=None):
        self.responses = list(responses)
        self.default = features() if default is None else default
        self.calls = []

    def measure(self, patch, polarity):
        self.calls.append((patch.copy(), polarity))
        result = self.responses.pop(0) if self.responses else self.default
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


def track(*, tid="bright:1", segment=0, measured=True, qualified=True, xy=(30.0, 30.0)):
    return {"track_id": tid, "segment": segment, "measured": measured,
            "qualified_moving": qualified,
            "measurement_source_xy": list(xy) if measured else None,
            "source_xy": [1000.0, 1000.0], "reference_xy": [-1000.0, -1000.0]}


def row(frame, tracks=(), *, segment=0, timestamp=None):
    return {"frame_index": frame, "timestamp_ns": frame * 100_000_000 if timestamp is None else timestamp,
            "segment": segment, "tracks": copy.deepcopy(list(tracks))}


def gray():
    y, x = np.indices((64, 80))
    return ((x * 3 + y * 7) % 251).astype(np.uint8)


class AccuracyV36ContextGateTests(unittest.TestCase):
    def make_gate(self, *responses):
        diagnostic = FakeDiagnostic(responses)
        return module.CausalPointContextGate(diagnostic=diagnostic), diagnostic

    def check_record(self, record, *, accepted, unknown=False, frame=None):
        self.assertIs(record["accepted"], accepted)
        self.assertIsInstance(record["reason"], str)
        self.assertTrue(record["reason"])
        if unknown:
            self.assertIn("unknown", record["reason"].lower())
        self.assertEqual(record["measurement_frame"], frame)
        self.assertIn("features", record)

    def test_qualified_measurement_pass_and_fail_with_strict_zero_boundary(self):
        for margin, expected in ((0.4, True), (0.0, False), (-0.4, False)):
            with self.subTest(margin=margin):
                gate, diagnostic = self.make_gate(features(margin))
                output = gate.update(row(0, [track()]), gray())
                self.check_record(output[(0, "bright:1")], accepted=expected, frame=0)
                self.assertEqual(output[(0, "bright:1")]["features"]["point_minus_edge_fraction"], margin)
                self.assertEqual(len(diagnostic.calls), 1)

    def test_unqualified_measurements_never_promoted_or_scored(self):
        gate, diagnostic = self.make_gate()
        output = gate.update(row(0, [track(qualified=False)]), gray())
        self.assertIs(output[(0, "bright:1")]["accepted"], False)
        self.assertEqual(diagnostic.calls, [])

    def test_prediction_inherits_latest_positive_eligibility_without_rescoring(self):
        gate, diagnostic = self.make_gate(features(0.4))
        initial = gate.update(row(0, [track()]), gray())
        for frame in range(1, 8):
            output = gate.update(row(frame, [track(measured=False)]), np.zeros_like(gray()))
            self.check_record(output[(0, "bright:1")], accepted=True, frame=0)
            # A coast may omit the stored feature details; it cannot introduce
            # a freshly evaluated feature or alter the originating decision.
            if output[(0, "bright:1")]["features"] is not None:
                self.assertEqual(output[(0, "bright:1")]["features"], initial[(0, "bright:1")]["features"])
        self.assertEqual(len(diagnostic.calls), 1)

    def test_prediction_inherits_latest_negative_eligibility_without_rescoring(self):
        gate, diagnostic = self.make_gate(features(-0.4))
        gate.update(row(0, [track()]), gray())
        for frame in range(1, 5):
            output = gate.update(row(frame, [track(measured=False)]), np.full_like(gray(), 255))
            self.check_record(output[(0, "bright:1")], accepted=False, frame=0)
        self.assertEqual(len(diagnostic.calls), 1)

    def test_new_measurement_overrides_cached_positive_and_negative_decisions(self):
        gate, diagnostic = self.make_gate(features(-0.2), features(0.2), features(-0.3))
        for frame, expected in enumerate((False, True, False)):
            output = gate.update(row(frame, [track(xy=(30 + frame, 30))]), gray())
            self.check_record(output[(0, "bright:1")], accepted=expected, frame=frame)
        self.assertEqual(len(diagnostic.calls), 3)

    def test_unqualified_measurement_clears_old_eligibility_cache(self):
        gate, diagnostic = self.make_gate(features(-0.5))
        gate.update(row(0, [track()]), gray())
        unqualified = gate.update(row(1, [track(qualified=False)]), gray())
        self.assertFalse(unqualified[(0, "bright:1")]["accepted"])
        output = gate.update(row(2, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=True, unknown=True)
        self.assertIsNone(output[(0, "bright:1")]["features"])
        self.assertEqual(len(diagnostic.calls), 1)

    def test_unqualified_prediction_remains_rejected_even_after_positive_measurement(self):
        gate, diagnostic = self.make_gate(features(0.5))
        gate.update(row(0, [track()]), gray())
        output = gate.update(row(1, [track(measured=False, qualified=False)]), gray())
        self.assertFalse(output[(0, "bright:1")]["accepted"])
        self.assertEqual(len(diagnostic.calls), 1)

    def test_missing_history_coast_preserves_baseline_and_flags_unknown(self):
        gate, diagnostic = self.make_gate()
        output = gate.update(row(0, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=True, unknown=True)
        self.assertIsNone(output[(0, "bright:1")]["features"])
        self.assertEqual(diagnostic.calls, [])

    def test_truncated_native_patch_is_unknown_not_target_absence(self):
        for xy in ((2, 30), (78, 30), (30, 2), (30, 62)):
            with self.subTest(xy=xy):
                gate, diagnostic = self.make_gate()
                output = gate.update(row(0, [track(xy=xy)]), gray())
                self.check_record(output[(0, "bright:1")], accepted=True, unknown=True, frame=0)
                self.assertIsNone(output[(0, "bright:1")]["features"])
                self.assertEqual(diagnostic.calls, [])

    def test_uninformative_features_preserve_baseline_and_flag_unknown(self):
        gate, diagnostic = self.make_gate(features(0.0, informative=False))
        output = gate.update(row(0, [track()]), gray())
        self.check_record(output[(0, "bright:1")], accepted=True, unknown=True, frame=0)
        self.assertFalse(output[(0, "bright:1")]["features"]["informative"])
        prediction = gate.update(row(1, [track(measured=False)]), gray())
        self.assertTrue(prediction[(0, "bright:1")]["accepted"])
        self.assertEqual(prediction[(0, "bright:1")]["measurement_frame"], 0)
        self.assertEqual(len(diagnostic.calls), 1)

    def test_disappearance_then_rebirth_never_reuses_rejected_cache(self):
        gate, diagnostic = self.make_gate(features(-0.2))
        gate.update(row(0, [track()]), gray())
        self.assertEqual(gate.update(row(1), gray()), {})
        output = gate.update(row(2, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=True, unknown=True)
        self.assertEqual(len(diagnostic.calls), 1)

    def test_segment_reset_never_reuses_old_id_history(self):
        gate, diagnostic = self.make_gate(features(-0.2))
        gate.update(row(0, [track()]), gray())
        output = gate.update(row(1, [track(segment=1, measured=False)], segment=1), gray())
        self.assertNotIn((0, "bright:1"), output)
        self.check_record(output[(1, "bright:1")], accepted=True, unknown=True)
        self.assertEqual(len(diagnostic.calls), 1)

    def test_source_fractional_rounding_not_filtered_coordinates_selects_patch(self):
        gate, diagnostic = self.make_gate()
        image = gray()
        gate.update(row(0, [track(xy=(21.49, 31.5))]), image)
        patch, polarity = diagnostic.calls[0]
        np.testing.assert_array_equal(patch, image[20:45, 9:34])
        self.assertEqual(patch.shape, (25, 25))
        self.assertEqual(patch.dtype, np.uint8)
        self.assertEqual(polarity, "bright")

    def test_dark_track_passes_actual_polarity_to_diagnostic(self):
        gate, diagnostic = self.make_gate()
        output = gate.update(row(0, [track(tid="dark:7")]), gray())
        self.assertTrue(output[(0, "dark:7")]["accepted"])
        self.assertEqual(diagnostic.calls[0][1], "dark")

    def test_nearby_track_decisions_do_not_share_or_suppress_identity(self):
        gate, diagnostic = self.make_gate(features(0.5), features(-0.5))
        output = gate.update(row(0, [track(), track(tid="bright:2", xy=(31, 30))]), gray())
        self.assertTrue(output[(0, "bright:1")]["accepted"])
        self.assertFalse(output[(0, "bright:2")]["accepted"])
        output = gate.update(row(1, [track(measured=False), track(tid="bright:2", measured=False)]), gray())
        self.assertTrue(output[(0, "bright:1")]["accepted"])
        self.assertFalse(output[(0, "bright:2")]["accepted"])
        self.assertEqual(len(diagnostic.calls), 2)

    def test_image_requires_uint8_two_dimensions(self):
        images = (np.zeros((64, 80), dtype=np.float32), np.zeros((64, 80), dtype=np.uint16),
                  np.zeros((64, 80, 1), dtype=np.uint8), np.zeros((64, 80, 3), dtype=np.uint8),
                  np.zeros((0, 80), dtype=np.uint8))
        for image in images:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                gate, diagnostic = self.make_gate()
                with self.assertRaises(ValueError):
                    gate.update(row(0, [track()]), image)
                self.assertEqual(diagnostic.calls, [])

    def test_frame_geometry_cannot_change_and_failure_does_not_advance_state(self):
        gate, diagnostic = self.make_gate(features(-0.5))
        gate.update(row(0, [track()]), gray())
        with self.assertRaises(ValueError):
            gate.update(row(1, [track()]), np.zeros((65, 80), dtype=np.uint8))
        output = gate.update(row(1, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=False, frame=0)
        self.assertEqual(len(diagnostic.calls), 1)

    def test_frame_indices_start_at_zero_and_remain_contiguous(self):
        gate, _ = self.make_gate()
        with self.assertRaises(ValueError):
            gate.update(row(1), gray())
        for bad_frame in (0, 2, -1):
            with self.subTest(bad_frame=bad_frame):
                gate, _ = self.make_gate()
                gate.update(row(0), gray())
                with self.assertRaises(ValueError):
                    gate.update(row(bad_frame, timestamp=100_000_000), gray())

    def test_timestamp_must_be_nonnegative_strictly_increasing(self):
        gate, _ = self.make_gate()
        with self.assertRaises(ValueError):
            gate.update(row(0, timestamp=-1), gray())
        for timestamp in (100, 99):
            with self.subTest(timestamp=timestamp):
                gate, _ = self.make_gate()
                gate.update(row(0, timestamp=100), gray())
                with self.assertRaises(ValueError):
                    gate.update(row(1, timestamp=timestamp), gray())

    def test_measured_coordinates_validated_even_when_unqualified(self):
        for xy in (None, [1], [1, 2, 3], [float("nan"), 2], [1, float("inf")]):
            with self.subTest(xy=xy):
                gate, diagnostic = self.make_gate()
                malformed = track(qualified=False)
                malformed["measurement_source_xy"] = xy
                with self.assertRaises(ValueError):
                    gate.update(row(0, [malformed]), gray())
                self.assertEqual(diagnostic.calls, [])

    def test_prediction_with_measurement_coordinate_is_invalid(self):
        gate, diagnostic = self.make_gate()
        malformed = track(measured=False)
        malformed["measurement_source_xy"] = [20.0, 30.0]
        with self.assertRaises(ValueError):
            gate.update(row(0, [malformed]), gray())
        self.assertEqual(diagnostic.calls, [])

    def test_duplicate_ids_and_wrong_segments_invalid_before_scoring(self):
        cases = ([track(), track()], [track(), track(tid="bright:2", segment=1)])
        for tracks in cases:
            with self.subTest(tracks=tracks):
                gate, diagnostic = self.make_gate()
                with self.assertRaises(ValueError):
                    gate.update(row(0, tracks), gray())
                self.assertEqual(diagnostic.calls, [])

    def test_non_boolean_flags_and_invalid_segment_are_rejected(self):
        for field in ("measured", "qualified_moving"):
            gate, _ = self.make_gate()
            malformed = track()
            malformed[field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                gate.update(row(0, [malformed]), gray())
        gate, _ = self.make_gate()
        with self.assertRaises(ValueError):
            gate.update(row(0, segment=-1), gray())

    def test_invalid_second_track_cannot_partially_replace_cached_decision(self):
        gate, diagnostic = self.make_gate(features(-0.2))
        gate.update(row(0, [track()]), gray())
        malformed = track(tid="bright:2")
        malformed["measurement_source_xy"] = [float("nan"), 30]
        with self.assertRaises(ValueError):
            gate.update(row(1, [track(), malformed]), gray())
        self.assertEqual(len(diagnostic.calls), 1)
        output = gate.update(row(1, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=False, frame=0)

    def test_diagnostic_failure_does_not_commit_part_of_frame(self):
        gate, diagnostic = self.make_gate(features(-0.2), features(0.3), ValueError("synthetic fit failure"))
        gate.update(row(0, [track()]), gray())
        with self.assertRaises(ValueError):
            gate.update(row(1, [track(), track(tid="bright:2")]), gray())
        output = gate.update(row(1, [track(measured=False)]), gray())
        self.check_record(output[(0, "bright:1")], accepted=False, frame=0)
        self.assertEqual(len(diagnostic.calls), 3)

    def test_source_inputs_and_past_outputs_are_not_mutated(self):
        gate, _ = self.make_gate()
        image = gray()
        image_before = image.copy()
        supplied = row(0, [track()])
        supplied_before = copy.deepcopy(supplied)
        output = gate.update(supplied, image)
        output_before = copy.deepcopy(output)
        gate.update(row(1, [track()]), np.zeros_like(image))
        np.testing.assert_array_equal(image, image_before)
        self.assertEqual(supplied, supplied_before)
        self.assertEqual(output, output_before)

    def test_diagnostic_receives_owned_patch_not_source_frame_view(self):
        class MutatingDiagnostic:
            def measure(self, patch, polarity):
                patch[:] = 255
                return features()

        gate = module.CausalPointContextGate(diagnostic=MutatingDiagnostic())
        image = gray()
        original = image.copy()
        gate.update(row(0, [track()]), image)
        np.testing.assert_array_equal(image, original)

    def test_malformed_diagnostic_result_fails_without_committing_frame(self):
        invalid = (None, {}, {"informative": 1, "point_minus_edge_fraction": 0.1},
                   {"informative": True, "point_minus_edge_fraction": float("nan")},
                   {"informative": True, "point_minus_edge_fraction": float("inf")},
                   {"informative": True, "point_minus_edge_fraction": True})
        for result in invalid:
            with self.subTest(result=result):
                gate, diagnostic = self.make_gate(features(-0.2), result)
                gate.update(row(0, [track()]), gray())
                with self.assertRaises(ValueError):
                    gate.update(row(1, [track()]), gray())
                output = gate.update(row(1, [track(measured=False)]), gray())
                self.check_record(output[(0, "bright:1")], accepted=False, frame=0)

    def test_default_diagnostic_integration_on_synthetic_point_and_edge(self):
        y, x = np.indices((64, 80), dtype=np.float64)
        point_image = np.rint(100 + 30 * np.exp(-((x - 30) ** 2 + (y - 30) ** 2) / 8)).astype(np.uint8)
        edge_image = np.rint(100 + 30 * np.tanh((x - 30) / 2)).astype(np.uint8)
        gate = module.CausalPointContextGate()
        point_result = gate.update(row(0, [track()]), point_image)
        edge_result = gate.update(row(1, [track()]), edge_image)
        self.assertTrue(point_result[(0, "bright:1")]["accepted"])
        self.assertFalse(edge_result[(0, "bright:1")]["accepted"])

    def test_causal_prefix_outputs_do_not_depend_on_future_frames(self):
        rows = [row(frame, [track(measured=frame % 3 == 0)]) for frame in range(12)]
        short, _ = self.make_gate(features(-0.2), features(0.2))
        full, _ = self.make_gate(features(-0.2), features(0.2), features(-0.2), features(0.2))
        prefix = [short.update(item, gray()) for item in copy.deepcopy(rows[:6])]
        sequence = [full.update(item, gray()) for item in copy.deepcopy(rows)]
        self.assertEqual(prefix, sequence[:6])

    def test_retained_state_is_bounded_by_current_track_population(self):
        gate, _ = self.make_gate()
        for frame in range(300):
            gate.update(row(frame, [track(), track(tid=f"dark:{frame}")]), gray())
            self.assertLessEqual(gate.retained_state_counts()["tracks"], 2)
        gate.update(row(300), gray())
        self.assertEqual(gate.retained_state_counts()["tracks"], 0)


if __name__ == "__main__":
    unittest.main()
