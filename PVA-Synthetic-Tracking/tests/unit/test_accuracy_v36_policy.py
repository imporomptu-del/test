"""Synthetic, causal safety tests for v36 qualification-only ablations.

No source media, clip identifiers, target coordinates, or experimental outputs
are used here. These checks do not establish airborne classification accuracy.
"""

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/accuracy_v36_policy.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v36_policy_under_test", MODULE_PATH)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
DECISIONS = ("baseline", "recent_support", "recent_excursion", "bounded_shape", "combined")
SHAPE = [[20.0, 30.0], [21.0, 30.0], [20.0, 31.0]]


def track(x=20.0, y=30.0, *, tid="bright:1", segment=0, measured=True,
          qualified=True, shape=SHAPE):
    return {
        "track_id": tid,
        "segment": segment,
        "measured": measured,
        "qualified_moving": qualified,
        "measurement_source_xy": [x, y] if measured else None,
        "learning_shape_reference_xy": copy.deepcopy(shape),
        # Intentionally misleading filtered states must never create evidence.
        "reference_xy": [99999.0, -99999.0],
        "source_xy": [-99999.0, 99999.0],
    }


def row(frame, tracks=(), *, segment=0, transform=IDENTITY, timestamp=None):
    return {
        "frame_index": frame,
        "timestamp_ns": frame * 100_000_000 if timestamp is None else timestamp,
        "segment": segment,
        "source_to_reference": copy.deepcopy(transform),
        "tracks": copy.deepcopy(list(tracks)),
    }


class AccuracyV36PolicyTests(unittest.TestCase):
    def decision(self, result, name, key=(0, "bright:1")):
        return result[key]["decisions"][name]

    def test_defaults_are_explicit_and_configuration_validated(self):
        config = policy.PolicyConfig()
        self.assertEqual(config.window_hits, 8)
        self.assertEqual(config.recent_frame_window, 8)
        self.assertEqual(config.minimum_recent_hits, 5)
        self.assertEqual(config.minimum_excursion_px, 12.0)
        invalid = (
            {"window_hits": 0}, {"window_hits": 2.5}, {"window_hits": True},
            {"recent_frame_window": -1}, {"recent_frame_window": 3.5},
            {"minimum_recent_hits": 1}, {"minimum_recent_hits": 9},
            {"window_hits": 4}, {"recent_frame_window": 4},
            {"minimum_excursion_px": 0}, {"minimum_excursion_px": -1},
            {"minimum_excursion_px": math.nan}, {"minimum_excursion_px": math.inf},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                policy.CausalQualification(policy.PolicyConfig(**kwargs))

    def test_sparse_old_hits_cannot_satisfy_recent_support(self):
        evaluator = policy.CausalQualification()
        for frame in range(29):
            measured = frame in {0, 7, 14, 21, 28}
            output = evaluator.update(row(frame, [track(20 + frame * 3, measured=measured)]))
            self.assertFalse(self.decision(output, "recent_support"))
            self.assertFalse(self.decision(output, "combined"))
            self.assertLessEqual(output[(0, "bright:1")]["recent_hits"], 2)

    def test_recent_frame_window_is_inclusive_of_current_frame(self):
        evaluator = policy.CausalQualification()
        for frame in range(9):
            output = evaluator.update(row(frame, [track(20 + frame * 4, measured=frame < 5)]))
            if frame == 7:
                self.assertEqual(output[(0, "bright:1")]["recent_hits"], 5)
                self.assertTrue(self.decision(output, "recent_support"))
            if frame == 8:
                self.assertEqual(output[(0, "bright:1")]["recent_hits"], 4)
                self.assertFalse(self.decision(output, "recent_support"))

    def test_former_motion_then_stopped_loses_excursion(self):
        evaluator = policy.CausalQualification()
        saw_combined = False
        for frame in range(24):
            output = evaluator.update(row(frame, [track(20 + min(frame, 7) * 4)]))
            saw_combined |= self.decision(output, "combined")
        self.assertTrue(saw_combined)
        self.assertTrue(self.decision(output, "recent_support"))
        self.assertFalse(self.decision(output, "recent_excursion"))
        self.assertFalse(self.decision(output, "combined"))
        self.assertEqual(output[(0, "bright:1")]["recent_excursion_px"], 0.0)

    def test_accelerating_turn_retains_qualified_measured_track(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            output = evaluator.update(row(frame, [track(20 + 4 * frame, 30 + (frame - 4) ** 2)]))
        for name in DECISIONS:
            self.assertTrue(self.decision(output, name), name)
        self.assertEqual(output[(0, "bright:1")]["recent_hits"], 8)
        self.assertEqual(output[(0, "bright:1")]["last_measurement_age_frames"], 0)

    def test_excursion_threshold_is_inclusive(self):
        evaluator = policy.CausalQualification()
        for frame in range(5):
            output = evaluator.update(row(frame, [track(20 + 3 * frame)]))
        self.assertEqual(output[(0, "bright:1")]["recent_excursion_px"], 12.0)
        self.assertTrue(self.decision(output, "recent_excursion"))
        self.assertTrue(self.decision(output, "combined"))

    def test_prediction_cannot_add_evidence_or_movement(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            output = evaluator.update(row(frame, [track(measured=frame == 0)]))
        diagnostic = output[(0, "bright:1")]
        self.assertEqual(diagnostic["recent_hits"], 1)
        self.assertEqual(diagnostic["recent_excursion_px"], 0.0)
        self.assertEqual(diagnostic["last_measurement_age_frames"], 7)
        self.assertFalse(self.decision(output, "recent_support"))
        self.assertFalse(self.decision(output, "recent_excursion"))

    def test_prediction_before_any_measurement_has_no_cached_shape(self):
        evaluator = policy.CausalQualification()
        output = evaluator.update(row(0, [track(measured=False, shape=SHAPE)]))
        diagnostic = output[(0, "bright:1")]
        self.assertEqual(diagnostic["recent_hits"], 0)
        self.assertIsNone(diagnostic["last_measurement_age_frames"])
        self.assertFalse(diagnostic["bounded_shape"])
        self.assertFalse(self.decision(output, "combined"))

    def test_prediction_caches_last_measured_shape_not_prediction_shape(self):
        evaluator = policy.CausalQualification()
        evaluator.update(row(0, [track(shape=SHAPE)]))
        output = evaluator.update(row(1, [track(measured=False, shape=None)]))
        self.assertTrue(output[(0, "bright:1")]["bounded_shape"])
        output = evaluator.update(row(2, [track(shape=None)]))
        self.assertFalse(output[(0, "bright:1")]["bounded_shape"])
        output = evaluator.update(row(3, [track(measured=False, shape=SHAPE)]))
        self.assertFalse(output[(0, "bright:1")]["bounded_shape"])

    def test_segment_change_never_reuses_old_identity_evidence(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            output = evaluator.update(row(frame, [track(20 + 4 * frame)]))
        self.assertTrue(self.decision(output, "combined"))
        output = evaluator.update(row(8, [track(segment=1, measured=False)], segment=1))
        self.assertNotIn((0, "bright:1"), output)
        diagnostic = output[(1, "bright:1")]
        self.assertEqual(diagnostic["recent_hits"], 0)
        self.assertIsNone(diagnostic["last_measurement_age_frames"])
        self.assertFalse(diagnostic["decisions"]["combined"])
        for frame in range(9, 17):
            output = evaluator.update(row(frame, [track(segment=1)], segment=1))
        self.assertFalse(output[(1, "bright:1")]["decisions"]["recent_excursion"])

    def test_disappeared_identity_is_pruned_not_reused(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            evaluator.update(row(frame, [track(20 + frame * 4)]))
        self.assertEqual(evaluator.update(row(8)), {})
        self.assertEqual(evaluator.retained_state_counts(), {"tracks": 0, "measurements": 0})
        output = evaluator.update(row(9, [track(80)]))
        self.assertEqual(output[(0, "bright:1")]["recent_hits"], 1)
        self.assertFalse(self.decision(output, "combined"))

    def test_camera_translation_does_not_create_residual_target_motion(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            transform = [[1, 0, -4 * frame], [0, 1, 0], [0, 0, 1]]
            output = evaluator.update(row(frame, [track(20 + frame * 4)], transform=transform))
        self.assertEqual(output[(0, "bright:1")]["recent_excursion_px"], 0.0)
        self.assertFalse(self.decision(output, "recent_excursion"))
        self.assertFalse(self.decision(output, "combined"))

    def test_raw_measurement_uses_homogeneous_transform_not_filtered_position(self):
        evaluator = policy.CausalQualification()
        transform = [[2, 0, 10], [0, 2, 20], [0, 0, 2]]
        for frame in range(5):
            output = evaluator.update(row(frame, [track(20 + frame * 3)], transform=transform))
        self.assertEqual(output[(0, "bright:1")]["recent_excursion_px"], 12.0)
        self.assertTrue(self.decision(output, "combined"))

    def test_all_ablations_are_subsets_of_baseline_qualification(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            output = evaluator.update(row(frame, [track(20 + frame * 4, qualified=False)]))
        self.assertEqual(set(output[(0, "bright:1")]["decisions"]), set(DECISIONS))
        for name in DECISIONS:
            self.assertIs(self.decision(output, name), False)

    def test_nearby_objects_keep_independent_evidence_without_suppression(self):
        evaluator = policy.CausalQualification()
        for frame in range(8):
            output = evaluator.update(row(frame, [
                track(20 + frame * 4, tid="bright:1"),
                track(21 + frame * 4, tid="bright:2"),
                track(22, tid="dark:1"),
            ]))
        self.assertTrue(self.decision(output, "combined", (0, "bright:1")))
        self.assertTrue(self.decision(output, "combined", (0, "bright:2")))
        self.assertFalse(self.decision(output, "combined", (0, "dark:1")))
        self.assertEqual(set(output), {(0, "bright:1"), (0, "bright:2"), (0, "dark:1")})

    def test_combined_is_conjunction_of_the_three_independent_ablations(self):
        evaluator = policy.CausalQualification()
        for frame in range(16):
            output = evaluator.update(row(frame, [track(20 + frame * 4, shape=None if frame % 3 == 0 else SHAPE)]))
            decisions = output[(0, "bright:1")]["decisions"]
            self.assertEqual(decisions["combined"], all(decisions[k] for k in DECISIONS[1:4]))

    def test_missing_empty_shape_is_unknown_not_proven_compact(self):
        for shape in (None, []):
            with self.subTest(shape=shape):
                evaluator = policy.CausalQualification()
                output = evaluator.update(row(0, [track(shape=shape)]))
                self.assertFalse(output[(0, "bright:1")]["bounded_shape"])
                self.assertFalse(self.decision(output, "bounded_shape"))
        missing = track()
        del missing["learning_shape_reference_xy"]
        output = policy.CausalQualification().update(row(0, [missing]))
        self.assertFalse(output[(0, "bright:1")]["bounded_shape"])

    def test_malformed_shape_raises_even_if_baseline_is_unqualified(self):
        for shape in ([1, 2], [[1]], [[1, 2, 3]], [[1, math.nan]], [[math.inf, 2]], [[]], "bad"):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                policy.CausalQualification().update(row(0, [track(shape=shape, qualified=False)]))

    def test_inputs_and_previously_returned_diagnostics_are_not_mutated(self):
        evaluator = policy.CausalQualification()
        supplied = row(0, [track()])
        before = copy.deepcopy(supplied)
        first = evaluator.update(supplied)
        frozen_first = copy.deepcopy(first)
        for frame in range(1, 20):
            evaluator.update(row(frame, [track(20 + frame * 4)]))
        self.assertEqual(supplied, before)
        self.assertEqual(first, frozen_first)

    def test_causal_prefix_outputs_independent_of_future_rows(self):
        frames = [row(frame, [track(20 + frame * 4)]) for frame in range(16)]
        short = policy.CausalQualification()
        prefix = [short.update(item) for item in copy.deepcopy(frames[:8])]
        full = policy.CausalQualification()
        complete = [full.update(item) for item in copy.deepcopy(frames)]
        self.assertEqual(prefix, complete[:8])

    def test_track_and_measurement_memory_are_bounded(self):
        config = policy.PolicyConfig(window_hits=5, recent_frame_window=9, minimum_recent_hits=3)
        evaluator = policy.CausalQualification(config)
        for frame in range(500):
            current = [track(frame * 4, tid="bright:1"), track(frame * 5, tid=f"dark:{frame}")]
            evaluator.update(row(frame, current))
            state = evaluator.retained_state_counts()
            self.assertEqual(state["tracks"], 2)
            self.assertLessEqual(state["measurements"], 2 * max(config.window_hits, config.recent_frame_window))
        evaluator.update(row(500))
        self.assertEqual(evaluator.retained_state_counts(), {"tracks": 0, "measurements": 0})

    def test_first_frame_must_be_zero_and_subsequent_frames_contiguous(self):
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(1))
        for bad_frame in (0, 2, -1):
            with self.subTest(bad_frame=bad_frame):
                evaluator = policy.CausalQualification()
                evaluator.update(row(0))
                with self.assertRaises(ValueError):
                    evaluator.update(row(bad_frame, timestamp=100_000_000))

    def test_timestamps_must_be_nonnegative_and_strictly_increasing(self):
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(0, timestamp=-1))
        for timestamp in (100, 99):
            with self.subTest(timestamp=timestamp):
                evaluator = policy.CausalQualification()
                evaluator.update(row(0, timestamp=100))
                with self.assertRaises(ValueError):
                    evaluator.update(row(1, timestamp=timestamp))
        evaluator = policy.CausalQualification()
        evaluator.update(row(0, timestamp=0))
        evaluator.update(row(1, timestamp=17))
        evaluator.update(row(2, timestamp=999_999))

    def test_nonfinite_noninvertible_or_wrong_shaped_transform_rejected(self):
        for transform in (
            [[1, 0], [0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, math.nan]],
            [[1, 0, 0], [0, math.inf, 0], [0, 0, 1]],
        ):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                policy.CausalQualification().update(row(0, transform=transform))

    def test_zero_homogeneous_denominator_rejected(self):
        transform = [[1, 0, 0], [0, 1, 0], [1, 0, -20]]
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(0, [track(20)], transform=transform))

    def test_measured_coordinates_must_be_finite_two_vector(self):
        for xy in (None, [], [1], [1, 2, 3], [1, math.nan], [math.inf, 1], "bad"):
            malformed = track()
            malformed["measurement_source_xy"] = xy
            with self.subTest(xy=xy), self.assertRaises(ValueError):
                policy.CausalQualification().update(row(0, [malformed]))

    def test_duplicate_identity_and_segment_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(0, [track(), track()]))
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(0, [track(segment=1)]))

    def test_predicted_state_with_measurement_coordinate_is_rejected(self):
        malformed = track(measured=False)
        malformed["measurement_source_xy"] = [30.0, 40.0]
        with self.assertRaises(ValueError):
            policy.CausalQualification().update(row(0, [malformed]))

    def test_malformed_new_segment_does_not_erase_current_segment(self):
        evaluator = policy.CausalQualification()
        for frame in range(5):
            evaluator.update(row(frame, [track(20 + frame * 4)]))
        malformed = track(segment=1, shape=[[math.nan, 0.0]])
        with self.assertRaises(ValueError):
            evaluator.update(row(5, [malformed], segment=1))
        output = evaluator.update(row(5, [track(40)]))
        self.assertEqual(output[(0, "bright:1")]["recent_hits"], 6)
        self.assertTrue(self.decision(output, "combined"))

    def test_invalid_row_does_not_partially_advance_state(self):
        evaluator = policy.CausalQualification()
        evaluator.update(row(0, [track()]))
        before = evaluator.retained_state_counts()
        malformed = track(tid="dark:2")
        malformed["measurement_source_xy"] = [math.nan, 3]
        with self.assertRaises(ValueError):
            evaluator.update(row(1, [track(24), malformed]))
        self.assertEqual(evaluator.retained_state_counts(), before)
        output = evaluator.update(row(1, [track(24)]))
        self.assertEqual(output[(0, "bright:1")]["recent_hits"], 2)
        self.assertEqual(output[(0, "bright:1")]["recent_excursion_px"], 4.0)

    def test_missing_required_fields_and_non_boolean_evidence_rejected(self):
        for field in ("frame_index", "timestamp_ns", "segment", "source_to_reference", "tracks"):
            malformed = row(0, [track()])
            del malformed[field]
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                policy.CausalQualification().update(malformed)
        for field in ("measured", "qualified_moving"):
            malformed = track()
            malformed[field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                policy.CausalQualification().update(row(0, [malformed]))


if __name__ == "__main__":
    unittest.main()
