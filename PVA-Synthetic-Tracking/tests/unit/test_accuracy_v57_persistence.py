"""Generated-record contracts for the nonpromoted output-only heuristic."""
import copy
from dataclasses import asdict, FrozenInstanceError
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v57_persistence import CausalEdgePersistence, PersistenceConfig


def track(identity="bright:1", *, measured=True, qualified=True, segment=0, xy=(30., 40.)):
    return dict(track_id=identity, segment=segment, measured=measured,
                qualified_moving=qualified,
                measurement_source_xy=list(xy) if measured else None,
                source_xy=[999., 888.], reference_xy=[-999., -888.],
                arbitrary_baseline_payload={"unchanged": [1, 2, 3]})


def row(frame, tracks=None, *, timestamp=None, segment=0, reset=False):
    return dict(frame_index=frame,
                timestamp_ns=frame * 100_000_000 if timestamp is None else timestamp,
                segment=segment, motion=dict(reset=reset),
                tracks=[] if tracks is None else tracks)


def evidence(tracks, margin=-.3, *, informative=True, reason=None):
    return {(item["segment"], item["track_id"]): dict(
        features=dict(informative=informative, point_minus_edge_fraction=margin),
        reason=(reason if reason is not None else
                "point_preferred" if margin > 0 else "edge_preferred_or_tie"))
        for item in tracks if item["qualified_moving"] and item["measured"]}


class PersistenceTests(unittest.TestCase):
    def step(self, policy, frame, *, margin=-.3, informative=True,
             timestamp=None, measured=True, qualified=True, segment=0, reset=False,
             identity="bright:1", xy=(30., 40.)):
        tracks = [track(identity, measured=measured, qualified=qualified, segment=segment, xy=xy)]
        result = policy.update(row(frame, tracks, timestamp=timestamp, segment=segment, reset=reset),
                               evidence(tracks, margin, informative=informative))
        return result[0] if result else None

    def test_defaults_are_exact_and_immutable(self):
        config = PersistenceConfig()
        self.assertEqual(asdict(config), dict(required_consecutive_edges=2,
                         maximum_edge_gap_ns=200_000_000,
                         maximum_coast_frames=7, maximum_coast_ns=700_000_000))
        with self.assertRaises(FrozenInstanceError):
            config.maximum_coast_frames = 99

    def test_config_rejects_bad_types_zero_and_negative_values(self):
        for name in asdict(PersistenceConfig()):
            for value in (True, False, 0, -1, 1.5, "2", None, float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PersistenceConfig(**{name: value})
        with self.assertRaises(ValueError):
            PersistenceConfig(required_consecutive_edges=1)
        with self.assertRaises(ValueError):
            CausalEdgePersistence(config={})

    def test_first_edge_retained_second_and_later_edges_rejected(self):
        policy = CausalEdgePersistence()
        for frame in range(5):
            decision = self.step(policy, frame)
            self.assertEqual(decision["accepted"], frame == 0)
            self.assertEqual(decision["edge_streak_count"], frame + 1)
            self.assertEqual(decision["edge_streak_start_frame"], 0)
            self.assertEqual(decision["edge_streak_start_timestamp_ns"], 0)
            self.assertEqual(decision["measurement_frame"], frame)
            self.assertEqual(decision["measurement_timestamp_ns"], frame * 100_000_000)
            self.assertEqual(decision["evidence_age_frames"], 0)
            self.assertFalse(decision["inherited"])
        self.assertEqual(decision["reason"], "edge_consecutive_rejected")
        self.assertEqual(decision["tier"], "edge_suppressed_measured")

    def test_zero_is_edge_but_smallest_positive_float_is_point(self):
        policy = CausalEdgePersistence()
        self.assertTrue(self.step(policy, 0, margin=-0.)["accepted"])
        self.assertFalse(self.step(policy, 1, margin=0.)["accepted"])
        point = self.step(policy, 2, margin=5e-324)
        self.assertTrue(point["accepted"])
        self.assertEqual(point["tier"], "point_supported_measured")
        self.assertEqual(point["edge_streak_count"], 0)
        self.assertIsNone(point["edge_streak_start_frame"])

    def test_positive_measurement_immediately_reacquires_and_resets_run(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.assertFalse(self.step(policy, 1)["accepted"])
        self.assertTrue(self.step(policy, 2, margin=.2)["accepted"])
        self.assertTrue(self.step(policy, 3)["accepted"])
        self.assertFalse(self.step(policy, 4)["accepted"])

    def test_alternating_point_and_edge_never_accumulates_negatives(self):
        policy = CausalEdgePersistence()
        for frame in range(20):
            decision = self.step(policy, frame, margin=.4 if frame % 2 else -.4)
            self.assertTrue(decision["accepted"])
            self.assertLessEqual(decision["edge_streak_count"], 1)

    def test_edge_gap_equality_passes_and_one_nanosecond_over_resets(self):
        for gap, accepted in ((200_000_000, False), (200_000_001, True)):
            with self.subTest(gap=gap):
                policy = CausalEdgePersistence()
                self.step(policy, 0, timestamp=19)
                decision = self.step(policy, 1, timestamp=19 + gap)
                self.assertEqual(decision["accepted"], accepted)
                self.assertEqual(decision["edge_streak_start_frame"], 1 if accepted else 0)

    def test_unknown_informative_false_retains_and_clears_run(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.assertFalse(self.step(policy, 1)["accepted"])
        decision = self.step(policy, 2, margin=-.9, informative=False)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "unknown_uninformative_patch")
        self.assertEqual(decision["tier"], "unknown_measured")
        self.assertIs(decision["evidence_informative"], False)
        self.assertEqual(decision["edge_streak_count"], 0)
        self.assertTrue(self.step(policy, 3)["accepted"])

    def test_unknown_missing_patch_is_explicit_retained_measurement(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        observed = {(0, "bright:1"): dict(features=None, reason="unknown_truncated_patch")}
        decision = policy.update(row(2, [track()]), observed)[0]
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "unknown_truncated_patch")
        self.assertEqual(decision["measurement_frame"], 2)
        self.assertIsNone(decision["features"])
        self.assertIsNone(decision["evidence_informative"])
        self.assertTrue(self.step(policy, 3)["accepted"])

    def test_unknown_measurement_coast_retains_unknown_origin(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0, informative=False)
        decision = self.step(policy, 1, measured=False)
        self.assertTrue(decision["accepted"])
        self.assertTrue(decision["inherited"])
        self.assertEqual(decision["inherited_reason"], "unknown_uninformative_patch")
        self.assertEqual(decision["inherited_tier"], "unknown_measured")
        self.assertIsNone(decision["features"])
        self.assertIsNone(decision["evidence_informative"])

    def test_coasts_inherit_rejection_but_never_refresh_evidence(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        for frame in range(2, 9):
            decision = self.step(policy, frame, measured=False)
            self.assertFalse(decision["accepted"])
            self.assertTrue(decision["inherited"])
            self.assertEqual(decision["reason"], "coast_edge_consecutive_rejected")
            self.assertEqual(decision["measurement_frame"], 1)
            self.assertEqual(decision["measurement_timestamp_ns"], 100_000_000)
            self.assertEqual(decision["evidence_age_frames"], frame - 1)
            self.assertEqual(decision["edge_streak_count"], 0)
            self.assertIsNone(decision["features"])
            self.assertIsNone(decision["measurement_source_xy"])
        self.assertEqual(policy.retained_state_counts(), dict(tracks=1, edge_streaks=0))

    def test_expiry_is_unknown_not_support_and_erases_stale_cache(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        for frame in range(2, 9):
            self.step(policy, frame, measured=False)
        decision = self.step(policy, 9, measured=False)
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["inherited"])
        self.assertEqual(decision["reason"], "unknown_expired_history")
        self.assertEqual(decision["expired_measurement_frame"], 1)
        self.assertEqual(decision["expired_measurement_timestamp_ns"], 100_000_000)
        self.assertIsNone(decision["measurement_frame"])
        self.assertEqual(decision["evidence_age_frames"], 8)
        self.assertEqual(policy.retained_state_counts(), dict(tracks=0, edge_streaks=0))
        following = self.step(policy, 10, measured=False)
        self.assertEqual(following["reason"], "unknown_missing_history")
        self.assertIsNone(following["evidence_age_frames"])

    def test_coast_time_budget_is_independent_of_frame_budget(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0, margin=.5)
        self.assertTrue(self.step(policy, 1, measured=False,
                                  timestamp=700_000_000)["inherited"])
        decision = self.step(policy, 2, measured=False, timestamp=700_000_001)
        self.assertFalse(decision["inherited"])
        self.assertEqual(decision["reason"], "unknown_expired_history")

    def test_coast_frame_budget_is_independent_of_time_budget(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0, timestamp=10)
        for frame in range(1, 8):
            self.assertTrue(self.step(policy, frame, measured=False,
                                      timestamp=10 + frame)["inherited"])
        decision = self.step(policy, 8, measured=False, timestamp=18)
        self.assertFalse(decision["inherited"])
        self.assertEqual(decision["evidence_age_ns"], 8)

    def test_one_coast_breaks_edge_run_even_if_it_inherits_rejection(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        self.assertFalse(self.step(policy, 2, measured=False)["accepted"])
        decision = self.step(policy, 3)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["edge_streak_count"], 1)
        self.assertEqual(decision["edge_streak_start_frame"], 3)

    def test_frame216_style_single_measurement_gap_is_not_fabricated_hit(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0, margin=.5)
        decision = self.step(policy, 1, measured=False)
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["measured"])
        self.assertIsNone(decision["measurement_source_xy"])
        self.assertIsNone(decision["features"])
        self.assertEqual(decision["measurement_frame"], 0)
        self.assertTrue(self.step(policy, 2, margin=.4)["measured"])

    def test_orphan_coast_unknown_retains_without_creating_cache(self):
        policy = CausalEdgePersistence()
        decision = self.step(policy, 0, measured=False)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["tier"], "prediction_unknown")
        self.assertEqual(decision["reason"], "unknown_missing_history")
        self.assertIsNone(decision["measurement_frame"])
        self.assertEqual(policy.retained_state_counts(), dict(tracks=0, edge_streaks=0))

    def test_absent_identity_clears_all_state_before_rebirth(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        self.assertEqual(policy.update(row(2), {}), [])
        self.assertEqual(policy.retained_state_counts()["tracks"], 0)
        self.assertEqual(self.step(policy, 3, measured=False)["reason"], "unknown_missing_history")
        self.assertTrue(self.step(policy, 4)["accepted"])

    def test_unqualified_actual_or_prediction_clears_all_state(self):
        for measured in (True, False):
            with self.subTest(measured=measured):
                policy = CausalEdgePersistence()
                self.step(policy, 0)
                self.step(policy, 1)
                self.assertIsNone(self.step(policy, 2, measured=measured, qualified=False))
                self.assertEqual(policy.retained_state_counts()["tracks"], 0)
                self.assertEqual(self.step(policy, 3, measured=False)["reason"],
                                 "unknown_missing_history")

    def test_unqualified_measurements_are_not_scored_or_promoted(self):
        policy = CausalEdgePersistence()
        original = row(0, [track(qualified=False)])
        self.assertEqual(policy.update(original, {}), [])
        self.assertFalse(original["tracks"][0]["qualified_moving"])

    def test_same_segment_motion_reset_clears_verdict_and_edge_lineage(self):
        for measured in (True, False):
            with self.subTest(measured=measured):
                policy = CausalEdgePersistence()
                self.step(policy, 0)
                self.step(policy, 1)
                decision = self.step(policy, 2, reset=True, measured=measured)
                self.assertTrue(decision["accepted"])
                self.assertEqual(decision["edge_streak_count"], int(measured))
                self.assertFalse(decision["inherited"])

    def test_segment_change_cannot_inherit_same_track_id(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        decision = self.step(policy, 2, segment=1, measured=False)
        self.assertEqual(decision["reason"], "unknown_missing_history")
        self.assertEqual(decision["segment"], 1)
        self.assertTrue(self.step(policy, 3, segment=1)["accepted"])

    def test_identity_and_polarity_do_not_share_history(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        rows = [track(), track("bright:2"), track("dark:1")]
        decisions = policy.update(row(1, rows), evidence(rows))
        self.assertEqual([d["accepted"] for d in decisions], [False, True, True])
        self.assertEqual([d["edge_streak_count"] for d in decisions], [2, 1, 1])

    def test_output_qualified_order_matches_input_and_track_order_is_irrelevant(self):
        first, second = CausalEdgePersistence(), CausalEdgePersistence()
        for frame in range(4):
            tracks = [track(), track("dark:2"), track("bright:3", qualified=False)]
            observed = evidence(tracks)
            observed[(0, "dark:2")]["features"]["point_minus_edge_fraction"] = .4
            a = first.update(row(frame, tracks), observed)
            b = second.update(row(frame, list(reversed(tracks))), observed)
            self.assertEqual(a, list(reversed(b)))
            self.assertEqual([d["track_id"] for d in a], ["bright:1", "dark:2"])

    def test_motion_and_location_are_not_additional_inference_gates(self):
        policy = CausalEdgePersistence()
        positions = [(10., 10.), (15., 8.), (10., 7.), (10., 7.), (-99., 1234.)]
        for frame, xy in enumerate(positions):
            decision = self.step(policy, frame, margin=.5, xy=xy)
            self.assertTrue(decision["accepted"])
            self.assertEqual(decision["measurement_source_xy"], list(xy))
            self.assertEqual(decision["physical_class"], "unknown")
            self.assertFalse(decision["airborne_confirmed"])

    def test_cached_decision_not_source_reference_or_review_labels(self):
        a, b = CausalEdgePersistence(), CausalEdgePersistence()
        original = row(0, [track()])
        changed = copy.deepcopy(original)
        changed.update(clip_id="totally_different", label="negative", reference_xy=[999., -3.])
        changed["tracks"][0].update(source_xy=[-4., 7.], reference_xy=[0., 0.], truth="airborne")
        self.assertEqual(a.update(original, evidence(original["tracks"])),
                         b.update(changed, evidence(changed["tracks"])))

    def test_upstream_v36_accepted_flag_cannot_override_features(self):
        policy = CausalEdgePersistence()
        for frame in range(2):
            observed = evidence([track()])
            observed[(0, "bright:1")]["accepted"] = True
            observed[(0, "bright:1")]["reason"] = "point_preferred"
            decision = policy.update(row(frame, [track()]), observed)[0]
        self.assertFalse(decision["accepted"])

    def test_frame_index_must_start_zero_and_be_contiguous(self):
        for invalid in (-1, 1, True, 0.0, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(invalid, timestamp=0), {})
        policy = CausalEdgePersistence()
        policy.update(row(0), {})
        for invalid in (0, 2):
            with self.assertRaises(ValueError):
                policy.update(row(invalid, timestamp=1), {})

    def test_timestamps_allow_nonzero_start_and_require_strict_increase(self):
        policy = CausalEdgePersistence()
        policy.update(row(0, timestamp=1000), {})
        for invalid in (-1, 1000, 999, True, 1001., None):
            malformed = row(1, timestamp=1001)
            malformed["timestamp_ns"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                policy.update(malformed, {})
        policy.update(row(1, timestamp=1001), {})

    def test_explicit_motion_reset_is_required_every_frame(self):
        for motion in (None, {}, {"reset": 0}, {"reset": "false"}, {"reset": None}):
            malformed = row(0)
            malformed["motion"] = motion
            with self.subTest(motion=motion), self.assertRaises(ValueError):
                CausalEdgePersistence().update(malformed, {})

    def test_segment_and_track_types_validated_before_mutation(self):
        for segment in (-1, True, 0., "0", None):
            malformed = row(0)
            malformed["segment"] = segment
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                CausalEdgePersistence().update(malformed, {})
        for tracks in (None, {}, [None], [1], ["bright:1"]):
            malformed = row(0)
            malformed["tracks"] = tracks
            with self.subTest(tracks=tracks), self.assertRaises(ValueError):
                CausalEdgePersistence().update(malformed, {})
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update([], {})

    def test_bad_ids_duplicate_ids_and_wrong_segment_rejected(self):
        for identity in (None, 1, "", "bright", "bright:", "other:1"):
            tracks = [track(identity)]
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, tracks), evidence(tracks))
        for tracks in ([track(), track()], [track(segment=1)]):
            with self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, tracks), evidence(tracks))
        tracks = [track()]
        tracks[0]["segment"] = False
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update(row(0, tracks), evidence(tracks))

    def test_nonboolean_track_evidence_rejected(self):
        for field in ("measured", "qualified_moving"):
            for value in (0, 1, None, "true"):
                tracks = [track()]
                tracks[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CausalEdgePersistence().update(row(0, tracks), {})

    def test_bad_actual_measurements_rejected_even_when_unqualified(self):
        for xy in (None, [1], [1, 2, 3], [True, 2], [float("nan"), 2], [1, float("inf")]):
            tracks = [track(qualified=False)]
            tracks[0]["measurement_source_xy"] = xy
            with self.subTest(xy=xy), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, tracks), {})

    def test_prediction_cannot_carry_actual_coordinate_or_pixel_evidence(self):
        tracks = [track(measured=False)]
        tracks[0]["measurement_source_xy"] = [1., 2.]
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update(row(0, tracks), {})
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update(row(0, [track(measured=False)]), evidence([track()]))

    def test_evidence_inventory_must_match_exact_qualified_measurements(self):
        for observed in ({}, None, [], {(0, "bright:2"): {}}):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, [track()]), observed)
        observed = evidence([track(), track("bright:2")])
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update(row(0, [track()]), observed)
        with self.assertRaises(ValueError):
            CausalEdgePersistence().update(row(0, [track(qualified=False)]), evidence([track()]))

    def test_evidence_keys_require_exact_nonboolean_integer_segment(self):
        for key in ((False, "bright:1"), (0., "bright:1"), "bright:1", (0,), (0, 1)):
            observed = {key: dict(features=None, reason="unknown_unavailable")}
            with self.subTest(key=key), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, [track()]), observed)

    def test_evidence_fields_and_unknown_reasons_validated(self):
        for record in (None, {}, {"features": None}, {"features": None, "reason": ""},
                       {"features": None, "reason": "edge"},
                       {"features": None, "reason": 1},
                       {"features": {}, "reason": "unknown"},
                       {"features": [], "reason": "unknown"}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                CausalEdgePersistence().update(row(0, [track()]), {(0, "bright:1"): record})

    def test_feature_types_and_nonfinite_scores_rejected(self):
        for field, values in (("informative", (1, "true", None)),
                              ("point_minus_edge_fraction", (True, None, "0", float("nan"),
                                                             float("inf"), -float("inf")))):
            for value in values:
                observed = evidence([track()])
                observed[(0, "bright:1")]["features"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    CausalEdgePersistence().update(row(0, [track()]), observed)

    def test_validation_failure_does_not_clear_reset_or_commit_first_track(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        tracks = [track(), track("dark:2")]
        observed = evidence(tracks, margin=.9)
        observed[(0, "dark:2")]["features"]["point_minus_edge_fraction"] = float("nan")
        with self.assertRaises(ValueError):
            policy.update(row(2, tracks, reset=True), observed)
        decision = self.step(policy, 2, measured=False)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["measurement_frame"], 1)

    def test_late_copy_failure_cannot_commit_partial_frame(self):
        class CannotCopy:
            def __deepcopy__(self, memo):
                raise RuntimeError("generated copy failure")

        policy = CausalEdgePersistence()
        self.step(policy, 0)
        self.step(policy, 1)
        tracks = [track(), track("dark:2")]
        observed = evidence(tracks, margin=.8)
        observed[(0, "dark:2")]["features"]["extra"] = CannotCopy()
        with self.assertRaises(RuntimeError):
            policy.update(row(2, tracks), observed)
        decision = self.step(policy, 2, measured=False)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["measurement_frame"], 1)

    def test_inputs_outputs_and_cache_do_not_share_mutable_objects(self):
        policy = CausalEdgePersistence()
        original = row(0, [track()])
        observed = evidence(original["tracks"])
        observed[(0, "bright:1")]["features"]["extra"] = [1, 2, 3]
        baseline, before = copy.deepcopy(original), copy.deepcopy(observed)
        output = policy.update(original, observed)
        self.assertEqual(original, baseline)
        self.assertEqual(observed, before)
        output[0]["features"]["extra"].append(99)
        output[0]["measurement_source_xy"][0] = -99
        output[0]["accepted"] = False
        self.assertEqual(original, baseline)
        self.assertEqual(observed, before)
        coast = self.step(policy, 1, measured=False)
        self.assertTrue(coast["accepted"])
        self.assertEqual(coast["inherited_reason"], "edge_first_or_interrupted")

    def test_previous_decisions_unchanged_after_new_measurements(self):
        policy = CausalEdgePersistence()
        previous = self.step(policy, 0)
        saved = copy.deepcopy(previous)
        self.step(policy, 1)
        self.step(policy, 2, measured=False)
        self.assertEqual(previous, saved)

    def test_state_exposes_only_counts_and_does_not_leak_mutable_cache(self):
        policy = CausalEdgePersistence()
        self.step(policy, 0)
        counts = policy.retained_state_counts()
        self.assertEqual(counts, dict(tracks=1, edge_streaks=1))
        counts["tracks"] = 99
        self.assertEqual(policy.retained_state_counts()["tracks"], 1)

    def test_unknown_is_not_airborne_or_proof_of_target_absence(self):
        policy = CausalEdgePersistence()
        for frame in range(3):
            decision = self.step(policy, frame)
            self.assertEqual(decision["physical_class"], "unknown")
            self.assertFalse(decision["airborne_confirmed"])
            self.assertTrue(decision["baseline_qualified"])
        self.assertFalse(decision["accepted"])
        self.assertTrue(decision["measured"])


if __name__ == "__main__":
    unittest.main()
