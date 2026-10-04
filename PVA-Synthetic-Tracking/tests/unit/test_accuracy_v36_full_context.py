"""Fail-closed full-shadow accounting with synthetic/mocked evidence only."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/evaluate_accuracy_v36_full_context.py"
SPEC = importlib.util.spec_from_file_location("accuracy_v36_full_context_under_test", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
with mock.patch.object(sys, "path", [str(MODULE_PATH.parent), *sys.path]):
    SPEC.loader.exec_module(runner)


def track(tid="bright:1", *, qualified=True, measured=True, segment=0, xy=(20, 30)):
    return {"track_id": tid, "segment": segment, "qualified_moving": qualified,
            "measured": measured, "measurement_source_xy": list(xy) if measured else None,
            "source_xy": list(xy)}


def decision(*, accepted=True, frame=4, features=None, reason="synthetic"):
    return {"accepted": accepted, "measurement_frame": frame,
            "features": features, "reason": reason}


def reference_score(hits, *, window="synthetic", identities=None):
    identities = identities or ["0/bright:1"] * len(hits)
    evidence = [{"frame_index": frame, "qualified_measured_hit": hit,
                 "assigned_track_id": identities[frame] if hit else None}
                for frame, hit in enumerate(hits)]
    return {"positive_windows": [{"window_id": window, "visible_samples": len(hits),
                                  "qualified_measured_hits": sum(hits), "evidence": evidence}]}


class AccuracyV36FullContextTests(unittest.TestCase):
    def test_missing_and_unknown_decision_identities_rejected(self):
        row = {"frame_index": 4, "tracks": [track()]}
        for decisions in ({}, {(0, "bright:2"): decision()},
                          {(0, "bright:1"): decision(), (0, "dark:9"): decision()}):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                runner.validate_decisions(row, decisions)

    def test_duplicate_journal_identity_rejected(self):
        row = {"frame_index": 4, "tracks": [track(), track()]}
        with self.assertRaises(ValueError):
            runner.validate_decisions(row, {(0, "bright:1"): decision()})

    def test_accepted_output_cannot_add_nonbaseline_qualified_track(self):
        row = {"frame_index": 4, "tracks": [track(qualified=False)]}
        with self.assertRaises(ValueError):
            runner.validate_decisions(row, {(0, "bright:1"): decision()})
        self.assertEqual(runner.validate_decisions(row, {(0, "bright:1"): decision(accepted=False)}), [])

    def test_gate_acceptance_and_reason_are_strict(self):
        row = {"frame_index": 4, "tracks": [track()]}
        invalid = [decision(accepted=value) for value in (1, 0, "yes", None)]
        invalid += [decision(reason=value) for value in ("", None, 7)]
        for malformed in invalid:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                runner.validate_decisions(row, {(0, "bright:1"): malformed})

    def test_invalid_or_future_measurement_provenance_rejected(self):
        row = {"frame_index": 4, "tracks": [track()]}
        for frame in (-1, 5, 4.0, True):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                runner.validate_decisions(row, {(0, "bright:1"): decision(frame=frame)})

    def test_qualified_current_measurement_cannot_use_missing_or_stale_provenance(self):
        row = {"frame_index": 4, "tracks": [track()]}
        for frame in (None, 0, 3):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                runner.validate_decisions(row, {(0, "bright:1"): decision(frame=frame)})

    def test_missing_required_decision_fields_rejected(self):
        row = {"frame_index": 4, "tracks": [track()]}
        for field in ("accepted", "reason", "measurement_frame", "features"):
            malformed = decision()
            del malformed[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_decisions(row, {(0, "bright:1"): malformed})

    def test_prediction_cannot_borrow_current_measurement(self):
        row = {"frame_index": 4, "tracks": [track(measured=False)]}
        with self.assertRaises(ValueError):
            runner.validate_decisions(row, {(0, "bright:1"): decision(frame=4)})
        self.assertEqual(runner.validate_decisions(row, {(0, "bright:1"): decision(frame=3)}), row["tracks"])
        self.assertEqual(runner.validate_decisions(row, {(0, "bright:1"): decision(frame=None, reason="unknown_missing_history")}), row["tracks"])

    def test_features_require_provenance_and_finite_serializable_values(self):
        row = {"frame_index": 4, "tracks": [track()]}
        invalid = (decision(frame=None, features={"informative": True}),
                   decision(features=[1, 2]), decision(features={"margin": float("nan")}),
                   decision(features={"margin": float("inf")}))
        for malformed in invalid:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                runner.validate_decisions(row, {(0, "bright:1"): malformed})

    def test_validate_returns_original_baseline_tracks_without_mutation(self):
        row = {"frame_index": 4, "tracks": [track(), track("dark:2", qualified=False)]}
        decisions = {(0, "bright:1"): decision(accepted=False),
                     (0, "dark:2"): decision(accepted=False, frame=None)}
        before = copy.deepcopy((row, decisions))
        relevant = runner.validate_decisions(row, decisions)
        self.assertEqual(relevant, row["tracks"][:1])
        self.assertIs(relevant[0], row["tracks"][0])
        self.assertEqual((row, decisions), before)

    def summarize(self, old_hits, new_hits, *, old_anchors=None, new_anchors=None,
                  new_identities=None, pilot_new_hits=None):
        old_anchors = old_anchors if old_anchors is not None else [
            {"event_id": "synthetic", "frame": i, "ids": ["0/bright:1"]}
            for i in range(len(old_hits))]
        new_anchors = copy.deepcopy(old_anchors) if new_anchors is None else new_anchors
        stats = runner.new_stats([])
        for value in stats.values():
            value.update(measured=len(old_hits), predicted=0, ids={(0, "bright:1")},
                         per_frame=[{"frame_index": i, "measured": 1, "predicted": 0}
                                    for i in range(len(old_hits))])
        scores = {
            "baseline": {"dense": reference_score(old_hits), "pilot": reference_score(old_hits)},
            "point_context": {"dense": reference_score(new_hits, identities=new_identities),
                              "pilot": reference_score(new_hits if pilot_new_hits is None else pilot_new_hits)},
        }
        references = {kind: (kind, {}) for kind in ("dense", "pilot")}
        with mock.patch.object(runner, "reference_spec", return_value=(references, set(), [], [])), \
             mock.patch.object(runner, "score_rows", side_effect=lambda rows, labels, *args: copy.deepcopy(scores[rows][labels])):
            return runner.summarize_clip("synthetic", len(old_hits), stats,
                                         {arm: arm for arm in runner.ARMS},
                                         {"baseline": old_anchors, "point_context": new_anchors},
                                         {"synthetic": True}, "decision-sha")

    def test_baseline_missed_samples_stay_in_denominator(self):
        result = self.summarize([True, False, True], [True, False, True])
        counts = result["arms"]["point_context"]["retention"]["dense"]
        self.assertEqual(counts["samples"], 3)
        self.assertEqual(counts["baseline_hits"], 2)
        self.assertEqual(counts["candidate_hits"], 2)
        self.assertEqual(counts["lost_visible_samples"], [])
        self.assertTrue(result["arms"]["point_context"]["known_reference_retention_passed"])

    def test_missing_baseline_reference_sample_is_rejected_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            self.summarize([True, False, True], [True, True])

    def test_eighty_percent_frame_retention_is_not_success(self):
        result = self.summarize([True] * 5, [True, True, True, True, False])
        candidate = result["arms"]["point_context"]
        self.assertFalse(candidate["known_reference_retention_passed"])
        self.assertEqual(candidate["retention"]["dense"]["lost_visible_samples"],
                         [{"window": "synthetic", "frame": 4}])

    def test_pilot_failure_not_hidden_by_unchanged_dense_samples(self):
        result = self.summarize([True] * 3, [True] * 3, pilot_new_hits=[True, False, True])
        candidate = result["arms"]["point_context"]
        self.assertTrue(candidate["retention"]["dense"]["no_new_misses"])
        self.assertFalse(candidate["retention"]["pilot"]["no_new_misses"])
        self.assertFalse(candidate["known_reference_retention_passed"])

    def test_losing_one_of_five_required_anchors_fails_strict_retention(self):
        anchors = [{"event_id": "synthetic", "frame": i, "ids": ["0/bright:1"] if i < 4 else []}
                   for i in range(5)]
        result = self.summarize([True] * 5, [True] * 5, new_anchors=anchors)
        candidate = result["arms"]["point_context"]
        self.assertFalse(candidate["known_reference_retention_passed"])
        self.assertEqual(candidate["lost_required_anchors"], [{"event_id": "synthetic", "frame": 4}])

    def test_missing_required_anchor_evidence_rejected(self):
        anchors = [{"event_id": "synthetic", "frame": 0, "ids": ["0/bright:1"]}]
        with self.assertRaises(ValueError):
            self.summarize([True, True], [True, True], new_anchors=anchors)

    def test_changed_measured_identity_fails_even_if_every_frame_hits(self):
        result = self.summarize([True] * 3, [True] * 3,
                                new_identities=["0/bright:1", "0/bright:2", "0/bright:1"])
        self.assertFalse(result["arms"]["point_context"]["known_reference_retention_passed"])
        self.assertEqual(len(result["arms"]["point_context"]["retention"]["dense"]["changed_assignments"]), 1)

    def test_prediction_only_never_matches_required_anchor(self):
        anchors = [{"event_id": "synthetic", "frame_index": 4, "xy": [20, 30],
                    "uncertainty_px": 2, "polarity": "bright"}]
        row = {"frame_index": 4}
        matched = runner.matched_anchors(row, [track(measured=False)], anchors)
        self.assertEqual(matched, [{"event_id": "synthetic", "frame": 4, "ids": []}])
        measured = runner.matched_anchors(row, [track()], anchors)
        self.assertEqual(measured[0]["ids"], ["0/bright:1"])

    def test_source_control_counts_separate_measurements_and_predictions(self):
        controls = [{"frames_inclusive": [4, 4], "crop_xywh": [10, 20, 20, 20], "label": "synthetic"}]
        stats = runner.new_stats(controls)["baseline"]
        tracks = [track(), track("bright:2", xy=(50, 30)), track("dark:3", measured=False)]
        runner.accumulate({"frame_index": 4}, tracks, stats, controls)
        self.assertEqual(stats["measured"], 2)
        self.assertEqual(stats["predicted"], 1)
        self.assertEqual(stats["controls"][0]["measured"], 1)
        self.assertEqual(stats["controls"][0]["predicted"], 1)

    def test_summary_does_not_claim_airborne_accuracy_or_feedback_change(self):
        result = self.summarize([True], [True])
        for name in ("airborne_precision", "airborne_recall", "false_alarms_per_minute"):
            self.assertIsNone(result[name])
        self.assertIs(result["detector_rerun"], False)
        self.assertIs(result["feedback_changed"], False)
        self.assertEqual(result["decisions_sha256"], "decision-sha")

    def bounded_fixture(self):
        y, x = np.indices((64, 64))
        gray = ((x * 3 + y * 7) % 251).astype(np.uint8)
        feature = {"informative": True, "point_minus_edge_fraction": 0.4,
                   "point_offset_xy": [1.0, -1.0]}
        expected = {"identity": "0/bright:1", "measurement_source_xy": [20, 30],
                    "patch_sha256": hashlib.sha256(np.ascontiguousarray(gray[18:43, 8:33]).tobytes()).hexdigest(),
                    "zero_margin_ablation_passed": True, "features": feature}
        row = {"frame_index": 4, "tracks": [track()]}
        decisions = {(0, "bright:1"): decision(features=copy.deepcopy(feature))}
        return expected, row, gray, decisions

    def test_bounded_patch_crosscheck_accepts_same_source_pixels_and_features(self):
        fixture = self.bounded_fixture()
        runner.check_bounded_observation(*fixture)
        fixture[3][(0, "bright:1")]["features"]["point_minus_edge_fraction"] += 1e-9
        runner.check_bounded_observation(*fixture)

    def test_bounded_patch_crosscheck_rejects_changed_pixel_identity_or_acceptance(self):
        for change in ("pixel", "identity", "coordinate", "acceptance"):
            with self.subTest(change=change):
                expected, row, gray, decisions = self.bounded_fixture()
                if change == "pixel":
                    gray[30, 20] = (int(gray[30, 20]) + 1) % 256
                elif change == "identity":
                    row["tracks"][0]["track_id"] = "bright:2"
                elif change == "coordinate":
                    row["tracks"][0]["measurement_source_xy"] = [21, 30]
                else:
                    decisions[(0, "bright:1")]["accepted"] = False
                with self.assertRaises(ValueError):
                    runner.check_bounded_observation(expected, row, gray, decisions)

    def test_bounded_feature_crosscheck_rejects_inventory_numeric_and_bool_changes(self):
        for change in ("inventory", "numeric", "nan", "boolean"):
            with self.subTest(change=change):
                expected, row, gray, decisions = self.bounded_fixture()
                actual = decisions[(0, "bright:1")]["features"]
                if change == "inventory":
                    actual["new_feature"] = 1
                elif change == "numeric":
                    actual["point_minus_edge_fraction"] += 0.01
                elif change == "nan":
                    actual["point_minus_edge_fraction"] = float("nan")
                else:
                    actual["informative"] = 1
                with self.assertRaises(ValueError):
                    runner.check_bounded_observation(expected, row, gray, decisions)

    def test_bounded_uninformative_patch_respects_conservative_full_gate_policy(self):
        expected, row, gray, decisions = self.bounded_fixture()
        expected["features"]["informative"] = False
        expected["features"]["point_minus_edge_fraction"] = 0.0
        expected["zero_margin_ablation_passed"] = False
        decisions[(0, "bright:1")].update(accepted=True, reason="unknown_uninformative_patch",
                                          features=copy.deepcopy(expected["features"]))
        runner.check_bounded_observation(expected, row, gray, decisions)

    def audit_fixture(self):
        audit_path = Path("/synthetic/context-independent-audit.json")
        context = runner.CONTEXT
        code_name = "scripts/accuracy_v36_context.py"
        external = "/synthetic/previously-audited-reference.json"
        hashes = {}

        def digest(path):
            canonical = str(Path(path).resolve())
            return hashes.setdefault(canonical, hashlib.sha256(canonical.encode()).hexdigest())

        freeze = {"pre_extraction": True, "classifier_promoted": False,
                  "inputs_sha256": {external: digest(external)},
                  "implementation_sha256": {code_name: digest(runner.ROOT / code_name)},
                  "selection_sha256": digest(context / "selection.json"),
                  "unit_log_sha256": digest(context / "unit.log")}
        hashes[str(context / "implementation" / code_name)] = digest(runner.ROOT / code_name)
        summary = {"completed": True, "freeze_sha256": digest(context / "freeze.json"),
                   "outputs_sha256": {"observations.json": digest(context / "observations.json"),
                                      "native_patches.npz": digest(context / "native_patches.npz")}}
        required = {str(context / name) for name in
                    ("freeze.json", "summary.json", "selection.json", "observations.json", "native_patches.npz", "unit.log")}
        required.update((external, str(runner.ROOT / code_name), str(context / "implementation" / code_name)))
        audit = {"schema": "seaqr.accuracy-v36-context-independent-audit.v1", "verified": True,
                 "experiment": str(context), "context_freeze_sha256": digest(context / "freeze.json"),
                 "complete_summary_verified": True, "zero_margin_any_alternative_decisions_match": True,
                 "auditor_sha256": digest(runner.ROOT / "scripts/audit_accuracy_v36_context.py"),
                 "checked_files_sha256": {path: digest(path) for path in required}}
        payloads = {str(audit_path): audit, str(context / "freeze.json"): freeze,
                    str(context / "summary.json"): summary}
        return audit_path, audit, freeze, summary, hashes, payloads, digest

    def verify_fixture(self, fixture):
        audit_path, _, _, _, _, payloads, digest = fixture
        with mock.patch.object(runner, "read", side_effect=lambda path: copy.deepcopy(payloads[str(Path(path).resolve())])), \
             mock.patch.object(runner, "sha", side_effect=digest), \
             mock.patch.object(runner.cv2, "VideoCapture") as capture:
            result = runner.verify_context_audit(audit_path)
            capture.assert_not_called()
            return result

    def test_complete_hash_bound_audit_accepted_without_opening_media(self):
        fixture = self.audit_fixture()
        files, freeze = self.verify_fixture(fixture)
        self.assertEqual(freeze, fixture[2])
        self.assertIn(str(fixture[0]), files)
        self.assertTrue(set(fixture[1]["checked_files_sha256"]) <= files.keys())

    def test_incomplete_or_wrong_audit_status_and_scope_fail_closed(self):
        changes = ((1, "verified", False), (1, "verified", "true"),
                   (1, "schema", "wrong"), (1, "experiment", "/synthetic/unrelated"),
                   (1, "context_freeze_sha256", "wrong"), (1, "auditor_sha256", "wrong"),
                   (1, "complete_summary_verified", False),
                   (1, "zero_margin_any_alternative_decisions_match", False),
                   (2, "pre_extraction", False), (2, "classifier_promoted", True),
                   (3, "completed", False), (3, "freeze_sha256", "wrong"))
        for target, field, value in changes:
            with self.subTest(target=target, field=field, value=value):
                fixture = self.audit_fixture()
                fixture[target][field] = value
                with self.assertRaises(ValueError):
                    self.verify_fixture(fixture)

    def test_missing_audit_bound_file_and_changed_actual_digest_rejected(self):
        fixture = self.audit_fixture()
        fixture[1]["checked_files_sha256"].pop(str(runner.CONTEXT / "native_patches.npz"))
        with self.assertRaises(ValueError):
            self.verify_fixture(fixture)
        fixture = self.audit_fixture()
        fixture[4][str(runner.CONTEXT / "observations.json")] = "changed"
        with self.assertRaises(ValueError):
            self.verify_fixture(fixture)

    def test_freeze_bound_selection_and_unit_log_hashes_rechecked(self):
        for field in ("selection_sha256", "unit_log_sha256"):
            with self.subTest(field=field):
                fixture = self.audit_fixture()
                fixture[2][field] = "wrong"
                with self.assertRaises(ValueError):
                    self.verify_fixture(fixture)

    def test_bad_audit_aborts_full_run_before_inputs_or_video_access(self):
        with mock.patch.object(Path, "exists", return_value=False), \
             mock.patch.object(runner, "verify_context_audit", side_effect=ValueError("synthetic failed audit")), \
             mock.patch.object(runner, "verified_inputs") as inputs, \
             mock.patch.object(runner, "sha") as digest, \
             mock.patch.object(runner.cv2, "VideoCapture") as capture:
            with self.assertRaises(ValueError):
                runner.run(Path("/synthetic/new-output"), Path("/synthetic/audit.json"))
            inputs.assert_not_called()
            digest.assert_not_called()
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
