"""Generated V57 independent-auditor tests; no real journal or media inputs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[2] / "scripts/audit_accuracy_v57.py"
SPEC = importlib.util.spec_from_file_location("independent_accuracy_v57_audit_test", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def track(identity="bright:1", measured=True, qualified=True, segment=0):
    return dict(track_id=identity, measured=measured, qualified_moving=qualified,
                segment=segment, source_xy=[10.0, 12.0],
                measurement_source_xy=[10.0, 12.0] if measured else None)


def row(frame=0, tracks=None, timestamp=None, segment=0, reset=False):
    return dict(frame_index=frame, timestamp_ns=frame * 100_000_000 if timestamp is None else timestamp,
                segment=segment, motion=dict(reset=reset),
                tracks=[track(segment=segment)] if tracks is None else tracks)


def feature(margin=-.3, informative=True, reason="edge_preferred_or_tie"):
    return dict(features=dict(informative=informative, point_minus_edge_fraction=margin), reason=reason)


def evidence(r, item=None):
    return {(t["segment"], t["track_id"]): copy.deepcopy(item if item is not None else feature())
            for t in r["tracks"] if t["measured"] and t["qualified_moving"]}


def saved(r, margin=-.3):
    compact = []
    for t in r["tracks"]:
        if not t["qualified_moving"]:
            continue
        current = t["measured"]
        compact.append({k: copy.deepcopy(t[k]) for k in (
            "track_id", "segment", "measured", "source_xy", "measurement_source_xy")})
        compact[-1].update(accepted=margin > 0, reason="edge_preferred_or_tie" if current else "coast_edge_preferred_or_tie",
                           measurement_frame=r["frame_index"] if current else None,
                           features=feature(margin)["features"] if current else None)
    return {k: copy.deepcopy(r[k]) for k in ("frame_index", "timestamp_ns", "segment")} | dict(tracks=compact)


class V57AuditIntegrityTests(unittest.TestCase):
    def test_no_import_of_experiment_policy_or_runner(self):
        tree = ast.parse(PATH.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(n and ("v57_policy" in n or "run_accuracy_v57" in n
                                    or "accuracy_v57_core" in n) for n in imports))

    def test_exact_comparison_distinguishes_boolean_and_numeric_types(self):
        for bad in (True, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit.exact(bad, 1, "typed number")
        audit.exact(dict(a=1, b=2), dict(b=2, a=1), "key order")

    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":-1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                audit.decode(raw)

    def test_integrity_detects_late_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.txt"
            path.write_text("before")
            integrity = audit.Integrity()
            integrity.bind(path)
            integrity.recheck()
            path.write_text("after")
            with self.assertRaises(ValueError):
                integrity.recheck()

    def test_integrity_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.txt"
            link = Path(directory) / "link.txt"
            original.write_text("content")
            link.symlink_to(original)
            with self.assertRaises(ValueError):
                audit.digest(link)

    def test_saved_features_bind_exact_track_and_current_coordinates(self):
        source = row(tracks=[track(), track("dark:2", qualified=False)])
        old = saved(source)
        before = copy.deepcopy((source, old))
        self.assertEqual(audit.saved_evidence(source, old), evidence(source))
        self.assertEqual((source, old), before)
        for field, changed in (("track_id", "bright:99"), ("measurement_source_xy", [11., 12.]),
                               ("source_xy", [99., 12.]), ("measured", False), ("segment", 1)):
            bad = copy.deepcopy(old)
            bad["tracks"][0][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.saved_evidence(source, bad)

    def test_saved_features_require_complete_qualified_inventory(self):
        source = row()
        old = saved(source)
        old["tracks"] = []
        with self.assertRaises(ValueError):
            audit.saved_evidence(source, old)

    def test_saved_current_features_require_current_provenance(self):
        source = row(3)
        old = saved(source)
        old["tracks"][0]["measurement_frame"] = 2
        with self.assertRaises(ValueError):
            audit.saved_evidence(source, old)

    def test_saved_coast_cannot_borrow_current_features(self):
        source = row(3, [track(measured=False)])
        old = saved(source)
        self.assertEqual(audit.saved_evidence(source, old), {})
        old["tracks"][0]["features"] = feature()["features"]
        with self.assertRaises(ValueError):
            audit.saved_evidence(source, old)


class V57IndependentStateTests(unittest.TestCase):
    def step(self, engine, frame, margin=-.3, tracks=None, **kwargs):
        current = row(frame, tracks=tracks, **kwargs)
        return engine.step(current, evidence(current, feature(margin)))[0]

    def test_two_consecutive_edges_reject_second_and_streak_is_not_capped(self):
        engine = audit.IndependentPersistence()
        output = [self.step(engine, frame) for frame in range(5)]
        self.assertEqual([item["accepted"] for item in output], [True, False, False, False, False])
        self.assertEqual([item["edge_streak_count"] for item in output], [1, 2, 3, 4, 5])
        self.assertEqual(output[-1]["edge_streak_start_frame"], 0)
        self.assertEqual(output[-1]["edge_streak_start_timestamp_ns"], 0)
        self.assertEqual(output[0]["reason"], "edge_first_or_interrupted")
        self.assertEqual(output[1]["reason"], "edge_consecutive_rejected")

    def test_zero_is_edge_and_arbitrarily_small_positive_resets(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0, margin=0.)
        self.assertFalse(self.step(engine, 1, margin=-0.)["accepted"])
        point = self.step(engine, 2, margin=1e-300)
        self.assertTrue(point["accepted"])
        self.assertEqual(point["edge_streak_count"], 0)
        self.assertEqual(point["tier"], "point_supported_measured")
        self.assertTrue(self.step(engine, 3)["accepted"])

    def test_edge_time_boundary_is_inclusive(self):
        for delta, rejected in ((200_000_000, True), (200_000_001, False)):
            with self.subTest(delta=delta):
                engine = audit.IndependentPersistence()
                self.step(engine, 0, timestamp=17)
                second = self.step(engine, 1, timestamp=17 + delta)
                self.assertEqual(not second["accepted"], rejected)

    def test_one_prediction_breaks_edge_streak(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        self.step(engine, 1)
        coast = self.step(engine, 2, tracks=[track(measured=False)])
        self.assertFalse(coast["accepted"])
        self.assertEqual(coast["edge_streak_count"], 0)
        next_edge = self.step(engine, 3)
        self.assertTrue(next_edge["accepted"])
        self.assertEqual(next_edge["edge_streak_count"], 1)

    def test_coast_does_not_refresh_origin_and_frame_budget_expires(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        self.step(engine, 1)
        for frame in range(2, 9):
            coast = self.step(engine, frame, tracks=[track(measured=False)], timestamp=frame * 10_000_000 + 100_000_000)
            self.assertFalse(coast["accepted"])
            self.assertEqual(coast["measurement_frame"], 1)
            self.assertEqual(coast["measurement_timestamp_ns"], 100_000_000)
            self.assertEqual(coast["evidence_age_frames"], frame - 1)
        expired = self.step(engine, 9, tracks=[track(measured=False)], timestamp=200_000_000)
        self.assertTrue(expired["accepted"])
        self.assertFalse(expired["inherited"])
        self.assertIsNone(expired["measurement_frame"])
        self.assertEqual(expired["expired_measurement_frame"], 1)
        self.assertEqual(expired["reason"], "unknown_expired_history")
        missing = self.step(engine, 10, tracks=[track(measured=False)], timestamp=210_000_000)
        self.assertEqual(missing["reason"], "unknown_missing_history")
        self.assertIsNone(missing["expired_measurement_frame"])

    def test_coast_time_budget_inclusive_and_independent_of_frames(self):
        for delta, inherited in ((700_000_000, True), (700_000_001, False)):
            with self.subTest(delta=delta):
                engine = audit.IndependentPersistence()
                self.step(engine, 0)
                self.step(engine, 1)
                out = self.step(engine, 2, tracks=[track(measured=False)], timestamp=100_000_000 + delta)
                self.assertEqual(out["inherited"], inherited)
                self.assertEqual(out["accepted"], not inherited)

    def test_current_measurement_after_expiry_can_start_new_run(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        self.step(engine, 1)
        self.step(engine, 2, tracks=[track(measured=False)], timestamp=1_000_000_000)
        out = self.step(engine, 3, timestamp=1_100_000_000)
        self.assertEqual(out["edge_streak_count"], 1)
        self.assertTrue(out["accepted"])

    def test_uninformative_current_measurement_keeps_baseline_and_resets(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        self.step(engine, 1)
        current = row(2)
        unknown = engine.step(current, evidence(current, feature(-100, informative=False)))[0]
        self.assertTrue(unknown["accepted"])
        self.assertEqual(unknown["tier"], "unknown_measured")
        self.assertEqual(unknown["reason"], "unknown_uninformative_patch")
        self.assertEqual(unknown["measurement_frame"], 2)
        self.assertTrue(self.step(engine, 3)["accepted"])

    def test_missing_current_features_are_explicit_unknown_not_negative(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        current = row(1)
        out = engine.step(current, evidence(current, dict(features=None, reason="unknown_truncated_patch")))[0]
        self.assertTrue(out["accepted"])
        self.assertEqual(out["reason"], "unknown_truncated_patch")
        self.assertIsNone(out["evidence_informative"])
        self.assertTrue(self.step(engine, 2)["accepted"])

    def test_unqualified_measured_or_predicted_state_clears_history(self):
        for measured in (False, True):
            with self.subTest(measured=measured):
                engine = audit.IndependentPersistence()
                self.step(engine, 0)
                self.step(engine, 1)
                current = row(2, [track(measured=measured, qualified=False)])
                self.assertEqual(engine.step(current, {}), [])
                coast = self.step(engine, 3, tracks=[track(measured=False)])
                self.assertTrue(coast["accepted"])
                self.assertEqual(coast["reason"], "unknown_missing_history")

    def test_disappearance_clears_history_even_if_id_returns(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        self.step(engine, 1)
        engine.step(row(2, []), {})
        self.assertTrue(self.step(engine, 3)["accepted"])

    def test_reset_same_segment_and_segment_change_clear_history(self):
        for kwargs in (dict(reset=True), dict(segment=1)):
            with self.subTest(kwargs=kwargs):
                engine = audit.IndependentPersistence()
                self.step(engine, 0)
                self.step(engine, 1)
                self.assertTrue(self.step(engine, 2, **kwargs)["accepted"])

    def test_track_identity_has_independent_history_and_preserves_input_order(self):
        engine = audit.IndependentPersistence()
        current = row(0, [track("dark:2"), track("bright:1")])
        engine.step(current, evidence(current))
        current = row(1, [track("bright:1"), track("dark:3"), track("dark:2")])
        output = engine.step(current, evidence(current))
        self.assertEqual([x["track_id"] for x in output], ["bright:1", "dark:3", "dark:2"])
        self.assertEqual([x["accepted"] for x in output], [False, True, False])

    def test_no_promotion_or_measured_coordinate_on_coast(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0, margin=.3)
        out = self.step(engine, 1, tracks=[track(measured=False)])
        self.assertIsNone(out["features"])
        self.assertIsNone(out["measurement_source_xy"])
        self.assertIsNone(out["evidence_informative"])
        self.assertEqual(out["physical_class"], "unknown")
        self.assertIs(out["airborne_confirmed"], False)

    def test_input_and_returned_features_do_not_mutate_state(self):
        engine = audit.IndependentPersistence()
        current = row()
        supplied = evidence(current)
        before = copy.deepcopy((current, supplied))
        output = engine.step(current, supplied)
        self.assertEqual((current, supplied), before)
        output[0]["reason"] = "tampered"
        output[0]["features"]["informative"] = False
        out = self.step(engine, 1, tracks=[track(measured=False)])
        self.assertEqual(out["reason"], "coast_edge_first_or_interrupted")

    def test_prefix_is_causal(self):
        short, long = audit.IndependentPersistence(), audit.IndependentPersistence()
        a = [self.step(short, frame) for frame in range(3)]
        b = [self.step(long, frame) for frame in range(9)]
        self.assertEqual(a, b[:3])

    def test_validation_failure_does_not_partially_commit(self):
        engine = audit.IndependentPersistence()
        self.step(engine, 0)
        current = row(1, [track(), track("dark:2")])
        supplied = evidence(current)
        supplied[0, "dark:2"]["features"]["informative"] = 1
        with self.assertRaises(ValueError):
            engine.step(current, supplied)
        self.assertFalse(self.step(engine, 1)["accepted"])

    def test_invalid_frame_time_reset_segment_and_identity_rejected(self):
        mutations = [dict(frame_index=True), dict(frame_index=1), dict(timestamp_ns=-1),
                     dict(timestamp_ns=0.), dict(segment=True), dict(motion={}), dict(motion=dict(reset=1))]
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                current = row() | changes
                audit.IndependentPersistence().step(current, evidence(current))
        for identity in ("bright", "bright:", "blue:1", "", 1):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                current = row(tracks=[track(identity)])
                audit.IndependentPersistence().step(current, evidence(current))

    def test_duplicate_original_track_and_nonboolean_fields_rejected(self):
        for tracks in ([track(), track()], [track(measured=1)], [track(qualified=1)],
                       [track(segment=True)]):
            with self.subTest(tracks=tracks), self.assertRaises(ValueError):
                current = row(tracks=tracks)
                audit.IndependentPersistence().step(current, evidence(current))

    def test_predicted_measurement_and_missing_actual_measurement_rejected(self):
        for measured, xy in ((False, [1, 2]), (True, None), (True, [True, 2]), (True, [float("inf"), 2])):
            with self.subTest(measured=measured, xy=xy), self.assertRaises(ValueError):
                item = track(measured=measured)
                item["measurement_source_xy"] = xy
                current = row(tracks=[item])
                audit.IndependentPersistence().step(current, evidence(current))

    def test_missing_extra_or_malformed_evidence_rejected(self):
        current = row()
        bad = [{}, {**evidence(current), (0, "dark:2"): feature()},
               {(False, "bright:1"): feature()}, {(0, "bright:1"): dict(features=None, reason="edge")},
               {(0, "bright:1"): feature(float("nan"))}, {(0, "bright:1"): feature(True)},
               {(0, "bright:1"): feature(-.3, informative=1)}]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                audit.IndependentPersistence().step(current, item)


class V57StreamAuditTests(unittest.TestCase):
    def fixture(self):
        originals = [row(0), row(1), row(2, [track(measured=False)]), row(3)]
        saved_rows = [saved(item, .3 if item["frame_index"] == 3 else -.3) for item in originals]
        engine = audit.IndependentPersistence()
        decisions = []
        for item, old in zip(originals, saved_rows):
            decisions.append({k: item[k] for k in ("frame_index", "timestamp_ns", "segment")} |
                             dict(tracks=engine.step(item, audit.saved_evidence(item, old))))
        return originals, saved_rows, decisions

    def test_independent_counts_distinguish_measured_predicted_and_v36(self):
        result = audit.audit_clip_rows(*self.fixture(), 4)
        expected = {
            "baseline": dict(qualified_measured_states=3, qualified_predicted_states=1, distinct_segment_track_ids=1),
            "v36_point_context": dict(qualified_measured_states=1, qualified_predicted_states=0, distinct_segment_track_ids=1),
            "v57_persistence": dict(qualified_measured_states=2, qualified_predicted_states=0, distinct_segment_track_ids=1),
        }
        self.assertEqual(result["arms"], expected)
        self.assertEqual(result["accepted_measured_states_rejected_by_v36"], 1)
        self.assertEqual(result["rejected_measured_states"], 1)
        self.assertEqual(result["rejected_predicted_states"], 1)
        self.assertIsNone(result["false_alarms_per_minute"])
        self.assertIs(result["promotion_allowed"], False)

    def test_changed_decision_is_rejected(self):
        original, features, decisions = self.fixture()
        decisions[1]["tracks"][0]["accepted"] = True
        with self.assertRaises(ValueError):
            audit.audit_clip_rows(original, features, decisions, 4)

    def test_extra_decision_field_is_not_ignored(self):
        original, features, decisions = self.fixture()
        decisions[0]["tracks"][0]["future_label"] = True
        with self.assertRaises(ValueError):
            audit.audit_clip_rows(original, features, decisions, 4)

    def test_unequal_or_incomplete_or_extra_streams_rejected(self):
        original, features, decisions = self.fixture()
        for a, b, c, frames in ((original, features[:-1], decisions, 4),
                                (original, features, decisions, 5), (original, features, decisions, 3)):
            with self.subTest(frames=frames, feature_count=len(b)), self.assertRaises(ValueError):
                audit.audit_clip_rows(a, b, c, frames)


class V57ArtifactAuditTests(unittest.TestCase):
    def summary(self, output):
        return dict(freeze_sha256=audit.digest(output / "freeze.json"), production_changed=False,
                    promotion_allowed=False, classifier_promoted=False, completed=True, frames=2741,
                    known_counterexample_rejected=True, known_counterexample_blocks_promotion=True,
                    clips={clip: dict(frames=count, tiers={}, reasons={}, workload={arm: dict(
                        measured=0, predicted=0, distinct_identities=0) for arm in ("baseline", "v36", "v57")})
                        for clip, count in audit.COUNTS.items()})

    def fixture(self, directory):
        repo = Path(directory) / "repo"
        output = repo / "generated_output"
        output.mkdir(parents=True)
        implementation = "scripts/audit_accuracy_v57.py"
        for base in (repo, output / "implementation"):
            path = base / implementation
            path.parent.mkdir(parents=True)
            shutil.copyfile(PATH, path)
        freeze = dict(schema="seaqr.accuracy_v57.freeze.v1", clips={}, inputs={},
                      config=dict(required_consecutive_edges=2, maximum_edge_gap_ns=200_000_000,
                                  maximum_coast_frames=7, maximum_coast_ns=700_000_000),
                      implementation={implementation: audit.digest(PATH)},
                      production_changed=False, promotion_allowed=False)
        for clip, count in audit.COUNTS.items():
            journal = "results/tiny_target/visible_validation_v34_20260923/audit_20260924/evidence/run/full_repeat0_" + clip + "/frames.jsonl"
            features = "results/tiny_target/accuracy_v36_20260924/full_context_01/" + clip + "_decisions.jsonl"
            originals = [row(frame, []) for frame in range(count)]
            compact = [{k: value[k] for k in ("frame_index", "timestamp_ns", "segment", "tracks")} for value in originals]
            for relative, values in ((journal, originals), (features, compact)):
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(value) + "\n" for value in values))
                freeze["inputs"][relative] = audit.digest(path)
            (output / (clip + "_decisions.jsonl")).write_text("".join(json.dumps(value) + "\n" for value in compact))
            freeze["clips"][clip] = dict(journal=journal, features=features, frames=count)
        (output / "freeze.json").write_text(json.dumps(freeze))
        return repo, output, freeze

    def test_complete_generated_artifacts_bind_every_input_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            receipt = audit.audit(output, repo)
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["frames"], 2741)
            self.assertFalse(receipt["production_changed"])
            self.assertFalse(receipt["promotion_allowed"])
            self.assertTrue(receipt["known_synthetic_counterexample"])
            self.assertEqual(len(receipt["checked_files_sha256"]), 15)

    def test_modified_implementation_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            (output / "implementation/scripts/audit_accuracy_v57.py").write_text("changed")
            with self.assertRaises(ValueError):
                audit.audit(output, repo)

    def test_scope_and_policy_cannot_be_changed_by_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, frozen = self.fixture(directory)
            changes = []
            bad = copy.deepcopy(frozen)
            bad["config"]["required_consecutive_edges"] = 1
            changes.append(bad)
            bad = copy.deepcopy(frozen)
            bad["promotion_allowed"] = True
            changes.append(bad)
            bad = copy.deepcopy(frozen)
            bad["clips"]["0126"]["frames"] = 128
            changes.append(bad)
            for freeze in changes:
                (output / "freeze.json").write_text(json.dumps(freeze))
                with self.subTest(freeze=freeze["config"]), self.assertRaises(ValueError):
                    audit.audit(output, repo)

    def test_summary_must_not_promote_or_hide_known_counterexample(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            healthy = self.summary(output)
            for key, value in (("production_changed", True), ("promotion_allowed", True),
                               ("classifier_promoted", True), ("known_counterexample_rejected", False),
                               ("known_counterexample_blocks_promotion", False),
                               ("freeze_sha256", "0" * 64)):
                (output / "summary.json").write_text(json.dumps(healthy | {key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit.audit(output, repo)
            (output / "summary.json").write_text(json.dumps(healthy))
            self.assertTrue(audit.audit(output, repo)["passed"])

    def test_summary_cannot_misreport_workload_reasons_tiers_or_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            healthy = self.summary(output)
            variants = []
            for field in ("measured", "predicted", "distinct_identities"):
                bad = copy.deepcopy(healthy)
                bad["clips"]["0126"]["workload"]["v57"][field] = 1
                variants.append(bad)
            for field in ("tiers", "reasons"):
                bad = copy.deepcopy(healthy)
                bad["clips"]["0126"][field] = {"fake": 1}
                variants.append(bad)
            variants.append(healthy | dict(frames=2740))
            for bad in variants:
                (output / "summary.json").write_text(json.dumps(bad))
                with self.subTest(bad=bad["frames"]), self.assertRaises(ValueError):
                    audit.audit(output, repo)

    def test_completion_receipt_cannot_claim_unbound_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            receipt = dict(completed=True, production_changed=False, promotion_allowed=False,
                           all_frozen_inputs_and_code_rehashed=True, outputs_sha256={"arbitrary.json": "0" * 64})
            (output / "completion_receipt.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                audit.audit(output, repo)

    def test_stress_cannot_hide_or_promote_known_counterexample(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, output, freeze = self.fixture(directory)
            stress = dict(completed=True, production_changed=False, promotion_allowed=False,
                          known_counterexample_rejected=False,
                          known_point_on_strong_edge_counterexample_reproduced=True, cases=[])
            (output / "stress.json").write_text(json.dumps(stress))
            with self.assertRaises(ValueError):
                audit.audit(output, repo)

    def test_unsafe_metadata_paths_cannot_access_media_or_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("../outside.json", "/tmp/elsewhere.json", "source.avi", "pixels.npz"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    audit._relative_file(root, relative)

    def test_symlink_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            (actual / "input.json").write_text("{}")
            (root / "link").symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ValueError):
                audit._relative_file(root, "link/input.json")


if __name__ == "__main__":
    unittest.main()
