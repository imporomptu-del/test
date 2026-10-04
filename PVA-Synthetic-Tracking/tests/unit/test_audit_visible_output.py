"""Independent-auditor tests using generated JSON only, never saved media."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("independent_visible_output_audit",
    Path(__file__).resolve().parents[2] / "scripts/audit_visible_output.py")
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


def track(*, measured=True, qualified=True, segment=0, tid="bright:1"):
    return dict(track_id=tid, segment=segment, measured=measured,
        qualified_moving=qualified, source_xy=[90.0, 91.0],
        measurement_source_xy=[10.0, 11.0] if measured else None)


def item(*, measured=True, segment=0, tid="bright:1", age_ns=0, age_frames=0, origin=0):
    return dict(identity=f"fixture/{segment}/{tid}", track_id=tid, segment=segment,
        measured=measured, qualified_moving=True,
        source_xy=[10.0, 11.0] if measured else [90.0, 91.0],
        coordinate_kind="measurement" if measured else "prediction",
        last_measurement_timestamp_ns=origin, last_measurement_age_ns=age_ns,
        last_measurement_age_frames=age_frames,
        physical_class="unknown", airborne_confirmed=False)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.paths = {name: self.root / (name + suffix) for name, suffix in
            (("journal", ".jsonl"), ("channels", ".jsonl"), ("summary", ".json"), ("output", ".json"))}
        timestamps = [0, 100, 250, 500, 600, 900, 1000]
        tracks = [[track(qualified=False)], [track(measured=False)], [track()], [],
                  [track(measured=False)], [track(measured=False, segment=1)], [track(segment=1)]]
        contexts = [[], [item(measured=False, age_ns=100, age_frames=1)],
                    [item(origin=250)], [],
                    [item(measured=False, age_ns=None, age_frames=None, origin=None)],
                    [item(measured=False, segment=1, age_ns=None, age_frames=None, origin=None)],
                    [item(segment=1, origin=1000)]]
        self.rows, self.channels = [], []
        for index, (timestamp, states, context) in enumerate(zip(timestamps, tracks, contexts)):
            segment = int(index >= 5)
            header = dict(frame_index=index, timestamp_ns=timestamp, segment=segment)
            self.rows.append(dict(header, tracks=states))
            self.channels.append(dict(header, schema="seaqr.visible-observation-output.v1",
                stream_id="fixture", track_context=context,
                observation_alerts=[dict(x, kind="qualified_motion_observation")
                                    for x in context if x["measured"]]))
        self.summary = dict(schema="seaqr.visible-output-replay.v1", completed=True, frames=7,
            stream_id="fixture", input_journal=str(self.paths["journal"]),
            detector_or_tracker_rerun=False, qualification_changed=False,
            all_qualified_measured_observations_preserved=True,
            all_qualified_prediction_context_preserved=True, predictions_in_observation_alerts=0,
            counts=dict(track_context_states=5, track_context_frames=5,
                observation_alerts_states=2, observation_alerts_frames=2,
                retained_prediction_states=3, alerts_without_current_observation=0,
                prediction_age_unknown_states=2),
            unique_identity_counts=dict(observation_alerts=2, track_context=2))
        self.save()

    def save(self):
        for name, rows in (("journal", self.rows), ("channels", self.channels)):
            self.paths[name].write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.summary["input_sha256"] = auditor.sha(self.paths["journal"])
        self.summary["channels_sha256"] = auditor.sha(self.paths["channels"])
        self.paths["summary"].write_text(json.dumps(self.summary))

    def run_audit(self):
        return auditor.audit(**self.paths)

    def assert_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit()
        self.assertFalse(self.paths["output"].exists())

    def test_valid_full_journal_and_independent_age_provenance(self):
        result = self.run_audit()
        self.assertTrue(result["passed"])
        self.assertEqual(result["frames"], 7)
        self.assertEqual(result["journal_track_states_checked"], 6)
        self.assertEqual(result["journal_actual_observations_checked"], 3)
        self.assertEqual(result["input_sha256_before"], result["input_sha256_after"])
        self.assertFalse(result["projection_or_replay_imported"])
        self.assertEqual(json.loads(self.paths["output"].read_text()), result)

    def test_output_is_exclusive(self):
        self.paths["output"].write_text("existing\n")
        with self.assertRaises(ValueError):
            self.run_audit()
        self.assertEqual(self.paths["output"].read_text(), "existing\n")

    def test_prediction_cannot_be_an_alert(self):
        self.channels[1]["observation_alerts"] = [dict(self.channels[1]["track_context"][0],
                                                    kind="qualified_motion_observation")]
        self.save()
        self.assert_rejected()

    def test_actual_alert_cannot_use_filtered_coordinate(self):
        self.channels[2]["observation_alerts"][0]["source_xy"] = [90.0, 91.0]
        self.save()
        self.assert_rejected()

    def test_every_qualified_prediction_context_is_required(self):
        self.channels[1]["track_context"] = []
        self.save()
        self.assert_rejected()

    def test_unqualified_measurement_still_supplies_age(self):
        self.channels[1]["track_context"][0]["last_measurement_age_ns"] = None
        self.save()
        self.assert_rejected()

    def test_nonuniform_timestamps_supply_actual_age(self):
        self.channels[1]["track_context"][0]["last_measurement_age_ns"] = 1
        self.save()
        self.assert_rejected()

    def test_absent_identity_cannot_inherit_previous_age(self):
        self.channels[4]["track_context"][0].update(last_measurement_timestamp_ns=250,
            last_measurement_age_ns=350, last_measurement_age_frames=2)
        self.save()
        self.assert_rejected()

    def test_segment_change_cannot_inherit_previous_age(self):
        self.channels[5]["track_context"][0].update(last_measurement_timestamp_ns=250,
            last_measurement_age_ns=650, last_measurement_age_frames=3)
        self.save()
        self.assert_rejected()

    def test_numeric_zero_cannot_replace_false_classification(self):
        self.channels[2]["track_context"][0]["airborne_confirmed"] = 0
        self.save()
        self.assert_rejected()

    def test_false_cannot_replace_zero_summary_count(self):
        self.summary["counts"]["alerts_without_current_observation"] = False
        self.save()
        self.assert_rejected()

    def test_unqualified_invalid_record_is_not_skipped(self):
        self.rows[0]["tracks"][0]["measurement_source_xy"] = None
        self.save()
        self.assert_rejected()

    def test_duplicate_journal_id_is_rejected(self):
        self.rows[0]["tracks"].append(copy.deepcopy(self.rows[0]["tracks"][0]))
        self.save()
        self.assert_rejected()

    def test_stale_summary_hash_rejected_before_report(self):
        with self.paths["journal"].open("a") as stream:
            stream.write("\n")
        self.assert_rejected()

    def test_truncated_journal_rejected_with_updated_hash(self):
        self.rows.pop()
        self.save()
        self.assert_rejected()

    def test_extra_export_record_rejected_with_updated_hash(self):
        self.channels.append(copy.deepcopy(self.channels[-1]))
        self.save()
        self.assert_rejected()

    def test_json_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            auditor.decode('{"frame_index": 0, "frame_index": 1}')

    def test_json_nonfinite_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                auditor.decode('{"value": ' + value + '}')

    def test_boolean_coordinates_rejected(self):
        self.rows[0]["tracks"][0]["measurement_source_xy"] = [True, 11.0]
        self.save()
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
