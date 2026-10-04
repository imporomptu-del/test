"""Generated-data checks for the observation-only output boundary."""

from copy import deepcopy
import random
import unittest

from tiny_target.visible_output import ObservationOutput


SCHEMA = "seaqr.visible-observation-output.v1"


def track(track_id="bright:7", *, segment=2, measured=True, qualified=True):
    return dict(
        track_id=track_id,
        segment=segment,
        measured=measured,
        qualified_moving=qualified,
        source_xy=[901.5, 902.5],
        measurement_source_xy=[11.25, 12.75] if measured else None,
    )


def row(*tracks, frame_index=7, timestamp_ns=1_000, segment=2):
    return dict(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        segment=segment,
        tracks=list(tracks),
    )


def expected_item(item, frame, stream_id, last_measurement, *, alert=False):
    """Reference projection, independent of policy implementation details."""
    measured = item["measured"]
    last_frame, last_timestamp = last_measurement or (None, None)
    result = dict(
        identity=f"{stream_id}/{frame['segment']}/{item['track_id']}",
        track_id=item["track_id"],
        segment=frame["segment"],
        measured=measured,
        qualified_moving=True,
        source_xy=list(item["measurement_source_xy"] if measured else item["source_xy"]),
        coordinate_kind="measurement" if measured else "prediction",
        last_measurement_timestamp_ns=last_timestamp,
        last_measurement_age_ns=(
            None if last_timestamp is None else frame["timestamp_ns"] - last_timestamp
        ),
        last_measurement_age_frames=(
            None if last_frame is None else frame["frame_index"] - last_frame
        ),
        physical_class="unknown",
        airborne_confirmed=False,
    )
    if alert:
        result["kind"] = "qualified_motion_observation"
    return result


class ObservationOutputTests(unittest.TestCase):
    def assert_age(self, item, timestamp, age_ns, age_frames):
        self.assertEqual(item["last_measurement_timestamp_ns"], timestamp)
        self.assertEqual(item["last_measurement_age_ns"], age_ns)
        self.assertEqual(item["last_measurement_age_frames"], age_frames)

    def test_current_qualification_and_measurement_control_emission(self):
        tracks = [
            track(f"test:{index}", measured=measured, qualified=qualified)
            for index, (measured, qualified) in enumerate(
                [(False, False), (False, True), (True, False), (True, True)]
            )
        ]
        source = row(*tracks)
        source["classification"] = "aircraft"
        for item in tracks:
            item.update(physical_class="aircraft", airborne_confirmed=True)
        output = ObservationOutput("stream-A").update(source)
        self.assertEqual(
            set(output),
            {"schema", "stream_id", "frame_index", "timestamp_ns", "segment",
             "observation_alerts", "track_context"},
        )
        self.assertEqual(output["schema"], SCHEMA)
        self.assertEqual(output["stream_id"], "stream-A")
        self.assertEqual(output["frame_index"], 7)
        self.assertEqual(output["timestamp_ns"], 1_000)
        self.assertEqual(output["segment"], 2)
        self.assertEqual(
            output["observation_alerts"],
            [expected_item(tracks[3], source, "stream-A", (7, 1_000), alert=True)],
        )
        self.assertEqual(
            {item["track_id"]: item for item in output["track_context"]},
            {
                "test:1": expected_item(tracks[1], source, "stream-A", None),
                "test:3": expected_item(tracks[3], source, "stream-A", (7, 1_000)),
            },
        )

    def test_generated_timeline_matches_independent_history_projection(self):
        rng = random.Random(83_521)
        policy = ObservationOutput("generated-stream")
        history = {}
        timestamp = 0
        segment = 0
        previous_segment = None
        ids = ("bright:7", "bright:8", "dark:7", "local:0")
        for offset in range(120):
            if offset in (40, 80):
                segment += 3
            timestamp += rng.randint(1, 1_000_000)
            tracks = []
            for track_id in ids:
                if rng.random() < 0.25:
                    continue
                item = track(
                    track_id,
                    segment=segment,
                    measured=rng.random() < 0.6,
                    qualified=rng.random() < 0.55,
                )
                item["source_xy"] = [rng.uniform(-500, 500), rng.uniform(-500, 500)]
                if item["measured"]:
                    item["measurement_source_xy"] = [
                        rng.uniform(-50, 50), rng.uniform(-50, 50)
                    ]
                tracks.append(item)
            rng.shuffle(tracks)
            source = row(
                *tracks, frame_index=19 + offset, timestamp_ns=timestamp, segment=segment
            )
            if previous_segment != segment:
                history = {}
            present = {item["track_id"] for item in tracks}
            history = {key: value for key, value in history.items() if key in present}
            for item in tracks:
                if item["measured"]:
                    history[item["track_id"]] = (source["frame_index"], timestamp)
            before = deepcopy(source)
            output = policy.update(source)
            with self.subTest(frame=source["frame_index"], segment=segment):
                self.assertEqual(source, before)
                self.assertEqual(output["schema"], SCHEMA)
                self.assertEqual(output["frame_index"], source["frame_index"])
                self.assertEqual(output["timestamp_ns"], timestamp)
                self.assertEqual(output["segment"], segment)
                for key, require_measurement in (
                    ("track_context", False), ("observation_alerts", True)
                ):
                    expected = {
                        item["track_id"]: expected_item(
                            item, source, "generated-stream", history.get(item["track_id"]),
                            alert=require_measurement,
                        )
                        for item in tracks
                        if item["qualified_moving"]
                        and (item["measured"] or not require_measurement)
                    }
                    self.assertIsInstance(output[key], list)
                    self.assertEqual(len(output[key]), len(expected))
                    self.assertEqual({item["track_id"]: item for item in output[key]}, expected)
            previous_segment = segment

    def test_unqualified_measurement_updates_history_without_emission(self):
        policy = ObservationOutput("stream-A")
        initial = policy.update(row(track(qualified=False)))
        self.assertEqual(initial["observation_alerts"], [])
        self.assertEqual(initial["track_context"], [])
        coast = policy.update(row(track(measured=False), frame_index=8, timestamp_ns=1_300))
        self.assertEqual(coast["observation_alerts"], [])
        self.assert_age(coast["track_context"][0], 1_000, 300, 1)
        hidden = policy.update(
            row(track(measured=False, qualified=False), frame_index=9, timestamp_ns=1_900)
        )
        self.assertEqual(hidden["track_context"], [])
        resumed = policy.update(row(track(measured=False), frame_index=10, timestamp_ns=2_100))
        self.assert_age(resumed["track_context"][0], 1_000, 1_100, 3)

    def test_first_prediction_has_unknown_measurement_age(self):
        output = ObservationOutput("stream-A").update(row(track(measured=False)))
        self.assertEqual(output["observation_alerts"], [])
        self.assert_age(output["track_context"][0], None, None, None)
        self.assertEqual(output["track_context"][0]["coordinate_kind"], "prediction")

    def test_measurement_refreshes_each_timestamp_and_is_not_new_object_event(self):
        policy = ObservationOutput("stream-A")
        identities = []
        for offset in range(12):
            timestamp = 1_000 + offset * 17
            output = policy.update(row(track(), frame_index=7 + offset, timestamp_ns=timestamp))
            self.assertEqual(len(output["observation_alerts"]), 1)
            alert = output["observation_alerts"][0]
            self.assertEqual(alert["kind"], "qualified_motion_observation")
            self.assert_age(alert, timestamp, 0, 0)
            identities.append(alert["identity"])
        self.assertEqual(set(identities), {"stream-A/2/bright:7"})

    def test_absence_drops_only_the_absent_track_history(self):
        policy = ObservationOutput("stream-A")
        policy.update(row(track(), track("dark:7")))
        policy.update(row(track("dark:7", measured=False), frame_index=8, timestamp_ns=1_300))
        output = policy.update(
            row(track(measured=False), track("dark:7", measured=False),
                frame_index=9, timestamp_ns=1_900)
        )
        by_id = {item["track_id"]: item for item in output["track_context"]}
        self.assert_age(by_id["bright:7"], None, None, None)
        self.assert_age(by_id["dark:7"], 1_000, 900, 2)

    def test_empty_frame_drops_history_and_advances_frame_sequence(self):
        policy = ObservationOutput("stream-A")
        policy.update(row(track()))
        output = policy.update(row(frame_index=8, timestamp_ns=1_300))
        self.assertEqual(output["track_context"], [])
        self.assertEqual(output["observation_alerts"], [])
        output = policy.update(row(track(measured=False), frame_index=9, timestamp_ns=1_400))
        self.assert_age(output["track_context"][0], None, None, None)

    def test_segment_change_clears_history_even_for_reused_local_id(self):
        policy = ObservationOutput("stream-A")
        first = policy.update(row(track()))
        output = policy.update(
            row(track(segment=8, measured=False), frame_index=8, timestamp_ns=1_300, segment=8)
        )
        self.assert_age(output["track_context"][0], None, None, None)
        self.assertEqual(first["track_context"][0]["identity"], "stream-A/2/bright:7")
        self.assertEqual(output["track_context"][0]["identity"], "stream-A/8/bright:7")

    def test_stream_instances_never_share_identity_or_measurement_history(self):
        first = ObservationOutput("stream-A").update(row(track()))
        other = ObservationOutput("stream-B").update(row(track(measured=False)))
        self.assertNotEqual(
            first["track_context"][0]["identity"], other["track_context"][0]["identity"]
        )
        self.assert_age(other["track_context"][0], None, None, None)

    def test_stream_id_may_include_slashes_without_ambiguous_identity_suffix(self):
        output = ObservationOutput("camera/group/stream-A").update(row(track()))
        identity = output["track_context"][0]["identity"]
        self.assertEqual(identity.rsplit("/", 2), ["camera/group/stream-A", "2", "bright:7"])

    def test_input_and_output_mutation_do_not_change_history_or_later_calls(self):
        policy = ObservationOutput("stream-A")
        source = row(track())
        before = deepcopy(source)
        output = policy.update(source)
        self.assertEqual(source, before)
        source["tracks"][0]["measurement_source_xy"][0] = -1_000
        source["tracks"][0]["source_xy"][0] = -2_000
        source["tracks"][0]["track_id"] = "changed"
        source["timestamp_ns"] = -1
        self.assertEqual(output["observation_alerts"][0]["source_xy"], [11.25, 12.75])
        self.assertEqual(output["track_context"][0]["source_xy"], [11.25, 12.75])
        for item in output["observation_alerts"] + output["track_context"]:
            item["source_xy"][0] = -3_000
            item["identity"] = "changed"
            item["last_measurement_timestamp_ns"] = -10
        output["frame_index"] = 100
        output["segment"] = 100
        next_output = policy.update(
            row(track(measured=False), frame_index=8, timestamp_ns=1_100)
        )
        self.assert_age(next_output["track_context"][0], 1_000, 100, 1)
        self.assertEqual(next_output["track_context"][0]["identity"], "stream-A/2/bright:7")
        self.assertEqual(next_output["track_context"][0]["source_xy"], [901.5, 902.5])
        snapshot = deepcopy(next_output)
        policy.update(row(track(), frame_index=9, timestamp_ns=1_200))
        self.assertEqual(next_output, snapshot)

    def test_initial_frame_and_timestamp_can_be_zero_or_start_later(self):
        for index, timestamp in ((0, 0), (1_000_000, 9_000_000_000)):
            with self.subTest(index=index, timestamp=timestamp):
                output = ObservationOutput("stream-A").update(
                    row(track(segment=0), frame_index=index, timestamp_ns=timestamp, segment=0)
                )
                self.assert_age(output["track_context"][0], timestamp, 0, 0)

    def test_invalid_stream_id_rejected(self):
        for invalid in ("", " \t\n", None, True, 0, [], {}):
            with self.subTest(stream_id=invalid):
                with self.assertRaises(ValueError):
                    ObservationOutput(invalid)

    def assert_rejected_without_state_change(self, invalid):
        policy = ObservationOutput("stream-A")
        policy.update(row(track()))
        with self.assertRaises(ValueError):
            policy.update(invalid)
        repaired = policy.update(row(track(measured=False), frame_index=8, timestamp_ns=1_100))
        self.assert_age(repaired["track_context"][0], 1_000, 100, 1)

    def test_row_validation_is_atomic(self):
        valid = row(track(), frame_index=8, timestamp_ns=1_100)
        invalid_rows = [("row-type", value) for value in (None, [], "frame", 1)]
        for field in ("frame_index", "timestamp_ns", "segment", "tracks"):
            invalid = deepcopy(valid)
            del invalid[field]
            invalid_rows.append((f"missing-{field}", invalid))
        bad_fields = {
            "frame_index": [True, False, -1, 8.0, "8", None, float("nan"), 7, 9],
            "timestamp_ns": [True, False, -1, 1_100.0, "1100", None, float("inf"), 1_000, 900],
            "segment": [True, False, -1, 2.0, "2", None, float("nan"), 1],
            "tracks": [None, "tracks", {}, [None], [1]],
        }
        for field, values in bad_fields.items():
            for value in values:
                invalid = deepcopy(valid)
                invalid[field] = value
                invalid_rows.append((f"{field}={value!r}", invalid))
        for name, invalid in invalid_rows:
            with self.subTest(case=name):
                self.assert_rejected_without_state_change(invalid)

    def test_all_tracks_validate_before_any_measurement_history_is_committed(self):
        valid = row(track(), track("dark:9"), frame_index=8, timestamp_ns=1_100)
        invalid_rows = []
        for field in (
            "track_id", "segment", "measured", "qualified_moving", "source_xy",
            "measurement_source_xy",
        ):
            invalid = deepcopy(valid)
            del invalid["tracks"][1][field]
            invalid_rows.append((f"missing-{field}", invalid))
        bad_fields = {
            "track_id": ["", " \t\n", "bad/id", None, True, 5, []],
            "segment": [True, False, -1, 2.0, "2", None, 3],
            "measured": [0, 1, None, "true"],
            "qualified_moving": [0, 1, None, "true"],
        }
        for field, values in bad_fields.items():
            for value in values:
                invalid = deepcopy(valid)
                invalid["tracks"][1][field] = value
                invalid_rows.append((f"{field}={value!r}", invalid))
        duplicate = deepcopy(valid)
        duplicate["tracks"][1] = deepcopy(duplicate["tracks"][0])
        invalid_rows.append(("duplicate-track-id", duplicate))
        for name, invalid in invalid_rows:
            with self.subTest(case=name):
                self.assert_rejected_without_state_change(invalid)

    def test_coordinate_validation_applies_to_hidden_and_predicted_tracks_too(self):
        invalid_coordinates = [
            None, [], [1], [1, 2, 3], "12", {"x": 1, "y": 2},
            [True, 2], [1, False], ["1", 2], [None, 2],
            [float("nan"), 2], [1, float("inf")], [float("-inf"), 2],
        ]
        for measured in (False, True):
            for qualified in (False, True):
                for field in ("source_xy", "measurement_source_xy"):
                    if field == "measurement_source_xy" and not measured:
                        values = [[1, 2], [], False]
                    else:
                        values = invalid_coordinates
                    for coordinates in values:
                        invalid = row(
                            track(), track("dark:9", measured=measured, qualified=qualified),
                            frame_index=8, timestamp_ns=1_100,
                        )
                        invalid["tracks"][1][field] = coordinates
                        with self.subTest(
                            measured=measured, qualified=qualified, field=field,
                            coordinates=coordinates,
                        ):
                            self.assert_rejected_without_state_change(invalid)

    def test_rejected_segment_change_does_not_clear_previous_history(self):
        invalid = row(
            track(segment=3), track(segment=3),
            frame_index=8, timestamp_ns=1_100, segment=3,
        )
        self.assert_rejected_without_state_change(invalid)

    def test_rejected_first_row_does_not_establish_frame_or_segment(self):
        policy = ObservationOutput("stream-A")
        invalid = row(track())
        invalid["tracks"][0]["measurement_source_xy"] = [float("nan"), 1]
        with self.assertRaises(ValueError):
            policy.update(invalid)
        output = policy.update(row(track(segment=0), frame_index=0, timestamp_ns=0, segment=0))
        self.assert_age(output["track_context"][0], 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
