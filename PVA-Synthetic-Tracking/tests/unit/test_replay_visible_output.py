"""Synthetic journal replay tests; no archived media or outcome claims."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replay_visible_output as runner


def frame(index, *, measured=True, qualified=True):
    return dict(
        frame_index=index,
        timestamp_ns=index * 100_000_000,
        segment=0,
        tracks=[dict(
            track_id="bright:7", segment=0,
            measured=measured, qualified_moving=qualified,
            source_xy=[900.0 + index, 901.0],
            measurement_source_xy=[10.0 + index, 11.0] if measured else None,
        )],
    )


def encode(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


class ReplayVisibleOutputTests(unittest.TestCase):
    def fixture(self, directory, rows=None, *, content=None):
        journal = Path(directory) / "frames.jsonl"
        journal.write_bytes(content if content is not None else encode(rows))
        digest = hashlib.sha256(journal.read_bytes()).hexdigest()
        return journal, digest, Path(directory) / "export"

    def assert_failed_without_summary(self, content, expected_frames):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(directory, content=content)
            original = journal.read_bytes()
            with self.assertRaises(ValueError):
                runner.replay(journal, digest, expected_frames, "generated-stream", output)
            self.assertFalse((output / "summary.json").exists())
            self.assertEqual(journal.read_bytes(), original)

    def test_measured_then_prediction_exports_separate_channels_and_bound_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(
                directory, [frame(0), frame(1, measured=False)]
            )
            original = journal.read_bytes()
            summary = runner.replay(journal, digest, 2, "generated-stream", output)
            self.assertEqual(journal.read_bytes(), original)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertIs(summary["completed"], True)
            self.assertEqual(summary["schema"], "seaqr.visible-output-replay.v1")
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["stream_id"], "generated-stream")
            self.assertEqual(summary["input_journal"], str(journal.resolve()))
            self.assertEqual(summary["input_sha256"], digest)
            channels_path = output / "channels.jsonl"
            self.assertEqual(
                summary["channels_sha256"],
                hashlib.sha256(channels_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(summary["code_sha256"]), 2)
            for path, code_digest in summary["code_sha256"].items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), code_digest)
            self.assertIs(summary["detector_or_tracker_rerun"], False)
            self.assertIs(summary["qualification_changed"], False)
            self.assertIs(summary["all_qualified_measured_observations_preserved"], True)
            self.assertIs(summary["all_qualified_prediction_context_preserved"], True)
            self.assertEqual(summary["predictions_in_observation_alerts"], 0)
            self.assertEqual(summary["counts"], dict(
                observation_alerts_states=1, observation_alerts_frames=1,
                track_context_states=2, track_context_frames=2,
                retained_prediction_states=1, alerts_without_current_observation=0,
                prediction_age_unknown_states=0,
            ))
            self.assertEqual(summary["unique_identity_counts"], dict(
                observation_alerts=1, track_context=1,
            ))
            rows = [json.loads(line) for line in channels_path.read_text().splitlines()]
            self.assertEqual([row["frame_index"] for row in rows], [0, 1])
            self.assertEqual(len(rows[0]["observation_alerts"]), 1)
            self.assertEqual(rows[0]["observation_alerts"][0]["source_xy"], [10.0, 11.0])
            self.assertEqual(rows[1]["observation_alerts"], [])
            context = rows[1]["track_context"][0]
            self.assertEqual(context["source_xy"], [901.0, 901.0])
            self.assertEqual(context["coordinate_kind"], "prediction")
            self.assertEqual(context["last_measurement_timestamp_ns"], 0)
            self.assertEqual(context["last_measurement_age_ns"], 100_000_000)
            self.assertEqual(context["last_measurement_age_frames"], 1)
            for row in rows:
                for item in row["observation_alerts"] + row["track_context"]:
                    self.assertEqual(item["physical_class"], "unknown")
                    self.assertIs(item["airborne_confirmed"], False)

    def test_false_qualification_is_valid_and_never_emits(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(
                directory, [frame(0, qualified=False), frame(1, measured=False, qualified=False)]
            )
            summary = runner.replay(journal, digest, 2, "generated-stream", output)
            self.assertIs(summary["completed"], True)
            self.assertEqual(summary["counts"]["observation_alerts_states"], 0)
            self.assertEqual(summary["counts"]["track_context_states"], 0)
            for line in (output / "channels.jsonl").read_text().splitlines():
                row = json.loads(line)
                self.assertEqual(row["observation_alerts"], [])
                self.assertEqual(row["track_context"], [])

    def test_initial_prediction_remains_context_with_unknown_age(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(directory, [frame(0, measured=False)])
            summary = runner.replay(journal, digest, 1, "generated-stream", output)
            self.assertEqual(summary["counts"]["observation_alerts_states"], 0)
            self.assertEqual(summary["counts"]["prediction_age_unknown_states"], 1)
            row = json.loads((output / "channels.jsonl").read_text())
            self.assertIsNone(row["track_context"][0]["last_measurement_age_ns"])

    def test_incomplete_extra_reordered_duplicate_and_nonzero_start_fail(self):
        cases = [
            ([], 1),
            ([frame(0)], 2),
            ([frame(0), frame(1)], 1),
            ([frame(1), frame(0)], 2),
            ([frame(0), frame(2)], 3),
            ([frame(0), frame(0)], 2),
            ([frame(5)], 1),
        ]
        for rows, expected in cases:
            with self.subTest(indices=[row["frame_index"] for row in rows], expected=expected):
                self.assert_failed_without_summary(encode(rows), expected)

    def test_malformed_json_duplicate_keys_and_nonfinite_values_fail(self):
        good = encode([frame(0)])
        malformed = [
            b"not-json\n", b"{\n", b"null\n", b"[]\n", b"true\n", b"\n",
            good + b"\n", good + b"{\n",
            b'{"frame_index":0,"frame_index":1}\n',
        ]
        for value in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
            invalid = json.dumps(frame(0)).replace("[10.0, 11.0]", f"[{value}, 11.0]")
            malformed.append((invalid + "\n").encode("utf-8"))
        nested_duplicate = json.dumps(frame(0)).replace(
            '"measured": true', '"measured": true, "measured": false'
        )
        malformed.append((nested_duplicate + "\n").encode("utf-8"))
        for content in malformed:
            with self.subTest(content=content):
                self.assert_failed_without_summary(content, 2 if content.startswith(good) else 1)

    def test_missing_metadata_and_nonboolean_flags_fail(self):
        bad_rows = []
        for field in ("frame_index", "timestamp_ns", "segment", "tracks"):
            invalid = frame(0)
            del invalid[field]
            bad_rows.append((f"missing-row-{field}", invalid))
        for field in (
            "track_id", "segment", "measured", "qualified_moving", "source_xy",
            "measurement_source_xy",
        ):
            invalid = frame(0)
            del invalid["tracks"][0][field]
            bad_rows.append((f"missing-track-{field}", invalid))
        for field in ("measured", "qualified_moving"):
            for value in (0, 1, "false", "true", None):
                invalid = frame(0)
                invalid["tracks"][0][field] = value
                bad_rows.append((f"{field}={value!r}", invalid))
        for field in ("frame_index", "timestamp_ns", "segment"):
            invalid = frame(0)
            invalid[field] = False
            bad_rows.append((f"row-{field}=False", invalid))
        for name, invalid in bad_rows:
            with self.subTest(case=name):
                self.assert_failed_without_summary(encode([invalid]), 1)

    def test_bad_second_row_never_publishes_completed_summary(self):
        invalid = frame(1, measured=False)
        invalid["tracks"][0]["measurement_source_xy"] = [20.0, 21.0]
        self.assert_failed_without_summary(encode([frame(0), invalid]), 2)

    def test_hash_mismatch_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(directory, [frame(0)])
            original = journal.read_bytes()
            wrong_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.replay(journal, wrong_digest, 1, "generated-stream", output)
            self.assertFalse(output.exists())
            self.assertEqual(journal.read_bytes(), original)

    def test_final_hash_mismatch_never_publishes_completed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(directory, [frame(0)])
            original = journal.read_bytes()
            real_sha = runner.sha
            journal_checks = []

            def changed_hash(path):
                if Path(path).resolve() == journal.resolve():
                    journal_checks.append(path)
                    if len(journal_checks) > 1:
                        return "0" * 64
                return real_sha(path)

            with patch.object(runner, "sha", side_effect=changed_hash):
                with self.assertRaisesRegex(ValueError, "changed input"):
                    runner.replay(journal, digest, 1, "generated-stream", output)
            self.assertGreaterEqual(len(journal_checks), 2)
            self.assertFalse((output / "summary.json").exists())
            self.assertEqual(journal.read_bytes(), original)

    def test_existing_destination_is_never_overwritten(self):
        for destination_kind in ("directory", "file", "journal"):
            with self.subTest(destination_kind=destination_kind), tempfile.TemporaryDirectory() as directory:
                journal, digest, output = self.fixture(directory, [frame(0)])
                original = journal.read_bytes()
                if destination_kind == "directory":
                    output.mkdir()
                    sentinel = output / "summary.json"
                    sentinel.write_bytes(b"original-sentinel\n")
                elif destination_kind == "file":
                    output.write_bytes(b"original-sentinel\n")
                    sentinel = output
                else:
                    output = journal
                    sentinel = journal
                sentinel_original = sentinel.read_bytes()
                with self.assertRaises(FileExistsError):
                    runner.replay(journal, digest, 1, "generated-stream", output)
                self.assertEqual(sentinel.read_bytes(), sentinel_original)
                self.assertEqual(journal.read_bytes(), original)
                if destination_kind == "directory":
                    self.assertEqual(sorted(path.name for path in output.iterdir()), ["summary.json"])

    def test_bad_expected_inventory_metadata_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            journal, digest, output = self.fixture(directory, [frame(0)])
            cases = [(digest, count) for count in (False, True, 0, -1, 1.0, "1", None)]
            cases.extend((value, 1) for value in (None, False, "", "0" * 63, "x" * 64, digest.upper()))
            for expected_digest, expected_frames in cases:
                with self.subTest(digest=expected_digest, frames=expected_frames):
                    with self.assertRaises(ValueError):
                        runner.replay(journal, expected_digest, expected_frames, "generated-stream", output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
