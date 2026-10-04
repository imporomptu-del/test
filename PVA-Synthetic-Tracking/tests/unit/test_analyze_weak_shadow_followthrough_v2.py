"""Generated metadata only; no real outcomes, media, GPU, or native arrays."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().with_name("analyze_weak_shadow_followthrough_v2.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_weak_shadow_followthrough_v2.py"
spec = importlib.util.spec_from_file_location("weak_followthrough_v2", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def record(tid=0, measured=True, xy=None, shadow=False, weak=False, segment=0, status=None, anchor=0):
    result = dict(track_id=f"bright:{tid}", segment=segment, measured=measured, qualified_moving=True,
        measurement_source_xy=(xy if xy is not None else [10., 20.]) if measured else None,
        source_xy=[11., 20.])
    if shadow:
        result["weak_evidence"] = dict(identity=f"{segment}/bright:{tid}", applied=weak,
            is_ordinary_measurement=False, physical_identity_verified=False,
            status=status or ("weak_kinematic_correction" if weak else "strong_measurement_priority" if measured else "missing_capture"))
        if weak:
            result["weak_evidence"].update(strong_anchor_timestamp_ns=anchor,
                measurement_reference_xy=[12., 20.], observations=dict(coverage_known=True, coverage_unknown_reasons=[]))
    return result


def rows():
    baseline, trace = [], []
    for frame in range(8):
        header = dict(frame_index=frame, timestamp_ns=frame * 100000000, segment=0)
        baseline.append(dict(header, tracks=[record(measured=frame != 1)]))
        trace.append(dict(header, capture_scheduled=1 <= frame <= 5,
                          records=[record(shadow=True, measured=frame != 1, weak=frame == 1)]))
    return baseline, trace


class FollowthroughTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.original_load_reader = m.load_reader
        self.reader = self.original_load_reader()
        self.patches = [patch.dict(self.reader.SOURCES, {clip: (8, str(i) * 64)
            for i, clip in enumerate(self.reader.SOURCES, 1)}, clear=True),
            patch.object(self.reader, "SCHEDULED_SLOTS", 15),
            patch.object(m, "load_reader", return_value=self.reader)]
        for p in self.patches:
            p.start()
        plan = dict(schema="seaqr.weak-continuation-shadow.plan.v1", full_causal_replay=True, concurrent_workers=1,
            clips={clip: dict(frames=count, source_sha256=digest, weak_windows_inclusive=[[1, 5]])
                   for clip, (count, digest) in self.reader.SOURCES.items()})
        write(self.root / "plan.json", plan)
        self.plan_sha = m.sha(self.root / "plan.json")
        write(self.root / "freeze.json", dict(schema="seaqr.weak-continuation-shadow.freeze.v1",
            pre_run=True, plan_sha256=self.plan_sha,
            files_sha256={self.reader.PRODUCER_NAME: self.reader.PRODUCER_SHA256}))
        self.freeze_sha = m.sha(self.root / "freeze.json")
        write(self.root / "batch_status.json", dict(schema="seaqr.weak-continuation-shadow.batch.v1",
            passed=True, error=None, concurrent_workers=1, production_changed=False,
            runs=[dict(clip=clip, arm=arm, returncode=0) for clip in self.reader.SOURCES for arm in ("clean", "shadow", "audit")]))
        for clip in self.reader.SOURCES:
            self.save(clip, *rows())

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def save(self, clip, baseline, trace):
        for row in trace:
            notes = [r["weak_evidence"] for r in row["records"]]
            row["metrics"] = dict(weak_continuation=dict(frame_index=row["frame_index"], decisions=notes,
                                                       applied_count=sum(n["applied"] for n in notes)))
        root = self.root / clip
        lines(root / "clean/frames.jsonl", baseline)
        lines(root / "shadow/shadow_trace.jsonl", trace)
        for arm in ("clean", "shadow"):
            write(root / (arm + ".shadow.json"), dict(schema="seaqr.weak-continuation-shadow.run.v1", passed=True,
                error=None, clip=clip, arm=arm, processed_frames=8, expected_frames=8,
                freeze_sha256=self.freeze_sha, plan_sha256=self.plan_sha, source_sha256=self.reader.SOURCES[clip][1],
                production_changed=False, weak_learning_enabled=False,
                trace_sha256=m.sha(root / "shadow/shadow_trace.jsonl") if arm == "shadow" else None,
                diagnostic_cost_ms=dict(capture=1, shadow=2, snapshot_write=1)))
        required = ("clean/frames.jsonl", "shadow/shadow_trace.jsonl", "clean.shadow.json", "shadow.shadow.json")
        write(root / "independent_audit.json", dict(schema="seaqr.weak-continuation-shadow.audit.v1", passed=True,
            clip=clip, frames=8, source_sha256=self.reader.SOURCES[clip][1], freeze_sha256=self.freeze_sha,
            plan_sha256=self.plan_sha, baseline_journal_non_timing_exact=True,
            baseline_output_state_learning_digests_exact=True, native_state_guards_unchanged=True,
            production_changed=False, weak_learning_enabled=False,
            files_sha256={name: m.sha(root / name) for name in required}))

    def analyze(self):
        return m.analyze(self.root, self.freeze_sha, self.plan_sha)

    def event(self):
        return self.analyze()["clips"]["0029"]["events"][0]

    def test_all_events_and_exact_coordinate_matching_independent_of_id(self):
        b, t = rows()
        b[2]["tracks"] = [record(tid=77), record(tid=99, xy=[99., 99.])]
        self.save("0029", b, t)
        result = self.analyze()
        self.assertEqual(result["event_count"], 3)
        self.assertTrue(result["exploratory"])
        self.assertFalse(result["physical_lineage_established"])
        event = result["clips"]["0029"]["events"][0]
        self.assertFalse(event["baseline_at_weak_frame"]["same_native_id_diagnostic"]["measured"])
        terminal = event["terminal"]
        self.assertEqual(terminal["kind"], "next_strong_measurement")
        self.assertEqual(terminal["exact_baseline_match_status"], "one")
        self.assertEqual(terminal["exact_baseline_actual_measurement_matches"][0]["track_id"], "bright:77")
        self.assertFalse(terminal["baseline_same_native_id_diagnostic"]["present"])

    def test_no_nearest_matching_and_no_polarity_collision(self):
        b, t = rows()
        opposite = record(tid=9)
        opposite["track_id"] = "dark:9"
        b[2]["tracks"] = [record(tid=0, xy=[10.0000000001, 20.]), opposite]
        self.save("0029", b, t)
        self.assertEqual(self.event()["terminal"]["exact_baseline_match_status"], "none")

    def test_multiple_exact_matches_remain_explicitly_ambiguous(self):
        b, t = rows()
        b[2]["tracks"] = [record(tid=5), record(tid=6)]
        self.save("0029", b, t)
        terminal = self.event()["terminal"]
        self.assertEqual(terminal["exact_baseline_match_status"], "multiple")
        self.assertEqual(terminal["exact_baseline_actual_measurement_match_count"], 2)

    def test_disappearance_not_bridged_to_later_reused_id(self):
        b, t = rows()
        t[2]["records"] = []
        self.save("0029", b, t)
        terminal = self.event()["terminal"]
        self.assertEqual((terminal["kind"], terminal["frame_index"]), ("disappearance", 2))
        self.assertTrue(terminal["disappearance_cause_unknown"])

    def test_explicit_expiry_distinct_from_unknown_disappearance(self):
        b, t = rows()
        for frame in range(2, 7):
            t[frame]["records"] = [record(shadow=True, measured=False)]
        t[7]["records"] = [record(shadow=True, measured=False, status="strong_age_expired")]
        self.save("0029", b, t)
        terminal = self.event()["terminal"]
        self.assertEqual(terminal["kind"], "explicit_strong_age_expiry")
        self.assertFalse(terminal["terminal_capture_scheduled"])

    def test_segment_reset_preempts_native_id_reuse(self):
        b, t = rows()
        for frame in range(2, 8):
            b[frame]["segment"] = t[frame]["segment"] = 1
            b[frame]["tracks"] = [record(segment=1)]
            t[frame]["records"] = [record(segment=1, shadow=True)]
        self.save("0029", b, t)
        self.assertEqual(self.event()["terminal"]["kind"], "segment_reset")

    def test_end_censoring_and_followthrough_beyond_window(self):
        b, t = rows()
        for frame in range(2, 8):
            t[frame]["records"] = [record(shadow=True, measured=False)]
        self.save("0029", b, t)
        self.assertEqual(self.event()["terminal"]["kind"], "end_of_clip_censored")
        t[7]["records"] = [record(shadow=True)]
        self.save("0029", b, t)
        terminal = self.event()["terminal"]
        self.assertEqual((terminal["kind"], terminal["frame_index"]), ("next_strong_measurement", 7))
        self.assertFalse(terminal["terminal_capture_scheduled"])

    def test_multiple_events_are_never_favorably_subselected(self):
        b, t = rows()
        t[4]["records"] = [record(shadow=True, measured=False, weak=True, anchor=300000000)]
        t[5]["records"] = []
        self.save("0029", b, t)
        events = self.analyze()["clips"]["0029"]["events"]
        self.assertEqual([e["frame_index"] for e in events], [1, 4])
        self.assertEqual([e["terminal"]["kind"] for e in events], ["next_strong_measurement", "disappearance"])

    def test_zero_events_is_valid_not_a_missing_cohort(self):
        for clip in self.reader.SOURCES:
            b, t = rows()
            t[1]["records"] = [record(shadow=True, measured=False)]
            self.save(clip, b, t)
        result = self.analyze()
        self.assertEqual(result["event_count"], 0)
        self.assertEqual(result["terminal_counts"], {})
        self.assertEqual(set(result["clips"]), set(self.reader.SOURCES))

    def test_missing_failed_audit_or_tampered_bound_journal_rejected(self):
        path = self.root / "0055/independent_audit.json"
        original = json.loads(path.read_text())
        write(path, dict(original, passed=False))
        with self.assertRaisesRegex(ValueError, "audit"):
            self.analyze()
        write(path, original)
        path.unlink()
        with self.assertRaises(ValueError):
            self.analyze()
        self.save("0055", *rows())
        path = self.root / "0029/shadow/shadow_trace.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed bound"):
            self.analyze()

    def test_invalid_weak_anchor_or_coordinate_rejected(self):
        for changes in (dict(strong_anchor_timestamp_ns=100000000), dict(measurement_reference_xy=[True, 20.])):
            b, t = rows()
            t[1]["records"][0]["weak_evidence"].update(changes)
            self.save("0029", b, t)
            with self.assertRaises(ValueError):
                self.analyze()

    def test_existing_output_refused_before_evidence_reads(self):
        output = self.root / "followthrough.json"
        m.write_analysis(self.root, self.freeze_sha, self.plan_sha, output)
        with patch.object(m, "analyze", side_effect=AssertionError("should not read")):
            with self.assertRaisesRegex(ValueError, "fresh follow-through"):
                m.write_analysis(self.root, self.freeze_sha, self.plan_sha, output)

    def test_reader_pin_is_enforced(self):
        with patch.object(m, "SUMMARY_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "source pin"):
                self.original_load_reader()

    def test_snapshot_write_can_exceed_already_exclusive_shadow_time(self):
        receipt_path = self.root / "0029/shadow.shadow.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["diagnostic_cost_ms"] = dict(capture=1, shadow=2, snapshot_write=3)
        write(receipt_path, receipt)
        audit_path = self.root / "0029/independent_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["files_sha256"]["shadow.shadow.json"] = m.sha(receipt_path)
        write(audit_path, audit)
        result = self.analyze()
        self.assertEqual(result["event_count"], 3)
        self.assertEqual(result["clips"]["0029"]["events"][0]["terminal"]["kind"], "next_strong_measurement")
        self.assertIn("after first compact receipt inspection", result["design_timing"])


if __name__ == "__main__":
    unittest.main()
