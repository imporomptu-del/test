"""Generated offline V56 parity fixtures; no real media or device access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" if (ROOT / "scripts").is_dir() else Path(__file__).resolve().parent))
import audit_accuracy_v56_replay as audit

COUNT = 4
FRAMES = (1, 2)
DIGEST = "a" * 64
OTHER = "b" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return hashlib.sha256(audit.encoded(value).encode()).hexdigest()


def runtime():
    before = dict(blas=[dict(threads=12, sha256=audit.BLAS_SHA, path="/frozen/blas", config="frozen", core="armv8")],
                  affinity=list(range(12)), numpy="1.26.1", opencv="4.10.0", opencv_threads=12,
                  clock_ticks=100, thread_environment=dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")))
    after = deepcopy(before)
    after["opencv_threads"] = 2
    return dict(before=before, after=after)


def fixture(root):
    for name, content in (("run_accuracy_v56_diagnostic.py", "generated runner"), ("plan.md", "generated plan"),
                          ("probes.json", "{}"), ("build/capture.so", "opaque generated bridge"),
                          ("build/layout.cu", "generated native layout"), ("generated_cuda_smoke.log", "pass")):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (root / "audit_accuracy_v56_replay.py").write_bytes(Path(audit.__file__).read_bytes())
    native = {"layout.cu": audit.file_sha(root / "build/layout.cu")}
    write(root / "build/build.json", dict(passed=True, returncode=0, native_sources_sha256=native))
    frozen = dict(schema="seaqr.accuracy-v56-freeze.v1", pre_run=True,
                  files_sha256={n: audit.file_sha(root / n) for n in ("run_accuracy_v56_diagnostic.py", "audit_accuracy_v56_replay.py", "plan.md", "probes.json")},
                  bridge_sha256=audit.file_sha(root / "build/capture.so"), plan_sha256=audit.file_sha(root / "plan.md"),
                  runner_sha256=audit.file_sha(root / "run_accuracy_v56_diagnostic.py"), auditor_sha256=audit.file_sha(root / "audit_accuracy_v56_replay.py"),
                  runtime_reference=runtime()["before"], native_sources_sha256=native,
                  v29_freeze_sha256=audit.V29_FREEZE_SHA, original_library_sha256=audit.LIBRARY_SHA,
                  build_sha256=audit.file_sha(root / "build/build.json"), generated_cuda_smoke_sha256=audit.file_sha(root / "generated_cuda_smoke.log"))
    write(root / "freeze.json", frozen)
    launch = dict(source=audit.SOURCE, source_sha256=audit.SOURCE_SHA, config_sha256=audit.CONFIG_SHA,
                  motion_config_sha256=audit.MOTION_CONFIG_SHA, fps=10.0, expected_frames=COUNT, max_frames=None,
                  annotations_supplied_to_detector=False, configuration={"policy": "frozen generated"},
                  code_sha256={"visible_baseline.py": DIGEST}, package_sha256={"visible_baseline.py": DIGEST},
                  exact_cuda_stabilization={"library_sha256": audit.LIBRARY_SHA},
                  external_accelerators={"median": {"library_sha256": audit.LIBRARY_SHA}},
                  frame_decode={"execution": "prefetch_one"}, source_probe={"generated": True})
    report = dict(completed=True, full_clip=True, frames=COUNT, source_sha256=audit.SOURCE_SHA,
                  faint_target_synthetic_branch_enabled=False, configuration=launch["configuration"],
                  counts={"candidate_count": 0}, qualified_tracks=[], qualified_track_count=0,
                  availability={"ready": True}, detection_status="ready", elapsed_seconds=2.0, processed_fps=2.0,
                  timings_ms={"detection": 1}, frame_decode=dict(contract=launch["frame_decode"], decoded_frames=COUNT,
                  consumed_frames=COUNT, worker_joined=True, capture_released=True, dropped_frames=0, read_calls=COUNT+1,
                  maximum_observed_frames_ahead=1))
    motion = dict(passed=True, error=None, closed=True, branch="visible", clip="0126", mode="reuse", frames=None,
                  injected=False, processed_frames=COUNT, reuse_hits=COUNT-2, reuse_misses=1,
                  **audit.MOTION_PINS, runtime_sha256={"visible_baseline.py": DIGEST},
                  motion=[dict(frame=i, identity={"segment": 0}, estimator_s=0.01) for i in range(1, COUNT)])
    row = dict(timestamp_ns=0, segment=0, source_to_reference=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               motion={"reset": False, "pva_timings_ms": [1]}, coverage={"detection_ms": 1, "warmup": False},
               candidates=[], tracks=[], tracking_metrics={"resolution_nms": {}}, timings_ms={"decode": 1})
    rows = [dict(deepcopy(row), frame_index=i, timestamp_ns=i*100000000) for i in range(COUNT)]
    captures = []
    for frame in FRAMES:
        npz = f"captures/frame_{frame:06d}.npz"
        meta = f"captures/frame_{frame:06d}.json"
        (root / "captures").mkdir(exist_ok=True)
        (root / npz).write_bytes(b"opaque generated NPZ, never loaded")
        write(root / meta, dict(frame=frame, prelearning=True, full_exposed_state_unchanged=True,
                               original_library_sha256=audit.LIBRARY_SHA, native_state_before={"background": DIGEST},
                               native_state_after={"background": DIGEST}, post_shape=[],
                               **{k: rows[frame][k] for k in ("tracks", "tracking_metrics", "source_to_reference", "segment")}))
        captures.append(dict(frame=frame, npz_path=npz, metadata_path=meta,
                             npz_sha256=audit.file_sha(root / npz), metadata_sha256=audit.file_sha(root / meta)))
    legacy = dict(schema="seaqr.visible-combined-v29.v1", passed=True, error=None, clip="0126", arm="combined",
                  frames=None, state_audit=False, processed_frames=COUNT, source_sha256={"runner.py": DIGEST},
                  freeze_sha256=audit.V29_FREEZE_SHA, config_sha256=audit.CONFIG_SHA, library_sha256=audit.LIBRARY_SHA,
                  geometry_library_sha256=audit.GEOMETRY_SHA, batch_library_sha256=audit.BATCH_SHA,
                  tracking_transformed_sha256=audit.TRANSFORMED_SHA, execution_policy="serial_reference",
                  gpu_front=True, tracking_stage=True, **dict.fromkeys(("raw16_accessed", "defaults_changed", "production_approved",
                  "new_accuracy_validated", "staged_v24_enabled", "native_motion_v25_enabled"), False))
    for arm in ("clean", "probe"):
        write(root / arm / "launch.json", launch)
        write(root / arm / "report.json", report)
        write(root / (arm + ".execution.json"), motion)
        (root / arm / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write(root / (arm + ".v29.json"), dict(legacy, comparison=dict(exact=True, frames=COUNT,
              reference_journal_sha256=DIGEST, journal_sha256=audit.file_sha(root / arm / "frames.jsonl"),
              execution_sha256=audit.file_sha(root / (arm + ".execution.json")))))
        write(root / (arm + ".v56.json"), dict(schema="seaqr.accuracy-v56-replay.v1", passed=True, error=None,
              arm=arm, processed_frames=COUNT, capture_frames=list(FRAMES) if arm == "probe" else [],
              identities={**{k: frozen[k] for k in ("runner_sha256", "bridge_sha256", "plan_sha256")},
                          "freeze_sha256": audit.file_sha(root / "freeze.json")}, frozen_code=frozen["files_sha256"],
              private_state_digests=[dict(frame=i, output=DIGEST, state=DIGEST, learning=DIGEST) for i in range(COUNT)],
              runtime=runtime(), captures=captures if arm == "probe" else []))
    pins = {k: canonical(launch[k]) for k in ("configuration", "package_sha256", "code_sha256", "frame_decode", "source_probe", "exact_cuda_stabilization", "external_accelerators")}
    pins.update(source_sha256=canonical(legacy["source_sha256"]), runtime_sha256=canonical(motion["runtime_sha256"]))
    return pins


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pins = fixture(self.root)
        self.mock_pins = patch.object(audit, "PINNED_FIELDS", self.pins)
        self.mock_pins.start()
        self.addCleanup(self.mock_pins.stop)

    def run_audit(self):
        return audit._audit_pair(self.root, expected_count=COUNT, capture_frames=FRAMES)

    def modify(self, name, change):
        path = self.root / name
        value = read(path)
        change(value)
        write(path, value)
        if name.endswith(".execution.json"):
            arm = name.split(".")[0]
            self.modify(arm + ".v29.json", lambda v: v["comparison"].update(execution_sha256=audit.file_sha(path)))
        if name.startswith("captures/"):
            self.modify("probe.v56.json", lambda v: [r.update(metadata_sha256=audit.file_sha(path)) for r in v["captures"] if r["metadata_path"] == name])

    def modify_journal(self, arm, change):
        path = self.root / arm / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.modify(arm + ".v29.json", lambda v: v["comparison"].update(journal_sha256=audit.file_sha(path)))

    def fails(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.run_audit()

    def test_complete_generated_pair(self):
        result = self.run_audit()
        self.assertTrue(result["passed"])
        self.assertEqual(result["frames"], COUNT)
        self.assertEqual(result["capture_frames"], list(FRAMES))
        self.assertFalse(result["source_media_accessed"])
        for name in ("freeze.json", "probes.json", "probe/frames.jsonl", "captures/frame_000001.npz", "captures/frame_000001.json"):
            self.assertEqual(result["files_sha256"][name], audit.file_sha(self.root / name))

    def test_production_entry_rejects_small_fixture(self):
        with self.assertRaises(ValueError):
            audit.audit_run(self.root)

    def test_only_named_timing_changes_allowed(self):
        self.modify_journal("probe", lambda rows: rows[0].update(timings_ms={"anything": 999}))
        self.modify_journal("probe", lambda rows: rows[0]["motion"].update(pva_timings_ms=[123]))
        self.assertTrue(self.run_audit()["passed"])

    def test_nonlisted_timing_like_field_not_ignored(self):
        self.modify_journal("probe", lambda rows: rows[0].update(latency_ms=123))
        self.fails()

    def test_mutated_candidate(self):
        self.modify_journal("probe", lambda rows: rows[0]["candidates"].append({"x": 1}))
        self.fails()

    def test_mutated_track(self):
        self.modify_journal("probe", lambda rows: rows[0]["tracks"].append({"track_id": "changed"}))
        self.fails()

    def test_mutated_coverage(self):
        self.modify_journal("probe", lambda rows: rows[0]["coverage"].update(warmup=True))
        self.fails()

    def test_float_and_int_not_equivalent(self):
        self.modify_journal("probe", lambda rows: rows[0].update(segment=0.0))
        self.fails()

    def test_bool_frame_not_integer(self):
        self.modify_journal("probe", lambda rows: rows[0].update(frame_index=False))
        self.fails()

    def test_truncated_journal(self):
        self.modify_journal("probe", lambda rows: rows.pop())
        self.fails()

    def test_both_truncated_journals(self):
        for arm in ("clean", "probe"):
            self.modify_journal(arm, lambda rows: rows.pop())
        self.fails()

    def test_extra_journal_row(self):
        self.modify_journal("probe", lambda rows: rows.append(dict(rows[-1], frame_index=COUNT)))
        self.fails()

    def test_duplicate_frame(self):
        self.modify_journal("probe", lambda rows: rows[1].update(frame_index=0))
        self.fails()

    def test_changed_timestamp(self):
        self.modify_journal("probe", lambda rows: rows[0].update(timestamp_ns=1))
        self.fails()

    def test_private_state_changed(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"][0].update(state=OTHER))
        self.fails()

    def test_learning_input_changed(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"][0].update(learning=OTHER))
        self.fails()

    def test_private_state_missing(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"].pop())
        self.fails()

    def test_private_state_bad_hash(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"][0].update(state="bad"))
        self.fails()

    def test_private_state_reordered(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"].reverse())
        self.fails()

    def test_failed_arm(self):
        self.modify("probe.v56.json", lambda v: v.update(passed=False))
        self.fails()

    def test_v29_failed(self):
        self.modify("probe.v29.json", lambda v: v.update(passed=False))
        self.fails()

    def test_v29_hash_not_bound(self):
        self.modify("probe.v29.json", lambda v: v["comparison"].update(journal_sha256=OTHER))
        self.fails()

    def test_missing_capture(self):
        self.modify("probe.v56.json", lambda v: v["captures"].pop())
        self.fails()

    def test_capture_reordered(self):
        self.modify("probe.v56.json", lambda v: v["captures"].reverse())
        self.fails()

    def test_clean_capture_forbidden(self):
        self.modify("clean.v56.json", lambda v: v.update(capture_frames=list(FRAMES)))
        self.fails()

    def test_capture_path_traversal(self):
        self.modify("probe.v56.json", lambda v: v["captures"][0].update(npz_path="../secret.npz"))
        self.fails()

    def test_capture_hash_changed(self):
        (self.root / "captures/frame_000001.npz").write_bytes(b"changed")
        self.fails()

    def test_capture_not_prelearning(self):
        self.modify("captures/frame_000001.json", lambda v: v.update(prelearning=False))
        self.fails()

    def test_capture_mutated_native_state(self):
        self.modify("captures/frame_000001.json", lambda v: v["native_state_after"].update(background=OTHER))
        self.fails()

    def test_capture_postshape_mismatch(self):
        self.modify("captures/frame_000001.json", lambda v: v.update(post_shape=[{"x": 0}]))
        self.fails()

    def test_capture_track_mismatch(self):
        self.modify("captures/frame_000001.json", lambda v: v.update(tracks=[{"id": "other"}]))
        self.fails()

    def test_motion_identity_changed(self):
        self.modify("probe.execution.json", lambda v: v["motion"][0]["identity"].update(segment=1))
        self.fails()

    def test_motion_closed_required(self):
        self.modify("probe.execution.json", lambda v: v.update(closed=False))
        self.fails()

    def test_missing_motion_frame(self):
        self.modify("probe.execution.json", lambda v: v["motion"].pop())
        self.fails()

    def test_motion_reuse_lifecycle(self):
        self.modify("probe.execution.json", lambda v: v.update(reuse_hits=0))
        self.fails()

    def test_equal_configuration_drift_rejected(self):
        for arm in ("clean", "probe"):
            self.modify(arm + "/launch.json", lambda v: v["configuration"].update(policy="different"))
        self.fails()

    def test_launch_extra_field_drift(self):
        self.modify("probe/launch.json", lambda v: v.update(unknown=1))
        self.fails()

    def test_source_identity_changed(self):
        self.modify("probe/launch.json", lambda v: v.update(source_sha256=OTHER))
        self.fails()

    def test_thread_environment_change(self):
        self.modify("probe.v56.json", lambda v: v["runtime"]["before"]["thread_environment"].update(OMP_NUM_THREADS="12"))
        self.fails()

    def test_opencv_post_policy(self):
        self.modify("probe.v56.json", lambda v: v["runtime"]["after"].update(opencv_threads=12))
        self.fails()

    def test_blas_identity_changed(self):
        self.modify("probe.v56.json", lambda v: v["runtime"]["before"]["blas"][0].update(sha256=OTHER))
        self.fails()

    def test_decode_drop(self):
        self.modify("probe/report.json", lambda v: v["frame_decode"].update(dropped_frames=1))
        self.fails()

    def test_decoder_unclosed(self):
        self.modify("probe/report.json", lambda v: v["frame_decode"].update(capture_released=False))
        self.fails()

    def test_report_semantic_change(self):
        self.modify("probe/report.json", lambda v: v["counts"].update(candidate_count=1))
        self.fails()

    def test_report_timing_allowed(self):
        self.modify("probe/report.json", lambda v: v.update(elapsed_seconds=9.0, processed_fps=COUNT/9))
        self.assertTrue(self.run_audit()["passed"])

    def test_bridge_changed(self):
        (self.root / "build/capture.so").write_bytes(b"changed")
        self.fails()

    def test_source_changed(self):
        (self.root / "plan.md").write_text("changed")
        self.fails()

    def test_symlink_journal_refused(self):
        path = self.root / "probe/frames.jsonl"
        path.unlink()
        path.symlink_to(self.root / "clean/frames.jsonl")
        self.fails()

    def test_missing_evidence(self):
        (self.root / "probe/report.json").unlink()
        self.fails()

    def test_duplicate_json_key(self):
        (self.root / "probe.v56.json").write_text('{"passed":true,"passed":true}')
        self.fails()

    def test_json_nan_rejected(self):
        with self.assertRaises(ValueError):
            audit.loads('{"value":NaN}')

    def test_json_infinity_rejected(self):
        with self.assertRaises(ValueError):
            audit.loads('{"value":Infinity}')

    def test_state_bool_frame_rejected(self):
        self.modify("probe.v56.json", lambda v: v["private_state_digests"][0].update(frame=False))
        self.fails()

    def test_freeze_bool_guard(self):
        self.modify("freeze.json", lambda v: v.update(pre_run=1))
        self.fails()


if __name__ == "__main__":
    unittest.main()
