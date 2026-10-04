"""Generated tracking metadata and isolated native CPU builds; no source media."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("replay_weak_divergence_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/replay_weak_divergence_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("weak_divergence_replay", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from tiny_target.visible_baseline import VisibleConfig, VisibleTracks, map_point
from tiny_target.tracking.kalman import KalmanTrackManager
from weak_continuation_shadow_v1 import joseph_weak_update


def fixture():
    cfg = VisibleConfig(confirmation_hits=2, minimum_moving_excursion_px=1,
        tracking_peak_nms_radius_px=2, motion_quality_enabled=True,
        tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity",
        tracking_association_appearance="log_response_coast", tracking_birth_policy="spatial_fair")
    trackers = {a: VisibleTracks(cfg, 10) for a in ("baseline", "shadow")}
    clean, trace = [], []
    for frame in range(9):
        proposals = [] if frame == 5 else [dict(x=float(80+2*frame), y=80., polarity="bright", score=10., response_dn=10.)]
        if frame == 3:
            proposals.append(dict(x=87., y=80., polarity="bright", score=9., response_dn=9.))
        b = dict(frame_index=frame, timestamp_ns=frame*100000000, segment=0,
                 source_to_reference=np.eye(3).tolist(), coverage=dict(full_shape_hw=[192, 192]), candidates=deepcopy(proposals))
        s = dict(frame_index=frame, timestamp_ns=frame*100000000, segment=0, strong_proposals=deepcopy(proposals))
        for arm in ("baseline", "shadow"):
            tracker = trackers[arm]
            records, metrics = tracker.update(deepcopy(proposals), frame, frame*100000000, 0, np.eye(3), (192, 192))
            if arm == "baseline":
                b.update(tracks=deepcopy(records), tracking_metrics=deepcopy(metrics))
                continue
            for record in records:
                record["weak_evidence"] = dict(applied=False)
            if frame == 5:
                track = tracker.managers["bright"]._tracks[0]
                raw = [float(track.mean[0]+2), float(track.mean[1])]
                note = dict(applied=True, identity="0/bright:0", strong_anchor_timestamp_ns=track.last_measurement_timestamp_ns,
                    measurement_reference_xy=raw, mean_before=track.mean.tolist(), covariance_before=track.covariance.tolist())
                mean, covariance = joseph_weak_update(track.mean, track.covariance, raw, tracker.managers["bright"]._measurement_covariance())
                note.update(mean_after=mean.tolist(), covariance_after=covariance.tolist())
                track.mean, track.covariance = mean, covariance
                records[0].update(reference_xy=mean[:2].tolist(), source_xy=map_point(np.eye(3), *mean[:2]),
                    velocity_reference_xy_px_s=mean[2:].tolist(), weak_evidence=note)
            s.update(records=deepcopy(records), metrics=deepcopy(metrics))
        clean.append(b); trace.append(s)
    return clean, trace, dict(configuration=asdict(cfg), fps=10)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from build_tracking_geometry_v20 import build as build_geometry
        from build_tracking_batch_v27 import build as build_batch
        cls.temp = tempfile.TemporaryDirectory(prefix="seaqr_divergence_generated_")
        root = Path(cls.temp.name)
        cls.geometry = build_geometry(root / "scalar")
        build_batch(root / "batch")
        cls.batch = root / "batch/libtracking_batch_v27.so"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, clean=None, shadow=None, reference=None):
        if clean is None:
            clean, shadow, reference = fixture()
        with m.native_backend(self.geometry, m.sha(self.geometry), self.batch, m.sha(self.batch)) as (method, optimized, scalar):
            result = m.replay_rows(clean, shadow, reference, method)
            self.assertGreater(optimized.geometry.calls, 0)
            self.assertEqual(scalar.calls, 0)
        return result

    def test_real_native_v28_replays_generated_weak_and_strong_history(self):
        original = KalmanTrackManager.update
        result = self.execute()
        self.assertIs(KalmanTrackManager.update, original)
        self.assertEqual(len(result), 9)
        self.assertTrue(all(row["arms"][a]["parity_passed"] for row in result for a in ("baseline", "shadow")))
        self.assertEqual(len(result[5]["arms"]["shadow"]["weak_events"]), 1)
        self.assertNotEqual(result[6]["arms"]["baseline"]["managers"]["bright"]["prior"][0]["predicted_mean"],
                            result[6]["arms"]["shadow"]["managers"]["bright"]["prior"][0]["predicted_mean"])
        self.assertEqual(result[3]["resolution_nms"][0]["candidate_index"], 1)
        self.assertEqual(result[3]["candidate_order"]["bright"][0]["original_proposal_index"], 0)
        self.assertTrue(result[4]["arms"]["baseline"]["visible_after"]["quality"]["bright:0"]["latest"]["ready"])

    def test_logged_weak_before_and_after_mismatch_fail_closed(self):
        for field in ("mean_before", "mean_after"):
            b, s, ref = fixture()
            s[5]["records"][0]["weak_evidence"][field][0] += .01
            with self.assertRaisesRegex(m.ParityError, "weak_mean_" + field.split("_")[1]):
                self.execute(b, s, ref)

    def test_discrete_measurement_and_quality_remain_exact(self):
        for key in ("measurement_source_xy", "motion_quality"):
            b, s, ref = fixture()
            if key == "measurement_source_xy":
                b[4]["tracks"][0][key][0] += 1e-9
            else:
                b[4]["tracks"][0][key]["quadratic_fit_rmse_px"] += 1e-9
            with self.assertRaisesRegex(m.ParityError, key):
                self.execute(b, s, ref)

    def test_posterior_tolerance_is_absolute_only_and_declared(self):
        b, s, ref = fixture()
        b[4]["tracks"][0]["reference_xy"][0] += 5e-8
        self.execute(b, s, ref)
        b[4]["tracks"][0]["reference_xy"][0] += 2e-7
        with self.assertRaisesRegex(m.ParityError, "reference_xy"):
            self.execute(b, s, ref)

    def test_reordered_proposals_rejected_before_tracking(self):
        b, s, ref = fixture()
        s[3]["strong_proposals"].reverse()
        with self.assertRaisesRegex(m.ParityError, "proposal_stream"):
            self.execute(b, s, ref)

    def test_prefix_must_start_zero_and_be_contiguous(self):
        b, s, ref = fixture()
        with self.assertRaisesRegex(ValueError, "Contiguous"):
            self.execute(b[1:], s[1:], ref)

    def test_library_pin_is_checked_before_loading(self):
        with self.assertRaisesRegex(ValueError, "library pin"):
            with m.native_backend(self.geometry, "0"*64, self.batch, m.sha(self.batch)):
                self.fail("must not enter")

    def test_changed_code_pin_rejected_and_json_is_strict(self):
        with patch.dict(m.CODE, {"tracking_stage_v28.py": "0"*64}):
            with self.assertRaisesRegex(ValueError, "Changed frozen adapter"):
                m.verify_runtime(dict(package_sha256={}))
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                m.loads(value)

    def test_failure_receipt_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed.json.gz"
            with patch.object(m, "read_inputs", side_effect=ValueError("generated bad binding")):
                with self.assertRaisesRegex(ValueError, "generated bad binding"):
                    m.run(Path(tmp), "0"*64, self.geometry, m.sha(self.geometry), self.batch, m.sha(self.batch), output)
            with gzip.open(output, "rt") as stream:
                report = json.load(stream)
            self.assertFalse(report["passed"])
            self.assertEqual(report["completed_frames"], 0)
            with patch.object(m, "read_inputs", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "Fresh replay"):
                    m.run(Path(tmp), "0"*64, self.geometry, m.sha(self.geometry), self.batch, m.sha(self.batch), output)

    def test_bound_compressed_prefix_and_raw_digest_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            audit = dict(schema="seaqr.weak-continuation-shadow.audit.v1", passed=True, clip="0126", frames=674,
                source_sha256=m.SOURCE_SHA, baseline_journal_non_timing_exact=True,
                baseline_output_state_learning_digests_exact=True, native_state_guards_unchanged=True,
                production_changed=False, weak_learning_enabled=False)
            reference = dict(source_sha256=m.SOURCE_SHA, fps=10, expected_frames=674, config_sha256="1"*64)
            artifacts = {}
            for name, value in (("original_audit.json", audit), ("reference_0126.json", reference),
                                ("freeze.json", {}), ("plan.json", {})):
                path = root / name
                path.write_text(json.dumps(value))
                artifacts[name] = dict(sha256=m.sha(path))
            raw = b"".join(json.dumps(dict(frame_index=i)).encode()+b"\n" for i in range(41))
            for name in ("clean_prefix.jsonl.gz", "shadow_prefix.jsonl.gz"):
                with gzip.open(root / name, "wb") as stream:
                    stream.write(raw)
                artifacts[name] = dict(sha256=m.sha(root/name), raw_bytes=len(raw), frames=41,
                                       raw_prefix_sha256=hashlib.sha256(raw).hexdigest())
            receipt = dict(schema="seaqr.weak-shadow.divergence-prefix.v1", passed=True, error=None,
                clip="0126", source_sha256=m.SOURCE_SHA, frame_start=0, frame_end_inclusive=40, frames=41,
                original_full_frames=674, rows_filtered=False, raw_prefix_lines_preserved=True, artifacts=artifacts,
                audit_sha256=artifacts["original_audit.json"]["sha256"], freeze_sha256=artifacts["freeze.json"]["sha256"],
                plan_sha256=artifacts["plan.json"]["sha256"], configuration_sha256="1"*64)
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(len(m.read_inputs(root, m.sha(path))[2]), 41)
            with self.assertRaisesRegex(ValueError, "Changed bound input"):
                m.read_inputs(root, "0"*64)
            receipt["artifacts"]["clean_prefix.jsonl.gz"]["raw_prefix_sha256"] = "0"*64
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Raw prefix binding"):
                m.read_inputs(root, m.sha(path))


if __name__ == "__main__":
    unittest.main()
