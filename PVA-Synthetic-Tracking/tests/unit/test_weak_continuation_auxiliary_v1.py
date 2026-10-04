"""Generated source-free ownership, weak evidence, and lifecycle tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("weak_continuation_auxiliary_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2]/"scripts/weak_continuation_auxiliary_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("weak_auxiliary", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from tiny_target.visible_baseline import VisibleConfig
import weak_continuation_shadow_v1 as old
from test_weak_continuation_shadow_v1 import capture, point, SHAPE, IDENTITY


class Harness:
    def __init__(self):
        self.cfg = VisibleConfig(confirmation_hits=2, minimum_moving_excursion_px=1)
        # Test-only strong tracker; no capture ever feeds its legacy weak path.
        self.primary = old.WeakContinuationShadow(self.cfg, 10)
        self.aux = m.AuxiliaryGapSupport(self.cfg, 10)
        self.frame = -1

    def prepare(self, frame, timestamp=None, segment=0):
        ts = frame*100000000 if timestamp is None else timestamp
        self.frame, self.timestamp, self.segment = frame, ts, segment
        self.priors = self.primary.prepare(frame, ts, segment)
        self.queries = self.aux.prepare(frame, ts, segment, self.priors)
        return self.queries

    def finish(self, proposals=(), provider=capture, matrix=IDENTITY, shape=SHAPE):
        self.records, _ = self.primary.step(list(proposals), matrix, shape, lambda _: None)
        inputs = (self.records, proposals, matrix, shape, self.priors)
        before = pickle.dumps(inputs)
        primary_before = pickle.dumps(vars(self.primary.tracker))
        result = self.aux.step(self.records, proposals, matrix, shape, provider)
        assert before == pickle.dumps(inputs), "Auxiliary modified caller inputs"
        assert primary_before == pickle.dumps(vars(self.primary.tracker)), "Auxiliary modified primary private state"
        self.output, self.metrics = result
        return result

    def advance(self, frame, proposals=(), provider=capture, timestamp=None, segment=0, matrix=IDENTITY):
        self.prepare(frame, timestamp, segment)
        return self.finish(proposals, provider, matrix)


def seeded():
    h = Harness()
    h.advance(0, [point(80)])
    h.advance(1, [point(82)])
    return h


def status(metrics, owner="0/bright:0"):
    return next(d for d in metrics["decisions"] if d["identity"] == owner)


class AuxiliaryTests(unittest.TestCase):
    def test_no_tracker_is_created_and_invalid_settings_fail(self):
        cfg = VisibleConfig()
        with patch.object(old, "VisibleTracks", side_effect=AssertionError("No owned primary tracker")):
            a = m.AuxiliaryGapSupport(cfg)
        self.assertEqual(a.snapshot()["states"], {})
        with self.assertRaisesRegex(ValueError, "Frozen"):
            m.AuxiliaryGapSupport(VisibleConfig(acceleration_sigma_px_s2=61))

    def test_weak_observation_changes_only_owned_auxiliary_state(self):
        h = seeded()
        output, metrics = h.advance(2)
        self.assertEqual(len(output), 1)
        r = output[0]
        self.assertTrue(r["current_weak_observation"])
        self.assertFalse(r["prediction_from_weak"])
        for name in ("ordinary_measurement", "qualified_detection", "physical_identity_verified"):
            self.assertFalse(r[name])
        self.assertEqual(metrics["actual_provider_calls"], 1)
        self.assertEqual(r["strong_anchor_timestamp_ns"], 100000000)
        primary = h.primary.tracker.managers["bright"]._tracks[0]
        np.testing.assert_array_equal(primary.mean, r["mean_before"])
        np.testing.assert_array_equal(primary.covariance, r["covariance_before"])
        self.assertFalse(np.array_equal(primary.mean, r["mean_after"]))
        expected = old.joseph_weak_update(primary.mean, primary.covariance,
            r["current_weak_measurement_reference_xy"], np.eye(2)*4)
        np.testing.assert_array_equal(expected[0], r["mean_after"])
        np.testing.assert_array_equal(expected[1], r["covariance_after"])
        self.assertFalse(h.records[0]["measured"])

    def test_prediction_uses_owned_state_no_second_capture_or_gap_extension(self):
        h = seeded(); first, _ = h.advance(2)
        h.prepare(3)
        self.assertEqual(h.queries, [])
        output, metrics = h.finish(provider=lambda _: (_ for _ in ()).throw(AssertionError("second query")))
        r = output[0]
        self.assertEqual(metrics["actual_provider_calls"], 0)
        self.assertTrue(r["prediction_from_weak"])
        self.assertIsNone(r["current_weak_measurement_reference_xy"])
        dt = .1
        transition = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        q = 60**2*np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],
                           [dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]])
        np.testing.assert_array_equal(r["mean_after"], transition@np.asarray(first[0]["mean_after"]))
        expected = transition@np.asarray(first[0]["covariance_after"])@transition.T+q
        np.testing.assert_array_equal(r["covariance_after"], .5*(expected+expected.T))
        self.assertEqual(r["origin_frame_index"], 2)
        self.assertEqual(r["strong_anchor_timestamp_ns"], 100000000)

    def test_strong_priority_drops_state_and_allows_only_new_strong_gap(self):
        h = seeded(); h.advance(2)
        output, metrics = h.advance(3, [point(86)], provider=lambda _: self.fail("strong frame must not query"))
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "strong_measurement_priority")
        self.assertEqual(h.aux.snapshot()["states"], {})
        self.assertEqual(h.aux.snapshot()["used_strong_gaps"], {})
        output, metrics = h.advance(4)
        self.assertEqual(output[0]["strong_anchor_timestamp_ns"], 300000000)
        self.assertEqual(metrics["current_weak_observation_count"], 1)

    def test_primary_deleted_owner_never_prolonged(self):
        h = seeded(); h.advance(2)
        for frame in range(3, 10):
            output, metrics = h.advance(frame, provider=lambda _: self.fail("used gap must not query"))
        self.assertEqual(h.records, [])
        self.assertEqual(output, [])
        self.assertEqual(h.aux.snapshot()["states"], {})
        self.assertIn(dict(identity="0/bright:0", reason="primary_deletion"), metrics["dropped_auxiliary"])

    def test_elapsed_strong_age_expiry_even_when_primary_still_alive(self):
        h = seeded(); h.advance(2)
        output, metrics = h.advance(3, timestamp=900000000, provider=lambda _: self.fail("expired gap query"))
        self.assertEqual(len(h.records), 1)
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "strong_age_expired")
        self.assertEqual(h.aux.snapshot()["states"], {})

    def test_segment_and_large_timestamp_gap_clear_state_and_budget(self):
        for kwargs, reason in ((dict(segment=1), "segment_reset"), (dict(timestamp=1400000000), "timestamp_gap_reset")):
            h = seeded(); h.advance(2)
            output, metrics = h.advance(3, provider=lambda _: self.fail("reset query"), **kwargs)
            self.assertEqual(output, [])
            self.assertEqual(metrics["reset_reason"], reason)
            self.assertEqual(h.aux.snapshot()["used_strong_gaps"], {})

    def test_missing_capture_does_not_establish_absence_or_spend_budget(self):
        h = seeded(); output, metrics = h.advance(2, provider=lambda _: None)
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "missing_capture")
        self.assertIsNone(status(metrics)["coverage_known"])
        self.assertEqual(h.aux.snapshot()["used_strong_gaps"], {})
        output, _ = h.advance(3)
        self.assertEqual(output[0]["evidence_type"], "current_weak_observation")

    def test_coverage_errors_and_invalid_flags_remain_unknown(self):
        h = seeded()
        output, metrics = h.advance(2, provider=lambda p: capture(p, truncated=True))
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "capture_coverage_unknown")
        self.assertFalse(status(metrics)["coverage_known"])
        def bad(p):
            packet = capture(p); packet["flags"][0,0,0] = 2
            return packet
        output, metrics = h.advance(3, provider=bad)
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "invalid_capture_or_weak_covariance")
        self.assertIsNone(status(metrics)["coverage_known"])

    def test_unique_weak_frozen_threshold_and_multiple_peak_abstentions(self):
        for provider, wanted in ((lambda p: capture(p, score=4), "original_threshold_peak_in_gate"),
                                 (lambda p: capture(p, offsets=(-8,8)), "no_unique_weak_peak"),
                                 (lambda p: capture(p, offsets=()), "no_unique_weak_peak")):
            h = seeded(); output, metrics = h.advance(2, provider=provider)
            self.assertEqual(output, [])
            self.assertEqual(status(metrics)["status"], wanted)
            self.assertTrue(status(metrics)["coverage_known"])

    def test_competing_unqualified_prior_is_not_filtered_out(self):
        h = Harness()
        h.advance(0, [point(80)])
        h.advance(1, [point(82), point(115)])
        h.prepare(2)
        self.assertEqual(len(h.priors), 2)
        self.assertEqual(len(h.queries), 1)
        output, metrics = h.finish()
        self.assertEqual(output, [])
        note = status(metrics)
        self.assertEqual(note["status"], "competing_prior_identity_gate")
        self.assertEqual(note["observations"]["observed_peaks"][0]["competing_prior_identity_gates"][0]["identity"], "0/bright:1")

    def test_any_current_strong_support_overlap_is_vetoed(self):
        h = seeded(); h.prepare(2)
        p = h.priors[0]
        xy = [round(p["reference_xy"][0]+8), round(p["reference_xy"][1])]
        output, metrics = h.finish([point(*xy, polarity="dark")])
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "overlaps_current_strong_evidence")

    def test_hostile_provider_and_returned_objects_cannot_alias_inputs_or_internal_state(self):
        h = seeded(); h.prepare(2)
        prepared = pickle.dumps(h.aux.snapshot())
        h.queries[0]["predicted_mean"][0] = -999
        self.assertEqual(prepared, pickle.dumps(h.aux.snapshot()))
        packets = []
        def hostile(p):
            packet = capture(p)
            packets.append((packet, packet["values"].tobytes(), packet["flags"].tobytes()))
            packet["values"].setflags(write=False); packet["flags"].setflags(write=False)
            p["reference_xy"][0] = -1000
            p["predicted_covariance"][0][0] = -1000
            return packet
        output, metrics = h.finish(provider=hostile)
        self.assertEqual(len(output), 1)
        internal = pickle.dumps(h.aux.snapshot())
        output[0]["mean_after"][0] = -1
        metrics["decisions"][0]["observations"]["observed_peaks"].clear()
        snapshot = h.aux.snapshot(); snapshot["states"]["0/bright:0"]["mean"][0] = -1
        self.assertEqual(internal, pickle.dumps(h.aux.snapshot()))
        for packet, values, flags in packets:
            self.assertEqual(packet["values"].tobytes(), values)
            self.assertEqual(packet["flags"].tobytes(), flags)
        json.dumps(h.aux.snapshot(), allow_nan=False)

    def test_priors_are_strong_only_complete_and_consistent(self):
        for change, message in ((lambda p: p.update(weak_budget_available=False), "strong-only"),
                                (lambda p: p.update(frame_index=99), "Stale"),
                                (lambda p: p["innovation_covariance_2x2"][0].__setitem__(0, 999), "inconsistent"),
                                (lambda p: p.update(strong_age_seconds=.99), "age inconsistent")):
            h = seeded()
            p = h.primary.prepare(2,200000000,0)
            change(p[0])
            before = h.aux.snapshot()
            with self.assertRaisesRegex(ValueError, message):
                h.aux.prepare(2,200000000,0,p)
            self.assertEqual(h.aux.snapshot(), before)

    def test_duplicate_identity_stale_records_and_singular_geometry_fail_closed(self):
        h = seeded()
        priors = h.primary.prepare(2,200000000,0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            h.aux.prepare(2,200000000,0,priors+deepcopy(priors))
        for invalid in ("record", "matrix"):
            h = seeded(); h.prepare(2)
            records, _ = h.primary.step([], IDENTITY, SHAPE, lambda _: None)
            if invalid == "record":
                records[0]["reference_xy"][0] += 1
            with self.assertRaises(ValueError):
                h.aux.step(records, [], np.zeros((3,3)) if invalid == "matrix" else IDENTITY, SHAPE, capture)
            self.assertTrue(h.aux.snapshot()["poisoned"])
            with self.assertRaises(ValueError):
                h.aux.prepare(3,300000000,0,[])

    def test_coordinate_censoring_retains_used_budget_and_never_emits_offframe_state(self):
        h = seeded(); h.advance(2)
        h.prepare(3)
        records, _ = h.primary.step([], IDENTITY, SHAPE, lambda _: None)
        # Valid but translated source transform puts the auxiliary source point outside.
        matrix = np.array([[1.,0,1000],[0,1.,0],[0,0,1.]])
        output, metrics = h.aux.step(records, [], matrix, SHAPE, lambda _: self.fail("used gap capture"))
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "auxiliary_coordinate_out_of_bounds")
        self.assertEqual(h.aux.snapshot()["used_strong_gaps"], {"0/bright:0":100000000})

    def test_first_weak_raw_source_outside_frame_abstains_even_if_posterior_inside(self):
        h = seeded(); h.prepare(2)
        prior = h.priors[0]
        measurement = [round(prior["reference_xy"][0]+8), round(prior["reference_xy"][1])]
        posterior, _ = old.joseph_weak_update(np.asarray(prior["predicted_mean"]),
            np.asarray(prior["predicted_covariance"]), measurement, np.eye(2)*4)
        # Put the right image boundary between raw weak evidence and its posterior.
        shift = SHAPE[1] - (posterior[0]+measurement[0])/2
        matrix = np.array([[1.,0,-shift],[0,1.,0],[0,0,1.]])
        self.assertTrue(m.in_frame(m.point_source(np.linalg.inv(matrix), posterior), SHAPE))
        self.assertFalse(m.in_frame(m.point_source(np.linalg.inv(matrix), measurement), SHAPE))
        output, metrics = h.finish(matrix=matrix)
        self.assertEqual(output, [])
        self.assertEqual(status(metrics)["status"], "auxiliary_coordinate_out_of_bounds")
        self.assertFalse(status(metrics)["applied"])
        self.assertEqual(h.aux.snapshot()["states"], {})
        self.assertEqual(h.aux.snapshot()["used_strong_gaps"], {})

    def test_deterministic_output_no_timing_labels_or_unbounded_population(self):
        h1, h2 = seeded(), seeded()
        a, b = h1.advance(2), h2.advance(2)
        self.assertEqual(a, b)
        self.assertEqual(h1.aux.snapshot(), h2.aux.snapshot())
        for forbidden in ("precision", "recall", "airborne", "false_positive", "timings_ms"):
            self.assertNotIn(forbidden, a[1])
        self.assertLessEqual(len(a[0]), len(h1.records))

    def test_protocol_repeated_prepare_and_noncontiguous_frames_fail(self):
        a = m.AuxiliaryGapSupport(VisibleConfig())
        with self.assertRaises(ValueError):
            a.prepare(1,100000000,0,[])
        a.prepare(0,0,0,[])
        with self.assertRaises(ValueError):
            a.prepare(0,0,0,[])
        a.step([], [], IDENTITY, SHAPE, lambda _: self.fail("empty query"))
        with self.assertRaises(ValueError):
            a.prepare(1,0,0,[])


if __name__ == "__main__":
    unittest.main()
