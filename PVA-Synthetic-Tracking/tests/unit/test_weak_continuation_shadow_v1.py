import copy
from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("weak_continuation_shadow_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/weak_continuation_shadow_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("weak_shadow", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from tiny_target.visible_baseline import VisibleConfig
from weak_continuation_information_v1 import FLOAT_FIELDS, FLAG_FIELDS

SHAPE = (192,192)
IDENTITY = np.eye(3)


def point(x, y=80, polarity="bright"):
    return dict(x=float(x), y=float(y), polarity=polarity, score=10.0, response_dn=10.0)


def capture(forecast, offsets=(8,), score=3, *, truncated=False):
    """Generated analytic fields only; not a source-media fixture."""
    values = np.zeros((*SHAPE,20),np.float32)
    flags = np.zeros((*SHAPE,13),np.uint8)
    v = {name:values[...,i] for i,name in enumerate(FLOAT_FIELDS)}
    f = {name:flags[...,i] for i,name in enumerate(FLAG_FIELDS)}
    for name in ("variance","tile_sigma_float","noise"):
        v[name][:] = 1
    v["image"][:] = 100
    v["temporal_threshold_dn"][:] = 4
    v["spatial_threshold_dn"][:] = 3
    sign = 1 if forecast["polarity"] == "bright" else -1
    cx,cy = forecast["reference_xy"]
    for offset in offsets:
        x,y = round(cx+offset),round(cy)
        v["temporal"][y,x] = sign*score
        v["spatial"][y,x] = sign*4
    v["centered_temporal"][:] = v["temporal"]
    for name in ("support","previous_support","native_eligible","eligible","ready","finite_neighborhood"):
        f[name][:] = 1
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(np.abs(v["temporal"]),2), (5,5))
    maximum = windows.max(axis=(-1,-2))
    v["neighborhood_max_abs"][:] = maximum
    f["raw_absolute_peak"][:] = np.abs(v["temporal"]) >= maximum
    for name, sign in (("positive",1),("negative",-1)):
        v[name+"_signed_temporal"][:] = sign*v["temporal"]
        v[name+"_signed_spatial"][:] = sign*v["spatial"]
        v[name+"_score"][:] = sign*v["temporal"]
        f[name+"_temporal_pass"][:] = v[name+"_signed_temporal"] >= 4
        f[name+"_spatial_pass"][:] = v[name+"_signed_spatial"] >= 3
        f[name+"_candidate"][:] = f[name+"_temporal_pass"] & f[name+"_spatial_pass"] & f["raw_absolute_peak"]
    metadata = dict(float_fields=list(FLOAT_FIELDS), flag_fields=list(FLAG_FIELDS), ready=True,
        temporal_threshold_sigma_float32=4.0, spatial_threshold_sigma_float32=3.0,
        frame=forecast["frame_index"], segment=forecast["segment"],
        rectangle=dict(shape_hw=list(SHAPE), capture_bounds_exclusive_xyxy=[0,0,192,192],
                       tile_bounds_exclusive_xyxy=[60,60,130,130] if truncated else [2,2,190,190]))
    return dict(values=values,flags=flags,metadata=metadata)


def seeded(**changes):
    cfg = VisibleConfig(confirmation_hits=2, minimum_moving_excursion_px=1,
                        learning_exclusion_radius_px=3, **changes)
    shadow = m.WeakContinuationShadow(cfg,10)
    for frame in range(2):
        shadow.prepare(frame,frame*100000000,0)
        shadow.step([point(80+2*frame)],IDENTITY,SHAPE,lambda f:None)
    return shadow


def advance(shadow,frame,proposals=(),provider=capture,timestamp=None,segment=0):
    shadow.prepare(frame, frame*100000000 if timestamp is None else timestamp, segment)
    return shadow.step(list(proposals),IDENTITY,SHAPE,provider)


def primary(records):
    return next(r for r in records if r["track_id"] == "bright:0")


class WeakShadowTests(unittest.TestCase):
    def test_weak_updates_kinematics_and_preserves_strong_only_counters(self):
        shadow = seeded()
        control = copy.deepcopy(shadow)
        records,_ = advance(shadow,2)
        baseline,_ = advance(control,2,provider=lambda f:None)
        result = primary(records)
        self.assertTrue(result["weak_evidence"]["applied"])
        self.assertFalse(result["measured"])
        self.assertIsNone(result["measurement_source_xy"])
        self.assertIsNone(result["measurement_score"])
        for key in ("hits","independent_hits","qualified_moving","confirmation_timestamp_ns","excursion_px","motion_quality"):
            self.assertEqual(result[key],primary(baseline)[key])
        weak = shadow.tracker.managers["bright"]._tracks[0]
        strong = control.tracker.managers["bright"]._tracks[0]
        for key in (field.name for field in fields(weak)):
            if key not in ("mean","covariance"):
                self.assertEqual(getattr(weak,key),getattr(strong,key),key)
        self.assertFalse(np.array_equal(weak.mean,strong.mean))
        self.assertEqual(result["reference_xy"],weak.mean[:2].tolist())
        self.assertEqual(shadow.tracker.learning_centers(300000000,0), [])
        self.assertEqual(shadow.tracker.extents,control.tracker.extents)
        self.assertEqual(shadow.tracker.qualified,control.tracker.qualified)

    def test_frozen_shape_appearance_maturity_quality_controls_unchanged(self):
        cfg = VisibleConfig(confirmation_hits=4, minimum_moving_excursion_px=12,
            pixel_noise_enabled=True, pixel_noise_model="background_residual", spatial_background="median5",
            shape_measurement_mode="mutual_half_height_r8", motion_quality_enabled=True,
            motion_quality_window_hits=8, motion_quality_minimum_hits=5, motion_quality_maximum_rmse_px=3,
            learning_protection_geometry="observed_shape", learning_protection_mode="variance_only",
            tracking_association_cost="gaussian_nll", tracking_association_prior="hit_maturity",
            tracking_association_appearance="log_response_coast", tracking_birth_policy="spatial_fair")
        shadow = m.WeakContinuationShadow(cfg,10)
        for frame in range(5):
            p = dict(point(80+4*frame),shape=dict(support_reference_xy=[[80+4*frame,80]]))
            advance(shadow,frame,[p],lambda f:None)
        control = copy.deepcopy(shadow)
        result,_ = advance(shadow,5)
        baseline,_ = advance(control,5,provider=lambda f:None)
        self.assertTrue(primary(result)["weak_evidence"]["applied"])
        self.assertEqual(primary(result)["motion_quality"],primary(baseline)["motion_quality"])
        self.assertIsNone(primary(result)["learning_shape_reference_xy"])
        self.assertEqual(shadow.tracker.learning_centers(600000000,0),[])
        weak = shadow.tracker.managers["bright"]._tracks[0]
        strong = control.tracker.managers["bright"]._tracks[0]
        for key in ("last_raw_response","last_detector_score_snr","independent_confirmation_hits","last_credited_frame_indices","missed_windows"):
            self.assertEqual(getattr(weak,key),getattr(strong,key))
        self.assertEqual(shadow.tracker.quality["bright:0"].latest,control.tracker.quality["bright:0"].latest)

    def test_optimized_predict_all_matches_prepared_forecasts_exactly(self):
        from tracking_stage_v28 import predict_all
        shadow = seeded()
        advance(shadow,2)
        forecasts = shadow.prepare(3,300000000,0)
        manager = copy.deepcopy(shadow.tracker.managers["bright"])
        predict_all(manager,300000000)
        prior = next(f for f in forecasts if f["polarity"] == "bright" and f["track_id"] == 0)
        np.testing.assert_array_equal(manager._tracks[0].mean,np.asarray(prior["predicted_mean"]))
        np.testing.assert_array_equal(manager._tracks[0].covariance,np.asarray(prior["predicted_covariance"]))

    def test_independent_next_strong_reassociation_uses_weak_posterior(self):
        shadow = seeded()
        control = copy.deepcopy(shadow)
        advance(shadow,2)
        advance(control,2,provider=lambda f:None)
        sf = shadow.prepare(3,300000000,0)[0]
        cf = control.prepare(3,300000000,0)[0]
        self.assertNotEqual(sf["reference_xy"],cf["reference_xy"])
        proposals = [point(*cf["reference_xy"]),point(*sf["reference_xy"])]
        a,_ = shadow.step(proposals,IDENTITY,SHAPE,lambda f:None)
        b,_ = control.step(proposals,IDENTITY,SHAPE,lambda f:None)
        self.assertEqual(primary(a)["measurement_source_xy"],sf["reference_xy"])
        self.assertEqual(primary(b)["measurement_source_xy"],cf["reference_xy"])

    def test_joseph_covariance_frozen_two_times_noise(self):
        mean = np.array([10,20,3,4.],float)
        covariance = np.array([[5,0,1,0],[0,6,0,1],[1,0,4,0],[0,1,0,4.]],float)
        strong_r = np.eye(2)*4
        before = [a.copy() for a in (mean,covariance,strong_r)]
        out_mean,out_cov = m.joseph_weak_update(mean,covariance,[14,22],strong_r)
        gain = covariance[:,:2] @ np.linalg.inv(covariance[:2,:2]+2*strong_r)
        transform = np.eye(4)-gain @ np.eye(4)[:2]
        expected = transform @ covariance @ transform.T+gain @ (2*strong_r) @ gain.T
        np.testing.assert_allclose(out_mean,mean+gain @ np.array([4,2]))
        np.testing.assert_allclose(out_cov,expected)
        for a,b in zip((mean,covariance,strong_r),before):
            np.testing.assert_array_equal(a,b)

    def test_budget_once_per_strong_gap_and_strong_rearms(self):
        shadow = seeded()
        self.assertTrue(primary(advance(shadow,2)[0])["weak_evidence"]["applied"])
        provider = Mock(side_effect=AssertionError("used budget must not query"))
        records,_ = advance(shadow,3,provider=provider)
        self.assertEqual(primary(records)["weak_evidence"]["status"],"weak_budget_used_for_strong_gap")
        provider.assert_not_called()
        forecast = shadow.prepare(4,400000000,0)[0]
        shadow.step([point(*forecast["reference_xy"])],IDENTITY,SHAPE,provider)
        self.assertTrue(primary(advance(shadow,5)[0])["weak_evidence"]["applied"])

    def test_strong_priority_never_queries_weak(self):
        shadow = seeded()
        forecast = shadow.prepare(2,200000000,0)[0]
        provider = Mock(side_effect=AssertionError("strong match must not query"))
        result,_ = shadow.step([point(*forecast["reference_xy"])],IDENTITY,SHAPE,provider)
        self.assertEqual(primary(result)["weak_evidence"]["status"],"strong_measurement_priority")
        provider.assert_not_called()

    def test_no_weak_births_or_revival_and_original_deletion(self):
        empty = m.WeakContinuationShadow(VisibleConfig(),10)
        provider = Mock(side_effect=AssertionError("no birth from weak"))
        self.assertEqual(advance(empty,0,provider=provider)[0],[])
        provider.assert_not_called()
        shadow = seeded(coast_seconds=.1)
        self.assertTrue(primary(advance(shadow,2)[0])["weak_evidence"]["applied"])
        records,metrics = advance(shadow,3,provider=provider)
        self.assertEqual(records,[])
        self.assertEqual(metrics["bright"]["deleted_track_count"],1)
        provider.assert_not_called()
        self.assertEqual(shadow._used_gap,{})

    def test_elapsed_strong_age_guard_before_missed_window_deletion(self):
        shadow = seeded()
        provider = Mock(side_effect=AssertionError("expired age must not query"))
        records,_ = advance(shadow,2,provider=provider,timestamp=900000000)
        self.assertEqual(primary(records)["weak_evidence"]["status"],"strong_age_expired")
        self.assertEqual(shadow.tracker.managers["bright"]._tracks[0].missed_windows,1)
        provider.assert_not_called()

    def test_missing_truncated_ambiguous_blank_and_strong_peak_abstain(self):
        cases = [(lambda f:None,"missing_capture"), (lambda f:capture(f,truncated=True),"capture_coverage_unknown"),
                 (lambda f:capture(f,offsets=(-3,3)),"no_unique_weak_peak"),
                 (lambda f:capture(f,offsets=()),"no_unique_weak_peak"),
                 (lambda f:capture(f,score=4),"original_threshold_peak_in_gate")]
        for provider,status in cases:
            shadow = seeded()
            result,_ = advance(shadow,2,provider=provider)
            self.assertEqual(primary(result)["weak_evidence"]["status"],status)
            self.assertFalse(primary(result)["weak_evidence"]["applied"])

    def test_other_prior_track_gate_competition_vetoes(self):
        shadow = m.WeakContinuationShadow(VisibleConfig(confirmation_hits=2,minimum_moving_excursion_px=1),10)
        for frame in range(2):
            advance(shadow,frame,[point(80+2*frame),point(100+2*frame)],lambda f:None)
        records,_ = advance(shadow,2,provider=lambda f:capture(f,offsets=(0,)))
        self.assertEqual(primary(records)["weak_evidence"]["status"],"competing_prior_identity_gate")

    def test_current_strong_centroid_cross_polarity_veto(self):
        shadow = seeded()
        forecast = shadow.prepare(2,200000000,0)[0]
        x,y = round(forecast["reference_xy"][0]+8),round(forecast["reference_xy"][1])
        records,_ = shadow.step([point(x,y,"dark")],IDENTITY,SHAPE,capture)
        self.assertEqual(primary(records)["weak_evidence"]["status"],"overlaps_current_strong_evidence")

    def test_current_strong_shape_support_veto_away_from_centroid(self):
        shadow = seeded()
        forecast = shadow.prepare(2,200000000,0)[0]
        pixel = [round(forecast["reference_xy"][0]+8),round(forecast["reference_xy"][1])]
        proposal = dict(point(160,160,"dark"),shape=dict(support_reference_xy=[pixel]))
        records,_ = shadow.step([proposal],IDENTITY,SHAPE,capture)
        self.assertEqual(primary(records)["weak_evidence"]["status"],"overlaps_current_strong_evidence")

    def test_input_proposals_arrays_and_prepared_snapshots_not_mutated(self):
        shadow = seeded()
        forecast = shadow.prepare(2,200000000,0)[0]
        data = capture(forecast)
        before = (data["values"].tobytes(),data["flags"].tobytes(),copy.deepcopy(data["metadata"]))
        forecast["predicted_mean"][0] = -999
        data["values"].setflags(write=False)
        data["flags"].setflags(write=False)
        proposals = [point(160,160,"dark")]
        original = copy.deepcopy(proposals)
        shadow.step(proposals,IDENTITY,SHAPE,lambda f:data)
        self.assertEqual(proposals,original)
        self.assertEqual(before,(data["values"].tobytes(),data["flags"].tobytes(),data["metadata"]))

    def test_stale_capture_and_invalid_provider_fail_closed(self):
        def stale(f):
            result = capture(f)
            result["metadata"]["frame"] -= 1
            return result
        for provider in (stale,lambda f:{"bad":True}):
            result,_ = advance(seeded(),2,provider=provider)
            self.assertEqual(primary(result)["weak_evidence"]["status"],"invalid_capture_or_weak_covariance")
            self.assertFalse(primary(result)["weak_evidence"]["applied"])
        with self.assertRaises(ValueError):
            m.joseph_weak_update(np.zeros(4),np.zeros((4,4)),[1,2],np.eye(2))

    def test_segment_reset_and_long_gap_have_no_prior_queries(self):
        for segment,timestamp in ((1,200000000),(0,2000000000)):
            shadow = seeded()
            self.assertEqual(shadow.prepare(2,timestamp,segment),[])
            records,_ = shadow.step([point(85)],IDENTITY,SHAPE,lambda f:None)
            self.assertTrue(all(r["weak_evidence"]["status"] == "strong_measurement_priority" for r in records))

    def test_sequence_guards(self):
        shadow = seeded()
        with self.assertRaisesRegex(ValueError,"contiguous"):
            shadow.prepare(3,300000000,0)
        with self.assertRaisesRegex(ValueError,"increasing"):
            shadow.prepare(2,100000000,0)
        shadow.prepare(2,200000000,0)
        with self.assertRaisesRegex(ValueError,"already prepared"):
            shadow.prepare(2,200000000,0)

    def test_same_true_and_decoy_observables_same_decision_explicit_risk(self):
        real = seeded()
        decoy = copy.deepcopy(real)
        a,_ = advance(real,2)
        b,metrics = advance(decoy,2)
        self.assertEqual(primary(a)["weak_evidence"],primary(b)["weak_evidence"])
        self.assertFalse(primary(a)["weak_evidence"]["physical_identity_verified"])
        self.assertIn("unrelated light",metrics["weak_continuation"]["identity_caveat"])


if __name__ == "__main__":
    unittest.main()
