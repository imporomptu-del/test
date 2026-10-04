"""Isolated causal shadow tracking over an unchanged strong-proposal stream.

Weak evidence changes only owned kinematic state, once per strong-anchored gap.
It never becomes ordinary measurement/confirmation/appearance/learning evidence.
This is not a closed-loop detector or a physical-identity guarantee.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from tiny_target.visible_baseline import VisibleTracks, map_point
from weak_continuation_information_v1 import enumerate_peaks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def joseph_weak_update(mean, covariance, measurement, strong_measurement_covariance):
    """Position-only Joseph correction with frozen Rweak=2*Rstrong."""
    mean = np.asarray(mean, np.float64)
    covariance = np.asarray(covariance, np.float64)
    point = np.asarray(measurement, np.float64)
    strong_r = np.asarray(strong_measurement_covariance, np.float64)
    require(mean.shape == (4,) and covariance.shape == (4,4) and point.shape == (2,)
            and strong_r.shape == (2,2), "weak state dimensions invalid")
    require(all(np.isfinite(a).all() for a in (mean, covariance, point, strong_r))
            and np.array_equal(covariance, covariance.T) and np.array_equal(strong_r, strong_r.T), "finite symmetric weak state required")
    try:
        np.linalg.cholesky(covariance)
        np.linalg.cholesky(strong_r)
        weak_r = 2.0*strong_r
        innovation = covariance[:2,:2]+weak_r
        gain = np.linalg.solve(innovation.T, covariance[:,:2].T).T
        transform = np.eye(4)-gain @ np.eye(4)[:2]
        updated_mean = mean+gain @ (point-mean[:2])
        updated_covariance = transform @ covariance @ transform.T+gain @ weak_r @ gain.T
        updated_covariance = 0.5*(updated_covariance+updated_covariance.T)
        np.linalg.cholesky(updated_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("invalid weak covariance") from error
    require(np.isfinite(updated_mean).all() and np.isfinite(updated_covariance).all(), "nonfinite weak correction")
    return updated_mean, updated_covariance


def _overlaps_strong(point, proposals, radius):
    x,y = point
    for proposal in proposals:
        if math.hypot(x-proposal["x"], y-proposal["y"]) <= radius:
            return True
        shape = proposal.get("shape", {})
        require(isinstance(shape, dict), "invalid strong shape")
        support = shape.get("support_reference_xy", [])
        require(isinstance(support, (list, tuple)), "invalid strong shape support")
        for value in support:
            require(isinstance(value, (list, tuple)) and len(value) == 2
                    and all(type(v) in (int,float) and math.isfinite(v) for v in value), "invalid strong support point")
            if x == value[0] and y == value[1]:
                return True
    return False


class WeakContinuationShadow:
    def __init__(self, config, fps):
        require(type(fps) in (int,float) and math.isfinite(fps) and fps > 0, "positive fps required")
        require(config.position_gate_px == 45 and config.mahalanobis_gate_squared == 25
                and config.temporal_threshold_sigma == 4 and config.spatial_threshold_sigma == 3,
                "frozen45px/d2=25/temporal4/spatial3 settings required")
        self.config, self.fps = deepcopy(config), fps
        self.tracker = VisibleTracks(self.config, fps)
        self._pending = None
        self._last_frame = None
        self._last_timestamp = None
        self._used_gap = {}
        self._poisoned = False

    def _key(self, segment, polarity, track_id):
        return f"{segment}/{polarity}:{track_id}"

    def prepare(self, frame_index, timestamp_ns, segment):
        """Freeze own prior forecasts; call before current strong/weak evidence."""
        require(not self._poisoned and self._pending is None, "shadow already prepared or failed")
        require(type(frame_index) is int and frame_index == (0 if self._last_frame is None else self._last_frame+1), "contiguous frames from zero required")
        require(type(timestamp_ns) is int and timestamp_ns >= 0 and (self._last_timestamp is None or timestamp_ns > self._last_timestamp), "strictly increasing timestamp required")
        require(type(segment) is int and segment >= 0, "nonnegative integer segment required")
        forecasts = []
        for polarity, manager in self.tracker.managers.items():
            if manager._segment_index != segment or manager._last_timestamp_ns is None:
                continue
            if (timestamp_ns-manager._last_timestamp_ns)/1e9 > manager.config.maximum_timestamp_gap_s:
                continue
            for tid, track in sorted(manager._tracks.items()):
                mean, covariance = manager._predicted_state(track, timestamp_ns)
                strong_r = manager._measurement_covariance()
                require(strong_r.shape == (2,2), "position-only shadow required")
                innovation = covariance[:2,:2]+strong_r
                require(np.isfinite(mean).all() and np.isfinite(covariance).all()
                        and np.array_equal(covariance, covariance.T), "invalid prior forecast")
                try:
                    np.linalg.cholesky(covariance)
                    np.linalg.cholesky(innovation)
                except np.linalg.LinAlgError as error:
                    raise ValueError("invalid prior covariance") from error
                identity = self._key(segment, polarity, tid)
                qualified = f"{polarity}:{tid}" in self.tracker.qualified
                age = (timestamp_ns-track.last_measurement_timestamp_ns)/1e9
                unused = self._used_gap.get(identity) != track.last_measurement_timestamp_ns
                forecasts.append(dict(identity=identity, polarity=polarity, track_id=tid, segment=segment,
                    frame_index=frame_index, timestamp_ns=timestamp_ns, reference_xy=mean[:2].tolist(),
                    innovation_covariance_2x2=innovation.tolist(), predicted_mean=mean.tolist(), predicted_covariance=covariance.tolist(),
                    strong_measurement_covariance=strong_r.tolist(), prior_qualified=qualified,
                    prior_last_strong_timestamp_ns=track.last_measurement_timestamp_ns,
                    prior_associated_update_count=track.associated_update_count,
                    prior_independent_confirmation_hits=track.independent_confirmation_hits,
                    prior_confirmation_timestamp_ns=track.confirmation_timestamp_ns,
                    weak_budget_available=unused, strong_age_seconds=age,
                    query_eligible=qualified and track.confirmation_timestamp_ns is not None and unused and 0 < age <= self.config.coast_seconds))
        self._pending = dict(frame_index=frame_index, timestamp_ns=timestamp_ns, segment=segment, forecasts=deepcopy(forecasts))
        return deepcopy(forecasts)

    def step(self, proposals, matrix, shape, capture_provider):
        """Own strong reassociation first; weak correction only on unmatched survivors.

        Provider(exact_detached_forecast) returns {values,flags,metadata} or None.
        It must supply a detached pre-learning snapshot; absent data mean abstain.
        """
        require(not self._poisoned and self._pending is not None, "prepare required before step")
        require(callable(capture_provider), "capture provider must be callable")
        require(isinstance(proposals, (list,tuple)), "strong proposal sequence required")
        strong = deepcopy(list(proposals))
        for proposal in strong:
            require(isinstance(proposal, dict) and proposal.get("polarity") in ("bright","dark")
                    and all(type(proposal.get(k)) in (int,float) and math.isfinite(proposal[k]) for k in ("x","y","score","response_dn")), "invalid strong proposal")
        geometry = np.asarray(matrix, np.float64).copy()
        require(geometry.shape == (3,3) and np.isfinite(geometry).all(), "finite geometry required")
        require(isinstance(shape, (tuple,list)) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), "native shape required")
        try:
            inverse = np.linalg.inv(geometry)
        except np.linalg.LinAlgError as error:
            raise ValueError("invalid geometry") from error
        context = self._pending
        frame, timestamp, segment = (context[k] for k in ("frame_index","timestamp_ns","segment"))
        priors = context["forecasts"]
        prior_by_id = {f["identity"]:f for f in priors}
        self._poisoned = True  # An interrupted ordinary strong update cannot be replayed.
        records, metrics = self.tracker.update(strong, frame, timestamp, segment, geometry, shape)
        evidence = []
        live = set()
        for record in records:
            polarity, raw_id = record["track_id"].split(":")
            tid = int(raw_id)
            identity = self._key(segment, polarity, tid)
            live.add(identity)
            manager = self.tracker.managers[polarity]
            track = manager._tracks[tid]
            note = dict(identity=identity, applied=False, status=None, is_ordinary_measurement=False,
                        physical_identity_verified=False, weak_noise_policy="Rweak=2*Rstrong; provisional safety hypothesis, not calibrated")
            prior = prior_by_id.get(identity)
            if record["measured"]:
                self._used_gap.pop(identity, None)
                note["status"] = "strong_measurement_priority"
            elif prior is None:
                note["status"] = "no_prior_same_segment_track"
            elif not prior["prior_qualified"] or prior["prior_confirmation_timestamp_ns"] is None:
                note["status"] = "not_prior_strong_qualified"
            elif (timestamp-track.last_measurement_timestamp_ns)/1e9 > self.config.coast_seconds:
                note["status"] = "strong_age_expired"
            elif not prior["weak_budget_available"]:
                note["status"] = "weak_budget_used_for_strong_gap"
            else:
                require(track.last_measurement_timestamp_ns == prior["prior_last_strong_timestamp_ns"]
                        and np.array_equal(track.mean, np.asarray(prior["predicted_mean"]))
                        and np.array_equal(track.covariance, np.asarray(prior["predicted_covariance"])), "unmatched prediction differs from prepared prior")
                note.update(self._attempt(prior, priors, track, strong, shape, capture_provider))
                if note["applied"]:
                    self._used_gap[identity] = track.last_measurement_timestamp_ns
                    # Only exposed kinematics follow the weak posterior. Actual
                    # measurement/quality/learning fields remain strong-only.
                    record["reference_xy"] = track.mean[:2].tolist()
                    record["source_xy"] = map_point(inverse, *track.mean[:2])
                    record["velocity_reference_xy_px_s"] = track.mean[2:].tolist()
            record["weak_evidence"] = deepcopy(note)
            evidence.append(note)
        self._used_gap = {key:value for key,value in self._used_gap.items() if key in live}
        metrics = deepcopy(metrics)
        metrics["weak_continuation"] = dict(schema="seaqr.weak-continuation-shadow.v1", frame_index=frame,
            prior_track_count=len(priors), prior_query_eligible_count=sum(f["query_eligible"] for f in priors),
            applied_count=sum(e["applied"] for e in evidence), decisions=evidence,
            ordinary_measured_is_strong_only=True, detector_learning_feedback=False,
            strong_associations_recomputed_by_shadow=True, frozen_detector_stream=True,
            identity_caveat="A unique weak point can be an unrelated light; this decision does not identify physical origin")
        self._last_frame, self._last_timestamp = frame, timestamp
        self._pending = None
        self._poisoned = False
        return deepcopy(records), metrics

    def _attempt(self, prior, priors, track, proposals, shape, capture_provider):
        result = dict(applied=False)
        try:
            capture = capture_provider(deepcopy(prior))
            if capture is None:
                return dict(result, status="missing_capture")
            require(isinstance(capture, dict) and set(capture) == {"values","flags","metadata"}, "invalid provider schema")
            require(isinstance(capture["values"], np.ndarray) and isinstance(capture["flags"], np.ndarray), "detached capture arrays required")
            require(capture["metadata"].get("frame") == prior["frame_index"]
                    and capture["metadata"].get("segment") == prior["segment"]
                    and capture["metadata"].get("rectangle", {}).get("shape_hw") == list(shape), "capture frame/segment/shape mismatch")
            observed = enumerate_peaks(capture["values"].copy(), capture["flags"].copy(), deepcopy(capture["metadata"]), deepcopy(prior), deepcopy(priors))
            result["observations"] = observed
            if not observed["coverage_known"]:
                return dict(result, status="capture_coverage_unknown")
            if observed["original_threshold_peak_count"]:
                return dict(result, status="original_threshold_peak_in_gate")
            if observed["observed_peak_count"] != 1:
                return dict(result, status="no_unique_weak_peak")
            peak = observed["observed_peaks"][0]
            require(peak["evidence_partition"] == "weak_temporal", "not a weak observation")
            if peak["competing_prior_identity_gates"]:
                return dict(result, status="competing_prior_identity_gate")
            if _overlaps_strong(peak["reference_xy"], proposals, max(2.0, self.config.tracking_peak_nms_radius_px)):
                return dict(result, status="overlaps_current_strong_evidence")
            mean, covariance = joseph_weak_update(track.mean, track.covariance, peak["reference_xy"], prior["strong_measurement_covariance"])
            result.update(status="weak_kinematic_correction", applied=True,
                measurement_reference_xy=peak["reference_xy"], score=peak["score"],
                mean_before=track.mean.tolist(), covariance_before=track.covariance.tolist(),
                mean_after=mean.tolist(), covariance_after=covariance.tolist(),
                strong_anchor_timestamp_ns=track.last_measurement_timestamp_ns)
            track.mean, track.covariance = mean, covariance
            return result
        except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
            return dict(result, status="invalid_capture_or_weak_covariance", error=str(error))
