"""Detached auxiliary gap support; never owns or updates a primary tracker.

Primary strong-only forecasts and records are copied inputs. One weak correction
per strong-anchored gap may seed an auxiliary trajectory, whose later predictions
have no measurement, confirmation, association, lifetime, or learning authority.
"""
from copy import deepcopy
import math
import re

import numpy as np

from weak_continuation_information_v1 import enumerate_peaks
from weak_continuation_shadow_v1 import joseph_weak_update, _overlaps_strong

PRIOR_KEYS = frozenset(("identity", "polarity", "track_id", "segment", "frame_index", "timestamp_ns",
    "reference_xy", "innovation_covariance_2x2", "predicted_mean", "predicted_covariance",
    "strong_measurement_covariance", "prior_qualified", "prior_last_strong_timestamp_ns",
    "prior_associated_update_count", "prior_independent_confirmation_hits", "prior_confirmation_timestamp_ns",
    "weak_budget_available", "strong_age_seconds", "query_eligible"))


def require(value, message):
    if not value:
        raise ValueError(message)


def array(value, shape, name, positive=False):
    require(isinstance(value, (list, tuple, np.ndarray)), name + " must be numeric array data")
    require(np.asarray(value).dtype.kind in "iuf", name + " must not contain booleans, strings, or objects")
    result = np.array(value, dtype=np.float64, copy=True)
    require(result.shape == shape and np.isfinite(result).all(), "Invalid " + name)
    if positive:
        require(np.array_equal(result, result.T), name + " must be symmetric")
        try:
            np.linalg.cholesky(result)
        except np.linalg.LinAlgError as exc:
            raise ValueError(name + " must be positive definite") from exc
    return result


def point_source(inverse, point):
    vector = inverse @ np.array([point[0], point[1], 1.], np.float64)
    require(np.isfinite(vector).all() and vector[2] != 0, "Invalid auxiliary source projection")
    result = vector[:2] / vector[2]
    require(np.isfinite(result).all(), "Nonfinite auxiliary source point")
    return result.tolist()


def in_frame(point, shape):
    return 0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]


class AuxiliaryGapSupport:
    def __init__(self, config, fps=10):
        self.config = deepcopy(config)
        cfg = self.config
        require(type(fps) in (int, float) and math.isfinite(fps) and fps > 0, "Finite positive fps required")
        require(cfg.position_gate_px == 45 and cfg.mahalanobis_gate_squared == 25
                and cfg.temporal_threshold_sigma == 4 and cfg.spatial_threshold_sigma == 3
                and cfg.acceleration_sigma_px_s2 == 60 and cfg.coast_seconds == .7,
                "Frozen 45px/d2=25/temporal4/spatial3/process60/coast0.7 required")
        require(math.isfinite(cfg.position_sigma_px) and cfg.position_sigma_px > 0
                and math.isfinite(cfg.tracking_peak_nms_radius_px) and cfg.tracking_peak_nms_radius_px >= 0,
                "Invalid fixed measurement or NMS geometry")
        self._strong_r = np.eye(2, dtype=np.float64) * cfg.position_sigma_px**2
        self.fps = fps
        self._maximum_gap = max(1., 3. / fps)
        self._states = {}
        self._pending = None
        self._last_frame = self._last_timestamp = self._last_segment = None
        self._poisoned = False

    def snapshot(self):
        """Owned JSON-serializable state/budget/config for passive audit only."""
        def detached(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: detached(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [detached(item) for item in value]
            return deepcopy(value)
        names = ("position_gate_px", "mahalanobis_gate_squared", "temporal_threshold_sigma",
                 "spatial_threshold_sigma", "acceleration_sigma_px_s2", "coast_seconds",
                 "position_sigma_px", "tracking_peak_nms_radius_px", "max_active_tracks_per_polarity")
        return dict(schema="seaqr.weak-auxiliary-state.v1", fps=self.fps,
            config={key: deepcopy(getattr(self.config, key)) for key in names},
            maximum_timestamp_gap_s=self._maximum_gap, last_frame_index=self._last_frame,
            last_timestamp_ns=self._last_timestamp, last_segment=self._last_segment, poisoned=self._poisoned,
            pending=detached(self._pending), states=detached(self._states),
            used_strong_gaps={key: state["strong_anchor_timestamp_ns"] for key, state in self._states.items()})

    def _prior(self, value, frame, timestamp, segment):
        require(type(value) is dict and set(value) == PRIOR_KEYS, "Complete detached primary forecast schema required")
        p = deepcopy(value)
        require(type(p["track_id"]) is int and p["track_id"] >= 0 and p["polarity"] in ("bright", "dark")
                and type(p["segment"]) is int and p["segment"] == segment
                and type(p["frame_index"]) is int and p["frame_index"] == frame
                and type(p["timestamp_ns"]) is int and p["timestamp_ns"] == timestamp,
                "Stale or invalid primary forecast owner")
        require(p["identity"] == f"{segment}/{p['polarity']}:{p['track_id']}", "Primary forecast identity differs")
        mean = array(p["predicted_mean"], (4,), "primary mean")
        covariance = array(p["predicted_covariance"], (4, 4), "primary covariance", True)
        strong_r = array(p["strong_measurement_covariance"], (2, 2), "primary measurement covariance", True)
        innovation = array(p["innovation_covariance_2x2"], (2, 2), "primary innovation covariance", True)
        require(np.array_equal(array(p["reference_xy"], (2,), "primary center"), mean[:2])
                and np.array_equal(strong_r, self._strong_r)
                and np.array_equal(innovation, covariance[:2, :2] + strong_r), "Primary forecast geometry inconsistent")
        anchor = p["prior_last_strong_timestamp_ns"]
        confirmation = p["prior_confirmation_timestamp_ns"]
        require(type(anchor) is int and 0 <= anchor < timestamp
                and (confirmation is None or type(confirmation) is int and 0 <= confirmation <= anchor),
                "Invalid strong-only primary timestamps")
        for name in ("prior_associated_update_count", "prior_independent_confirmation_hits"):
            require(type(p[name]) is int and p[name] >= 1, "Invalid primary hit history")
        require(p["prior_independent_confirmation_hits"] <= p["prior_associated_update_count"]
                and type(p["prior_qualified"]) is bool and type(p["query_eligible"]) is bool
                and (not p["prior_qualified"] or confirmation is not None)
                and p["weak_budget_available"] is True, "Primary priors must be strong-only, not auxiliary-budget filtered")
        age = (timestamp-anchor)/1e9
        require(type(p["strong_age_seconds"]) in (int, float) and math.isfinite(p["strong_age_seconds"])
                and p["strong_age_seconds"] == age, "Primary strong age inconsistent")
        eligible = p["prior_qualified"] and confirmation is not None and 0 < age <= self.config.coast_seconds
        require(p["query_eligible"] is eligible, "Primary strong-only eligibility inconsistent")
        for name, value in (("predicted_mean", mean), ("predicted_covariance", covariance),
                            ("strong_measurement_covariance", strong_r), ("innovation_covariance_2x2", innovation)):
            p[name] = value.tolist()
        p["reference_xy"] = mean[:2].tolist()
        return p

    def prepare(self, frame_index, timestamp_ns, segment, primary_forecasts):
        """Store all strong-only gates; return owned first-weak query forecasts.

        No auxiliary propagation occurs here. Every frame must finish with step.
        The returned list is not the internal competitor list and cannot mutate it.
        """
        require(not self._poisoned and self._pending is None, "Auxiliary step pending or instance failed")
        require(type(frame_index) is int and frame_index == (0 if self._last_frame is None else self._last_frame+1),
                "Contiguous auxiliary frames from zero required")
        require(type(timestamp_ns) is int and timestamp_ns >= 0
                and (self._last_timestamp is None or timestamp_ns > self._last_timestamp), "Strict timestamp increase required")
        require(type(segment) is int and segment >= 0 and type(primary_forecasts) in (list, tuple), "Invalid primary forecast collection")
        reset = ("segment_reset" if self._last_segment is not None and segment != self._last_segment else
                 "timestamp_gap_reset" if self._last_timestamp is not None and
                 (timestamp_ns-self._last_timestamp)/1e9 > self._maximum_gap else None)
        priors = [self._prior(p, frame_index, timestamp_ns, segment) for p in primary_forecasts]
        require(len({p["identity"] for p in priors}) == len(priors), "Duplicate primary prior identity")
        require(len(priors) <= 2*self.config.max_active_tracks_per_polarity, "Primary forecast capacity exceeded")
        if reset or frame_index == 0:
            require(not priors, "Reset/cold-start frame must not carry stale primary priors")
        states = {} if reset else self._states
        queries = [deepcopy(p) for p in priors if p["query_eligible"] and
                   (p["identity"] not in states or states[p["identity"]]["strong_anchor_timestamp_ns"] != p["prior_last_strong_timestamp_ns"])]
        self._pending = dict(frame=frame_index, timestamp=timestamp_ns, segment=segment, priors=priors,
                             prepared_query_count=len(queries), reset=reset)
        return queries

    def _records(self, records, pending):
        require(type(records) in (list, tuple), "Detached primary record sequence required")
        result = {}
        priors = {p["identity"]: p for p in pending["priors"]}
        for record in deepcopy(records):
            require(type(record) is dict and type(record.get("track_id")) is str
                    and re.fullmatch(r"(bright|dark):[0-9]+", record["track_id"])
                    and type(record.get("segment")) is int and record["segment"] == pending["segment"]
                    and type(record.get("measured")) is bool and type(record.get("qualified_moving")) is bool,
                    "Invalid primary record")
            identity = f"{pending['segment']}/{record['track_id']}"
            require(identity not in result, "Duplicate primary live identity")
            position = array(record.get("reference_xy"), (2,), "primary output position")
            velocity = array(record.get("velocity_reference_xy_px_s"), (2,), "primary output velocity")
            if record["measured"]:
                array(record.get("measurement_source_xy"), (2,), "actual primary measurement")
            else:
                require(record.get("measurement_source_xy") is None, "Unmeasured primary carries a measurement")
                prior = priors.get(identity)
                if prior is not None:
                    require(np.array_equal(np.concatenate((position, velocity)), np.asarray(prior["predicted_mean"]))
                            and type(record.get("hits")) is int and type(record.get("independent_hits")) is int
                            and record.get("hits") == prior["prior_associated_update_count"]
                            and record.get("independent_hits") == prior["prior_independent_confirmation_hits"]
                            and record.get("confirmation_timestamp_ns") == prior["prior_confirmation_timestamp_ns"],
                            "Unmatched primary record differs from strong-only prior")
            result[identity] = record
        require(len(result) <= 2*self.config.max_active_tracks_per_polarity, "Primary record capacity exceeded")
        return result

    def _propagate(self, state, timestamp):
        dt = (timestamp-state["state_timestamp_ns"])/1e9
        require(0 < dt <= self._maximum_gap, "Invalid auxiliary propagation interval")
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
        process = self.config.acceleration_sigma_px_s2**2 * np.array(
            [[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
             [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]], np.float64)
        before, covariance_before = state["mean"].copy(), state["covariance"].copy()
        mean = transition @ before
        covariance = transition @ covariance_before @ transition.T + process
        covariance = .5*(covariance+covariance.T)
        state.update(mean=array(mean, (4,), "auxiliary prediction"),
                     covariance=array(covariance, (4, 4), "auxiliary prediction covariance", True), state_timestamp_ns=timestamp)
        return before, covariance_before

    def _observe(self, prior, priors, proposals, shape, provider):
        result = dict(applied=False, observations=None, coverage_known=None)
        try:
            capture = provider(deepcopy(prior))
            if capture is None:
                return dict(result, status="missing_capture"), None
            require(type(capture) is dict and set(capture) == {"values", "flags", "metadata"}, "Invalid capture provider schema")
            metadata = deepcopy(capture["metadata"])
            require(type(metadata) is dict and metadata.get("frame") == prior["frame_index"]
                    and type(metadata.get("frame")) is int and metadata.get("segment") == prior["segment"]
                    and type(metadata.get("segment")) is int
                    and metadata.get("rectangle", {}).get("shape_hw") == list(shape), "Capture frame/segment/shape differs")
            require(isinstance(capture["values"], np.ndarray) and isinstance(capture["flags"], np.ndarray), "Detached numeric capture arrays required")
            observed = enumerate_peaks(capture["values"].copy(), capture["flags"].copy(), metadata, deepcopy(prior), deepcopy(priors))
            result.update(observations=observed, coverage_known=observed["coverage_known"])
            if not observed["coverage_known"]:
                return dict(result, status="capture_coverage_unknown"), None
            if observed["original_threshold_peak_count"]:
                return dict(result, status="original_threshold_peak_in_gate"), None
            if observed["observed_peak_count"] != 1:
                return dict(result, status="no_unique_weak_peak"), None
            peak = observed["observed_peaks"][0]
            require(peak["evidence_partition"] == "weak_temporal", "Not a weak observation")
            if peak["competing_prior_identity_gates"]:
                return dict(result, status="competing_prior_identity_gate"), None
            if _overlaps_strong(peak["reference_xy"], proposals, max(2., self.config.tracking_peak_nms_radius_px)):
                return dict(result, status="overlaps_current_strong_evidence"), None
            mean, covariance = joseph_weak_update(np.asarray(prior["predicted_mean"]), np.asarray(prior["predicted_covariance"]),
                                                 peak["reference_xy"], np.asarray(prior["strong_measurement_covariance"]))
            return dict(result, status="weak_auxiliary_correction", applied=True,
                        measurement_reference_xy=deepcopy(peak["reference_xy"]), score=peak["score"]), (mean, covariance)
        except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            return dict(result, applied=False, status="invalid_capture_or_weak_covariance", coverage_known=None, error=str(exc)), None

    def step(self, primary_records, strong_proposals, matrix, shape, capture_provider):
        """Return (auxiliary_records, metrics), separately from untouched primary.

        A malformed structural input poisons this instance and aborts. Missing,
        censored, or invalid capture evidence is reported as unknown/abstention.
        """
        require(not self._poisoned and self._pending is not None, "Prepare required; failed instances cannot resume")
        self._poisoned = True
        pending = self._pending
        require(callable(capture_provider) and type(shape) in (list, tuple) and len(shape) == 2
                and all(type(v) is int and v > 0 for v in shape), "Valid provider and native shape required")
        geometry = array(matrix, (3, 3), "source-to-reference matrix")
        try:
            inverse = np.linalg.inv(geometry)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular source-to-reference matrix") from exc
        require(np.isfinite(inverse).all(), "Nonfinite inverse geometry")
        records = self._records(primary_records, pending)
        require(type(strong_proposals) in (list, tuple), "Detached strong proposal sequence required")
        proposals = deepcopy(list(strong_proposals))
        for p in proposals:
            require(type(p) is dict and p.get("polarity") in ("bright", "dark")
                    and all(type(p.get(k)) in (int, float) and math.isfinite(p[k]) for k in ("x", "y", "score", "response_dn"))
                    and in_frame((p["x"], p["y"]), shape), "Invalid or out-of-frame strong proposal")
            _overlaps_strong((math.inf, math.inf), [p], 2.)  # Validate supplied shape support without using it.
            require(all(in_frame(value, shape) for value in p.get("shape", {}).get("support_reference_xy", [])),
                    "Strong shape support outside native bounds")
        states = deepcopy(self._states)
        drops = []
        if pending["reset"]:
            drops.extend(dict(identity=k, reason=pending["reset"]) for k in states)
            states = {}
        for identity in list(states):
            if identity not in records:
                drops.append(dict(identity=identity, reason="primary_deletion"))
                del states[identity]
        priors = {p["identity"]: p for p in pending["priors"]}
        decisions, outputs, provider_calls = [], [], 0
        for identity, record in records.items():
            prior = priors.get(identity)
            note = dict(identity=identity, applied=False, observations=None, coverage_known=None)
            reason = ("strong_measurement_priority" if record["measured"] else
                      "no_prior_same_segment_track" if prior is None else
                      "strong_age_expired" if prior["strong_age_seconds"] > self.config.coast_seconds else
                      "not_prior_strong_qualified" if not prior["query_eligible"] or not record["qualified_moving"] else None)
            if reason:
                if identity in states:
                    drops.append(dict(identity=identity, reason=reason))
                    del states[identity]
                decisions.append(dict(note, status=reason))
                continue
            if identity in states and states[identity]["strong_anchor_timestamp_ns"] != prior["prior_last_strong_timestamp_ns"]:
                drops.append(dict(identity=identity, reason="primary_strong_anchor_changed"))
                del states[identity]
            state = states.get(identity)
            current_weak = state is None
            if state is not None:
                before, covariance_before = self._propagate(state, pending["timestamp"])
                note.update(status="auxiliary_prediction_from_weak")
            else:
                if not in_frame(prior["reference_xy"], shape):
                    decisions.append(dict(note, status="query_prior_out_of_bounds"))
                    continue
                provider_calls += 1
                decision, posterior = self._observe(prior, pending["priors"], proposals, shape, capture_provider)
                note.update(decision)
                if posterior is None:
                    decisions.append(note)
                    continue
                before = np.asarray(prior["predicted_mean"], dtype=np.float64).copy()
                covariance_before = np.asarray(prior["predicted_covariance"], dtype=np.float64).copy()
                mean, covariance = posterior
                state = dict(mean=mean, covariance=covariance, state_timestamp_ns=pending["timestamp"],
                    strong_anchor_timestamp_ns=prior["prior_last_strong_timestamp_ns"],
                    origin_frame_index=pending["frame"], origin_timestamp_ns=pending["timestamp"],
                    origin_measurement_reference_xy=deepcopy(note["measurement_reference_xy"]),
                    origin_measurement_source_xy=point_source(inverse, note["measurement_reference_xy"]),
                    primary_prior_at_weak=deepcopy(prior))
            source_xy = point_source(inverse, state["mean"][:2])
            if (not in_frame(state["mean"][:2], shape) or not in_frame(source_xy, shape)
                    or current_weak and not in_frame(state["origin_measurement_source_xy"], shape)):
                note.update(status="auxiliary_coordinate_out_of_bounds", applied=False)
                decisions.append(note)
                continue
            states[identity] = state
            output = dict(record_type="auxiliary_gap_support", identity=identity, primary_track_id=record["track_id"],
                segment=pending["segment"], frame_index=pending["frame"], timestamp_ns=pending["timestamp"],
                evidence_type="current_weak_observation" if current_weak else "prediction_from_weak",
                current_weak_observation=current_weak, prediction_from_weak=not current_weak,
                ordinary_measurement=False, qualified_detection=False, physical_identity_verified=False,
                reference_xy=state["mean"][:2].tolist(), source_xy=source_xy,
                velocity_reference_xy_px_s=state["mean"][2:].tolist(),
                mean_before=before.tolist(), covariance_before=covariance_before.tolist(),
                mean_after=state["mean"].tolist(), covariance_after=state["covariance"].tolist(),
                strong_anchor_timestamp_ns=state["strong_anchor_timestamp_ns"],
                origin_frame_index=state["origin_frame_index"], origin_timestamp_ns=state["origin_timestamp_ns"],
                origin_measurement_reference_xy=deepcopy(state["origin_measurement_reference_xy"]),
                origin_measurement_source_xy=deepcopy(state["origin_measurement_source_xy"]),
                current_weak_measurement_reference_xy=deepcopy(note.get("measurement_reference_xy")) if current_weak else None,
                primary_prior_at_weak=deepcopy(state["primary_prior_at_weak"]))
            outputs.append(output)
            decisions.append(note)
        require(len(outputs) <= len(states) <= len(records), "Auxiliary state exceeded live primary population")
        self._states, self._pending = states, None
        self._last_frame, self._last_timestamp, self._last_segment = pending["frame"], pending["timestamp"], pending["segment"]
        self._poisoned = False
        metrics = dict(schema="seaqr.weak-auxiliary.v1", frame_index=pending["frame"],
            prior_count=len(pending["priors"]), prior_strong_eligible_count=sum(p["query_eligible"] for p in pending["priors"]),
            prepared_query_count=pending["prepared_query_count"], actual_provider_calls=provider_calls,
            primary_live_count=len(records), auxiliary_state_count=len(states), auxiliary_record_count=len(outputs),
            current_weak_observation_count=sum(r["current_weak_observation"] for r in outputs),
            prediction_from_weak_count=sum(r["prediction_from_weak"] for r in outputs),
            decisions=decisions, dropped_auxiliary=drops, reset_reason=pending["reset"],
            primary_feedback=False, ordinary_measurement_credit=False, detector_learning_feedback=False,
            physical_identity_established=False, timestamps_are_inputs_not_wall_clock=True)
        return deepcopy(outputs), deepcopy(metrics)
