"""Nonpromoted causal persistence shadow for existing qualified output.

This development heuristic is not a physical-class classifier or evidence of
target absence. Two consecutive edge-preferred measurements can reject a real
point on an edge; that known counterexample blocks production promotion.

The adapter never creates a measurement, changes qualification, associates
tracks, or supplies learning feedback. Current pixel features are supplied only
for actual baseline-qualified measurements. Predictions can inherit a bounded
measured verdict but cannot refresh evidence or extend an edge run.
"""
import copy
from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class PersistenceConfig:
    required_consecutive_edges: int = 2
    maximum_edge_gap_ns: int = 200_000_000
    maximum_coast_frames: int = 7
    maximum_coast_ns: int = 700_000_000

    def __post_init__(self):
        for name in ("required_consecutive_edges", "maximum_edge_gap_ns",
                     "maximum_coast_frames", "maximum_coast_ns"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer persistence budgets required")
        if self.required_consecutive_edges < 2:
            raise ValueError("Persistence requires at least two measurements")


@dataclass(frozen=True)
class _Verdict:
    accepted: bool
    reason: str
    tier: str
    frame: int
    timestamp: int


@dataclass(frozen=True)
class _History:
    verdict: _Verdict
    edge_count: int = 0
    edge_start_frame: int | None = None
    edge_start_timestamp: int | None = None


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _position(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_finite(component) for component in value))


class CausalEdgePersistence:
    """Replay one contiguous baseline stream without modifying its records."""

    def __init__(self, config=PersistenceConfig()):
        if not isinstance(config, PersistenceConfig):
            raise ValueError("Validated PersistenceConfig required")
        self.config = config
        self._frame = -1
        self._timestamp = None
        self._segment = None
        self._state = {}

    def _validate(self, row, evidence):
        if not isinstance(row, dict):
            raise ValueError("Frame record must be a dictionary")
        frame, timestamp, segment = (row.get(name) for name in
                                     ("frame_index", "timestamp_ns", "segment"))
        if (not _integer(frame) or frame != self._frame + 1
                or not _integer(timestamp)
                or self._timestamp is not None and timestamp <= self._timestamp
                or not _integer(segment)
                or not isinstance(row.get("motion"), dict)
                or type(row["motion"].get("reset")) is not bool
                or not isinstance(row.get("tracks"), list)):
            raise ValueError("Contiguous frames, increasing time and explicit segment/reset required")
        identities, measured_keys = set(), set()
        for track in row["tracks"]:
            if not isinstance(track, dict):
                raise ValueError("Track record must be a dictionary")
            identity = track.get("track_id")
            components = identity.split(":", 1) if isinstance(identity, str) else []
            if (len(components) != 2 or components[0] not in ("bright", "dark")
                    or not components[1] or not _integer(track.get("segment"))
                    or track["segment"] != segment
                    or (segment, identity) in identities
                    or type(track.get("measured")) is not bool
                    or type(track.get("qualified_moving")) is not bool):
                raise ValueError("Unique polarity IDs and boolean baseline evidence required")
            identities.add((segment, identity))
            xy = track.get("measurement_source_xy")
            if track["measured"]:
                if not _position(xy):
                    raise ValueError("Finite actual source measurement required")
                if track["qualified_moving"]:
                    measured_keys.add((segment, identity))
            elif xy is not None:
                raise ValueError("Prediction cannot contain a current measurement")
        if not isinstance(evidence, Mapping):
            raise ValueError("Qualified-measurement evidence mapping required")
        for key in evidence:
            if (not isinstance(key, tuple) or len(key) != 2 or not _integer(key[0])
                    or not isinstance(key[1], str)):
                raise ValueError("Evidence keys must be segment/identity tuples")
        if set(evidence) != measured_keys:
            raise ValueError("Evidence must cover exactly qualified actual measurements")
        for record in evidence.values():
            if (not isinstance(record, dict) or not isinstance(record.get("reason"), str)
                    or not record["reason"] or "features" not in record):
                raise ValueError("Explicit evidence features and reason required")
            features = record["features"]
            if features is None:
                if not record["reason"].startswith("unknown_"):
                    raise ValueError("Unavailable features require an explicit unknown reason")
            elif (not isinstance(features, dict)
                    or type(features.get("informative")) is not bool
                    or not _finite(features.get("point_minus_edge_fraction"))):
                raise ValueError("Finite signed point/edge diagnostic required")
        return identities

    @staticmethod
    def _decision(track):
        return dict(
            segment=track["segment"], track_id=track["track_id"],
            measured=track["measured"], baseline_qualified=True,
            accepted=True, reason=None, tier=None,
            measurement_source_xy=copy.deepcopy(track["measurement_source_xy"]),
            features=None, evidence_informative=None,
            measurement_frame=None, measurement_timestamp_ns=None,
            evidence_age_frames=None, evidence_age_ns=None,
            edge_streak_count=0, edge_streak_start_frame=None,
            edge_streak_start_timestamp_ns=None,
            inherited=False, inherited_reason=None, inherited_tier=None,
            expired_measurement_frame=None, expired_measurement_timestamp_ns=None,
            physical_class="unknown", airborne_confirmed=False,
        )

    def update(self, row, evidence):
        """Return qualified-only decisions in input order; commit atomically.

        ``evidence[(segment, track_id)]`` is a V36-shaped record containing
        ``features`` and ``reason``. It must exist for every qualified actual
        measurement, and for no other track. Unknown observations are explicit
        records, never omitted observations or inferred target absence.
        """
        identities = self._validate(row, evidence)
        frame, timestamp, segment = (row[name] for name in
                                     ("frame_index", "timestamp_ns", "segment"))
        state = ({} if segment != self._segment or row["motion"]["reset"] else
                 {key: value for key, value in self._state.items() if key in identities})
        output = []
        cfg = self.config
        for track in row["tracks"]:
            key = (segment, track["track_id"])
            if not track["qualified_moving"]:
                state.pop(key, None)
                continue
            decision = self._decision(track)
            prior = state.get(key)
            if not track["measured"]:
                if prior is None:
                    decision.update(reason="unknown_missing_history", tier="prediction_unknown")
                else:
                    verdict = prior.verdict
                    age_frames, age_ns = frame - verdict.frame, timestamp - verdict.timestamp
                    if (0 < age_frames <= cfg.maximum_coast_frames
                            and 0 < age_ns <= cfg.maximum_coast_ns):
                        decision.update(
                            accepted=verdict.accepted, reason="coast_" + verdict.reason,
                            tier="prediction_inherited", inherited=True,
                            inherited_reason=verdict.reason, inherited_tier=verdict.tier,
                            measurement_frame=verdict.frame,
                            measurement_timestamp_ns=verdict.timestamp,
                            evidence_age_frames=age_frames, evidence_age_ns=age_ns,
                        )
                        # The actual-measurement origin is unchanged. No edge
                        # streak survives even a single prediction.
                        state[key] = _History(verdict)
                    else:
                        decision.update(
                            reason="unknown_expired_history", tier="prediction_unknown",
                            evidence_age_frames=age_frames, evidence_age_ns=age_ns,
                            expired_measurement_frame=verdict.frame,
                            expired_measurement_timestamp_ns=verdict.timestamp,
                        )
                        state.pop(key, None)
                output.append(decision)
                continue

            observed = evidence[key]
            features = copy.deepcopy(observed["features"])
            informative = None if features is None else features["informative"]
            count, start_frame, start_timestamp = 0, None, None
            if features is None or not informative:
                accepted, tier = True, "unknown_measured"
                reason = (observed["reason"] if features is None else
                          "unknown_uninformative_patch")
            elif features["point_minus_edge_fraction"] > 0:
                accepted, reason, tier = True, "point_preferred", "point_supported_measured"
            else:
                consecutive = (prior is not None and prior.edge_count > 0
                               and prior.verdict.frame == frame - 1
                               and 0 < timestamp - prior.verdict.timestamp <= cfg.maximum_edge_gap_ns)
                count = prior.edge_count + 1 if consecutive else 1
                start_frame = prior.edge_start_frame if consecutive else frame
                start_timestamp = prior.edge_start_timestamp if consecutive else timestamp
                accepted = count < cfg.required_consecutive_edges
                reason = "edge_first_or_interrupted" if accepted else "edge_consecutive_rejected"
                tier = "edge_pending_measured" if accepted else "edge_suppressed_measured"
            verdict = _Verdict(accepted, reason, tier, frame, timestamp)
            state[key] = _History(verdict, count, start_frame, start_timestamp)
            decision.update(
                accepted=accepted, reason=reason, tier=tier,
                features=features, evidence_informative=informative,
                measurement_frame=frame, measurement_timestamp_ns=timestamp,
                evidence_age_frames=0, evidence_age_ns=0,
                edge_streak_count=count, edge_streak_start_frame=start_frame,
                edge_streak_start_timestamp_ns=start_timestamp,
            )
            output.append(decision)
        # No object shared with the previous state has been mutated. Validation
        # or feature-copy failure therefore cannot commit part of a frame.
        self._state = state
        self._frame, self._timestamp, self._segment = frame, timestamp, segment
        return output

    def retained_state_counts(self):
        return dict(tracks=len(self._state),
                    edge_streaks=sum(value.edge_count > 0 for value in self._state.values()))
