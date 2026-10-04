"""Bounded measured continuity, without promoting degraded evidence to confirmed.

Offline opt-in experiment. No source labels, clip identifiers, truth coordinates,
new detections, reassociation, predicted replacement measurements, or classifier.
Baseline qualification is never removed or redefined. A short grace state is
explicitly different from baseline-qualified output and cannot refresh itself.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContinuityConfig:
    maximum_gap_frames: int = 2
    maximum_gap_ns: int = 200_000_000
    minimum_excursion_px: float = 12.0

    def __post_init__(self):
        for name in ("maximum_gap_frames", "maximum_gap_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Positive integer grace budget required")
        value = self.minimum_excursion_px
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite baseline excursion threshold required")


def _position(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


class CausalMeasuredContinuity:
    def __init__(self, config=ContinuityConfig()):
        if not isinstance(config, ContinuityConfig):
            raise ValueError("Validated continuity config required")
        self.config = config
        self._frame = -1
        self._timestamp = None
        self._segment = None
        self._anchors = {}

    def update(self, row):
        frame, timestamp, segment = (row[k] for k in ("frame_index", "timestamp_ns", "segment"))
        if (type(frame) is not int or frame != self._frame + 1
                or type(timestamp) is not int or timestamp < 0
                or self._timestamp is not None and timestamp <= self._timestamp
                or type(segment) is not int or segment < 0
                or not isinstance(row.get("motion"), dict)
                or type(row["motion"].get("reset")) is not bool
                or not isinstance(row.get("tracks"), list)):
            raise ValueError("Contiguous frames, increasing time and explicit segment/reset required")
        observations = {}
        for track in row["tracks"]:
            if not isinstance(track, dict):
                raise ValueError("Track record must be a dictionary")
            identity = track.get("track_id")
            key = (segment, identity)
            if (not isinstance(identity, str) or not identity or track.get("segment") != segment
                    or type(track.get("segment")) is not int or key in observations
                    or type(track.get("measured")) is not bool
                    or type(track.get("qualified_moving")) is not bool):
                raise ValueError("Unique same-segment identity and explicit measurement/qualification required")
            if track["measured"]:
                if not _position(track.get("measurement_source_xy")):
                    raise ValueError("Measured state requires finite actual source coordinates")
            elif track.get("measurement_source_xy") is not None:
                raise ValueError("A prediction is not an actual measurement")
            quality = track.get("motion_quality")
            if quality is not None:
                if (not isinstance(quality, dict) or type(quality.get("ready")) is not bool
                        or type(quality.get("passed")) is not bool):
                    raise ValueError("Motion quality must have explicit readiness/pass state")
                if quality["ready"]:
                    rmse, limit = quality.get("quadratic_fit_rmse_px"), quality.get("maximum_rmse_px")
                    if (type(rmse) not in (int, float) or not math.isfinite(rmse) or rmse < 0
                            or type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
                            or quality["passed"] != (rmse <= limit)):
                        raise ValueError("Ready quality state must agree with its finite fit and limit")
                elif quality["passed"]:
                    raise ValueError("Unready quality cannot pass")
            observations[key] = track

        # Validation above is atomic. A reset, absent identity, or any coast
        # breaks grace lineage, even if a future row reuses the same ID.
        anchors = {} if segment != self._segment or row["motion"]["reset"] else {
            key: value for key, value in self._anchors.items() if key in observations
        }
        decisions = {}
        cfg = self.config
        for key, track in observations.items():
            measured, qualified = track["measured"], track["qualified_moving"]
            anchor = anchors.get(key)
            if not measured:
                anchors.pop(key, None)
                anchor = None
            if qualified and measured:
                anchor = (frame, timestamp)
                anchors[key] = anchor
            age_frames = None if anchor is None else frame-anchor[0]
            age_ns = None if anchor is None else timestamp-anchor[1]
            quality = track.get("motion_quality")
            excursion = track.get("excursion_px")
            confirmation = track.get("confirmation_timestamp_ns")
            quality_failed = (isinstance(quality, dict) and quality.get("ready") is True
                              and quality.get("passed") is False)
            other_conditions = (type(confirmation) is int and 0 <= confirmation <= timestamp
                                and type(excursion) in (int, float) and math.isfinite(excursion)
                                and excursion >= cfg.minimum_excursion_px)
            eligible_failure = measured and not qualified and quality_failed and other_conditions
            within_budget = (anchor is not None and 0 < age_frames <= cfg.maximum_gap_frames
                             and 0 < age_ns <= cfg.maximum_gap_ns)
            degraded = bool(eligible_failure and within_budget)
            if qualified:
                status = "baseline_qualified_measured" if measured else "baseline_qualified_prediction"
                reason = "unchanged_baseline_qualification"
            elif degraded:
                status, reason = "quality_degraded_measured", "bounded_recent_qualified_measurement"
            else:
                status = "not_output_eligible"
                reason = ("no_actual_measurement" if not measured else
                          "not_only_a_ready_motion_quality_failure" if not eligible_failure else
                          "no_unbroken_qualified_measurement_anchor" if anchor is None else
                          "grace_budget_expired")
            decisions[key] = dict(status=status, reason=reason, baseline_qualified=qualified,
                measured=measured, renderable=qualified or degraded,
                added_degraded_measurement=degraded, confirmed_output=qualified,
                physical_class="unknown", airborne_confirmed=False,
                anchor_frame=anchor[0] if anchor else None,
                anchor_timestamp_ns=anchor[1] if anchor else None,
                anchor_age_frames=age_frames, anchor_age_ns=age_ns,
                measurement_source_xy=track.get("measurement_source_xy"))
            # Only an actual baseline-qualified measurement can refresh grace.
            # Other failures break lineage instead of lending evidence onward.
            if not qualified and not degraded:
                anchors.pop(key, None)
        self._anchors = anchors
        self._frame, self._timestamp, self._segment = frame, timestamp, segment
        return decisions
