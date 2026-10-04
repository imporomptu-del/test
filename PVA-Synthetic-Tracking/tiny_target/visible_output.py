"""Observation-backed output channels; not a detector or airborne classifier.

Consumes completed tracker rows without changing them. Qualification is always
the current row's decision. Only measurement age is carried between calls;
neither this history nor the emitted channels feed back into the tracker.
"""

import math


SCHEMA = "seaqr.visible-observation-output.v1"


def _integer(value):
    return type(value) is int and value >= 0


def _point(value):
    valid = isinstance(value, (list, tuple)) and len(value) == 2
    try:
        valid = valid and all(type(v) in (int, float) and math.isfinite(v) for v in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Finite two-number source coordinate required")
    return tuple(value)


class ObservationOutput:
    """One instance per ordered stream; alert observations are NOT object counts.

    Call once per frame (including empty frames). Arbitrary starting frame is
    permitted, but prior measurement age is unknown until observed locally.
    Gaps/duplicates/reordered frames are rejected instead of inventing age.
    Stream timestamps, not wall-clock time, determine observation age.
    """

    def __init__(self, stream_id):
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("Nonempty stream identity required")
        self.stream_id = stream_id
        self._frame = self._timestamp = self._segment = None
        self._measurements = {}

    def update(self, row):
        if not isinstance(row, dict):
            raise ValueError("Completed tracker row required")
        frame, timestamp, segment = (row.get(k) for k in
                                      ("frame_index", "timestamp_ns", "segment"))
        if (not _integer(frame) or not _integer(timestamp) or not _integer(segment)
                or self._frame is not None and frame != self._frame + 1
                or self._timestamp is not None and timestamp <= self._timestamp
                or self._segment is not None and segment < self._segment):
            raise ValueError("Contiguous frames, increasing time, and nondecreasing segments required")
        tracks = row.get("tracks")
        if not isinstance(tracks, list):
            raise ValueError("Explicit track list required")
        validated = []
        seen = set()
        for track in tracks:
            if not isinstance(track, dict):
                raise ValueError("Track record required")
            tid = track.get("track_id")
            if (not isinstance(tid, str) or not tid.strip() or "/" in tid
                    or not _integer(track.get("segment")) or track["segment"] != segment
                    or tid in seen):
                raise ValueError("Unique track identity in the current segment required")
            seen.add(tid)
            measured, qualified = track.get("measured"), track.get("qualified_moving")
            if type(measured) is not bool or type(qualified) is not bool:
                raise ValueError("Explicit measured and qualified booleans required")
            filtered = _point(track.get("source_xy"))
            if "measurement_source_xy" not in track:
                raise ValueError("Explicit measurement source coordinate or null required")
            if measured:
                actual = _point(track["measurement_source_xy"])
            else:
                if track["measurement_source_xy"] is not None:
                    raise ValueError("Prediction cannot contain a current measurement")
                actual = None
            validated.append((tid, measured, qualified, actual, filtered))

        # All validation precedes state mutation. Missing IDs and new segments
        # discard age history; later reuse must not inherit stale observations.
        previous = self._measurements if segment == self._segment else {}
        next_measurements = {}
        alerts, context = [], []
        for tid, measured, qualified, actual, filtered in validated:
            last = (frame, timestamp) if measured else previous.get(tid)
            if last is not None:
                next_measurements[tid] = last
            if not qualified:
                continue
            record = dict(
                identity=f"{self.stream_id}/{segment}/{tid}", track_id=tid, segment=segment,
                measured=measured, qualified_moving=qualified,
                source_xy=list(actual if measured else filtered),
                coordinate_kind="measurement" if measured else "prediction",
                last_measurement_timestamp_ns=last[1] if last is not None else None,
                last_measurement_age_ns=timestamp-last[1] if last is not None else None,
                last_measurement_age_frames=frame-last[0] if last is not None else None,
                physical_class="unknown", airborne_confirmed=False,
            )
            context.append(record)
            if measured:
                # Separate coordinate list prevents cross-channel aliasing.
                alerts.append(dict(record, source_xy=list(actual),
                                   kind="qualified_motion_observation"))
        self._measurements = next_measurements
        self._frame, self._timestamp, self._segment = frame, timestamp, segment
        return dict(schema=SCHEMA, stream_id=self.stream_id, frame_index=frame,
                    timestamp_ns=timestamp, segment=segment,
                    observation_alerts=alerts, track_context=context)
