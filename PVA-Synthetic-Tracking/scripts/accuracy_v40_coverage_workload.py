"""Exact-ROI saved-journal accounting, not a false-positive classifier.

No rendering, model inference, media access, or references. Measurements use
actual source coordinates; predictions use their separate source positions.
Caller must freeze source-review records before using this on the real grid.
"""
from collections import Counter
import math


def inside(xy, crop):
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)):
        raise ValueError("Finite source position required")
    x, y, width, height = crop
    return x <= xy[0] < x + width and y <= xy[1] < y + height


def summarize_window(window, rows):
    first, last = window["frame_start"], window["frame_end_inclusive"]
    crop = window["crop_xywh"]
    if (type(first) is not int or type(last) is not int or first < 0 or last < first
            or not isinstance(crop, list) or len(crop) != 4
            or any(type(v) is not int for v in crop)
            or min(crop[:2]) < 0 or min(crop[2:]) <= 0):
        raise ValueError("Valid window indices and native integer ROI required")
    if [row["frame_index"] for row in rows] != list(range(first, last + 1)):
        raise ValueError("Complete ordered window frames required")
    counts = Counter(candidate_states=0, actual_measured_states=0,
                     qualified_measured_states=0, qualified_prediction_states=0)
    identities = set()
    frames = []
    for row in rows:
        measured, predicted, candidates = [], [], []
        for index, candidate in enumerate(row["candidates"]):
            if inside(candidate["source_xy"], crop):
                candidates.append(dict(candidate_index=index, source_xy=candidate["source_xy"],
                                       polarity=candidate["polarity"]))
        seen = set()
        for track in row["tracks"]:
            identity = (track["segment"], track["track_id"])
            if identity in seen or track["segment"] != row["segment"]:
                raise ValueError("Unique current-segment track IDs required")
            seen.add(identity)
            if type(track["measured"]) is not bool or type(track["qualified_moving"]) is not bool:
                raise ValueError("Explicit measurement and qualification required")
            if not track["measured"] and track.get("measurement_source_xy") is not None:
                raise ValueError("Prediction cannot contain an actual measurement")
            xy = track["measurement_source_xy"] if track["measured"] else track["source_xy"]
            if not inside(xy, crop):
                continue
            item = dict(segment=identity[0], track_id=identity[1], source_xy=xy,
                        qualified_moving=track["qualified_moving"])
            if track["measured"]:
                measured.append(item)
                if track["qualified_moving"]:
                    identities.add(identity)
            elif track["qualified_moving"]:
                predicted.append(item)
        counts["candidate_states"] += len(candidates)
        counts["actual_measured_states"] += len(measured)
        counts["qualified_measured_states"] += sum(t["qualified_moving"] for t in measured)
        counts["qualified_prediction_states"] += len(predicted)
        frames.append(dict(frame_index=row["frame_index"], candidates=candidates,
                           actual_measurements=measured, qualified_predictions=predicted))
    return dict(window_id=window["window_id"], frame_count=len(rows), crop_xywh=crop,
                counts=dict(counts), distinct_qualified_measured_ids=len(identities),
                frames=frames, false_positive_count=None, physical_class_inferred=False,
                verified_negative_exposure=False)
