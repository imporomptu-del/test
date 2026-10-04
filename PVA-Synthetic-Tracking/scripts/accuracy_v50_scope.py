"""Pure metadata chronology for the V50 development diagnostic.

This module never opens files and never reads image, score, source-position,
reference, or archive-path values.  Its output is a scope, not an estimator.
Rows at a common clip/segment/frame always receive the same partition.

The primary calibration units are greedily spaced archived response frames;
each unit includes every archived state at that frame.  The sensitivity units
include all archived calibration response frames.  Neither policy is a fallback
for the other, and spacing does not establish statistical independence.
"""
from collections.abc import Mapping, Sequence


HISTORY_FRAMES = 8
EMBARGO_FRAMES = 8
ANCHOR_SPACING_FRAMES = 9
PARTITIONS = ("calibration", "embargo", "evaluation")
StateKey = tuple[str, int, int, str]


def _integer(value, name, *, nonnegative=True):
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _clip(value):
    if not isinstance(value, str) or not value:
        raise ValueError("clip must be a nonempty string")
    return value


def _cutoff_map(cutoffs):
    """Accept our JSON records or an explicit {(clip, segment): cutoff} map."""
    result = {}
    if isinstance(cutoffs, Mapping):
        records = []
        for key, cutoff in cutoffs.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("cutoff mapping keys must be (clip, segment)")
            records.append((key[0], key[1], cutoff))
    elif isinstance(cutoffs, Sequence) and not isinstance(cutoffs, (str, bytes)):
        try:
            records = [(r["clip"], r["segment"], r["cutoff_frame_index"])
                       for r in cutoffs]
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid cutoff record") from exc
    else:
        raise ValueError("cutoffs must be records or a (clip, segment) mapping")
    for clip, segment, cutoff in records:
        key = (_clip(clip), _integer(segment, "segment"))
        if key in result:
            raise ValueError("duplicate cutoff group")
        result[key] = _integer(cutoff, "cutoff_frame_index")
    return result


def partition_for(clip, segment, frame, cutoffs):
    """Partition any response frame, including a reference with no state row.

    The first evaluation frame is cutoff+9, so its eight-frame history starts
    at cutoff+1, strictly after every calibration response frame.
    """
    group = (_clip(clip), _integer(segment, "segment"))
    frame = _integer(frame, "frame")
    cutoff_by_group = _cutoff_map(cutoffs)
    if group not in cutoff_by_group:
        raise ValueError("missing cutoff for clip/segment")
    cutoff = cutoff_by_group[group]
    if frame <= cutoff:
        return "calibration"
    if frame <= cutoff + EMBARGO_FRAMES:
        return "embargo"
    return "evaluation"


def _minimal_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("each row must be a metadata mapping")
    try:
        clip = _clip(row["clip"])
        frame = _integer(row["frame_index"], "frame_index")
        segment = _integer(row["segment"], "segment")
        track = row["track_id"]
        archived = row["archive"] is not None
    except KeyError as exc:
        raise ValueError(f"missing scope field: {exc.args[0]}") from exc
    if not isinstance(track, str) or not track:
        raise ValueError("track_id must be a nonempty string")
    if archived:
        try:
            prior = row["geometry"]["geometry"]["prior_frame_indices"]
        except (KeyError, TypeError) as exc:
            raise ValueError("archived row requires eight prior frame indices") from exc
        if (frame < HISTORY_FRAMES or not isinstance(prior, (list, tuple))
                or any(type(x) is not int for x in prior)
                or list(prior) != list(range(frame - HISTORY_FRAMES, frame))):
            raise ValueError("archived history must be exactly frame-8 through frame-1")
    return (clip, frame, segment, track), archived


def _frame_unit(group, frame, row_keys, archived_by_key):
    archived = [list(k) for k in row_keys if archived_by_key[k]]
    unknown = [list(k) for k in row_keys if not archived_by_key[k]]
    return {
        "clip": group[0], "segment": group[1], "frame_index": frame,
        "input_span": [frame - HISTORY_FRAMES, frame],
        "state_keys": archived,
        "history_unknown_state_keys": unknown,
    }


def _greedy_anchors(units):
    selected = []
    previous_by_group = {}
    for unit in units:
        group = (unit["clip"], unit["segment"])
        previous = previous_by_group.get(group)
        if previous is None or unit["frame_index"] >= previous + ANCHOR_SPACING_FRAMES:
            if previous is not None and unit["input_span"][0] <= previous:
                raise ValueError("anchor input spans intersect")
            selected.append(unit)
            previous_by_group[group] = unit["frame_index"]
    return selected


def split_scope(rows):
    """Build a deterministic, JSON-ready split using only needed metadata.

    Cutoffs use the lower median of *all* unique selected response frames,
    including frames with no archives.  Duplicate state keys are rejected.
    Null-archive rows remain in partitions but never become scored units.

    ``state_keys`` in a frame unit are only archived states; unknown states at
    that same frame are separately preserved.  Only the presence of an archive
    is inspected: its path, hash, and contents are never accessed here.
    """
    archived_by_key = {}
    for row in rows:
        key, archived = _minimal_row(row)
        if key in archived_by_key:
            raise ValueError("duplicate state key")
        archived_by_key[key] = archived
    keys = sorted(archived_by_key, key=lambda k: (k[0], k[2], k[1], k[3]))
    groups = sorted({(k[0], k[2]) for k in keys})
    frames_by_group = {
        group: sorted({k[1] for k in keys if (k[0], k[2]) == group})
        for group in groups
    }
    cutoff_by_group = {
        group: frames[(len(frames)-1)//2]
        for group, frames in frames_by_group.items()
    }
    cutoffs = [{"clip": group[0], "segment": group[1], "cutoff_frame_index": cutoff}
               for group, cutoff in cutoff_by_group.items()]
    partitions = {name: [] for name in PARTITIONS}
    assignments = []
    frame_rows = {name: {} for name in PARTITIONS}
    for key in keys:
        clip, frame, segment, _ = key
        name = partition_for(clip, segment, frame, cutoff_by_group)
        partitions[name].append(list(key))
        assignments.append({"state_key": list(key), "partition": name,
                            "archived": archived_by_key[key]})
        frame_rows[name].setdefault((clip, segment, frame), []).append(key)
        if name == "evaluation" and archived_by_key[key]:
            if frame - HISTORY_FRAMES <= cutoff_by_group[(clip, segment)]:
                raise ValueError("evaluation history intersects calibration responses")
    units = {name: [] for name in PARTITIONS}
    no_archive_frames = {name: [] for name in PARTITIONS}
    counts = {}
    for name in PARTITIONS:
        for (clip, segment, frame), row_keys in frame_rows[name].items():
            if any(archived_by_key[k] for k in row_keys):
                units[name].append(_frame_unit((clip, segment), frame,
                                               row_keys, archived_by_key))
            else:
                no_archive_frames[name].append({
                    "clip": clip, "segment": segment, "frame_index": frame,
                    "state_keys": [list(k) for k in row_keys],
                })
        archived_count = sum(archived_by_key[tuple(k)] for k in partitions[name])
        counts[name] = {
            "states": len(partitions[name]), "archived_states": archived_count,
            "history_unknown_states": len(partitions[name]) - archived_count,
            "unique_response_frames": len(frame_rows[name]),
            "archived_response_frames": len(units[name]),
            "frames_without_archives": len(no_archive_frames[name]),
        }
    return {
        "schema_version": 1,
        "split_rule": "lower_median_of_all_unique_selected_response_frames_per_clip_segment",
        "history_frames": HISTORY_FRAMES,
        "embargo_frames": EMBARGO_FRAMES,
        "anchor_spacing_frames": ANCHOR_SPACING_FRAMES,
        "cutoffs": cutoffs,
        "assignments": assignments,
        "partitions": partitions,
        "anchors": {name: _greedy_anchors(units[name])
                    for name in ("calibration", "evaluation")},
        "calibration_all_archived_frames": units["calibration"],
        "frames_without_archives": no_archive_frames,
        "counts": counts,
        "quantile_policies": {
            "primary": "disjoint_input_span_calibration_frame_anchors",
            "sensitivity": "all_archived_calibration_response_frames_dependent",
            "automatic_fallback": False,
            "statistical_independence_established": False,
            "production_policy": False,
        },
        "scope_only_no_scores_or_image_values_read": True,
    }
