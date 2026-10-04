"""Exploratory metadata-only follow-through of every applied weak correction.

Developed after the frozen run started, before its weak outcomes were inspected.
This is not an inference-policy change, an identity matcher, or an accuracy test.
Only audit-bound JSON/JSONL is read; no media, native arrays, or detector imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SCHEMA = "seaqr.weak-shadow-followthrough.v1"
SUMMARY_SHA256 = "959a00002953ac4667f6efbfe665cfc879239614ddecefab72c0409a025ab0be"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reader():
    path = Path(__file__).resolve().with_name("summarize_weak_continuation_shadow_v1.py")
    require(path.is_file() and not path.is_symlink() and sha(path) == SUMMARY_SHA256,
            "metadata reader source pin differs")
    spec = importlib.util.spec_from_file_location("weak_followthrough_pinned_reader", path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    require(sha(path) == SUMMARY_SHA256, "metadata reader changed while loading")
    return reader


def point(value, description, optional=False):
    if value is None and optional:
        return None
    require(type(value) is list and len(value) == 2 and
            all(type(x) in (int, float) and math.isfinite(x) for x in value),
            "invalid " + description)
    return list(value)


def state(record):
    if record is None:
        return dict(present=False)
    return dict(present=True, track_id=record["track_id"], segment=record["segment"],
                measured=record["measured"], qualified_moving=record["qualified_moving"],
                measurement_source_xy=point(record.get("measurement_source_xy"), "strong source coordinate", True),
                posterior_source_xy=point(record.get("source_xy"), "posterior source coordinate", True))


def keyed(records):
    return {f"{r['segment']}/{r['track_id']}": r for r in records}


def exact_baseline_matches(records, track_id, xy):
    """Coordinate equality, not proximity or physical-identity association."""
    polarity = track_id.split(":", 1)[0]
    return [state(r) for r in records if r["measured"] and
            r["track_id"].split(":", 1)[0] == polarity and
            r["measurement_source_xy"] == xy]


def follow_events(baseline, trace):
    """Inspect already validated contiguous journals; never select an event subset."""
    require(len(baseline) == len(trace) and len(trace) > 0, "complete paired journals required")
    for frame, (b, t) in enumerate(zip(baseline, trace)):
        require(b["frame_index"] == t["frame_index"] == frame and
                b["timestamp_ns"] == t["timestamp_ns"] and b["segment"] == t["segment"],
                "paired journal headers differ")
    bmaps = [keyed(row["tracks"]) for row in baseline]
    tmaps = [keyed(row["records"]) for row in trace]
    events = []
    for frame, row in enumerate(trace):
        for record in row["records"]:
            note = record["weak_evidence"]
            if not note["applied"]:
                continue
            identity = f"{row['segment']}/{record['track_id']}"
            require(note["identity"] == identity and not record["measured"] and
                    note["status"] == "weak_kinematic_correction", "invalid applied weak event")
            anchor = note.get("strong_anchor_timestamp_ns")
            require(type(anchor) is int and 0 <= anchor < row["timestamp_ns"], "invalid strong anchor")
            event = dict(event_index=len(events), frame_index=frame, timestamp_ns=row["timestamp_ns"],
                shadow_native_identity=identity, segment=row["segment"],
                weak_measurement_reference_xy=point(note.get("measurement_reference_xy"), "weak reference coordinate"),
                weak_posterior_source_xy=point(record.get("source_xy"), "weak posterior source coordinate", True),
                strong_anchor_timestamp_ns=anchor, capture_scheduled=row["capture_scheduled"],
                baseline_at_weak_frame=dict(same_native_id_diagnostic=state(bmaps[frame].get(identity))),
                lineage_established=False)
            terminal = None
            last_present = frame
            for future in range(frame + 1, len(trace)):
                following = trace[future]
                own = tmaps[future].get(identity)
                if following["segment"] != row["segment"]:
                    kind = "segment_reset"
                elif own is None:
                    kind = "disappearance"
                elif own["measured"]:
                    kind = "next_strong_measurement"
                elif own["weak_evidence"]["status"] == "strong_age_expired":
                    kind = "explicit_strong_age_expiry"
                else:
                    last_present = future
                    continue
                terminal = dict(kind=kind, frame_index=future, timestamp_ns=following["timestamp_ns"],
                    elapsed_since_weak_seconds=(following["timestamp_ns"]-row["timestamp_ns"])/1e9,
                    elapsed_since_strong_anchor_seconds=(following["timestamp_ns"]-anchor)/1e9,
                    terminal_capture_scheduled=following["capture_scheduled"],
                    shadow_state=state(own),
                    baseline_same_native_id_diagnostic=state(bmaps[future].get(identity)),
                    disappearance_cause_unknown=kind == "disappearance")
                if kind == "next_strong_measurement":
                    xy = point(own["measurement_source_xy"], "next strong source coordinate")
                    matches = exact_baseline_matches(baseline[future]["tracks"], own["track_id"], xy)
                    terminal.update(measurement_source_xy=xy,
                        exact_baseline_actual_measurement_matches=matches,
                        exact_baseline_actual_measurement_match_count=len(matches),
                        exact_baseline_match_status="none" if not matches else "one" if len(matches) == 1 else "multiple",
                        baseline_exact_match_is_physical_identity=False)
                break
            if terminal is None:
                final = trace[-1]
                terminal = dict(kind="end_of_clip_censored", frame_index=final["frame_index"],
                    timestamp_ns=final["timestamp_ns"],
                    elapsed_since_weak_seconds=(final["timestamp_ns"]-row["timestamp_ns"])/1e9,
                    elapsed_since_strong_anchor_seconds=(final["timestamp_ns"]-anchor)/1e9,
                    terminal_capture_scheduled=final["capture_scheduled"],
                    shadow_state=state(tmaps[-1].get(identity)),
                    baseline_same_native_id_diagnostic=state(bmaps[-1].get(identity)))
            event.update(terminal=terminal, last_present_frame_before_terminal=last_present)
            events.append(event)
    return events


def analyze(directory, freeze_sha256, plan_sha256):
    reader = load_reader()
    cohort = reader.summarize(directory, freeze_sha256, plan_sha256)
    require(not cohort["invalid_or_unknown_decisions_present"], "invalid decisions prevent interpretation")
    evidence = reader.BoundMetadata(directory)
    # Re-bind all summary/audit inputs so an in-between change cannot be accepted.
    for relative, digest in cohort["files_sha256"].items():
        evidence.path(relative, digest)
    clips = {}
    total = Counter()
    for clip in sorted(cohort["clips"]):
        journals = []
        for suffix in ("clean/frames.jsonl", "shadow/shadow_trace.jsonl"):
            relative = clip + "/" + suffix
            path = evidence.path(relative, cohort["files_sha256"][relative])
            with path.open() as stream:
                journals.append([reader.loads(line) for line in stream])
        events = follow_events(*journals)
        require(len(events) == cohort["clips"][clip]["state_totals"]["shadow"]["weak_corrected"],
                "not every applied weak correction was followed")
        counts = Counter(event["terminal"]["kind"] for event in events)
        total.update(counts)
        clips[clip] = dict(event_count=len(events), terminal_counts=dict(counts), events=events)
    evidence.unchanged()
    return dict(schema=SCHEMA, passed=True, exploratory=True,
        design_timing="Developed after run started, before weak outcomes inspected; not a predeclared success gate",
        freeze_sha256=freeze_sha256, plan_sha256=plan_sha256, analyzer_code_sha256=sha(Path(__file__)),
        metadata_reader_code_sha256=SUMMARY_SHA256, files_sha256=evidence.hashes,
        all_three_passed_audits_required=True, all_applied_events_included=True,
        event_count=sum(c["event_count"] for c in clips.values()), terminal_counts=dict(total), clips=clips,
        media_accessed=False, native_arrays_interpreted=False, policy_changed=False,
        physical_lineage_established=False, accuracy_improvement_established=False,
        limitations=["Own follow-through means the same segment/native shadow ID, not verified physical identity",
            "Baseline same-ID presence is diagnostic only; independent associations may already have diverged",
            "Exact source-coordinate and polarity matches across all baseline IDs are evidence reuse, not lineage or accuracy",
            "No distance tolerance, nearest match, favorable subset, or inferred disappearance cause is used",
            "Explicit strong-age expiry is an observed status, not an independently reconstructed deletion reason",
            "End-of-clip follow-through is censored; later reacquisition or failure is unknown",
            "Weak posterior source coordinates are not raw weak measurement coordinates, which are reported in reference space",
            "Qualified tracker history and source-visible motion do not establish airborne class"])


def write_analysis(directory, freeze_sha256, plan_sha256, output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh follow-through output required")
    result = analyze(directory, freeze_sha256, plan_sha256)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_analysis(args.directory, args.freeze_sha256, args.plan_sha256, args.output)
