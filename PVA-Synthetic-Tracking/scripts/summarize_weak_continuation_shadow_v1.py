"""Metadata-only workload summary of three completed, independently audited runs.

No producer/core, detector, NumPy, media decoder or native arrays are imported.
Native IDs and exact assignment changes are diagnostics, never physical truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

SCHEMA = "seaqr.weak-continuation-shadow.summary.v1"
SOURCES = {
    "0029": (687, "0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359"),
    "0126": (674, "c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344"),
    "0055": (689, "c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f"),
}
SCHEDULED_SLOTS = 182
STATUSES = frozenset(("strong_measurement_priority", "no_prior_same_segment_track", "not_prior_strong_qualified",
    "strong_age_expired", "weak_budget_used_for_strong_gap", "missing_capture", "capture_coverage_unknown",
    "original_threshold_peak_in_gate", "no_unique_weak_peak", "competing_prior_identity_gate",
    "overlaps_current_strong_evidence", "weak_kinematic_correction", "invalid_capture_or_weak_covariance"))
PROVIDER_STATUSES = STATUSES - {"strong_measurement_priority", "no_prior_same_segment_track",
    "not_prior_strong_qualified", "strong_age_expired", "weak_budget_used_for_strong_gap"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def loads(text):
    def pairs(items):
        result = {}
        for key,value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite JSON constant")))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)


class BoundMetadata:
    def __init__(self, root):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve() == self.root and self.root.is_dir()
                and not self.root.is_symlink(), "canonical evidence directory required")
        self.hashes = {}

    def path(self, relative, expected=None):
        part = Path(relative)
        require(type(relative) is str and relative and not part.is_absolute() and ".." not in part.parts
                and "." not in part.parts and str(part) == relative, "unsafe metadata path")
        path = self.root
        for name in part.parts:
            path /= name
            require(not path.is_symlink(), "symlink metadata refused")
        require(path.is_file() and path.suffix in (".json", ".jsonl"), "regular metadata required")
        actual = sha(path)
        if expected is not None:
            require(type(expected) is str and re.fullmatch("[0-9a-f]{64}",expected) and actual == expected, "changed bound metadata: "+relative)
        require(relative not in self.hashes or self.hashes[relative] == actual, "metadata changed while summarizing")
        self.hashes[relative] = actual
        return path

    def read(self, relative, expected=None):
        path = self.path(relative, expected)
        result = loads(path.read_text())
        require(type(result) is dict, "metadata JSON object required")
        self.path(relative,self.hashes[relative])
        return result

    def unchanged(self):
        for relative,digest in list(self.hashes.items()):
            self.path(relative,digest)


def frame_header(row, frame):
    require(type(row) is dict and type(row.get("frame_index")) is int and row["frame_index"] == frame,
            "incomplete/noncontiguous frame sequence")
    require(type(row.get("timestamp_ns")) is int and row["timestamp_ns"] == frame*100000000,
            "frame cadence differs")
    require(type(row.get("segment")) is int and row["segment"] >= 0, "invalid coordinate segment")


def state_counts(records, segment, shadow=False):
    require(type(records) is list, "track records required")
    counts = Counter(dict(states=0, qualified_states=0, strong_measured=0, strong_measured_qualified=0,
        weak_corrected=0, weak_corrected_qualified=0, predicted_only=0, predicted_only_qualified=0))
    identities, assignments, notes = {}, {}, []
    for record in records:
        require(type(record) is dict and record.get("segment") == segment and type(record.get("segment")) is int,
                "track segment differs")
        name = record.get("track_id")
        require(type(name) is str and re.fullmatch("(bright|dark):[0-9]+",name), "invalid native track ID")
        identity = f"{segment}/{name}"
        require(identity not in identities, "duplicate track identity")
        measured, qualified = record.get("measured"), record.get("qualified_moving")
        require(type(measured) is bool and type(qualified) is bool, "boolean measured/qualified required")
        weak = False
        if shadow:
            note = record.get("weak_evidence")
            require(type(note) is dict and note.get("identity") == identity and type(note.get("applied")) is bool
                    and type(note.get("status")) is str and note.get("is_ordinary_measurement") is False
                    and note.get("physical_identity_verified") is False, "weak evidence provenance invalid")
            weak = note["applied"]
            require(not weak or (not measured and note["status"] == "weak_kinematic_correction"), "weak/strong measurement conflation")
            notes.append(note)
        position = record.get("measurement_source_xy")
        if measured:
            require(type(position) is list and len(position) == 2 and all(type(x) in (int,float) and math.isfinite(x) for x in position),
                    "actual strong measurement coordinate required")
            assignments[identity] = position
        else:
            require(position is None, "unmeasured state carries an actual measurement")
        category = "strong_measured" if measured else "weak_corrected" if weak else "predicted_only"
        counts["states"] += 1
        counts["qualified_states"] += qualified
        counts[category] += 1
        counts[category+"_qualified"] += qualified
        identities[identity] = dict(qualified=qualified, category=category)
    return dict(counts), identities, assignments, notes


def assignment_difference(baseline, shadow):
    left, right = set(baseline),set(shadow)
    same = left & right
    changed = sum(encoded(baseline[k]) != encoded(shadow[k]) for k in same)
    shared = len(same)-changed
    return dict(changed=encoded(baseline) != encoded(shadow), exact_id_and_position_pairs_shared=shared,
        same_native_id_different_actual_measurement=changed, baseline_only_native_ids=len(left-right),
        shadow_only_native_ids=len(right-left), baseline_pairs_not_exactly_shared=len(left)-shared,
        shadow_pairs_not_exactly_shared=len(right)-shared)


def update_lifetimes(store, identities, frame):
    for identity,state in identities.items():
        item = store.setdefault(identity,dict(first_frame=frame,last_frame=frame,visible_state_frames=0,
            qualified_frame_count=0,first_qualified_frame=None,last_qualified_frame=None,
            strong_measured_frames=0,weak_corrected_frames=0,predicted_only_frames=0))
        item["last_frame"] = frame
        item["visible_state_frames"] += 1
        item[state["category"]+"_frames"] += 1
        if state["qualified"]:
            item["qualified_frame_count"] += 1
            if item["first_qualified_frame"] is None:
                item["first_qualified_frame"] = frame
            item["last_qualified_frame"] = frame


def qualified_lifetimes(store):
    return {identity:dict(item, journal_visible_lifetime_span_seconds=(item["last_frame"]-item["first_frame"])/10,
            qualification_span_seconds=(item["last_qualified_frame"]-item["first_qualified_frame"])/10)
            for identity,item in sorted(store.items()) if item["qualified_frame_count"]}


def ratio(numerator,denominator):
    return dict(numerator=numerator,denominator=denominator,value=numerator/denominator if denominator else None)


def decisions(notes):
    counts = Counter()
    reasons = Counter()
    for note in notes:
        status = note["status"]
        counts["provider_attempts"] += status in PROVIDER_STATUSES
        counts["missing_capture"] += status == "missing_capture"
        counts["invalid_decisions"] += status.startswith("invalid")
        counts["unknown_statuses"] += status not in STATUSES
        if "observations" in note:
            observed = note["observations"]
            require(type(observed) is dict and type(observed.get("coverage_known")) is bool
                    and type(observed.get("coverage_unknown_reasons")) is list
                    and all(type(s) is str for s in observed["coverage_unknown_reasons"]), "invalid observation coverage")
            counts["observed_capture_decisions"] += 1
            counts["coverage_known"] += observed["coverage_known"]
            counts["coverage_censored"] += not observed["coverage_known"]
            reasons.update(observed["coverage_unknown_reasons"])
    return counts,reasons


def coverage_summary(counts,reasons):
    return dict(counts, coverage_unknown_reasons=dict(reasons),
        available_capture_fraction=ratio(counts["observed_capture_decisions"],counts["provider_attempts"]),
        known_coverage_fraction_of_observed_captures=ratio(counts["coverage_known"],counts["observed_capture_decisions"]),
        censored_fraction_of_observed_captures=ratio(counts["coverage_censored"],counts["observed_capture_decisions"]))


def summarize_clip(evidence,clip,spec,freeze_sha,plan_sha):
    count,source_sha = SOURCES[clip]
    audit = evidence.read(clip+"/independent_audit.json")
    require(audit.get("schema") == "seaqr.weak-continuation-shadow.audit.v1" and audit.get("passed") is True
            and audit.get("clip") == clip and type(audit.get("frames")) is int and audit["frames"] == count,
            "absent/failed/partial independent audit")
    for key,expected in (("freeze_sha256",freeze_sha),("plan_sha256",plan_sha),("source_sha256",source_sha),
        ("baseline_journal_non_timing_exact",True),("baseline_output_state_learning_digests_exact",True),
        ("native_state_guards_unchanged",True),("production_changed",False),("weak_learning_enabled",False)):
        require(type(audit.get(key)) is type(expected) and audit[key] == expected,"audit identity/isolation differs: "+key)
    hashes = audit.get("files_sha256")
    require(type(hashes) is dict,"audit file bindings missing")
    required = ("clean/frames.jsonl","shadow/shadow_trace.jsonl","clean.shadow.json","shadow.shadow.json")
    require(all(name in hashes for name in required),"audit lacks summary input binding")
    paths = {name:evidence.path(clip+"/"+name,hashes[name]) for name in required}
    receipts = {}
    for arm in ("clean","shadow"):
        receipt = evidence.read(clip+"/"+arm+".shadow.json",hashes[arm+".shadow.json"])
        require(receipt.get("schema") == "seaqr.weak-continuation-shadow.run.v1" and receipt.get("passed") is True and receipt.get("error") is None and receipt.get("clip") == clip
                and receipt.get("arm") == arm and receipt.get("processed_frames") == count and receipt.get("expected_frames") == count,
                "incomplete run receipt")
        require(receipt.get("freeze_sha256") == freeze_sha and receipt.get("plan_sha256") == plan_sha
                and receipt.get("source_sha256") == source_sha and receipt.get("production_changed") is False
                and receipt.get("weak_learning_enabled") is False, "run receipt identity differs")
        receipts[arm] = receipt
    require(receipts["shadow"].get("trace_sha256") == hashes["shadow/shadow_trace.jsonl"],"trace receipt binding differs")
    windows = spec["weak_windows_inclusive"]
    scheduled = {i for first,last in windows for i in range(first,last+1)}
    totals = {"baseline":Counter(),"shadow":Counter()}
    lifetime = {"baseline":{},"shadow":{}}
    reason_counts,scheduled_reasons,invalid_statuses = Counter(),Counter(),Counter()
    coverage,scheduled_coverage,coverage_reasons,scheduled_coverage_reasons = Counter(),Counter(),Counter(),Counter()
    rows,change_totals = [],Counter()
    with paths["clean/frames.jsonl"].open() as clean,paths["shadow/shadow_trace.jsonl"].open() as trace:
        for frame,lines in enumerate(itertools.zip_longest(clean,trace)):
            require(frame < count and all(type(line) is str and line.strip() for line in lines),"partial/extra/blank journal")
            baseline,shadow = map(loads,lines)
            frame_header(baseline,frame);frame_header(shadow,frame)
            require(baseline["segment"] == shadow["segment"] and type(shadow.get("capture_scheduled")) is bool
                    and shadow["capture_scheduled"] == (frame in scheduled),"trace segment/schedule differs")
            bcount,bids,bassign,_ = state_counts(baseline.get("tracks"),baseline["segment"])
            scount,sids,sassign,notes = state_counts(shadow.get("records"),shadow["segment"],True)
            require(frame in scheduled or not scount["weak_corrected"],"weak correction outside frozen schedule")
            wm = shadow.get("metrics",{}).get("weak_continuation",{})
            require(wm.get("frame_index") == frame and type(wm.get("decisions")) is list
                    and encoded(wm["decisions"]) == encoded(notes) and wm.get("applied_count") == scount["weak_corrected"], "decision/record mismatch")
            for name,counts,ids in (("baseline",bcount,bids),("shadow",scount,sids)):
                totals[name].update(counts)
                update_lifetimes(lifetime[name],ids,frame)
            difference = assignment_difference(bassign,sassign)
            change_totals.update(difference)
            reasons = Counter(n["status"] for n in notes)
            reason_counts.update(reasons)
            invalid_statuses.update(n["status"] for n in notes if n["status"].startswith("invalid") or n["status"] not in STATUSES)
            c,r = decisions(notes)
            coverage.update(c);coverage_reasons.update(r)
            if frame in scheduled:
                scheduled_reasons.update(reasons);scheduled_coverage.update(c);scheduled_coverage_reasons.update(r)
            rows.append(dict(frame_index=frame,capture_scheduled=frame in scheduled,baseline=bcount,shadow=scount,
                             exact_native_assignment_diagnostic=difference,decision_reasons=dict(reasons)))
    require(len(rows) == count,"incomplete journal extent")
    timing = receipts["shadow"].get("diagnostic_cost_ms",{})
    require(all(type(timing.get(k)) in (int,float) and math.isfinite(timing[k]) and timing[k] >= 0
                for k in ("capture","shadow","snapshot_write")),"incomplete diagnostic timing")
    require(timing["shadow"] >= timing["snapshot_write"],"nested snapshot timing exceeds shadow elapsed")
    assignment_totals = dict(change_totals)
    assignment_totals["frames_with_different_exact_assignments"] = assignment_totals.pop("changed",0)
    return dict(clip=clip,frames=count,scheduled_frame_slots=len(scheduled),decoded_frame_instances=2*count,
        state_totals={k:dict(v) for k,v in totals.items()},per_frame=rows,
        qualified_identities={k:qualified_lifetimes(v) for k,v in lifetime.items()},
        qualified_identity_counts={k:sum(bool(item["qualified_frame_count"]) for item in v.values()) for k,v in lifetime.items()},
        exact_native_assignment_diagnostic_totals=assignment_totals,decision_reason_counts=dict(reason_counts),
        scheduled_decision_reason_counts=dict(scheduled_reasons),invalid_or_unknown_status_counts=dict(invalid_statuses),
        all_frame_capture_coverage=coverage_summary(coverage,coverage_reasons),
        scheduled_capture_coverage=coverage_summary(scheduled_coverage,scheduled_coverage_reasons),
        diagnostic_timing_ms=dict(capture=timing["capture"],shadow_excluding_snapshot_write=timing["shadow"]-timing["snapshot_write"],
            snapshot_write=timing["snapshot_write"],shadow_inclusive_snapshot_write=timing["shadow"]),
        timing_is_not_pipeline_fps=True,truth_metrics=dict(true_positives=None,false_positives=None,airborne_identity=None,precision=None,recall=None))


def summarize(directory,freeze_sha256,plan_sha256):
    evidence = BoundMetadata(directory)
    freeze = evidence.read("freeze.json",freeze_sha256)
    plan = evidence.read("plan.json",plan_sha256)
    require(freeze.get("schema") == "seaqr.weak-continuation-shadow.freeze.v1" and freeze.get("pre_run") is True and freeze.get("plan_sha256") == plan_sha256,"freeze/plan binding differs")
    require(plan.get("schema") == "seaqr.weak-continuation-shadow.plan.v1" and set(plan.get("clips",{})) == set(SOURCES)
            and plan.get("full_causal_replay") is True and plan.get("concurrent_workers") == 1,"frozen three-clip cohort required")
    batch = evidence.read("batch_status.json")
    require(batch.get("schema") == "seaqr.weak-continuation-shadow.batch.v1" and batch.get("passed") is True and batch.get("error") is None and batch.get("concurrent_workers") == 1
            and batch.get("production_changed") is False,"batch incomplete/failed")
    stages = [(r.get("clip"),r.get("arm")) for r in batch.get("runs",[])]
    require(len(stages) == 9 and set(stages) == {(clip,arm) for clip in SOURCES for arm in ("clean","shadow","audit")}
            and all(type(r.get("returncode")) is int and r["returncode"] == 0 for r in batch["runs"]),"nine completed stages required")
    scheduled = 0
    for clip,(count,digest) in SOURCES.items():
        spec = plan["clips"][clip]
        require(spec.get("frames") == count and spec.get("source_sha256") == digest,"plan source identity differs")
        previous = -1
        for interval in spec.get("weak_windows_inclusive",[]):
            require(type(interval) is list and len(interval) == 2 and all(type(x) is int for x in interval)
                    and previous < interval[0] <= interval[1] < count,"invalid frozen schedule")
            scheduled += interval[1]-interval[0]+1
            previous = interval[1]
    require(scheduled == SCHEDULED_SLOTS,"scheduled slot count differs")
    clips = {clip:summarize_clip(evidence,clip,plan["clips"][clip],freeze_sha256,plan_sha256) for clip in SOURCES}
    evidence.unchanged()
    unique = sum(count for count,_ in SOURCES.values())
    return dict(schema=SCHEMA,passed=True,complete_audited_metadata_summary=True,freeze_sha256=freeze_sha256,
        plan_sha256=plan_sha256,summary_code_sha256=sha(Path(__file__)),files_sha256=evidence.hashes,
        unique_source_frames=unique,decoded_frame_instances=2*unique,scheduled_shadow_frame_slots=scheduled,
        clips=clips,invalid_or_unknown_decisions_present=any(c["invalid_or_unknown_status_counts"] for c in clips.values()),
        physical_identity_established=False,accuracy_improvement_established=False,producer_imported=False,native_arrays_interpreted=False,
        limits=["Native IDs can diverge; exact ID/actual-position differences are assignment/workload diagnostics, not physical identity or accuracy",
            "Strong/weak/predicted states remain separate; qualified is tracker history, not verified airborne class",
            "Outside-schedule missing captures are planned inactivity; scheduled capture coverage is reported separately",
            "Lifetimes are observed journal spans, not verified object survival; qualification can be intermittent",
            "Diagnostic timing excludes nested snapshot writing from shadow compute and is not detector throughput",
            "No true/false-positive labels, historical-reference rescoring, threshold fitting or holdout access"])


def write_summary(directory,freeze_sha256,plan_sha256,output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(),"fresh summary output required")
    result = summarize(directory,freeze_sha256,plan_sha256)
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--freeze-sha256",required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    write_summary(args.directory,args.freeze_sha256,args.plan_sha256,args.output)
