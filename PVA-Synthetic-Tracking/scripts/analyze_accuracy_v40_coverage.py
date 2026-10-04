"""Post-source-review ROI workload, with no automatic truth or policy decisions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from accuracy_v40_coverage_workload import summarize_window
from score_phase20_accuracy import assign_one_to_one

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/tiny_target"
MANIFEST = BASE / "accuracy_v39_20260925/validation_coverage_plan_v1.json"
MANIFEST_SHA = "cff39ab26a03178e4a350111dae78aef58ea24483e4856582c3663d46f808ebc"
JOURNALS = BASE / "visible_validation_v34_20260923/audit_20260924/evidence/run"
AUDIT = BASE / "accuracy_v39_20260925/continuity_independent_audit_01.json"
AUDIT_SHA = "074af6d1e5752280be79de40d911e3aa268949ecc58d3fcddfd09adfcd4cb3e6"
REFERENCE = BASE / "accuracy_v40_20260925/grid_visible_reference_v1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def run(reviews, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh output directory required")
    files = {}

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Changed input: " + path)
        if path in files and files[path] != digest:
            raise ValueError("Input changed during analysis")
        files[path] = digest

    bind(MANIFEST, MANIFEST_SHA)
    bind(AUDIT, AUDIT_SHA)
    manifest, audit = read(MANIFEST), read(AUDIT)
    windows = {w["window_id"]: w for w in manifest["windows"]}
    reviewed, clips = {}, set()
    if len(reviews) != 4:
        raise ValueError("Exactly four completed source review records required")
    for path in reviews:
        bind(path)
        review = read(path)
        clip = review["clip_id"]
        if clip not in manifest["allowlisted_clips"] or clip in clips:
            raise ValueError("Unique allowlisted clip review required")
        clips.add(clip)
        if (review["source_only"] is not True or
                review["window_scores_seen_before_review"] is not False):
            raise ValueError("Initial source review must precede window scores")
        bind(review["packet_manifest_path"], review["packet_manifest_sha256"])
        for record in review["windows"]:
            wid = record["window_id"]
            if wid not in windows or wid in reviewed or windows[wid]["clip"] != clip:
                raise ValueError("Unexpected/duplicate review window")
            w = windows[wid]
            if record["reviewed_frames"] != list(range(w["frame_start"], w["frame_end_inclusive"] + 1)):
                raise ValueError("Every selected source frame must be reviewed")
            bind(record["sheet_path"], record["sheet_sha256"])
            # This workload tool cannot validate a physical-class/absence label.
            # Such a proposed label requires its own adjudication and scorer.
            if record["authoritative_negative"] is not False:
                raise ValueError("Negative claims require separate adjudicated evaluation")
            reviewed[wid] = record
    if set(reviewed) != set(windows):
        raise ValueError("Incomplete source-review denominator")
    bind(REFERENCE)
    reference = read(REFERENCE)
    if reference["sample_count"] != 11 or len(reference["samples"]) != 11:
        raise ValueError("Keep the declared11-sample provisional denominator")
    for path, digest in reference["source_reviews_sha256"].items():
        bind(path, digest)
    for name in ("scripts/analyze_accuracy_v40_coverage.py",
                 "scripts/accuracy_v40_coverage_workload.py",
                 "scripts/score_phase20_accuracy.py",
                 "tests/unit/test_accuracy_v40_coverage_workload.py", "docs/accuracy_v40_plan.md"):
        bind(ROOT / name)
    output.mkdir(parents=True)
    # Persist chronology before opening any original journal in this run.
    write(output / "source_review_freeze.json", dict(created_at_utc=now(),
          inputs_sha256=files.copy(), windows=108, native_crop_frames=2160,
          review_precedes_window_workload=True, candidate_policy=None,
          prior_v39_aggregate_results_exposed=True, held_out=False))
    bind(output / "source_review_freeze.json")
    results, matched = [], []
    for clip in manifest["allowlisted_clips"]:
        path = JOURNALS / ("full_repeat0_" + clip) / "frames.jsonl"
        bind(path, audit["checked_files_sha256"][str(path)])
        active = [w for w in manifest["windows"] if w["clip"] == clip]
        needed = {i for w in active for i in range(w["frame_start"], w["frame_end_inclusive"] + 1)}
        selected = {}
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["frame_index"] != count:
                    raise ValueError("Noncontiguous saved journal")
                if count in needed:
                    selected[count] = row
                count += 1
        if count != manifest["source_claims"][clip]["declared_frame_count"] or set(selected) != needed:
            raise ValueError("Changed source frame scope")
        for w in active:
            result = summarize_window(w, [selected[i] for i in range(w["frame_start"], w["frame_end_inclusive"] + 1)])
            results.append(dict(clip_id=clip, source_review=reviewed[w["window_id"]], **result))
            for frame in result["frames"]:
                samples = [s for s in reference["samples"] if s["window_id"] == w["window_id"]
                           and s["frame_index"] == frame["frame_index"]]
                if not samples:
                    continue
                truth = [dict(xy=s["source_xy"], radius=s["position_uncertainty_px"] + 2,
                              polarity=s["polarity"]) for s in samples]
                measured = [dict(id=f'{t["segment"]}/{t["track_id"]}', xy=t["source_xy"],
                                 polarity=t["track_id"].split(":")[0], qualified=t["qualified_moving"])
                            for t in frame["actual_measurements"]]
                stages = dict(candidate=[dict(id=f'candidate:{c["candidate_index"]}',
                                              xy=c["source_xy"], polarity=c["polarity"])
                                         for c in frame["candidates"]],
                              actual_measurement=measured,
                              strict_qualified_measurement=[t for t in measured if t["qualified"]])
                evidence = [dict(**s, stages={}) for s in samples]
                for name, values in stages.items():
                    assignments, neighbors = assign_one_to_one(truth, values)
                    for i, e in enumerate(evidence):
                        e["stages"][name] = dict(hit=i in assignments,
                            assigned_id=values[assignments[i]]["id"] if i in assignments else None,
                            all_gated_ids=[values[j]["id"] for j in neighbors[i]])
                matched.extend(evidence)
    totals = Counter()
    by_clip = {}
    for clip in manifest["allowlisted_clips"]:
        counts = Counter()
        for result in results:
            if result["clip_id"] == clip:
                counts.update(result["counts"])
        by_clip[clip] = dict(counts)
        totals.update(counts)
    summary = dict(completed=True, created_at_utc=now(), windows=len(results), crop_frames=2160,
                   totals=dict(totals), by_clip=by_clip,
                   output_states_are_false_positives=False, authoritative_negative_exposure=0,
                   airborne_accuracy_established=False, candidate_policy=None, production_changed=False)
    if len(matched) != 11:
        raise ValueError("Changed provisional sample denominator")
    summary["provisional_visible_reference"] = dict(samples=11,
        hits={stage:sum(s["stages"][stage]["hit"] for s in matched)
              for stage in ("candidate", "actual_measurement", "strict_qualified_measurement")},
        airborne_truth=False, physical_identity_established=False)
    write(output / "window_evidence.json", results)
    write(output / "visible_reference_evidence.json", matched)
    write(output / "summary.json", summary)
    for name in ("window_evidence.json", "visible_reference_evidence.json", "summary.json"):
        bind(output / name)
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Changed bound file: " + path)
    write(output / "completion_receipt.json", dict(completed=True, all_bound_files_rehashed=True,
                                                    inputs_outputs_sha256=files))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviews", nargs=4, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.reviews, args.output)
