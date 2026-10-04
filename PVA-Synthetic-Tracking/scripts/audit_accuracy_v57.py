"""Independent, metadata-only audit of the V57 temporal output shadow.

The state machine below does not import the candidate policy or runner.  It
reconstructs every decision from unchanged V34 track rows and the previously
saved V36 current-measurement features.  It never opens media or treats reduced
unlabelled workload as a false-positive rate or airborne classification result.
"""
import argparse
from collections import Counter
import copy
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"0029": 687, "0126": 674, "0055": 689, "0082": 691}
MAX_EDGE_GAP_NS = 200_000_000
MAX_COAST_AGE_FRAMES = 7
MAX_COAST_AGE_NS = 700_000_000
CAVEAT = ("Output-only development shadow; detector, association and feedback are "
          "unchanged. Workload is unlabelled, not a false-positive rate. Known "
          "synthetic counterexamples preclude promotion as an airborne classifier.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def _json_float(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite JSON exponent: " + value)
    return result


def decode(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_json_float)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exact(actual, expected, description):
    require(encoded(actual) == encoded(expected), "Independent comparison failed: " + description)


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _xy(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(map(_finite, value))


def _sha_ok(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular non-symlink file required: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class Integrity:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        actual = digest(path)
        require(expected is None or (_sha_ok(expected) and actual == expected),
                "SHA256 mismatch: " + str(path))
        require(self.files.setdefault(str(path), actual) == actual,
                "File changed during audit: " + str(path))
        return actual

    def recheck(self):
        for path, expected in self.files.items():
            require(digest(path) == expected, "File changed during audit: " + path)


def read(path):
    return decode(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            require(bool(line.strip()), "Blank JSONL row: " + str(index))
            value = decode(line)
            require(type(value) is dict, "JSON object row required")
            yield value


def saved_evidence(row, saved):
    """Bind each saved feature to an original qualified track, never to labels."""
    for name in ("frame_index", "timestamp_ns", "segment"):
        exact(saved.get(name), row.get(name), "V36 " + name)
    require(type(saved.get("tracks")) is list and type(row.get("tracks")) is list,
            "Original and V36 track lists required")
    original = [track for track in row["tracks"] if track.get("qualified_moving") is True]
    require(len(original) == len(saved["tracks"]), "Missing or additional V36 qualified track")
    evidence = {}
    for track, old in zip(original, saved["tracks"]):
        require(type(old) is dict, "Saved V36 decision object required")
        for name in ("track_id", "segment", "measured", "source_xy", "measurement_source_xy"):
            exact(old.get(name), track.get(name), "V36 track binding " + name)
        require(type(old.get("accepted")) is bool and type(old.get("reason")) is str
                and bool(old["reason"]), "Malformed V36 verdict")
        if track["measured"]:
            require(type(old.get("measurement_frame")) is int
                    and old["measurement_frame"] == row["frame_index"],
                    "Current saved features lack current measurement provenance")
            key = (track["segment"], track["track_id"])
            require(key not in evidence, "Duplicate saved feature identity")
            evidence[key] = dict(features=copy.deepcopy(old.get("features")), reason=old["reason"])
        else:
            require(old.get("features") is None, "Saved coast cannot supply current pixel features")
            prior = old.get("measurement_frame")
            require(prior is None or (type(prior) is int and 0 <= prior < row["frame_index"]),
                    "Saved coast has future or current measurement provenance")
    return evidence


class IndependentPersistence:
    """List-based reconstruction, separate from the candidate's verdict cache.

    A retained list contains the most recent measured decision and, only while
    uninterrupted, earlier edge measurements.  Predictions never enter it.
    """
    def __init__(self):
        self.frame = -1
        self.timestamp = None
        self.segment = None
        self.history = {}

    def _validate(self, row, evidence):
        require(type(row) is dict and type(row.get("tracks")) is list,
                "Original frame and track list required")
        require(_integer(row.get("frame_index")) and row["frame_index"] == self.frame + 1,
                "Contiguous frames starting at zero required")
        require(_integer(row.get("timestamp_ns")) and
                (self.timestamp is None or row["timestamp_ns"] > self.timestamp),
                "Strictly increasing nonnegative integer timestamp required")
        require(_integer(row.get("segment")) and type(row.get("motion")) is dict
                and type(row["motion"].get("reset")) is bool,
                "Explicit segment and motion-reset required")
        identities, measured = set(), set()
        for item in row["tracks"]:
            require(type(item) is dict, "Track object required")
            identity = item.get("track_id")
            parts = identity.split(":", 1) if type(identity) is str else []
            require(len(parts) == 2 and parts[0] in ("bright", "dark") and bool(parts[1])
                    and _integer(item.get("segment")) and item["segment"] == row["segment"],
                    "Current-segment polarity-bearing identity required")
            key = (item["segment"], identity)
            require(key not in identities, "Duplicate original track identity")
            identities.add(key)
            require(type(item.get("measured")) is bool and
                    type(item.get("qualified_moving")) is bool, "Boolean original evidence required")
            require(_xy(item.get("measurement_source_xy")) if item["measured"] else
                    item.get("measurement_source_xy") is None, "Invalid actual/predicted coordinate")
            if item["measured"] and item["qualified_moving"]:
                measured.add(key)
        require(type(evidence) is dict, "Qualified current-feature mapping required")
        require(all(type(key) is tuple and len(key) == 2 and _integer(key[0])
                    and type(key[1]) is str for key in evidence), "Invalid evidence key")
        require(set(evidence) == measured, "Features must cover exactly qualified actual measurements")
        for observation in evidence.values():
            require(type(observation) is dict and "features" in observation
                    and type(observation.get("reason")) is str and bool(observation["reason"]),
                    "Explicit saved feature and reason required")
            features = observation["features"]
            if features is None:
                require(observation["reason"].startswith("unknown_"), "Missing features must be unknown")
            else:
                require(type(features) is dict and type(features.get("informative")) is bool
                        and _finite(features.get("point_minus_edge_fraction")), "Finite signed feature required")
        return identities

    def step(self, row, evidence):
        identities = self._validate(row, evidence)
        frame, timestamp, segment = (row[k] for k in ("frame_index", "timestamp_ns", "segment"))
        # Reconstruct with private copies and commit only after all decisions.
        histories = ({} if row["motion"]["reset"] or segment != self.segment else
                     {key: copy.deepcopy(value) for key, value in self.history.items() if key in identities})
        answer = []
        for track in row["tracks"]:
            key = (segment, track["track_id"])
            if not track["qualified_moving"]:
                histories.pop(key, None)
                continue
            decision = {
                "segment": segment, "track_id": track["track_id"], "measured": track["measured"],
                "baseline_qualified": True, "accepted": True, "reason": None, "tier": None,
                "measurement_source_xy": copy.deepcopy(track["measurement_source_xy"]),
                "features": None, "evidence_informative": None,
                "measurement_frame": None, "measurement_timestamp_ns": None,
                "evidence_age_frames": None, "evidence_age_ns": None,
                "edge_streak_count": 0, "edge_streak_start_frame": None,
                "edge_streak_start_timestamp_ns": None, "inherited": False,
                "inherited_reason": None, "inherited_tier": None,
                "expired_measurement_frame": None, "expired_measurement_timestamp_ns": None,
                "physical_class": "unknown", "airborne_confirmed": False,
            }
            history = histories.get(key, [])
            if not track["measured"]:
                decision["tier"] = "prediction_unknown"
                decision["reason"] = "unknown_missing_history"
                if history:
                    last = history[-1]
                    frame_age = frame - last["measurement_frame"]
                    time_age = timestamp - last["measurement_timestamp_ns"]
                    decision["evidence_age_frames"] = frame_age
                    decision["evidence_age_ns"] = time_age
                    if 1 <= frame_age <= MAX_COAST_AGE_FRAMES and 0 < time_age <= MAX_COAST_AGE_NS:
                        decision["accepted"] = last["accepted"]
                        decision["reason"] = "coast_" + last["reason"]
                        decision["tier"] = "prediction_inherited"
                        decision["inherited"] = True
                        decision["inherited_reason"] = last["reason"]
                        decision["inherited_tier"] = last["tier"]
                        decision["measurement_frame"] = last["measurement_frame"]
                        decision["measurement_timestamp_ns"] = last["measurement_timestamp_ns"]
                        # Retain a measured origin, but mark its edge run broken.
                        last = copy.deepcopy(last)
                        last["edge_streak_count"] = 0
                        histories[key] = [last]
                    else:
                        decision["reason"] = "unknown_expired_history"
                        decision["expired_measurement_frame"] = last["measurement_frame"]
                        decision["expired_measurement_timestamp_ns"] = last["measurement_timestamp_ns"]
                        histories.pop(key, None)
            else:
                observed = evidence[key]
                features = copy.deepcopy(observed["features"])
                decision["features"] = features
                decision["evidence_informative"] = None if features is None else features["informative"]
                decision["measurement_frame"] = frame
                decision["measurement_timestamp_ns"] = timestamp
                decision["evidence_age_frames"] = 0
                decision["evidence_age_ns"] = 0
                if features is None or not features["informative"]:
                    decision["reason"] = observed["reason"] if features is None else "unknown_uninformative_patch"
                    decision["tier"] = "unknown_measured"
                    history = []
                elif features["point_minus_edge_fraction"] > 0:
                    decision["reason"] = "point_preferred"
                    decision["tier"] = "point_supported_measured"
                    history = []
                else:
                    if not history or history[-1]["edge_streak_count"] == 0 or \
                            history[-1]["measurement_frame"] != frame - 1 or \
                            timestamp - history[-1]["measurement_timestamp_ns"] > MAX_EDGE_GAP_NS:
                        history = []
                    decision["edge_streak_count"] = len(history) + 1
                    decision["edge_streak_start_frame"] = history[0]["measurement_frame"] if history else frame
                    decision["edge_streak_start_timestamp_ns"] = history[0]["measurement_timestamp_ns"] if history else timestamp
                    decision["accepted"] = not history
                    decision["reason"] = "edge_first_or_interrupted" if not history else "edge_consecutive_rejected"
                    decision["tier"] = "edge_pending_measured" if not history else "edge_suppressed_measured"
                histories[key] = history + [copy.deepcopy(decision)]
            answer.append(decision)
        self.history = histories
        self.frame, self.timestamp, self.segment = frame, timestamp, segment
        return answer


def audit_clip_rows(original_rows, feature_rows, decision_rows, expected_frames):
    """Independently compare complete streams; iterables make generated tests cheap."""
    require(type(expected_frames) is int and expected_frames > 0, "Positive frame count required")
    engine = IndependentPersistence()
    counts = {arm: {"measured": 0, "predicted": 0, "identities": set()}
              for arm in ("baseline", "v36_point_context", "v57_persistence")}
    reasons, tiers = Counter(), Counter()
    rejected_current, rejected_predictions, recovered_current = 0, 0, 0
    total = 0
    sentinel = object()
    for row, saved, emitted in zip_longest(original_rows, feature_rows, decision_rows, fillvalue=sentinel):
        require(all(item is not sentinel for item in (row, saved, emitted)), "Unequal full-journal stream lengths")
        require(total < expected_frames, "Extra frame beyond frozen clip count")
        evidence = saved_evidence(row, saved)
        independent = engine.step(row, evidence)
        expected = {k: row[k] for k in ("frame_index", "timestamp_ns", "segment")}
        expected["tracks"] = independent
        exact(emitted, expected, "frame decision " + str(total))
        qualified = [track for track in row["tracks"] if track["qualified_moving"]]
        for original, old, new in zip(qualified, saved["tracks"], independent):
            kind = "measured" if original["measured"] else "predicted"
            key = (original["segment"], original["track_id"])
            for arm, accepted in (("baseline", True), ("v36_point_context", old["accepted"]),
                                  ("v57_persistence", new["accepted"])):
                if accepted:
                    counts[arm][kind] += 1
                    counts[arm]["identities"].add(key)
            reasons[new["reason"]] += 1
            tiers[new["tier"]] += 1
            rejected_current += original["measured"] and not new["accepted"]
            rejected_predictions += not original["measured"] and not new["accepted"]
            recovered_current += original["measured"] and new["accepted"] and not old["accepted"]
        total += 1
    require(total == expected_frames, "Incomplete frozen clip")
    summary = {arm: dict(qualified_measured_states=data["measured"],
                         qualified_predicted_states=data["predicted"],
                         distinct_segment_track_ids=len(data["identities"]))
               for arm, data in counts.items()}
    return dict(frames=total, exact_decisions=True, arms=summary,
                decision_reasons=dict(sorted(reasons.items())), decision_tiers=dict(sorted(tiers.items())),
                rejected_measured_states=rejected_current, rejected_predicted_states=rejected_predictions,
                accepted_measured_states_rejected_by_v36=recovered_current,
                output_only_shadow=True, promotion_allowed=False, production_changed=False,
                airborne_precision=None, airborne_recall=None, false_alarms_per_minute=None)


def _relative_file(root, relative):
    require(type(relative) is str and bool(relative), "Relative metadata path required")
    component = Path(relative)
    require(not component.is_absolute() and ".." not in component.parts
            and component.suffix in (".json", ".jsonl", ".py", ".md", ".log"),
            "Only safe relative metadata/code paths may be audited")
    target = root
    for part in component.parts:
        target = target / part
        require(not target.is_symlink(), "Symlink path component refused")
    require(target.is_file(), "Missing bound metadata: " + relative)
    return target


def audit(output_dir, repo_root=ROOT):
    output, repo = Path(output_dir), Path(repo_root)
    require(output.is_dir() and not output.is_symlink() and repo.is_dir(), "Regular output/repository directories required")
    integrity = Integrity()
    freeze_path = output / "freeze.json"
    freeze_sha = integrity.bind(freeze_path)
    frozen = read(freeze_path)
    require(frozen.get("schema") == "seaqr.accuracy_v57.freeze.v1", "Unknown V57 freeze schema")
    exact(frozen.get("config"), dict(required_consecutive_edges=2, maximum_edge_gap_ns=MAX_EDGE_GAP_NS,
                                  maximum_coast_frames=MAX_COAST_AGE_FRAMES, maximum_coast_ns=MAX_COAST_AGE_NS),
          "frozen persistence policy")
    require(frozen.get("production_changed") is False and frozen.get("promotion_allowed") is False,
            "V57 is an explicitly nonpromoted output-only experiment")
    for name in ("inputs", "implementation"):
        registry = frozen.get(name)
        require(type(registry) is dict and bool(registry), "Nonempty frozen " + name + " map required")
        for relative, expected in registry.items():
            integrity.bind(_relative_file(repo, relative), expected)
            if name == "implementation":
                integrity.bind(_relative_file(output, "implementation/" + relative), expected)
    auditor_relative = "scripts/audit_accuracy_v57.py"
    require(frozen["implementation"].get(auditor_relative) == digest(__file__),
            "Executing independent auditor not bound to freeze")
    summary = None
    summary_path = output / "summary.json"
    if summary_path.exists():
        integrity.bind(summary_path)
        summary = read(summary_path)
        require(summary.get("freeze_sha256") == freeze_sha, "Summary bound to another freeze")
        for name in ("production_changed", "promotion_allowed"):
            require(summary.get(name) is False, "Nonproduction summary guard changed: " + name)
        require(summary.get("classifier_promoted", False) is False,
                "Classifier promotion is prohibited")
        require(summary.get("known_counterexample_rejected") is True
                and summary.get("known_counterexample_blocks_promotion") is True,
                "Known counterexample must remain explicit in summary")
    clips = frozen.get("clips")
    require(type(clips) is dict and set(clips) == set(COUNTS), "Only original four development clips allowed")
    per_clip = {}
    for clip, count in COUNTS.items():
        spec = clips[clip]
        require(type(spec) is dict and set(spec) == {"journal", "features", "frames"}, "Changed clip metadata schema")
        exact(spec["frames"], count, "frozen complete clip count")
        exact(spec["journal"], "results/tiny_target/visible_validation_v34_20260923/"
              "audit_20260924/evidence/run/full_repeat0_" + clip + "/frames.jsonl", "original V34 journal path")
        exact(spec["features"], "results/tiny_target/accuracy_v36_20260924/full_context_01/"
              + clip + "_decisions.jsonl", "saved V36 feature path")
        require(spec["journal"] in frozen["inputs"] and spec["features"] in frozen["inputs"],
                "Unbound original journal or saved features")
        journal_path = _relative_file(repo, spec["journal"])
        feature_path = _relative_file(repo, spec["features"])
        decisions_path = output / (clip + "_decisions.jsonl")
        integrity.bind(decisions_path)
        per_clip[clip] = audit_clip_rows(jsonl(journal_path), jsonl(feature_path), jsonl(decisions_path), count)
        if summary is not None:
            reported = summary.get("clips", {}).get(clip)
            require(type(reported) is dict, "Missing per-clip summary")
            exact(reported.get("frames"), count, "summary complete clip count")
            expected_workload = {}
            for arm, alias in (("baseline", "baseline"), ("v36_point_context", "v36"),
                               ("v57_persistence", "v57")):
                counts = per_clip[clip]["arms"][arm]
                expected_workload[alias] = dict(measured=counts["qualified_measured_states"],
                                               predicted=counts["qualified_predicted_states"],
                                               distinct_identities=counts["distinct_segment_track_ids"])
            exact(reported.get("workload"), expected_workload, "summary workload " + clip)
            exact(reported.get("tiers"), per_clip[clip]["decision_tiers"], "summary tiers " + clip)
            exact(reported.get("reasons"), per_clip[clip]["decision_reasons"], "summary reasons " + clip)
    if summary is not None:
        require(summary.get("completed") is True, "Summary must be complete")
        exact(summary.get("frames"), sum(COUNTS.values()), "summary full frame count")
        require(set(summary["clips"]) == set(COUNTS), "Summary clip inventory changed")
    stress_path = output / "stress.json"
    if stress_path.exists():
        integrity.bind(stress_path)
        stress = read(stress_path)
        require(stress.get("completed") is True and stress.get("production_changed") is False
                and stress.get("promotion_allowed") is False
                and stress.get("known_counterexample_rejected") is True
                and stress.get("known_point_on_strong_edge_counterexample_reproduced") is True,
                "Known generated counterexample must remain nonpromoted")
        known = [case for case in stress.get("cases", [])
                 if case.get("case_id") == "point_on_strong_edge_bright"]
        require(len(known) == 1 and known[0].get("analytic_localized_source_present") is True,
                "Missing known-present analytic point-on-edge counterexample")
        case = known[0]
        require(type(case.get("sequence")) is list and len(case["sequence"]) == 4,
                "Four generated counterexample observations required")
        machine = IndependentPersistence()
        accepted = []
        for item in case["sequence"]:
            original = dict(frame_index=item["frame_index"], timestamp_ns=item["timestamp_ns"],
                            segment=0, motion=dict(reset=False), tracks=[item["assumed_baseline_track"]])
            evidence = {(0, item["assumed_baseline_track"]["track_id"]):
                        dict(features=case["generated_features"], reason="edge_preferred_or_tie")}
            expected = machine.step(original, evidence)[0]
            exact(item["verdict"], expected, "generated known-present counterexample verdict")
            accepted.append(expected["accepted"])
        exact(accepted, [True, False, False, False], "known source veto pattern")
    receipt_path = output / "completion_receipt.json"
    if receipt_path.exists():
        integrity.bind(receipt_path)
        receipt = read(receipt_path)
        require(receipt.get("completed") is True and receipt.get("production_changed") is False
                and receipt.get("promotion_allowed") is False
                and receipt.get("all_frozen_inputs_and_code_rehashed") is True,
                "Incomplete or promoted completion receipt")
        expected_names = {"freeze.json", "stress.json", "summary.json"} | {
            clip + "_decisions.jsonl" for clip in COUNTS}
        require(type(receipt.get("outputs_sha256")) is dict
                and set(receipt["outputs_sha256"]) == expected_names, "Completion output inventory changed")
        for relative, expected in receipt["outputs_sha256"].items():
            integrity.bind(_relative_file(output, relative), expected)
    integrity.recheck()
    return dict(schema="seaqr.accuracy_v57.independent_audit.v1", passed=True,
                freeze_sha256=freeze_sha, frames=sum(item["frames"] for item in per_clip.values()),
                clips=per_clip, checked_files_sha256=integrity.files,
                exact_decisions=True, inputs_unchanged=True, output_only_shadow=True,
                production_changed=False, promotion_allowed=False, known_synthetic_counterexample=True,
                classifier_promoted=False, airborne_precision=None, airborne_recall=None,
                false_alarms_per_minute=None, caveat=CAVEAT,
                summary_workload_verified=summary is not None,
                reference_retention_and_control_scoring_independently_verified=False,
                saved_pixel_features_independently_refitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory, args.repo_root)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
