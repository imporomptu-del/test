"""Metadata-only regression intake checks; no media, detector or accuracy scoring."""

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "configs/evaluation/discovery_pair_regression_20260929.json"
EVIDENCE = ROOT.parent / "outputs/seaqr_discovery_pair_20260928/evidence"
SCHEMA = "seaqr.discovery-pair.regression-intake.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_contract(value):
    """Test-local contract validator, not a production matcher or GT importer."""
    require(value["schema"] == SCHEMA, "schema")
    scope = value["scope"]
    require(scope["allowed_clip_ids"] == ["0170", "0240"], "scope")
    for key in ("raw16_allowed", "sealed_holdout_access_allowed", "production_algorithm_changed",
                "original_reports_modified", "legacy_references_modified", "untouched_validation"):
        require(scope[key] is False, key)
    require(scope["development_only"] is True and scope["prior_detector_exposure"] is True,
            "discovery exposure")
    recordings = value["recordings"]
    require([r["clip_id"] for r in recordings] == ["0170", "0240"], "recordings")
    for recording in recordings:
        require(type(recording["frame_count"]) is int and recording["frame_count"] == 673, "frames")
        require(recording["frame_index_bounds_inclusive"] == [0, 672], "bounds")
        require(recording["native_shape_hw"] == [3190, 4784], "shape")
        require(recording["nominal_fps"] == 10 and recording["decoded_luma_bit_depth"] == 8,
                "nominal input")
        require(re.fullmatch(r"[0-9a-f]{64}", recording["source_sha256"]) is not None, "source hash")
    positive = value["positive_pass"]
    require((positive["clip_id"], positive["first_frame"], positive["last_frame_inclusive"])
            == ("0240", 430, 464), "positive interval")
    require(positive["class_status"] == "user_confirmed_airborne", "class provenance")
    for key in ("independent_per_frame_ground_truth", "independent_physical_identity_ground_truth",
                "recall_estimate_available"):
        require(positive[key] is False, key)
    require(positive["coordinate_provenance"] == "frozen_baseline_actual_measurements", "coordinate provenance")
    require(positive["coordinate_field"] == "tracks[].measurement_source_xy", "measurement field")
    require(positive["baseline_identity"] == {"segment": 13, "polarity": "dark", "track_id": "dark:7238"},
            "baseline identity")
    require(positive["reference_frame_count"] == 35, "reference count")
    anchors = positive["baseline_measurements"]
    require(len(anchors) == 35, "anchor count")
    require([a["frame_index"] for a in anchors] == list(range(430, 465)), "anchor chronology")
    for anchor in anchors:
        require(type(anchor["frame_index"]) is int and type(anchor["timestamp_ns"]) is int, "exact integers")
        require(anchor["timestamp_ns"] == anchor["frame_index"] * 100_000_000, "nominal timestamp")
        xy = anchor["measurement_source_xy"]
        require(len(xy) == 2, "coordinate dimension")
        require(all(type(n) in (int, float) and math.isfinite(n) for n in xy), "finite coordinates")
        require(0 <= xy[0] < 4784 and 0 <= xy[1] < 3190, "native coordinates")
    contract = positive["retention_contract"]
    require(type(contract["radius_native_px"]) is int and contract["radius_native_px"] == 8, "frozen radius")
    for key in ("predeclared_before_candidate_run", "require_qualified", "require_measured", "same_polarity",
                "missing_unready_or_prediction_only_frames_count_as_retention_misses", "not_recall",
                "not_detector_promotion_criterion"):
        require(contract[key] is True, key)
    require(contract["candidate_evidence_field"] == "measurement_source_xy", "candidate actual measurements")
    require(contract["required_polarity"] == "dark" and contract["boundary"] == "inclusive", "gate")
    require(contract["distance"] == "euclidean_native_source_xy", "metric")
    require(contract["required_reference_frame_count"] == 35, "fixed denominator")
    require(contract["identity_fields"] == ["segment", "polarity", "track_id"], "identity scope")
    require(contract["require_baseline_identity_string"] is False, "candidate ID independence")
    require(contract["primary"] == "one_coherent_candidate_identity_covers_all_reference_frames", "coherent primary")
    require(set(contract["secondary_metrics"]) == {
        "any_identity_covered_frame_count", "frames_with_multiple_qualifying_identities",
        "qualifying_identity_count_by_frame", "coherent_identity_covered_frame_counts"}, "ambiguity metrics")
    expected = [("0170", 0, 672), ("0240", 50, 105)]
    require(len(value["diagnostic_cases"]) == 2, "diagnostic count")
    for case, bounds in zip(value["diagnostic_cases"], expected):
        require((case["clip_id"], case["first_frame"], case["last_frame_inclusive"]) == bounds, "diagnostic interval")
        require(case["frame_count"] == bounds[2] - bounds[1] + 1, "diagnostic denominator")
        require(case["verified_airborne_negative"] is False and case["false_positive_rate_available"] is False,
                "unverified negatives")
        require(case["class_status"] == "unknown" and case["pass_fail_thresholds"] is None, "diagnostic only")
        require(case["baseline"]["ready_frame_count"] + case["baseline"]["unready_frame_count"]
                == case["frame_count"], "availability denominator")
    for binding in all_bindings(value):
        require(re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is not None, "artifact hash")
        require(Path(binding["path"]).suffix in (".json", ".jsonl"), "metadata only")


def all_bindings(value):
    yield from value["metadata_bindings"].values()
    yield from value["legacy_regression"]["references"]
    for recording in value["recordings"]:
        yield from recording["artifacts"].values()


def metadata_hash(path):
    require(path.suffix in (".json", ".jsonl"), "refuse media reads")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IntakeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(INTAKE.read_text())

    def test_intake_contract(self):
        validate_contract(self.value)

    def test_actual_measurement_coordinates_not_filtered_positions(self):
        rows = self.value["positive_pass"]["baseline_measurements"]
        self.assertEqual(rows[0]["measurement_source_xy"], [2719.6782899774016, 2845.7376550513654])
        self.assertEqual(rows[-1]["measurement_source_xy"], [3051.5895458666837, 2696.672527548792])
        self.assertNotEqual(rows[-1]["measurement_source_xy"], [3050.116614156623, 2697.604717077921])

    def test_class_and_limited_source_review_provenance(self):
        positive = self.value["positive_pass"]
        self.assertEqual(positive["class_provenance"]["reviewer"], "user")
        self.assertTrue(positive["class_provenance"]["prior_detector_or_overlay_exposure"])
        review = positive["source_only_review"]
        self.assertEqual(review["reported_review_frames"], [430, 441, 453, 464])
        self.assertFalse(review["independent_per_frame_positions_available"])
        self.assertFalse(review["exhaustive_interval_source_review"])
        self.assertEqual(review["coarse_scene_checks"]["status"], "not_annotated")

    def test_original_failed_batch_is_not_reclassified(self):
        history = self.value["execution_history"]
        self.assertTrue(history["both_inference_receipts_passed"])
        self.assertFalse(history["remote_batch_passed"])
        self.assertTrue(history["original_delivery_precedes_this_user_class_feedback"])

    def test_previous_four_clip_references_unchanged(self):
        legacy = self.value["legacy_regression"]
        self.assertEqual(legacy["clip_ids"], ["0029", "0126", "0055", "0082"])
        self.assertTrue(legacy["unchanged"])
        self.assertTrue(legacy["new_intake_does_not_relabel_legacy_panels"])
        for binding in legacy["references"]:
            self.assertEqual(metadata_hash(ROOT / binding["path"]), binding["sha256"])

    def test_reject_missing_duplicate_reordered_anchors(self):
        for operation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, rows[0]),
                          lambda rows: rows.reverse()):
            value = copy.deepcopy(self.value)
            operation(value["positive_pass"]["baseline_measurements"])
            with self.assertRaises(ValueError):
                validate_contract(value)

    def test_reject_invalid_coordinates_and_timestamps(self):
        for key, bad in [("measurement_source_xy", [math.nan, 0]), ("measurement_source_xy", [0, math.inf]),
                         ("measurement_source_xy", [True, 0]), ("measurement_source_xy", [-1, 0]),
                         ("measurement_source_xy", [4784, 0]), ("measurement_source_xy", [0, 3190]),
                         ("timestamp_ns", 43000000000.0), ("timestamp_ns", 43000000001)]:
            value = copy.deepcopy(self.value)
            value["positive_pass"]["baseline_measurements"][0][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_contract(value)

    def test_reject_recall_or_ground_truth_claims(self):
        for key in ("independent_per_frame_ground_truth", "independent_physical_identity_ground_truth", "recall_estimate_available"):
            value = copy.deepcopy(self.value)
            value["positive_pass"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(value)

    def test_reject_coasts_wrong_gate_or_identity_shortcuts(self):
        for key, bad in [("require_measured", False), ("require_qualified", False), ("radius_native_px", 9),
                         ("same_polarity", False), ("require_baseline_identity_string", True),
                         ("candidate_evidence_field", "source_xy"), ("required_reference_frame_count", 34),
                         ("identity_fields", ["track_id"])]:
            value = copy.deepcopy(self.value)
            value["positive_pass"]["retention_contract"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(value)

    def test_reject_negative_labels_on_diagnostic_intervals(self):
        for index in range(2):
            value = copy.deepcopy(self.value)
            value["diagnostic_cases"][index]["verified_airborne_negative"] = True
            with self.assertRaises(ValueError):
                validate_contract(value)

    def test_burst_mixed_scene_review_is_warning_not_ground_truth(self):
        warning = self.value["diagnostic_cases"][1]["mixed_scene_warning"]
        self.assertFalse(warning["entire_burst_is_noise"])
        self.assertFalse(warning["suppress_all_post_reset_outputs_justified"])
        self.assertFalse(warning["manual_ground_truth_coordinates_created"])

    def test_reject_scope_expansion_or_malformed_hash(self):
        value = copy.deepcopy(self.value)
        value["scope"]["raw16_allowed"] = True
        with self.assertRaises(ValueError):
            validate_contract(value)
        value = copy.deepcopy(self.value)
        value["recordings"][0]["source_sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            validate_contract(value)

    def test_metadata_reader_refuses_media(self):
        with self.assertRaisesRegex(ValueError, "refuse media"):
            metadata_hash(Path("not-opened.avi"))


@unittest.skipUnless(EVIDENCE.is_dir(), "Immutable local discovery evidence is not distributed with the test suite")
class BoundEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(INTAKE.read_text())

    def test_all_bound_metadata_hashes(self):
        for binding in all_bindings(self.value):
            with self.subTest(path=binding["path"]):
                self.assertEqual(metadata_hash(ROOT / binding["path"]), binding["sha256"])

    def test_native_journals_and_exact_baseline_anchors(self):
        positive = self.value["positive_pass"]
        wanted = {row["frame_index"]: row for row in positive["baseline_measurements"]}
        found = []
        for recording in self.value["recordings"]:
            diagnostic = next(c for c in self.value["diagnostic_cases"] if c["clip_id"] == recording["clip_id"])
            counts = dict(ready_frame_count=0, unready_frame_count=0, reset_count=0,
                          candidate_count=0, qualified_measurement_count=0, qualified_prediction_count=0)
            identities, ready, reset, indices = set(), [], [], []
            with (ROOT / recording["artifacts"]["journal"]["path"]).open() as handle:
                for line in handle:
                    row = json.loads(line)
                    index = row["frame_index"]
                    indices.append(index)
                    self.assertEqual(row["timestamp_ns"], index * 100_000_000)
                    self.assertEqual(row["coverage"]["full_shape_hw"], [3190, 4784])
                    self.assertTrue(row["coverage"]["native_pixel_sampling"])
                    if recording["clip_id"] == "0240" and index in wanted:
                        track = [t for t in row["tracks"] if t["segment"] == 13 and t["track_id"] == "dark:7238"]
                        self.assertEqual(len(track), 1)
                        self.assertTrue(track[0]["measured"])
                        self.assertTrue(track[0]["qualified_moving"])
                        self.assertTrue(row["coverage"]["detection_ready"])
                        self.assertFalse(row["motion"]["reset"])
                        self.assertEqual(track[0]["measurement_source_xy"], wanted[index]["measurement_source_xy"])
                        found.append(index)
                    if not diagnostic["first_frame"] <= index <= diagnostic["last_frame_inclusive"]:
                        continue
                    is_ready = row["coverage"]["detection_ready"]
                    counts["ready_frame_count" if is_ready else "unready_frame_count"] += 1
                    if is_ready:
                        ready.append(index)
                    if row["motion"]["reset"]:
                        counts["reset_count"] += 1
                        reset.append(index)
                    counts["candidate_count"] += len(row["candidates"])
                    for track in row["tracks"]:
                        if track["qualified_moving"]:
                            counts["qualified_measurement_count" if track["measured"] else "qualified_prediction_count"] += 1
                            identities.add((row["segment"], track["track_id"]))
            self.assertEqual(indices, list(range(673)))
            counts["qualified_identity_count"] = len(identities)
            for key, number in counts.items():
                self.assertEqual(number, diagnostic["baseline"][key], (recording["clip_id"], key))
            if "ready_frames" in diagnostic["baseline"]:
                self.assertEqual(ready, diagnostic["baseline"]["ready_frames"])
            if "reset_frames" in diagnostic["baseline"]:
                self.assertEqual(reset, diagnostic["baseline"]["reset_frames"])
        self.assertEqual(found, list(range(430, 465)))


if __name__ == "__main__":
    unittest.main()
