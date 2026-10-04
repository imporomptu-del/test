"""V38 extraction/selection tests using synthetic journals and fake decoders.

Temporary files contain generated fixtures only. These tests never open real
source media, inspect truth outcomes, or alter any frozen experiment.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import diagnose_accuracy_v38_global_multilag as runner


def track(identity="bright:1", xy=(33., 33.), *, measured=True, qualified=True,
          segment=0, accepted=False):
    return dict(track_id=identity, segment=segment, measured=measured,
                qualified_moving=qualified, measurement_source_xy=list(xy),
                source_xy=[999., 999.], accepted=accepted)


def row(frame, tracks=()):
    return dict(frame_index=frame, timestamp_ns=frame*100_000_000, segment=0,
                motion=dict(reset=False, accepted=True),
                source_to_reference=np.eye(3).tolist(), tracks=copy.deepcopy(list(tracks)))


def observation(frame, identity="0/bright:1", *, cid="0029", xy=(33., 33.), controls=()):
    return dict(key=f"{cid}/{frame}/{identity}", clip=cid, frame=frame, identity=identity,
                measurement_source_xy=list(xy), polarity="bright",
                provisional_control_indices=list(controls))


def reference(visible=(12, 13, 14), ambiguous=(15,)):
    return dict(clip="0029", physical_class="unknown", frames=[
        dict(frame_index=f, visibility="visible" if f in visible else "ambiguous" if f in ambiguous else "not_visible",
             source_xy=[100., 200.] if f in visible else None,
             position_uncertainty_radius_px=1. if f in visible else None)
        for f in range(12, 25)])


def selection_with_old_denominators():
    observations = [observation(100+i, cid="0126") for i in range(285) if i != 116]
    groups = []
    for kind, count in (("dense", 285), ("pilot", 28), ("anchor", 24)):
        for i in range(count):
            key = f"0126/{100+i}/0/bright:1"
            keys = [] if kind == "dense" and i == 116 else [key]
            groups.append(dict(kind=kind, clip="0126", window="original_"+kind,
                               frame=100+i, keys=keys, baseline_assigned_id="0/bright:1" if keys else None))
    return dict(observations=observations, groups=groups,
                controls=[dict(label="fixture_control_"+str(i)) for i in range(7)])


def minimal_selection(observations=(), groups=(), controls=()):
    return dict(observations=copy.deepcopy(list(observations)),
                groups=copy.deepcopy(list(groups)), controls=copy.deepcopy(list(controls)))


def fake_evidence(*, nominal=True, envelope=True, reason=None, value=.4):
    probes = [dict(reasons=[] if reason is None else [reason],
                   source_supported=True, contrast_informative=nominal,
                   conditional_pair=dict(available=False, reasons=["missing_previous_actual_measurement"]))
              for _ in range(9)]
    return dict(nominal_contrast_available=nominal, envelope_available=envelope,
                source_supported_probes=9, informative_probes=9 if nominal else 0,
                probes=probes,
                envelope={metric:dict(minimum=value, median=value+.1, maximum=value+.2, span=.2)
                          for metric in runner.METRICS} if envelope else None)


def diagnostic_record(item, *, geometric=(1, 2, 4, 8), nominal=(1, 2, 4, 8), envelopes=(1, 2, 4, 8)):
    result = copy.deepcopy(item)
    result["lags"] = [dict(lag=lag, available=lag in geometric,
        previous_actual_measurement_available=False,
        reasons=[] if lag in geometric else ["fixture_geometry_unknown"],
        evidence=fake_evidence(nominal=lag in nominal, envelope=lag in envelopes,
                               reason=None if lag in nominal else "uninformative_difference")
                 if lag in geometric else None) for lag in (1, 2, 4, 8)]
    return result


class FakeCapture:
    """Large declared BGR views have zero strides; no large video is allocated."""
    def __init__(self, count=14, fps=10., opened=True, fail_at=None, bad_shape_at=None, bad_dtype_at=None):
        self.count, self.fps, self.opened = count, fps, opened
        self.fail_at, self.bad_shape_at, self.bad_dtype_at = fail_at, bad_shape_at, bad_dtype_at
        self.next_frame = 0
        self.read_frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, key):
        if key == runner.cv2.CAP_PROP_FPS:
            return self.fps
        if key == runner.cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        raise AssertionError("Unexpected decoder metadata request")

    def read(self):
        frame = self.next_frame
        self.next_frame += 1
        self.read_frames.append(frame)
        if frame == self.fail_at or frame >= self.count:
            return False, None
        shape = (3, 4, 3) if frame == self.bad_shape_at else (3190, 4784, 3)
        dtype = np.float32 if frame == self.bad_dtype_at else np.uint8
        return True, np.broadcast_to(np.asarray(frame, dtype=dtype), shape)

    def getBackendName(self):
        return "synthetic_fake_capture"

    def release(self):
        self.released = True


def fake_gray(bgr, code):
    if code != runner.cv2.COLOR_BGR2GRAY:
        raise AssertionError("Unexpected color conversion")
    return np.full((67, 67), 80+int(bgr[0, 0, 0]), np.uint8)


class AccuracyV38ReferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.selection = selection_with_old_denominators()
        self.reference = reference()
        self.journal = [row(f) for f in range(25)]
        self.decisions = [row(f) for f in range(25)]
        self.journal[12]["tracks"] = [
            track("bright:near", (100.5, 200)), track("bright:far", (102, 200)),
            track("bright:outside", (104, 200)), track("bright:coast", (100, 200), measured=False),
            track("bright:unqualified", (100, 200), qualified=False), track("dark:wrong_polarity", (100, 200))]
        self.decisions[12]["tracks"] = [track("bright:far", (102, 200), accepted=True)]
        self.journal[14]["tracks"] = [track("bright:boundary", (103, 200))]

    def run_selection(self):
        return runner.add_reference(self.selection, self.reference, self.journal, self.decisions)

    def test_old_285_28_24_groups_and_observations_are_unchanged(self):
        original = copy.deepcopy(self.selection)
        result = self.run_selection()
        self.assertEqual(self.selection, original)
        self.assertEqual(result["groups"][:len(original["groups"])], original["groups"])
        by_key = {o["key"]:o for o in result["observations"]}
        for o in original["observations"]:
            self.assertEqual(by_key[o["key"]], o)
        self.assertEqual({kind:sum(g["kind"] == kind for g in result["groups"])
                          for kind in ("dense", "pilot", "anchor")}, dict(dense=285, pilot=28, anchor=24))
        self.assertEqual(result["controls"], original["controls"])

    def test_all_visible_samples_including_unmatched_are_retained(self):
        result = self.run_selection()
        groups = [g for g in result["groups"] if g["kind"] == "compact_light"]
        self.assertEqual([g["frame"] for g in groups], [12, 13, 14])
        self.assertEqual(groups[1]["keys"], [])
        self.assertIsNone(groups[1]["baseline_assigned_id"])
        self.assertEqual(groups[1]["v36_retained_keys"], [])
        self.assertEqual(result["additional_reference"]["visibility_counts"],
                         dict(visible=3, ambiguous=1, not_visible=9))
        self.assertIs(result["additional_reference"]["authoritative_airborne_truth"], False)
        self.assertIs(result["additional_reference"]["independently_held_out"], False)

    def test_all_gated_actual_measurements_not_only_nearest_or_v36_retained_are_selected(self):
        result = self.run_selection()
        group = next(g for g in result["groups"] if g["kind"] == "compact_light" and g["frame"] == 12)
        self.assertEqual(group["keys"], ["0029/12/0/bright:near", "0029/12/0/bright:far"])
        self.assertEqual(group["baseline_assigned_id"], "0/bright:near")
        self.assertEqual(group["v36_retained_keys"], ["0029/12/0/bright:far"])
        self.assertEqual(group["source_reference_xy"], [100., 200.])
        self.assertEqual(group["matching_radius_px"], 3.)
        boundary = next(g for g in result["groups"] if g["kind"] == "compact_light" and g["frame"] == 14)
        self.assertEqual(boundary["keys"], ["0029/14/0/bright:boundary"])

    def test_tracker_motion_changes_match_assignment_not_reference_coordinates(self):
        reference_snapshot = copy.deepcopy(self.reference)
        self.journal[12]["tracks"][0]["measurement_source_xy"] = [140., 200.]
        result = self.run_selection()
        group = next(g for g in result["groups"] if g["kind"] == "compact_light" and g["frame"] == 12)
        self.assertEqual(group["baseline_assigned_id"], "0/bright:far")
        self.assertEqual(group["source_reference_xy"], reference_snapshot["frames"][0]["source_xy"])
        self.assertEqual(self.reference, reference_snapshot)

    def test_v36_acceptance_does_not_choose_reference_or_diagnostic_samples(self):
        first = self.run_selection()
        self.decisions[12]["tracks"] = [track("bright:near", (100.5, 200), accepted=True)]
        second = self.run_selection()
        self.assertEqual(first["observations"], second["observations"])
        a = next(g for g in first["groups"] if g["kind"] == "compact_light" and g["frame"] == 12)
        b = next(g for g in second["groups"] if g["kind"] == "compact_light" and g["frame"] == 12)
        self.assertEqual(a["keys"], b["keys"])
        self.assertEqual(a["source_reference_xy"], b["source_reference_xy"])
        self.assertNotEqual(a["v36_retained_keys"], b["v36_retained_keys"])

    def test_matching_reuses_original_observation_without_duplicate(self):
        existing = observation(12, "0/bright:near", xy=(100.5, 200.), controls=(2, 5))
        self.selection["observations"].append(existing)
        result = self.run_selection()
        self.assertEqual(sum(o["key"] == existing["key"] for o in result["observations"]), 1)
        self.assertEqual(next(o for o in result["observations"] if o["key"] == existing["key"]), existing)

    def test_reference_domain_and_visible_location_validation(self):
        for field, value in (("clip", "0055"), ("physical_class", "airborne")):
            altered = copy.deepcopy(self.reference)
            altered[field] = value
            with self.assertRaises(ValueError):
                runner.add_reference(self.selection, altered, self.journal, self.decisions)
        for field, value in (("source_xy", [np.nan, 200]), ("source_xy", [100]),
                             ("position_uncertainty_radius_px", 0),
                             ("position_uncertainty_radius_px", True),
                             ("position_uncertainty_radius_px", np.inf), ("visibility", "negative")):
            altered = copy.deepcopy(self.reference)
            altered["frames"][0][field] = value
            with self.assertRaises(ValueError):
                runner.add_reference(self.selection, altered, self.journal, self.decisions)
        altered = copy.deepcopy(self.reference)
        altered["frames"][0]["frame_index"] = 11
        with self.assertRaises(ValueError):
            runner.add_reference(self.selection, altered, self.journal, self.decisions)


class AccuracyV38ExtractionTests(unittest.TestCase):
    def extract(self, *, frames=(2, 8, 10), cap=None, mutate_rows=None, originals=None):
        cap = FakeCapture() if cap is None else cap
        rows = [row(f, [track()]) for f in range(14)]
        if mutate_rows:
            mutate_rows(rows)
        observations = [observation(f) for f in frames]
        if originals is None:
            originals = {o["key"]:hashlib.sha256(np.full((25, 25), 80+o["frame"], np.uint8).tobytes()).hexdigest()
                         for o in observations}
        arrays, records = {}, []
        with mock.patch.object(runner.cv2, "VideoCapture", return_value=cap) as constructor, \
             mock.patch.object(runner.cv2, "cvtColor", side_effect=fake_gray), \
             mock.patch.dict(runner.COUNTS, {"0029":14}), \
             mock.patch.dict(runner.sha_cache, {"0029":"fixture_source_sha"}):
            result = runner.extract_clip("0029", Path("synthetic-not-a-real-video"), rows,
                                         observations, originals, arrays, records)
        constructor.assert_called_once_with("synthetic-not-a-real-video")
        return result, arrays, records, cap

    def test_sequential_decode_and_exact_fixed_lag_arrays_never_use_future_frames(self):
        result, arrays, records, cap = self.extract()
        self.assertEqual(cap.read_frames, list(range(11)))
        self.assertTrue(cap.released)
        self.assertEqual(result["decoded_frames"], 11)
        self.assertEqual(result["last_frame"], 10)
        self.assertEqual([r["frame"] for r in records], [2, 8, 10])
        for record in records:
            frame = record["frame"]
            self.assertEqual(record["identity"], "0/bright:1")
            self.assertEqual(record["current_frame"]["frame_index"], frame)
            self.assertTrue(record["v36_native_patch_crosschecked"])
            self.assertLessEqual(record["buffer_scope"]["frames"], 9)
            self.assertEqual([lag["lag"] for lag in record["lags"]], [1, 2, 4, 8])
            np.testing.assert_array_equal(arrays[record["current25_array"]["array_key"]],
                                          np.full((25, 25), 80+frame, dtype=np.float64))
            for lag in record["lags"]:
                expected = frame-lag["lag"]
                self.assertEqual(lag["requested_prior_frame_index"], expected)
                self.assertTrue(all(meta["frame_index"] <= frame for meta in lag["intervening_frames"]))
                if expected >= 0:
                    self.assertTrue(lag["available"])
                    self.assertEqual(lag["prior_frame"]["frame_index"], expected)
                    np.testing.assert_array_equal(arrays[lag["prior27_array"]["array_key"]],
                                                  np.full((27, 27), 80+expected, dtype=np.float64))
                else:
                    self.assertFalse(lag["available"])
                    self.assertIsNone(lag["prior27_array"])
                    self.assertIsNone(lag["evidence"])
        self.assertEqual(records[-1]["buffer_scope"]["oldest_frame"], 2)
        json.dumps(records, allow_nan=False)

    def test_missing_previous_measurement_does_not_prevent_source_extraction(self):
        def remove_prior(rows):
            for r in rows[:10]:
                r["tracks"] = []
        _, _, records, _ = self.extract(frames=(10,), mutate_rows=remove_prior)
        for lag in records[0]["lags"]:
            self.assertTrue(lag["available"])
            self.assertFalse(lag["previous_actual_measurement_available"])
            self.assertIsNotNone(lag["prior27_array"])
            self.assertIsNotNone(lag["evidence"])

    def test_reference_reset_keeps_unknown_lag_records_without_reusing_old_geometry(self):
        def reset(rows):
            rows[7]["motion"]["reset"] = True
        _, _, records, _ = self.extract(frames=(8,), mutate_rows=reset)
        lags = records[0]["lags"]
        self.assertTrue(lags[0]["available"])
        for lag in lags[1:]:
            self.assertFalse(lag["available"])
            self.assertIn("intervening_reference_reset", lag["reasons"])
            self.assertIsNone(lag["evidence"])

    def test_bad_frozen_current_native_hash_fails_and_releases_capture(self):
        cap = FakeCapture()
        with self.assertRaisesRegex(ValueError, "native pixels changed"):
            self.extract(frames=(2,), cap=cap, originals={observation(2)["key"]:"wrong_hash"})
        self.assertTrue(cap.released)
        self.assertEqual(cap.read_frames, [0, 1, 2])

    def test_additional_observation_is_explicitly_not_original_patch_crosschecked(self):
        _, _, records, _ = self.extract(frames=(2,), originals={})
        self.assertFalse(records[0]["v36_native_patch_crosschecked"])

    def test_bad_decoder_metadata_fails_before_any_read_and_releases(self):
        for cap in (FakeCapture(fps=9.9), FakeCapture(count=13), FakeCapture(opened=False)):
            with self.assertRaisesRegex(ValueError, "decoder metadata"):
                self.extract(cap=cap)
            self.assertTrue(cap.released)
            self.assertEqual(cap.read_frames, [])

    def test_decode_failure_bad_shape_and_bad_dtype_are_fatal_and_release(self):
        for cap in (FakeCapture(fail_at=1), FakeCapture(bad_shape_at=1), FakeCapture(bad_dtype_at=1)):
            with self.assertRaisesRegex(ValueError, "Sequential source decode failed"):
                self.extract(frames=(2,), cap=cap)
            self.assertTrue(cap.released)
            self.assertEqual(cap.read_frames, [0, 1])

    def test_future_or_misindexed_journal_row_is_not_consumed_as_current(self):
        def change(rows):
            rows[1]["frame_index"] = 9
            rows[1]["timestamp_ns"] = 900_000_000
        cap = FakeCapture()
        with self.assertRaises(ValueError):
            self.extract(frames=(2,), cap=cap, mutate_rows=change)
        self.assertEqual(cap.read_frames, [0, 1])
        self.assertTrue(cap.released)

    def test_selected_identity_must_equal_actual_source_measurement(self):
        def change(rows):
            rows[2]["tracks"][0]["measurement_source_xy"] = [34., 33.]
        cap = FakeCapture()
        with self.assertRaisesRegex(ValueError, "actual measurement"):
            self.extract(frames=(2,), cap=cap, mutate_rows=change)
        self.assertTrue(cap.released)

    def test_source_pair_identity_cannot_silently_overwrite_selected_identity(self):
        real_buffer = runner.CausalSourceBuffer
        class WrongIdentityBuffer(real_buffer):
            def extract(self, *args, **kwargs):
                result = super().extract(*args, **kwargs)
                result["identity"] = "0/bright:wrong"
                return result
        cap = FakeCapture()
        with mock.patch.object(runner, "CausalSourceBuffer", WrongIdentityBuffer):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                self.extract(frames=(2,), cap=cap)
        self.assertTrue(cap.released)


class AccuracyV38AggregationTests(unittest.TestCase):
    def test_all_four_lag_group_availability_requires_one_same_observation_key(self):
        a, b = observation(12, "0/bright:a"), observation(12, "0/bright:b")
        group = dict(kind="compact_light", clip="0029", window="fixture", frame=12,
                     keys=[a["key"], b["key"]], baseline_assigned_id="0/bright:a", v36_retained_keys=[])
        selection = minimal_selection([a,b], [group])
        records = [diagnostic_record(a, envelopes=(1,2)), diagnostic_record(b, envelopes=(4,8))]
        result = runner.summarize(selection, records)
        report = result["reference_provenance"]["compact_light"]
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["baseline_matched_samples"], 1)
        self.assertEqual(report["v36_matched_samples"], 0)
        self.assertEqual([report["per_lag"][str(lag)]["envelope_available"] for lag in (1,2,4,8)], [1,1,1,1])
        self.assertEqual(report["all_four_lag_envelope_samples"], 0)
        self.assertEqual(result["groups"][0]["all_four_lag_envelope_keys"], [])
        self.assertFalse(report["detection_retention_claimed"])

    def test_unmatched_reference_denominators_and_unknowns_are_not_dropped(self):
        item = observation(12)
        groups = [dict(kind="dense", clip="0029", window="fixture", frame=12, keys=[item["key"]]),
                  dict(kind="dense", clip="0029", window="fixture", frame=13, keys=[])]
        result = runner.summarize(minimal_selection([item], groups),
                                  [diagnostic_record(item, geometric=(), nominal=(), envelopes=())])
        dense = result["reference_provenance"]["dense"]
        self.assertEqual(dense["samples"], 2)
        self.assertEqual(dense["baseline_matched_samples"], 1)
        self.assertEqual(dense["all_four_lag_envelope_samples"], 0)
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][1]["diagnostic_available_keys"]["geometry"],
                         {str(lag):[] for lag in (1,2,4,8)})
        self.assertFalse(result["classifier_promoted"])
        self.assertFalse(result["airborne_accuracy_established"])

    def test_unknown_probe_and_geometry_reasons_remain_separate(self):
        a, b = observation(12), observation(13)
        unknown_geometry = diagnostic_record(a, geometric=(), nominal=(), envelopes=())
        uninformative = diagnostic_record(b, nominal=(), envelopes=())
        report = runner.scope_stats([unknown_geometry, uninformative])
        for lag in (1,2,4,8):
            value = report["per_lag"][str(lag)]
            self.assertEqual(value["geometry_available"], 1)
            self.assertEqual(value["nominal_contrast_available"], 0)
            self.assertEqual(value["all_nine_envelope_available"], 0)
            self.assertEqual(value["geometry_unknown_reasons"], {"fixture_geometry_unknown":1})
            self.assertEqual(value["probe_unknown_reasons"], {"uninformative_difference":9})
            self.assertEqual(value["nominal_source_supported"], 1)
            self.assertEqual(value["all_nine_source_supported"], 1)
            self.assertEqual(value["nominal_joint_available"], 0)
            self.assertEqual(value["all_nine_joint_available"], 0)
            self.assertEqual(value["joint_unknown_reasons"], {"missing_previous_actual_measurement":9})
            self.assertTrue(all(entry == {"count":0} for entry in value["envelope_distributions"].values()))

    def test_joint_availability_is_not_previous_identity_presence(self):
        record = diagnostic_record(observation(12))
        for lag in record["lags"]:
            lag["previous_actual_measurement_available"] = True
            for index, probe in enumerate(lag["evidence"]["probes"]):
                probe["conditional_pair"] = dict(available=index == 4,
                    reasons=[] if index == 4 else ["subpixel_relative_displacement"])
        report = runner.scope_stats([record])
        for lag in (1, 2, 4, 8):
            value = report["per_lag"][str(lag)]
            self.assertEqual(value["previous_actual_measurement_available"], 1)
            self.assertEqual(value["nominal_joint_available"], 1)
            self.assertEqual(value["all_nine_joint_available"], 0)
            self.assertEqual(value["joint_unknown_reasons"], {"subpixel_relative_displacement":8})

    def test_envelope_distribution_excludes_unknown_not_zero_fills(self):
        records = [diagnostic_record(observation(12)),
                   diagnostic_record(observation(13), nominal=(), envelopes=())]
        report = runner.scope_stats(records)
        for lag in (1,2,4,8):
            for metric in runner.METRICS:
                distribution = report["per_lag"][str(lag)]["envelope_distributions"][metric]
                self.assertEqual(distribution["minimum"], dict(count=1, minimum=.4, median=.4, maximum=.4))

    def test_seven_control_scopes_are_kept_even_when_empty(self):
        item = observation(12, controls=(2,5))
        selection = minimal_selection([item], controls=[dict(label=f"control_{i}") for i in range(7)])
        result = runner.summarize(selection, [diagnostic_record(item)])
        control_scopes = {k:v for k,v in result["scopes"].items() if k.startswith("control/")}
        self.assertEqual(len(control_scopes), 7)
        for i in range(7):
            self.assertEqual(control_scopes[f"control/{i}/control_{i}"]["observations"], int(i in (2,5)))

    def test_duplicate_missing_or_extra_record_fails(self):
        item, other = observation(12), observation(13)
        selection = minimal_selection([item])
        for records in ([], [diagnostic_record(item),diagnostic_record(item)],
                        [diagnostic_record(item),diagnostic_record(other)]):
            with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
                runner.summarize(selection, records)

    def test_saved_array_bytes_hash_copy_and_duplicate_protection(self):
        arrays = {}
        original = np.arange(20, dtype=float).reshape(4,5)[:,::2]
        result = runner.save_array(arrays, "fixture", "patch", original)
        expected = np.ascontiguousarray(original).tobytes()
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(result["finite_pixels"], 12)
        original[0,0] = -999
        self.assertEqual(arrays[result["array_key"]][0,0], 0)
        with self.assertRaisesRegex(ValueError, "Duplicate array key"):
            runner.save_array(arrays, "fixture", "patch", original)


class AccuracyV38PreflightTests(unittest.TestCase):
    @staticmethod
    def reference_fixture(base):
        annotation_input = base/"source_annotation_fixture.txt"
        annotation_input.write_text("synthetic source-only annotation input")
        review_input = base/"root_review_fixture.txt"
        review_input.write_text("synthetic second review input")
        annotation = base/"reference.json"
        contents = reference()
        contents["inputs_sha256"] = {str(annotation_input):runner.sha(annotation_input)}
        annotation.write_text(json.dumps(contents))
        receipt = base/"review_receipt.json"
        review = dict(approved_use="provisional_class_unknown_image_feature_regression",
                      before_v38_scoring=True, airborne_truth=False,
                      annotation_sha256=runner.sha(annotation),
                      inputs_sha256={str(review_input):runner.sha(review_input)})
        receipt.write_text(json.dumps(review))
        return annotation, receipt, contents, annotation_input, review_input

    def test_second_reference_review_pins_annotation_and_both_input_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation, receipt, contents, annotation_input, review_input = self.reference_fixture(Path(temporary))
            files = {}
            with mock.patch.object(runner, "REFERENCE", annotation), \
                 mock.patch.object(runner, "REFERENCE_REVIEW", receipt), \
                 mock.patch.object(runner.cv2, "VideoCapture") as decoder:
                result = runner.verify_reference(files)
            self.assertEqual(result, contents)
            self.assertEqual(files, {str(p.resolve()):runner.sha(p)
                                     for p in (annotation, receipt, annotation_input, review_input)})
            decoder.assert_not_called()

    def test_second_reference_review_rejects_truth_promotion_or_postscore_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation, receipt, _, _, _ = self.reference_fixture(Path(temporary))
            original = json.loads(receipt.read_text())
            for field, value in (("approved_use", "airborne_truth"), ("before_v38_scoring", False),
                                 ("before_v38_scoring", 1), ("airborne_truth", True), ("airborne_truth", 0)):
                changed = {**original, field:value}
                receipt.write_text(json.dumps(changed))
                with mock.patch.object(runner, "REFERENCE", annotation), \
                     mock.patch.object(runner, "REFERENCE_REVIEW", receipt):
                    with self.assertRaisesRegex(ValueError, "source-only reference review"):
                        runner.verify_reference({})

    def test_second_reference_review_rejects_changed_annotation_or_review_inputs(self):
        for changed_item in ("annotation", "annotation_input", "review_input"):
            with self.subTest(changed_item=changed_item), tempfile.TemporaryDirectory() as temporary:
                annotation, receipt, _, annotation_input, review_input = self.reference_fixture(Path(temporary))
                target = dict(annotation=annotation, annotation_input=annotation_input,
                              review_input=review_input)[changed_item]
                target.write_text(target.read_text()+"\n")
                with mock.patch.object(runner, "REFERENCE", annotation), \
                     mock.patch.object(runner, "REFERENCE_REVIEW", receipt):
                    with self.assertRaisesRegex(ValueError, "Changed bound input"):
                        runner.verify_reference({})

    def test_existing_output_is_never_overwritten_or_decoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"existing"
            output.mkdir()
            sentinel = output/"keep.txt"
            sentinel.write_text("keep unchanged")
            with mock.patch.object(runner, "bind_parents") as parents, \
                 mock.patch.object(runner.cv2, "VideoCapture") as decoder:
                with self.assertRaises(FileExistsError):
                    runner.run(output)
            self.assertEqual(sentinel.read_text(), "keep unchanged")
            parents.assert_not_called()
            decoder.assert_not_called()

    def test_parent_preflight_binds_source_bytes_and_rejects_changed_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            checked = base/"checked.json"
            checked.write_text("{}")
            source = base/"synthetic_source_bytes.bin"
            source.write_bytes(b"not a video; hash fixture only")
            audit_path = base/"audit.json"
            audit = dict(verified=True, selected_observations=358,
                         complete_summary_independently_reconstructed=True,
                         source_video_files_opened=False,
                         checked_files_sha256={str(checked):runner.sha(checked)})
            audit_path.write_text(json.dumps(audit))
            expected_source = runner.sha(source)
            parent = dict(inputs={"0029":dict(source_sha256=expected_source)})
            with mock.patch.object(runner, "verify_parent", side_effect=lambda _:({}, {}, parent)), \
                 mock.patch.object(runner, "AUDIT37", audit_path), \
                 mock.patch.object(runner, "AUDIT37_SHA", runner.sha(audit_path)), \
                 mock.patch.object(runner, "SOURCES", {"0029":source}), \
                 mock.patch.object(runner.cv2, "VideoCapture") as decoder:
                _, files, returned_parent = runner.bind_parents()
                self.assertEqual(files[str(source.resolve())], expected_source)
                self.assertEqual(files[str(checked.resolve())], runner.sha(checked))
                self.assertEqual(returned_parent, parent)
                source.write_bytes(b"changed fixture bytes")
                with self.assertRaisesRegex(ValueError, "Changed bound input"):
                    runner.bind_parents()
            decoder.assert_not_called()

    def test_failed_parent_preflight_creates_no_output_and_opens_no_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"new_output"
            with mock.patch.object(runner, "bind_parents", side_effect=ValueError("bad source hash")), \
                 mock.patch.object(runner.cv2, "VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "bad source hash"):
                    runner.run(output)
            self.assertFalse(output.exists())
            decoder.assert_not_called()

    def test_unit_results_code_inputs_and_selection_are_frozen_before_extraction(self):
        class ReachedFrozenExtraction(Exception):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, full, context = base/"root", base/"full", base/"context"
            root.mkdir(); full.mkdir(); context.mkdir()
            journal_dir = base/"journal"
            journal_dir.mkdir()
            journal = journal_dir/"frames.jsonl"
            journal.write_text("".join(json.dumps(row(f))+"\n" for f in range(25)))
            (full/"0029_decisions.jsonl").write_text(journal.read_text())
            (context/"observations.json").write_text("[]")
            refpath = base/"reference.json"
            ref = reference(visible=(), ambiguous=())
            ref["inputs_sha256"] = {}
            refpath.write_text(json.dumps(ref))
            reviewpath = base/"review_receipt.json"
            reviewpath.write_text(json.dumps(dict(
                approved_use="provisional_class_unknown_image_feature_regression",
                before_v38_scoring=True, airborne_truth=False,
                annotation_sha256=runner.sha(refpath), inputs_sha256={})))
            source = base/"synthetic_source_bytes.bin"
            source.write_bytes(b"not decoded")
            code = root/"scripts/fixture.py"
            code.parent.mkdir()
            code.write_text("# synthetic snapshot fixture\n")
            parent = dict(inputs={"0029":dict(path=str(journal_dir), source_sha256=runner.sha(source))})
            files = {str(source.resolve()):runner.sha(source), str(journal.resolve()):runner.sha(journal)}
            output = base/"new_output"
            def stop_after_freeze(*args, **kwargs):
                self.assertTrue((output/"freeze.json").is_file())
                self.assertTrue((output/"unit.log").is_file())
                self.assertTrue((output/"selection.json").is_file())
                snapshot = output/"implementation/scripts/fixture.py"
                self.assertEqual(snapshot.read_bytes(), code.read_bytes())
                freeze = json.loads((output/"freeze.json").read_text())
                self.assertIs(freeze["pre_extraction"], True)
                self.assertEqual(freeze["lags"], [1,2,4,8])
                self.assertEqual(freeze["inputs_sha256"][str(source.resolve())], runner.sha(source))
                self.assertEqual(freeze["inputs_sha256"][str(snapshot.resolve())], runner.sha(snapshot))
                self.assertEqual(freeze["inputs_sha256"][str(reviewpath.resolve())], runner.sha(reviewpath))
                self.assertEqual(freeze["inputs_sha256"][str(refpath.resolve())], runner.sha(refpath))
                raise ReachedFrozenExtraction()
            with mock.patch.object(runner, "ROOT", root), mock.patch.object(runner, "FULL", full), \
                 mock.patch.object(runner, "CONTEXT", context), mock.patch.object(runner, "REFERENCE", refpath), \
                 mock.patch.object(runner, "REFERENCE_REVIEW", reviewpath), \
                 mock.patch.object(runner, "SOURCES", {"0029":source}), \
                 mock.patch.object(runner, "COUNTS", {"0029":25}), \
                 mock.patch.object(runner, "IMPLEMENTATION", ("scripts/fixture.py",)), \
                 mock.patch.object(runner, "bind_parents", return_value=(minimal_selection(), files, parent)), \
                 mock.patch.dict(runner.sha_cache, {}, clear=True), \
                 mock.patch.object(runner.subprocess, "run") as tests, \
                 mock.patch.object(runner, "extract_clip", side_effect=stop_after_freeze), \
                 mock.patch.object(runner.cv2, "VideoCapture") as decoder:
                with self.assertRaises(ReachedFrozenExtraction):
                    runner.run(output)
            tests.assert_called_once()
            self.assertIn("test_accuracy_v38*.py", tests.call_args.args[0])
            decoder.assert_not_called()

    def test_extraction_test_source_is_in_implementation_freeze_inventory(self):
        self.assertIn("tests/unit/test_accuracy_v38_extraction.py", runner.IMPLEMENTATION)


if __name__ == "__main__":
    unittest.main()
