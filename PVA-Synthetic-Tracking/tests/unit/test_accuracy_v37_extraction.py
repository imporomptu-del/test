"""Synthetic extraction/provenance tests; never open original media."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import diagnose_accuracy_v37_temporal as module


def track(*, measured=True, qualified=True, xy=(32.25, 31.75), segment=0, identity="bright:1"):
    return dict(track_id=identity, segment=segment, measured=measured,
                qualified_moving=qualified, measurement_source_xy=list(xy) if measured else None)


def row(frame, tracks=None, segment=0, transform=None):
    return dict(frame_index=frame, timestamp_ns=frame*module.FRAME_NS, segment=segment,
                source_to_reference=np.eye(3).tolist() if transform is None else transform,
                motion={"reset": False}, tracks=[track(segment=segment)] if tracks is None else tracks)


def observation(frame=1, xy=(32.25, 31.75)):
    return dict(key=f"0029/{frame}/0/bright:1", clip="0029", frame=frame,
                identity="0/bright:1", measurement_source_xy=list(xy), polarity="bright",
                provisional_control_indices=[])


def pixels():
    y, x = np.indices((72, 80))
    return (10+x+2*y).astype(np.uint8)


class Registration:
    def __init__(self, *, available=True, shift=(0, 0), gain=1.0, offset=0.0):
        self.calls = []
        self.available, self.shift, self.gain, self.offset = available, shift, gain, offset

    def measure(self, current, prior, *, prior_point_xy):
        self.calls.append((current.copy(), prior.copy(), list(prior_point_xy)))
        return dict(available=self.available, status="ok" if self.available else "ambiguous",
                    reasons=[] if self.available else ["ambiguous"], shift_xy=list(self.shift),
                    gain=self.gain, offset=self.offset, mse=0,
                    current_patch=current[12:37, 12:37].copy(),
                    registered_prior_patch=prior[14:39, 14:39].copy(),
                    registered_prior_patch_photometrically_corrected=False)


class Temporal:
    def __init__(self, available=True):
        self.calls = []
        self.available = available

    def measure(self, current, prior, **kwargs):
        self.calls.append((current.copy(), prior.copy(), copy.deepcopy(kwargs)))
        return dict(available=self.available, status="ok" if self.available else "uninformative",
                    reasons=[], point_minus_edge_fraction=0.25, nested={"example_margin": -0.1})


class ExtractionTest(unittest.TestCase):
    def process(self, *, current_row=None, previous_row=None, obs=None, registration=None, temporal=None,
                prior_missing=False, original_sha=None):
        registration = registration or Registration()
        temporal = temporal or Temporal()
        arrays = {}
        image = pixels()
        result = module.process_observation(obs or observation(), current_row or row(1),
            None if prior_missing else previous_row or row(0), image, None if prior_missing else image,
            registration, temporal, arrays, "synthetic", original_sha)
        return result, arrays, registration, temporal

    def test_current_crop_native_values_and_fractional_rounding(self):
        result, arrays, _, _ = self.process()
        self.assertEqual(result["integer_center_xy"], [32, 32])
        self.assertEqual(result["current_xy"], [0.25, -0.25])
        np.testing.assert_array_equal(arrays["synthetic_current49"], pixels()[8:57, 8:57])
        self.assertEqual(result["patches"]["current49"]["dtype"], "float32")

    def test_native_crop_border_is_nan_without_resizing_or_wrapping(self):
        patch = module.crop_native(pixels(), (1, 2), 24)
        self.assertEqual(patch.shape, (49, 49))
        self.assertTrue(np.isnan(patch[:22]).all())
        self.assertTrue(np.isnan(patch[:, :23]).all())
        np.testing.assert_array_equal(patch[22:, 23:], pixels()[:27, :26])

    def test_prior_identity_resampling_preserves_native_interior(self):
        patch = module.resample_prior(pixels(), (32, 32), np.eye(3))
        np.testing.assert_array_equal(patch, pixels()[6:59, 6:59])

    def test_warp_direction_point_mapping_and_shift_sign(self):
        previous = np.array([[1, 0, 4], [0, 1, -3], [0, 0, 1.0]])
        current = np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1.0]])
        registration = Registration(shift=(0.5, -1))
        result, arrays, reg, temp = self.process(
            current_row=row(1, transform=current.tolist()),
            previous_row=row(0, [track(xy=(30, 37))], transform=previous.tolist()), registration=registration)
        np.testing.assert_array_equal(result["current_to_previous_matrix"], [[1, 0, -3], [0, 1, 5], [0, 0, 1]])
        self.assertEqual(reg.calls[0][2], [1.0, 0.0])
        self.assertEqual(temp.calls[0][2]["previous_xy"], [0.5, 1.0])
        self.assertEqual(result["previous_point_registered_grid_xy"], [0.5, 1.0])
        self.assertEqual(arrays["synthetic_prior53"][26, 26], pixels()[37, 29])

    def test_fractional_prior_warp_matches_analytic_linear_field(self):
        warp = np.array([[1, 0, .5], [0, 1, -.25], [0, 0, 1]])
        prior = module.resample_prior(pixels(), (32, 32), warp)
        y, x = np.mgrid[-26:27, -26:27]
        np.testing.assert_allclose(prior, 10+(32+x+.5)+2*(32+y-.25), atol=0, rtol=0)

    def test_photometric_correction_applied_exactly_once_and_saved(self):
        result, arrays, _, temporal = self.process(registration=Registration(gain=1.5, offset=3))
        expected = 1.5*arrays["synthetic_registered_prior25_raw"]+3
        np.testing.assert_array_equal(temporal.calls[0][1], expected)
        np.testing.assert_array_equal(arrays["synthetic_registered_prior25_corrected"], expected)
        self.assertTrue(result["available"])

    def test_previous_prediction_never_substitutes_for_actual_measurement(self):
        result, arrays, registration, temporal = self.process(previous_row=row(0, [track(measured=False)]))
        self.assertEqual(result["reason"], "missing_previous_actual_measurement")
        self.assertFalse(result["previous_actual_measurement_available"])
        self.assertEqual(registration.calls, [])
        self.assertEqual(temporal.calls, [])
        self.assertEqual(set(arrays), {"synthetic_current49", "synthetic_prior53"})
        self.assertTrue(np.isfinite(arrays["synthetic_prior53"]).all())

    def test_prior_unqualified_actual_measurement_is_valid_evidence(self):
        result, _, registration, _ = self.process(previous_row=row(0, [track(qualified=False)]))
        self.assertTrue(result["previous_actual_measurement_available"])
        self.assertEqual(len(registration.calls), 1)

    def test_missing_previous_frame_keeps_observation_and_nan_prior(self):
        result, arrays, reg, _ = self.process(prior_missing=True)
        self.assertEqual(result["reason"], "missing_previous_frame")
        self.assertTrue(np.isnan(arrays["synthetic_prior53"]).all())
        self.assertEqual(reg.calls, [])

    def test_gap_or_future_previous_row_is_unknown_never_last_observation(self):
        for previous in (row(-1), row(1), row(2)):
            with self.subTest(previous=previous["frame_index"]):
                result, _, reg, _ = self.process(previous_row=previous)
                self.assertEqual(result["reason"], "nonadjacent_or_stale_previous_frame")
                self.assertEqual(reg.calls, [])

    def test_segment_and_reference_reset_prohibit_temporal_fit(self):
        result, _, reg, _ = self.process(previous_row=row(0, segment=1))
        self.assertEqual(result["reason"], "different_reference_segment")
        self.assertEqual(reg.calls, [])
        current = row(1)
        current["motion"]["reset"] = True
        result, _, reg, _ = self.process(current_row=current)
        self.assertEqual(result["reason"], "different_reference_segment")
        self.assertEqual(reg.calls, [])

    def test_invalid_transform_is_explicit_unknown(self):
        previous = row(0, transform=np.zeros((3, 3)).tolist())
        result, arrays, reg, _ = self.process(previous_row=previous)
        self.assertEqual(result["reason"], "invalid_source_to_reference_transform")
        self.assertTrue(np.isnan(arrays["synthetic_prior53"]).all())
        self.assertEqual(reg.calls, [])

    def test_registration_unavailable_preserves_arrays_and_never_calls_temporal(self):
        result, arrays, _, temporal = self.process(registration=Registration(available=False))
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "registration_ambiguous")
        self.assertEqual(result["registration"]["reasons"], ["ambiguous"])
        self.assertEqual(result["unavailable_reasons"], ["registration/ambiguous"])
        self.assertIn("synthetic_registered_prior25_raw", arrays)
        self.assertIsNone(result["features"])
        self.assertEqual(temporal.calls, [])

    def test_temporal_unavailable_is_not_a_retained_or_failed_detection(self):
        result, _, _, _ = self.process(temporal=Temporal(available=False))
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "temporal_uninformative")
        self.assertNotIn("accepted", result)
        self.assertNotIn("qualified", result)

    def test_original_v36_pixel_hash_is_checked_before_any_fit(self):
        digest = hashlib.sha256(pixels()[20:45, 20:45].tobytes()).hexdigest()
        result, _, _, _ = self.process(original_sha=digest)
        self.assertTrue(result["v36_current_patch_bytes_crosschecked"])
        reg = Registration()
        with self.assertRaisesRegex(ValueError, "differ from audited"):
            self.process(original_sha="0"*64, registration=reg)
        self.assertEqual(reg.calls, [])

    def test_identity_coordinate_or_qualification_mismatch_fails_closed(self):
        bad = (row(1, [track(xy=(32.3, 31.75))]), row(1, [track(measured=False)]),
               row(1, [track(qualified=False)]), row(1, []))
        for current in bad:
            with self.subTest(current=current):
                with self.assertRaisesRegex(ValueError, "no longer matches"):
                    self.process(current_row=current)

    def test_json_metadata_has_no_nan_and_inputs_are_unchanged(self):
        current, previous, obs, image = row(1), row(0), observation(), pixels()
        originals = copy.deepcopy((current, previous, obs))
        image_before = image.copy()
        result = module.process_observation(obs, current, previous, image, image,
                                            Registration(), Temporal(), {}, "owned")
        self.assertEqual((current, previous, obs), originals)
        np.testing.assert_array_equal(image, image_before)
        json.dumps(result, allow_nan=False)

    def test_validation_rejects_duplicate_ids_and_nonchronological_rows(self):
        with self.assertRaises(ValueError):
            module.validate_row(row(1, [track(), track()]), 1)
        for changed in ({"timestamp_ns": 1}, {"frame_index": 2}, {"segment": True}):
            current = row(1)
            current.update(changed)
            with self.assertRaises(ValueError):
                module.validate_row(current, 1)
        for value in (None, 0, 1, "false"):
            current = row(1)
            current["motion"]["reset"] = value
            with self.assertRaises(ValueError):
                module.validate_row(current, 1)

    def test_fitted_prior_already_corrected_is_rejected_not_corrected_twice(self):
        registration = Registration()
        original = registration.measure
        def corrected(*args, **kwargs):
            value = original(*args, **kwargs)
            value["registered_prior_patch_photometrically_corrected"] = True
            return value
        registration.measure = corrected
        with self.assertRaisesRegex(ValueError, "one photometric correction"):
            self.process(registration=registration)

    def test_each_model_unavailable_reason_is_preserved(self):
        temporal = Temporal()
        temporal.measure = lambda *args, **kwargs: dict(
            available=False, status="unavailable", reasons=["small_displacement", "rank_deficient"])
        result, _, _, _ = self.process(temporal=temporal)
        self.assertEqual(result["unavailable_reasons"],
                         ["temporal/small_displacement", "temporal/rank_deficient"])


class FakeCapture:
    def __init__(self, frames, shape):
        self.frames, self.shape, self.reads, self.released = frames, shape, 0, False

    def isOpened(self):
        return True

    def getBackendName(self):
        return "synthetic-no-media"

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_HEIGHT: self.shape[0],
                cv2.CAP_PROP_FRAME_WIDTH: self.shape[1], cv2.CAP_PROP_FRAME_COUNT: len(self.frames)}[prop]

    def read(self):
        index = self.reads
        self.reads += 1
        if index >= len(self.frames):
            return False, None
        return True, self.frames[index].copy()

    def release(self):
        self.released = True


class StreamTest(unittest.TestCase):
    def test_sequential_walk_uses_immediately_previous_frame_without_seek(self):
        base = pixels()
        frames = [np.repeat((base+index)[:, :, None], 3, axis=2) for index in range(4)]
        cap = FakeCapture(frames, base.shape)
        obs = observation(frame=2)
        records, arrays, registration, temporal = [], {}, Registration(), Temporal()
        digest = hashlib.sha256((base+2)[20:45, 20:45].tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)/"synthetic.jsonl"
            journal.write_text("".join(json.dumps(row(index))+"\n" for index in range(4)))
            metadata = module.extract_clip("0029", {"observations": [obs]}, journal, "not-a-video",
                registration, temporal, {obs["key"]: digest}, records, arrays,
                capture_factory=lambda _: cap, shape=base.shape, frame_count=4)
        self.assertEqual(cap.reads, 3)
        self.assertTrue(cap.released)
        self.assertEqual(metadata["last_frame"], 2)
        self.assertEqual(records[0]["previous_frame"]["frame_index"], 1)
        np.testing.assert_array_equal(registration.calls[0][1], (base+1)[6:59, 6:59])

    def test_decode_failure_releases_capture_and_never_skips_selected_observation(self):
        base = pixels()
        cap = FakeCapture([np.repeat(base[:, :, None], 3, axis=2)], base.shape)
        cap.get = lambda prop: {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_HEIGHT: base.shape[0],
                              cv2.CAP_PROP_FRAME_WIDTH: base.shape[1], cv2.CAP_PROP_FRAME_COUNT: 4}[prop]
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)/"synthetic.jsonl"
            journal.write_text("".join(json.dumps(row(index))+"\n" for index in range(2)))
            with self.assertRaisesRegex(ValueError, "decode failed"):
                module.extract_clip("0029", {"observations": [observation()]}, journal, "not-a-video",
                    Registration(), Temporal(), {}, [], {}, capture_factory=lambda _: cap,
                    shape=base.shape, frame_count=4)
        self.assertTrue(cap.released)

    def test_bad_audit_prevents_output_creation_or_source_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"fresh"
            with patch.object(module, "verify_parent", side_effect=ValueError("bad audit")), \
                    patch.object(module, "extract_clip") as extract:
                with self.assertRaisesRegex(ValueError, "bad audit"):
                    module.run(output, Path(temporary)/"not-an-audit")
                extract.assert_not_called()
                self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(module, "verify_parent") as verify:
                with self.assertRaises(FileExistsError):
                    module.run(Path(temporary), "unused")
                verify.assert_not_called()

    def test_unknowns_remain_in_summary_and_reference_denominators(self):
        obs = [observation(frame=1), observation(frame=2)]
        obs[1]["provisional_control_indices"] = [0]
        selected = dict(observations=obs, groups=[
            dict(kind="dense", window="synthetic", frame=1, keys=[obs[0]["key"]]),
            dict(kind="dense", window="synthetic", frame=2, keys=[obs[1]["key"]]),
            dict(kind="dense", window="synthetic", frame=3, keys=[])],
            controls=[dict(label="synthetic")])
        results = [dict(**item, available=i==0, previous_actual_measurement_available=i==0,
                        reason="available" if i==0 else "missing_previous_actual_measurement",
                        features={"available": True, "margin": .2} if i==0 else None)
                   for i, item in enumerate(obs)]
        summary = module.summarize(selected, results)
        provenance = summary["reference_provenance"]["dense"]
        self.assertEqual(provenance["samples"], 3)
        self.assertEqual(provenance["original_baseline_matched_samples"], 2)
        self.assertEqual(provenance["samples_with_available_diagnostic"], 1)
        self.assertEqual(summary["provisional_controls"][0]["baseline_selected_measurements"], 1)
        self.assertEqual(summary["provisional_controls"][0]["diagnostic_available_measurements"], 0)
        self.assertFalse(summary["accuracy_retention_claimed"])
        self.assertFalse(summary["output_gate_applied"])
        self.assertIsNone(summary["airborne_recall"])
        self.assertEqual(summary["feature_distributions"]["all"]["feature_distributions"]["margin"]["count"], 1)
        self.assertNotIn("available", summary["feature_distributions"]["all"]["feature_distributions"])

    def test_summary_rejects_missing_or_duplicate_selected_records(self):
        obs = observation()
        selected = {"observations": [obs]}
        for records in ([], [obs, obs]):
            with self.subTest(records=records):
                with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                    module.summarize(selected, records)

    def test_multireason_counts_are_per_observation_and_may_overlap(self):
        records = [dict(available=False, reason="temporal_unavailable",
                        unavailable_reasons=["temporal/low_motion", "temporal/rank", "temporal/rank"]),
                   dict(available=False, reason="temporal_unavailable", unavailable_reasons=["temporal/rank"]),
                   dict(available=True, reason="diagnostic_available", unavailable_reasons=[])]
        self.assertEqual(module.unavailable_reason_counts(records),
                         {"temporal/low_motion": 1, "temporal/rank": 2})


if __name__ == "__main__":
    unittest.main()
