"""Synthetic-only causal source-pair tests. No media or existing data access."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v38_source_pairs import (CausalSourceBuffer, FRAME_NS, LAGS, SHIFTS,
                                     bilinear_sample, shifted_prior, shifted_previous_point)


def pixels():
    y, x = np.indices((80, 96))
    return (x+2*y).astype(np.uint8)


def track(xy=(40.25, 39.75), *, measured=True, qualified=False, segment=0):
    return dict(track_id="bright:1", segment=segment, measured=measured,
                qualified_moving=qualified, measurement_source_xy=list(xy) if measured else None)


def row(frame, *, transform=None, segment=0, reset=False, tracks=None):
    return dict(frame_index=frame, timestamp_ns=frame*FRAME_NS, segment=segment,
                source_to_reference=np.eye(3).tolist() if transform is None else copy.deepcopy(transform),
                motion=dict(reset=reset, accepted=True, status="accepted", motion_fit={
                    "metrics": {"inlier_ratio": 0.9, "median_reprojection_error_px": 0.2}}),
                tracks=[] if tracks is None else tracks)


def pair(buffer, xy=(40.25, 39.75), identity=None, lag=1):
    result = buffer.extract(xy, identity=identity)
    return result, next(item for item in result["lags"] if item["lag"] == lag)


class BufferTest(unittest.TestCase):
    def test_empty_buffer_and_invalid_capacity_rejected(self):
        with self.assertRaises(ValueError):
            CausalSourceBuffer().extract((20, 20))
        for value in (0, 9, -1, True, 1.0, "8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CausalSourceBuffer(value)

    def test_exact_fixed_lags_unknown_warmup_and_missing_association(self):
        buffer = CausalSourceBuffer()
        buffer.update(row(0), pixels())
        result = buffer.extract((40, 40))
        self.assertEqual([item["lag"] for item in result["lags"]], list(LAGS))
        for item in result["lags"]:
            self.assertFalse(item["available"])
            self.assertEqual(item["reasons"], ["prior_frame_not_yet_available"])
            self.assertIsNone(item["prior27"])
        buffer.update(row(1), pixels())
        _, item = pair(buffer)
        self.assertTrue(item["available"])
        self.assertFalse(item["previous_actual_measurement_available"])
        self.assertEqual(item["previous_measurement_status"], "identity_not_requested")

    def test_reject_future_duplicate_gap_and_timestamp_without_state_commit(self):
        buffer = CausalSourceBuffer()
        with self.assertRaises(ValueError):
            buffer.update(row(1), pixels())
        buffer.update(row(0), pixels())
        bad_rows = [row(0), row(2), row(8)]
        bad = row(1)
        bad["timestamp_ns"] += 1
        bad_rows.append(bad)
        bad = row(1)
        bad["timestamp_ns"] = True
        bad_rows.append(bad)
        for bad in bad_rows:
            with self.subTest(frame=bad["frame_index"]), self.assertRaises(ValueError):
                buffer.update(bad, pixels())
            self.assertEqual(buffer.retained_state_counts()["current_frame"], 0)
        buffer.update(row(1), pixels())
        self.assertEqual(buffer.retained_state_counts()["current_frame"], 1)

    def test_invalid_gray_and_shape_change_are_transactional(self):
        buffer = CausalSourceBuffer()
        buffer.update(row(0), pixels())
        for image in (pixels().astype(np.float64), pixels()[..., None], np.zeros((0, 0), np.uint8), pixels()[:-1]):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                buffer.update(row(1), image)
        self.assertEqual(buffer.retained_state_counts()["frames"], 1)

    def test_only_nine_owned_frames_retained_after_long_stream(self):
        buffer = CausalSourceBuffer()
        for frame in range(30):
            state = buffer.update(row(frame), pixels())
            self.assertLessEqual(state["frames"], 9)
        state = buffer.retained_state_counts()
        self.assertEqual(state["frames"], 9)
        self.assertEqual(state["oldest_frame"], 21)
        self.assertEqual(state["current_frame"], 29)
        self.assertEqual(state["owned_uint8_bytes"], 9*pixels().nbytes)
        result, item = pair(buffer, lag=8)
        self.assertTrue(item["available"])
        self.assertEqual(item["prior_frame"]["frame_index"], 21)
        self.assertEqual(item["delta_time_ns"], 8*FRAME_NS)

    def test_smaller_capacity_explicitly_abstains_without_changing_bank(self):
        buffer = CausalSourceBuffer(max_lag=2)
        for frame in range(10):
            buffer.update(row(frame), pixels())
        result = buffer.extract((40, 40))
        self.assertEqual(buffer.retained_state_counts()["frames"], 3)
        for item in result["lags"]:
            if item["lag"] > 2:
                self.assertEqual(item["reasons"], ["lag_exceeds_buffer_capacity"])
            else:
                self.assertTrue(item["available"])

    def test_lag_parameter_rejects_arbitrary_future_reordered_and_bool_values(self):
        buffer = CausalSourceBuffer()
        buffer.update(row(0), pixels())
        for lags in ((1,), (-1, 2, 4, 8), (1, 2, 4, 9), (8, 4, 2, 1), (True, 2, 4, 8), (1, 1, 4, 8)):
            with self.subTest(lags=lags), self.assertRaises(ValueError):
                buffer.extract((40, 40), lags=lags)

    def test_copies_frames_rows_outputs_and_does_not_mutate_input(self):
        buffer = CausalSourceBuffer()
        image, prior = pixels(), row(0)
        image_before, prior_before = image.copy(), copy.deepcopy(prior)
        buffer.update(prior, image)
        np.testing.assert_array_equal(image, image_before)
        self.assertEqual(prior, prior_before)
        image[:] = 255
        prior["motion"]["motion_fit"]["metrics"]["inlier_ratio"] = 0
        buffer.update(row(1), pixels())
        first, item = pair(buffer)
        np.testing.assert_array_equal(item["prior27"], pixels()[27:54, 27:54])
        self.assertEqual(item["prior_frame"]["motion"]["motion_fit"]["metrics"]["inlier_ratio"], .9)
        first["current25"][:] = 0
        item["prior27"][:] = 0
        item["intervening_frames"][0]["motion"]["accepted"] = False
        second, second_item = pair(buffer)
        self.assertTrue(second_item["intervening_frames"][0]["motion"]["accepted"])
        self.assertGreater(second["current25"].max(), 0)
        self.assertGreater(second_item["prior27"].max(), 0)

    def test_source_dtype_is_owned_uint8_even_readonly_noncontiguous_input(self):
        image = pixels()[:, ::2]
        image.flags.writeable = False
        buffer = CausalSourceBuffer()
        buffer.update(row(0), image)
        buffer.update(row(1), image)
        output, _ = pair(buffer, xy=(20, 30))
        self.assertEqual(output["current25"].dtype, np.float64)
        np.testing.assert_array_equal(output["current25"], image[18:43, 8:33])

    def test_invalid_tracks_and_reset_metadata_rejected(self):
        buffer = CausalSourceBuffer()
        for bad in (row(0, tracks=[track(), track()]), row(0, reset="false"), row(0, tracks=[track(segment=1)])):
            with self.assertRaises(ValueError):
                buffer.update(bad, pixels())
        self.assertEqual(buffer.retained_state_counts()["frames"], 0)

    def test_identical_prefix_unchanged_by_future_inputs(self):
        buffers = [CausalSourceBuffer(), CausalSourceBuffer()]
        for buffer in buffers:
            for frame in range(5):
                buffer.update(row(frame), pixels())
        prefix = buffers[0].extract((40, 40))
        other = buffers[1].extract((40, 40))
        for frame in range(5, 10):
            buffers[1].update(row(frame, reset=frame==7), np.zeros_like(pixels()))
        np.testing.assert_array_equal(prefix["current25"], other["current25"])
        for a, b in zip(prefix["lags"], other["lags"]):
            self.assertEqual(a["available"], b["available"])
            if a["prior27"] is not None:
                np.testing.assert_array_equal(a["prior27"], b["prior27"])


class GeometryTest(unittest.TestCase):
    def buffer(self, prior_h=None, current_h=None, *, prior_tracks=None, current_tracks=None):
        result = CausalSourceBuffer()
        result.update(row(0, transform=prior_h, tracks=prior_tracks), pixels())
        result.update(row(1, transform=current_h, tracks=current_tracks), pixels())
        return result

    def test_native_rounding_identity_sampling_and_all_nine_shifts(self):
        result, item = pair(self.buffer())
        self.assertEqual(result["current_integer_center_xy"], [40, 40])
        self.assertEqual(result["current_xy"], [.25, -.25])
        np.testing.assert_array_equal(result["current25"], pixels()[28:53, 28:53])
        np.testing.assert_array_equal(item["prior27"], pixels()[27:54, 27:54])
        for dx, dy in SHIFTS:
            shifted = shifted_prior(item["prior27"], (dx, dy))
            np.testing.assert_array_equal(shifted, pixels()[28+dy:53+dy, 28+dx:53+dx])
        self.assertTrue(all(record["complete"] for record in item["shift_support"]))

    def test_direction_solve_matrices_and_optional_point_minus_shift(self):
        prior = [[1, 0, 4], [0, 1, -3], [0, 0, 1]]
        current = [[1, 0, 1], [0, 1, 2], [0, 0, 1]]
        buffer = self.buffer(prior, current, prior_tracks=[track((38, 45))], current_tracks=[track()])
        _, item = pair(buffer, identity="0/bright:1")
        np.testing.assert_array_equal(item["current_to_prior_matrix"], [[1, 0, -3], [0, 1, 5], [0, 0, 1]])
        self.assertEqual(item["previous_point_current_grid_xy"], [1.0, 0.0])
        self.assertEqual(shifted_previous_point(item["previous_point_current_grid_xy"], (1, -1)), [0.0, 1.0])
        self.assertEqual(item["prior27"][13, 13], pixels()[45, 37])

    def test_exact_fractional_bilinear_not_quantized_cv_table(self):
        current = [[1, 0, .137], [0, 1, -.219], [0, 0, 1]]
        _, item = pair(self.buffer(current_h=current))
        y, x = np.mgrid[-13:14, -13:14]
        expected = (40+x+.137)+2*(40+y-.219)
        np.testing.assert_allclose(item["prior27"], expected, atol=1e-12, rtol=0)

    def test_projective_sampling_matches_independent_scalar_projection(self):
        current = np.array([[1.02, .01, -1], [.005, 1.01, 1], [.001, -.0003, 1]])
        _, item = pair(self.buffer(current_h=current.tolist()))
        expected = np.zeros((27, 27))
        for iy, y in enumerate(range(-13, 14)):
            for ix, x in enumerate(range(-13, 14)):
                den = .001*(40+x)-.0003*(40+y)+1
                sx = (1.02*(40+x)+.01*(40+y)-1)/den
                sy = (.005*(40+x)+1.01*(40+y)+1)/den
                expected[iy, ix] = sx+2*sy
        self.assertTrue(item["available"])
        np.testing.assert_allclose(item["prior27"], expected, atol=1e-12, rtol=0)

    def test_negative_homogeneous_scale_preserves_mapping_without_false_horizon(self):
        _, item = pair(self.buffer(prior_h=(-np.eye(3)).tolist()))
        self.assertTrue(item["available"])
        self.assertFalse(item["support"]["projective_horizon_crosses_patch"])
        np.testing.assert_array_equal(item["prior27"], pixels()[27:54, 27:54])

    def test_composed_transform_condition_checked_even_if_endpoints_pass(self):
        prior = np.diag([1e-7, 1, 1]).tolist()
        current = np.diag([1e7, 1, 1]).tolist()
        _, item = pair(self.buffer(prior_h=prior, current_h=current))
        self.assertFalse(item["available"])
        self.assertEqual(item["reasons"], ["invalid_or_ill_conditioned_current_to_prior_transform"])

    def test_projective_horizon_crossing_is_unavailable_with_nan_support(self):
        warp = [[1, 0, 0], [0, 1, 0], [.05, 0, -2]]
        _, item = pair(self.buffer(current_h=warp), xy=(40, 40))
        self.assertFalse(item["available"])
        self.assertIn("projective_horizon_crosses_patch", item["reasons"])
        self.assertTrue(item["support"]["projective_horizon_crosses_patch"])
        self.assertTrue(np.isnan(item["prior27"][:, 13]).all())

    def test_border_native_and_prior_support_never_zero_filled(self):
        result, item = pair(self.buffer(), xy=(0, 0))
        self.assertEqual(result["current25"].shape, (25, 25))
        self.assertTrue(np.isnan(result["current25"][:12]).all())
        self.assertTrue(np.isnan(result["current25"][:, :12]).all())
        self.assertEqual(result["current25"][12, 12], pixels()[0, 0])
        self.assertTrue(item["available"])  # Geometry valid, patch support incomplete.
        self.assertFalse(any(record["complete"] for record in item["shift_support"]))
        self.assertTrue(np.isnan(item["prior27"][:13]).all())

    def test_exact_last_source_pixel_ignores_zero_weight_outside_neighbor(self):
        gray = pixels()
        sampled = bilinear_sample(gray, [[95, 95.1, float("nan")]], [[79, 79, 0]])
        self.assertEqual(sampled[0, 0], gray[79, 95])
        self.assertTrue(np.isnan(sampled[0, 1:]).all())

    def test_invalid_nonfinite_and_illconditioned_transforms_are_explicit_unknown(self):
        values = [np.zeros((3, 3)).tolist(), [[1, 0], [0, 1]],
                  [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
                  [[1, 0, 0], [0, 1e-14, 0], [0, 0, 1]]]
        for value in values:
            with self.subTest(transform=value):
                result, item = pair(self.buffer(current_h=value))
                self.assertFalse(item["available"])
                self.assertTrue(item["reasons"][0].startswith("current_"))
                self.assertIsNone(item["prior27"])
                metadata = copy.deepcopy(result)
                metadata.pop("current25")
                for branch in metadata["lags"]:
                    branch.pop("prior27")
                json.dumps(metadata, allow_nan=False)

    def test_shift_helpers_reject_nonbank_values_and_preserve_input(self):
        prior = np.arange(729, dtype=float).reshape(27, 27)
        before = prior.copy()
        output = shifted_prior(prior, (0, 0))
        output[:] = 0
        np.testing.assert_array_equal(prior, before)
        for shift in ((0.5, 0), (2, 0), (True, 0), (0, float("nan"))):
            with self.assertRaises(ValueError):
                shifted_prior(prior, shift)
        self.assertIsNone(shifted_previous_point(None, (0, 0)))

    def test_invalid_current_coordinates_or_identity_are_not_silently_recentered(self):
        buffer = self.buffer(current_tracks=[track()])
        for xy in ((40, 40), (True, 40), (float("nan"), 40)):
            with self.assertRaises(ValueError):
                buffer.extract(xy, identity="0/bright:1")
        for identity in ("bright:1", "0/bright:2", (0, "bright:1")):
            with self.assertRaises(ValueError):
                buffer.extract((40.25, 39.75), identity=identity)


class ProvenanceTest(unittest.TestCase):
    def test_missing_prior_id_or_prediction_does_not_block_source_pair(self):
        for prior_tracks, expected in (([], "identity_not_present"),
                                       ([track(measured=False)], "identity_present_without_actual_measurement")):
            buffer = CausalSourceBuffer()
            buffer.update(row(0, tracks=prior_tracks), pixels())
            buffer.update(row(1, tracks=[track()]), pixels())
            _, item = pair(buffer, identity="0/bright:1")
            self.assertTrue(item["available"])
            self.assertFalse(item["previous_actual_measurement_available"])
            self.assertIsNone(item["previous_point_current_grid_xy"])
            self.assertEqual(item["previous_measurement_status"], expected)

    def test_actual_prior_needs_no_class_or_qualification(self):
        buffer = CausalSourceBuffer()
        buffer.update(row(0, tracks=[track((38, 39), qualified=False)]), pixels())
        buffer.update(row(1, tracks=[track(qualified=False)]), pixels())
        _, item = pair(buffer, identity="0/bright:1")
        self.assertTrue(item["available"])
        self.assertTrue(item["previous_actual_measurement_available"])
        self.assertEqual(item["previous_point_current_grid_xy"], [-2, -1])

    def test_every_intermediate_motion_dict_is_retained_without_quality_reclassification(self):
        buffer = CausalSourceBuffer()
        originals = []
        for frame in range(9):
            current = row(frame)
            current["motion"].update(accepted=False, status="reused", detail={"frame": frame, "flags": [False, True]})
            originals.append(copy.deepcopy(current["motion"]))
            buffer.update(current, pixels())
        _, item = pair(buffer, lag=8)
        self.assertTrue(item["available"])
        self.assertEqual(item["prior_frame"]["motion"], originals[0])
        self.assertEqual([entry["motion"] for entry in item["intervening_frames"]], originals[1:])
        self.assertEqual([entry["frame_index"] for entry in item["intervening_frames"]], list(range(1, 9)))

    def test_intervening_reset_blocks_crossing_lag_but_not_new_reference_prior(self):
        buffer = CausalSourceBuffer()
        for frame in range(9):
            buffer.update(row(frame, reset=frame==4), pixels())
        result = buffer.extract((40, 40))
        decisions = {item["lag"]: item for item in result["lags"]}
        self.assertFalse(decisions[8]["available"])
        self.assertIn("intervening_reference_reset", decisions[8]["reasons"])
        self.assertTrue(decisions[4]["available"])  # Reset at prior frame starts the shared reference.
        self.assertTrue(decisions[1]["available"])

    def test_segment_changed_then_returned_still_invalidates_longer_pair(self):
        buffer = CausalSourceBuffer()
        for frame in range(5):
            buffer.update(row(frame, segment=1 if frame==2 else 0), pixels())
        _, item = pair(buffer, lag=4)
        self.assertFalse(item["available"])
        self.assertIn("intervening_reference_segment_change", item["reasons"])

    def test_intervening_invalid_transform_cannot_be_hidden_by_valid_endpoints(self):
        buffer = CausalSourceBuffer()
        for frame in range(5):
            buffer.update(row(frame, transform=np.zeros((3, 3)).tolist() if frame==2 else None), pixels())
        _, item = pair(buffer, lag=4)
        self.assertFalse(item["available"])
        self.assertTrue(any(reason.startswith("intervening_frame_2_") for reason in item["reasons"]))

    def test_actual_prior_coordinates_remain_auditable_when_geometry_unavailable(self):
        buffer = CausalSourceBuffer()
        buffer.update(row(0, tracks=[track((38, 39))]), pixels())
        buffer.update(row(1, reset=True, tracks=[track()]), pixels())
        _, item = pair(buffer, identity="0/bright:1")
        self.assertFalse(item["available"])
        self.assertTrue(item["previous_actual_measurement_available"])
        self.assertEqual(item["previous_actual_measurement_source_xy"], [38, 39])
        self.assertIsNone(item["previous_point_current_grid_xy"])

    def test_nonfinite_quality_field_is_explicitly_sanitized_not_hidden_or_used_as_gate(self):
        buffer = CausalSourceBuffer()
        previous = row(0)
        previous["motion"]["motion_fit"]["metrics"]["median_reprojection_error_px"] = float("nan")
        buffer.update(previous, pixels())
        buffer.update(row(1), pixels())
        _, item = pair(buffer)
        self.assertTrue(item["available"])
        meta = item["prior_frame"]
        self.assertIsNone(meta["motion"]["motion_fit"]["metrics"]["median_reprojection_error_px"])
        self.assertEqual(meta["nonfinite_metadata_fields_replaced_with_null"],
                         ["motion.motion_fit.metrics.median_reprojection_error_px"])
        json.dumps(meta, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
