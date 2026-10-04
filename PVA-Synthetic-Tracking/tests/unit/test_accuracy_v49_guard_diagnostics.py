"""Generated-only checks for the bounded V49 diagnostic, never real packet reads."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_accuracy_v49_guard_diagnostics as diagnostics


def generated_row(frame, track="bright:1", *, clip="generated", segment=0,
                  stamp=None, guard=True, archive=True, sign="positive"):
    return dict(
        clip=clip, frame_index=frame, segment=segment, track_id=track,
        archive={"path": f"generated_{frame}_{track}.npz", "sha256": "a"*64} if archive else None,
        geometry={"geometry": {
            "current_timestamp_ns": frame * 100_000_000 if stamp is None else stamp,
            "current_center_xy": [64, 64],
        }},
        qualified_moving=False, reference_samples=[],
        arms={diagnostics.ARM: {
            "gain_calibration": None if guard is None else {"available": guard},
            "raw_adapter_result": {"coefficient_sign": sign, "reasons": ["generated"]},
        }},
    )


def selection(rows, case=None):
    if case is None:
        case = ("generated", 100, 0, "bright:1")
    return diagnostics.select_scope(rows, cases=(case,))


def decision(result, kind, direction):
    return next(x for x in result["decisions"]
                if x["kind"] == kind and x["direction"] == direction)


class ScopeTests(unittest.TestCase):
    def test_prefers_same_track_even_when_other_track_is_nearer(self):
        rows = [generated_row(100, guard=False), generated_row(99, "bright:2"),
                generated_row(95), generated_row(101, "bright:2"), generated_row(107)]
        result = selection(rows)
        self.assertEqual(decision(result, "guard_control", "before")["selected_key"],
                         ["generated", 95, 0, "bright:1"])
        self.assertEqual(decision(result, "guard_control", "after")["selected_key"],
                         ["generated", 107, 0, "bright:1"])

    def test_rejects_different_segment_polarity_clip_and_same_timestamp(self):
        rows = [generated_row(100, guard=False), generated_row(99, segment=1),
                generated_row(99, "dark:1"), generated_row(99, clip="other"),
                generated_row(98, "bright:2", stamp=10_000_000_000),
                generated_row(96, "bright:3")]
        result = selection(rows)
        before = decision(result, "guard_control", "before")
        self.assertEqual(before["selected_key"], ["generated", 96, 0, "bright:3"])
        self.assertFalse(before["same_track"])
        self.assertIsNone(decision(result, "guard_control", "after")["selected_key"])

    def test_maximum_gap_is_inclusive_and_timestamp_based(self):
        rows = [generated_row(100, guard=False),
                generated_row(99, stamp=7_999_999_999),
                generated_row(98, "bright:2", stamp=8_000_000_000),
                generated_row(101, stamp=12_000_000_001)]
        result = selection(rows)
        before = decision(result, "guard_control", "before")
        self.assertEqual(before["selected_key"], ["generated", 98, 0, "bright:2"])
        self.assertEqual(before["nominal_timestamp_gap_seconds"], 2.0)
        self.assertIsNone(decision(result, "guard_control", "after")["selected_key"])

    def test_tie_breaks_by_full_state_key_not_input_order(self):
        rows = [generated_row(100, guard=False), generated_row(99, "bright:3"),
                generated_row(99, "bright:2"), generated_row(98, "bright:9", stamp=9_900_000_000)]
        expected = ["generated", 98, 0, "bright:9"]
        for order in (rows, list(reversed(rows))):
            self.assertEqual(decision(selection(order), "guard_control", "before")["selected_key"],
                             expected)

    def test_source_sign_and_reference_metadata_cannot_change_selection(self):
        rows = [generated_row(100, guard=False), generated_row(99, "bright:2", sign="zero"),
                generated_row(98, "bright:3", sign="positive")]
        before = selection(rows)
        altered = deepcopy(rows)
        for row in altered:
            del row["arms"][diagnostics.ARM]["raw_adapter_result"]
            row["qualified_moving"] = True
            row["reference_samples"] = [12345]
        self.assertEqual(selection(altered), before)
        self.assertFalse(before["source_signs_or_references_used_for_selection"])

    def test_guard_not_reached_neighbor_is_kept_but_not_a_passing_control(self):
        rows = [generated_row(100, guard=False), generated_row(99, guard=None),
                generated_row(98), generated_row(101, guard=False)]
        result = selection(rows)
        self.assertEqual(decision(result, "guard_control", "before")["selected_key"][1], 98)
        self.assertEqual(decision(result, "same_track_neighbor", "before")["selected_key"][1], 99)
        self.assertEqual(decision(result, "same_track_neighbor", "after")["selected_key"][1], 101)
        row99 = next(x for x in result["selected"] if x["state_key"][1] == 99)
        self.assertFalse(row99["guard_reached"])
        self.assertIsNone(decision(result, "guard_control", "after")["selected_key"])

    def test_missing_archive_is_neither_control_nor_neighbor(self):
        rows = [generated_row(100, guard=False), generated_row(99, archive=False)]
        result = selection(rows)
        self.assertEqual(len(result["selected"]), 1)
        self.assertTrue(all(x["selected_key"] is None for x in result["decisions"]))

    def test_case_without_archive_and_duplicate_keys_fail(self):
        with self.assertRaisesRegex(ValueError, "no frozen archive"):
            selection([generated_row(100, archive=False)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            selection([generated_row(100), generated_row(100)])

    def test_one_state_retains_multiple_roles_and_does_not_duplicate(self):
        result = selection([generated_row(100, guard=False), generated_row(99)])
        self.assertEqual(len(result["selected"]), 2)
        control = next(x for x in result["selected"] if x["state_key"][1] == 99)
        self.assertEqual({x["kind"] for x in control["roles"]},
                         {"guard_control", "same_track_neighbor"})


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.points = [[8, 8], [16, 8], [24, 8], [16, 16], [16, 24]]
        self.stencils = [
            dict(axis="x", center_xy=[16, 8], pixels_xy=self.points[:3], weights=[1, -2, 1]),
            dict(axis="y", center_xy=[16, 16], pixels_xy=[self.points[i] for i in [1, 3, 4]],
                 weights=[1, -2, 1]),
        ]

    def test_exact_binary_float_contrasts_are_recomputed_without_rounding(self):
        y = np.array([0.1, np.nextafter(0.2, np.inf), 0.3, 1.25, 8.75])
        b = np.array([7.5, 1.125, 2.5, np.nextafter(0.7, -np.inf), 1.0])
        result = diagnostics.contrasts_from_points(y, b, self.points, self.stencils)
        for i, indices in enumerate(([0, 1, 2], [1, 3, 4])):
            expected = {}
            for name, values in (("response", y), ("background", b)):
                expected[name] = str(Fraction(float(values[indices[0]]))
                                     - 2*Fraction(float(values[indices[1]]))
                                     + Fraction(float(values[indices[2]])))
            expected.update(response_error="2", background_error="2")
            self.assertEqual(result[i], expected)

    def test_nonfinite_and_incomplete_support_never_drop_rows(self):
        y = np.arange(5, dtype=float)
        for malformed in (np.array([1., 2., np.nan, 4., 5.]), y[:-1], y[:, None]):
            with self.assertRaisesRegex(ValueError, "Finite complete fixed support"):
                diagnostics.contrasts_from_points(malformed, y, self.points, self.stencils)

    def test_stencil_weight_changes_are_rejected(self):
        changed = deepcopy(self.stencils)
        changed[0]["weights"] = [1, -1, 0]
        with self.assertRaisesRegex(ValueError, "unchanged V47"):
            diagnostics.contrasts_from_points(np.zeros(5), np.zeros(5), self.points, changed)

    def test_duplicate_fractional_and_missing_points_are_rejected(self):
        for coords in (self.points[:4]+[self.points[0]],
                       [[8.5, 8]]+self.points[1:]):
            with self.assertRaisesRegex(ValueError, "Unique integer"):
                diagnostics.contrasts_from_points(np.zeros(5), np.zeros(5), coords, self.stencils)
        changed = deepcopy(self.stencils)
        changed[0]["pixels_xy"][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, "missing from fixed support"):
            diagnostics.contrasts_from_points(np.zeros(5), np.zeros(5), self.points, changed)

    def test_wrong_stencil_size_is_not_silently_zipped(self):
        changed = deepcopy(self.stencils)
        changed[0]["pixels_xy"].append([16, 16])
        with self.assertRaisesRegex(ValueError, "unchanged V47"):
            diagnostics.contrasts_from_points(np.zeros(5), np.zeros(5), self.points, changed)

    def test_unit_gain_violation_keeps_exact_boundary_and_empty_case(self):
        cases = [
            dict(response="6", background="2", response_error="2", background_error="2"),
            dict(response="9007199254740993/2251799813685248", background="0",
                 response_error="2", background_error="2"),
            dict(response="-8", background="1", response_error="2", background_error="2"),
        ]
        result = diagnostics.unit_gain_violations(cases)
        self.assertEqual(result["violating_stencil_indices"], [1, 2])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["maximum_excess_exact"], "5")
        self.assertGreater(result["all_excess_contrast_dn"][1], 0)
        self.assertEqual(diagnostics.unit_gain_violations([])["maximum_excess_exact"], "0")


class PlaneFitTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[8,8], [24,16], [72,8], [120,16], [8,72],
                                [24,120], [80,120], [120,120], [112,80]], dtype=float)
        self.b = np.array([1., 5., 2., 9., 3., 8., 7., 4., 6.])
        self.affine = np.column_stack((np.ones(len(self.points)),
                                      (self.points[:, 0]-64)/56,
                                      (self.points[:, 1]-64)/56))

    def test_positive_gain_affine_is_exact_on_all_points(self):
        y = 1.7*self.b + self.affine@np.array([6., -3., 2.])
        result = diagnostics.fit_guard_plane(y, self.b, self.points)
        self.assertTrue(result["available"])
        self.assertEqual(result["points"], len(self.points))
        self.assertEqual(result["rank"], 4)
        np.testing.assert_allclose(result["coefficients"], [1.7, 6., -3., 2.], atol=1e-12)
        self.assertLess(result["rmse_dn"], 1e-12)
        self.assertFalse(result["gain_nonnegative_constraint_active"])

    def test_negative_unconstrained_gain_refits_boundary_using_all_rows(self):
        y = -2.5*self.b + self.affine@np.array([8., 1., -2.])
        result = diagnostics.fit_guard_plane(y, self.b, self.points)
        expected_beta = np.linalg.lstsq(self.affine, y, rcond=None)[0]
        expected_residual = y-self.affine@expected_beta
        self.assertTrue(result["gain_nonnegative_constraint_active"])
        self.assertEqual(result["coefficients"][0], 0.)
        np.testing.assert_allclose(result["coefficients"][1:], expected_beta, atol=1e-12)
        np.testing.assert_allclose(result["residual_dn"], expected_residual, atol=1e-12)
        self.assertEqual(len(result["residual_dn"]), len(self.points))

    def test_gradient_addition_keeps_all_points_and_is_marked_diagnostic(self):
        gradient = np.array([[0.,1.], [1.,0.], [2.,1.], [-1.,2.], [3.,4.],
                             [2.,-3.], [-2.,1.], [1.,4.], [3.,2.]])
        beta = np.array([1.3, 2., 1., -1., 0.2, -0.4])
        design = np.column_stack((self.b, self.affine, gradient))
        result = diagnostics.fit_guard_plane(design@beta, self.b, self.points, gradient)
        self.assertEqual(result["points"], len(self.points))
        self.assertEqual(result["parameters"], 6)
        self.assertTrue(result["in_sample_diagnostic_not_validated_registration"])
        np.testing.assert_allclose(result["coefficients"], beta, atol=1e-12)

    def test_missing_gradient_makes_whole_diagnostic_unknown(self):
        y = self.b + 2
        bad = np.zeros((len(self.points), 2))
        bad[4, 0] = np.nan
        for gradient in (bad, np.zeros((len(self.points)-1, 2))):
            result = diagnostics.fit_guard_plane(y, self.b, self.points, gradient)
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], "gradient_neighbors_missing_on_fixed_support")
            self.assertNotIn("coefficients", result)

    def test_nonfinite_base_values_raise_instead_of_dropping_a_point(self):
        y = self.b+3
        for target in ("current", "background", "points"):
            args = [y.copy(), self.b.copy(), self.points.copy()]
            index = ["current", "background", "points"].index(target)
            if index == 2:
                args[index][0, 1] = np.nan
            else:
                args[index][0] = np.nan
            with self.assertRaisesRegex(ValueError, "No row dropping|finite Nx2"):
                diagnostics.fit_guard_plane(*args)

    def test_malformed_shapes_cannot_broadcast(self):
        for y, b, p in ((self.b[:, None], self.b, self.points),
                        (self.b, self.b[:-1], self.points),
                        (self.b, self.b, self.points[:, 0]),
                        (np.empty(0), np.empty(0), np.empty((0, 2)))):
            with self.assertRaisesRegex(ValueError, "Matching one-dimensional"):
                diagnostics.fit_guard_plane(y, b, p)


class GeneratedIntegrationTests(unittest.TestCase):
    def test_generated_packet_reconstructs_and_renders_without_mutation(self):
        yy, xx = np.indices((129, 129), dtype=float)
        scene = 100. + 2*np.cos(xx/7) + 3*np.sin(yy/9)
        centers = np.array([[50.+i, 64.] for i in range(8)])
        history = np.array([scene + 12*np.exp(-((xx-cx)**2+(yy-cy)**2)/2)
                            for cx, cy in centers])
        current = scene + 12*np.exp(-((xx-58.)**2+(yy-64.)**2)/2)
        current[0, 0] = np.nan
        packet = dict(history129=history, current129=current, prior_centers_xy=centers,
                      predicted_offset_xy=np.array([0., 0.]))
        components = diagnostics.prepare_components(history, centers.tolist(), [0., 0.], "bright")
        background = components["background"]
        guard = diagnostics.estimate_guard_gain(current, history, background,
                  np.where(np.isfinite(background), .5, np.nan), centers.tolist())
        row = generated_row(100)
        row["geometry"]["geometry"]["current_to_prior_matrices"] = np.tile(np.eye(3), (8, 1, 1)).tolist()
        row["arms"][diagnostics.ARM]["gain_calibration"] = guard
        row["arms"][diagnostics.ARM]["raw_adapter_result"]["learned_design_sha256"] = {
            "background": diagnostics._hash_array(background)}
        initial = {k: v.copy() for k, v in packet.items()}
        result, samples, rebuilt, repeated = diagnostics.analyze_packet(row, packet)
        self.assertEqual(repeated, guard)
        np.testing.assert_array_equal(rebuilt, background)
        self.assertTrue(result["complete_guard_record_reproduced"])
        self.assertTrue(result["independent_exact_contrasts_reproduced"])
        self.assertEqual(len(result["prior_leave_one_out"]), 8)
        self.assertEqual(samples["contrast_constraints"], guard["contrast_constraints"])
        self.assertEqual(result["used_points"], len(samples["current_values"]))
        self.assertTrue(result["support_unchanged"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"generated_history.png"
            diagnostics.render_case(path, packet, background, guard, result)
            with diagnostics.Image.open(path) as image:
                self.assertEqual(image.size, (1136, 962))
                self.assertEqual(image.mode, "RGB")
                image.load()
        for k, values in initial.items():
            np.testing.assert_array_equal(packet[k], values)


if __name__ == "__main__":
    unittest.main()
