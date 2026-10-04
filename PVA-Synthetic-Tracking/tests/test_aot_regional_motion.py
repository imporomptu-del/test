"""Generated correspondences only: no AOT outcomes, pixels, or remote access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        "generated_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lattice():
    return np.array([[x, y] for y in range(24, 2048, 48)
                     for x in range(24, 2448, 48)], dtype=np.float64)


def background(points, affine=False):
    points = np.asarray(points, dtype=np.float64)
    if not affine:
        return np.tile([3., -2.], (len(points), 1))
    x, y = points[:, 0] - 1224, points[:, 1] - 1024
    return np.column_stack((3 + .008*x + .004*y, -2 - .003*x + .006*y))


class RegionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script("compare_aot_regional_motion")
        cls.helper = load_script("compare_aot_motion_models")


class RegionalGeometryTests(RegionalBase):
    def test_all_48_half_open_cells_partition_points_exactly_once(self):
        points = np.array([[306*c, 2048*r/6] for r in range(6) for c in range(8)]
                          + [[2447.999, 2047.999]], float)
        assignments = np.zeros(len(points), int)
        for cell in range(48):
            masks = self.analysis.cell_masks(points, cell)
            expected = [cell] + ([48] if cell == 47 else [])
            self.assertEqual(masks["test_indices"].tolist(), expected)
            assignments[masks["test_indices"]] += 1
            train, test, guard = (set(masks[key].tolist()) for key in
                                 ("global_train_indices", "test_indices", "guard_indices"))
            self.assertFalse(train & test or train & guard or test & guard)
            self.assertEqual(train | test | guard, set(range(len(points))))
            self.assertTrue(set(masks["local_train_indices"].tolist()) <= train)
        np.testing.assert_array_equal(assignments, 1)

    def test_guard_is_clipped_expanded_rectangle_not_cross_strips(self):
        points = lattice()
        for cell in range(48):
            row, column = divmod(cell, 8)
            x0, x1 = 306*column, 306*(column+1)
            y0, y1 = 2048*row/6, 2048*(row+1)/6
            excluded = ((points[:, 0] >= max(0, x0-64)) &
                        (points[:, 0] < min(2448, x1+64)) &
                        (points[:, 1] >= max(0, y0-64)) &
                        (points[:, 1] < min(2048, y1+64)))
            masks = self.analysis.cell_masks(points, cell)
            np.testing.assert_array_equal(masks["global_train_indices"], np.flatnonzero(~excluded))
            center = np.array([(x0+x1)/2, (y0+y1)/2])
            local = ~excluded & (np.linalg.norm(points-center, axis=1) <= 512)
            np.testing.assert_array_equal(masks["local_train_indices"], np.flatnonzero(local))

    def test_outer_guard_boundary_is_trainable_and_radius_512_is_inclusive(self):
        cy = 2048*3.5/6
        points = np.array([[1224+64-1e-5, cy], [1224+64, cy],
                           [1071+512, cy], [1071+512+1e-5, cy],
                           [918-64, cy], [918-64-1e-5, cy]], float)
        masks = self.analysis.cell_masks(points, 27)
        self.assertEqual(masks["guard_indices"].tolist(), [0, 4])
        self.assertEqual(masks["global_train_indices"].tolist(), [1, 2, 3, 5])
        self.assertEqual(masks["local_train_indices"].tolist(), [1, 2, 5])

    def test_invalid_native_coordinates_and_cell_ids_fail_closed(self):
        for point in ([-1, 0], [2448, 0], [0, 2048], [np.nan, 0]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                self.analysis.cell_masks(np.array([point]), 0)
        for cell in (-1, 48, True, 1.5):
            with self.subTest(cell=cell), self.assertRaises((ValueError, TypeError)):
                self.analysis.cell_masks(np.array([[100., 100.]]), cell)

    def test_hull_is_deterministic_ccw_and_deduplicates_collinear_edges(self):
        points = np.array([[1, 1], [0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, 0]], float)
        expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        for order in (np.arange(len(points)), np.arange(len(points))[::-1]):
            np.testing.assert_array_equal(self.analysis.convex_hull(points[order]), expected)

    def test_degenerate_hulls_never_supply_support(self):
        for points in (np.empty((0, 2)), [[0, 0]], [[0, 0], [1, 1]],
                       [[0, 0], [1, 1], [2, 2], [1, 1]]):
            hull = self.analysis.convex_hull(np.asarray(points, float))
            self.assertEqual(len(hull), 0)
            np.testing.assert_array_equal(self.analysis.hull_contains(hull, np.array([[0., 0.]])), [False])

    def test_hull_boundary_tolerance_is_cross_product_not_pixel_expansion(self):
        hull = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        query = np.array([[0, 0], [10, 5], [5, 5], [-.5e-9, 5], [-2e-9, 5], [11, 5]], float)
        np.testing.assert_array_equal(self.analysis.hull_contains(hull, query),
                                      [True, True, True, True, False, False])


def support_points(counts=(6, 6, 6, 6)):
    centers = [(340, 380), (660, 380), (340, 740), (660, 740), (980, 740)]
    return np.array([[x+i*.1, y+i*.2] for (x, y), count in zip(centers, counts)
                     for i in range(count)], float)


class RegionalSupportTests(RegionalBase):
    def patched_fit(self, points, residuals=None, condition=None, arm="local_translation"):
        original = self.helper.fit_model
        def replace(*args, **kwargs):
            fit = original(*args, **kwargs)
            if residuals is not None:
                fit["training_residual_xy"] = np.asarray(residuals).tolist()
            if condition is not None:
                fit["condition_number"] = condition
            return fit
        with patch.object(self.helper, "fit_model", side_effect=replace) as spy:
            fit = self.analysis.fit_arm(self.helper, points, points+[3, -2],
                                        np.arange(len(points)), arm)
        self.assertEqual(spy.call_count, 1, "Coherent points must never trigger refitting")
        return fit

    def test_clean_local_translation_and_affine_recover_analytic_background(self):
        points = lattice()
        for arm, affine in (("local_translation", False), ("local_affine", True)):
            current = points + background(points, affine)
            indices = self.analysis.cell_masks(points, 27)["local_train_indices"]
            fitted = self.analysis.fit_arm(self.helper, points, current, indices, arm)
            self.assertTrue(fitted["training_eligible"], fitted["gate_reasons"])
            self.assertEqual(fitted["fit"]["solver_calls"], 21)
            self.assertEqual(fitted["fit"]["iterations_completed"], 20)
            query = np.array([[1071, 2048*3.5/6]])
            out = self.analysis.predict_arm(self.helper, fitted, query)
            self.assertEqual(out["available"], [True])
            np.testing.assert_allclose(out["predicted_displacement_xy"], background(query, affine), atol=1e-8)

    def test_minimum_training_count_and_cell_gates_are_independent(self):
        for points, expected in ((support_points()[:-1], "training_points_below_24"),
                                 (support_points((8, 8, 8)), "training_cells_below_4")):
            for arm in ("local_translation", "local_affine"):
                with patch.object(self.helper, "fit_model") as solver:
                    fit = self.analysis.fit_arm(self.helper, points, points+[1, 0],
                                                np.arange(len(points)), arm)
                self.assertFalse(fit["training_eligible"])
                self.assertIn(expected, fit["gate_reasons"])
                solver.assert_not_called()
        points = support_points()
        fit = self.analysis.fit_arm(self.helper, points, points+[1, 0], np.arange(24), "local_translation")
        self.assertTrue(fit["training_eligible"], fit["gate_reasons"])

    def test_coherent_residual_and_condition_bounds_are_inclusive(self):
        points = support_points()
        fit = self.patched_fit(points, np.tile([2., 0.], (24, 1)), 1000.)
        self.assertTrue(fit["training_eligible"])
        self.assertEqual(fit["coherent_indices"], list(range(24)))
        fit = self.patched_fit(points, np.tile([2.+1e-8, 0.], (24, 1)), 1000.+1e-8)
        self.assertFalse(fit["training_eligible"])
        self.assertEqual(fit["coherent_indices"], [])
        self.assertIn("weighted_design_condition_above_1000_or_nonfinite", fit["gate_reasons"])
        self.assertIn("coherent_points_below_12", fit["gate_reasons"])

    def test_coherent_mass_uses_original_cell_balance_not_robust_weights_or_count(self):
        points = support_points((40, 4, 4, 4))
        residuals = np.tile([3., 0.], (len(points), 1))
        residuals[40:] = 0
        residuals[:4] = 0
        fit = self.patched_fit(points, residuals)
        self.assertTrue(fit["training_eligible"], fit["gate_reasons"])
        self.assertAlmostEqual(fit["coherent_base_mass_fraction"], .775)
        self.assertLess(fit["coherent_count"] / len(points), .6)

    def test_exact_mass_threshold_and_coherent_count_cell_failures(self):
        points = support_points((10, 10, 10, 10, 10))
        residuals = np.tile([3., 0.], (50, 1))
        coherent = np.array([i for cell in range(5) for i in range(cell*10, cell*10+6)])
        residuals[coherent] = 0
        fit = self.patched_fit(points, residuals)
        self.assertTrue(fit["training_eligible"], fit["gate_reasons"])
        self.assertAlmostEqual(fit["coherent_base_mass_fraction"], .6)
        residuals[coherent[-1]] = [3, 0]
        fit = self.patched_fit(points, residuals)
        self.assertIn("coherent_original_base_mass_below_0_6", fit["gate_reasons"])
        residuals[:] = [3, 0]
        residuals[:30] = 0
        fit = self.patched_fit(points, residuals)
        self.assertIn("coherent_cells_below_4", fit["gate_reasons"])
        residuals[:] = [3, 0]
        residuals[[0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31]] = 0
        fit = self.patched_fit(points, residuals)
        self.assertIn("coherent_points_below_12", fit["gate_reasons"])

    def test_global_reference_does_not_receive_local_consistency_or_hull_gate(self):
        points = lattice()
        fit = self.patched_fit(points, np.tile([20., 30.], (len(points), 1)), 1e5,
                               arm="global_translation")
        self.assertTrue(fit["training_eligible"])
        self.assertIsNone(fit["hull"])
        out = self.analysis.predict_arm(self.helper, fit, np.array([[0., 0.], [2447., 2047.]]))
        self.assertEqual(out["available"], [True, True])

    def test_pointwise_hull_abstains_without_nearest_or_global_fallback(self):
        points = support_points()
        fitted = self.analysis.fit_arm(self.helper, points, points+[3, -2],
                                       np.arange(len(points)), "local_translation")
        query = np.array([[500, 550], [900, 550], [340, 380]], float)
        out = self.analysis.predict_arm(self.helper, fitted, query)
        self.assertEqual(out["available"], [True, False, True])
        self.assertIsNone(out["predicted_displacement_xy"][1])
        self.assertEqual(out["unavailable_reason"][1], "outside_coherent_training_hull")
        self.assertNotIn([900., 550.], fitted["hull"])

    def test_collinear_training_abstains_for_both_local_models(self):
        points = np.array([[i*50+30, i*30+30] for i in range(24)], float)
        for arm in ("local_translation", "local_affine"):
            fitted = self.analysis.fit_arm(self.helper, points, points+[1, 0], np.arange(24), arm)
            self.assertFalse(fitted["training_eligible"])
            out = self.analysis.predict_arm(self.helper, fitted, points[:1])
            self.assertEqual(out["available"], [False])
            self.assertEqual(out["predicted_displacement_xy"], [None])

    def test_guard_and_heldout_q_change_neither_fit_support_weights_nor_predictions(self):
        points = lattice()
        current = points + background(points, True)
        masks = self.analysis.cell_masks(points, 27)
        changed = current.copy()
        changed[masks["test_indices"]] += [123, -99]
        changed[masks["guard_indices"]] += [-234, 321]
        before_p, before_q = points.copy(), current.copy()
        first = self.analysis.evaluate_cell(self.helper, points, current, 27)
        second = self.analysis.evaluate_cell(self.helper, points, changed, 27)
        self.assertEqual(first, second)
        local = [first["arms"][arm] for arm in ("local_translation", "local_affine")]
        self.assertEqual(local[0]["train_indices"], local[1]["train_indices"])
        self.assertEqual(local[0]["fit"]["base_weights"], local[1]["fit"]["base_weights"])
        np.testing.assert_array_equal(points, before_p)
        np.testing.assert_array_equal(current, before_q)

    def test_cell_balance_recomputed_only_on_selected_training_inventory(self):
        points = support_points()
        extra = np.tile(points[0], (100, 1))
        all_points = np.vstack((extra, points))
        indices = np.arange(100, 124)
        fitted = self.analysis.fit_arm(self.helper, all_points, all_points+[3, -2], indices,
                                       "local_translation")
        np.testing.assert_allclose(fitted["fit"]["base_weights"], np.repeat(1/6, 24))
        self.assertEqual(fitted["coherent_indices"], indices.tolist())


class RegionalTargetTests(RegionalBase):
    def test_all_18_clean_and_target_cases_preserve_excluded_independent_vectors(self):
        cy = 2048*3.5/6
        placements = (np.array([[1071., cy]]),
                      np.array([[1071+dx, cy+dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]),
                      np.array([[1224+dx, cy+dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]))
        case_count = 0
        for affine in (False, True):
            for queries in placements:
                base = lattice()
                previous = np.vstack((base, queries))
                current = previous + background(previous, affine)
                query_indices = np.arange(len(base), len(previous))
                cells = np.unique(self.helper.native_cells(queries))
                clean = {int(cell): self.analysis.evaluate_cell(self.helper, previous, current, int(cell))
                         for cell in cells}
                for offset in ([0, 0], [1, 0], [4, -2]):
                    case_count += 1
                    moved = current.copy()
                    moved[query_indices] += offset
                    for cell in cells:
                        result = self.analysis.evaluate_cell(self.helper, previous, moved, int(cell))
                        self.assertEqual(result, clean[int(cell)], "Excluded target q must not alter any fit or support")
                        for arm in self.analysis.ARMS:
                            training = result["arms"][arm]["train_indices"]
                            self.assertFalse(set(query_indices) & set(training))
                            for index in set(query_indices) & set(result["test_indices"]):
                                position = result["test_indices"].index(index)
                                prediction = result["arms"][arm]["queries"]["predicted_displacement_xy"][position]
                                if prediction is None:
                                    continue
                                predicted = np.asarray(prediction)
                                true_background = background(previous[index:index+1], affine)[0]
                                residual = moved[index]-previous[index]-predicted
                                np.testing.assert_allclose(residual, np.asarray(offset)+true_background-predicted,
                                                           rtol=0, atol=1e-12)
                                if arm == "local_affine" or (arm == "local_translation" and not affine):
                                    np.testing.assert_allclose(predicted, true_background, rtol=0, atol=1e-8)
                                    np.testing.assert_allclose(residual, offset, rtol=0, atol=1e-8)
                for cell in cells:
                    for arm in ("local_translation", "local_affine") if not affine else ("local_affine",):
                        for index in set(query_indices) & set(clean[int(cell)]["test_indices"]):
                            position = clean[int(cell)]["test_indices"].index(index)
                            self.assertTrue(clean[int(cell)]["arms"][arm]["queries"]["available"][position])
        self.assertEqual(case_count, 18)

    def test_sparse_target_is_unavailable_never_counted_as_preserved_zero_motion(self):
        previous = np.array([[1071., 2048*3.5/6]])
        result = self.analysis.evaluate_cell(self.helper, previous, previous+[7, -4], 27)
        self.assertEqual(result["test_indices"], [0])
        for arm in self.analysis.ARMS:
            out = result["arms"][arm]["queries"]
            self.assertEqual(out["available"], [False])
            self.assertEqual(out["predicted_displacement_xy"], [None])
            self.assertEqual(out["unavailable_reason"], ["training_ineligible"])


def actual_fixture():
    previous = np.array([[x, y] for y in np.linspace(40, 2008, 12)
                         for x in np.linspace(40, 2408, 16)], float)
    current = previous + [3, -2]
    selected = np.vstack((previous, [[500, 500], [501, 500], [502, 500]]))
    return dict(case_id="generated_adjacent", source_kind="aot_adjacent", previous_index=0,
        current_index=1, arm="half_gain16_complete", selected_count=len(selected),
        accepted_count=len(previous), lost_count=3, expected_shift_xy=None,
        original_fit=dict(quality_status="rejected"), saved_candidate_translation=None,
        points=dict(selected_previous_xy=selected.tolist(), accepted_previous_xy=previous.tolist(),
            accepted_current_xy=current.tolist(), accepted_selected_indices=list(range(len(previous))),
            saved_inlier_mask=[False]*len(previous), fixed_support_mask=None))


def patch_fixture(actual, index=0):
    return dict(id="generated_patch_"+str(index), source_kind="actual_residual_extremum",
        case_id=actual["case_id"], previous_index=actual["previous_index"], current_index=actual["current_index"],
        accepted_index=index, selected_index=actual["points"]["accepted_selected_indices"][index],
        previous_xy=actual["points"]["accepted_previous_xy"][index],
        current_xy=actual["points"]["accepted_current_xy"][index], roles=["minimum", "maximum"],
        comparison=dict(verdict="disagreement", saved_lk_distances=[99., 99.]),
        scales=[dict(template_size=size, qualified=True, available=True, best_offset_xy=[20, 10])
                for size in (33, 65)])


class RegionalCohortTests(RegionalBase):
    def test_independent_ncc_cohort_never_conditions_on_saved_lk_verdict(self):
        actual = actual_fixture()
        rows = [patch_fixture(actual, index) for index in range(4)]
        rows[1]["scales"][1]["best_offset_xy"] = [21.5, 10]
        rows[2]["scales"][1]["best_offset_xy"] = [21.500001, 10]
        rows[3]["scales"][0]["qualified"] = False
        result = self.analysis.join_patch_evidence(actual, rows)
        self.assertEqual(result["patch_indices"], [0, 1, 2, 3])
        self.assertEqual(result["independent_indices"], [0, 1])
        self.assertEqual(result["unresolved_patch_indices"], [2, 3])
        for row in rows:
            row["comparison"] = dict(verdict="agreement", saved_lk_distances=[0., 0.])
        self.assertEqual(self.analysis.join_patch_evidence(actual, rows), result)
        self.assertEqual(result["independent_offsets"][1], dict(ncc33=[20., 10.], ncc65=[21.5, 10.]))

    def test_multiple_roles_stay_single_observation_and_duplicate_or_mismatch_fails(self):
        actual = actual_fixture()
        row = patch_fixture(actual)
        self.assertEqual(self.analysis.join_patch_evidence(actual, [row])["patch_indices"], [0])
        with self.assertRaises(ValueError):
            self.analysis.join_patch_evidence(actual, [row, copy.deepcopy(row)])
        for field, value in (("accepted_index", True), ("selected_index", 99),
                             ("previous_xy", [0, 0]), ("current_xy", [0, 0])):
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analysis.join_patch_evidence(actual, [bad])
        for mutate in (lambda row: row["scales"].reverse(),
                       lambda row: row["scales"][0].update(available=False),
                       lambda row: row["scales"][0].update(best_offset_xy=[np.nan, 0])):
            bad = copy.deepcopy(row)
            mutate(bad)
            with self.assertRaises(ValueError):
                self.analysis.join_patch_evidence(actual, [bad])

    def test_other_case_and_synthetic_patch_records_do_not_enter_actual_cohort(self):
        actual = actual_fixture()
        first = patch_fixture(actual)
        other = copy.deepcopy(first)
        other["case_id"] = "other_case"
        synthetic = copy.deepcopy(first)
        synthetic["source_kind"] = "known_counterexample"
        result = self.analysis.join_patch_evidence(actual, [other, first, synthetic])
        self.assertEqual(result["patch_indices"], [0])

    def test_ncc_changes_never_influence_routing_fits_support_or_lk_population(self):
        actual = actual_fixture()
        rows = [patch_fixture(actual, i) for i in range(3)]
        original = copy.deepcopy((actual, rows))
        first = self.analysis.compare_case(self.helper, actual, rows)
        self.assertEqual((actual, rows), original)
        for row in rows:
            row["scales"][0].update(qualified=False, best_offset_xy=[-100, 50])
            row["comparison"] = dict(verdict="ambiguous")
        actual["points"]["saved_inlier_mask"] = [True]*actual["accepted_count"]
        second = self.analysis.compare_case(self.helper, actual, rows)
        self.assertEqual(first["cells"], second["cells"])
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["cohorts"]["all_accepted_lk"], second["cohorts"]["all_accepted_lk"])
        self.assertEqual(len(first["cells"]), 48)
        self.assertEqual((first["selected_count"], first["accepted_count"], first["lost_count"]), (195, 192, 3))
        self.assertEqual(second["cohorts"]["all_patch_samples"]["image_evidence_counts"]["unresolved"], 3)
        self.assertEqual(second["cohorts"]["independent_patch_references"]["total_count"], 0)

    def test_pairwise_and_all_three_use_identical_queries_and_retain_missing_denominators(self):
        masks = ([True, True, True, True], [False, True, True, False], [True, False, True, False])
        predictions = {arm: dict(available=list(mask), predicted_displacement_xy=
            [[float(i+1), 0.] if available else None for available in mask])
            for i, (arm, mask) in enumerate(zip(self.analysis.ARMS, masks))}
        refs = dict(ncc33=np.zeros((4, 2)), ncc65=np.tile([1., 0.], (4, 1)))
        result = self.analysis.cohort_summary(predictions, [0, 1, 2, 3], refs)
        expected = ([1, 2], [0, 2], [2])
        for (first, second), indices in zip(self.analysis.PAIRS, expected):
            comparison = result["pairwise"][first+"__"+second]
            self.assertEqual(comparison["common_indices"], indices)
            self.assertEqual(comparison["total_count"], 4)
            self.assertEqual(comparison["missing_count"], 4-len(indices))
            values = comparison["references"]["ncc33"]["paired_difference"]
            self.assertEqual(values["direction"], "second minus first")
            self.assertEqual(values["values"], [float(self.analysis.ARMS.index(second)-self.analysis.ARMS.index(first))]*len(indices))
        self.assertEqual(result["all_three_common"]["common_indices"], [2])
        self.assertEqual(result["all_three_common"]["missing_count"], 3)
        global_errors = result["arms"]["global_translation"]["errors"]
        self.assertEqual(global_errors["ncc33"]["values"], [1.]*4)
        self.assertEqual(global_errors["ncc65"]["values"], [0.]*4)
        local = result["arms"]["local_translation"]
        self.assertEqual(local["errors"]["ncc33"]["values"], [None, 2., 2., None])
        self.assertEqual(local["unavailable_count"], 2)

    def test_empty_cohorts_and_zero_common_support_stay_unavailable(self):
        predictions = {arm: dict(available=[False], predicted_displacement_xy=[None])
                       for arm in self.analysis.ARMS}
        for indices, reference in (([0], np.array([[1., 0.]])), ([], np.empty((0, 2)))):
            result = self.analysis.cohort_summary(predictions, indices, dict(ncc33=reference))
            self.assertEqual(result["all_three_common"]["common_count"], 0)
            for arm in self.analysis.ARMS:
                summary = result["arms"][arm]["errors"]["ncc33"]["summary"]
                self.assertEqual(summary["count"], 0)
                self.assertIsNone(summary["median"])
                self.assertIsNone(summary["maximum"])

    def test_collect_predictions_keeps_original_positions_and_rejects_missing_duplicate_tests(self):
        def cell(indices):
            return dict(test_indices=indices, arms={arm: dict(queries=dict(available=[False]*len(indices),
                predicted_displacement_xy=[None]*len(indices), unavailable_reason=["training_ineligible"]*len(indices)))
                for arm in self.analysis.ARMS})
        cells = [cell([1]), cell([]), cell([0, 2])]
        result = self.analysis.collect_predictions(cells, 3)
        for arm in self.analysis.ARMS:
            self.assertEqual(result[arm]["predicted_displacement_xy"], [None]*3)
        for bad in ([cell([0, 1]), cell([1, 2])], [cell([0, 1])]):
            with self.assertRaises(ValueError):
                self.analysis.collect_predictions(bad, 3)


class RegionalGeneratedInventoryTests(RegionalBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.controls = load_script("aot_regional_generated_controls")
        cls.result = cls.controls.run_generated(cls.helper, cls.analysis.evaluate_cell, cls.analysis.ARMS)

    def test_exact_fixed_18_target_and_8_contaminant_cases(self):
        targets = self.result["target_preservation"]
        nuisance = self.result["training_contamination"]
        self.assertTrue(self.result["passed"])
        self.assertFalse(self.result["source_images_used"])
        self.assertEqual((len(targets["cases"]), len(targets["paired_checks"]),
                          len(nuisance["cases"]), len(nuisance["paired_checks"])), (18, 12, 8, 4))
        self.assertEqual(len({case["id"] for case in targets["cases"]+nuisance["cases"]}), 26)
        expected = {(kind, placement, offset) for kind in ("translation", "affine")
                    for placement in ("center_single", "center_3x3", "right_boundary_3x3")
                    for offset in ((0, 0), (1, 0), (4, -2))}
        self.assertEqual({(case["background"], case["placement"], tuple(case["target_offset_xy"]))
                          for case in targets["cases"]}, expected)
        np.testing.assert_array_equal(self.controls.lattice(), lattice())
        for case in targets["cases"]:
            p = np.asarray(case["query_xy"])
            np.testing.assert_allclose(case["analytic_background_displacement_xy"],
                                       background(p, case["background"] == "affine"), atol=1e-12)
            if case["placement"] == "right_boundary_3x3":
                self.assertEqual(sorted({cell["cell_id"] for cell in case["cells"]}), [27, 28])

    def test_nuisance_controls_enter_training_while_query_remains_excluded(self):
        for case in self.result["training_contamination"]["cases"]:
            nuisance = set(case["nuisance_indices"])
            self.assertEqual(len(nuisance), case["nuisance_size_points"])
            self.assertEqual(case["target_offset_xy"], [4, -2])
            for cell in case["cells"]:
                self.assertFalse(nuisance & set(cell["guard_indices"]))
                for arm in self.analysis.ARMS:
                    train = set(cell["arms"][arm]["train_indices"])
                    self.assertTrue(nuisance <= train)
                    self.assertFalse(set(case["query_indices"]) & train)
        for pair in self.result["training_contamination"]["paired_checks"]:
            for arm in self.analysis.ARMS:
                for row in pair["arms"][arm]:
                    if row["clean_available"] and row["contaminated_available"]:
                        np.testing.assert_allclose(np.asarray(row["background_prediction_drift_xy"])
                            + row["compensated_target_change_xy"], [0, 0], atol=1e-8)
                    else:
                        self.assertIsNone(row["background_prediction_drift_px"])

    def test_target_checks_reject_fit_changes_and_vector_loss(self):
        baseline, moving = copy.deepcopy(self.result["target_preservation"]["cases"][:2])
        altered = copy.deepcopy(moving)
        altered["cells"][0]["arms"]["local_translation"]["coherent_count"] = -1
        with self.assertRaises(ValueError):
            self.controls.target_checks(baseline, altered, self.analysis.ARMS)
        altered = copy.deepcopy(moving)
        altered["query_results"]["local_translation"][0]["compensated_target_xy"] = [0., 0.]
        with self.assertRaises(ValueError):
            self.controls.target_checks(baseline, altered, self.analysis.ARMS)

    def test_generated_unavailable_query_never_becomes_preservation_pass(self):
        p = np.array([[1071., 2048*3.5/6]])
        cases = []
        for offset in ([0, 0], [4, -2]):
            case = self.controls.evaluate_queries(self.helper, self.analysis.evaluate_cell, self.analysis.ARMS,
                p, p+[3, -2]+offset, [0], "translation", offset, True)
            case["id"] = str(offset)
            cases.append(case)
        result = self.controls.target_checks(*cases, self.analysis.ARMS)
        for arm in self.analysis.ARMS:
            self.assertEqual(result["arms"][arm]["supported_preservation_checks"], 0)
            self.assertEqual(result["arms"][arm]["unavailable_not_passed"], 1)
            self.assertIsNone(result["arms"][arm]["max_vector_difference_error_px"])


class RegionalGuardTests(RegionalBase):
    def manifest_fixture(self):
        hashes = {key: expected for key, (_, expected) in self.analysis.INPUT_PINS.items()}
        hashes["model_helper_sha256"] = self.analysis.HELPER_SHA
        hashes.update({key: "a"*64 for key in ("script_sha256", "tests_sha256", "plan_sha256", "generated_controls_sha256")})
        manifest = dict(schema=self.analysis.PLAN_SCHEMA, design=copy.deepcopy(self.analysis.DESIGN), **hashes)
        return manifest, hashes

    def test_manifest_exact_design_input_helper_and_all_new_artifacts_are_bound(self):
        manifest, hashes = self.manifest_fixture()
        self.analysis.validate_manifest(manifest, hashes)
        for key in hashes:
            changed = copy.deepcopy(manifest)
            changed[key] = "b"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)
            changed = copy.deepcopy(hashes)
            changed[key] = "b"*64
            with self.subTest(actual=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(manifest, changed)
        for key, value in (("guard_px", 0), ("local_radius_px", 513), ("huber_iterations", 19),
                           ("coherent_base_mass_fraction", .5), ("automatic_model_selection", True),
                           ("target_case_count", 17), ("grid_rows", 6.0)):
            changed = copy.deepcopy(manifest)
            changed["design"][key] = value
            with self.subTest(design=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)

    def test_scope_and_existing_result_failure_or_symlink_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "regional_motion_01"
            output.mkdir()
            other = root / "other"
            other.mkdir()
            with patch.object(self.analysis, "OUTPUT_DIR", output):
                self.assertEqual(self.analysis.scope_output(output), output)
                for bad in (root, other, output / "nested"):
                    with self.assertRaises(ValueError):
                        self.analysis.scope_output(bad)
                for filename in ("result.json", "failure.json"):
                    path = output / filename
                    path.write_text("preserved")
                    with self.assertRaises(ValueError):
                        self.analysis.scope_output(output)
                    self.assertEqual(path.read_text(), "preserved")
                    path.unlink()
                    path.symlink_to(root / "missing")
                    with self.assertRaises(ValueError):
                        self.analysis.scope_output(output)
                    path.unlink()
            linked = root / "linked"
            linked.symlink_to(output, target_is_directory=True)
            with patch.object(self.analysis, "OUTPUT_DIR", linked), self.assertRaises(ValueError):
                self.analysis.scope_output(linked)

    def test_input_hash_and_module_import_fail_before_untrusted_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.txt"
            source.write_bytes(b"generated evidence")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(self.analysis, "EVIDENCE_ROOT", root), \
                    patch.object(self.analysis, "INPUT_PINS", {"fixture_sha256": (source.name, digest)}):
                self.assertEqual(self.analysis.input_hashes(), {"fixture_sha256": digest})
                source.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    self.analysis.input_hashes()
            with patch.object(self.analysis.importlib.util, "spec_from_file_location") as loader, self.assertRaises(ValueError):
                self.analysis.load_module(source, digest, "must_not_import")
            loader.assert_not_called()

    def test_exclusive_json_cannot_replace_existing_or_serialize_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"result.json"
            self.analysis.write_exclusive(output, {"generated": True})
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                self.analysis.write_exclusive(output, {"changed": True})
            self.assertEqual(output.read_bytes(), before)
            with self.assertRaises(ValueError):
                self.analysis.write_exclusive(Path(tmp)/"invalid.json", {"invalid": float("nan")})

    def test_run_stops_before_any_data_access_if_output_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            (output/"result.json").write_text("existing")
            with patch.object(self.analysis, "OUTPUT_DIR", output), \
                    patch.object(self.analysis, "bound_hashes") as hashes, self.assertRaises(ValueError):
                self.analysis.run(output)
            hashes.assert_not_called()
            self.assertEqual((output/"result.json").read_text(), "existing")

    def test_after_hash_mismatch_retains_failure_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            before = {"generated_controls_sha256": "a"*64}
            generated = SimpleNamespace(run_generated=lambda *args: dict(
                target_preservation=dict(cases=[{}]*18), training_contamination=dict(cases=[{}]*8)))
            with patch.object(self.analysis, "OUTPUT_DIR", output), \
                    patch.object(self.analysis, "bound_hashes", side_effect=[before, {**before, "changed": True}]), \
                    patch.object(self.analysis, "load_module", side_effect=[self.helper, generated]), \
                    patch.object(self.analysis, "json_read", return_value={}), \
                    patch.object(self.analysis, "validate_inputs", return_value=([dict(case_id=str(i)) for i in range(8)], [])), \
                    patch.object(self.analysis, "compare_case", side_effect=lambda helper, row, patches: row), \
                    patch("builtins.print"), self.assertRaises(ValueError):
                self.analysis.run(output)
            self.assertFalse((output/"result.json").exists())
            failure = json.loads((output/"failure.json").read_text())
            self.assertFalse(failure["passed"])
            self.assertEqual(failure["completed_cell_pairs"], 384)
            self.assertEqual(len(failure["rows"]), 8)
            self.assertIn("changed frozen", failure["error"])


if __name__ == "__main__":
    unittest.main()
