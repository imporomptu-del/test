"""Generated motion fields only; no saved experiment outcomes or media reads."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_aot_motion_models.py"


def load_comparison():
    spec = importlib.util.spec_from_file_location("aot_models_generated_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_grid():
    return np.array([[x, y] for y in np.linspace(40, 2008, 24)
                     for x in np.linspace(40, 2408, 30)], dtype=np.float64)


def transform(points, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:2, :2].T + matrix[:2, 2]


class MotionModelGeneratedFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_comparison()

    def test_all_models_recover_known_translation_without_truth_in_the_solver(self):
        points = native_grid()
        current = points + [3.25, -2.5]
        for model in ("translation", "similarity", "affine"):
            with self.subTest(model=model):
                fit = self.analysis.fit_model(points, current, model)
                self.assertTrue(fit["valid"], fit)
                predicted = self.analysis.predict(points, fit["parameters"], model)
                np.testing.assert_allclose(predicted, np.tile([3.25, -2.5], (len(points), 1)), atol=1e-8)
                np.testing.assert_allclose(fit["native_matrix"], [[1, 0, 3.25], [0, 1, -2.5], [0, 0, 1]], atol=1e-8)

    def test_similarity_recovers_rotation_scale_and_translation(self):
        points = native_grid()
        theta, scale = .012, 1.003
        a, b = scale*np.cos(theta), scale*np.sin(theta)
        matrix = [[a, -b, 2.5], [b, a, -3.0], [0, 0, 1]]
        current = transform(points, matrix)
        fit = self.analysis.fit_model(points, current, "similarity")
        self.assertTrue(fit["valid"], fit)
        np.testing.assert_allclose(fit["native_matrix"], matrix, atol=1e-8)
        np.testing.assert_allclose(self.analysis.predict(points, fit["parameters"], "similarity"), current-points, atol=1e-8)

    def test_affine_recovers_shear_and_anisotropic_scale(self):
        points = native_grid()
        matrix = [[1.002, .004, 2.0], [-.003, .998, -1.0], [0, 0, 1]]
        current = transform(points, matrix)
        fit = self.analysis.fit_model(points, current, "affine")
        self.assertTrue(fit["valid"], fit)
        np.testing.assert_allclose(fit["native_matrix"], matrix, atol=1e-8)
        np.testing.assert_allclose(self.analysis.predict(points, fit["parameters"], "affine"), current-points, atol=1e-8)

    def test_lower_order_model_does_not_gain_undeclared_affine_parameters(self):
        points = native_grid()
        matrix = [[1.002, .004, 2.0], [-.003, .998, -1.0], [0, 0, 1]]
        current = transform(points, matrix)
        for model in ("translation", "similarity"):
            fit = self.analysis.fit_model(points, current, model)
            errors = np.linalg.norm(self.analysis.predict(points, fit["parameters"], model)-(current-points), axis=1)
            self.assertGreater(float(np.median(errors)), .1)

    def test_affine_degeneracy_is_unavailable_not_zero_error(self):
        points = np.column_stack((np.linspace(0, 2000, 150), np.linspace(0, 1500, 150)))
        fit = self.analysis.fit_model(points, points+[2, -1], "affine")
        self.assertFalse(fit["valid"])
        self.assertIsNone(fit.get("parameters"))
        self.assertIsNone(fit.get("native_matrix"))

    def test_exact_twenty_huber_refits_match_independent_translation_reference(self):
        points = native_grid()[:20]
        displacements = np.column_stack((np.linspace(0, 3, 20), np.linspace(2, -1, 20)))
        displacements[0] = [40, -30]
        base = np.linspace(.1, 1, len(points))
        translation = np.sum(base[:, None]*displacements, axis=0) / base.sum()
        solver_weights = base.copy()
        for _ in range(20):
            error = np.linalg.norm(displacements-translation, axis=1)
            robust = np.ones(len(error))
            robust[error > 1] = 1/error[error > 1]
            solver_weights = base * robust
            translation = np.sum(solver_weights[:, None]*displacements, axis=0) / solver_weights.sum()
        final_error = np.linalg.norm(displacements-translation, axis=1)
        final_huber = np.ones(len(final_error))
        final_huber[final_error > 1] = 1/final_error[final_error > 1]
        fit = self.analysis.fit_model(points, points+displacements, "translation", base)
        self.assertTrue(fit["valid"])
        self.assertEqual(fit["iterations_completed"], 20)
        self.assertEqual(fit["solver_calls"], 21)
        np.testing.assert_allclose(fit["parameters"], translation, atol=1e-12)
        np.testing.assert_allclose(fit["base_weights"], base, atol=1e-15)
        np.testing.assert_allclose(fit["final_solver_weights"], solver_weights, atol=1e-12)
        np.testing.assert_allclose(fit["final_residual_huber_weights"], final_huber, atol=1e-12)

    def test_joint_euclidean_huber_is_not_separate_component_clipping(self):
        residuals = np.array([[0, 0], [1, 0], [.8, .8], [3, 4]], float)
        np.testing.assert_allclose(self.analysis.huber_weights(residuals),
                                   [1, 1, 1/np.sqrt(1.28), .2])

    def test_perfect_fit_still_runs_initial_solve_and_twenty_refits(self):
        points = native_grid()
        for model in ("translation", "similarity", "affine"):
            fit = self.analysis.fit_model(points, points+[2, -1], model)
            self.assertEqual(fit["solver_calls"], 21)
            self.assertEqual(fit["iterations_completed"], 20)


class MotionModelFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_comparison()

    def test_half_open_quadrants_assign_every_point_to_exactly_one_test_fold(self):
        points = np.array([[0, 0], [1223.999, 1023.999], [1224, 0], [2447.999, 1023.999],
                           [0, 1024], [1223.999, 2047.999], [1224, 1024], [2447.999, 2047.999]])
        assignment = np.zeros(len(points), int)
        for fold in range(4):
            masks = self.analysis.fold_masks(points, fold)
            self.assertEqual(masks["test_indices"].tolist(), [2*fold, 2*fold+1])
            assignment[masks["test_indices"]] += 1
            train, test, guard = (set(masks[key].tolist()) for key in ("train_indices", "test_indices", "guard_indices"))
            self.assertFalse(train & test or train & guard or test & guard)
            self.assertEqual(train | test | guard, set(range(len(points))))
        np.testing.assert_array_equal(assignment, np.ones(len(points), int))

    def test_guard_is_expanded_test_rectangle_not_whole_image_cross(self):
        points = np.array([[1223, 1000], [1224, 1000], [1287.999, 1000], [1288, 1000],
                           [1000, 1087.999], [1000, 1088], [2000, 1024], [1224, 1600]])
        masks = self.analysis.fold_masks(points, 0)
        self.assertEqual(masks["test_indices"].tolist(), [0])
        self.assertEqual(masks["guard_indices"].tolist(), [1, 2, 4])
        self.assertEqual(masks["train_indices"].tolist(), [3, 5, 6, 7])

    def test_all_quadrant_guards_are_clipped_and_use_same_fixed_geometry(self):
        points = native_grid()
        for fold in range(4):
            x0, x1 = (0, 1224) if fold % 2 == 0 else (1224, 2448)
            y0, y1 = (0, 1024) if fold < 2 else (1024, 2048)
            excluded = ((points[:, 0] >= max(0, x0-64)) & (points[:, 0] < min(2448, x1+64))
                        & (points[:, 1] >= max(0, y0-64)) & (points[:, 1] < min(2048, y1+64)))
            masks = self.analysis.fold_masks(points, fold)
            np.testing.assert_array_equal(masks["train_indices"], np.flatnonzero(~excluded))

    def test_each_occupied_training_cell_has_equal_total_base_weight(self):
        points = np.array([[10, 10], [20, 10], [400, 10],
                           [2400, 2000], [2410, 2000], [2420, 2000]], float)
        result = self.analysis.cell_weights(points)
        self.assertEqual(result["occupied_cells"], 3)
        self.assertEqual(result["cell_indices"].tolist(), [0, 0, 1, 47, 47, 47])
        np.testing.assert_allclose(result["weights"], [.5, .5, 1, 1/3, 1/3, 1/3], atol=1e-15)
        for cell in (0, 1, 47):
            self.assertAlmostEqual(float(result["weights"][result["cell_indices"] == cell].sum()), 1)

    def test_heldout_and_guard_values_cannot_change_training_fit_or_cell_weights(self):
        previous = native_grid()
        current = transform(previous, [[1.002, .004, 2], [-.003, .998, -1], [0, 0, 1]])
        masks = self.analysis.fold_masks(previous, 0)
        perturbed_previous, perturbed_current = previous.copy(), current.copy()
        perturbed_previous[masks["test_indices"]] += [1, 1]
        perturbed_current[masks["test_indices"]] += [100, -90]
        perturbed_current[masks["guard_indices"]] += [-123, 321]
        for model in ("translation", "similarity", "affine"):
            left = self.analysis.evaluate_fold(previous, current, model, 0)
            right = self.analysis.evaluate_fold(perturbed_previous, perturbed_current, model, 0)
            with self.subTest(model=model):
                self.assertTrue(left["valid"] and right["valid"])
                self.assertEqual(left["train_indices"], right["train_indices"])
                self.assertEqual(left["train_cell_counts"], right["train_cell_counts"])
                self.assertEqual(left["fit"], right["fit"])
                self.assertNotEqual(left["test_error_px"], right["test_error_px"])

    def test_failed_training_preserves_test_count_with_unavailable_errors(self):
        points = np.array([[x, y] for y in range(50, 950, 100) for x in range(50, 1150, 100)], float)
        fold = self.analysis.evaluate_fold(points, points+[1, 0], "translation", 0)
        self.assertFalse(fold["valid"])
        self.assertEqual(fold["test_count"], len(points))
        self.assertEqual(fold["train_count"], 0)
        self.assertIsNone(fold["test_error_px"])
        self.assertIsNone(fold["test_error_summary"])
        self.assertIn("fewer_than_100_training_points", fold["unavailable_reasons"])
        self.assertIn("fewer_than_12_occupied_training_cells", fold["unavailable_reasons"])

    def test_empty_test_fold_is_unavailable_even_with_valid_training(self):
        points = native_grid()
        points = points[~((points[:, 0] < 1224) & (points[:, 1] < 1024))]
        fold = self.analysis.evaluate_fold(points, points+[1, 0], "translation", 0)
        self.assertFalse(fold["valid"])
        self.assertEqual(fold["test_count"], 0)
        self.assertTrue(fold["fit"]["valid"])
        self.assertIsNone(fold["test_error_px"])
        self.assertIsNone(fold["test_error_summary"])
        self.assertIn("empty_test_fold", fold["unavailable_reasons"])

    def test_all_models_use_identical_folds_and_training_balance(self):
        points = native_grid()
        for fold_id in range(4):
            folds = [self.analysis.evaluate_fold(points, points+[1, 0], model, fold_id)
                     for model in ("translation", "similarity", "affine")]
            for result in folds[1:]:
                for key in ("train_indices", "test_indices", "guard_indices", "train_cell_counts"):
                    self.assertEqual(result[key], folds[0][key])
                self.assertEqual(result["fit"]["base_weights"], folds[0]["fit"]["base_weights"])

    def test_minimum_training_point_and_cell_thresholds_are_inclusive_and_separate(self):
        centers = [(306*(column+.5), (2048/6)*(row+.5)) for row in range(3) for column in range(4, 8)]
        train = np.array([[*centers[i % 12]] for i in range(100)], float)
        train += np.column_stack((np.arange(100)*.001, np.arange(100)*.002))
        points = np.vstack(([100, 100], train))
        accepted = self.analysis.evaluate_fold(points, points+[1, 0], "translation", 0)
        self.assertEqual(accepted["train_count"], 100)
        self.assertEqual(accepted["train_occupied_cells"], 12)
        self.assertTrue(accepted["valid"])
        too_few = self.analysis.evaluate_fold(points[:-1], points[:-1]+[1, 0], "translation", 0)
        self.assertEqual(too_few["train_count"], 99)
        self.assertEqual(too_few["train_occupied_cells"], 12)
        self.assertFalse(too_few["valid"])
        self.assertIn("fewer_than_100_training_points", too_few["unavailable_reasons"])
        clustered = np.vstack(([100, 100], np.array([[1500+i*.01, 100] for i in range(100)])))
        low_coverage = self.analysis.evaluate_fold(clustered, clustered+[1, 0], "translation", 0)
        self.assertEqual(low_coverage["train_count"], 100)
        self.assertFalse(low_coverage["valid"])
        self.assertIn("fewer_than_12_occupied_training_cells", low_coverage["unavailable_reasons"])


def aggregate_fixture():
    points = np.array([[100, 100], [1300, 100], [100, 1200], [1300, 1200]], float)
    folds = [dict(fold_id=i, test_indices=[i], valid=True,
                  test_predicted_current_xy=[(points[i]+[1, 0]).tolist()], test_error_px=[2.0])
             for i in range(4)]
    return points, folds


def source_case_fixture(known=False):
    previous = native_grid()
    selected = np.vstack((previous, [[500+i*.1, 1500] for i in range(10)]))
    truth = np.array([1, 0], float)
    current = previous + truth + [.25, -.25]
    support = np.all((selected >= 128) & (selected < [2320, 1920])
                     & (selected+truth >= 128) & (selected+truth < [2320, 1920]), axis=1)
    return dict(ordinal=8 if known else 0, case_id="aot_prev000_dx+1_dy+0" if known else "aot_001_adjacent",
        source_kind="aot_known_shift" if known else "aot_adjacent", previous_index=0, current_index=1,
        arm="half_gain16_complete", selected_count=len(selected), accepted_count=len(previous), lost_count=10,
        expected_shift_xy=[1.0, 0.0] if known else None,
        original_fit=dict(quality_status="rejected", parameters=None), saved_candidate_translation=None,
        points=dict(selected_previous_xy=selected.tolist(), accepted_previous_xy=previous.tolist(),
            accepted_current_xy=current.tolist(), accepted_selected_indices=list(range(len(previous))),
            saved_inlier_mask=[False]*len(previous), fixed_support_mask=support.tolist() if known else None))


class MotionModelOutOfFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_comparison()

    def test_missing_fit_preserves_full_denominator_and_null_positions(self):
        points, folds = aggregate_fixture()
        folds[1].update(valid=False, test_predicted_current_xy=None, test_error_px=None)
        result = self.analysis.aggregate_folds(points, folds, known_shift=[1, 0])
        self.assertFalse(result["complete"])
        self.assertEqual((result["expected_count"], result["scored_count"], result["unscored_count"]), (4, 3, 1))
        self.assertEqual(result["error_px"], [2.0, None, 2.0, 2.0])
        self.assertIsNone(result["predicted_current_xy"][1])
        self.assertEqual(result["unscored_indices"], [1])
        self.assertEqual(result["error_summary"]["count"], 3)
        self.assertIn("not rankable", result["summary_scope"])
        self.assertEqual(result["known_transform_prediction_error_px"], [0.0, None, 0.0, 0.0])
        self.assertEqual(sum(cell["expected_count"] for cell in result["cell_errors"]), 4)
        self.assertEqual(sum(cell["unscored_count"] for cell in result["cell_errors"]), 1)
        self.assertIsNone(folds[1]["known_transform_prediction_error_px"])

    def test_all_failed_folds_are_unavailable_not_zero_error(self):
        points, folds = aggregate_fixture()
        for fold in folds:
            fold.update(valid=False, test_predicted_current_xy=None, test_error_px=None)
        result = self.analysis.aggregate_folds(points, folds)
        self.assertFalse(result["complete"])
        self.assertEqual(result["unscored_count"], 4)
        self.assertEqual(result["error_px"], [None]*4)
        self.assertIsNone(result["error_summary"]["median"])
        self.assertIsNone(result["error_summary"]["quantiles"])
        self.assertIsNone(result["worst_supported_cell_median"])
        self.assertEqual(result["supported_cell_count"], 0)

    def test_duplicate_missing_and_reordered_test_assignments_fail_closed(self):
        for mode in ("duplicate", "reordered", "missing", "count_mismatch"):
            points, folds = aggregate_fixture()
            if mode == "duplicate": folds[1]["test_indices"] = [0]
            elif mode == "reordered": folds.reverse()
            elif mode == "missing": folds.pop()
            else: folds[0]["test_error_px"] = []
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.analysis.aggregate_folds(points, folds)

    def test_cells_below_five_scored_points_are_unavailable_but_keep_counts(self):
        points = np.array([[10+i, 10] for i in range(5)] + [[2440, 2040]], float)
        cells = self.analysis.cell_errors(points, np.array([0, 1, 2, 3, 4, 9], float))
        self.assertEqual(len(cells), 48)
        self.assertEqual(cells[0]["count"], 5)
        self.assertEqual(cells[0]["median"], 2)
        self.assertAlmostEqual(cells[0]["p90"], 3.6)
        self.assertEqual(cells[-1]["count"], 1)
        self.assertIsNone(cells[-1]["median"])
        self.assertIsNone(cells[-1]["p90"])
        self.assertEqual(cells[1]["count"], 0)
        self.assertIsNone(cells[1]["median"])


class MotionModelCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_comparison()

    def test_mapping_truth_and_observed_correspondence_error_remain_distinct(self):
        source = source_case_fixture(known=True)
        before = copy.deepcopy(source)
        result = self.analysis.compare_case(source)
        self.assertEqual(source, before)
        self.assertEqual((result["selected_count"], result["accepted_count"], result["lost_count"]), (730, 720, 10))
        self.assertEqual(result["original_fixed_support"]["lost_count"], 10)
        self.assertEqual(result["original_fit"]["quality_status"], "rejected")
        for model in ("translation", "similarity", "affine"):
            out = result["models"][model]["out_of_fold"]
            self.assertTrue(out["complete"])
            self.assertEqual((out["expected_count"], out["scored_count"], out["unscored_count"]), (720, 720, 0))
            self.assertLess(out["error_summary"]["quantiles"]["max"], 1e-8)
            np.testing.assert_allclose(out["known_transform_prediction_error_px"], np.sqrt(.125), atol=1e-8)
            self.assertEqual(len(out["predicted_current_xy"]), 720)

    def test_original_inliers_do_not_select_training_and_actual_truth_is_absent(self):
        source = source_case_fixture()
        first = self.analysis.compare_case(source)
        source["points"]["saved_inlier_mask"] = [True]*source["accepted_count"]
        second = self.analysis.compare_case(source)
        self.assertEqual(first["models"], second["models"])
        self.assertIsNone(first["original_fixed_support"])
        self.assertIsNone(first["expected_shift_xy"])
        for model in ("translation", "similarity", "affine"):
            self.assertNotIn("known_transform_prediction_error_px", first["models"][model]["out_of_fold"])

    def test_incomplete_models_cannot_be_ranked_by_survivor_summaries(self):
        source = source_case_fixture()
        actual_evaluator = self.analysis.evaluate_fold
        def missing_one_fold(previous, current, model, fold_id):
            result = actual_evaluator(previous, current, model, fold_id)
            if model == "affine" and fold_id == 0:
                result.update(valid=False, test_predicted_current_xy=None, test_error_px=None,
                              unavailable_reasons=["generated_failure"])
            return result
        with patch.object(self.analysis, "evaluate_fold", side_effect=missing_one_fold):
            result = self.analysis.compare_case(source)
        difference = result["descriptive_differences"]["affine"]
        self.assertFalse(difference["comparable_complete_cases"])
        self.assertIsNone(difference["median_error_difference_vs_translation"])
        self.assertIsNone(difference["p90_error_difference_vs_translation"])
        self.assertGreater(result["models"]["affine"]["out_of_fold"]["unscored_count"], 0)

    def test_corrupt_selected_inventory_or_invented_actual_truth_is_rejected(self):
        for mode in ("lost_count", "reordered_indices", "previous_mismatch", "actual_truth", "known_support", "boolean_shift"):
            row = source_case_fixture(known=mode in ("known_support", "boolean_shift"))
            if mode == "lost_count": row["lost_count"] = 0
            elif mode == "reordered_indices": row["points"]["accepted_selected_indices"].reverse()
            elif mode == "previous_mismatch": row["points"]["accepted_previous_xy"][0] = [99, 99]
            elif mode == "actual_truth": row["expected_shift_xy"] = [1, 0]
            elif mode == "boolean_shift": row["expected_shift_xy"] = [True, 0.0]
            else: row["points"]["fixed_support_mask"][0] = not row["points"]["fixed_support_mask"][0]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.analysis.validate_case(row)


class MotionModelScopeManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_comparison()

    def manifest_fixture(self):
        hashes = {key: value[1] for key, value in self.analysis.INPUT_PINS.items()}
        hashes.update(script_sha256="a"*64, tests_sha256="b"*64, plan_sha256="c"*64)
        return dict(schema=self.analysis.PLAN_SCHEMA, design=copy.deepcopy(self.analysis.DESIGN), **hashes), hashes

    def test_exact_design_and_all_source_artifact_identities_are_bound(self):
        manifest, hashes = self.manifest_fixture()
        self.analysis.validate_manifest(manifest, hashes)
        for key in hashes:
            changed = copy.deepcopy(manifest)
            changed[key] = "f"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)
        for key, value in (("guard_px", 0), ("huber_iterations", 19), ("huber_delta_px", 2),
                           ("original_inlier_filtering", True), ("min_training_points", 99),
                           ("min_training_cells", 11), ("case_count", 64.0), ("models", ["translation"])):
            changed = copy.deepcopy(manifest)
            changed["design"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)

    def test_bad_source_identity_is_rejected_before_reading_case_content(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(self.analysis, "sha", return_value="f"*64):
            with self.assertRaisesRegex(ValueError, "identity"):
                self.analysis.input_hashes()

    def test_exact_output_scope_and_exclusive_preservation(self):
        with tempfile.TemporaryDirectory(prefix="aot-model-test-") as directory:
            output = Path(directory).resolve()
            with patch.object(self.analysis, "OUTPUT_DIR", output):
                self.assertEqual(self.analysis.scope_output(output), output)
                with self.assertRaisesRegex(ValueError, "scope"):
                    self.analysis.scope_output(output.parent)
                for name in ("result.json", "failure.json"):
                    with patch.object(Path, "exists", return_value=False), \
                            patch.object(Path, "is_symlink", lambda p: p.name == name):
                        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                            self.analysis.scope_output(output)
                path = output / "result.json"
                self.analysis.write_exclusive(path, dict(generated=True))
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                    self.analysis.scope_output(output)
                with self.assertRaises(FileExistsError):
                    self.analysis.write_exclusive(path, dict(replacement=True))
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
