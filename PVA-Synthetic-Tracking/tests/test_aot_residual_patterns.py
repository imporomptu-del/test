"""Generated-only residual-analysis checks; no media or experiment outcomes."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_aot_residual_patterns.py"


def load_analysis():
    spec = importlib.util.spec_from_file_location("aot_residual_generated_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ramp_fixture():
    """A generated regular spatial field, not evidence about any image pair."""
    points = np.array([[30*x, 30*y] for y in range(6) for x in range(6)], dtype=np.float64)
    vectors = np.column_stack((points[:, 0] * .01, points[:, 1] * -.02))
    return points, vectors


class ResidualNeighborTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_five_required_neighbors_self_excluded_and_boundary_is_inclusive(self):
        points = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [200, 0]], float)
        graph = self.analysis.neighbor_graph(points)
        self.assertIn(0, graph["anchors"])
        row = list(graph["anchors"]).index(0)
        self.assertEqual(graph["neighbors"][row].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(graph["neighbors"].shape[1], 5)
        for anchor, neighbors in zip(graph["anchors"], graph["neighbors"]):
            self.assertNotIn(anchor, neighbors)
            self.assertTrue(np.all(np.linalg.norm(points[neighbors]-points[anchor], axis=1) <= 200))
        points[5, 0] = 200.0001
        self.assertNotIn(0, self.analysis.neighbor_graph(points)["anchors"])

    def test_tied_distances_use_stable_original_index_order(self):
        points = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]], float)
        graph = self.analysis.neighbor_graph(points)
        row = list(graph["anchors"]).index(0)
        self.assertEqual(graph["neighbors"][row].tolist(), [1, 2, 3, 4, 5])

    def test_sparse_and_isolated_points_are_reported_not_given_fewer_neighbors(self):
        for points in (np.zeros((0, 2)), np.arange(10).reshape(5, 2),
                       np.array([[i*1000, 0] for i in range(8)])):
            graph = self.analysis.neighbor_graph(points)
            with self.subTest(count=len(points)):
                self.assertEqual(graph["anchor_count"], 0)
                self.assertEqual(graph["directed_edge_count"], 0)
                self.assertEqual(graph["point_count"], len(points))
                self.assertEqual(graph["neighbors"].shape, (0, 5))

    def test_each_eligible_anchor_is_counted_once(self):
        points, _ = ramp_fixture()
        graph = self.analysis.neighbor_graph(points)
        self.assertEqual(graph["anchor_count"], len(points))
        self.assertEqual(len(set(graph["anchors"].tolist())), len(points))
        self.assertEqual(graph["directed_edge_count"], 5*len(points))
        self.assertEqual(graph["anchor_fraction"], 1.0)


class ResidualCoherenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_uniform_vectors_have_zero_statistic_and_undefined_zero_null_ratio(self):
        points, vectors = ramp_fixture()
        vectors[:] = [3.0, -2.0]
        result = self.analysis.spatial_coherence(points, vectors, seed=20260928)
        self.assertEqual(result["observed"], 0.0)
        self.assertEqual(result["reference_quantiles"], dict(p5=0.0, p50=0.0, p95=0.0))
        self.assertIsNone(result["observed_to_reference_median"])

    def test_fixed_seed_reproduces_joint_vector_permutations_on_fixed_graph(self):
        points, vectors = ramp_fixture()
        original_points, original_vectors = points.copy(), vectors.copy()
        result = self.analysis.spatial_coherence(points, vectors, seed=20260928)
        graph = self.analysis.neighbor_graph(points)
        anchors, neighbors = graph["anchors"], graph["neighbors"]
        def statistic(values):
            return float(np.median(np.linalg.norm(values[anchors] - np.median(values[neighbors], axis=1), axis=1)))
        rng = np.random.default_rng(20260928)
        references = [statistic(vectors[rng.permutation(len(vectors))]) for _ in range(199)]
        self.assertAlmostEqual(result["observed"], statistic(vectors))
        for name, value in zip(("p5", "p50", "p95"), np.percentile(references, [5, 50, 95])):
            self.assertAlmostEqual(result["reference_quantiles"][name], value)
        self.assertEqual(result, self.analysis.spatial_coherence(points, vectors, seed=20260928))
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(vectors, original_vectors)

    def test_generated_spatial_ramp_is_more_coherent_than_its_permuted_reference(self):
        points, vectors = ramp_fixture()
        result = self.analysis.spatial_coherence(points, vectors, seed=20260928)
        self.assertLess(result["observed"], result["reference_quantiles"]["p5"])

    def test_subtracting_a_saved_translation_does_not_create_new_neighbor_evidence(self):
        points, vectors = ramp_fixture()
        left = self.analysis.spatial_coherence(points, vectors, seed=20260928)
        right = self.analysis.spatial_coherence(points, vectors-[11.0, -7.0], seed=20260928)
        self.assertAlmostEqual(left["observed"], right["observed"])
        self.assertAlmostEqual(left["observed_to_reference_median"], right["observed_to_reference_median"])
        for key in ("p5", "p50", "p95"):
            self.assertAlmostEqual(left["reference_quantiles"][key], right["reference_quantiles"][key])

    def test_no_eligible_anchors_is_unavailable_not_zero_coherence(self):
        points = np.array([[0, 0], [1000, 1000]], float)
        result = self.analysis.spatial_coherence(points, np.zeros((2, 2)), seed=20260928)
        self.assertIsNone(result["observed"])
        self.assertIsNone(result["reference_quantiles"])
        self.assertIsNone(result["observed_to_reference_median"])
        self.assertEqual(result["reference_values"], [])


class ResidualSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_norm_quantiles_are_native_euclidean_norms_not_component_quantiles(self):
        vectors = np.array([[3, 4], [-3, 4], [0, 0], [5, 12]], dtype=np.float64)
        result = self.analysis.norm_summary(vectors)
        self.assertEqual(result["count"], 4)
        expected = np.percentile([5, 5, 0, 13], [10, 50, 90, 95, 99, 100])
        for key, value in zip(("p10", "p50", "p90", "p95", "p99", "max"), expected):
            self.assertAlmostEqual(result["norm_quantiles"][key], value)

    def test_score_cuts_are_preflow_linear_quantiles_with_ties_retained(self):
        scores = np.array([0, 0, 0, 0, 4, 4, 10, 4294967295], dtype=np.uint32)
        cuts = self.analysis.score_cuts(scores)
        np.testing.assert_array_equal(cuts, np.quantile(scores, [.25, .5, .75], method="linear"))
        self.assertEqual(self.analysis.score_cuts(np.full(8, 1234, np.uint32)), [1234, 1234, 1234])

    def test_empty_summaries_and_under_supported_cells_are_unavailable(self):
        self.assertEqual(self.analysis.norm_summary(np.empty((0, 2))),
                         dict(count=0, component_median=None, norm_quantiles=None))
        points = np.array([[10+i, 10] for i in range(5)] + [[2447.5, 2047.5]], float)
        values = np.array([[3, -2]]*5 + [[99, 99]], float)
        grid = self.analysis.cell_medians(points, values)
        self.assertEqual(len(grid["cells"]), 48)
        self.assertEqual(grid["extent_xy"], [0, 2448, 0, 2048])
        self.assertEqual(grid["cells"][0]["median_vector"], [3, -2])
        self.assertEqual(grid["cells"][0]["median_scatter_px"], 0)
        self.assertEqual(grid["cells"][-1]["count"], 1)
        self.assertIsNone(grid["cells"][-1]["median_vector"])
        self.assertIsNone(grid["cells"][1]["median_vector"])

    def test_native_grid_edges_are_half_open(self):
        points = np.array([[0, 0], [305.999, 0], [306, 0], [2447.999, 2047.999]], float)
        self.assertEqual(self.analysis.cell_indices(points).tolist(), [0, 0, 1, 47])
        for point in ([-.01, 0], [2448, 0], [0, 2048]):
            with self.assertRaises(ValueError):
                self.analysis.cell_indices([point])


class ResidualScoreStrataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_ties_stay_together_and_empty_bins_remain_explicit(self):
        scores = np.full(8, 1234, np.uint32)
        result = self.analysis.score_strata(scores, np.ones(8, bool), self.analysis.score_cuts(scores))
        self.assertEqual([b["selected_count"] for b in result["bins"]], [0, 0, 0, 8])
        for row in result["bins"][:3]:
            self.assertIsNone(row["acceptance_fraction"])
            self.assertIsNone(row["loss_fraction"])
        self.assertEqual(result["bins"][3]["acceptance_fraction"], 1)

    def test_acceptance_loss_cannot_shrink_score_bin_denominators_or_move_cuts(self):
        scores = np.arange(8, dtype=np.uint32)
        cuts = self.analysis.score_cuts(scores)
        full = self.analysis.score_strata(scores, np.ones(8, bool), cuts)
        lost = self.analysis.score_strata(scores, np.array([True, False]*4), cuts)
        self.assertEqual(full["cuts"], lost["cuts"])
        self.assertEqual([b["selected_count"] for b in full["bins"]], [2]*4)
        self.assertEqual([b["selected_count"] for b in lost["bins"]], [2]*4)
        self.assertEqual([b["lost_count"] for b in lost["bins"]], [1]*4)
        self.assertEqual([b["acceptance_fraction"] for b in lost["bins"]], [.5]*4)

    def test_truth_cohort_is_separate_from_all_selected_primary_counts(self):
        scores = np.full(8, 10, np.uint32)
        accepted = np.array([True, False]*4)
        cohort = np.array([True]*6 + [False]*2)
        result = self.analysis.score_strata(scores, accepted, [10]*3,
            accepted_values=np.array([[0, 0], [3, 4], [6, 8], [9, 12]], float),
            inlier_mask=np.array([False, True, False, True]), cohort_mask=cohort,
            truth_errors=np.array([0, .2, .7, 0]),
            points=np.array([[10+i*306, 10] for i in range(8)], float))
        row = result["bins"][3]
        self.assertEqual((row["selected_count"], row["accepted_count"], row["lost_count"]), (8, 4, 4))
        self.assertEqual((row["saved_inlier_count"], row["saved_outlier_count"]), (2, 2))
        self.assertEqual(row["saved_inlier_value_summary"]["norm_quantiles"]["p50"], 10)
        self.assertEqual(row["selected_grid_coverage"]["occupied_cells"], 8)
        self.assertEqual(row["accepted_grid_coverage"]["occupied_cells"], 4)
        truth = row["fixed_support_truth"]
        self.assertEqual((truth["selected_count"], truth["accepted_count"], truth["lost_count"]), (6, 3, 3))
        self.assertEqual(truth["accepted_within_0_1_count"], 1)
        self.assertEqual(truth["accepted_within_0_5_count"], 2)
        self.assertEqual(truth["within_0_5_fraction_of_selected"], 2/6)
        actual = self.analysis.score_strata(scores, accepted, [10]*3)
        self.assertTrue(all("fixed_support_truth" not in b for b in actual["bins"]))

    def test_score_and_mask_types_fail_closed(self):
        for scores, accepted in ((np.arange(8, dtype=float), np.ones(8, bool)),
                                 (np.arange(8, dtype=np.uint32), np.ones(8, int)),
                                 (np.arange(8, dtype=np.uint32), np.ones(7, bool))):
            with self.assertRaises(ValueError):
                self.analysis.score_strata(scores, accepted, [1, 3, 5])


class ResidualScopeManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def manifest_fixture(self):
        hashes = {k: v[1] for k, v in self.analysis.INPUT_PINS.items()}
        hashes.update(script_sha256="a"*64, tests_sha256="b"*64, plan_sha256="c"*64)
        return dict(schema=self.analysis.PLAN_SCHEMA, design=copy.deepcopy(self.analysis.DESIGN), **hashes), hashes

    def test_fixed_design_and_every_input_artifact_hash_are_bound(self):
        manifest, hashes = self.manifest_fixture()
        self.analysis.validate_manifest(manifest, hashes)
        for key in hashes:
            changed = copy.deepcopy(manifest)
            changed[key] = "f"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)
        for key, value in (("neighbor_k", 4), ("neighbor_radius_px", 201), ("case_count", 64.0),
                           ("new_fit", 0), ("permutation_references", 200), ("seed", 20260929)):
            changed = copy.deepcopy(manifest)
            changed["design"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analysis.validate_manifest(changed, hashes)

    def test_wrong_input_bytes_are_rejected_without_reading_capture_contents(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "is_symlink", return_value=False), \
                patch.object(self.analysis, "sha", return_value="f"*64):
            with self.assertRaisesRegex(ValueError, "input identity"):
                self.analysis.input_hashes()

    def test_output_scope_is_exact_and_existing_success_or_failure_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="aot-residual-test-") as directory:
            output = Path(directory).resolve()
            with patch.object(self.analysis, "OUTPUT_DIR", output):
                self.assertEqual(self.analysis.scope_output(output), output)
                with self.assertRaisesRegex(ValueError, "scope"):
                    self.analysis.scope_output(output.parent)
                for name in ("result.json", "failure.json"):
                    with patch.object(Path, "exists", lambda p: p.name == name), \
                            patch.object(Path, "is_symlink", return_value=False):
                        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                            self.analysis.scope_output(output)
                    with patch.object(Path, "exists", return_value=False), \
                            patch.object(Path, "is_symlink", lambda p: p.name == name):
                        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                            self.analysis.scope_output(output)

    def test_exclusive_writer_cannot_replace_an_existing_result(self):
        with tempfile.TemporaryDirectory(prefix="aot-residual-test-") as directory:
            path = Path(directory) / "result.json"
            self.analysis.write_exclusive(path, dict(generated=True, count=3))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.analysis.write_exclusive(path, dict(replacement=True))
            self.assertEqual(path.read_bytes(), before)


def array_record(array):
    return dict(dtype=str(array.dtype), shape=list(array.shape), values=array.tolist(),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def unpack_fixture():
    raw_points = np.array([[100+30*i, 200] for i in range(8)], np.float32)
    raw_scores = np.array([101, 4000000001, 99, 1, 17, 22, 45, 63], np.uint32)
    indices = np.array([6, 1, 4, 0, 7, 3], np.int64)
    selected = raw_points[indices]
    native = selected * np.float32(2) + np.float32(.5)
    accepted_indices = [1, 3, 5]
    pairs = [dict(previous_xy=native[index].tolist(), current_xy=(native[index]+[dx, 0]).tolist())
             for index, dx in zip(accepted_indices, [1, 5, 3])]
    return dict(case_id="aot_001_adjacent", previous_index=0, current_index=1,
        arm=dict(id="half_gain16_complete"), completed=True, kind="aot_adjacent",
        native_pixel_sha256=dict(previous="a"*64),
        capture=dict(proxy=dict(previous=dict(shape=[1024, 1224], sha256="b"*64)),
            s16=dict(previous=dict(sha256="c"*64)),
            harris=dict(coordinates=array_record(raw_points), scores=array_record(raw_scores)),
            selection=dict(selected_count=6, selected_indices=array_record(indices),
                           coordinates=array_record(selected))),
        correspondence=dict(accepted_count=3, correspondences=pairs),
        global_fit=dict(model="translation", quality_status="rejected", rejection_reasons=["generated_rejection"],
            parameters=dict(translation_x_px=1.0, translation_y_px=0.0), inlier_indices=[1],
            previous_to_current_matrix=[[1, 0, 1], [0, 1, 0], [0, 0, 1]]))


class ResidualSavedSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_raw_score_selection_and_saved_inliers_index_the_correct_lists(self):
        data = self.analysis.unpack_case(unpack_fixture())
        self.assertEqual(data["selected_scores"].dtype, np.uint32)
        self.assertEqual(data["selected_scores"].tolist(), [45, 4000000001, 17, 101, 63, 1])
        self.assertEqual(data["accepted_indices"].tolist(), [1, 3, 5])
        self.assertEqual(data["accepted_mask"].tolist(), [False, True, False, True, False, True])
        self.assertEqual(data["inlier_mask"].tolist(), [False, True, False])
        self.assertIsNone(data["known_truth"])
        self.assertIsNone(data["cohort_mask"])

    def test_rejected_but_finite_candidate_is_preserved_not_refitted_or_promoted(self):
        row = unpack_fixture()
        data = self.analysis.unpack_case(row)
        result = self.analysis.analyze_case(row, data, self.analysis.score_cuts(data["selected_scores"]), 7)
        self.assertTrue(result["residual_available"])
        self.assertEqual(result["saved_candidate_translation"], [1, 0])
        self.assertEqual(result["original_fit"]["quality_status"], "rejected")
        self.assertEqual(result["points"]["candidate_residual_xy"], [[0, 0], [4, 0], [2, 0]])
        self.assertEqual(result["saved_inlier_residual_summary"]["norm_quantiles"]["p50"], 4)
        self.assertEqual(result["saved_inlier_residual_summary"]["count"], 1)
        self.assertEqual(result["coherence"]["seed"], 20260935)
        self.assertEqual(result["accepted_displacement_fraction_above_sqrt20"], 1/3)

    def test_null_saved_translation_stays_unavailable_without_substitute(self):
        row = unpack_fixture()
        row["global_fit"].update(parameters=None, previous_to_current_matrix=None, inlier_indices=[])
        data = self.analysis.unpack_case(row)
        result = self.analysis.analyze_case(row, data, self.analysis.score_cuts(data["selected_scores"]), 0)
        self.assertIsNone(data["translation"])
        self.assertFalse(result["residual_available"])
        for key in ("candidate_residual_summary", "saved_inlier_residual_summary", "grid", "coherence"):
            self.assertIsNone(result[key])
        self.assertEqual(result["displacement_summary"]["count"], 3)
        self.assertEqual(result["lost_count"], 3)

    def test_invalid_saved_inlier_or_accepted_order_fails_closed(self):
        for change in ("selected_index_not_accepted_index", "boolean_index", "duplicate_inlier",
                       "reordered_pairs", "unknown_previous", "nonfinite_current", "wrong_matrix"):
            row = unpack_fixture()
            if change == "selected_index_not_accepted_index": row["global_fit"]["inlier_indices"] = [5]
            elif change == "boolean_index": row["global_fit"]["inlier_indices"] = [True]
            elif change == "duplicate_inlier": row["global_fit"]["inlier_indices"] = [1, 1]
            elif change == "reordered_pairs": row["correspondence"]["correspondences"].reverse()
            elif change == "unknown_previous": row["correspondence"]["correspondences"][0]["previous_xy"] = [10, 20]
            elif change == "nonfinite_current": row["correspondence"]["correspondences"][0]["current_xy"] = [np.nan, 20]
            else: row["global_fit"]["previous_to_current_matrix"][0][2] = 5
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.analysis.unpack_case(row)

    def test_synthetic_truth_uses_preflow_support_not_observed_current_crop(self):
        row = unpack_fixture()
        row.update(kind="aot_known_shift", expected_shift_xy=[1, 0])
        row["capture"]["preflow_support"] = dict(margin_px=128, expected_shift_xy=[1, 0],
            selected_count=6, fixed_support_count=6, mask=[True]*6, selected_indices=list(range(6)))
        row["correspondence"]["correspondences"][0]["current_xy"] = [0, 0]
        data = self.analysis.unpack_case(row)
        self.assertEqual(data["cohort_mask"].tolist(), [True]*6)
        self.assertEqual(data["known_truth"].tolist(), [1, 0])
        row["capture"]["preflow_support"]["fixed_support_count"] = 5
        with self.assertRaisesRegex(ValueError, "pre-flow"):
            self.analysis.unpack_case(row)

    def test_corrupt_raw_scores_cannot_silently_change_quartiles(self):
        row = unpack_fixture()
        row["capture"]["harris"]["scores"]["values"][0] += 1
        with self.assertRaisesRegex(ValueError, "hash"):
            self.analysis.unpack_case(row)


class ResidualInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis()

    def test_only_ordered_actual8_then_natural56_are_selected(self):
        indices = (0, 42, 85, 127, 170, 212, 255, 298)
        shifts = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))
        actual = [dict(case_id=f"aot_{i+1:03d}_adjacent", previous_index=i,
            kind="aot_adjacent", arm=dict(id="half_gain16_complete")) for i in indices]
        known = [dict(case_id=f"aot_prev{i:03d}_dx{x:+d}_dy{y:+d}", previous_index=i,
            expected_shift_xy=[x, y], kind="aot_known_shift", arm=dict(id="half_gain16_complete"))
            for i in indices for x, y in shifts]
        factor = dict(schema="seaqr.aot.feature-factors.v1", passed=True,
            rows=[dict(kind="aot_adjacent", arm=dict(id="other"))] + actual)
        natural = dict(schema="seaqr.aot.natural-shifts.v1", passed=True,
            rows=known + [dict(kind="generated_control", arm=dict(id="half_gain16_complete"))])
        selected = self.analysis.select_rows(factor, natural)
        self.assertEqual(selected, actual + known)
        self.assertEqual(len(selected), 64)
        self.assertEqual([r["case_id"] for r in selected], self.analysis.case_ids())
        for mode in ("missing", "reorder", "wrong_shift", "failed_receipt"):
            left, right = copy.deepcopy(factor), copy.deepcopy(natural)
            if mode == "missing": right["rows"].pop(0)
            elif mode == "reorder": left["rows"][1], left["rows"][2] = left["rows"][2], left["rows"][1]
            elif mode == "wrong_shift": right["rows"][0]["expected_shift_xy"] = [2, 0]
            else: left["passed"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.analysis.select_rows(left, right)


if __name__ == "__main__":
    unittest.main()
