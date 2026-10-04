"""Generated-only tests; no source media, PVA, saved endpoints or global fits."""
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/validate_motion_patch_controls.py"
SPEC = importlib.util.spec_from_file_location("motion_patch_controls_under_test", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def reference_patch(image, point):
    """Separate scalar-row interpolation, not the vectorized implementation."""
    xs = point[0] + np.arange(-15, 16, dtype=float)
    ys = point[1] + np.arange(-15, 16, dtype=float)
    columns = np.arange(image.shape[1], dtype=float)
    rows = []
    for y in ys:
        low = int(np.floor(y))
        high = min(low+1, image.shape[0]-1)
        first = np.interp(xs, columns, image[low].astype(float))
        second = np.interp(xs, columns, image[high].astype(float))
        rows.append(first + (y-low)*(second-first))
    return np.array(rows)


class PatchControlsTests(unittest.TestCase):
    def test_fixed_contract_and_twenty_case_inventory(self):
        inventory = M.control_inventory()
        self.assertEqual(inventory["shape_hw"], [512, 640])
        cases = inventory["cases"]
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c["case_id"] for c in cases}), 20)
        self.assertEqual([c["case_id"] for c in cases[-4:]],
                         ["flat", "straight_edge", "periodic_ambiguity", "out_of_range_shift4"])
        self.assertTrue(all(c["point_queries_xy"] == [[320.25, 256.5]] for c in cases))
        self.assertEqual(M.CONTRACT["search_candidates"], 625)
        self.assertEqual(M.CONTRACT["minimum_runner_gap"], .02)
        self.assertEqual(M.CONTRACT["minimum_negative_ncc_hessian_eigenvalue_per_px2"], .001)
        self.assertEqual(M.CONTRACT["maximum_negative_ncc_hessian_condition"], 100)
        self.assertEqual([c["truth_displacement_xy"] for c in cases[:4]],
                         [[0., 0.], [.5, -.75], [2., -1.], [-1.25, .5]])
        inventory["contract"]["minimum_ncc"] = -1
        self.assertEqual(M.CONTRACT["minimum_ncc"], .8)
        json.dumps(M.control_inventory(), allow_nan=False)

    def test_bad_generated_dimensions(self):
        for shape in ((True, 640), (512., 640), (50, 640), (512, 9000), (512,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                M.control_inventory(shape)

    def test_scalar_bilinear_reference_nonsquare_and_border(self):
        rng = np.random.default_rng(20260930)
        image = rng.integers(0, 256, (83, 117), dtype=np.uint8)
        centers = np.array([[40.125, 40.625], [15., 15.], [101., 67.]])
        result = M.bilinear_patches(image, centers)
        for p, patch in zip(centers, result):
            np.testing.assert_allclose(patch, reference_patch(image, p), rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            M.bilinear_patches(image, [[14.999, 40.]])
        with self.assertRaises(ValueError):
            M.bilinear_patches(image, np.repeat([[40., 40.]], 65, axis=0))

    def test_full_score_surface_against_independent_spatial_calculation(self):
        rng = np.random.default_rng(460)
        previous = rng.integers(10, 230, (93, 119), dtype=np.uint8)
        current = rng.integers(10, 230, previous.shape, dtype=np.uint8)
        p = np.array([56.3, 43.7])
        result = M.score_surface(previous, current, p)
        template = reference_patch(previous, p)
        template -= template.mean()
        reference = np.zeros((25, 25))
        deviations = np.zeros((25, 25))
        for row, dy in enumerate(np.arange(-12, 13)/4):
            for col, dx in enumerate(np.arange(-12, 13)/4):
                patch = reference_patch(current, p+[dx, dy])
                deviations[row, col] = patch.std()
                patch -= patch.mean()
                reference[row, col] = np.sum(patch*template)/np.sqrt(np.sum(patch**2)*np.sum(template**2))
        np.testing.assert_allclose(result["scores"], reference, rtol=0, atol=2e-14)
        np.testing.assert_allclose(result["current_std"], deviations, rtol=0, atol=3e-13)

    def test_candidate_batches_are_bounded_and_surface_invariant(self):
        pair = next(M.generated_controls((96, 120)))
        args = pair["previous_gray"], pair["current_gray"], pair["case"]["point_queries_xy"][0]
        original = M.bilinear_patches
        sizes = []
        def checked(image, centers):
            sizes.append(len(centers))
            return original(image, centers)
        with mock.patch.object(M, "bilinear_patches", side_effect=checked):
            large = M.score_surface(*args)
        self.assertLessEqual(max(sizes), 64)
        self.assertEqual(large["candidate_batches"], 10)
        small = M.score_surface(*args, batch_size=1)
        np.testing.assert_array_equal(large["scores"], small["scores"])
        for batch in (0, 65, True, 1.0):
            with self.assertRaises(ValueError):
                M.score_surface(*args, batch_size=batch)

    def test_support_failure_is_null_no_partial_search_or_padding(self):
        image = np.full((96, 100), 120, np.uint8)
        for p in ([17.99, 45.], [50., 78.01], [14., 50.]):
            record = M.measure_patch(image, image, p)
            self.assertFalse(record["qualified"])
            self.assertIsNone(record["displacement_xy"])
            self.assertIsNone(record["raw_best_displacement_xy"])
            self.assertEqual(record["cost"]["candidate_count"], 0)
        self.assertTrue(M.search_geometry([18., 18.], image.shape)["support_available"])

    def test_flat_never_confident_zero(self):
        image = np.full((96, 100), 120, np.uint8)
        record = M.measure_patch(image, image, [50., 48.], return_surface=True)
        self.assertFalse(record["qualified"])
        self.assertIsNone(record["displacement_xy"])
        self.assertIsNone(record["raw_best_displacement_xy"])
        self.assertEqual(record["finite_candidates"], 0)
        self.assertTrue(all(x is None for row in record["score_surface_dy_dx"] for x in row))
        json.dumps(record, allow_nan=False)

    def test_qualification_hessian_and_runner_exclusion(self):
        y, x = np.mgrid[-12:13, -12:13]/4
        scores = 1 - .1*(x*x+y*y)
        record = M.peak_diagnostics(scores, np.full((25, 25), 10.), 10.)
        self.assertTrue(record["qualified"])
        self.assertEqual(record["displacement_xy"], [0., 0.])
        self.assertAlmostEqual(record["runner_gap"], .1)
        np.testing.assert_allclose(record["hessian_eigenvalues_ascending"], [.2, .2], atol=1e-12)
        self.assertAlmostEqual(record["hessian_condition"], 1)
        # Chebyshev .75 belongs to excluded central basin, distance1 does not.
        scores[12, 15] = .9999
        record = M.peak_diagnostics(scores, np.full((25, 25), 10.), 10.)
        self.assertAlmostEqual(record["runner_gap"], .1)
        scores[12, 16] = .999
        self.assertIn("runner_gap", M.peak_diagnostics(scores, np.full((25, 25), 10.), 10.)["abstention_reasons"])

    def test_boundary_and_row_major_tie_not_qualified(self):
        y, x = np.mgrid[-12:13, -12:13]/4
        scores = 1-.1*((x-3)**2+y*y)
        record = M.peak_diagnostics(scores, np.full((25,25), 10.), 10.)
        self.assertFalse(record["qualified"])
        self.assertTrue(record["best_at_search_boundary"])
        self.assertEqual(record["raw_best_displacement_xy"], [3., 0.])
        record = M.peak_diagnostics(np.ones((25,25)), np.full((25,25), 10.), 10.)
        self.assertEqual(record["raw_best_displacement_xy"], [-3., -3.])
        self.assertIsNone(record["displacement_xy"])

    def test_flat_curvature_and_ill_conditioned_hessian_abstain(self):
        y, x = np.mgrid[-12:13, -12:13]/4
        for scores, reason in ((1-.0001*(x*x+y*y), "hessian_minimum"),
                               (1-.001*x*x-.2*y*y, "hessian_condition")):
            record = M.peak_diagnostics(scores, np.full((25,25), 10.), 10.)
            self.assertFalse(record["qualified"])
            self.assertIsNone(record["displacement_xy"])
            self.assertIn(reason, record["abstention_reasons"])

    def test_nonfinite_and_malformed_input_rejected(self):
        image = np.zeros((96,100), np.uint8)
        with self.assertRaises(ValueError):
            M.measure_patch(image, image, [np.nan, 40])
        with self.assertRaises(ValueError):
            M.measure_patch(image.astype(float), image, [50, 40])
        with self.assertRaises(ValueError):
            M.measure_patch(image, image[:80], [50, 40])
        for std in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                M.peak_diagnostics(np.ones((25,25)), np.ones((25,25)), std)
        with self.assertRaises(ValueError):
            M.peak_diagnostics(np.full((25,25), np.inf), np.ones((25,25)), 10.)

    def test_all_fixed_textured_positive_truths_and_photometry(self):
        for pair in M.generated_controls((96,120)):
            if not pair["case"]["case_id"].startswith("texture__"):
                continue
            case = pair["case"]
            with self.subTest(case=case["case_id"]):
                result = M.measure_patch(pair["previous_gray"], pair["current_gray"], case["point_queries_xy"][0])
                self.assertTrue(result["qualified"], result["abstention_reasons"])
                self.assertTrue(M.assess_control(case, result)["passed"])
                photo = result["photometry_at_raw_peak"]
                self.assertFalse(photo["feeds_search_or_qualification"])
                self.assertLess(photo["affine_adjusted_rmse_dn"], 2.)

    def test_generation_deterministic_uint8_no_clipping_and_independent_truth(self):
        first = list(M.generated_controls((96,120)))
        second = list(M.generated_controls((96,120)))
        for a, b in zip(first, second):
            self.assertEqual(a["case"], b["case"])
            self.assertEqual(a["previous_pixel_sha256"], b["previous_pixel_sha256"])
            self.assertEqual(a["current_pixel_sha256"], b["current_pixel_sha256"])
            self.assertEqual(a["quantization"]["clipped_pixels"], 0)
            self.assertLessEqual(a["quantization"]["current_maximum_abs_rounding_error_dn"], .5)
            self.assertEqual(a["previous_gray"].dtype, np.uint8)
            self.assertEqual(a["previous_gray"].shape, (96,120))
            self.assertFalse(a["previous_gray"].flags.writeable)
            self.assertFalse(a["current_gray"].flags.writeable)
        np.testing.assert_array_equal(first[0]["previous_gray"], first[1]["previous_gray"])
        self.assertNotEqual(first[1]["current_pixel_sha256"], first[0]["current_pixel_sha256"])

    def test_all_four_degenerate_or_out_of_range_controls_abstain(self):
        for pair in M.generated_controls((96,120)):
            case = pair["case"]
            if case["expectation"] != "abstain":
                continue
            with self.subTest(case=case["case_id"]):
                result = M.measure_patch(pair["previous_gray"], pair["current_gray"], case["point_queries_xy"][0])
                self.assertFalse(result["qualified"])
                self.assertIsNone(result["displacement_xy"])
                self.assertTrue(M.assess_control(case, result)["passed"])

    def test_no_saved_endpoint_input_and_no_source_mutation(self):
        self.assertEqual(list(inspect.signature(M.measure_patch).parameters),
                         ["previous", "current", "previous_xy", "return_surface"])
        pair = next(M.generated_controls((96,120)))
        p, q = pair["previous_gray"], pair["current_gray"]
        initial = (M.array_sha(p), M.array_sha(q))
        M.measure_patch(p, q, pair["case"]["point_queries_xy"][0])
        self.assertEqual(initial, (M.array_sha(p), M.array_sha(q)))

    def test_bad_quality_record_retains_scientific_failure_not_zero_error(self):
        case = M.control_inventory()["cases"][0]
        record = dict(qualified=False, raw_best_displacement_xy=[0.,0.], displacement_xy=None)
        assessed = M.assess_control(case, record)
        self.assertFalse(assessed["passed"])
        self.assertIsNone(assessed["qualified_vector_error_px"])
        self.assertEqual(assessed["raw_argmax_error_px_not_a_valid_measurement"], 0.)

    def test_fresh_result_retains_all_twenty_and_rejects_overwrite_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"generated.json"
            result = M.run_generated(path, (96,120))
            saved = json.loads(path.read_text())
            self.assertEqual(len(saved["rows"]), 20)
            self.assertEqual(saved, result)
            self.assertTrue(saved["generated_only"])
            self.assertFalse(saved["source_media_accessed"])
            self.assertEqual(saved["passed"], all(q["assessment"]["passed"] for row in saved["rows"] for q in row["queries"]))
            self.assertTrue(saved["scientific_failure_is_retained_without_retuning"])
            with self.assertRaises(ValueError):
                M.run_generated(path, (96,120))
            linked = Path(directory).resolve()/"linked.json"
            linked.symlink_to(Path(directory).resolve()/"missing.json")
            with self.assertRaises(ValueError):
                M.run_generated(linked, (96,120))


if __name__ == "__main__":
    unittest.main()
