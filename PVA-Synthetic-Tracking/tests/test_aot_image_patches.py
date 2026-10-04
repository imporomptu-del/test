"""Generated-only patch checks; no source imagery or experiment result reads."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_aot_image_patches.py"


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("aot_patches_generated_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generated_texture(size=320):
    return np.random.default_rng(20260928).integers(16, 240, size=(size, size), dtype=np.uint8)


def translated_copy(source, dx, dy, fill=128):
    """Independent integer copy fixture, with no roll, wrap, or resampling."""
    result = np.full_like(source, fill)
    height, width = source.shape
    x0, x1 = max(0, -dx), min(width, width-dx)
    y0, y1 = max(0, -dy), min(height, height-dy)
    if x0 < x1 and y0 < y1:
        result[y0+dy:y1+dy, x0+dx:x1+dx] = source[y0:y1, x0:x1]
    return result


class PatchAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_half_up_rounding_is_not_bankers_rounding(self):
        for point, expected in (([100.49, 200.49], [100, 200]),
                                 ([100.5, 200.5], [101, 201]),
                                 ([101.5, 201.5], [102, 202]),
                                 ([100.0, 200.0], [100, 200])):
            np.testing.assert_array_equal(self.diagnostic.half_up(point), expected)

    def test_generated_copy_has_correct_sign_without_wrap_or_source_changes(self):
        previous = np.arange(120, dtype=np.uint8).reshape(10, 12)
        before = previous.copy()
        for dx, dy in ((0, 0), (1, -1), (-1, 1), (4, -2), (-4, 2)):
            current = translated_copy(previous, dx, dy)
            np.testing.assert_array_equal(self.diagnostic.translate_no_wrap(previous, [dx, dy]), current)
            for y in range(10):
                for x in range(12):
                    expected = previous[y-dy, x-dx] if 0 <= x-dx < 12 and 0 <= y-dy < 10 else 128
                    self.assertEqual(current[y, x], expected)
            np.testing.assert_array_equal(previous, before)


class PatchGeneratedMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_both_template_sizes_recover_signed_integer_translations(self):
        previous = generated_texture()
        original = previous.copy()
        for size in (33, 65):
            for shift in ((0, 0), (1, -1), (-1, 1), (4, -2), (-4, 2)):
                with self.subTest(size=size, shift=shift):
                    current = translated_copy(previous, *shift)
                    result = self.diagnostic.zncc_search(previous, current, [160.5, 160.5], size)
                    self.assertEqual(result["best_offset_xy"], list(shift))
                    self.assertAlmostEqual(result["best_ncc"], 1.0, places=8)
                    self.assertTrue(result["qualified"])
                    self.assertFalse(result["best_at_search_boundary"])
                    self.assertFalse(result["geometry"]["search_clipped"])
        np.testing.assert_array_equal(previous, original)

    def test_integer_anchor_offset_is_not_mistaken_for_original_subpixel_motion(self):
        previous = generated_texture()
        current = translated_copy(previous, 4, -2)
        for p in ([160.0, 160.0], [160.49, 160.49], [160.5, 160.5], [160.75, 160.75]):
            result = self.diagnostic.zncc_search(previous, current, p, 33)
            self.assertEqual(result["best_offset_xy"], [4, -2])
            np.testing.assert_allclose(result["estimated_current_xy"], np.asarray(p)+[4, -2], atol=0)
            self.assertTrue(result["qualified"])

    def test_flat_template_never_qualifies_as_zero_motion(self):
        flat = np.full((320, 320), 128, np.uint8)
        for size in (33, 65):
            result = self.diagnostic.zncc_search(flat, flat, [160.5, 160.5], size)
            self.assertFalse(result["qualified"])
            self.assertEqual(result["template_std"], 0)

    def test_repeated_texture_has_no_unique_peak(self):
        y, x = np.indices((320, 320))
        previous = (40+160*((x+y) % 2)).astype(np.uint8)
        current = translated_copy(previous, 4, -2)
        result = self.diagnostic.zncc_search(previous, current, [160.5, 160.5], 33)
        self.assertFalse(result["qualified"])
        self.assertAlmostEqual(result["gap"], 0, places=8)
        self.assertAlmostEqual(result["runner_up_ncc"], 1, places=8)

    def test_exact_search_boundary_match_is_not_qualified(self):
        previous = generated_texture()
        for shift in ((64, 0), (-64, 0), (0, 64), (0, -64)):
            result = self.diagnostic.zncc_search(previous, translated_copy(previous, *shift), [160.5, 160.5], 33)
            with self.subTest(shift=shift):
                self.assertEqual(result["best_offset_xy"], list(shift))
                self.assertAlmostEqual(result["best_ncc"], 1, places=8)
                self.assertTrue(result["best_at_search_boundary"])
                self.assertFalse(result["qualified"])


class PatchSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_full_fft_surface_matches_independent_direct_float64_zncc(self):
        rng = np.random.default_rng(270928)
        template = rng.integers(10, 240, (5, 5), dtype=np.uint8)
        search = rng.integers(10, 240, (13, 17), dtype=np.uint8)
        scores, deviations = self.diagnostic.zncc_surface(template, search)
        centered = template.astype(float) - template.mean()
        direct = np.empty((9, 13))
        std = np.empty((9, 13))
        for y in range(9):
            for x in range(13):
                patch_values = search[y:y+5, x:x+5].astype(float)
                current_centered = patch_values-patch_values.mean()
                direct[y, x] = np.sum(centered*current_centered) / np.sqrt(np.sum(centered**2)*np.sum(current_centered**2))
                std[y, x] = np.std(patch_values)
        np.testing.assert_allclose(scores, direct, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(deviations, std, atol=1e-12, rtol=1e-12)

    def test_zncc_removes_offset_and_positive_gain_without_changing_pixels(self):
        template = np.random.default_rng(7).integers(10, 90, (9, 9), dtype=np.uint8)
        search = template*2+7
        before = search.copy()
        scores, std = self.diagnostic.zncc_surface(template, search)
        self.assertAlmostEqual(scores[0, 0], 1, places=12)
        self.assertAlmostEqual(std[0, 0], 2*np.std(template), places=12)
        np.testing.assert_array_equal(search, before)

    def test_missing_second_peak_is_unqualified_not_infinite_confidence(self):
        scores = np.zeros((3, 3))
        scores[1, 1] = .95
        result = self.diagnostic.peak_summary(scores, np.full((3, 3), 10), [-1, -1, 1, 1], 10)
        self.assertEqual(result["best_offset_xy"], [0, 0])
        self.assertIsNone(result["runner_up_ncc"])
        self.assertIsNone(result["gap"])
        self.assertFalse(result["qualified"])

    def test_runner_excludes_chebyshev_three_inclusive_and_keeps_four(self):
        scores = np.zeros((9, 9))
        scores[4, 4], scores[7, 7], scores[8, 4] = .95, .94, .8
        result = self.diagnostic.peak_summary(scores, np.full((9, 9), 10), [-4, -4, 4, 4], 10)
        self.assertEqual(result["best_offset_xy"], [0, 0])
        self.assertEqual(result["runner_up_ncc"], .8)
        self.assertAlmostEqual(result["gap"], .15)
        self.assertTrue(result["qualified"])

    def test_equal_maxima_use_lowest_dy_then_dx_without_resolving_ambiguity(self):
        scores = np.zeros((11, 11))
        scores[2, 3] = scores[2, 8] = scores[8, 3] = .9
        result = self.diagnostic.peak_summary(scores, np.full((11, 11), 10), [-5, -5, 5, 5], 10)
        self.assertEqual(result["best_offset_xy"], [-2, -3])
        self.assertEqual(result["gap"], 0)
        self.assertFalse(result["qualified"])

    def test_each_frozen_quality_gate_is_required_including_current_patch_std(self):
        scores = np.zeros((9, 9))
        scores[4, 4], scores[8, 4] = .8, .75
        good = self.diagnostic.peak_summary(scores, np.ones((9, 9)), [-4, -4, 4, 4], 1)
        self.assertTrue(good["qualified"])
        for mode in ("ncc", "gap", "template_std", "current_std", "boundary"):
            values, std, template_std = scores.copy(), np.ones((9, 9)), 1
            if mode == "ncc": values[4, 4] = np.nextafter(.8, 0)
            elif mode == "gap": values[8, 4] = .76
            elif mode == "template_std": template_std = .999
            elif mode == "current_std": std[4, 4] = .999
            else: values[0, 0] = .99
            result = self.diagnostic.peak_summary(values, std, [-4, -4, 4, 4], template_std)
            with self.subTest(mode=mode):
                self.assertFalse(result["qualified"])

    def test_nonfinite_patch_deviation_cannot_pass_a_quality_threshold(self):
        scores = np.zeros((9, 9))
        scores[4, 4] = .95
        for value in (np.inf, np.nan, -1):
            with self.subTest(value=value, field="template"), self.assertRaises(ValueError):
                self.diagnostic.peak_summary(scores, np.ones((9, 9)), [-4, -4, 4, 4], value)
            std = np.ones((9, 9))
            std[4, 4] = value
            with self.subTest(value=value, field="current"), self.assertRaises(ValueError):
                self.diagnostic.peak_summary(scores, std, [-4, -4, 4, 4], 1)


class PatchClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_clipped_search_retains_scores_but_cannot_qualify_or_pad(self):
        previous = generated_texture()
        for point, size in (([40.5, 160.5], 65), ([280.5, 160.5], 33), ([160.5, 40.5], 65)):
            result = self.diagnostic.zncc_search(previous, previous, point, size)
            with self.subTest(point=point, size=size):
                self.assertTrue(result["available"])
                self.assertTrue(result["geometry"]["search_clipped"])
                self.assertFalse(result["qualified"])
                self.assertEqual(result["best_offset_xy"], [0, 0])
                bounds = result["geometry"]["current_search_bounds"]
                self.assertTrue(all(0 <= coordinate <= 320 for coordinate in bounds))
                offsets = result["geometry"]["offset_bounds"]
                self.assertEqual(result["total_candidate_count"], (offsets[2]-offsets[0]+1)*(offsets[3]-offsets[1]+1))

    def test_outside_template_is_unavailable_without_reading_any_surface(self):
        previous = generated_texture()
        with patch.object(self.diagnostic, "zncc_surface", side_effect=AssertionError("must not pad invalid template")):
            result = self.diagnostic.zncc_search(previous, previous, [10.5, 100.5], 65)
        self.assertFalse(result["available"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["unavailable_reason"], "previous_template_outside_image")


class PatchTwoScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def records(self, offsets=([0, 0], [1, 1])):
        return [dict(template_size=size, available=True, qualified=True, best_offset_xy=list(offset))
                for size, offset in zip((33, 65), offsets)]

    def test_both_sizes_must_qualify_and_have_consistent_offsets(self):
        records = self.records()
        result = self.diagnostic.two_scale_verdict(records, [0, 0])
        self.assertTrue(result["two_sizes_qualified_and_consistent"])
        self.assertEqual(result["verdict"], "agrees_with_saved_lk")
        records[1]["qualified"] = False
        self.assertEqual(self.diagnostic.two_scale_verdict(records, [0, 0])["verdict"], "ambiguous")
        inconsistent = self.records(([0, 0], [1, 2]))
        result = self.diagnostic.two_scale_verdict(inconsistent, [0, 0])
        self.assertFalse(result["two_sizes_qualified_and_consistent"])
        self.assertEqual(result["verdict"], "ambiguous")

    def test_mixed_lk_agreement_cannot_be_called_definite_disagreement(self):
        records = self.records()
        mixed = self.diagnostic.two_scale_verdict(records, [-.4, -.4])
        self.assertTrue(mixed["two_sizes_qualified_and_consistent"])
        self.assertEqual(mixed["verdict"], "ambiguous")
        self.assertEqual(self.diagnostic.two_scale_verdict(records, [10, 0])["verdict"], "disagrees_with_saved_lk")

    def test_lk_agreement_threshold_is_inclusive_without_rounding(self):
        records = self.records(([0, 0], [0, 0]))
        self.assertEqual(self.diagnostic.two_scale_verdict(records, [1.5, 0])["verdict"], "agrees_with_saved_lk")
        self.assertEqual(self.diagnostic.two_scale_verdict(records, [np.nextafter(1.5, np.inf), 0])["verdict"],
                         "disagrees_with_saved_lk")

    def test_missing_scale_and_unavailable_peak_do_not_become_agreement(self):
        records = self.records()
        with self.assertRaises(ValueError):
            self.diagnostic.two_scale_verdict(records[:1], [0, 0])
        records[1].update(available=False, qualified=False, best_offset_xy=None)
        result = self.diagnostic.two_scale_verdict(records, [0, 0])
        self.assertEqual(result["verdict"], "ambiguous")
        self.assertIsNone(result["saved_lk_error_by_size_px"][1])
        self.assertIsNone(result["offset_difference_px"])


class PatchIndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_saved_lk_perturbation_changes_comparison_but_cannot_move_ncc_search(self):
        previous = generated_texture()
        current = translated_copy(previous, 4, -2)
        good = dict(id="generated", source_kind="actual_residual_extremum",
                    previous_xy=[160.5, 160.5], current_xy=[164.5, 158.5])
        bad = dict(good, current_xy=[200.5, 200.5])
        before_previous, before_current = previous.copy(), current.copy()
        agreed = self.diagnostic.evaluate_point(previous, current, good)
        disagreed = self.diagnostic.evaluate_point(previous, current, bad)
        self.assertEqual(agreed["scales"], disagreed["scales"])
        self.assertEqual(agreed["comparison"]["verdict"], "agrees_with_saved_lk")
        self.assertEqual(disagreed["comparison"]["verdict"], "disagrees_with_saved_lk")
        for result in disagreed["scales"]:
            self.assertEqual(result["best_offset_xy"], [4, -2])
            self.assertEqual(result["geometry"]["anchor_xy"], [161, 161])
            self.assertEqual(result["estimated_current_xy"], [164.5, 158.5])
        np.testing.assert_array_equal(previous, before_previous)
        np.testing.assert_array_equal(current, before_current)
        self.assertEqual(good["current_xy"], [164.5, 158.5])
        self.assertEqual(bad["current_xy"], [200.5, 200.5])

    def test_known_bad_lk_counterexample_truth_is_consulted_only_after_search(self):
        previous = generated_texture()
        current = translated_copy(previous, 48, 0)
        record = dict(id="generated_counterexample", source_kind="known_shift_counterexample",
            previous_xy=[160.5, 160.5], current_xy=[210.5, 160.5], synthetic_current=dict(shift_xy=[48, 0]))
        result = self.diagnostic.evaluate_point(previous, current, record)
        self.assertEqual(result["comparison"]["verdict"], "disagrees_with_saved_lk")
        self.assertEqual(result["counterexample_truth"]["saved_lk_error_px"], 2)
        self.assertEqual(result["counterexample_truth"]["independent_error_by_size_px"], [0, 0])
        changed_truth = copy.deepcopy(record)
        changed_truth["synthetic_current"]["shift_xy"] = [40, 0]
        changed = self.diagnostic.evaluate_point(previous, current, changed_truth)
        self.assertEqual(changed["scales"], result["scales"])
        self.assertEqual(changed["comparison"], result["comparison"])
        self.assertEqual(changed["counterexample_truth"]["independent_error_by_size_px"], [8, 8])


def residual_selection_fixture(equal=False):
    rows = []
    previous = np.array([[10.5, 10.5], [20.5, 20.5], [30.5, 30.5], [350.5, 20.5]])
    residual = np.array([[5, 0], [1, 0], [5, 0], [2, 0]], float)
    if equal:
        residual[:] = [2, 0]
    for index in (0, 42, 85, 127, 170, 212, 255, 298):
        rows.append(dict(case_id=f"aot_{index+1:03d}_adjacent", previous_index=index, current_index=index+1,
            source_kind="aot_adjacent", arm="half_gain16_complete", accepted_count=4,
            saved_candidate_translation=[10, 0],
            previous_identity=dict(native_previous="a"*64),
            points=dict(accepted_previous_xy=previous.tolist(), accepted_current_xy=(previous+residual+[10, 0]).tolist(),
                        candidate_residual_xy=residual.tolist(), accepted_selected_indices=[10, 20, 30, 40])))
    return dict(rows=rows)


class PatchSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_cell_extrema_ties_are_deterministic_and_singletons_are_deduplicated(self):
        source = residual_selection_fixture()
        before = copy.deepcopy(source)
        result = self.diagnostic.select_actual_points(source)
        self.assertEqual(len(result["cell_inventory"]), 384)
        self.assertEqual(len(result["points"]), 24)
        first = result["points"][:3]
        self.assertEqual([p["accepted_index"] for p in first], [1, 0, 3])
        self.assertEqual([p["selected_index"] for p in first], [20, 10, 40])
        self.assertEqual([p["selection_roles"] for p in first], [["minimum"], ["maximum"], ["minimum", "maximum"]])
        self.assertEqual(len({p["id"] for p in result["points"]}), 24)
        self.assertEqual(result, self.diagnostic.select_actual_points(source))
        self.assertEqual(source, before)

    def test_equal_extrema_do_not_force_a_different_second_representative(self):
        result = self.diagnostic.select_actual_points(residual_selection_fixture(equal=True))
        self.assertEqual(len(result["points"]), 16)
        self.assertEqual(result["cell_inventory"][0]["status"], "same_extremum")
        self.assertEqual(result["points"][0]["accepted_index"], 0)
        self.assertEqual(result["points"][0]["selection_roles"], ["minimum", "maximum"])

    def test_empty_cells_and_invalid_template_geometry_never_trigger_replacement(self):
        result = self.diagnostic.select_actual_points(residual_selection_fixture())
        self.assertEqual(result["cell_inventory"][2]["status"], "empty")
        self.assertEqual(result["cell_inventory"][2]["selected_point_ids"], [])
        self.assertEqual(sum(row["status"] == "empty" for row in result["cell_inventory"]), 8*46)
        chosen = result["points"][0]
        self.assertEqual(chosen["previous_xy"], [20.5, 20.5])
        self.assertFalse(chosen["planned_geometry"][1]["available"])
        self.assertEqual(chosen["planned_geometry"][1]["unavailable_reason"], "previous_template_outside_image")

    def test_missing_reordered_or_wrong_arm_actual_cases_fail_closed(self):
        for mode in ("missing", "reordered", "wrong_arm", "nonfinite"):
            source = residual_selection_fixture()
            if mode == "missing": source["rows"].pop()
            elif mode == "reordered": source["rows"].reverse()
            elif mode == "wrong_arm": source["rows"][0]["arm"] = "full_gain16_complete"
            else: source["rows"][0]["points"]["candidate_residual_xy"][0] = [np.nan, 0]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.diagnostic.select_actual_points(source)

    def test_two_per_cell_selection_respects_exact_768_point_upper_bound(self):
        source = residual_selection_fixture()
        points = np.array([[306*(column+.5)+offset, (2048/6)*(row+.5)]
                           for row in range(6) for column in range(8) for offset in (-5, 5)])
        residuals = np.array([[value, 0] for _ in range(48) for value in (0, 1)], float)
        for row in source["rows"]:
            row["accepted_count"] = 96
            row["points"] = dict(accepted_previous_xy=points.tolist(),
                accepted_current_xy=(points+residuals+[2, 0]).tolist(), candidate_residual_xy=residuals.tolist(),
                accepted_selected_indices=list(range(96)))
        result = self.diagnostic.select_actual_points(source)
        self.assertEqual(len(result["points"]), 768)
        self.assertEqual(len(result["cell_inventory"]), 384)
        self.assertTrue(all(record["status"] == "two_extrema" for record in result["cell_inventory"]))
        self.assertTrue(all(len(record["selected_point_ids"]) == 2 for record in result["cell_inventory"]))

    def test_visual_sample_is_preselected_by_roles_not_patch_outcomes(self):
        points = self.diagnostic.select_actual_points(residual_selection_fixture())["points"]
        chosen = self.diagnostic.visual_selection(points)
        self.assertEqual(len(chosen), 16)
        self.assertEqual(chosen[0]["point_id"], points[0]["id"])
        self.assertEqual(chosen[1]["point_id"], points[1]["id"])
        self.assertTrue(all(role["quadrant"] == 0 for record in chosen for role in record["visual_roles"]))
        for point in points:
            point["comparison"] = dict(verdict="disagrees_with_saved_lk")
            point["scales"] = [dict(best_ncc=0)]
        self.assertEqual(chosen, self.diagnostic.visual_selection(points))
        equal = self.diagnostic.select_actual_points(residual_selection_fixture(equal=True))["points"]
        deduplicated = self.diagnostic.visual_selection(equal)
        self.assertEqual(len(deduplicated), 8)
        self.assertEqual(len(deduplicated[0]["visual_roles"]), 2)


def metadata_fixture():
    return dict(images=[dict(img_name=f"generated_{index:03d}.png", png_sha256="b"*64,
        pixel_sha256="a"*64, bytes=1024, source_frame=index+10,
        timestamp_ns=str(10**18+index*100_000_000), entities=["must not enter patch selection"])
        for index in range(300)])


class PatchMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def test_metadata_whitelist_preserves_exact_timestamps_and_drops_annotations(self):
        metadata = self.diagnostic.image_metadata(metadata_fixture())
        self.assertEqual(len(metadata), 300)
        self.assertEqual(set(metadata[0]), {"frame_index", "img_name", "png_sha256", "pixel_sha256",
                                            "bytes", "source_frame", "timestamp_ns"})
        self.assertEqual(metadata[-1]["timestamp_ns"], str(10**18+299*100_000_000))
        self.assertEqual(metadata[-1]["frame_index"], 299)

    def test_synthetic_counterexample_does_not_bind_the_actual_next_frame(self):
        metadata = self.diagnostic.image_metadata(metadata_fixture())
        points = [dict(source_kind="actual_residual_extremum", previous_index=0, current_index=1,
                       expected_previous_pixel_sha256="a"*64),
                  dict(source_kind="known_shift_counterexample", previous_index=85, current_index=86,
                       expected_previous_pixel_sha256="a"*64)]
        inventory = self.diagnostic.bind_images(points, metadata)
        self.assertEqual([row["frame_index"] for row in inventory], [0, 1, 85])
        self.assertEqual(points[0]["current_image"]["frame_index"], 1)
        self.assertIsNone(points[1]["current_image"])
        self.assertEqual(points[1]["previous_image"]["frame_index"], 85)
        points[0]["expected_previous_pixel_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "pixel identity"):
            self.diagnostic.bind_images(points, metadata)

    def test_missing_inventory_unsafe_names_wrong_hashes_and_boolean_sizes_reject(self):
        for mode in ("missing", "path", "hash", "bool_size"):
            validation = metadata_fixture()
            if mode == "missing": validation["images"].pop()
            elif mode == "path": validation["images"][0]["img_name"] = "../generated.png"
            elif mode == "hash": validation["images"][0]["png_sha256"] = "not-a-hash"
            else: validation["images"][0]["bytes"] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.diagnostic.image_metadata(validation)


class PatchScopeFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_diagnostic()

    def manifest_fixture(self):
        hashes = {key: value[1] for key, value in self.diagnostic.INPUT_PINS.items()}
        hashes.update(script_sha256="a"*64, tests_sha256="b"*64, plan_sha256="c"*64)
        return dict(schema=self.diagnostic.PLAN_SCHEMA, design=copy.deepcopy(self.diagnostic.DESIGN), **hashes), hashes

    def test_all_inputs_artifacts_and_fixed_matcher_choices_are_hash_bound(self):
        manifest, hashes = self.manifest_fixture()
        self.diagnostic.validate_manifest(manifest, hashes)
        for key in hashes:
            changed = copy.deepcopy(manifest)
            changed[key] = "f"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.diagnostic.validate_manifest(changed, hashes)
        for key, value in (("patch_sizes", [33]), ("search_radius_px", 65), ("min_ncc", .7),
                           ("min_gap", .04), ("agreement_px", 2), ("clipped_search_unqualified", False),
                           ("maximum_actual_points", 769), ("maximum_actual_visual_points", 65)):
            changed = copy.deepcopy(manifest)
            changed["design"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.diagnostic.validate_manifest(changed, hashes)

    def test_wrong_input_hash_is_rejected_without_reading_image_content(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(self.diagnostic, "sha", return_value="f"*64):
            with self.assertRaisesRegex(ValueError, "frozen input"):
                self.diagnostic.input_hashes()

    def test_selection_freeze_binds_both_selection_and_manifest(self):
        freeze = dict(schema=self.diagnostic.FREEZE_SCHEMA, selection_sha256="a"*64, manifest_sha256="b"*64)
        self.diagnostic.validate_selection_freeze(freeze, "a"*64, "b"*64)
        for key in ("schema", "selection_sha256", "manifest_sha256"):
            changed = dict(freeze, **{key: "changed"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.diagnostic.validate_selection_freeze(changed, "a"*64, "b"*64)

    def test_wrong_selection_freeze_prevents_any_source_image_decode(self):
        output = Path("/generated/image_patches_01")
        freeze = dict(schema=self.diagnostic.FREEZE_SCHEMA, selection_sha256="wrong", manifest_sha256="a"*64)
        with patch.object(self.diagnostic, "scope_output", return_value=output), \
                patch.object(self.diagnostic, "bound_hashes", return_value=dict(manifest_sha256="a"*64)), \
                patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(self.diagnostic, "sha", return_value="b"*64), \
                patch.object(self.diagnostic, "json_read", return_value=freeze), \
                patch.object(self.diagnostic, "load_approved_image", side_effect=AssertionError("unfrozen image read")) as load:
            with self.assertRaisesRegex(ValueError, "freeze differs"):
                self.diagnostic.run(output)
            load.assert_not_called()

    def test_exact_scope_and_select_run_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="aot-patch-test-") as directory:
            output = Path(directory).resolve()
            with patch.object(self.diagnostic, "OUTPUT_DIR", output):
                for phase in ("select", "run"):
                    self.assertEqual(self.diagnostic.scope_output(output, phase), output)
                    with self.assertRaisesRegex(ValueError, "scope"):
                        self.diagnostic.scope_output(output.parent, phase)
                    names = ("result.json", "failure.json", "selection.json", "selection_freeze.json") if phase == "select" else ("result.json", "failure.json")
                    for name in names:
                        with patch.object(Path, "exists", lambda p: p.name == name), patch.object(Path, "is_symlink", return_value=False):
                            with self.subTest(phase=phase, name=name), self.assertRaisesRegex(ValueError, "overwrite"):
                                self.diagnostic.scope_output(output, phase)
                        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_symlink", lambda p: p.name == name):
                            with self.assertRaisesRegex(ValueError, "overwrite"):
                                self.diagnostic.scope_output(output, phase)
                target = output / "selection.json"
                self.diagnostic.write_exclusive(target, dict(generated=True))
                original = target.read_bytes()
                with self.assertRaises(FileExistsError):
                    self.diagnostic.write_exclusive(target, dict(replacement=True))
                self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
