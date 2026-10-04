"""Generated-only checks of V52's frozen snapshot formulas and missingness."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import accuracy_v52_benchmark as benchmark


def case(family="stable", background="planar", noise=0):
    spec = next(spec for spec in benchmark.specifications()
                if spec["family"] == family and spec["background"] == background
                and spec["noise_level"] == noise)
    return benchmark.generate_case(spec)


class BenchmarkContractTest(unittest.TestCase):
    def test_frozen_factorial_count_and_unique_ids(self):
        specs = benchmark.specifications()
        self.assertEqual(120, len(specs))
        self.assertEqual(120, len({spec["case_id"] for spec in specs}))
        self.assertEqual(20, len(benchmark.FAMILIES))
        self.assertEqual(3, len(benchmark.BACKGROUNDS))
        for family in benchmark.FAMILIES:
            self.assertEqual(6, sum(spec["family"] == family for spec in specs))

    def test_specifications_and_metadata_are_fresh_json_safe(self):
        specs = benchmark.specifications()
        specs[0]["family"] = "bad"
        self.assertEqual("stable", benchmark.specifications()[0]["family"])
        meta = benchmark.metadata()
        self.assertEqual(120, meta["case_count"])
        self.assertEqual(144, meta["point_count"])
        self.assertEqual(set(benchmark.FAMILIES), set(meta["family_formulas"]))
        self.assertFalse(meta["truth_and_labels_allowed_as_fit_inputs"])
        self.assertFalse(meta["physical_sensor_model"])
        self.assertFalse(meta["clipped"])
        self.assertFalse(meta["quantized"])
        self.assertEqual(meta, json.loads(json.dumps(meta, allow_nan=False)))
        meta["points_xy"][0][0] = 999
        self.assertNotEqual(999, benchmark.metadata()["points_xy"][0][0])

    def test_exact_guard_ring_and_order(self):
        points = case()["points_xy"]
        expected = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                    if 40 <= max(abs(x - 64), abs(y - 64)) <= 56]
        np.testing.assert_array_equal(points, expected)
        self.assertEqual((144, 2), points.shape)
        self.assertEqual(144, len(set(map(tuple, points))))
        self.assertTrue(np.issubdtype(points.dtype, np.integer))

    def test_all_outputs_have_expected_shapes_and_types(self):
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            with self.subTest(case_id=spec["case_id"]):
                self.assertEqual((8, 144), data["history"].shape)
                for name in ("median8", "median3", "current", "clean_current_background",
                             "injected_contamination", "contamination_mask"):
                    self.assertEqual((144,), data[name].shape)
                self.assertEqual(np.bool_, data["contamination_mask"].dtype)
                for name in ("history", "median8", "median3", "current",
                             "clean_current_background", "injected_contamination"):
                    self.assertEqual(np.float64, data[name].dtype)
                self.assertTrue(np.isfinite(data["clean_current_background"]).all())

    def test_all_arrays_are_readonly(self):
        data = case()
        for name, value in data.items():
            if isinstance(value, np.ndarray):
                with self.subTest(field=name):
                    self.assertFalse(value.flags.writeable)
                    with self.assertRaises(ValueError):
                        value.flat[0] = 0

    def test_spec_return_is_detached(self):
        spec = benchmark.specifications()[0]
        data = benchmark.generate_case(spec)
        data["spec"]["family"] = "changed"
        self.assertEqual("stable", spec["family"])
        self.assertEqual("stable", benchmark.generate_case(spec)["spec"]["family"])

    def test_reproducibility_all_cases(self):
        for spec in benchmark.specifications():
            first, second = benchmark.generate_case(spec), benchmark.generate_case(spec)
            for name in first:
                if isinstance(first[name], np.ndarray):
                    np.testing.assert_array_equal(first[name], second[name])
                else:
                    self.assertEqual(first[name], second[name])

    def test_rejects_unknown_or_malformed_specifications(self):
        invalid = [None, [], {}, {"case_id": "unknown"}]
        for field, value in (("family", "other"), ("noise_level", 99), ("seed", 12),
                             ("history_length", 7), ("point_count", 145), ("extra", 1),
                             ("analytic_simulation_only", False)):
            changed = deepcopy(benchmark.specifications()[0])
            changed[field] = value
            invalid.append(changed)
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                benchmark.generate_case(spec)

    def test_canonical_json_roundtrip_and_key_order(self):
        spec = benchmark.specifications()[99]
        reordered = dict(reversed(list(spec.items())))
        data = benchmark.generate_case(json.loads(json.dumps(reordered)))
        self.assertEqual(spec, data["spec"])


class AnalyticFormulaTest(unittest.TestCase):
    def test_background_formulas(self):
        points = case()["points_xy"]
        dx, dy = points[:, 0] - 64, points[:, 1] - 64
        planar = 96 + 0.05 * dx + 0.03 * dy
        expected = {
            "constant": np.full(144, 96.),
            "planar": planar,
            "textured": planar + 12 * np.sin(dx / 12) + 9 * np.cos(dy / 15) + 6 * np.sin((dx + dy) / 17),
        }
        for background, values in expected.items():
            data = case(background=background)
            np.testing.assert_array_equal(data["clean_current_background"], values)
            np.testing.assert_array_equal(data["current"], values)
            np.testing.assert_array_equal(data["history"], np.tile(values, (8, 1)))

    def test_all_broad_current_transforms(self):
        for background in benchmark.BACKGROUNDS:
            data = case(background=background)
            base = data["current"]
            dx, dy = data["points_xy"].T - 64
            expected = {
                "offset_plus8": base + 8, "offset_minus8": base - 8,
                "gain125": base * 1.25, "gain075": base * .75,
                "gain_zero": np.full(144, 96.), "gain_negative": 192 - base,
                "plane": base + (4 * dx.astype(float) / 56 - 3 * dy.astype(float) / 56),
                "gain_plane": 1.15 * base + 8 + 4 * dx / 56 - 3 * dy / 56,
            }
            for family, current in expected.items():
                with self.subTest(background=background, family=family):
                    actual = case(family, background)
                    np.testing.assert_array_equal(actual["history"], data["history"])
                    np.testing.assert_array_equal(actual["current"], current)
                    np.testing.assert_array_equal(actual["clean_current_background"], current)
                    self.assertFalse(actual["contamination_mask"].any())

    def test_recent_step_and_ended_pulse_identical_histories(self):
        for background in benchmark.BACKGROUNDS:
            for noise in (0, 1):
                step = case("recent_step", background, noise)
                pulse = case("pulse_ended", background, noise)
                np.testing.assert_array_equal(step["history"], pulse["history"])
                np.testing.assert_array_equal(step["median8"], pulse["median8"])
                np.testing.assert_array_equal(step["median3"], pulse["median3"])
                np.testing.assert_allclose(step["current"] - pulse["current"], 8, atol=3e-14, rtol=0)

    def test_history_and_current_for_temporal_cases(self):
        for background in benchmark.BACKGROUNDS:
            base = case(background=background)["current"]
            for family in ("recent_step", "pulse_ended", "long_pulse_ended"):
                data = case(family, background)
                expected_history = np.tile(base, (8, 1))
                expected_history[-2:] += 8
                if family == "long_pulse_ended":
                    expected_history[:6] += 8
                np.testing.assert_array_equal(data["history"], expected_history)
                np.testing.assert_array_equal(data["current"], base + 8 if family == "recent_step" else base)
                np.testing.assert_array_equal(data["median8"], base + 8 if family == "long_pulse_ended" else base)
                np.testing.assert_array_equal(data["median3"], base + 8)

    def test_localized_contamination_formula_and_truth(self):
        data = case("localized_change")
        x, y = data["points_xy"].T
        mask = (x >= 64) & (y >= 64)
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        np.testing.assert_array_equal(data["contamination_mask"], mask)
        np.testing.assert_array_equal(data["injected_contamination"], 16 * mask)
        np.testing.assert_array_equal(data["current"], data["clean_current_background"] + 16 * mask)
        np.testing.assert_array_equal(data["clean_current_background"], case()["current"])

    def test_sparse_contamination_formula_and_truth(self):
        mask = np.arange(144) % 17 == 0
        self.assertEqual(9, mask.sum())
        for family, amplitude in (("sparse_bright", 40), ("sparse_dark", -40)):
            data = case(family)
            np.testing.assert_array_equal(data["contamination_mask"], mask)
            np.testing.assert_array_equal(data["injected_contamination"], amplitude * mask)
            np.testing.assert_array_equal(data["current"], data["clean_current_background"] + amplitude * mask)
            np.testing.assert_array_equal(data["clean_current_background"], case()["current"])

    def test_stripe_contamination_formula_and_truth(self):
        data = case("stripe")
        mask = np.abs(data["points_xy"][:, 0] - 64) <= 8
        np.testing.assert_array_equal(data["contamination_mask"], mask)
        np.testing.assert_array_equal(data["injected_contamination"], 32 * mask)
        np.testing.assert_array_equal(data["current"], data["clean_current_background"] + 32 * mask)
        np.testing.assert_array_equal(data["clean_current_background"], case()["current"])

    def test_only_four_families_mark_contamination(self):
        contaminated = {"localized_change", "sparse_bright", "sparse_dark", "stripe"}
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            self.assertEqual(spec["family"] in contaminated, bool(data["contamination_mask"].any()))
            np.testing.assert_array_equal(data["contamination_mask"], data["injected_contamination"] != 0)


class NoiseAndMissingnessTest(unittest.TestCase):
    def test_complete_noise_realization_shared_across_every_case(self):
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            noiseless = case(spec["family"], spec["background"], 0)
            noise = np.random.default_rng(spec["seed"]).uniform(-spec["noise_level"], spec["noise_level"], (9, 144))
            # Arithmetic addition to distinct baselines can round differently;
            # compare complete observed values, rather than subtracting noise.
            expected_history = noiseless["history"] + noise[:8]
            expected_current = noiseless["current"] + noise[-1]
            np.testing.assert_array_equal(data["history"], expected_history)
            np.testing.assert_array_equal(data["current"], expected_current)
            np.testing.assert_array_equal(data["clean_current_background"], noiseless["clean_current_background"])

    def test_noise_is_post_transform_not_scaled_by_gain(self):
        for family in ("gain125", "gain075", "gain_negative", "gain_plane"):
            data = case(family, "textured", 1)
            noise = np.random.default_rng(991).uniform(-1, 1, (9, 144))
            np.testing.assert_array_equal(data["current"], data["clean_current_background"] + noise[-1])

    def test_median_values_match_strict_reductions_all_cases(self):
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            np.testing.assert_array_equal(data["median8"], np.median(data["history"], axis=0))
            np.testing.assert_array_equal(data["median3"], np.median(data["history"][-3:], axis=0))

    def test_missing_prior_exact_scope_and_median_availability(self):
        for noise in (0, 1):
            data = case("missing_prior", "textured", noise)
            missing = np.zeros((8, 144), dtype=bool)
            missing[0, :8] = True
            np.testing.assert_array_equal(np.isnan(data["history"]), missing)
            np.testing.assert_array_equal(np.isnan(data["median8"]), np.arange(144) < 8)
            self.assertTrue(np.isfinite(data["median3"]).all())
            self.assertTrue(np.isfinite(data["current"]).all())

    def test_missing_current_left_and_right_are_complementary(self):
        left, right = case("missing_current_left"), case("missing_current_right")
        x = left["points_xy"][:, 0]
        np.testing.assert_array_equal(np.isnan(left["current"]), x < 64)
        np.testing.assert_array_equal(np.isnan(right["current"]), x >= 64)
        np.testing.assert_array_equal(np.isnan(left["current"]), ~np.isnan(right["current"]))
        self.assertTrue(np.isfinite(left["history"]).all())
        self.assertTrue(np.isfinite(right["history"]).all())
        self.assertTrue(np.isfinite(left["clean_current_background"]).all())
        self.assertTrue(np.isfinite(right["clean_current_background"]).all())

    def test_all_missing_current_preserves_history_and_truth(self):
        data = case("all_current_missing", "textured", 1)
        baseline = case("stable", "textured", 1)
        self.assertTrue(np.isnan(data["current"]).all())
        np.testing.assert_array_equal(data["history"], baseline["history"])
        np.testing.assert_array_equal(data["clean_current_background"], baseline["clean_current_background"])

    def test_no_unrequested_missing_values(self):
        missing_families = {"missing_prior", "missing_current_left", "missing_current_right", "all_current_missing"}
        for spec in benchmark.specifications():
            data = benchmark.generate_case(spec)
            if spec["family"] not in missing_families:
                self.assertTrue(np.isfinite(data["history"]).all())
                self.assertTrue(np.isfinite(data["current"]).all())


if __name__ == "__main__":
    unittest.main()
