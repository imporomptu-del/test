"""Generated-only checks of the frozen V54 paired source-preservation design."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import accuracy_v54_benchmark as benchmark


def case(condition="stable", motion="appearing", amplitude=4, background="constant", noise=0, missingness=None):
    spec = next(spec for spec in benchmark.specifications()
        if spec["condition"] == condition and spec["motion"] == motion and spec["amplitude"] == amplitude
        and spec["background"] == background and spec["noise_level"] == noise and spec["missingness"] == missingness)
    return benchmark.generate_case(spec)


def full(data, prefix):
    return np.vstack((data[prefix + "_history"], data[prefix + "_current"]))


def core_full(data, side):
    return np.vstack((data["core_history_" + side], data["core_current_" + side]))


def profile(points, center):
    return np.array([max(1 - abs(x - center[0]) / 2, 0) * max(1 - abs(y - center[1]) / 2, 0)
                     for x, y in points], dtype=float)


class FrozenDesignTests(unittest.TestCase):
    def test_exact_case_counts_unique_ids_and_factorial(self):
        specs = benchmark.specifications()
        self.assertEqual(420, len(specs))
        self.assertEqual(420, len({spec["case_id"] for spec in specs}))
        self.assertEqual(416, sum(spec["stratum"] == "factorial" for spec in specs))
        self.assertEqual(4, sum(spec["stratum"] == "availability" for spec in specs))
        self.assertEqual(13, len(benchmark.SOURCE_CONFIGURATIONS))
        self.assertEqual(52, sum(spec["condition"] == "stable" and spec["stratum"] == "factorial" for spec in specs))
        for condition in benchmark.CONDITIONS:
            rows = [spec for spec in specs if spec["condition"] == condition and spec["stratum"] == "factorial"]
            self.assertEqual(52, len(rows))
            self.assertEqual(52, len({(spec["motion"], spec["amplitude"], spec["background"], spec["noise_level"]) for spec in rows}))

    def test_declared_case_order_and_availability_defaults(self):
        specs = benchmark.specifications()
        expected = [(condition, motion, amplitude, background, level, seed)
            for condition in benchmark.CONDITIONS for motion, amplitude in benchmark.SOURCE_CONFIGURATIONS
            for background in benchmark.BACKGROUNDS for level, seed in benchmark.NOISE_PROFILES]
        self.assertEqual(expected, [(spec["condition"], spec["motion"], spec["amplitude"], spec["background"], spec["noise_level"], spec["seed"]) for spec in specs[:416]])
        for missingness, spec in zip(benchmark.AVAILABILITY_CASES, specs[416:]):
            self.assertEqual("availability_" + missingness, spec["case_id"])
            self.assertEqual(("stable", "appearing", 4, "textured", 0, 71),
                (spec["condition"], spec["motion"], spec["amplitude"], spec["background"], spec["noise_level"], spec["seed"]))

    def test_specifications_and_metadata_fresh_json_safe(self):
        specs = benchmark.specifications()
        self.assertEqual(specs, json.loads(json.dumps(specs, allow_nan=False)))
        specs[0]["condition"] = "changed"
        self.assertEqual("stable", benchmark.specifications()[0]["condition"])
        meta = benchmark.metadata()
        self.assertEqual(meta, json.loads(json.dumps(meta, allow_nan=False)))
        self.assertEqual(420, meta["case_count"])
        self.assertFalse(meta["labels_and_truth_allowed_as_fit_inputs"])
        self.assertFalse(meta["physical_sensor_model"])
        self.assertFalse(meta["quantized"])
        self.assertFalse(meta["clipped"])
        meta["guard_xy"][0][0] = 999
        self.assertNotEqual(999, benchmark.metadata()["guard_xy"][0][0])

    def test_reject_modified_unknown_or_noncanonical_cases(self):
        invalid = [None, [], {}, {"case_id": "unknown"}]
        for field, value in (("amplitude", 40), ("noise_level", 1), ("condition", "tuned"),
                             ("history_length", 7), ("extra", 1), ("physical_sensor_model", True)):
            changed = deepcopy(benchmark.specifications()[0])
            changed[field] = value
            invalid.append(changed)
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                benchmark.generate_case(spec)

    def test_output_spec_is_detached(self):
        spec = benchmark.specifications()[0]
        data = benchmark.generate_case(spec)
        data["spec"]["condition"] = "changed"
        self.assertEqual("stable", spec["condition"])
        self.assertEqual("stable", benchmark.generate_case(spec)["spec"]["condition"])


class GeometryAndObservationTests(unittest.TestCase):
    def test_exact_disjoint_y_major_grids(self):
        data = case()
        core = [(x, y) for y in range(52, 77) for x in range(52, 77)]
        guard = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                 if 40 <= max(abs(x - 64), abs(y - 64)) <= 56]
        np.testing.assert_array_equal(data["core_xy"], core)
        np.testing.assert_array_equal(data["guard_xy"], guard)
        self.assertEqual((625, 2), data["core_xy"].shape)
        self.assertEqual((144, 2), data["guard_xy"].shape)
        self.assertFalse(set(core) & set(guard))
        self.assertEqual((64, 64), tuple(data["core_xy"][312]))

    def test_array_shapes_types_and_readonly_detachment(self):
        data = case()
        expected = dict(guard_xy=(144, 2), core_xy=(625, 2), guard_history=(8, 144), guard_current=(144,),
            core_history_on=(8, 625), core_history_off=(8, 625), core_current_on=(625,), core_current_off=(625,),
            clean_current_background=(625,), current_source=(625,), source_template=(625,), guard_contamination_current=(144,))
        self.assertEqual(set(expected) | {"spec"}, set(data))
        for name, shape in expected.items():
            with self.subTest(name=name):
                self.assertEqual(shape, data[name].shape)
                self.assertEqual(np.int64 if name.endswith("_xy") else np.float64, data[name].dtype)
                self.assertFalse(data[name].flags.writeable)
                self.assertIsNone(data[name].base)
                with self.assertRaises(ValueError):
                    data[name].flat[0] = 0
        self.assertFalse(np.shares_memory(data["core_history_on"], data["core_history_off"]))

    def test_background_formula_applied_separately_to_both_grids(self):
        for background in benchmark.BACKGROUNDS:
            data = case(motion="absent", amplitude=0, background=background)
            for prefix in ("core", "guard"):
                points = data[prefix + "_xy"]
                dx, dy = points[:, 0] - 64., points[:, 1] - 64.
                expected = np.full(len(points), 96.) if background == "constant" else (
                    96 + .05 * dx + .03 * dy + 12 * np.sin(dx / 12) + 9 * np.cos(dy / 15) + 6 * np.sin((dx + dy) / 17))
                actual = core_full(data, "off") if prefix == "core" else full(data, "guard")
                np.testing.assert_array_equal(actual, np.tile(expected, (9, 1)))
            np.testing.assert_array_equal(data["clean_current_background"], data["core_current_off"])

    def test_recent_and_ended_global_shift_schedules(self):
        schedules = {
            "stable": [0] * 9,
            "recent_plus8": [0] * 6 + [8, 8, 8],
            "recent_minus8": [0] * 6 + [-8, -8, -8],
            "ended_short_plus8": [0] * 6 + [8, 8, 0],
            "ended_long_plus8": [8] * 8 + [0],
        }
        for condition, shift in schedules.items():
            data = case(condition=condition, motion="absent", amplitude=0)
            for array in (full(data, "guard"), core_full(data, "off"), core_full(data, "on")):
                np.testing.assert_array_equal(array, np.tile(96 + np.asarray(shift)[:, None], (1, array.shape[1])))
            np.testing.assert_array_equal(data["clean_current_background"], np.full(625, 96 + shift[-1]))

    def test_local_guard_contamination_never_changes_core_or_guard_history(self):
        baseline = case(background="textured", noise=.5)
        for condition, amplitude in (("local_guard_plus16", 16), ("local_guard_minus16", -16)):
            data = case(condition, background="textured", noise=.5)
            x, y = data["guard_xy"].T
            mask = (x >= 64) & (y >= 64)
            np.testing.assert_array_equal(data["guard_contamination_current"], amplitude * mask)
            np.testing.assert_array_equal(data["guard_current"], baseline["guard_current"] + amplitude * mask)
            for name in ("guard_history", "core_history_on", "core_history_off", "core_current_on", "core_current_off", "clean_current_background"):
                np.testing.assert_array_equal(data[name], baseline[name])

    def test_all_guard_contamination_never_changes_core(self):
        baseline, data = case(), case("all_guard_plus8")
        np.testing.assert_array_equal(data["guard_contamination_current"], np.full(144, 8.))
        np.testing.assert_array_equal(data["guard_current"], baseline["guard_current"] + 8)
        np.testing.assert_array_equal(data["guard_history"], baseline["guard_history"])
        for name in ("core_history_on", "core_history_off", "core_current_on", "core_current_off", "clean_current_background"):
            np.testing.assert_array_equal(data[name], baseline[name])

    def test_rng_guard_then_core_draw_order_all_profiles(self):
        for level, seed in benchmark.NOISE_PROFILES:
            data = case(motion="absent", amplitude=0, noise=level)
            rng = np.random.default_rng(seed)
            expected_guard = 96 + rng.uniform(-level, level, (9, 144))
            expected_core = 96 + rng.uniform(-level, level, (9, 625))
            np.testing.assert_array_equal(full(data, "guard"), expected_guard)
            np.testing.assert_array_equal(core_full(data, "off"), expected_core)
            np.testing.assert_array_equal(core_full(data, "on"), expected_core)

    def test_shared_noise_across_conditions_backgrounds_and_sources(self):
        for spec in benchmark.specifications()[:416]:
            data = benchmark.generate_case(spec)
            level, seed = spec["noise_level"], spec["seed"]
            rng = np.random.default_rng(seed)
            guard_noise = rng.uniform(-level, level, (9, 144))
            core_noise = rng.uniform(-level, level, (9, 625))
            noiseless = case(spec["condition"], spec["motion"], spec["amplitude"], spec["background"], 0)
            # Source addition comes after noise, so source-on arithmetic order
            # can differ at machine precision from adding noise to no-noise on.
            np.testing.assert_allclose(full(data, "guard"), full(noiseless, "guard") + guard_noise, rtol=0, atol=3e-14)
            np.testing.assert_array_equal(core_full(data, "off"), core_full(noiseless, "off") + core_noise)
            np.testing.assert_allclose(core_full(data, "on"), core_full(noiseless, "on") + core_noise, rtol=0, atol=3e-14)

    def test_all_cases_reproducible(self):
        for spec in benchmark.specifications():
            first, second = benchmark.generate_case(spec), benchmark.generate_case(spec)
            for name in first:
                if name == "spec":
                    self.assertEqual(first[name], second[name])
                else:
                    np.testing.assert_array_equal(first[name], second[name])


class SourceMotionTests(unittest.TestCase):
    def test_fixed_template_peak_support_and_squared_norm(self):
        data = case()
        template = data["source_template"]
        self.assertEqual(1., template[312])
        self.assertEqual(9, np.count_nonzero(template))
        self.assertEqual(4., template.sum())
        self.assertEqual(2.25, np.square(template).sum())
        np.testing.assert_array_equal(template, profile(data["core_xy"], (64, 64)))

    def test_all_motion_trajectories_in_every_prior_and_current(self):
        def center(motion, t):
            if motion == "appearing": return (64, 64) if t == 0 else None
            if motion == "stationary": return 64, 64
            if motion == "slow_linear": return 64 + t / 4, 64
            if motion == "linear": return 64 + t, 64
            if motion == "turning": return (68 + t, 60) if t <= -4 else (64, 64 + t)
            return 64 + min(t + 3, 0), 64
        for motion in benchmark.MOTIONS:
            for amplitude in (4, -4):
                data = case(motion=motion, amplitude=amplitude)
                source = core_full(data, "on") - core_full(data, "off")
                for index, time in enumerate(range(-8, 1)):
                    location = center(motion, time)
                    expected = np.zeros(625) if location is None else amplitude * profile(data["core_xy"], location)
                    np.testing.assert_array_equal(source[index], expected)
                np.testing.assert_array_equal(data["current_source"], amplitude * data["source_template"])

    def test_source_profile_confined_to_core_never_guard(self):
        for motion in benchmark.MOTIONS:
            data = case(motion=motion)
            for time in benchmark.TIMES:
                center = benchmark._center(motion, time)
                if center is not None:
                    self.assertTrue(54 <= center[0] <= 74 and 54 <= center[1] <= 74)
                    self.assertFalse(np.any(profile(data["guard_xy"], center)))
            baseline = case(motion="absent", amplitude=0)
            np.testing.assert_array_equal(data["guard_history"], baseline["guard_history"])
            np.testing.assert_array_equal(data["guard_current"], baseline["guard_current"])

    def test_absent_keeps_unit_template_but_identical_observed_pair(self):
        for background in benchmark.BACKGROUNDS:
            for noise in (0, .5):
                data = case(motion="absent", amplitude=0, background=background, noise=noise)
                np.testing.assert_array_equal(data["core_history_on"], data["core_history_off"])
                np.testing.assert_array_equal(data["core_current_on"], data["core_current_off"])
                np.testing.assert_array_equal(data["current_source"], np.zeros(625))
                self.assertEqual(1., data["source_template"].max())

    def test_pairing_cancels_noise_and_preserves_source_addition_order(self):
        for motion in benchmark.MOTIONS:
            positive = case(motion=motion, amplitude=4, background="textured", noise=.5)
            negative = case(motion=motion, amplitude=-4, background="textured", noise=.5)
            np.testing.assert_array_equal(positive["core_history_off"], negative["core_history_off"])
            np.testing.assert_array_equal(positive["core_current_off"], negative["core_current_off"])
            np.testing.assert_array_equal(positive["core_current_on"], positive["core_current_off"] + positive["current_source"])
            np.testing.assert_array_equal(negative["core_current_on"], negative["core_current_off"] + negative["current_source"])

    def test_turn_corner_and_stop_times_are_explicit(self):
        self.assertEqual((64, 60), benchmark._center("turning", -4))
        self.assertEqual((64, 61), benchmark._center("turning", -3))
        self.assertEqual((63, 64), benchmark._center("move_stop", -4))
        self.assertEqual((64, 64), benchmark._center("move_stop", -3))
        self.assertEqual((64, 64), benchmark._center("move_stop", -1))


class AvailabilityTests(unittest.TestCase):
    def test_guard_missing_cases_exact_scope(self):
        for name in ("guard_current_left_missing", "guard_current_all_missing"):
            data = case(background="textured", missingness=name)
            expected = data["guard_xy"][:, 0] < 64 if name.endswith("left_missing") else np.ones(144, dtype=bool)
            np.testing.assert_array_equal(np.isnan(data["guard_current"]), expected)
            self.assertTrue(np.isfinite(data["guard_history"]).all())
            self.assertTrue(np.isfinite(data["core_history_on"]).all())
            self.assertTrue(np.isfinite(data["core_current_on"]).all())

    def test_core_prior_missing_same_center_in_both_pairs(self):
        data = case(background="textured", missingness="core_first_prior_center_missing")
        expected = np.zeros((8, 625), dtype=bool); expected[0, 312] = True
        for side in ("on", "off"):
            np.testing.assert_array_equal(np.isnan(data["core_history_" + side]), expected)
            self.assertTrue(np.isfinite(data["core_current_" + side]).all())
            self.assertTrue(np.isnan(np.median(data["core_history_" + side], axis=0)[312]))
            self.assertTrue(np.isfinite(np.median(data["core_history_" + side][-3:], axis=0)).all())

    def test_core_current_missing_same_center_in_both_pairs(self):
        data = case(background="textured", missingness="core_current_center_missing")
        expected = np.arange(625) == 312
        for side in ("on", "off"):
            np.testing.assert_array_equal(np.isnan(data["core_current_" + side]), expected)
            self.assertTrue(np.isfinite(data["core_history_" + side]).all())

    def test_truth_remains_finite_and_source_known_under_missingness(self):
        for missingness in benchmark.AVAILABILITY_CASES:
            data = case(background="textured", missingness=missingness)
            for field in ("clean_current_background", "current_source", "source_template", "guard_contamination_current"):
                self.assertTrue(np.isfinite(data[field]).all())
            self.assertEqual(4., data["current_source"][312])

    def test_no_missing_values_in_any_factorial_case(self):
        for spec in benchmark.specifications()[:416]:
            data = benchmark.generate_case(spec)
            for name, value in data.items():
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.isfinite(value).all(), (spec["case_id"], name))


if __name__ == "__main__":
    unittest.main()
