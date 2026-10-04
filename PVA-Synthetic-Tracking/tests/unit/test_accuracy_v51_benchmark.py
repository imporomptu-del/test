from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import accuracy_v51_benchmark as benchmark


class V51BenchmarkTests(unittest.TestCase):
    def case(self, family, amplitude=8, level=0):
        spec = next(spec for spec in benchmark.specifications()
                    if spec["family"] == family and spec["amplitude"] == amplitude
                    and spec["noise_level"] == level)
        return benchmark.generate_case(spec)

    def test_exact_case_count_and_balanced_design(self):
        specs = benchmark.specifications()
        self.assertEqual(len(specs), 98)
        self.assertEqual(len({spec["case_id"] for spec in specs}), 98)
        self.assertEqual(sum(spec["family"] == "stable" for spec in specs), 2)
        for family in benchmark.FAMILIES:
            members = [spec for spec in specs if spec["family"] == family]
            self.assertEqual(len(members), 8)
            self.assertEqual({(s["amplitude"], s["noise_level"], s["seed"]) for s in members},
                             {(a, n, seed) for a in (-32, -8, 8, 32)
                              for n, seed in ((0, 71), (1, 991))})

    def test_points_are_exact_fixed_ring_in_y_then_x_order(self):
        points = tuple((x, y) for y in range(8, 121, 8)
                       for x in range(8, 121, 8)
                       if 40 <= max(abs(x - 64), abs(y - 64)) <= 56)
        self.assertEqual(benchmark.POINTS_XY, points)
        self.assertEqual(benchmark.POINT_COUNT, 144)
        self.assertEqual(len(set(points)), 144)

    def test_metadata_is_json_serializable_and_returns_fresh_objects(self):
        metadata = benchmark.benchmark_metadata()
        json.dumps(metadata, allow_nan=False)
        metadata["specifications"][0]["event_window"]["start_inclusive"] = 9
        self.assertIsNone(benchmark.specifications()[0]["event_window"]["start_inclusive"])
        self.assertEqual(benchmark.benchmark_metadata()["case_count"], 98)

    def test_all_cases_shape_dtype_and_readonly(self):
        for spec in benchmark.specifications():
            with self.subTest(case=spec["case_id"]):
                case = benchmark.generate_case(spec)
                self.assertEqual(case["values"].shape, (64, 144))
                self.assertEqual(case["values"].dtype, np.float64)
                self.assertEqual(case["signal_delta"].shape, (64, 144))
                self.assertEqual(case["points_xy"].shape, (144, 2))
                for value in case.values():
                    if isinstance(value, np.ndarray):
                        self.assertFalse(value.flags.writeable)
                self.assertEqual(case["spec"], spec)
                self.assertIsNot(case["spec"], spec)
                self.assertTrue(spec["analytic_simulation_only"])
                self.assertFalse(spec["physical_sensor_model"])

    def test_stable_base_and_no_events(self):
        case = self.case("stable", amplitude=0)
        points = case["points_xy"]
        expected = 96 + .05 * (points[:, 0] - 64) + .03 * (points[:, 1] - 64)
        np.testing.assert_array_equal(case["base"], expected)
        np.testing.assert_array_equal(case["values"], np.broadcast_to(expected, (64, 144)))
        self.assertFalse(case["event_active"].any())
        self.assertFalse(case["history_has_prior_event"].any())
        self.assertFalse(case["affected_points_mask"].any())
        self.assertTrue((case["response_phase"] == "baseline").all())

    def test_step_exact_onset_persistence_and_both_signs(self):
        for amplitude in benchmark.AMPLITUDES:
            case = self.case("step", amplitude)
            self.assertTrue((case["signal_delta"][:20] == 0).all())
            self.assertTrue((case["signal_delta"][20:] == amplitude).all())
            self.assertFalse(case["event_active"][:20].any())
            self.assertTrue(case["event_active"][20:].all())

    def test_ramp_takes_sixteen_intervals_then_plateaus(self):
        case = self.case("ramp", amplitude=32)
        for frame in range(64):
            expected = 0 if frame < 20 else 32 * min((frame - 20) / 16, 1)
            np.testing.assert_array_equal(case["signal_delta"][frame], np.full(144, expected))
        self.assertTrue(case["event_active"][20])
        self.assertEqual(case["signal_delta"][20, 0], 0)
        self.assertEqual(case["signal_delta"][35, 0], 30)
        self.assertEqual(case["signal_delta"][36, 0], 32)

    def test_every_pulse_has_exact_duration_and_return(self):
        for duration in (1, 2, 4, 8, 12):
            case = self.case(f"pulse{duration}", amplitude=-32)
            active = np.zeros(64, dtype=bool)
            active[20:20 + duration] = True
            np.testing.assert_array_equal(case["event_active"], active)
            self.assertTrue((case["signal_delta"][active] == -32).all())
            self.assertTrue((case["signal_delta"][~active] == 0).all())
            self.assertEqual(case["response_phase"][20 + duration], "post_event")

    def test_localized_families_use_inclusive_bottom_right_quadrant(self):
        for family, stop in (("localized_step", 64), ("localized_pulse2", 22)):
            case = self.case(family)
            x, y = case["points_xy"].T
            mask = (x >= 64) & (y >= 64)
            self.assertTrue(mask.any())
            self.assertTrue((case["signal_delta"][20:stop, mask] == 8).all())
            self.assertTrue((case["signal_delta"][:, ~mask] == 0).all())
            self.assertTrue((case["signal_delta"][:20] == 0).all())
            self.assertTrue((case["signal_delta"][stop:] == 0).all())
            np.testing.assert_array_equal(case["affected_points_mask"][20], mask)

    def test_moving_stripe_wraps_and_stops_exactly(self):
        case = self.case("moving_stripe", amplitude=-8)
        x = case["points_xy"][:, 0]
        for frame in range(20, 44):
            center = 8 + ((frame - 20) % 15) * 8
            mask = np.abs(x - center) <= 8
            np.testing.assert_array_equal(case["affected_points_mask"][frame], mask)
            np.testing.assert_array_equal(case["signal_delta"][frame], np.where(mask, -8, 0))
        np.testing.assert_array_equal(case["signal_delta"][20], case["signal_delta"][35])
        self.assertTrue((case["signal_delta"][:20] == 0).all())
        self.assertTrue((case["signal_delta"][44:] == 0).all())

    def test_registration_formula_and_zero_shift_slots(self):
        case = self.case("registration_shift", amplitude=32)
        x = case["points_xy"][:, 0]
        shifts = (0, 1, -1, 2, -2, 1, 0, -1)
        for frame in range(20, 44):
            shift = shifts[(frame - 20) % 8]
            expected = 32 * (np.sin((x - 64 + shift) / 8) - np.sin((x - 64) / 8))
            np.testing.assert_array_equal(case["signal_delta"][frame], expected)
        self.assertTrue(case["event_active"][20])
        self.assertTrue((case["signal_delta"][20] == 0).all())
        self.assertTrue((case["signal_delta"][44:] == 0).all())

    def test_missing_window_is_exact_and_underlying_step_unchanged(self):
        case = self.case("missing_window", amplitude=32, level=1)
        step = self.case("step", amplitude=32, level=1)
        missing = np.zeros((64, 144), dtype=bool)
        missing[18:21, :8] = True
        np.testing.assert_array_equal(case["missing_mask"], missing)
        np.testing.assert_array_equal(np.isnan(case["values"]), missing)
        np.testing.assert_array_equal(case["values"][~missing], step["values"][~missing])
        np.testing.assert_array_equal(case["signal_delta"], step["signal_delta"])
        self.assertEqual(int(missing.sum()), 24)

    def test_noise_profile_reuse_across_every_family_and_amplitude(self):
        for level, seed in benchmark.NOISE_PROFILES:
            expected_noise = np.random.default_rng(seed).uniform(-level, level, (64, 144))
            for spec in benchmark.specifications():
                if spec["noise_level"] != level:
                    continue
                case = benchmark.generate_case(spec)
                expected = case["base"][None, :] + case["signal_delta"] + expected_noise
                expected[case["missing_mask"]] = np.nan
                np.testing.assert_array_equal(case["values"], expected)

    def test_all_forty_twins_have_identical_priors_but_different_response(self):
        specs = {spec["case_id"]: spec for spec in benchmark.specifications()}
        twins = benchmark.benchmark_metadata()["twin_pairs"]
        self.assertEqual(len(twins), 40)
        for twin in twins:
            first = benchmark.generate_case(specs[twin["step_case_id"]])
            second = benchmark.generate_case(specs[twin["pulse_case_id"]])
            frame = twin["response_frame"]
            self.assertEqual(frame, 20 + twin["duration"])
            np.testing.assert_array_equal(first["values"][frame - 8:frame],
                                          second["values"][frame - 8:frame])
            self.assertFalse(np.array_equal(first["values"][frame], second["values"][frame]))
            np.testing.assert_allclose(first["values"][frame] - second["values"][frame],
                                       twin["amplitude"], rtol=0, atol=6e-14)

    def test_temporal_categories_include_onset_and_post_event_history(self):
        case = self.case("pulse2")
        self.assertTrue((case["response_phase"][:20] == "baseline").all())
        self.assertEqual(case["response_phase"][20], "onset")
        self.assertEqual(case["response_phase"][21], "event")
        self.assertTrue((case["response_phase"][22:] == "post_event").all())
        self.assertFalse(case["history_has_prior_event"][:21].any())
        self.assertTrue(case["history_has_prior_event"][21:30].all())
        self.assertFalse(case["history_has_prior_event"][30:].any())
        for frame in range(8, 64):
            self.assertEqual(case["history_has_prior_event"][frame],
                             case["event_active"][frame - 8:frame].any())

    def test_all_response_frame_indices_are_retained(self):
        for spec in benchmark.specifications():
            self.assertEqual(spec["response_frame_indices"], list(range(8, 64)))
            case = benchmark.generate_case(spec)
            self.assertEqual(len(case["response_phase"][8:]), 56)
        self.assertEqual(98 * 56, 5488)

    def test_no_clipping_or_quantization_is_applied(self):
        case = self.case("registration_shift", level=1)
        self.assertTrue((case["values"] != np.round(case["values"])).any())
        # The frozen amplitudes do not exceed 8-bit bounds; exact analytic
        # equality proves no quantization, while source design declares no clip.
        metadata = benchmark.benchmark_metadata()
        self.assertFalse(metadata["clipped"])
        self.assertFalse(metadata["quantized"])

    def test_generation_deterministic_and_global_rng_independent(self):
        spec = benchmark.specifications()[-1]
        first = benchmark.generate_case(spec)
        np.random.seed(938)
        np.random.uniform(size=39)
        second = benchmark.generate_case(spec)
        np.testing.assert_array_equal(first["values"], second["values"])
        self.assertEqual(benchmark.case_content_hash(first), benchmark.case_content_hash(second))

    def test_hash_binds_values_and_metadata_fields(self):
        first = self.case("step")
        original = benchmark.case_content_hash(first)
        changed = dict(first)
        changed["values"] = first["values"].copy()
        changed["values"][40, 0] += 1
        self.assertNotEqual(original, benchmark.case_content_hash(changed))
        changed = dict(first)
        changed["response_phase"] = first["response_phase"].copy()
        changed["response_phase"][40] = "baseline"
        self.assertNotEqual(original, benchmark.case_content_hash(changed))

    def test_spec_mutation_or_extra_field_is_rejected(self):
        for key, value in (("amplitude", 99), ("noise_level", True), ("seed", 0),
                           ("frame_count", 65), ("extra", 1), ("family", "unknown"),
                           ("amplitude", float("nan"))):
            spec = deepcopy(benchmark.specifications()[2])
            spec[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    benchmark.generate_case(spec)

    def test_noncanonical_spec_forms_rejected(self):
        for spec in (None, [], {}, {"case_id": "unknown"}, "step"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    benchmark.generate_case(spec)


if __name__ == "__main__":
    unittest.main()
