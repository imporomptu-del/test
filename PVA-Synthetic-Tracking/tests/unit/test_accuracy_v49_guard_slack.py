from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from accuracy_v49_guard_slack import minimum_guard_response_slack, verify_guard_response_slack_certificate


def contrast(y, b, ey=0, eb=0):
    return dict(response=y, background=b, response_error=ey, background_error=eb)


def brute_force(rows):
    """Independent small-case oracle: every pair intersection, direct abs cost."""
    exact = [tuple(Fraction(row[key]) for key in
                   ("response", "background", "response_error", "background_error")) for row in rows]
    lines = [(Fraction(0), Fraction(0))]
    for y, b, ey, eb in exact:
        lines.extend([((-b-eb)/4, (y-ey)/4), ((b-eb)/4, (-y-ey)/4)])
    candidates = {Fraction(0)}
    for index, (a, b) in enumerate(lines):
        for c, d in lines[index+1:]:
            if a != c:
                intersection = (d-b)/(a-c)
                if intersection >= 0:
                    candidates.add(intersection)
    return min((max([Fraction(0)]+[(abs(y-g*b)-ey-g*eb)/4 for y, b, ey, eb in exact]), g)
               for g in candidates)


class GuardSlackTests(unittest.TestCase):
    def check(self, rows, expected_gain, expected_slack):
        result = minimum_guard_response_slack(rows)
        self.assertEqual(Fraction(result["gain_exact"]), expected_gain)
        self.assertEqual(Fraction(result["slack_exact"]), expected_slack)
        self.assertTrue(verify_guard_response_slack_certificate(rows, result))
        json.dumps(result, allow_nan=False)
        return result

    def test_exact_inconsistent_pair_has_straddling_witness(self):
        result = self.check([contrast(10, 1), contrast(0, 1)], Fraction(5), Fraction(5, 4))
        witness = result["optimality_witness"]
        self.assertEqual(witness["kind"], "straddling_active_slopes")
        self.assertEqual(witness["convex_weights_exact"], ["1/2", "1/2"])
        self.assertEqual(witness["slopes_exact"], ["-1/4", "1/4"])

    def test_boundary_minimum_requires_no_negative_gain(self):
        result = self.check([contrast(-10, 1)], Fraction(0), Fraction(5, 2))
        self.assertEqual(result["optimality_witness"]["kind"], "nonnegative_boundary_slope")
        self.assertEqual(result["active_signed_constraints"][0]["residual_sign"], -1)

    def test_zero_slack_interval_uses_smallest_nonnegative_gain(self):
        result = self.check([contrast(10, 2, 2)], Fraction(4), Fraction(0))
        self.assertEqual(result["optimality_witness"]["kind"], "zero_slack_global_lower_bound")
        self.assertEqual(result["active_signed_constraints"][0]["residual_sign"], 1)

    def test_positive_flat_plateau_returns_earliest_point_and_zero_slope(self):
        result = self.check([contrast(8, 0), contrast(12, 1)], Fraction(4), Fraction(2))
        self.assertEqual(result["optimality_witness"]["convex_weights_exact"], ["0", "1"])
        self.assertEqual(result["optimality_witness"]["slopes_exact"], ["-1/4", "0"])

    def test_constant_positive_constraint_minimizes_at_boundary(self):
        result = self.check([contrast(8, 0)], Fraction(0), Fraction(2))
        self.assertEqual(result["optimality_witness"]["supporting_slope_exact"], "0")

    def test_empty_zero_and_redundant_constraints_are_vacuous_or_zero(self):
        result = self.check([], Fraction(0), Fraction(0))
        self.assertTrue(result["vacuous_empty_constraints"])
        for rows in ([contrast(0, 0)], [contrast(0, 0, 10, 5)], [contrast(0, 1)]):
            result = self.check(rows, Fraction(0), Fraction(0))
            self.assertFalse(result["vacuous_empty_constraints"])

    def test_all_decreasing_lines_reach_the_zero_floor(self):
        self.check([contrast(12, 0, 0, 1)], Fraction(12), Fraction(0))
        self.check([contrast(10, 2, 0, 3)], Fraction(2), Fraction(0))

    def test_all_tied_active_lines_survive_envelope_deduplication(self):
        rows = [contrast(10, 1), contrast(0, 1)]*7
        result = self.check(rows, Fraction(5), Fraction(5, 4))
        self.assertEqual(len(result["active_signed_constraints"]), 14)

    def test_equal_slope_dominated_intercepts_and_three_way_ties(self):
        rows = [contrast(8, 0), contrast(10, 1), contrast(-6, 1), contrast(4, 0)]
        result = self.check(rows, Fraction(2), Fraction(2))
        self.assertEqual(len(result["active_signed_constraints"]), 3)

    def test_sign_and_scale_symmetries(self):
        rows = [contrast(10, 1, 2), contrast(0, 1, 1)]
        base = minimum_guard_response_slack(rows)
        signed = [contrast(-r["response"], -r["background"], r["response_error"], r["background_error"])
                  for r in rows]
        self.check(signed, Fraction(base["gain_exact"]), Fraction(base["slack_exact"]))
        for scale in (Fraction(1, 7), Fraction(3), Fraction(1000)):
            scaled = [{key: value*scale for key, value in row.items()} for row in rows]
            self.check(scaled, Fraction(base["gain_exact"]), Fraction(base["slack_exact"])*scale)

    def test_float_inputs_are_exact_binary_ratios_including_numpy_longdouble(self):
        gain = Fraction.from_float(.3)/Fraction.from_float(.1)
        self.assertNotEqual(gain, 3)
        self.check([contrast(.3, .1)], gain, Fraction(0))
        extended = np.nextafter(np.longdouble(1), np.longdouble(2))
        self.check([contrast(extended, np.int64(1))], Fraction(*extended.as_integer_ratio()), Fraction(0))

    def test_recorded_rational_strings_and_generator_input(self):
        rows = [contrast("7/3", "2/5", "1/3", "1/5")]
        self.check(rows, Fraction(10, 3), Fraction(0))
        result = minimum_guard_response_slack(row for row in rows)
        self.assertEqual(result["gain_exact"], "10/3")

    def test_nonfinite_boolean_bad_type_missing_and_negative_errors_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, np.longdouble("nan"), True, np.bool_(False),
                    "nan", "inf", "1/0", "1.2", "1e3", " 1", "1/2/3", [1, 2], None):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    minimum_guard_response_slack([contrast(bad, 1)])
        for field in ("response_error", "background_error"):
            row = contrast(1, 1)
            row[field] = -Fraction(1, 100)
            with self.assertRaises(ValueError):
                minimum_guard_response_slack([row])
        for row in ({}, [], {"response": 1}):
            with self.assertRaises(ValueError):
                minimum_guard_response_slack([row])

    def test_display_overflow_and_underflow_never_change_exact_output(self):
        enormous = Fraction(10**400)
        result = self.check([contrast(enormous, 1)], enormous, Fraction(0))
        self.assertIsNone(result["gain_display"])
        result = self.check([contrast(enormous, 0)], Fraction(0), enormous/4)
        self.assertIsNone(result["slack_display"])
        tiny = Fraction(1, 10**400)
        result = self.check([contrast(tiny, 0)], Fraction(0), tiny/4)
        self.assertEqual(result["slack_display"], 0.)
        self.assertNotEqual(result["slack_exact"], "0")

    def test_generated_small_cases_match_independent_pair_enumeration(self):
        rng = random.Random(4901)
        for index in range(250):
            rows = [contrast(Fraction(rng.randint(-20, 20), rng.randint(1, 7)),
                             Fraction(rng.randint(-12, 12), rng.randint(1, 5)),
                             Fraction(rng.randint(0, 9), rng.randint(1, 5)),
                             Fraction(rng.randint(0, 8), rng.randint(1, 5)))
                    for _ in range(rng.randrange(9))]
            expected_slack, expected_gain = brute_force(rows)
            with self.subTest(index=index):
                self.check(rows, expected_gain, expected_slack)

    def test_many_lines_and_duplicate_slopes(self):
        rows = [contrast(index, 1) for index in range(4001)]
        self.check(rows, Fraction(2000), Fraction(500))

    def test_certificate_verifier_rejects_false_objective_active_set_or_witness(self):
        rows = [contrast(10, 1), contrast(0, 1)]
        original = minimum_guard_response_slack(rows)
        mutations = [lambda r: r.update(gain_exact="4"),
                     lambda r: r.update(slack_exact="0"),
                     lambda r: r.update(stencil_l1_weight=3),
                     lambda r: r.update(active_signed_constraints=[]),
                     lambda r: r["optimality_witness"].update(convex_weights_exact=["1", "0"]),
                     lambda r: r["optimality_witness"].update(line_ids=["slack_floor", "slack_floor"]),
                     lambda r: r["optimality_witness"].update(kind="not_a_certificate"),
                     lambda r: r["optimality_witness"].update(slopes_exact=["0", "0"])]
        for mutate in mutations:
            result = deepcopy(original)
            mutate(result)
            with self.assertRaises(ValueError):
                verify_guard_response_slack_certificate(rows, result)

    def test_result_explicitly_disclaims_physical_and_production_conclusions(self):
        result = minimum_guard_response_slack([contrast(1, 1)])
        for key in ("production_bounds_changed", "calibrated_noise_estimate", "proposed_threshold",
                    "sufficient_for_joint_pixel_or_affine_feasibility", "proves_guard_cleanliness",
                    "proves_guard_to_core_transfer", "proves_source_presence"):
            self.assertFalse(result["interpretation"][key])


if __name__ == "__main__":
    unittest.main()
