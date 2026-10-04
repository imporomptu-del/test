"""Exact construction witnesses and immutable-invalid-fixture regressions."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v47_synthetic_cases as old
import accuracy_v48_synthetic_cases as new


class ExactObservationRenderingTests(unittest.TestCase):
    def assertInside(self, latent, error):
        result = new.round_inside_error_box(latent, error)
        self.assertTrue(math.isfinite(result))
        self.assertLessEqual(abs(Fraction(result)-latent), Fraction(1, 2))
        return result

    def test_exact_positive_negative_and_zero_endpoints(self):
        for latent in (Fraction(0), Fraction(50), Fraction(-50), Fraction(1, 4)):
            for error in (Fraction(-1, 2), Fraction(0), Fraction(1, 2)):
                self.assertEqual(Fraction(self.assertInside(latent, error)), latent+error)

    def test_inward_adjustment_only_when_nearest_endpoint_escaped(self):
        gain = Fraction(1.2)
        seen_adjustment = 0
        for value in (42., 43.5, 57.25, 61., 68., 70., 77.5):
            for sign in (-1, 1):
                latent, intended = gain*Fraction(value), sign*Fraction(1, 2)
                nearest = float(latent+intended)
                repaired = self.assertInside(latent, intended)
                if abs(Fraction(nearest)-latent) <= Fraction(1, 2):
                    self.assertEqual(repaired, nearest)
                else:
                    seen_adjustment += 1
                    self.assertEqual(repaired, math.nextafter(nearest, -math.inf if sign > 0 else math.inf))
                self.assertGreaterEqual((Fraction(repaired)-latent)*intended, 0)
        self.assertGreater(seen_adjustment, 0)

    def test_no_arbitrary_tolerance_accepts_outside_endpoint(self):
        latent = Fraction(1)+Fraction(1, 2**54)
        intended = Fraction(-1, 2)
        self.assertLess(Fraction(float(latent+intended)), latent-Fraction(1, 2))
        self.assertGreater(self.assertInside(latent, intended), float(latent+intended))

    def test_subnormal_ties_and_small_exact_latent_values(self):
        smallest = Fraction(math.nextafter(0., 1.))
        self.assertEqual(self.assertInside(smallest/2, Fraction(0)), 0.)
        self.assertEqual(self.assertInside(3*smallest/2, Fraction(0)), float(2*smallest))
        for sign in (-1, 1):
            self.assertInside(sign*smallest/3, Fraction(1, 2))
            self.assertInside(sign*smallest/3, Fraction(-1, 2))

    def test_finite_maximum_can_be_the_only_representable_witness(self):
        largest = Fraction(sys.float_info.max)
        self.assertEqual(self.assertInside(largest+Fraction(1, 4), Fraction(-1, 2)), sys.float_info.max)
        self.assertEqual(self.assertInside(-largest-Fraction(1, 4), Fraction(1, 2)), -sys.float_info.max)

    def test_overflow_and_empty_representable_boxes_fail_closed(self):
        for latent in (Fraction(2**2000), Fraction(-2**2000),
                       Fraction(2**54+1), Fraction(-2**54-1),
                       Fraction(sys.float_info.max)+1):
            with self.assertRaisesRegex(ValueError, "no finite binary64"):
                new.round_inside_error_box(latent, Fraction(0))

    def test_rejects_inexact_arguments_and_outside_error_contract(self):
        for latent, error in ((1., Fraction(0)), (Fraction(1), .5), (1, Fraction(0))):
            with self.assertRaises(TypeError):
                new.round_inside_error_box(latent, error)
        for error in (Fraction(1, 2)+Fraction(1, 2**200), Fraction(-1)):
            with self.assertRaisesRegex(ValueError, "unchanged"):
                new.round_inside_error_box(Fraction(1), error)


class V48SyntheticConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.original = new.build_cases(), old.build_cases()
        cls.corrected = cls.cases[-1]
        cls.uncorrected = cls.original[-1]
        cls.yy, cls.xx = np.indices((129, 129), dtype=float)
        cls.radius = np.maximum(abs(cls.xx-64), abs(cls.yy-64))
        cls.band = (cls.radius >= 40) & (cls.radius <= 56)
        cls.signs = np.where((np.floor(cls.xx/8)+np.floor(cls.yy/8)) % 2 == 0, 1, -1)
        cls.background = 50.+20.*(cls.xx >= 64.)+8.*np.sin(cls.yy/12.)
        cls.gain = Fraction(1.2)

    def test_only_current_annulus_and_additive_metadata_of_one_case_change(self):
        self.assertEqual(tuple(c["case_id"] for c in self.cases), old.CASE_IDS)
        for actual, previous in zip(self.cases[:-1], self.original[:-1]):
            self.assertEqual({k: v for k, v in actual.items() if k != "adapter_inputs"},
                             {k: v for k, v in previous.items() if k != "adapter_inputs"})
            for key in old.ADAPTER_KEYS:
                np.testing.assert_array_equal(actual["adapter_inputs"][key], previous["adapter_inputs"][key])
        for key in set(old.ADAPTER_KEYS)-{"current129"}:
            np.testing.assert_array_equal(self.corrected["adapter_inputs"][key], self.uncorrected["adapter_inputs"][key])
        np.testing.assert_array_equal(self.corrected["adapter_inputs"]["current129"][~self.band],
                                      self.uncorrected["adapter_inputs"]["current129"][~self.band])
        stripped = deepcopy({k: v for k, v in self.corrected.items() if k != "adapter_inputs"})
        for group, key in (("generator_truth", "current_error_rendering_v48"), ("provenance", "renderer_correction_v48")):
            metadata = stripped[group].pop(key)
            self.assertFalse(metadata["detector_or_gain_solver_used_for_rendering"])
            self.assertEqual(metadata["error_radius_dn"], .5)
        self.assertEqual(stripped, {k: v for k, v in self.uncorrected.items() if k != "adapter_inputs"})

    def test_full_band_exact_current_witness_and_error_direction(self):
        current = self.corrected["adapter_inputs"]["current129"]
        altered = 0
        for y, x in zip(*np.where(self.band)):
            latent = self.gain*Fraction(float(self.background[y, x]))
            intended = -int(self.signs[y, x])*Fraction(1, 2)
            error = Fraction(float(current[y, x]))-latent
            self.assertLessEqual(abs(error), Fraction(1, 2))
            self.assertGreaterEqual(error*intended, 0)
            nearest = float(latent+intended)
            # Independent expression: the inward neighbor is the nearest
            # endpoint itself if legal, otherwise the one toward the latent.
            expected = nearest if abs(Fraction(nearest)-latent) <= Fraction(1, 2) else math.nextafter(nearest, -math.inf if intended > 0 else math.inf)
            self.assertEqual(current[y, x], expected)
            altered += current[y, x] != self.uncorrected["adapter_inputs"]["current129"][y, x]
        self.assertEqual(int(self.band.sum()), 6528)
        self.assertEqual(altered, 3294)

    def test_source_free_witness_is_not_a_claim_about_the_source_core(self):
        source = 30*np.exp(-((self.xx-64)**2+(self.yy-64)**2)/2)
        self.assertEqual(np.count_nonzero(source[self.band]), 0)
        self.assertEqual(source[64, 64], 30)
        self.assertEqual(self.corrected["adapter_inputs"]["current129"][64, 64],
                         self.uncorrected["adapter_inputs"]["current129"][64, 64])

    def test_unchanged_prior_error_witness_excludes_all_fixed_footprints(self):
        prior_domain = self.band.copy()
        a = self.corrected["adapter_inputs"]
        for cx, cy in a["prior_centers_xy"]:
            prior_domain &= np.maximum(abs(self.xx-cx), abs(self.yy-cy)) > 12
        self.assertLess(int(prior_domain.sum()), int(self.band.sum()))
        for i, (cx, cy) in enumerate(a["prior_centers_xy"]):
            clean = self.background+30*np.exp(-((self.xx-cx)**2+(self.yy-cy)**2)/2)
            np.testing.assert_array_equal(clean[prior_domain], self.background[prior_domain])
            for y, x in zip(*np.where(prior_domain)):
                error = Fraction(float(a["history129"][i, y, x]))-Fraction(float(self.background[y, x]))
                intended = int(self.signs[y, x])*Fraction(1, 2)
                self.assertLessEqual(abs(error), Fraction(1, 2))
                self.assertGreaterEqual(error*intended, 0)

    def test_preserved_original_failure_has_73_used_pixel_and_59_contrast_violations(self):
        a = self.uncorrected["adapter_inputs"]
        points = {(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                  if 40 <= max(abs(x-64), abs(y-64)) <= 56
                  and all(max(abs(x-cx), abs(y-cy)) > 12 for cx, cy in a["prior_centers_xy"])}
        triplets = []
        for x, y in sorted(points):
            for dx, dy in ((8, 0), (0, 8)):
                triplet = [(x-dx, y-dy), (x, y), (x+dx, y+dy)]
                if all(p in points for p in triplet):
                    triplets.append(triplet)
        used = set(p for t in triplets for p in t)
        f = lambda value: Fraction(float(value))
        bad_pixels = sum(abs(f(a["current129"][y, x])-self.gain*f(self.background[y, x])) > Fraction(1, 2) for x, y in used)
        bad_contrasts = 0
        for triplet in triplets:
            response = sum(w*f(a["current129"][y, x]) for w, (x, y) in zip((1, -2, 1), triplet))
            prior = sum(w*f(a["history129"][0, y, x]) for w, (x, y) in zip((1, -2, 1), triplet))
            bad_contrasts += abs(response-self.gain*prior) > 2+2*self.gain
        self.assertEqual((len(used), len(triplets)), (141, 184))
        self.assertEqual(bad_pixels, 73)
        self.assertEqual(bad_contrasts, 59)
        # The legacy declaration stays unchanged rather than being relabeled.
        self.assertEqual(self.uncorrected["generator_truth"]["current_error_max_abs_dn"], .5)
        self.assertNotIn("current_error_rendering_v48", self.uncorrected["generator_truth"])

    def test_pre_score_witness_passes_and_is_json_serializable(self):
        report = new.pre_score_contract(self.cases)
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(report["issues"], [])
        self.assertFalse(report["uses_scores"])
        self.assertEqual(report["unchanged_cases"], 19)
        self.assertEqual(report["changed_current_pixel_count"], 3294)
        self.assertEqual(report["preserved_original_current_full_annulus_violation_count"], 3294)
        self.assertEqual(report["prior_max_abs_error_exact"], "1/2")
        self.assertEqual(report["prior_error_not_exact_endpoint_count"], 128)
        for key in ("current_witness_violation_count", "prior_witness_violation_count",
                    "current_error_sign_violation_count", "prior_error_sign_violation_count",
                    "deterministic_renderer_mismatch_count",
                    "current_source_stored_nonzero_pixel_count", "prior_source_changes_stored_background_count"):
            self.assertEqual(report[key], 0, key)
        json.dumps(report, allow_nan=False)

    def test_pre_score_contract_rejects_invalid_legacy_case_and_tampering(self):
        legacy = deepcopy(self.cases)
        legacy[-1]["adapter_inputs"]["current129"] = self.uncorrected["adapter_inputs"]["current129"].copy()
        report = new.pre_score_contract(legacy)
        self.assertFalse(report["passed"])
        self.assertEqual(report["current_witness_violation_count"], 3294)
        tampered = deepcopy(self.cases)
        tampered[0]["generator_truth"]["current_source_peak_dn"] = 0
        tampered[-1]["adapter_inputs"]["current129"][64, 64] += 1
        tampered[-1]["adapter_inputs"]["history129"][0, 8, 8] += 1
        report = new.pre_score_contract(tampered)
        self.assertFalse(report["passed"])
        self.assertTrue(any("outside the declared" in message for message in report["issues"]))
        self.assertGreater(report["prior_witness_violation_count"], 0)
        self.assertFalse(new.pre_score_contract(self.cases[:-1])["passed"])

    def test_repeat_builds_and_manifest_are_deterministic_no_score_inputs(self):
        repeated = new.build_cases()
        for a, b in zip(self.cases, repeated):
            self.assertEqual({k: v for k, v in a.items() if k != "adapter_inputs"},
                             {k: v for k, v in b.items() if k != "adapter_inputs"})
            self.assertEqual(set(a["adapter_inputs"]), set(old.ADAPTER_KEYS))
            for key in old.ADAPTER_KEYS:
                np.testing.assert_array_equal(a["adapter_inputs"][key], b["adapter_inputs"][key])
        manifest = new.scenario_manifest()
        for key, value in old.scenario_manifest().items():
            self.assertEqual(manifest[key], value)
        self.assertEqual(manifest["renderer_revision"], 48)
        json.dumps(manifest, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
