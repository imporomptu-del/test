"""Synthetic V38 evidence checks, with no source media or truth labels.

Independent least-squares oracles test the fractional-coordinate model.
Counterexamples check honest availability and provenance, not accuracy.
"""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import accuracy_v38_evidence as evidence


Y, X = np.mgrid[-12:13, -12:13].astype(np.float64)
PY, PX = np.mgrid[-13:14, -13:14].astype(np.float64)
DESIGN = np.column_stack([v.ravel() for v in (np.ones_like(X), X, Y, X*X, X*Y, Y*Y)])
FIXED_SHIFTS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def gaussian(x=X, y=Y, *, xy=(0., 0.), sigma=2.):
    return np.exp(-((x-xy[0])**2 + (y-xy[1])**2)/(2*sigma*sigma))


def background(x=X, y=Y):
    return 95 + .15*x - .2*y + .006*x*x - .004*x*y + .008*y*y


def edge(width, theta, offset):
    return np.tanh((X*math.cos(theta)+Y*math.sin(theta)-offset)/width)


def fit(columns, vector):
    matrix = np.column_stack((DESIGN, *columns)) if columns else DESIGN
    coefficients = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    residual = vector-matrix@coefficients
    return float(residual@residual), coefficients


def independent_banks(patch, current_xy, polarity):
    """Exhaust original, unnormalized designs without implementation helpers."""
    vector = np.asarray(patch, dtype=np.float64).ravel()
    energy, _ = fit([], vector)
    sign = 1 if polarity == "bright" else -1
    point_candidates = []
    for sigma in (1., 2., 3.):
        for dx in (-1., 0., 1.):
            for dy in (-1., 0., 1.):
                g = gaussian(xy=(current_xy[0]+dx, current_xy[1]+dy), sigma=sigma).ravel()
                sse, coefficients = fit([sign*g], vector)
                point_candidates.append((sse if coefficients[-1] >= 0 else energy,
                                         sigma, dx, dy, max(float(coefficients[-1]), 0.)))
    edge_candidates = []
    for width in (1., 2., 4.):
        for theta in (k*math.pi/8 for k in range(8)):
            for offset in (-2., 0., 2.):
                column = edge(width, theta, offset).ravel()
                sse, coefficients = fit([column], vector)
                edge_candidates.append((sse, width, theta, offset, float(coefficients[-1])))
    return energy, min(point_candidates, key=lambda v: v[0]), min(edge_candidates, key=lambda v: v[0])


class AccuracyV38EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = evidence.SourceEvidenceDiagnostic()

    @staticmethod
    def probe(result, shift=(0, 0)):
        return next(p for p in result["probes"] if p["shift_xy"] == list(shift))

    def assert_metadata(self, value):
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.assertIs(value["diagnostic_only"], True)
        for name in ("classifier_promoted", "uncertainty_envelope_is_calibrated",
                     "independent_votes", "lag_or_shift_selection_applied"):
            self.assertIs(value[name], False, name)
        self.assertEqual([p["shift_xy"] for p in value["probes"]],
                         [list(s) for s in FIXED_SHIFTS])
        self.assertEqual(value["source_supported_probes"],
                         sum(p["source_supported"] for p in value["probes"]))
        self.assertEqual(value["informative_probes"],
                         sum(p["contrast_informative"] for p in value["probes"]))
        self.assertEqual(value["nominal_contrast_available"],
                         self.probe(value)["contrast_informative"])
        self.assertEqual(value["envelope_available"],
                         all(p["contrast_informative"] for p in value["probes"]))
        self.assertIsNone(value["current_source"]["presence_classification"])
        self.assertEqual(value["current_source"]["physical_class"], "unknown")
        self.assertEqual(value["photometric_gain"], 1.)
        self.assertEqual(value["photometric_offset"], 0.)
        if not value["envelope_available"]:
            self.assertIsNone(value["envelope"])
            self.assertTrue(value["reasons"])
        for p in value["probes"]:
            self.assertIsInstance(p["source_supported"], bool)
            self.assertIsInstance(p["contrast_informative"], bool)
            self.assertFalse({"accepted", "rejected", "is_object", "is_noise", "classification"} & p.keys())
            self.assertEqual(p["contrast_informative"],
                             bool(p["difference_features"] and p["difference_features"]["informative"]))

    def test_fractional_gaussian_bank_recovers_exact_actual_center_and_offsets(self):
        for sigma, xy, offset, amplitude in ((1., (.37, -.26), (0, 0), 13.),
                                            (2., (-.5, .5), (-1, 1), 21.),
                                            (3., (.12, .4), (1, -1), 9.)):
            for polarity, sign in (("bright", 1), ("dark", -1)):
                with self.subTest(sigma=sigma, xy=xy, polarity=polarity):
                    center = (xy[0]+offset[0], xy[1]+offset[1])
                    patch = background()+sign*amplitude*gaussian(xy=center, sigma=sigma)
                    result = evidence.FractionalPointEdge(xy).measure(patch, polarity)
                    self.assertTrue(result["informative"])
                    self.assertAlmostEqual(result["point_gain_fraction"], 1., places=11)
                    self.assertAlmostEqual(result["point_amplitude_dn"], amplitude, places=9)
                    self.assertEqual(result["point_sigma_px"], sigma)
                    self.assertEqual(result["point_offset_xy"], list(offset))

    def test_fractional_noisy_point_and_edge_gains_match_independent_full_banks(self):
        rng = np.random.default_rng(38001)
        xy = (.37, -.26)
        for polarity, sign in (("bright", 1), ("dark", -1)):
            patch = (background()+sign*11*gaussian(xy=(xy[0]+1, xy[1]), sigma=2)
                     + 17*edge(2, math.pi/4, -2) + rng.normal(0, .3, (25, 25)))
            result = evidence.FractionalPointEdge(xy).measure(patch, polarity)
            energy, point_fit, edge_fit = independent_banks(patch, xy, polarity)
            self.assertAlmostEqual(result["background_residual_energy"], energy, places=7)
            self.assertAlmostEqual(result["point_gain_fraction"], 1-point_fit[0]/energy, places=10)
            self.assertAlmostEqual(result["edge_gain_fraction"], 1-edge_fit[0]/energy, places=10)
            self.assertAlmostEqual(result["point_amplitude_dn"], point_fit[-1], places=8)
            self.assertAlmostEqual(result["edge_amplitude_dn"], edge_fit[-1], places=8)
            self.assertEqual(result["point_sigma_px"], point_fit[1])
            self.assertEqual(result["point_offset_xy"], list(point_fit[2:4]))

    def test_fractional_conditional_point_after_edge_matches_independent_joint_design(self):
        xy = (-.31, .24)
        patch = background()+12*gaussian(xy=xy, sigma=1)+23*edge(2, math.pi/8, 0)
        result = evidence.FractionalPointEdge(xy).measure(patch, "bright")
        edge_column = edge(result["edge_width_px"], result["edge_orientation_rad"],
                           result["edge_offset_px"]).ravel()
        edge_sse, _ = fit([edge_column], patch.ravel())
        candidates = []
        for sigma in (1., 2., 3.):
            for dx in (-1., 0., 1.):
                for dy in (-1., 0., 1.):
                    g = gaussian(xy=(xy[0]+dx, xy[1]+dy), sigma=sigma).ravel()
                    sse, coefficients = fit([edge_column, g], patch.ravel())
                    candidates.append((sse if coefficients[-1] >= 0 else edge_sse,
                                       max(float(coefficients[-1]), 0.)))
        best = min(candidates, key=lambda v: v[0])
        self.assertAlmostEqual(result["point_gain_after_edge_fraction"], 1-best[0]/edge_sse, places=10)
        self.assertAlmostEqual(result["point_after_edge_amplitude_dn"], best[1], places=8)

    def test_flat_background_and_no_previous_association_do_not_prevent_evidence(self):
        current = 90+15*gaussian(xy=(.2, -.3))
        prior = np.full((27, 27), 90.)
        result = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright")
        self.assert_metadata(result)
        self.assertEqual(result["informative_probes"], 9)
        self.assertTrue(result["envelope_available"])
        self.assertFalse(result["previous_actual_measurement_available"])
        self.assertTrue(result["missing_prior_point_may_contaminate_difference"])
        for p in result["probes"]:
            self.assertAlmostEqual(p["difference_features"]["point_amplitude_dn"], 15., places=9)
            self.assertFalse(p["conditional_pair"]["available"])
            self.assertEqual(p["conditional_pair"]["reasons"], ["missing_previous_actual_measurement"])

    def test_moving_point_without_previous_association_retains_all_probes(self):
        current = 90+15*gaussian(xy=(.2, -.3))
        prior = 90+18*gaussian(PX, PY, xy=(-5.2, 2.4), sigma=1)
        result = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright")
        self.assert_metadata(result)
        self.assertEqual(result["source_supported_probes"], 9)
        self.assertEqual(result["informative_probes"], 9)
        self.assertTrue(result["nominal_contrast_available"])

    def test_exact_shift_crops_and_fractional_current_coordinate_stay_fixed(self):
        rng = np.random.default_rng(38002)
        current = 100+rng.normal(0, 2, (25, 25))
        prior = 99+rng.normal(0, 2, (27, 27))
        xy = (.43, -.17)
        result = self.model.measure(current, prior, current_xy=xy, polarity="bright")
        for p in result["probes"]:
            dx, dy = p["shift_xy"]
            difference = current-prior[1+dy:26+dy, 1+dx:26+dx]
            expected = evidence.FractionalPointEdge(xy).measure(difference, "bright")
            self.assertEqual({k:v for k,v in p["difference_features"].items()
                              if k != "signed_point_amplitude_dn"}, expected)
            independent_energy, pf, ef = independent_banks(difference, xy, "bright")
            self.assertAlmostEqual(expected["background_residual_energy"], independent_energy, places=8)
            self.assertAlmostEqual(expected["point_gain_fraction"], 1-pf[0]/independent_energy, places=10)
            self.assertAlmostEqual(expected["edge_gain_fraction"], 1-ef[0]/independent_energy, places=10)

    def test_envelopes_include_all_nine_in_fixed_order_not_best_shift(self):
        current = 90+17*gaussian(xy=(.3, -.2), sigma=1)
        prior = 90+12*gaussian(PX, PY, xy=(-4.1, 2.3), sigma=3)
        result = self.model.measure(current, prior, current_xy=(.3, -.2), polarity="bright")
        self.assert_metadata(result)
        self.assertTrue(result["envelope_available"])
        for metric in evidence.METRICS:
            values = sorted(p["difference_features"][metric] for p in result["probes"])
            self.assertEqual(result["envelope"][metric], dict(
                minimum=values[0], median=values[4], maximum=values[-1], span=values[-1]-values[0]))
        self.assertGreater(result["envelope"]["point_gain_fraction"]["span"], 0)

    def test_stationary_point_cancellation_is_unknown_despite_other_shift_residuals(self):
        prior = 100+20*gaussian(PX, PY)
        current = prior[1:26, 1:26].copy()
        result = self.model.measure(current, prior, polarity="bright", previous_xy=(0, 0))
        self.assert_metadata(result)
        self.assertTrue(result["current_source"]["informative"])
        self.assertFalse(result["nominal_contrast_available"])
        self.assertEqual(result["source_supported_probes"], 9)
        self.assertEqual(result["informative_probes"], 8)
        self.assertFalse(result["envelope_available"])
        self.assertEqual(self.probe(result)["reasons"], ["uninformative_difference"])

    def test_one_pixel_motion_cancels_one_probe_without_becoming_negative(self):
        current = 100+20*gaussian()
        prior = 100+20*gaussian(PX, PY, xy=(-1, 0))
        result = self.model.measure(current, prior, polarity="bright")
        self.assert_metadata(result)
        self.assertTrue(result["nominal_contrast_available"])
        self.assertFalse(self.probe(result, (-1, 0))["contrast_informative"])
        self.assertEqual(result["informative_probes"], 8)
        self.assertFalse(result["envelope_available"])

    def test_constant_and_quadratic_differences_are_not_shape_evidence(self):
        result = self.model.measure(background(), 120-.1*PX+.3*PY+.01*PX*PY,
                                    polarity="bright")
        self.assert_metadata(result)
        self.assertFalse(result["current_source"]["informative"])
        self.assertEqual(result["source_supported_probes"], 9)
        self.assertEqual(result["informative_probes"], 0)
        for p in result["probes"]:
            self.assertEqual(p["reasons"], ["uninformative_difference"])

    def test_blink_on_and_dimming_preserve_source_snapshot_without_classification(self):
        current = 90+9*gaussian()
        for prior_amplitude in (0, 18):
            with self.subTest(prior_amplitude=prior_amplitude):
                prior = 90+prior_amplitude*gaussian(PX, PY)
                result = self.model.measure(current, prior, polarity="bright")
                self.assert_metadata(result)
                self.assertTrue(result["current_source"]["informative"])
                self.assertTrue(result["nominal_contrast_available"])
                amplitude = self.probe(result)["difference_features"]["point_amplitude_dn"]
                self.assertAlmostEqual(amplitude, 9 if prior_amplitude == 0 else 0, places=9)

    def test_bright_dark_symmetry_including_signed_amplitude(self):
        current = 90+15*gaussian(xy=(.2, -.3))
        prior = 90+18*gaussian(PX, PY, xy=(-5.2, 2.4), sigma=1)
        bright = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright")
        dark = self.model.measure(255-current, 255-prior, current_xy=(.2, -.3), polarity="dark")
        self.assert_metadata(dark)
        for a, b in zip(bright["probes"], dark["probes"]):
            af, bf = a["difference_features"], b["difference_features"]
            for key in ("background_residual_energy", "point_gain_fraction", "edge_gain_fraction",
                        "point_minus_edge_fraction", "point_amplitude_dn"):
                self.assertAlmostEqual(af[key], bf[key], places=8, msg=key)
            self.assertAlmostEqual(af["signed_point_amplitude_dn"], -bf["signed_point_amplitude_dn"], places=8)
            self.assertAlmostEqual(af["edge_amplitude_dn"], -bf["edge_amplitude_dn"], places=8)

    def test_previous_association_changes_only_conditional_model_and_metadata(self):
        current = 90+15*gaussian(xy=(.2, -.3))
        prior_xy = (-5.2, 2.4)
        prior = 90+18*gaussian(PX, PY, xy=prior_xy, sigma=1)
        missing = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright")
        present = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright", previous_xy=prior_xy)
        self.assert_metadata(present)
        self.assertEqual(missing["current_source"], present["current_source"])
        self.assertEqual(missing["envelope"], present["envelope"])
        for a, b in zip(missing["probes"], present["probes"]):
            self.assertEqual(a["difference_features"], b["difference_features"])
            dx, dy = b["shift_xy"]
            self.assertEqual(b["previous_point_xy"], [prior_xy[0]-dx, prior_xy[1]-dy])
            self.assertEqual(b["conditional_pair"]["previous_xy"], b["previous_point_xy"])
            self.assertEqual(b["conditional_pair"]["current_xy"], [.2, -.3])
            self.assertTrue(b["conditional_pair"]["available"])
            self.assertAlmostEqual(b["conditional_pair"]["point_current_amplitude_dn"], 15., places=8)
            self.assertAlmostEqual(b["conditional_pair"]["point_previous_amplitude_dn"], 18., places=8)

    def test_unavailable_conditional_pair_does_not_block_source_contrast(self):
        current = 90+15*gaussian()
        prior = np.full((27, 27), 90.)
        for prior_xy in ((40, 40), (0, 0)):
            with self.subTest(prior_xy=prior_xy):
                result = self.model.measure(current, prior, polarity="bright", previous_xy=prior_xy)
                self.assert_metadata(result)
                self.assertTrue(result["envelope_available"])
                self.assertFalse(self.probe(result)["conditional_pair"]["available"])

    def test_nan_prior_corner_disables_only_affected_probe_and_robust_envelope(self):
        current = 90+15*gaussian()
        prior = np.full((27, 27), 90.)
        prior[0, 0] = np.nan
        result = self.model.measure(current, prior, polarity="bright")
        self.assert_metadata(result)
        self.assertEqual(result["source_supported_probes"], 8)
        self.assertEqual(result["informative_probes"], 8)
        self.assertTrue(result["nominal_contrast_available"])
        unsupported = self.probe(result, (-1, -1))
        self.assertIsNone(unsupported["difference_features"])
        self.assertEqual(unsupported["conditional_pair"]["reasons"], ["unsupported_source_patch"])

    def test_nan_current_or_all_prior_is_unknown_and_serializable(self):
        current = 90+15*gaussian()
        bad_current = current.copy()
        bad_current[12, 12] = np.nan
        for c, p in ((bad_current, np.full((27, 27), 90.)),
                     (current, np.full((27, 27), np.nan))):
            result = self.model.measure(c, p, polarity="bright")
            self.assert_metadata(result)
            self.assertEqual(result["source_supported_probes"], 0)
            self.assertEqual(result["informative_probes"], 0)
        self.assertTrue(result["current_source"]["source_supported"])
        self.assertTrue(result["current_source"]["informative"])

    def test_saturation_counts_preserved_per_actual_shifted_support(self):
        current = np.full((25, 25), 90.)
        current[3, 4], current[7, 8] = 0, 255
        prior = np.full((27, 27), 90.)
        prior[0, 0], prior[26, 26] = 0, 255
        result = self.model.measure(current, prior, polarity="bright")
        self.assert_metadata(result)
        for p in result["probes"]:
            dx, dy = p["shift_xy"]
            self.assertEqual(p["saturation_counts"], dict(
                current_zero=1, current_255=1,
                interpolated_prior_zero=int((dx, dy) == (-1, -1)),
                interpolated_prior_255=int((dx, dy) == (1, 1))))

    def test_persistent_strong_edge_cancels_nominally_without_erasing_point(self):
        current = 100+40*np.tanh((X+.3*Y)/2)+8*gaussian(xy=(.2, -.3), sigma=1)
        prior = 100+40*np.tanh((PX+.3*PY)/2)
        result = self.model.measure(current, prior, current_xy=(.2, -.3), polarity="bright")
        self.assert_metadata(result)
        nominal = self.probe(result)["difference_features"]
        self.assertAlmostEqual(nominal["point_gain_fraction"], 1., places=10)
        self.assertAlmostEqual(nominal["point_amplitude_dn"], 8., places=9)
        self.assertGreater(result["envelope"]["edge_gain_fraction"]["span"], .01)

    def test_gain_change_and_deforming_cloud_are_described_not_classified(self):
        cases = (
            (110+45*np.tanh(X/2), 100+30*np.tanh(PX/2)),
            (100+35*np.tanh((X-.025*Y*Y)/2),
             100+35*np.tanh((PX-.01*PY*PY-.2)/2)),
        )
        for current, prior in cases:
            result = self.model.measure(current, prior, polarity="bright")
            self.assert_metadata(result)
            self.assertTrue(result["nominal_contrast_available"])
            self.assertGreater(self.probe(result)["difference_features"]["background_residual_energy"], 0)
            self.assertEqual(result["photometric_gain"], 1.)
            # These are deliberately not asserted to be objects or negatives.

    def test_faint_noisy_quantized_pairs_are_valid_without_detection_assertions(self):
        rng = np.random.default_rng(38003)
        for amplitude in (.2, 1., 3.):
            current = np.clip(np.rint(95+amplitude*gaussian()+rng.normal(0, .35, (25, 25))), 0, 255).astype(np.uint8)
            prior = np.clip(np.rint(95+2*amplitude*gaussian(PX, PY, xy=(-3, 2))
                                    +rng.normal(0, .35, (27, 27))), 0, 255).astype(np.uint8)
            result = self.model.measure(current, prior, polarity="bright")
            self.assert_metadata(result)
            self.assertEqual(result["source_supported_probes"], 9)

    def test_repeatability_inputs_not_mutated_and_previous_results_independent(self):
        current = 90+15*gaussian()
        prior = 90+18*gaussian(PX, PY, xy=(-5.2, 2.4), sigma=1)
        current.setflags(write=False)
        prior.setflags(write=False)
        c, p = current.copy(), prior.copy()
        first = self.model.measure(current, prior, polarity="bright", previous_xy=(-5.2, 2.4))
        snapshot = copy.deepcopy(first)
        second = self.model.measure(current, prior, polarity="bright", previous_xy=(-5.2, 2.4))
        self.assertEqual(first, second)
        second["probes"][0]["difference_features"]["point_amplitude_dn"] = -999
        self.assertEqual(first, snapshot)
        np.testing.assert_array_equal(current, c)
        np.testing.assert_array_equal(prior, p)

    def test_invalid_source_shapes_ranges_infinity_and_complex_are_rejected(self):
        current, prior = np.full((25, 25), 90.), np.full((27, 27), 90.)
        for replacement in (np.ones((24, 25)), np.full((25, 25), np.inf),
                            np.full((25, 25), -1), np.full((25, 25), 256),
                            np.ones((25, 25), dtype=complex), [["not numeric"]]):
            with self.subTest(shape=np.shape(replacement)):
                with self.assertRaises(ValueError):
                    self.model.measure(replacement, prior, polarity="bright")
        for replacement in (np.ones((27, 26)), np.full((27, 27), -np.inf),
                            np.full((27, 27), -.01), np.full((27, 27), 255.01),
                            np.ones((27, 27), dtype=complex)):
            with self.assertRaises(ValueError):
                self.model.measure(current, replacement, polarity="bright")

    def test_invalid_coordinates_and_polarity_rejected(self):
        current, prior = np.full((25, 25), 90.), np.full((27, 27), 90.)
        for center in ((.5001, 0), (0, -.5001), (np.nan, 0), (np.inf, 0), (0j, 0), (0,), "xy"):
            with self.subTest(center=center):
                with self.assertRaises(ValueError):
                    self.model.measure(current, prior, polarity="bright", current_xy=center)
        for old in ((np.nan, 0), (np.inf, 0), (0j, 0), (0,), "xy"):
            with self.assertRaises(ValueError):
                self.model.measure(current, prior, polarity="bright", previous_xy=old)
        for polarity in (None, "both", "Bright", 1):
            with self.assertRaises(ValueError):
                self.model.measure(current, prior, polarity=polarity)

    def test_envelope_rejects_partial_nonfinite_or_wrong_shape(self):
        for values in ([1]*8, [1]*10, [1]*8+[np.nan], [1]*8+[np.inf], [[1]*9]):
            with self.assertRaises(ValueError):
                evidence.envelope(values)
        self.assertEqual(evidence.envelope([9, 1, 8, 2, 7, 3, 6, 4, 5]),
                         dict(minimum=1., median=5., maximum=9., span=8.))


if __name__ == "__main__":
    unittest.main()
