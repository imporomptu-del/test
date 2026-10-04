import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import accuracy_v44_synthetic_cases as cases_module


class V44SyntheticCasesTests(unittest.TestCase):
    def setUp(self):
        self.cases = cases_module.build_cases()
        self.by_id = {case["case_id"]: case for case in self.cases}

    def test_manifest_is_prespecified_json_and_matches_all_cases(self):
        manifest = cases_module.scenario_manifest()
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(len(self.cases), 16)
        self.assertEqual([c["case_id"] for c in self.cases], [c["case_id"] for c in manifest["scenarios"]])
        self.assertEqual(len(self.by_id), len(self.cases))
        self.assertFalse(manifest["uses_real_media_journals_or_scores"])
        for declaration in manifest["scenarios"]:
            case = self.by_id[declaration["case_id"]]
            self.assertEqual(case["truth"]["numerical_role"], declaration["numerical_role"])
            self.assertEqual(case["integration"]["unknown_reasons"], declaration["external_integration_unknown_reasons"])

    def test_full_finite_array_shapes_nonnegative_bounds_and_unit_source(self):
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(case["y"].shape, (625,))
                self.assertEqual(case["m"].shape, (625,))
                self.assertEqual(case["P"].shape, (625, 3))
                self.assertEqual(case["Z"].shape[0], 625)
                for name in ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound"):
                    self.assertTrue(np.all(np.isfinite(case[name])))
                    self.assertEqual(case[name].dtype, np.float64)
                for bound, array in (("response_bound", "y"), ("nuisance_bound", "Z"), ("source_bound", "m")):
                    self.assertEqual(case[bound].shape, case[array].shape)
                    self.assertTrue(np.all(case[bound] >= 0))
                self.assertAlmostEqual(np.linalg.norm(case["m"]), 1.0, places=14)
                self.assertEqual(case["truth"]["source_template_origin"], "oracle_solver_unit_fixture")

    def test_truth_reconstructs_nominal_vectors_exactly(self):
        for case in self.cases:
            t = case["truth"]
            expected = case["P"]@t["plane_coefficients"] + case["Z"]@t["nuisance_coefficients"] + t["source_amplitude"]*case["m"]
            np.testing.assert_array_equal(case["y"], expected)
            np.testing.assert_array_equal(case["synthetic_observations"]["current_image"].ravel(), case["y"])
            self.assertFalse(t["template_learned_from_current_or_history"])
            self.assertFalse(case["synthetic_observations"]["used_to_learn_core_templates"])

    def test_deterministic_and_fresh_arrays_metadata_and_manifest(self):
        again = cases_module.build_cases()
        for left, right in zip(self.cases, again):
            for name in ("y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound"):
                self.assertEqual(left[name].tobytes(), right[name].tobytes())
            self.assertEqual(left["truth"], right["truth"])
        self.cases[0]["m"][0] = 999
        self.cases[0]["truth"]["nuisance_coefficients"][0] = 999
        self.assertNotEqual(again[0]["m"][0], 999)
        self.assertEqual(again[0]["truth"]["nuisance_coefficients"], [1.0])
        manifest = cases_module.scenario_manifest()
        manifest["scenarios"][0]["external_integration_unknown_reasons"].append("injected")
        self.assertEqual(cases_module.scenario_manifest()["scenarios"][0]["external_integration_unknown_reasons"], [])

    def test_affine_changes_are_exactly_shared_plane_without_template_changes(self):
        base = self.by_id["ordinary_moving"]
        for name in ("uniform_brightness_change", "plane_brightness_change"):
            case = self.by_id[name]
            subtraction_roundoff = 4*np.finfo(np.float64).eps*np.max(np.abs(case["y"]))
            np.testing.assert_allclose(case["y"]-base["y"], case["P"]@case["truth"]["plane_coefficients"], atol=subtraction_roundoff, rtol=0)
            for key in ("Z", "m", "P", "source_bound", "nuisance_bound"):
                np.testing.assert_array_equal(case[key], base[key])

    def test_background_and_fixed_only_controls_have_zero_source_amplitude(self):
        for name in ("background_only", "fixed_flicker_separate"):
            self.assertEqual(self.by_id[name]["truth"]["source_amplitude"], 0)
        fixed = self.by_id["fixed_flicker_separate"]
        self.assertEqual(fixed["Z"].shape[1], 2)
        self.assertFalse(np.array_equal(fixed["Z"][:, 1], fixed["m"]))
        self.assertEqual(fixed["truth"]["nuisance_coefficients"], [1.0, 40.0])

    def test_overlap_and_template_error_zero_are_explicit_nonidentifiability_controls(self):
        overlap = self.by_id["fixed_source_exact_overlap"]
        np.testing.assert_array_equal(overlap["Z"][:, 1], overlap["m"])
        uncertain = self.by_id["uncertainty_contains_zero"]
        self.assertTrue(np.all(np.abs(-uncertain["m"]) <= uncertain["source_bound"]))
        self.assertAlmostEqual(np.linalg.norm(uncertain["m"]), 1.0)

    def test_missing_metadata_does_not_silently_change_numerical_inputs(self):
        base = self.by_id["ordinary_moving"]
        for name in ("raw_normalization_unsupported_metadata", "short_history", "unknown_nuisance_support", "slow_motion", "hovering"):
            case = self.by_id[name]
            self.assertTrue(case["integration"]["unknown_reasons"])
            for key in ("y", "Z", "m", "P", "response_bound", "source_bound", "nuisance_bound"):
                np.testing.assert_array_equal(case[key], base[key])
        raw = self.by_id["raw_normalization_unsupported_metadata"]
        self.assertIn("not simulated", raw["truth"]["raw_normalization_limit"])
        self.assertFalse(self.by_id["short_history"]["integration"]["history_complete"])
        self.assertEqual(self.by_id["short_history"]["synthetic_observations"]["history_images"].shape, (3, 25, 25))
        self.assertFalse(self.by_id["unknown_nuisance_support"]["integration"]["nuisance_support_known"])

    def test_curved_acceleration_has_independent_nonconstant_velocity_centers(self):
        case = self.by_id["curved_accelerating_motion"]
        provenance = case["temporal_provenance"]
        positions = np.array(provenance["centers_xy"])
        times = np.array(provenance["time_indices"])
        np.testing.assert_array_equal(positions[:, 0], 1.4*times+0.06*times**2)
        np.testing.assert_array_equal(positions[:, 1], 0.07*times**2)
        self.assertTrue(np.any(np.abs(np.diff(positions, n=2, axis=0)) > 0.1))
        self.assertEqual(provenance["current_center_xy"], [0.0, 0.0])
        self.assertFalse(provenance["physical_identity_certified_from_images"])
        # Current patch equality is intentional: differing histories cannot be
        # inferred by a routine that is given only the same current vectors.
        np.testing.assert_array_equal(case["y"], self.by_id["ordinary_moving"]["y"])

    def test_identifiability_twins_have_byte_identical_observations_and_core_inputs(self):
        moving = self.by_id["identifiability_moving_world"]
        fixed = self.by_id["identifiability_fixed_emitter_world"]
        for key in ("y", "Z", "m", "P", "response_bound", "source_bound", "nuisance_bound"):
            self.assertEqual(moving[key].tobytes(), fixed[key].tobytes())
        for key in ("history_images", "current_image"):
            self.assertEqual(moving["synthetic_observations"][key].tobytes(), fixed["synthetic_observations"][key].tobytes())
        self.assertEqual(moving["temporal_provenance"], fixed["temporal_provenance"])
        self.assertEqual(moving["integration"], fixed["integration"])
        self.assertFalse(moving["integration"]["temporal_identity_available"])
        self.assertFalse(moving["integration"]["nuisance_support_known"])
        self.assertNotEqual(moving["truth"]["physical_interpretation"], fixed["truth"]["physical_interpretation"])
        for field in set(moving["truth"]) - {"physical_interpretation"}:
            self.assertEqual(moving["truth"][field], fixed["truth"][field])

    def test_stationary_emitter_bank_reconstructs_twin_images_without_motion(self):
        fixed = self.by_id["identifiability_fixed_emitter_world"]
        yy, xx = np.indices((25, 25), dtype=np.float64)
        xx -= 12
        yy -= 12
        positions = np.array(fixed["temporal_provenance"]["centers_xy"])
        source_norm = np.linalg.norm(np.exp(-(xx**2+yy**2)/2))
        # These spatial basis functions never move. Only their one-hot light
        # coefficients vary with time, yielding the same entire observed film.
        emitters = np.stack([np.exp(-((xx-cx)**2+(yy-cy)**2)/2)/source_norm for cx, cy in positions])
        background = fixed["Z"][:, 0].reshape((25, 25))
        observations = np.concatenate((fixed["synthetic_observations"]["history_images"],
                                       fixed["synthetic_observations"]["current_image"][None]))
        for frame, observation in enumerate(observations):
            light_coefficients = np.zeros(len(positions))
            light_coefficients[frame] = fixed["truth"]["source_amplitude"]
            reconstruction = background + np.einsum("k,kij->ij", light_coefficients, emitters)
            np.testing.assert_allclose(reconstruction, observation, rtol=0, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
