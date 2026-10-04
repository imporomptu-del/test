import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_independent_motion_pairs as check


class IndependentPairs(unittest.TestCase):
    def _control_fixture(self):
        inventory = check.matcher.control_inventory()
        rows = []
        for case in inventory["cases"]:
            qualified = case["expectation"] != "abstain"
            measurement = dict(qualified=qualified,
                raw_best_displacement_xy=case["truth_displacement_xy"] if qualified else None,
                displacement_xy=case["truth_displacement_xy"] if qualified else None)
            rows.append(dict(case=case, queries=[dict(previous_xy=case["point_queries_xy"][0],
                measurement=measurement, assessment=check.matcher.assess_control(case, measurement))]))
        return dict(schema=check.matcher.SCHEMA+".result", passed=True, generated_only=True,
            source_media_accessed=False, production_changes=False, model_fits=0,
            inventory=inventory, rows=rows,
            input_sha256={str(path): check.pixels.sha(path) for path in (
                Path(check.matcher.__file__).resolve(),
                check.pixels.REPOSITORY/"tests/unit/test_motion_patch_controls.py")})

    def test_empty_summary_has_no_invented_values(self):
        result = check.summarize_points([])
        self.assertEqual(result["qualified"], 0)
        self.assertIsNone(result["qualified_lk_disagreement_px"]["median"])

    def test_failed_controls_prevent_source_access(self):
        with patch.object(check, "validate_controls", side_effect=ValueError("failed control")), \
             patch.object(check.pixels, "load_verified_analysis") as load, \
             patch.object(check.pixels, "decode_verified_pairs") as decode:
            with self.assertRaisesRegex(ValueError, "failed control"):
                check.run("unused", "unused", "unused", "unused")
            load.assert_not_called()
            decode.assert_not_called()

    def test_real_failed_control_validation_stops_before_any_other_hash_or_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"failed_controls.json"
            value = self._control_fixture()
            value["passed"] = False
            path.write_text(json.dumps(value))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes = []
            def only_control_hash(requested):
                self.assertEqual(Path(requested), path, "A later metadata/source file was accessed")
                hashes.append(Path(requested))
                return digest
            with patch.object(check.pixels, "sha", side_effect=only_control_hash), \
                 patch.object(check.pixels, "load_verified_analysis") as analysis, \
                 patch.object(check.pixels, "decode_verified_pairs") as decode, \
                 patch.object(check.matcher, "measure_patch") as match:
                with self.assertRaisesRegex(ValueError, "Generated controls have not passed"):
                    check.run(path, digest, "must-not-read-analysis", Path(directory)/"must-not-create")
            self.assertEqual(hashes, [path])
            analysis.assert_not_called()
            decode.assert_not_called()
            match.assert_not_called()
            self.assertFalse((Path(directory)/"must-not-create").exists())

    def test_top_level_pass_cannot_hide_a_failed_required_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"controls.json"
            value = self._control_fixture()
            query = value["rows"][8]["queries"][0]
            query["measurement"]["qualified"] = False
            query["measurement"]["displacement_xy"] = None
            query["assessment"] = check.matcher.assess_control(value["rows"][8]["case"], query["measurement"])
            self.assertFalse(query["assessment"]["passed"])
            path.write_text(json.dumps(value))
            with patch.object(check.pixels, "load_verified_analysis") as load, \
                 patch.object(check.pixels, "decode_verified_pairs") as decode:
                with self.assertRaisesRegex(ValueError, "Generated control assessment differs"):
                    check.run(path, check.pixels.sha(path), "not-read", "not-created")
            load.assert_not_called()
            decode.assert_not_called()

    def test_control_inventory_and_assessment_changes_fail_before_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"controls.json"
            for change in ("contract", "missing_case", "wrong_assessment", "wrong_query", "code_hash"):
                value = self._control_fixture()
                if change == "contract":
                    value["inventory"]["contract"]["minimum_runner_gap"] = .001
                elif change == "missing_case":
                    value["rows"].pop()
                elif change == "wrong_assessment":
                    value["rows"][0]["queries"][0]["assessment"]["qualified_vector_error_px"] = 1.
                elif change == "wrong_query":
                    value["rows"][0]["queries"][0]["previous_xy"] = [1., 1.]
                else:
                    value["input_sha256"][str(Path(check.matcher.__file__).resolve())] = "0"*64
                path.write_text(json.dumps(value))
                with self.subTest(change=change), \
                     patch.object(check.pixels, "load_verified_analysis") as load, \
                     patch.object(check.pixels, "decode_verified_pairs") as decode:
                    with self.assertRaises(ValueError):
                        check.run(path, check.pixels.sha(path), "not-read", "not-created")
                    load.assert_not_called()
                    decode.assert_not_called()

    def test_only_previous_points_reach_matcher(self):
        image = np.zeros((64, 64), np.uint8)
        p = np.array([[31.25, 30.5], [32.0, 32.0]], np.float64)
        q = p + [1, -1]
        arrays = (p, q, np.array([1., 2.]), np.zeros(2), np.array([True, False]), np.full(2, np.sqrt(2)))
        measured = dict(qualified=True, displacement_xy=[.5, -.5], abstention_reasons=[], cost={"elapsed_seconds": .01})
        missing = dict(qualified=False, displacement_xy=None, raw_best_displacement_xy=[3., 3.],
                       abstention_reasons=["interior_peak"], cost={"elapsed_seconds": .02})
        with patch.object(check.matcher, "measure_patch", side_effect=[measured, missing]) as match:
            result = check.measure_pair(image, image, arrays, np.eye(3))
        self.assertEqual(len(match.call_args_list), 2)
        for i, call in enumerate(match.call_args_list):
            self.assertEqual(len(call.args), 3)
            self.assertEqual(call.kwargs, {})
            np.testing.assert_array_equal(call.args[2], p[i])
        self.assertIsNone(result["points"][1]["comparison"])
        self.assertAlmostEqual(result["points"][0]["comparison"]["lk_difference_norm_px"], np.sqrt(.5))
        self.assertEqual(result["summary"]["all"]["qualified"], 1)
        self.assertEqual(result["summary"]["all"]["unavailable"], 1)

    def test_original_fit_binding_checked_before_match(self):
        p = np.array([[32., 32.]])
        arrays = (p, p, np.ones(1), np.zeros(1), np.array([True]), np.ones(1))
        with patch.object(check.matcher, "measure_patch") as match:
            with self.assertRaisesRegex(ValueError, "Original residuals differ"):
                check.measure_pair(np.zeros((64,64), np.uint8), np.zeros((64,64), np.uint8), arrays, np.eye(3))
            match.assert_not_called()

    def test_actual_generated_search_invariant_to_saved_endpoint_and_fit_metadata(self):
        generated = next(check.matcher.generated_controls((96,120)))
        before, after = generated["previous_gray"], generated["current_gray"]
        p = np.asarray(generated["case"]["point_queries_xy"], np.float64)
        q1, q2 = p+[.5, -.75], p+[2., -1.]
        matrix1, matrix2 = np.eye(3), np.array([[1., 0., .8], [0., 1., -.3], [0., 0., 1.]])
        residual1 = np.linalg.norm(q1-p, axis=1)
        residual2 = np.linalg.norm(q2-(p+[.8, -.3]), axis=1)
        first = check.measure_pair(before, after,
            (p, q1, np.array([1.]), np.array([.01]), np.array([True]), residual1), matrix1)
        second = check.measure_pair(before, after,
            (p, q2, np.array([100000.]), np.array([2.]), np.array([False]), residual2), matrix2)
        one, two = first["points"][0], second["points"][0]
        def scientific(record):
            return {key: value for key, value in record.items() if key != "cost"}
        self.assertEqual(scientific(one["independent"]), scientific(two["independent"]))
        self.assertEqual(one["independent"]["displacement_xy"], [0., 0.])
        self.assertNotEqual(one["comparison"], two["comparison"])

    def test_missing_control_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Control result identity differs"):
                check.validate_controls(Path(directory) / "missing.json", "0"*64)


if __name__ == "__main__":
    unittest.main()
