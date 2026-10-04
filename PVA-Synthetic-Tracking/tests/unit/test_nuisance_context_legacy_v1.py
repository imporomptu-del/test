import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().with_name("run_nuisance_context_legacy_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_nuisance_context_legacy_v1.py"
spec = importlib.util.spec_from_file_location("legacy_context", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
core, helper, scorer = m.load_dependencies()
H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def history():
    return [dict(frame_index=f, segment=0, source_to_reference=copy.deepcopy(H),
                 motion=dict(accepted=True, reset=False), tracks=[]) for f in range(12)]


def track(number=2641, xy=None, measured=True, qualified=True, segment=0):
    return dict(segment=segment, track_id=f"dark:{number}", measured=measured,
                qualified_moving=qualified, measurement_source_xy=(xy or [64.25, 64.25]) if measured else None,
                source_xy=[999, 999], predicted_source_xy=[888, 888])


def reference(panel="dense", window="window", frame=8, identity="0/dark:2641", hit=True):
    stage = dict(hit=hit, assigned_id=identity if hit else None,
                 all_gated_ids=["0/dark:2641", "0/dark:2724"] if hit else [],
                 multiple_gated_alternatives=hit, shared_gated_observation=False)
    sample = dict(panel=panel, clip_id="0029", window_id=window, frame_index=frame,
                  source_xy=[500, 600], position_uncertainty_px=4, polarity="dark")
    return dict(sample=sample, detection_ready=True,
                stages=dict(qualified_measurement=stage,
                            actual_measurement=dict(stage, assigned_id="0/dark:2725")))


def inputs():
    rows = history()
    rows[8]["tracks"] = [track(), track(2724, [77, 77]), track(2725, [88, 88])]
    rows[4]["tracks"] = [track(xy=[62.5, 64.25], qualified=False)]
    return {"0029": [reference()], "0126": []}, {"0029": rows, "0126": history()}


def request():
    scored, rows = inputs()
    return m.build_requests(scored, rows, helper)[0][0]


class LegacyContextTests(unittest.TestCase):
    def test_exact_pinned_dependencies(self):
        self.assertEqual(m.sha(m.SCRIPT_DIR / "nuisance_context_features_v1.py"), m.PINNED_CODE["nuisance_context_features_v1.py"])
        self.assertEqual(m.sha(m.SCRIPT_DIR / "run_nuisance_context_v1.py"), m.PINNED_CODE["run_nuisance_context_v1.py"])

    def test_original_qualified_not_actual_assignment_or_manual_center(self):
        scored, rows = inputs()
        reqs, refs = m.build_requests(scored, rows, helper)
        self.assertEqual(reqs[0]["original_identity"], "0/dark:2641")
        self.assertEqual(reqs[0]["current_source_xy"], [64.25, 64.25])
        self.assertNotEqual(reqs[0]["current_source_xy"], refs[0]["sample"]["source_xy"])
        self.assertEqual(refs[0]["stages"]["actual_measurement"]["assigned_id"], "0/dark:2725")
        self.assertTrue(refs[0]["stages"]["qualified_measurement"]["multiple_gated_alternatives"])

    def test_panel_overlap_deduplicates_but_keeps_all_mappings(self):
        scored, rows = inputs()
        scored["0029"].append(reference(panel="pilot"))
        reqs, refs = m.build_requests(scored, rows, helper)
        self.assertEqual(len(reqs), 1)
        self.assertEqual(len(refs), 2)
        self.assertEqual(reqs[0]["reference_ids"], [r["reference_id"] for r in refs])
        self.assertEqual(refs[0]["request_id"], refs[1]["request_id"])

    def test_distinct_object_at_same_frame_not_deduplicated(self):
        scored, rows = inputs()
        scored["0029"].append(reference(window="other", identity="0/dark:2724"))
        reqs, refs = m.build_requests(scored, rows, helper)
        self.assertEqual(len(reqs), 2)
        self.assertNotEqual(refs[0]["request_id"], refs[1]["request_id"])

    def test_misses_never_substitute_an_actual_or_alternative(self):
        scored, rows = inputs()
        scored["0029"] = [reference(hit=False)]
        reqs, refs = m.build_requests(scored, rows, helper)
        self.assertEqual(reqs, [])
        self.assertIsNone(refs[0]["request_id"])
        self.assertEqual(refs[0]["descriptor_unavailable_reason"], "no_original_qualified_assignment")

    def test_missing_or_ambiguous_identity_remains_unavailable(self):
        for tracks, reason in (([], "assigned_identity_missing"), ([track(), track()], "assigned_identity_ambiguous"),
                               ([track(measured=False)], "assigned_identity_not_actual"),
                               ([track(qualified=False)], "assigned_identity_not_qualified")):
            scored, rows = inputs()
            rows["0029"][8]["tracks"] = tracks
            reqs, refs = m.build_requests(scored, rows, helper)
            self.assertEqual(reqs, [])
            self.assertEqual(refs[0]["descriptor_unavailable_reason"], reason)

    def test_prior_unqualified_actual_allowed(self):
        req = request()
        self.assertEqual(req["previous_source_xy"], [62.5, 64.25])
        self.assertTrue(req["prior_measured"])
        self.assertFalse(req["prior_qualified"])
        self.assertIsNone(req["temporal_unavailable_reason"])

    def test_prior_prediction_other_identity_other_segment_not_used(self):
        for prior in (track(measured=False), track(2724), track(segment=1)):
            scored, rows = inputs()
            rows["0029"][4]["tracks"] = [prior]
            reqs, _ = m.build_requests(scored, rows, helper)
            self.assertIsNone(reqs[0]["previous_source_xy"])
            self.assertIsNotNone(reqs[0]["temporal_unavailable_reason"])

    def test_reset_rejection_segment_barriers(self):
        for field, value, reason in (("reset", True, "reset_barrier"), ("accepted", False, "geometry_not_accepted")):
            scored, rows = inputs()
            rows["0029"][6]["motion"][field] = value
            reqs, _ = m.build_requests(scored, rows, helper)
            self.assertEqual(reqs[0]["temporal_unavailable_reason"], reason)
        scored, rows = inputs()
        rows["0029"][6]["segment"] = 1
        self.assertEqual(m.build_requests(scored, rows, helper)[0][0]["temporal_unavailable_reason"], "segment_change")

    def test_duplicate_reference_rejected_no_inputs_mutated(self):
        scored, rows = inputs()
        before = copy.deepcopy((scored, rows))
        m.build_requests(scored, rows, helper)
        self.assertEqual((scored, rows), before)
        scored["0029"].append(copy.deepcopy(scored["0029"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            m.build_requests(scored, rows, helper)

    def test_denominators_preserve_misses_ambiguity_and_history(self):
        scored, rows = inputs()
        scored["0029"].extend([reference(panel="pilot"), reference(frame=9, hit=False)])
        reqs, refs = m.build_requests(scored, rows, helper)
        counts = m.denominators(reqs, refs)["0029"]
        self.assertEqual([counts[k] for k in ("references", "qualified_hits", "requests", "lag4_eligible", "prior_actual_unqualified")], [3, 2, 1, 1, 1])
        self.assertEqual(counts["qualified_gated_ambiguity_reference_count"], 2)

    def test_outside_scope_rejected(self):
        scored, rows = inputs()
        scored["0029"][0]["sample"]["clip_id"] = "0055"
        with self.assertRaisesRegex(ValueError, "clip differs"):
            m.build_requests(scored, rows, helper)

    def test_sampling_matches_core_and_pair_formula(self):
        y, x = np.mgrid[:128, :128]
        gray = (100 + 10*np.exp(-((x-64)**2+(y-64)**2)/8)).astype(np.uint8)
        record, arrays = m.sample_request(request(), gray, gray, 0, core)
        p = record["array_prefix"]
        expected = core.measure_pair(arrays[p+"current"], arrays[p+"prior_background"], arrays[p+"prior_actual"], "dark")
        self.assertEqual(record["pair"]["temporal"]["D"], expected["temporal"]["D"])
        self.assertEqual(record["spatial"]["A_dn"], core.measure_patch(arrays[p+"current"], "dark")["A_dn"])

    def test_native_positive_corner_censor_survives_fractional_mixing(self):
        gray = np.full((128, 128), 100, np.uint8)
        gray[64, 64] = 255
        record, arrays = m.sample_request(request(), gray, gray, 0, core)
        self.assertLess(np.max(arrays["record000_current"]), 255)
        self.assertGreater(record["native_censor_counts"]["current"], 0)
        self.assertFalse(record["spatial"]["interpretation_available"])
        self.assertFalse(record["pair"]["interpretation_available"])

    def test_native_edge_no_clamp_or_padding(self):
        req = request()
        req["current_source_xy"] = [31.99, 64]
        record, arrays = m.sample_request(req, np.full((128, 128), 100, np.uint8), None, 0, core)
        self.assertEqual(record["temporal_status"], "current_patch_out_of_support")
        self.assertTrue(np.isnan(arrays["record000_current"][:, 0]).all())
        self.assertIsNone(record["spatial"])
        self.assertIsNone(record["pair"])

    def test_prior_support_unavailable_explicit(self):
        req = request()
        req["previous_source_xy"] = [30, 64]
        gray = np.full((128, 128), 100, np.uint8)
        record, arrays = m.sample_request(req, gray, gray, 0, core)
        self.assertEqual(record["temporal_status"], "prior_patch_out_of_support")
        self.assertIsNone(record["pair"])
        self.assertIsNotNone(record["spatial"])

    def test_saved_arrays_exact_bytes_and_feature_recomputation(self):
        gray = np.full((128, 128), 100, np.uint8)
        gray[64, 64] = 90
        record, arrays = m.sample_request(request(), gray, gray, 0, core)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patches.npz"
            np.savez_compressed(path, **arrays)
            self.assertTrue(m.verify_array_parity(path, arrays, [record], core)["passed"])
            record["spatial"]["A_dn"] += 1
            with self.assertRaisesRegex(ValueError, "feature parity"):
                m.verify_array_parity(path, arrays, [record], core)
            arrays["record000_current"] = arrays["record000_current"].astype(np.float32)
            with self.assertRaisesRegex(ValueError, "array parity"):
                m.verify_array_parity(path, arrays, [], core)

    def test_strict_json_no_overwrite_and_hash(self):
        for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                helper.read_json(content)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "plan.json"
            helper.write_json(path, {})
            with self.assertRaises(FileExistsError):
                helper.write_json(path, {})
            with self.assertRaisesRegex(ValueError, "input changed"):
                m.verify(path, "0"*64)

    def test_existing_output_never_decodes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cv2.VideoCapture") as decoder:
            with self.assertRaisesRegex(ValueError, "fresh canonical output"):
                m.run(Path(tmp)/"absent.json", Path(tmp).resolve())
            decoder.assert_not_called()

    def test_changed_plan_never_decodes(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cv2.VideoCapture") as decoder:
            root = Path(tmp).resolve()
            path = root / "plan.json"
            helper.write_json(path, {"changed": True})
            with patch.object(m, "build_plan", return_value={"changed": False}):
                with self.assertRaisesRegex(ValueError, "plan differs"):
                    m.run(path, root/"out")
            decoder.assert_not_called()
            self.assertFalse((root/"out").exists())

    def test_source_hash_mismatch_never_decodes_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cv2.VideoCapture") as decoder:
            root = Path(tmp).resolve()
            source = root / "source.avi"
            source.write_bytes(b"not real media")
            value = dict(source_bindings={"0029": dict(path=str(source), sha256="0"*64)})
            plan = root / "plan.json"
            helper.write_json(plan, value)
            with patch.object(m, "build_plan", return_value=value):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    m.run(plan, root/"out")
            decoder.assert_not_called()
            self.assertFalse((root/"out").exists())


if __name__ == "__main__":
    unittest.main()
