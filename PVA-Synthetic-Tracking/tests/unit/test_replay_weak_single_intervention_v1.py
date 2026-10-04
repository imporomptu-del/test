"""Generated-only single-intervention tests using the existing native fixture."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_replay_weak_divergence_v1 as generated

SCRIPT = Path(__file__).resolve().with_name("replay_weak_single_intervention_v1.py")
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts/replay_weak_single_intervention_v1.py"
spec = importlib.util.spec_from_file_location("single_weak_intervention", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SingleInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generated.ReplayTests.setUpClass()
        cls.geometry, cls.batch = generated.ReplayTests.geometry, generated.ReplayTests.batch

    @classmethod
    def tearDownClass(cls):
        generated.ReplayTests.tearDownClass()

    def setUp(self):
        self.h = m.helper()
        self.patches = [patch.object(m, "EVENT_FRAME", 5), patch.object(m, "EVENT_TRACK", "bright:0"),
                        patch.object(m, "FRAME_COUNT", 9)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def data(self):
        b, s, ref = generated.fixture()
        with self.h.native_backend(self.geometry, self.h.sha(self.geometry), self.batch, self.h.sha(self.batch)) as (method, _, _):
            parent = self.h.replay_rows(b, s, ref, method)
        return b, s, ref, parent

    def execute(self, data):
        b, s, ref, parent = data
        with self.h.native_backend(self.geometry, self.h.sha(self.geometry), self.batch, self.h.sha(self.batch)) as (method, _, _):
            return m.intervene_rows(b, s, ref, method, parent, self.h)

    def test_exactly_one_correction_and_preintervention_baseline_parity(self):
        result = self.execute(self.data())
        self.assertEqual(result["weak_corrections_applied"], 1)
        self.assertEqual(result["later_weak_corrections_applied"], 0)
        self.assertEqual(len(result["frames"]), 9)
        self.assertEqual([r["frame_index"] for r in result["frames"] if r["intervention"]], [5])
        self.assertTrue(all(not r["arms"]["single"]["comparison_to_original_baseline"]["exact_assignment_changed"]
                            for r in result["frames"][:5]))
        self.assertTrue(all(not r["arms"]["single"]["comparison_to_all_weak_shadow"]["exact_assignment_changed"]
                            for r in result["frames"]))
        self.assertEqual(len(result["strong_histories"]["single"]["bright:0"]), 8)
        self.assertEqual(result["strong_histories"]["single"], result["strong_histories"]["baseline"])

    def test_later_note_is_not_an_additional_intervention_input(self):
        data = self.data()
        # Deliberately malformed later note proves it is never consumed as an
        # intervention. Real inputs additionally require the passed parent audit.
        data[1][8]["records"][0]["weak_evidence"] = dict(applied=True, mean_before="DO NOT READ")
        result = self.execute(data)
        self.assertEqual(result["weak_corrections_applied"], 1)
        self.assertEqual(result["later_weak_corrections_applied"], 0)

    def test_fixed_owner_prior_and_computational_birth_must_match(self):
        data = self.data()
        data[1][5]["records"][0]["weak_evidence"]["mean_before"][0] += 1
        with self.assertRaisesRegex(ValueError, "weak_mean_before"):
            self.execute(data)
        data = self.data()
        data[3][5]["arms"]["shadow"]["managers"]["bright"]["prior"][0]["birth_timestamp_ns"] += 1
        with self.assertRaisesRegex(ValueError, "owner history differs"):
            self.execute(data)

    def test_fixed_observation_cannot_be_missing(self):
        data = self.data()
        data[1][5]["records"][0]["weak_evidence"]["applied"] = False
        with self.assertRaisesRegex(ValueError, "intervention absent"):
            self.execute(data)

    def test_baseline_parity_failure_stops_new_experiment(self):
        data = self.data()
        data[0][7]["tracks"][0]["measurement_source_xy"][0] += .001
        with self.assertRaisesRegex(ValueError, "baseline.records"):
            self.execute(data)

    def test_qualified_coordinates_ignore_id_but_assignments_do_not(self):
        records = [dict(track_id="bright:1", measured=True, qualified_moving=True, measurement_source_xy=[1., 2.])]
        other = deepcopy(records)
        other[0]["track_id"] = "bright:77"
        result = m.changes(records, other)
        self.assertTrue(result["exact_assignment_changed"])
        self.assertFalse(result["qualified_actual_coordinate_multiset_changed"])

    def test_parent_requires_pass_same_prefix_helper_and_native_pins(self):
        report = dict(schema="seaqr.weak-divergence-replay.v1", passed=True, error=None, completed_frames=9,
            frames=[{}]*9, script_sha256=m.PARENT_HELPER_SHA, input_receipt_sha256="1"*64,
            media_accessed=False, weak_selection_reevaluated=False, production_changed=False,
            adapter=dict(geometry_library_sha256="2"*64, batch_library_sha256="3"*64))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"parent.json.gz"
            def save(value):
                with gzip.open(path, "wt") as stream:
                    json.dump(value, stream)
            save(report)
            self.assertTrue(m.load_parent(path, self.h.sha(path), "1"*64, "2"*64, "3"*64, self.h)["passed"])
            for key, value in (("passed", False), ("input_receipt_sha256", "0"*64), ("script_sha256", "0"*64)):
                changed = dict(report, **{key: value})
                save(changed)
                with self.assertRaises(ValueError):
                    m.load_parent(path, self.h.sha(path), "1"*64, "2"*64, "3"*64, self.h)
            save(report)
            with self.assertRaisesRegex(ValueError, "backend differs"):
                m.load_parent(path, self.h.sha(path), "1"*64, "0"*64, "3"*64, self.h)

    def test_failure_is_saved_exclusively_and_no_parent_result_is_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"single.json.gz"
            with patch.object(m, "load_parent", side_effect=ValueError("generated failed parent")):
                with self.assertRaisesRegex(ValueError, "generated failed parent"):
                    m.run(tmp, "1"*64, "parent", "2"*64, self.geometry, "3"*64, self.batch, "4"*64, output)
            with gzip.open(output, "rt") as stream:
                self.assertFalse(json.load(stream)["passed"])
            with patch.object(m, "load_parent", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "Fresh single"):
                    m.run(tmp, "1"*64, "parent", "2"*64, self.geometry, "3"*64, self.batch, "4"*64, output)


if __name__ == "__main__":
    unittest.main()
