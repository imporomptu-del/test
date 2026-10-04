"""Generated-only policy and exact-source isolation tests; no journal/media reads."""
from collections import Counter
from pathlib import Path
import random
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_tracking_batch_v27 import build
from build_tracking_geometry_v20 import build as build_scalar
from check_tracker_capacity_shadow import CONFIG, CONFIG_SHA, make_adapter, read, value_sha
from test_kalman_tracking import batch, candidate, config
import tracker_maturity_candidate_v1 as experiment
from tracking_stage_v28 import VictimIndex
from tiny_target.tracking import KalmanTrackManager


def configured(**changes):
    return config(confirmation_independent_hits=4, max_missed_windows=7,
                  birth_policy="spatial_fair", measurement_model="position_only",
                  association_cost="gaussian_nll", association_prior="hit_maturity",
                  **changes)


def observable(result):
    result = result.to_dict()
    result.pop("timings_ms")
    return value_sha(result)


class VictimOrderTests(unittest.TestCase):
    def test_exact_frozen_maturity_order_with_dynamic_cell_counts(self):
        rng = random.Random(20261001)
        for _ in range(30):
            tracks = {i: SimpleNamespace(independent_confirmation_hits=rng.randint(1, 3),
                                        missed_windows=rng.randint(1, 7)) for i in range(100)}
            cells = {i: (rng.randrange(8), 0) for i in tracks}
            occupancy = Counter(cells.values())
            replaceable = {i for i in tracks if i % 3}
            index = experiment.MaturityFirstVictimIndex(tracks, cells, replaceable)
            for _ in range(60):
                chosen = (rng.randrange(8), 0)
                eligible = [i for i in replaceable if occupancy[cells[i]] > occupancy[chosen] + 1]
                expected = max(eligible, key=lambda i: (-tracks[i].independent_confirmation_hits,
                    occupancy[cells[i]], tracks[i].missed_windows, -i)) if eligible else None
                self.assertEqual(index.take(occupancy, occupancy[chosen] + 1), expected)
                if expected is not None:
                    replaceable.remove(expected)
                    occupancy[cells[expected]] -= 1
                    occupancy[chosen] += 1

    def test_ineligible_one_hit_does_not_protect_eligible_two_hit(self):
        tracks = {0: SimpleNamespace(independent_confirmation_hits=1, missed_windows=7),
                  1: SimpleNamespace(independent_confirmation_hits=2, missed_windows=1)}
        cells = {0: "sparse", 1: "dense"}
        index = experiment.MaturityFirstVictimIndex(tracks, cells, set(tracks))
        self.assertEqual(index.take(Counter(sparse=1, dense=3), 1), 1)
        self.assertIsNone(index.take(Counter(sparse=1, dense=2), 1))

    def test_ties_use_occupancy_then_misses_then_lower_id(self):
        tracks = {i: SimpleNamespace(independent_confirmation_hits=1, missed_windows=2) for i in (9, 4, 7)}
        tracks[9].missed_windows = 7
        index = experiment.MaturityFirstVictimIndex(tracks, {9: "a", 4: "b", 7: "b"}, set(tracks))
        occupancy = Counter(a=2, b=3)
        self.assertEqual(index.take(occupancy, 1), 4)
        occupancy["b"] -= 1
        self.assertEqual(index.take(occupancy, 1), 9)
        self.assertEqual(index.take(occupancy, 1), 7)
        self.assertIsNone(index.take(occupancy, 1))

    def test_empty_index_is_unavailable_not_a_fabricated_victim(self):
        self.assertIsNone(experiment.MaturityFirstVictimIndex({}, {}, set()).take(Counter(), 0))


class FrozenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="seaqr_maturity_generated_")
        root = Path(cls.temp.name)
        build(root / "batch")
        cls.libraries = {"batch": root / "batch/libtracking_batch_v27.so",
                         "geometry": build_scalar(root / "scalar")}
        cls.module_sha = experiment.sha(Path(experiment.__file__).resolve())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def methods(self):
        original, _, _ = make_adapter(self.libraries)
        adapted, _, _ = make_adapter(self.libraries)
        changed, binding = experiment.make_candidate_method(adapted, expected_module_sha256=self.module_sha)
        return original, changed, adapted, binding

    def seed(self, manager, items):
        for i, item in enumerate(items):
            x, y, hits, missed, confirmed = item
            track = manager._new_track(candidate(i, x, y), batch(0, (0,)))
            track.independent_confirmation_hits = hits
            track.associated_update_count = hits
            track.missed_windows = missed
            if confirmed:
                track.confirmation_timestamp_ns = 0
                track.lifecycle_state = "coasted" if missed else "confirmed"
            manager._tracks[track.track_id] = track

    def test_isolated_binding_keeps_source_bytecode_defaults_and_baseline_globals(self):
        original, changed, adapted, binding = self.methods()
        old, new = adapted.__globals__, changed.__globals__
        self.assertIsNot(old, new)
        self.assertIs(old["_VictimIndex_v28"], VictimIndex)
        self.assertIs(new["_VictimIndex_v28"], experiment.MaturityFirstVictimIndex)
        self.assertIs(new["_births_v28"].__globals__, new)
        self.assertIs(new["_record_v28"].__globals__, new)
        self.assertIs(old["_births_v28"].__globals__, old)
        self.assertIs(KalmanTrackManager.update.__globals__["KalmanTrackManager"], KalmanTrackManager)
        structural = set(binding["isolated_function_rebindings"] + binding["changed_global_bindings"])
        self.assertEqual(old.keys(), new.keys())
        for key in old.keys() - structural:
            self.assertIs(old[key], new[key], key)
        self.assertIs(changed.__code__, adapted.__code__)
        self.assertEqual(changed.__defaults__, adapted.__defaults__)
        self.assertEqual(changed.__kwdefaults__, adapted.__kwdefaults__)
        self.assertEqual(binding["sha256"], self.module_sha)
        self.assertEqual(len(binding["class_source_sha256"]), 64)
        self.assertEqual(set(binding["class_code_sha256"]), {"__init__", "take"})
        self.assertEqual(binding["unchanged_transformed_source_sha256"], experiment.METHOD_SHA)
        self.assertTrue(binding["requires_causal_rerun_after_learning_input_divergence"])

    def test_wrong_candidate_hash_rejected_before_baseline_use(self):
        with self.assertRaisesRegex(ValueError, "candidate module differs"):
            experiment.make_candidate_method(None, expected_module_sha256="0" * 64)

    def test_frozen_visible_configuration_is_unmodified_on_generated_points(self):
        from tiny_target.visible_baseline import VisibleConfig
        self.assertEqual(experiment.sha(CONFIG), CONFIG_SHA)
        cfg = VisibleConfig(**read(CONFIG)).tracker(10)
        self.assertEqual((cfg.confirmation_independent_hits, cfg.max_missed_windows,
                          cfg.max_active_tracks, cfg.birth_cell_size_px), (4, 7, 256, 256))
        original, changed, _, _ = self.methods()
        managers = [KalmanTrackManager(cfg) for _ in range(2)]
        before = value_sha(cfg)
        for frame in range(12):
            points = (candidate(0, 10+frame, 10),) if frame < 5 or frame == 11 else ()
            outputs = [method(manager, batch(frame*100_000_000, (frame,), points))
                       for method, manager in zip((original, changed), managers)]
            self.assertEqual(observable(outputs[0]), observable(outputs[1]))
            self.assertEqual(value_sha(vars(managers[0])), value_sha(vars(managers[1])))
            self.assertIs(managers[1].config, cfg)
            self.assertEqual(value_sha(cfg), before)

    def test_changed_generated_function_is_not_trusted_by_stale_source_hash(self):
        original, _, _ = make_adapter(self.libraries)
        scope = dict(original.__globals__)
        wrong = FunctionType((lambda: None).__code__, scope, "update")
        scope["update"] = wrong
        for name in ("_births_v28", "_record_v28"):
            scope[name] = experiment._clone(original.__globals__[name], scope)
        with self.assertRaisesRegex(ValueError, "generated function differs"):
            experiment.make_candidate_method(wrong, expected_module_sha256=self.module_sha)

    def test_refuses_already_changed_victim_binding(self):
        _, changed, _, _ = self.methods()
        with self.assertRaisesRegex(ValueError, "victim binding"):
            experiment.make_candidate_method(changed, expected_module_sha256=self.module_sha)

    def test_one_hit_replaced_before_two_three_even_with_fewer_misses(self):
        original, changed, _, _ = self.methods()
        items = [(10, 10, 3, 7, False), (40, 10, 2, 6, False),
                 (70, 10, 1, 1, False), (100, 10, 1, 3, False)]
        victims = []
        for method in (original, changed):
            manager = KalmanTrackManager(configured(max_active_tracks=4))
            self.seed(manager, items)
            deleted = []
            born, audit = method.__globals__["_births_v28"](manager,
                batch(100_000_000, (1,), (candidate(0, 1200, 10),)), [0], set(), deleted)
            self.assertEqual(len(born), 1)
            self.assertEqual(audit["confirmed_tracks_evicted"], 0)
            self.assertEqual(len(manager._tracks), 4)
            victims.append(deleted[0]["track_id"])
        self.assertEqual(victims, [0, 3])

    def test_assigned_never_missed_and_ever_confirmed_protections(self):
        _, changed, _, _ = self.methods()
        manager = KalmanTrackManager(configured(max_active_tracks=5))
        self.seed(manager, [(10, 10, 1, 1, False), (40, 10, 1, 0, False),
                            (70, 10, 4, 7, True), (100, 10, 3, 1, False), (130, 10, 2, 1, False)])
        # Even a misleading lifecycle label cannot erase ever-confirmed status.
        manager._tracks[2].lifecycle_state = "tentative"
        deleted = []
        changed.__globals__["_births_v28"](manager,
            batch(100_000_000, (1,), (candidate(0, 1200, 10),)), [0], {0}, deleted)
        self.assertEqual([r["track_id"] for r in deleted], [4])
        self.assertTrue({0, 1, 2, 3} <= set(manager._tracks))

    def test_newborns_and_dynamic_quota_protected_in_same_admission(self):
        _, changed, _, _ = self.methods()
        manager = KalmanTrackManager(configured(max_active_tracks=4))
        self.seed(manager, [(10 + 30*i, 10, 1, 1, False) for i in range(4)])
        deleted = []
        points = tuple(candidate(i, 1200 + 5*i, 10) for i in range(5))
        born, audit = changed.__globals__["_births_v28"](manager,
            batch(100_000_000, (1,), points), list(range(5)), set(), deleted)
        self.assertEqual(born, [4, 5])
        self.assertEqual([r["track_id"] for r in deleted], [0, 1])
        self.assertEqual(audit["rejected_candidate_indices"], [2, 3, 4])
        self.assertEqual(set(manager._tracks), {2, 3, 4, 5})

    def test_four_independent_hits_and_overlap_credit_unchanged(self):
        original, changed, _, _ = self.methods()
        managers = [KalmanTrackManager(configured()) for _ in range(2)]
        windows = [(0, 1), (1, 2), (3, 4), (5, 6), (7, 8)]
        for frame, window in enumerate(windows):
            outputs = [method(manager, batch(frame*100_000_000, window, (candidate(0, 10, 10),)))
                       for method, manager in zip((original, changed), managers)]
            self.assertEqual(observable(outputs[0]), observable(outputs[1]))
            track = managers[1]._tracks[0]
            self.assertEqual(track.independent_confirmation_hits, (1, 1, 2, 3, 4)[frame])
            self.assertEqual(track.lifecycle_state, "confirmed" if frame == 4 else "tentative")

    def test_intermittent_target_survives_seven_misses_then_reacquires(self):
        original, changed, _, _ = self.methods()
        for method in (original, changed):
            manager = KalmanTrackManager(configured())
            for frame in range(3):
                method(manager, batch(frame*100_000_000, (frame,), (candidate(0, 10, 10),)))
            for frame in range(3, 10):
                result = method(manager, batch(frame*100_000_000, (frame,)))
                self.assertFalse(result.deleted_tracks)
                self.assertEqual(manager._tracks[0].missed_windows, frame-2)
            result = method(manager, batch(1_000_000_000, (10,), (candidate(0, 10, 10),)))
            self.assertEqual(result.born_track_ids, ())
            self.assertEqual(manager._tracks[0].independent_confirmation_hits, 4)
            self.assertEqual(manager._tracks[0].lifecycle_state, "confirmed")

    def test_eighth_miss_expires_tentative_and_confirmed_without_prolongation(self):
        original, changed, _, _ = self.methods()
        for hits in (1, 4):
            managers = [KalmanTrackManager(configured()) for _ in range(2)]
            for frame in range(hits + 8):
                points = (candidate(0, 10, 10),) if frame < hits else ()
                outputs = [method(manager, batch(frame*100_000_000, (frame,), points))
                           for method, manager in zip((original, changed), managers)]
                self.assertEqual(observable(outputs[0]), observable(outputs[1]))
                if frame == hits + 6:
                    self.assertIn(0, managers[1]._tracks)
                if frame == hits + 7:
                    self.assertNotIn(0, managers[1]._tracks)
                    self.assertEqual(outputs[1].deleted_tracks[0]["reason"], "maximum_missed_windows_exceeded")

    def test_crossing_association_and_full_state_exact_when_no_replacement(self):
        original, changed, _, _ = self.methods()
        managers = [KalmanTrackManager(configured(initial_velocity_sigma_px_s=30)) for _ in range(2)]
        for frame in range(30):
            points = (candidate(0, 20+frame, 10, score=5 if frame % 2 else 50),
                      candidate(1, 49-frame, 13, score=50 if frame % 2 else 5))
            outputs = [method(manager, batch(frame*100_000_000, (frame,), points))
                       for method, manager in zip((original, changed), managers)]
            self.assertEqual(observable(outputs[0]), observable(outputs[1]))
            self.assertEqual(value_sha(vars(managers[0])), value_sha(vars(managers[1])))
            if frame:
                self.assertEqual({a["track_id"]: a["candidate_index"] for a in outputs[1].associations}, {0: 0, 1: 1})

    def test_generated_crowding_is_deterministic_and_keeps_capacity_audit(self):
        _, changed, _, _ = self.methods()
        managers = [KalmanTrackManager(configured(max_active_tracks=8)) for _ in range(2)]
        rng = random.Random(2901)
        for frame in range(45):
            points = tuple(candidate(i, rng.randrange(1800), rng.randrange(900), score=4+rng.random()*10)
                           for i in range(24))
            outputs = [changed(manager, batch(frame*100_000_000, (frame,), points)) for manager in managers]
            self.assertEqual(observable(outputs[0]), observable(outputs[1]))
            self.assertEqual(value_sha(vars(managers[0])), value_sha(vars(managers[1])))
            result = outputs[1]
            self.assertLessEqual(len(result.tracks), 8)
            metrics = result.metrics
            self.assertEqual(metrics["unmatched_candidate_count"], metrics["birth_count"] + metrics["dropped_birth_count_at_active_track_cap"])
            self.assertEqual(metrics["birth_admission"]["confirmed_tracks_evicted"], 0)
            self.assertTrue(all(t.missed_windows <= 7 for t in managers[1]._tracks.values()))


if __name__ == "__main__":
    unittest.main()
