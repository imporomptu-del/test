"""Generated metadata only: V50 split, grouped units, and causal boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from accuracy_v50_scope import partition_for, split_scope


def row(frame, *, clip="a", segment=0, track="bright:1", archived=True):
    result = {"clip": clip, "frame_index": frame, "segment": segment,
              "track_id": track, "archive": {} if archived else None}
    if archived:
        result["geometry"] = {"geometry": {"prior_frame_indices": list(range(frame-8, frame))}}
    return result


class RestrictedMetadata(dict):
    """Explode on accidental iteration/copy or access outside the allowlist."""
    def __getitem__(self, key):
        if key not in {"clip", "frame_index", "segment", "track_id", "archive", "geometry"}:
            raise AssertionError(f"forbidden field {key}")
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("must not iterate source metadata")

    def items(self):
        raise AssertionError("must not copy source metadata")


class UnreadableArchive:
    def __bool__(self):
        raise AssertionError("archive truthiness must not be inspected")

    def __getitem__(self, key):
        raise AssertionError("archive values must not be inspected")


class ScopeTests(unittest.TestCase):
    def test_lower_median_counts_unique_frames_not_rows_or_archives(self):
        rows = [row(f, archived=False) for f in (9, 10, 40, 70)]
        rows += [row(70, track=f"other:{i}") for i in range(30)]
        result = split_scope(rows)
        self.assertEqual(result["cutoffs"], [{"clip": "a", "segment": 0, "cutoff_frame_index": 10}])
        self.assertEqual(result["counts"]["calibration"]["states"], 2)

    def test_all_tracks_at_one_frame_share_partition(self):
        result = split_scope([row(f, track=t) for f in range(8, 30) for t in ("a", "b")])
        parts = {}
        for x in result["assignments"]:
            parts.setdefault(x["state_key"][1], set()).add(x["partition"])
        self.assertTrue(all(len(p) == 1 for p in parts.values()))

    def test_eight_frame_embargo_exact_boundaries_and_unselected_reference(self):
        cutoffs = [{"clip": "a", "segment": 0, "cutoff_frame_index": 100}]
        self.assertEqual(partition_for("a", 0, 100, cutoffs), "calibration")
        for f in range(101, 109):
            self.assertEqual(partition_for("a", 0, f, cutoffs), "embargo")
        self.assertEqual(partition_for("a", 0, 109, cutoffs), "evaluation")
        self.assertEqual(partition_for("a", 0, 216, cutoffs), "evaluation")

    def test_clips_and_segments_have_independent_cutoffs(self):
        rows = [row(f, clip=c, segment=s) for c, s, fs in
                (("a", 0, [8, 9, 10]), ("a", 1, [50, 60, 70]), ("b", 0, [80, 90])) for f in fs]
        result = split_scope(rows)
        self.assertEqual([r["cutoff_frame_index"] for r in result["cutoffs"]], [9, 60, 80])

    def test_unknown_history_rows_and_empty_archive_dict_are_preserved(self):
        rows = [row(8), row(9, archived=False), row(20, archived=False), row(30)]
        result = split_scope(rows)
        self.assertEqual(sum(v["states"] for v in result["counts"].values()), 4)
        self.assertEqual(sum(v["archived_states"] for v in result["counts"].values()), 2)
        self.assertEqual(sum(v["history_unknown_states"] for v in result["counts"].values()), 2)
        self.assertEqual(result["frames_without_archives"]["calibration"][0]["frame_index"], 9)

    def test_greedy_anchors_do_not_backfill_from_unknown_frames(self):
        rows = [row(f, archived=f not in (8, 17)) for f in range(8, 80)]
        result = split_scope(rows)
        frames = [x["frame_index"] for x in result["anchors"]["calibration"]]
        self.assertEqual(frames, [9, 18, 27, 36])
        self.assertGreater(len(result["calibration_all_archived_frames"]), len(frames))

    def test_anchors_include_all_archived_tracks_and_separate_unknowns(self):
        rows = [row(f) for f in (8, 17, 26, 50, 70)]
        rows += [row(8, track="b"), row(8, track="c", archived=False)]
        unit = split_scope(rows)["anchors"]["calibration"][0]
        self.assertEqual(unit["state_keys"], [["a", 8, 0, "b"], ["a", 8, 0, "bright:1"]])
        self.assertEqual(unit["history_unknown_state_keys"], [["a", 8, 0, "c"]])
        self.assertEqual(unit["input_span"], [0, 8])

    def test_evaluation_anchors_restart_after_embargo_and_all_eval_rows_survive(self):
        result = split_scope([row(f) for f in range(8, 80)])
        self.assertEqual(result["cutoffs"][0]["cutoff_frame_index"], 43)
        self.assertEqual([x["frame_index"] for x in result["anchors"]["evaluation"]], [52, 61, 70, 79])
        self.assertEqual(len(result["partitions"]["evaluation"]), 28)
        for unit in result["anchors"]["evaluation"]:
            self.assertGreater(unit["input_span"][0], 43)

    def test_deterministic_under_input_reordering_and_no_mutation(self):
        rows = [row(f, clip=c, track=t) for c in ("a", "b") for f in range(8, 31) for t in ("b", "a")]
        original = deepcopy(rows)
        baseline = split_scope(rows)
        self.assertEqual(rows, original)
        random.Random(47).shuffle(rows)
        self.assertEqual(split_scope(rows), baseline)
        json.dumps(baseline, allow_nan=False)

    def test_only_minimal_metadata_is_accessed(self):
        rows = [RestrictedMetadata(row(f)) for f in (8, 9, 20)]
        for r in rows:
            r["archive"] = UnreadableArchive()
            r["arms"] = object()
            r["reference_samples"] = object()
            r["actual_source_xy"] = object()
        result = split_scope(rows)
        self.assertEqual(sum(x["archived_states"] for x in result["counts"].values()), 3)

    def test_source_values_cannot_change_split(self):
        rows = [row(f) for f in range(8, 40)]
        baseline = split_scope(rows)
        for r in rows:
            r.update(actual_source_xy=[999, -999], arms={"fake": {"positive": True}},
                     qualified_moving=True, reference_samples=[123], grid_windows=["selected"])
        self.assertEqual(split_scope(rows), baseline)

    def test_duplicate_key_rejected_even_when_archive_differs(self):
        with self.assertRaisesRegex(ValueError, "duplicate state"):
            split_scope([row(8), row(8, archived=False)])

    def test_missing_required_metadata_rejected(self):
        for field in ("clip", "frame_index", "segment", "track_id", "archive"):
            r = row(8); del r[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                split_scope([r])

    def test_invalid_key_types_rejected(self):
        for field, value in (("frame_index", True), ("frame_index", 8.0), ("frame_index", -1),
                             ("segment", False), ("segment", -1), ("clip", ""),
                             ("clip", 29), ("track_id", ""), ("track_id", None)):
            r = row(8); r[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                split_scope([r])

    def test_archived_prior_sequence_must_be_complete_ordered_and_causal(self):
        for prior in (list(range(1, 9)), list(range(7)), list(reversed(range(8))),
                      [0, 1, 2, 3, 4, 5, 6, 6], [False, 1, 2, 3, 4, 5, 6, 7],
                      [0.0, 1, 2, 3, 4, 5, 6, 7], None, "01234567"):
            r = row(8); r["geometry"]["geometry"]["prior_frame_indices"] = prior
            with self.subTest(prior=prior), self.assertRaisesRegex(ValueError, "history"):
                split_scope([r])

    def test_archived_row_requires_history_but_unknown_does_not(self):
        r = row(8); del r["geometry"]
        with self.assertRaisesRegex(ValueError, "prior frame"):
            split_scope([r])
        result = split_scope([row(0, archived=False)])
        self.assertEqual(result["counts"]["calibration"]["history_unknown_states"], 1)

    def test_archived_frame_cannot_have_negative_source_history(self):
        with self.assertRaisesRegex(ValueError, "history"):
            split_scope([row(7)])

    def test_cutoff_map_supported_and_invalid_records_rejected(self):
        self.assertEqual(partition_for("a", 0, 30, {("a", 0): 10}), "evaluation")
        for cutoffs in ({"a": 10}, [{"clip": "a"}], [
                {"clip": "a", "segment": 0, "cutoff_frame_index": 10},
                {"clip": "a", "segment": 0, "cutoff_frame_index": 20}], "bad"):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                partition_for("a", 0, 8, cutoffs)
        with self.assertRaisesRegex(ValueError, "missing cutoff"):
            partition_for("b", 0, 8, {("a", 0): 10})

    def test_empty_scope_is_serializable_and_has_no_units(self):
        result = split_scope([])
        self.assertEqual(result["cutoffs"], [])
        self.assertEqual(result["anchors"], {"calibration": [], "evaluation": []})
        self.assertEqual(sum(x["states"] for x in result["counts"].values()), 0)
        json.dumps(result, allow_nan=False)

    def test_random_metadata_partition_and_nonoverlap_invariants(self):
        rng = random.Random(50)
        for _ in range(30):
            frames = sorted(rng.sample(range(8, 300), 30))
            rows = [row(f, track=str(t), archived=rng.choice((False, True)))
                    for f in frames for t in range(rng.randint(1, 4))]
            result = split_scope(rows)
            cut = frames[14]
            self.assertEqual(result["cutoffs"][0]["cutoff_frame_index"], cut)
            self.assertEqual(sum(x["states"] for x in result["counts"].values()), len(rows))
            for name, units in result["anchors"].items():
                for previous, current in zip(units, units[1:]):
                    self.assertLess(previous["input_span"][1], current["input_span"][0])
                if name == "evaluation":
                    self.assertTrue(all(u["input_span"][0] > cut for u in units))
            for key in result["partitions"]["evaluation"]:
                self.assertGreater(key[1]-8, cut)


if __name__ == "__main__":
    unittest.main()
