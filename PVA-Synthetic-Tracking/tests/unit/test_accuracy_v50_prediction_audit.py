"""Generated-only independent V50 audit arithmetic and fail-closed checks."""
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_accuracy_v50_prediction as audit
from accuracy_v50_scope import split_scope


def row(frame, archived=True, track="bright:1", clip="0029"):
    archive = None if not archived else dict(
        path=f'inputs/{clip}_{frame:04d}_0_{track.replace(":", "_")}.npz', sha256="a"*64)
    return dict(clip=clip, frame_index=frame, segment=0, track_id=track, archive=archive,
                geometry=dict(geometry=dict(prior_frame_indices=list(range(frame-8, frame)))))


def generated_forecast():
    history = np.stack([np.full((129, 129), float(x)) for x in range(8)])
    result = audit.reconstruct_forecast(history, [None]*8)
    return history, result


def record(frame, measurement, track="bright:1"):
    return dict(state_key=["0029", frame, 0, track], clip="0029", segment=0,
                frame_index=frame, forecast_available=True, measurement=measurement)


def measurement(score, missing=False):
    return dict(available=not missing, reasons=[] if not missing else ["test_missing"],
        arms={} if missing else {a: dict(max_score=float(score), mae_dn=float(score), rmse_dn=float(score),
                                        residuals=[float(score)], normalized_absolute_errors=[float(score)])
                               for a in audit.ARMS})


class PredictionAuditTests(unittest.TestCase):
    def test_no_execute_never_reads_files_or_packets(self):
        with mock.patch.object(audit, "read_json") as read, mock.patch.object(audit, "digest") as digest:
            with self.assertRaisesRegex(ValueError, "execution"):
                audit.audit("/tmp/never-read", "/tmp/never-write")
            read.assert_not_called(); digest.assert_not_called()

    def test_no_producer_module_imports(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(m and (m.startswith("accuracy_v") or m.startswith("run_accuracy_v")) for m in modules))

    def test_scalar_order_statistics_match_known_medians_and_mad(self):
        _, f = generated_forecast()
        self.assertTrue(f["available"])
        self.assertGreater(f["used_count"], 0)
        for a in audit.ARMS[:2]:
            self.assertEqual(set(f["arms"][a]["prediction"]), {3.5})
        self.assertEqual(set(f["arms"][audit.ARMS[2]]["prediction"]), {6.0})
        self.assertEqual(set(f["arms"][audit.ARMS[0]]["scale"]), {1.0})
        self.assertEqual(set(f["arms"][audit.ARMS[1]]["scale"]), {2.0})
        self.assertEqual(f["arms"][audit.ARMS[1]]["scale"], f["arms"][audit.ARMS[2]]["scale"])

    def test_prior_core_cannot_affect_forecast(self):
        h, f = generated_forecast()
        h[:, 50:79, 50:79] = np.nan
        self.assertEqual(audit.reconstruct_forecast(h, [None]*8), f)

    def test_generated_random_forecasts_and_measurements_agree_with_producer(self):
        # The independent auditor imports no producer; this generated-only
        # differential test checks both separate implementations, never real data.
        from accuracy_v50_predictive_background import forecast, measure_current
        rng = np.random.default_rng(50)
        for _ in range(5):
            history = rng.uniform(0, 255, (8, 129, 129))
            centers = [[float(rng.uniform(0, 128)), float(rng.uniform(0, 128))]]+[None]*7
            expected = audit.reconstruct_forecast(history, centers)
            actual = forecast(history, centers)
            for name, value in expected.items():
                def lists(x):
                    if isinstance(x, np.ndarray): return x.tolist()
                    if isinstance(x, dict): return {k: lists(v) for k, v in x.items()}
                    return x
                audit.same(lists(actual[name]), value, name)
            current = rng.uniform(0, 255, (129, 129))
            measured = measure_current(current, actual)
            reconstructed = audit.reconstruct_measurement(current, expected, actual)
            for name, value in reconstructed.items():
                audit.same(lists(measured[name]), value, name)

    def test_guard_support_is_triplet_union_with_center_and_nonfinite_exclusions(self):
        h, _ = generated_forecast(); h[:, 8, 8] = np.nan
        f = audit.reconstruct_forecast(h, [[24., 8.]]+[None]*7)
        used = {tuple(p) for p in f["used_points_xy"]}
        union = {tuple(p) for t in f["stencils"] for p in t["pixels_xy"]}
        self.assertEqual(used, union)
        self.assertNotIn((8, 8), used)
        self.assertTrue(all(max(abs(x-24), abs(y-8)) > 12 for x, y in used))
        self.assertEqual(f["prior_rejection_counts_nonexclusive"]["nonfinite_prior_history"], 1)

    def test_empty_support_is_unknown_not_zero_error(self):
        h = np.full((8, 129, 129), np.nan)
        f = audit.reconstruct_forecast(h, [None]*8)
        self.assertFalse(f["available"])
        self.assertEqual(f["arms"], {})
        m = audit.reconstruct_measurement(np.zeros((129, 129)), f, dict(forecast_sha256="x"))
        self.assertFalse(m["available"])
        self.assertEqual(m["reasons"], ["prior_forecast_unavailable"])

    def test_measurement_reads_only_used_guard_and_recomputes_errors(self):
        _, f = generated_forecast()
        current = np.full((129, 129), np.nan)
        for x, y in f["used_points_xy"]:
            current[y, x] = 7
        m = audit.reconstruct_measurement(current, f, dict(forecast_sha256="x"))
        self.assertTrue(m["available"])
        self.assertEqual(m["arms"][audit.ARMS[0]]["mae_dn"], 3.5)
        self.assertEqual(m["arms"][audit.ARMS[1]]["max_score"], 1.75)
        self.assertEqual(m["arms"][audit.ARMS[2]]["max_score"], .5)
        x, y = f["used_points_xy"][0]; current[y, x] = np.nan
        missing = audit.reconstruct_measurement(current, f, dict(forecast_sha256="x"))
        self.assertFalse(missing["available"])
        self.assertEqual(missing["current_nonfinite_used_point_count"], 1)

    def test_same_rejects_structural_and_arithmetic_tampering(self):
        for a, b in ((True, 1), ([1], [1, 2]), ({"x": 1}, {"y": 1}), (1.1, 1.0), (math.nan, 1.0)):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                audit.same(a, b)
        audit.same(1.0+1e-14, 1.0)

    def test_independent_scope_matches_metadata_producer_and_preserves_unknowns(self):
        rows = [row(f, archived=f%3 != 0) for f in range(8, 55)]
        saved = split_scope(rows)
        cuts, parts, counts = audit.validate_scope(saved, rows)
        self.assertEqual(cuts, {("0029", 0): 31})
        self.assertEqual(sum(c["states"] for c in counts.values()), len(rows))
        self.assertEqual(parts[("0029", 39, 0, "bright:1")], "embargo")
        self.assertEqual(parts[("0029", 40, 0, "bright:1")], "evaluation")
        changed = deepcopy(saved); changed["partitions"]["evaluation"].pop()
        with self.assertRaises(ValueError):
            audit.validate_scope(changed, rows)

    def test_packet_allowlist_excludes_embargo_and_rejects_extra_before_file_read(self):
        rows = [row(8), row(9), row(10), row(18), row(30)]
        _, parts, _ = audit.metadata_scope(rows)
        all_paths = {str(audit.CACHE/r["archive"]["path"]): "a"*64 for r in rows}
        allowed = {str(audit.CACHE/r["archive"]["path"]): "a"*64 for r in rows
                   if parts[audit.state_key(r)] != "embargo"}
        freeze = {"packet_files_sha256": allowed}; old = {"files_sha256": all_paths}
        with mock.patch.object(audit, "exact_file") as opened:
            self.assertEqual(audit.validate_packet_bindings(rows, parts, freeze, old, len(allowed)), allowed)
            bad = deepcopy(freeze); bad["packet_files_sha256"] = all_paths
            with self.assertRaises(ValueError):
                audit.validate_packet_bindings(rows, parts, bad, old, len(allowed))
            opened.assert_not_called()

    def test_packet_literal_path_tampering_is_rejected(self):
        rows = [row(8), row(30)]
        _, parts, _ = audit.metadata_scope(rows)
        old = {"files_sha256": {str(audit.CACHE/r["archive"]["path"]): "a"*64 for r in rows}}
        rows[0]["archive"]["path"] = "../other.npz"
        with self.assertRaisesRegex(ValueError, "literal"):
            audit.validate_packet_bindings(rows, parts, {"packet_files_sha256": {}}, old, 2)

    def test_frame_maximum_includes_all_tracks_and_missing_packet_poisoning(self):
        rs = [record(8, measurement(1)), record(8, measurement(9), "bright:2"), record(9, measurement(2))]
        units = audit.frame_units(rs)
        self.assertEqual(units[0]["scores"][audit.ARMS[0]], 9.)
        rs[1]["measurement"] = measurement(0, missing=True)
        units = audit.frame_units(rs)
        self.assertIsNone(units[0]["scores"][audit.ARMS[0]])
        self.assertEqual(units[0]["unavailable_packet_count"], 1)

    def test_quantile_coarse_rank_missing_and_no_rank_clamping(self):
        self.assertEqual(audit.quantile(list(range(12)))["rank"], 12)
        self.assertEqual(audit.quantile(list(range(10)))["q"], 9)
        self.assertFalse(audit.quantile(list(range(8)))["available"])
        self.assertIsNone(audit.quantile(list(range(8)))["q"])
        q = audit.quantile([None]+[2.]*9)
        self.assertEqual((q["rank"], q["finite_units"], q["missing_units"], q["q"]), (9, 9, 1, 2.))

    def test_missing_anchor_is_not_replaced_by_a_finite_nonanchor(self):
        records = [record(f, measurement(float(f), missing=f == 8)) for f in range(8, 110)]
        policies = audit.calibration_policies(records)
        main = policies[audit.POLICIES[0]][0]
        self.assertEqual(main["selected_frame_indices"][:3], [8, 17, 26])
        self.assertIsNone(main["frames"][0]["scores"][audit.ARMS[0]])
        self.assertEqual(main["arms"][audit.ARMS[0]]["missing_units"], 1)

    def test_interval_coverage_is_inclusive_and_width_is_not_clipped(self):
        key = ("0029", 100, 0, "bright:1")
        f = dict(available=True, arms={a: dict(prediction=[127.5], scale=[100.]) for a in audit.ARMS})
        rs = [record(100, measurement(2.))]
        policies = {p: [dict(clip="0029", segment=0, arms={a: dict(available=True, q=2.) for a in audit.ARMS})]
                    for p in audit.POLICIES}
        items = audit.reconstruct_intervals(rs, {key: f}, policies)
        arm = items[0]["policies"][audit.POLICIES[0]][audit.ARMS[0]]
        self.assertTrue(arm["whole_packet_covered"])
        self.assertEqual(arm["half_width_values"], [200.])
        self.assertEqual(arm["full_8bit_range_included_points"], 1)
        summary = audit.interval_summary(items, audit.POLICIES[0], audit.ARMS[0])
        self.assertEqual(summary["covered_whole_frames"], 1)

    def test_interval_unavailability_reasons_keep_precedence(self):
        key = ("0029", 100, 0, "bright:1")
        policies = {p: [dict(clip="0029", segment=0, arms={a: dict(available=False, q=None) for a in audit.ARMS})]
                    for p in audit.POLICIES}
        for forecast_ok, current_ok, expected in ((False, False, "forecast_unavailable"),
                (True, False, "current_support_unavailable"), (True, True, "calibration_unavailable")):
            f = dict(available=forecast_ok)
            records = [record(100, measurement(0, missing=not current_ok))]
            item = audit.reconstruct_intervals(records, {key: f}, policies)[0]
            self.assertEqual(item["policies"][audit.POLICIES[0]][audit.ARMS[0]]["reason"], expected)

    def test_empty_interval_summary_preserves_zero_denominator(self):
        summary = audit.interval_summary([], audit.POLICIES[0], audit.ARMS[0])
        self.assertEqual(summary["archived_packets"], 0)
        self.assertIsNone(summary["conditional_whole_frame_coverage"])
        self.assertEqual(summary["half_width_dn"]["count"], 0)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
