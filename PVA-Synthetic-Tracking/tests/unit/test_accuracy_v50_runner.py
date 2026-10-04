"""Generated-only runner integration tests; no archived experiment data is read."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_accuracy_v50_prediction as runner


def state(frame=100, track="bright:1", clip="0126", archived=True):
    return dict(clip=clip, segment=0, frame_index=frame, track_id=track,
        archive={"path": f"generated_{clip}_{frame}_{track}.npz", "sha256": "0" * 64}
            if archived else None,
        geometry={"geometry": {"prior_centers_xy": [[64., 64.]] * 8,
            "predicted_offset_xy": [0., 0.],
            "prior_frame_indices": list(range(frame - 8, frame))}},
        qualified_moving=False, reference_samples=[], arms={"old_evidence": "unchanged"})


def prior_arrays():
    return dict(history129=np.full((8, 129, 129), 100., dtype=np.float64),
        prior_centers_xy=np.full((8, 2), 64., dtype=np.float64),
        predicted_offset_xy=np.zeros(2, dtype=np.float64))


def record(row, errors=(1., 2.), available=True):
    arms = {arm: dict(max_score=max(errors), normalized_absolute_errors=list(errors),
                     residuals=list(errors),
                     mae_dn=float(np.mean(errors)),
                     rmse_dn=float(np.sqrt(np.mean(np.square(errors)))))
            for arm in runner.ARMS} if available else {}
    return dict(state_key=list(runner.cache.state_key(row)), clip=row["clip"],
        segment=row["segment"], frame_index=row["frame_index"], forecast_available=True,
        measurement=dict(available=available,
                         reasons=[] if available else ["synthetic_missing_guard"], arms=arms))


def prediction(points=2, scale=1., center=100., available=True):
    return dict(available=available, arms={arm: dict(
        prediction=[center] * points, scale=[scale] * points) for arm in runner.ARMS})


def calibration_for(clip="0126", primary_available=True, q=2.):
    return {"policies": {policy: [dict(clip=clip, segment=0, arms={
        arm: dict(available=primary_available if policy == runner.POLICIES[0] else True,
                  q=q if primary_available or policy != runner.POLICIES[0] else None)
        for arm in runner.ARMS})] for policy in runner.POLICIES}}


class V50RunnerTests(unittest.TestCase):
    def write_archive(self, directory, current=None, **overrides):
        row = state()
        arrays = prior_arrays()
        arrays["current129"] = (np.full((129, 129), 101., dtype=np.float64)
                                if current is None else current)
        arrays.update(overrides)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        path = Path(directory).resolve() / "generated_packet.npz"
        path.write_bytes(payload)
        row["archive"]["sha256"] = hashlib.sha256(payload).hexdigest()
        return path, row

    def test_prior_loader_does_not_decode_pickle_only_current_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory, current=np.array([object()], dtype=object))
            arrays = runner.load_prior_arrays(path, row)
            self.assertEqual(set(arrays), {"history129", "prior_centers_xy", "predicted_offset_xy"})
            self.assertTrue(all(not a.flags.writeable for a in arrays.values()))
            with np.load(path, allow_pickle=False) as archive:
                with self.assertRaises(ValueError):
                    archive["current129"]

    def test_prior_loader_hashes_before_numpy_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory)
            row["archive"]["sha256"] = "f" * 64
            with mock.patch.object(runner.np, "load") as decode:
                with self.assertRaisesRegex(ValueError, "changed before prior decode"):
                    runner.load_prior_arrays(path, row)
            decode.assert_not_called()

    def test_prior_loader_rejects_extra_members_and_metadata_geometry_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory, extra=np.zeros(1))
            with self.assertRaisesRegex(ValueError, "members"):
                runner.load_prior_arrays(path, row)
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory)
            row["geometry"]["geometry"]["predicted_offset_xy"] = [1., 0.]
            with self.assertRaisesRegex(ValueError, "geometry changed"):
                runner.load_prior_arrays(path, row)

    def test_prior_loader_rejects_bad_prior_shape_and_dtype(self):
        for history in (np.zeros((7, 129, 129)), np.zeros((8, 129, 129), dtype=np.float32)):
            with self.subTest(shape=history.shape, dtype=history.dtype):
                with tempfile.TemporaryDirectory() as directory:
                    path, row = self.write_archive(directory, history129=history)
                    with self.assertRaisesRegex(ValueError, "Invalid prior"):
                        runner.load_prior_arrays(path, row)

    def test_prior_loader_rejects_partial_centers_and_nonfinite_offset(self):
        for point in ([64., float("nan")], [float("inf"), float("inf")]):
            with tempfile.TemporaryDirectory() as directory:
                centers = np.array([point] * 8, dtype=np.float64)
                path, row = self.write_archive(directory, prior_centers_xy=centers)
                row["geometry"]["geometry"]["prior_centers_xy"] = [point] * 8
                with self.assertRaisesRegex(ValueError, "geometry coordinates"):
                    runner.load_prior_arrays(path, row)
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory, predicted_offset_xy=np.array([np.inf, 0.]))
            row["geometry"]["geometry"]["predicted_offset_xy"] = [np.inf, 0.]
            with self.assertRaisesRegex(ValueError, "geometry coordinates"):
                runner.load_prior_arrays(path, row)

    def test_current_loader_is_lazy_about_priors_and_does_not_scan_core_values(self):
        current = np.full((129, 129), 101., dtype=np.float64)
        current[64, 64] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory, current=current,
                                          history129=np.array([object()], dtype=object))
            observed = runner.load_current_array(path, row)
            self.assertFalse(observed.flags.writeable)
            self.assertTrue(np.isinf(observed[64, 64]))
            frozen = runner.forecast(prior_arrays()["history129"], [[64., 64.]] * 8)
            self.assertTrue(runner.measure_current(observed, frozen)["available"])

    def test_prior_loader_does_not_scan_prior_core_values(self):
        history = prior_arrays()["history129"]
        history[:, 64, 64] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path, row = self.write_archive(directory, history129=history)
            loaded = runner.load_prior_arrays(path, row)
            frozen = runner.forecast(loaded["history129"], [[64., 64.]] * 8)
            self.assertTrue(frozen["available"])

    def test_all_forecasts_are_frozen_before_any_current_packet_is_decoded(self):
        rows = [state(frame) for frame in (100, 109, 118)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()), \
                 mock.patch.object(runner, "load_current_array") as current:
                predictions, digest = runner.build_forecasts(rows, output, "calibration")
                current.assert_not_called()
                self.assertEqual(len(predictions), 3)
                frozen = runner.cache.read_json(output / "calibration_forecasts_frozen.json")
                self.assertTrue(frozen["completed"])
                self.assertFalse(frozen["current129_members_decoded"])
                self.assertEqual(frozen["forecasts_sha256"], digest)
                self.assertEqual(frozen["packet_count"], len(rows))

                def decode_after_freeze(*args):
                    self.assertTrue((output / "calibration_forecasts_frozen.json").is_file())
                    self.assertEqual(runner.cache.sha(output / "calibration_forecasts.jsonl"), digest)
                    return np.full((129, 129), 101.)

                current.side_effect = decode_after_freeze
                records = runner.score_forecasts(rows, predictions, output, "calibration", digest)
                self.assertEqual(current.call_count, 3)
                self.assertTrue(all(r["measurement"]["available"] for r in records))

    def test_changed_forecast_file_stops_before_response_access(self):
        rows = [state()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()), \
                 mock.patch.object(runner, "load_current_array") as current:
                predictions, digest = runner.build_forecasts(rows, output, "evaluation")
                with (output / "evaluation_forecasts.jsonl").open("a") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "Forecast file changed"):
                    runner.score_forecasts(rows, predictions, output, "evaluation", digest)
                current.assert_not_called()

    def test_replaced_valid_in_memory_forecast_is_not_the_saved_forecast(self):
        rows = [state()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()), \
                 mock.patch.object(runner, "load_current_array") as current:
                predictions, digest = runner.build_forecasts(rows, output, "calibration")
                predictions[runner.cache.state_key(rows[0])] = runner.forecast(
                    np.full((8, 129, 129), 200.), [[64., 64.]] * 8)
                with self.assertRaisesRegex(ValueError, "In-memory forecast differs"):
                    runner.score_forecasts(rows, predictions, output, "calibration", digest)
                current.assert_not_called()

    def test_missing_saved_forecast_membership_rejected_before_response_access(self):
        rows = [state(), state(109)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()), \
                 mock.patch.object(runner, "load_current_array") as current:
                predictions, digest = runner.build_forecasts(rows, output, "calibration")
                predictions.pop(runner.cache.state_key(rows[-1]))
                with self.assertRaisesRegex(ValueError, "membership differs"):
                    runner.score_forecasts(rows, predictions, output, "calibration", digest)
                current.assert_not_called()

    def test_forecast_response_mutation_rejected(self):
        rows = [state()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()), \
                 mock.patch.object(runner, "load_current_array", return_value=np.zeros((129, 129))):
                predictions, digest = runner.build_forecasts(rows, output, "calibration")

                def bad_measure(current, frozen):
                    frozen["metadata"]["unexpected_response_change"] = True
                    return {"available": False}

                with mock.patch.object(runner, "measure_current", side_effect=bad_measure):
                    with self.assertRaisesRegex(ValueError, "Response mutated"):
                        runner.score_forecasts(rows, predictions, output, "calibration", digest)

    def test_thirty_frame_calibration_uses_all_packet_maxima_and_missing_anchors(self):
        rows = [record(state(frame), errors=(float(frame),)) for frame in range(100, 130)]
        rows.append(record(state(100, track="bright:2"), errors=(500.,)))
        rows.append(record(state(109, track="bright:2"), available=False))
        frozen = deepcopy(rows)
        units = runner.frame_units(rows)
        by_frame = {u["frame_index"]: u for u in units}
        self.assertEqual(len(units), 30)
        self.assertEqual(len(by_frame[100]["state_keys"]), 2)
        self.assertEqual(by_frame[100]["scores"][runner.ARMS[0]], 500.)
        self.assertIsNone(by_frame[109]["scores"][runner.ARMS[0]])
        self.assertEqual(by_frame[109]["unavailable_packet_count"], 1)
        result = runner.calibrate(rows)
        primary = result["policies"][runner.POLICIES[0]][0]
        sensitivity = result["policies"][runner.POLICIES[1]][0]
        self.assertEqual(primary["selected_frame_indices"], [100, 109, 118, 127])
        for arm in runner.ARMS:
            self.assertEqual(primary["arms"][arm]["total_units"], 4)
            self.assertEqual(primary["arms"][arm]["missing_units"], 1)
            self.assertFalse(primary["arms"][arm]["available"])
            self.assertIsNone(primary["arms"][arm]["q"])
            self.assertEqual(sensitivity["arms"][arm]["total_units"], 30)
            self.assertEqual(sensitivity["arms"][arm]["finite_units"], 29)
            self.assertEqual(sensitivity["arms"][arm]["rank"], 27)
            self.assertEqual(sensitivity["arms"][arm]["q"], 128.)
        self.assertEqual(rows, frozen)

    def test_calibration_never_pools_clips_or_segments(self):
        records = [record(state(frame, clip=clip), errors=(1.,))
                   for clip in ("0029", "0126") for frame in range(100, 109)]
        extra = deepcopy(records[-1])
        extra["segment"] = 1
        extra["state_key"][2] = 1
        records.append(extra)
        calibrated = runner.calibrate(records)
        sensitivity = calibrated["policies"][runner.POLICIES[1]]
        self.assertEqual([(e["clip"], e["segment"]) for e in sensitivity],
                         [("0029", 0), ("0126", 0), ("0126", 1)])
        self.assertEqual([e["arms"][runner.ARMS[0]]["finite_units"] for e in sensitivity],
                         [9, 9, 1])
        self.assertFalse(sensitivity[-1]["arms"][runner.ARMS[0]]["available"])

    def test_detector_and_reference_metadata_do_not_enter_forecasts_or_calibration(self):
        class Poison:
            def __repr__(self):
                raise AssertionError("Unrelated detector/reference data inspected")

        rows = [state()]
        rows[0]["arms"] = Poison()
        rows[0]["reference_samples"] = Poison()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            with mock.patch.object(runner.cache, "packet_path", return_value=output / "synthetic"), \
                 mock.patch.object(runner, "load_prior_arrays", side_effect=lambda *args: prior_arrays()):
                result, _ = runner.build_forecasts(rows, output, "calibration")
            self.assertTrue(result[runner.cache.state_key(rows[0])]["available"])
        records = [record(state(frame), errors=(1.,)) for frame in range(100, 130)]
        for item in records:
            item["reference_samples"] = Poison()
            item["old_detector_evidence"] = Poison()
        result = runner.calibrate(records)
        self.assertTrue(result["policies"][runner.POLICIES[1]][0]["arms"][runner.ARMS[0]]["available"])

    def test_interval_policy_does_not_fallback_and_all_arms_remain_reported(self):
        row = state()
        measured = record(row, errors=(0.5, 3.))
        measured["measurement"]["arms"][runner.ARMS[0]]["normalized_absolute_errors"] = [0.5, 0.5]
        predictions = {runner.cache.state_key(row): prediction()}
        result = runner.interval_records([measured], predictions,
                                        calibration_for(primary_available=False))[0]
        self.assertEqual(set(result["policies"]), set(runner.POLICIES))
        self.assertEqual(set(result["policies"][runner.POLICIES[1]]), set(runner.ARMS))
        for arm in runner.ARMS:
            primary = result["policies"][runner.POLICIES[0]][arm]
            self.assertFalse(primary["available"])
            self.assertEqual(primary["reason"], "calibration_unavailable")
        sensitivity = result["policies"][runner.POLICIES[1]]
        self.assertTrue(sensitivity[runner.ARMS[0]]["whole_packet_covered"])
        self.assertFalse(sensitivity[runner.ARMS[1]]["whole_packet_covered"])
        self.assertFalse(sensitivity[runner.ARMS[2]]["whole_packet_covered"])

    def test_intervals_are_inclusive_and_not_clipped_to_eight_bit_range(self):
        row = state()
        predictions = {runner.cache.state_key(row): prediction(points=2, scale=100., center=200.)}
        result = runner.interval_records([record(row, errors=(2., 2.))], predictions,
                                        calibration_for())[0]
        for policy in runner.POLICIES:
            for interval in result["policies"][policy].values():
                self.assertTrue(interval["whole_packet_covered"])
                self.assertEqual(interval["half_width_values"], [200., 200.])
                self.assertEqual(interval["half_width_dn"]["max"], 200.)
                self.assertEqual(interval["full_8bit_range_included_points"], 2)

    def test_interval_missing_forecast_current_and_calibration_are_distinct(self):
        row = state()
        for forecast_ok, current_ok, primary_ok, expected in (
                (False, False, False, "forecast_unavailable"),
                (True, False, False, "current_support_unavailable"),
                (True, True, False, "calibration_unavailable")):
            with self.subTest(expected=expected):
                result = runner.interval_records([record(row, available=current_ok)],
                    {runner.cache.state_key(row): prediction(available=forecast_ok)},
                    calibration_for(primary_available=primary_ok))[0]
                self.assertEqual(result["policies"][runner.POLICIES[0]][runner.ARMS[0]]["reason"], expected)

    def test_frame_packet_and_point_denominators_are_separate(self):
        rows = [state(100, "bright:1"), state(100, "bright:2"),
                state(101, "bright:1"), state(101, "bright:2"), state(102)]
        records = [record(rows[0], errors=(1., 1.)), record(rows[1], errors=(1., 3.)),
                   record(rows[2], errors=(1.,)), record(rows[3], available=False),
                   record(rows[4], available=False)]
        predictions = {runner.cache.state_key(row): prediction(points=1 if i == 2 else 2)
                       for i, row in enumerate(rows)}
        intervals = runner.interval_records(records, predictions, calibration_for())
        report = runner.summarize_intervals(intervals, runner.POLICIES[0], runner.ARMS[0])
        self.assertEqual(report["archived_packets"], 5)
        self.assertEqual(report["interval_available_packets"], 3)
        self.assertEqual(report["covered_packets"], 2)
        self.assertEqual(report["conditional_packet_coverage"], 2 / 3)
        self.assertEqual(report["guard_state_point_pairs"], 5)
        self.assertEqual(report["covered_guard_state_point_pairs"], 4)
        self.assertEqual(report["conditional_point_coverage"], 4 / 5)
        self.assertEqual(report["archived_frames"], 3)
        self.assertEqual(report["interval_available_whole_frames"], 1)
        self.assertEqual(report["covered_whole_frames"], 0)
        self.assertEqual(report["conditional_whole_frame_coverage"], 0.)
        self.assertEqual(report["half_width_dn"]["count"], 5)
        self.assertEqual(report["unavailable_reasons"], {"current_support_unavailable": 2})
        metrics = runner.metrics(records, intervals)["combined"]
        self.assertEqual(metrics["archived_packets"], 5)
        self.assertEqual(metrics["scorable_packets"], 3)
        self.assertEqual(metrics["complete_scorable_frames"], 1)
        self.assertEqual(metrics["arms"][runner.ARMS[0]]["packet_mae_dn"]["count"], 3)
        self.assertEqual(metrics["arms"][runner.ARMS[0]]["complete_frame_macro_mae_dn"]["mean"], 1.5)
        self.assertEqual(metrics["arms"][runner.ARMS[0]]["packet_maximum_absolute_error_dn"]["max"], 3.)
        self.assertEqual(metrics["arms"][runner.ARMS[0]]["packet_maximum_normalized_error"]["max"], 3.)

    def test_all_unavailable_coverage_is_unknown_not_zero(self):
        row = state()
        records = [record(row, available=False)]
        intervals = runner.interval_records(records, {runner.cache.state_key(row): prediction()}, calibration_for())
        summary = runner.summarize_intervals(intervals, runner.POLICIES[0], runner.ARMS[0])
        for key in ("conditional_point_coverage", "conditional_packet_coverage", "conditional_whole_frame_coverage"):
            self.assertIsNone(summary[key])
        self.assertEqual(summary["archived_packets"], 1)
        self.assertEqual(summary["interval_available_packets"], 0)

    def test_original_references_and_unassigned_216_stay_unchanged(self):
        original = dict(samples=[
            dict(sample_index=0, original={"clip_id": "0126", "frame_index": 216},
                 original_strict_assigned_identity=None, measured_alternatives=["do not select"]),
            dict(sample_index=1, original={"clip_id": "0126", "frame_index": 150},
                 original_strict_assigned_identity="0/bright:1", measured_alternatives=[]),
        ], other_evidence={"frozen": True})
        snapshot = deepcopy(original)
        scope = {"cutoffs": [dict(clip="0126", segment=0, cutoff_frame_index=161)]}
        result = runner.reference_context(original, scope, [
            dict(state_key=["0126", 150, 0, "bright:1"], v50_status="background_measured")])
        missed = result["samples"][0]["v50_background_context"]
        self.assertEqual(missed["partition"], "evaluation")
        self.assertTrue(missed["original_unassigned_stays_unassigned"])
        self.assertIsNone(missed["original_assigned_state_key"])
        self.assertIsNone(missed["original_assigned_state_status"])
        self.assertEqual(result["samples"][1]["v50_background_context"]["partition"], "calibration")
        for sample in result["samples"]:
            del sample["v50_background_context"]
        self.assertEqual(result, snapshot)
        self.assertEqual(original, snapshot)

    def test_jsonl_refuses_overwrite_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.jsonl"
            runner.write_jsonl(path, [{"value": np.array([1.])}])
            self.assertEqual(json.loads(path.read_text()), {"value": [1.]})
            with self.assertRaises(FileExistsError):
                runner.write_jsonl(path, [])
            with self.assertRaises(ValueError):
                runner.write_jsonl(Path(directory) / "invalid.jsonl", [{"value": float("nan")}])

    def test_nine_frame_bins_retain_all_unknown_states_without_coverage(self):
        states = [dict(state_key=["0126", 216, 0, "bright:1"], partition="evaluation", v50_status="history_unknown"),
                  dict(state_key=["0126", 216, 0, "bright:2"], partition="evaluation", v50_status="history_unknown"),
                  dict(state_key=["0126", 218, 0, "bright:3"], partition="evaluation", v50_status="history_unknown"),
                  dict(state_key=["0126", 100, 0, "bright:4"], partition="calibration", v50_status="history_unknown")]
        report = runner.nine_frame_bins(states, [])
        self.assertEqual(len(report["bins"]), 1)
        item = report["bins"][0]
        self.assertEqual(item["nine_frame_bin"], 24)
        self.assertEqual(item["selected_states"], 3)
        self.assertEqual(item["selected_response_frames"], 2)
        self.assertEqual(item["state_status_counts"], {"history_unknown": 3})
        self.assertEqual(item["response_frames_without_archives"], [216, 218])
        self.assertEqual(item["frames_with_history_unknown_states"], [216, 218])
        for policy in runner.POLICIES:
            for arm in runner.ARMS:
                summary = item["policies"][policy][arm]
                self.assertEqual(summary["archived_packets"], 0)
                self.assertIsNone(summary["conditional_whole_frame_coverage"])

    def test_full_orchestration_freezes_calibration_before_later_forecasts(self):
        # Full metadata denominators are cheap; no actual image archives are
        # used. Forecast/measurement helpers are mocked only for this ordering
        # test; their actual numerical integration is exercised above.
        rows = [state(frame) for frame in range(100, 290)]
        rows += [state(frame, f"bright:{track}") for frame in range(290, 298) for track in (1, 2)]
        rows += [state(frame) for frame in range(298, 480)]
        rows += [state(frame, "bright:2") for frame in range(298, 419)]
        rows += [state(100 + i % 190, f"bright:{1000+i}", archived=False) for i in range(232)]
        rows += [state(298 + i % 182, f"bright:{2000+i}", archived=False) for i in range(470)]
        scope = runner.split_scope(rows)
        self.assertEqual(scope["counts"]["calibration"]["archived_states"], 190)
        self.assertEqual(scope["counts"]["evaluation"]["archived_states"], 303)
        refs = dict(samples=[dict(sample_index=i,
            original={"clip_id": "0126", "frame_index": 216+i if i < 3 else 298},
            original_strict_assigned_identity=None if i < 3 else "0/bright:1",
            measured_alternatives=[]) for i in range(355)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, output_root = root / "baseline", root / "v50"
            base.mkdir(); (base / "inputs").mkdir()
            packet_paths = {}
            for row in rows:
                if row["archive"] is not None:
                    path = base / "inputs" / row["archive"]["path"]
                    path.write_bytes(b"generated fixture: not an image archive")
                    row["archive"]["sha256"] = runner.cache.sha(path)
                    packet_paths[runner.cache.state_key(row)] = path
            runner.write_jsonl(base / "states.jsonl", rows)
            runner.cache.write_json(base / "selected_ledger.json",
                {"states": [{k: v for k, v in row.items() if k != "arms"} for row in rows]})
            runner.cache.write_json(base / "reference_evidence.json", refs)
            paths = [base / name for name in ("states.jsonl", "selected_ledger.json", "reference_evidence.json")]
            bindings = {str(path): runner.cache.sha(path) for path in paths + list(packet_paths.values())}
            runner.cache.write_json(base / "completion_receipt.json", {"files_sha256": bindings})
            receipt_hash = runner.cache.sha(base / "completion_receipt.json")
            events = []
            saved_calibration = []

            def get_packet_path(row):
                self.assertNotIn(row["frame_index"], range(290, 298))
                return packet_paths[runner.cache.state_key(row)]

            def build(subset, output, partition):
                self.assertTrue((output / "freeze.json").is_file())
                if partition == "calibration":
                    self.assertFalse((output / "calibration.json").exists())
                else:
                    self.assertTrue((output / "calibration.json").is_file())
                    self.assertTrue((output / "calibration_measurements.jsonl").is_file())
                    saved_calibration.append(runner.cache.sha(output / "calibration.json"))
                values = {runner.cache.state_key(row): prediction() for row in subset}
                path = output / (partition + "_forecasts.jsonl")
                runner.write_jsonl(path, [dict(state_key=list(key), forecast=value) for key, value in values.items()])
                digest = runner.cache.sha(path)
                runner.cache.write_json(output / (partition + "_forecasts_frozen.json"),
                                        {"completed": True, "forecasts_sha256": digest})
                events.append(partition + "_forecasts_frozen")
                return values, digest

            def score(subset, predictions, output, partition, digest):
                self.assertTrue((output / (partition + "_forecasts_frozen.json")).is_file())
                self.assertEqual(digest, runner.cache.sha(output / (partition + "_forecasts.jsonl")))
                if partition == "evaluation":
                    self.assertEqual(saved_calibration[0], runner.cache.sha(output / "calibration.json"))
                result = [record(row) for row in subset]
                runner.write_jsonl(output / (partition + "_measurements.jsonl"), result)
                events.append(partition + "_responses")
                return result

            with mock.patch.object(runner, "BASE", base), mock.patch.object(runner, "OUTPUT", output_root), \
                 mock.patch.object(runner, "RECEIPT_SHA", receipt_hash), \
                 mock.patch.object(runner, "source_dependencies", return_value=[]), \
                 mock.patch.object(runner.cache, "packet_path", side_effect=get_packet_path), \
                 mock.patch.object(runner, "build_forecasts", side_effect=build), \
                 mock.patch.object(runner, "score_forecasts", side_effect=score):
                summary = runner.run(output_root / "generated")
            self.assertEqual(events, ["calibration_forecasts_frozen", "calibration_responses",
                                      "evaluation_forecasts_frozen", "evaluation_responses"])
            self.assertEqual(summary["opened_packets"], 493)
            self.assertEqual(summary["embargo_archives_unread"], 16)
            self.assertEqual(summary["partitions"]["embargo"], {"embargo_not_scored": 16})
            self.assertEqual(summary["partitions"]["calibration"]["history_unknown"], 232)
            self.assertEqual(summary["partitions"]["evaluation"]["history_unknown"], 470)
            self.assertEqual(len(summary["original_unassigned_references"]), 3)
            self.assertFalse(summary["production_changed"])


if __name__ == "__main__":
    unittest.main()
