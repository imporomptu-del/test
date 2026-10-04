"""Generated-only checks for frozen custom AOT point-observation scoring.

No pilot annotations, journals, media, detector imports, or network are read.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_aot_pilot.py"
SPEC = importlib.util.spec_from_file_location("generated_aot_scorer", SCRIPT)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)
FLIGHT = "0123456789abcdef0123456789abcdef"
STAMP = 1573043646380340792


def annotation(oid="Airplane1", box=(10, 20, 6, 6), distance=650):
    return {"id": oid, "bb": list(box), "range": distance}


def manifest(labeled=None):
    labeled = {8: [annotation()]} if labeled is None else labeled
    frames = []
    for i in range(300):
        stamp = STAMP + i * 100_000_000 + (i % 3) * 1_000_000
        name = f"{stamp}{FLIGHT}.png"
        base = {"time": stamp, "blob": {"frame": i + 3}, "flight_id": FLIGHT, "img_name": name}
        entities = []
        for obj in labeled.get(i, []):
            entity = copy.deepcopy(base)
            entity.update(id=obj["id"], bb=obj["bb"], labels={"is_above_horizon": -1})
            if obj.get("range") is not None:
                entity["blob"]["range_distance_m"] = obj["range"]
            entities.append(entity)
        frames.append({"source_frame": i + 3, "timestamp_ns": str(stamp), "img_name": name,
                       "airborne_label_count": len(entities), "entities": entities or [base]})
    return {"part": "part1", "flight_id": FLIGHT, "bb_convention": "left,top,width,height", "frames": frames}


def track(tid="bright:1", xy=(13, 23), *, filtered=(1300, 1300), measured=True, qualified=True, segment=0):
    return {"track_id": tid, "segment": segment, "measured": measured,
            "qualified_moving": qualified, "measurement_source_xy": list(xy) if measured else None,
            "source_xy": list(filtered)}


def journal():
    return [{"frame_index": i, "timestamp_ns": i * 100_000_000, "segment": 0,
             "coverage": {"full_shape_hw": [2048, 2448], "configured_crop": None,
                          "native_pixel_sampling": True, "warmup": i < 8,
                          "detection_ready": i >= 8, "searchable_pixels": 2048 * 2448,
                          "unavailable_reason": "warmup" if i < 8 else None},
             "motion": {"backend": "pva", "reset": False, "pva_failure": False,
                        "status": "initial_reference" if i == 0 else "accepted", "accepted": True},
             "tracks": [], "candidates": []} for i in range(300)]


def result_stage(result, *, stage="all_measured", window="all_300", gate="primary_box"):
    return result["windows"][window]["gates"][gate][stage]


def run_metadata():
    cfg = {"input_bit_depth": 8, "warmup_frames": 8, "motion_backend": "pva",
           "state_update_backend": "cuda_resident", "spatial_filter_backend": "cuda_median5",
           "stabilization_execution": "cuda_cubic_resident", "frame_decode_execution": "prefetch_one"}
    launch = {"source_sha256": scorer.VIDEO_SHA256, "config_sha256": scorer.CONFIG_SHA256,
              "motion_config_sha256": scorer.MOTION_SHA256,
              "code_sha256": {"visible_baseline.py": scorer.BASELINE_SHA256},
              "annotations_supplied_to_detector": False, "configuration": cfg,
              "expected_frames": 300, "fps": 10, "max_frames": None,
              "source_probe": {"width": 2448, "height": 2048, "codec": "ffv1", "pixel_format": "gray",
                               "frame_rate": "10", "declared_frame_count": 300}}
    report = {"source_sha256": scorer.VIDEO_SHA256, "configuration": copy.deepcopy(cfg),
              "completed": True, "frames": 300, "full_clip": True,
              "frame_decode": {"decoded_frames": 300, "consumed_frames": 300, "dropped_frames": 0,
                               "worker_joined": True, "capture_released": True}}
    return launch, report


def execution_fixture():
    hashes = {key: str(i) * 64 for i, key in enumerate(("journal", "launch", "report", "preflight", "scoring_freeze", "script"), 1)}
    runtime = {"numpy": "1.26.1", "opencv": "4.10.0", "blas": [{"threads": 12}],
               "affinity": [0, 1], "thread_environment": {"OPENBLAS_NUM_THREADS": None},
               "clock_ticks": 100, "opencv_threads": 2}
    inputs = {"frozen_image_manifest.json": scorer.MANIFEST_SHA256,
              "pilot_gray8_ffv1_10fps.avi": scorer.VIDEO_SHA256, "scoring_freeze.json": hashes["scoring_freeze"]}
    adapters = {name: {"sha256": value, "path": "/tmp/generated/" + name + ".py"} for name, value in scorer.ADAPTERS.items()}
    receipt = {"schema": "seaqr.aot.frozen-baseline.v1", "passed": True, "error": None,
               "processed_frames": 300, "pixel_hashes_verified": 300,
               **{key: False for key in ("algorithm_changed", "detector_configuration_changed", "annotations_supplied_to_detector",
                                         "raw16_accessed", "private_camera_media_accessed", "clocks_changed")},
               **scorer.RUNTIME_IDENTITIES, "libraries": copy.deepcopy(scorer.LIBRARIES), "adapters": adapters,
               **{key + "_sha256": value for key, value in hashes.items()}, "input_sha256": inputs,
               "workspace": "/tmp/generated_aot", "gpu_fronts": [{"closed": True, "calls": 300, "device_calls": 300,
                                                                      "finish_calls": 300, "host_calls": 0}],
               "native_mask_calls": 0,
               "tracking": {"geometry_calls": 0, "geometry_fallbacks": 0, "batch_fallbacks": 0, "innovation_fallbacks": 0,
                            "batch_track_rows": 7, "innovation_tracks": 7, "batch_calls": 300, "exercised": True},
               "motion_instances": [{"failed": False, "closed": True}], "cleanup_errors": [],
               "motion_attempts": [{"frame": i, "error": None, "expected_unavailable": False} for i in range(1, 300)],
               "execution": {"policy": "reference", "mode": "reference", "frames": [{"frame": i} for i in range(300)]},
               "runtime_before": runtime, "runtime_after": copy.deepcopy(runtime),
               "vpi_version": "3.2.4", "remote_clocks_unchanged": True,
               "clock_policy_before": {"generated": "fixed"}, "clock_policy_after": {"generated": "fixed"}}
    preflight = {"schema": "seaqr.aot.frozen-baseline.v1.preflight", "passed": True, "detector_run": False,
                 "script_sha256": hashes["script"], "input_sha256": copy.deepcopy(inputs),
                 "workspace": receipt["workspace"], "decode": {"passed": True, "pixel_hashes_verified": 300}}
    for key in ("libraries", "adapters", "config_sha256", "motion_config_sha256", "v29_freeze_sha256", "reuse_method_sha256", "vpi_version"):
        preflight[key] = copy.deepcopy(receipt[key])
    return receipt, preflight, hashes


class AotScoringTests(unittest.TestCase):
    def test_only_stdlib_and_no_production_or_media_imports(self):
        tree = ast.parse(SCRIPT.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module.split(".")[0])
        self.assertFalse(set(modules) & {"tiny_target", "cv2", "numpy", "PIL", "urllib", "socket", "subprocess"})

    def test_measured_source_not_filtered_state_and_stage_separation(self):
        rows = journal()
        rows[8]["tracks"] = [track(qualified=False)]
        rows[8]["candidates"] = [{"source_xy": [13, 23], "polarity": "dark"}]
        original = copy.deepcopy(rows)
        result = scorer.score_rows(manifest(), rows)
        self.assertEqual(result_stage(result)["strata"]["all_labeled"]["matched_annotations"], 1)
        self.assertEqual(result_stage(result, stage="qualified_measured")["strata"]["all_labeled"]["matched_annotations"], 0)
        self.assertEqual(result_stage(result, stage="raw_candidates_diagnostic")["strata"]["all_labeled"]["matched_annotations"], 1)
        self.assertEqual(rows, original)

    def test_filtered_state_inside_label_cannot_replace_actual_measurement(self):
        rows = journal()
        rows[8]["tracks"] = [track(xy=(100, 100), filtered=(13, 23))]
        scored = result_stage(scorer.score_rows(manifest(), rows))
        self.assertEqual(scored["strata"]["all_labeled"]["matched_annotations"], 0)
        self.assertEqual(scored["counts"]["unmatched_outside_all_gt_gates"], 1)

    def test_coast_inside_label_never_counts_as_hit(self):
        rows = journal()
        rows[8]["tracks"] = [track(measured=False, filtered=(13, 23))]
        result = scorer.score_rows(manifest(), rows)
        self.assertEqual(result_stage(result)["counts"]["matched_annotations"], 0)
        self.assertEqual(result["windows"]["all_300"]["coasts_diagnostic_only"]["qualified_states"], 1)
        self.assertEqual(result["windows"]["all_300"]["coasts_diagnostic_only"]["counted_as_hits"], 0)

    def test_primary_box_closed_edges_and_fixed_three_pixel_sensitivity(self):
        for xy in [(10, 20), (16, 26), (7, 20), (19, 26), (6.999, 20), (19.001, 26)]:
            rows = journal()
            rows[8]["tracks"] = [track(xy=xy)]
            result = scorer.score_rows(manifest(), rows)
            with self.subTest(xy=xy):
                self.assertEqual(result_stage(result)["counts"]["matched_annotations"], int(xy in [(10, 20), (16, 26)]))
                self.assertEqual(result_stage(result, gate="sensitivity_box_plus_3px")["counts"]["matched_annotations"],
                                 int(xy not in [(6.999, 20), (19.001, 26)]))

    def test_ltwh_is_not_swapped_or_center_form(self):
        rows = journal()
        rows[8]["tracks"] = [track(xy=(102, 21))]
        result = scorer.score_rows(manifest({8: [annotation(box=(100, 20, 4, 2))]}), rows)
        self.assertEqual(result_stage(result)["counts"]["matched_annotations"], 1)

    def test_one_observation_cannot_cover_two_overlapping_labels(self):
        rows = journal()
        rows[8]["tracks"] = [track()]
        result = scorer.score_rows(manifest({8: [annotation(), annotation("Bird2")]}), rows)
        scored = result_stage(result)
        self.assertEqual(scored["strata"]["all_labeled"]["matched_annotations"], 1)
        self.assertEqual(scored["strata"]["all_labeled"]["missed_annotations"], 1)
        self.assertTrue(all(obj["ambiguous_gate_present"] for obj in scored["per_object"].values()))

    def test_augmenting_paths_find_maximum_cardinality_not_greedy(self):
        objects = [{"box_ltwh": [0, 0, 4, 4]}, {"box_ltwh": [0, 0, 1, 1]}]
        observations = [{"identity": "0/bright:1", "xy": [0.5, 0.5]}, {"identity": "0/bright:2", "xy": [4, 4]}]
        matches, _ = scorer.assign(objects, observations, 0)
        self.assertEqual(matches, {0: 1, 1: 0})

    def test_multiple_detections_near_one_label_are_excess_not_extra_hits(self):
        rows = journal()
        rows[8]["tracks"] = [track(), track("dark:1")]
        result = result_stage(scorer.score_rows(manifest(), rows))
        self.assertEqual(result["counts"]["matched_annotations"], 1)
        self.assertEqual(result["counts"]["unmatched_inside_any_gt_gate"], 1)
        self.assertEqual(result["counts"]["unmatched_outside_all_gt_gates"], 0)
        self.assertFalse(result["per_object"]["Airplane1"]["identity_fragmentation_available"])
        self.assertIsNone(result["per_object"]["Airplane1"]["first_unambiguous_hit"])

    def test_unknown_and_far_labels_are_matched_before_range_stratum(self):
        labels = {8: [annotation("Airplane1", distance=701), annotation("Airborne2", (100, 100, 10, 10), None),
                      annotation("Helicopter3", (200, 200, 10, 10), 700)]}
        rows = journal()
        rows[8]["tracks"] = [track(xy=(13, 23)), track("dark:1", (105, 105)), track("bright:2", (205, 205))]
        scored = result_stage(scorer.score_rows(manifest(labels), rows))
        self.assertEqual(scored["strata"]["all_labeled"]["matched_annotations"], 3)
        self.assertEqual(scored["strata"]["known_range_le_700m"]["matched_annotations"], 1)
        self.assertEqual(scored["strata"]["tiny_box_area_le_100"]["matched_annotations"], 3)
        self.assertEqual(scored["counts"]["unmatched_observations"], 0)
        self.assertEqual(scored["publisher_empty_context"]["publisher_empty_frames"], 299)

    def test_warmup_views_are_fixed_not_runtime_availability_filters(self):
        labels = {i: [annotation()] for i in range(12)}
        rows = journal()
        rows[9]["coverage"].update(warmup=True, detection_ready=False, unavailable_reason="warmup")
        rows[10]["coverage"].update(searchable_pixels=0, detection_ready=False, unavailable_reason="no_valid_search_support")
        rows[10]["motion"].update(reset=True, pva_failure=True, accepted=False, status="reset_reference")
        result = scorer.score_rows(manifest(labels), rows)
        all_view, eligible = result["windows"]["all_300"], result["windows"]["fixed_eligible_8_299"]
        self.assertEqual(all_view["frame_count"], 300)
        self.assertEqual(eligible["frame_count"], 292)
        self.assertEqual(result_stage(result)["strata"]["all_labeled"]["labeled_annotations"], 12)
        self.assertEqual(result_stage(result, window="fixed_eligible_8_299")["strata"]["all_labeled"]["labeled_annotations"], 4)
        self.assertEqual(eligible["coverage_context"]["runtime_warmup_frames"], 1)
        self.assertEqual(eligible["coverage_context"]["pva_failure_labeled_annotations"], 1)
        self.assertEqual(eligible["coverage_context"]["frames_excluded_for_motion_or_availability"], 0)

    def test_empty_exposure_is_bounded_and_does_not_infer_physical_false_targets(self):
        rows = journal()
        rows[100]["tracks"] = [track(), track("dark:1", measured=False)]
        scored = scorer.score_rows(manifest({}), rows)
        negative = result_stage(scored)["publisher_empty_context"]
        self.assertEqual(negative["nominal_exposure_seconds"], 30.0)
        self.assertEqual(negative["observations_on_publisher_empty_frames"], 1)
        self.assertEqual(negative["publisher_empty_frames_with_observations"], 1)
        self.assertIsNone(result_stage(scored)["strata"]["all_labeled"]["annotation_match_fraction"])
        for key in ("official_aot_metrics", "independent_validation", "physical_class_inferred", "deployment_accuracy_claim"):
            self.assertIs(scored[key], False)
        self.assertNotIn("hourly_false_alarm_rate", json.dumps(scored))

    def test_histories_preserve_exact_timestamps_and_first_hit_reference(self):
        labels = {i: [annotation()] for i in range(8, 12)}
        rows = journal()
        rows[10]["tracks"] = [track()]
        history = result_stage(scorer.score_rows(manifest(labels), rows))["per_object"]["Airplane1"]
        first = history["first_unambiguous_hit"]
        self.assertEqual(first["frame_index"], 10)
        self.assertEqual(first["source_frame"], 13)
        self.assertEqual(first["source_timestamp_ns"], str(STAMP + 1_001_000_000))
        self.assertEqual(first["source_ns_after_first_label_in_window"], "199000000")
        self.assertEqual(first["nominal_seconds_after_first_label_in_window"], 0.2)
        self.assertTrue(first["not_physical_onset_latency"])

    def test_fragments_preserve_segments_polarities_and_missed_frames(self):
        labels = {i: [annotation()] for i in range(8, 14)}
        rows = journal()
        rows[8]["tracks"] = [track()]
        rows[10]["tracks"] = [track()]
        rows[11]["tracks"] = [track("dark:1")]
        for row in rows[12:]:
            row["segment"] = 1
        rows[12]["tracks"] = [track("dark:1", segment=1)]
        rows[13]["tracks"] = [track("dark:1", segment=1)]
        history = result_stage(scorer.score_rows(manifest(labels), rows))["per_object"]["Airplane1"]
        self.assertEqual(history["identity_fragments"], 4)
        self.assertEqual(history["identity_fragmentation_additional_fragments"], 3)
        self.assertEqual(history["matched_identities"], ["0/bright:1", "0/dark:1", "1/dark:1"])

    def test_gt_gaps_and_raw_candidates_do_not_receive_fragmentation_claim(self):
        rows = journal()
        for i in (8, 10):
            rows[i]["tracks"] = [track()]
            rows[i]["candidates"] = [{"source_xy": [13, 23], "polarity": "bright"}]
        result = scorer.score_rows(manifest({8: [annotation()], 10: [annotation()]}), rows)
        for stage in ("all_measured", "raw_candidates_diagnostic"):
            obj = result_stage(result, stage=stage)["per_object"]["Airplane1"]
            self.assertIsNone(obj["identity_fragments"])
            self.assertFalse(obj["identity_fragmentation_available"])

    def test_missing_reordered_duplicate_extra_journal_rows_fail_closed(self):
        mutations = [lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[-1])),
                     lambda r: r.__setitem__(8, copy.deepcopy(r[7])), lambda r: r.reverse()]
        for mutate in mutations:
            rows = journal()
            mutate(rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                scorer.score_rows(manifest(), rows)

    def test_wrong_timestamp_cropped_or_missing_availability_fail_closed(self):
        cases = [("timestamp", lambda r: r[8].update(timestamp_ns=STAMP)),
                 ("crop", lambda r: r[8]["coverage"].update(configured_crop=[0, 0, 100, 100])),
                 ("shape", lambda r: r[8]["coverage"].update(full_shape_hw=[1024, 1224])),
                 ("coverage", lambda r: r[8].pop("coverage")),
                 ("ready", lambda r: r[8]["coverage"].update(detection_ready=False))]
        for name, mutate in cases:
            rows = journal()
            mutate(rows)
            with self.subTest(name=name), self.assertRaises(ValueError):
                scorer.validate_rows(rows)

    def test_missing_false_nonfinite_and_duplicate_measurements_fail_closed(self):
        cases = [track(xy=(float("nan"), 23)), track(xy=(True, 23)), track(segment=1)]
        absent = track()
        del absent["measurement_source_xy"]
        cases.append(absent)
        fake = track(measured=False)
        fake["measurement_source_xy"] = [13, 23]
        cases.append(fake)
        for item in cases:
            rows = journal()
            rows[8]["tracks"] = [item]
            with self.subTest(track=item), self.assertRaises(ValueError):
                scorer.validate_rows(rows)
        rows = journal()
        rows[8]["tracks"] = [track(), track()]
        with self.assertRaises(ValueError):
            scorer.validate_rows(rows)

    def test_gt_incomplete_identity_or_census_never_becomes_negative(self):
        for change in ("missing_label", "missing_frame", "bad_name", "count", "nan_range"):
            data = manifest()
            if change == "missing_label":
                del data["frames"][8]["entities"][0]["bb"]
            elif change == "missing_frame":
                data["frames"].pop(8)
            elif change == "bad_name":
                data["frames"][8]["img_name"] = "wrong.png"
            elif change == "count":
                data["frames"][8]["airborne_label_count"] = 0
            else:
                data["frames"][8]["entities"][0]["blob"]["range_distance_m"] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                scorer.validate_manifest(data)

    def test_strict_json_rejects_duplicates_and_nonfinite_numbers(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                scorer.strict_json(text)

    def test_run_metadata_requires_frozen_hardware_source_and_complete_decode(self):
        launch, report = run_metadata()
        scorer.validate_run_metadata(launch, report)
        mutations = [lambda l, r: l.update(source_sha256="0" * 64),
                     lambda l, r: l.update(config_sha256="0" * 64),
                     lambda l, r: l.update(annotations_supplied_to_detector=True),
                     lambda l, r: r.update(completed=False),
                     lambda l, r: r["frame_decode"].update(dropped_frames=1),
                     lambda l, r: l["configuration"].update(motion_backend="cpu_translation"),
                     lambda l, r: l.update(max_frames=300), lambda l, r: r.update(full_clip=False),
                     lambda l, r: l["source_probe"].update(pixel_format="rgb24"),
                     lambda l, r: l["source_probe"].update(width=1224),
                     lambda l, r: l["source_probe"].update(codec="h264")]
        for mutate in mutations:
            launch, report = run_metadata()
            mutate(launch, report)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                scorer.validate_run_metadata(launch, report)

    def test_freeze_binds_policy_manifest_and_source_files_without_reading_outputs(self):
        with tempfile.TemporaryDirectory(prefix="seaqr_generated_scoring_") as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest()))
            for name in (*scorer.FREEZE_FILES, scorer.HARNESS_FILE):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("generated freeze fixture: " + name)
            with mock.patch.object(scorer, "ROOT", root), mock.patch.object(scorer, "MANIFEST_SHA256", scorer.digest(path)):
                frozen = scorer.make_freeze(path)
                self.assertIs(frozen["detector_outputs_read"], False)
                scorer.verify_freeze(frozen, path)
                (root / scorer.FREEZE_FILES[0]).write_text("changed source")
                with self.assertRaisesRegex(ValueError, "changed after freeze"):
                    scorer.verify_freeze(frozen, path)

    def test_execution_receipt_requires_exact_combined_stack_and_complete_lifecycle(self):
        receipt, preflight, hashes = execution_fixture()
        scorer.validate_execution(receipt, preflight, hashes)
        mutations = [lambda r: r.update(passed=False), lambda r: r.update(error="failed"),
                     lambda r: r.update(algorithm_changed=True), lambda r: r.update(pixel_hashes_verified=299),
                     lambda r: r["gpu_fronts"][0].update(host_calls=1), lambda r: r["gpu_fronts"][0].update(closed=False),
                     lambda r: r["tracking"].update(batch_fallbacks=1), lambda r: r["tracking"].update(innovation_tracks=6),
                     lambda r: r["motion_instances"][0].update(failed=True), lambda r: r.update(cleanup_errors=["error"]),
                     lambda r: r["execution"].update(policy="bounded"), lambda r: r["motion_attempts"].pop(),
                     lambda r: r["runtime_after"].update(numpy="2.0"), lambda r: r["libraries"].clear(),
                     lambda r: r["adapters"].pop("motion_reuse_v12"), lambda r: r.update(scoring_freeze_sha256="0" * 64),
                     lambda r: r.update(vpi_version="wrong"), lambda r: r.update(remote_clocks_unchanged=False),
                     lambda r: r["clock_policy_after"].update(generated="changed")]
        for mutate in mutations:
            receipt, preflight, hashes = execution_fixture()
            mutate(receipt)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                scorer.validate_execution(receipt, preflight, hashes)

    def test_expected_motion_unavailability_is_not_silently_a_run_failure(self):
        receipt, preflight, hashes = execution_fixture()
        receipt["motion_attempts"][8].update(error="PVA zero features", expected_unavailable=True)
        scorer.validate_execution(receipt, preflight, hashes)
        receipt["motion_attempts"][8]["expected_unavailable"] = False
        with self.assertRaises(ValueError):
            scorer.validate_execution(receipt, preflight, hashes)

    def test_preflight_must_bind_same_input_freeze_harness_and_dependencies(self):
        mutations = [lambda p: p.update(detector_run=True), lambda p: p.update(passed=False),
                     lambda p: p["decode"].update(pixel_hashes_verified=299),
                     lambda p: p["input_sha256"].update(**{"scoring_freeze.json": "0" * 64}),
                     lambda p: p.update(script_sha256="0" * 64), lambda p: p["adapters"].clear()]
        for mutate in mutations:
            receipt, preflight, hashes = execution_fixture()
            mutate(preflight)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                scorer.validate_execution(receipt, preflight, hashes)

    def test_write_new_never_overwrites_and_journal_loader_rejects_blank_rows(self):
        with tempfile.TemporaryDirectory(prefix="seaqr_generated_scoring_") as directory:
            path = Path(directory) / "result.json"
            scorer.write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                scorer.write_new(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})
            source = Path(directory) / "generated.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in journal()) + "\n")
            self.assertEqual(len(scorer.read_journal(source)), 300)
            source.write_text(source.read_text() + "\n")
            with self.assertRaises(ValueError):
                scorer.read_journal(source)


if __name__ == "__main__":
    unittest.main()
