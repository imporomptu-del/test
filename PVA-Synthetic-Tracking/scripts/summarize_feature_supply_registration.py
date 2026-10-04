"""Frozen-pair registration diagnostics from saved metadata, never media or fits.

These comparisons are descriptive. Feature cohorts change between algorithms;
inlier residuals and same-pair transform agreement are not independent truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
BASELINE_ROOT = OUTPUTS / "seaqr_discovery_pair_20260928/evidence"
CANDIDATE_ROOT = OUTPUTS / "seaqr_feature_supply_20260929/evidence"
OUTPUT_ROOT = CANDIDATE_ROOT.parent
CLIPS = ("0170", "0240")
FRAMES = 673
SOURCE_SHA = {"0170": "12848c0f0caedd697a3da51776ab1579bd634a7ae94343f8cbd2a8830ee340bc",
              "0240": "2f86f28785e302572a86e23688143edbd7f5f1f65e8a3434b86a427e79c6a585"}
CONFIG_SHA = "7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f"
MOTION_SHA = "fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1"
METHOD_SHA = "80aabc85b25b9204bc9de1838972ddd78a2b639d2b7b933609724d5ca2d7c733"
GAIN_METHOD_SHA = "bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b"
CANDIDATE = dict(harris_gain=16, harris_capacity_policy="complete_grid", feature_image_scale=0.5)
BACKENDS = dict(intensity_conversion="CPU", motion_image_rescale="CUDA", gaussian_pyramid="PVA",
                harris_input_conversion="CUDA", harris="PVA", optical_flow_pyrlk="PVA", cpu_fallback=False)
FEATURE_FIELDS = ("detected_count", "selected_count", "accepted_count", "rejected_count",
                  "track_survival_ratio", "grid_coverage.occupied_cells", "grid_coverage.fraction",
                  "median_displacement_px", "p95_displacement_px", "median_forward_backward_error_px")
FIT_FIELDS = ("correspondence_count", "inlier_count", "inlier_ratio", "inlier_grid_coverage.occupied_cells",
              "inlier_grid_coverage.fraction", "median_reprojection_error_px", "p90_reprojection_error_px",
              "maximum_reprojection_error_px", "sparse_translation_support.supported_cells",
              "sparse_translation_support.consensus_cells", "sparse_translation_support.supported_point_fraction",
              "sparse_translation_support.consensus_point_fraction",
              "sparse_translation_support.fit_to_balanced_error_px")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or linked metadata")
    return parse(path.read_text())


def get(value, dotted):
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def summary(values):
    """Linear quantiles across frames; missing values are counted, never zeroed."""
    values = list(values)
    present = [v for v in values if v is not None]
    require(all(number(v) for v in present), "nonfinite or nonnumeric diagnostic")
    valid = sorted(present)
    result = dict(total=len(values), count=len(valid), missing=len(values) - len(valid), mean=None,
                  minimum=None, p10=None, median=None, p90=None, p95=None, maximum=None)
    if valid:
        result.update(mean=statistics.fmean(valid), minimum=valid[0], maximum=valid[-1])
        for name, q in (("p10", .1), ("median", .5), ("p90", .9), ("p95", .95)):
            location = q * (len(valid) - 1)
            lo, hi = math.floor(location), math.ceil(location)
            result[name] = valid[lo] + (location - lo) * (valid[hi] - valid[lo])
    return result


def validate_scope(baseline_root, candidate_root, output):
    baseline_root, candidate_root, output = map(Path, (baseline_root, candidate_root, output))
    require(baseline_root == BASELINE_ROOT and candidate_root == CANDIDATE_ROOT,
            "only the two fixed evidence roots are authorized")
    require(output.parent == OUTPUT_ROOT and output.suffix == ".json"
            and output.name not in {"freeze.json", "batch_status.json"}, "output outside diagnostic root")
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    for path in (baseline_root, candidate_root, output.parent):
        require(path.is_dir() and all(not p.is_symlink() for p in (path, *path.parents)), "missing or linked root")
    return baseline_root, candidate_root, output


def source_spec(clip):
    require(clip in CLIPS, "unapproved clip")
    return dict(path=f"/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_{clip}.avi",
                sha256=SOURCE_SHA[clip], frames=FRAMES, width=4784, height=3190,
                fps=10, codec="mjpeg", pixel_format="yuvj420p")


def validate_artifacts(receipt, preflight, report, launch, arm, clip):
    require(arm in ("baseline", "candidate"), "invalid arm")
    schema = "seaqr.discovery-pair.baseline.v1" if arm == "baseline" else "seaqr.discovery-feature-supply.v1"
    require(receipt.get("schema") == schema and receipt.get("passed") is True
            and receipt.get("clip") == clip and receipt.get("source") == source_spec(clip)
            and receipt.get("processed_frames") == receipt.get("decoded_frames_verified") == FRAMES,
            "failed, incomplete or wrong-source receipt")
    require(preflight.get("schema") == schema + ".preflight" and preflight.get("passed") is True
            and preflight.get("clip") == clip and preflight.get("source") == source_spec(clip)
            and preflight.get("workspace") == receipt.get("workspace")
            and preflight.get("input_sha256") == receipt.get("input_sha256"), "preflight binding differs")
    require(receipt.get("algorithm_changed") is (arm == "candidate"), "wrong algorithm declaration")
    require(receipt.get("detector_configuration_changed") is False
            and receipt.get("annotations_supplied_to_detector") is False
            and receipt.get("raw16_accessed") is False and receipt.get("sealed_holdouts_accessed") is False,
            "execution scope differs")
    if arm == "candidate":
        require(receipt.get("candidate") == CANDIDATE and receipt.get("feature_algorithm_changed") is True
                and receipt.get("global_motion_gates_changed") is False
                and receipt.get("tracker_configuration_changed") is False
                and receipt.get("production_promotion") is False, "candidate policy differs")
        audit = receipt["feature_adapter"]
        original, effective = audit["original_motion_configuration"], audit["effective_motion_configuration"]
        require(original["harris_capacity_policy"] == "legacy_default"
                and dict(original, harris_capacity_policy="complete_grid") == effective
                and effective["feature_image_scale"] == .5, "unexpected effective configuration")
        transform = audit["source_transformation"]
        require(transform.get("original_method_sha256") == METHOD_SHA
                and transform.get("transformed_method_sha256") == GAIN_METHOD_SHA
                and transform.get("exact_original_recovered") is True, "candidate source transformation differs")
    require(report.get("schema") == "seaqr.visible-baseline.v1" and report.get("completed") is True
            and report.get("full_clip") is True and report.get("frames") == FRAMES
            and report.get("source_sha256") == launch.get("source_sha256") == SOURCE_SHA[clip]
            and launch.get("source") == source_spec(clip)["path"], "incomplete or wrong-source report/launch")
    require(report.get("configuration") == launch.get("configuration"), "detector configuration mismatch")
    for key, value in (("config_sha256", CONFIG_SHA), ("motion_config_sha256", MOTION_SHA)):
        require(receipt.get(key) == launch.get(key) == value, "frozen configuration identity differs")
    probe = launch["source_probe"]
    require((probe["width"], probe["height"], probe["declared_frame_count"]) == (4784, 3190, FRAMES)
            and probe["codec"] == "mjpeg" and probe["pixel_format"] == "yuvj420p"
            and Fraction(probe["frame_rate"]) == 10, "native probe contract differs")
    decode = report["frame_decode"]
    require(decode["decoded_frames"] == decode["consumed_frames"] == FRAMES
            and decode["dropped_frames"] == 0 and decode["worker_joined"] is True
            and decode["capture_released"] is True, "decode inventory/lifecycle differs")


def validate_rows(rows, attempts):
    require(len(rows) == FRAMES and len(attempts) == FRAMES - 1, "incomplete frame/pair inventory")
    require([r.get("frame") for r in attempts] == list(range(1, FRAMES)), "attempt order differs")
    for i, row in enumerate(rows):
        require(type(row.get("frame_index")) is int and row["frame_index"] == i
                and row.get("timestamp_ns") == i * 100_000_000, "frame index/timeline differs")
        coverage, motion = row["coverage"], row["motion"]
        require(coverage["full_shape_hw"] == [3190, 4784] and coverage["native_pixel_sampling"] is True
                and coverage["configured_crop"] is None, "non-native/cropped frame")
        require(coverage["detection_ready"] is (not coverage["warmup"] and coverage["searchable_pixels"] > 0),
                "readiness fields disagree")
        require(motion.get("backend") == "pva" and motion.get("cpu_warp_fallback") is False
                and motion.get("stabilization_execution") == "cuda_cubic_resident", "motion backend differs")
        if i == 0:
            require(motion.get("status") == "initial_reference" and motion.get("reset") is False,
                    "first frame is not initial reference")
            continue
        attempt = attempts[i - 1]
        require(type(attempt["expected_unavailable"]) is bool and type(motion["pva_failure"]) is bool,
                "invalid unavailable flags")
        require(motion["pva_failure"] is (attempt["error"] is not None), "journal/attempt error mismatch")
        if motion["pva_failure"]:
            require(attempt["expected_unavailable"] is True and motion["reset"] is True
                    and "motion_fit" not in motion, "unexpected execution failure or fabricated fit")
            continue
        require(attempt["expected_unavailable"] is False and motion["motion_backends"] == BACKENDS,
                "backend/attempt contract differs")
        fit, corr = motion["motion_fit"], motion["correspondence_metrics"]
        accepted = motion["accepted"]
        require(type(accepted) is bool and fit["quality_status"] == ("accepted" if accepted else "rejected")
                and motion["reset"] is (not accepted) and motion["status"] == ("accepted" if accepted else "reset_reference")
                and bool(fit["rejection_reasons"]) is (not accepted)
                and motion["rejection_reasons"] == fit["rejection_reasons"], "inconsistent saved fit acceptance")
        require(fit["model"] == "translation" and fit["previous_frame_index"] == i - 1
                and fit["current_frame_index"] == i
                and fit["mapping"] == "previous_frame_pixels_to_current_frame_pixels", "pair/model/mapping differs")
        require(fit["metrics"]["correspondence_count"] == corr["accepted_count"]
                and corr["accepted_count"] + corr["rejected_count"] == corr["selected_count"], "feature accounting differs")
        parameters, matrix = fit["parameters"], fit["previous_to_current_matrix"]
        if parameters is None:
            require(not accepted and matrix is None, "missing accepted transform")
        else:
            tx, ty = parameters["translation_x_px"], parameters["translation_y_px"]
            require(number(tx) and number(ty) and matrix == [[1, 0, tx], [0, 1, ty], [0, 0, 1]],
                    "translation parameter/matrix disagreement")


def load_arm(root, arm, clip, hashes):
    base = root / clip
    require(base.is_dir() and not base.is_symlink() and not (base / "run").is_symlink(), "linked/missing clip")
    paths = dict(receipt=base / "execution_receipt.json", preflight=base / "preflight.json",
                 report=base / "run/report.json", launch=base / "run/launch.json", journal=base / "run/frames.jsonl")
    for path in paths.values():
        require(path.is_file() and not path.is_symlink(), "missing/linked evidence")
        hashes[str(path)] = sha(path)
    receipt, preflight, report, launch = (read(paths[key]) for key in ("receipt", "preflight", "report", "launch"))
    for key in ("preflight", "report", "launch", "journal"):
        require(receipt[key + "_sha256"] == hashes[str(paths[key])], "receipt artifact hash mismatch: " + key)
    validate_artifacts(receipt, preflight, report, launch, arm, clip)
    rows = []
    with paths["journal"].open() as stream:
        for line in stream:
            value = parse(line)
            rows.append({key: value[key] for key in ("frame_index", "timestamp_ns", "motion", "coverage", "timings_ms")})
    validate_rows(rows, receipt["motion_attempts"])
    require(sum(row["coverage"]["detection_ready"] for row in rows)
            == report["availability"]["counts"]["detection_ready_frames"], "report readiness count differs")
    return dict(receipt=receipt, report=report, launch=launch, rows=rows)


def metrics(rows, prefix, fields):
    return {field: summary(get(row, prefix + "." + field) for row in rows) for field in fields}


def fit_summary(rows):
    return dict(frames=len(rows), frame_indices=[r["frame_index"] for r in rows],
                metrics=metrics(rows, "motion.motion_fit.metrics", FIT_FIELDS),
                acceptance_paths=dict(Counter(get(r, "motion.motion_fit.metrics.coverage_acceptance_path")
                                               or "unavailable" for r in rows)))


def timing_summary(rows):
    stages = sorted({key for row in rows for key in row.get("timings_ms", {})})
    pva_stages = ("motion_image_prepare", "gaussian_pyramids", "harris_input_conversion_cuda", "harris_pva",
                  "harris_readback_cpu", "feature_eligibility_cpu", "spatial_quota_cpu",
                  "feature_readback_and_grid_selection", "forward_pyrlk_pva", "backward_pyrlk_pva",
                  "flow_readback_cpu", "coordinate_lift_and_filter_cpu", "total_including_metrics")
    return dict(frames=len(rows), outer_stages_ms=metrics(rows, "timings_ms", stages),
                pva_work_ms=metrics(rows, "motion.pva_timings_ms", pva_stages),
                global_fit_ms=summary(get(r, "motion.motion_fit.timing_ms") for r in rows),
                resident_warp_ms=summary(get(r, "motion.warp_timings_ms.exact_cuda_total") for r in rows))


def arm_summary(data):
    rows, report = data["rows"], data["report"]
    pairs = rows[1:]
    accepted = [r for r in pairs if r["motion"].get("accepted") is True]
    fitted = [r for r in pairs if get(r, "motion.motion_fit.parameters") is not None]
    reasons, paths = Counter(), Counter()
    losses, exclusions = Counter(), Counter()
    for row in pairs:
        reasons.update(row["motion"].get("rejection_reasons", []))
        paths.update([get(row, "motion.motion_fit.metrics.coverage_acceptance_path") or "unavailable"])
        losses.update(get(row, "motion.correspondence_metrics.rejections") or {})
        exclusions.update(get(row, "motion.correspondence_metrics.feature_exclusions_before_tracking") or {})
    selected = sum(get(r, "motion.correspondence_metrics.selected_count") or 0 for r in pairs)
    survivors = sum(get(r, "motion.correspondence_metrics.accepted_count") or 0 for r in pairs)
    return dict(pair_count=len(pairs), accepted_pairs=len(accepted), accepted_pair_fraction=len(accepted) / len(pairs),
                fitted_pairs=len(fitted), unavailable_fit_parameters=len(pairs) - len(fitted),
                resets=sum(r["motion"]["reset"] for r in pairs),
                pva_errors=sum(r["motion"]["pva_failure"] for r in pairs),
                ready_frames=sum(r["coverage"]["detection_ready"] for r in rows),
                rejection_reason_counts=dict(reasons), acceptance_paths=dict(paths),
                feature_supply=metrics(pairs, "motion.correspondence_metrics", FEATURE_FIELDS),
                selected_total=selected, accepted_correspondence_total=survivors,
                point_weighted_survival=survivors / selected if selected else None,
                selected_1000_frames=sum(get(r, "motion.correspondence_metrics.selected_count") == 1000 for r in pairs),
                flow_rejection_totals=dict(losses), pretracking_exclusion_totals=dict(exclusions),
                harris_capacity=summary(get(r, "motion.correspondence_metrics.harris_output.capacity") for r in pairs),
                harris_exhausted_frames=sum(get(r, "motion.correspondence_metrics.harris_output.capacity_exhausted") is True
                                            for r in pairs),
                fits=dict(all_fitted=fit_summary(fitted), own_accepted=fit_summary(accepted)),
                timing=dict(all_frames=timing_summary(rows), first_frame=timing_summary(rows[:1]),
                            first_pair=timing_summary(rows[1:2]), later_pairs=timing_summary(rows[2:]),
                            processed_fps=report["processed_fps"], elapsed_seconds=report["elapsed_seconds"],
                            recorded_semantics=report.get("timing_semantics"),
                            instrumentation=data["receipt"].get("timing_instrumentation")))


def compare_rows(baseline, candidate):
    require(len(baseline) == len(candidate) == FRAMES, "comparison inventory differs")
    groups = {key: [] for key in ("both_accepted", "baseline_only", "candidate_only", "neither_accepted")}
    for b, c in zip(baseline[1:], candidate[1:]):
        require(b["frame_index"] == c["frame_index"], "comparison frame join differs")
        ba, ca = b["motion"].get("accepted") is True, c["motion"].get("accepted") is True
        key = "both_accepted" if ba and ca else "baseline_only" if ba else "candidate_only" if ca else "neither_accepted"
        groups[key].append(b["frame_index"])
    differences = []
    for index in groups["both_accepted"]:
        b, c = (rows[index]["motion"]["motion_fit"] for rows in (baseline, candidate))
        require(b["model"] == c["model"] == "translation", "nontranslation paired comparison")
        dx, dy = (c["parameters"][key] - b["parameters"][key] for key in ("translation_x_px", "translation_y_px"))
        require(number(dx) and number(dy), "nonfinite paired transform")
        differences.append(dict(frame_index=index, dx_px=dx, dy_px=dy, norm_px=math.hypot(dx, dy)))
    paired_quality = {}
    for field in FIT_FIELDS:
        values = []
        for index in groups["both_accepted"]:
            b, c = (get(rows[index], "motion.motion_fit.metrics." + field) for rows in (baseline, candidate))
            values.append(c - b if b is not None and c is not None else None)
        paired_quality[field] = summary(values)
    return dict(transition_counts={key: len(indices) for key, indices in groups.items()}, frame_indices=groups,
                joint_accepted=dict(baseline=fit_summary([baseline[i] for i in groups["both_accepted"]]),
                                    candidate=fit_summary([candidate[i] for i in groups["both_accepted"]])),
                newly_accepted=fit_summary([candidate[i] for i in groups["candidate_only"]]),
                lost_acceptance=fit_summary([baseline[i] for i in groups["baseline_only"]]),
                translation_disagreement=dict(direction="candidate minus baseline, previous-to-current native pixels",
                    points=summary(d["norm_px"] for d in differences),
                    dx_px=summary(d["dx_px"] for d in differences), dy_px=summary(d["dy_px"] for d in differences),
                    per_pair=differences,
                    worst_10=sorted(differences, key=lambda d: (-d["norm_px"], d["frame_index"]))[:10]),
                paired_fit_metric_difference=dict(direction="candidate minus baseline", metrics=paired_quality))


def run(baseline_root, candidate_root, output):
    baseline_root, candidate_root, output = validate_scope(baseline_root, candidate_root, output)
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    results = {}
    for clip in CLIPS:
        base = load_arm(baseline_root, "baseline", clip, hashes)
        candidate = load_arm(candidate_root, "candidate", clip, hashes)
        for key in ("configuration", "config_sha256", "motion_config_sha256", "code_sha256", "package_sha256"):
            require(base["launch"][key] == candidate["launch"][key], "baseline/candidate frozen launch differs: " + key)
        for key in ("adapters", "libraries", "v29_freeze_sha256", "tracking_transformed_sha256"):
            require(base["receipt"][key] == candidate["receipt"][key], "unchanged execution dependency differs: " + key)
        b, c = arm_summary(base), arm_summary(candidate)
        require(all(number(data["report"][key]) and data["report"][key] > 0
                    for data in (base, candidate) for key in ("processed_fps", "elapsed_seconds")), "invalid wall timing")
        results[clip] = dict(baseline=b, candidate=c, comparison=compare_rows(base["rows"], candidate["rows"]),
            observed_timing_ratio=dict(candidate_over_baseline_fps=c["timing"]["processed_fps"] / b["timing"]["processed_fps"],
                candidate_over_baseline_wall_time=c["timing"]["elapsed_seconds"] / b["timing"]["elapsed_seconds"]))
    result = dict(schema="seaqr.feature-supply.registration-summary.v1", passed_integrity=True,
        input_sha256=hashes, clips=results, independent_geometry_truth=False, new_acceptance_gate=False,
        new_fits_performed=False, media_read=False, production_promotion=False,
        interpretation=["Quantiles summarize saved per-frame metrics, not pooled individual-point residuals.",
            "Saved reprojection summaries are inlier-only and feature/inlier cohorts differ between arms.",
            "Same-pair translation disagreement is not independent registration error or target preservation.",
            "Pair acceptance comes from the saved global fit, not correspondence usable_for_transform.",
            "Missing fit/metric values remain unavailable, not zero. Rejection reasons can overlap.",
            "Sparse held-out-cell checks reuse the same image-derived correspondence population.",
            "Do not compare composed source_to_reference matrices across different reset histories.",
            "Stage work overlaps decode; nested PVA/warp/fit timings overlap parent motion_and_warp. Do not add them.",
            "Wall throughput includes decode, processing, journaling and lifecycle overhead; timing ratios are uncontrolled cross-run observations.",
            "Nominal10Hz container timestamps do not verify physical acquisition cadence."])
    require(all(sha(path) == value for path, value in hashes.items()), "metadata changed during summary")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "candidate-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps(dict(passed_integrity=result["passed_integrity"], clips=list(result["clips"]))))
