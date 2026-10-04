#!/usr/bin/env python3
"""Independent metadata-only audit of the frozen 26-case, two-depth PVA test.

No media, VPI, estimator imports, model fitting, or threshold search occurs.
Stored accepted populations are joined through selected slots, never by point
coordinates or accepted-array position. Scientific failure is a saved result.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "seaqr.pva-robustness.summary.v1"
SHAPE = (512, 640)
DEPTHS = (4, 2)
MODES = ("base", "trace")
MARGIN = 128
ERROR_LIMIT = .25
ARITHMETIC_ATOL = 1e-12  # Serialized residual identity only; never a quality gate.
SOURCES = {"probe_pva_robustness.py", "batch_pva_robustness.py", "summarize_pva_robustness.py",
           "test_pva_robustness.py", "test_batch_pva_robustness.py", "test_summarize_pva_robustness.py",
           "batch_discovery_pair.py"}
EXPECTED_GLOBAL = dict(model="translation", ransac_iterations=500, ransac_reprojection_px=1.0,
    random_seed=75, minimum_correspondences=30, minimum_inliers=30, minimum_inlier_ratio=.6,
    grid_rows=6, grid_cols=8, minimum_inlier_grid_coverage=.2, maximum_median_reprojection_px=.35,
    maximum_p90_reprojection_px=.75, maximum_reprojection_px=2., maximum_translation_px=120.,
    maximum_rotation_deg=2., minimum_scale=.98, maximum_scale=1.02, minimum_noncollinearity_ratio=.01,
    minimum_sample_separation_px=32., failure_policy="reset_reference", maximum_reuse_pairs=1,
    coverage_policy="translation_consensus", sparse_minimum_cells=4, sparse_minimum_points_per_cell=4,
    sparse_minimum_span_fraction=.25)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024*1024,
            "Missing, linked, or oversized metadata: " + str(path))
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def number(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON number")
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: require(False, "Nonfinite JSON constant"))


def array(record):
    require(isinstance(record, dict) and set(record) == {"dtype", "shape", "data_base64", "sha256"},
            "Unexpected array descriptor")
    dtype = np.dtype(record["dtype"])
    shape = record["shape"]
    require(dtype.kind in "buif" and dtype.itemsize <= 8 and isinstance(shape, list)
            and len(shape) <= 2 and all(type(x) is int and x >= 0 for x in shape), "Invalid array type/shape")
    count = math.prod(shape)
    require(count <= 512*640 and isinstance(record["data_base64"], str)
            and len(record["data_base64"]) <= 4*512*640*8//3+8, "Unbounded capture")
    raw = base64.b64decode(record["data_base64"], validate=True)
    require(len(raw) == count*dtype.itemsize and hashlib.sha256(raw).hexdigest() == record["sha256"],
            "Array bytes/hash differ")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def same_array(left, right):
    a, b = array(left), array(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def statistics(values):
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "Invalid error population")
    return dict(count=len(values), median_px=float(np.median(values)) if len(values) else None,
                p90_px=float(np.percentile(values, 90)) if len(values) else None,
                maximum_px=float(values.max()) if len(values) else None,
                above_0_25px=int(np.count_nonzero(values > ERROR_LIMIT)))


def truth_mask(points, truth):
    points = np.asarray(points, np.float64)
    truth = np.asarray(truth, np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and truth.shape == (2,)
            and np.isfinite(points).all() and np.isfinite(truth).all(), "Invalid native truth coordinates")
    expected = points + truth
    limit = np.array([SHAPE[1], SHAPE[0]]) - MARGIN
    return ((points >= MARGIN) & (points < limit) & (expected >= MARGIN) & (expected < limit)).all(axis=1)


def fit_analysis(result, p, q, truth, config):
    """Check the returned original model; never refit or substitute a model."""
    fit = result["fit"]
    require(isinstance(fit, dict) and fit.get("model") == "translation"
            and fit.get("mapping") == "previous_frame_pixels_to_current_frame_pixels"
            and fit.get("previous_frame_index") == 0 and fit.get("current_frame_index") == 1,
            "Original fit mapping/model differs")
    require(fit.get("quality_status") in {"accepted", "rejected"}, "Invalid fit status")
    mask, residuals = array(result["inlier_mask"]), array(result["residuals_px"])
    require(mask.dtype == np.dtype(bool) and residuals.dtype == np.dtype("float64")
            and mask.shape == residuals.shape == (len(p),), "Original mask/residual representation differs")
    require(fit.get("inlier_indices") == np.flatnonzero(mask).tolist(), "Original inlier indices differ")
    matrix, parameters = fit.get("previous_to_current_matrix"), fit.get("parameters")
    accepted = fit["quality_status"] == "accepted"
    if matrix is None:
        require(not accepted and parameters is None and not mask.any() and np.isnan(residuals).all(),
                "Unavailable fit is not a measured zero model")
        return dict(fit_ran=True, accepted=False, model_available=False, translation_xy=None, translation_error_px=None,
                    inlier_count=0, mask_is_measured_inlier_classification=False), mask
    matrix = np.asarray(matrix, np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and isinstance(parameters, dict),
            "Invalid original model")
    expected = np.eye(3)
    expected[:2, 2] = matrix[:2, 2]
    require(np.array_equal(matrix, expected), "Nontranslation model cannot use translation-only truth error")
    vector = np.array([parameters.get("translation_x_px"), parameters.get("translation_y_px")], np.float64)
    require(np.isfinite(vector).all() and np.array_equal(vector, matrix[:2, 2]), "Matrix/parameter translation differs")
    calculated = np.linalg.norm(p + vector - q, axis=1)
    require(np.isfinite(residuals).all() and np.allclose(calculated, residuals, rtol=0, atol=ARITHMETIC_ATOL),
            "Original residuals disagree with saved model")
    require(np.array_equal(mask, residuals <= config["ransac_reprojection_px"]), "Original residual/inlier mask differs")
    metrics = fit["metrics"]
    require(metrics.get("correspondence_count") == len(p) and metrics.get("inlier_count") == int(mask.sum()),
            "Original fit counts differ")
    require(metrics.get("inlier_ratio") == (int(mask.sum())/len(p) if len(p) else 0), "Original inlier ratio differs")
    require((not accepted) or (not fit["rejection_reasons"] and len(p) >= config["minimum_correspondences"]
            and int(mask.sum()) >= config["minimum_inliers"]), "Impossible accepted original fit")
    return dict(fit_ran=True, accepted=accepted, model_available=True, translation_xy=vector.tolist(),
                translation_error_px=float(np.linalg.norm(vector-truth)), inlier_count=int(mask.sum()),
                mask_is_measured_inlier_classification=True, rejection_reasons=fit["rejection_reasons"]), mask


def accepted_analysis(canonical, case):
    result = canonical["result"]
    truth = np.asarray(case["truth_displacement_xy"], np.float64)
    if result.get("correspondence") is None:
        require(result.get("status") == "unavailable" and result.get("fit") is None, "Invalid pre-LK unavailable result")
        return dict(status="unavailable_before_correspondence", accepted_count=None,
                    all_accepted=statistics([]), truth_interior=statistics([]),
                    original_fit=dict(fit_ran=False, accepted=None, model_available=False, translation_xy=None,
                                      translation_error_px=None, inlier_count=None),
                    point_guard_passed=False), None
    correspondence = result["correspondence"]
    require(correspondence["full_image_size"] == [640, 512]
            and correspondence["motion_image_size"] == [320, 256], "Native/proxy coordinate dimensions differ")
    values = {name: array(correspondence["arrays"][name]) for name in
              ("previous_points", "current_points", "harris_scores", "forward_backward_error_px")}
    p, q = values["previous_points"], values["current_points"]
    require(p.dtype == q.dtype == np.dtype("float32") and p.ndim == 2 and p.shape[1] == 2
            and p.shape == q.shape and len(p) <= 384 and np.isfinite(p).all() and np.isfinite(q).all(),
            "Invalid accepted native points")
    for name in ("harris_scores", "forward_backward_error_px"):
        require(values[name].dtype == np.dtype("float32") and values[name].shape == (len(p),)
                and np.isfinite(values[name]).all(), "Invalid accepted scores/FB")
    p, q = p.astype(np.float64), q.astype(np.float64)
    require(np.all(p >= 0) and np.all(q >= 0) and np.all(p < [640, 512]) and np.all(q < [640, 512]),
            "Accepted native point outside source")
    require(correspondence["metrics"]["accepted_count"] == len(p), "Accepted count differs")
    errors = np.linalg.norm(q-(p+truth), axis=1)
    interior = truth_mask(p, truth)
    fit, inlier = fit_analysis(result, p, q, truth, canonical["global_configuration"])
    row = dict(status="measured" if len(p) else "unavailable_zero_accepted", accepted_count=len(p),
        all_accepted=statistics(errors), truth_interior=statistics(errors[interior]),
        truth_interior_indices=np.flatnonzero(interior).tolist(), truth_interior_uses_measured_endpoint=False,
        original_fit=fit, point_guard_passed=bool(interior.any() and np.all(errors[interior] <= ERROR_LIMIT)))
    row["original_fit_inlier_truth_interior"] = statistics(errors[interior & inlier]) if fit["model_available"] else None
    row["original_fit_outlier_truth_interior"] = statistics(errors[interior & ~inlier]) if fit["model_available"] else None
    return row, dict(arrays=values, p=p, q=q, errors=errors, interior=interior)


def capture_analysis(trace, case, accepted):
    capture = trace.get("capture")
    require(isinstance(capture, dict) and capture.get("error") is None and capture.get("method_calls") == 1,
            "Missing or invalid single-call trace")
    data = capture["data"]
    require("pyramids" in data, "Pyramids not captured")
    if capture.get("partial") is True:
        require(accepted is None and capture.get("expected_feature_unavailable") is True
                and capture["stages"] == ["pyramids"], "Unexpected partial capture")
        return dict(selected_count=None, truth_interior_selected_count=None,
                    immediate_forward_valid_count=None, final_accepted_count=None,
                    truth_interior_missing_count=None, selection_available=False), None
    require(capture["stages"] == ["pyramids", "before_forward", "after_forward", "after_backward", "readback", "final_filter"]
            and accepted is not None, "Incomplete full capture")
    seeds, final = data["before_forward"], data["final_filter"]
    points, scores, indices = (array(seeds[key]) for key in ("selected_points", "selected_scores", "selected_indices"))
    count = len(points)
    require(0 < count <= 384 and seeds["selected_count"] == count and points.dtype == np.dtype("float32")
            and points.shape == (count, 2) and scores.dtype == np.dtype("float32") and scores.shape == (count,)
            and indices.dtype.kind in "iu" and indices.shape == (count,) and len(np.unique(indices)) == count
            and np.isfinite(points).all() and np.isfinite(scores).all(), "Invalid selected seed inventory")
    mask = array(final["accepted_mask"])
    slots = array(final["selected_indices_of_accepted_points"])
    rejected = array(final["selected_indices_of_rejected_points"])
    require(mask.dtype == np.dtype(bool) and mask.shape == (count,) and slots.dtype.kind in "iu"
            and np.array_equal(slots, np.flatnonzero(mask)) and np.array_equal(rejected, np.flatnonzero(~mask)),
            "Accepted/rejected slot map differs")
    previous, current = array(final["previous_full_points"]), array(final["current_full_points"])
    require(previous.shape == current.shape == (count, 2) and np.array_equal(previous, (points+.5)*2-.5),
            "Selected proxy points were not converted to exact native coordinates")
    fields = dict(previous_points="accepted_previous", current_points="accepted_current", harris_scores="accepted_scores",
                  forward_backward_error_px="accepted_fb_error")
    for name, field in fields.items():
        require(same_array(final[field], trace["canonical_nontiming"]["result"]["correspondence"]["arrays"][name]),
                "Returned accepted array differs from captured final filter: "+name)
    require(np.array_equal(accepted["arrays"]["previous_points"], previous[mask])
            and np.array_equal(accepted["arrays"]["current_points"], current[mask])
            and np.array_equal(accepted["arrays"]["harris_scores"], scores[mask]), "Mask order does not reconstruct returned correspondence")
    fb = array(final["fb_error"])
    require(fb.shape == (count,) and np.array_equal(accepted["arrays"]["forward_backward_error_px"], fb[mask]), "FB mask mapping differs")
    status = array(data["after_forward"]["forward_status"]).reshape(-1)
    require(status.dtype == np.dtype("uint8") and status.shape == (count,), "Invalid immediate-forward status")
    cohort = truth_mask(previous, case["truth_displacement_xy"])
    accepted_cohort = cohort[slots]
    immediate = status == 0
    result = dict(selection_available=True, selected_count=count, truth_interior_selected_count=int(cohort.sum()),
        immediate_forward_valid_count=int(immediate.sum()), truth_interior_forward_valid_count=int((immediate & cohort).sum()),
        final_accepted_count=len(slots), truth_interior_final_accepted_count=int(accepted_cohort.sum()),
        truth_interior_missing_count=int(cohort.sum()-accepted_cohort.sum()),
        truth_interior_accepted_fraction=float(accepted_cohort.sum()/cohort.sum()) if cohort.any() else None,
        final_accepted_selected_slots=slots.tolist())
    return result, dict(points=points, scores=scores, indices=indices, slots=slots, cohort=cohort,
                       errors=accepted["errors"], displacement=accepted["q"]-accepted["p"],
                       accepted_interior=accepted_cohort)


def cross_depth_identity(four, two):
    a, b = four["canonical_nontiming"], two["canonical_nontiming"]
    for field in ("case_metadata", "input_pair", "frame_pixel_sha256", "global_configuration"):
        require(a[field] == b[field], "Cross-depth identity differs: "+field)
    ac, bc = a["effective_motion_configuration"], b["effective_motion_configuration"]
    require(ac.get("pyramid_levels") == 4 and bc.get("pyramid_levels") == 2 and set(ac) == set(bc)
            and {k for k in ac if ac[k] != bc[k]} == {"pyramid_levels"}, "Configuration differs beyond depth4 to2")
    ad, bd = four["capture"]["data"], two["capture"]["data"]
    for direction in ("previous", "current"):
        left, right = ad["pyramids"][direction], bd["pyramids"][direction]
        require(len(left) == 4 and len(right) == 2, "Pyramid depths differ")
        for i, item in enumerate(left):
            value = array(item["array"])
            require(value.dtype == np.dtype("uint8") and value.shape == ((256,320),(128,160),(64,80),(32,40))[i],
                    "Original pyramid dimensions differ")
        for i in range(2):
            require(same_array(left[i]["array"], right[i]["array"]), "Cross-depth pyramid prefix differs")
        require(same_array(ad["pyramids"]["proxy"][direction]["array"], bd["pyramids"]["proxy"][direction]["array"]),
                "Cross-depth proxy differs")
    seed_available = "before_forward" in ad and "before_forward" in bd
    require(("before_forward" in ad) == ("before_forward" in bd), "Cross-depth seed availability differs")
    if seed_available:
        for field in ("selected_points", "selected_scores", "selected_indices"):
            require(same_array(ad["before_forward"][field], bd["before_forward"][field]), "Cross-depth feature seed differs: "+field)
    return dict(pixels_and_configuration_equal_except_depth=True, proxy_and_pyramid_prefix_exact=True,
                selected_seed_identity_exact=True if seed_available else None,
                selection_unavailable_at_both_depths=not seed_available)


def matched_populations(left, right):
    """One-to-one selected-slot join; retain recoveries and losses explicitly."""
    if left is None or right is None:
        return dict(available=False, reason="Selected features unavailable; no fabricated denominator")
    require(np.array_equal(left["points"], right["points"]) and np.array_equal(left["scores"], right["scores"])
            and np.array_equal(left["indices"], right["indices"]), "Unmatched selected seed identities")
    li = {int(slot): i for i, slot in enumerate(left["slots"])}
    ri = {int(slot): i for i, slot in enumerate(right["slots"])}
    common = sorted(li.keys() & ri.keys())
    lost, recovered = sorted(li.keys()-ri.keys()), sorted(ri.keys()-li.keys())
    def values(item, lookup, slots):
        return item["errors"][[lookup[i] for i in slots]]
    common_interior = [i for i in common if left["cohort"][i] and right["cohort"][i]]
    vector_delta = np.array([np.linalg.norm(right["displacement"][ri[i]]-left["displacement"][li[i]]) for i in common_interior])
    return dict(available=True, selected_count=len(left["points"]), common_accepted_count=len(common),
        left_only_accepted_count=len(lost), right_only_accepted_count=len(recovered),
        common_accepted_selected_slots=common, left_only_selected_slots=lost, right_only_selected_slots=recovered,
        common_truth_interior_count=len(common_interior),
        left_common_truth_interior_error=statistics(values(left, li, common_interior)),
        right_common_truth_interior_error=statistics(values(right, ri, common_interior)),
        common_truth_interior_endpoint_delta=statistics(vector_delta),
        left_only_all_error=statistics(values(left, li, lost)), right_only_all_error=statistics(values(right, ri, recovered)),
        common_survivors_are_not_an_availability_metric=True)


def gates(case, analysis):
    kind = case["scientific_class"]
    fit = analysis["original_fit"]
    error = fit["translation_error_px"]
    return dict(global_positive_required=kind == "global_positive",
        global_positive_passed=bool(fit["accepted"] is True and error is not None and error <= ERROR_LIMIT) if kind == "global_positive" else None,
        degenerate_rejection_required=kind == "degenerate",
        degenerate_rejection_passed=fit["accepted"] is not True if kind == "degenerate" else None,
        point_guard_required=kind != "degenerate",
        point_guard_passed=analysis["point_guard_passed"] if kind != "degenerate" else None)


def inventory():
    """Independent exact corpus declaration; no executable worker import."""
    conditions = (("static", [0., 0.], 1., 0.), ("shift_half", [.5, -.75], 1., 0.),
        ("shift_two", [2., -1.], 1., 0.), ("shift_fractional", [-1.25, .5], 1., 0.),
        ("static_gain08_offset12", [0., 0.], .8, 12.), ("shift_half_gain08_offset12", [.5, -.75], .8, 12.),
        ("static_gain12_offsetm12", [0., 0.], 1.2, -12.), ("shift_half_gain12_offsetm12", [.5, -.75], 1.2, -12.))
    def item(identifier, family, truth, gain, offset, group, kind, source, ncc, reference):
        return dict(id=identifier, family=family, truth_displacement_xy=truth, gain=gain, offset_dn=offset,
            scientific_class=group, shape_hw=list(SHAPE), input_kind=kind, source_case_id=source,
            inherited_ncc_expectation=ncc, photometric_reference_id=reference)
    cases = []
    for family in ("texture", "corner"):
        for name, truth, gain, offset in conditions:
            identifier = family+"__"+name
            reference = family+"__"+("static" if truth == [0., 0.] else "shift_half") if gain != 1. else None
            cases.append(item(identifier, family, truth, gain, offset,
                "global_positive" if family == "texture" else "sparse_corner", "analytic", identifier,
                "qualified_with_error_at_most_0.25px", reference))
    for identifier, family, truth, group in (("flat", "flat", [.5, -.75], "degenerate"),
        ("straight_edge", "edge", [.5, -.75], "degenerate"), ("periodic_ambiguity", "periodic", [.5, -.75], "degenerate"),
        ("out_of_range_shift4", "texture", [4., 0.], "global_positive")):
        cases.append(item(identifier, family, truth, 1., 0., group, "analytic", identifier, "abstain", None))
    for name, source, truth in (("static", "low_contrast_static", [0., 0.]),
                              ("shift_4_m2", "low_contrast_translated", [4., -2.])):
        for suffix, gain, offset in (("", 1., 0.), ("_gain08_offset12", .8, 12.), ("_gain12_offsetm12", 1.2, -12.)):
            cases.append(item("bridge__"+name+suffix, "bridge", truth, gain, offset, "global_positive", "bridge",
                              source, None, "bridge__"+name if suffix else None))
    return cases


def validate_freeze(frozen):
    require(frozen.get("schema") == "pva_robustness.v1" and frozen.get("scope") == "generated_only_no_production_change"
            and frozen.get("cases") == inventory() and frozen.get("shape_hw") == list(SHAPE)
            and frozen.get("depths") == list(DEPTHS) and frozen.get("modes") == list(MODES), "Frozen corpus differs")
    require(frozen.get("safety") == dict(start_c=65, stop_c=75, phase_seconds=900, batch_seconds=3600,
            workers=1, automatic_retries=0) and frozen.get("gates") == dict(fit_translation_error_px=.25,
            point_error_px=.25, truth_interior_margin_px=128), "Frozen limits/gates differ")
    require(frozen.get("helpers") == dict(
        depth_path="/tmp/seaqr_pva_depth_control_20261001_HzshKd/probe_pva_depth_control.py",
        depth_sha256="48345393d7f2e8c760169946cdd6e88a8e018152fd403891d2fe26730de2ed0f",
        static_sha256="48cff34b8248fdcfbc447eda32774ff91f33f88b78bb62c6986b65c50d634907",
        photometric_sha256="569220932c8132944120ffc6026f59e01c47cb81cfd11d9c2875f890566f32f1",
        generator_sha256="b1d915604327cca4a5a7bf83a3eef9eca197d43ecd9cc8e2cfc63a1d61aeb97c"),
        "Original helper pins differ")
    sources = frozen.get("source_sha256")
    require(isinstance(sources, dict) and set(sources) == SOURCES
            and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values())
            and sources["batch_discovery_pair.py"] == "a8c5ebe468f638af7ae39cf876e816a462c0fab1f3673add0eb81e037a8504bf",
            "Frozen source identities differ")
    protocol = frozen.get("protocol", {})
    for key, expected in dict(fresh_processes=104, focal_estimate_calls_per_child=1, preflight_pva_calls=0,
        status_policy="legacy_default", only_depth_configuration_difference="pyramid_levels",
        base_trace_exact_parity_required=True, cross_depth_input_seed_prefix_identity_required=True,
        science_failure_is_execution_failure=False, production_promotion=False, ncc_expectations_are_not_pva_gates=True).items():
        require(protocol.get(key) == expected and type(protocol.get(key)) is type(expected), "Frozen protocol differs: "+key)


def phase_name(case, depth, mode):
    return case+"_depth"+str(depth)+"_"+mode


def validate_child(row, case, depth, mode, digest, frozen):
    require(row.get("schema") == "seaqr.pva-robustness.v1" and row.get("completed") is True
            and row.get("passed_integrity") is True and row.get("case") == case["id"]
            and row.get("case_metadata") == case and row.get("depth") == depth and row.get("mode") == mode
            and row.get("input_sha256", {}).get("freeze_sha256") == digest
            and row.get("input_sha256", {}).get("source_sha256") == frozen["source_sha256"], "Child identity differs")
    for key, expected in dict(generated_only=True, source_media_accessed=False, detector_run=False,
                              production_changes=False, production_promotion=False, preflight_pva_calls=0,
                              focal_estimate_calls=1, closed=True, clocks_changed=False).items():
        require(row.get(key) == expected and type(row.get(key)) is type(expected), "Child scope differs: "+key)
    require(row.get("error") is None and row.get("cleanup_error") is None
            and row["input_sha256"].get("helpers") == frozen["helpers"], "Child error/helper identity differs")
    canonical = row.get("canonical_nontiming")
    require(isinstance(canonical, dict) and canonical_sha(canonical) == row.get("canonical_nontiming_sha256")
            and canonical.get("case_metadata") == case, "Child canonical identity differs")
    motion = canonical.get("effective_motion_configuration", {})
    require(motion.get("pyramid_levels") == depth and motion.get("feature_image_scale") == .5
            and motion.get("pyramid_scale") == .5 and motion.get("flow_status_policy") == "legacy_default"
            and motion.get("forward_backward_check") is True and motion.get("max_features") == 384
            and motion.get("max_features_per_cell") == 8 and motion.get("minimum_accepted_features") == 30
            and motion.get("minimum_grid_coverage") == .2 and motion.get("window_size") == 11,
            "Child feature/motion configuration differs")
    require(canonical.get("global_configuration") == EXPECTED_GLOBAL, "Original global quality gates differ")
    require(row.get("global_fits") == (0 if canonical["result"].get("correspondence") is None else 1),
            "Exactly one original fit per returned correspondence required")
    observation = row.get("observation_contract", {})
    require(observation.get("method_sha256") == "bfe680841a214acdddb16c3f12bc166921760613c3446a0a6c2ca8ac37e7286b"
            and observation.get("estimator_source_changed") is False and observation.get("initial_status_changed") is False
            and observation.get("extra_pva_compute_calls") == 0 and observation.get("sys_settrace") is (mode == "trace")
            and observation.get("additional_cpu_readbacks") is (mode == "trace"), "Observer contract differs")
    pair = canonical.get("input_pair", {})
    require(pair.get("native_shape_hw") == list(SHAPE) and pair.get("source_media_accessed") is False,
            "Generated input pair scope differs")
    for field in ("previous_pixel_sha256", "current_pixel_sha256", "original_current_pixel_sha256"):
        require(isinstance(pair.get(field), str) and re.fullmatch(r"[a-f0-9]{64}", pair[field]), "Invalid generated pixel identity")
    quant = pair.get("quantization", {})
    require(quant.get("clipped_pixels") == 0 and all(type(value) in (int, float) and math.isfinite(value)
            and 0 <= value <= .5 for key, value in quant.items() if key.endswith("rounding_error_dn")), "Quantization differs")
    require("current_maximum_abs_rounding_error_dn" in quant, "Missing quantization evidence")
    frame = canonical.get("frame_pixel_sha256", {})
    require(set(frame) == {"previous", "current"} and all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                                                       for v in frame.values()), "Invalid Frame pixel identity")


def summarize(directory, freeze_sha256, bundle):
    directory, bundle = Path(directory).absolute(), Path(bundle).absolute()
    require(directory.is_dir() and bundle.is_dir() and not directory.is_symlink() and not bundle.is_symlink(),
            "Real metadata and bundle directories required")
    require(isinstance(freeze_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", freeze_sha256), "Caller freeze SHA required")
    bindings = {}
    def bound(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, "Evidence hash differs: "+path.name)
        value = read(path)
        require(sha(path) == digest, "Evidence changed during read: "+path.name)
        bindings[str(path)] = digest
        return value
    frozen = bound(directory/"freeze.json", freeze_sha256)
    validate_freeze(frozen)
    for name, digest in frozen["source_sha256"].items():
        path = bundle/name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Frozen source file differs: "+name)
        bindings[str(path)] = digest
    require(sha(Path(__file__)) == frozen["source_sha256"]["summarize_pva_robustness.py"], "Running summarizer differs from freeze")
    batch = bound(directory/"batch_status.json")
    require(batch.get("schema") == "seaqr.pva-robustness.batch.v1" and batch.get("complete") is True
            and batch.get("execution_passed") is True and batch.get("generated_only") is True
            and batch.get("camera_media_accessed") is False and batch.get("clock_writes") is False
            and batch.get("error") is None and batch.get("not_run") == [] and batch.get("current") is None
            and batch.get("freeze_sha256") == freeze_sha256 and batch.get("source_sha256") == frozen["source_sha256"],
            "Complete, unmodified generated batch required; no partial summary")
    expected_phases = [(case["id"], depth, mode) for case in inventory() for depth in DEPTHS for mode in MODES]
    phases = batch.get("phases", [])
    require([(p["case"], p["depth"], p["mode"]) for p in phases] == expected_phases
            and len(phases) == 104 and len({p["pid"] for p in phases}) == 104
            and all(type(p["pid"]) is int and p["pid"] > 0 and p["returncode"] == 0 for p in phases),
            "104 completed fresh processes required")
    require(0 <= batch["elapsed_seconds"] < 3600 and all(0 <= p["elapsed_seconds"] < 900 for p in phases),
            "Recorded execution exceeded frozen bounds")
    require(batch.get("execution") == dict(workers=1, phases=104, phase_deadline_seconds=900,
        batch_deadline_seconds=3600, start_below_celsius=65, stop_at_celsius=75, automatic_retries=0),
        "Batch execution policy differs")
    parity = bound(directory/"parity.json", batch["parity_sha256"])
    require(parity.get("schema") == "seaqr.pva-robustness.batch.v1.parity"
            and [(p["case"], p["depth"]) for p in parity.get("cases", [])]
            == [(case["id"], depth) for case in inventory() for depth in DEPTHS], "52 parity pairs required")
    rows, definitions = {}, {case["id"]: case for case in inventory()}
    for phase in phases:
        key = (phase["case"], phase["depth"], phase["mode"])
        require(phase["name"] == phase_name(*key), "Phase filename differs")
        command = phase.get("command", [])
        require(isinstance(command, list) and len(command) in (13, 14) and command[0] == "/usr/bin/python3"
                and command[1] == "-I" and command[3] == "--workspace"
                and re.fullmatch(r"/tmp/seaqr_pva_robustness_20261001_[A-Za-z0-9]{6}", command[4]),
                "Scoped isolated child command differs")
        workspace = command[4]
        expected_command = ["/usr/bin/python3", "-I", workspace+"/probe_pva_robustness.py", "--workspace", workspace,
                            "--freeze", workspace+"/freeze.json", "--freeze-sha256", freeze_sha256,
                            "--case", key[0], "--depth", str(key[1])] + (["--trace"] if key[2] == "trace" else [])
        require(command == expected_command, "Recorded case/depth/mode/freeze command differs")
        row = bound(directory/(phase["name"]+".json"), phase["result_sha256"])
        validate_child(row, definitions[key[0]], key[1], key[2], freeze_sha256, frozen)
        rows[key] = row
    parity_checks = {}
    for item in parity["cases"]:
        identifier, depth = item["case"], item["depth"]
        base, trace = (rows[identifier, depth, mode] for mode in MODES)
        same = base["canonical_nontiming"] == trace["canonical_nontiming"]
        require(same == (base["canonical_nontiming_sha256"] == trace["canonical_nontiming_sha256"])
                and item["passed"] is same and item["exact_nontiming_parity"] is same
                and item["baseline_sha256"] == base["canonical_nontiming_sha256"]
                and item["trace_sha256"] == trace["canonical_nontiming_sha256"], "Recorded parity differs")
        parity_checks[identifier, depth] = same
    require(batch["parity_passed"] is all(parity_checks.values()) and parity["passed"] is all(parity_checks.values()),
            "Overall parity differs")
    cases, internals, identity_failures = [], {}, []
    for case in inventory():
        identifier = case["id"]
        comparison = dict(case=case, depths={}, cross_depth_identity=None, cross_depth_populations=None)
        for depth in DEPTHS:
            base, trace = (rows[identifier, depth, mode] for mode in MODES)
            analysis, accepted = accepted_analysis(base["canonical_nontiming"], case)
            same = parity_checks[identifier, depth]
            analysis.update(exact_base_trace_parity=same, gates=gates(case, analysis), availability=None)
            if same:
                availability, internals[identifier, depth] = capture_analysis(trace, case, accepted)
                analysis["availability"] = availability
            else:
                internals[identifier, depth] = None
                identity_failures.append(dict(case=identifier, depth=depth, reason="Base/trace canonical parity failed"))
            comparison["depths"][str(depth)] = analysis
        if all(parity_checks[identifier, depth] for depth in DEPTHS):
            try:
                comparison["cross_depth_identity"] = cross_depth_identity(rows[identifier, 4, "trace"], rows[identifier, 2, "trace"])
                comparison["cross_depth_populations"] = matched_populations(internals[identifier, 4], internals[identifier, 2])
            except ValueError as exc:
                identity_failures.append(dict(case=identifier, reason=str(exc)))
        cases.append(comparison)
    photometric = []
    for case in inventory():
        reference = case["photometric_reference_id"]
        if reference is None:
            continue
        for depth in DEPTHS:
            a, b = rows[reference, depth, "trace"], rows[case["id"], depth, "trace"]
            item = dict(control_case=reference, photometric_case=case["id"], depth=depth, common_seed_comparison=None)
            if parity_checks[reference, depth] and parity_checks[case["id"], depth]:
                require(a["canonical_nontiming"]["input_pair"]["previous_pixel_sha256"]
                        == b["canonical_nontiming"]["input_pair"]["previous_pixel_sha256"], "Photometric previous pixels differ")
                try:
                    item["common_seed_comparison"] = matched_populations(internals[reference, depth], internals[case["id"], depth])
                except ValueError as exc:
                    item["seed_identity_failed"] = str(exc)
                    identity_failures.append(dict(case=case["id"], depth=depth, reason="Photometric "+str(exc)))
            photometric.append(item)
    gate_failures = []
    totals = {}
    for depth in DEPTHS:
        total = {}
        for category, required in (("global_positive", 15), ("degenerate_rejection", 3), ("point_guard", 23)):
            evaluated = [row for row in cases if row["depths"][str(depth)]["gates"][category+"_required"]]
            require(len(evaluated) == required, "Scientific gate inventory differs")
            failures = [row["case"]["id"] for row in evaluated if not row["depths"][str(depth)]["gates"][category+"_passed"]]
            total[category] = dict(required=required, passed=required-len(failures), failed_case_ids=failures)
            if depth == 2:
                gate_failures.extend(dict(case=case, gate=category) for case in failures)
        totals[str(depth)] = total
    require(all(sha(Path(path)) == digest for path, digest in bindings.items()), "Inputs changed during metadata summary")
    ready = not identity_failures and not gate_failures
    return dict(schema=SCHEMA, completed=True, metadata_only=True, source_media_accessed=False,
        production_changes=False, production_promotion=False, numerical_refits=0, freeze_sha256=freeze_sha256,
        input_sha256=bindings, cases_completed=26, children_completed=104, base_trace_pairs_checked=52,
        all_base_trace_parity_passed=all(parity_checks.values()), identity_failures=identity_failures,
        scientific_gates=totals, depth2_scientific_failures=gate_failures,
        ready_for_bounded_real_video_validation=ready, cases=cases, photometric_comparisons=photometric,
        limitations=["Generated truth does not establish real-video accuracy, airborne identity, FPR, or throughput.",
            "Depth2 passes only if all15 global positives,3 negative guards,23 nonvacuous point guards and all identities pass.",
            "The shared128px truth-only interior is not an exact pyramid receptive-field model; all-accepted errors are also reported.",
            "Missing/rejected points have no error measurement, never zero error; common survivors are a selected population.",
            "Periodic control includes Nyquist aliasing; corner controls lack30-point global support.",
            "Four-pixel motion is outside NCC search, not outside PVA's motion range.",
            "Affine gain and offset change together with rounding; this does not isolate their causal contributions.",
            "Legacy initial forward status is implicit; base/trace parity does not establish all backend state independence.",
            "Old failed controls remain separate immutable evidence; this report does not supersede their outcomes."],
        arithmetic_identity_atol_px=ARITHMETIC_ATOL, scientific_error_gate_atol_px=0.)


def save_summary(directory, freeze_sha256, bundle, output):
    output = Path(output).absolute()
    require(output.suffix == ".json" and output.parent.is_dir() and output.parent.resolve() == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh JSON output required; no overwrite")
    result = summarize(directory, freeze_sha256, bundle)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = save_summary(args.directory, args.freeze_sha256, args.bundle, args.output)
    print(json.dumps({key: result[key] for key in ("completed", "children_completed",
        "all_base_trace_parity_passed", "ready_for_bounded_real_video_validation", "depth2_scientific_failures")}))
