#!/usr/bin/env python3
"""Frozen, local-only descriptions of saved AOT correspondences; never refits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "seaqr.aot.residual-patterns.v1"
PLAN_SCHEMA = "seaqr.aot.residual-patterns-plan.v1"
REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO.parent / "outputs/seaqr_aot_pilot_20260927"
OUTPUT_DIR = EVIDENCE_ROOT / "residual_patterns_01"
PREVIOUS_INDICES = (0, 42, 85, 127, 170, 212, 255, 298)
SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))
ARM = "half_gain16_complete"
WIDTH, HEIGHT = 2448, 2048
SEED, REFERENCES, CASE_COUNT = 20260928, 199, 64
INPUT_PINS = {
    "factor_result_sha256": ("feature_factors_01/result.json", "aaeb0cb9898199548afe9c8303d27717c1694c596dec2b775df6355d24584f56"),
    "factor_audit_sha256": ("feature_factors_01/audit.json", "3f351436a132d4eedc00225aebf3f3ddeacaaf1ab6034edfb563601dd240f478"),
    "natural_result_sha256": ("natural_shifts_01/result.json", "56b9f1a363b449706c38aee4453cae1e4eb9c0316576203fbf8f2a03721bf54f"),
    "natural_audit_sha256": ("natural_shifts_01/audit.json", "a1b6632b4a635311d89dedc0bbe8a6cc85a17b9ed870a5a2d123ec714c71e51d"),
}
DESIGN = dict(arm=ARM, previous_indices=list(PREVIOUS_INDICES), shifts=[list(x) for x in SHIFTS],
    case_count=CASE_COUNT, case_order="actual8_then_natural56", width=WIDTH, height=HEIGHT,
    grid_rows=6, grid_cols=8, grid_min_points=5, neighbor_k=5, neighbor_radius_px=200,
    permutation_references=REFERENCES, seed=SEED, score_quantile_method="linear",
    score_bin_side="right", known_truth_margin_px=128, new_fit=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scope_output(path):
    path = Path(path)
    require(path == OUTPUT_DIR and path.resolve() == OUTPUT_DIR, "outside fixed local output scope")
    require(path.is_dir() and not path.is_symlink(), "missing/linked output directory")
    for name in ("result.json", "failure.json"):
        require(not (path / name).exists() and not (path / name).is_symlink(), "existing output; refusing overwrite")
    return path


def artifact_hashes():
    paths = dict(script_sha256=Path(__file__), tests_sha256=REPO / "tests/test_aot_residual_patterns.py",
                 plan_sha256=REPO / "docs/aot_residual_patterns_plan_20260928.md")
    require(all(p.is_file() and not p.is_symlink() for p in paths.values()), "missing/linked artifact")
    return {key: sha(path) for key, path in paths.items()}


def input_hashes():
    found = {}
    for key, (relative, expected) in INPUT_PINS.items():
        path = EVIDENCE_ROOT / relative
        require(path.is_file() and not path.is_symlink(), "missing/linked capture or audit")
        found[key] = sha(path)
        require(found[key] == expected, f"frozen input identity differs: {key}")
    return found


def validate_manifest(manifest, actual_hashes):
    require(manifest.get("schema") == PLAN_SCHEMA, "wrong residual plan schema")
    # Canonical JSON distinguishes booleans from integer values in fixed design.
    require(json.dumps(manifest.get("design"), sort_keys=True) == json.dumps(DESIGN, sort_keys=True),
            "fixed residual design differs")
    for key, (_, expected) in INPUT_PINS.items():
        require(manifest.get(key) == actual_hashes.get(key) == expected, f"manifest input differs: {key}")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        value = actual_hashes.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                and manifest.get(key) == value, f"manifest artifact differs: {key}")


def vectors(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all(), "finite Nx2 required")
    return value


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "finite scalar array required")
    keys, quantiles = ("p10", "p50", "p90", "p95", "p99", "max"), (.1, .5, .9, .95, .99, 1)
    return dict(count=len(values), quantiles=dict(zip(keys, np.quantile(values, quantiles).tolist())) if len(values) else None)


def norm_summary(value):
    value = vectors(value)
    summary = scalar_summary(np.linalg.norm(value, axis=1))
    return dict(count=len(value), component_median=np.median(value, axis=0).tolist() if len(value) else None,
                norm_quantiles=summary["quantiles"])


def cell_indices(points, rows=6, cols=8):
    points = vectors(points)
    require(np.all((points >= 0) & (points < [WIDTH, HEIGHT])), "previous points outside native extent")
    return (np.floor(points[:, 1] * rows / HEIGHT).astype(int) * cols
            + np.floor(points[:, 0] * cols / WIDTH).astype(int))


def grid_coverage(points):
    counts = np.bincount(cell_indices(points), minlength=48)
    return dict(occupied_cells=int(np.count_nonzero(counts)), total_cells=48, counts=counts.tolist())


def cell_medians(points, value, rows=6, cols=8, min_points=5):
    points, value = vectors(points), vectors(value)
    require(len(points) == len(value), "grid vector count differs")
    indices = cell_indices(points, rows, cols)
    output = []
    for index in range(rows * cols):
        selected = value[indices == index]
        median = np.median(selected, axis=0) if len(selected) >= min_points else None
        output.append(dict(row=index // cols, column=index % cols, count=len(selected),
            median_vector=None if median is None else median.tolist(),
            median_scatter_px=None if median is None else float(np.median(np.linalg.norm(selected - median, axis=1)))))
    return dict(rows=rows, cols=cols, min_points=min_points, extent_xy=[0, WIDTH, 0, HEIGHT], cells=output)


def neighbor_graph(points, k=5, radius=200):
    points = vectors(points)
    require(type(k) is int and k > 0 and np.isfinite(radius) and radius >= 0, "invalid neighborhood")
    anchors, neighbors, distances = [], [], []
    indices = np.arange(len(points))
    for index, point in enumerate(points):
        squared = np.sum((points - point) ** 2, axis=1)
        order = np.lexsort((indices, squared))
        order = order[(order != index) & (squared[order] <= radius ** 2)]
        if len(order) >= k:
            anchors.append(index)
            neighbors.append(order[:k])
            distances.extend(np.sqrt(squared[order[:k]]).tolist())
    return dict(anchors=np.asarray(anchors, dtype=np.int64),
        neighbors=np.asarray(neighbors, dtype=np.int64).reshape(-1, k), point_count=len(points),
        anchor_count=len(anchors), excluded_count=len(points) - len(anchors),
        anchor_fraction=len(anchors) / len(points) if len(points) else None,
        directed_edge_count=len(anchors) * k, neighbor_distance_summary=scalar_summary(distances))


def coherence_statistic(value, graph):
    value = vectors(value)
    require(len(value) == graph["point_count"], "graph vector count differs")
    if not len(graph["anchors"]):
        return None
    local = np.median(value[graph["neighbors"]], axis=1)
    return float(np.median(np.linalg.norm(value[graph["anchors"]] - local, axis=1)))


def spatial_coherence(points, value, seed, references=199):
    points, value = vectors(points), vectors(value)
    require(type(references) is int and references > 0, "invalid reference count")
    graph = neighbor_graph(points)
    observed = coherence_statistic(value, graph)
    rng = np.random.default_rng(seed)
    null = [coherence_statistic(value[rng.permutation(len(value))], graph) for _ in range(references)] if observed is not None else []
    quantiles = dict(zip(("p5", "p50", "p95"), np.quantile(null, [.05, .5, .95]).tolist())) if null else None
    median = quantiles["p50"] if quantiles is not None else None
    return dict(**{key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in graph.items()},
        seed=seed, requested_references=references, observed=observed, reference_values=null,
        reference_quantiles=quantiles, observed_to_reference_median=observed / median if median else None,
        interpretation="joint random-label references on fixed graph; not a p-value; translation cancels in neighbor differences")


def score_cuts(scores):
    scores = np.asarray(scores)
    require(scores.ndim == 1 and scores.dtype == np.uint32 and len(scores) > 0, "nonempty exact U32 scores required")
    return np.quantile(scores, [.25, .5, .75], method="linear").tolist()


def score_strata(scores, accepted_mask, cuts, accepted_values=None, inlier_mask=None,
                 cohort_mask=None, points=None, truth_errors=None):
    """Primary counts use all selected points; truth has a separate fixed cohort."""
    scores, accepted = np.asarray(scores), np.asarray(accepted_mask)
    require(scores.ndim == 1 and scores.dtype == np.uint32 and accepted.dtype == bool
            and accepted.shape == scores.shape, "invalid selected scores/acceptance")
    cuts = np.asarray(cuts, dtype=float)
    require(cuts.shape == (3,) and np.isfinite(cuts).all() and np.all(np.diff(cuts) >= 0), "invalid score cuts")
    bins = np.searchsorted(cuts, scores, side="right")
    count = int(accepted.sum())
    if accepted_values is not None:
        accepted_values = vectors(accepted_values)
        require(len(accepted_values) == count, "accepted values must follow selected acceptance order")
    if inlier_mask is not None:
        inlier_mask = np.asarray(inlier_mask)
        require(inlier_mask.dtype == bool and inlier_mask.shape == (count,), "invalid accepted inlier mask")
    if points is not None:
        points = vectors(points)
        require(len(points) == len(scores), "selected point count differs")
    if cohort_mask is not None:
        cohort_mask = np.asarray(cohort_mask)
        require(cohort_mask.dtype == bool and cohort_mask.shape == scores.shape, "invalid fixed cohort")
        truth_errors = np.asarray(truth_errors, dtype=float)
        require(truth_errors.shape == (count,) and np.isfinite(truth_errors).all()
                and np.all(truth_errors >= 0), "finite accepted truth errors required")
    output = []
    for index in range(4):
        selected_bin = bins == index
        accepted_bin = selected_bin[accepted]
        selected_count, accepted_count = int(selected_bin.sum()), int(accepted_bin.sum())
        row = dict(bin=index, selected_count=selected_count, accepted_count=accepted_count,
            lost_count=selected_count - accepted_count,
            acceptance_fraction=accepted_count / selected_count if selected_count else None,
            loss_fraction=(selected_count - accepted_count) / selected_count if selected_count else None,
            accepted_value_summary=None if accepted_values is None else norm_summary(accepted_values[accepted_bin]))
        if points is not None:
            row.update(selected_grid_coverage=grid_coverage(points[selected_bin]),
                       accepted_grid_coverage=grid_coverage(points[selected_bin & accepted]))
        if inlier_mask is not None:
            row.update(saved_inlier_count=int(np.sum(accepted_bin & inlier_mask)),
                saved_outlier_count=int(np.sum(accepted_bin & ~inlier_mask)),
                saved_inlier_value_summary=None if accepted_values is None else norm_summary(accepted_values[accepted_bin & inlier_mask]),
                saved_outlier_value_summary=None if accepted_values is None else norm_summary(accepted_values[accepted_bin & ~inlier_mask]))
        if cohort_mask is not None:
            cohort_bin = selected_bin & cohort_mask
            surviving = cohort_bin[accepted]
            denominator, numerator = int(cohort_bin.sum()), int(surviving.sum())
            errors = truth_errors[surviving]
            row["fixed_support_truth"] = dict(selected_count=denominator, accepted_count=numerator,
                lost_count=denominator - numerator, acceptance_fraction=numerator / denominator if denominator else None,
                accepted_error_summary=scalar_summary(errors), accepted_within_0_1_count=int(np.sum(errors <= .1)),
                accepted_within_0_5_count=int(np.sum(errors <= .5)),
                within_0_1_fraction_of_selected=float(np.sum(errors <= .1)) / denominator if denominator else None,
                within_0_5_fraction_of_selected=float(np.sum(errors <= .5)) / denominator if denominator else None)
        output.append(row)
    return dict(cuts=cuts.tolist(), tie_policy="searchsorted right; equal scores stay together; empty bins retained",
                primary_denominator="all selected previous features", bins=output)


def captured_array(record, dtype, columns=None, limit=19866):
    """Only finite, losslessly readable selection/Harris arrays, never failed flow."""
    require(record.get("dtype") == dtype, "captured array dtype differs")
    shape = record.get("shape")
    require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape)
            and len(shape) == (1 if columns is None else 2) and shape[0] <= limit
            and (columns is None or shape[1] == columns), "invalid captured array shape")
    array = np.asarray(record["values"], dtype=np.dtype(dtype)).reshape(shape)
    require(np.isfinite(array).all(), "nonfinite selection/Harris array")
    require(hashlib.sha256(array.tobytes()).hexdigest() == record.get("sha256"), "captured finite array hash differs")
    return array


def unpack_case(row):
    require(row.get("arm", {}).get("id") == ARM and row.get("completed") is True,
            "wrong/incomplete case arm")
    capture = row["capture"]
    require(capture["proxy"]["previous"]["shape"] == [HEIGHT // 2, WIDTH // 2], "wrong proxy dimensions")
    selected = capture["selection"]
    raw_points = captured_array(capture["harris"]["coordinates"], "float32", 2)
    raw_scores = captured_array(capture["harris"]["scores"], "uint32")
    indices = captured_array(selected["selected_indices"], "int64", limit=1000)
    proxy_points = captured_array(selected["coordinates"], "float32", 2, 1000)
    require(len(raw_points) == len(raw_scores) and len(indices) == len(proxy_points) == selected["selected_count"]
            and len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(raw_points)))
            and np.array_equal(proxy_points, raw_points[indices]), "selected/raw feature correspondence differs")
    # Match frozen float32 pixel-center lifting, then describe with float64 arithmetic.
    native = ((proxy_points + np.float32(.5)) * np.float32(2) - np.float32(.5)).astype(np.float64)
    cell_indices(native)
    selected_scores = raw_scores[indices]
    lookup = {tuple(point): index for index, point in enumerate(native)}
    require(len(lookup) == len(native), "duplicate selected native coordinate")
    correspondence = row.get("correspondence") or {}
    pairs = correspondence.get("correspondences", [])
    p = vectors(np.asarray([pair["previous_xy"] for pair in pairs]).reshape(-1, 2))
    q = vectors(np.asarray([pair["current_xy"] for pair in pairs]).reshape(-1, 2))
    require(len(p) == correspondence.get("accepted_count"), "accepted count differs")
    require(all(tuple(point) in lookup for point in p), "accepted point not in selected inventory")
    accepted_indices = np.asarray([lookup[tuple(point)] for point in p], dtype=np.int64)
    require(np.all(np.diff(accepted_indices) > 0), "accepted pairs must preserve unique selected order")
    accepted_mask = np.zeros(len(native), dtype=bool)
    accepted_mask[accepted_indices] = True
    fit = row.get("global_fit")
    inliers = [] if fit is None else fit.get("inlier_indices", [])
    require(isinstance(inliers, list) and all(type(index) is int and 0 <= index < len(p) for index in inliers)
            and len(set(inliers)) == len(inliers), "saved inliers must index accepted pair list")
    inlier_mask = np.zeros(len(p), dtype=bool)
    inlier_mask[inliers] = True
    translation = None
    if fit is not None:
        require(fit.get("model") == "translation", "saved model is not the frozen translation")
        parameters = fit.get("parameters") or {}
        xy = [parameters.get("translation_x_px"), parameters.get("translation_y_px")]
        if all(value is not None for value in xy):
            translation = np.asarray(xy, dtype=float)
            require(translation.shape == (2,) and np.isfinite(translation).all(), "invalid saved translation")
            matrix = fit.get("previous_to_current_matrix")
            if matrix is not None:
                require(np.array_equal(np.asarray(matrix, dtype=float),
                    np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])), "saved translation matrix differs")
    truth, cohort = None, None
    if row.get("kind") == "aot_known_shift":
        shift = row["expected_shift_xy"]
        require(isinstance(shift, list) and len(shift) == 2 and all(type(x) is int for x in shift)
                and tuple(shift) in SHIFTS, "unexpected known shift")
        truth = np.asarray(shift, dtype=float)
        target = native + truth
        cohort = np.all((native >= [128, 128]) & (native < [WIDTH - 128, HEIGHT - 128])
                        & (target >= [128, 128]) & (target < [WIDTH - 128, HEIGHT - 128]), axis=1)
        support = capture["preflow_support"]
        require(support["margin_px"] == 128 and support["expected_shift_xy"] == shift
                and support["selected_count"] == len(native) and support["fixed_support_count"] == int(cohort.sum())
                and support["mask"] == cohort.tolist()
                and support["selected_indices"] == np.flatnonzero(cohort).tolist(), "fixed pre-flow cohort differs")
    else:
        require(row.get("kind") == "aot_adjacent", "out-of-scope pair kind")
    identity = dict(native_previous=row["native_pixel_sha256"]["previous"],
        proxy_previous=capture["proxy"]["previous"]["sha256"], s16_previous=capture["s16"]["previous"]["sha256"],
        raw_points=capture["harris"]["coordinates"]["sha256"], raw_scores=capture["harris"]["scores"]["sha256"],
        selected_indices=selected["selected_indices"]["sha256"], selected_coordinates=selected["coordinates"]["sha256"],
        selected_exact_scores=hashlib.sha256(selected_scores.tobytes()).hexdigest())
    return dict(selected_points=native, selected_scores=selected_scores, accepted_mask=accepted_mask,
        accepted_indices=accepted_indices, accepted_points=p, current_points=q, inlier_mask=inlier_mask,
        translation=translation, known_truth=truth, cohort_mask=cohort, identity=identity)


def case_ids():
    return ([f"aot_{index + 1:03d}_adjacent" for index in PREVIOUS_INDICES]
            + [f"aot_prev{index:03d}_dx{shift[0]:+d}_dy{shift[1]:+d}" for index in PREVIOUS_INDICES for shift in SHIFTS])


def select_rows(factor, natural):
    require(factor.get("schema") == "seaqr.aot.feature-factors.v1" and factor.get("passed") is True
            and natural.get("schema") == "seaqr.aot.natural-shifts.v1" and natural.get("passed") is True,
            "source execution integrity not passed")
    actual = [row for row in factor["rows"] if row.get("kind") == "aot_adjacent" and row.get("arm", {}).get("id") == ARM]
    synthetic = [row for row in natural["rows"] if row.get("kind") == "aot_known_shift" and row.get("arm", {}).get("id") == ARM]
    rows = actual + synthetic
    require([row["case_id"] for row in rows] == case_ids(), "fixed 64-case inventory/order differs")
    require([row["previous_index"] for row in actual] == list(PREVIOUS_INDICES)
            and [row["previous_index"] for row in synthetic] == [index for index in PREVIOUS_INDICES for _ in SHIFTS]
            and [row["expected_shift_xy"] for row in synthetic] == [list(s) for _ in PREVIOUS_INDICES for s in SHIFTS],
            "fixed case source indices/shifts differ")
    return rows


def analyze_case(row, unpacked, cuts, ordinal):
    data = unpacked
    p, q = data["accepted_points"], data["current_points"]
    displacements = q - p
    residuals = None if data["translation"] is None else displacements - data["translation"]
    truth_error = None if data["known_truth"] is None else np.linalg.norm(displacements - data["known_truth"], axis=1)
    inliers = data["inlier_mask"]
    strata = score_strata(data["selected_scores"], data["accepted_mask"], cuts, residuals, inliers,
        data["cohort_mask"], data["selected_points"], truth_error)
    selected_count, accepted_count = len(data["selected_points"]), len(p)
    return dict(ordinal=ordinal, case_id=row["case_id"], source_kind=row["kind"], arm=ARM,
        previous_index=row["previous_index"], current_index=row["current_index"],
        source_previous=row.get("source_previous"), synthetic=row.get("synthetic"),
        previous_identity=data["identity"], original_fit=row.get("global_fit"),
        original_outcome=row.get("outcome"), original_error=row.get("error"),
        original_correspondence_metrics=(row.get("correspondence") or {}).get("metrics"),
        original_scientific_gates=row.get("scientific_gates"),
        expected_shift_xy=None if data["known_truth"] is None else data["known_truth"].tolist(),
        saved_candidate_translation=None if data["translation"] is None else data["translation"].tolist(),
        residual_available=residuals is not None,
        residual_unavailable_reason="saved candidate translation unavailable; no replacement fit" if residuals is None else None,
        selected_count=selected_count, accepted_count=accepted_count, lost_count=selected_count - accepted_count,
        acceptance_fraction=accepted_count / selected_count if selected_count else None,
        accepted_grid_coverage=grid_coverage(p),
        displacement_summary=norm_summary(displacements),
        accepted_displacement_fraction_above_sqrt20=float(np.mean(np.linalg.norm(displacements, axis=1) > np.sqrt(20))) if len(p) else None,
        candidate_residual_summary=None if residuals is None else norm_summary(residuals),
        saved_inlier_residual_summary=None if residuals is None else norm_summary(residuals[inliers]),
        saved_outlier_residual_summary=None if residuals is None else norm_summary(residuals[~inliers]),
        grid=None if residuals is None else cell_medians(p, residuals),
        coherence=None if residuals is None else spatial_coherence(p, residuals, SEED + ordinal, REFERENCES),
        score_strata=strata,
        points=dict(selected_previous_xy=data["selected_points"].tolist(), selected_harris_u32=data["selected_scores"].tolist(),
            accepted_selected_indices=data["accepted_indices"].tolist(), accepted_previous_xy=p.tolist(), accepted_current_xy=q.tolist(),
            displacement_xy=displacements.tolist(), candidate_residual_xy=None if residuals is None else residuals.tolist(),
            saved_inlier_mask=inliers.tolist(), fixed_support_mask=None if data["cohort_mask"] is None else data["cohort_mask"].tolist(),
            accepted_truth_error_px=None if truth_error is None else truth_error.tolist()))


def write_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def run(output):
    output = scope_output(output)
    manifest_path = output / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "missing/linked frozen manifest")
    before = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, before)
    receipt = dict(schema=SCHEMA, passed=False, design=DESIGN, hashes_before=before, rows=[],
        new_fit=False, media_accessed=False, labels_accessed=False, detector_run=False,
        remote_accessed=False, production_changed=False, production_promotion=False,
        limitations=["One development sequence; synthetic cases reuse eight textures, not independent scenes.",
            "Spatial metrics are conditional on original accepted matches, censored by status/bounds/120px/FB checks.",
            "Actual residuals use a saved, possibly rejected candidate; they are not truth errors.",
            "Saved inlier/outlier partitions are threshold-defined, not independent accuracy evidence.",
            "Old factor failed-flow JSON cannot recover nonfinite/subnormal bytes; no failed-flow vectors analyzed.",
            "Natural raw-byte preservation remains verified by pinned prior audit; only accepted finite points used here.",
            "Joint permutation references are descriptive, not population p-values; neighbor differences cancel translation.",
            "Score/location association and spatially coherent wrong matches can confound interpretation.",
            "Above sqrt(20) means outside the largest tested discrete shift magnitude, not a validated coverage boundary."])
    try:
        factor_audit = json.loads((EVIDENCE_ROOT / INPUT_PINS["factor_audit_sha256"][0]).read_text())
        natural_audit = json.loads((EVIDENCE_ROOT / INPUT_PINS["natural_audit_sha256"][0]).read_text())
        require(factor_audit.get("integrity_audit_passed") is True and natural_audit.get("passed") is True
                and factor_audit.get("result_sha256") == before["factor_result_sha256"]
                and natural_audit.get("input_sha256", {}).get("raw") == before["natural_result_sha256"],
                "pinned audits do not attest frozen results")
        with (EVIDENCE_ROOT / INPUT_PINS["factor_result_sha256"][0]).open() as stream:
            factor = json.load(stream)
        # Drop unrelated arms before loading the second large source JSON.
        factor["rows"] = [row for row in factor["rows"] if row.get("kind") == "aot_adjacent" and row.get("arm", {}).get("id") == ARM]
        with (EVIDENCE_ROOT / INPUT_PINS["natural_result_sha256"][0]).open() as stream:
            natural = json.load(stream)
        rows = select_rows(factor, natural)
        del factor, natural
        identities, cuts_by_source = {}, {}
        for ordinal, row in enumerate(rows):
            data = unpack_case(row)
            previous = row["previous_index"]
            cuts = score_cuts(data["selected_scores"])
            if ordinal < len(PREVIOUS_INDICES):
                identities[previous], cuts_by_source[previous] = data["identity"], cuts
            else:
                require(data["identity"] == identities[previous] and cuts == cuts_by_source[previous],
                        "previous-image/raw-feature/selected-score invariance differs")
            receipt["rows"].append(analyze_case(row, data, cuts_by_source[previous], ordinal))
            print(json.dumps(dict(case_id=row["case_id"], completed=ordinal + 1, total=CASE_COUNT)), flush=True)
        require(len(receipt["rows"]) == CASE_COUNT, "incomplete fixed analysis")
        receipt["source_invariance"] = dict(passed=True, groups=8, cases_per_group=8,
            identity_fields=list(next(iter(identities.values()))), score_cuts_by_previous_index=cuts_by_source)
        receipt["hashes_after"] = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(manifest_path)}
        require(before == receipt["hashes_after"], "inputs/artifacts changed during analysis")
        receipt["passed"] = True
        write_exclusive(output / "result.json", receipt)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        write_exclusive(output / "failure.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
