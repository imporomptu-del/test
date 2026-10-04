#!/usr/bin/env python3
"""Predeclared offline spatial holdout comparison; no production model change."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 2448, 2048
MODELS = ("translation", "similarity", "affine")
PARAMETER_COUNTS = dict(translation=2, similarity=4, affine=6)
CENTER = np.array([WIDTH / 2, HEIGHT / 2], dtype=np.float64)
COORDINATE_SCALE = float(WIDTH)
GUARD_PX, MIN_TRAIN, MIN_CELLS = 64, 100, 12
ITERATIONS, HUBER_DELTA = 20, 1.0
PREVIOUS_INDICES = (0, 42, 85, 127, 170, 212, 255, 298)
SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (4, -2), (-4, 2))
REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO.parent / "outputs/seaqr_aot_pilot_20260927"
OUTPUT_DIR = EVIDENCE_ROOT / "motion_models_01"
SCHEMA = "seaqr.aot.motion-models.v1"
PLAN_SCHEMA = "seaqr.aot.motion-models-plan.v1"
INPUT_PINS = {
    "input_result_sha256": ("residual_patterns_01/result.json", "a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc"),
    "input_audit_sha256": ("residual_patterns_01/audit.json", "af5ea40caf5956abf9753b1e05c570527be4a416de7bb9863355024809e6da0a"),
}
DESIGN = dict(models=list(MODELS), previous_indices=list(PREVIOUS_INDICES), shifts=[list(s) for s in SHIFTS],
    case_count=64, case_order="actual8_then_natural56", input_arm="half_gain16_complete",
    width=WIDTH, height=HEIGHT, fold_count=4, fold_order="top_left_top_right_bottom_left_bottom_right",
    guard_px=GUARD_PX, min_training_points=MIN_TRAIN, min_training_cells=MIN_CELLS,
    grid_rows=6, grid_cols=8, minimum_cell_evaluation_support=5,
    coordinate_center=CENTER.tolist(), coordinate_scale=COORDINATE_SCALE,
    initial_weighted_least_squares=True, huber_iterations=ITERATIONS, huber_delta_px=HUBER_DELTA,
    base_weight="inverse_training_cell_count", fit_point_deletion=False,
    original_inlier_filtering=False, automatic_model_selection=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def points_array(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all(), "finite Nx2 points required")
    return value


def native_cells(points):
    points = points_array(points)
    require(np.all((points >= 0) & (points < [WIDTH, HEIGHT])), "previous point outside native image")
    return np.floor(points[:, 1] * 6 / HEIGHT).astype(int) * 8 + np.floor(points[:, 0] * 8 / WIDTH).astype(int)


def fold_masks(points, fold_id):
    """Four half-open image quadrants, with a clipped 64px training exclusion."""
    points = points_array(points)
    native_cells(points)
    require(type(fold_id) is int and 0 <= fold_id < 4, "invalid spatial fold")
    row, column = divmod(fold_id, 2)
    x0, x1 = column * WIDTH / 2, (column + 1) * WIDTH / 2
    y0, y1 = row * HEIGHT / 2, (row + 1) * HEIGHT / 2
    ex0, ex1 = max(0, x0 - GUARD_PX), min(WIDTH, x1 + GUARD_PX)
    ey0, ey1 = max(0, y0 - GUARD_PX), min(HEIGHT, y1 + GUARD_PX)
    test = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    excluded = ((points[:, 0] >= ex0) & (points[:, 0] < ex1)
                & (points[:, 1] >= ey0) & (points[:, 1] < ey1))
    return dict(fold_id=fold_id, test_rectangle=[x0, y0, x1, y1],
        excluded_rectangle=[ex0, ey0, ex1, ey1], train_indices=np.flatnonzero(~excluded),
        test_indices=np.flatnonzero(test), guard_indices=np.flatnonzero(excluded & ~test))


def cell_weights(points):
    cells = native_cells(points)
    counts = np.bincount(cells, minlength=48)
    return dict(weights=1.0 / counts[cells], cell_indices=cells, counts=counts.tolist(),
                occupied_cells=int(np.count_nonzero(counts)))


def design_matrix(points, model):
    """Linear displacement basis; one fixed isotropic coordinate normalization."""
    points = points_array(points)
    require(model in MODELS, "unknown model")
    design = np.zeros((len(points), 2, PARAMETER_COUNTS[model]), dtype=np.float64)
    design[:, 0, 0] = 1
    design[:, 1, 1] = 1
    z = (points - CENTER) / COORDINATE_SCALE
    if model == "similarity":
        design[:, 0, 2], design[:, 0, 3] = z[:, 0], -z[:, 1]
        design[:, 1, 2], design[:, 1, 3] = z[:, 1], z[:, 0]
    elif model == "affine":
        design[:, 0, 2:4] = z
        design[:, 1, 4:6] = z
    return design


def predict(points, parameters, model):
    parameters = np.asarray(parameters, dtype=np.float64)
    require(model in MODELS and parameters.shape == (PARAMETER_COUNTS[model],)
            and np.isfinite(parameters).all(), "invalid model parameters")
    return np.einsum("nkp,p->nk", design_matrix(points, model), parameters)


def native_matrix(parameters, model):
    parameters = np.asarray(parameters, dtype=np.float64)
    require(model in MODELS and parameters.shape == (PARAMETER_COUNTS[model],)
            and np.isfinite(parameters).all(), "invalid matrix parameters")
    linear = np.zeros((2, 2), dtype=float)
    if model == "similarity":
        a, b = parameters[2:]
        linear[:] = [[a, -b], [b, a]]
    elif model == "affine":
        linear[:] = parameters[2:].reshape(2, 2)
    linear /= COORDINATE_SCALE
    matrix = np.eye(3)
    matrix[:2, :2] += linear
    matrix[:2, 2] = parameters[:2] - linear @ CENTER
    return matrix


def huber_weights(residuals, delta=1.0):
    residuals = points_array(residuals)
    require(np.isfinite(delta) and delta > 0, "invalid Huber delta")
    norms = np.linalg.norm(residuals, axis=1)
    result = np.ones(len(norms), dtype=float)
    large = norms > delta
    result[large] = delta / norms[large]
    return result


def fit_model(previous, current, model, base_weights=None, iterations=20, delta=1.0):
    """Initial weighted LS followed by exactly N joint-residual Huber solves."""
    previous, current = points_array(previous), points_array(current)
    require(len(previous) == len(current) and len(previous) > 0, "invalid training pair count")
    require(type(iterations) is int and iterations >= 0 and np.isfinite(delta) and delta > 0,
            "invalid fixed IRLS settings")
    design = design_matrix(previous, model)
    base = cell_weights(previous)["weights"] if base_weights is None else np.asarray(base_weights, dtype=float)
    require(base.shape == (len(previous),) and np.isfinite(base).all() and np.all(base > 0), "invalid training base weights")
    target = current - previous
    matrix = design.reshape(-1, design.shape[-1])
    final_weights = base.copy()
    parameters = None
    receipt = dict(valid=False, reason=None, model=model, iterations_requested=iterations,
        iterations_completed=0, solver_calls=0, huber_delta_px=float(delta), base_weights=base.tolist())
    try:
        for iteration in range(iterations + 1):
            if iteration:
                final_weights = base * huber_weights(target - predict(previous, parameters, model), delta)
            root = np.repeat(np.sqrt(final_weights), 2)
            parameters, _, rank, singular = np.linalg.lstsq(matrix * root[:, None], target.ravel() * root, rcond=None)
            receipt.update(solver_calls=iteration + 1, iterations_completed=iteration, rank=int(rank),
                singular_values=singular.tolist(),
                condition_number=float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None,
                final_solver_weights=final_weights.tolist())
            if rank != PARAMETER_COUNTS[model] or not np.isfinite(parameters).all():
                receipt["reason"] = "rank_deficient_or_nonfinite_solution"
                return receipt
        residual = target - predict(previous, parameters, model)
        native = native_matrix(parameters, model)
        receipt.update(valid=True, parameters=parameters.tolist(), native_matrix=native.tolist(),
            native_linear_determinant=float(np.linalg.det(native[:2, :2])),
            training_residual_xy=residual.tolist(), final_residual_huber_weights=huber_weights(residual, delta).tolist(),
            final_solver_weight_note="weights actually used for final solve; final-residual weights reported separately, not another solve")
    except np.linalg.LinAlgError as exc:
        receipt.update(reason="linear_algebra_failure", error=repr(exc))
    return receipt


def error_summary(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all() and np.all(values >= 0), "finite nonnegative errors required")
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
        mean=float(np.mean(values)) if len(values) else None,
        rms=float(np.sqrt(np.mean(values * values))) if len(values) else None,
        quantiles=dict(zip(("p10", "p50", "p90", "p95", "p99", "max"),
            np.quantile(values, [.1, .5, .9, .95, .99, 1]).tolist())) if len(values) else None)


def cell_errors(points, errors):
    cells = native_cells(points)
    errors = np.asarray(errors, dtype=float)
    require(errors.shape == (len(cells),), "cell error count differs")
    output = []
    for index in range(48):
        sample = errors[cells == index]
        require(np.isfinite(sample).all() and np.all(sample >= 0), "invalid per-cell errors")
        output.append(dict(row=index // 8, column=index % 8, count=len(sample),
            median=float(np.median(sample)) if len(sample) >= 5 else None,
            p90=float(np.quantile(sample, .9)) if len(sample) >= 5 else None))
    return output


def evaluate_fold(previous, current, model, fold_id):
    previous, current = points_array(previous), points_array(current)
    require(len(previous) == len(current), "pair count differs")
    fold = fold_masks(previous, fold_id)
    train, test = fold["train_indices"], fold["test_indices"]
    balance = cell_weights(previous[train])
    result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fold.items()}
    result.update(model=model, train_count=len(train), test_count=len(test), guard_count=len(fold["guard_indices"]),
        train_cell_counts=balance["counts"], train_occupied_cells=balance["occupied_cells"],
        valid=False, fit=None, unavailable_reasons=[], test_residual_xy=None, test_error_px=None,
        test_predicted_current_xy=None, test_error_summary=None, test_cell_errors=None)
    if len(train) < MIN_TRAIN:
        result["unavailable_reasons"].append("fewer_than_100_training_points")
    if balance["occupied_cells"] < MIN_CELLS:
        result["unavailable_reasons"].append("fewer_than_12_occupied_training_cells")
    if result["unavailable_reasons"]:
        return result
    fit = fit_model(previous[train], current[train], model, balance["weights"], ITERATIONS, HUBER_DELTA)
    result["fit"] = fit
    if not fit["valid"]:
        result["unavailable_reasons"].append(fit["reason"])
        return result
    if not len(test):
        result["unavailable_reasons"].append("empty_test_fold")
        return result
    predicted = previous[test] + predict(previous[test], fit["parameters"], model)
    residual = current[test] - predicted
    errors = np.linalg.norm(residual, axis=1)
    result.update(valid=True, test_predicted_current_xy=predicted.tolist(),
                  test_residual_xy=residual.tolist(), test_error_px=errors.tolist(),
                  test_error_summary=error_summary(errors), test_cell_errors=cell_errors(previous[test], errors))
    return result


def aggregate_folds(previous, folds, known_shift=None):
    """Keep the original evaluation denominator even when a fold cannot be fit."""
    previous = points_array(previous)
    expected = len(previous)
    require(len(folds) == 4 and [fold["fold_id"] for fold in folds] == list(range(4)), "incomplete fold inventory")
    inventory = [index for fold in folds for index in fold["test_indices"]]
    require(sorted(inventory) == list(range(expected)), "test folds do not partition accepted points exactly once")
    predicted = [None] * expected
    errors = [None] * expected
    truth_errors = [None] * expected if known_shift is not None else None
    truth = None if known_shift is None else np.asarray(known_shift, dtype=float)
    require(truth is None or truth.shape == (2,) and np.isfinite(truth).all(), "invalid known transform")
    for fold in folds:
        if truth is not None:
            fold["known_transform_prediction_error_px"] = None
            fold["known_transform_prediction_error_summary"] = None
        if not fold["valid"]:
            continue
        indices = np.asarray(fold["test_indices"], dtype=int)
        fold_predicted = points_array(fold["test_predicted_current_xy"])
        fold_errors = np.asarray(fold["test_error_px"], dtype=float)
        require(len(indices) == len(fold_predicted) == len(fold_errors), "fold prediction count differs")
        error_summary(fold_errors)
        for index, point, error in zip(indices, fold_predicted, fold_errors):
            predicted[index] = point.tolist()
            errors[index] = float(error)
        if truth is not None:
            mapping_errors = np.linalg.norm(fold_predicted - (previous[indices] + truth), axis=1)
            fold["known_transform_prediction_error_px"] = mapping_errors.tolist()
            fold["known_transform_prediction_error_summary"] = error_summary(mapping_errors)
            for index, error in zip(indices, mapping_errors):
                truth_errors[index] = float(error)
    scored = np.asarray([index for index, error in enumerate(errors) if error is not None], dtype=int)
    unscored = [index for index, error in enumerate(errors) if error is None]
    selected_errors = np.asarray([errors[index] for index in scored], dtype=float)
    cells = cell_errors(previous[scored], selected_errors)
    expected_cells = np.bincount(native_cells(previous), minlength=48)
    for index, cell in enumerate(cells):
        cell.update(expected_count=int(expected_cells[index]), unscored_count=int(expected_cells[index]) - cell["count"])
    complete = not unscored
    result = dict(complete=complete, expected_count=expected, scored_count=len(scored), unscored_count=len(unscored),
        valid_fold_count=sum(fold["valid"] for fold in folds), unavailable_fold_count=sum(not fold["valid"] for fold in folds),
        scored_indices=scored.tolist(), unscored_indices=unscored, predicted_current_xy=predicted, error_px=errors,
        error_summary=error_summary(selected_errors), summary_scope="full accepted case" if complete else "conditional scored subset; case incomplete, not rankable",
        cell_errors=cells, supported_cell_count=sum(cell["median"] is not None for cell in cells),
        worst_supported_cell_median=max((cell["median"] for cell in cells if cell["median"] is not None), default=None),
        worst_supported_cell_p90=max((cell["p90"] for cell in cells if cell["p90"] is not None), default=None))
    if truth is not None:
        result.update(known_transform_prediction_error_px=truth_errors,
            known_transform_prediction_error_summary=error_summary([truth_errors[index] for index in scored]),
            known_transform_interpretation="predicted mapping versus imposed shift on originally accepted test points; not raw LK error or recovered lost tracks")
    return result


def case_ids():
    return ([f"aot_{index + 1:03d}_adjacent" for index in PREVIOUS_INDICES]
            + [f"aot_prev{index:03d}_dx{shift[0]:+d}_dy{shift[1]:+d}" for index in PREVIOUS_INDICES for shift in SHIFTS])


def validate_case(row):
    points = row["points"]
    previous, current = points_array(points["accepted_previous_xy"]), points_array(points["accepted_current_xy"])
    selected = points_array(points["selected_previous_xy"])
    indices = np.asarray(points["accepted_selected_indices"])
    require(indices.ndim == 1 and indices.dtype.kind in "iu" and np.all(np.diff(indices) > 0)
            and len(indices) == len(previous) == len(current) == row["accepted_count"]
            and len(selected) == row["selected_count"] and row["lost_count"] == len(selected) - len(previous)
            and np.all((indices >= 0) & (indices < len(selected)))
            and np.array_equal(previous, selected[indices]), "source accepted/selected inventory differs")
    native_cells(previous)
    require(row.get("arm") == "half_gain16_complete", "wrong input arm")
    truth, support = None, None
    if row["source_kind"] == "aot_known_shift":
        shift = row["expected_shift_xy"]
        require(isinstance(shift, list) and len(shift) == 2
                and all(type(x) in (int, float) and np.isfinite(x) and float(x).is_integer() for x in shift)
                and tuple(shift) in SHIFTS, "unexpected known shift")
        truth = np.asarray(shift, dtype=float)
        mask = np.asarray(points["fixed_support_mask"])
        require(mask.dtype == bool and mask.shape == (len(selected),), "invalid frozen support mask")
        expected = np.all((selected >= 128) & (selected < [WIDTH - 128, HEIGHT - 128])
                          & (selected + truth >= 128) & (selected + truth < [WIDTH - 128, HEIGHT - 128]), axis=1)
        require(np.array_equal(mask, expected), "frozen pre-flow support differs")
        support = dict(selected_count=int(mask.sum()), accepted_count=int(mask[indices].sum()),
            lost_count=int(mask.sum() - mask[indices].sum()),
            denominator_basis="original selected p and expected p+shift within128px support; not new fit/test survival")
    else:
        require(row["source_kind"] == "aot_adjacent" and row.get("expected_shift_xy") is None
                and points.get("fixed_support_mask") is None, "actual case must not invent known truth")
    return previous, current, truth, support


def compare_case(row):
    previous, current, truth, support = validate_case(row)
    models = {}
    for model in MODELS:
        folds = [evaluate_fold(previous, current, model, fold_id) for fold_id in range(4)]
        models[model] = dict(folds=folds, out_of_fold=aggregate_folds(previous, folds, truth))
    baseline = models["translation"]["out_of_fold"]
    differences = {}
    for model in ("similarity", "affine"):
        candidate = models[model]["out_of_fold"]
        comparable = baseline["complete"] and candidate["complete"] and baseline["scored_count"] > 0
        differences[model] = dict(comparable_complete_cases=comparable,
            median_error_difference_vs_translation=(candidate["error_summary"]["median"] - baseline["error_summary"]["median"]) if comparable else None,
            p90_error_difference_vs_translation=(candidate["error_summary"]["quantiles"]["p90"] - baseline["error_summary"]["quantiles"]["p90"]) if comparable else None,
            interpretation="signed model minus common-protocol translation error; no automatic model selection")
    return dict(ordinal=row["ordinal"], case_id=row["case_id"], source_kind=row["source_kind"],
        previous_index=row["previous_index"], current_index=row["current_index"], arm=row["arm"],
        selected_count=row["selected_count"], accepted_count=row["accepted_count"], lost_count=row["lost_count"],
        original_fixed_support=support, expected_shift_xy=None if truth is None else truth.tolist(),
        original_fit=row.get("original_fit"), original_scientific_gates=row.get("original_scientific_gates"),
        original_candidate_translation=row.get("saved_candidate_translation"), models=models,
        descriptive_differences=differences)


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


def input_hashes():
    result = {}
    for key, (relative, expected) in INPUT_PINS.items():
        path = EVIDENCE_ROOT / relative
        require(path.is_file() and not path.is_symlink(), "missing/linked frozen input")
        result[key] = sha(path)
        require(result[key] == expected, f"input identity differs: {key}")
    return result


def artifact_hashes():
    paths = dict(script_sha256=Path(__file__), tests_sha256=REPO / "tests/test_aot_motion_models.py",
        plan_sha256=REPO / "docs/aot_motion_models_plan_20260928.md")
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()), "missing/linked frozen artifact")
    return {key: sha(path) for key, path in paths.items()}


def validate_manifest(manifest, hashes):
    require(manifest.get("schema") == PLAN_SCHEMA, "wrong motion-model plan schema")
    require(json.dumps(manifest.get("design"), sort_keys=True) == json.dumps(DESIGN, sort_keys=True), "fixed model design differs")
    for key, (_, expected) in INPUT_PINS.items():
        require(manifest.get(key) == hashes.get(key) == expected, f"manifest input differs: {key}")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        value = hashes.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                and manifest.get(key) == value, f"manifest artifact differs: {key}")


def write_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def run(output):
    output = scope_output(output)
    manifest_path = output / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "missing/linked frozen manifest")
    before = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(manifest_path)}
    validate_manifest(json.loads(manifest_path.read_text()), before)
    receipt = dict(schema=SCHEMA, passed=False, design=DESIGN, hashes_before=before, rows=[],
        numpy_version=np.__version__, local_saved_correspondences_only=True, media_accessed=False, remote_accessed=False,
        production_model_changed=False, original_quality_status_changed=False, detector_run=False,
        automatic_model_selection=False, production_promotion=False,
        limitations=["Originally accepted correspondences are censored by preceding feature/flow filters; lost matches remain lost.",
            "Agreement with saved actual correspondences is not physical camera truth, object truth, or detection accuracy.",
            "Spatial blocks are not independent recordings; eight scenes from one sequence and their reused synthetic textures.",
            "All three diagnostic estimators use identical fixed training weights/Huber protocol, not the old RANSAC fit.",
            "Similarity/affine numerical validity does not imply motion-quality acceptance or production suitability.",
            "Incomplete cases retain unscored denominators; conditional summaries cannot rank incomplete against complete cases."])
    try:
        with (EVIDENCE_ROOT / INPUT_PINS["input_result_sha256"][0]).open() as stream:
            source = json.load(stream)
        require(source.get("schema") == "seaqr.aot.residual-patterns.v1" and source.get("passed") is True,
                "source residual capture is not passed")
        rows = source["rows"]
        require([row["case_id"] for row in rows] == case_ids()
                and [row["ordinal"] for row in rows] == list(range(64)), "fixed case inventory/order differs")
        for ordinal, row in enumerate(rows):
            expected_previous = PREVIOUS_INDICES[ordinal] if ordinal < 8 else PREVIOUS_INDICES[(ordinal - 8) // 7]
            require(row["previous_index"] == expected_previous, "fixed source index differs")
            require(row["source_kind"] == ("aot_adjacent" if ordinal < 8 else "aot_known_shift"), "fixed source kind differs")
            if ordinal >= 8:
                require(row["expected_shift_xy"] == list(SHIFTS[(ordinal - 8) % 7]), "fixed synthetic shift differs")
            receipt["rows"].append(compare_case(row))
            print(json.dumps(dict(case_id=row["case_id"], completed=ordinal + 1, total=64)), flush=True)
        receipt["hashes_after"] = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(manifest_path)}
        require(receipt["hashes_after"] == before, "input/code/plan/manifest changed during comparison")
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
