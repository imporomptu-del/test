#!/usr/bin/env python3
"""Frozen numerical regional-motion study with explicit support abstention."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, GRID_ROWS, GRID_COLS = 2448, 2048, 6, 8
GUARD, RADIUS, ITERATIONS, HUBER_DELTA = 64, 512, 20, 1.0
COHERENT_RESIDUAL, COHERENT_MASS, MAX_CONDITION, HULL_TOLERANCE = 2.0, .6, 1000.0, 1e-8
ARMS = ("global_translation", "local_translation", "local_affine")
MODEL = dict(global_translation="translation", local_translation="translation", local_affine="affine")
PAIRS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))
PREVIOUS_INDICES = (0, 42, 85, 127, 170, 212, 255, 298)
REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO.parent / "outputs/seaqr_aot_pilot_20260927"
OUTPUT_DIR = EVIDENCE_ROOT / "regional_motion_01"
HELPER_PATH = REPO / "scripts/compare_aot_motion_models.py"
GENERATED_PATH = REPO / "scripts/aot_regional_generated_controls.py"
HELPER_SHA = "b62b4b47dd4a767be8e3dd7d3157be01418870d3e993b54e116cf06500dfc4a9"
SCHEMA = "seaqr.aot.regional-motion.v1"
PLAN_SCHEMA = "seaqr.aot.regional-motion-plan.v1"
INPUT_PINS = {
    "residual_result_sha256": ("residual_patterns_01/result.json", "a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc"),
    "patch_result_sha256": ("image_patches_01/result.json", "6fd7beb2e83ae8915e2ec9f5d96ae9f7fa956df66a3cee4e5a2c6146f0a82f3c"),
    "patch_selection_sha256": ("image_patches_01/selection.json", "874313f96e3eb2d537326eda46e5b52deb1458eb892474637cc4c3d9c926b6a6"),
}
DESIGN = dict(arms=list(ARMS), previous_indices=list(PREVIOUS_INDICES), case_count=8,
    grid_rows=GRID_ROWS, grid_cols=GRID_COLS, width=WIDTH, height=HEIGHT, guard_px=GUARD,
    local_radius_px=RADIUS, global_min_points=100, global_min_cells=12, local_min_points=24, local_min_cells=4,
    coherent_residual_px=COHERENT_RESIDUAL, coherent_min_points=12, coherent_min_cells=4,
    coherent_base_mass_fraction=COHERENT_MASS, local_max_condition=MAX_CONDITION,
    hull_cross_product_tolerance=HULL_TOLERANCE, huber_iterations=ITERATIONS, huber_delta_px=HUBER_DELTA,
    actual_patch_sample_count=758, independent_patch_reference_count=172,
    target_backgrounds=["translation", "affine"], target_placements=["center_single", "center_3x3", "right_boundary_3x3"],
    target_offsets=[[0, 0], [1, 0], [4, -2]], target_case_count=18,
    nuisance_sizes=[1, 9], nuisance_offsets=[[0, 0], [20, -10]], nuisance_case_count=8,
    lattice_origin=[24, 24], lattice_step=48, analytic_recovery_tolerance_px=1e-8,
    automatic_model_selection=False, production_changed=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def points_array(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all(), "finite Nx2 required")
    return value


def cell_masks(points, cell_id):
    points = points_array(points)
    require(np.all((points >= 0) & (points < [WIDTH, HEIGHT])), "previous point outside native image")
    require(type(cell_id) is int and 0 <= cell_id < 48, "invalid native cell")
    row, column = divmod(cell_id, GRID_COLS)
    x0, x1 = column * WIDTH / GRID_COLS, (column + 1) * WIDTH / GRID_COLS
    y0, y1 = row * HEIGHT / GRID_ROWS, (row + 1) * HEIGHT / GRID_ROWS
    ex0, ey0, ex1, ey1 = max(0, x0 - GUARD), max(0, y0 - GUARD), min(WIDTH, x1 + GUARD), min(HEIGHT, y1 + GUARD)
    test = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    excluded = (points[:, 0] >= ex0) & (points[:, 0] < ex1) & (points[:, 1] >= ey0) & (points[:, 1] < ey1)
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    radius = np.sum((points - center) ** 2, axis=1) <= RADIUS ** 2
    return dict(cell_id=cell_id, cell_bounds=[x0, y0, x1, y1], excluded_bounds=[ex0, ey0, ex1, ey1],
        cell_center=center.tolist(), test_indices=np.flatnonzero(test), guard_indices=np.flatnonzero(excluded & ~test),
        global_train_indices=np.flatnonzero(~excluded), local_train_indices=np.flatnonzero(~excluded & radius),
        local_radius_indices=np.flatnonzero(radius))


def convex_hull(points):
    points = points_array(points)
    ordered = sorted(set(map(tuple, points.tolist())))
    if len(ordered) < 3:
        return np.empty((0, 2), dtype=float)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    chains = []
    for sequence in (ordered, reversed(ordered)):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        chains.append(chain)
    hull = chains[0][:-1] + chains[1][:-1]
    return np.asarray(hull, dtype=float) if len(hull) >= 3 else np.empty((0, 2), dtype=float)


def hull_contains(hull, queries, tolerance=1e-8):
    hull, queries = points_array(hull), points_array(queries)
    require(np.isfinite(tolerance) and tolerance >= 0, "invalid hull tolerance")
    if len(hull) < 3:
        return np.zeros(len(queries), dtype=bool)
    edges = np.roll(hull, -1, axis=0) - hull
    relative = queries[:, None, :] - hull[None, :, :]
    crosses = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(crosses >= -tolerance, axis=1)


def fit_arm(helper, previous, current, train_indices, arm):
    previous, current = points_array(previous), points_array(current)
    indices = np.asarray(train_indices, dtype=np.int64)
    require(arm in ARMS and previous.shape == current.shape and indices.ndim == 1
            and np.all((indices >= 0) & (indices < len(previous))) and len(np.unique(indices)) == len(indices), "invalid training inventory")
    p, q = previous[indices], current[indices]
    balance = helper.cell_weights(p)
    global_arm = arm == "global_translation"
    minimum_count, minimum_cells = (100, 12) if global_arm else (24, 4)
    result = dict(arm=arm, model=MODEL[arm], train_indices=indices.tolist(), train_count=len(indices),
        train_occupied_cells=balance["occupied_cells"], train_cell_counts=balance["counts"],
        training_eligible=False, gate_reasons=[], fit=None, coherent_indices=None, coherent_count=None, coherent_occupied_cells=None,
        coherent_base_mass_fraction=None, hull=None,
        availability_policy="global numerical reference" if global_arm else "local training consistency plus pointwise coherent-training hull")
    if len(indices) < minimum_count:
        result["gate_reasons"].append(f"training_points_below_{minimum_count}")
    if balance["occupied_cells"] < minimum_cells:
        result["gate_reasons"].append(f"training_cells_below_{minimum_cells}")
    if result["gate_reasons"]:
        return result
    fit = helper.fit_model(p, q, MODEL[arm], balance["weights"], ITERATIONS, HUBER_DELTA)
    result["fit"] = fit
    if not fit["valid"]:
        result["gate_reasons"].append(fit["reason"])
        return result
    if global_arm:
        result["training_eligible"] = True
        return result
    condition = fit["condition_number"]
    if condition is None or not np.isfinite(condition) or condition > MAX_CONDITION:
        result["gate_reasons"].append("weighted_design_condition_above_1000_or_nonfinite")
    residual = np.linalg.norm(np.asarray(fit["training_residual_xy"], dtype=float), axis=1)
    coherent = residual <= COHERENT_RESIDUAL
    cells = helper.native_cells(p)
    coherent_cells = len(np.unique(cells[coherent]))
    mass = float(np.sum(balance["weights"][coherent]) / np.sum(balance["weights"]))
    hull = convex_hull(p[coherent])
    result.update(coherent_indices=indices[coherent].tolist(), coherent_count=int(coherent.sum()),
        coherent_occupied_cells=coherent_cells, coherent_base_mass_fraction=mass, hull=hull.tolist())
    if int(coherent.sum()) < 12:
        result["gate_reasons"].append("coherent_points_below_12")
    if coherent_cells < 4:
        result["gate_reasons"].append("coherent_cells_below_4")
    if mass < COHERENT_MASS:
        result["gate_reasons"].append("coherent_original_base_mass_below_0_6")
    if len(hull) < 3:
        result["gate_reasons"].append("degenerate_coherent_training_hull")
    result["training_eligible"] = not result["gate_reasons"]
    return result


def predict_arm(helper, fitted, queries):
    queries = points_array(queries)
    available = np.zeros(len(queries), dtype=bool)
    reasons = ["training_ineligible"] * len(queries)
    predictions = [None] * len(queries)
    if fitted["training_eligible"]:
        available = (np.ones(len(queries), dtype=bool) if fitted["arm"] == "global_translation"
                     else hull_contains(np.asarray(fitted["hull"]).reshape(-1, 2), queries, HULL_TOLERANCE))
        values = helper.predict(queries[available], fitted["fit"]["parameters"], fitted["model"])
        for index, value in zip(np.flatnonzero(available), values):
            predictions[int(index)] = value.tolist()
        reasons = [None if present else "outside_coherent_training_hull" for present in available]
    return dict(available=available.tolist(), predicted_displacement_xy=predictions, unavailable_reason=reasons)


def evaluate_cell(helper, previous, current, cell_id):
    masks = cell_masks(previous, cell_id)
    result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in masks.items()}
    result["arms"] = {}
    for arm in ARMS:
        train = masks["global_train_indices"] if arm == "global_translation" else masks["local_train_indices"]
        fitted = fit_arm(helper, previous, current, train, arm)
        fitted["queries"] = predict_arm(helper, fitted, np.asarray(previous)[masks["test_indices"]])
        result["arms"][arm] = fitted
    return result


def join_patch_evidence(actual_row, patch_rows):
    points = actual_row["points"]
    previous, current = points_array(points["accepted_previous_xy"]), points_array(points["accepted_current_xy"])
    relevant = [row for row in patch_rows if row["source_kind"] == "actual_residual_extremum" and row["case_id"] == actual_row["case_id"]]
    joined = {}
    independent = {}
    for row in relevant:
        index = row["accepted_index"]
        require(type(index) is int and 0 <= index < len(previous) and index not in joined, "duplicate/invalid patch accepted index")
        require(row["previous_xy"] == previous[index].tolist() and row["current_xy"] == current[index].tolist()
                and row["selected_index"] == points["accepted_selected_indices"][index], "patch/LK point identity differs")
        require([scale["template_size"] for scale in row["scales"]] == [33, 65], "patch scale inventory differs")
        joined[index] = row["id"]
        scales = row["scales"]
        if all(scale["qualified"] is True for scale in scales):
            require(all(scale["available"] is True for scale in scales), "qualified but unavailable patch reference")
            offsets = points_array([scale["best_offset_xy"] for scale in scales])
            if np.linalg.norm(offsets[0] - offsets[1]) <= 1.5:
                independent[index] = dict(ncc33=offsets[0].tolist(), ncc65=offsets[1].tolist())
    return dict(patch_indices=sorted(joined), independent_indices=sorted(independent),
        unresolved_patch_indices=sorted(set(joined) - set(independent)), patch_ids=joined, independent_offsets=independent,
        cohort_rule="both old scales qualify and agree within1.5px; saved-LK agreement/verdict never used for membership")


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "finite scalar observations required")
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
        p90=float(np.quantile(values, .9)) if len(values) else None,
        minimum=float(np.min(values)) if len(values) else None, maximum=float(np.max(values)) if len(values) else None)


def cohort_summary(predictions, query_indices, references):
    indices = list(query_indices)
    require(len(set(indices)) == len(indices), "duplicate cohort queries")
    refs = {name: points_array(values) for name, values in references.items()}
    require(all(len(values) == len(indices) for values in refs.values()), "reference/cohort length differs")
    errors = {arm: {name: [] for name in refs} for arm in ARMS}
    availability = {arm: [predictions[arm]["available"][index] for index in indices] for arm in ARMS}
    arms = {}
    for arm in ARMS:
        for position, index in enumerate(indices):
            prediction = predictions[arm]["predicted_displacement_xy"][index]
            require((prediction is not None) == availability[arm][position], "prediction/abstention mismatch")
            for name, reference in refs.items():
                errors[arm][name].append(None if prediction is None else float(np.linalg.norm(np.asarray(prediction) - reference[position])))
        arms[arm] = dict(available_count=sum(availability[arm]), unavailable_count=len(indices) - sum(availability[arm]),
            available_indices=[index for position, index in enumerate(indices) if availability[arm][position]],
            unavailable_indices=[index for position, index in enumerate(indices) if not availability[arm][position]],
            errors={name: dict(values=values, summary=numeric_summary([value for value in values if value is not None])) for name, values in errors[arm].items()})
    comparisons = {}
    for first, second in PAIRS:
        positions = [position for position in range(len(indices)) if availability[first][position] and availability[second][position]]
        reference_errors = {}
        for name in refs:
            a, b = [errors[first][name][position] for position in positions], [errors[second][name][position] for position in positions]
            differences = [right - left for left, right in zip(a, b)]
            reference_errors[name] = dict(first_errors=a, second_errors=b, first_summary=numeric_summary(a), second_summary=numeric_summary(b),
                paired_difference=dict(direction="second minus first", values=differences, summary=numeric_summary(differences)))
        comparisons[f"{first}__{second}"] = dict(first=first, second=second, total_count=len(indices), common_count=len(positions),
            missing_count=len(indices) - len(positions), common_indices=[indices[position] for position in positions], references=reference_errors)
    all_positions = [position for position in range(len(indices)) if all(availability[arm][position] for arm in ARMS)]
    all_references = {}
    for name in refs:
        values = {arm: [errors[arm][name][position] for position in all_positions] for arm in ARMS}
        all_references[name] = dict(arm_errors=values, arm_summaries={arm: numeric_summary(v) for arm, v in values.items()},
            paired_differences={f"{first}__{second}": dict(direction="second minus first",
                values=[b - a for a, b in zip(values[first], values[second])],
                summary=numeric_summary([b - a for a, b in zip(values[first], values[second])])) for first, second in PAIRS})
    return dict(total_count=len(indices), query_indices=indices, arms=arms, pairwise=comparisons,
        all_three_common=dict(total_count=len(indices), common_count=len(all_positions), missing_count=len(indices) - len(all_positions),
            common_indices=[indices[position] for position in all_positions], references=all_references),
        interpretation="errors conditional on reported eligibility; identical-query comparisons only; missing predictions never counted correct")


def collect_predictions(cells, count):
    inventory = [index for cell in cells for index in cell["test_indices"]]
    require(sorted(inventory) == list(range(count)), "held-out cells do not partition accepted observations exactly once")
    predictions = {arm: dict(available=[False] * count, predicted_displacement_xy=[None] * count,
        unavailable_reason=["not_evaluated"] * count) for arm in ARMS}
    for cell in cells:
        for arm in ARMS:
            query = cell["arms"][arm]["queries"]
            require(len(query["available"]) == len(cell["test_indices"]), "cell query length differs")
            for position, index in enumerate(cell["test_indices"]):
                for key in predictions[arm]:
                    predictions[arm][key][index] = query[key][position]
    return predictions


def compare_case(helper, actual_row, patch_rows):
    previous, current, truth, _ = helper.validate_case(actual_row)
    require(truth is None and actual_row["source_kind"] == "aot_adjacent", "only actual adjacent pairs in regional study")
    evidence = join_patch_evidence(actual_row, patch_rows)
    cells = [evaluate_cell(helper, previous, current, cell_id) for cell_id in range(48)]
    predictions = collect_predictions(cells, len(previous))
    displacement = current - previous
    all_indices = list(range(len(previous)))
    patch_indices, independent_indices = evidence["patch_indices"], evidence["independent_indices"]
    cohorts = dict(all_accepted_lk=cohort_summary(predictions, all_indices, dict(saved_lk=displacement)),
        all_patch_samples=cohort_summary(predictions, patch_indices, dict(saved_lk=displacement[patch_indices])),
        independent_patch_references=cohort_summary(predictions, independent_indices,
            {name: np.asarray([evidence["independent_offsets"][index][name] for index in independent_indices]).reshape(-1, 2) for name in ("ncc33", "ncc65")}))
    cohorts["all_patch_samples"]["image_evidence_counts"] = dict(selected=len(patch_indices), independent=len(independent_indices),
        unresolved=len(evidence["unresolved_patch_indices"]), unresolved_indices=evidence["unresolved_patch_indices"],
        interpretation="unresolved prior patch evidence retained, not reclassified by model support or error")
    return dict(case_id=actual_row["case_id"], previous_index=actual_row["previous_index"], current_index=actual_row["current_index"],
        selected_count=actual_row["selected_count"], accepted_count=actual_row["accepted_count"], lost_count=actual_row["lost_count"],
        original_fit=actual_row.get("original_fit"), points=dict(accepted_previous_xy=previous.tolist(), accepted_current_xy=current.tolist()),
        cells=cells, predictions=predictions, cohorts=cohorts, patch_evidence=evidence)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def scope_output(path):
    path = Path(path)
    require(path == OUTPUT_DIR and path.resolve() == OUTPUT_DIR and path.is_dir() and not path.is_symlink(), "outside fixed regional output scope")
    require(all(not (path / name).exists() and not (path / name).is_symlink() for name in ("result.json", "failure.json")), "existing output; refusing overwrite")
    return path


def input_hashes():
    hashes = {}
    for key, (relative, expected) in INPUT_PINS.items():
        path = EVIDENCE_ROOT / relative
        require(path.is_file() and not path.is_symlink(), "missing/linked numerical input")
        hashes[key] = sha(path)
        require(hashes[key] == expected, f"frozen input differs: {key}")
    return hashes


def artifact_hashes():
    paths = dict(script_sha256=Path(__file__), tests_sha256=REPO / "tests/test_aot_regional_motion.py",
        plan_sha256=REPO / "docs/aot_regional_motion_plan_20260928.md", model_helper_sha256=HELPER_PATH,
        generated_controls_sha256=GENERATED_PATH)
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()), "missing/linked regional artifact")
    hashes = {key: sha(path) for key, path in paths.items()}
    require(hashes["model_helper_sha256"] == HELPER_SHA, "frozen robust-fitting helper changed")
    return hashes


def validate_manifest(manifest, hashes):
    require(manifest.get("schema") == PLAN_SCHEMA, "wrong regional plan schema")
    require(json.dumps(manifest.get("design"), sort_keys=True) == json.dumps(DESIGN, sort_keys=True), "fixed regional design differs")
    for key, (_, expected) in INPUT_PINS.items():
        require(manifest.get(key) == hashes.get(key) == expected, f"manifest input differs: {key}")
    require(manifest.get("model_helper_sha256") == hashes.get("model_helper_sha256") == HELPER_SHA, "manifest helper identity differs")
    for key in ("script_sha256", "tests_sha256", "plan_sha256", "generated_controls_sha256"):
        value = hashes.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                and manifest.get(key) == value, f"manifest artifact differs: {key}")


def bound_hashes(output):
    manifest_path = output / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "missing/linked frozen manifest")
    hashes = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(manifest_path)}
    validate_manifest(json_read(manifest_path), hashes)
    return hashes


def load_module(path, expected_hash, name):
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_hash, "helper changed before import")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "unavailable frozen helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(residual, patches, selection):
    require(residual.get("schema") == "seaqr.aot.residual-patterns.v1" and residual.get("passed") is True
            and patches.get("schema") == "seaqr.aot.image-patches.v1" and patches.get("passed") is True
            and selection.get("schema") == "seaqr.aot.image-patch-selection.v1" and selection.get("passed") is True,
            "prior numerical/image-evidence receipts not passed")
    require(patches["selection_sha256"] == INPUT_PINS["patch_selection_sha256"][1]
            and patches["hashes_before"] == patches["hashes_after"]
            and patches["hashes_before"]["residual_result_sha256"] == INPUT_PINS["residual_result_sha256"][1], "patch provenance differs")
    rows = [row for row in residual["rows"] if row["source_kind"] == "aot_adjacent"]
    require([row["previous_index"] for row in rows] == list(PREVIOUS_INDICES)
            and [row["case_id"] for row in rows] == [f"aot_{index + 1:03d}_adjacent" for index in PREVIOUS_INDICES], "fixed eight actual cases differ")
    patch_rows = [row for row in patches["rows"] if row["source_kind"] == "actual_residual_extremum"]
    selected = [row for row in selection["points"] if row["source_kind"] == "actual_residual_extremum"]
    require(len(patch_rows) == len(selected) == 758 and [row["id"] for row in patch_rows] == [row["id"] for row in selected], "frozen patch sample differs")
    require(all(all(row.get(key) == value for key, value in chosen.items()) for row, chosen in zip(patch_rows, selected)), "patch selection records changed")
    joins = [join_patch_evidence(row, patch_rows) for row in rows]
    require(sum(len(join["patch_indices"]) for join in joins) == 758
            and sum(len(join["independent_indices"]) for join in joins) == 172
            and sum(len(join["unresolved_patch_indices"]) for join in joins) == 586, "fixed patch-cohort denominators differ")
    return rows, patch_rows


def run(output):
    output = scope_output(output)
    before = bound_hashes(output)
    receipt = dict(schema=SCHEMA, passed=False, passed_interpretation="execution/provenance completion only, not model acceptance",
        design=DESIGN, hashes_before=before, rows=[], generated=None, numpy_version=np.__version__,
        planned_cell_pairs=384, planned_arm_evaluations=1152, completed_cell_pairs=0,
        media_accessed=False, annotations_used=False, remote_accessed=False, detector_run=False,
        production_changed=False, production_promotion=False, automatic_model_selection=False,
        limitations=["Originally accepted LK observations remain censored by preceding feature/flow acceptance; lost features are not recovered.",
            "Local consistency and hull eligibility are training-derived descriptors, not calibrated correctness or uncertainty.",
            "Patch references are purposefully selected, correlated, integer-grid evidence; both scales retained separately, not physical truth.",
            "Unavailable predictions remain in denominators; comparisons use identical eligible queries without fallback.",
            "Generated target cases are vector-level checks, not image contrast, warp, temporal, identity or detector-recall validation.",
            "Eight pairs from one development sequence cannot establish independent-scene or stationary/night deployment accuracy."])
    try:
        helper = load_module(HELPER_PATH, HELPER_SHA, "_aot_regional_frozen_fit")
        generated = load_module(GENERATED_PATH, before["generated_controls_sha256"], "_aot_regional_generated")
        rows, patches = validate_inputs(json_read(EVIDENCE_ROOT / INPUT_PINS["residual_result_sha256"][0]),
            json_read(EVIDENCE_ROOT / INPUT_PINS["patch_result_sha256"][0]), json_read(EVIDENCE_ROOT / INPUT_PINS["patch_selection_sha256"][0]))
        for ordinal, row in enumerate(rows):
            receipt["rows"].append(compare_case(helper, row, patches))
            receipt["completed_cell_pairs"] += 48
            print(json.dumps(dict(case_id=row["case_id"], completed=ordinal + 1, total=8)), flush=True)
        receipt["generated"] = generated.run_generated(helper, evaluate_cell, ARMS)
        require(len(receipt["generated"]["target_preservation"]["cases"]) == 18
                and len(receipt["generated"]["training_contamination"]["cases"]) == 8, "generated control inventory differs")
        receipt["hashes_after"] = bound_hashes(output)
        require(before == receipt["hashes_after"] and receipt["completed_cell_pairs"] == 384, "incomplete run or changed frozen inputs/artifacts")
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
