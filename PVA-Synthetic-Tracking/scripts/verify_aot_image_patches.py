#!/usr/bin/env python3
"""Frozen local image-patch evidence, independent of the saved feature/LK method."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 2448, 2048
PATCH_SIZES = (33, 65)
SEARCH_RADIUS, RUNNER_EXCLUSION = 64, 3
MIN_NCC, MIN_GAP, MIN_STD, AGREEMENT_PX = .8, .05, 1.0, 1.5
PREVIOUS_INDICES = (0, 42, 85, 127, 170, 212, 255, 298)
REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO.parent / "outputs/seaqr_aot_pilot_20260927"
OUTPUT_DIR = EVIDENCE_ROOT / "image_patches_01"
SOURCE_DIR = EVIDENCE_ROOT / "source_png"
SCHEMA = "seaqr.aot.image-patches.v1"
PLAN_SCHEMA = "seaqr.aot.image-patches-plan.v1"
SELECTION_SCHEMA = "seaqr.aot.image-patch-selection.v1"
FREEZE_SCHEMA = "seaqr.aot.image-patch-selection-freeze.v1"
INPUT_PINS = {
    "residual_result_sha256": ("residual_patterns_01/result.json", "a18cb9614b02847e5a9e34a16e5f3fab3ec1963541cb2684444e28134c0d85fc"),
    "large_result_sha256": ("large_shifts_01/result.json", "3342c3440dd1d190653a9aa46b5fd0210157ffd3faa811ed76506a78b09b6703"),
    "counterexamples_sha256": ("large_shifts_01/accepted_large_shift_outliers.json", "87f79aa1b39992819d607ebfbb2436ff1d601f8d762b722ec8f85f0b49792992"),
    "download_validation_sha256": ("download_validation.json", "b602b89755e60122f1cac200808d19bc58d55443697ff5d16f83da8afde530d9"),
}
COUNTEREXAMPLES = ((85, 48, 402, 348), (127, 48, 691, 599), (127, -48, 386, 342), (298, -48, 526, 488))
DESIGN = dict(previous_indices=list(PREVIOUS_INDICES), actual_arm="half_gain16_complete",
    grid_rows=6, grid_cols=8, selection="within_cell_minimum_and_maximum_saved_candidate_residual_norm",
    tie_policy="lowest_accepted_index_each_extremum_deduplicate", maximum_actual_points=768,
    counterexamples=[list(record) for record in COUNTEREXAMPLES], patch_sizes=list(PATCH_SIZES),
    search_radius_px=SEARCH_RADIUS, runner_exclusion_chebyshev_px=RUNNER_EXCLUSION,
    min_ncc=MIN_NCC, min_gap=MIN_GAP, min_template_and_current_std_dn=MIN_STD,
    agreement_px=AGREEMENT_PX, clipped_search_unqualified=True, coordinate_rounding="floor(p+0.5)",
    visual_selection="quadrant_minimum_of_min_roles_and_maximum_of_max_roles_tie_accepted_index_deduplicate",
    maximum_actual_visual_points=64, synthetic_fill=128, image_width=WIDTH, image_height=HEIGHT,
    production_changed=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def half_up(point):
    point = np.asarray(point, dtype=float)
    require(point.shape == (2,) and np.isfinite(point).all() and np.all(point >= 0), "finite nonnegative source xy required")
    return np.floor(point + .5).astype(np.int64)


def search_geometry(point, size, image_shape, search_radius=64):
    require(type(size) is int and size > 0 and size % 2 == 1, "positive odd template size required")
    require(type(search_radius) is int and 0 <= search_radius <= SEARCH_RADIUS, "invalid search radius")
    require(len(image_shape) == 2 and all(type(v) in (int, np.int64) and v > 0 for v in image_shape), "invalid grayscale image dimensions")
    height, width = image_shape
    cx, cy = (int(v) for v in half_up(point))
    radius = size // 2
    record = dict(anchor_xy=[cx, cy], template_size=size, search_radius_px=search_radius,
        template_bounds=[cx - radius, cy - radius, cx + radius + 1, cy + radius + 1],
        bounds_convention="image rectangles half-open xyxy; offset bounds inclusive dx_min,dy_min,dx_max,dy_max")
    if cx - radius < 0 or cy - radius < 0 or cx + radius >= width or cy + radius >= height:
        return dict(**record, available=False, unavailable_reason="previous_template_outside_image")
    dx0, dx1 = max(-search_radius, radius - cx), min(search_radius, width - 1 - radius - cx)
    dy0, dy1 = max(-search_radius, radius - cy), min(search_radius, height - 1 - radius - cy)
    require(dx0 <= dx1 and dy0 <= dy1, "empty valid current search")
    return dict(**record, available=True, unavailable_reason=None, offset_bounds=[dx0, dy0, dx1, dy1],
        current_search_bounds=[cx + dx0 - radius, cy + dy0 - radius, cx + dx1 + radius + 1, cy + dy1 + radius + 1],
        search_clipped=[dx0, dy0, dx1, dy1] != [-search_radius, -search_radius, search_radius, search_radius])


def window_sums(image, size):
    image = np.asarray(image, dtype=np.float64)
    integral = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def zncc_surface(template, search):
    """Float64 valid correlation, no image padding/resampling or model guidance."""
    template, search = np.asarray(template), np.asarray(search)
    require(template.dtype == np.uint8 and search.dtype == np.uint8 and template.ndim == search.ndim == 2
            and template.shape[0] == template.shape[1] and template.shape[0] > 0
            and all(a <= b for a, b in zip(template.shape, search.shape)), "invalid U8 template/search arrays")
    size, count = template.shape[0], template.size
    source = search.astype(np.float64)
    centered = template.astype(np.float64) - float(np.mean(template))
    template_energy = float(np.sum(centered * centered))
    sums, squared = window_sums(source, size), window_sums(source * source, size)
    variance_energy = np.maximum(squared - sums * sums / count, 0.0)
    current_std = np.sqrt(variance_energy / count)
    scores = np.full(current_std.shape, np.nan, dtype=np.float64)
    if template_energy == 0:
        return scores, current_std
    # FFT padding is computational only. Valid correlation indices never wrap,
    # so every candidate uses exactly one fully in-image patch of original U8s.
    fft_shape = tuple(1 << (dimension - 1).bit_length() for dimension in source.shape)
    correlation = np.fft.irfft2(np.fft.rfft2(source, s=fft_shape)
        * np.conj(np.fft.rfft2(centered, s=fft_shape)), s=fft_shape)
    numerator = correlation[:scores.shape[0], :scores.shape[1]] - sums * float(centered.sum()) / count
    denominator = np.sqrt(template_energy * variance_energy)
    valid = denominator > 0
    scores[valid] = numerator[valid] / denominator[valid]
    require(np.all(np.abs(scores[valid]) <= 1 + 1e-7), "numerically invalid ZNCC coefficient")
    # Preserve raw computed coefficients (including harmless FP epsilon), not a
    # calibrated probability, rounded score, or a forced perfect-match value.
    return scores, current_std


def peak_summary(scores, current_std, offset_bounds, template_std):
    scores, current_std = np.asarray(scores, dtype=float), np.asarray(current_std, dtype=float)
    require(scores.ndim == 2 and scores.shape == current_std.shape and len(offset_bounds) == 4,
            "invalid peak surface metadata")
    require(np.isfinite(template_std) and template_std >= 0 and np.isfinite(current_std).all()
            and np.all(current_std >= 0), "finite nonnegative patch standard deviations required")
    dx0, dy0, dx1, dy1 = offset_bounds
    require(scores.shape == (dy1 - dy0 + 1, dx1 - dx0 + 1), "offset bounds do not describe score surface")
    valid = np.isfinite(scores)
    base = dict(valid_candidate_count=int(valid.sum()), total_candidate_count=int(scores.size))
    if not valid.any():
        return dict(**base, available=False, qualified=False, unavailable_reason="no_nonzero_variance_correlation",
            best_offset_xy=None, best_ncc=None, runner_up_ncc=None, gap=None,
            template_std=float(template_std), best_current_std=None, best_at_search_boundary=None)
    flat = int(np.argmax(np.where(valid, scores, -np.inf)))
    row, column = np.unravel_index(flat, scores.shape)
    yy, xx = np.indices(scores.shape)
    outside = np.maximum(np.abs(yy - row), np.abs(xx - column)) > RUNNER_EXCLUSION
    runner = valid & outside
    second = float(np.max(scores[runner])) if runner.any() else None
    best = float(scores[row, column])
    gap = None if second is None else best - second
    boundary = bool(row in (0, scores.shape[0] - 1) or column in (0, scores.shape[1] - 1))
    patch_std = float(current_std[row, column])
    qualified = (best >= MIN_NCC and gap is not None and gap >= MIN_GAP
                 and template_std >= MIN_STD and patch_std >= MIN_STD and not boundary)
    return dict(**base, available=True, unavailable_reason=None, best_offset_xy=[int(dx0 + column), int(dy0 + row)],
        best_ncc=best, runner_up_ncc=second, gap=gap, template_std=float(template_std),
        best_current_std=patch_std, best_at_search_boundary=boundary, qualified=bool(qualified),
        qualification=dict(ncc=best >= MIN_NCC, gap=gap is not None and gap >= MIN_GAP,
            template_std=template_std >= MIN_STD, current_std=patch_std >= MIN_STD, interior_search_peak=not boundary),
        tie_policy="first row-major valid maximum: lowest dy then lowest dx")


def zncc_search(previous, current, point, size, search_radius=64):
    previous, current = np.asarray(previous), np.asarray(current)
    require(previous.dtype == current.dtype == np.uint8 and previous.ndim == current.ndim == 2
            and previous.shape == current.shape, "matching native U8 grayscale images required")
    geometry = search_geometry(point, size, previous.shape, search_radius)
    if not geometry["available"]:
        return dict(geometry=geometry, template_size=size, available=False, qualified=False,
                    unavailable_reason=geometry["unavailable_reason"])
    x0, y0, x1, y1 = geometry["template_bounds"]
    template = previous[y0:y1, x0:x1]
    x0, y0, x1, y1 = geometry["current_search_bounds"]
    search = current[y0:y1, x0:x1]
    scores, std = zncc_surface(template, search)
    peak = peak_summary(scores, std, geometry["offset_bounds"], float(np.std(template.astype(np.float64))))
    if peak["available"]:
        peak["qualification"]["full_unclipped_search"] = not geometry["search_clipped"]
        peak["qualified"] = peak["qualified"] and not geometry["search_clipped"]
    return dict(geometry=geometry, template_size=size, **peak,
        estimated_current_xy=None if not peak["available"] else (np.asarray(point, dtype=float) + peak["best_offset_xy"]).tolist())


def two_scale_verdict(records, saved_displacement):
    require([record["template_size"] for record in records] == list(PATCH_SIZES), "two fixed template sizes required")
    saved = np.asarray(saved_displacement, dtype=float)
    require(saved.shape == (2,) and np.isfinite(saved).all(), "finite saved displacement required")
    errors = [None if not record.get("available") else float(np.linalg.norm(np.asarray(record["best_offset_xy"]) - saved)) for record in records]
    offsets = [record.get("best_offset_xy") for record in records]
    offset_difference = None if any(offset is None for offset in offsets) else float(np.linalg.norm(np.asarray(offsets[0]) - offsets[1]))
    qualify = all(record.get("qualified") is True for record in records) and offset_difference is not None and offset_difference <= AGREEMENT_PX
    verdict = "ambiguous"
    if qualify and all(error <= AGREEMENT_PX for error in errors):
        verdict = "agrees_with_saved_lk"
    elif qualify and all(error > AGREEMENT_PX for error in errors):
        verdict = "disagrees_with_saved_lk"
    return dict(verdict=verdict, two_sizes_qualified_and_consistent=bool(qualify),
        offset_difference_px=offset_difference, saved_lk_error_by_size_px=errors,
        interpretation="diagnostic image-patch agreement only; ambiguity is not rejection and disagreement is not ground truth")


def select_actual_points(residual_result):
    """Numerical-only extrema selection; never consult image values or NCC."""
    actual = [row for row in residual_result["rows"] if row["source_kind"] == "aot_adjacent"]
    require([row["previous_index"] for row in actual] == list(PREVIOUS_INDICES), "eight fixed actual pairs required")
    inventory, chosen = [], []
    for row in actual:
        require(row["arm"] == "half_gain16_complete" and row["current_index"] == row["previous_index"] + 1,
                "wrong actual source arm/indices")
        points = row["points"]
        previous = np.asarray(points["accepted_previous_xy"], dtype=float)
        current = np.asarray(points["accepted_current_xy"], dtype=float)
        residual = np.asarray(points["candidate_residual_xy"], dtype=float)
        require(previous.ndim == 2 and previous.shape[1] == 2 and previous.shape == current.shape == residual.shape
                and len(previous) == row["accepted_count"] and np.isfinite([previous, current, residual]).all(), "invalid saved actual points/residuals")
        require(np.all((previous >= 0) & (previous < [WIDTH, HEIGHT])), "accepted source point outside image")
        cells = np.floor(previous[:, 1] * 6 / HEIGHT).astype(int) * 8 + np.floor(previous[:, 0] * 8 / WIDTH).astype(int)
        norms = np.linalg.norm(residual, axis=1)
        for cell in range(48):
            members = np.flatnonzero(cells == cell)
            record = dict(case_id=row["case_id"], previous_index=row["previous_index"],
                cell_row=cell // 8, cell_column=cell % 8, accepted_point_count=len(members), selected_point_ids=[])
            if not len(members):
                record["status"] = "empty"
            else:
                # np.argmin/argmax choose the first (lowest accepted index) tie.
                low, high = int(members[np.argmin(norms[members])]), int(members[np.argmax(norms[members])])
                record["status"] = "singleton" if len(members) == 1 else "same_extremum" if low == high else "two_extrema"
                for index in dict.fromkeys((low, high)):
                    roles = [name for name, candidate in (("minimum", low), ("maximum", high)) if index == candidate]
                    identifier = f"{row['case_id']}__cell{cell:02d}__accepted{index:04d}"
                    chosen.append(dict(id=identifier, source_kind="actual_residual_extremum", case_id=row["case_id"],
                        previous_index=row["previous_index"], current_index=row["current_index"],
                        accepted_index=index, selected_index=points["accepted_selected_indices"][index],
                        cell_row=cell // 8, cell_column=cell % 8, selection_roles=roles,
                        previous_xy=previous[index].tolist(), current_xy=current[index].tolist(),
                        saved_candidate_residual_xy=residual[index].tolist(), saved_candidate_residual_norm_px=float(norms[index]),
                        original_candidate_displacement_xy=row["saved_candidate_translation"],
                        expected_previous_pixel_sha256=row["previous_identity"]["native_previous"],
                        planned_geometry=[search_geometry(previous[index], size, (HEIGHT, WIDTH)) for size in PATCH_SIZES]))
                    record["selected_point_ids"].append(identifier)
            inventory.append(record)
    require(len(inventory) == 384 and len(chosen) <= 768, "representative selection exceeds fixed scope")
    return dict(cell_inventory=inventory, points=chosen,
        tie_policy="lowest accepted-list index independently at minimum and maximum; identical chosen points deduplicated",
        population_caveat="purposive within-cell residual extremes, not a random sample or a population track-error estimate")


def evaluate_point(previous, current, selected_record):
    point = np.asarray(selected_record["previous_xy"], dtype=float)
    saved = np.asarray(selected_record["current_xy"], dtype=float) - point
    records = [zncc_search(previous, current, point, size, SEARCH_RADIUS) for size in PATCH_SIZES]
    result = dict(**selected_record, saved_lk_displacement_xy=saved.tolist(), scales=records,
                  comparison=two_scale_verdict(records, saved))
    if selected_record.get("source_kind") == "known_shift_counterexample":
        truth = np.asarray(selected_record["synthetic_current"]["shift_xy"], dtype=float)
        result["counterexample_truth"] = dict(expected_shift_xy=truth.tolist(), saved_lk_error_px=float(np.linalg.norm(saved - truth)),
            independent_error_by_size_px=[None if not record["available"] else float(np.linalg.norm(np.asarray(record["best_offset_xy"]) - truth)) for record in records],
            interpretation="known shift consulted only after both unguided searches, never for qualification")
    return result


def visual_selection(points):
    """Pre-image visual subset; never choose illustrations by patch outcomes."""
    selected = {}
    for previous in PREVIOUS_INDICES:
        source = [point for point in points if point["source_kind"] == "actual_residual_extremum" and point["previous_index"] == previous]
        for quadrant in range(4):
            members = [point for point in source if int(point["previous_xy"][1] >= HEIGHT / 2) * 2
                       + int(point["previous_xy"][0] >= WIDTH / 2) == quadrant]
            for role in ("minimum", "maximum"):
                candidates = [point for point in members if role in point["selection_roles"]]
                if candidates:
                    sign = 1 if role == "minimum" else -1
                    point = min(candidates, key=lambda item: (sign * item["saved_candidate_residual_norm_px"], item["accepted_index"]))
                    selected.setdefault(point["id"], []).append(dict(quadrant=quadrant, role=role))
    require(len(selected) <= 64, "visual selection exceeds actual scope")
    for point in points:
        if point["source_kind"] == "known_shift_counterexample":
            selected[point["id"]] = [dict(quadrant=None, role="counterexample")]
        point["visual_roles"] = selected.get(point["id"], [])
    return [dict(point_id=identifier, visual_roles=roles) for identifier, roles in selected.items()]


def select_counterexamples(large, listing):
    require(large.get("passed") is True and listing.get("source_result_sha256") == INPUT_PINS["large_result_sha256"][1]
            and len(listing["outliers"]) == 4, "counterexample source identity/count differs")
    records = []
    for previous, dx, selected_index, accepted_index in COUNTEREXAMPLES:
        case_id = f"aot_prev{previous:03d}_dx{dx:+d}_dy+0"
        source = [row for row in large["rows"] if row["case_id"] == case_id]
        reference = [row for row in listing["outliers"] if row["case_id"] == case_id and row["selected_index"] == selected_index]
        require(len(source) == len(reference) == 1, "fixed counterexample missing/duplicated")
        source, reference = source[0], reference[0]
        pair = source["correspondence"]["correspondences"][accepted_index]
        coordinates = np.asarray(source["capture"]["selection"]["coordinates"]["values"], dtype=np.float32)
        native = ((coordinates[selected_index] + np.float32(.5)) * np.float32(2) - np.float32(.5)).astype(float)
        require(pair["previous_xy"] == reference["previous_xy"] == native.tolist()
                and pair["current_xy"] == reference["current_xy"] and reference["known_shift_xy"] == [dx, 0]
                and source["previous_index"] == previous and source["expected_shift_xy"] == [dx, 0]
                and source["arm"]["id"] == "half_gain16_complete", "counterexample point/shift identity differs")
        records.append(dict(id=f"{case_id}__accepted{accepted_index:04d}", source_kind="known_shift_counterexample",
            case_id=case_id, previous_index=previous, current_index=source["current_index"],
            accepted_index=accepted_index, selected_index=selected_index, previous_xy=pair["previous_xy"], current_xy=pair["current_xy"],
            cell_row=int(native[1] * 6 / HEIGHT), cell_column=int(native[0] * 8 / WIDTH), selection_roles=["counterexample"],
            expected_previous_pixel_sha256=source["native_pixel_sha256"]["previous"],
            synthetic_current=dict(shift_xy=[dx, 0], fill=128, expected_pixel_sha256=source["native_pixel_sha256"]["current"]),
            saved_known_shift_error_px=reference["true_error_px"],
            planned_geometry=[search_geometry(native, size, (HEIGHT, WIDTH)) for size in PATCH_SIZES]))
    return records


def image_metadata(validation):
    keys = ("img_name", "png_sha256", "pixel_sha256", "bytes", "source_frame", "timestamp_ns")
    require(len(validation["images"]) == 300, "approved image inventory is not 300 frames")
    rows = [dict(frame_index=index, **{key: row[key] for key in keys}) for index, row in enumerate(validation["images"])]
    for row in rows:
        require(Path(row["img_name"]).name == row["img_name"] and row["img_name"].endswith(".png")
                and type(row["bytes"]) is int and row["bytes"] > 0, "invalid approved image metadata")
        require(all(isinstance(row[key], str) and len(row[key]) == 64 and all(c in "0123456789abcdef" for c in row[key])
                    for key in ("png_sha256", "pixel_sha256")), "invalid approved image hash")
    return rows


def bind_images(points, metadata):
    inventory = {}
    for point in points:
        previous = metadata[point["previous_index"]]
        require(previous["pixel_sha256"] == point["expected_previous_pixel_sha256"], "previous numeric/pixel identity differs")
        point["previous_image"] = previous
        inventory[previous["frame_index"]] = previous
        point["current_image"] = None
        if point["source_kind"] == "actual_residual_extremum":
            current = metadata[point["current_index"]]
            point["current_image"] = current
            inventory[current["frame_index"]] = current
    return [inventory[index] for index in sorted(inventory)]


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


def scope_output(output, phase):
    output = Path(output)
    require(output == OUTPUT_DIR and output.resolve() == OUTPUT_DIR and output.is_dir() and not output.is_symlink(), "outside fixed patch output scope")
    require(phase in ("select", "run"), "invalid diagnostic phase")
    names = ["result.json", "failure.json"] + (["selection.json", "selection_freeze.json"] if phase == "select" else [])
    require(all(not (output / name).exists() and not (output / name).is_symlink() for name in names), "existing phase output; refusing overwrite")
    return output


def input_hashes():
    result = {}
    for key, (relative, expected) in INPUT_PINS.items():
        path = EVIDENCE_ROOT / relative
        require(path.is_file() and not path.is_symlink(), "missing/linked numeric or metadata input")
        result[key] = sha(path)
        require(result[key] == expected, f"frozen input differs: {key}")
    return result


def artifact_hashes():
    paths = dict(script_sha256=Path(__file__), tests_sha256=REPO / "tests/test_aot_image_patches.py",
        plan_sha256=REPO / "docs/aot_image_patches_plan_20260928.md")
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()), "missing/linked diagnostic artifact")
    return {key: sha(path) for key, path in paths.items()}


def validate_manifest(manifest, hashes):
    require(manifest.get("schema") == PLAN_SCHEMA, "wrong image-patch plan schema")
    require(json.dumps(manifest.get("design"), sort_keys=True) == json.dumps(DESIGN, sort_keys=True), "frozen patch design differs")
    for key, (_, expected) in INPUT_PINS.items():
        require(manifest.get(key) == hashes.get(key) == expected, f"manifest input differs: {key}")
    for key in ("script_sha256", "tests_sha256", "plan_sha256"):
        value = hashes.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                and manifest.get(key) == value, f"manifest artifact differs: {key}")


def bound_hashes(output):
    path = output / "manifest.json"
    require(path.is_file() and not path.is_symlink(), "missing/linked frozen manifest")
    hashes = {**input_hashes(), **artifact_hashes(), "manifest_sha256": sha(path)}
    validate_manifest(json_read(path), hashes)
    return hashes


def validate_selection_freeze(freeze, selection_hash, manifest_hash):
    require(freeze.get("schema") == FREEZE_SCHEMA and freeze.get("selection_sha256") == selection_hash
            and freeze.get("manifest_sha256") == manifest_hash, "selection/manifest freeze differs")


def select(output):
    output = scope_output(output, "select")
    before = bound_hashes(output)
    receipt = dict(schema=SELECTION_SCHEMA, passed=False, hashes_before=before, design=DESIGN,
        source_images_opened=False, patch_metrics_computed=False)
    try:
        residual = json_read(EVIDENCE_ROOT / INPUT_PINS["residual_result_sha256"][0])
        require(residual.get("schema") == "seaqr.aot.residual-patterns.v1" and residual.get("passed") is True, "residual source not passed")
        receipt.update(select_actual_points(residual))
        del residual
        large = json_read(EVIDENCE_ROOT / INPUT_PINS["large_result_sha256"][0])
        listing = json_read(EVIDENCE_ROOT / INPUT_PINS["counterexamples_sha256"][0])
        receipt["points"].extend(select_counterexamples(large, listing))
        del large, listing
        metadata = image_metadata(json_read(EVIDENCE_ROOT / INPUT_PINS["download_validation_sha256"][0]))
        receipt["image_inventory"] = bind_images(receipt["points"], metadata)
        receipt["visual_selection"] = visual_selection(receipt["points"])
        require(len(receipt["points"]) <= 772 and len({point["id"] for point in receipt["points"]}) == len(receipt["points"]), "invalid bounded selected inventory")
        receipt["hashes_after"] = bound_hashes(output)
        require(before == receipt["hashes_after"], "inputs/artifacts changed during numerical selection")
        receipt["passed"] = True
        write_exclusive(output / "selection.json", receipt)
        print(json.dumps(dict(selection=str(output / "selection.json"), selection_sha256=sha(output / "selection.json"),
            selected_points=len(receipt["points"]), visual_points=len(receipt["visual_selection"]), source_images_opened=False)))
    except BaseException as exc:
        receipt["error"] = repr(exc)
        write_exclusive(output / "failure.json", receipt)
        raise


def load_approved_image(metadata):
    """Run-phase only: exact approved PNG decode, no conversion or rescaling."""
    from PIL import Image
    allowed = {index for previous in PREVIOUS_INDICES for index in (previous, previous + 1)}
    require(metadata["frame_index"] in allowed, "source frame outside fixed actual-pair scope")
    path = SOURCE_DIR / metadata["img_name"]
    require(SOURCE_DIR.is_dir() and not SOURCE_DIR.is_symlink() and path.parent == SOURCE_DIR
            and path.is_file() and not path.is_symlink() and path.resolve().parent == SOURCE_DIR.resolve(), "missing/linked/out-of-scope source PNG")
    require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["png_sha256"], "approved PNG file identity differs")
    with Image.open(path) as image:
        require(image.format == "PNG" and image.mode == "L" and image.size == (WIDTH, HEIGHT), "native grayscale PNG representation differs")
        array = np.asarray(image).copy()
    require(array.dtype == np.uint8 and array.shape == (HEIGHT, WIDTH)
            and hashlib.sha256(array.tobytes()).hexdigest() == metadata["pixel_sha256"], "decoded native pixels differ")
    require(sha(path) == metadata["png_sha256"], "PNG changed during decode")
    array.setflags(write=False)
    return array


def translate_no_wrap(previous, shift, fill=128):
    require(previous.dtype == np.uint8 and previous.ndim == 2 and len(shift) == 2
            and all(type(value) is int for value in shift) and fill == 128, "invalid exact synthetic translation")
    dx, dy = shift
    height, width = previous.shape
    result = np.full_like(previous, fill)
    sx0, sx1, sy0, sy1 = max(0, -dx), min(width, width - dx), max(0, -dy), min(height, height - dy)
    if sx0 < sx1 and sy0 < sy1:
        result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = previous[sy0:sy1, sx0:sx1]
    return result


def descriptive_summary(rows):
    def counts(sample):
        return dict(selected_count=len(sample), mutually_supported_count=sum(row["comparison"]["two_sizes_qualified_and_consistent"] for row in sample),
            verdict_counts={verdict: sum(row["comparison"]["verdict"] == verdict for row in sample)
                for verdict in ("agrees_with_saved_lk", "disagrees_with_saved_lk", "ambiguous")},
            scales=[dict(template_size=size, available_count=sum(row["scales"][index]["available"] for row in sample),
                unavailable_count=sum(not row["scales"][index]["available"] for row in sample),
                unavailable_reason_counts={reason: sum(row["scales"][index].get("unavailable_reason") == reason for row in sample)
                    for reason in ("previous_template_outside_image", "no_nonzero_variance_correlation")},
                qualified_count=sum(row["scales"][index]["qualified"] for row in sample),
                failed_qualification_counts={key: sum(row["scales"][index].get("qualification", {}).get(key) is False for row in sample)
                    for key in ("ncc", "gap", "template_std", "current_std", "interior_search_peak", "full_unclipped_search")})
                for index, size in enumerate(PATCH_SIZES)])
    output = []
    for previous in PREVIOUS_INDICES:
        sample = [row for row in rows if row["source_kind"] == "actual_residual_extremum" and row["previous_index"] == previous]
        supported = [row for row in sample if row["comparison"]["two_sizes_qualified_and_consistent"]]
        descriptors = []
        for index, size in enumerate(PATCH_SIZES):
            offsets = np.asarray([row["scales"][index]["best_offset_xy"] for row in supported], dtype=float).reshape(-1, 2)
            candidates = np.asarray([row["original_candidate_displacement_xy"] for row in supported], dtype=float).reshape(-1, 2)
            residual = offsets - candidates
            norms = np.linalg.norm(residual, axis=1)
            descriptors.append(dict(template_size=size, count=len(supported),
                original_candidate_residual_median_px=float(np.median(norms)) if len(norms) else None,
                original_candidate_residual_p90_px=float(np.quantile(norms, .9)) if len(norms) else None,
                displacement_component_min=offsets.min(axis=0).tolist() if len(offsets) else None,
                displacement_component_max=offsets.max(axis=0).tolist() if len(offsets) else None,
                residual_component_min=residual.min(axis=0).tolist() if len(residual) else None,
                residual_component_max=residual.max(axis=0).tolist() if len(residual) else None))
        output.append(dict(previous_index=previous, **counts(sample),
            selection_roles={role: counts([row for row in sample if role in row["selection_roles"]]) for role in ("minimum", "maximum")},
            supported_subset_descriptors=descriptors))
    return dict(actual_pairs=output, counterexamples=counts([row for row in rows if row["source_kind"] == "known_shift_counterexample"]),
        role_denominator_note="a deduplicated record may have both roles; role denominators overlap and must not be summed",
        interpretation="purposive supported subset; residuals to original candidate are descriptive, not a refit, truth, or population estimate")


def run(output):
    output = scope_output(output, "run")
    before = bound_hashes(output)
    selection_path, freeze_path = output / "selection.json", output / "selection_freeze.json"
    require(all(path.is_file() and not path.is_symlink() for path in (selection_path, freeze_path)), "missing/linked selection freeze")
    selection_hash, freeze_hash = sha(selection_path), sha(freeze_path)
    validate_selection_freeze(json_read(freeze_path), selection_hash, before["manifest_sha256"])
    selection = json_read(selection_path)
    require(selection.get("schema") == SELECTION_SCHEMA and selection.get("passed") is True
            and selection["hashes_before"] == selection["hashes_after"] == before
            and selection.get("source_images_opened") is False and selection.get("patch_metrics_computed") is False,
            "selection is not a frozen pre-image artifact")
    metadata = image_metadata(json_read(EVIDENCE_ROOT / INPUT_PINS["download_validation_sha256"][0]))
    require(0 < len(selection["points"]) <= 772 and len(selection["image_inventory"]) <= 16, "selection exceeds fixed scope")
    for row in selection["image_inventory"]:
        require(row == metadata[row["frame_index"]], "selection image metadata differs")
    receipt = dict(schema=SCHEMA, passed=False, hashes_before=before, selection_sha256=selection_hash,
        selection_freeze_sha256=freeze_hash, design=DESIGN, rows=[], visual_selection=selection["visual_selection"],
        image_inventory=selection["image_inventory"], images_verified=[], numpy_version=np.__version__,
        annotations_used=False, detector_run=False, private_media_accessed=False, raw16_accessed=False,
        remote_accessed=False, production_changed=False, production_promotion=False,
        interpretation="independent integer-grid image evidence, not subpixel truth; ambiguous does not mean rejected or static")
    try:
        # All hashes and the pre-image selection freeze are checked before here.
        images = {}
        for meta in selection["image_inventory"]:
            images[meta["frame_index"]] = load_approved_image(meta)
            receipt["images_verified"].append(meta)
        synthetic = {}
        for ordinal, point in enumerate(selection["points"]):
            previous = images[point["previous_index"]]
            if point["source_kind"] == "actual_residual_extremum":
                current = images[point["current_index"]]
            else:
                require(point["source_kind"] == "known_shift_counterexample", "unplanned source kind")
                if point["case_id"] not in synthetic:
                    spec = point["synthetic_current"]
                    current = translate_no_wrap(previous, spec["shift_xy"], spec["fill"])
                    require(hashlib.sha256(current.tobytes()).hexdigest() == spec["expected_pixel_sha256"], "synthetic pixels differ from frozen large-shift run")
                    synthetic[point["case_id"]] = current
                current = synthetic[point["case_id"]]
            receipt["rows"].append(evaluate_point(previous, current, point))
            if (ordinal + 1) % 48 == 0 or ordinal + 1 == len(selection["points"]):
                print(json.dumps(dict(completed=ordinal + 1, total=len(selection["points"]))), flush=True)
        receipt["summary"] = descriptive_summary(receipt["rows"])
        receipt["hashes_after"] = bound_hashes(output)
        require(before == receipt["hashes_after"] and sha(selection_path) == selection_hash and sha(freeze_path) == freeze_hash,
                "inputs/artifacts/selection/freeze changed during patch execution")
        receipt["image_png_sha256_after"] = {meta["img_name"]: sha(SOURCE_DIR / meta["img_name"]) for meta in selection["image_inventory"]}
        require(all(receipt["image_png_sha256_after"][meta["img_name"]] == meta["png_sha256"] for meta in selection["image_inventory"]), "source PNG changed during patch execution")
        receipt["passed"] = True
        write_exclusive(output / "result.json", receipt)
    except BaseException as exc:
        receipt["error"] = repr(exc)
        write_exclusive(output / "failure.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--select", action="store_true")
    phases.add_argument("--run", action="store_true")
    args = parser.parse_args()
    (select if args.select else run)(args.output)


if __name__ == "__main__":
    main()
