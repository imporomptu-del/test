#!/usr/bin/env python3
"""Independent, bounded native-patch translation measurements; generated-only CLI.

The search knows only the previous point. It does not import PVA, load real
media, fit global motion, consume saved LK endpoints, or change detector gates.
Qualification is a fixed diagnostic observability check, not calibrated truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SCHEMA = "seaqr.motion-patch-controls.v1"
PATCH_SIZE = 31
PATCH_RADIUS = PATCH_SIZE // 2
SEARCH_RADIUS = 3.0
SEARCH_STEP = 0.25
MAX_BATCH = 64
DEFAULT_SHAPE = (512, 640)
QUERY_OFFSET_XY = (0.25, 0.5)
CONTRACT = {
    "patch_size": PATCH_SIZE, "search_radius_native_px": SEARCH_RADIUS,
    "search_step_native_px": SEARCH_STEP, "search_candidates": 625,
    "search_center": "exact previous point only; never saved LK q or a global transform",
    "sampling": "float64 bilinear, no padding, no warp, no additional peak refinement",
    "tie_policy": "first row-major maximum: lowest dy, then lowest dx",
    "full_previous_and_current_search_support_required": True,
    "minimum_template_and_winning_current_std_dn": 1.0,
    "minimum_ncc": 0.8,
    "runner_exclusion_chebyshev_native_px": 0.75,
    "minimum_runner_gap": 0.02,
    "interior_peak_required": True,
    "minimum_negative_ncc_hessian_eigenvalue_per_px2": 0.001,
    "maximum_negative_ncc_hessian_condition": 100.0,
    "hessian_definition": "central 3x3 finite differences at grid peak, step .25 native px",
    "maximum_candidate_batch": MAX_BATCH,
    "positive_control_maximum_vector_error_native_px": 0.25,
    "observable_positive_controls_must_qualify": True,
    "degenerate_and_out_of_range_controls_must_abstain": True,
    "query_offset_from_integer_image_center_xy": list(QUERY_OFFSET_XY),
    "image_rounding": "numpy rint (ties to even), once after analytic intensity transformation",
    "generated_nominal_intensity_bounds_dn": [80.0, 160.0],
    "generated_output_has_no_clipped_pixels": True,
    "hessian_is_calibrated_uncertainty": False,
    "global_model_fits": 0, "production_changes": False,
}
POSITIVE_CONDITIONS = (
    ("static", (0.0, 0.0), 1.0, 0.0),
    ("shift_half", (0.5, -0.75), 1.0, 0.0),
    ("shift_two", (2.0, -1.0), 1.0, 0.0),
    ("shift_fractional", (-1.25, 0.5), 1.0, 0.0),
    ("static_gain08_offset12", (0.0, 0.0), 0.8, 12.0),
    ("shift_half_gain08_offset12", (0.5, -0.75), 0.8, 12.0),
    ("static_gain12_offsetm12", (0.0, 0.0), 1.2, -12.0),
    ("shift_half_gain12_offsetm12", (0.5, -0.75), 1.2, -12.0),
)
ABSTENTION_CONDITIONS = (
    ("flat", "flat", (0.5, -0.75)),
    ("straight_edge", "edge", (0.5, -0.75)),
    ("periodic_ambiguity", "periodic", (0.5, -0.75)),
    ("out_of_range_shift4", "texture", (4.0, 0.0)),
)
LIMITATIONS = [
    "A qualified patch match is a diagnostic measurement, not independent scene-motion truth.",
    "Only translations inside +/-3 native pixels are searched on a .25-pixel grid.",
    "Boundary rejection does not detect every possible out-of-range or aliased match.",
    "Hessian/gap checks are fixed observability heuristics, not probabilities or confidence intervals.",
    "ZNCC removes positive affine intensity gain/offset but not non-affine changes, clipping, occlusion or moving texture.",
    "Bilinear sampling can introduce phase bias; generated analytic images are sampled independently, not by this sampler.",
    "No saved raw LK rejects, global-motion model, or detector is evaluated here.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha(image):
    image = np.ascontiguousarray(image)
    prefix = json.dumps({"dtype": image.dtype.str, "shape": list(image.shape)},
                        sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return hashlib.sha256(prefix + image.tobytes()).hexdigest()


def _image(image):
    image = np.asarray(image)
    require(image.ndim == 2 and image.dtype == np.uint8 and min(image.shape) >= PATCH_SIZE,
            "Native two-dimensional uint8 grayscale image required")
    return image


def bilinear_patches(image, centers):
    """At most 64 complete 31x31 supports, exact fractional weights, no padding."""
    image = _image(image)
    centers = np.asarray(centers, dtype=np.float64)
    require(centers.ndim == 2 and centers.shape[1] == 2 and 0 < len(centers) <= MAX_BATCH
            and np.isfinite(centers).all(), "Invalid bounded batch of finite native points")
    offsets = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=np.float64)
    xs, ys = centers[:, 0, None] + offsets, centers[:, 1, None] + offsets
    height, width = image.shape
    require(xs.min() >= 0 and ys.min() >= 0 and xs.max() <= width - 1 and ys.max() <= height - 1,
            "Entire bilinear patch must be in the source image")
    x0, y0 = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
    # At an exact final pixel only, the clipped neighbor has zero weight.
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = xs - x0, ys - y0
    upper = image[y0[:, :, None], x0[:, None, :]] * (1 - wx[:, None, :])
    upper += image[y0[:, :, None], x1[:, None, :]] * wx[:, None, :]
    lower = image[y1[:, :, None], x0[:, None, :]] * (1 - wx[:, None, :])
    lower += image[y1[:, :, None], x1[:, None, :]] * wx[:, None, :]
    return upper * (1 - wy[:, :, None]) + lower * wy[:, :, None]


def search_geometry(point, shape):
    point = np.asarray(point, dtype=np.float64)
    require(point.shape == (2,) and np.isfinite(point).all(), "Finite previous native x/y required")
    require(len(shape) == 2 and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                  and v >= PATCH_SIZE for v in shape), "Invalid source image shape")
    limit = np.array([shape[1] - 1, shape[0] - 1], dtype=np.float64)
    previous_ok = bool(np.all(point - PATCH_RADIUS >= 0) and np.all(point + PATCH_RADIUS <= limit))
    full_ok = bool(np.all(point - PATCH_RADIUS - SEARCH_RADIUS >= 0)
                   and np.all(point + PATCH_RADIUS + SEARCH_RADIUS <= limit))
    return dict(previous_xy=point.tolist(), source_shape_hw=list(shape),
                previous_template_supported=previous_ok, full_search_supported=full_ok,
                support_available=previous_ok and full_ok,
                unavailable_reason=None if previous_ok and full_ok else
                "previous_template_outside_image" if not previous_ok else "incomplete_current_search_support")


def score_surface(previous, current, previous_xy, batch_size=MAX_BATCH):
    """Return the complete fixed score surface; never consult an LK endpoint."""
    started = time.perf_counter()
    previous, current = _image(previous), _image(current)
    require(previous.shape == current.shape, "Native image dimensions differ")
    require(type(batch_size) is int and 1 <= batch_size <= MAX_BATCH, "Candidate batch exceeds fixed cap")
    geometry = search_geometry(previous_xy, previous.shape)
    result = dict(geometry=geometry, scores=None, current_std=None, template_std=None,
                  template_mean=None, template=None, candidate_batches=0)
    if not geometry["support_available"]:
        result["elapsed_seconds"] = time.perf_counter() - started
        return result
    p = np.asarray(previous_xy, np.float64)
    template = bilinear_patches(previous, p[None])[0]
    template_mean = float(template.mean())
    centered = template - template_mean
    template_energy = float(np.sum(centered * centered))
    axis = np.arange(-12, 13, dtype=np.float64) * SEARCH_STEP
    dx, dy = np.meshgrid(axis, axis)
    displacement = np.column_stack((dx.ravel(), dy.ravel()))
    scores, stds = np.full(625, np.nan, np.float64), np.zeros(625, np.float64)
    for start in range(0, len(displacement), batch_size):
        patches = bilinear_patches(current, p + displacement[start:start + batch_size])
        patches -= np.mean(patches, axis=(1, 2), keepdims=True)
        energies = np.sum(patches * patches, axis=(1, 2))
        stds[start:start + len(patches)] = np.sqrt(energies / template.size)
        denominator = np.sqrt(template_energy * energies)
        numerator = np.sum(patches * centered, axis=(1, 2))
        valid = denominator > 0
        values = np.full(len(patches), np.nan, np.float64)
        values[valid] = numerator[valid] / denominator[valid]
        require(np.all(np.abs(values[valid]) <= 1 + 1e-10), "Invalid ZNCC arithmetic")
        scores[start:start + len(patches)] = values
        result["candidate_batches"] += 1
    result.update(scores=scores.reshape(25, 25), current_std=stds.reshape(25, 25),
                  template_std=math.sqrt(template_energy / template.size), template_mean=template_mean,
                  template=template, elapsed_seconds=time.perf_counter() - started)
    return result


def peak_diagnostics(scores, current_std, template_std):
    scores, current_std = np.asarray(scores, np.float64), np.asarray(current_std, np.float64)
    require(scores.shape == current_std.shape == (25, 25), "Fixed 25x25 score surface required")
    require(not np.isinf(scores).any() and np.isfinite(current_std).all() and np.all(current_std >= 0)
            and math.isfinite(template_std) and template_std >= 0, "Invalid patch statistics")
    finite = np.isfinite(scores)
    result = dict(qualified=False, displacement_xy=None, raw_best_displacement_xy=None,
        best_ncc=None, runner_ncc=None, runner_gap=None, best_current_std_dn=None,
        template_std_dn=float(template_std), best_at_search_boundary=None,
        negative_ncc_hessian=None, hessian_eigenvalues_ascending=None, hessian_condition=None,
        finite_candidates=int(finite.sum()), qualification={}, abstention_reasons=[])
    if not finite.any():
        result["abstention_reasons"] = ["no_finite_nonzero_variance_correlation"]
        return result
    row, col = np.unravel_index(int(np.argmax(np.where(finite, scores, -np.inf))), scores.shape)
    best = float(scores[row, col])
    yy, xx = np.indices(scores.shape)
    runner_mask = finite & (np.maximum(np.abs(yy - row), np.abs(xx - col)) * SEARCH_STEP > .75)
    runner = float(np.max(scores[runner_mask])) if runner_mask.any() else None
    gap = None if runner is None else best - runner
    boundary = bool(row in (0, 24) or col in (0, 24))
    eigenvalues, condition, hessian = None, None, None
    if not boundary and np.isfinite(scores[row-1:row+2, col-1:col+2]).all():
        hxx = (scores[row, col+1] - 2*best + scores[row, col-1]) / SEARCH_STEP**2
        hyy = (scores[row+1, col] - 2*best + scores[row-1, col]) / SEARCH_STEP**2
        hxy = (scores[row+1, col+1] - scores[row+1, col-1]
               - scores[row-1, col+1] + scores[row-1, col-1]) / (4*SEARCH_STEP**2)
        hessian = -np.array([[hxx, hxy], [hxy, hyy]], np.float64)
        eigenvalues = np.linalg.eigvalsh(hessian)
        if eigenvalues[0] > 0:
            condition = float(eigenvalues[1] / eigenvalues[0])
    checks = dict(template_std=template_std >= 1.0, current_std=current_std[row, col] >= 1.0,
        ncc=best >= .8, runner_gap=gap is not None and gap >= .02, interior_peak=not boundary,
        hessian_minimum=eigenvalues is not None and eigenvalues[0] >= .001,
        hessian_condition=condition is not None and condition <= 100.0)
    qualified = bool(all(checks.values()))
    offset = [(int(col)-12)*SEARCH_STEP, (int(row)-12)*SEARCH_STEP]
    result.update(qualified=qualified, displacement_xy=offset if qualified else None,
        raw_best_displacement_xy=offset, best_ncc=best, runner_ncc=runner, runner_gap=gap,
        best_current_std_dn=float(current_std[row, col]), best_at_search_boundary=boundary,
        negative_ncc_hessian=None if hessian is None else hessian.tolist(),
        hessian_eigenvalues_ascending=None if eigenvalues is None else eigenvalues.tolist(),
        hessian_condition=condition, qualification={k: bool(v) for k, v in checks.items()},
        abstention_reasons=[k for k, v in checks.items() if not v])
    return result


def measure_patch(previous, current, previous_xy, return_surface=False):
    started = time.perf_counter()
    surface = score_surface(previous, current, previous_xy)
    result = dict(geometry=surface["geometry"], qualified=False, displacement_xy=None,
        raw_best_displacement_xy=None, best_ncc=None, runner_ncc=None, runner_gap=None,
        template_std_dn=None, best_current_std_dn=None, best_at_search_boundary=None,
        negative_ncc_hessian=None, hessian_eigenvalues_ascending=None, hessian_condition=None,
        finite_candidates=0, qualification={}, abstention_reasons=[], photometry_at_raw_peak=None)
    if surface["scores"] is None:
        result["abstention_reasons"] = [surface["geometry"]["unavailable_reason"]]
    else:
        result.update(peak_diagnostics(surface["scores"], surface["current_std"], surface["template_std"]))
        offset = result["raw_best_displacement_xy"]
        if offset is not None:
            template = surface["template"]
            winning = bilinear_patches(current, (np.asarray(previous_xy) + offset)[None])[0]
            t0, w0 = template - template.mean(), winning - winning.mean()
            gain = float(np.sum(t0*w0) / np.sum(t0*t0))
            intercept = float(winning.mean() - gain * template.mean())
            result["photometry_at_raw_peak"] = dict(
                template_mean_dn=float(template.mean()), current_mean_dn=float(winning.mean()),
                mean_difference_dn=float(winning.mean()-template.mean()),
                affine_current_from_template_gain=gain, affine_current_from_template_offset_dn=intercept,
                raw_rmse_dn=float(np.sqrt(np.mean((winning-template)**2))),
                zero_mean_rmse_dn=float(np.sqrt(np.mean((w0-t0)**2))),
                affine_adjusted_rmse_dn=float(np.sqrt(np.mean((winning-(gain*template+intercept))**2))),
                feeds_search_or_qualification=False)
        if return_surface:
            result["score_surface_dy_dx"] = [[float(x) if np.isfinite(x) else None for x in row]
                                               for row in surface["scores"]]
    result["cost"] = dict(elapsed_seconds=time.perf_counter()-started,
        score_surface_seconds=surface["elapsed_seconds"], candidate_batches=surface["candidate_batches"],
        candidate_count=625 if surface["scores"] is not None else 0, maximum_candidate_batch=MAX_BATCH,
        includes_python_and_diagnostic_statistics=True, not_production_runtime=True)
    return result


def control_inventory(shape=DEFAULT_SHAPE):
    require(isinstance(shape, (tuple, list)) and len(shape) == 2
            and all(type(v) is int for v in shape)
            and 96 <= shape[0] <= 3190 and 96 <= shape[1] <= 4784, "Unsupported generated image shape")
    height, width = shape
    point = [width//2 + QUERY_OFFSET_XY[0], height//2 + QUERY_OFFSET_XY[1]]
    cases = []
    for family in ("texture", "corner"):
        for name, shift, gain, offset in POSITIVE_CONDITIONS:
            cases.append(dict(case_id=family+"__"+name, family=family, source_shape_hw=list(shape),
                point_queries_xy=[point.copy()], truth_displacement_xy=list(shift),
                current_gain=gain, current_offset_dn=offset, expectation="qualified_with_error_at_most_0.25px"))
    for name, family, shift in ABSTENTION_CONDITIONS:
        cases.append(dict(case_id=name, family=family, source_shape_hw=list(shape),
            point_queries_xy=[point.copy()], truth_displacement_xy=list(shift),
            current_gain=1.0, current_offset_dn=0.0, expectation="abstain"))
    return dict(schema=SCHEMA+".inventory", default_shape_hw=list(DEFAULT_SHAPE), shape_hw=list(shape),
                contract=json.loads(json.dumps(CONTRACT)), cases=cases,
                positive_pairs=16, abstention_pairs=4, total_pairs=20, total_queries=20)


def _analytic(family, x, y):
    if family == "texture":
        return 120 + 10*np.cos(.67*x+.23*y) + 9*np.sin(-.31*x+.79*y) \
            + 8*np.cos(1.03*x-.57*y) + 7*np.sin(.15*x+1.13*y)
    if family == "corner":
        return 120 + 20*np.tanh(x/1.2) + 20*np.tanh(y/1.2)
    if family == "edge":
        return 120 + 40*np.tanh(x/1.2) + np.zeros_like(y)
    if family == "periodic":
        return 120 + 18*np.cos(np.pi*x) + 18*np.cos(np.pi*y)
    if family == "flat":
        return np.zeros(np.broadcast_shapes(x.shape, y.shape), np.float64) + 120
    raise ValueError("Unknown generated family")


def generated_controls(shape=DEFAULT_SHAPE):
    """One pair at a time; current analytic image is sampled independently."""
    inventory = control_inventory(shape)
    height, width = shape
    x = np.arange(width, dtype=np.float64)[None, :] - width//2
    y = np.arange(height, dtype=np.float64)[:, None] - height//2
    for case in inventory["cases"]:
        dx, dy = case["truth_displacement_xy"]
        previous = _analytic(case["family"], x, y)
        current = case["current_gain"]*_analytic(case["family"], x-dx, y-dy) + case["current_offset_dn"]
        require(np.isfinite(previous).all() and np.isfinite(current).all()
                and previous.min() >= 0 and previous.max() <= 255
                and current.min() >= 0 and current.max() <= 255, "Generated photometry would require clipping")
        before, after = np.rint(previous).astype(np.uint8), np.rint(current).astype(np.uint8)
        before.flags.writeable = after.flags.writeable = False
        yield dict(case=case, previous_gray=before, current_gray=after,
            previous_pixel_sha256=array_sha(before), current_pixel_sha256=array_sha(after),
            quantization=dict(previous_maximum_abs_rounding_error_dn=float(np.max(np.abs(previous-before))),
                current_maximum_abs_rounding_error_dn=float(np.max(np.abs(current-after))), clipped_pixels=0))


def assess_control(case, measurement):
    expected = case["expectation"]
    require(expected in ("abstain", "qualified_with_error_at_most_0.25px"), "Unknown control expectation")
    truth = np.asarray(case["truth_displacement_xy"], np.float64)
    raw = measurement["raw_best_displacement_xy"]
    raw_error = None if raw is None else float(np.linalg.norm(np.asarray(raw)-truth))
    qualified = measurement["qualified"] is True
    error = raw_error if qualified else None
    passed = not qualified if expected == "abstain" else qualified and error is not None and error <= .25
    return dict(passed=bool(passed), expectation=expected, qualified=qualified,
        qualified_vector_error_px=error, raw_argmax_error_px_not_a_valid_measurement=raw_error)


def run_generated(output, shape=DEFAULT_SHAPE):
    output = Path(output).absolute()
    require(output.suffix == ".json" and output.parent.is_dir() and output.parent.resolve() == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh JSON output in an existing real directory required")
    script = Path(__file__).resolve()
    test = script.parents[1] / "tests/unit/test_motion_patch_controls.py"
    bindings = {str(path): sha(path) for path in (script, test)}
    started = time.perf_counter()
    rows = []
    for pair in generated_controls(shape):
        case = pair["case"]
        measurements = []
        for point in case["point_queries_xy"]:
            measured = measure_patch(pair["previous_gray"], pair["current_gray"], point, return_surface=True)
            measurements.append(dict(previous_xy=point, measurement=measured,
                                     assessment=assess_control(case, measured)))
        rows.append(dict(case=case, previous_pixel_sha256=pair["previous_pixel_sha256"],
            current_pixel_sha256=pair["current_pixel_sha256"], quantization=pair["quantization"], queries=measurements))
    require(all(sha(Path(path)) == digest for path, digest in bindings.items()), "Code changed during generated validation")
    inventory = control_inventory(shape)
    require([r["case"]["case_id"] for r in rows] == [c["case_id"] for c in inventory["cases"]], "Control inventory differs")
    passed = all(q["assessment"]["passed"] for row in rows for q in row["queries"])
    result = dict(schema=SCHEMA+".result", passed=passed, generated_only=True, source_media_accessed=False,
        production_changes=False, model_fits=0, inventory=inventory, rows=rows,
        input_sha256=bindings, numpy_version=np.__version__, limitations=LIMITATIONS,
        elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat(),
        scientific_failure_is_retained_without_retuning=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[1])
    options = parser.parse_args()
    result = run_generated(options.output, (options.height, options.width))
    print(json.dumps(dict(passed=result["passed"], generated_pairs=len(result["rows"]),
                          elapsed_seconds=result["elapsed_seconds"], output=str(options.output))))
    raise SystemExit(0 if result["passed"] else 1)
