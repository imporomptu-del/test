"""Independent numerical audit of V42 saved metadata and preselected NPZs.

This implementation imports neither production nor experiment inference code.
It uses scalar medians, direct box sums, scalar bilinear sampling and QR solves
to check the saved causal forecast and localized-template calculations. It
does not decode raw media, verify global-motion accuracy, or classify objects.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


N = 129
MID = 64
ATOL = 2e-6
RTOL = 2e-8


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def project(matrix, point):
    homogeneous = [float(point[0]), float(point[1]), 1.0]
    mapped = [math.fsum(float(v)*p for v, p in zip(row, homogeneous)) for row in matrix]
    if mapped[2] == 0:
        raise ValueError("Unsupported homogeneous point")
    return np.asarray([mapped[0]/mapped[2], mapped[1]/mapped[2]])


def solve(design, target):
    """QR for full column rank; independent pinv for deficient dictionaries."""
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.shape[1] == 0:
        return np.empty((0, *target.shape[1:])), 0
    rank = int(np.linalg.matrix_rank(design))
    if rank == design.shape[1]:
        q, r = np.linalg.qr(design, mode="reduced")
        return np.linalg.solve(r, q.T @ target), rank
    return np.linalg.pinv(design, rcond=np.finfo(float).eps*max(design.shape)) @ target, rank


def scalar_sample(image, x, y):
    """No clipped/wrapped support: require exactly positive-weight corners."""
    if not math.isfinite(x) or not math.isfinite(y):
        return math.nan
    left, top = math.floor(x), math.floor(y)
    fx, fy = x-left, y-top
    total = 0.0
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        if weight == 0:
            continue
        xx, yy = left+dx, top+dy
        if xx < 0 or xx >= image.shape[1] or yy < 0 or yy >= image.shape[0]:
            return math.nan
        value = float(image[yy, xx])
        if not math.isfinite(value):
            return math.nan
        total += weight*value
    return total


def scalar_stamp(image, center):
    return np.asarray([[scalar_sample(image, center[0]+x-8, center[1]+y-8)
                        for x in range(17)] for y in range(17)], dtype=float)


def aperture():
    one = [0.5-0.5*math.cos(2*math.pi*i/16) for i in range(17)]
    return np.outer(one, one)


APERTURE = aperture()


def normalize_stamp(stamp):
    support = np.isfinite(stamp) & (APERTURE > 0)
    if int(support.sum()) < 64:
        return None
    weighted = np.maximum(stamp, 0)*APERTURE
    weighted[APERTURE == 0] = 0
    energy = math.fsum(float(v)*float(v) for v in weighted.flat if math.isfinite(v))
    if not energy > 1e-12:
        return None
    return weighted/math.sqrt(energy)


def combine(stamps):
    if len(stamps) < 3:
        return None
    combined = np.full((17, 17), np.nan)
    for y in range(17):
        for x in range(17):
            values = [float(stamp[y, x]) for stamp in stamps if np.isfinite(stamp[y, x])]
            if len(values) >= 3:
                combined[y, x] = statistics.median(values)
    combined[APERTURE == 0] = 0
    if int((np.isfinite(combined) & (APERTURE > 0)).sum()) < 64:
        return None
    energy = math.fsum(float(v)*float(v) for v in combined.flat if math.isfinite(v))
    return combined/math.sqrt(energy) if energy > 1e-12 else None


def place(stamp, center):
    result = np.zeros((N, N))
    for y in range(max(0, math.ceil(center[1]-8)), min(N-1, math.floor(center[1]+8))+1):
        for x in range(max(0, math.ceil(center[0]-8)), min(N-1, math.floor(center[0]+8))+1):
            result[y, x] = scalar_sample(stamp, x-center[0]+8, y-center[1]+8)
    return result


def direct_highpass(image):
    # Direct per-window reduction, not the inference code's integral-image path.
    result = np.full((N, N), np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(image, (9, 9))
    valid = np.isfinite(windows).all(axis=(-2, -1))
    mean = windows.sum(axis=(-2, -1))/81.0
    result[4:-4, 4:-4] = np.where(valid, image[4:-4, 4:-4]-mean, np.nan)
    return result


def find_anchors(highpasses):
    observations = []
    for frame, image in enumerate(highpasses):
        for y in range(44, 85):
            for x in range(44, 85):
                value = float(image[y, x])
                if not math.isfinite(value) or value < 1:
                    continue
                neighbours = image[y-1:y+2, x-1:x+2]
                finite = neighbours[np.isfinite(neighbours)]
                if len(finite) and value == float(finite.max()):
                    observations.append((frame, x, y, value))
    proposed = []
    seeds = sorted({(x, y) for _, x, y, _ in observations}, key=lambda xy: (xy[1], xy[0]))
    for x, y in seeds:
        strongest = {}
        for frame, px, py, value in observations:
            if max(abs(px-x), abs(py-y)) <= 2:
                strongest[frame] = max(strongest.get(frame, -math.inf), value)
        if len(strongest) >= 3:
            proposed.append(dict(x=x, y=y, prior_frame_indices=sorted(strongest),
                                 repeat_frame_count=len(strongest),
                                 median_contrast_dn=statistics.median(strongest.values())))
    proposed.sort(key=lambda a: (-a["repeat_frame_count"], -a["median_contrast_dn"], a["y"], a["x"]))
    chosen = []
    for item in proposed:
        if all(max(abs(item["x"]-other["x"]), abs(item["y"]-other["y"])) > 2 for other in chosen):
            chosen.append(item)
    return chosen


def independent_components(history, centers, offset, polarity):
    sign = 1 if polarity == "bright" else -1
    background = np.full((N, N), np.nan)
    counts = np.zeros((N, N), dtype=int)
    for y in range(N):
        for x in range(N):
            values = []
            for frame in range(8):
                center = centers[frame]
                if center is not None and max(abs(x-center[0]), abs(y-center[1])) <= 8:
                    continue
                value = float(history[frame, y, x])
                if math.isfinite(value):
                    values.append(value)
            counts[y, x] = len(values)
            if len(values) >= 3:
                background[y, x] = statistics.median(values)
    stamps, usable = [], []
    for frame, center in enumerate(centers):
        if center is None:
            continue
        stamp = normalize_stamp(scalar_stamp(sign*(history[frame]-background), center))
        if stamp is not None:
            stamps.append(stamp)
            usable.append(frame)
    moving_stamp = combine(stamps)
    predicted = np.asarray(offset)+64
    moving = None if moving_stamp is None else place(moving_stamp, predicted)
    highpasses = [sign*direct_highpass(image) for image in history]
    discovered = find_anchors(highpasses)
    overlapping = [[item["x"], item["y"]] for item in discovered
                   if max(abs(item["x"]-predicted[0]), abs(item["y"]-predicted[1])) <= 2]
    fixed, anchors = [], []
    for anchor in discovered[:4]:
        stamps = [normalize_stamp(scalar_stamp(image, (anchor["x"], anchor["y"])))
                  for image in highpasses]
        stamps = [stamp for stamp in stamps if stamp is not None]
        merged = combine(stamps)
        anchors.append(dict(anchor, usable_stamp_count=len(stamps), template_available=merged is not None))
        if merged is not None:
            fixed.append(place(merged, (anchor["x"], anchor["y"])))
    reasons = []
    if moving is None:
        reasons.append("insufficient_causal_foreground_template")
    if any(not anchor["template_available"] for anchor in anchors):
        reasons.append("incomplete_persistent_anchor_template")
    metadata = dict(polarity=polarity, prior_count=8,
                    usable_moving_stamp_indices=usable, usable_moving_stamp_count=len(usable),
                    background_available_pixel_count=int(np.isfinite(background).sum()),
                    background_observation_count_min=int(counts.min()),
                    background_observation_count_max=int(counts.max()),
                    persistent_anchor_count_before_cap=len(discovered),
                    causal_predicted_center_xy=predicted.tolist(),
                    persistent_anchor_centres_near_prediction=overlapping,
                    persistent_anchor_overlaps_prediction=bool(overlapping),
                    fixed_anchor_count=len(fixed), anchors=anchors,
                    anchor_dictionary_truncated=len(discovered) > 4,
                    causal_component_reasons=reasons)
    return dict(metadata=metadata, background=background, moving_template=moving,
                fixed_templates=np.stack(fixed) if fixed else np.empty((0, N, N)),
                background_observation_counts=counts, moving_stamp=moving_stamp)


def independent_localized(current, history, centers, offset, polarity):
    components = independent_components(history, centers, offset, polarity)
    metadata = components["metadata"]
    result = dict(available=False, ambiguous=False, reasons=[], ambiguity_reasons=[],
                  components=metadata, background_fit=None, common_support_count=0,
                  common_support_sha256=None, fold_support_counts=[0, 0], folds=[],
                  moving_relative_energy_outside_fixed_span=None, mse_stationary=None,
                  mse_augmented=None, advantage_stationary_minus_augmented=None)
    if metadata["anchor_dictionary_truncated"]:
        result["ambiguity_reasons"].append("persistent_anchor_dictionary_truncated")
    if metadata["persistent_anchor_overlaps_prediction"]:
        result["ambiguity_reasons"].append("causal_prediction_overlaps_persistent_fixed_anchor")
    result["ambiguous"] = bool(result["ambiguity_reasons"])
    if metadata["causal_component_reasons"]:
        result["reasons"].extend(metadata["causal_component_reasons"])
        return result
    background = components["background"]
    coordinates = [(x, y) for y in range(34, 95) for x in range(34, 95)
                   if max(abs(x-64), abs(y-64)) >= 16
                   and math.isfinite(background[y, x]) and math.isfinite(current[y, x])]
    if len(coordinates) < 64:
        result["reasons"].append("insufficient_annulus_support")
        return result
    plane = np.asarray([[1, (x-64)/32, (y-64)/32] for x, y in coordinates])
    design = np.column_stack(([background[y, x] for x, y in coordinates], plane))
    response = np.asarray([current[y, x] for x, y in coordinates])
    beta, rank = solve(design, response)
    result["background_fit"] = dict(support_count=len(coordinates), design_rank=rank,
                                    gain=None, plane_coefficients=None,
                                    nonnegative_gain_constraint_active=None)
    if rank != 4:
        result["reasons"].append("flat_or_rank_deficient_annulus_background")
        return result
    active = bool(beta[0] < 0)
    if active:
        beta = np.r_[0.0, solve(plane, response)[0]]
    result["background_fit"].update(gain=float(beta[0]), plane_coefficients=beta[1:].tolist(),
                                     nonnegative_gain_constraint_active=active)
    ys, xs = np.indices((N, N))
    baseline = beta[0]*background+beta[1]+beta[2]*(xs-64)/32+beta[3]*(ys-64)/32
    residual = (current-baseline)*(1 if polarity == "bright" else -1)
    current_core = residual[52:77, 52:77]
    moving = components["moving_template"][52:77, 52:77]
    fixed = components["fixed_templates"][:, 52:77, 52:77]
    common = np.isfinite(current_core) & np.isfinite(moving)
    for image in fixed:
        common &= np.isfinite(image)
    yy, xx = np.indices((25, 25))
    parity = (yy+xx) % 2
    counts = [int((common & (parity == i)).sum()) for i in (0, 1)]
    result.update(common_support_count=int(common.sum()), fold_support_counts=counts,
                  common_support_sha256=hashlib.sha256(common.astype(np.uint8).tobytes()).hexdigest())
    if int(common.sum()) < 64 or min(counts) < 32:
        result["reasons"].append("insufficient_common_core_or_checkerboard_support")
        return result

    def outside_energy(column, dictionary):
        energy = float(column @ column)
        if energy <= 0:
            return 0.0
        remainder = column-dictionary @ solve(dictionary, column)[0]
        return float((remainder @ remainder)/energy)

    result["moving_relative_energy_outside_fixed_span"] = outside_energy(moving[common], fixed[:, common].T)
    if result["moving_relative_energy_outside_fixed_span"] <= 1e-8:
        result["ambiguity_reasons"].append("moving_template_redundant_with_fixed_dictionary")
    errors_sum = [0.0, 0.0]
    for fold in (0, 1):
        train = common & (parity == fold)
        test = common & (parity != fold)
        a, b = fixed[:, train].T, fixed[:, test].T
        stationary, stationary_rank = solve(a, current_core[train])
        augmented, augmented_rank = solve(np.column_stack((a, moving[train])), current_core[train])
        energy = outside_energy(moving[train], a)
        redundant, active = energy <= 1e-8, bool(augmented[-1] < 0)
        if redundant or active:
            augmented = np.r_[stationary, 0.0]
        error_stationary = current_core[test]-b @ stationary
        error_augmented = current_core[test]-b @ augmented[:-1]-moving[test]*augmented[-1]
        sse = [math.fsum(float(v)*float(v) for v in error) for error in (error_stationary, error_augmented)]
        errors_sum = [total+error for total, error in zip(errors_sum, sse)]
        if stationary_rank != len(fixed):
            result["ambiguity_reasons"].append(f"fixed_dictionary_rank_deficient:{fold}")
        if redundant or augmented_rank != len(fixed)+1:
            result["ambiguity_reasons"].append(f"moving_template_unidentifiable_on_training_fold:{fold}")
        result["folds"].append(dict(training_parity=fold, training_count=counts[fold], heldout_count=counts[1-fold],
                                    stationary_rank=stationary_rank, augmented_rank=augmented_rank,
                                    moving_relative_energy_outside_fixed_span=energy,
                                    stationary_signed_amplitudes=stationary.tolist(),
                                    augmented_signed_fixed_amplitudes=augmented[:-1].tolist(),
                                    moving_nonnegative_amplitude=float(augmented[-1]),
                                    moving_amplitude_identifiable=not redundant,
                                    moving_nonnegative_constraint_active=active and not redundant,
                                    redundant_moving_column_zeroed_by_convention=redundant,
                                    mse_stationary=sse[0]/counts[1-fold], mse_augmented=sse[1]/counts[1-fold]))
    result.update(available=True, ambiguous=bool(result["ambiguity_reasons"]),
                  mse_stationary=errors_sum[0]/int(common.sum()), mse_augmented=errors_sum[1]/int(common.sum()),
                  advantage_stationary_minus_augmented=(errors_sum[0]-errors_sum[1])/int(common.sum()))
    return result


def compare(expected, observed, path="", issues=None):
    """Expected keys form the independently audited subset of output fields."""
    issues = [] if issues is None else issues
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            issues.append(dict(path=path, reason="not_mapping"))
            return issues
        for key, value in expected.items():
            if key not in observed:
                issues.append(dict(path=path+"."+key, reason="missing_key"))
            else:
                compare(value, observed[key], path+"."+key, issues)
    elif isinstance(expected, (tuple, list, np.ndarray)):
        if not isinstance(observed, (tuple, list, np.ndarray)) or len(expected) != len(observed):
            issues.append(dict(path=path, reason="wrong_sequence_length"))
        else:
            for i, (a, b) in enumerate(zip(expected, observed)):
                compare(a, b, path+f"[{i}]", issues)
    elif isinstance(expected, (float, np.floating)):
        if (isinstance(observed, bool) or not isinstance(observed, (int, float, np.number))
                or not math.isclose(float(expected), float(observed), rel_tol=RTOL, abs_tol=ATOL)):
            issues.append(dict(path=path, reason="numeric_mismatch", expected=float(expected), observed=observed))
    elif expected != observed:
        issues.append(dict(path=path, reason="exact_mismatch", expected=expected, observed=observed))
    return issues


def forecast_audit(geometry):
    records = geometry["prior_measurements"]
    issues = []
    if len(records) < 5:
        raise ValueError("Available forecast lacks five prior actual measurements")
    timestamps = [item["timestamp_ns"] for item in records]
    current_ns = geometry["current_timestamp_ns"]
    if any(timestamp >= current_ns for timestamp in timestamps):
        raise ValueError("Noncausal measurement timestamp")
    references = np.asarray([project(item["source_to_reference"], item["source_xy"]) for item in records])
    compare(references, [item["reference_xy"] for item in records], "prior_reference_xy", issues)
    u = np.asarray([(timestamp-current_ns)/800_000_000 for timestamp in timestamps])
    design = np.column_stack((np.ones(len(u)), u, u*u))
    coefficients, rank = solve(design, references)
    forecast = project(np.linalg.inv(geometry["current_source_to_reference"]), coefficients[0])
    rmse = float(np.sqrt(np.mean(np.sum((references-design @ coefficients)**2, axis=1))))
    # The independently fitted float may straddle an exact half-pixel tie by
    # roundoff. Audit source forecast numerically, but derive grid from the
    # saved source forecast to test its actual documented rounding contract.
    saved_forecast = np.asarray(geometry["predicted_source_xy"])
    center = np.floor(saved_forecast+.5)
    compare(dict(measured_prior_count=len(records), forecast_fit_rank=rank,
                 forecast_fit_rmse_reference_px=rmse,
                 predicted_reference_xy=coefficients[0].tolist(), predicted_source_xy=forecast.tolist(),
                 current_center_xy=center.tolist(), predicted_offset_xy=(saved_forecast-center).tolist()),
            geometry, "geometry", issues)
    inverse = np.linalg.inv(geometry["current_source_to_reference"])
    frame_to_reference = {item["frame_index"]: ref for item, ref in zip(records, references)}
    centers = [None if frame not in frame_to_reference else
               (project(inverse, frame_to_reference[frame])-center+64).tolist()
               for frame in geometry["prior_frame_indices"]]
    compare(centers, geometry["prior_centers_xy"], "geometry.prior_centers_xy", issues)
    return dict(passed=not issues, measurement_count=len(records), issues=issues,
                forecast_source_max_abs_difference=float(np.max(np.abs(forecast-saved_forecast))))


def archive_geometry_audit(inputs, geometry):
    """Connect saved model inputs to the separately audited forecast metadata."""
    shapes = dict(current129=(129, 129), history129=(8, 129, 129),
                  prior_centers_xy=(8, 2), predicted_offset_xy=(2,))
    if set(inputs) != set(shapes):
        raise ValueError("Unexpected inference-input archive members")
    for key, shape in shapes.items():
        value = inputs[key]
        if value.shape != shape or value.dtype != np.float64 or np.isinf(value).any():
            raise ValueError("Invalid inference-input archive member: "+key)
    raw_centers = inputs["prior_centers_xy"]
    if not np.all(np.isfinite(raw_centers).all(axis=1) | np.isnan(raw_centers).all(axis=1)):
        raise ValueError("A missing center must have both coordinates NaN")
    centers = [point.tolist() if np.isfinite(point).all() else None for point in raw_centers]
    if not np.isfinite(inputs["predicted_offset_xy"]).all():
        raise ValueError("Forecast offset must be finite")
    return compare(dict(prior_centers_xy=centers,
                        predicted_offset_xy=inputs["predicted_offset_xy"].tolist(),
                        current_supported_pixels=int(np.isfinite(inputs["current129"]).sum()),
                        history_supported_pixels=np.isfinite(inputs["history129"]).sum(axis=(1, 2)).tolist()),
                   geometry, "archive_geometry")


def run(replay, output):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Independent audit output must be new")
    receipt_path = replay/"completion_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt["completed"]:
        raise ValueError("Replay is not complete")
    tests_path = Path(__file__).resolve().parents[1]/"tests/unit/test_accuracy_v42_numerical_audit.py"
    checked = {str(receipt_path): digest(receipt_path), str(Path(__file__).resolve()): digest(__file__),
               str(tests_path): digest(tests_path)}
    for name in ("selection.json", "states.jsonl"):
        path = replay/name
        value = digest(path)
        if receipt["files_sha256"].get(str(path)) != value:
            raise ValueError("Unbound or changed replay artifact: "+str(path))
        checked[str(path)] = value
    selection = json.loads((replay/"selection.json").read_text())
    records = [json.loads(line) for line in (replay/"states.jsonl").read_text().splitlines()]
    if len(records) != selection["unique_states"]:
        raise ValueError("Incomplete states")
    state_keys = [(r["clip"], r["frame_index"], r["segment"], r["track_id"]) for r in records]
    expected_keys = {(r["clip"], r["frame_index"], r["segment"], r["track_id"]) for r in selection["states"]}
    if len(set(state_keys)) != len(state_keys) or set(state_keys) != expected_keys:
        raise ValueError("Duplicate, missing or unexpected state identities")
    selected = {tuple(key) for key in selection["audit_state_keys"]}
    counts = Counter(states=len(records))
    details, all_issues = [], []
    for record in records:
        key = (record["clip"], record["frame_index"], record["segment"], record["track_id"])
        detail = dict(key=list(key), forecast=None, component_and_score=None)
        if record["geometry"]["available"]:
            counts["forecast_checked"] += 1
            detail["forecast"] = forecast_audit(record["geometry"]["geometry"])
            all_issues.extend(dict(key=list(key), **issue) for issue in detail["forecast"]["issues"])
        if key in selected:
            counts["preselected_audit_states"] += 1
            archive = record["audit_inputs"]
            if archive is None:
                if record["geometry"]["available"]:
                    raise ValueError("Available preselected state missing audit archive")
                counts["preselected_geometry_unavailable"] += 1
            else:
                path = (replay/archive["path"]).resolve()
                if path.parent != replay/"audit_inputs":
                    raise ValueError("Unexpected audit archive location")
                value = digest(path)
                if value != archive["sha256"] or receipt["files_sha256"].get(str(path)) != value:
                    raise ValueError("Changed or unbound numerical input archive")
                checked[str(path)] = value
                with np.load(path, allow_pickle=False) as inputs:
                    issues = archive_geometry_audit(inputs, record["geometry"]["geometry"])
                    centers = [None if not np.isfinite(point).all() else point.tolist()
                               for point in inputs["prior_centers_xy"]]
                    independently = independent_localized(inputs["current129"], inputs["history129"],
                                                          centers, inputs["predicted_offset_xy"],
                                                          record["track_id"].split(":")[0])
                compare(independently, record["localized"], "localized", issues)
                detail["component_and_score"] = dict(passed=not issues, issues=issues,
                                                      localized_available=independently["available"])
                counts["component_assembly_checked"] += 1
                counts["heldout_fit_available_checked"] += int(independently["available"])
                all_issues.extend(dict(key=list(key), **issue) for issue in issues)
                print("Audited "+str(key)+"; issues "+str(len(issues)), flush=True)
        if detail["forecast"] is not None or key in selected:
            details.append(detail)
    if counts["preselected_audit_states"] != len(selected):
        raise ValueError("Missing preselected audit states")
    for path, value in checked.items():
        if digest(path) != value:
            raise ValueError("Input changed during audit: "+path)
    result = dict(schema="seaqr.accuracy-v42-independent-numerical-audit.v1",
                  completed=True, passed=not all_issues, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  checked_files_sha256=checked, counts=dict(counts), issues=all_issues, details=details,
                  comparison_tolerances=dict(absolute=ATOL, relative=RTOL),
                  independent_method="Scalar medians/interpolation; direct box means; QR least squares; pinv only rank-deficient designs",
                  scope_limitations=["No raw-media sampling verification", "No global-motion accuracy verification",
                                     "No ground-truth labels or classifier acceptance", "Component/heldout checks only frozen preselected NPZs"],
                  production_changed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], counts=result["counts"], issues=len(all_issues))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outcome = run(args.replay, args.output)
    raise SystemExit(0 if outcome["passed"] else 1)
