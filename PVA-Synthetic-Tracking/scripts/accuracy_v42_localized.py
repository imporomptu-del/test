"""Causal localized-template evidence, NOT a detection or rejection policy.

Only the eight already globally aligned prior images, prior actual same-ID
centres and a caller-supplied causal predicted subpixel offset learn templates.
The current image fits a shared gain/plane on an outer annulus, then two models
fit checkerboard halves of the central 25 square and score the other halves.
The augmented model has one EXTRA parameter, so an improvement is not a motion
probability. Broad deformation and lookalike fixed lights remain confounders.

All arrays use native DN; NaN denotes unavailable/saturated support. The caller
must mask every saturation-contaminated interpolation footprint. No missing
pixels are filled. Missing, flat, slowly moving or hovering histories can be
unknown; that is never evidence for rejecting a feature. A redundant moving
template, a truncated fixed-light dictionary, or a persistent fixed-light anchor
within the two-pixel repeat radius of the causal prediction is ambiguous. The
last condition is conservative physical ambiguity: empirical template-shape
differences must not make a co-located fixed light look identifiable as motion.
"""

import hashlib
import warnings

import numpy as np


PATCH_SHAPE = (129, 129)
HISTORY_SHAPE = (8, 129, 129)
MIN_BACKGROUND_OBSERVATIONS = 3
EXCLUSION_RADIUS = 8
STAMP_RADIUS = 8
STAMP_SHAPE = (17, 17)
MIN_STAMP_SUPPORT = 64
MIN_USABLE_STAMPS = 3
MIN_COMMON_SUPPORT = 64
MIN_FOLD_SUPPORT = 32
MAX_FIXED_ANCHORS = 4
ANCHOR_REPEAT_RADIUS = 2
MIN_ANCHOR_FRAMES = 3
MIN_ANCHOR_CONTRAST_DN = 1.0
COLLINEAR_RELATIVE_ENERGY = 1.0e-8


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    array = np.asarray(array, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinity")
    return array


def _coordinates(centres, offset):
    if len(centres) != 8:
        raise ValueError("prior_centers_xy must contain exactly eight entries")
    parsed = []
    for centre in centres:
        if centre is None:
            parsed.append(None)
            continue
        point = _array(centre, (2,), "prior centre")
        if not np.isfinite(point).all():
            raise ValueError("a missing prior centre must be None, not NaN")
        parsed.append(point)
    offset = _array(offset, (2,), "predicted_offset_xy")
    if not np.isfinite(offset).all() or (np.abs(offset) > 0.5).any():
        raise ValueError("predicted_offset_xy must be finite and within [-0.5, 0.5]")
    return parsed, offset


def _median(values, minimum):
    count = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(values, axis=0)
    result[count < minimum] = np.nan
    return result, count


def _sample(image, xx, yy):
    """Bilinear sampling; only positive-weight finite inputs are required."""
    height, width = image.shape
    x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
    dx, dy = xx - x0, yy - y0
    output = np.zeros(xx.shape, dtype=float)
    valid = np.ones(xx.shape, dtype=bool)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        x, y = x0 + ox, y0 + oy
        needed = weight > 0.0
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        samples = image[np.clip(y, 0, height-1), np.clip(x, 0, width-1)]
        valid &= ~needed | (inside & np.isfinite(samples))
        output += np.where(needed & inside & np.isfinite(samples), samples, 0) * weight
    output[~valid] = np.nan
    return output


def _stamp(image, centre):
    y, x = np.indices(STAMP_SHAPE)
    return _sample(image, x + centre[0] - STAMP_RADIUS,
                   y + centre[1] - STAMP_RADIUS)


def _positive_unit_stamp(stamp):
    aperture = np.outer(np.hanning(17), np.hanning(17))
    finite = np.isfinite(stamp) & (aperture > 0)
    if np.count_nonzero(finite) < MIN_STAMP_SUPPORT:
        return None
    weighted = np.maximum(stamp, 0.0) * aperture
    weighted[aperture == 0] = 0.0
    energy = float(np.nansum(weighted ** 2))
    if not np.isfinite(energy) or energy <= 1.0e-12:
        return None
    return weighted / np.sqrt(energy)


def _combine_stamps(stamps):
    if len(stamps) < MIN_USABLE_STAMPS:
        return None
    combined, _ = _median(np.stack(stamps), MIN_USABLE_STAMPS)
    aperture = np.outer(np.hanning(17), np.hanning(17))
    combined[aperture == 0] = 0.0
    if np.count_nonzero(np.isfinite(combined) & (aperture > 0)) < MIN_STAMP_SUPPORT:
        return None
    energy = float(np.nansum(combined ** 2))
    if energy <= 1.0e-12 or not np.isfinite(energy):
        return None
    return combined / np.sqrt(energy)


def _place(stamp, centre):
    y, x = np.indices(PATCH_SHAPE)
    sx, sy = x - centre[0] + STAMP_RADIUS, y - centre[1] + STAMP_RADIUS
    inside = (sx >= 0) & (sx <= 16) & (sy >= 0) & (sy <= 16)
    result = np.zeros(PATCH_SHAPE, dtype=float)
    result[inside] = _sample(stamp, sx[inside], sy[inside])
    return result


def _box_highpass(image):
    finite = np.isfinite(image)
    values = np.pad(np.where(finite, image, 0.0), 4)
    counts = np.pad(finite.astype(np.int64), 4)

    def sums(array):
        integral = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return integral[9:, 9:] - integral[:-9, 9:] - integral[9:, :-9] + integral[:-9, :-9]

    count = sums(counts)
    result = image - sums(values) / 81.0
    result[count != 81] = np.nan
    return result


def _anchors(highpasses):
    """Deterministic repeat-supported seeds; no current image is consulted."""
    observations = []
    for frame, hp in enumerate(highpasses):
        safe = np.where(np.isfinite(hp), hp, -np.inf)
        neighbours = np.lib.stride_tricks.sliding_window_view(np.pad(safe, 1, constant_values=-np.inf), (3, 3))
        maxima = neighbours.max(axis=(-1, -2))
        yy, xx = np.indices(PATCH_SHAPE)
        selected = ((safe == maxima) & (safe >= MIN_ANCHOR_CONTRAST_DN)
                    & (np.maximum(np.abs(xx-64), np.abs(yy-64)) <= 20))
        for y, x in np.argwhere(selected):
            observations.append((int(frame), int(x), int(y), float(hp[y, x])))
    if not observations:
        return []
    candidates = []
    observed = np.asarray(observations)
    # Repeated observations of an identical pixel define one seed, not several
    # objects. Each seed is stationary; it never walks while points are added.
    for x, y in sorted({(entry[1], entry[2]) for entry in observations}, key=lambda v: (v[1], v[0])):
        near = np.maximum(np.abs(observed[:, 1]-x), np.abs(observed[:, 2]-y)) <= ANCHOR_REPEAT_RADIUS
        close = observed[near]
        frames = sorted(int(frame) for frame in np.unique(close[:, 0]))
        if len(frames) < MIN_ANCHOR_FRAMES:
            continue
        strength = float(np.median([float(close[close[:, 0] == frame, 3].max()) for frame in frames]))
        candidates.append({"x": x, "y": y, "prior_frame_indices": frames,
                           "repeat_frame_count": len(frames), "median_contrast_dn": strength})
    candidates.sort(key=lambda item: (-item["repeat_frame_count"], -item["median_contrast_dn"], item["y"], item["x"]))
    clustered = []
    for item in candidates:
        if all(max(abs(item["x"]-old["x"]), abs(item["y"]-old["y"])) > ANCHOR_REPEAT_RADIUS for old in clustered):
            clustered.append(item)
    return clustered


def prepare_components(history129, prior_centers_xy, predicted_offset_xy, polarity):
    """Learn causal components only; returned arrays are for numerical audit.

    This function intentionally has no current-image argument. ``metadata`` is
    JSON-safe. Arrays preserve NaNs and are new values, never views to mutate.
    """
    history = _array(history129, HISTORY_SHAPE, "history129")
    centres, offset = _coordinates(prior_centers_xy, predicted_offset_xy)
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity must be bright or dark")
    sign = 1.0 if polarity == "bright" else -1.0
    y, x = np.indices(PATCH_SHAPE)
    masked = history.copy()
    for frame, centre in enumerate(centres):
        if centre is not None:
            mask = np.maximum(np.abs(x-centre[0]), np.abs(y-centre[1])) <= EXCLUSION_RADIUS
            masked[frame, mask] = np.nan
    background, counts = _median(masked, MIN_BACKGROUND_OBSERVATIONS)
    stamps = []
    usable_indices = []
    for frame, centre in enumerate(centres):
        if centre is None:
            continue
        stamp = _positive_unit_stamp(_stamp(sign * (history[frame] - background), centre))
        if stamp is not None:
            stamps.append(stamp)
            usable_indices.append(frame)
    moving_stamp = _combine_stamps(stamps)
    moving = None if moving_stamp is None else _place(moving_stamp, 64.0 + offset)
    highpasses = np.stack([sign * _box_highpass(frame) for frame in history])
    discovered = _anchors(highpasses)
    predicted_centre = 64.0 + offset
    overlapping = [[item["x"], item["y"]] for item in discovered
                   if max(abs(item["x"]-predicted_centre[0]),
                          abs(item["y"]-predicted_centre[1])) <= ANCHOR_REPEAT_RADIUS]
    fixed = []
    anchors = []
    for anchor in discovered[:MAX_FIXED_ANCHORS]:
        anchor_stamps = []
        for frame in highpasses:
            stamp = _positive_unit_stamp(_stamp(frame, (anchor["x"], anchor["y"])))
            if stamp is not None:
                anchor_stamps.append(stamp)
        combined = _combine_stamps(anchor_stamps)
        record = dict(anchor, usable_stamp_count=len(anchor_stamps), template_available=combined is not None)
        anchors.append(record)
        if combined is not None:
            fixed.append(_place(combined, (anchor["x"], anchor["y"])))
    reasons = []
    if moving is None:
        reasons.append("insufficient_causal_foreground_template")
    if any(not item["template_available"] for item in anchors):
        reasons.append("incomplete_persistent_anchor_template")
    metadata = {
        "polarity": polarity, "prior_count": 8,
        "usable_moving_stamp_indices": usable_indices,
        "usable_moving_stamp_count": len(stamps),
        "background_available_pixel_count": int(np.isfinite(background).sum()),
        "background_observation_count_min": int(counts.min()),
        "background_observation_count_max": int(counts.max()),
        "persistent_anchor_count_before_cap": len(discovered),
        "causal_predicted_center_xy": predicted_centre.tolist(),
        "persistent_anchor_centres_near_prediction": overlapping,
        "persistent_anchor_overlaps_prediction": bool(overlapping),
        "fixed_anchor_count": len(fixed), "anchors": anchors,
        "anchor_dictionary_truncated": len(discovered) > MAX_FIXED_ANCHORS,
        "causal_component_reasons": reasons,
    }
    return {"metadata": metadata, "background": background,
            "background_observation_counts": counts, "moving_template": moving,
            "fixed_templates": np.stack(fixed) if fixed else np.empty((0, 129, 129)),
            "moving_stamp": moving_stamp}


def _residual_energy(column, design):
    energy = float(column @ column)
    if not energy > 0:
        return 0.0
    residual = column if design.shape[1] == 0 else column - design @ np.linalg.lstsq(design, column, rcond=None)[0]
    return float((residual @ residual) / energy)


def evaluate_localized(current129, history129, prior_centers_xy,
                       predicted_offset_xy, polarity):
    """Return finite JSON-safe scores, explicit unknowns and ambiguity flags.

    ``available`` means scores were calculable, never that a detection passed.
    ``ambiguous`` can coexist with availability. There is no class, confidence,
    accept/reject decision, current detection coordinate or reference label.
    """
    current = _array(current129, PATCH_SHAPE, "current129")
    components = prepare_components(history129, prior_centers_xy, predicted_offset_xy, polarity)
    metadata = components["metadata"]
    result = {"available": False, "ambiguous": False, "reasons": [],
              "ambiguity_reasons": [], "components": metadata,
              "background_fit": None, "common_support_count": 0,
              "common_support_sha256": None, "fold_support_counts": [0, 0],
              "moving_relative_energy_outside_fixed_span": None,
              "model_complexity": {"shared_annulus_parameters": 4,
                                   "stationary_core_parameters": metadata["fixed_anchor_count"],
                                   "augmented_core_parameters": metadata["fixed_anchor_count"] + 1,
                                   "extra_augmented_parameters": 1},
              "folds": [], "mse_stationary": None, "mse_augmented": None,
              "advantage_stationary_minus_augmented": None}
    if metadata["anchor_dictionary_truncated"]:
        result["ambiguity_reasons"].append("persistent_anchor_dictionary_truncated")
        result["ambiguous"] = True
    if metadata["persistent_anchor_overlaps_prediction"]:
        result["ambiguity_reasons"].append("causal_prediction_overlaps_persistent_fixed_anchor")
        result["ambiguous"] = True
    if metadata["causal_component_reasons"]:
        result["reasons"].extend(metadata["causal_component_reasons"])
        return result
    y, x = np.indices(PATCH_SHAPE)
    plane = np.stack((np.ones(PATCH_SHAPE), (x-64)/32.0, (y-64)/32.0), axis=-1)
    distance = np.maximum(np.abs(x-64), np.abs(y-64))
    background = components["background"]
    annulus = ((distance >= 16) & (distance <= 30)
               & np.isfinite(background) & np.isfinite(current))
    if int(annulus.sum()) < MIN_COMMON_SUPPORT:
        result["reasons"].append("insufficient_annulus_support")
        return result
    design = np.column_stack((background[annulus], plane[annulus]))
    beta, _, rank, _ = np.linalg.lstsq(design, current[annulus], rcond=None)
    result["background_fit"] = {"support_count": int(annulus.sum()), "design_rank": int(rank),
                                 "gain": None, "plane_coefficients": None,
                                 "nonnegative_gain_constraint_active": None}
    if rank != 4:
        result["reasons"].append("flat_or_rank_deficient_annulus_background")
        return result
    constrained = bool(beta[0] < 0)
    if constrained:
        beta = np.r_[0.0, np.linalg.lstsq(plane[annulus], current[annulus], rcond=None)[0]]
    result["background_fit"].update(gain=float(beta[0]), plane_coefficients=beta[1:].tolist(),
                                    nonnegative_gain_constraint_active=constrained)
    fixed_background = beta[0] * background + plane @ beta[1:]
    residual = (current - fixed_background) * (1.0 if polarity == "bright" else -1.0)
    core = (slice(52, 77), slice(52, 77))
    response = residual[core]
    moving = components["moving_template"][core]
    fixed = components["fixed_templates"][:, 52:77, 52:77]
    common = np.isfinite(response) & np.isfinite(moving)
    if len(fixed):
        common &= np.isfinite(fixed).all(axis=0)
    iy, ix = np.indices((25, 25))
    parity = (iy + ix) % 2
    counts = [int((common & (parity == fold)).sum()) for fold in (0, 1)]
    result.update(common_support_count=int(common.sum()), fold_support_counts=counts,
                  common_support_sha256=hashlib.sha256(common.astype(np.uint8).tobytes()).hexdigest())
    if int(common.sum()) < MIN_COMMON_SUPPORT or min(counts) < MIN_FOLD_SUPPORT:
        result["reasons"].append("insufficient_common_core_or_checkerboard_support")
        return result
    fixed_all = fixed[:, common].T
    energy = _residual_energy(moving[common], fixed_all)
    result["moving_relative_energy_outside_fixed_span"] = energy
    if energy <= COLLINEAR_RELATIVE_ENERGY:
        result["ambiguity_reasons"].append("moving_template_redundant_with_fixed_dictionary")
    squared_errors = np.zeros(2)
    for fold in (0, 1):
        train = common & (parity == fold)
        test = common & (parity != fold)
        a, b = fixed[:, train].T, fixed[:, test].T
        yy, yt = response[train], response[test]
        m, mt = moving[train], moving[test]
        stationary_beta, _, stationary_rank, _ = np.linalg.lstsq(a, yy, rcond=None)
        augmented_design = np.column_stack((a, m))
        augmented_beta, _, augmented_rank, _ = np.linalg.lstsq(augmented_design, yy, rcond=None)
        fold_energy = _residual_energy(m, a)
        redundant = fold_energy <= COLLINEAR_RELATIVE_ENERGY
        constrained = bool(augmented_beta[-1] < 0)
        if redundant or constrained:
            augmented_beta = np.r_[stationary_beta, 0.0]
        predicted_stationary = b @ stationary_beta
        predicted_augmented = b @ augmented_beta[:-1] + mt * augmented_beta[-1]
        errors = [yt-predicted_stationary, yt-predicted_augmented]
        for index, error in enumerate(errors):
            squared_errors[index] += float(error @ error)
        if stationary_rank != len(fixed):
            result["ambiguity_reasons"].append(f"fixed_dictionary_rank_deficient:{fold}")
        if redundant or augmented_rank != len(fixed)+1:
            result["ambiguity_reasons"].append(f"moving_template_unidentifiable_on_training_fold:{fold}")
        result["folds"].append({
            "training_parity": fold, "training_count": counts[fold], "heldout_count": counts[1-fold],
            "stationary_rank": int(stationary_rank), "augmented_rank": int(augmented_rank),
            "moving_relative_energy_outside_fixed_span": fold_energy,
            "stationary_signed_amplitudes": stationary_beta.tolist(),
            "augmented_signed_fixed_amplitudes": augmented_beta[:-1].tolist(),
            "moving_nonnegative_amplitude": float(augmented_beta[-1]),
            "moving_amplitude_identifiable": not redundant,
            "moving_nonnegative_constraint_active": constrained and not redundant,
            "redundant_moving_column_zeroed_by_convention": redundant,
            "mse_stationary": float(np.mean(errors[0]**2)),
            "mse_augmented": float(np.mean(errors[1]**2)),
        })
    result.update(available=True, ambiguous=bool(result["ambiguity_reasons"]),
                  mse_stationary=float(squared_errors[0]/common.sum()),
                  mse_augmented=float(squared_errors[1]/common.sum()),
                  advantage_stationary_minus_augmented=float((squared_errors[0]-squared_errors[1])/common.sum()))
    return result
