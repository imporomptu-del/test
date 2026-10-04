"""Pure, continuous template-comparison diagnostic; not a detection gate.

All inputs are 25 x 25 native-DN patches already placed on the same current
coordinate grid by the caller. Geometry, registration and track correspondence
are outside this helper. A stationary and a transported prior each get exactly
four fit parameters: a nonnegative texture amplitude and a planar background.
The plane-only baseline has three parameters. Parameters are fitted on one
checkerboard parity and evaluated on the other, then the roles are swapped.
Every hypothesis uses the intersection of finite, nonsaturated (0 < DN < 255)
pixels in all three inputs, including the plane-only baseline.

MIN_RESIDUAL_TEXTURE_VARIANCE = 1/12 DN squared is the variance of a uniform
one-DN quantization interval. It is a fixed numerical availability condition,
not a learned decision threshold. A missing/flat prior or deficient design is
unavailable, never evidence for rejecting an object. At least 64 common pixels
and 32 per checkerboard fold are required. NaN means missing; infinity, complex
values, Boolean values and wrong shapes are malformed and raise ValueError.

The support hash is SHA-256 over 625 row-major uint8 mask bytes (0 or 1).
Returned MSEs are in DN squared, weighted across all held-out common pixels.
A positive stationary-minus-transported MSE means only that the supplied
transported template predicted these pixels better. Camera-registration error,
evolving broad texture, or alternating identical fixed lights can produce the
same sign as a moving feature. These are not class or confidence estimates.
Spatial pixels are correlated; the geometry hypotheses may come from a tracker
that has already seen the current image. Checkerboard prediction is not an
independent statistical validation or a probability of physical motion.
"""

import hashlib

import numpy as np


PATCH_SHAPE = (25, 25)
MIN_COMMON_SUPPORT = 64
MIN_FOLD_SUPPORT = 32
MIN_RESIDUAL_TEXTURE_VARIANCE = 1.0 / 12.0


def _patch(value, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric 25 x 25 array") from exc
    if array.shape != PATCH_SHAPE or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric 25 x 25 array")
    array = np.asarray(array, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinity")
    return array


def compare_templates(current, stationary_prior, transported_prior):
    """Return JSON-safe diagnostic scores or explicit unavailability reasons.

    Inputs are never modified. This function performs no source access, temporal
    interpolation, model-selection sweep, thresholding or object classification.
    Fold labels denote training parity; the opposite parity is held out.
    """
    patches = {
        "current": _patch(current, "current"),
        "stationary": _patch(stationary_prior, "stationary_prior"),
        "transported": _patch(transported_prior, "transported_prior"),
    }
    mask = np.ones(PATCH_SHAPE, dtype=bool)
    for patch in patches.values():
        mask &= np.isfinite(patch) & (patch > 0.0) & (patch < 255.0)
    iy, ix = np.indices(PATCH_SHAPE)
    parity = (iy + ix) % 2
    # Unit-scaled coordinates improve conditioning without changing the plane.
    plane = np.stack((np.ones(PATCH_SHAPE), (ix - 12) / 12.0,
                      (iy - 12) / 12.0), axis=-1)
    counts = [int(np.count_nonzero(mask & (parity == fold))) for fold in (0, 1)]
    result = {
        "available": False,
        "reasons": [],
        "common_support_count": int(np.count_nonzero(mask)),
        "common_support_sha256": hashlib.sha256(
            mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
        "fold_support_counts": counts,
        "folds": [],
        "mse_stationary": None,
        "mse_transported": None,
        "mse_plane": None,
        "advantage_stationary_minus_transported": None,
    }
    if result["common_support_count"] < MIN_COMMON_SUPPORT:
        result["reasons"].append("insufficient_common_support")
    for fold, count in enumerate(counts):
        if count < MIN_FOLD_SUPPORT:
            result["reasons"].append(f"insufficient_fold_support:{fold}")
    if result["reasons"]:
        return result

    squared_errors = {name: 0.0 for name in ("stationary", "transported", "plane")}
    for fold in (0, 1):
        train = mask & (parity == fold)
        heldout = mask & (parity != fold)
        a_train, a_test = plane[train], plane[heldout]
        y_train, y_test = patches["current"][train], patches["current"][heldout]
        beta_plane, _, plane_rank, _ = np.linalg.lstsq(a_train, y_train, rcond=None)
        item = {
            "training_parity": fold,
            "training_count": counts[fold],
            "heldout_count": counts[1 - fold],
            "plane_rank": int(plane_rank),
            "models": {},
            "mse_plane": None,
        }
        result["folds"].append(item)
        if plane_rank != 3:
            result["reasons"].append(f"plane_rank_deficient:{fold}")
            continue
        errors_plane = y_test - a_test @ beta_plane
        item["mse_plane"] = float(np.mean(errors_plane ** 2))
        squared_errors["plane"] += float(errors_plane @ errors_plane)

        for name in ("stationary", "transported"):
            texture = patches[name][train]
            texture_plane, _, _, _ = np.linalg.lstsq(a_train, texture, rcond=None)
            residual_texture = texture - a_train @ texture_plane
            variance = float(np.mean(residual_texture ** 2))
            design_train = np.column_stack((texture, a_train))
            beta, _, rank, _ = np.linalg.lstsq(design_train, y_train, rcond=None)
            model = {
                "design_rank": int(rank),
                "residual_texture_variance": variance,
                "amplitude": None,
                "nonnegative_constraint_active": None,
                "mse": None,
            }
            item["models"][name] = model
            if rank != 4:
                result["reasons"].append(f"template_rank_deficient:{name}:{fold}")
            if variance <= MIN_RESIDUAL_TEXTURE_VARIANCE:
                result["reasons"].append(f"insufficient_prior_texture:{name}:{fold}")
            if rank != 4 or variance <= MIN_RESIDUAL_TEXTURE_VARIANCE:
                continue
            # There is only one constrained coefficient. If its unconstrained
            # optimum is negative, the exact constrained optimum is amplitude
            # zero with the remaining plane coefficients refitted freely.
            constrained = bool(beta[0] < 0.0)
            if constrained:
                amplitude = 0.0
                predicted = a_test @ beta_plane
            else:
                amplitude = float(beta[0])
                predicted = amplitude * patches[name][heldout] + a_test @ beta[1:]
            errors = y_test - predicted
            model.update(amplitude=amplitude, nonnegative_constraint_active=constrained,
                         mse=float(np.mean(errors ** 2)))
            squared_errors[name] += float(errors @ errors)

    if result["reasons"]:
        return result
    count = result["common_support_count"]
    for name, sse in squared_errors.items():
        result[f"mse_{name}"] = float(sse / count)
    result["advantage_stationary_minus_transported"] = (
        result["mse_stationary"] - result["mse_transported"])
    result["available"] = True
    return result
