#!/usr/bin/env python3
"""Isolated saved-field cubic compensation; no fitting, I/O, or detector calls."""
from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

WIDTH, HEIGHT = 2448, 2048
ARMS = ("global_translation", "local_translation", "local_affine")
EPSILON = 1e-12
PSF_TRUNCATION = 6.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _shape(value):
    require(len(value) == 2 and all(type(x) is int and 0 < x < 32767 for x in value), "invalid image shape")
    return tuple(value)


def _origin(value):
    require(len(value) == 2 and all(type(x) is int for x in value), "integer native origin required")
    return tuple(value)


def _hull_mask(hull, x, y):
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    require(np.isfinite(hull).all(), "nonfinite saved hull")
    if len(hull) < 3:
        return np.zeros(x.shape, bool)
    result = np.ones(x.shape, bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        result &= ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) >= -1e-8
    return result


def field_from_maps(qx, qy, *, model_support=None, numerical=None,
                    source_shape=None, erosion_px=2, origin_xy=(0, 0)):
    """Freeze one float32 map, its quantized cubic footprint, and output support.

    Maps are global source coordinates. Erosion is performed here, once, on the
    supplied output domain. Real-image callers build the FULL field first and
    obtain smaller output domains with field_roi; never erode each small ROI.
    """
    qx, qy = np.asarray(qx, dtype=np.float32), np.asarray(qy, dtype=np.float32)
    require(qx.ndim == 2 and qx.shape == qy.shape and all(qx.shape), "matching nonempty 2D maps required")
    source_shape = _shape(tuple(source_shape or qx.shape))
    origin_xy = _origin(origin_xy)
    require(type(erosion_px) is int and erosion_px >= 0, "invalid erosion")
    finite = np.isfinite(qx) & np.isfinite(qy)
    numerical = finite if numerical is None else np.asarray(numerical, dtype=bool) & finite
    require(numerical.shape == qx.shape, "numerical mask shape differs")
    model = numerical.copy() if model_support is None else np.asarray(model_support, dtype=bool) & numerical
    require(model.shape == qx.shape, "model support shape differs")
    # Undefined fits have safe map storage, but no numerical oracle or support.
    qx = np.ascontiguousarray(np.where(numerical, qx, 0), dtype=np.float32)
    qy = np.ascontiguousarray(np.where(numerical, qy, 0), dtype=np.float32)
    fixed, coefficients = cv2.convertMaps(qx, qy, cv2.CV_16SC2)
    height, width = source_shape
    bx, by = fixed[..., 0].astype(np.int32), fixed[..., 1].astype(np.int32)
    kernel = numerical & (bx >= 1) & (by >= 1) & (bx + 2 < width) & (by + 2 < height)
    pre = model & kernel
    valid = (cv2.erode(pre.astype(np.uint8), np.ones((2 * erosion_px + 1,) * 2, np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
             if erosion_px else pre.copy())
    x0, y0 = origin_xy
    result = dict(qx=qx, qy=qy, fixed_xy=fixed, coefficients=coefficients,
        numerical=numerical.copy(), model_support=model.copy(), kernel_valid=kernel,
        valid_pre=pre, valid=valid, shape=qx.shape, source_shape=source_shape,
        roi_xyxy=(x0, y0, x0 + qx.shape[1], y0 + qx.shape[0]), erosion_px=erosion_px)
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def build_field(row, arm, *, shape=(HEIGHT, WIDTH), stripe_rows=128):
    """Apply all 48 existing cell transforms without fitting or blending."""
    height, width = _shape(tuple(shape))
    require(arm in ARMS and type(stripe_rows) is int and stripe_rows > 0, "invalid arm/stripe bound")
    cells = row["cells"]
    require(len(cells) == 48 and [c["cell_id"] for c in cells] == list(range(48)), "saved 48-cell inventory differs")
    qx, qy = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    numerical, support = np.zeros(shape, bool), np.zeros(shape, bool)
    matrices = [None] * 48
    for cell in cells:
        cell_id = cell["cell_id"]
        r, c = divmod(cell_id, 8)
        # Integer pixel centers obey the same half-open native cell rectangles.
        x0, x1 = math.ceil(c * width / 8), math.ceil((c + 1) * width / 8)
        y0, y1 = math.ceil(r * height / 6), math.ceil((r + 1) * height / 6)
        fitted = cell["arms"][arm]
        fit = fitted.get("fit")
        if not fit or not fit.get("valid", False):
            continue
        matrix = np.asarray(fit["native_matrix"], dtype=np.float64)
        require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
                and np.array_equal(matrix[2], [0, 0, 1]), "finite saved affine matrix required")
        matrices[cell_id] = matrix.tolist()
        for sy in range(y0, y1, stripe_rows):
            ey = min(y1, sy + stripe_rows)
            xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(sy, ey, dtype=np.float64))
            tx = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
            ty = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
            require(np.isfinite(tx).all() and np.isfinite(ty).all()
                    and np.max(np.abs(tx), initial=0) <= np.finfo(np.float32).max
                    and np.max(np.abs(ty), initial=0) <= np.finfo(np.float32).max, "map overflow")
            qx[sy:ey, x0:x1], qy[sy:ey, x0:x1] = tx, ty
            numerical[sy:ey, x0:x1] = True
            if fitted["training_eligible"]:
                support[sy:ey, x0:x1] = (True if arm == "global_translation"
                    else _hull_mask(fitted["hull"], xx, yy))
    result = field_from_maps(qx, qy, model_support=support, numerical=numerical, source_shape=shape)
    result.update(arm=arm, case_id=row.get("case_id"), cell_matrices=matrices)
    return result


def field_roi(field, roi_xyxy):
    """Extract native output ROI with false padding; never re-erode support."""
    require(len(roi_xyxy) == 4 and all(type(x) is int for x in roi_xyxy), "integer ROI required")
    x0, y0, x1, y1 = roi_xyxy
    require(x1 > x0 and y1 > y0, "empty output ROI")
    fx0, fy0, fx1, fy1 = field["roi_xyxy"]
    ax0, ay0, ax1, ay1 = max(x0, fx0), max(y0, fy0), min(x1, fx1), min(y1, fy1)
    result = {key: value for key, value in field.items() if not isinstance(value, np.ndarray)}
    for key, value in field.items():
        if not isinstance(value, np.ndarray):
            continue
        out = np.zeros((y1 - y0, x1 - x0) + value.shape[2:], dtype=value.dtype)
        if ax0 < ax1 and ay0 < ay1:
            out[ay0-y0:ay1-y0, ax0-x0:ax1-x0] = value[ay0-fy0:ay1-fy0, ax0-fx0:ax1-fx0]
        out.setflags(write=False)
        result[key] = out
    result.update(shape=(y1-y0, x1-x0), roi_xyxy=tuple(roi_xyxy))
    return result


def pull(image, field, *, origin_xy=(0, 0), source_valid=None):
    """Sample exactly once using the frozen quantized cubic map, no fallback.

    An integer-aligned source context is permitted if all valid 4x4 footprints
    are present. Optional source_valid further invalidates any affected tap.
    """
    image = np.asarray(image)
    require(image.ndim == 2 and image.dtype in (np.uint8, np.float32) and np.isfinite(image).all(), "finite U8/float32 image required")
    ox, oy = _origin(origin_xy)
    valid = np.asarray(field["valid"], dtype=bool).copy()
    output = np.full(valid.shape, np.nan, np.float32)
    if not np.any(valid):
        return output
    require(all(image.shape), "nonempty source context required")
    fixed = field["fixed_xy"].astype(np.int32)
    fixed -= [ox, oy]
    bx, by = fixed[..., 0], fixed[..., 1]
    inside = (bx >= 1) & (by >= 1) & (bx + 2 < image.shape[1]) & (by + 2 < image.shape[0])
    require(np.all(inside[valid]), "source context omits valid cubic footprint")
    if source_valid is not None:
        source_valid = np.asarray(source_valid, dtype=bool)
        require(source_valid.shape == image.shape, "source validity shape differs")
        yy, xx = np.nonzero(valid)
        footprint = np.ones(len(xx), bool)
        for dy in (-1, 0, 1, 2):
            for dx in (-1, 0, 1, 2):
                footprint &= source_valid[by[yy, xx] + dy, bx[yy, xx] + dx]
        valid[yy, xx] &= footprint
    # Invalid-map storage is harmless and cannot wrap/saturate into evidence.
    fixed[~valid] = 0
    require(np.all((fixed[valid] >= -32768) & (fixed[valid] <= 32767)), "context map exceeds OpenCV range")
    sampled = cv2.remap(np.ascontiguousarray(image, dtype=np.float32), fixed.astype(np.int16),
        field["coefficients"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output[valid] = sampled[valid]
    return output


def psf_at(x, y, center, sigma, amplitude):
    require(len(center) == 2 and np.isfinite(center).all() and np.isfinite(sigma) and sigma > 0
            and np.isfinite(amplitude), "invalid analytic PSF")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    radius2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return np.where(radius2 <= (PSF_TRUNCATION * sigma) ** 2,
                    amplitude * np.exp(-radius2 / (2 * sigma * sigma)), 0.0)


def psf(shape, center, sigma, amplitude, *, origin_xy=(0, 0)):
    height, width = _shape(tuple(shape))
    ox, oy = _origin(origin_xy)
    return psf_at(np.arange(width)[None, :] + ox, np.arange(height)[:, None] + oy,
                  center, sigma, amplitude)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if math.isfinite(denominator) and abs(denominator) > EPSILON else None


def oracle_coordinates(field):
    """Unquantized float64 F(p) for saved matrices, separate from remap tables."""
    if "cell_matrices" not in field:
        # A maps-only synthetic field defines its supplied float32 coordinates.
        return field["qx"].astype(np.float64), field["qy"].astype(np.float64)
    x0, y0, x1, y1 = field["roi_xyxy"]
    xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    height, width = field["source_shape"]
    ids = np.floor(yy * 6 / height).astype(int) * 8 + np.floor(xx * 8 / width).astype(int)
    qx, qy = np.zeros(xx.shape, dtype=np.float64), np.zeros(xx.shape, dtype=np.float64)
    for cell_id in np.unique(ids[field["numerical"]]):
        matrix = np.asarray(field["cell_matrices"][cell_id], dtype=np.float64)
        mask = field["numerical"] & (ids == cell_id)
        qx[mask] = matrix[0, 0] * xx[mask] + matrix[0, 1] * yy[mask] + matrix[0, 2]
        qy[mask] = matrix[1, 0] * xx[mask] + matrix[1, 1] * yy[mask] + matrix[1, 2]
    return qx, qy


def array_metrics(array, valid=None, *, origin_xy=(0, 0), polarity=1, template=None):
    """Signed DN/lobe statistics; positive centroid is polarity-normalized."""
    array = np.asarray(array, dtype=np.float64)
    require(array.ndim == 2 and polarity in (-1, 1), "invalid metric array/polarity")
    valid = np.isfinite(array) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(array)
    require(valid.shape == array.shape, "metric support differs")
    values = array[valid]
    result = dict(count=int(valid.sum()), minimum_dn=None, maximum_dn=None, peak_abs_dn=None,
        polarity_peak_dn=None, positive_mass_dn=None, negative_mass_dn=None, signed_mass_dn=None,
        l1_dn=None, l2_dn=None, rms_dn=None, centroid_xy=None, template_response_dn=None, template_gain=None)
    if not len(values):
        return result
    positive, negative = np.maximum(values, 0), np.maximum(-values, 0)
    l2 = float(np.sqrt(np.sum(values * values)))
    result.update(minimum_dn=float(values.min()), maximum_dn=float(values.max()), peak_abs_dn=float(np.max(np.abs(values))),
        polarity_peak_dn=float(np.max(polarity * values)), positive_mass_dn=float(positive.sum()),
        negative_mass_dn=float(negative.sum()), signed_mass_dn=float(values.sum()),
        l1_dn=float(np.abs(values).sum()), l2_dn=l2, rms_dn=float(l2 / np.sqrt(len(values))))
    weight = np.where(valid, np.maximum(polarity * array, 0), 0)
    mass = float(weight.sum())
    if mass > EPSILON:
        ox, oy = origin_xy
        result["centroid_xy"] = [float(np.sum(weight * (np.arange(array.shape[1])[None, :] + ox)) / mass),
                                 float(np.sum(weight * (np.arange(array.shape[0])[:, None] + oy)) / mass)]
    if template is not None:
        template = np.asarray(template, dtype=np.float64)
        require(template.shape == array.shape and np.isfinite(template[valid]).all(), "template undefined on metric support")
        energy = float(np.sum(template[valid] ** 2))
        dot = float(np.dot(values, template[valid]))
        result.update(template_response_dn=_ratio(dot, math.sqrt(energy)), template_gain=_ratio(dot, energy))
    return result


def inject_patch(image, origin_xy, center, sigma, amp, *, source_valid=None):
    image = np.asarray(image)
    require(image.dtype == np.uint8 and image.ndim == 2, "U8 injection context required")
    valid = np.ones(image.shape, bool) if source_valid is None else np.asarray(source_valid, dtype=bool)
    require(valid.shape == image.shape, "injection source support differs")
    intended = (psf(image.shape, center, sigma, amp, origin_xy=origin_xy) if image.size
                else np.zeros(image.shape, dtype=np.float64))
    unrounded = image.astype(np.float64) + intended
    rounded = np.rint(unrounded)
    injected = np.clip(rounded, 0, 255).astype(np.uint8)
    injected[~valid] = image[~valid]
    effective = injected.astype(np.float32) - image.astype(np.float32)
    quantization = rounded - unrounded
    polarity = 1 if amp >= 0 else -1
    radius = PSF_TRUNCATION * sigma
    px0, py0 = math.floor(center[0] - radius), math.floor(center[1] - radius)
    px1, py1 = math.ceil(center[0] + radius) + 1, math.ceil(center[1] + radius) + 1
    full_psf = psf((py1-py0, px1-px0), center, sigma, amp, origin_xy=(px0, py0))
    full_mass, native_mass = float(np.abs(full_psf).sum()), float(np.abs(intended[valid]).sum())
    stats = dict(intended=array_metrics(intended, valid, origin_xy=origin_xy, polarity=polarity),
        effective=array_metrics(effective, valid, origin_xy=origin_xy, polarity=polarity),
        clipped_pixel_count=int(np.count_nonzero(valid & ((rounded < 0) | (rounded > 255)))),
        intended_outside_u8_range_pixel_count=int(np.count_nonzero(valid & ((unrounded < 0) | (unrounded > 255)))),
        quantization_l1_dn=float(np.abs(quantization[valid]).sum()),
        quantization_l2_dn=float(np.sqrt(np.sum(quantization[valid] ** 2))),
        realized_minus_intended_l1_dn=float(np.abs(effective[valid] - intended[valid]).sum()),
        full_discrete_intended_l1_dn=full_mass, native_intended_l1_dn=native_mass,
        excluded_intended_l1_dn=max(0.0, full_mass-native_mass),
        source_mass_coverage=_ratio(native_mass, full_mass), source_psf_partial=bool(full_mass-native_mass > EPSILON),
        empty_native_context=not bool(image.size),
        realization="clip(rint(U8 + continuous Gaussian),0,255)", origin_xy=list(origin_xy), shape=list(image.shape))
    return dict(image=injected, intended_delta=intended, effective_delta=effective, stats=stats)


def _padded_roi(image, roi):
    x0, y0, x1, y1 = roi
    out = np.zeros((y1-y0, x1-x0), dtype=image.dtype)
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x1, image.shape[1]), min(y1, image.shape[0])
    if ax0 < ax1 and ay0 < ay1:
        out[ay0-y0:ay1-y0, ax0-x0:ax1-x0] = image[ay0:ay1, ax0:ax1]
    native = np.zeros(out.shape, bool)
    if ax0 < ax1 and ay0 < ay1:
        native[ay0-y0:ay1-y0, ax0-x0:ax1-x0] = True
    return out, native


def source_context(field, center, sigma):
    """Bound current crop by required valid cubic taps plus entire finite PSF."""
    height, width = field["source_shape"]
    radius = PSF_TRUNCATION * sigma
    x0, y0 = math.floor(center[0] - radius), math.floor(center[1] - radius)
    x1, y1 = math.ceil(center[0] + radius) + 1, math.ceil(center[1] + radius) + 1
    mask = field["kernel_valid"]
    if np.any(mask):
        fixed = field["fixed_xy"][mask].astype(np.int32)
        x0, y0 = min(x0, int(fixed[:, 0].min()) - 1), min(y0, int(fixed[:, 1].min()) - 1)
        x1, y1 = max(x1, int(fixed[:, 0].max()) + 3), max(y1, int(fixed[:, 1].max()) + 3)
    # Empty intersections remain empty, not a substituted identity sample.
    return (max(0, min(width, x0)), max(0, min(height, y0)),
            max(0, min(width, x1)), max(0, min(height, y1)))


def probe(previous, current, field, previous_center, current_center, sigma, amp,
          roi_radius=32, return_arrays=False):
    """Paired clean/injected image probe using one unchanged spatial operator."""
    previous, current = np.asarray(previous), np.asarray(current)
    require(previous.dtype == current.dtype == np.uint8 and previous.ndim == current.ndim == 2
            and previous.shape == current.shape == tuple(field["source_shape"]), "native paired U8 inputs required")
    require(type(roi_radius) is int and roi_radius >= 0 and np.isfinite(previous_center).all()
            and np.isfinite(current_center).all() and amp != 0, "invalid probe geometry")
    cx, cy = math.floor(previous_center[0]), math.floor(previous_center[1])
    roi = (cx-roi_radius, cy-roi_radius, cx+roi_radius+1, cy+roi_radius+1)
    local = field_roi(field, roi)
    original_previous, native = _padded_roi(previous, roi)
    previous_injection = inject_patch(original_previous, roi[:2], previous_center, sigma, amp, source_valid=native)
    context = source_context(local, current_center, sigma)
    x0, y0, x1, y1 = context
    clean_context = current[y0:y1, x0:x1]
    current_injection = inject_patch(clean_context, context[:2], current_center, sigma, amp)
    current_injection_stats = current_injection["stats"]
    if clean_context.size:
        clean_warp = pull(clean_context, local, origin_xy=context[:2])
        injected_warp = pull(current_injection["image"], local, origin_xy=context[:2])
    else:
        require(not local["valid"].any(), "empty context for valid source samples")
        clean_warp = np.full(local["shape"], np.nan, np.float32)
        injected_warp = clean_warp.copy()
    valid = local["valid"]
    previous_float = original_previous.astype(np.float32)
    clean_residual = np.where(valid, clean_warp - previous_float, np.nan).astype(np.float32)
    injected_residual = np.where(valid, injected_warp - previous_injection["image"].astype(np.float32), np.nan).astype(np.float32)
    delta = injected_residual - clean_residual
    current_target_delta = injected_warp - clean_warp
    previous_oracle = psf(local["shape"], previous_center, sigma, amp, origin_xy=roi[:2])
    oracle_qx, oracle_qy = oracle_coordinates(local)
    current_oracle = psf_at(oracle_qx, oracle_qy, current_center, sigma, amp)
    numerical = local["numerical"]
    oracle = np.where(numerical, current_oracle - previous_oracle, np.nan)
    current_oracle = np.where(numerical, current_oracle, np.nan)
    polarity = 1 if amp > 0 else -1
    metrics = dict(delta=array_metrics(delta, valid, origin_xy=roi[:2], polarity=polarity, template=oracle),
        oracle=array_metrics(oracle, valid, origin_xy=roi[:2], polarity=polarity, template=oracle),
        current_target_delta=array_metrics(current_target_delta, valid, origin_xy=roi[:2], polarity=polarity, template=current_oracle),
        current_target_oracle=array_metrics(current_oracle, valid, origin_xy=roi[:2], polarity=polarity, template=current_oracle),
        clean_residual=array_metrics(clean_residual, valid, origin_xy=roi[:2], polarity=polarity, template=oracle),
        oracle_error=array_metrics(delta.astype(np.float64) - oracle, valid, origin_xy=roi[:2], polarity=polarity))
    for name, reference in (("delta", "oracle"), ("current_target_delta", "current_target_oracle")):
        observed, expected = metrics[name], metrics[reference]
        ratios = {key: None if observed[key] is None or expected[key] is None else _ratio(observed[key], expected[key])
                  for key in ("peak_abs_dn", "positive_mass_dn", "negative_mass_dn", "l1_dn", "l2_dn")}
        c1, c2 = observed["centroid_xy"], expected["centroid_xy"]
        metrics[name + "_comparison"] = dict(ratios=ratios,
            centroid_error_xy=None if c1 is None or c2 is None else (np.asarray(c1)-c2).tolist(),
            centroid_error_px=None if c1 is None or c2 is None else float(np.linalg.norm(np.asarray(c1)-c2)))
    oracle_mass = float(np.abs(oracle[numerical]).sum())
    oracle_valid_mass = float(np.abs(oracle[valid]).sum())
    fraction = float(valid.mean())
    significant = numerical & (np.abs(oracle) > abs(amp) * 1e-4)
    current_footprint = numerical & (np.abs(current_oracle) > abs(amp) * 1e-4)
    previous_footprint = numerical & (np.abs(previous_oracle) > abs(amp) * 1e-4)
    target_footprint = current_footprint | previous_footprint
    roi_edge = np.zeros(valid.shape, bool)
    roi_edge[[0, -1], :] = True
    roi_edge[:, [0, -1]] = True
    oracle_roi_truncated = bool(np.any(target_footprint & roi_edge)
                               or not current_footprint.any() or not previous_footprint.any())
    source_partial = previous_injection["stats"]["source_psf_partial"] or current_injection_stats["source_psf_partial"]
    response, clean_response = metrics["delta"]["template_response_dn"], metrics["clean_residual"]["template_response_dn"]
    clean_rms = metrics["clean_residual"]["rms_dn"]
    metrics["visibility"] = dict(isolated_template_response_dn=response, clean_template_response_dn=clean_response,
        clean_rms_dn=clean_rms,
        isolated_response_over_clean_rms=None if response is None or clean_rms is None else _ratio(abs(response), clean_rms),
        isolated_response_over_abs_clean_template=None if response is None or clean_response is None else _ratio(abs(response), abs(clean_response)),
        calibrated_snr=False, detector_pass=None)
    reasons = []
    for key, label in (("numerical", "undefined_saved_fit_or_outside_native"), ("model_support", "training_gate_or_hull_unsupported"),
                       ("kernel_valid", "cubic_footprint_outside_native"), ("valid", "masked_or_erosion_lost")):
        if not local[key].all():
            reasons.append(label)
    if source_partial:
        reasons.append("source_psf_truncated_by_native_image")
    if oracle_roi_truncated:
        reasons.append("oracle_footprint_touches_roi_edge_or_target_missing")
    result = dict(previous_center_xy=list(previous_center), current_center_xy=list(current_center),
        sigma_px=float(sigma), amplitude_dn=float(amp), roi_xyxy=list(roi), source_context_xyxy=list(context),
        total_roi_pixels=int(valid.size), numerical_pixels=int(numerical.sum()), numerical_fraction=float(numerical.mean()),
        model_support_pixels=int(local["model_support"].sum()), kernel_valid_pixels=int(local["kernel_valid"].sum()),
        valid_pre_pixels=int(local["valid_pre"].sum()), valid_pixels=int(valid.sum()), valid_fraction=fraction,
        oracle_absolute_mass_dn=oracle_mass, oracle_valid_absolute_mass_dn=oracle_valid_mass,
        oracle_absolute_mass_coverage=_ratio(oracle_valid_mass, oracle_mass),
        significant_oracle_pixels=int(significant.sum()), significant_oracle_valid_pixels=int((significant & valid).sum()),
        entire_significant_oracle_supported=bool(np.all(valid[significant])),
        full_roi_support=bool(valid.all()),
        full_target_footprint_support=bool(numerical.all() and target_footprint.any() and np.all(valid[target_footprint])),
        significant_target_union_pixels=int(target_footprint.sum()),
        significant_target_union_valid_pixels=int((target_footprint & valid).sum()),
        current_oracle_present=bool(current_footprint.any()), previous_oracle_present=bool(previous_footprint.any()),
        oracle_roi_truncated=oracle_roi_truncated,
        partial=not bool(valid.all() and numerical.all()) or source_partial or oracle_roi_truncated,
        full_support_eligible=bool(numerical.all() and target_footprint.any() and np.all(valid[target_footprint])
                                   and not source_partial and not oracle_roi_truncated),
        source_psf_partial=source_partial,
        unavailable=not bool(valid.any()), support_reasons=reasons,
        previous_injection=previous_injection["stats"], current_injection=current_injection_stats, metrics=metrics)
    if return_arrays:
        result["arrays"] = dict(previous=np.where(native, previous_float, np.nan), current_aligned=clean_warp,
            clean_residual=clean_residual, injected_residual=injected_residual, delta=delta, oracle=oracle,
            current_target_delta=current_target_delta, current_target_oracle=current_oracle,
            valid=valid, numerical=numerical, model_support=local["model_support"], kernel_valid=local["kernel_valid"])
    return result


def field_hashes(field):
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in field.items() if isinstance(value, np.ndarray)}
