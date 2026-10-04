"""Frozen descriptive patch features and transport geometry, without a classifier.

The caller owns media scope, coordinate-reset checks, selection and provenance.
This module neither reads files nor clips observations or predicted coordinates.
"""
from __future__ import annotations

import math

import numpy as np

SIZE, RADIUS = 65, 32
SIGMAS = (1.5, 4.0, 8.0)
EPS = np.finfo(np.float64).eps
MAX_HOMOGRAPHY_CONDITION = 1e12
YY, XX = np.mgrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1].astype(np.float64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _gaussian(sigma):
    values = np.exp(-(XX * XX + YY * YY) / (2 * sigma * sigma))
    return values / np.sum(values, dtype=np.float64)


GAUSSIANS = tuple(_gaussian(sigma) for sigma in SIGMAS)
KERNEL_A = GAUSSIANS[0] - GAUSSIANS[1]
KERNEL_B = GAUSSIANS[1] - GAUSSIANS[2]
for _values in (XX, YY, *GAUSSIANS, KERNEL_A, KERNEL_B):
    _values.setflags(write=False)


def _image(image, shape=None, *, as_float64=True):
    value = np.asarray(image)
    require(value.ndim == 2 and min(value.shape) >= 1 and (shape is None or value.shape == shape),
            "finite two-dimensional image with the declared shape required")
    require(value.dtype == np.uint8 or value.dtype.kind == "f", "uint8 or floating-point image required")
    # Native U8 is finite by construction. Sampling only needs selected corner
    # values; do not copy or scan a full native camera frame for each65x65 site.
    if value.dtype == np.uint8:
        return value.astype(np.float64) if as_float64 else value
    require(np.isfinite(value).all(), "nonfinite image input")
    result = np.asarray(value, dtype=np.float64) if as_float64 else value
    require(not as_float64 or np.isfinite(result).all(), "float64 conversion overflow")
    return result


def _sign(polarity):
    require(polarity in ("bright", "dark"), "polarity must be bright or dark")
    return 1.0 if polarity == "bright" else -1.0


def measure_patch(patch, polarity):
    """Return two signed DoG amplitudes and a bounded descriptive scale ratio.

    Numerical guard per amplitude is16*eps*4225*max(1,max(abs(I)))DN.
    The ratio is null when |A|+|B| is at most twice that conservative guard.
    No observed value is zeroed, thresholded into a class, or intensity-clipped.
    """
    values, sign = _image(patch, (SIZE, SIZE)), _sign(polarity)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean = float(np.mean(values, dtype=np.float64))
            centered = values - mean
            a = float(np.sum(centered * KERNEL_A, dtype=np.float64))
            b = float(np.sum(centered * KERNEL_B, dtype=np.float64))
            denominator = abs(a) + abs(b)
            tolerance = float(16 * EPS * SIZE * SIZE * max(1.0, float(np.max(np.abs(values)))))
    except FloatingPointError as error:
        raise ValueError("numerical overflow in patch arithmetic") from error
    require(all(math.isfinite(v) for v in (mean, a, b, denominator, tolerance)), "nonfinite patch result")
    zero = denominator <= 2 * tolerance
    at_zero, at_255 = bool(np.any(values == 0)), bool(np.any(values == 255))
    saturated = at_zero or at_255
    return dict(A_dn=a, B_dn=b, signed_A_dn=sign * a, signed_B_dn=sign * b,
        polarity=polarity, absolute_amplitude_denominator_dn=denominator,
        R=None if zero else abs(a) / denominator, numerical_amplitude_tolerance_dn=tolerance,
        numerical_denominator_tolerance_dn=2 * tolerance, denominator_numerically_zero=zero,
        mean_dn=mean, minimum_dn=float(np.min(values)), maximum_dn=float(np.max(values)),
        saturation=dict(any=saturated, any_zero=at_zero, any_255=at_255),
        interpretation_available=not saturated and not zero,
        unavailable_reasons=(["saturated_patch"] if saturated else []) + (["numerically_zero_scale_denominator"] if zero else []),
        classifier_applied=False)


def measure_pair(curpatch, bgpatch, trackpatch, polarity):
    """Compare current amplitude with two actual prior-image transported patches.

    Acur/ApriorBG/ApriorActual use polarity-signed A. PriorActual means the
    previous image sampled under the measured residual-displacement hypothesis,
    not an independent truth label. D has no acceptance/class threshold.
    """
    spatial = dict(current=measure_patch(curpatch, polarity), prior_background=measure_patch(bgpatch, polarity),
                   prior_actual=measure_patch(trackpatch, polarity))
    c, b, t = [spatial[key]["signed_A_dn"] for key in ("current", "prior_background", "prior_actual")]
    err_bg, err_actual = abs(c - b), abs(c - t)
    denominator = err_bg + err_actual
    tolerance = (2 * spatial["current"]["numerical_amplitude_tolerance_dn"]
                 + spatial["prior_background"]["numerical_amplitude_tolerance_dn"]
                 + spatial["prior_actual"]["numerical_amplitude_tolerance_dn"])
    require(all(math.isfinite(v) for v in (c, b, t, err_bg, err_actual, denominator, tolerance)), "nonfinite temporal result")
    zero = denominator <= tolerance
    saturated = any(v["saturation"]["any"] for v in spatial.values())
    temporal = dict(Acur_dn=c, ApriorBG_dn=b, ApriorActual_dn=t, error_background_dn=err_bg,
        error_actual_dn=err_actual, error_sum_denominator_dn=denominator,
        numerical_denominator_tolerance_dn=tolerance, denominator_numerically_zero=zero,
        D=None if zero else (err_bg - err_actual) / denominator, any_patch_saturated=saturated,
        interpretation_available=not saturated and not zero,
        unavailable_reasons=(["saturated_patch"] if saturated else []) + (["numerically_zero_temporal_denominator"] if zero else []))
    return dict(spatial=spatial, temporal=temporal, classifier_applied=False,
                interpretation_available=temporal["interpretation_available"])


def bilinear_sample(gray, map_x, map_y):
    """Float64 interpolation; unsupported/nonfinite map locations return NaN.

    Exact final-row/column integer samples are valid: their outboard corner
    weights are zero. Positive-weight corners must all lie inside the image.
    This is not border replication or padding; coordinates are never clamped.
    """
    values = _image(gray, as_float64=False)
    x, y = np.asarray(map_x, dtype=np.float64), np.asarray(map_y, dtype=np.float64)
    require(x.shape == y.shape, "map shapes differ")
    out = np.full(x.shape, np.nan, np.float64)
    h, w = values.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if not np.any(valid):
        return out
    vx, vy = x[valid], y[valid]
    x0, y0 = np.floor(vx).astype(np.int64), np.floor(vy).astype(np.int64)
    # Duplication only at exact edge samples whose outboard weight is zero.
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = vx - x0, vy - y0
    out[valid] = ((1 - wy) * ((1 - wx) * values[y0, x0] + wx * values[y0, x1])
                  + wy * ((1 - wx) * values[y1, x0] + wx * values[y1, x1]))
    require(np.isfinite(out[valid]).all(), "bilinear arithmetic overflow")
    return out


def bilinear_saturation_mask(gray, map_x, map_y):
    """Flag censored native contributors, even when interpolation hides clipping.

    Any positive-weight source corner <=0 or >=255 marks a supported output
    sample. Exact zero-weight neighbors do not count. Unsupported locations
    returnFalse here; bilinear_sample independently returnsNaN for availability.
    """
    values = _image(gray, as_float64=False)
    x, y = np.asarray(map_x, dtype=np.float64), np.asarray(map_y, dtype=np.float64)
    require(x.shape == y.shape, "map shapes differ")
    out = np.zeros(x.shape, np.bool_)
    h, w = values.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if not np.any(valid):
        return out
    vx, vy = x[valid], y[valid]
    x0, y0 = np.floor(vx).astype(np.int64), np.floor(vy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = vx - x0, vy - y0
    flagged = np.zeros(len(vx), np.bool_)
    for yy, xx, weight in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x1, (1 - wy) * wx),
                           (y1, x0, wy * (1 - wx)), (y1, x1, wy * wx)):
        native = values[yy, xx]
        flagged |= (weight > 0) & ((native <= 0) | (native >= 255))
    out[valid] = flagged
    return out


def _xy(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (2,) and np.isfinite(result).all(), "finite x/y point required")
    return result


def _homography(value):
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "finite3x3 homography required")
    scale = float(np.max(np.abs(matrix)))
    require(scale > 0, "zero homography")
    condition = float(np.linalg.cond(matrix / scale))
    require(math.isfinite(condition) and condition <= MAX_HOMOGRAPHY_CONDITION, "ill-conditioned homography")
    return matrix


def transported_maps(cur_xy, prev_xy, Hcur, Hprev):
    """Transport the current65x65 grid to the prior image under two hypotheses.

    F=inv(Hprev)*Hcur maps current source pixels to prior source pixels. The
    measured-position hypothesis adds the uniform residual prev_xy-F(cur_xy).
    Coordinate-reset compatibility is the caller's responsibility.
    """
    current, previous = _xy(cur_xy), _xy(prev_xy)
    hcur, hprev = _homography(Hcur), _homography(Hprev)
    try:
        with np.errstate(over="raise", invalid="raise"):
            forward = np.linalg.solve(hprev, hcur)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("homography transport failed") from error
    _homography(forward)
    cx, cy = current[0] + XX, current[1] + YY
    grid = np.stack((cx, cy, np.ones_like(cx)))
    transported = np.einsum("ij,jhw->ihw", forward, grid)
    require(np.isfinite(transported).all(), "nonfinite transported homogeneous grid")
    denominator = transported[2]
    denominator_scale = float(np.max(np.sum(np.abs(forward[2, :, None, None] * grid), axis=0)))
    tolerance = 64 * EPS * max(np.finfo(np.float64).tiny, denominator_scale)
    require(np.all(np.abs(denominator) > tolerance)
            and (np.all(denominator > 0) or np.all(denominator < 0)), "projective horizon intersects or approaches patch")
    bx, by = transported[0] / denominator, transported[1] / denominator
    center = np.array([bx[RADIUS, RADIUS], by[RADIUS, RADIUS]])
    residual = previous - center
    tx, ty = bx + residual[0], by + residual[1]
    require(all(np.isfinite(a).all() for a in (bx, by, tx, ty)), "nonfinite transported map")
    return dict(current_x=cx.copy(), current_y=cy.copy(), prior_background_x=bx, prior_background_y=by,
        prior_track_x=tx, prior_track_y=ty, F=forward, transported_current_center=center,
        residual_displacement_xy=residual, projective_denominator_tolerance=tolerance,
        coordinate_reset_checked=False)


def synthetic_cases():
    """Analytic case geometry, independent of measured feature results.

    These describe controlled signal constructions, not classifier labels or a
    claim that a ratio separates physical objects from real scene artifacts.
    """
    cases = []
    def blob(sigma, dx=0.0, dy=0.0):
        return np.exp(-((XX - dx) ** 2 + (YY - dy) ** 2) / (2 * sigma * sigma))
    def add(name, polarity, signal_cur, signal_bg, signal_actual, scope):
        sign = _sign(polarity)
        cases.append(dict(name=name + "_" + polarity, polarity=polarity,
            current=80 + sign * signal_cur, prior_background=80 + sign * signal_bg,
            prior_actual=80 + sign * signal_actual, expected_scope=scope))
    for polarity in ("bright", "dark"):
        point = 6 * blob(2)
        dim_point = 2 * blob(2)
        add("dim_point", polarity, dim_point, 2 * blob(2, -2), dim_point,
            dict(construction="moving2DNsigma2 compact Gaussian", residual_displacement_xy=[-2, 0], temporal="D_positive_one_if_numerically_defined"))
        edge = 20 * np.tanh(XX / 2)
        add("point_on_strong_edge", polarity, point + edge, 6 * blob(2, -2) + edge,
            point + 20 * np.tanh((XX - 2) / 2),
            dict(construction="6DNsigma2 point plus20DNtanh(x/2) stationary edge", residual_displacement_xy=[-2, 0], temporal="no_required_D_ordering_on_composite_signal"))
        add("slow_point", polarity, point, 6 * blob(2, -.25), point,
            dict(construction="quarter-pixel compact point displacement", residual_displacement_xy=[-.25, 0], temporal="D_positive_one_but_small_raw_denominator"))
        add("stopped_point", polarity, point, point, point,
            dict(construction="identical stopped compact point", residual_displacement_xy=[0, 0], temporal="numerically_zero_denominator_D_null"))
        broad = 6 * blob(8)
        add("broad_moving_blob", polarity, broad, 6 * blob(8, -2), broad,
            dict(construction="moving sigma8 Gaussian", residual_displacement_xy=[-2, 0], temporal="D_positive_one_if_defined; not_a_classifier"))
        add("alternating_fixed_lights", polarity, point, 6 * blob(2, 12), 6 * blob(2, 12),
            dict(construction="two fixed sites12px apart alternate illumination; zero residual hypothesis", residual_displacement_xy=[0, 0], temporal="D_zero_if_denominator_defined"))
        add("alternating_sites_false_association", polarity, point, 6 * blob(2, -12), point,
            dict(construction="two fixed sites12px apart alternate illumination with deliberately false cross-site association",
                residual_displacement_xy=[-12, 0], physical_entities=2, single_moving_object=False,
                temporal="D_positive_one_despite_two_physical_sites; temporal_descriptor_is_not_identity_truth"))
        add("static_point_camera_motion", polarity, point, point, point,
            dict(construction="static scene point aligned by known camera transport", camera_translation_current_minus_previous_xy=[2, -1],
                residual_displacement_xy=[0, 0], temporal="both_transports_identical_D_null"))
    return cases
