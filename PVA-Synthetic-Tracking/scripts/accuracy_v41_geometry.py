"""Saved-crop geometry for an offline source-transport diagnostic.

Sampling is not a visual enhancement and missing pixels remain unknown. Global
geometry and same-ID displacement are exposed tracker estimates, not truth.
"""
import math
import numpy as np
from accuracy_v38_source_pairs import bilinear_sample

SHIFTS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
LAGS = (1, 2, 4, 8)


def _xy(value, name):
    if (np.iscomplexobj(value) or np.shape(value) != (2,)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                   or not math.isfinite(float(v)) for v in value)):
        raise ValueError(name + " must have two finite real coordinates")
    return np.asarray(value, dtype=float)


def _matrix(value):
    if np.iscomplexobj(value):
        raise ValueError("Real transform required")
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Finite 3x3 transform required")
    if np.linalg.cond(matrix) > 1e12:
        raise ValueError("Singular or ill-conditioned transform")
    return matrix


def _sample(gray, mx, my):
    """Exclude every saturated source corner with strictly positive weight."""
    values = bilinear_sample(gray, mx, my)
    inside = (np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0)
              & (mx <= gray.shape[1]-1) & (my <= gray.shape[0]-1))
    xx, yy = np.where(inside, mx, 0), np.where(inside, my, 0)
    x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
    x1, y1 = np.minimum(x0+1, gray.shape[1]-1), np.minimum(y0+1, gray.shape[0]-1)
    fx, fy = xx-x0, yy-y0
    valid = inside.copy()
    for x, y, weight in ((x0, y0, (1-fx)*(1-fy)), (x1, y0, fx*(1-fy)),
                          (x0, y1, (1-fx)*fy), (x1, y1, fx*fy)):
        valid &= (weight == 0) | ((gray[y, x] > 0) & (gray[y, x] < 255))
    return np.where(valid, values, np.nan)


def source_templates(current, prior, current_origin, prior_origin,
                     current_to_reference, prior_to_reference,
                     current_actual_xy, prior_actual_xy, shift=(0, 0)):
    """Two prior hypotheses share one 25x25 current native sampling grid.

The transported offset is prior_actual - warp(current_actual), not a filtered
position or velocity extrapolation. Positive offsets sample further right/down
in the PRIOR frame. A camera motion can thus yield zero residual transport.
"""
    for pixels in (current, prior):
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 2 or not pixels.size:
            raise ValueError("Nonempty uint8 grayscale crops required")
    origin_c, origin_p = _xy(current_origin, "current_origin"), _xy(prior_origin, "prior_origin")
    actual_c, actual_p = _xy(current_actual_xy, "current_actual_xy"), _xy(prior_actual_xy, "prior_actual_xy")
    shift = _xy(shift, "shift")
    if tuple(shift) not in SHIFTS:
        raise ValueError("Only the frozen nine integer sensitivity shifts are supported")
    hc, hp = _matrix(current_to_reference), _matrix(prior_to_reference)
    warp = np.linalg.solve(hp, hc)
    center = np.floor(actual_c + 0.5).astype(np.int64)
    y, x = np.mgrid[-12:13, -12:13]
    xx, yy = x + center[0], y + center[1]
    mapped = np.einsum("ij,jkl->ikl", warp, np.stack((xx, yy, np.ones(x.shape))))
    denom = mapped[2]
    if not np.isfinite(mapped).all() or denom.min() <= 0 <= denom.max():
        raise ValueError("Projective horizon or nonfinite mapping within patch")
    mapped_x, mapped_y = mapped[0] / denom, mapped[1] / denom
    point = warp @ np.array([*actual_c, 1.])
    if not np.isfinite(point).all() or point[2] == 0:
        raise ValueError("Invalid actual-point mapping")
    point = point[:2] / point[2]
    offset = actual_p - point
    current_patch = _sample(current, xx-origin_c[0], yy-origin_c[1])
    stationary = _sample(prior, mapped_x-origin_p[0]+shift[0], mapped_y-origin_p[1]+shift[1])
    transported = _sample(prior, mapped_x-origin_p[0]+offset[0]+shift[0],
                                 mapped_y-origin_p[1]+offset[1]+shift[1])
    return dict(current=current_patch, stationary_prior=stationary, transported_prior=transported,
        geometry=dict(current_center_xy=center.tolist(), current_actual_xy=actual_c.tolist(),
            prior_actual_xy=actual_p.tolist(), current_to_prior=warp.tolist(),
            mapped_current_actual_xy=point.tolist(), prior_transport_offset_xy=offset.tolist(),
            prior_transport_offset_norm_px=float(np.linalg.norm(offset)),
            common_prior_shift_xy=shift.tolist(), current_native_pixels=int(np.isfinite(current_patch).sum()),
            stationary_supported_pixels=int(np.isfinite(stationary).sum()),
            transported_supported_pixels=int(np.isfinite(transported).sum())))
