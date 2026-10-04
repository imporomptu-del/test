"""Pure descriptive weak-evidence enumeration; no IO, assignment or acceptance.

The caller must freeze forecasts using past-only state and verify capture
provenance. This module never estimates, updates or repairs a forecast.
"""
from __future__ import annotations

import numpy as np

POSITION_GATE_PX = 45.0
MAHALANOBIS_SQUARED_GATE = 25.0
FLOAT_FIELDS = (
    "image", "blur", "median", "spatial", "background", "temporal", "variance",
    "tile_center", "tile_sigma_float", "noise", "centered_temporal",
    "temporal_threshold_dn", "spatial_threshold_dn", "positive_score", "negative_score",
    "neighborhood_max_abs", "positive_signed_temporal", "negative_signed_temporal",
    "positive_signed_spatial", "negative_signed_spatial",
)
FLAG_FIELDS = (
    "support", "previous_support", "native_eligible", "ready", "eligible",
    "positive_temporal_pass", "negative_temporal_pass", "positive_spatial_pass",
    "negative_spatial_pass", "raw_absolute_peak", "positive_candidate", "negative_candidate",
    "finite_neighborhood",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def _bounds(value, name):
    require(isinstance(value, (list, tuple)) and len(value) == 4
            and all(type(v) is int for v in value), name+" must contain four integers")
    require(0 <= value[0] < value[2] and 0 <= value[1] < value[3], name+" is invalid")
    return tuple(value)


def _forecast(value):
    require(isinstance(value, dict) and isinstance(value.get("identity"), str)
            and bool(value["identity"]), "forecast identity required")
    require(value.get("polarity") in ("bright", "dark"), "forecast polarity invalid")
    center = np.asarray(value.get("reference_xy"), dtype=np.float64)
    cov = np.asarray(value.get("innovation_covariance_2x2"), dtype=np.float64)
    require(center.shape == (2,) and np.isfinite(center).all(), "finite forecast center required")
    require(cov.shape == (2, 2) and np.isfinite(cov).all() and np.array_equal(cov, cov.T), "finite symmetric innovation covariance required")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as error:
        raise ValueError("positive definite innovation covariance required") from error
    return dict(identity=value["identity"], polarity=value["polarity"], center=center.copy(), chol=chol)


def _gate(forecast, x, y):
    dx, dy = x-forecast["center"][0], y-forecast["center"][1]
    distance2 = dx*dx + dy*dy
    # Cholesky avoids an explicit inverse; no covariance inflation/fallback.
    lower = forecast["chol"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z0 = dx/lower[0, 0]
        z1 = (dy-lower[1, 0]*z0)/lower[1, 1]
        d2 = z0*z0+z1*z1
    require(np.isfinite(distance2).all() and np.isfinite(d2).all(), "unusable forecast arithmetic")
    return (distance2 <= POSITION_GATE_PX**2) & (d2 <= MAHALANOBIS_SQUARED_GATE), distance2, d2


def _inside_disk(center, bounds):
    x, y = center
    left, top, right, bottom = bounds
    return bool(x-POSITION_GATE_PX >= left and x+POSITION_GATE_PX <= right-1
                and y-POSITION_GATE_PX >= top and y+POSITION_GATE_PX <= bottom-1)


def _neighborhood(temporal, x0, y0, shape_hw):
    """Recompute original raw absolute5x5 peak on available capture neighbors."""
    height, width = temporal.shape
    yy, xx = np.mgrid[:height, :width]
    image_h, image_w = shape_hw
    complete = ((np.maximum(xx+x0-2, 0) >= x0) & (np.minimum(xx+x0+2, image_w-1) < x0+width)
                & (np.maximum(yy+y0-2, 0) >= y0) & (np.minimum(yy+y0+2, image_h-1) < y0+height))
    absolute = np.abs(temporal)
    peak = np.ones_like(temporal, dtype=bool)
    finite = np.ones_like(peak)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sy = slice(max(0, -dy), min(height, height-dy))
            sx = slice(max(0, -dx), min(width, width-dx))
            neighbors = absolute[max(0, dy):min(height, height+dy), max(0, dx):min(width, width+dx)]
            peak[sy, sx] &= ~(neighbors > absolute[sy, sx])
            finite[sy, sx] &= np.isfinite(neighbors)
    return complete, peak, finite


def enumerate_peaks(values, flags, metadata, focal_forecast, prior_forecasts):
    """Return every observed frozen-gate peak; identity remains unknown.

    Metadata uses the archived V56 ``rectangle`` and named field lists.
    Every forecast has identity, reference_xy, innovation_covariance_2x2 and
    polarity. Competing forecasts need not have centers inside the capture.
    Inputs are read-only; no source/reference positions are used for selection.
    """
    require(isinstance(metadata, dict), "metadata required")
    fields, bits = metadata.get("float_fields"), metadata.get("flag_fields")
    require(isinstance(fields, list) and len(fields) == len(FLOAT_FIELDS) and set(fields) == set(FLOAT_FIELDS), "float field schema differs")
    require(isinstance(bits, list) and len(bits) == len(FLAG_FIELDS) and set(bits) == set(FLAG_FIELDS), "flag field schema differs")
    vindex, findex = {n:i for i,n in enumerate(fields)}, {n:i for i,n in enumerate(bits)}
    rectangle = metadata.get("rectangle", {})
    capture = _bounds(rectangle.get("capture_bounds_exclusive_xyxy"), "capture bounds")
    tile = _bounds(rectangle.get("tile_bounds_exclusive_xyxy"), "full tile bounds")
    shape = rectangle.get("shape_hw")
    require(isinstance(shape, (list, tuple)) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), "native shape invalid")
    require(capture[2] <= shape[1] and capture[3] <= shape[0] and tile[2] <= shape[1] and tile[3] <= shape[0], "bounds exceed native frame")
    require(isinstance(values, np.ndarray) and values.dtype == np.float32
            and values.shape == (capture[3]-capture[1], capture[2]-capture[0], len(fields)), "float32 capture shape required")
    require(isinstance(flags, np.ndarray) and flags.dtype == np.uint8
            and flags.shape == values.shape[:2]+(len(bits),) and np.all(flags <= 1), "binary uint8 flag shape required")
    require(type(metadata.get("ready")) is bool, "capture readiness required")
    for name, expected in (("temporal_threshold_sigma_float32", 4.0), ("spatial_threshold_sigma_float32", 3.0)):
        number = metadata.get(name)
        require(type(number) in (int, float) and np.isfinite(number) and number == expected,
                "frozen temporal4/spatial3 sigma thresholds required")
    focal = _forecast(focal_forecast)
    require(isinstance(prior_forecasts, (list, tuple)), "prior forecast list required")
    forecasts = [_forecast(f) for f in prior_forecasts]
    require(len({f["identity"] for f in forecasts}) == len(forecasts), "duplicate prior identity")
    for f in forecasts:
        if f["identity"] == focal["identity"]:
            require(f["polarity"] == focal["polarity"] and np.array_equal(f["center"], focal["center"])
                    and np.array_equal(f["chol"], focal["chol"]), "focal forecast differs in prior list")
    field = lambda name: values[..., vindex[name]]
    flag = lambda name: flags[..., findex[name]].astype(bool)
    yy, xx = np.mgrid[:values.shape[0], :values.shape[1]]
    gx, gy = xx+capture[0], yy+capture[1]
    gate, distance2, d2 = _gate(focal, gx, gy)
    interior = (gx >= tile[0]) & (gx < tile[2]) & (gy >= tile[1]) & (gy < tile[3])
    observed_gate = gate & interior
    finite_pixel = np.isfinite(values).all(axis=2) & (field("noise") > 0) & (field("variance") >= 0)
    complete, raw_peak, finite_neighbors = _neighborhood(field("temporal"), capture[0], capture[1], shape)
    # Validate the archived predicates where finite data make them comparable.
    eligible = metadata["ready"] & flag("support") & flag("previous_support")
    require(np.array_equal(flag("eligible"), eligible) and np.array_equal(flag("native_eligible"), eligible)
            and np.all(flag("ready") == metadata["ready"]), "eligibility flags inconsistent")
    comparable = complete & finite_neighbors & finite_pixel
    require(np.array_equal(flag("finite_neighborhood")[complete], finite_neighbors[complete]), "finite-neighborhood flag inconsistent")
    require(np.array_equal(flag("raw_absolute_peak")[comparable], raw_peak[comparable]), "raw peak flag inconsistent")
    centered = np.subtract(field("temporal"), field("tile_center"), dtype=np.float32)
    require(np.array_equal(field("centered_temporal")[finite_pixel], centered[finite_pixel]), "centered temporal inconsistent")
    for polarity, sign in (("positive", np.float32(1)), ("negative", np.float32(-1))):
        signed_temporal = np.multiply(centered, sign, dtype=np.float32)
        signed_spatial = np.multiply(field("spatial"), sign, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.divide(signed_temporal, field("noise"), dtype=np.float32)
        for name, expected in ((polarity+"_signed_temporal", signed_temporal), (polarity+"_signed_spatial", signed_spatial), (polarity+"_score", score)):
            require(np.array_equal(field(name)[finite_pixel], expected[finite_pixel]), "signed amplitude/score inconsistent")
        tp = signed_temporal >= field("temporal_threshold_dn")
        sp = signed_spatial >= field("spatial_threshold_dn")
        require(np.array_equal(flag(polarity+"_temporal_pass")[finite_pixel], tp[finite_pixel])
                and np.array_equal(flag(polarity+"_spatial_pass")[finite_pixel], sp[finite_pixel]), "threshold flags inconsistent")
        expected_candidate = eligible & tp & sp & flag("raw_absolute_peak")
        require(np.array_equal(flag(polarity+"_candidate")[finite_pixel], expected_candidate[finite_pixel]), "candidate flags inconsistent")
    for domain in ("temporal", "spatial"):
        expected = np.multiply(np.float32(metadata[domain+"_threshold_sigma_float32"]), field("noise"), dtype=np.float32)
        require(np.array_equal(field(domain+"_threshold_dn")[finite_pixel], expected[finite_pixel]), "threshold magnitude inconsistent")
    reasons = []
    if not _inside_disk(focal["center"], tile):
        reasons.append("focal_45px_disk_outside_full_tile")
    if not _inside_disk(focal["center"], capture):
        reasons.append("focal_45px_disk_outside_capture")
    for mask, reason in ((~eligible, "gate_pixels_not_eligible"), (~finite_pixel, "gate_pixels_nonfinite"),
                         (~complete, "peak_neighborhood_missing"), (~finite_neighbors, "peak_neighborhood_nonfinite")):
        if np.any(observed_gate & mask):
            reasons.append(reason)
    name = "positive" if focal["polarity"] == "bright" else "negative"
    selected = (observed_gate & eligible & finite_pixel & complete & finite_neighbors & flag("finite_neighborhood")
                & flag("raw_absolute_peak") & flag(name+"_spatial_pass") & (field(name+"_signed_temporal") > 0))
    iy, ix = np.nonzero(selected)
    order = np.lexsort((gx[iy,ix], gy[iy,ix], -field(name+"_score")[iy,ix]))
    # Competing gates need evaluation only at observed peaks, not260x260 grids.
    prior_gates = [(f, *_gate(f, gx[iy,ix], gy[iy,ix])) for f in forecasts
                   if f["polarity"] == focal["polarity"] and f["identity"] != focal["identity"]]
    peaks = []
    for rank, index in enumerate(order, 1):
        y, x = int(iy[index]), int(ix[index])
        competing = []
        for forecast, member, prior_distance2, prior_d2 in prior_gates:
            if member[index]:
                cx, cy = forecast["center"]
                competing.append(dict(identity=forecast["identity"], distance_px=float(np.sqrt(prior_distance2[index])),
                    mahalanobis_squared=float(prior_d2[index]), forecast_center_outside_capture=not(capture[0] <= cx < capture[2] and capture[1] <= cy < capture[3])))
        competing.sort(key=lambda f:f["identity"])
        original = bool(flag(name+"_temporal_pass")[y,x])
        peaks.append(dict(descriptive_rank=rank, reference_xy=[int(gx[y,x]), int(gy[y,x])], capture_xy=[x,y],
            polarity=focal["polarity"], score=float(field(name+"_score")[y,x]),
            signed_centered_temporal_dn=float(field(name+"_signed_temporal")[y,x]), signed_spatial_dn=float(field(name+"_signed_spatial")[y,x]),
            raw_temporal_dn=float(field("temporal")[y,x]), noise_dn=float(field("noise")[y,x]),
            temporal_threshold_dn=float(field("temporal_threshold_dn")[y,x]), spatial_threshold_dn=float(field("spatial_threshold_dn")[y,x]),
            distance_px=float(np.sqrt(distance2[y,x])), mahalanobis_squared=float(d2[y,x]),
            original_temporal_threshold_pass=original, evidence_partition="original_threshold_pass" if original else "weak_temporal",
            competing_prior_identity_gates=competing, identity_assignment=None, acceptance_decision=None))
    return dict(schema="seaqr.weak-continuation-information.v1", focal_identity=focal["identity"],
        position_gate_px=POSITION_GATE_PX, mahalanobis_squared_gate=MAHALANOBIS_SQUARED_GATE,
        observed_peaks=peaks, observed_peak_count=len(peaks), weak_peak_count=sum(not p["original_temporal_threshold_pass"] for p in peaks),
        original_threshold_peak_count=sum(p["original_temporal_threshold_pass"] for p in peaks),
        coverage_known=not reasons, coverage_unknown_reasons=reasons,
        observed_gate_pixel_count=int(observed_gate.sum()), eligible_observed_gate_pixel_count=int((observed_gate & eligible).sum()),
        consistency_checked_finite_pixels=int(finite_pixel.sum()), identity_assignment=None, acceptance_decision=None,
        identity_observability="not_identified_from_these_observations", forecast_causality_verified_by_this_function=False,
        limits=["Descriptive candidates only; even a unique peak can be an unrelated source", "Forecasts and input provenance must be frozen/verified by caller",
                "Incomplete or censored coverage does not establish candidate absence", "No current/future data changes forecasts; no classifier, identity decision or production modification"])
