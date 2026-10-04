"""Pure causal fixed-lag source-pair extraction for an offline diagnostic.

No video I/O, local registration, photometric fit, detector, or classifier is
implemented here. Global geometry being available is not camera confidence or
evidence of an object. NaN pixels remain missing support, never zero padding.
"""
from collections import deque
import copy
import math

import numpy as np

LAGS = (1, 2, 4, 8)
SHIFTS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
FRAME_NS = 100_000_000
MAX_TRANSFORM_CONDITION = 1e12


def _xy(value, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,)
            or np.iscomplexobj(value)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                   or not math.isfinite(float(v)) for v in value)):
        raise ValueError(name + " must contain two finite real coordinates")
    return np.asarray(value, dtype=np.float64)


def _shift(value):
    value = _xy(value, "shift_xy")
    if tuple(value) not in SHIFTS:
        raise ValueError("Only the fixed nine integer shifts are supported")
    return int(value[0]), int(value[1])


def shifted_prior(prior27, shift_xy):
    """Return prior(q + shift) at q in [-12,12]^2, as an owned 25x25 copy."""
    if np.iscomplexobj(prior27):
        raise ValueError("Real prior pixels required")
    prior = np.asarray(prior27, dtype=np.float64)
    if prior.shape != (27, 27) or np.isinf(prior).any():
        raise ValueError("A 27x27 real prior patch without infinities is required")
    dx, dy = _shift(shift_xy)
    return prior[1+dy:26+dy, 1+dx:26+dx].copy()


def shifted_previous_point(previous_xy, shift_xy):
    """An actual previous point at p appears at p-shift in that shifted patch."""
    dx, dy = _shift(shift_xy)
    if previous_xy is None:
        return None
    return (_xy(previous_xy, "previous_xy") - (dx, dy)).tolist()


def _json_copy(value, path="", nonfinite=None):
    """Preserve quality fields; explicitly record any nonfinite-to-null change."""
    nonfinite = [] if nonfinite is None else nonfinite
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            nonfinite.append(path)
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_json_copy(item, f"{path}[{i}]", nonfinite) for i, item in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_copy(item, path+"."+key if path else key, nonfinite)
                for key, item in value.items()}
    raise ValueError("Journal metadata must use JSON-compatible values")


def _metadata(row):
    nonfinite = []
    result = _json_copy({name: row.get(name) for name in
                        ("frame_index", "timestamp_ns", "segment", "source_to_reference", "motion")},
                        nonfinite=nonfinite)
    result["nonfinite_metadata_fields_replaced_with_null"] = nonfinite
    return result


def _matrix(row):
    value = row.get("source_to_reference")
    if np.iscomplexobj(value):
        return None, "nonreal_transform"
    try:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (3, 3):
            return None, "invalid_transform_shape"
        if not np.isfinite(value).all():
            return None, "nonfinite_transform"
        condition = float(np.linalg.cond(value))
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None, "invalid_transform"
    if not math.isfinite(condition) or condition > MAX_TRANSFORM_CONDITION:
        return None, "singular_or_ill_conditioned_transform"
    return value, None


def _native_crop(gray, center):
    x, y = center
    output = np.full((25, 25), np.nan, dtype=np.float64)
    left, top, right, bottom = max(0, x-12), max(0, y-12), min(gray.shape[1], x+13), min(gray.shape[0], y+13)
    if right > left and bottom > top:
        output[top-y+12:bottom-y+12, left-x+12:right-x+12] = gray[top:bottom, left:right]
    return output


def bilinear_sample(gray, map_x, map_y):
    """Exact float64 bilinear interpolation; only positive-weight corners count.

    Unlike OpenCV's quantized INTER_LINEAR table, this uses the full fractional
    coordinates. An exact border pixel is valid when its zero-weight neighbor
    falls outside the source. Nonfinite/outside coordinates remain NaN.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 1:
        raise ValueError("Nonempty native uint8 source required")
    map_x, map_y = np.asarray(map_x, dtype=np.float64), np.asarray(map_y, dtype=np.float64)
    if map_x.shape != map_y.shape:
        raise ValueError("Sampling maps must have the same shape")
    height, width = gray.shape
    valid = (np.isfinite(map_x) & np.isfinite(map_y)
             & (map_x >= 0) & (map_x <= width-1) & (map_y >= 0) & (map_y <= height-1))
    safe_x, safe_y = np.where(valid, map_x, 0), np.where(valid, map_y, 0)
    x0, y0 = np.floor(safe_x).astype(np.int64), np.floor(safe_y).astype(np.int64)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    fx, fy = safe_x-x0, safe_y-y0
    output = ((1-fx)*(1-fy)*gray[y0, x0] + fx*(1-fy)*gray[y0, x1]
              + (1-fx)*fy*gray[y1, x0] + fx*fy*gray[y1, x1])
    return np.where(valid, output, np.nan).astype(np.float64)


def _prior_patch(gray, center, warp):
    y, x = np.mgrid[-13:14, -13:14]
    coordinates = np.stack((x+float(center[0]), y+float(center[1]), np.ones(x.shape)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = np.einsum("ij,jkl->ikl", warp, coordinates)
        denominator = mapped[2]
        valid = np.isfinite(mapped).all(axis=0) & (denominator != 0)
        map_x = np.divide(mapped[0], denominator, out=np.full(x.shape, np.nan, dtype=float), where=valid)
        map_y = np.divide(mapped[1], denominator, out=np.full(y.shape, np.nan, dtype=float), where=valid)
    finite_den = denominator[np.isfinite(denominator)]
    horizon = bool(finite_den.size and finite_den.min() <= 0 <= finite_den.max())
    output = bilinear_sample(gray, map_x, map_y)
    return output, dict(
        interpolation="exact_float64_bilinear_positive_weight_support",
        mapping_finite_nonzero_denominator_pixels=int(valid.sum()),
        prior_finite_pixels=int(np.isfinite(output).sum()),
        projective_horizon_crosses_patch=horizon,
        denominator_min=float(finite_den.min()) if finite_den.size else None,
        denominator_max=float(finite_den.max()) if finite_den.size else None)


def _point(warp, point):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = warp @ np.array([point[0], point[1], 1.0])
        if not np.isfinite(mapped).all() or mapped[2] == 0:
            return None
        mapped = mapped[:2]/mapped[2]
    return mapped if np.isfinite(mapped).all() else None


def _actual_measurement(row, identity):
    if identity is None:
        return None, "identity_not_requested"
    found = [track for track in row["tracks"]
             if f'{track["segment"]}/{track["track_id"]}' == identity]
    if not found:
        return None, "identity_not_present"
    track = found[0]
    if track.get("measured") is not True:
        return None, "identity_present_without_actual_measurement"
    try:
        xy = _xy(track.get("measurement_source_xy"), "actual measurement")
    except ValueError:
        return None, "invalid_actual_measurement_coordinates"
    return xy, "actual_measurement"


class CausalSourceBuffer:
    """Own at most max_lag+1 source frames and rows, with a hard limit of nine.

    Updates must begin at frame zero and use the original 10 Hz timestamp grid.
    Extraction always reports the complete fixed lag bank. A smaller configured
    capacity explicitly makes larger lags unavailable; it never changes lags.
    """
    def __init__(self, max_lag=8):
        if type(max_lag) is not int or not 1 <= max_lag <= max(LAGS):
            raise ValueError("max_lag must be an integer from one through eight")
        self.max_lag = max_lag
        self._entries = deque(maxlen=max_lag+1)
        self._frame = -1
        self._shape = None

    def update(self, row, gray):
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 1:
            raise ValueError("Nonempty native uint8 grayscale frame required")
        if self._shape is not None and gray.shape != self._shape:
            raise ValueError("Source geometry cannot change within a buffer")
        if (not isinstance(row, dict) or type(row.get("frame_index")) is not int
                or row["frame_index"] != self._frame+1
                or type(row.get("timestamp_ns")) is not int
                or row["timestamp_ns"] != row["frame_index"]*FRAME_NS
                or type(row.get("segment")) is not int or row["segment"] < 0
                or not isinstance(row.get("motion"), dict)
                or type(row["motion"].get("reset")) is not bool
                or not isinstance(row.get("tracks"), list)):
            raise ValueError("Contiguous frame-zero 10Hz journal with explicit segment/reset required")
        identities = set()
        for track in row["tracks"]:
            if (not isinstance(track, dict) or type(track.get("segment")) is not int
                    or track["segment"] != row["segment"]
                    or not isinstance(track.get("track_id"), str) or not track["track_id"]
                    or type(track.get("measured")) is not bool):
                raise ValueError("Journal track identity/measurement metadata invalid")
            identity = f'{track["segment"]}/{track["track_id"]}'
            if identity in identities:
                raise ValueError("Duplicate journal track identity")
            identities.add(identity)
        # Validate metadata and create all owned state before committing.
        _metadata(row)
        owned_row, owned_gray = copy.deepcopy(row), np.array(gray, dtype=np.uint8, order="C", copy=True)
        owned_gray.flags.writeable = False
        self._entries.append((owned_row, owned_gray))
        self._frame = row["frame_index"]
        self._shape = gray.shape
        return self.retained_state_counts()

    def retained_state_counts(self):
        return dict(frames=len(self._entries), maximum_frames=self.max_lag+1, max_lag=self.max_lag,
                    oldest_frame=self._entries[0][0]["frame_index"] if self._entries else None,
                    current_frame=self._frame if self._entries else None,
                    owned_uint8_bytes=sum(gray.nbytes for _, gray in self._entries))

    def extract(self, current_xy, identity=None, lags=LAGS):
        if not self._entries:
            raise ValueError("No current frame is buffered")
        if (not isinstance(lags, (tuple, list)) or len(lags) != len(LAGS)
                or any(type(lag) is not int for lag in lags) or tuple(lags) != LAGS):
            raise ValueError("The exact fixed lag bank (1,2,4,8) is required")
        xy = _xy(current_xy, "current_xy")
        row, gray = self._entries[-1]
        if identity is not None:
            if not isinstance(identity, str) or "/" not in identity:
                raise ValueError("Identity must be the original 'segment/track_id' string")
            actual, reason = _actual_measurement(row, identity)
            if actual is None or not np.array_equal(actual, xy):
                raise ValueError("Requested current identity/coordinate must be an actual measurement")
        center = [math.floor(float(value)+.5) for value in xy]
        current25 = _native_crop(gray, center)
        result = dict(current25=current25, current_integer_center_xy=center,
                      current_xy=[float(xy[i])-center[i] for i in range(2)],
                      actual_current_source_xy=xy.tolist(), identity=identity,
                      current_frame=_metadata(row), source_shape_hw=list(gray.shape),
                      current_finite_pixels=int(np.isfinite(current25).sum()),
                      buffer_scope=self.retained_state_counts(), lags=[])
        entries = {past["frame_index"]: (past, image) for past, image in self._entries}
        current_matrix, current_error = _matrix(row)
        for lag in LAGS:
            previous_index = self._frame-lag
            interval = [past for past, _ in self._entries if previous_index < past["frame_index"] <= self._frame]
            item = dict(lag=lag, available=False, reasons=[], prior27=None,
                        current_frame_index=self._frame, requested_prior_frame_index=previous_index,
                        requested_prior_timestamp_ns=previous_index*FRAME_NS,
                        delta_time_ns=lag*FRAME_NS, prior_frame=None,
                        intervening_frames=[_metadata(past) for past in interval],
                        current_to_prior_matrix=None, optional_identity=identity,
                        previous_actual_measurement_available=False,
                        previous_actual_measurement_source_xy=None,
                        previous_point_current_grid_xy=None,
                        previous_measurement_status="not_evaluated",
                        support=None, shift_support=[],
                        geometry_availability_is_not_motion_confidence=True,
                        no_local_registration=True, no_photometric_correction=True)
            result["lags"].append(item)
            if lag > self.max_lag:
                item["reasons"].append("lag_exceeds_buffer_capacity")
            elif previous_index not in entries:
                item["reasons"].append("prior_frame_not_yet_available")
            if item["reasons"]:
                continue
            prior, prior_gray = entries[previous_index]
            item["prior_frame"] = _metadata(prior)
            previous_xy, status = _actual_measurement(prior, identity)
            item["previous_measurement_status"] = status
            if previous_xy is not None:
                item["previous_actual_measurement_available"] = True
                item["previous_actual_measurement_source_xy"] = previous_xy.tolist()
            if any(past["segment"] != prior["segment"] for past in interval):
                item["reasons"].append("intervening_reference_segment_change")
            if any(past["motion"]["reset"] for past in interval):
                item["reasons"].append("intervening_reference_reset")
            prior_matrix, prior_error = _matrix(prior)
            if prior_error:
                item["reasons"].append("prior_" + prior_error)
            if current_error:
                item["reasons"].append("current_" + current_error)
            # Intermediate transform failures are preserved as unavailable too;
            # merely reusing a finite transform with accepted=False is not one.
            for past in interval[:-1]:
                _, error = _matrix(past)
                if error:
                    item["reasons"].append(f'intervening_frame_{past["frame_index"]}_{error}')
            if item["reasons"]:
                continue
            try:
                warp = np.linalg.solve(prior_matrix, current_matrix)
                if not np.isfinite(warp).all():
                    raise np.linalg.LinAlgError("Nonfinite composed geometry")
                condition = float(np.linalg.cond(warp))
                if not math.isfinite(condition) or condition > MAX_TRANSFORM_CONDITION:
                    raise np.linalg.LinAlgError("Invalid composed geometry")
                inverse = np.linalg.inv(warp)
                if not np.isfinite(inverse).all():
                    raise np.linalg.LinAlgError("Nonfinite inverse geometry")
            except np.linalg.LinAlgError:
                item["reasons"].append("invalid_or_ill_conditioned_current_to_prior_transform")
                continue
            item["current_to_prior_matrix"] = warp.tolist()
            patch, support = _prior_patch(prior_gray, center, warp)
            item["prior27"], item["support"] = patch, support
            if support["projective_horizon_crosses_patch"]:
                item["reasons"].append("projective_horizon_crosses_patch")
            if support["mapping_finite_nonzero_denominator_pixels"] == 0:
                item["reasons"].append("no_finite_projective_mapping_support")
            if previous_xy is not None:
                mapped = _point(inverse, previous_xy)
                if mapped is not None:
                    item["previous_point_current_grid_xy"] = (mapped-center).tolist()
                else:
                    item["previous_measurement_status"] = "actual_measurement_mapping_unavailable"
            for shift in SHIFTS:
                shifted = shifted_prior(patch, shift)
                finite = np.isfinite(current25) & np.isfinite(shifted)
                item["shift_support"].append(dict(shift_xy=list(shift), common_finite_pixels=int(finite.sum()),
                    complete=bool(finite.all()), previous_point_xy=shifted_previous_point(
                        item["previous_point_current_grid_xy"], shift)))
            item["available"] = not item["reasons"]
        return result
