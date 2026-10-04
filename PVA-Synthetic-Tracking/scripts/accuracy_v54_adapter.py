"""Observed-only synthetic guard-to-core adapter for six frozen V54 branches.

Core extrapolation is a new experimental numeric operation. The frozen V53
guard-only prediction API is unchanged and is never called with core pixels.
No current core, labels, source parameters, truth or scoring enter this module.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V53_MODEL_SHA256 = "aa490ecf8305dd0c5f3facff83a5fa8fef43c67460eb55572f516b298a079d04"
V53_TEST_SHA256 = "d5c53052fca8f47bf1f380582702e62217a353587b906fe759d564b460e70f05"
INHERITED_FILES = (
    (ROOT/"scripts/accuracy_v53_offset.py", V53_MODEL_SHA256),
    (ROOT/"tests/unit/test_accuracy_v53_offset.py", V53_TEST_SHA256),
)
GUARD_POINTS = tuple((x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                     if 40 <= max(abs(x-64), abs(y-64)) <= 56)
CORE_POINTS = tuple((x, y) for y in range(52, 77) for x in range(52, 77))
METHODS = ("median8", "median3", "offset_left_right_fold0", "offset_left_right_fold1",
           "offset_checkerboard_fold0", "offset_checkerboard_fold1")


def _read_pinned(path, expected):
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
        raise ValueError("Inherited source must be a canonical regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen V53 source/test hash differs: "+str(path))
    return raw


def _load_pinned_v53():
    # Verify BOTH bindings before any inherited source executes. Executing the
    # verified bytes also avoids stale import-cache/pyc or re-read races.
    verified = [_read_pinned(path, expected) for path, expected in INHERITED_FILES]
    inherited = ModuleType("_accuracy_v54_pinned_v53_offset")
    inherited.__file__ = str(INHERITED_FILES[0][0])
    exec(compile(verified[0], inherited.__file__, "exec"), inherited.__dict__)
    return inherited


_V53 = _load_pinned_v53()


def model_constants():
    return dict(methods=list(METHODS), guard_count=144, core_count=625,
        prior_count=8, median3_uses_latest_priors=3,
        guard_order="y_major_then_x_major", core_order="y_major_then_x_major",
        guard_grid="8..120_step8_Chebyshev_radius40..56_about64",
        core_grid="52..76_step1",
        medians_require_every_used_history_sample_finite=True,
        median_algorithm="numpy_median_float64",
        offset_core_prior="median8", guard_fits_shared_across_source_on_off=True,
        inherited_model_sha256=V53_MODEL_SHA256, inherited_test_sha256=V53_TEST_SHA256,
        inherited_constants=_V53.model_constants(),
        offset_unavailability_precedence=["fit_unavailable", "nonfinite_core_median8", "nonfinite_offset_addition"],
        clipping_selection_blending_or_fallback=False)


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fingerprint(value, excluded=()):
    payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(json.dumps(_plain(payload), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prediction_fingerprint(result):
    return _fingerprint(result, ("prediction_sha256",))


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(array, dtype=np.float64, copy=True)


def _geometry(value, points, name):
    xy = _numeric(value, (len(points), 2), name)
    if not np.array_equal(xy, np.asarray(points)):
        raise ValueError(name+" must equal the full canonical grid in y-major order")
    return xy


def _median(history):
    finite = np.isfinite(history).all(axis=0)
    values = np.full(history.shape[1], np.nan)
    available = np.zeros(history.shape[1], dtype=bool)
    reasons = [None if good else "nonfinite_history" for good in finite]
    indices = np.flatnonzero(finite)
    if len(indices):
        with np.errstate(over="ignore", invalid="ignore"):
            medians = np.median(history[:, finite], axis=0)
        valid = np.isfinite(medians)
        values[indices[valid]] = medians[valid]
        available[indices[valid]] = True
        for index in indices[~valid]:
            reasons[int(index)] = "nonfinite_median_arithmetic"
    return dict(values=_readonly(values), available=_readonly(available),
                unavailable_reasons=tuple(reasons))


def _apply_offset(fitted, baseline):
    """New core numeric adapter; intentionally not the V53 guard predict API."""
    if fitted["model_sha256"] != _V53.model_fingerprint(fitted):
        raise ValueError("Guard fit fingerprint differs before core application")
    count = len(baseline["values"])
    values = np.full(count, np.nan)
    available = np.zeros(count, dtype=bool)
    reasons = [None]*count
    if not fitted["available"]:
        reasons = ["fit_unavailable:"+str(fitted["unavailable_reason"])]*count
    else:
        finite = baseline["available"]
        for index in np.flatnonzero(~finite):
            reasons[int(index)] = "nonfinite_core_median8"
        indices = np.flatnonzero(finite)
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = baseline["values"][finite]+fitted["offset_dn"]
        valid = np.isfinite(predicted)
        values[indices[valid]] = predicted[valid]
        available[indices[valid]] = True
        for index in indices[~valid]:
            reasons[int(index)] = "nonfinite_offset_addition"
    return dict(values=_readonly(values), available=_readonly(available),
                unavailable_reasons=tuple(reasons))


def predict(guard_xy, core_xy, guard_history, guard_current, core_history_on, core_history_off):
    """Freeze all six core-background predictions from observed inputs only."""
    guard_xy = _geometry(guard_xy, GUARD_POINTS, "guard_xy")
    core_xy = _geometry(core_xy, CORE_POINTS, "core_xy")
    guard_history = _numeric(guard_history, (8, 144), "guard_history")
    guard_current = _numeric(guard_current, (144,), "guard_current")
    core_history_on = _numeric(core_history_on, (8, 625), "core_history_on")
    core_history_off = _numeric(core_history_off, (8, 625), "core_history_off")
    input_hash = _fingerprint(dict(guard_xy=guard_xy, core_xy=core_xy,
        guard_history=guard_history, guard_current=guard_current,
        core_history_on=core_history_on, core_history_off=core_history_off))
    guard_median8 = _median(guard_history)
    predictions = {
        "median8": dict(on=_median(core_history_on), off=_median(core_history_off), guard_fit_sha256=None),
        "median3": dict(on=_median(core_history_on[-3:]), off=_median(core_history_off[-3:]), guard_fit_sha256=None),
    }
    guard_crossfits = {}
    for split in ("left_right", "checkerboard"):
        crossfit = _V53.crossfit(guard_xy, guard_median8["values"], guard_current, split)
        if crossfit["crossfit_sha256"] != _V53.crossfit_fingerprint(crossfit):
            raise ValueError("Guard crossfit fingerprint differs")
        guard_crossfits[split] = crossfit
        for fold in (0, 1):
            fitted = crossfit["fits"]["median_offset"][str(fold)]
            predictions[f"offset_{split}_fold{fold}"] = dict(
                on=_apply_offset(fitted, predictions["median8"]["on"]),
                off=_apply_offset(fitted, predictions["median8"]["off"]),
                guard_fit_sha256=fitted["model_sha256"])
    result = dict(schema_version=1, total_guard_count=144, total_core_count=625,
        input_sha256=input_hash, constants=model_constants(), guard_median8=guard_median8,
        guard_crossfits=guard_crossfits, predictions=predictions,
        metadata=dict(current_core_or_truth_argument_accepted=False,
            core_values_enter_guard_fit=False, guard_fits_shared_across_source_on_off=True,
            all_four_guard_folds_preserved_without_selection=True,
            new_guard_to_core_extrapolation_adapter=True,
            inherited_guard_only_predict_api_unchanged=True,
            source_core_safety_or_real_detection_accuracy_certified=False,
            production_decisions_modified=False, scoring_performed=False,
            unknown_predictions_preserved_without_fallback=True))
    result["prediction_sha256"] = prediction_fingerprint(result)
    return result
