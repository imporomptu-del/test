"""Frozen analytic snapshots for V52's background-correction experiment.

These generated arrays are not camera data, a physical sensor model, object
labels, or detection-accuracy evidence. Only point coordinates, temporal
medians, and observed current guard samples may enter the fitted correction.
Scenario names, clean background truth, and injected-contamination masks are
evaluation metadata and must never be supplied to the fitting procedure.
"""

from copy import deepcopy
import json

import numpy as np


SCHEMA = "accuracy_v52_generated_benchmark_v1"
HISTORY_LENGTH = 8
BACKGROUNDS = ("constant", "planar", "textured")
NOISE_PROFILES = ((0, 71), (1, 991))
FAMILIES = (
    "stable", "offset_plus8", "offset_minus8", "gain125", "gain075",
    "gain_zero", "gain_negative", "plane", "gain_plane", "recent_step",
    "pulse_ended", "long_pulse_ended", "localized_change", "sparse_bright",
    "sparse_dark", "stripe", "missing_prior", "missing_current_left",
    "missing_current_right", "all_current_missing",
)
POINTS_XY = tuple(
    (x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
    if 40 <= max(abs(x - 64), abs(y - 64)) <= 56
)
POINT_COUNT = len(POINTS_XY)


def _spec(family, background, noise_level, seed):
    return {
        "schema": SCHEMA,
        "case_id": f"{family}_{background}_noise{noise_level}_seed{seed}",
        "family": family,
        "background": background,
        "noise_level": noise_level,
        "seed": seed,
        "history_length": HISTORY_LENGTH,
        "point_count": POINT_COUNT,
        "analytic_simulation_only": True,
        "physical_sensor_model": False,
    }


def specifications():
    """Return fresh JSON-safe canonical specifications for all 120 cases."""
    return [_spec(family, background, level, seed)
            for family in FAMILIES for background in BACKGROUNDS
            for level, seed in NOISE_PROFILES]


def _validated_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("A canonical benchmark specification is required")
    canonical_by_id = {item["case_id"]: item for item in specifications()}
    try:
        canonical = canonical_by_id[spec["case_id"]]
        supplied = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Unknown or malformed benchmark specification") from error
    expected = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if supplied != expected:
        raise ValueError("Specification differs from its frozen case")
    return canonical


def _readonly(array):
    array.setflags(write=False)
    return array


def generate_case(spec):
    """Generate one frozen case, preserving missing observations as NaN.

    ``history`` has shape (8, 144), ``points_xy`` has shape (144, 2), and
    all other arrays have shape (144,). Temporal medians use strict median,
    not NaN-skipping reduction. ``clean_current_background`` is noiseless
    background truth, excluding only the explicitly injected local/sparse/
    stripe contamination; it remains finite when observations are missing.
    Measurement noise is added after analytic transformations, with exactly
    the same (9, 144) draw reused for every case with the same noise profile.
    The final noise row belongs to the current observation.
    """
    spec = _validated_spec(spec)
    points = np.asarray(POINTS_XY, dtype=np.int64)
    x, y = points[:, 0], points[:, 1]
    dx, dy = x.astype(np.float64) - 64, y.astype(np.float64) - 64
    planar = 96.0 + 0.05 * dx + 0.03 * dy
    background = spec["background"]
    if background == "constant":
        base = np.full(POINT_COUNT, 96.0)
    elif background == "planar":
        base = planar
    else:
        base = planar + 12 * np.sin(dx / 12) + 9 * np.cos(dy / 15) + 6 * np.sin((dx + dy) / 17)
    history = np.tile(base, (HISTORY_LENGTH, 1))
    clean_current = base.copy()
    contamination = np.zeros(POINT_COUNT, dtype=np.float64)
    family = spec["family"]
    if family == "offset_plus8":
        clean_current += 8
    elif family == "offset_minus8":
        clean_current -= 8
    elif family == "gain125":
        clean_current *= 1.25
    elif family == "gain075":
        clean_current *= 0.75
    elif family == "gain_zero":
        clean_current[:] = 96
    elif family == "gain_negative":
        clean_current = 192 - base
    elif family == "plane":
        clean_current += 4 * dx / 56 - 3 * dy / 56
    elif family == "gain_plane":
        clean_current = 1.15 * base + 8 + 4 * dx / 56 - 3 * dy / 56
    elif family == "recent_step":
        history[-2:] += 8
        clean_current += 8
    elif family == "pulse_ended":
        history[-2:] += 8
    elif family == "long_pulse_ended":
        history += 8
    elif family == "localized_change":
        contamination[(x >= 64) & (y >= 64)] = 16
    elif family == "sparse_bright":
        contamination[np.arange(POINT_COUNT) % 17 == 0] = 40
    elif family == "sparse_dark":
        contamination[np.arange(POINT_COUNT) % 17 == 0] = -40
    elif family == "stripe":
        contamination[np.abs(x - 64) <= 8] = 32
    # Re-seeding is intentional: cases share the entire noise realization.
    noise = np.random.default_rng(spec["seed"]).uniform(
        -spec["noise_level"], spec["noise_level"], size=(HISTORY_LENGTH + 1, POINT_COUNT))
    history += noise[:-1]
    current = clean_current + contamination + noise[-1]
    if family == "missing_prior":
        history[0, :8] = np.nan
    elif family == "missing_current_left":
        current[x < 64] = np.nan
    elif family == "missing_current_right":
        current[x >= 64] = np.nan
    elif family == "all_current_missing":
        current[:] = np.nan
    return {
        "spec": deepcopy(spec),
        "points_xy": _readonly(points),
        "history": _readonly(history),
        "median8": _readonly(np.median(history, axis=0)),
        "median3": _readonly(np.median(history[-3:], axis=0)),
        "current": _readonly(current),
        "clean_current_background": _readonly(clean_current),
        "injected_contamination": _readonly(contamination),
        "contamination_mask": _readonly(contamination != 0),
    }


def metadata():
    """Return the full generated design, without computed experiment outcomes."""
    return {
        "schema": SCHEMA,
        "case_count": len(specifications()),
        "history_length": HISTORY_LENGTH,
        "point_count": POINT_COUNT,
        "points_xy": [list(point) for point in POINTS_XY],
        "background_formulas": {
            "constant": "96",
            "planar": "96 + 0.05*dx + 0.03*dy",
            "textured": "96 + 0.05*dx + 0.03*dy + 12*sin(dx/12) + 9*cos(dy/15) + 6*sin((dx+dy)/17)",
        },
        "coordinate_definition": "dx=x-64, dy=y-64; y-major then x-major points",
        "point_support": "x,y in 8..120 step8; 40 <= max(abs(dx),abs(dy)) <= 56",
        "noise_formula": "numpy.default_rng(seed).uniform(-level, level, size=(9,144))",
        "noise_profiles": [{"level": level, "seed": seed} for level, seed in NOISE_PROFILES],
        "noise_reused_for_profile_across_all_cases": True,
        "noise_application": "added after analytic transformations; first8 rows history, last row current; then missing masks",
        "temporal_median_semantics": "strict np.median over all8 or last3 rows, not nanmedian",
        "clean_current_background_semantics": "noiseless mathematical current excluding only explicit contamination; remains finite even when observation missing",
        "family_formulas": {
            "stable": "history=base; current=base",
            "offset_plus8": "history=base; current=base+8",
            "offset_minus8": "history=base; current=base-8",
            "gain125": "history=base; current=1.25*base",
            "gain075": "history=base; current=0.75*base",
            "gain_zero": "history=base; current=96",
            "gain_negative": "history=base; current=192-base",
            "plane": "history=base; current=base+4*dx/56-3*dy/56",
            "gain_plane": "history=base; current=1.15*base+8+4*dx/56-3*dy/56",
            "recent_step": "last2priors=base+8, otherpriors=base; current=base+8",
            "pulse_ended": "last2priors=base+8, otherpriors=base; current=base",
            "long_pulse_ended": "all8priors=base+8; current=base",
            "localized_change": "history=base; current=base+16 where x>=64 and y>=64, else base; injected contamination",
            "sparse_bright": "history=base; current=base+40 where index%17==0, else base; injected contamination",
            "sparse_dark": "history=base; current=base-40 where index%17==0, else base; injected contamination",
            "stripe": "history=base; current=base+32 where abs(x-64)<=8, else base; injected contamination",
            "missing_prior": "history=base with firstprior first8points NaN; current=base",
            "missing_current_left": "history=base; current=base with x<64 NaN",
            "missing_current_right": "history=base; current=base with x>=64 NaN",
            "all_current_missing": "history=base; allcurrent NaN",
        },
        "truth_and_labels_allowed_as_fit_inputs": False,
        "analytic_simulation_only": True,
        "physical_sensor_model": False,
        "quantized": False,
        "clipped": False,
        "specifications": specifications(),
    }
