"""Frozen, generated-only temporal examples for the V51 background diagnostic.

These analytic simulations are not camera data, a physical sensor/noise model,
object labels, or evidence of detection accuracy. The scenario labels, signal
truth, and response values must not be supplied to the prior-only predictor.
Every noise profile reuses the same complete noise realization across all its
families and amplitudes. This deliberately creates indistinguishable prior
histories for step and each pulse at its first post-pulse response, with
different response values.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np


FRAME_COUNT = 64
HISTORY_LENGTH = 8
ONSET = 20
AMPLITUDES = (-32, -8, 8, 32)
NOISE_PROFILES = ((0, 71), (1, 991))
FAMILIES = (
    "step", "ramp", "pulse1", "pulse2", "pulse4", "pulse8", "pulse12",
    "localized_step", "localized_pulse2", "moving_stripe",
    "registration_shift", "missing_window",
)
POINTS_XY = tuple(
    (x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
    if 40 <= max(abs(x - 64), abs(y - 64)) <= 56
)
POINT_COUNT = len(POINTS_XY)
SCHEMA = "accuracy_v51_generated_benchmark_v1"


def _case_id(family, amplitude, noise_level, seed):
    amplitude_label = f"neg{abs(amplitude)}" if amplitude < 0 else f"pos{amplitude}"
    if family == "stable":
        return f"stable_noise{noise_level}_seed{seed}"
    return f"{family}_a_{amplitude_label}_noise{noise_level}_seed{seed}"


def _event_stop(family):
    if family == "stable":
        return None
    if family.startswith("pulse"):
        return ONSET + int(family.removeprefix("pulse"))
    if family == "localized_pulse2":
        return ONSET + 2
    if family in ("moving_stripe", "registration_shift"):
        return 44
    return FRAME_COUNT


def _spec(family, amplitude, noise_level, seed):
    onset = None if family == "stable" else ONSET
    return {
        "schema": SCHEMA,
        "case_id": _case_id(family, amplitude, noise_level, seed),
        "family": family,
        "amplitude": amplitude,
        "noise_level": noise_level,
        "seed": seed,
        "frame_count": FRAME_COUNT,
        "point_count": POINT_COUNT,
        "history_length": HISTORY_LENGTH,
        "response_frame_indices": list(range(HISTORY_LENGTH, FRAME_COUNT)),
        "event_window": {"start_inclusive": onset,
                         "stop_exclusive": _event_stop(family)},
        "missing_window": {"start_inclusive": 18, "stop_exclusive": 21,
                           "point_indices": list(range(8))}
                          if family == "missing_window" else None,
        "analytic_simulation_only": True,
        "physical_sensor_model": False,
    }


def specifications():
    """Return fresh JSON-safe metadata for all 98 predeclared cases."""
    stable = [_spec("stable", 0, level, seed) for level, seed in NOISE_PROFILES]
    varied = [_spec(family, amplitude, level, seed)
              for family in FAMILIES for amplitude in AMPLITUDES
              for level, seed in NOISE_PROFILES]
    return stable + varied


def _validated_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("A specification must be a canonical benchmark dictionary")
    candidates = {item["case_id"]: item for item in specifications()}
    try:
        canonical = candidates[spec["case_id"]]
        supplied_json = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Unknown or malformed benchmark specification") from error
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if supplied_json != canonical_json:
        raise ValueError("The supplied specification differs from its frozen case")
    return canonical


def _readonly(array):
    array.setflags(write=False)
    return array


def generate_case(spec):
    """Generate one canonical case; metadata/truth are diagnostic outputs only.

    ``values`` is float64 (64, 144), with no quantization or clipping. Predictors
    must receive only values[t-8:t], not this dictionary, labels, or values[t].
    event_active marks scheduled support, including zero ramp/shift positions;
    affected_points_mask marks scheduled spatial support, not nonzero change.
    response_phase and history_has_prior_event are defined for all 64 indices,
    but evaluation is restricted to response indices 8..63.
    """
    spec = _validated_spec(spec)
    points = np.asarray(POINTS_XY, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    base = 96.0 + 0.05 * (x - 64.0) + 0.03 * (y - 64.0)
    family, amplitude = spec["family"], spec["amplitude"]
    event_active = np.zeros(FRAME_COUNT, dtype=np.bool_)
    phase = np.full(FRAME_COUNT, "baseline", dtype="<U10")
    delta = np.zeros((FRAME_COUNT, POINT_COUNT), dtype=np.float64)
    affected = np.zeros(delta.shape, dtype=np.bool_)
    if family != "stable":
        stop = spec["event_window"]["stop_exclusive"]
        event_active[ONSET:stop] = True
        phase[ONSET] = "onset"
        phase[ONSET + 1:stop] = "event"
        phase[stop:] = "post_event"
        quadrant = (x >= 64) & (y >= 64)
        for frame in range(ONSET, stop):
            mask = quadrant if family.startswith("localized_") else np.ones(POINT_COUNT, dtype=bool)
            if family == "moving_stripe":
                center_x = 8 + ((frame - ONSET) % 15) * 8
                mask = np.abs(x - center_x) <= 8
            affected[frame] = mask
            if family == "ramp":
                delta[frame, mask] = amplitude * min((frame - ONSET) / 16.0, 1.0)
            elif family == "registration_shift":
                shift = (0, 1, -1, 2, -2, 1, 0, -1)[(frame - ONSET) % 8]
                delta[frame] = amplitude * (np.sin((x - 64 + shift) / 8.0)
                                             - np.sin((x - 64) / 8.0))
            else:
                delta[frame, mask] = amplitude
    # Re-seeding every case is intentional: equal profiles have equal noise,
    # including stable, every family, and both signs/all amplitudes.
    noise = np.random.default_rng(spec["seed"]).uniform(
        -spec["noise_level"], spec["noise_level"], size=delta.shape)
    values = base[None, :] + delta + noise
    missing = np.zeros(delta.shape, dtype=np.bool_)
    if family == "missing_window":
        missing[18:21, :8] = True
        values[missing] = np.nan
    history_has_event = np.array([
        event_active[max(0, frame - HISTORY_LENGTH):frame].any()
        for frame in range(FRAME_COUNT)], dtype=np.bool_)
    return {
        "spec": deepcopy(spec),
        "values": _readonly(values),
        "points_xy": _readonly(np.asarray(POINTS_XY, dtype=np.int64)),
        "base": _readonly(base),
        "signal_delta": _readonly(delta),
        "event_active": _readonly(event_active),
        "response_phase": _readonly(phase),
        "history_has_prior_event": _readonly(history_has_event),
        "affected_points_mask": _readonly(affected),
        "missing_mask": _readonly(missing),
    }


def case_content_hash(case):
    """Bind canonical metadata, arrays, types, and shapes in fixed field order."""
    digest = hashlib.sha256()
    spec = _validated_spec(case["spec"])
    digest.update(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in ("values", "points_xy", "base", "signal_delta", "event_active",
                 "response_phase", "history_has_prior_event", "affected_points_mask",
                 "missing_mask"):
        array = np.asarray(case[name])
        digest.update(name.encode("ascii"))
        digest.update(json.dumps({"dtype": array.dtype.str,
                                  "shape": list(array.shape)}, sort_keys=True).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def benchmark_metadata():
    """Describe the fixed design, including all 40 causal ambiguity twins."""
    return {
        "schema": SCHEMA,
        "case_count": len(specifications()),
        "frame_count": FRAME_COUNT,
        "point_count": POINT_COUNT,
        "points_xy": [list(point) for point in POINTS_XY],
        "history_length": HISTORY_LENGTH,
        "response_frame_indices": list(range(HISTORY_LENGTH, FRAME_COUNT)),
        "base_formula": "96 + 0.05*(x-64) + 0.03*(y-64)",
        "noise_formula": "numpy.default_rng(seed).uniform(-level, level, size=(64,144))",
        "noise_reused_for_profile_across_all_cases": True,
        "ramp_formula": "A*min((t-20)/16,1) for t>=20, else 0; plateau starts t=36",
        "localized_support_formula": "x>=64 and y>=64, inclusive boundaries",
        "moving_stripe_formula": "20<=t<44; center_x=8+((t-20)%15)*8; abs(x-center_x)<=8",
        "registration_shift_formula": "20<=t<44; shift=[0,1,-1,2,-2,1,0,-1][(t-20)%8]; A*(sin((x-64+shift)/8)-sin((x-64)/8))",
        "event_active_semantics": "scheduled temporal support, including zero ramp/shift positions",
        "affected_points_mask_semantics": "scheduled spatial support, not nonzero signal change",
        "response_phase_semantics": "baseline before onset; onset at t=20; event later inside scheduled support; post_event afterward; stable always baseline",
        "history_has_prior_event_semantics": "any(event_active[max(0,t-8):t]); evaluate only t>=8",
        "analytic_simulation_only": True,
        "physical_sensor_model": False,
        "quantized": False,
        "clipped": False,
        "twin_pairs": [{
            "step_case_id": _case_id("step", amplitude, level, seed),
            "pulse_case_id": _case_id(f"pulse{duration}", amplitude, level, seed),
            "duration": duration,
            "amplitude": amplitude,
            "noise_level": level,
            "seed": seed,
            "history_start_inclusive": ONSET + duration - HISTORY_LENGTH,
            "history_stop_exclusive": ONSET + duration,
            "response_frame": ONSET + duration,
            "identical_prior_values": True,
            "different_response_values": True,
            "response_step_minus_pulse_analytic": amplitude,
        } for duration in (1, 2, 4, 8, 12)
          for amplitude in AMPLITUDES for level, seed in NOISE_PROFILES],
        "specifications": specifications(),
    }
