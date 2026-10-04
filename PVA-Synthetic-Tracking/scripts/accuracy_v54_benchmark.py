"""Frozen V54 paired synthetic source-preservation stress-test inputs.

These are analytic arrays, not camera data, calibrated optical/noise models,
airborne labels, independent trials, or detector-accuracy evidence. Scenario
metadata and truth must not enter background fitting or prediction. Source-on
and source-off observations share identical backgrounds, noise and missingness.
"""

from copy import deepcopy
import json

import numpy as np


SCHEMA = "accuracy_v54_generated_benchmark_v1"
TIMES = tuple(range(-8, 1))
HISTORY_LENGTH = 8
CONDITIONS = (
    "stable", "recent_plus8", "recent_minus8", "ended_short_plus8",
    "ended_long_plus8", "local_guard_plus16", "local_guard_minus16",
    "all_guard_plus8",
)
MOTIONS = ("appearing", "stationary", "slow_linear", "linear", "turning", "move_stop")
SOURCE_CONFIGURATIONS = (("absent", 0),) + tuple((motion, amplitude)
    for motion in MOTIONS for amplitude in (4, -4))
BACKGROUNDS = ("constant", "textured")
NOISE_PROFILES = ((0, 71), (0.5, 991))
AVAILABILITY_CASES = ("guard_current_left_missing", "guard_current_all_missing",
                      "core_first_prior_center_missing", "core_current_center_missing")
GUARD_XY = tuple((x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
                 if 40 <= max(abs(x - 64), abs(y - 64)) <= 56)
CORE_XY = tuple((x, y) for y in range(52, 77) for x in range(52, 77))
GUARD_COUNT = len(GUARD_XY)
CORE_COUNT = len(CORE_XY)


def _spec(condition, motion, amplitude, background, level, seed, missingness=None):
    amplitude_label = "0" if amplitude == 0 else ("p4" if amplitude > 0 else "n4")
    noise_label = "0" if level == 0 else "0p5"
    case_id = (f"{condition}_{motion}_a{amplitude_label}_{background}_noise{noise_label}_seed{seed}"
               if missingness is None else "availability_" + missingness)
    return dict(schema=SCHEMA, case_id=case_id,
        stratum="factorial" if missingness is None else "availability",
        condition=condition, motion=motion, amplitude=amplitude, background=background,
        noise_level=level, seed=seed, missingness=missingness,
        history_length=HISTORY_LENGTH, guard_point_count=GUARD_COUNT, core_point_count=CORE_COUNT,
        analytic_simulation_only=True, physical_sensor_model=False)


def specifications():
    """Fresh JSON-safe frozen420-case design; availability cases are a separate stratum."""
    factorial = [_spec(condition, motion, amplitude, background, level, seed)
        for condition in CONDITIONS for motion, amplitude in SOURCE_CONFIGURATIONS
        for background in BACKGROUNDS for level, seed in NOISE_PROFILES]
    return factorial + [_spec("stable", "appearing", 4, "textured", 0, 71, missingness)
                        for missingness in AVAILABILITY_CASES]


def _validated_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("A canonical V54 benchmark specification is required")
    by_id = {entry["case_id"]: entry for entry in specifications()}
    try:
        expected = by_id[spec["case_id"]]
        supplied = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Unknown or malformed V54 specification") from error
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if supplied != canonical:
        raise ValueError("Specification differs from its frozen V54 case")
    return expected


def _background(points, name):
    if name == "constant":
        return np.full(len(points), 96.0)
    dx, dy = points[:, 0].astype(float) - 64, points[:, 1].astype(float) - 64
    return 96.0 + .05 * dx + .03 * dy + 12 * np.sin(dx / 12) + 9 * np.cos(dy / 15) + 6 * np.sin((dx + dy) / 17)


def _profile(points, center):
    cx, cy = center
    return np.maximum(1 - np.abs(points[:, 0] - cx) / 2, 0) * np.maximum(1 - np.abs(points[:, 1] - cy) / 2, 0)


def _center(motion, time):
    if motion == "appearing":
        return (64., 64.) if time == 0 else None
    if motion == "absent":
        return None
    if motion == "stationary":
        return 64., 64.
    if motion == "slow_linear":
        return 64 + .25 * time, 64.
    if motion == "linear":
        return 64 + time, 64.
    if motion == "turning":
        return (64 + time + 4, 60.) if time <= -4 else (64., 64 + time)
    if motion == "move_stop":
        return 64 + min(time + 3, 0), 64.
    raise ValueError("Unknown source motion")


def _readonly(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def generate_case(spec):
    """Generate one canonical observation pair, with truth explicitly separate.

    Every array is readonly and detached. Coordinates are integer y-major grids;
    observed values and truth are float64. Histories are times-8..-1 and current
    is time0. The positive unit ``source_template`` always centers(64,64), even
    for absent cases; absent target retention/sign metrics must remain null.
    Ordinary strict temporal medians are intentionally left to the adapter.
    """
    spec = _validated_spec(spec)
    guard_xy = np.asarray(GUARD_XY, dtype=np.int64)
    core_xy = np.asarray(CORE_XY, dtype=np.int64)
    guard_base = _background(guard_xy, spec["background"])
    core_base = _background(core_xy, spec["background"])
    shift = np.zeros(9, dtype=np.float64)
    condition = spec["condition"]
    if condition in ("recent_plus8", "recent_minus8"):
        shift[-3:] = 8 if condition == "recent_plus8" else -8
    elif condition == "ended_short_plus8":
        shift[-3:-1] = 8
    elif condition == "ended_long_plus8":
        shift[:-1] = 8
    contamination = np.zeros(GUARD_COUNT, dtype=np.float64)
    if condition in ("local_guard_plus16", "local_guard_minus16"):
        mask = (guard_xy[:, 0] >= 64) & (guard_xy[:, 1] >= 64)
        contamination[mask] = 16 if condition == "local_guard_plus16" else -16
    elif condition == "all_guard_plus8":
        contamination[:] = 8
    # Every case of a profile restarts the RNG and consumes guard first, then
    # core. Pairing and shared realizations are deliberate controlled contrasts.
    rng = np.random.default_rng(spec["seed"])
    guard_noise = rng.uniform(-spec["noise_level"], spec["noise_level"], (9, GUARD_COUNT))
    core_noise = rng.uniform(-spec["noise_level"], spec["noise_level"], (9, CORE_COUNT))
    guard = guard_base[None, :] + shift[:, None] + guard_noise
    guard[-1] += contamination
    core_off = core_base[None, :] + shift[:, None] + core_noise
    source = np.zeros((9, CORE_COUNT), dtype=np.float64)
    for index, time in enumerate(TIMES):
        center = _center(spec["motion"], time)
        if center is not None:
            source[index] = spec["amplitude"] * _profile(core_xy, center)
    core_on = core_off + source
    template = _profile(core_xy, (64., 64.))
    missingness = spec["missingness"]
    center_index = CORE_XY.index((64, 64))
    if missingness == "guard_current_left_missing":
        guard[-1, guard_xy[:, 0] < 64] = np.nan
    elif missingness == "guard_current_all_missing":
        guard[-1] = np.nan
    elif missingness == "core_first_prior_center_missing":
        core_on[0, center_index] = core_off[0, center_index] = np.nan
    elif missingness == "core_current_center_missing":
        core_on[-1, center_index] = core_off[-1, center_index] = np.nan
    return dict(spec=deepcopy(spec), guard_xy=_readonly(guard_xy), core_xy=_readonly(core_xy),
        guard_history=_readonly(guard[:-1]), guard_current=_readonly(guard[-1]),
        core_history_on=_readonly(core_on[:-1]), core_history_off=_readonly(core_off[:-1]),
        core_current_on=_readonly(core_on[-1]), core_current_off=_readonly(core_off[-1]),
        clean_current_background=_readonly(core_base + shift[-1]), current_source=_readonly(source[-1]),
        source_template=_readonly(template), guard_contamination_current=_readonly(contamination))


def metadata():
    """Complete analytic design description, without computed performance outcomes."""
    return dict(schema=SCHEMA, case_count=420, factorial_case_count=416,
        availability_case_count=4, history_length=8, times=list(TIMES),
        guard_point_count=GUARD_COUNT, core_point_count=CORE_COUNT,
        guard_xy=[list(point) for point in GUARD_XY], core_xy=[list(point) for point in CORE_XY],
        point_order="y-major then x-major",
        conditions=list(CONDITIONS), source_configurations=[dict(motion=motion, amplitude=amplitude)
            for motion, amplitude in SOURCE_CONFIGURATIONS],
        background_formulas=dict(constant="96", textured="96+.05*dx+.03*dy+12*sin(dx/12)+9*cos(dy/15)+6*sin((dx+dy)/17)"),
        coordinate_definition="dx=x-64,dy=y-64",
        source_profile="max(1-abs(x-cx)/2,0)*max(1-abs(y-cy)/2,0), multiplied by signed amplitude",
        source_centers=dict(appearing="no prior source; time0:(64,64)", stationary="(64,64)",
            slow_linear="(64+.25*t,64)", linear="(64+t,64)",
            turning="t<=-4:(64+t+4,60), otherwise:(64,64+t)", move_stop="(64+min(t+3,0),64)"),
        condition_formulas=dict(stable="no change", recent_plus8="times-2,-1,0 globally+8",
            recent_minus8="times-2,-1,0 globally-8", ended_short_plus8="times-2,-1 globally+8; current unchanged",
            ended_long_plus8="times-8..-1 globally+8; current unchanged",
            local_guard_plus16="current guard x>=64,y>=64:+16; core unchanged",
            local_guard_minus16="current guard x>=64,y>=64:-16; core unchanged",
            all_guard_plus8="all current guard:+8; core unchanged"),
        noise_profiles=[dict(level=level, seed=seed) for level, seed in NOISE_PROFILES],
        noise_formula="default_rng(seed); uniform(-level,level,(9,144)) guard first; uniform(-level,level,(9,625)) core second",
        noise_reused_across_cases_of_profile=True, on_off_noise_identical=True,
        application_order="analytic background shifts, noise, core-on source, then missingness",
        missing_core_masks_identical_on_off=True, truth_remains_finite_when_observations_missing=True,
        current_template_center=[64, 64], absent_has_positive_unit_template_but_no_retention_ratio=True,
        labels_and_truth_allowed_as_fit_inputs=False, analytic_simulation_only=True,
        physical_sensor_model=False, quantized=False, clipped=False, specifications=specifications())
