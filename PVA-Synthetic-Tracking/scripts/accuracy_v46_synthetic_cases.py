"""Predeclared prior-image stress cases; no solver, media, or oracle templates.

Only reported prior measurements determine the forecast and crop. Independent
world truth renders image observations into that crop and remains separate from
the adapter's five input fields. These are floating-point native-DN-domain
synthetics, not sensor-quantized 8-bit recordings or calibrated noise models.
"""
from copy import deepcopy

import numpy as np


ADAPTER_KEYS = ("current129", "history129", "prior_centers_xy",
                "predicted_offset_xy", "polarity")
PRIOR_TIMES = tuple(range(-8, 0))
JITTER_PATTERN = ((1, 0), (-1, 1), (0, -1), (1, 1),
                  (-1, 0), (0, 1), (1, -1), (-1, -1))
CURRENT_ONLY_GROUP = (
    "ordinary_prior_forecast", "current_absent", "current_departure_halfpx",
    "current_departure_twopx", "current_departure_sixpx", "current_psf_wide",
    "current_psf_elliptic", "current_dim_quarter", "observational_twin_moving",
    "observational_twin_sequential_fixed",
)
TWIN_IDS = ("observational_twin_moving", "observational_twin_sequential_fixed")


def _specifications():
    """Declared order and parameters, never selected using an observed score."""
    specs = []

    def add(case_id, family, *, mismatch=(), **changes):
        parameters = dict(
            prior_measurement_bias_xy=[0., 0.], prior_jitter_scale=0.,
            current_departure_xy=[0., 0.], missing_measurement_indices=[],
            measurement_order="chronological", incorrect_measurement_offsets={},
            prior_peak_dn=[30.]*8, current_peak_dn=30.,
            prior_psf_sigma_xy=[[1., 1.] for _ in range(8)],
            prior_psf_angle_radians=[0.]*8,
            current_psf_sigma_xy=[1., 1.], current_psf_angle_radians=0.,
            fixed_emitter=None, identity_world="single_moving_source")
        parameters.update(deepcopy(changes))
        specs.append(dict(case_id=case_id, family=family, generation_parameters=parameters,
                          deliberate_model_mismatch_axes=list(mismatch)))

    add("ordinary_prior_forecast", "reference")
    add("current_absent", "current_source_control", current_peak_dn=0.,
        mismatch=("source_disappears",))
    for label, amount in (("halfpx", .5), ("twopx", 2.)):
        add("prior_bias_"+label, "measurement_bias", prior_measurement_bias_xy=[amount, 0.],
            mismatch=("incorrect_prior_positions", "forecast_position_error"))
    for label, amount in (("halfpx", .5), ("twopx", 2.)):
        add("prior_jitter_"+label, "measurement_jitter", prior_jitter_scale=amount,
            mismatch=("incorrect_prior_positions", "forecast_position_error"))
    for label, amount in (("halfpx", .5), ("twopx", 2.), ("sixpx", 6.)):
        add("current_departure_"+label, "forecast_departure", current_departure_xy=[amount, 0.],
            mismatch=("forecast_position_error",))
    add("combined_measurement_and_current_error", "combined_geometry",
        prior_measurement_bias_xy=[1., 0.], prior_jitter_scale=.5,
        current_departure_xy=[2., 1.],
        mismatch=("incorrect_prior_positions", "forecast_position_error"))
    add("current_psf_wide", "psf_change", current_psf_sigma_xy=[1.5, 1.5],
        mismatch=("current_psf_change",))
    add("current_psf_elliptic", "psf_change", current_psf_sigma_xy=[2., .75],
        current_psf_angle_radians=float(np.pi/4), mismatch=("current_psf_change",))
    add("psf_broadens_over_time", "psf_change",
        prior_psf_sigma_xy=[[.75+.75*i/8]*2 for i in range(8)],
        current_psf_sigma_xy=[1.5, 1.5], mismatch=("historical_and_current_psf_change",))
    add("current_dim_quarter", "brightness_change", current_peak_dn=7.5,
        mismatch=("current_amplitude_change",))
    add("brightness_ramp", "brightness_change", prior_peak_dn=[15.+30*i/8 for i in range(8)],
        current_peak_dn=45., mismatch=("historical_and_current_amplitude_change",))
    add("intermittent_source_history", "brightness_change",
        prior_peak_dn=[30.*a for a in (1, 0, 1, .5, 1, 0, .5, 1)],
        mismatch=("historical_amplitude_change", "reported_measurements_on_invisible_frames"))
    add("missing_three_measurements", "measurement_history", missing_measurement_indices=[0, 3, 6],
        mismatch=("missing_prior_measurements",))
    add("missing_four_measurements", "measurement_history", missing_measurement_indices=[0, 2, 4, 6],
        mismatch=("missing_prior_measurements", "below_adapter_five_measurement_minimum"))
    add("incorrect_last_two_measurements", "measurement_history",
        incorrect_measurement_offsets={"6": [0., 12.], "7": [0., 12.]},
        mismatch=("incorrect_prior_positions", "forecast_position_error"))
    add("reversed_measurement_order", "measurement_history", measurement_order="reversed",
        mismatch=("incorrect_prior_frame_associations", "forecast_position_error"))

    def fixed(center, blink=False):
        return dict(center_world_xy=list(center),
                    prior_peak_dn=[30.*a for a in ((1, 0, 1, 0, 1, 0, 1, 0) if blink else (1,)*8)],
                    current_peak_dn=30., psf_sigma_xy=[1., 1.], psf_angle_radians=0.)

    add("persistent_fixed_at_forecast", "fixed_confuser", current_peak_dn=0.,
        fixed_emitter=fixed((64., 64.)), mismatch=("source_disappears", "fixed_light_confuser"))
    add("blinking_fixed_at_forecast", "fixed_confuser", current_peak_dn=0.,
        fixed_emitter=fixed((64., 64.), True), mismatch=("source_disappears", "fixed_light_confuser"))
    add("persistent_fixed_near_forecast", "fixed_confuser", current_peak_dn=0.,
        fixed_emitter=fixed((66., 64.)), mismatch=("source_disappears", "fixed_light_confuser"))
    add("blinking_fixed_near_forecast", "fixed_confuser", current_peak_dn=0.,
        fixed_emitter=fixed((68., 64.), True), mismatch=("source_disappears", "fixed_light_confuser"))
    add("moving_plus_fixed_at_forecast", "mixed_source_fixed", fixed_emitter=fixed((64., 64.)),
        mismatch=("fixed_light_confuser", "coincident_current_emitters"))
    add("moving_plus_fixed_near_forecast", "mixed_source_fixed", fixed_emitter=fixed((68., 64.)),
        mismatch=("fixed_light_confuser",))
    add(TWIN_IDS[0], "observational_identity", identity_world="single_moving_source",
        mismatch=("physical_identity_not_identifiable_from_observations",))
    add(TWIN_IDS[1], "observational_identity", identity_world="sequential_fixed_emitters",
        mismatch=("physical_identity_not_identifiable_from_observations", "incorrect_same_id_identity"))
    return specs


def scenario_manifest():
    """JSON-safe specification; no expected numerical signs or scoring results."""
    return dict(
        schema_version=1, synthetic_only=True, real_media_read=False,
        case_count=28, prior_frame_count=8, patch_shape=[129, 129],
        image_dtype="float64", image_units="native DN domain, not sensor-quantized",
        prior_times=list(PRIOR_TIMES), current_time=0,
        true_prior_world_formula="c_i=(32+4*i,64), i=0..7 at times -8..-1",
        true_current_world_formula="(64,64)+independently declared current_departure_xy",
        base_image_formula="50 + 20*(world_x>=64) + 8*sin(world_y/12)",
        point_formula="A*exp(-0.5*(u^2/sigma_x^2+v^2/sigma_y^2)); u=cos(theta)*dx+sin(theta)*dy; v=-sin(theta)*dx+cos(theta)*dy",
        forecast_policy="OLS intercept at time0 using last four available reported prior measurements at original times; no current inputs",
        crop_policy="integer center=floor(predicted_world_xy+0.5); world origin=center-(64,64); pixel world=origin+local index",
        adapter_offset_policy="predicted_world_xy-integer_crop_center_world_xy",
        jitter_pattern_xy=[list(p) for p in JITTER_PATTERN],
        missing_measurement_policy="None removes only a reported measurement, never its image or original timestamp",
        current_only_same_history_geometry_group=list(CURRENT_ONLY_GROUP),
        prior_equivalence_groups={"base_prior_world": list(CURRENT_ONLY_GROUP)},
        observational_twins=list(TWIN_IDS),
        observational_twin_reference="ordinary_prior_forecast",
        uncertainty_contract="Immutable V45 +/-0.5 DN response/prior-image error contract; no new noise or location budget",
        limitations=[
            "No current truth, current target localization, or oracle source template is passed to the adapter",
            "No camera motion, image quantization, detector, association, or complete tracking stack is simulated",
            "Measurement, crop/forecast, trajectory and PSF errors are not covered by fixed-geometry V45 image-error bounds",
            "A fitted source sign is not a physical motion, airborne, identity, or detection decision",
            "Intermittent invisible frames intentionally retain imperfect reported measurements",
            "Observational twins differ only in latent identity; their inputs must produce identical evidence",
            "Legacy V45 comparators are supplied separately by the runner, not duplicated here",
        ],
        scenarios=_specifications())


def _forecast_geometry(reported_world):
    """Forecast and crop depend only on the eight reported prior measurements."""
    available = [i for i, point in enumerate(reported_world) if point is not None]
    if len(available) < 4:
        raise ValueError("The predeclared generator needs four prior measurements to forecast")
    used = available[-4:]
    times = np.asarray([PRIOR_TIMES[i] for i in used], dtype=float)
    points = np.asarray([reported_world[i] for i in used], dtype=float)
    delta = times-times.mean()
    denominator = float(delta@delta)
    weights = np.full(4, .25)-times.mean()*delta/denominator
    forecast = points.mean(axis=0)-times.mean()*(delta@points/denominator)
    center = np.floor(forecast+.5).astype(np.int64)
    origin = center-64
    return dict(ols_history_indices=used, ols_times=times.tolist(), ols_weights=weights.tolist(),
                predicted_world_xy=forecast.tolist(), integer_crop_center_world_xy=center.tolist(),
                crop_origin_world_xy=origin.tolist(), predicted_offset_xy=(forecast-center).tolist(),
                local_reported_prior_centers_xy=[None if p is None else (np.asarray(p)-origin).tolist()
                                                for p in reported_world])


def _point(xx, yy, center, amplitude, sigma, angle):
    dx, dy = xx-center[0], yy-center[1]
    u = np.cos(angle)*dx+np.sin(angle)*dy
    v = -np.sin(angle)*dx+np.cos(angle)*dy
    return amplitude*np.exp(-.5*((u/sigma[0])**2+(v/sigma[1])**2))


def build_cases():
    """Build all 28 cases, with adapter inputs separate from generator truth."""
    cases = []
    true_prior = np.asarray([[32.+4*i, 64.] for i in range(8)])
    local_y, local_x = np.indices((129, 129))
    for spec in _specifications():
        p = spec["generation_parameters"]
        reported = true_prior.copy()
        if p["measurement_order"] == "reversed":
            reported = reported[::-1].copy()
        reported += np.asarray(p["prior_measurement_bias_xy"])
        reported += p["prior_jitter_scale"]*np.asarray(JITTER_PATTERN)
        for index, shift in p["incorrect_measurement_offsets"].items():
            reported[int(index)] += shift
        reported_list = [None if i in p["missing_measurement_indices"] else position.tolist()
                         for i, position in enumerate(reported)]
        geometry = _forecast_geometry(reported_list)
        origin = geometry["crop_origin_world_xy"]
        xx, yy = local_x+origin[0], local_y+origin[1]
        background = 50+20*(xx >= 64)+8*np.sin(yy/12)
        true_current = np.asarray([64., 64.])+p["current_departure_xy"]
        history = np.stack([
            background+_point(xx, yy, true_prior[i], p["prior_peak_dn"][i],
                              p["prior_psf_sigma_xy"][i], p["prior_psf_angle_radians"][i])
            for i in range(8)])
        current = background+_point(xx, yy, true_current, p["current_peak_dn"],
                                    p["current_psf_sigma_xy"], p["current_psf_angle_radians"])
        fixed = p["fixed_emitter"]
        if fixed is not None:
            history += np.stack([
                _point(xx, yy, fixed["center_world_xy"], amplitude,
                       fixed["psf_sigma_xy"], fixed["psf_angle_radians"])
                for amplitude in fixed["prior_peak_dn"]])
            current += _point(xx, yy, fixed["center_world_xy"], fixed["current_peak_dn"],
                              fixed["psf_sigma_xy"], fixed["psf_angle_radians"])
        sequential = p["identity_world"] == "sequential_fixed_emitters"
        moving_current = p["current_peak_dn"] > 0 and not sequential
        fixed_current = bool((fixed is not None and fixed["current_peak_dn"] > 0) or
                             (sequential and p["current_peak_dn"] > 0))
        ambiguity = fixed is not None or spec["family"] == "observational_identity"
        adapter_inputs = dict(current129=current, history129=history,
                              prior_centers_xy=deepcopy(geometry["local_reported_prior_centers_xy"]),
                              predicted_offset_xy=list(geometry["predicted_offset_xy"]), polarity="bright")
        truth = dict(
            apparent_point_prior_world_centers_xy=true_prior.tolist(),
            apparent_point_current_world_center_xy=true_current.tolist(),
            prior_peak_dn=list(p["prior_peak_dn"]), current_peak_dn=p["current_peak_dn"],
            prior_psf_sigma_xy=deepcopy(p["prior_psf_sigma_xy"]),
            prior_psf_angle_radians=list(p["prior_psf_angle_radians"]),
            current_psf_sigma_xy=list(p["current_psf_sigma_xy"]),
            current_psf_angle_radians=p["current_psf_angle_radians"],
            latent_world_interpretation=p["identity_world"], fixed_emitter=deepcopy(fixed),
            current_moving_source_present=bool(moving_current),
            current_fixed_emitter_present=fixed_current,
            any_current_emitter_present=bool(moving_current or fixed_current),
            physical_identity_ambiguous=ambiguity, physical_identity_certified_from_images=False,
            deliberate_model_mismatch_axes=list(spec["deliberate_model_mismatch_axes"]),
            v45_fixed_geometry_bound_covers_deliberate_mismatch=False)
        provenance = dict(
            synthetic_only=True, real_media_read=False, prior_times=list(PRIOR_TIMES), current_time=0,
            reported_prior_world_centers_xy=deepcopy(reported_list), **geometry,
            forecast_uses_current_image=False, forecast_uses_current_truth=False,
            crop_uses_current_image=False, crop_uses_current_truth=False,
            prior_template_source="prior images and supplied prior measurements only; generator supplies no template",
            prior_equivalence_group=("base_prior_world" if spec["case_id"] in CURRENT_ONLY_GROUP
                                     else "unique_"+spec["case_id"]),
            current_only_same_history_geometry_group=("ordinary_current_variants"
                if spec["case_id"] in CURRENT_ONLY_GROUP else None),
            forecast_is_synthetic_linear_test_not_real_quadratic_tracker=True,
            generation_parameters=deepcopy(p), image_domain="float64 native DN domain, not sensor-quantized")
        cases.append(dict(case_id=spec["case_id"], family=spec["family"],
                          adapter_inputs=adapter_inputs, generator_truth=truth, provenance=provenance))
    return cases
