"""Twenty predeclared synthetic guard-calibration cases; no scoring or media I/O.

Current source amplitude is independent of photometric background gain. Truth
and latent interpretations are never adapter arguments. The regional/twin
controls explicitly violate the guard-to-core brightness-transfer model; an
algorithm cannot discover every such violation from a disjoint guard alone.
"""
from copy import deepcopy

import numpy as np


CASE_IDS = (
    "ordinary_g1", "shared_g08", "shared_g12", "shared_g12_plus_plane",
    "background_only_g12", "dim_source_g12", "dark_source_g12",
    "flat_affine_guard", "near_affine_guard", "prior_only_counterexample_g1",
    "prior_only_counterexample_g2", "persistent_guard_light_shared_gain",
    "independently_blinking_guard_light", "new_current_object_in_guard",
    "broad_current_psf_reaches_guard", "core_only_gain_change",
    "guard_only_gain_change", "guard_core_observational_twin",
    "predeclared_guard_nan", "correlated_guard_error_extremes",
)
TWIN_IDS = ("guard_only_gain_change", "guard_core_observational_twin")
COUNTEREXAMPLE_IDS = ("prior_only_counterexample_g1", "prior_only_counterexample_g2")
ADAPTER_KEYS = ("current129", "history129", "prior_centers_xy", "predicted_offset_xy", "polarity")
PRIOR_TIMES = tuple(range(-8, 0))
GUARD_LIGHT_XY = (112., 64.)


def _specifications():
    specs = []

    def add(case_id, family, **changes):
        p = dict(background_kind="step_sine", gain_core=1., gain_guard=1.,
                 current_affine_plane=[0., 0., 0.], prior_source_peak_dn=30.,
                 current_source_peak_dn=30., prior_source_sigma=1., current_source_sigma=1.,
                 prior_guard_light_peak_dn=0., current_guard_light_peak_dn=0.,
                 current_new_guard_object_peak_dn=0., guard_light_sigma=1.,
                 polarity="bright", current_guard_nan_xy=None,
                 correlated_guard_error=False, latent_global_gain=1.,
                 latent_additive_illumination="none", known_guard_model_violation=False,
                 known_core_model_violation=False, gain_transfer_valid_in_declared_world=True)
        p.update(deepcopy(changes))
        specs.append(dict(case_id=case_id, family=family, generation_parameters=p))

    add("ordinary_g1", "shared_gain")
    add("shared_g08", "shared_gain", gain_core=.8, gain_guard=.8, latent_global_gain=.8)
    add("shared_g12", "shared_gain", gain_core=1.2, gain_guard=1.2, latent_global_gain=1.2)
    add("shared_g12_plus_plane", "shared_gain_affine", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, current_affine_plane=[6., .08, -.04])
    add("background_only_g12", "source_control", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, current_source_peak_dn=0.)
    add("dim_source_g12", "source_control", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, current_source_peak_dn=7.5)
    add("dark_source_g12", "source_control", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, prior_source_peak_dn=-30., current_source_peak_dn=-30., polarity="dark")
    add("flat_affine_guard", "gain_unidentifiable", background_kind="affine_guard_core_step",
        gain_core=1.2, gain_guard=1.2, latent_global_gain=1.2)
    add("near_affine_guard", "gain_unidentifiable", background_kind="near_affine_guard_core_step",
        gain_core=1.2, gain_guard=1.2, latent_global_gain=1.2)
    add("prior_only_counterexample_g1", "past_gain_not_current_gain")
    add("prior_only_counterexample_g2", "past_gain_not_current_gain", gain_core=2., gain_guard=2., latent_global_gain=2.)
    add("persistent_guard_light_shared_gain", "guard_light", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, prior_guard_light_peak_dn=20., current_guard_light_peak_dn=24.)
    add("independently_blinking_guard_light", "guard_contamination", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, prior_guard_light_peak_dn=20., current_guard_light_peak_dn=0.,
        known_guard_model_violation=True)
    add("new_current_object_in_guard", "guard_contamination", current_new_guard_object_peak_dn=20.,
        known_guard_model_violation=True)
    add("broad_current_psf_reaches_guard", "guard_contamination", current_source_sigma=16.,
        known_guard_model_violation=True, known_core_model_violation=True)
    add("core_only_gain_change", "guard_core_transfer_violation", gain_core=1.2, gain_guard=1.,
        current_source_peak_dn=0., latent_global_gain=1.,
        latent_additive_illumination="+0.2*B within Chebyshev radius<=12; zero outside",
        known_core_model_violation=True, gain_transfer_valid_in_declared_world=False)
    add("guard_only_gain_change", "guard_core_transfer_violation", gain_core=1., gain_guard=1.2,
        current_source_peak_dn=0., latent_global_gain=1.,
        latent_additive_illumination="+0.2*B outside Chebyshev radius<=12; zero inside",
        known_core_model_violation=True, gain_transfer_valid_in_declared_world=False)
    add("guard_core_observational_twin", "observational_identity", gain_core=1., gain_guard=1.2,
        current_source_peak_dn=0., latent_global_gain=1.2,
        latent_additive_illumination="-0.2*B within Chebyshev radius<=12; zero outside",
        known_core_model_violation=True, gain_transfer_valid_in_declared_world=False)
    add("predeclared_guard_nan", "missing_guard_support", current_guard_nan_xy=list(GUARD_LIGHT_XY))
    add("correlated_guard_error_extremes", "simultaneous_error_control", gain_core=1.2, gain_guard=1.2,
        latent_global_gain=1.2, correlated_guard_error=True)
    assert tuple(s["case_id"] for s in specs) == CASE_IDS
    return specs


def scenario_manifest():
    """Full deterministic case declaration, independent of any solver outcome."""
    return dict(schema_version=1, case_count=20, synthetic_only=True, real_media_read=False,
        patch_shape=[129, 129], prior_frame_count=8, image_dtype="float64",
        image_domain="unquantized native DN synthetics, not 8-bit camera recordings",
        prior_times=list(PRIOR_TIMES), current_time=0,
        source_trajectory="prior(32+4*i,64), i=0..7 at -8..-1; current(64,64)",
        source_formula="peak_dn*exp(-((x-cx)^2+(y-cy)^2)/(2*sigma^2))",
        source_amplitude_independent_of_background_gain=True,
        base_background_formula="50+20*(x>=64)+8*sin(y/12)",
        affine_guard_background_formula="50+.03*(x-64)-.02*(y-64)+20*(x>=64)*(Chebyshev_radius<=24)",
        near_affine_guard_background_formula="affine_guard_background+.1*sin(y/12)",
        gain_map_formula="gain_core when max(abs(x-64),abs(y-64))<=12; gain_guard otherwise",
        current_formula="gain_map*B+affine_plane+independent_current_source+current_guard_light+new_guard_object+declared_error",
        prior_formula="B+moving_prior_source+persistent_prior_guard_light+declared_error; same B in all eight priors",
        affine_plane_formula="p0+p1*(x-64)+p2*(y-64)",
        guard_light_center_xy=list(GUARD_LIGHT_XY), guard_light_sigma=1.,
        error_formula="in 40<=Chebyshev_radius<=56 only: prior +.5*s, current -.5*s; s=(-1)^(floor(x/8)+floor(y/8)); same error in all eight priors",
        error_contract="simultaneous deterministic +/-0.5DN; no independence or averaging reduction",
        missing_guard_pixel_xy=list(GUARD_LIGHT_XY),
        forecast_policy="test-only last-four prior linear OLS at time0; weights[-.5,0,.5,1] give(64,64); no current truth/image",
        crop_origin_world_xy=[0., 0.], predicted_offset_xy=[0., 0.],
        twin_case_ids=list(TWIN_IDS), identical_prior_current_gain_counterexample_ids=list(COUNTEREXAMPLE_IDS),
        observational_twin_explanations={
            TWIN_IDS[0]: "global gain1 plus coherent outer-region +.2*B illumination",
            TWIN_IDS[1]: "global gain1.2 plus compensating core-local -.2*B illumination"},
        limitations=[
            "The stencil interval is an outer necessary-constraint relaxation, not full affine feasibility or verified background validity",
            "Current guard observations make calibration target-core-blind, not prior-only",
            "Past gains cannot certify arbitrary current gain without a temporal-change assumption",
            "Global gain plus local illumination is not identifiable from these observational twins",
            "Neither clean-guard validity nor guard-to-core transfer is proved by nonempty calibration",
            "The fixed 8-pixel grid can miss between-grid contamination",
            "Horizontal/vertical second differences annihilate an xy cross-term even though xy is not an affine plane",
            "Broad PSF, spatially different gains and unknown contaminants are outside the compact shared-photometric model",
            "Physical motion and airborne class remain unknown, including on valid numerical source evidence",
        ], scenarios=_specifications())


def _background(kind, xx, yy):
    radius = np.maximum(np.abs(xx-64.), np.abs(yy-64.))
    if kind == "step_sine":
        return 50.+20.*(xx >= 64.)+8.*np.sin(yy/12.)
    if kind in ("affine_guard_core_step", "near_affine_guard_core_step"):
        result = 50.+.03*(xx-64.)-.02*(yy-64.)+20.*(xx >= 64.)*(radius <= 24.)
        if kind == "near_affine_guard_core_step":
            result = result+.1*np.sin(yy/12.)
        return result
    raise ValueError("Unknown predeclared synthetic background")


def _point(xx, yy, center, peak, sigma):
    return peak*np.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2.*sigma*sigma))


def build_cases():
    """Build twenty arrays-only adapter calls plus separate truth/provenance."""
    yy, xx = np.indices((129, 129), dtype=np.float64)
    radius = np.maximum(np.abs(xx-64.), np.abs(yy-64.))
    guard_band = (radius >= 40.) & (radius <= 56.)
    checkerboard = np.where((np.floor(xx/8.)+np.floor(yy/8.)) % 2 == 0, 1., -1.)
    prior_centers = [[32.+4.*i, 64.] for i in range(8)]
    cases = []
    for specification in _specifications():
        case_id, p = specification["case_id"], specification["generation_parameters"]
        background = _background(p["background_kind"], xx, yy)
        prior_light = _point(xx, yy, GUARD_LIGHT_XY, p["prior_guard_light_peak_dn"], p["guard_light_sigma"])
        history = np.stack([background+_point(xx, yy, center, p["prior_source_peak_dn"],
                           p["prior_source_sigma"])+prior_light for center in prior_centers])
        gain = np.where(radius <= 12., p["gain_core"], p["gain_guard"])
        a0, ax, ay = p["current_affine_plane"]
        current = gain*background+a0+ax*(xx-64.)+ay*(yy-64.)
        current += _point(xx, yy, (64., 64.), p["current_source_peak_dn"], p["current_source_sigma"])
        current += _point(xx, yy, GUARD_LIGHT_XY, p["current_guard_light_peak_dn"], p["guard_light_sigma"])
        current += _point(xx, yy, GUARD_LIGHT_XY, p["current_new_guard_object_peak_dn"], 1.)
        if p["correlated_guard_error"]:
            error = .5*checkerboard*guard_band
            history += error[None, :, :]
            current -= error
        if p["current_guard_nan_xy"] is not None:
            x, y = p["current_guard_nan_xy"]
            current[int(y), int(x)] = np.nan
        model_valid = not (p["known_guard_model_violation"] or p["known_core_model_violation"])
        truth = dict(background_kind=p["background_kind"],
            effective_background_gain_guard=p["gain_guard"], effective_background_gain_core=p["gain_core"],
            latent_global_photometric_gain=p["latent_global_gain"],
            latent_additive_illumination=p["latent_additive_illumination"],
            current_affine_plane=list(p["current_affine_plane"]),
            prior_source_peak_dn=p["prior_source_peak_dn"], current_source_peak_dn=p["current_source_peak_dn"],
            current_source_present=p["current_source_peak_dn"] != 0,
            prior_source_sigma=p["prior_source_sigma"], current_source_sigma=p["current_source_sigma"],
            prior_source_centers_xy=deepcopy(prior_centers), current_source_center_xy=[64., 64.],
            prior_guard_light_peak_dn=p["prior_guard_light_peak_dn"], current_guard_light_peak_dn=p["current_guard_light_peak_dn"],
            current_new_guard_object_peak_dn=p["current_new_guard_object_peak_dn"],
            guard_light_center_xy=list(GUARD_LIGHT_XY), guard_light_sigma=p["guard_light_sigma"],
            known_guard_model_violation=p["known_guard_model_violation"],
            known_core_model_violation=p["known_core_model_violation"],
            gain_transfer_valid_in_declared_world=p["gain_transfer_valid_in_declared_world"],
            shared_gain_affine_background_model_valid_in_declared_world=model_valid,
            guard_validity_or_transfer_certified_from_observations=False,
            correlated_guard_error=p["correlated_guard_error"],
            prior_error_max_abs_dn=.5 if p["correlated_guard_error"] else 0.,
            current_error_max_abs_dn=.5 if p["correlated_guard_error"] else 0.,
            current_guard_nan_xy=deepcopy(p["current_guard_nan_xy"]),
            observed_arrays_cannot_distinguish_latent_explanation=case_id in TWIN_IDS,
            truth_is_not_an_adapter_argument=True, physical_motion_and_class_certified=False)
        provenance = dict(synthetic_only=True, real_media_read=False, prior_times=list(PRIOR_TIMES), current_time=0,
            forecast_uses_current_image=False, forecast_uses_current_truth=False,
            crop_uses_current_image=False, crop_uses_current_truth=False,
            reported_prior_world_centers_xy=deepcopy(prior_centers), predicted_world_xy=[64., 64.],
            integer_crop_center_world_xy=[64, 64], crop_origin_world_xy=[0., 0.],
            ols_history_indices=[4, 5, 6, 7], ols_times=[-4, -3, -2, -1], ols_weights=[-.5, 0., .5, 1.],
            forecast_is_synthetic_linear_test_not_real_quadratic_tracker=True,
            source_template_supplied=False, image_domain="unquantized float64 native DN",
            generation_parameters=deepcopy(p),
            intended_guard_geometry="annulus40..56 on grid8..120 step8; actual eligibility determined only by frozen prior data",
            guard_selection_uses_current_values=False,
            guard_calibration_may_use_current_guard_values=True,
            current_source_amplitude_independent_of_background_gain=True)
        adapter = dict(current129=current, history129=history, prior_centers_xy=deepcopy(prior_centers),
                       predicted_offset_xy=[0., 0.], polarity=p["polarity"])
        cases.append(dict(case_id=case_id, family=specification["family"], adapter_inputs=adapter,
                          generator_truth=truth, provenance=provenance))
    return cases
