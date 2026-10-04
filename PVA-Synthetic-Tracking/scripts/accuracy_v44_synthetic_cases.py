"""Predeclared V44 oracle-vector controls, with no media or result dependencies.

These arrays test a conditional errors-in-variables contrast calculation, not
source learning or motion detection.  In particular, the image-space source and
fixed-light template directions are supplied by the synthetic generator.  The
response is never used to learn a direction.  Integration flags are declarations
of external information, not a claim that an image algorithm inferred it.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np


IMAGE_SHAPE = (25, 25)
SOURCE_AMPLITUDE = 30.0
RESPONSE_BOUND_DN = 0.5
BACKGROUND_DESIGN_BOUND_DN = 0.5
TEMPLATE_RELATIVE_BOUND = 0.02
SOURCE_TEMPLATE_ORIGIN = "oracle_solver_unit_fixture"

# Roles and perturbations are declared before running the contrast core.  There
# is deliberately no expected score, decision threshold, or availability label.
_SCENARIOS = (
    ("ordinary_moving", "positive_source", "linear_motion", ()),
    ("background_only", "zero_source", "no_source", ()),
    ("uniform_brightness_change", "positive_source_affine_shift", "linear_motion", ()),
    ("plane_brightness_change", "positive_source_affine_shift", "linear_motion", ()),
    ("fixed_flicker_separate", "zero_source_represented_fixed_flicker", "fixed_flicker", ()),
    ("moving_with_separate_fixed_flicker", "positive_source_represented_fixed_flicker", "linear_motion", ()),
    ("fixed_source_exact_overlap", "source_in_nuisance_span", "unresolved_overlap",
     ("source_and_fixed_template_indistinguishable",)),
    ("raw_normalization_unsupported_metadata", "positive_source_oracle_direction_placeholder", "linear_motion",
     ("raw_template_normalization_unsupported_external_metadata",)),
    ("uncertainty_contains_zero", "source_error_set_contains_zero", "linear_motion", ()),
    ("short_history", "positive_source_incomplete_external_history", "linear_motion",
     ("insufficient_external_history",)),
    ("unknown_nuisance_support", "positive_source_incomplete_external_dictionary", "linear_motion",
     ("external_nuisance_support_unknown",)),
    ("slow_motion", "positive_source_temporal_identity_unresolved", "slow_motion",
     ("slow_motion_identity_unresolved_external_metadata",)),
    ("hovering", "positive_source_temporal_identity_unresolved", "hovering",
     ("hovering_source_or_fixed_light_unresolved",)),
    ("curved_accelerating_motion", "positive_source", "curved_accelerating_motion", ()),
    ("identifiability_moving_world", "observationally_identical_physical_interpretations", "identifiability_twins",
     ("sequential_fixed_emitter_alternative_unresolved",)),
    ("identifiability_fixed_emitter_world", "observationally_identical_physical_interpretations", "identifiability_twins",
     ("sequential_fixed_emitter_alternative_unresolved",)),
)


def _grid():
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    return xx - 12.0, yy - 12.0


def _gaussian(xx, yy, center_xy):
    cx, cy = center_xy
    return np.exp(-((xx-cx)**2+(yy-cy)**2)/2.0)


def _unit_direction(xx, yy, center_xy):
    stamp = _gaussian(xx, yy, center_xy)
    return stamp / np.linalg.norm(stamp)


def _positions(temporal_role, history_count=8):
    times = np.arange(-history_count, 1, dtype=np.float64)
    if temporal_role == "curved_accelerating_motion":
        positions = np.column_stack((1.4*times+0.06*times**2, 0.07*times**2))
    elif temporal_role == "slow_motion":
        positions = np.column_stack((0.05*times, np.zeros_like(times)))
    elif temporal_role == "hovering":
        positions = np.zeros((len(times), 2), dtype=np.float64)
    else:
        positions = np.column_stack((1.25*times, np.zeros_like(times)))
    return times, positions


def scenario_manifest():
    """Return a fresh JSON-safe declaration; does not build or score any case."""
    scenarios = []
    for case_id, role, temporal_role, unknown_reasons in _SCENARIOS:
        scenarios.append({
            "case_id": case_id, "numerical_role": role,
            "temporal_role": temporal_role,
            "external_integration_unknown_reasons": list(unknown_reasons),
            "source_template_origin": SOURCE_TEMPLATE_ORIGIN,
        })
    return {
        "schema": "seaqr.accuracy-v44-oracle-synthetic-scenario-manifest.v1",
        "synthetic_only": True,
        "uses_real_media_journals_or_scores": False,
        "source_template_origin": SOURCE_TEMPLATE_ORIGIN,
        "image_shape": list(IMAGE_SHAPE),
        "array_interface": ["y", "Z", "m", "P", "response_bound", "nuisance_bound", "source_bound"],
        "model": "y = P @ plane_coefficients + Z @ nuisance_coefficients + source_amplitude * m",
        "array_order": "C-order row-major flattening of the complete 25x25 patch; no support selection",
        "source_template_norm": "unit L2 on the declared patch",
        "source_amplitude": SOURCE_AMPLITUDE,
        "source_amplitude_units": "DN coefficient multiplying a unit-L2 image template; not peak DN",
        "exact_plane_columns": ["1", "x/12", "y/12"],
        "background_formula": "50 + 20*(x>=0) + 8*sin((y+64)/12), x,y=-12..12",
        "gaussian_formula": "exp(-((x-cx)^2+(y-cy)^2)/2), then unit-L2 normalization on patch",
        "fixed_light_center_xy": [7.0, -5.0],
        "fixed_light_coefficient": 40.0,
        "uniform_plane_coefficients": [1.5, 0.0, 0.0],
        "tilted_plane_coefficients": [1.5, 3.0, -2.0],
        "response_bound_dn": RESPONSE_BOUND_DN,
        "background_design_entry_bound_dn": BACKGROUND_DESIGN_BOUND_DN,
        "source_and_fixed_design_relative_bound": TEMPLATE_RELATIVE_BOUND,
        "uncertainty_contains_zero_source_bound": "abs(m), so the admitted source-error box contains -m",
        "temporal_coordinates_origin": "independent deterministic generator, not fitted from current response",
        "linear_positions": "(1.25*t, 0), t=-8..0",
        "slow_positions": "(0.05*t, 0), t=-8..0; unresolved role declared, no fitted threshold",
        "hovering_positions": "(0, 0), t=-8..0",
        "curved_accelerating_positions": "(1.4*t+0.06*t^2, 0.07*t^2), t=-8..0",
        "short_history_prior_count": 3,
        "identifiability_pair": ["identifiability_moving_world", "identifiability_fixed_emitter_world"],
        "scenarios": scenarios,
        "limitations": [
            "Templates are oracle solver-unit inputs, not causal learned image templates",
            "Declared bounded design errors are engineering fixture assumptions, not camera-noise calibration",
            "Temporal metadata does not certify motion or airborne class from pixels",
            "Metadata-only unknowns do not demonstrate an algorithm detecting missing history, support, or raw normalization",
            "The raw-normalization case supplies no raw stamps and does not test normalization arithmetic",
            "A moving point and sequentially blinking fixed emitters can have identical observations",
            "No pass threshold or empirical tuning is defined by this manifest",
        ],
    }


def build_cases():
    """Build fresh deterministic oracle vectors and separately declared metadata.

    ``Z`` contains only uncertain nuisance columns. ``P`` is the separate exact
    affine plane.  All three error bounds have the full corresponding array
    shape; there are no NaNs, silent support restrictions, or random draws.
    ``synthetic_observations`` is a diagnostic history, not a core input or a
    source-learning procedure.  Its trajectory is generated independently of y.
    """
    xx, yy = _grid()
    plane = np.column_stack((np.ones(xx.size), xx.ravel()/12, yy.ravel()/12))
    background = 50.0 + 20.0*(xx >= 0) + 8.0*np.sin((yy+64.0)/12.0)
    source_image = _unit_direction(xx, yy, (0.0, 0.0))
    fixed_image = _unit_direction(xx, yy, (7.0, -5.0))
    cases = []
    for case_id, role, temporal_role, unknown_reasons in _SCENARIOS:
        source_amplitude = 0.0 if case_id in ("background_only", "fixed_flicker_separate") else SOURCE_AMPLITUDE
        nuisance_images = [background]
        nuisance_bounds = [np.full(IMAGE_SHAPE, BACKGROUND_DESIGN_BOUND_DN)]
        nuisance_coefficients = [1.0]
        nuisance_names = ["step_and_sinusoidal_background"]
        if case_id in ("fixed_flicker_separate", "moving_with_separate_fixed_flicker"):
            nuisance_images.append(fixed_image)
            nuisance_bounds.append(TEMPLATE_RELATIVE_BOUND*np.abs(fixed_image))
            nuisance_coefficients.append(40.0)
            nuisance_names.append("separate_fixed_light_oracle_template")
        elif case_id == "fixed_source_exact_overlap":
            nuisance_images.append(source_image)
            nuisance_bounds.append(TEMPLATE_RELATIVE_BOUND*np.abs(source_image))
            nuisance_coefficients.append(40.0)
            nuisance_names.append("fixed_light_identical_to_source_oracle_template")
        z = np.column_stack([value.ravel() for value in nuisance_images])
        z_bound = np.column_stack([value.ravel() for value in nuisance_bounds])
        plane_coefficients = np.zeros(3)
        if case_id == "uniform_brightness_change":
            plane_coefficients = np.array([1.5, 0.0, 0.0])
        elif case_id == "plane_brightness_change":
            plane_coefficients = np.array([1.5, 3.0, -2.0])
        source = source_image.ravel().copy()
        source_bound = TEMPLATE_RELATIVE_BOUND*np.abs(source)
        if case_id == "uncertainty_contains_zero":
            source_bound = np.abs(source)
        response = plane@plane_coefficients + z@np.asarray(nuisance_coefficients) + source_amplitude*source
        history_count = 3 if case_id == "short_history" else 8
        times, positions = _positions(temporal_role, history_count)
        # Fixed peak scale across the history is generated from the current
        # oracle unit-template normalization.  There is no per-frame fitted
        # direction, alignment, normalization, or current-response feedback.
        gaussian_norm = float(np.linalg.norm(_gaussian(xx, yy, (0.0, 0.0))))
        history = np.stack([
            background + source_amplitude/gaussian_norm*_gaussian(xx, yy, position)
            for position in positions[:-1]
        ])
        if case_id in ("fixed_flicker_separate", "moving_with_separate_fixed_flicker"):
            # A fixed location with deterministic varying brightness, explicitly
            # represented in the nuisance dictionary even at the current time.
            history += np.arange(1, history_count+1)[:, None, None]*5.0*fixed_image
        elif case_id == "fixed_source_exact_overlap":
            history += 40.0*source_image
        unknown = list(unknown_reasons)
        identity_available = temporal_role not in (
            "slow_motion", "hovering", "unresolved_overlap", "identifiability_twins")
        support_known = case_id not in (
            "unknown_nuisance_support", "identifiability_moving_world", "identifiability_fixed_emitter_world")
        interpretation = "synthetic Gaussian source following declared centers"
        if case_id == "background_only":
            interpretation = "no source"
        elif case_id == "fixed_flicker_separate":
            interpretation = "only a separate stationary emitter with changing brightness"
        elif case_id == "identifiability_fixed_emitter_world":
            interpretation = "stationary Gaussian emitters at every declared center; only emitter for each time is lit"
        elif case_id == "identifiability_moving_world":
            interpretation = "one Gaussian point moving through the declared centers"
        truth = {
            "numerical_role": role, "temporal_role": temporal_role,
            "source_template_origin": SOURCE_TEMPLATE_ORIGIN,
            "source_amplitude": source_amplitude,
            "source_amplitude_units": "DN coefficient multiplying unit-L2 source template",
            "source_peak_dn": float(source_amplitude*np.max(source)),
            "source_l2_norm": float(np.linalg.norm(source)),
            "plane_coefficients": plane_coefficients.tolist(),
            "nuisance_coefficients": nuisance_coefficients,
            "nuisance_columns": nuisance_names,
            "physical_interpretation": interpretation,
            "airborne_class_claimed": False,
            "template_learned_from_current_or_history": False,
        }
        if case_id == "raw_normalization_unsupported_metadata":
            truth["raw_normalization_limit"] = "metadata-only near-zero/uncertain raw stamp condition; raw normalization not simulated"
        cases.append({
            "case_id": case_id,
            "y": response.copy(), "Z": z.copy(), "m": source.copy(), "P": plane.copy(),
            "response_bound": np.full(source.shape, RESPONSE_BOUND_DN),
            "nuisance_bound": z_bound.copy(), "source_bound": source_bound.copy(),
            "truth": truth,
            "integration": {
                "history_complete": history_count == 8,
                "nuisance_support_known": support_known,
                "temporal_identity_available": identity_available,
                "unknown_reasons": unknown,
                "flags_origin": "declared synthetic external metadata; not inferred from image evidence",
            },
            "temporal_provenance": {
                "origin": "independent deterministic generator; not current-response estimation",
                "time_indices": times.tolist(), "centers_xy": positions.tolist(),
                "current_time_index": 0.0, "current_center_xy": positions[-1].tolist(),
                "history_count": history_count, "physical_identity_certified_from_images": False,
            },
            "synthetic_observations": {
                "history_images": history.copy(), "current_image": response.reshape(IMAGE_SHAPE).copy(),
                "used_to_learn_core_templates": False,
            },
        })
    return deepcopy(cases)
