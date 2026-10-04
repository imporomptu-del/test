"""Synthetic-only adapter for prior-learned localized contrast evidence.

No file, journal, camera or video decoding is implemented here. A caller supplies
already aligned native-DN synthetic arrays and actual prior same-ID positions.
The unchanged V43 component learner uses priors only. The supplied forecast
offset is NOT independently verified as causal by this adapter; no current
detection/position argument exists. Current finite pixels select evaluation
support and their values supply the response, never the learned templates.

Numerical source-coefficient evidence is separate from physical motion/class.
In particular, incomplete fixed-light coverage remains ambiguous even when a
source coefficient is numerically observable. This is not a production gate.
"""

import hashlib

import numpy as np

import accuracy_v42_localized as v42
from accuracy_v43_bounds import component_bounds
from accuracy_v43_components import prepare_components


MIN_ACTUAL_PRIOR_CENTERS = 5
INPUT_BOUND_DN = 0.5
MIN_CORE_SUPPORT = 64


def _source_contrast(*args, **kwargs):
    # A lazy import permits independent adapter tests without replacing or
    # otherwise modifying the mathematical solver's module globals.
    from accuracy_v44_contrast import source_contrast
    return source_contrast(*args, **kwargs)


def _hash_array(value):
    array = np.asarray(value, dtype=np.float64, order="C")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def evaluate_causal_probe(current129, history129, prior_centers_xy,
                          predicted_offset_xy, polarity):
    """Return JSON-safe synthetic evidence, never a motion/class decision.

    Inadequate history/template/bounds/support remain unavailable. Numerical
    contrast is retained despite physical/fixed-alternative ambiguity. No
    model, threshold, regularization or predicted position is selected from
    current response values. The historical same-ID association and supplied
    forecast offset are caller assumptions, not validated by this function.
    """
    current = v42._array(current129, v42.PATCH_SHAPE, "current129")
    history = v42._array(history129, v42.HISTORY_SHAPE, "history129")
    centers, offset = v42._coordinates(prior_centers_xy, predicted_offset_xy)
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity must be bright or dark")
    actual_count = sum(point is not None for point in centers)
    result = {
        "synthetic_only": True, "available": False, "reasons": [],
        "numerical_contrast": None, "motion_status": "unknown", "physical_class": "unknown",
        "is_motion_or_classification_gate": False, "production_changed": False,
        "ambiguity_reasons": ["physical_motion_and_class_not_certified_by_source_contrast"],
        "prior_context": {"frames": 8, "actual_prior_centers": actual_count,
                          "minimum_actual_prior_centers": MIN_ACTUAL_PRIOR_CENTERS,
                          "templates_use_prior_images_only": True,
                          "same_id_association_and_forecast_causality_are_caller_assumptions": True,
                          "supplied_forecast_offset_independently_verified": False,
                          "current_values_used_for": "response only; finite mask also selects common support"},
        "components": None, "component_bounds": None, "learned_design_sha256": None,
        "common_support_count": 0, "common_support_sha256": None,
        "conditional_on": ["fixed_geometry", "fixed_prior_associations", "fixed_anchor_identities",
                           "fixed_finite_support", "fixed_used_stamp_membership"],
        "uncertainty_excludes": ["unknown_camera_noise", "registration_error", "temporal_deformation",
                                 "discarded_stamp_membership_changes", "physical_identity"],
    }
    if actual_count < MIN_ACTUAL_PRIOR_CENTERS:
        result["reasons"].append("insufficient_actual_prior_centers")
        return result
    components = prepare_components(history, centers, offset, polarity)
    metadata = components["metadata"]
    result["components"] = metadata
    result["ambiguity_reasons"].extend(metadata.get("fixed_learning_ambiguity_reasons", []))
    for key, reason in (("anchor_dictionary_truncated", "persistent_anchor_dictionary_truncated"),
                        ("persistent_anchor_overlaps_prediction", "causal_prediction_overlaps_persistent_fixed_anchor")):
        if metadata.get(key, False):
            result["ambiguity_reasons"].append(reason)
    result["ambiguity_reasons"] = list(dict.fromkeys(result["ambiguity_reasons"]))
    if metadata["causal_component_reasons"]:
        result["reasons"].extend(metadata["causal_component_reasons"])
        return result
    background = components["background"]
    fixed = components["fixed_templates"]
    moving = components["moving_template"] * (1.0 if polarity == "bright" else -1.0)
    result["learned_design_sha256"] = {
        "background": _hash_array(background), "fixed_templates": _hash_array(fixed),
        "signed_moving_template": _hash_array(moving),
    }
    core = (slice(52, 77), slice(52, 77))
    common = np.isfinite(current[core]) & np.isfinite(background[core]) & np.isfinite(moving[core])
    if len(fixed):
        common &= np.isfinite(fixed[:, 52:77, 52:77]).all(axis=0)
    result.update(common_support_count=int(common.sum()),
                  common_support_sha256=hashlib.sha256(common.astype(np.uint8).tobytes()).hexdigest())
    if common.sum() < MIN_CORE_SUPPORT:
        result["reasons"].append("insufficient_original_common_core_support")
        return result
    bounds = component_bounds(history, centers, offset, polarity, components, protected=True)
    result["component_bounds"] = {key: value for key, value in bounds.items()
                                  if key not in ("background_bound129", "moving_template_bound129", "fixed_template_bounds")}
    if not bounds["available"]:
        result["reasons"].extend("component_bounds:"+reason for reason in bounds["reasons"])
        return result
    nuisance = np.column_stack((background[core][common], fixed[:, 52:77, 52:77][:, common].T))
    nuisance_bound = np.column_stack((bounds["background_bound129"][core][common],
                                     bounds["fixed_template_bounds"][:, 52:77, 52:77][:, common].T))
    source_bound = bounds["moving_template_bound129"][core][common]
    if not np.isfinite(nuisance_bound).all() or not np.isfinite(source_bound).all():
        result["reasons"].append("component_bounds_missing_on_original_common_support")
        return result  # Never discard inconvenient support to improve a bound.
    iy, ix = np.indices((25, 25))
    exact_affine = np.stack((np.ones((25, 25)), (ix-12)/12, (iy-12)/12), axis=-1)[common]
    contrast = _source_contrast(current[core][common], nuisance, moving[core][common], exact_affine,
                                response_bound=INPUT_BOUND_DN, nuisance_bound=nuisance_bound,
                                source_bound=source_bound)
    result["numerical_contrast"] = contrast
    result["available"] = bool(contrast["available"])
    if not contrast["available"]:
        result["reasons"].extend("source_contrast:"+reason for reason in contrast["reasons"])
    return result
