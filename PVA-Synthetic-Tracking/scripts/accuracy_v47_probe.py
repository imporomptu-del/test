"""Separate conditional guard-gain experiment over immutable V44/V45 components.

The source/background/fixed templates and core support are unchanged. Same-frame
outer guard samples constrain a background gain only under explicitly unverified
guard-cleanliness and spatial-transfer assumptions. No physical decision follows.
"""
from types import FunctionType

import accuracy_v44_causal_probe as base
from accuracy_v45_bounds import component_bounds
from accuracy_v47_guard_gain import estimate_guard_gain
from accuracy_v47_bounded_background import bounded_background_presence


ARM = "bounded_background_guard_gain"
QUANTITY = "bounded_gain_partial_regression_numerator"
CURRENT_USE_DESCRIPTION = ("core values supply response and original finite support; "
                           "outer guard values supply conditional gain bounds only")


def evaluate_probe(current129, history129, prior_centers_xy, predicted_offset_xy, polarity):
    captured = {}

    def bounds(history, centers, offset, sign, components, *, protected):
        result = component_bounds(history, centers, offset, sign, components, protected=protected)
        captured["components"] = components
        captured["bounds"] = result
        return result

    def contrast(y, nuisance, moving, plane, *, response_bound, nuisance_bound, source_bound):
        components, learned_bounds = captured["components"], captured["bounds"]
        gain = estimate_guard_gain(current129, history129, components["background"],
            learned_bounds["background_bound129"], prior_centers_xy, response_bound=response_bound)
        captured["gain"] = gain
        # Column zero is the unchanged background. All remaining fixed-light
        # alternatives retain their order, membership, support and uncertainty.
        return bounded_background_presence(y, nuisance[:, 0], nuisance[:, 1:], moving, plane,
            gain_interval=gain["gain_interval"] if gain["available"] else None,
            response_bound=response_bound, background_bound=nuisance_bound[:, 0],
            fixed_bound=nuisance_bound[:, 1:], source_bound=source_bound,
            gain_provenance={"method": "current_outer_guard_integer_stencil_relaxation",
                             "available": gain["available"], "reasons": gain["reasons"],
                             "background_validity_and_core_transfer_certified": False})

    environment = dict(base.evaluate_causal_probe.__globals__)
    environment.update(component_bounds=bounds, _source_contrast=contrast)
    evaluate = FunctionType(base.evaluate_causal_probe.__code__, environment,
                            "v47_isolated_guard_probe", base.evaluate_causal_probe.__defaults__,
                            base.evaluate_causal_probe.__closure__)
    result = evaluate(current129, history129, prior_centers_xy, predicted_offset_xy, polarity)
    old_current_use = result["prior_context"]["current_values_used_for"]
    result["prior_context"]["current_values_used_for"] = CURRENT_USE_DESCRIPTION
    result["ambiguity_reasons"].extend([
        "outer_guard_background_validity_not_certified",
        "guard_to_core_shared_photometric_model_not_certified"])
    result["conditional_on"].extend([
        "nonnegative_background_gain_inside_guard_outer_interval",
        "guard_background_validity", "guard_to_core_shared_photometric_gain"])
    return dict(arm=ARM, quantity=QUANTITY, bound_version="v45_plus_guard_gain",
        raw_adapter_result=result, gain_calibration=captured.get("gain"),
        legacy_numerical_contrast_field_contains=QUANTITY,
        current_use_metadata_override={"path":"raw_adapter_result.prior_context.current_values_used_for",
                                       "from":old_current_use,"to":CURRENT_USE_DESCRIPTION},
        synthetic_only=True, production_changed=False,
        is_motion_or_classification_gate=False, motion_status="unknown", physical_class="unknown",
        prior_components_and_core_support_unchanged=True,
        current_guard_values_used_for="conditional gain bounds only; never templates or forecast",
        current_core_values_used_for="response only; finite mask also selects original common support",
        old_unrestricted_background_estimand_preserved=False,
        uncertainty_excludes_guard_contamination_and_spatial_transfer_failure=True)
