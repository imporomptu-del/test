"""Four synthetic-only mathematical arms over the unchanged V44 adapter.

Function-local dependency injection preserves the frozen adapter bytecode and
does not mutate module globals. The returned raw adapter result is retained so
the original arm can be checked exactly against its immutable V44 baseline.
"""
from types import FunctionType

import accuracy_v44_causal_probe as v44
from accuracy_v43_bounds import component_bounds as old_bounds
from accuracy_v44_contrast import source_contrast
from accuracy_v45_bounds import component_bounds as box_bounds
from accuracy_v45_presence import source_presence


ARMS = {
    "amplitude_old_bounds": ("amplitude", "v43"),
    "presence_old_bounds": ("numerator", "v43"),
    "amplitude_box_bounds": ("amplitude", "v45"),
    "presence_box_bounds": ("numerator", "v45"),
}


def evaluate_causal_probe(current129, history129, prior_centers_xy,
                          predicted_offset_xy, polarity, *, arm):
    if arm not in ARMS:
        raise ValueError("Unknown predeclared V45 arm")
    quantity, version = ARMS[arm]
    environment = dict(v44.evaluate_causal_probe.__globals__)
    environment["component_bounds"] = old_bounds if version == "v43" else box_bounds
    environment["_source_contrast"] = source_contrast if quantity == "amplitude" else source_presence
    evaluate = FunctionType(v44.evaluate_causal_probe.__code__, environment,
                            "v45_isolated_causal_probe", v44.evaluate_causal_probe.__defaults__,
                            v44.evaluate_causal_probe.__closure__)
    result = evaluate(current129, history129, prior_centers_xy, predicted_offset_xy, polarity)
    return {"arm": arm, "quantity": quantity, "bound_version": version,
            "raw_adapter_result": result,
            "legacy_numerical_contrast_field_contains": quantity,
            "is_motion_or_classification_gate": False,
            "motion_status": "unknown", "physical_class": "unknown"}
