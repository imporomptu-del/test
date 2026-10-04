"""Reproduce V43's ordinary synthetic absolute-budget counterexample.

This is a post-run synthetic diagnosis, not a changed V43 arm, new gate or V44
experiment. It reads no journals, media or real-data results. Output must be a
new path outside the frozen stability_01 directory.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

import accuracy_v42_localized as v42
from accuracy_v43_localized import evaluate_arm


ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_BOUND_DN = 0.5


def synthetic_fixture():
    """Exactly the ordinary fixture already declared in V43 integration tests."""
    y, x = np.indices((129, 129))
    background = 50+20*(x >= 64)+8*np.sin(y/12)
    centers = [[29+4*i, 64] for i in range(8)]
    def point(cx, cy):
        return 30*np.exp(-((x-cx)**2+(y-cy)**2)/2)
    return (background+point(64, 64),
            np.stack([background+point(*position) for position in centers]),
            centers, [0, 0], "bright")


def diagnose():
    current, history, centers, offset, polarity = synthetic_fixture()
    components = v42.prepare_components(history, centers, offset, polarity)
    y, x = np.indices((129, 129))
    plane = np.stack((np.ones((129, 129)), (x-64)/32, (y-64)/32), axis=-1)
    distance = np.maximum(abs(x-64), abs(y-64))
    annulus = (distance >= 16) & (distance <= 30) & np.isfinite(components["background"])
    core = (slice(52, 77), slice(52, 77))
    common = np.isfinite(components["background"][core])
    a = np.column_stack((plane[annulus], components["background"][annulus]))
    b = np.column_stack((plane[core][common], components["background"][core][common]))
    target = current[annulus]
    beta = np.linalg.lstsq(a, target, rcond=None)[0]
    nominal = b@beta

    # Construct admitted +/-0.5 changes in the aligned input images themselves,
    # not merely arbitrary perturbations of a fitted coefficient. The common
    # pixelwise shift in all eight priors commutes with their masked median.
    current_perturbed = current.copy()
    current_perturbed[annulus] += PERTURBATION_BOUND_DN
    history_perturbed = history.copy()
    history_perturbed[:, annulus] -= PERTURBATION_BOUND_DN
    history_perturbed[:, 52:77, 52:77] += PERTURBATION_BOUND_DN
    components_perturbed = v42.prepare_components(history_perturbed, centers, offset, polarity)
    ap = np.column_stack((plane[annulus], components_perturbed["background"][annulus]))
    bp = np.column_stack((plane[core][common], components_perturbed["background"][core][common]))
    beta_perturbed = np.linalg.lstsq(ap, current_perturbed[annulus], rcond=None)[0]
    actual_shift = bp@beta_perturbed-nominal
    expected_background = components["background"].copy()
    expected_background[annulus] -= PERTURBATION_BOUND_DN
    expected_background[52:77, 52:77] += PERTURBATION_BOUND_DN
    background_difference = components_perturbed["background"]-expected_background
    background_error = float(np.nanmax(abs(background_difference)))

    # With fixed design, the exact worst-case response change at pixel i is
    # epsilon*||H_i||_1, attained by dy=epsilon*sign(H_i), H=B A+.
    operator = b@np.linalg.pinv(a)
    exact_response_bounds = PERTURBATION_BOUND_DN*np.sum(abs(operator), axis=1)
    extreme_index = int(np.argmax(exact_response_bounds))
    response_vertex = PERTURBATION_BOUND_DN*np.sign(operator[extreme_index])
    beta_vertex = np.linalg.lstsq(a, target+response_vertex, rcond=None)[0]
    vertex_shift = b@beta_vertex-nominal
    positions_yx = np.argwhere(common)
    pixel = positions_yx[extreme_index]+52

    stable = evaluate_arm(current, history, centers, offset, polarity, "stable_only")
    fit = stable["stability"]["annulus"]["diagnostics"]["free_fit"]
    conservative = np.asarray(fit["prediction_total_bound_dn"])
    worst_index = int(np.argmax(conservative))

    # Fixed-design illustration only: use the causal template projected away
    # from the shared plane/background subspace. No uncertain-design guarantee
    # or moving-versus-fixed interpretation is asserted for this contrast.
    nuisance = b
    moving = components["moving_template"][core][common]
    residual_template = moving-nuisance@np.linalg.lstsq(nuisance, moving, rcond=None)[0]
    contrast = residual_template/(residual_template@residual_template)
    contrast_value = float(contrast@current[core][common])
    contrast_response_bound = float(PERTURBATION_BOUND_DN*np.sum(abs(contrast)))

    return {
        "schema": "seaqr.accuracy-v43-absolute-budget-synthetic-diagnosis.v1",
        "synthetic_only": True,
        "uses_real_media_journals_or_scores": False,
        "changes_frozen_v43_or_production": False,
        "is_new_classifier_or_gate": False,
        "fixture": {"source": "ordinary fixture in tests/unit/test_accuracy_v43_localized.py",
                    "history_frames": 8, "shape": [129, 129],
                    "foreground_peak_dn": 30.0, "current_center_xy": [64, 64],
                    "prior_centers_xy": centers, "annulus_pixels": int(annulus.sum()),
                    "core_pixels": int(common.sum()),
                    "background_formula": "50+20*(x>=64)+8*sin(y/12)",
                    "foreground_formula": "30*exp(-((x-cx)^2+(y-cy)^2)/2)"},
        "contract": {"aligned_input_absolute_bound_dn": PERTURBATION_BOUND_DN,
                     "prediction_budget_dn": 2*PERTURBATION_BOUND_DN,
                     "fixed_geometry_masks_and_support": True,
                     "not_actual_camera_noise_or_physical_confidence": True},
        "nominal_annulus": {"coefficient_order": ["offset", "x_plane", "y_plane", "background_gain"],
                            "coefficients": beta.tolist(),
                            "residual_l2_norm_dn": float(np.linalg.norm(target-a@beta))},
        "admissible_input_construction": {
            "current_annulus_change_dn": PERTURBATION_BOUND_DN,
            "all_prior_annulus_change_dn": -PERTURBATION_BOUND_DN,
            "all_prior_core_change_dn": PERTURBATION_BOUND_DN,
            "maximum_current_input_change_dn": float(np.max(abs(current_perturbed-current))),
            "maximum_prior_input_change_dn": float(np.max(abs(history_perturbed-history))),
            "background_median_translation_error_dn": background_error,
            "background_support_unchanged": bool(np.array_equal(np.isfinite(components["background"]),
                                                np.isfinite(components_perturbed["background"]))),
            "coefficients_after_perturbation": beta_perturbed.tolist(),
            "gain_remains_nonnegative": bool(beta_perturbed[-1] >= 0),
            "actual_prediction_shift_min_dn": float(actual_shift.min()),
            "actual_prediction_shift_max_dn": float(actual_shift.max()),
            "expected_prediction_shift_dn": 1.5,
            "violates_one_dn_absolute_budget": bool(np.all(actual_shift > 1.0)),
        },
        "exact_response_only_extremum": {
            "formula": "epsilon*sum(abs((B@pinv(A))[pixel,:])); attained by epsilon*sign(row)",
            "pixel_xy_in_129_patch": [int(pixel[1]), int(pixel[0])],
            "analytic_maximum_shift_dn": float(exact_response_bounds[extreme_index]),
            "attained_prediction_shift_dn": float(vertex_shift[extreme_index]),
            "maximum_training_response_change_dn": float(np.max(abs(response_vertex))),
            "gain_remains_nonnegative": bool(beta_vertex[-1] >= 0),
        },
        "v43_conservative_screen": {
            "available": stable["available"], "reasons": stable["reasons"],
            "maximum_prediction_bound_dn": float(conservative[worst_index]),
            "response_term_at_worst_pixel_dn": fit["prediction_response_bound_dn"][worst_index],
            "design_term_at_worst_pixel_dn": fit["prediction_design_bound_dn"][worst_index],
            "scaled_train_condition_number": fit["scaled_train_condition_number"],
            "scaled_train_singular_values": fit["train_singular_values"],
            "design_perturbation_spectral_bound": fit["train_design_perturbation_spectral_bound"],
            "robust_minimum_singular_value": fit["robust_minimum_singular_value"],
            "interpretation": "The conservative bound is loose, but even exact admitted changes exceed the budget; tighter bounds alone cannot make this fixture pass.",
        },
        "fixed_design_contrast_illustration": {
            "formula": "r=m-Z@lstsq(Z,m); contrast=r/(r@r), Z=[plane,nominal_background]",
            "nominal_unit_template_contrast_coefficient": contrast_value,
            "response_only_absolute_coefficient_bound": contrast_response_bound,
            "coefficient_change_from_uniform_1p5_dn": float(contrast@np.full(int(common.sum()), 1.5)),
            "contrast_sum": float(contrast.sum()),
            "nuisance_orthogonality_norm": float(np.linalg.norm(contrast@nuisance)),
            "design_uncertainty_certified": False,
            "moving_versus_fixed_light_discrimination_certified": False,
            "scope_limit": "An observability illustration with nominal fixed designs, not a validated detector, calibrated confidence or acceptance/rejection policy.",
        },
        "recommended_next_check": "Synthetic-only localized-contrast uncertainty analysis and fixed-light counterexamples before any new broad replay; not a relaxed absolute-background threshold.",
    }


def write_diagnosis(output):
    output = Path(output).resolve()
    frozen = ROOT/"results/tiny_target/accuracy_v43_20260925/stability_01"
    if output == frozen or frozen in output.parents:
        raise ValueError("Diagnosis must remain outside the frozen stability_01 output")
    if output.exists():
        raise FileExistsError("Use a fresh diagnosis output; existing evidence is preserved")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    inputs = [Path(__file__), ROOT/"tests/unit/test_accuracy_v43_absolute_budget.py",
              ROOT/"tests/unit/test_accuracy_v43_localized.py", ROOT/"scripts/accuracy_v42_localized.py",
              ROOT/"scripts/accuracy_v43_localized.py", ROOT/"scripts/accuracy_v43_stable_fit.py",
              ROOT/"scripts/accuracy_v43_components.py", ROOT/"scripts/accuracy_v43_bounds.py"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    report = diagnose()
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Diagnostic dependency changed during execution")
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), inputs_sha256=hashes,
                  runtime={"python": platform.python_version(), "numpy": np.__version__})
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_diagnosis(arguments.output)
    print(str(arguments.output.resolve()))
