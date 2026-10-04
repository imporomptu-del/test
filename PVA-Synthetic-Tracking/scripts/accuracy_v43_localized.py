"""Four offline ablation arms; no classification or suppression policy.

The stable arm checks conditional measurement perturbations, not total camera,
registration or model uncertainty. Unsupported scores are None, never negatives.
"""
import hashlib
from types import FunctionType

import numpy as np

import accuracy_v42_localized as v42
from accuracy_v43_bounds import component_bounds
from accuracy_v43_components import prepare_components as protected_components
from accuracy_v43_stable_fit import supported_fit

ARMS = ('baseline', 'stable_only', 'protected_only', 'combined')
INPUT_BOUND_DN = .5


def legacy_with_components(current, history, centers, offset, polarity, components):
    """Inject components into an isolated copy of frozen V42 function globals.

    No module globals are patched, including during concurrent calls. This keeps
    the numerical reference byte-for-byte identical while changing one factor.
    The frozen V42 code and this adapter are both hashed by the replay.
    """
    namespace = dict(v42.evaluate_localized.__globals__)
    namespace['prepare_components'] = lambda *args: components
    evaluate = FunctionType(v42.evaluate_localized.__code__, namespace)
    return evaluate(current, history, centers, offset, polarity)


def add_protection_ambiguity(result):
    extra = result['components'].get('fixed_learning_ambiguity_reasons', [])
    result['ambiguity_reasons'] = list(dict.fromkeys(result['ambiguity_reasons'] + extra))
    result['ambiguous'] = bool(result['ambiguity_reasons'])
    return result


def compact_fit(fit):
    # Predictions and coefficient arrays stay inside the evaluator. Diagnostics
    # retain sensitivity/support; unavailable fits offer no operative estimates.
    return {k: v for k, v in fit.items() if k not in ('coefficients', 'prediction', 'prediction_bound')}


def evaluate_arm(current129, history129, prior_centers_xy, predicted_offset_xy, polarity, arm):
    if arm not in ARMS:
        raise ValueError('Unknown ablation arm')
    if arm == 'baseline':
        return v42.evaluate_localized(current129, history129, prior_centers_xy, predicted_offset_xy, polarity)
    protected = arm in ('protected_only', 'combined')
    build = protected_components if protected else v42.prepare_components
    components = build(history129, prior_centers_xy, predicted_offset_xy, polarity)
    if arm == 'protected_only':
        return add_protection_ambiguity(legacy_with_components(current129, history129,
            prior_centers_xy, predicted_offset_xy, polarity, components))
    current = v42._array(current129, (129, 129), 'current129')
    metadata = components['metadata']
    result = dict(available=False, ambiguous=False, reasons=[], ambiguity_reasons=[],
        components=metadata, background_fit=None, common_support_count=0,
        common_support_sha256=None, fold_support_counts=[0, 0], folds=[],
        mse_stationary=None, mse_augmented=None, advantage_stationary_minus_augmented=None,
        moving_relative_energy_outside_fixed_span=None,
        model_complexity=dict(shared_annulus_parameters=4,
            stationary_core_parameters=metadata['fixed_anchor_count'],
            augmented_core_parameters=metadata['fixed_anchor_count']+1, extra_augmented_parameters=1),
        stability=dict(input_perturbation_bound_dn=INPUT_BOUND_DN,
            conditional_on_fixed_geometry_support_anchor_and_used_stamp_membership=True,
            actual_camera_noise_or_model_error_calibrated=False,
            annulus=None, core_fits=[], component_bounds=None, core_response_bound_dn=None))
    for flag, reason in [('anchor_dictionary_truncated', 'persistent_anchor_dictionary_truncated'),
                         ('persistent_anchor_overlaps_prediction', 'causal_prediction_overlaps_persistent_fixed_anchor')]:
        if metadata[flag]:
            result['ambiguity_reasons'].append(reason)
    add_protection_ambiguity(result)
    if metadata['causal_component_reasons']:
        result['reasons'].extend(metadata['causal_component_reasons'])
        return result
    y, x = np.indices((129, 129))
    plane = np.stack((np.ones((129, 129)), (x-64)/32, (y-64)/32), axis=-1)
    distance = np.maximum(abs(x-64), abs(y-64))
    background = components['background']
    core = (slice(52, 77), slice(52, 77))
    moving, fixed = components['moving_template'][core], components['fixed_templates'][:, 52:77, 52:77]
    common = np.isfinite(current[core]) & np.isfinite(background[core]) & np.isfinite(moving)
    if len(fixed):
        common &= np.isfinite(fixed).all(axis=0)
    yy, xx = np.indices((25, 25))
    parity = (yy+xx) % 2
    counts = [int((common & (parity == fold)).sum()) for fold in (0, 1)]
    result.update(common_support_count=int(common.sum()), fold_support_counts=counts,
        common_support_sha256=hashlib.sha256(common.astype(np.uint8).tobytes()).hexdigest())
    if int(common.sum()) < 64 or min(counts) < 32:
        result['reasons'].append('insufficient_common_core_or_checkerboard_support')
        return result
    annulus = (distance >= 16) & (distance <= 30) & np.isfinite(background) & np.isfinite(current)
    if int(annulus.sum()) < 64:
        result['reasons'].append('insufficient_annulus_support')
        return result
    # Exact analytic plane columns have zero measurement perturbation. Background
    # median has conditional ±0.5DN, independent of observed current core values.
    a = np.column_stack((plane[annulus], background[annulus]))
    b = np.column_stack((plane[core][common], background[core][common]))
    ua, ub = np.zeros_like(a), np.zeros_like(b)
    ua[:, -1] = INPUT_BOUND_DN; ub[:, -1] = INPUT_BOUND_DN
    annulus_fit = supported_fit(a, current[annulus], b, INPUT_BOUND_DN, nonnegative_last=True,
        train_design_bound=ua, test_design_bound=ub)
    result['stability']['annulus'] = compact_fit(annulus_fit)
    if not annulus_fit['available']:
        result['reasons'].extend('annulus:'+r for r in annulus_fit['reasons'])
        return result
    beta = annulus_fit['coefficients']
    result['background_fit'] = dict(support_count=int(annulus.sum()), gain=float(beta[-1]),
        plane_coefficients=beta[:-1].tolist(), conditional_prediction_bound_max_dn=float(
            np.max(annulus_fit['prediction_bound'])))
    response = np.full((25, 25), np.nan)
    response[common] = (current[core][common]-annulus_fit['prediction']) * (1 if polarity=='bright' else -1)
    response_bound = INPUT_BOUND_DN + float(np.max(annulus_fit['prediction_bound']))
    result['stability']['core_response_bound_dn'] = response_bound
    bounds = component_bounds(history129, prior_centers_xy, predicted_offset_xy, polarity,
                              components, protected=protected)
    result['stability']['component_bounds'] = {k: v for k, v in bounds.items() if k not in
        ('background_bound129', 'moving_template_bound129', 'fixed_template_bounds')}
    if not bounds['available']:
        result['reasons'].extend('component_bounds:'+r for r in bounds['reasons'])
        return result
    moving_bound = bounds['moving_template_bound129'][core]
    fixed_bound = bounds['fixed_template_bounds'][:, 52:77, 52:77]
    if not np.isfinite(moving_bound[common]).all() or not np.isfinite(fixed_bound[:, common]).all():
        result['reasons'].append('component_bounds_missing_on_original_common_support')
        return result  # never narrow to easier pixels
    energy = v42._residual_energy(moving[common], fixed[:, common].T)
    result['moving_relative_energy_outside_fixed_span'] = energy
    if energy <= v42.COLLINEAR_RELATIVE_ENERGY:
        result['ambiguity_reasons'].append('moving_template_redundant_with_fixed_dictionary')
        result['ambiguous'] = True
    sums = np.zeros(2)
    for fold in (0, 1):
        train, test = common & (parity == fold), common & (parity != fold)
        stationary = supported_fit(fixed[:, train].T, response[train], fixed[:, test].T, response_bound,
            train_design_bound=fixed_bound[:, train].T, test_design_bound=fixed_bound[:, test].T)
        augmented = supported_fit(np.column_stack((fixed[:, train].T, moving[train])), response[train],
            np.column_stack((fixed[:, test].T, moving[test])), response_bound, nonnegative_last=True,
            train_design_bound=np.column_stack((fixed_bound[:, train].T, moving_bound[train])),
            test_design_bound=np.column_stack((fixed_bound[:, test].T, moving_bound[test])))
        result['stability']['core_fits'].append(dict(training_parity=fold,
            stationary=compact_fit(stationary), augmented=compact_fit(augmented)))
        for label, fit in [('stationary', stationary), ('augmented', augmented)]:
            if not fit['available']:
                result['reasons'].extend(f'core:{fold}:{label}:'+r for r in fit['reasons'])
        if not stationary['available'] or not augmented['available']:
            # Diagnose both folds, but do not publish a partial held-out MSE.
            continue
        errors = [response[test]-stationary['prediction'], response[test]-augmented['prediction']]
        for index, error in enumerate(errors): sums[index] += float(error @ error)
        result['folds'].append(dict(training_parity=fold, training_count=counts[fold], heldout_count=counts[1-fold],
            stationary_signed_amplitudes=stationary['coefficients'].tolist(),
            augmented_signed_fixed_amplitudes=augmented['coefficients'][:-1].tolist(),
            moving_nonnegative_amplitude=float(augmented['coefficients'][-1]),
            mse_stationary=float(np.mean(errors[0]**2)), mse_augmented=float(np.mean(errors[1]**2))))
    if result['reasons']:
        # Partial fold scores are not an operative comparison either.
        result['folds'] = []
        return result
    result.update(available=True, mse_stationary=float(sums[0]/common.sum()),
        mse_augmented=float(sums[1]/common.sum()),
        advantage_stationary_minus_augmented=float((sums[0]-sums[1])/common.sum()))
    return result
