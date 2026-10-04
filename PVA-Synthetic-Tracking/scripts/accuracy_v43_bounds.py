"""Conditional component sensitivity to aligned-image errors bounded by 0.5 DN.

These are deterministic perturbation bounds, NOT camera noise estimates, target
probabilities, or physical confidence. Geometry, identities, masks, finite support,
and the exact set of stamps retained by the nominal algorithm are held fixed.
Changes in discarded-stamp membership, peak selection, registration, clipping,
model misspecification, and physical identity are outside this contract.
"""
import warnings

import numpy as np

import accuracy_v42_localized as legacy


ALIGNED_VALUE_BOUND_DN = 0.5
RESIDUAL_STAMP_BOUND_DN = 1.0
HIGHPASS_STAMP_BOUND_DN = 1.0
MIN_NORMALIZATION_NORM = 1.0e-6
APERTURE = np.outer(np.hanning(17), np.hanning(17))


def normalization_bound(vector, entry_bound, minimum_norm=MIN_NORMALIZATION_NORM):
    """Bound entrywise changes of v/||v|| on fixed finite coordinates.

    If ||dv|| <= delta and r=||v||, reverse triangle inequality gives
    ||v+dv|| >= r-delta. Then each change is at most
    e/(r-delta) + |v| delta/[r(r-delta)]. A lower norm that reaches the
    nominal component acceptance threshold is explicitly unsupported.
    """
    v = np.asarray(vector)
    e = np.asarray(entry_bound)
    if (v.shape != e.shape or v.dtype.kind not in 'iuf' or e.dtype.kind not in 'iuf'
            or np.isinf(v).any() or np.isinf(e).any()):
        raise ValueError('Matching real arrays without infinity required')
    v, e = v.astype(float), e.astype(float)
    support = np.isfinite(v)
    if np.any(np.isfinite(e) != support) or np.any(e[support] < 0):
        raise ValueError('Nonnegative bounds must have exactly the vector finite support')
    if not np.isfinite(minimum_norm) or minimum_norm < 0:
        raise ValueError('Finite nonnegative normalization threshold required')
    r = float(np.linalg.norm(v[support]))
    delta = float(np.linalg.norm(e[support]))
    lower = r - delta
    metadata = dict(nominal_norm=r, perturbation_l2_bound=delta,
                    norm_lower_bound=lower, minimum_accepted_norm=minimum_norm,
                    finite_support_count=int(support.sum()))
    if (not np.isfinite(r) or not np.isfinite(delta) or r <= 0
            or lower <= minimum_norm):
        return None, dict(**metadata, available=False,
                          reason='normalization_lower_norm_not_above_acceptance_threshold')
    bound = np.full(v.shape, np.nan, dtype=float)
    bound[support] = e[support] / lower + np.abs(v[support]) * delta / (r * lower)
    if not np.isfinite(bound[support]).all():
        return None, dict(**metadata, available=False, reason='nonfinite_normalization_bound')
    return bound, dict(**metadata, available=True, reason=None)


def _record_from_raw(raw_stamps, indices):
    normalized, norms, used = [], [], []
    for raw, index in zip(raw_stamps, indices):
        stamp = legacy._positive_unit_stamp(raw)
        if stamp is None:
            continue
        weighted = np.maximum(raw, 0.0) * APERTURE
        weighted[APERTURE == 0] = 0.0
        normalized.append(stamp)
        norms.append(float(np.sqrt(np.nansum(weighted * weighted))))
        used.append(index)
    return dict(normalized_stamps=np.stack(normalized) if normalized else np.empty((0, 17, 17)),
                raw_weighted_norms=norms, history_indices=used)


def _legacy_history(history, centres, background, sign, metadata):
    frames = [i for i, centre in enumerate(centres) if centre is not None]
    moving = _record_from_raw([
        legacy._stamp(sign * (history[i] - background), centres[i]) for i in frames], frames)
    if moving['history_indices'] != metadata['usable_moving_stamp_indices']:
        raise ValueError('Nominal moving stamp membership changed')
    highpasses = np.stack([sign * legacy._box_highpass(frame) for frame in history])
    fixed = []
    for anchor in metadata['anchors']:
        if not anchor['template_available']:
            continue
        centre = [anchor['x'], anchor['y']]
        entry = _record_from_raw([legacy._stamp(frame, centre) for frame in highpasses], list(range(8)))
        if len(entry['history_indices']) != anchor['usable_stamp_count']:
            raise ValueError('Nominal fixed stamp membership changed')
        entry['centre_xy'] = centre
        fixed.append(entry)
    return dict(moving=moving, fixed=fixed)


def _normalization_record(record):
    stamps = np.asarray(record['normalized_stamps'])
    indices = list(record['history_indices'])
    norms = np.asarray(record['raw_weighted_norms'])
    if (stamps.dtype.kind not in 'iuf' or stamps.shape != (len(indices), 17, 17)
            or norms.dtype.kind not in 'iuf' or norms.shape != (len(indices),)
            or np.isinf(stamps).any() or not np.isfinite(norms).all()
            or (norms <= 0).any() or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 8 for i in indices)):
        raise ValueError('Invalid used normalized stamp provenance')
    stamps = stamps.astype(float)
    if any(not np.isclose(np.nansum(s*s), 1.0, rtol=1e-12, atol=1e-12) for s in stamps):
        raise ValueError('Used normalized stamps must have unit finite-support energy')
    if (stamps[:, APERTURE == 0] != 0).any():
        raise ValueError('Hann boundary must have exact zero value')
    if (stamps[np.isfinite(stamps)] < 0).any():
        raise ValueError('Positive normalized stamp expected')
    return stamps, norms.astype(float), indices


def _protected_history(provenance, metadata):
    """Translate the component producer's explicit, lossless history schema."""
    if not isinstance(provenance, dict) or 'moving' not in provenance or 'fixed' not in provenance:
        raise ValueError('Protected components require exact used-stamp template_history')

    def entry(record):
        norms = np.asarray(record['raw_weighted_l2_norm_dn'], dtype=float)
        energies = np.asarray(record['raw_weighted_energy_dn2'], dtype=float)
        if energies.shape != norms.shape or not np.allclose(energies, norms*norms, rtol=1e-12, atol=1e-12):
            raise ValueError('Protected raw norm/energy provenance disagrees')
        out = dict(normalized_stamps=record['normalized_stamps'],
                   history_indices=record['prior_history_indices'], raw_weighted_norms=norms)
        if 'anchor_xy' in record:
            out['centre_xy'] = record['anchor_xy']
        return out

    moving = entry(provenance['moving'])
    if moving['history_indices'] != metadata['usable_moving_stamp_indices']:
        raise ValueError('Protected moving stamp membership disagrees with metadata')
    fixed = [entry(record) for record in provenance['fixed']]
    anchors = [anchor for anchor in metadata['anchors'] if anchor['template_available']]
    if len(anchors) != len(fixed):
        raise ValueError('Protected fixed history omits a nominal template')
    for record, anchor in zip(fixed, anchors):
        if (record['centre_xy'] != [anchor['x'], anchor['y']]
                or record['history_indices'] != anchor['usable_stamp_history_indices']
                or len(record['history_indices']) != anchor['usable_stamp_count']):
            raise ValueError('Protected fixed stamp membership/placement disagrees with metadata')
    return dict(moving=moving, fixed=fixed)


def _combined_bound(record, nominal_stamp, entry_dn_bound):
    stamps, norms, indices = _normalization_record(record)
    combined = legacy._combine_stamps(list(stamps))
    if combined is None or nominal_stamp is None:
        return None, dict(available=False, reasons=['nominal_combined_template_missing'],
                          used_history_indices=indices)
    if not np.allclose(combined, nominal_stamp, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError('Used stamp provenance does not reconstruct nominal template')
    bounds, details = [], []
    for index, stamp, norm in zip(indices, stamps, norms):
        vector = stamp * norm
        error = np.where(np.isfinite(vector), entry_dn_bound * APERTURE, np.nan)
        bound, detail = normalization_bound(vector, error)
        details.append(dict(history_index=index, **detail))
        if bound is None:
            # Do not drop an uncertain nominally used stamp and improve the median.
            return None, dict(available=False, reasons=['used_stamp_normalization_unbounded'],
                              used_history_indices=indices, stamp_bounds=details,
                              uncertain_used_stamp_omitted=False)
        bounds.append(bound)
    stacked = np.stack(stamps)
    median, counts = legacy._median(stacked, legacy.MIN_USABLE_STAMPS)
    median[APERTURE == 0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # A coordinatewise median is L-infinity Lipschitz. No independence or
        # sample-count reduction is assumed, including correlated quantization.
        median_error = np.nanmax(np.stack(bounds), axis=0)
    median_error[counts < legacy.MIN_USABLE_STAMPS] = np.nan
    median_error[APERTURE == 0] = 0.0
    final, detail = normalization_bound(median, median_error)
    return final, dict(available=final is not None,
        reasons=[] if final is not None else ['combined_median_normalization_unbounded'],
        used_history_indices=indices, stamp_bounds=details,
        median_rule='maximum bound over all finite used stamps per coordinate; no independence reduction',
        final_normalization=detail)


def _validate_placed(stamp, centre, nominal):
    placed = legacy._place(stamp, centre)
    if not np.allclose(placed, nominal, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError('Used stamp placement does not reconstruct component template')


def component_bounds(history129, prior_centers_xy, predicted_offset_xy,
                     polarity, components, protected=False):
    """Conditional finite-support bounds; unavailable components remain unknown.

    ``protected=True`` requires lossless ``components['template_history']``:
    moving and each fixed record have normalized_stamps, prior_history_indices,
    raw_weighted_energy_dn2 and raw_weighted_l2_norm_dn; fixed records also have
    anchor_xy and align exactly with
    fixed_templates. A used stamp cannot silently disappear from this record.
    """
    history = legacy._array(history129, legacy.HISTORY_SHAPE, 'history129')
    centres, offset = legacy._coordinates(prior_centers_xy, predicted_offset_xy)
    if polarity not in ('bright', 'dark') or type(protected) is not bool:
        raise ValueError('Known polarity and boolean protected flag required')
    background = legacy._array(components['background'], legacy.PATCH_SHAPE, 'background')
    fixed = np.asarray(components['fixed_templates'])
    if fixed.ndim != 3 or fixed.shape[1:] != legacy.PATCH_SHAPE or np.isinf(fixed).any():
        raise ValueError('fixed_templates must be Kx129x129 without infinity')
    background_bound = np.where(np.isfinite(background), ALIGNED_VALUE_BOUND_DN, np.nan)
    metadata = dict(conditional_only=True, aligned_value_error_bound_dn=ALIGNED_VALUE_BOUND_DN,
        moving_residual_stamp_error_bound_dn=RESIDUAL_STAMP_BOUND_DN,
        fixed_highpass_stamp_error_bound_dn=HIGHPASS_STAMP_BOUND_DN,
        protected_fixed_components=protected,
        fixed_geometry=True, fixed_anchor_identities=True, fixed_finite_support=True,
        fixed_used_stamp_membership=True, no_independence_assumption=True,
        numerical_roundoff_not_a_camera_noise_model=True,
        omitted_stamp_selection_uncertainty='Not bounded: a discarded zero-energy stamp can become used under perturbation.',
        interpretation='Conditional deterministic component-array sensitivity; not total camera noise, physical confidence or a rejection policy.')
    provenance = _protected_history(components.get('template_history'), components['metadata']) if protected else _legacy_history(
        history, centres, background, 1.0 if polarity == 'bright' else -1.0, components['metadata'])
    if not isinstance(provenance, dict) or 'moving' not in provenance or 'fixed' not in provenance:
        raise ValueError('Protected components require exact used-stamp template_history')
    if len(provenance['fixed']) != len(fixed):
        raise ValueError('Fixed template provenance must preserve every nominal template')
    reasons = []
    moving_bound = None
    moving_stamp = components['moving_stamp']
    if components['moving_template'] is None or moving_stamp is None:
        reasons.append('nominal_moving_template_missing')
        metadata['moving'] = dict(available=False, reasons=['nominal_moving_template_missing'])
    else:
        _validate_placed(moving_stamp, 64.0 + offset, components['moving_template'])
        stamp_bound, detail = _combined_bound(provenance['moving'], moving_stamp, RESIDUAL_STAMP_BOUND_DN)
        metadata['moving'] = detail
        if stamp_bound is None:
            reasons.append('moving_component_uncertainty_unavailable')
        else:
            moving_bound = legacy._place(stamp_bound, 64.0 + offset)
            assert np.array_equal(np.isfinite(moving_bound), np.isfinite(components['moving_template']))
    fixed_bounds, fixed_metadata = [], []
    for index, (entry, nominal) in enumerate(zip(provenance['fixed'], fixed)):
        stamps, unused_norms, unused_indices = _normalization_record(entry)
        stamp = legacy._combine_stamps(list(stamps))
        if stamp is None:
            raise ValueError('Nominal fixed template lacks reconstructible used stamps')
        centre = legacy._array(entry['centre_xy'], (2,), 'fixed centre')
        if not np.isfinite(centre).all():
            raise ValueError('Finite fixed center required')
        _validate_placed(stamp, centre, nominal)
        stamp_bound, detail = _combined_bound(entry, stamp, HIGHPASS_STAMP_BOUND_DN)
        fixed_metadata.append(dict(template_index=index, centre_xy=centre.tolist(), **detail))
        if stamp_bound is None:
            fixed_bounds.append(np.full((129, 129), np.nan))
            reasons.append('fixed_component_uncertainty_unavailable:' + str(index))
        else:
            bound = legacy._place(stamp_bound, centre)
            assert np.array_equal(np.isfinite(bound), np.isfinite(nominal))
            fixed_bounds.append(bound)
    metadata['fixed'] = fixed_metadata
    return dict(available=not reasons, reasons=reasons,
        background_bound129=background_bound, moving_template_bound129=moving_bound,
        fixed_template_bounds=np.stack(fixed_bounds) if fixed_bounds else np.empty((0, 129, 129)),
        metadata=metadata)
