"""Prior-only V43 components with foreground-safe fixed-light learning.

V42 background and moving-template arrays are returned unchanged. Only fixed
anchor observation and template eligibility change. A missing actual same-ID
position makes that entire prior frame ineligible for fixed learning. Each
retained 17-square fixed stamp requires all of its high-pass dependencies:
every native 9-square box footprint must be finite and outside the prior
foreground exclusion. Integer fixed anchors therefore require a finite,
foreground-free 25-square native dependency footprint.

Absent eligible anchors are not evidence that a fixed explanation is absent.
Untestable seed locations and lost raw repeat-supported alternatives are
explicitly marked ambiguous. Returned historical templates are empirical
samples, not calibrated uncertainty bounds or guarantees about future frames.
"""
import hashlib

import numpy as np

from accuracy_v42_localized import (
    ANCHOR_REPEAT_RADIUS, EXCLUSION_RADIUS, HISTORY_SHAPE, MAX_FIXED_ANCHORS,
    MIN_ANCHOR_CONTRAST_DN, MIN_ANCHOR_FRAMES, PATCH_SHAPE, STAMP_RADIUS,
    _anchors, _array, _box_highpass, _combine_stamps, _coordinates, _place,
    _positive_unit_stamp, _stamp, prepare_components as prepare_v42_components,
)


def _raw_peak_masks(highpasses):
    """Find original maxima before any dependency-eligibility deletion."""
    y, x = np.indices(PATCH_SHAPE)
    search = np.maximum(np.abs(x-64), np.abs(y-64)) <= 20
    masks = []
    for highpass in highpasses:
        safe = np.where(np.isfinite(highpass), highpass, -np.inf)
        neighbours = np.lib.stride_tricks.sliding_window_view(
            np.pad(safe, 1, constant_values=-np.inf), (3, 3))
        maxima = neighbours.max(axis=(-1, -2))
        masks.append(search & (safe == maxima) & (safe >= MIN_ANCHOR_CONTRAST_DN))
    return np.stack(masks)


def _full_stamp_eligible(highpass):
    """All finite 17x17 samples, including zero-aperture border, are required."""
    padded = np.pad(np.isfinite(highpass), STAMP_RADIUS, constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (17, 17))
    return windows.all(axis=(-1, -2))


def _stamp_record(stamps, indices, raw_stamps, anchor=None):
    aperture = np.outer(np.hanning(17), np.hanning(17))
    energies = []
    for raw in raw_stamps:
        weighted = np.maximum(raw, 0.0)*aperture
        weighted[aperture == 0] = 0.0
        energies.append(float(np.nansum(weighted*weighted)))
    record = dict(normalized_stamps=np.stack(stamps) if stamps else np.empty((0, 17, 17)),
                  prior_history_indices=list(indices),
                  raw_weighted_energy_dn2=np.asarray(energies, dtype=float),
                  raw_weighted_l2_norm_dn=np.sqrt(np.asarray(energies, dtype=float)))
    if anchor is not None:
        record["anchor_xy"] = list(anchor)
    return record


def _overlaps_core(x, y):
    # Full placed stamp support is conservative; its outer aperture is zero.
    return x+STAMP_RADIUS >= 52 and x-STAMP_RADIUS <= 76 and y+STAMP_RADIUS >= 52 and y-STAMP_RADIUS <= 76


def prepare_components(history129, prior_centers_xy, predicted_offset_xy, polarity):
    """Same primary arrays as V42, plus prior-only template_history records.

    template_history.moving and each entry in template_history.fixed contain
    normalized_stamps (K,17,17), prior_history_indices (oldest=0/newest=7), and
    raw weighted energies (DN squared) / L2 norms (DN). Fixed records correspond
    exactly, in order, to fixed_templates. No current image enters this API.
    """
    base = prepare_v42_components(history129, prior_centers_xy, predicted_offset_xy, polarity)
    history = _array(history129, HISTORY_SHAPE, "history129")
    centers, offset = _coordinates(prior_centers_xy, predicted_offset_xy)
    sign = 1.0 if polarity == "bright" else -1.0
    y, x = np.indices(PATCH_SHAPE)
    raw_highpasses = np.stack([sign*_box_highpass(frame) for frame in history])
    dependency_safe_highpasses, eligible_stamps = [], []
    excluded_source_counts = []
    missing = []
    for index, center in enumerate(centers):
        source = history[index].copy()
        if center is None:
            source[:] = np.nan
            missing.append(index)
            excluded_source_counts.append(int(source.size))
        else:
            excluded = np.maximum(np.abs(x-center[0]), np.abs(y-center[1])) <= EXCLUSION_RADIUS
            excluded_source_counts.append(int(excluded.sum()))
            source[excluded] = np.nan
        safe_highpass = sign*_box_highpass(source)
        dependency_safe_highpasses.append(safe_highpass)
        eligible_stamps.append(_full_stamp_eligible(safe_highpass))
    dependency_safe_highpasses = np.stack(dependency_safe_highpasses)
    eligible_stamps = np.stack(eligible_stamps)
    original_peaks = _raw_peak_masks(raw_highpasses)
    eligible_peaks = original_peaks & eligible_stamps
    # Only peaks already present in the original highpass enter this map. In
    # particular, deleting an unsafe brighter neighbour cannot promote a new
    # local maximum. Reusing V42 clustering preserves its repeat and cap math.
    peak_only_maps = np.where(eligible_peaks, raw_highpasses, np.nan)
    raw_discovered = _anchors(raw_highpasses)
    discovered = _anchors(peak_only_maps)

    fixed, anchors, fixed_histories = [], [], []
    for anchor in discovered[:MAX_FIXED_ANCHORS]:
        ax, ay = anchor["x"], anchor["y"]
        stamps, indices, raw_stamps = [], [], []
        for index, highpass in enumerate(dependency_safe_highpasses):
            if not eligible_stamps[index, ay, ax]:
                continue
            raw = _stamp(highpass, (ax, ay))
            stamp = _positive_unit_stamp(raw)
            if stamp is not None:
                stamps.append(stamp)
                raw_stamps.append(raw)
                indices.append(index)
        combined = _combine_stamps(stamps)
        record = dict(anchor, usable_stamp_count=len(stamps), template_available=combined is not None,
                      eligible_stamp_history_indices=np.flatnonzero(eligible_stamps[:, ay, ax]).tolist(),
                      usable_stamp_history_indices=indices)
        anchors.append(record)
        if combined is not None:
            fixed.append(_place(combined, (ax, ay)))
            fixed_histories.append(_stamp_record(stamps, indices, raw_stamps, (ax, ay)))

    # Expose moving-template empirical history without changing V42 outputs.
    moving_stamps, moving_indices, moving_raw = [], [], []
    for index, center in enumerate(centers):
        if center is None:
            continue
        raw = _stamp(sign*(history[index]-base["background"]), center)
        stamp = _positive_unit_stamp(raw)
        if stamp is not None:
            moving_stamps.append(stamp)
            moving_raw.append(raw)
            moving_indices.append(index)

    opportunity_counts = eligible_stamps[:, 44:85, 44:85].sum(axis=0).astype(np.uint8)
    incomplete = opportunity_counts < MIN_ANCHOR_FRAMES
    # Every seed in the frozen +/-20 region has at least boundary support
    # overlapping the 25-square tested core. Keep the explicit overlap test.
    overlap = np.asarray([[_overlaps_core(xx, yy) for xx in range(44, 85)] for yy in range(44, 85)])
    incomplete_overlap = incomplete & overlap
    removed = []
    for anchor in raw_discovered:
        ax, ay = anchor["x"], anchor["y"]
        near = eligible_peaks[:, ay-2:ay+3, ax-2:ax+3].any(axis=(1, 2))
        repeat_count = int(near.sum())
        stamp_count = int(eligible_stamps[:, ay, ax].sum())
        represented = any(max(abs(other["x"]-ax), abs(other["y"]-ay)) <= ANCHOR_REPEAT_RADIUS
                          for other in discovered)
        reasons = []
        if repeat_count < MIN_ANCHOR_FRAMES:
            reasons.append("fewer_than_three_dependency_safe_repeat_observations")
        if stamp_count < MIN_ANCHOR_FRAMES:
            reasons.append("fewer_than_three_dependency_safe_stamp_opportunities")
        if not represented:
            reasons.append("not_represented_by_safe_repeat_anchor")
        for retained in anchors:
            if (retained["x"], retained["y"]) == (ax, ay) and not retained["template_available"]:
                reasons.append("insufficient_uncontaminated_positive_template_history")
        if reasons:
            removed.append(dict(anchor_xy=[ax, ay], raw_repeat_frame_count=anchor["repeat_frame_count"],
                                dependency_safe_repeat_frame_count=repeat_count,
                                dependency_safe_stamp_frame_count=stamp_count,
                                represented_by_safe_repeat_anchor=represented,
                                overlaps_current_core=_overlaps_core(ax, ay), reasons=reasons))
    unresolved = any(item["overlaps_current_core"] for item in removed)
    ambiguity = []
    if incomplete_overlap.any():
        ambiguity.append("fixed_alternative_prior_coverage_incomplete")
    if unresolved:
        ambiguity.append("foreground_excluded_raw_fixed_alternative_unresolved")
    predicted = 64.0+offset
    overlapping = [[item["x"], item["y"]] for item in discovered
                   if max(abs(item["x"]-predicted[0]), abs(item["y"]-predicted[1])) <= ANCHOR_REPEAT_RADIUS]
    reasons = [reason for reason in base["metadata"]["causal_component_reasons"]
               if reason != "incomplete_persistent_anchor_template"]
    if any(not item["template_available"] for item in anchors):
        reasons.append("incomplete_persistent_anchor_template")
    metadata = dict(base["metadata"])
    metadata.update(
        persistent_anchor_count_before_cap=len(discovered),
        persistent_anchor_centres_near_prediction=overlapping,
        persistent_anchor_overlaps_prediction=bool(overlapping),
        fixed_anchor_count=len(fixed), anchors=anchors,
        anchor_dictionary_truncated=len(discovered) > MAX_FIXED_ANCHORS,
        causal_component_reasons=reasons,
        fixed_learning_version="v43_prior_actual_foreground_and_full_dependency_exclusion",
        fixed_learning_uses_current_image=False,
        background_and_moving_templates_unchanged_from_v42=True,
        missing_actual_position_history_indices=missing,
        foreground_excluded_source_pixel_counts=excluded_source_counts,
        raw_peak_observation_counts=original_peaks.sum(axis=(1, 2)).tolist(),
        dependency_safe_peak_observation_counts=eligible_peaks.sum(axis=(1, 2)).tolist(),
        raw_persistent_anchor_count_before_exclusion=len(raw_discovered),
        fixed_alternative_coverage_incomplete=bool(incomplete_overlap.any()),
        unresolved_removed_fixed_alternative=unresolved,
        fixed_learning_ambiguity_reasons=ambiguity,
        removed_raw_fixed_alternatives=removed,
        fixed_seed_opportunity_coverage=dict(
            seed_x_min=44, seed_x_max=84, seed_y_min=44, seed_y_max=84,
            required_history_frames=MIN_ANCHOR_FRAMES,
            minimum=int(opportunity_counts.min()), maximum=int(opportunity_counts.max()),
            sufficient_seed_count=int((~incomplete).sum()), insufficient_seed_count=int(incomplete.sum()),
            insufficient_overlapping_core_seed_count=int(incomplete_overlap.sum()),
            per_history_eligible_seed_count=eligible_stamps[:, 44:85, 44:85].sum(axis=(1, 2)).tolist(),
            counts_row_major_uint8_sha256=hashlib.sha256(opportunity_counts.tobytes()).hexdigest()),
        fixed_dependency_contract=dict(foreground_chebyshev_radius=8, highpass_native_radius=4,
                                       fixed_stamp_radius=8, total_native_dependency_radius=12,
                                       all_stamp_samples_required=True,
                                       original_local_maxima_computed_before_eligibility=True,
                                       missing_actual_center_excludes_entire_fixed_learning_frame=True,
                                       sufficient_opportunities_do_not_establish_fixed_source_absence=True),
        template_history_contract=dict(prior_history_index_zero="oldest", prior_history_index_seven="newest",
                                       energies_units="DN squared before unit normalization",
                                       l2_norm_units="DN before unit normalization",
                                       empirical_only_not_an_uncertainty_guarantee=True),
    )
    result = dict(base)
    result.update(metadata=metadata,
                  fixed_templates=np.stack(fixed) if fixed else np.empty((0, *PATCH_SHAPE)),
                  template_history=dict(moving=_stamp_record(moving_stamps, moving_indices, moving_raw),
                                        fixed=fixed_histories))
    return result
