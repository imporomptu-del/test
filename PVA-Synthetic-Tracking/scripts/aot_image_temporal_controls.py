"""Frozen generated image trajectories; no actual media or filesystem IO.

Each current crop is rendered independently from native analytic coordinates.
The continuous oracle is evaluated directly, never through the tested remapper.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

CAMERAS = ('identity', 'translation', 'affine')
SIGMAS = (.6, 1.2)
AMPLITUDES = (-8., 8., -32., 32.)
PHASES = ((0., 0.), (.25, .25), (.5, .5))
FRAME_COUNT = 9
CENTER = np.array([1224., 1024.])
SOURCE_ORIGIN = (1160, 960)
SOURCE_SHAPE = (128, 128)
REFERENCE_ROI = (1192, 992, 1257, 1057)
PADDED_ORIGIN = (1190, 990)
PADDED_SHAPE = (69, 69)
EPSILON = 1e-12
NUMERICAL_TOLERANCE_DN = 1e-4


def native_grid(shape, origin):
    y, x = np.indices(shape, dtype=np.float64)
    return np.stack((x+origin[0], y+origin[1]), axis=-1)


def camera(kind, t):
    if kind not in CAMERAS or type(t) is not int or not 0<=t<FRAME_COUNT:
        raise ValueError('unplanned camera/frame')
    linear = np.eye(2, dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    if kind!='identity':
        translation = np.array([.25*t, -.125*t])
    if kind=='affine':
        linear = np.array([[1+.001*t, .002*t], [-.001*t, 1-.0005*t]])
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = CENTER-linear@CENTER+translation
    return matrix


def forward(points, matrix):
    return np.asarray(points, dtype=np.float64)@matrix[:2, :2].T+matrix[:2, 2]


def continuous_psf(points, center, sigma, amplitude):
    """Independent circular current-domain Gaussian, with fixed 6sigma cutoff."""
    squared = np.sum((np.asarray(points, dtype=np.float64)-center)**2, axis=-1)
    return np.where(squared <= (6*sigma)**2, amplitude*np.exp(-squared/(2*sigma*sigma)), 0.)


def render_original(kind, t, sigma, amplitude, phase, static=False):
    """Render from analytic truth at t, never from a preceding warped image."""
    transform = camera(kind, t)
    position = np.array([1222., 1024.])+phase
    if not static:
        position += [.5*t, .0625*t*t]
    target_current = forward(position, transform)
    current_grid = native_grid(SOURCE_SHAPE, SOURCE_ORIGIN)
    reference_grid = (current_grid-transform[:2, 2])@np.linalg.inv(transform[:2, :2]).T
    texture = 128+12*np.sin(reference_grid[..., 0]/17)+9*np.cos(reference_grid[..., 1]/23)
    clean = np.clip(np.rint(texture), 0, 255).astype(np.uint8)
    intended = continuous_psf(current_grid, target_current, sigma, amplitude)
    before_clip = np.rint(clean.astype(np.float64)+intended)
    injected = np.clip(before_clip, 0, 255).astype(np.uint8)
    effective = injected.astype(np.float64)-clean
    stats = dict(intended_signed_peak_dn=float(np.sign(amplitude)*np.max(np.sign(amplitude)*intended)),
        realized_signed_peak_dn=float(np.sign(amplitude)*np.max(np.sign(amplitude)*effective)),
        intended_positive_mass_dn=float(np.maximum(intended, 0).sum()),
        intended_negative_mass_dn=float(np.maximum(-intended, 0).sum()),
        realized_positive_mass_dn=float(np.maximum(effective, 0).sum()),
        realized_negative_mass_dn=float(np.maximum(-effective, 0).sum()),
        clipped_pixel_count=int(np.count_nonzero((before_clip<0)|(before_clip>255))),
        quantization_difference_l1_dn=float(np.sum(abs(effective-intended))),
        quantization_difference_l2_dn=float(np.linalg.norm(effective-intended)))
    return transform, position, target_current, clean, injected, stats


def independent_stats(array, valid, polarity):
    valid = np.asarray(valid, bool)
    values = np.asarray(array, dtype=np.float64)[valid]
    if not np.isfinite(values).all():
        raise ValueError('nonfinite supported generated observation')
    if not len(values):
        return dict(count=0, polarity_peak_dn=None, peak_xy=None, positive_mass_dn=None,
            negative_mass_dn=None, l1_dn=None, energy_dn_squared=None, rms_dn=None, centroid_xy=None)
    coordinates = native_grid(valid.shape, REFERENCE_ROI[:2])[valid]
    normalized = polarity*values
    weights = np.maximum(normalized, 0)
    mass = float(weights.sum())
    centroid = None if mass<=EPSILON else (np.sum(coordinates*weights[:, None], axis=0)/mass).tolist()
    peak = float(np.max(normalized))
    return dict(count=len(values), polarity_peak_dn=peak,
        peak_xy=None if peak<=EPSILON else coordinates[int(np.argmax(normalized))].tolist(),
        positive_mass_dn=float(np.maximum(values, 0).sum()), negative_mass_dn=float(np.maximum(-values, 0).sum()),
        l1_dn=float(np.sum(abs(values))), energy_dn_squared=float(values@values),
        rms_dn=float(np.sqrt(np.mean(values*values))), centroid_xy=centroid)


def comparison(core, measured, oracle, valid, polarity):
    actual = independent_stats(measured, valid, polarity)
    ideal = independent_stats(oracle, valid, polarity)
    record = dict(observed=actual, continuous_oracle=ideal,
        core_observed=core.array_metrics(measured, valid, origin_xy=REFERENCE_ROI[:2], polarity=polarity, template=oracle),
        core_oracle=core.array_metrics(oracle, valid, origin_xy=REFERENCE_ROI[:2], polarity=polarity, template=oracle),
        peak_amplitude_error_dn=None, peak_position_error_px=None, energy_error_dn_squared=None,
        energy_ratio=None, oracle_rmse_dn=None, oracle_max_abs_error_dn=None,
        centroid_error_xy=None, centroid_error_px=None)
    if not actual['count']:
        return record
    difference = np.asarray(measured, float)[valid]-np.asarray(oracle, float)[valid]
    record.update(peak_amplitude_error_dn=actual['polarity_peak_dn']-ideal['polarity_peak_dn'],
        energy_error_dn_squared=actual['energy_dn_squared']-ideal['energy_dn_squared'],
        energy_ratio=None if ideal['energy_dn_squared']<=EPSILON else actual['energy_dn_squared']/ideal['energy_dn_squared'],
        oracle_rmse_dn=float(np.sqrt(np.mean(difference*difference))),
        oracle_max_abs_error_dn=float(np.max(abs(difference))))
    # A zero/near-zero oracle has no meaningful peak or centroid location.
    if ideal['peak_xy'] is not None and actual['peak_xy'] is not None:
        record['peak_position_error_px'] = float(np.linalg.norm(np.asarray(actual['peak_xy'])-ideal['peak_xy']))
    if actual['centroid_xy'] is not None and ideal['centroid_xy'] is not None:
        error = np.asarray(actual['centroid_xy'])-ideal['centroid_xy']
        record.update(centroid_error_xy=error.tolist(), centroid_error_px=float(np.linalg.norm(error)))
    return record


def trajectory(core, kind, sigma, amplitude, phase, *, dropout=False, static=False):
    if static and (kind!='identity' or dropout):
        raise ValueError('static control must use identity and complete support')
    padded = native_grid(PADDED_SHAPE, PADDED_ORIGIN)
    reference = native_grid((65, 65), REFERENCE_ROI[:2])
    frames, adjacent, previous = [], [], None
    static_hashes = None
    for t in range(FRAME_COUNT):
        transform, target_reference, target_current, clean, injected, injection = render_original(kind, t, sigma, amplitude, phase, static)
        identities = dict(clean=hashlib.sha256(clean.tobytes()).hexdigest(), injected=hashlib.sha256(injected.tobytes()).hexdigest())
        if static:
            if static_hashes is None: static_hashes = identities
            if identities!=static_hashes: raise ValueError('static originals are not repeated exactly')
        support_present = not (dropout and t in (3, 4))
        q = forward(padded, transform)
        support = np.full(PADDED_SHAPE, support_present, bool)
        field = core.field_from_maps(q[..., 0], q[..., 1], model_support=support,
            source_shape=(2048, 2448), erosion_px=2, origin_xy=PADDED_ORIGIN)
        field = core.field_roi(field, REFERENCE_ROI)
        valid = np.asarray(field['valid'], bool)
        if valid.shape!=(65, 65) or bool(np.all(valid))!=support_present or bool(np.any(valid))!=support_present:
            raise ValueError('generated full-support/dropout routing differs')
        # Each source is an original independently rendered frame, sampled once.
        aligned_clean = core.pull(clean, field, origin_xy=SOURCE_ORIGIN)
        aligned_injected = core.pull(injected, field, origin_xy=SOURCE_ORIGIN)
        if not np.array_equal(np.isnan(aligned_clean), ~valid) or not np.array_equal(np.isnan(aligned_injected), ~valid):
            raise ValueError('generated invalid samples are not explicit NaNs')
        if identities!=dict(clean=hashlib.sha256(clean.tobytes()).hexdigest(), injected=hashlib.sha256(injected.tobytes()).hexdigest()):
            raise ValueError('original generated source modified')
        isolated = aligned_injected-aligned_clean
        # Current PSF stays circular in current sensor coordinates. Pulling an
        # affine field therefore produces an anisotropic reference-domain oracle.
        oracle = continuous_psf(forward(reference, transform), target_current, sigma, amplitude)
        oracle[~valid] = np.nan
        measured = comparison(core, isolated, oracle, valid, float(np.sign(amplitude)))
        observed_centroid = measured['observed']['centroid_xy']
        measured['centroid_error_to_continuous_target_px'] = (None if observed_centroid is None else
            float(np.linalg.norm(np.asarray(observed_centroid)-target_reference)))
        frame = dict(frame_index=t, available=support_present, valid_pixel_count=int(valid.sum()),
            target_reference_xy=target_reference.tolist(), target_current_xy=target_current.tolist(),
            previous_to_current_matrix=transform.tolist(), original_source_sha256=identities,
            injection=injection, current_target_increment=measured)
        frames.append(frame)
        if previous is not None:
            common = previous['valid'] & valid
            injected_difference = aligned_injected-previous['injected']
            clean_difference = aligned_clean-previous['clean']
            target_difference = injected_difference-clean_difference
            oracle_difference = oracle-previous['oracle']
            pair = dict(previous_frame=t-1, current_frame=t, available=bool(common.any()),
                common_pixel_count=int(common.sum()),
                isolated_target_residual=comparison(core, target_difference, oracle_difference, common, float(np.sign(amplitude))),
                injected_adjacent_residual=core.array_metrics(injected_difference, common, origin_xy=REFERENCE_ROI[:2], polarity=float(np.sign(amplitude)), template=oracle_difference),
                clean_adjacent_residual=core.array_metrics(clean_difference, common, origin_xy=REFERENCE_ROI[:2], polarity=float(np.sign(amplitude)), template=oracle_difference),
                observed_centroid_step_xy=None, oracle_centroid_step_xy=None, centroid_step_error_px=None)
            left = previous['measurement']
            centroids = [left['observed']['centroid_xy'], measured['observed']['centroid_xy'],
                         left['continuous_oracle']['centroid_xy'], measured['continuous_oracle']['centroid_xy']]
            if common.any() and all(c is not None for c in centroids):
                observed_step = np.asarray(centroids[1])-centroids[0]
                oracle_step = np.asarray(centroids[3])-centroids[2]
                pair.update(observed_centroid_step_xy=observed_step.tolist(), oracle_centroid_step_xy=oracle_step.tolist(),
                    centroid_step_error_px=float(np.linalg.norm(observed_step-oracle_step)))
            if static and (not common.all() or max(float(np.max(abs(v[common]))) for v in
                    (injected_difference, clean_difference, target_difference, oracle_difference))>NUMERICAL_TOLERANCE_DN):
                raise ValueError('static repeated-frame numerical cancellation failed')
            adjacent.append(pair)
        # Retain the immediately preceding frame even when unavailable: this
        # prevents accidentally connecting frame2 directly to recovery frame5.
        previous = dict(valid=valid, clean=aligned_clean, injected=aligned_injected, oracle=oracle, measurement=measured)
    expected_missing = 2 if dropout else 0
    if sum(not f['available'] for f in frames)!=expected_missing or sum(not a['available'] for a in adjacent)!=(3 if dropout else 0):
        raise ValueError('generated missing-frame/adjacency accounting differs')
    return dict(camera=kind, sigma_px=sigma, signed_peak_dn=amplitude, initial_phase_xy=list(phase),
        dropout=dropout, static=static, frame_count=len(frames), missing_frames=expected_missing,
        adjacent_pair_count=len(adjacent), missing_adjacent_pairs=sum(not a['available'] for a in adjacent),
        original_source_crop_xyxy=[1160, 960, 1288, 1088], reference_roi_xyxy=list(REFERENCE_ROI),
        frames=frames, adjacent=adjacent,
        interpretation='Generated original-image trajectory; metrics are descriptive, not an image-quality acceptance threshold.')


def run_generated(core):
    trajectories = [trajectory(core, kind, sigma, amplitude, phase)
        for kind, sigma, amplitude, phase in itertools.product(CAMERAS, SIGMAS, AMPLITUDES, PHASES)]
    dropouts = [trajectory(core, kind, .6, 16., (0., 0.), dropout=True) for kind in CAMERAS]
    static = [trajectory(core, 'identity', sigma, amplitude, (0., 0.), static=True)
        for sigma, amplitude in itertools.product(SIGMAS, AMPLITUDES)]
    if (len(trajectories), len(dropouts), len(static))!=(72, 3, 8):
        raise ValueError('generated temporal inventory differs')
    return dict(schema='seaqr.aot.image-temporal-controls.v1', passed=True,
        passed_interpretation='Numerical conformance/inventory completion only; no preservation-quality threshold.',
        numerical_static_cancellation_tolerance_dn=NUMERICAL_TOLERANCE_DN,
        reference_target_path='(1222+0.5t,1024+0.0625t^2)+initial_phase, t=0..8; static controls omit t terms',
        oracle='Independent continuous circular current-domain PSF at exact float64 F_t(p); not passed through remap.',
        source_generation='Every original U8 crop independently rendered from analytic inverse-mapped background, then quantized PSF injection; no recursive warp.',
        source_images_used=False, detector_run=False, production_changed=False,
        trajectories=trajectories, support_dropouts=dropouts, static_repeats=static,
        counts=dict(trajectories=72, trajectory_frames=648, dropout_trajectories=3, dropout_frames=27,
            dropout_missing_frames=6, dropout_missing_adjacent_pairs=9, static_trajectories=8, static_frames=72,
            total_frames=747, total_adjacent_pairs=664),
        limitations='Generated temporal data are not consecutive actual AOT pairs, estimator-contamination tests, target identities, detector recall or validated sensor PSFs.')
