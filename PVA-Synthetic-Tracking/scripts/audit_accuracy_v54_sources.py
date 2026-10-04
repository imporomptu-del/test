"""Standalone independent audit of generated V54 source-preservation evidence.

No producer, adapter, model or earlier auditor is imported. All reads are literal
generated-run artifacts or the eleven frozen source/test/plan paths below.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/tiny_target/accuracy_v54_20260926'
SOURCE_NAMES = (
    'docs/accuracy_v54_plan.md', 'scripts/accuracy_v54_benchmark.py',
    'scripts/accuracy_v54_adapter.py', 'scripts/run_accuracy_v54_sources.py',
    'scripts/audit_accuracy_v54_sources.py', 'tests/unit/test_accuracy_v54_benchmark.py',
    'tests/unit/test_accuracy_v54_adapter.py', 'tests/unit/test_accuracy_v54_runner.py',
    'tests/unit/test_accuracy_v54_audit.py', 'scripts/accuracy_v53_offset.py',
    'tests/unit/test_accuracy_v53_offset.py',
)
INHERITED_SHA256 = {
    'scripts/accuracy_v53_offset.py': 'aa490ecf8305dd0c5f3facff83a5fa8fef43c67460eb55572f516b298a079d04',
    'tests/unit/test_accuracy_v53_offset.py': 'd5c53052fca8f47bf1f380582702e62217a353587b906fe759d564b460e70f05',
}
SPLITS = ('left_right', 'checkerboard')
METHODS = ('median8', 'median3', 'offset_left_right_fold0', 'offset_left_right_fold1',
           'offset_checkerboard_fold0', 'offset_checkerboard_fold1')
GUARD_XY = [(x, y) for y in range(8, 121, 8) for x in range(8, 121, 8)
            if 40 <= max(abs(x-64), abs(y-64)) <= 56]
CORE_XY = [(x, y) for y in range(52, 77) for x in range(52, 77)]
RUN_NAMES = ('freeze.json', 'inputs.jsonl', 'predictions.jsonl', 'predictions_frozen.json', 'scores.jsonl', 'summary.json')
CONDITIONS = ('stable', 'recent_plus8', 'recent_minus8', 'ended_short_plus8', 'ended_long_plus8',
              'local_guard_plus16', 'local_guard_minus16', 'all_guard_plus8')
MOTIONS = ('appearing', 'stationary', 'slow_linear', 'linear', 'turning', 'move_stop')
MISSINGNESS = ('guard_current_left_missing', 'guard_current_all_missing',
               'core_first_prior_center_missing', 'core_current_center_missing')


def specifications():
    result = []

    def spec(condition, motion, amplitude, background, level, seed, missing=None):
        amp_label = '0' if amplitude == 0 else 'p4' if amplitude > 0 else 'n4'
        noise_label = '0' if level == 0 else '0p5'
        case_id = (f'{condition}_{motion}_a{amp_label}_{background}_noise{noise_label}_seed{seed}'
                   if missing is None else 'availability_'+missing)
        return dict(schema='accuracy_v54_generated_benchmark_v1', case_id=case_id,
            stratum='factorial' if missing is None else 'availability', condition=condition,
            motion=motion, amplitude=amplitude, background=background, noise_level=level, seed=seed,
            missingness=missing, history_length=8, guard_point_count=144, core_point_count=625,
            analytic_simulation_only=True, physical_sensor_model=False)

    sources = [('absent', 0)]+[(motion, amplitude) for motion in MOTIONS for amplitude in (4, -4)]
    for condition in CONDITIONS:
        for motion, amplitude in sources:
            for background in ('constant', 'textured'):
                for level, seed in ((0, 71), (.5, 991)):
                    result.append(spec(condition, motion, amplitude, background, level, seed))
    for missing in MISSINGNESS:
        result.append(spec('stable', 'appearing', 4, 'textured', 0, 71, missing))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value, exclude=()):
    if exclude:
        value = {k: v for k, v in value.items() if k not in exclude}
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def compare(expected, actual, context='root', *, rtol=1e-10, atol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), context+': keys differ')
        for key, value in expected.items():
            compare(value, actual[key], context+'.'+str(key), rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(expected) == len(actual), context+': length differs')
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{context}[{index}]', rtol=rtol, atol=atol)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol), context+': numeric value differs')
    else:
        require(type(expected) is type(actual) and expected == actual, context+': value/type differs')


def raw_file(path):
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'Canonical regular file required: '+str(path))
    return path.read_bytes()


def file_sha(path):
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'Canonical regular file required: '+str(path))
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_read(path, expected_hash, bindings):
    raw = raw_file(path)
    require(hashlib.sha256(raw).hexdigest() == expected_hash, 'File hash differs: '+str(path))
    bindings[str(path)] = expected_hash
    return ([json.loads(line) for line in raw.splitlines()]
            if path.name.endswith('.jsonl') else json.loads(raw))


def verify_inherited_sources(root):
    for name, expected in INHERITED_SHA256.items():
        require(hashlib.sha256(raw_file(root/name)).hexdigest() == expected, 'Inherited V53 source changed: '+name)


def offset_constants():
    return dict(loss='absolute_deviation', fixed_gain=1.0, fixed_x_slope=0.0, fixed_y_slope=0.0,
        minimum_finite_training_rows=4, guard_coordinate_min=8, guard_coordinate_max=120,
        guard_coordinate_step=8, guard_center=64, guard_chebyshev_radius_min=40,
        guard_chebyshev_radius_max=56,
        median_rule='correctly_rounded_exact_rational_midpoint_of_middle_residuals',
        objective_rule='correctly_rounded_exact_mean_of_finite_absolute_deviations',
        optimality_rule='zero_in_mean_L1_subgradient_interval_from_exact_order_counts',
        clipping_or_tuning=False, implicit_fallback=False, arithmetic_failure_discards_entire_fit=True)


def reconstructed_fit(xy, slow, current):
    xy = np.asarray(xy, dtype=float)
    slow, current = np.asarray(slow, dtype=float), np.asarray(current, dtype=float)
    require(xy.shape == (len(slow), 2) and current.shape == slow.shape == (len(xy),), 'Training shape differs')
    used = [math.isfinite(s) and math.isfinite(c) for s, c in zip(slow, current)]
    n = sum(used)
    result = dict(schema_version=1, loss='median_offset', available=False, unavailable_reason=None,
        training_count=len(xy), training_used_count=n, training_used_mask=used,
        training_unavailable_reasons=[None if selected else 'nonfinite_training_slow_or_current' for selected in used],
        training_input_sha256=digest(dict(points_xy=xy, slow=slow, current=current)),
        offset_dn=None, candidate_offset_dn=None, median_interval_dn=[None, None], objective_mae_dn=None,
        subgradient_interval=[None, None], residual_order_counts=None, constants=offset_constants())

    def finish(reason=None):
        result['unavailable_reason'] = reason
        result['model_sha256'] = digest(result)
        return result

    if n < 4:
        return finish('insufficient_finite_training_rows')
    residuals = [float(c)-float(s) for s, c, selected in zip(slow, current, used) if selected]
    if not all(math.isfinite(v) for v in residuals):
        return finish('nonfinite_training_residual_arithmetic')
    ordered = sorted(residuals)
    lower, upper = ordered[(n-1)//2], ordered[n//2]
    candidate = float((Fraction(lower)+Fraction(upper))/2)
    if not math.isfinite(candidate) or not lower <= candidate <= upper:
        return finish('nonfinite_or_outside_median_interval')
    below = sum(v < candidate for v in residuals)
    above = sum(v > candidate for v in residuals)
    tied = sum(v == candidate for v in residuals)
    interval = [(below-above-tied)/n, (below-above+tied)/n]
    result.update(candidate_offset_dn=candidate, median_interval_dn=[lower, upper],
        subgradient_interval=interval, residual_order_counts=dict(below=below, above=above, tied=tied))
    if below+above+tied != n or not interval[0] <= 0 <= interval[1]:
        return finish('median_subgradient_certificate_failed')
    deviations = [abs(v-candidate) for v in residuals]
    if not all(math.isfinite(v) for v in deviations):
        return finish('nonfinite_objective_arithmetic')
    objective = float(sum(map(Fraction, deviations), Fraction())/n)
    if not math.isfinite(objective):
        return finish('nonfinite_objective_arithmetic')
    result.update(available=True, offset_dn=candidate, objective_mae_dn=objective)
    return finish()


def reconstructed_crossfit(points, slow, current, split):
    require(split in SPLITS, 'Unknown split')
    xy, slow, current = np.asarray(points, dtype=float), np.asarray(slow, dtype=float), np.asarray(current, dtype=float)
    require(xy.shape == (len(slow), 2) and current.shape == slow.shape == (len(xy),), 'Guard input shape differs')
    ids = ((xy[:, 0] >= 64).astype(int) if split == 'left_right'
           else ((xy[:, 0]//8+xy[:, 1]//8) % 2).astype(int))
    fits, values, availability, reasons = {}, [None]*len(xy), [False]*len(xy), [None]*len(xy)
    for fold in (0, 1):
        heldout = ids == fold
        fit = reconstructed_fit(xy[~heldout], slow[~heldout], current[~heldout])
        fits[str(fold)] = fit
        for index in np.flatnonzero(heldout):
            if not fit['available']:
                reasons[index] = 'fit_unavailable:'+fit['unavailable_reason']
            elif not math.isfinite(slow[index]):
                reasons[index] = 'nonfinite_prediction_slow'
            else:
                value = float(slow[index])+fit['offset_dn']
                if math.isfinite(value):
                    values[index], availability[index] = value, True
                else:
                    reasons[index] = 'nonfinite_prediction_arithmetic'
    result = dict(schema_version=1, split=split, total_count=len(xy), fold_id=ids.tolist(),
        fits={'median_offset': fits}, predictions={'median_offset': dict(values=values, available=availability,
            unavailable_reasons=reasons, model_sha256=None)}, constants=offset_constants(),
        metadata=dict(current_complementary_guard_values_used=True, heldout_current_argument_accepted_by_predict=False,
            fit_dictionary_key_names_heldout_fold=True, core_pixels_accepted=False,
            scoring_or_forecast_selection_performed=False, unavailable_indices_preserved_without_fallback=True,
            prior_only_or_online_camera_causality_certified=False, guard_purity_or_guard_to_core_transfer_certified=False,
            production_detection_modified=False))
    result['crossfit_sha256'] = digest(result)
    return result


def base_values(points, background):
    xy = np.asarray(points, dtype=float)
    dx, dy = xy[:, 0]-64, xy[:, 1]-64
    if background == 'constant':
        return np.full(len(xy), 96.)
    require(background == 'textured', 'Unknown analytic background')
    return (96.+.05*dx+.03*dy+12*np.sin(dx/12)+9*np.cos(dy/15)
            +6*np.sin((dx+dy)/17))


def source_center(motion, time):
    if motion == 'absent' or (motion == 'appearing' and time < 0):
        return None
    if motion in ('appearing', 'stationary'):
        return 64., 64.
    if motion == 'slow_linear':
        return 64.+.25*time, 64.
    if motion == 'linear':
        return 64.+time, 64.
    if motion == 'turning':
        return (64.+time+4, 60.) if time <= -4 else (64., 64.+time)
    require(motion == 'move_stop', 'Unknown source motion')
    return 64.+min(time+3, 0), 64.


def template(points, center):
    if center is None:
        return np.zeros(len(points))
    xy = np.asarray(points, dtype=float)
    return np.maximum(1-abs(xy[:, 0]-center[0])/2, 0)*np.maximum(1-abs(xy[:, 1]-center[1])/2, 0)


def generated_input(spec):
    guard, core = np.asarray(GUARD_XY), np.asarray(CORE_XY)
    guard_series = np.tile(base_values(guard, spec['background']), (9, 1))
    core_background = np.tile(base_values(core, spec['background']), (9, 1))
    condition = spec['condition']
    if condition in ('recent_plus8', 'recent_minus8'):
        amount = 8 if condition == 'recent_plus8' else -8
        guard_series[-3:] += amount
        core_background[-3:] += amount
    elif condition == 'ended_short_plus8':
        guard_series[-3:-1] += 8
        core_background[-3:-1] += 8
    elif condition == 'ended_long_plus8':
        guard_series[:-1] += 8
        core_background[:-1] += 8
    contamination = np.zeros(144)
    if condition in ('local_guard_plus16', 'local_guard_minus16'):
        contamination[(guard[:, 0] >= 64) & (guard[:, 1] >= 64)] = 16 if condition == 'local_guard_plus16' else -16
    elif condition == 'all_guard_plus8':
        contamination[:] = 8
    else:
        require(condition in ('stable', 'recent_plus8', 'recent_minus8', 'ended_short_plus8', 'ended_long_plus8'),
                'Unknown analytic condition')
    rng = np.random.default_rng(spec['seed'])
    guard_noise = rng.uniform(-spec['noise_level'], spec['noise_level'], (9, 144))
    core_noise = rng.uniform(-spec['noise_level'], spec['noise_level'], (9, 625))
    guard_series[-1] += contamination
    guard_series += guard_noise
    off = core_background+core_noise
    on = off.copy()
    for index, time in enumerate(range(-8, 1)):
        center = source_center(spec['motion'], time)
        on[index] += spec['amplitude']*template(core, center)
        require(np.count_nonzero(template(guard, center)) == 0, 'Generated source reaches guard')
    missing = spec['missingness']
    if missing == 'guard_current_left_missing':
        guard_series[-1, guard[:, 0] < 64] = np.nan
    elif missing == 'guard_current_all_missing':
        guard_series[-1] = np.nan
    elif missing == 'core_first_prior_center_missing':
        on[0, 312] = off[0, 312] = np.nan
    elif missing == 'core_current_center_missing':
        on[-1, 312] = off[-1, 312] = np.nan
    else:
        require(missing is None, 'Unknown missingness condition')
    unit = template(core, (64., 64.))
    return plain(dict(case_id=spec['case_id'], spec=spec,
        observed=dict(guard_xy=guard, core_xy=core, guard_history=guard_series[:-1],
            guard_current=guard_series[-1], core_history_on=on[:-1], core_history_off=off[:-1],
            core_current_on=on[-1], core_current_off=off[-1]),
        truth=dict(clean_current_background=core_background[-1], current_source=spec['amplitude']*unit,
                   source_template=unit, guard_contamination_current=contamination)))


def check_generated_input(spec, row):
    expected = generated_input(spec)
    compare(expected, row, 'independent generated input', rtol=0, atol=1e-12)
    # Exactly representable tent/source/contamination values determine support.
    # An epsilon must not turn a zero template location into a required pixel.
    for field in ('source_template', 'current_source', 'guard_contamination_current'):
        compare(expected['truth'][field], row['truth'][field], 'exact analytic truth.'+field, rtol=0, atol=0)


def strict_median(history, count):
    values = np.asarray(history, dtype=float)[-count:]
    require(values.ndim == 2 and len(values) == count, 'Median history shape differs')
    predictions, available, reasons = [], [], []
    for column in values.T:
        if not np.isfinite(column).all():
            predictions.append(None); available.append(False); reasons.append('nonfinite_history')
        else:
            ordered = sorted(float(v) for v in column)
            n = len(ordered)
            with np.errstate(over='ignore', invalid='ignore'):
                value = (ordered[n//2] if n % 2 else
                         float(np.float64(ordered[n//2-1])+np.float64(ordered[n//2]))/2)
            good = math.isfinite(value)
            predictions.append(value if good else None); available.append(good)
            reasons.append(None if good else 'nonfinite_median_arithmetic')
    return dict(values=predictions, available=available, unavailable_reasons=reasons)


def offset_prediction(median, fit):
    values, available, reasons = [], [], []
    for value, good in zip(median['values'], median['available']):
        if not fit['available']:
            values.append(None); available.append(False)
            reasons.append('fit_unavailable:'+fit['unavailable_reason'])
        elif not good:
            values.append(None); available.append(False); reasons.append('nonfinite_core_median8')
        else:
            predicted = value+fit['offset_dn']
            finite = math.isfinite(predicted)
            values.append(predicted if finite else None); available.append(finite)
            reasons.append(None if finite else 'nonfinite_offset_addition')
    return dict(values=values, available=available, unavailable_reasons=reasons)


def adapter_constants():
    return dict(methods=list(METHODS), guard_count=144, core_count=625, prior_count=8,
        median3_uses_latest_priors=3, guard_order='y_major_then_x_major', core_order='y_major_then_x_major',
        guard_grid='8..120_step8_Chebyshev_radius40..56_about64', core_grid='52..76_step1',
        medians_require_every_used_history_sample_finite=True, median_algorithm='numpy_median_float64',
        offset_core_prior='median8', guard_fits_shared_across_source_on_off=True,
        inherited_model_sha256=INHERITED_SHA256['scripts/accuracy_v53_offset.py'],
        inherited_test_sha256=INHERITED_SHA256['tests/unit/test_accuracy_v53_offset.py'],
        inherited_constants=offset_constants(),
        offset_unavailability_precedence=['fit_unavailable', 'nonfinite_core_median8', 'nonfinite_offset_addition'],
        clipping_selection_blending_or_fallback=False)


def reconstructed_prediction(row):
    observed = row['observed']
    fields = ('guard_xy', 'core_xy', 'guard_history', 'guard_current', 'core_history_on', 'core_history_off')
    arrays = {k: np.asarray(observed[k], dtype=float) for k in fields}
    require(np.array_equal(arrays['guard_xy'], GUARD_XY) and np.array_equal(arrays['core_xy'], CORE_XY),
            'Canonical core/guard grids differ')
    for name, shape in (('guard_history', (8, 144)), ('guard_current', (144,)),
                        ('core_history_on', (8, 625)), ('core_history_off', (8, 625))):
        require(arrays[name].shape == shape, 'Observed history shape differs: '+name)
    guard_median = strict_median(arrays['guard_history'], 8)
    predictions = {name: dict(on=strict_median(arrays['core_history_on'], count),
        off=strict_median(arrays['core_history_off'], count), guard_fit_sha256=None)
        for name, count in (('median8', 8), ('median3', 3))}
    crossfits = {}
    for split in SPLITS:
        crossfits[split] = crossfit = reconstructed_crossfit(arrays['guard_xy'], guard_median['values'],
                                                           arrays['guard_current'], split)
        for fold in (0, 1):
            fit = crossfit['fits']['median_offset'][str(fold)]
            predictions[f'offset_{split}_fold{fold}'] = dict(
                on=offset_prediction(predictions['median8']['on'], fit),
                off=offset_prediction(predictions['median8']['off'], fit), guard_fit_sha256=fit['model_sha256'])
    result = dict(schema_version=1, total_guard_count=144, total_core_count=625,
        input_sha256=digest(arrays), constants=adapter_constants(), guard_median8=guard_median,
        guard_crossfits=crossfits, predictions=predictions,
        metadata=dict(current_core_or_truth_argument_accepted=False, core_values_enter_guard_fit=False,
            guard_fits_shared_across_source_on_off=True, all_four_guard_folds_preserved_without_selection=True,
            new_guard_to_core_extrapolation_adapter=True, inherited_guard_only_predict_api_unchanged=True,
            source_core_safety_or_real_detection_accuracy_certified=False, production_decisions_modified=False,
            scoring_performed=False, unknown_predictions_preserved_without_fallback=True))
    result['prediction_sha256'] = digest(result)
    return result


def check_prediction(row, forecast):
    require(forecast.get('prediction_sha256') == digest(forecast, ('prediction_sha256',)), 'Prediction fingerprint differs')
    compare(reconstructed_prediction(row), forecast, 'independent complete prediction', rtol=0, atol=0)


ERROR_NAMES = ('on_background_error', 'off_background_error', 'on_residual', 'off_residual', 'paired_increment_error')
AMP_NAMES = ('raw_on_dn', 'raw_off_dn', 'paired_increment_dn', 'oracle_on_dn', 'raw_retention',
             'paired_retention', 'oracle_retention', 'raw_amplitude_error_dn', 'paired_amplitude_error_dn')


def evaluation_constants():
    return dict(sign_tolerance_dn=1e-10, sign_tolerance_is_detection_threshold=False,
        amplitude_requires_entire_positive_template=True, absent_retention_is_null=True,
        paired_on_off_noise_shared=True, matched_controls=['median8', 'median3'],
        source_on_off_common_scoring_support=True, methods=list(METHODS))


def point_metrics(values):
    values = [float(v) for v in values]
    require(all(math.isfinite(v) for v in values), 'Nonfinite scoring error')
    n = len(values)
    result = dict(count=n, mae_dn=math.fsum(abs(v) for v in values)/n if n else None,
        rmse_dn=math.sqrt(math.fsum(v*v for v in values)/n) if n else None,
        max_abs_dn=max(abs(v) for v in values) if n else None)
    require(all(v is None or math.isfinite(v) for v in result.values()), 'Nonfinite scoring arithmetic')
    return result


def sign_category(value, amplitude):
    normalized = value if amplitude > 0 else -value
    return 'reversed' if normalized < -1e-10 else 'zero' if abs(normalized) <= 1e-10 else 'same'


def metric_bundle(row, on, off, mask):
    observed, truth = row['observed'], row['truth']
    current_on = np.asarray(observed['core_current_on'], dtype=float)
    current_off = np.asarray(observed['core_current_off'], dtype=float)
    clean = np.asarray(truth['clean_current_background'], dtype=float)
    source = np.asarray(truth['current_source'], dtype=float)
    profile = np.asarray(truth['source_template'], dtype=float)
    amplitude = float(row['spec']['amplitude'])
    require(current_on.shape == current_off.shape == on.shape == off.shape == mask.shape == clean.shape == source.shape == profile.shape,
            'Scoring shapes differ')
    require(np.isfinite(profile).all() and (profile >= 0).all() and (profile > 0).any(), 'Invalid source template')
    require(math.isfinite(amplitude) and np.isfinite(clean).all() and np.isfinite(source).all(), 'Nonfinite analytic truth')
    require(np.allclose(source, amplitude*profile, rtol=0, atol=1e-12), 'Declared source/template differ')
    selected = np.flatnonzero(mask).tolist()
    on_res = [float(current_on[i])-float(on[i]) for i in selected]
    off_res = [float(current_off[i])-float(off[i]) for i in selected]
    errors = dict(on_background_error=point_metrics(float(on[i])-float(clean[i]) for i in selected),
        off_background_error=point_metrics(float(off[i])-float(clean[i]) for i in selected),
        on_residual=point_metrics(on_res), off_residual=point_metrics(off_res),
        paired_increment_error=point_metrics(a-b-float(source[i]) for a, b, i in zip(on_res, off_res, selected)))
    footprint = np.flatnonzero(profile > 0).tolist()
    available = all(bool(mask[i]) for i in footprint)
    oracle_available = all(math.isfinite(current_on[i]) and math.isfinite(clean[i]) for i in footprint)
    amps = dict(source_present=amplitude != 0, template_support_count=len(footprint), available=available,
        oracle_available=oracle_available,
        ideal_source_dn=amplitude, raw_on_dn=None, raw_off_dn=None, paired_increment_dn=None, oracle_on_dn=None,
        raw_retention=None, paired_retention=None, oracle_retention=None,
        raw_amplitude_error_dn=None, paired_amplitude_error_dn=None, raw_sign=None, paired_sign=None)
    denominator = math.fsum(float(profile[i])**2 for i in footprint)
    if oracle_available:
        oracle = math.fsum(float(profile[i])*(float(current_on[i])-float(clean[i])) for i in footprint)/denominator
        require(math.isfinite(oracle), 'Nonfinite oracle arithmetic')
        amps.update(oracle_on_dn=oracle, oracle_retention=oracle/amplitude if amplitude else None)
    if available:
        raw_on = math.fsum(float(profile[i])*(float(current_on[i])-float(on[i])) for i in footprint)/denominator
        raw_off = math.fsum(float(profile[i])*(float(current_off[i])-float(off[i])) for i in footprint)/denominator
        paired = math.fsum(float(profile[i])*((float(current_on[i])-float(on[i]))-(float(current_off[i])-float(off[i])))
                           for i in footprint)/denominator
        require(all(math.isfinite(v) for v in (raw_on, raw_off, paired)), 'Nonfinite projection arithmetic')
        amps.update(raw_on_dn=raw_on, raw_off_dn=raw_off, paired_increment_dn=paired)
        if amplitude:
            amps.update(raw_retention=raw_on/amplitude, paired_retention=paired/amplitude,
                raw_amplitude_error_dn=raw_on-amplitude, paired_amplitude_error_dn=paired-amplitude,
                raw_sign=sign_category(raw_on, amplitude), paired_sign=sign_category(paired, amplitude))
    return dict(point_errors=errors, amplitudes=amps)


def expected_score(row, forecast):
    observed = row['observed']
    current_on = np.asarray(observed['core_current_on'], dtype=float)
    current_off = np.asarray(observed['core_current_off'], dtype=float)
    n = len(current_on)
    result = dict(case_id=row['case_id'], spec=row['spec'], total_core_points=n, methods={}, matched={})
    masks, values = {}, {}
    for name in METHODS:
        prediction = forecast['predictions'][name]
        on = np.asarray(prediction['on']['values'], dtype=float)
        off = np.asarray(prediction['off']['values'], dtype=float)
        on_valid = np.asarray(prediction['on']['available'], dtype=bool)
        off_valid = np.asarray(prediction['off']['available'], dtype=bool)
        mask = on_valid & off_valid & np.isfinite(current_on) & np.isfinite(current_off)
        masks[name], values[name] = mask, (on, off)
        result['methods'][name] = dict(prediction_available_on=int(on_valid.sum()),
            prediction_available_off=int(off_valid.sum()), current_available_on=int(np.isfinite(current_on).sum()),
            current_available_off=int(np.isfinite(current_off).sum()), scored_indices=np.flatnonzero(mask).tolist(),
            complete=bool(n and mask.all()), metrics=metric_bundle(row, on, off, mask))
    for name in METHODS[2:]:
        mask = masks[name] & masks['median8'] & masks['median3']
        result['matched'][name] = dict(scored_indices=np.flatnonzero(mask).tolist(), complete=bool(n and mask.all()),
            methods={key: metric_bundle(row, *values[method], mask)
                     for key, method in (('corrected', name), ('median8', 'median8'), ('median3', 'median3'))})
    return result


def distribution(values):
    ordered = sorted(float(v) for v in values)
    require(all(math.isfinite(v) for v in ordered), 'Nonfinite summary metric')
    n = len(ordered)
    if not n:
        return dict(count=0, mean=None, median=None, p10=None, p90=None, min=None, max=None)

    def quantile(q):
        at = (n-1)*q
        lower, upper = math.floor(at), math.ceil(at)
        return ordered[lower]+(at-lower)*(ordered[upper]-ordered[lower])

    return dict(count=n, mean=math.fsum(ordered)/n, median=quantile(.5),
                p10=quantile(.1), p90=quantile(.9), min=ordered[0], max=ordered[-1])


def aggregate_metrics(bundles, completes):
    amps = [b['amplitudes'] for b in bundles]
    errors = {}
    for name in ERROR_NAMES:
        records = [b['point_errors'][name] for b in bundles]
        good = [r for r in records if r['count']]
        errors[name] = dict(scored_points=sum(r['count'] for r in records),
            case_mae=distribution(r['mae_dn'] for r in good), case_rmse=distribution(r['rmse_dn'] for r in good),
            case_max_abs=distribution(r['max_abs_dn'] for r in good))
    return dict(case_count=len(bundles), complete_cases=sum(completes),
        amplitude_available_cases=sum(a['available'] for a in amps),
        oracle_available_cases=sum(a['oracle_available'] for a in amps), present_cases=sum(a['source_present'] for a in amps),
        present_amplitude_available_cases=sum(a['source_present'] and a['available'] for a in amps),
        present_amplitude_unknown_cases=sum(a['source_present'] and not a['available'] for a in amps),
        present_oracle_available_cases=sum(a['source_present'] and a['oracle_available'] for a in amps), point_errors=errors,
        amplitudes={name: distribution(a[name] for a in amps if a[name] is not None) for name in AMP_NAMES},
        raw_sign_counts=dict(Counter(a['raw_sign'] for a in amps if a['raw_sign'] is not None)),
        paired_sign_counts=dict(Counter(a['paired_sign'] for a in amps if a['paired_sign'] is not None)))


def aggregate(rows):
    return dict(case_count=len(rows), methods={name: aggregate_metrics([r['methods'][name]['metrics'] for r in rows],
        [r['methods'][name]['complete'] for r in rows]) for name in METHODS},
        matched={name: {control: aggregate_metrics([r['matched'][name]['methods'][control] for r in rows],
            [r['matched'][name]['complete'] for r in rows]) for control in ('corrected', 'median8', 'median3')}
            for name in METHODS[2:]})


def expected_summary(rows, created):
    result = dict(completed=True, created_at_utc=created, case_count=len(rows), strata={}, groups={},
        no_detector_or_object_accuracy_claim=True, no_production_change=True, paired_source_counterfactual=True)
    for stratum in ('factorial', 'availability'):
        result['strata'][stratum] = aggregate([r for r in rows if r['spec']['stratum'] == stratum])
    factorial = [r for r in rows if r['spec']['stratum'] == 'factorial']
    for field in ('condition', 'motion', 'amplitude', 'background', 'noise_level'):
        groups = defaultdict(list)
        for row in factorial:
            groups[str(row['spec'][field])].append(row)
        result['groups'][field] = {k: aggregate(v) for k, v in sorted(groups.items())}
    groups = defaultdict(list)
    for row in factorial:
        spec = row['spec']
        groups[f"{spec['condition']}/{spec['motion']}/{spec['amplitude']}"] .append(row)
    result['groups']['condition_source'] = {k: aggregate(v) for k, v in sorted(groups.items())}
    result['groups']['missingness'] = {r['spec']['missingness']: aggregate([r]) for r in rows if r['spec']['stratum'] == 'availability'}
    return result


def read_lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def audit_run(run):
    verify_inherited_sources(ROOT)
    run = Path(run).absolute()
    require(run.parent == OUTPUT and run.resolve() == run and run.is_dir(), 'Canonical immediate V54 child required')
    require({p.name for p in run.iterdir()} == set(RUN_NAMES) | {'completion_receipt.json'},
            'Unexpected or missing run artifacts')
    bindings = {}
    receipt_path = run/'completion_receipt.json'
    receipt_hash = hashlib.sha256(raw_file(receipt_path)).hexdigest()
    receipt = checked_read(receipt_path, receipt_hash, bindings)
    require(receipt['completed'] is True and all(receipt[k] is False for k in
            ('source_decisions_changed', 'production_changed', 'real_data_accessed')), 'Invalid completion declarations')
    sources = {str(ROOT/name) for name in SOURCE_NAMES}
    artifacts = {str(run/name) for name in RUN_NAMES}
    require(set(receipt['files_sha256']) == sources | artifacts, 'Completion path allowlist differs')
    # Literal lists, never paths supplied by an old/current receipt map.
    for name in SOURCE_NAMES:
        path = ROOT/name
        expected = receipt['files_sha256'][str(path)]
        require(hashlib.sha256(raw_file(path)).hexdigest() == expected, 'Source changed: '+name)
        bindings[str(path)] = expected
    small = {}
    for name in RUN_NAMES:
        path = run/name
        expected = receipt['files_sha256'][str(path)]
        if name.endswith('.jsonl'):
            require(file_sha(path) == expected, 'Artifact hash differs: '+name)
            bindings[str(path)] = expected
        else:
            small[name] = checked_read(path, expected, bindings)
    freeze, manifest, summary = small['freeze.json'], small['predictions_frozen.json'], small['summary.json']
    specs = specifications()
    compare({p: bindings[p] for p in sources}, freeze['source_files_sha256'], 'frozen source bindings')
    compare(specs, freeze['specifications'], 'all frozen specifications', rtol=0, atol=0)
    compare(adapter_constants(), freeze['adapter_constants'], 'frozen adapter constants', rtol=0, atol=0)
    compare(evaluation_constants(), freeze['evaluation_constants'], 'frozen evaluation constants', rtol=0, atol=0)
    require(freeze['before_actual_fitting'] is True and freeze['before_truth_scoring'] is True, 'Freeze chronology flags differ')
    require(manifest['completed'] is True and manifest['case_count'] == 420
            and manifest['truth_scoring_started'] is False and manifest['current_guard_training_used'] is True,
            'Prediction freeze contract differs')
    compare({str(run/name): bindings[str(run/name)] for name in ('inputs.jsonl', 'predictions.jsonl')},
            manifest['files_sha256'], 'prediction freeze bindings')
    times = [freeze['created_at_utc'], manifest['created_at_utc'], summary['created_at_utc'], receipt['created_at_utc']]
    require([datetime.fromisoformat(v) for v in times] == sorted(datetime.fromisoformat(v) for v in times),
            'Saved freeze/scoring/completion chronology differs')
    scores, unavailable, method_unknown = [], Counter(), Counter()
    for index, (spec, row, prediction, saved_score) in enumerate(zip(specs, read_lines(run/'inputs.jsonl'),
            read_lines(run/'predictions.jsonl'), read_lines(run/'scores.jsonl'), strict=True), 1):
        require(row['case_id'] == prediction['case_id'] == saved_score['case_id'] == spec['case_id'], 'Case identity/order differs')
        compare(spec, row['spec'], 'input spec', rtol=0, atol=0)
        check_generated_input(spec, row)
        require(set(prediction) == {'case_id', 'forecast'}, 'Prediction wrapper fields differ')
        check_prediction(row, prediction['forecast'])
        expected = expected_score(row, prediction['forecast'])
        compare(expected, saved_score, 'complete source score')
        scores.append(expected)
        for crossfit in prediction['forecast']['guard_crossfits'].values():
            for fit in crossfit['fits']['median_offset'].values():
                if not fit['available']:
                    unavailable[fit['unavailable_reason']] += 1
        for method in METHODS:
            if not expected['methods'][method]['metrics']['amplitudes']['available']:
                method_unknown[method] += 1
        if index % 50 == 0 or index == len(specs):
            print(f'V54 audit: independently checked {index}/{len(specs)} cases', flush=True)
    require(len(scores) == 420, 'Frozen case denominator differs')
    compare(expected_summary(scores, summary['created_at_utc']), summary, 'entire independent summary')
    for path, expected in bindings.items():
        require(file_sha(Path(path)) == expected, 'Bound file changed during audit')
    return dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        verified_cases=420, verified_factorial_cases=416, verified_availability_cases=4,
        verified_guard_crossfits=840, verified_guard_fits=1680, verified_method_case_predictions=2520,
        verified_score_rows=420, unavailable_guard_fit_reasons=dict(unavailable),
        amplitude_unavailable_cases_by_method=dict(method_unknown),
        all_generated_arrays_and_paired_source_profiles_checked=True,
        all_guard_fit_certificates_and_core_predictions_checked=True,
        all_own_and_matched_masks_projections_and_metrics_checked=True, entire_summary_recomputed=True,
        no_producer_or_prior_auditor_imports=True, no_real_data_or_media_access=True,
        no_detector_or_object_accuracy_claim=True, generated_input_atol=1e-12, exact_source_template_truth_atol=0,
        fitted_prediction_atol=0, fitted_prediction_rtol=0,
        reduction_comparison_atol=1e-9, reduction_comparison_rtol=1e-10, files_sha256=bindings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.absolute()
    require(destination.parent == OUTPUT and destination.resolve() == destination and not destination.exists(),
            'Fresh immediate V54 audit artifact required')
    result = audit_run(args.run)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
