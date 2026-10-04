"""Frozen generated history diagnostic; no adaptive forecast or detector change."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np

from accuracy_v51_benchmark import specifications, generate_case, benchmark_metadata
from accuracy_v51_history_diagnostic import diagnose, diagnostic_fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/tiny_target/accuracy_v51_20260926'
ARMS = ('median8', 'median3')
RESPONSE_FRAMES = tuple(range(8, 64))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    path = Path(path)
    if path.resolve() != path or not path.is_file():
        raise ValueError('Expected canonical regular file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    for a in result.values():
        a.setflags(write=False)
    return result


def check_bindings(bindings):
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen file changed: ' + path)


def sources():
    return sorted(set((ROOT/'scripts').glob('*accuracy_v51*.py')) |
                  set((ROOT/'tests/unit').glob('test_accuracy_v51*.py')) |
                  {ROOT/'docs/accuracy_v51_plan.md'})


def prepare(specs, output):
    """Save ALL diagnostics before any response scoring; metadata stays separate."""
    entries = []; bindings = {}
    for case_number,spec in enumerate(specs,1):
        name = spec['case_id']
        if not re.fullmatch('[a-z0-9_]+', name):
            raise ValueError('Unsafe generated case id')
        case = generate_case(spec)
        values = case['values']
        if values.shape != (64, 144) or values.dtype != np.float64:
            raise ValueError('Generated input schema differs')
        input_path = output/(name+'_input.npz')
        write_npz(input_path, {k: case[k] for k in ('values', 'event_active',
            'response_phase', 'history_has_prior_event', 'signal_delta',
            'affected_points_mask', 'missing_mask', 'points_xy')})
        diagnostics = [diagnose(values[t-8:t]) for t in RESPONSE_FRAMES]
        array_keys = sorted(k for k, v in diagnostics[0].items() if isinstance(v, np.ndarray))
        if any(sorted(k for k,v in d.items() if isinstance(v,np.ndarray)) != array_keys for d in diagnostics):
            raise ValueError('Diagnostic schema changed between windows')
        diagnostic_path = output/(name+'_diagnostics.npz')
        write_npz(diagnostic_path, {k: np.stack([d[k] for d in diagnostics]) for k in array_keys})
        context = []
        for t, d in zip(RESPONSE_FRAMES, diagnostics):
            if diagnostic_fingerprint(d) != d['diagnostic_sha256']:
                raise ValueError('Diagnostic fingerprint differs before saving')
            context.append(dict(frame_index=t, diagnostic={k:v for k,v in d.items() if k not in array_keys}))
        context_path = output/(name+'_context.json')
        write_json(context_path, context)
        for path in (input_path, diagnostic_path, context_path):
            bindings[str(path)] = sha(path)
        entries.append(dict(spec=spec, input_path=str(input_path),
            diagnostic_path=str(diagnostic_path), context_path=str(context_path), array_keys=array_keys))
        if case_number%14==0 or case_number==len(specs):
            print(f'V51: prior-only diagnostics saved {case_number}/{len(specs)} cases',flush=True)
    manifest = dict(completed=True, created_at_utc=now(), cases=entries, files_sha256=bindings,
        case_count=len(entries), response_window_count=len(entries)*len(RESPONSE_FRAMES),
        response_scoring_started=False, adaptive_forecast_selection=False)
    write_json(output/'forecasts_frozen.json', manifest)
    return manifest


def restore_diagnostic(arrays, context, index):
    value = dict(context[index]['diagnostic'])
    value.update({k:a[index] for k,a in arrays.items()})
    if diagnostic_fingerprint(value) != value['diagnostic_sha256']:
        raise ValueError('Reloaded diagnostic differs from saved fingerprint')
    return value


def diagnostic_summary(d):
    good = d['point_available']
    recent = d['recent_center_defined']
    returned = d['return_fraction_defined']
    suffix = d['recent_center_suffix_length'][recent]
    return dict(prior_available_points=int(good.sum()),
        recent_center_defined_points=int(recent.sum()),
        recent_center_suffix_counts={str(i):int((suffix==i).sum()) for i in range(4)},
        mean_absolute_fast_slow_delta_dn=float(np.abs(d['fast_slow_delta'][good]).mean()) if good.any() else None,
        mean_absolute_early_recent_delta_dn=float(np.abs(d['early_recent_delta'][good]).mean()) if good.any() else None,
        mean_absolute_recent_center_margin_dn=float(np.abs(d['recent_center_margins'][:,recent]).mean()) if recent.any() else None,
        observed_return_defined_points=int(returned.sum()),
        observed_return_fraction_sum=float(d['return_fraction'][returned].sum()) if returned.any() else 0.,
        observed_return_fraction_mean=float(d['return_fraction'][returned].mean()) if returned.any() else None,
        no_regime_or_object_label=True)


def measure(current, d):
    """Responses never enter diagnose; use an identical mask for both comparators."""
    current = np.asarray(current)
    if current.shape != d['point_available'].shape or current.dtype.kind not in 'iuf':
        raise ValueError('Current shape differs')
    before = diagnostic_fingerprint(d)
    if before != d['diagnostic_sha256']:
        raise ValueError('Diagnostic differs from its frozen fingerprint')
    good = d['point_available'] & np.isfinite(current)
    errors = {arm: current[good]-d[key][good] for arm,key in zip(ARMS,('slow8','fast3'))}
    if any(not np.isfinite(e).all() for e in errors.values()):
        raise ValueError('Nonfinite scoring arithmetic')
    arms = {}
    for arm, error in errors.items():
        absolute = np.abs(error)
        with np.errstate(over='ignore',invalid='ignore'):
            absolute_sum = float(absolute.sum())
            squared_sum = float(np.square(error).sum())
        if not np.isfinite([absolute_sum,squared_sum]).all():
            raise ValueError('Nonfinite scoring summary')
        arms[arm] = dict(point_count=int(good.sum()), absolute_error_sum_dn=absolute_sum,
            squared_error_sum_dn2=squared_sum,
            mae_dn=float(absolute.mean()) if len(error) else None,
            max_absolute_error_dn=float(absolute.max()) if len(error) else None)
    if diagnostic_fingerprint(d) != before:
        raise ValueError('Current scoring mutated history diagnostic')
    return dict(total_points=len(current), prior_unavailable_points=int((~d['point_available']).sum()),
        current_nonfinite_points=int((~np.isfinite(current)).sum()), scorable_points=int(good.sum()),
        unscorable_points=int((~good).sum()), complete_window=bool(len(good) and good.all()), arms=arms)


def aggregate(rows):
    total = sum(r['measurement']['total_points'] for r in rows)
    scored = sum(r['measurement']['scorable_points'] for r in rows)
    complete = [r for r in rows if r['measurement']['complete_window']]
    arms = {}
    for arm in ARMS:
        absolute = sum(r['measurement']['arms'][arm]['absolute_error_sum_dn'] for r in rows)
        squared = sum(r['measurement']['arms'][arm]['squared_error_sum_dn2'] for r in rows)
        maxima = [r['measurement']['arms'][arm]['max_absolute_error_dn'] for r in rows
                  if r['measurement']['arms'][arm]['point_count']]
        arms[arm] = dict(conditional_point_mae_dn=absolute/scored if scored else None,
            conditional_point_rmse_dn=float(np.sqrt(squared/scored)) if scored else None,
            max_absolute_error_dn=max(maxima) if maxima else None,
            mean_complete_window_mae_dn=float(np.mean([r['measurement']['arms'][arm]['mae_dn'] for r in complete])) if complete else None)
    returned = sum(r['diagnostic']['observed_return_defined_points'] for r in rows)
    return dict(response_windows=len(rows), complete_windows=len(complete), total_point_opportunities=total,
        scorable_points=scored, unscorable_points=total-scored,
        prior_unavailable_points=sum(r['measurement']['prior_unavailable_points'] for r in rows),
        current_nonfinite_points=sum(r['measurement']['current_nonfinite_points'] for r in rows),
        missing_counts_overlap=True, arms=arms,
        recent_center_defined_points=sum(r['diagnostic']['recent_center_defined_points'] for r in rows),
        recent_center_suffix_counts={str(i):sum(r['diagnostic']['recent_center_suffix_counts'][str(i)] for r in rows) for i in range(4)},
        observed_return_defined_points=returned,
        observed_return_fraction_mean=sum(r['diagnostic']['observed_return_fraction_sum'] for r in rows)/returned if returned else None,
        frames_points_and_cases_not_independent=True, descriptors_are_not_regime_classifications=True)


def score(manifest, metadata, output):
    check_bindings(manifest['files_sha256'])
    manifest_path = output/'forecasts_frozen.json'
    manifest_sha = sha(manifest_path)
    if read_json(manifest_path) != manifest:
        raise ValueError('Memory manifest differs from frozen file')
    records = []; twin_windows = {}; twins = metadata['twin_pairs']
    wanted = {(p[k],p['response_frame']) for p in twins for k in ('step_case_id','pulse_case_id')}
    for case_number,entry in enumerate(manifest['cases'],1):
        case = load_npz(entry['input_path']); arrays = load_npz(entry['diagnostic_path'])
        context = read_json(entry['context_path']); spec = entry['spec']
        if [c['frame_index'] for c in context] != list(RESPONSE_FRAMES):
            raise ValueError('Window membership differs')
        for i,t in enumerate(RESPONSE_FRAMES):
            d = restore_diagnostic(arrays,context,i)
            measurement = measure(case['values'][t],d)
            row = dict(spec, frame_index=t, scheduled_phase=str(case['response_phase'][t]),
                history_has_prior_event=bool(case['history_has_prior_event'][t]),
                measurement=measurement, diagnostic=diagnostic_summary(d))
            records.append(row)
            if (spec['case_id'],t) in wanted:
                twin_windows[(spec['case_id'],t)] = dict(history=case['values'][t-8:t].copy(),
                    current=case['values'][t].copy(), fingerprint=d['diagnostic_sha256'],
                    forecasts={a:d[k].copy() for a,k in zip(ARMS,('slow8','fast3'))},
                    diagnostics=diagnostic_summary(d))
        if case_number%14==0 or case_number==len(manifest['cases']):
            print(f'V51: frozen comparator scoring {case_number}/{len(manifest["cases"])} cases',flush=True)
    twin_results = []
    for pair in twins:
        t = pair['response_frame']
        a = twin_windows[(pair['step_case_id'],t)]; b = twin_windows[(pair['pulse_case_id'],t)]
        if not np.array_equal(a['history'],b['history'],equal_nan=True) or a['fingerprint'] != b['fingerprint']:
            raise ValueError('Indistinguishable histories produced different descriptors')
        if (not np.isfinite(a['current']).all() or not np.isfinite(b['current']).all()
                or not np.allclose(a['current']-b['current'], pair['amplitude'], rtol=0, atol=6e-14)):
            raise ValueError('Twin response difference differs from declared amplitude')
        separation = np.abs(a['current']-b['current']); lower = separation/2
        if not (separation>0).all():
            raise ValueError('Twin responses must differ at every point')
        arms = {}
        for arm in ARMS:
            if not np.array_equal(a['forecasts'][arm],b['forecasts'][arm],equal_nan=True):
                raise ValueError('Identical prefix forecasts differ')
            worst = np.maximum(np.abs(a['forecasts'][arm]-a['current']),np.abs(a['forecasts'][arm]-b['current']))
            if not np.all(worst+1e-12>=lower):
                raise ValueError('Twin minimax arithmetic failed')
            arms[arm] = dict(mean_worst_world_absolute_error_dn=float(worst.mean()),
                max_worst_world_absolute_error_dn=float(worst.max()))
        twin_results.append(dict(pair, identical_priors=True, identical_diagnostics=True,
            identical_forecasts=True, point_count=len(lower),
            possible_response_separation_dn=dict(min=float(separation.min()),max=float(separation.max())),
            necessary_worst_world_error_dn=dict(min=float(lower.min()),max=float(lower.max())),
            minimum_interval_width_to_cover_both_dn=dict(min=float(separation.min()),max=float(separation.max())),
            arms=arms, shared_history_diagnostics=a['diagnostics'], numerical_check_tolerance_dn=1e-12,
            analytic_response_difference_check_tolerance_dn=6e-14,
            not_a_physical_bound_or_predicted_class=True))
    check_bindings(manifest['files_sha256'])
    if sha(manifest_path) != manifest_sha:
        raise ValueError('Manifest changed during scoring')
    with (output/'responses.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False, separators=(',',':'))+'\n')
    write_json(output/'identical_prefix_pairs.json',dict(pairs=twin_results,all_pairs_passed=True))
    summary = dict(completed=True,created_at_utc=now(),overall=aggregate(records),
        by_case={name:aggregate([r for r in records if r['case_id']==name]) for name in sorted({r['case_id'] for r in records})},
        by_family={name:aggregate([r for r in records if r['family']==name]) for name in sorted({r['family'] for r in records})},
        by_family_phase={family:{phase:aggregate([r for r in records if r['family']==family and r['scheduled_phase']==phase])
            for phase in sorted({r['scheduled_phase'] for r in records if r['family']==family})}
            for family in sorted({r['family'] for r in records})},
        by_noise_level={str(level):aggregate([r for r in records if r['noise_level']==level]) for level in sorted({r['noise_level'] for r in records})},
        twin_pair_count=len(twin_results),all_identical_prefix_checks_passed=True,
        no_new_predictor_or_model_selection=True,no_production_or_object_decision=True,
        generated_only_not_real_camera_or_airborne_validation=True)
    write_json(output/'summary.json',summary)
    return summary


def run(output):
    output = Path(output).absolute()
    if output.parent != OUTPUT or output.resolve()!=output or output.exists():
        raise ValueError('Fresh immediate V51 output child required')
    specs = specifications(); metadata = benchmark_metadata()
    if len(specs)!=98 or len({s['case_id'] for s in specs})!=98 or len(metadata['twin_pairs'])!=40:
        raise ValueError('Frozen benchmark denominator differs')
    bindings = {str(path):sha(path) for path in sources()}
    output.mkdir(parents=True)
    write_json(output/'freeze.json',dict(created_at_utc=now(),specifications=specs,benchmark=metadata,
        source_files_sha256=bindings,before_generation_and_scoring=True,
        diagnostic_receives_only_eight_prior_values=True,production_changed=False,
        runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    freeze_sha = sha(output/'freeze.json')
    print('V51: specifications frozen; generating all prior-only diagnostics',flush=True)
    manifest = prepare(specs,output)
    print('V51: all diagnostics frozen; scoring fixed comparator forecasts',flush=True)
    summary = score(manifest,metadata,output)
    if (summary['overall']['response_windows'],summary['overall']['total_point_opportunities'])!=(5488,790272):
        raise ValueError('Scoring denominator differs')
    check_bindings(bindings)
    if sha(output/'freeze.json') != freeze_sha:
        raise ValueError('Freeze changed during experiment')
    write_json(output/'completion_receipt.json',dict(completed=True,created_at_utc=now(),
        files_sha256={**bindings,**{str(p):sha(p) for p in sorted(output.iterdir()) if p.is_file()}},
        production_changed=False,real_data_accessed=False))
    print('V51: generated experiment complete; no adaptive predictor or production change',flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
