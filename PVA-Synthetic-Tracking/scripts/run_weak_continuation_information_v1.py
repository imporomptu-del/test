"""Frozen metadata/cached-array feasibility diagnostic; no detector decisions."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v56_20260926/evidence_01/seaqr_accuracy_v56_GFs5W1'
AUDIT_SHA = '86c069029c08dda31503bfb318c440a7e82de06e008bcf1f717ee5a4b4c9a063'
FRAMES = tuple(range(213, 219))
FOCAL = '0/bright:1001'
CODE = [ROOT/'scripts'/n for n in ('run_weak_continuation_information_v1.py', 'weak_continuation_information_v1.py')]
CODE += [ROOT/'tests/unit'/n for n in ('test_weak_continuation_information_v1.py', 'test_run_weak_continuation_information_v1.py')]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def number(s):
        result = float(s)
        require(math.isfinite(result), 'nonfinite JSON')
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(binding):
    p = Path(binding['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_file() and not p.is_symlink(), 'canonical regular input required')
    require(sha(p) == binding['sha256'], 'input changed: '+str(p))


def transition(dt):
    require(0 < dt <= 1, 'unsupported tracker time gap')
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q = 3600*np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                       [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
    return f, q


def covariance_predict(p, dt):
    f, q = transition(dt)
    value = f @ p @ f.T + q
    return (value+value.T)/2


def covariance_correct(p):
    s = p[:2, :2]+np.eye(2)*4
    k = np.linalg.solve(s.T, p[:, :2].T).T
    a = np.eye(4)-k @ np.eye(4)[:2]
    value = a @ p @ a.T + k @ (np.eye(2)*4) @ k.T
    return k, (value+value.T)/2


def derive_forecasts(journal):
    """Only t-1 posterior state enters forecasts; current updates only audit them."""
    with Path(journal).open() as stream:
        return _derive_forecasts_rows(stream)


def _derive_forecasts_rows(stream):
    states, saved, diagnostics = {}, [], dict(posterior_states_checked=0, gaussian_costs_checked=0,
        max_posterior_absolute_error=0., max_gaussian_cost_absolute_error=0., last_journal_frame_read=FRAMES[-1])
    for index, line in enumerate(itertools.islice(stream, FRAMES[-1]+1)):
        row = json.loads(line)
        require(row['frame_index'] == index and row['timestamp_ns'] == index*100000000, 'journal cadence/continuity')
        ts, segment = row['timestamp_ns'], row['segment']
        if row['motion'].get('reset') is True:
            states = {}
        require(not states or all(s['segment'] == segment for s in states.values()), 'segment without reset')
        predictions = {}
        for identity, state in states.items():
            dt = (ts-state['timestamp_ns'])/1e9
            f, _ = transition(dt)
            mean, p = f @ state['mean'], covariance_predict(state['covariance'], dt)
            predictions[identity] = (mean, p)
        # This snapshot is made before reading current tracks/candidates/references.
        if index in FRAMES:
            require(row['motion'].get('reset') is False and row['motion'].get('accepted') is True,
                    'capture geometry not accepted')
            forecasts = []
            for identity, (mean, p) in sorted(predictions.items()):
                old = states[identity]
                forecasts.append(dict(identity=identity, polarity=old['polarity'], reference_xy=mean[:2].tolist(),
                    innovation_covariance_2x2=(p[:2, :2]+np.eye(2)*4).tolist(),
                    previous_frame_index=old['frame_index'], previous_timestamp_ns=old['timestamp_ns'],
                    previous_reference_xy=old['mean'][:2].tolist(), previous_velocity_reference_xy_px_s=old['mean'][2:].tolist(),
                    previous_covariance_4x4=old['covariance'].tolist(), previous_measured=old['measured'],
                    previous_qualified=old['qualified'], previous_lifecycle=old['lifecycle'],
                    previous_independent_hits=old['independent_hits'], previous_confirmation_timestamp_ns=old['confirmation_timestamp_ns']))
            require(any(f['identity'] == FOCAL and f['previous_qualified'] for f in forecasts), 'focal past qualification missing')
            saved.append(dict(frame_index=index, timestamp_ns=ts, forecasts=forecasts))
        next_states = {}
        for track in row['tracks']:
            identity = f'{track["segment"]}/{track["track_id"]}'
            require(identity not in next_states, 'duplicate track identity')
            observed_mean = np.array(track['reference_xy']+track['velocity_reference_xy_px_s'], dtype=float)
            require(observed_mean.shape == (4,) and np.isfinite(observed_mean).all(), 'invalid state')
            polarity, number = track['track_id'].split(':')
            if identity not in states:
                require(track['hits'] == 1 and track['measured'] is True, 'incomplete birth history')
                require(np.array_equal(observed_mean[2:], [0, 0]), 'birth velocity differs')
                p = np.diag([4., 4., 22500., 22500.])
            else:
                mean, p = predictions[identity]
                if track['measured']:
                    point = np.array([*track['measurement_source_xy'], 1.])
                    z = np.array(row['source_to_reference']) @ point
                    z = z[:2]/z[2]
                    innovation = z-mean[:2]
                    s = p[:2, :2]+np.eye(2)*4
                    gauss = float(innovation @ np.linalg.solve(s, innovation)+np.linalg.slogdet(s)[1])
                    matches = [a for a in row['tracking_metrics'][polarity]['association_audit'] if a['track_id'] == int(number)]
                    require(len(matches) == 1, 'missing association audit')
                    err = abs(gauss-matches[0]['gaussian_twice_nll_without_constant'])
                    require(err < 1e-7, 'reconstructed Gaussian cost mismatch')
                    diagnostics['max_gaussian_cost_absolute_error'] = max(diagnostics['max_gaussian_cost_absolute_error'], err)
                    diagnostics['gaussian_costs_checked'] += 1
                    k, p = covariance_correct(p)
                    mean = mean+k @ innovation
                err = float(np.max(np.abs(mean-observed_mean)))
                require(err < 1e-7, 'reconstructed posterior mismatch')
                diagnostics['max_posterior_absolute_error'] = max(diagnostics['max_posterior_absolute_error'], err)
                diagnostics['posterior_states_checked'] += 1
            next_states[identity] = dict(mean=observed_mean, covariance=p, timestamp_ns=ts, frame_index=index,
                segment=track['segment'], polarity=polarity, measured=track['measured'], qualified=track['qualified_moving'],
                lifecycle=track['lifecycle'], independent_hits=track['independent_hits'],
                confirmation_timestamp_ns=track['confirmation_timestamp_ns'])
        states = next_states
    require([r['frame_index'] for r in saved] == list(FRAMES), 'incomplete capture forecast sequence')
    return saved, diagnostics


def build_plan():
    verify(dict(path=str(BASE/'independent_audit.json'), sha256=AUDIT_SHA))
    audit = read(BASE/'independent_audit.json')
    require(audit['passed'] and audit['frames'] == 674 and audit['exact_full_journal_semantics'], 'original parity missing')
    names = ['clean/frames.jsonl', 'clean/launch.json', 'probe.v56.json', 'probe/implementation/tracking/kalman.py',
             'probe/implementation/visible_baseline.py']
    names += [f'captures/frame_{f:06d}.{ext}' for f in FRAMES for ext in ('json', 'npz')]
    # Library-source hashes are bound by the already-pinned launch if not directly
    # enumerated in the original independent audit file list.
    launch_binding = dict(path=str(BASE/'clean/launch.json'), sha256=audit['files_sha256']['clean/launch.json'])
    verify(launch_binding)
    launch = read(BASE/'clean/launch.json')
    bindings = [dict(path=str(BASE/'independent_audit.json'), sha256=AUDIT_SHA)]
    for name in names:
        digest = audit['files_sha256'].get(name)
        if digest is None and '/implementation/' in name:
            digest = launch['package_sha256'][name.split('/implementation/')[1]]
        require(digest is not None, 'missing pinned input: '+name)
        binding = dict(path=str(BASE/name), sha256=digest)
        verify(binding)
        bindings.append(binding)
    config = launch['configuration']
    for key, expected in dict(position_sigma_px=2, initial_velocity_sigma_px_s=150, acceleration_sigma_px_s2=60,
                              position_gate_px=45, mahalanobis_gate_squared=25, temporal_threshold_sigma=4, spatial_threshold_sigma=3).items():
        require(config[key] == expected, 'frozen configuration differs')
    forecasts, validation = derive_forecasts(BASE/'clean/frames.jsonl')
    oracle_path = ROOT.parent/'outputs/seaqr_weak_continuation_20261002/forecast_oracle.json'
    oracle_binding = dict(path=str(oracle_path), sha256='354af4bfc270e41931ab7a3b11e73514865923b19827aee06c2e37f5eac69885')
    verify(oracle_binding)
    oracle = read(oracle_path)
    require(oracle['passed'] and len(oracle['forecasts']) == len(FRAMES), 'independent oracle missing')
    for ours, other in zip(forecasts, oracle['forecasts']):
        focal = next(t for t in ours['forecasts'] if t['identity'] == FOCAL)
        require(ours['frame_index'] == other['frame'] and focal['reference_xy'] == other['mean_reference_xy_vxy'][:2]
                and focal['innovation_covariance_2x2'] == other['innovation_covariance'], 'forecast oracle differs')
    validation['focal_matches_independent_six_frame_oracle_exactly'] = True
    bindings.append(oracle_binding)
    snapshots = []
    for f in FRAMES:
        meta = read(BASE/f'captures/frame_{f:06d}.json')
        require(meta['frame'] == f and meta['prelearning'] and meta['full_exposed_state_unchanged'] and
                meta['native_state_before'] == meta['native_state_after'], 'capture provenance differs')
        # Deliberately omit source_probe, its reference coordinates and all current
        # tracking/candidate values from query input. Rectangle only bounds coverage.
        snapshots.append(dict(frame_index=f, metadata={k: meta[k] for k in ('rectangle', 'float_fields', 'flag_fields',
            'ready', 'temporal_threshold_sigma_float32', 'spatial_threshold_sigma_float32')},
            npz_path=str(BASE/f'captures/frame_{f:06d}.npz')))
        snapshots[-1]['metadata']['rectangle'] = {k: meta['rectangle'][k] for k in
            ('shape_hw', 'tile_bounds_exclusive_xyxy', 'capture_bounds_exclusive_xyxy', 'halo_px')}
    codes = [dict(path=str(p), sha256=sha(p)) for p in CODE]
    for b in bindings+codes:
        verify(b)
    return dict(schema='seaqr.weak-continuation-information.plan.v1', focal_identity=FOCAL, frames=list(FRAMES),
        input_bindings=bindings, code_bindings=codes, snapshots=snapshots, forecast_frames=forecasts,
        covariance_reconstruction_validation=validation,
        frozen_rule='Every eligible raw-absolute peak with positive same-polarity centered temporal evidence and inherited spatial pass; unchanged45px AND Mahalanobis25 geometry. Report all, no weak cutoff, selection, acceptance or state update.',
        query_scope='All previously qualified tracks whose45px outer bounding boxes intersect the captured full tile (a conservative superset of disk intersection); all prior-active tracks remain potential competing owners even if their centers lie outside capture.',
        limits=['Six adjacent familiar, originally reference-selected tiles; not six independent encounters or an unconditioned scene sample.',
                'Past covariance reconstructed and posterior/Gaussian costs checked; no replay of association decisions or closed-loop weak updates.',
                'This query uses raw prequota/pre-shape pixels, not final shape-centroid measurements. Pixel multiplicity is not object count.',
                'Unique local evidence is not identity. Generated true-point/decoy ambiguity prevents automatic continuation claims.',
                'No source-video/RAW16/holdout/Jetson access, new-object threshold change, classifier, physical-class or speed claim.'],
        production_changed=False, classifier_applied=False)


def run(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists() and not output.is_symlink(), 'fresh output required')
    digest = sha(plan_path)
    plan = read(plan_path)
    require(sha(plan_path) == digest, 'plan changed while reading')
    require(plan == build_plan(), 'frozen plan changed')
    require(sha(plan_path) == digest, 'plan changed before evaluation')
    core_path = ROOT/'scripts/weak_continuation_information_v1.py'
    source = core_path.read_bytes()
    core_hash = next(b['sha256'] for b in plan['code_bindings'] if b['path'] == str(core_path))
    require(hashlib.sha256(source).hexdigest() == core_hash, 'core changed before import')
    core = types.ModuleType('weak_core')
    core.__file__ = str(core_path)
    exec(compile(source, str(core_path), 'exec'), core.__dict__)
    output.mkdir()
    (output/'implementation').mkdir()
    for b in plan['code_bindings']:
        dest = output/'implementation'/Path(b['path']).name
        shutil.copyfile(b['path'], dest)
        verify(dict(path=str(dest), sha256=b['sha256']))
    shutil.copyfile(plan_path, output/'plan.json')
    verify(dict(path=str(output/'plan.json'), sha256=digest))
    results = []
    for snap, history in zip(plan['snapshots'], plan['forecast_frames']):
        require(snap['frame_index'] == history['frame_index'], 'snapshot/forecast frame mismatch')
        with np.load(snap['npz_path'], allow_pickle=False) as arrays:
            values, flags = arrays['values'], arrays['flags']
            x0, y0, x1, y1 = snap['metadata']['rectangle']['tile_bounds_exclusive_xyxy']
            queries = []
            for f in history['forecasts']:
                x, y = f['reference_xy']
                if f['previous_qualified'] and x+45 >= x0 and x-45 < x1 and y+45 >= y0 and y-45 < y1:
                    queries.append(core.enumerate_peaks(values, flags, snap['metadata'], f, history['forecasts']))
            results.append(dict(frame_index=snap['frame_index'], prior_active_tracks=len(history['forecasts']),
                queries=queries, focal_identity=FOCAL))
    for b in plan['input_bindings']+plan['code_bindings']:
        verify(b)
    require(sha(plan_path) == digest, 'plan changed during evaluation')
    write(output/'results.json', results)
    summary = dict(schema='seaqr.weak-continuation-information.summary.v1', passed=True, frames=list(FRAMES),
        plan_sha256=digest, result_sha256=sha(output/'results.json'), query_counts=[len(r['queries']) for r in results],
        production_changed=False, new_measurements_created=0, limits=plan['limits'])
    write(output/'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-output', type=Path)
    mode.add_argument('--run-plan', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.plan_output:
        require(args.output is None, 'output only for run')
        write(args.plan_output, build_plan())
    else:
        require(args.output is not None, 'output required')
        run(args.run_plan, args.output)
