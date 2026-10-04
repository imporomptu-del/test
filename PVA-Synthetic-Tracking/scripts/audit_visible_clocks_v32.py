"""Read-only audit of completed trials in the stopped v32 clock diagnostic."""
import argparse
from collections import defaultdict
import datetime
import math
from pathlib import Path

from profile_visible_interaction_v30 import read, sha, write
from analyze_visible_interaction_v30 import require, statistics
from verify_visible_combined_v29 import verify_sources, transformed_hash, ROOT
from verify_visible_interaction_v30 import V29_LOCAL, audit_relocated_trial

V31_LOCAL = ROOT/'results/tiny_target/visible_threads_v31_20260920/evidence'
REMOTE = Path('/tmp/seaqr_visible_clocks_v32_Y1gbvz')
OLD_RUNNER = Path('/tmp/seaqr_visible_threads_v31_QvwNyq/run_visible_threads_v31.py')
MODES = ('v26_default', 'combined_default', 'v26', 'combined')


def specs():
    result = [dict(name='smoke_'+c, clip=c, mode='combined', fixed=True, audit=True) for c in ('0126', '0082')]
    orders = (MODES, ('combined', 'v26', 'combined_default', 'v26_default'),
              ('combined_default', 'v26', 'combined', 'v26_default'))
    for repeat, order in enumerate(orders):
        for clip in (('0082', '0126') if repeat == 1 else ('0126', '0082')):
            clock_order = (False, True) if (repeat + (clip == '0082')) % 2 == 0 else (True, False)
            for fixed in clock_order:
                for mode in order:
                    result.append(dict(name=f'{clip}_repeat{repeat}_{"fixed" if fixed else "auto"}_{mode}',
                        clip=clip, mode=mode, fixed=fixed, audit=False))
    return result


def summarize(trials, samples):
    result = {}
    for clip in ('0126', '0082'):
        present = {(fixed, mode): [i for i in range(3)
            if f'{clip}_repeat{i}_{"fixed" if fixed else "auto"}_{mode}' in trials]
            for fixed in (False, True) for mode in MODES}
        paired = sorted(set.intersection(*(set(v) for v in present.values())))
        require(paired, 'No complete matched repeat for '+clip)
        cells = {}
        for fixed in (False, True):
            clock = 'fixed' if fixed else 'auto'
            cells[clock] = {}
            for mode in MODES:
                names = [f'{clip}_repeat{i}_{clock}_{mode}' for i in paired]
                rows = [trials[n] for n in names]
                pooled = sum(r['count'] for r in rows)/sum(r['wall_s'] for r in rows)
                cells[clock][mode] = dict(pooled_fps=pooled, individual_fps=[r['fps'] for r in rows],
                    matched_repeats=paired, all_available_repeats=present[(fixed, mode)],
                    stage_statistics={k: statistics([x for n in names for x in samples[n][k]]) for k in samples[names[0]]})
        gain = {mode: dict(pooled=cells['fixed'][mode]['pooled_fps']/cells['auto'][mode]['pooled_fps'],
            paired=[a/b for a,b in zip(cells['fixed'][mode]['individual_fps'], cells['auto'][mode]['individual_fps'])]) for mode in MODES}
        thread = {clock: {arm: cells[clock][arm]['pooled_fps']/cells[clock][arm+'_default']['pooled_fps']
            for arm in ('v26', 'combined')} for clock in ('auto', 'fixed')}
        integration = {clock: {setting: cells[clock]['combined'+setting]['pooled_fps']/cells[clock]['v26'+setting]['pooled_fps']
            for setting in ('', '_default')} for clock in ('auto', 'fixed')}
        result[clip] = dict(matched_repeats=paired, cells=cells, clock_speedups=gain,
            one_thread_vs_inherited=thread, combined_vs_gpu=integration)
    return result


def telemetry(rows, trials, execution):
    require(rows and all(a['monotonic_ns'] < b['monotonic_ns'] for a,b in zip(rows, rows[1:])), 'Telemetry sequence invalid')
    require(all(r['trial'] in trials and r['phase'] in ('settle', 'video') for r in rows), 'Telemetry outside completed scope')
    peak = max(v for r in rows for v in r['temperatures'].values() if type(v) is int)
    require(peak < 75000, 'Thermal cutoff violated in recorded samples')
    results = {}
    for n in trials:
        frames = execution[n]['frames']
        begin, end = frames[32]['cpu_prepare_start_ns'], frames[-1]['consumer_complete_ns']
        subset = [r for r in rows if r['trial'] == n and r['phase'] == 'video' and begin <= r['monotonic_ns'] <= end]
        require(len(subset) >= 2, 'Missing steady processing telemetry '+n)
        clocks = {}
        for name in ('cpu0', 'cpu4', 'cpu8', 'gpu'):
            values = [r['clocks'][name] for r in subset if type(r['clocks'].get(name)) is int]
            require(values, 'No readable steady frequency '+n+' '+name)
            clocks[name] = statistics(values)
            target = 1300500000 if name == 'gpu' else 2201600
            clocks[name]['samples_at_fixed_target'] = sum(x == target for x in values)
        results[n] = dict(samples=len(subset), frames='32 through 127', clocks=clocks,
            maximum_temperature_c=max(v for r in subset for v in r['temperatures'].values() if type(v) is int)/1000)
    return dict(recorded_samples=len(rows), maximum_temperature_c=peak/1000, trials=results,
        caveat='500 ms snapshots, not execution-weighted counters or proof against every transient throttle. CPU policy frequencies are not consumer-affinity measurements.')


def verify(evidence):
    f = read(evidence/'freeze.json'); run = evidence/'run'
    batch, status = read(run/'batch.json'), read(run/'status.json')
    schedule = specs()
    require(f['pre_run'] and f['schedule'] == batch['schedule'] == schedule, 'Changed planned schedule')
    require(not f['settings_changed'] and not f['media_accessed'], 'Preparation scope changed')
    require(f['v31_freeze_sha256'] == sha(V31_LOCAL/'freeze.json') and f['v31_batch_sha256'] == sha(V31_LOCAL/'batch.json'), 'Changed v31 dependency')
    expected_sources = {'visible_clocks_v32.py', 'test_visible_clocks_v32.py', 'visible_clocks_v32_plan.md'}
    require(set(f['sources']) == expected_sources, 'Source freeze incomplete')
    for name, digest in f['sources'].items():
        sub = 'tests/unit' if name.startswith('test_') else 'docs' if name.endswith('.md') else 'scripts'
        require(digest == sha(evidence/name) == sha(ROOT/sub/name), 'Changed frozen source '+name)
    require(f['unit_log_sha256'] == sha(evidence/'unit.log') and '\nOK\n' in (evidence/'unit.log').read_text(), 'Unit gate failed')
    require(f['dependency_preflight_sha256'] == sha(evidence/'dependency_preflight.json') and not read(evidence/'dependency_preflight.json')['returncode'], 'Dependency preflight failed')
    identity = read(run/'run_identity.json')
    require(identity['freeze_sha256'] == sha(evidence/'freeze.json') and identity['child_uid'] == 1000 and identity['uid'] == 0, 'Controller/child provenance changed')
    require(not status['running'] and not batch['completed'] and len(batch['rows']) == status['completed'] == 46, 'Unexpected run state')
    require(batch['error'] == status['error'] == "ValueError('Clock policy no longer matches declared arm')", 'Different failure needs investigation')
    require(not batch['settings_restored'] and not status['settings_restored'] and batch['diagnostic_only']
        and not any(batch[k] for k in ('raw16_accessed', 'defaults_changed', 'full_regression_run')), 'Scope/restoration flag changed')
    original = read(run/'original_policy.json')
    controller, watchdog = read(run/'controller_restoration.json'), read(run/'watchdog_restoration.json')
    require(original == f['policy'] == controller['saved'] == watchdog['saved'], 'Saved policies differ')
    require(not controller['restored'] and controller['actual'] != original and not controller['errors'], 'Unexpected controller restoration result')
    require(watchdog['restored'] and not watchdog['errors'] and watchdog['actual'] == original
        and watchdog['reason'] == 'normal_exit', 'Final watchdog restoration not verified')
    frozen, generated, geometry = verify_sources(V29_LOCAL)
    tracking_hash = transformed_hash()
    reference = read(V31_LOCAL/'smoke_0126.v31.json')['runtime_before']
    def runtime(info, one):
        expected_env = dict.fromkeys(('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','GOTO_NUM_THREADS'))
        if one:
            expected_env['OPENBLAS_NUM_THREADS'] = '1'
        require(info['blas'] == [dict(reference['blas'][0], threads=1 if one else 12)]
            and info['thread_environment'] == expected_env and info['affinity'] == reference['affinity']
            and info['numpy'] == reference['numpy'] and info['opencv'] == reference['opencv'], 'Runtime/library/thread identity changed')
    trials, samples, execution = {}, {}, {}
    for spec, row in zip(schedule, batch['rows']):
        n, mode = spec['name'], spec['mode']
        require(all(row[k] == v for k,v in spec.items()) and row['returncode'] == 0, 'Order/scope/return code changed')
        r = read(run/(n+'.v31.json'))
        require(r['passed'] and r['error'] is None and r['frames'] == r['count'] == 128
            and r['clip'] == spec['clip'] and r['mode'] == mode and r['audit'] == spec['audit'] and not r['traced'], 'Invalid child receipt')
        require(r['arm'] == mode.removesuffix('_default') and r['thread_policy'] == ('inherited' if mode.endswith('_default') else 'one'), 'Wrong child mode')
        require(not any(r[k] for k in ('raw16_accessed','defaults_changed','algorithm_changed')), 'Changed child scope')
        require(r['freeze_sha256'] == sha(V31_LOCAL/'freeze.json') and row['v31_sha256'] == sha(run/(n+'.v31.json'))
            and row['receipt_sha256'] == r['receipt_sha256'] == sha(run/(n+'.v29.json'))
            and row['log_sha256'] == sha(run/(n+'.log')), 'Changed child/source/timing provenance')
        cmd = ['/usr/bin/python3', str(OLD_RUNNER), '--clip', spec['clip'], '--mode', mode,
            '--output', str(REMOTE/'run'/n), '--frames', '128'] + (['--audit'] if spec['audit'] else [])
        require(row['command'] == cmd, 'Command changed')
        for info in (r['runtime_before'], r['runtime_after']):
            runtime(info, not mode.endswith('_default'))
        require(r['runtime_after']['opencv_threads'] == 2, 'OpenCV policy changed')
        trial_spec = dict(name=n, clip=spec['clip'], arm=r['arm'], frames=128, state_audit=spec['audit'])
        trial, values = audit_relocated_trial(run, trial_spec, row, frozen, generated, geometry, tracking_hash)
        require(trial['count'] == r['count'] and trial['fps'] == r['fps'] and trial['wall_s'] == r['wall_s'], 'Timing mismatch')
        trial.update(mode=mode, fixed=spec['fixed'])
        trials[n], samples[n] = trial, values
        execution[n] = read(run/(n+'.v29.json'))['execution']
        print('Verified '+n, flush=True)
    rows = [__import__('json').loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    timing_trials = {n:r for n,r in trials.items() if not r['state_audit']}
    missing = schedule[len(batch['rows']):]
    require(not any(r['trial'] == missing[0]['name'] for r in rows), 'Unexpected failed trial telemetry')
    return dict(verified_completed_trials=True, experiment_completed=False, partial=True, trials=trials,
        completed_trials=len(trials), scheduled_trials=50, missing=missing, matched_comparisons=summarize(timing_trials, samples),
        telemetry=telemetry(rows, trials, execution), frame_instances=sum(r['count'] for r in trials.values()),
        unique_development_frames=256, controller_restoration=controller, watchdog_restoration=watchdog,
        original_recorded_settings_restored=batch['settings_restored'], current_settings_require_separate_readback=True,
        start_utc=identity['started_utc'], end_utc=datetime.datetime.fromtimestamp(batch['finished_ns']/1e9, datetime.timezone.utc).isoformat(),
        error=batch['error'], raw16_accessed=False, full_regression_run=False, defaults_changed=False, production_approved=False,
        verifier_sha256=sha(__file__), freeze_sha256=sha(evidence/'freeze.json'), batch_sha256=sha(run/'batch.json'),
        caveat='Partial file-replay diagnostic. Heavy has 3 matched repeats, light 2; extra fixed light repeat is retained but excluded from paired summaries. Not a completed gate, holdout accuracy evaluation, or live camera latency.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    write(args.output, verify(args.evidence))
