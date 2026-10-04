"""Independent, report-only full-video v34 audit. Never decodes media."""
import argparse
from collections import defaultdict
import datetime
import json
import math
from pathlib import Path

from profile_visible_interaction_v30 import read, write
from export_visible_validation_v34 import files_for, sha, SOURCES
from analyze_visible_interaction_v30 import analyze, require, statistics
from audit_visible_clocks_v33 import verify_stability
from verify_visible_combined_v29 import verify_sources, transformed_hash, ROOT, ARCHIVE
from verify_visible_interaction_v30 import V29_LOCAL, audit_relocated_trial

REMOTE = Path('/tmp/seaqr_visible_validation_v34_aqGC5d')
V31 = ROOT/'results/tiny_target/visible_threads_v31_20260920/evidence'
V33 = ROOT/'results/tiny_target/visible_clocks_v33_20260922/audit_20260923'
PREP = ROOT/'results/tiny_target/visible_validation_v34_20260923/preparation'
COUNTS = {'0029': 687, '0126': 674, '0055': 689, '0082': 691}


def specs():
    # Independent declaration, not imported from the supervisor or its receipts.
    rows = [(f'smoke_{c}', c, 'smoke', None) for c in ('0126', '0082')]
    for i, order in enumerate((('0029', '0126', '0055', '0082'),
            ('0082', '0055', '0126', '0029'), ('0055', '0029', '0082', '0126'))):
        rows += [(f'full_repeat{i}_{c}', c, 'full', i) for c in order]
    rows += [(f'trace_{c}', c, 'trace', None) for c in ('0126', '0082')]
    return [dict(name=n, clip=c, kind=k, repeat=i, mode='combined_default', fixed=True,
        audit=k == 'smoke', traced=k == 'trace', frames=None if k == 'full' else 128)
        for n, c, k, i in rows]


def distribution(values):
    result = statistics(values)
    ordered = sorted(values)
    index = (len(ordered)-1)*.99
    lo, hi = math.floor(index), math.ceil(index)
    result.update(p99=ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo),
        fraction_over_100_ms=sum(v > 100 for v in values)/len(values))
    return result


def occupancy(admission):
    endpoints = defaultdict(int)
    for e in admission['events']:
        require(0 < e['admitted_ns'] <= e['released_ns'], 'Invalid occupancy interval')
        endpoints[e['admitted_ns']] += 1
        endpoints[e['released_ns']] -= 1
    times = sorted(endpoints)
    require(len(times) >= 2, 'Missing occupancy interval')
    held = peak = 0
    durations = defaultdict(int)
    for i, tick in enumerate(times):
        held += endpoints[tick]
        require(0 <= held <= 2, 'Occupancy exceeds capacity')
        peak = max(peak, held)
        if i+1 < len(times):
            durations[held] += times[i+1]-tick
    require(held == 0 and peak == admission['maximum'], 'Occupancy receipt mismatch')
    total = times[-1]-times[0]
    return dict(maximum=peak, observed_span_s=total/1e9,
        time_fraction={str(k): durations[k]/total for k in (0, 1, 2)},
        mean_held=sum(k*v for k, v in durations.items())/total,
        caveat='Admission reservations include decoding, ready and processing frames, plus EOF; not ready-queue depth or live-camera backlog.')


def windows(frames, width=10):
    ticks = [f['consumer_complete_ns'] for f in frames]
    require(len(ticks) > width and all(b > a for a, b in zip(ticks, ticks[1:])),
        'Invalid completion sequence')
    return [width*1e9/(ticks[i+width]-ticks[i]) for i in range(len(ticks)-width)]


def summarize(trials, samples, executions):
    result = {}
    for c, count in COUNTS.items():
        names = [f'full_repeat{i}_{c}' for i in range(3)]
        rows = [trials[n] for n in names]
        require(all(r['kind'] == 'full' and not r['traced'] and not r['state_audit']
            and r['frames'] is None and r['count'] == count for r in rows), 'Contaminated clean timings')
        metrics = {k: distribution([v for n in names for v in samples[n][k]])
            for k in samples[names[0]]}
        result[c] = dict(frames_per_run=count, repeats=3,
            pooled_fps=sum(r['count'] for r in rows)/sum(r['wall_s'] for r in rows),
            individual_fps=[r['fps'] for r in rows], individual_wall_s=[r['wall_s'] for r in rows],
            frame_and_stage_ms=metrics,
            sliding_10_completion_intervals_fps={n: statistics(windows(executions[n]['frames'])) for n in names},
            maximum_process_peak_rss_mib=max(r['process_peak_rss_kib'] for r in rows)/1024,
            admission={n: occupancy(executions[n]['admission']) for n in names})
    return result


def transitions(evidence, batch, saved):
    run = evidence/'run'
    pre = read(run/'transition_preflight.json')
    require(sha(run/'transition_preflight.json') == batch['transition_preflight_sha256']
        and pre['passed'] and pre['error'] is None and not pre['media_accessed']
        and 0 <= pre['elapsed_s'] <= 30, 'Transition preflight failed')
    require(len(pre['transitions']) == 6 and len(batch['transitions']) == 16, 'Incomplete transitions')
    expected = [(f'preflight_{i}_{mode}', mode == 'fixed')
        for i in range(3) for mode in ('fixed', 'auto')]
    expected += [(s['name'], True) for s in specs()]
    checked = []
    for (label, fixed), row in zip(expected, pre['transitions']+batch['transitions']):
        path = run/'transitions'/(label+'.json')
        r = read(path)
        require(row['label'] == r['label'] == label and row['fixed'] == r['fixed'] == fixed
            and row['sha256'] == sha(path) and r['verified'] and r['error'] is None,
            'Changed/failed transition')
        require(all(r['before'][n][k] == saved[n][k] for n in saved for k in ('maximum', 'governor'))
            and row['elapsed_s'] == r['elapsed_s'] >= r['verification']['elapsed_s'], 'Changed transition bounds/timing')
        checked.append(dict(label=label, fixed=fixed, **verify_stability(r, saved, fixed, True)))
    restored = {}
    for who in ('controller', 'watchdog'):
        r = read(run/(who+'_restoration.json'))
        require(r['restored'] and not r['errors'] and r['actual'] == r['saved'] == saved,
            'Restoration failed')
        if who == 'watchdog':
            require(r['reason'] == 'normal_exit', 'Abnormal watchdog exit')
        restored[who] = verify_stability(r, saved, False, False)
    return dict(verified=True, transitions=checked, restorations=restored)


def runtime(actual, expected, after=False):
    for key in ('blas', 'affinity', 'numpy', 'opencv', 'thread_environment', 'clock_ticks'):
        require(actual[key] == expected[key], 'Changed runtime '+key)
    require(len(actual['blas']) == 1 and actual['blas'][0]['threads'] == 12
        and actual['thread_environment'] == dict.fromkeys(('OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GOTO_NUM_THREADS')), 'Wrong thread policy')
    if after:
        require(actual['opencv_threads'] == 2, 'Wrong OpenCV worker count')


def telemetry(path, executions):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(rows and all(a['monotonic_ns'] < b['monotonic_ns'] for a, b in zip(rows, rows[1:])), 'Invalid telemetry sequence')
    for r in rows:
        require(r['trial'] in executions and r['phase'] in ('settle', 'video'), 'Telemetry outside scope')
        require(all(type(r['temperatures'].get(n)) is int and 0 < r['temperatures'][n] < 75000
            for n in ('cpu-thermal', 'tj-thermal')), 'Unsafe/missing mandatory temperature')
        require(all(0 < v < 75000 for v in r['temperatures'].values() if type(v) is int), 'Unsafe optional temperature')
    result = {}
    for n, execution in executions.items():
        frames = execution['frames']
        begin, end = frames[32]['cpu_prepare_start_ns'], frames[-1]['consumer_complete_ns']
        subset = [r for r in rows if r['trial'] == n and r['phase'] == 'video'
            and begin <= r['monotonic_ns'] <= end]
        require(len(subset) >= 2, 'Missing steady telemetry '+n)
        clocks = {}
        for policy in ('cpu0', 'cpu4', 'cpu8', 'gpu'):
            values = [r['clocks'][policy] for r in subset if type(r['clocks'].get(policy)) is int]
            target = 1300500000 if policy == 'gpu' else 2201600
            clocks[policy] = dict(**statistics(values), samples_at_fixed_target=sum(v == target for v in values))
        result[n] = dict(first_frame=32, last_frame=len(frames)-1, samples=len(subset), clocks=clocks,
            maximum_temperature_c=max(v for r in subset for v in r['temperatures'].values() if type(v) is int)/1000)
    return dict(recorded_samples=len(rows), trials=result,
        maximum_temperature_c=max(v for r in rows for v in r['temperatures'].values() if type(v) is int)/1000,
        caveat='500 ms snapshots, not execution-weighted utilization or proof against all transient throttling.')


def verify(evidence):
    schedule = specs()
    manifest = read(evidence/'export_manifest_v34_01.json')
    require(manifest['post_run'] and not manifest['media_included']
        and set(manifest['files']) == set(files_for(schedule)), 'Invalid export scope')
    for name, digest in manifest['files'].items():
        require(sha(evidence/name) == digest, 'Changed export '+name)
    require(sha(evidence/'export_visible_validation_v34.py') == sha(ROOT/'scripts/export_visible_validation_v34.py'), 'Changed exporter')
    f = read(evidence/'freeze.json')
    require(sha(evidence/'freeze.json') == sha(PREP/'freeze.json') and f['pre_run']
        and not f['settings_changed'] and not f['media_accessed'], 'Changed preparation')
    require(set(f['sources']) == set(SOURCES), 'Incomplete source freeze')
    for name, digest in f['sources'].items():
        local = V33/'summary_verified_01.json' if name == 'v33_audit.json' else ROOT/(
            'tests/unit' if name.startswith('test_') else 'docs' if name.endswith('.md') else 'scripts')/name
        require(digest == sha(evidence/name) == sha(local), 'Changed frozen source '+name)
    require(f['v31_freeze_sha256'] == sha(V31/'freeze.json')
        and f['v31_batch_sha256'] == sha(V31/'batch.json'), 'Changed v31 dependency')
    require(f['unit_log_sha256'] == sha(evidence/'unit.log') and '\nOK\n' in (evidence/'unit.log').read_text(), 'Failed unit gate')
    require(f['dependency_preflight_sha256'] == sha(evidence/'dependency_preflight.json')
        and read(evidence/'dependency_preflight.json')['returncode'] == 0, 'Failed dependency preflight')
    prior = read(evidence/'v33_audit.json')
    require(prior['verified_completed_trials'] and prior['experiment_completed'] and not prior['partial']
        and prior['completed_trials'] == prior['scheduled_trials'] == 50 and not prior['missing']
        and prior['original_recorded_settings_restored'] and prior['transition_audit']['verified']
        and prior['freeze_sha256'] == sha(V33/'evidence/freeze.json')
        and prior['batch_sha256'] == sha(V33/'evidence/run/batch.json')
        and prior['verifier_sha256'] == sha(evidence/'audit_visible_clocks_v33.py'), 'Failed prerequisite')
    run = evidence/'run'
    batch, status, identity = (read(run/n) for n in ('batch.json', 'status.json', 'run_identity.json'))
    require(f['schedule'] == batch['schedule'] == schedule and len(batch['rows']) == 16
        and not status['running'] and status['completed'] == status['scheduled'] == 16
        and batch['completed'] and batch['error'] is status['error'] is None, 'Incomplete or changed run')
    require(batch['settings_restored'] and status['settings_restored'] and batch['full_regression_run']
        and batch['diagnostic_only'] and not batch['raw16_accessed'] and not batch['defaults_changed'], 'Changed scope')
    require(identity['freeze_sha256'] == sha(evidence/'freeze.json') and identity['uid'] == 0
        and identity['child_uid'] == 1000, 'Changed execution identity')
    saved = read(run/'original_policy.json')
    require(saved == f['policy'], 'Changed saved policy')
    transition_audit = transitions(evidence, batch, saved)
    frozen, generated, geometry = verify_sources(V29_LOCAL)
    tracking_hash = transformed_hash()
    reference_runtime = read(V33/'evidence/run/0126_repeat0_fixed_combined_default.v31.json')['runtime_before']
    require(reference_runtime == f['runtime_reference'], 'Changed inherited runtime reference')
    references = {}
    for c, count in COUNTS.items():
        path = ARCHIVE/f'visible_{c}_full_reuse'
        launch, report = read(path/'launch.json'), read(path/'report.json')
        require(report['frames'] == count and report['full_clip'] and report['completed']
            and launch['fps'] == 10, 'Changed full reference')
        references[c] = dict(frames=count, nominal_file_fps=10, source_sha256=launch['source_sha256'],
            launch_sha256=sha(path/'launch.json'), report_sha256=sha(path/'report.json'))
    trials, samples, executions, traces = {}, {}, {}, {}
    for spec, row in zip(schedule, batch['rows']):
        n = spec['name']
        r = read(run/(n+'.v34.json'))
        require(all(row[k] == r[k] == v for k, v in spec.items()) and row['returncode'] == 0
            and r['passed'] and r['error'] is None and r['count'] == (COUNTS[spec['clip']] if spec['frames'] is None else 128), 'Invalid child receipt')
        require(not any(r[k] for k in ('raw16_accessed', 'defaults_changed', 'algorithm_changed'))
            and r['references'] == references, 'Changed child scope/reference')
        require(r['freeze_sha256'] == sha(evidence/'freeze.json') and row['v34_sha256'] == sha(run/(n+'.v34.json'))
            and row['receipt_sha256'] == r['receipt_sha256'] == sha(run/(n+'.v29.json'))
            and row['log_sha256'] == sha(run/(n+'.log')), 'Changed provenance')
        cmd = ['/usr/bin/python3', str(REMOTE/'run_visible_validation_v34.py'), '--trial', n, '--output', str(REMOTE/'run'/n)]
        if spec['traced']:
            cmd = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=process-tree',
                '--osrt-threshold=100000', '--cuda-flush-interval=0', '--kill=none', '--force-overwrite=false',
                '--export=sqlite', '--output='+str(REMOTE/'run'/n)] + cmd
        require(row['command'] == cmd, 'Changed command')
        runtime(r['runtime_before'], reference_runtime)
        runtime(r['runtime_after'], reference_runtime, after=True)
        trial_spec = dict(name=n, clip=spec['clip'], arm='combined', frames=spec['frames'], state_audit=spec['audit'])
        trial, values = audit_relocated_trial(run, trial_spec, row, frozen, generated, geometry, tracking_hash)
        require(trial['count'] == r['count'] and trial['fps'] == r['fps'] and trial['wall_s'] == r['wall_s'], 'Changed timing')
        trial.update(kind=spec['kind'], traced=spec['traced'], repeat=spec['repeat'])
        trials[n], samples[n] = trial, values
        executions[n] = read(run/(n+'.v29.json'))['execution']
        if spec['traced']:
            for suffix in ('.trace30.json', '.sqlite', '.nsys-rep'):
                require(row[suffix[1:]+'_sha256'] == sha(run/(n+suffix)), 'Changed trace artifact')
            tr = read(run/(n+'.trace30.json'))
            require(r['trace_sha256'] == sha(run/(n+'.trace30.json'))
                and tr['clip'] == spec['clip'] and tr['arm'] == 'combined' and tr['frames'] == 128
                and tr['script_sha256'] == sha(ROOT/'scripts/profile_visible_interaction_v30.py')
                == read(V31/'freeze.json')['sources']['profile_visible_interaction_v30.py']
                and tr['baseline_freeze_sha256'] == sha(V29_LOCAL/'freeze.json')
                and tr['bridge_sha256'] == read(ROOT/'results/tiny_target/visible_interaction_v30_20260920/evidence/freeze.json')['bridge_sha256']
                and tr['receipt_sha256'] == row['receipt_sha256']
                and not any(tr[k] for k in ('raw16_accessed', 'defaults_changed', 'numerical_threads_changed')), 'Changed trace provenance')
            runtime(tr['runtime_before'], reference_runtime)
            runtime(tr['runtime_after'], reference_runtime, after=True)
            traces[n] = analyze(run/(n+'.sqlite'), run/(n+'.trace30.json'))
        print('Verified '+n, flush=True)
    return dict(verified=True, completed=True, scheduled_trials=16, completed_trials=len(trials),
        frame_instances=sum(r['count'] for r in trials.values()), unique_development_frames=sum(COUNTS.values()),
        clean_full_frame_instances=sum(r['count'] for r in trials.values() if r['kind'] == 'full'),
        clean_full=summarize(trials, samples, executions), trials=trials, traces=traces,
        transition_audit=transition_audit, telemetry=telemetry(run/'telemetry.jsonl', executions),
        start_utc=identity['started_utc'], end_utc=datetime.datetime.fromtimestamp(batch['finished_ns']/1e9, datetime.timezone.utc).isoformat(),
        recorded_settings_restored=True, live_policy_readback_is_separate=True,
        new_airborne_accuracy_validated=False, production_approved=False, raw16_accessed=False, defaults_changed=False,
        verifier_sha256=sha(__file__), analyzer_sha256=sha(ROOT/'scripts/analyze_visible_interaction_v30.py'),
        export_manifest_sha256=sha(evidence/'export_manifest_v34_01.json'), freeze_sha256=sha(evidence/'freeze.json'),
        caveat='Full development replay parity, not new detection accuracy or live-camera latency. Clean FPS includes decode/processing/journaling, not process startup/hash checks. Traces are separate 128-frame prefixes. Stage timings overlap decode and contain nested work; do not add across overlapping domains.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    write(a.output, verify(a.evidence))
