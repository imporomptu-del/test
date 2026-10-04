"""Independent report-only v23 audit; never opens video or holdout manifests."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_visible_v17 import read, sha, write
from verify_visible_v17 import require, distribution
from verify_visible_v20 import baseline
from run_visible_overlap_v23 import FROZEN, validate_snapshot
from run_visible_v20 import FROZEN as V20_FROZEN
from tracking_geometry_v20 import OLD, NEW, REFERENCE_SHA
from check_tracking_geometry_v20 import primitive_cases
from tiny_target.tracking.kalman import KalmanTrackManager

V20 = ROOT/'results/tiny_target/visible_speed_v20_20260918/evidence'
V22 = ROOT/'results/tiny_target/gpu_timeline_v22_20260920/evidence'
CLIPS = ('0029', '0126', '0055', '0082')
WORKLOADS = ('0126', '0082')


def local_source(name):
    if name.startswith('test_'):
        return ROOT/'tests/unit'/name
    return ROOT/('docs' if name.endswith('.md') else 'scripts')/name


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, 'Invalid '+label)


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        require(start <= end, 'Reversed host interval')
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersection_ns(left, right):
    left, right = union(left), union(right)
    total = i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1])-max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def host_overlap(rows):
    samples, preparation = [], 0
    for previous, current in zip(rows, rows[1:]):
        work = [(current['prepare_start_ns'], current['prepare_end_ns'])]
        main = [(previous[stage+'_start_ns'], previous[stage+'_end_ns'])
                for stage in ('detector', 'tracking')]
        samples.append(intersection_ns(work, main)/1e6)
        preparation += current['prepare_end_ns']-current['prepare_start_ns']
    return dict(pairs=len(samples), pairs_with_overlap=sum(v > 0 for v in samples),
        overlap=distribution(samples) if samples else None, total_overlap_ms=sum(samples),
        fraction_of_next_frame_preparation=(sum(samples)*1e6/preparation if preparation else 0.0),
        interval_definition='Preparation of frame n+1 intersected with the union of detector '
            'and tracker host intervals for frame n. Nested/overlapped time is not double counted.',
        proves_gpu_overlap=False)


def prefix_schedule():
    result = [dict(name='serial_'+clip, clip=clip, mode='serial', frames=128, traced=False)
              for clip in WORKLOADS]
    result += [dict(name='smoke_overlap_'+clip, clip=clip, mode='overlap', frames=128, traced=False)
               for clip in WORKLOADS]
    for clip in WORKLOADS:
        for i in range(3):
            for mode in (('reference', 'overlap') if i % 2 == 0 else ('overlap', 'reference')):
                result.append(dict(name=f'{clip}_repeat{i}_{mode}', clip=clip,
                                   mode=mode, frames=128, traced=False))
    return result


def full_schedule():
    return [dict(name='full_'+clip+'_overlap', clip=clip, mode='overlap', frames=None, traced=False)
            for clip in CLIPS]


def trace_schedule():
    return [dict(name='trace_'+clip+'_overlap', clip=clip, mode='overlap', frames=128, traced=True)
            for clip in WORKLOADS]


def verify_dependencies():
    """Recheck the frozen v20 helper identity, not an unrelated full benchmark."""
    f = read(V20/'freeze.json')
    require(set(f['files']) == set(V20_FROZEN), 'Incomplete v20 dependency freeze')
    for name, digest in f['files'].items():
        require(digest == sha(V20/name) == sha(local_source(name)), 'Changed v20 dependency '+name)
    g, b = read(V20/'generated_01.json'), read(V20/'build/build.json')
    require(f['gate_sha256'] == sha(V20/'generated_01.json')
            and f['build_sha256'] == sha(V20/'build/build.json'), 'Changed v20 dependency gate')
    require(g['passed'] and g['error'] is None and not g['real_media_read']
            and g['reference_sha256'] == REFERENCE_SHA == sha(ROOT/'tiny_target/tracking/kalman.py'),
            'Invalid v20 generated reference')
    require([c['name'] for c in g['cases']] == [r[0] for r in primitive_cases()]
            and len(g['cases']) == 136 and all(c['exact'] and c['inputs_unchanged'] for c in g['cases'])
            and {c['native'] for c in g['cases']} == {True, False}, 'Incomplete v20 primitive gate')
    require([r['scenario'] for r in g['replays']] == list(range(12))
            and all(r['exact'] and r['frames'] == 32 for r in g['replays']), 'Incomplete v20 replay gate')
    require(set(g['source_sha256']) == {'tracking_geometry_v20.cpp', 'tracking_geometry_v20.py',
            'build_tracking_geometry_v20.py', 'check_tracking_geometry_v20.py'}, 'Incomplete helper provenance')
    for name, digest in g['source_sha256'].items():
        require(digest == sha(V20/name) == sha(local_source(name)), 'Changed helper source '+name)
    require(b['returncode'] == 0 and b['source_sha256'] == g['source_sha256']['tracking_geometry_v20.cpp']
            and b['builder_sha256'] == g['source_sha256']['build_tracking_geometry_v20.py']
            and b['library_sha256'] == g['library_sha256'] == sha(V20/'build/libtracking_geometry_v20.so')
            and '-fno-fast-math' in b['command'] and '-ffp-contract=off' in b['command'],
            'Changed or unsafe helper build')
    source = inspect.getsource(KalmanTrackManager.update)
    require(source.count(OLD) == 1, 'Reference geometry anchor changed')
    transformed = hashlib.sha256(textwrap.dedent(source.replace(OLD, NEW)).encode()).hexdigest()
    require(g['transformed_sha256'] == transformed, 'Changed helper transformation')
    return dict(freeze_sha256=sha(V20/'freeze.json'), generated_sha256=sha(V20/'generated_01.json'),
                library_sha256=g['library_sha256'], transformed_sha256=transformed)


def verify_sources(evidence):
    frozen, generated = read(evidence/'freeze.json'), read(evidence/'generated_01.json')
    require(set(frozen['files']) == set(FROZEN), 'Incomplete v23 freeze')
    for name, digest in frozen['files'].items():
        require(digest == sha(evidence/name) == sha(local_source(name)), 'Changed v23 source '+name)
    require(generated['passed'] and generated['returncode'] == 0 and not generated['real_media_read']
            and generated['test_sha256'] == frozen['files']['test_frame_lookahead_v23.py']
            and generated['adapter_test_sha256'] == frozen['files']['test_visible_overlap_v23.py']
            and generated['adapter_sha256'] == frozen['files']['visible_overlap_v23.py']
            and generated['controller_sha256'] == frozen['files']['frame_lookahead_v23.py']
            and generated['log_sha256'] == sha(evidence/'generated_01.log')
            and frozen['generated_sha256'] == sha(evidence/'generated_01.json'), 'Ownership tests failed/changed')
    return frozen, generated


def audit_trial(evidence, spec, row, frozen, dependency):
    name = spec['name']
    require(re.fullmatch(r'[a-zA-Z0-9_]+', name) is not None, 'Unsafe trial name')
    path = evidence/name
    r = read(path.with_suffix('.v23.json'))
    require(row['returncode'] == 0 and r['passed'] and r['error'] is None, 'Claimed successful trial failed')
    require(all(r[k] == spec[k] for k in ('clip', 'mode', 'frames', 'traced')), 'Trial scope changed')
    require(r['schema'] == 'seaqr.visible-overlap-v23.v1'
            and not any(r[k] for k in ('raw16_accessed', 'defaults_changed', 'gpu_changed',
                                      'algorithm_changed', 'production_approved')), 'Algorithm or scope changed')
    require(r['source_sha256'] == frozen['files'] and r['freeze_sha256'] == sha(evidence/'freeze.json')
            and r['generated_sha256'] == sha(evidence/'generated_01.json')
            and r['baseline_freeze_sha256'] == dependency['freeze_sha256']
            and row['v23_sha256'] == sha(path.with_suffix('.v23.json'))
            and r['baseline_receipt_sha256'] == sha(path.with_suffix('.v20.json')), 'v23 receipt chain changed')
    v20 = read(path.with_suffix('.v20.json'))
    require(v20['passed'] and v20['error'] is None and v20['mode'] == 'candidate'
            and all(v20[k] == spec[k] for k in ('clip', 'frames'))
            and not any(v20[k] for k in ('raw16_accessed', 'defaults_changed', 'gpu_changed',
                                       'noise_v18_enabled', 'median_v19_enabled')), 'v20 reference candidate changed')
    require(v20['script_sha256'] == sha(V20/'run_visible_v20.py')
            and v20['freeze_sha256'] == dependency['freeze_sha256']
            and v20['gate_sha256'] == dependency['generated_sha256']
            and v20['library_sha256'] == dependency['library_sha256']
            and v20['transformed_sha256'] == dependency['transformed_sha256']
            and v20['baseline_receipt_sha256'] == sha(path.with_suffix('.v17.json'))
            and v20['native_geometry_calls'] > 0 and v20['geometry_fallbacks'] >= 0, 'v20 identity/geometry execution changed')
    v17, report, motion = baseline(path, spec['clip'], spec['frames'])
    count = report['frames']
    require(count == r['processed_frames'] == v20['processed_frames'], 'Frame count changed')
    for key in ('fps', 'wall_s'):
        positive(r[key], key)
        require(r[key] == row[key] == v20[key] == v17[key], 'Timing receipt chain changed: '+key)
    require(math.isclose(r['fps'], count/r['wall_s'], rel_tol=1e-12), 'FPS denominator changed')
    snap = r['execution']
    require(snap['mode'] == spec['mode'], 'Wrong adapter mode')
    validate_snapshot(snap, count)
    consumer = snap['consumer_thread_id']
    require(isinstance(consumer, int) and consumer > 0, 'Missing recorded consumer identity')
    require((snap['engine'] is None) == (spec['mode'] == 'reference'), 'Unexpected worker execution')
    if snap['engine'] is not None:
        require(snap['engine']['owner_thread_id'] != consumer, 'VPI worker ran on consumer thread')
    push_pop = r['nvtx_push_pop_counts']
    require(push_pop == ([3*count, 3*count] if spec['traced'] else [0, 0]), 'Annotation coverage changed')
    if spec['traced']:
        require(r['bridge_sha256'] == sha(V22/'libnvtx_bridge.so'), 'Unrecognized diagnostic bridge')
    else:
        require(r['bridge_sha256'] is None, 'Unprofiled benchmark used annotation bridge')
    positive(r['process_peak_rss_kib'], 'process peak RSS')
    rows = snap['frames']
    for item in rows:
        for key, value in item.items():
            if key.endswith('_ns'):
                require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                        'Invalid event timestamp '+key)
    for left, right in zip(rows, rows[1:]):
        require(left['consumer_complete_ns'] <= right['consumer_received_ns']
                and left['prepare_end_ns'] <= right['prepare_start_ns'], 'Frame stages ran out of order')
    times = dict(cadence=v17['consumer_frame_ms'],
        queue_aware=[(x['consumer_complete_ns']-x['ready_ns'])/1e6 for x in rows],
        preparation=[(x['prepare_end_ns']-x['prepare_start_ns'])/1e6 for x in rows],
        queue_wait=[x['queue_wait_ms'] for x in rows])
    metrics = dict(**spec, count=count, fps=r['fps'], wall_s=r['wall_s'], exact=True,
        geometry_calls=v20['native_geometry_calls'], geometry_fallbacks=v20['geometry_fallbacks'],
        process_peak_rss_kib=r['process_peak_rss_kib'], engine=snap['engine'],
        cadence=distribution(times['cadence']), queue_aware_latency=distribution(times['queue_aware']),
        preparation=distribution(times['preparation']), queue_wait=distribution(times['queue_wait']),
        host_overlap=host_overlap(rows), stage_means_ms={k:v['mean'] for k,v in report['timings_ms'].items()},
        old_motion_timing_is_handoff_only=spec['mode'] != 'reference',
        v23_sha256=sha(path.with_suffix('.v23.json')))
    metrics['latency_decomposition'] = dict(
        ready_to_prepare_start=distribution([(x['prepare_start_ns']-x['ready_ns'])/1e6 for x in rows]),
        preparation=distribution(times['preparation']),
        prepare_end_to_completion=distribution([(x['consumer_complete_ns']-x['prepare_end_ns'])/1e6 for x in rows]),
        prepared_queue=(distribution([(x['consumer_received_ns']-x['prepare_end_ns'])/1e6 for x in rows])
                        if spec['mode'] != 'reference' else None),
        warning='The first three per-frame durations sum to queue-aware latency. Their percentiles '
                'do not sum. Prepared queue is a subset of prepare-end-to-completion, not an extra term.')
    return metrics, times


def performance(trials, samples):
    per_clip = {}
    for clip in WORKLOADS:
        names = [f'{clip}_repeat{i}_{mode}' for i in range(3) for mode in ('reference', 'overlap')]
        if not all(name in trials for name in names):
            continue
        arms = {}
        for mode in ('reference', 'overlap'):
            selected = [trials[f'{clip}_repeat{i}_{mode}'] for i in range(3)]
            count, wall = sum(r['count'] for r in selected), sum(r['wall_s'] for r in selected)
            require(count == 384 and not any(r['traced'] for r in selected), 'Invalid paired timing scope')
            arms[mode] = dict(frames=count, wall_s=wall, fps=count/wall,
                cadence=distribution([v for r in selected for v in samples[r['name']]['cadence']]),
                queue_aware_latency=distribution([v for r in selected for v in samples[r['name']]['queue_aware']]),
                preparation=distribution([v for r in selected for v in samples[r['name']]['preparation']]),
                process_peak_rss_kib=[r['process_peak_rss_kib'] for r in selected])
        paired = [trials[f'{clip}_repeat{i}_overlap']['fps']/trials[f'{clip}_repeat{i}_reference']['fps']
                  for i in range(3)]
        regressions = {}
        for key in ('cadence', 'queue_aware_latency'):
            ratios = [trials[f'{clip}_repeat{i}_overlap'][key]['p95_ms']/
                      trials[f'{clip}_repeat{i}_reference'][key]['p95_ms'] for i in range(3)]
            pooled_worse = arms['overlap'][key]['p95_ms'] > arms['reference'][key]['p95_ms']
            regressions[key] = dict(pooled_worse=pooled_worse, paired_p95_ratios=ratios,
                worse_pairs=sum(x > 1 for x in ratios), consistent=pooled_worse and sum(x > 1 for x in ratios) >= 2)
        speedup = arms['overlap']['fps']/arms['reference']['fps']
        per_clip[clip] = dict(arms=arms, speedup=speedup, paired_speedups=paired,
            p95_regressions=regressions,
            passed=speedup >= 1.2 and all(x > 1 for x in paired)
                and not any(r['consistent'] for r in regressions.values()))
    complete = set(per_clip) == set(WORKLOADS)
    return dict(complete=complete, passed=complete and all(v['passed'] for v in per_clip.values()),
        by_clip=per_clip, criterion='At least 1.2 pooled FPS ratio on each workload, all three paired ratios > 1; '
            'reject any cadence or queue-aware p95 that is worse pooled and in at least two of three pairs.')


def verify(evidence):
    evidence = evidence.resolve(strict=True)
    frozen, generated = verify_sources(evidence)
    dependency = verify_dependencies()
    batch_path = evidence/'batch.json'
    batch = read(batch_path)
    require(batch['source_sha256'] == frozen['files'] and not batch['raw16_accessed']
            and not batch['defaults_changed'], 'Batch identity or scope changed')
    specs, rows = batch['schedule'], batch['rows']
    prefix, full, traces = prefix_schedule(), full_schedule(), trace_schedule()
    valid_schedules = [prefix, prefix+traces, prefix+full, prefix+full+traces]
    require(specs in valid_schedules, 'Changed or unbounded batch schedule')
    require(len(rows) <= len(specs), 'More batch rows than scheduled runs')
    trials, samples, failed = {}, {}, []
    for spec, row in zip(specs, rows):
        require(all(row[k] == v for k,v in spec.items()), 'Batch order/scope changed')
        log_name = Path(row['log']).name
        require(log_name == spec['name']+'.log'
                and row['log_sha256'] == sha(evidence/log_name), 'Trial log changed')
        path = evidence/spec['name']
        receipt_path = path.with_suffix('.v23.json')
        receipt = read(receipt_path) if receipt_path.exists() else None
        if row['returncode'] != 0:
            # Nsight post-processing can fail after a valid pipeline receipt.
            # Audit that receipt too, but never turn the failed command into a
            # clean performance sample or claim the diagnostic is complete.
            pipeline_verified = False
            if receipt is not None and receipt.get('passed'):
                synthetic_row = dict(row, returncode=0, v23_sha256=sha(receipt_path),
                                     fps=receipt['fps'], wall_s=receipt['wall_s'])
                audit_trial(evidence, spec, synthetic_row, frozen, dependency)
                pipeline_verified = True
            if receipt is not None and row.get('v23_sha256') is not None:
                require(row['v23_sha256'] == sha(receipt_path), 'Failed receipt changed')
            failed.append(dict(**spec, returncode=row['returncode'],
                error=None if receipt is None else receipt.get('error'), log=log_name,
                receipt_present=receipt is not None, pipeline_receipt_verified=pipeline_verified,
                receipt_sha256=sha(receipt_path) if receipt is not None else None))
            require(len(rows) == len(trials)+len(failed), 'Batch continued after a failed trial')
            break
        metrics, durations = audit_trial(evidence, spec, row, frozen, dependency)
        trials[spec['name']], samples[spec['name']] = metrics, durations
    gate = performance(trials, samples)
    if batch['performance'] is not None:
        require(gate['complete'], 'Recorded performance before all clean pairs completed')
        projected = dict(passed=gate['passed'], clips={})
        for clip, item in gate['by_clip'].items():
            latency = {}
            for remote, local in (('queue_aware', 'queue_aware_latency'), ('consumer_cadence', 'cadence')):
                measure = item['p95_regressions'][local]
                latency[remote] = dict(pooled_p95_ms={mode:item['arms'][mode][local]['p95_ms']
                                                    for mode in ('reference', 'overlap')},
                    paired_p95_ratios=measure['paired_p95_ratios'], consistent_regression=measure['consistent'])
            projected['clips'][clip] = dict(pooled_fps={mode:item['arms'][mode]['fps']
                                                       for mode in ('reference', 'overlap')},
                speedup=item['speedup'], paired_speedups=item['paired_speedups'], latency=latency, passed=item['passed'])
        require(batch['performance'] == projected, 'Batch performance gate differs from independent computation')
    if full[0] in specs:
        require(gate['complete'] and gate['passed'], 'Full regressions scheduled without earned prefix gate')
    if any(s['traced'] for s in specs):
        require(gate['complete'] and all(s['name'] in trials for s in prefix), 'Traces scheduled before clean prefix checks')
    observed_schedule_complete = len(rows) == len(specs) and not failed
    completed = observed_schedule_complete and bool(batch.get('passed'))
    if batch.get('passed'):
        require(completed and batch.get('error') is None and batch['performance'] is not None,
                'Batch claimed success before verified completion')
        require(specs == prefix+(full if gate['passed'] else [])+traces,
                'Completed batch omitted required validation/diagnostics')
        require(batch['decision'] == ('candidate_requires_independent_audit' if gate['passed'] else 'reject_v23_keep_v20'),
                'Batch decision contradicts its acceptance gate')
    full_trials = [r for r in trials.values() if r['frames'] is None]
    full_complete = {r['clip'] for r in full_trials} == set(CLIPS)
    require(batch['full_regression_run'] == full_complete, 'Full regression completion flag changed')
    traced = [r for r in trials.values() if r['traced']]
    full_count = sum(r['count'] for r in full_trials)
    full_wall = sum(r['wall_s'] for r in full_trials)
    decision = ('incomplete_or_failed' if not completed else
                'reject_overlap_retain_v20' if not gate['passed'] else
                'prefix_gate_passed_full_regression_pending' if not full_complete else
                'opt_in_exact_overlap_candidate')
    return dict(schema='seaqr.visible-overlap-v23-summary.v1', verified=True,
        completed=completed, observed_schedule_complete=observed_schedule_complete,
        batch_passed=bool(batch.get('passed')), batch_error=batch.get('error'),
        completed_successful_runs=len(trials), scheduled_runs=len(specs), failed_runs=failed,
        pending_names=[r['name'] for r in specs[len(rows):]], decision=decision,
        prefix_performance=gate, trials=list(trials.values()),
        full_regression=dict(complete=full_complete, frames=full_count,
            clips=[r['clip'] for r in full_trials], fps=(full_count/full_wall if full_wall else None),
            scope='All four development clips' if full_complete else
                  'Not run: clean prefix performance gate rejected this candidate' if gate['complete'] and not gate['passed'] else
                  'Incomplete/not yet authorized by a passing clean prefix gate',
            fresh_paired_full_reference=False),
        traced_runs=[r['name'] for r in traced], trace_device_timeline_verified=False,
        generated_ownership_gate=generated, frozen_sources_sha256=frozen['files'],
        v20_dependency=dependency, batch_sha256=sha(batch_path), verifier_sha256=sha(__file__),
        verified_frame_instances=sum(r['count'] for r in trials.values()),
        exact_non_timing_outputs_for_successful_runs=True, raw16_paused=True,
        defaults_changed=False, production_approved=False, new_airborne_accuracy_validated=False,
        warning='Development regression only, not new accuracy/generalization or real-time validation. '
            'Host overlap is not CUDA execution overlap. Traced FPS is excluded from the performance gate. '
            'Queue-aware latency starts at grayscale completion, not camera acquisition. Peak RSS is a '
            'process-level high-water mark, not GPU allocation size. Existing motion stage timings in '
            'prepared modes measure proxy handoff only; use independently recorded preparation durations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.evidence)
    write(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials', 'generated_ownership_gate')}, indent=2))
