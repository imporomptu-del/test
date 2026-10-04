"""Independent report-only four-arm integration, journal and latency audit."""
import argparse
import json
import math
from pathlib import Path

from verify_visible_overlap_v23 import ROOT, local_source, verify_dependencies, positive
from verify_visible_v17 import ARCHIVE, read, sha, write, require, distribution
from verify_visible_front_v26 import verify_sources as verify_front_sources
from verify_visible_stage_v24 import verify_sources as verify_stage_sources
from verify_tracking_v28 import verify as verify_tracking
from run_visible_combined_v29 import SOURCES
from run_visible_stage_v24 import validate_snapshot
from video_checks_v19 import check

V26 = ROOT/'results/tiny_target/visible_front_v26_20260920/evidence'
V28 = ROOT/'results/tiny_target/tracking_v28_20260920/evidence'
V27 = ROOT/'results/tiny_target/tracking_v27_20260920/evidence'
V24 = ROOT/'results/tiny_target/visible_stage_v24_20260920/evidence'
ARMS = ('v20', 'v26', 'v28', 'combined')
WORKLOADS = ('0126', '0082')
CLIPS = ('0029', '0126', '0055', '0082')
LATENCY = ('consumer_cadence', 'queue_aware', 'request_to_complete')


def schedules():
    prefix = [dict(name='smoke_'+c, clip=c, arm='combined', frames=128, state_audit=True) for c in WORKLOADS]
    orders = (ARMS, tuple(reversed(ARMS)), ('v28', 'v20', 'combined', 'v26'))
    prefix += [dict(name=f'{c}_repeat{i}_{a}', clip=c, arm=a, frames=128, state_audit=False)
               for c in WORKLOADS for i, order in enumerate(orders) for a in order]
    full = [dict(name='full_'+c, clip=c, arm='combined', frames=None, state_audit=False) for c in CLIPS]
    return prefix, full


def performance(trials, samples):
    result = {}
    for clip in WORKLOADS:
        names = {a: [f'{clip}_repeat{i}_{a}' for i in range(3)] for a in ARMS}
        require(all(n in trials for values in names.values() for n in values), 'Incomplete timing schedule')
        arms = {a: [trials[n] for n in values] for a, values in names.items()}
        for a, values in arms.items():
            require(all(r['frames'] == r['count'] == 128 and not r['state_audit'] and r['arm'] == a
                        and r['clip'] == clip for r in values), 'Wrong timing scope')
            for r in values:
                positive(r['wall_s'], 'wall time')
                require(math.isclose(r['fps'], 128/r['wall_s'], rel_tol=1e-12), 'FPS denominator changed')
        fps = {a: 384/sum(r['wall_s'] for r in values) for a, values in arms.items()}
        comparisons = {}
        for a in ARMS[1:]:
            paired = [arms[a][i]['fps']/arms['v20'][i]['fps'] for i in range(3)]
            latency = {}
            for label in LATENCY:
                lists = {m: [samples[n][label] for n in names[m]] for m in ('v20', a)}
                require(all(len(values) == 128 for arm in lists.values() for values in arm), 'Missing latency samples')
                require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                            for arm in lists.values() for values in arm for v in values), 'Invalid latency duration')
                p95 = {m: distribution([v for values in arm for v in values])['p95_ms'] for m, arm in lists.items()}
                ratios = [distribution(lists[a][i])['p95_ms']/distribution(lists['v20'][i])['p95_ms'] for i in range(3)]
                latency[label] = dict(pooled_p95_ms=p95, paired_p95_ratios=ratios,
                    consistent_regression=p95[a] > p95['v20'] and sum(x > 1 for x in ratios) >= 2)
            comparisons[a] = dict(speedup=fps[a]/fps['v20'], paired_speedups=paired, latency=latency)
        combined = comparisons['combined']
        singles = fps['combined'] >= max(fps['v26'], fps['v28'])
        passed = (combined['speedup'] >= 1.2 and all(x > 1 for x in combined['paired_speedups'])
            and not any(x['consistent_regression'] for x in combined['latency'].values()) and singles)
        result[clip] = dict(pooled_fps=fps, vs_v20=comparisons, combined_not_worse_than_singles=singles, passed=passed)
    return dict(passed=all(v['passed'] for v in result.values()), clips=result)


def transformed_hash():
    # Build guarded Python code only; do not load or execute a native binary.
    from tracking_geometry_v20 import GeometryV20
    from tracking_batch_v27 import BatchGeometryV27
    from tracking_stage_v28 import TrackingStageV28
    from tiny_target.tracking.kalman import KalmanTrackManager
    scalar = object.__new__(GeometryV20)
    stage = object.__new__(TrackingStageV28)
    stage.geometry = object.__new__(BatchGeometryV27)
    stage.adapt(scalar.adapter(KalmanTrackManager.update))
    return stage.transformed_sha256


def verify_sources(evidence):
    front, generated = verify_front_sources(V26)
    verify_stage_sources(V24)
    tracking = verify_tracking(V28)
    geometry = verify_dependencies()
    frozen = read(evidence/'freeze.json')
    require(frozen['pre_run'] and set(frozen['files']) == set(SOURCES), 'Missing pre-run source freeze')
    for name, value in frozen['files'].items():
        require(value == sha(evidence/name) == sha(local_source(name)), 'Changed integration source '+name)
    require(frozen['v26_freeze_sha256'] == sha(V26/'freeze.json')
            and frozen['v28_freeze_sha256'] == sha(V28/'freeze_01.json')
            and frozen['v28_manifest_sha256'] == sha(V28/'post_run_01.json')
            and frozen['v28_gate_sha256'] == tracking['run_sha256']
            and frozen['library_sha256'] == sha(V27/'build_01/libtracking_batch_v27.so'), 'Changed verified dependency chain')
    require(frozen['baseline_profiles'] == {c: sha(V27/f'profile_{c}_01.json') for c in WORKLOADS},
            'Changed private-state baseline')
    require(frozen['unit_gate_sha256'] == sha(evidence/'unit_gate.json'), 'Changed harness gate')
    unit = read(evidence/'unit_gate.json')
    require(unit['passed'] and unit['returncode'] == 0 and unit['test_sha256'] == frozen['files']['test_combined_v29.py']
            and unit['log_sha256'] == sha(evidence/'unit_gate.log'), 'Harness unit gate failed')
    return frozen, generated, geometry


def audit_trial(evidence, spec, row, frozen, generated, geometry, tracking_hash):
    path = evidence/spec['name']
    r = read(path.with_suffix('.v29.json'))
    gpu, tracking = spec['arm'] in ('v26', 'combined'), spec['arm'] in ('v28', 'combined')
    require(row['returncode'] == 0 and r['passed'] and r['error'] is None
            and r['schema'] == 'seaqr.visible-combined-v29.v1'
            and all(r[k] == spec[k] for k in ('clip', 'arm', 'frames', 'state_audit')), 'Trial failed/scope changed')
    require(r['source_sha256'] == frozen['files'] and r['freeze_sha256'] == sha(evidence/'freeze.json')
            and row['receipt_sha256'] == sha(path.with_suffix('.v29.json')), 'Trial receipt/source changed')
    require(r['gpu_front'] == gpu and r['tracking_stage'] == tracking and r['execution_policy'] == 'serial_reference'
            and not any(r[k] for k in ('raw16_accessed', 'defaults_changed', 'production_approved',
                                      'new_accuracy_validated', 'staged_v24_enabled', 'native_motion_v25_enabled')),
            'Wrong algorithm/scheduling arm')
    reference = ARCHIVE/f"visible_{spec['clip']}_full_reuse"
    original = read(reference/'launch.json')
    config = read(V26/'candidate_config.json') if gpu else original['configuration']
    config_sha = sha(V26/'candidate_config.json') if gpu else original['config_sha256']
    count = spec['frames'] or read(reference/'report.json')['frames']
    comparison = check(reference, path, count, spec['frames'] is None, 'candidate' if gpu else 'reference',
                       config, config_sha, read(V26/'transition.json'))
    require(comparison == r['comparison'] and r['processed_frames'] == count and r['config_sha256'] == config_sha,
            'Exact journal comparison changed')
    require(r['library_sha256'] == (generated['library_sha256'] if gpu else generated['reference_library_sha256'])
            and r['geometry_library_sha256'] == geometry['library_sha256']
            and r['batch_library_sha256'] == (frozen['library_sha256'] if tracking else None)
            and r['tracking_transformed_sha256'] == (tracking_hash if tracking else geometry['transformed_sha256']),
            'Actual implementation identity changed')
    require(r['geometry_fallbacks'] == 0, 'Unexpected geometry fallback')
    if tracking:
        counts = r['optimized_tracking']
        require(r['geometry_calls'] == 0 and counts['geometry_batches'] > 0 and counts['geometry_tracks'] > 0
                and counts['geometry_fallbacks'] == counts['innovation_fallbacks'] == 0
                and counts['innovation_tracks'] == counts['geometry_tracks']
                and counts['innovation_batches'] == counts['geometry_batches'], 'Incomplete optimized tracking')
    else:
        require(r['geometry_calls'] > 0 and r['optimized_tracking'] is None, 'Original tracking not exercised')
    if gpu:
        require(r['native_mask_calls'] == 0 and len(r['fronts']) == 1, 'Incorrect GPU learning path')
        front = r['fronts'][0]
        require(front['calls'] == front['device_calls'] == front['finish_calls'] == count and front['host_calls'] == 0
                and front['closed'] and front['learning_points'] >= 0, 'Incomplete GPU front lifecycle')
    else:
        require(r['native_mask_calls'] > 0 and not r['fronts'], 'Original front not exercised')
    if spec['state_audit']:
        baseline = read(V27/f"profile_{spec['clip']}_01.json")['replay']
        require(r['private_state_digests'] == baseline['digests'] and len(r['private_state_digests']) == count,
                'Private state or actual learning inputs changed')
        require(r['optimized_tracking']['geometry_tracks'] == baseline['geometry_calls'], 'Smoke track population changed')
    else:
        require(r['private_state_digests'] is None, 'State instrumentation present in clean timing')
    report = read(path/'report.json')
    for key, report_key in (('fps', 'processed_fps'), ('wall_s', 'elapsed_seconds')):
        positive(r[key], key)
        require(r[key] == row[key] == report[report_key], 'Timing receipt mismatch')
    require(math.isclose(r['fps'], count/r['wall_s'], rel_tol=1e-12), 'Wrong FPS denominator')
    snap = r['execution']
    validate_snapshot(snap, count)
    require(snap['policy'] == snap['mode'] == 'reference' and snap['engine'] is None, 'Nonserial scheduling')
    require(len(r['consumer_frame_ms']) == count, 'Missing cadence samples')
    frame_rows = snap['frames']
    def spans(start, end):
        return [(f[end]-f[start])/1e6 for f in frame_rows]
    samples = dict(consumer_cadence=r['consumer_frame_ms'], queue_aware=spans('ready_ns','consumer_complete_ns'),
        request_to_complete=spans('request_ns','consumer_complete_ns'), decode=spans('admitted_ns','ready_ns'),
        admission_wait=spans('request_ns','admitted_ns'), cpu_prepare=spans('cpu_prepare_start_ns','cpu_prepare_end_ns'),
        warp_gpu_host=spans('warp_gpu_start_ns','warp_gpu_end_ns'), detector=spans('detector_start_ns','detector_end_ns'),
        tracking=spans('tracking_start_ns','tracking_end_ns'))
    for label in LATENCY:
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples[label]),
                'Invalid latency sample ' + label)
    positive(r['process_peak_rss_kib'], 'RSS')
    return dict(**spec, count=count, exact=True, fps=r['fps'], wall_s=r['wall_s'],
        metrics={k: distribution(v) for k, v in samples.items()}, process_peak_rss_kib=r['process_peak_rss_kib'],
        tracking_counters=r['optimized_tracking'], fronts=r['fronts'], geometry_calls=r['geometry_calls']), samples


def verify(evidence):
    manifest = read(evidence/'export_manifest.json')
    require(manifest['post_run'] and not manifest['includes_media'], 'Invalid evidence export')
    for name, expected in manifest['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe evidence path')
        require(sha(evidence/path) == expected, 'Exported artifact changed '+name)
    require(sha(evidence/'export_visible_combined_v29.py') == sha(ROOT/'scripts/export_visible_combined_v29.py'),
            'Changed evidence exporter')
    frozen, generated, geometry = verify_sources(evidence)
    tracking_hash = transformed_hash()
    batch = read(evidence/'batch.json')
    prefix, full = schedules()
    specs, rows = batch['schedule'], batch['rows']
    require(specs in (prefix, prefix+full) and len(rows) <= len(specs), 'Unexpected schedule')
    expected_files = set(SOURCES) | {'freeze.json', 'unit_gate.json', 'unit_gate.log', 'batch.json',
        'batch.log', 'status.json', 'export_visible_combined_v29.py'}
    for s in specs:
        expected_files.update(s['name']+ending for ending in ('.v29.json', '.execution.json', '.log',
            '/launch.json', '/report.json', '/frames.jsonl'))
    require(set(manifest['files']) == expected_files, 'Incomplete exported evidence')
    require(batch['source_sha256'] == frozen['files'] and batch['freeze_sha256'] == sha(evidence/'freeze.json')
            and not batch['raw16_accessed'] and not batch['defaults_changed'], 'Batch source/scope changed')
    trials, samples, failed = {}, {}, []
    remote = Path(rows[0]['command'][1]).parent if rows else None
    require(remote is not None and str(remote).startswith('/tmp/seaqr_visible_combined_v29_'), 'Unknown experiment directory')
    for spec, row in zip(specs, rows):
        require(all(row[k] == v for k, v in spec.items()), 'Reordered video schedule')
        command = ['/usr/bin/python3', str(remote/'run_visible_combined_v29.py'), '--clip', spec['clip'],
                   '--arm', spec['arm'], '--output', str(remote/spec['name'])]
        if spec['frames'] is not None: command += ['--frames', str(spec['frames'])]
        if spec['state_audit']: command += ['--state-audit']
        require(row['command'] == command and row['log'] == spec['name']+'.log'
                and row['log_sha256'] == sha(evidence/row['log']), 'Changed trial command/log')
        if row['returncode']:
            failed.append(dict(name=spec['name'], returncode=row['returncode']))
            require(len(trials)+1 == len(rows), 'Continued after failure')
            break
        trials[spec['name']], samples[spec['name']] = audit_trial(evidence, spec, row, frozen, generated, geometry, tracking_hash)
    gate = performance(trials, samples) if all(s['name'] in trials for s in prefix) else None
    require(batch['performance'] == gate, 'Independent performance calculation differs')
    full_complete = all(s['name'] in trials for s in full)
    if len(specs) > len(prefix): require(gate and gate['passed'], 'Full regression before passing prefix gate')
    require(batch['full_regression_run'] == full_complete, 'Wrong full-regression flag')
    completed = not failed and batch['passed'] and batch['error'] is None and len(rows) == len(specs)
    if batch['passed']:
        require(completed and gate is not None and specs == prefix+(full if gate['passed'] else []), 'False completion')
        require(batch['decision'] == ('combined_requires_independent_audit' if gate['passed'] else 'reject_combined_keep_v20'),
                'Wrong decision')
    pooled = {}
    if gate is not None:
        for c in WORKLOADS:
            pooled[c] = {a: {key: distribution([v for i in range(3) for v in samples[f'{c}_repeat{i}_{a}'][key]])
                              for key in samples[f'{c}_repeat0_{a}']} for a in ARMS}
    return dict(verified=True, completed=completed, completed_runs=len(trials), scheduled_runs=len(specs), failed_runs=failed,
        prefix_performance=gate, full_regression_complete=full_complete, paired_stage_metrics=pooled,
        verified_frame_instances=sum(t['count'] for t in trials.values()),
        unique_development_frames=sum(max((t['count'] for t in trials.values() if t['clip']==c), default=0) for c in CLIPS),
        full_regression_frames=sum(t['count'] for t in trials.values() if t['frames'] is None),
        decision=('incomplete_or_failed' if not completed else 'opt_in_combined_candidate' if full_complete
                  else 'reject_combined_keep_v20'), trials=list(trials.values()),
        batch_sha256=sha(evidence/'batch.json'), verifier_sha256=sha(__file__),
        export_manifest_sha256=sha(evidence/'export_manifest.json'),
        production_approved=False, defaults_changed=False, raw16_paused=True, new_airborne_accuracy_validated=False,
        note='FPS includes video decode/processing/journaling, excludes process startup/hash checks. '
             'State-audited smokes excluded; full runs have archived, not fresh paired full timing references. '
             'Latency starts at a file decode request, not sensor exposure. Development parity is not new accuracy.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.evidence)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('trials', 'paired_stage_metrics')}, indent=2))
