"""Independent, report-only v28 schedule/provenance/state/timing audit."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from profile_visible_v17 import read, sha, write
from freeze_tracking_v28 import SOURCES
from finalize_tracking_v28 import EXTRA
from verify_tracking_v27 import verify as verify_v27
from verify_visible_overlap_v23 import ROOT, local_source
from verify_visible_v17 import require, distribution

V27 = ROOT / 'results/tiny_target/tracking_v27_20260920/evidence'
V26 = ROOT / 'results/tiny_target/visible_front_v26_20260920/evidence'
REMOTE_V27 = Path('/tmp/seaqr_tracking_v27_pmUXGZ')
REMOTE_V26 = Path('/tmp/seaqr_visible_front_v26_retry_24sEU7')
REMOTE_RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
REMOTE_GEOMETRY = Path('/tmp/seaqr_visible_speed_v20_XUf1LR/build/libtracking_geometry_v20.so')


def summarize(rows):
    expected = [('chunk_' + c, i, m) for c in ('0126', '0082') for i in range(2)
                for m in (('v20', 'v27', 'v28') if i == 0 else ('v28', 'v27', 'v20'))]
    require([(r['clip'], r['repeat'], r['mode']) for r in rows] == expected,
            'Incomplete or reordered replay schedule')
    for r in rows:
        require(r['frames'] == 128 and not r['profiled'] and len(r['tracking_ms']) == 128,
                'Wrong replay timing scope')
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in r['tracking_ms']),
                'Invalid duration')
    result = {}
    for clip in ('0126', '0082'):
        arms = {m: [r for r in rows if r['clip'] == 'chunk_' + clip and r['mode'] == m]
                for m in ('v20', 'v27', 'v28')}
        stats = {m: distribution([v for r in arm for v in r['tracking_ms']]) for m, arm in arms.items()}
        comparisons = {}
        for reference in ('v20', 'v27'):
            a, b = stats[reference]['mean_ms'], stats['v28']['mean_ms']
            comparisons[reference] = dict(saved_ms=a-b, time_reduction_percent=100*(a-b)/a,
                throughput_ratio=a/b, paired_ratios=[sum(arms[reference][i]['tracking_ms']) /
                    sum(arms['v28'][i]['tracking_ms']) for i in range(2)])
        result[clip] = dict(tracking_ms=stats, v28_vs=comparisons)
    return result


def verify(evidence):
    previous = verify_v27(V27)
    manifest = read(evidence / 'post_run_01.json')
    require(manifest['post_run'] and not manifest['media_read'] and not manifest['defaults_changed'],
            'Invalid post-run manifest')
    required = set(SOURCES + EXTRA + ('freeze_01.json', 'replays_01.json', 'unit_01.log',
        'run_01.log', 'supplemental_unit_01.json', 'supplemental_unit_01.log'))
    require(set(manifest['files']) == required, 'Incomplete artifact manifest')
    for name, expected_hash in manifest['files'].items():
        require(sha(evidence / name) == expected_hash, 'Changed artifact ' + name)
    supplemental = read(evidence / 'supplemental_unit_01.json')
    require(supplemental['passed'] and supplemental['returncode'] == 0
            and not supplemental['media_read'] and supplemental['post_timing']
            and supplemental['log_sha256'] == sha(evidence / 'supplemental_unit_01.log'),
            'Supplemental error-state unit gate failed')
    require(set(supplemental['source_sha256']) == set(SOURCES + EXTRA), 'Incomplete supplemental source set')
    for name, value in supplemental['source_sha256'].items():
        require(value == sha(evidence / name) == sha(local_source(name)), 'Supplemental source changed ' + name)
    supplemental_log = (evidence / 'supplemental_unit_01.log').read_text()
    require('Ran 3 tests' in supplemental_log and supplemental_log.rstrip().endswith('OK'),
            'Missing supplemental tests')
    freeze = read(evidence / 'freeze_01.json')
    run = read(evidence / 'replays_01.json')
    require(freeze['pre_run'] and not freeze['media_read'] and not freeze['defaults_changed']
            and freeze['frames_per_prefix'] == 128 and freeze['clips'] == ['0126', '0082']
            and freeze['schedule'] == ['v20', 'v27', 'v28', 'v28', 'v27', 'v20'], 'Wrong frozen scope')
    require(run['passed'] and run['error'] is None and not run['media_read']
            and not run['defaults_changed'] and not run['full_pipeline_tested'], 'Run incomplete/failed')
    require(run['freeze_sha256'] == sha(evidence / 'freeze_01.json')
            and freeze['created_ns'] <= run['started_ns'] < run['finished_ns'], 'Invalid pre-run freeze')
    roots = [Path(p).parent for p in freeze['files'] if Path(p).name == 'tracking_stage_v28.py']
    require(len(roots) == 1, 'Unknown experiment source root')
    remote = roots[0]
    expected = {}
    for name in SOURCES:
        expected[str(remote / name)] = sha(evidence / name)
        require(expected[str(remote / name)] == sha(local_source(name)), 'Local/source mismatch ' + name)
    expected[str(REMOTE_V27 / 'build_01/libtracking_batch_v27.so')] = previous['library_sha256']
    expected[str(REMOTE_GEOMETRY)] = read(V27 / 'profile_0126_01.json')['replay']['geometry_library_sha256']
    baselines = {}
    for clip in ('0126', '0082'):
        parent = V26 / (clip + '_repeat0_reference')
        for name in ('frames.jsonl', 'launch.json', 'report.json'):
            expected[str(REMOTE_V26 / parent.name / name)] = sha(parent / name)
        profile = 'profile_' + clip + '_01.json'
        expected[str(REMOTE_V27 / profile)] = sha(V27 / profile)
        baselines['chunk_' + clip] = read(V27 / profile)['replay']
        for name, value in read(parent / 'launch.json')['package_sha256'].items():
            expected[str(REMOTE_RUNTIME / 'tiny_target' / name)] = value
            require(sha(ROOT / 'tiny_target' / name) == value, 'Runtime source changed ' + name)
    require(freeze['files'] == expected, 'Incomplete or changed dependency freeze')
    generated = run['generated']
    require([(r['scenario'], r['frames'], r['exact']) for r in generated['scenarios']]
            == [(i, 40, True) for i in range(36)], 'Incomplete generated scenarios')
    require(generated['innovation_batches'] > 0 and generated['innovation_tracks'] > 0
            and generated['innovation_fallbacks'] == 0, 'Generated batching not exercised')
    timing = summarize(run['replays'])
    require(len(run['profiles']) == 2 and [r['clip'] for r in run['profiles']] == list(baselines),
            'Missing candidate profiles')
    for r in run['replays'] + run['profiles']:
        baseline = baselines[r['clip']]
        require(r['exact'] and r['frames'] == 128 and r['geometry_fallbacks'] == 0, 'Inexact replay')
        for name in ('digests', 'populations', 'journal_sha256', 'launch_sha256', 'report_sha256', 'geometry_library_sha256'):
            require(r[name] == baseline[name], 'Replay changed ' + name)
        if r.get('mode') == 'v20':
            require(r['geometry_calls'] == baseline['geometry_calls'] and r['counters'] == {}, 'Wrong reference path')
        else:
            counters = r['counters']
            count = sum(v['tracks'] > 0 for frame in baseline['populations'] for v in frame.values())
            require(r['geometry_calls'] == 0 and counters['geometry_tracks'] == baseline['geometry_calls']
                    and counters['geometry_batches'] == count and counters['geometry_fallbacks'] == 0,
                    'Batch geometry not fully exercised')
            if r.get('mode', 'v28') == 'v28':
                require(counters['innovation_tracks'] == baseline['geometry_calls']
                        and counters['innovation_batches'] == count and counters['innovation_fallbacks'] == 0,
                        'Innovation batch not fully exercised')
    require(all(r['profiled'] and r['profile'] for r in run['profiles']), 'Invalid candidate profile')
    unit = (evidence / 'unit_01.log').read_text()
    require('Ran 9 tests' in unit and unit.rstrip().endswith('OK'), 'Jetson unit gate failed')
    return dict(verified=True, completed=True, timing=timing, replays=12,
        replay_frame_instances=1536, unique_development_frames=256, generated_scenarios=36,
        generated_frames_per_scenario=40, private_state_and_learning_exact=True,
        media_decoded=False, full_pipeline_tested=False, defaults_changed=False,
        new_accuracy_validated=False, raw16_paused=True, production_approved=False,
        profiles={r['clip']: r['profile'] for r in run['profiles']},
        freeze_sha256=sha(evidence / 'freeze_01.json'), run_sha256=sha(evidence / 'replays_01.json'),
        unit_log_sha256=sha(evidence / 'unit_01.log'), verifier_sha256=sha(__file__),
        post_run_manifest_sha256=sha(evidence / 'post_run_01.json'),
        supplemental_unit_sha256=sha(evidence / 'supplemental_unit_01.json'),
        previous_v27_gate_sha256=previous['gate_sha256'], note='Tracking-only timings, not new pipeline FPS.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.evidence)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'profiles'}, indent=2))
