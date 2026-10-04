"""Aggregate additive wall timings and exact-output gates without opening media."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import compact, sha, write_json
from profile_raw16_v6 import BASELINE_HASHES, RUNTIME_SHA
from run_raw16_cpu_v6 import CONFIG_SHA
from summarize_raw16_cpu_v6 import compare, compare_source_motion


def read(path):
    return json.loads(Path(path).read_text())


def validate_accounting(timing):
    total = timing['wall_s']
    groups = timing['groups']
    spans = timing['spans']
    if not math.isfinite(total) or total <= 0 or timing['accounting_error_ns'] != 0:
        raise ValueError('Invalid root wall time/accounting')
    if not math.isclose(total, sum(row['exclusive_s'] for row in groups.values()), abs_tol=1e-9, rel_tol=0):
        raise ValueError('Group times do not sum to wall time')
    if not math.isclose(total, timing['exclusive_sum_s'], abs_tol=1e-9, rel_tol=0):
        raise ValueError('Reported sum differs from wall time')
    if len({row['path'] for row in spans}) != len(spans):
        raise ValueError('Duplicate timing paths')
    roots = [row for row in spans if '/' not in row['path']]
    if len(roots) != 1 or roots[0]['calls'] != 1 or roots[0]['inclusive_s'] != total:
        raise ValueError('Incomplete or ambiguous root timing')
    if {row['group'] for row in spans} != set(groups):
        raise ValueError('Group inventory differs from spans')
    for row in spans:
        if not (0 <= row['exclusive_s'] <= row['inclusive_s'] and row['calls'] > 0):
            raise ValueError('Invalid span duration/count')
        children = [child for child in spans if child['path'].rsplit('/', 1)[0] == row['path']
                    and child['path'] != row['path']]
        if not math.isclose(row['inclusive_s'], row['exclusive_s'] + sum(c['inclusive_s'] for c in children),
                            abs_tol=1e-9, rel_tol=0):
            raise ValueError('Nested span durations do not reconcile')
    for name, row in groups.items():
        observed = sum(s['exclusive_s'] for s in spans if s['group'] == name)
        if not math.isclose(row['exclusive_s'], observed, abs_tol=1e-9, rel_tol=0):
            raise ValueError('Group total differs from measured spans')
        if not math.isclose(row['fraction'], observed/total, abs_tol=1e-12, rel_tol=0):
            raise ValueError('Incorrect wall-time fraction')


def one(directory, baseline):
    profile = read(directory/'stage_profile.json')
    clip = profile['clip']
    for name, expected in BASELINE_HASHES[clip].items():
        if sha(baseline/name) != expected:
            raise ValueError('Baseline artifact changed')
    validate_accounting(profile['timing'])
    if profile['error'] is not None or profile['runtime_archive_sha256'] != RUNTIME_SHA:
        raise ValueError('Incomplete run or unfrozen runtime')
    if profile['script_sha256'] != sha(ROOT/'scripts/profile_raw16_v6.py'):
        raise ValueError('Profiling script changed since measurement')
    provenance = read(directory/'provenance.json')
    package = {key:value for key,value in profile['frozen_files'].items()
               if key.startswith('tiny_target/') and key.endswith('.py')}
    if provenance['package_sha256'] != package:
        raise ValueError('Profile provenance differs from frozen runtime')
    if provenance['frozen_sha256']['configs/evaluation/raw16_motion_v6.json'] != CONFIG_SHA:
        raise ValueError('Motion configuration changed')
    if provenance['frozen_sha256']['build/cuda/libtiny_target_cuda.so'] != 'e29dc8bae949e41497aff82cfa2fe07d52e1c039b5c337187bdb8c2e87b1fc65':
        raise ValueError('CUDA library changed')
    parity = compare(baseline, directory)
    parity.update(compare_source_motion(baseline, directory))
    checks = read(directory/'checks.json')
    if not (read(directory/'profile_parity.json')['passed'] is True
            and parity['exact_semantics'] and parity['source_frames_exact'] and parity['motion_points_exact']
            and checks['checks']['processing_integrity_passed']
            and checks['checks']['detection_availability_passed']):
        raise ValueError('Profile changed outputs or failed processing')
    report = read(directory/'report.json')
    motions = read(directory/'motion_profile.json')
    if [row['frame_index'] for row in motions] != list(range(1,64)):
        raise ValueError('Incomplete motion sequence')
    screen = report['screening']
    windows = screen['synthetic_tracking']['windows']
    if screen['frames_seen'] != 64 or len(windows) != 6:
        raise ValueError('Unexpected workload size')
    cuda = {name:sum(w['synthetic_tracking_timings_ms'][name] for w in windows)/1000
            for name in ('total','native_call_host','cuda_h2d','cuda_displacement',
                         'cuda_kernel','cuda_d2h','cuda_gpu_total')}
    cuda['outer_integration_host'] = profile['timing']['groups']['synthetic_integration']['exclusive_s']
    cuda['outer_minus_legacy_total'] = cuda['outer_integration_host'] - cuda['total']
    prior_elapsed = read(baseline/'checks.json')['elapsed_wall_s']
    return dict(clip=clip, cprofile_enabled=profile['cprofile_enabled'], parity=parity,
        checks_elapsed_wall_s=checks['elapsed_wall_s'],
        prior_v6_checks_elapsed_wall_s=prior_elapsed,
        elapsed_ratio_to_prior_v6=checks['elapsed_wall_s']/prior_elapsed,
        frames_per_instrumented_wall_second=64/checks['elapsed_wall_s'],
        timing=profile['timing'], cuda_nested_totals_s=cuda,
        accepted_motion_pairs=report['source']['pva_stabilization']['metrics']['accepted_transforms_applied'],
        filter_supported_frames=screen['availability']['frames_with_valid_filter_support'],
        search_windows=len(windows), retained_tracks=len(screen['synthetic_tracking']['track_pool']),
        files={name:sha(directory/name) for name in ('stage_profile.json','report.json',
               'source_frames.json','motion_profile.json','profile_parity.json','checks.json')})


def summarize(current, baseline):
    trials = {clip:one(current/f'{clip}_stages', baseline/f'full_frame_{clip}_v6')
              for clip in ('0040','0029')}
    if any(row['cprofile_enabled'] for row in trials.values()):
        raise ValueError('Main stage profiles must exclude cProfile')
    diagnostic = None
    if (current/'0040_cprofile').is_dir():
        diagnostic = one(current/'0040_cprofile', baseline/'full_frame_0040_v6')
        if not diagnostic['cprofile_enabled']:
            raise ValueError('Diagnostic must identify cProfile overhead')
    generated = []
    for seed in (75316, 129827, 85723):
        new = read(current/'generated_evidence'/f'status_controls_seed{seed}.json')
        old = read(baseline/f'status_controls_seed{seed}.json')
        exact = compact(new['cases']) == compact(old['cases'])
        if not (new['passed'] is True and len(new['cases']) == 16 and exact
                and new['config_sha256'] == old['config_sha256'] == CONFIG_SHA):
            raise ValueError('Regenerated controls changed or failed')
        generated.append(dict(seed=seed, cases=16, passed=True, exact_saved_v6_cases=exact))
    group_names = set().union(*(row['timing']['groups'] for row in trials.values()))
    combined_total = sum(row['timing']['wall_s'] for row in trials.values())
    groups = {}
    for name in group_names:
        seconds = sum(row['timing']['groups'].get(name, {}).get('exclusive_s',0) for row in trials.values())
        fraction = seconds/combined_total
        groups[name] = dict(total_s=seconds, fraction=fraction,
            ideal_whole_pipeline_speedup_if_eliminated=1/(1-fraction),
            hypothetical_whole_pipeline_speedup_if_stage_2x=1/(1-fraction/2),
            hypothetical_whole_pipeline_speedup_if_stage_4x=1/(1-3*fraction/4))
    return dict(schema_version='seaqr.raw16-frozen-v6-profile-summary.v1',
        passed=True, frozen_runtime_sha256=RUNTIME_SHA,
        trials=trials, diagnostic_cprofile=diagnostic,
        regenerated_controls=generated,
        combined_exclusive_groups=dict(sorted(groups.items(), key=lambda item:-item[1]['total_s'])),
        real_airborne_accuracy_validated=False, runtime_or_defaults_changed=False,
        warning='Only two 64-frame development prefixes. Exact non-timing outputs and source/motion '
                'hashes match frozen v6. Host wall intervals, not utilization. CUDA timers are nested '
                'and non-additive with host spans. Relative time to earlier runs is not an isolated '
                'profiling-overhead estimate. Amdahl projections are arithmetic scenarios, not measured '
                'optimization speedups or real-time promises.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.current, args.baseline)
    write_json(args.output, result)
    print(json.dumps(dict(passed=result['passed'], groups=result['combined_exclusive_groups']), indent=2))
