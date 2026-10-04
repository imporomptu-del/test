"""Report-only audit, exactness, and alternating-pair timing summary for v7."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import compact, sha, write_json
from run_raw16_background_v7 import (BASELINE_HASHES, CONFIG_SHA, INJECTED_HASHES,
    LIBRARY_SHA, compare, compare_audits, read)
from summarize_raw16_motion_v3 import trial
from batch_raw16_background_v7 import schedule
from summarize_raw16_profile_v6 import validate_accounting


def exact_trial(folder, baseline, component):
    comparison = compare(baseline, folder)
    experiment = read(folder/'background_experiment.json')
    provenance = read(folder/'provenance.json')
    checks = read(folder/'checks.json')
    if not (read(folder/'background_parity.json')['passed'] is True
            and comparison['exact_semantics'] and comparison['source_frames_exact']
            and comparison['motion_points_exact'] and experiment['error'] is None
            and checks['checks']['processing_integrity_passed'] is True
            and checks['checks']['detection_availability_passed'] is True):
        raise ValueError(f'Incomplete or non-exact trial: {folder.name}')
    if experiment['wrapper_sha256'] != sha(ROOT/'scripts/run_raw16_background_v7.py'):
        raise ValueError('Experiment harness changed after measurement')
    runtime = experiment['runtime']
    if not (runtime['cpu_oracle_ast_exact'] is True
            and runtime['library_sha256'] == component['library_sha256'] == LIBRARY_SHA
            and runtime['package_sha256'] == provenance['package_sha256'] == component['package_sha256']):
        raise ValueError('Trial and generated controls used different runtime')
    report = read(folder/'report.json')
    observed = trial(folder.parent, folder.name)
    if not (observed['frames'] == 64 and observed['applied_transforms'] == 63
            and observed['reference_resets'] == observed['runtime_failures'] == 0
            and observed['ready_filter_frames'] == 60 and observed['windows'] == 6):
        raise ValueError('Workload/availability changed')
    expected_mode = 'cuda_temporal_exact_v1' if experiment['gpu'] else 'masked_ufunc'
    if report['configuration']['effective']['background_execution'] != expected_mode:
        raise ValueError('Trial execution arm does not match report')
    elapsed = checks['elapsed_wall_s']
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Invalid elapsed time')
    return dict(name=folder.name, gpu=experiment['gpu'], audit=experiment['audit'], profile=experiment['profile'],
        elapsed_wall_s=elapsed, frames_per_wall_second=64/elapsed, exact=comparison,
        observation=observed, files={n:sha(folder/n) for n in ('report.json','source_frames.json',
            'motion_profile.json','background_experiment.json','background_parity.json','checks.json')})


def summarize(current, baseline, pairs):
    if pairs != 2:
        raise ValueError('Exactly two predeclared alternating pairs per clip are required')
    plan = read(current/'timing_plan.json')
    journal = [json.loads(line) for line in (current/'timing_journal.jsonl').read_text().splitlines()]
    expected = schedule()
    if not (plan['schedule'] == expected and len(journal) == len(expected)
            and plan['script_sha256'] == sha(ROOT/'scripts/batch_raw16_background_v7.py')
            and plan['wrapper_sha256'] == sha(ROOT/'scripts/run_raw16_background_v7.py')):
        raise ValueError('Missing or changed predeclared timing schedule')
    for planned, observed in zip(expected, journal):
        if any(observed[k] != v for k,v in planned.items()) or observed['returncode'] != 0:
            raise ValueError('Batch order changed or a trial failed')
    component = read(current/'component_final.json')
    if not (component['passed'] is True and component['frame_comparisons'] == 1128
            and component['native_enabled'] is True and len(component['cases']) == 28
            and all(r['exact'] is True for r in component['cases'])):
        raise ValueError('Generated component gate failed')
    if component['script_sha256'] != sha(ROOT/'scripts/check_raw16_background_cuda.py'):
        raise ValueError('Component harness changed')
    for name, expected in component['package_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Current runtime differs from measured runtime')
    if sha(ROOT/'configs/evaluation/raw16_background_v7.json') != CONFIG_SHA:
        raise ValueError('GPU experiment configuration changed')
    if component['cuda_source_sha256'] != sha(ROOT/'tiny_target/detection/cuda/raw_background.cu'):
        raise ValueError('CUDA source differs from tested source')
    audits, trials, speeds = {}, {}, {}
    for clip in ('0040', '0029'):
        previous = baseline/f'full_frame_{clip}_v6'
        for name, expected in BASELINE_HASHES[clip].items():
            if sha(previous/name) != expected:
                raise ValueError('Frozen baseline changed')
        for arm in ('cpu', 'gpu'):
            name = f'audit_{clip}_{arm}'
            result = exact_trial(current/name, previous, component)
            if result['audit'] is not True:
                raise ValueError('Expected an array audit')
            trials[name] = result
        audits[clip] = compare_audits(current/f'audit_{clip}_cpu.audit.jsonl', current/f'audit_{clip}_gpu.audit.jsonl')
        if not audits[clip]['passed']:
            raise ValueError('Intermediate array audit is not complete and exact')
        measured = []
        for index in range(1, pairs + 1):
            pair = {}
            for arm in ('cpu', 'gpu'):
                name = f'timed_{clip}_{index}_{arm}'
                result = exact_trial(current/name, previous, component)
                if result['audit'] is not False or result['profile'] is not False or result['gpu'] != (arm == 'gpu'):
                    raise ValueError('Timing contains audit overhead or mislabeled arm')
                trials[name] = result
                pair[arm] = result['elapsed_wall_s']
            pair.update(pair=index, execution_order=['cpu','gpu'] if index % 2 else ['gpu','cpu'],
                        speedup=pair['cpu']/pair['gpu'], wall_time_reduction_fraction=1-pair['gpu']/pair['cpu'])
            measured.append(pair)
        cpu = statistics.median(r['cpu'] for r in measured)
        gpu = statistics.median(r['gpu'] for r in measured)
        speeds[clip] = dict(pairs=measured, cpu_median_s=cpu, gpu_median_s=gpu,
            cpu_median_rate_fps=64/cpu, gpu_median_rate_fps=64/gpu,
            ratio_of_medians=cpu/gpu, median_paired_speedup=statistics.median(r['speedup'] for r in measured),
            minimum_paired_speedup=min(r['speedup'] for r in measured),
            maximum_paired_speedup=max(r['speedup'] for r in measured),
            median_wall_time_reduction_fraction=1-gpu/cpu)
    previous = baseline/'full_frame_0040_injected_v6'
    for name, expected in INJECTED_HASHES.items():
        if sha(previous/name) != expected:
            raise ValueError('Frozen injected reference changed')
    injected = exact_trial(current/'injected_0040_gpu', previous, component)
    if not injected['gpu'] or injected['audit']:
        raise ValueError('Unexpected injected experiment mode')
    generated = []
    for seed in (75316,129827,85723):
        fresh = read(current/'evidence'/f'status_controls_seed{seed}.json')
        old = read(baseline/f'status_controls_seed{seed}.json')
        exact = compact(fresh['cases']) == compact(old['cases'])
        if not (fresh['passed'] is True and len(fresh['cases']) == 16 and exact):
            raise ValueError('Frozen generated motion controls changed')
        generated.append(dict(seed=seed, cases=16, exact_saved_v6=True))
    diagnostic = exact_trial(current/'profile_0040_gpu', baseline/'full_frame_0040_v6', component)
    if not diagnostic['profile'] or diagnostic['audit'] or not diagnostic['gpu']:
        raise ValueError('Expected separate GPU stage profile')
    stage = read(current/'profile_0040_gpu/stage_profile.json')
    if stage['error'] is not None:
        raise ValueError('Incomplete stage profile')
    validate_accounting(stage['timing'])
    diagnostic['timing'] = stage['timing']
    # Setup/preflight and array audits are excluded from timing comparisons;
    # legacy timer still includes source hashing, reports and decoder cleanup.
    return dict(schema_version='seaqr.raw16-gpu-background-summary.v7', passed=True,
        audits=audits, trials=trials, measured_speed=speeds, injected=injected, diagnostic_profile=diagnostic,
        component=dict(passed=True, cases=28, frame_comparisons=1128,
            file_sha256=sha(current/'component_final.json'), library_sha256=LIBRARY_SHA,
            direct_gpu_point_filter_probe=component['direct_gpu_point_filter_probe']),
        regenerated_motion_controls=generated, timing_plan_sha256=sha(current/'timing_plan.json'),
        timing_journal_sha256=sha(current/'timing_journal.jsonl'), default_configuration_changed=False,
        real_airborne_accuracy_validated=False, production_ready=False,
        warning='Two previously used 64-frame RAW16 development prefixes only. '
                'Exact arithmetic/tracking parity is not a new real-airborne recall/FAR evaluation. '
                'Timed runs include source hashing, progress, report writes, cold setup and FFmpeg cleanup; '
                'not sustained streaming throughput. GPU handles temporal state/support only; '
                'the CPU OpenCV point filter is unchanged. Existing upper injected-control miss remains.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pairs', type=int, choices=(2,), default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.current, args.baseline, args.pairs)
    write_json(args.output, result)
    print(result['measured_speed'])
