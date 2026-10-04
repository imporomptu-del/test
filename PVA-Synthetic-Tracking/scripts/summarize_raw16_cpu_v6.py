"""Report-only exactness and bounded performance comparison; never reads media."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_raw16_efficiency import compact, sha, write_json
from summarize_raw16_motion_v3 import read, trial
from run_raw16_cpu_v6 import CONFIG_SHA
from validate_raw16_motion_v5 import CANDIDATE_SHA256 as V5_SHA


def semantic(report):
    """Discard only known runtime timings and verified execution-path metadata."""
    source = deepcopy(report['source'])
    motion_identity = source['pva_stabilization'].pop('configuration')
    if motion_identity['sha256'] not in (V5_SHA, CONFIG_SHA):
        raise ValueError('Unknown motion configuration')
    screen = deepcopy(report['screening'])
    for window in screen['synthetic_tracking']['windows']:
        cuda = window['synthetic_tracking_metrics']['cuda']
        if not cuda['library_path'].endswith('/build/cuda/libtiny_target_cuda.so'):
            raise ValueError('Unexpected CUDA library identity')
        cuda['library_path'] = '<verified_runtime>/build/cuda/libtiny_target_cuda.so'
    injection = deepcopy(report['injection'])
    if injection is not None:
        if not injection['identity']['path'].endswith('/configs/evaluation/raw16_full_frame_controls_v2.json'):
            raise ValueError('Unknown injected control layout')
        injection['identity']['path'] = '<verified_runtime>/configs/evaluation/raw16_full_frame_controls_v2.json'
    return compact(dict(source=source, screening=screen, injection=injection,
                        detector=report['configuration']['effective']))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def difference(a, b, path=''):
    if type(a) is not type(b): return path + ':type'
    if isinstance(a, dict):
        if a.keys() != b.keys(): return path + ':keys'
        for key in a:
            found = difference(a[key], b[key], path + '/' + str(key))
            if found is not None: return found
    elif isinstance(a, list):
        if len(a) != len(b): return path + ':length'
        for index, (left,right) in enumerate(zip(a,b)):
            found = difference(left,right,path+'/'+str(index))
            if found is not None: return found
    elif a != b: return path
    return None


def compare(left, right):
    a, b = semantic(read(left/'report.json')), semantic(read(right/'report.json'))
    return dict(exact_semantics=a == b, first_difference=difference(a,b),
        left_semantic_sha256=digest(a), right_semantic_sha256=digest(b),
        source_frames_exact=read(left/'source_frames.json') == read(right/'source_frames.json'))


def timing(directory):
    profile = read(directory/'motion_profile.json')
    if len(profile) != 63 or [r['frame_index'] for r in profile] != list(range(1,64)):
        raise ValueError('Incomplete per-pair timing evidence')
    elapsed = read(directory/'checks.json')['elapsed_wall_s']
    keys = set.intersection(*(set(row['timings_ms']) for row in profile))
    medians = {key: statistics.median(r['timings_ms'][key] for r in profile) for key in sorted(keys)}
    totals = {key: sum(r['timings_ms'][key] for r in profile)/1000 for key in sorted(keys)}
    measured_motion = sum(r['wall_ms'] for r in profile)/1000
    return dict(instrumented_wall_s=elapsed, frames_per_wall_second=64/elapsed,
        median_motion_wall_ms=statistics.median(r['wall_ms'] for r in profile),
        stage_median_ms=medians, stage_total_s=totals, motion_wall_total_s=measured_motion,
        outside_estimator_wall_s=elapsed-measured_motion,
        warning='Stage medians are not additive; per-stage submit/sync/total fields overlap. '
                'Outside-estimator time includes decode, motion fitting, stabilization, detector and instrumentation.')


def summarize(current, previous):
    names=('full_frame_0029','full_frame_0040','full_frame_0040_injected')
    comparisons={name: compare(previous/(name+'_v5'), current/(name+'_v6')) for name in names}
    fresh={name: compare(current/(name+'_reference'),current/(name+'_v6')) for name in names[:2]}
    profiles={}
    speed={}
    for name in names[:2]:
        reference, candidate=current/(name+'_reference'),current/(name+'_v6')
        a,b=read(reference/'motion_profile.json'),read(candidate/'motion_profile.json')
        fresh[name]['motion_points_exact']=[r['identity'] for r in a] == [r['identity'] for r in b]
        left,right=timing(reference),timing(candidate)
        profiles[name]=dict(reference=left,candidate=right)
        speed[name]=dict(speedup=left['instrumented_wall_s']/right['instrumented_wall_s'],
            wall_time_reduction_fraction=1-right['instrumented_wall_s']/left['instrumented_wall_s'])
    injected=current/'full_frame_0040_injected_v6'
    pair=compare_source_motion(current/'full_frame_0040_v6',injected)
    sequence=read(current/'sequence_0040_v6.parity.json')
    cpu=read(current/'cpu_parity.json')
    generated=[]
    for seed in (75316,129827,85723):
        path=current/f'status_controls_seed{seed}.json'
        r=read(path)
        old=read(previous/f'status_controls_seed{seed}.json')
        generated.append(dict(seed=seed,passed=r['passed'],cases=len(r['cases']),sha256=sha(path),
            exact_v5_semantics=compact(r['cases']) == compact(old['cases'])))
    trials={name:trial(current,name+'_v6') for name in names}
    gates=dict(
        archived_v5_full_output_parity=all(r['exact_semantics'] and r['source_frames_exact'] for r in comparisons.values()),
        fresh_reference_full_output_parity=all(r['exact_semantics'] and r['source_frames_exact'] and r['motion_points_exact'] for r in fresh.values()),
        generated_controls=all(r['passed'] and r['cases']==16 and r['exact_v5_semantics'] for r in generated),
        frozen_cpu_parity=cpu['passed'],
        sequence_exact=sequence['passed'] and sequence['exact_v5_points_and_source'],
        injected_motion_unchanged=all(pair.values()),
        processing_integrity=all(t['checks']['processing_integrity_passed'] for t in trials.values()),
        search_available=all(t['checks']['detection_availability_passed'] for t in trials.values()))
    exact_pass=all(gates.values())
    gates.update(all_injected_controls_recovered=trials['full_frame_0040_injected']['checks']['synthetic_controls_passed'],
        real_airborne_accuracy_validated=False, production_ready=False, default_configuration_changed=False)
    return dict(schema_version='seaqr.raw16-cpu-summary.v6',exact_development_parity_passed=exact_pass,
        gates=gates, archived_comparisons=comparisons, fresh_comparisons=fresh,
        trials=trials, timings=profiles, measured_speed=speed, generated=generated,
        cpu_microbenchmark=cpu, sequence_parity=sequence,
        exclusions=['Known timing/performance fields from compact()',
            'Motion config identity (validated frozen v5/v6 SHA; execution policy only)',
            'Isolated workspace prefix for CUDA library and injected config'],
        warning='Single matched 64-frame pairs, not a throughput guarantee or new real-airborne accuracy evidence. '
                'The cache-isolation workaround and detector policy remain unchanged.')


def compare_source_motion(left,right):
    a,b=read(left/'motion_profile.json'),read(right/'motion_profile.json')
    return dict(source_frames_exact=read(left/'source_frames.json')==read(right/'source_frames.json'),
                motion_points_exact=[r['identity'] for r in a]==[r['identity'] for r in b])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.current,args.previous)
    write_json(args.output,result)
    print(json.dumps(dict(gates=result['gates'],speed=result['measured_speed']),indent=2))
