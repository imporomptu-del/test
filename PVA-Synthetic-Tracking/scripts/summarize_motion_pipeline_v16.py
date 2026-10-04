"""Independent local full-pipeline identity and exclusive timing verification."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_motion_pipeline_v16 import read, sha, write
from batch_motion_pipeline_v16 import schedule
from motion_front_pixels_v15 import logical_identity
from summarize_motion_front_v15 import validate_execution_fields
from run_raw16_background_v7 import normalized_report
from summarize_raw16_cpu_v6 import compare_source_motion

ARCHIVE = ROOT/'results/tiny_target/motion_video_v13_20260916/evidence/results'
EXACT = ROOT/'results/tiny_target/raw16_exact_v9_20260916/evidence/results/final'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def compare(reference, candidate):
    reports = [logical_identity(normalized_report(read(p/'report.json'))) for p in (reference, candidate)]
    require(reports[0] == reports[1], 'Report/track identity mismatch')
    require(all(compare_source_motion(reference, candidate).values()), 'Source/point identity mismatch')
    for name in ('candidate_decisions.json', 'global_fit_identities.json'):
        require(read(reference/name) == read(candidate/name), 'Changed '+name)


def summarize(evidence):
    batch = read(evidence/'batch.json')
    require(batch['passed'] and batch['error'] is None, 'Batch incomplete')
    require(batch['schedule'] == schedule(), 'Schedule changed')
    require(batch['script_sha256'] == sha(ROOT/'scripts/batch_motion_pipeline_v16.py'), 'Batch source changed')
    require(len(batch['rows']) == 11, 'Missing trials')
    front_gate = ROOT/'results/tiny_target/motion_front_v15_20260916/evidence/generated_03.json'
    records, profiles = {}, {}
    for spec, row in zip(schedule(), batch['rows']):
        require(all(row[k] == v for k, v in spec.items()), 'Trial order/scope mismatch')
        directory = evidence/spec['name']
        path = directory.with_suffix('.execution.json')
        require(sha(path) == row['sha256'], 'Trial receipt changed')
        r = read(path)
        require(all(r[k] == spec[k] for k in ('clip', 'mode', 'injected', 'profile')), 'Execution scope differs from schedule')
        require(r['wrapper_sha256'] == sha(ROOT/'scripts/run_motion_pipeline_v16.py')
                and r['plan_sha256'] == sha(ROOT/'docs/motion_pipeline_v16_plan.md')
                and r['gate_sha256'] == sha(front_gate), 'Provenance changed')
        require(r['passed'] and r['closed'] and r['error'] is None
                and r['reuse_hits'] == 62 and r['reuse_misses'] == 1, 'Lifecycle failure')
        require([p['frame'] for p in r['motion']] == list(range(1, 64)), 'Incomplete motion record')
        validate_execution_fields(r['motion'], spec['mode'])
        reference = EXACT/'injected_0040_exact' if spec['injected'] else ARCHIVE/f'raw_{spec["clip"]}_repeat0_reuse'
        compare(reference, directory)
        old_motion = read((ARCHIVE/f'raw_{spec["clip"]}_repeat0_reuse').with_suffix('.execution.json'))['motion']
        require([logical_identity(p['identity']) for p in r['motion']] ==
                [logical_identity(p['identity']) for p in old_motion], 'Full motion identity changed')
        require(read(directory/'comparison.json')['exact_gate_passed'], 'Original pipeline gate failed')
        checks = read(directory/'checks.json')
        require(checks['checks'] == r['checks'] and checks['elapsed_wall_s'] == r['wall_s'] == row['wall_s']
                and 64/r['wall_s'] == r['fps'] == row['fps'], 'Timing/check receipt mismatch')
        require(r['checks']['processing_integrity_passed'] and r['checks']['detection_availability_passed'], 'Integrity gate failed')
        records[spec['name']] = r
        if spec['profile']:
            profile = read(directory/'stage_profile.json')
            t = profile['timing']
            require(profile['error'] is None and t['accounting_error_ns'] == 0
                    and abs(sum(g['exclusive_s'] for g in t['groups'].values())-t['wall_s']) < 1e-7,
                    'Invalid exclusive timing accounting')
            profiles[spec['mode']] = dict(wall_s=t['wall_s'],
                ms_per_input_frame={k: g['exclusive_s']*1000/64 for k, g in t['groups'].items()},
                fraction={k: g['fraction'] for k, g in t['groups'].items()})
    performance = {}
    for clip in ('0029', '0040', 'combined'):
        arms = {}
        for mode in ('reference', 'candidate'):
            selected = [r for r in records.values() if not r['profile'] and not r['injected']
                        and r['mode'] == mode and (clip == 'combined' or r['clip'] == clip)]
            wall = sum(r['wall_s'] for r in selected)
            frames = 64*len(selected)
            arms[mode] = dict(trials=len(selected), frames=frames, wall_s=wall, fps=frames/wall,
                ms_per_input_frame=wall*1000/frames,
                motion_ms_per_input_frame=1000*sum(p['estimator_s'] for r in selected for p in r['motion'])/frames)
        performance[clip] = dict(arms=arms, speedup=arms['reference']['wall_s']/arms['candidate']['wall_s'],
            wall_time_reduction_percent=100*(1-arms['candidate']['wall_s']/arms['reference']['wall_s']))
    return dict(schema='seaqr.motion-pipeline-v16-summary.v1', verified=True,
        full_pipeline_trials=11, input_frames_across_runs=704, unique_real_frames=128,
        exact_source_point_fit_candidate_track_parity=True,
        performance=performance, separate_profiles=profiles,
        injected_checks=records['raw_0040_injected_candidate']['checks'],
        injected_evaluation=read(evidence/'raw_0040_injected_candidate/report.json')['injection']['synthetic_track_pool_evaluation'],
        timing_trials=[dict(name=row['name'], wall_s=row['wall_s'], fps=row['fps']) for row in batch['rows']],
        batch_sha256=sha(evidence/'batch.json'), summarizer_sha256=sha(__file__),
        defaults_changed=False, production_approved=False, airborne_accuracy_validated=False,
        warning='Bounded offline regression with source hashing/logging/report writing; not live camera throughput. '
                'Separate instrumented profiles excluded from throughput means. Exactness does not repair existing misses.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evidence)
    write(args.output, result)
    print(json.dumps(result, indent=2))
