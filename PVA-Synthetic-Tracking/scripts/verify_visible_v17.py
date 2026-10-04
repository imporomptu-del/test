"""Report-only independent verification; never opens video or a RAW manifest."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from profile_visible_v17 import read, sha, write
from batch_visible_v17 import schedule
from compare_phase20_exact_runs import validate_decode, shape_accelerator
from repeat_phase20_kernel_speed import check_prefix

ARCHIVE = ROOT/'results/tiny_target/motion_video_v13_20260916/evidence/results'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def distribution(values):
    require(len(values)>0 and all(np.isfinite(v) and v>=0 for v in values), 'Invalid timing samples')
    return dict(count=len(values), mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)), maximum_ms=float(max(values)))


def verify(evidence):
    generated = read(evidence/'generated_01.json')
    require(generated['passed'] and generated['error'] is None and not generated['real_media_read'], 'Generated gate failed')
    require(generated['reference_sha256']==sha(ROOT/'tiny_target/visible_learning.py'), 'Reference mask changed')
    require(len(generated['cases'])==150 and len(generated['invalid'])==3 and len(generated['timings'])==8, 'Generated schedule incomplete')
    require(all(r['exact'] and r['inputs_unchanged'] for r in generated['cases']), 'Mask mismatch')
    require({r['native'] for r in generated['cases']} == {True, False}, 'Both sparse and dense paths required')
    for name, digest in generated['source_sha256'].items():
        path = ROOT/('docs' if name.endswith('.md') else 'scripts')/name
        require(sha(path) == digest, 'Generated code/plan changed: '+name)
    require(sha(evidence/'build/liblearning_mask_v17.so') == generated['library_sha256'], 'Library changed')
    build = read(evidence/'build/build.json')
    require(build['returncode']==0 and build['library_sha256']==generated['library_sha256']
            and build['source_sha256']==sha(ROOT/'scripts/learning_mask_v17.cpp'), 'Build mismatch')
    profiles = {}
    for clip in ('0126','0082'):
        r = read(evidence/f'profile_{clip}.profile.json')
        require(r['passed'] and r['error'] is None and r['script_sha256']==sha(ROOT/'scripts/profile_visible_v17.py'), 'Baseline profile failed')
        require(r['baseline_launch_sha256']==sha(ARCHIVE/f'visible_{clip}_full_reuse/launch.json')
                and r['plan_sha256']==sha(ROOT/'docs/visible_speed_v17_plan.md'), 'Profile provenance changed')
        check_prefix(ARCHIVE/f'visible_{clip}_full_reuse', evidence/f'profile_{clip}', 128)
        profiles[clip] = dict(instrumented=True, calls={k:{q:v[q] for q in ('calls','mean_ms','total_ms')} for k,v in r['calls'].items()})
    directory = evidence/'trials_01'
    batch = read(directory/'batch.json')
    require(batch['passed'] and batch['error'] is None and batch['schedule']==schedule()
            and len(batch['rows'])==12 and batch['script_sha256']==sha(ROOT/'scripts/batch_visible_v17.py'), 'Incomplete or changed batch')
    trials, stored = [], {}
    for spec, row in zip(schedule(), batch['rows']):
        require(all(row[k]==v for k,v in spec.items()), 'Changed schedule order')
        name = spec['name']; path = directory/name
        r = read(path.with_suffix('.v17.json'))
        require(sha(path.with_suffix('.v17.json'))==row['sha256'] and r['passed'] and r['error'] is None, 'Trial failed/changed')
        require(all(r[k]==spec[k] for k in ('clip','mode','frames')), 'Execution scope mismatch')
        require(r['script_sha256']==sha(ROOT/'scripts/run_visible_v17.py') and r['gate_sha256']==sha(evidence/'generated_01.json')
                and r['library_sha256']==generated['library_sha256'] and not r['raw16_accessed']
                and not r['defaults_changed'] and not r['production_approved'], 'Trial provenance changed')
        require(r['candidate_plan_sha256']==sha(ROOT/'docs/visible_speed_v17_candidate.md'), 'Candidate plan changed')
        reference = ARCHIVE/f'visible_{spec["clip"]}_full_reuse'
        report, launch = read(path/'report.json'), read(path/'launch.json')
        old_report, old_launch = read(reference/'report.json'), read(reference/'launch.json')
        count = spec['frames'] or old_report['frames']
        require(report['completed'] and report['frames']==count and report['full_clip']==(spec['frames'] is None), 'Incomplete video')
        require(report['configuration']==old_report['configuration'], 'Detector policy changed')
        require(report['faint_target_synthetic_branch_enabled'] is False
                and old_report['faint_target_synthetic_branch_enabled'] is False, 'Visible branch contract changed')
        for key in ('source_sha256','fps','configuration','package_sha256','motion_config_sha256','external_accelerators','exact_cuda_stabilization'):
            require(launch[key]==old_launch[key], 'Launch identity changed: '+key)
        validate_decode(launch,report); shape_accelerator(launch)
        check_prefix(reference,path,count)
        require(sha(path/'frames.jsonl')==r['journal_sha256'] and sha(path.with_suffix('.execution.json'))==r['execution_sha256'], 'Journal/motion receipt mismatch')
        motion = read(path.with_suffix('.execution.json'))
        old_motion = read(reference.with_suffix('.execution.json'))
        for key in ('adapter_sha256','method_sha256','wrapper_sha256','runtime_sha256'):
            require(motion[key]==old_motion[key], 'Frozen motion harness changed: '+key)
        require(motion['branch']=='visible' and motion['mode']=='reuse' and motion['clip']==spec['clip']
                and motion['processed_frames']==count, 'Wrong motion execution scope')
        require(motion['passed'] and motion['closed'] and motion['error'] is None and motion['reuse_hits']==count-2
                and motion['reuse_misses']==1, 'Motion lifecycle failed')
        require([(p['frame'],p['identity']) for p in motion['motion']] ==
                [(p['frame'],p['identity']) for p in old_motion['motion'][:count-1]], 'Motion identity changed')
        require(len(r['consumer_frame_ms'])==count and r['fps']==row['fps']==report['processed_fps']
                and r['wall_s']==row['wall_s']==report['elapsed_seconds'], 'Timing mismatch')
        require(r['native_mask_calls']==0 if spec['mode']=='reference' else r['native_mask_calls']>0, 'Candidate not exercised')
        if spec['frames'] is None:
            for key in ('counts','qualified_tracks','qualified_track_count','availability','detection_status'):
                require(report[key]==old_report[key], 'Full aggregate changed: '+key)
        result = dict(**spec, count=count, fps=r['fps'], wall_s=r['wall_s'], exact=True,
                      native_mask_calls=r['native_mask_calls'], mask_call=distribution(r['mask_call_ms']),
                      consumer_service=distribution(r['consumer_frame_ms']), stage_means_ms={k:v['mean'] for k,v in report['timings_ms'].items()})
        trials.append(result); stored[name]=r
    performance = {}
    for clip in ('0126','0082'):
        arms = {}
        for mode in ('reference','candidate'):
            rows = [r for r in trials if r['frames']==128 and r['clip']==clip and r['mode']==mode]
            wall = sum(r['wall_s'] for r in rows)
            names = [r['name'] for r in rows]
            arms[mode] = dict(frames=256, wall_s=wall, fps=256/wall,
                consumer_service=distribution([v for name in names for v in stored[name]['consumer_frame_ms']]),
                mask_call=distribution([v for name in names for v in stored[name]['mask_call_ms']]))
        performance[clip] = dict(arms=arms, speedup=arms['candidate']['fps']/arms['reference']['fps'],
            paired_speedups=[stored[f'{clip}_repeat{i}_candidate']['fps']/stored[f'{clip}_repeat{i}_reference']['fps'] for i in range(2)])
    full = [r for r in trials if r['frames'] is None]
    full_wall=sum(r['wall_s'] for r in full); full_frames=sum(r['count'] for r in full)
    return dict(schema='seaqr.visible-speed-v17-summary.v1', verified=True, raw16_paused=True,
        generated_cases=150, baseline_profiles=profiles, prefix_performance=performance, trials=trials,
        full_regression_frames=full_frames, full_candidate_fps=full_frames/full_wall,
        full_candidate_service=distribution([v for r in full for v in stored[r['name']]['consumer_frame_ms']]),
        exact_non_timing_outputs=True, defaults_changed=False, production_approved=False,
        new_airborne_accuracy_validated=False, summarizer_sha256=sha(__file__),
        warning='Full candidate runs have archived, not fresh paired full timing references. '
                'Prefix repeats are two existing workloads, not independent accuracy data. '
                'Consumer service time includes decode wait/processing/journaling, not capture-to-alert latency.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.evidence);write(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trials','baseline_profiles')},indent=2))
