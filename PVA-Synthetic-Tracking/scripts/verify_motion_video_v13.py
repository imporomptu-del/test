"""Independent report-only recheck; no source video access or VPI required."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from batch_motion_video_v13 import schedule, check_pair, check_visible, check_raw
from run_motion_video_v13 import read, sha, write
from motion_reuse_v12 import generated_method

VISIBLE = ROOT/'results/tiny_target/phase20/decode_overlap_v10_20260914/jetson/pipeline'
RAW = ROOT/'results/tiny_target/raw16_exact_v9_20260916/evidence/results/final'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def verify(directory):
    results, trials = {}, {}
    runtime = read(RAW/'timed_0040_1_exact/experiment.json')['provenance']['package_sha256']
    methods = hashlib.sha256(generated_method().encode()).hexdigest()
    for stage in ('smoke', 'raw_checks', 'full', 'visible_repeats', 'raw_repeats'):
        record = read(directory/(stage+'.json'))
        require(record['passed'] is True and record['error'] is None, 'Stage failed: '+stage)
        expected = schedule(stage)
        require(record['schedule'] == [list(r) for r in expected], 'Schedule mismatch')
        require(record['script_sha256'] == sha(ROOT/'scripts/batch_motion_video_v13.py'), 'Batch changed')
        require(len(record['rows']) == len(expected), 'Incomplete stage')
        paired, comparisons = {}, []
        for row, spec in zip(record['rows'], expected):
            branch, clip, frames, injected, tag, mode = spec
            key = f'{branch}_{clip}_{tag}'+('_injected' if injected else '')
            name = key+'_'+mode
            require(row['name'] == name and row['returncode'] == 0, 'Missing/failed job')
            output = directory/name
            execution = read(output.with_suffix('.execution.json'))
            require(execution['passed'] is True and execution['closed'] is True
                    and execution['error'] is None, 'Failed lifecycle')
            require(execution['runtime_sha256'] == runtime, 'Frozen runtime changed')
            for field, file in (('wrapper_sha256', 'run_motion_video_v13.py'),
                                ('adapter_sha256', 'motion_reuse_v12.py')):
                require(execution[field] == sha(ROOT/'scripts'/file), 'Harness changed')
            require(execution['method_sha256'] == methods, 'Generated method changed')
            require([execution[k] for k in ('branch', 'clip', 'frames', 'injected', 'mode')]
                    == [branch, clip, frames, injected, mode], 'Run spec changed')
            require(execution['instances'] == 1 and not execution['defaults_changed']
                    and not execution['production_approved'], 'Unexpected execution scope')
            count = execution['processed_frames']
            require([r['frame'] for r in execution['motion']] == list(range(1, count)), 'Missing/reordered motion')
            require(all('identity' in r and 'error' not in r for r in execution['motion']), 'Motion error')
            require(mode == 'reference' or (execution['reuse_hits'] == count-2
                    and execution['reuse_misses'] == 1), 'Reuse not exercised')
            for value in [execution['pipeline_fps'], execution['process_wall_s']]+[r['estimator_s'] for r in execution['motion']]:
                require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid timing')
            if branch == 'visible':
                extent = frames or read(VISIBLE/('pva_'+clip)/'report.json')['frames']
                comparison = check_visible(VISIBLE/('pva_'+clip), output, extent, frames is None)
                report = read(output/'report.json')
                require(execution['pipeline_fps'] == report['processed_fps'], 'FPS mismatch')
                launch = read(output/'launch.json')
                require({'tiny_target/'+k: v for k, v in launch['package_sha256'].items()}
                        == runtime, 'Visible runtime provenance mismatch')
            else:
                reference = RAW/('injected_0040_exact' if injected else 'timed_'+clip+'_1_exact')
                comparison = check_raw(reference, output)
                require(execution['pipeline_fps'] == 64/read(output/'checks.json')['elapsed_wall_s'], 'RAW FPS mismatch')
            require(comparison == row['archived'], 'Archived comparison receipt mismatch')
            paired.setdefault(key, {})[mode] = output
            if set(paired[key]) == {'reference', 'reuse'}:
                comparisons.append(dict(key=key, **check_pair(paired[key]['reference'], paired[key]['reuse'])))
            samples = execution['motion']
            rss = [int(r['rss']['VmRSS'].split()[0]) for r in samples if r.get('rss')]
            trials[name] = dict(branch=branch, clip=clip, mode=mode, frames=count,
                fps=execution['pipeline_fps'], reuse_hits=execution['reuse_hits'],
                estimator_median_ms=1000*statistics.median(r['estimator_s'] for r in samples),
                estimator_mean_ms=1000*statistics.mean(r['estimator_s'] for r in samples),
                rss_sample_min_kib=min(rss), rss_sample_max_kib=max(rss),
                rss_first_kib=rss[0], rss_last_kib=rss[-1],
                execution_sha256=sha(output.with_suffix('.execution.json')))
        require(record['comparisons'] == comparisons, 'Pair comparison receipt mismatch')
        results[stage] = dict(passed=True, comparisons=comparisons, runs=len(expected))
    full = {mode: [t for name, t in trials.items() if name.startswith('visible_')
                  and '_full_' in name and t['mode'] == mode] for mode in ('reference', 'reuse')}
    fps = {mode: sum(t['frames'] for t in rows)/sum(t['frames']/t['fps'] for t in rows)
           for mode, rows in full.items()}
    repeats = {}
    for branch, stage in (('visible', 'visible_repeats'), ('raw', 'raw_repeats')):
        rows = results[stage]['comparisons']
        repeats[branch] = {}
        for clip in sorted({r['key'].split('_')[1] for r in rows}):
            group = [r for r in rows if r['key'].split('_')[1] == clip]
            repeats[branch][clip] = dict(
                reference_fps_median=statistics.median(r['reference_fps'] for r in group),
                reuse_fps_median=statistics.median(r['reuse_fps'] for r in group),
                paired_speedups=[r['speedup'] for r in group],
                all_pairs_faster=all(r['speedup'] > 1 for r in group))
    return dict(verified=True, verifier_sha256=sha(__file__), stages=results, trials=trials,
                full_visible_weighted_fps=fps, full_visible_speedup=fps['reuse']/fps['reference'],
                full_visible_frames_per_arm=sum(t['frames'] for t in full['reuse']),
                repeated_prefixes=repeats,
                real_airborne_accuracy_validated=False, production_approved=False, defaults_changed=False,
                warning='Development execution parity, not general recall/FAR. Full visible runs are single pairs; '
                'prefix repeats are short. Includes motion identity instrumentation and normal journaling. '
                'RAW prefixes include source hashing. RSS is sampled process memory, not GPU leak certification.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    write(args.output, verify(args.directory))
