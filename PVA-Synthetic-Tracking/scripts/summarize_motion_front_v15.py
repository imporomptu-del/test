"""Locally recheck v15 point/fit/source identities and complete timing schedules."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from motion_front_pixels_v15 import logical_identity

ARCHIVE = ROOT/'results/tiny_target/motion_video_v13_20260916/evidence/results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_execution_fields(value, mode):
    if isinstance(value, dict):
        if 'previous_points' in value and 'current_points' in value and 'motion_image_size' in value:
            expected = 'CUDA' if mode == 'reference' else 'CPU_fused_half_linear_v15'
            require(value['backends']['motion_image_rescale'] == expected, 'Unexpected preparation backend')
            memory = value['metrics']['memory_bytes']
            fields = [k for k in ('motion_u8_frames_created', 'motion_u16_frames_created') if k in memory]
            require(len(fields) == 1, 'Ambiguous preparation dtype')
            element = 2 if fields[0] == 'motion_u16_frames_created' else 1
            w, h = value['full_image_size'] if mode == 'reference' else value['motion_image_size']
            require(memory[fields[0]] == 2*w*h*element, 'Incorrect allocated-buffer metadata')
        for child in value.values():
            validate_execution_fields(child, mode)
    elif isinstance(value, list):
        for child in value:
            validate_execution_fields(child, mode)


def compare_generated_pair(row):
    a, b = row['reference'], row['candidate']
    validate_execution_fields(a['identity'], 'reference')
    validate_execution_fields(b['identity'], 'candidate')
    require(a['logical'] == logical_identity(a['identity']) and b['logical'] == logical_identity(b['identity']), 'Invalid identity normalization')
    require(a['logical'] == b['logical'] and row['exact'], 'Point/fit mismatch')
    require(a['accepted'] == b['accepted'], 'Decision mismatch')


def summarize(evidence):
    gate = read(evidence/'generated_03.json')
    require(gate['passed'] and gate['error'] is None and not gate['real_media_read'], 'Generated gate failed')
    require(gate['runtime_motion_sha256'] == sha(ROOT/'tiny_target/motion/pva_pyrlk.py'), 'Motion runtime changed')
    require(gate['config_sha256'] == sha(ROOT/'configs/evaluation/raw16_motion_v6.json'), 'Configuration changed')
    require(set(gate['source_sha256']) == {'motion_front_v15.py', 'motion_front_pixels_v15.py',
            'motion_reuse_v12.py', 'check_motion_front_v15.py', 'motion_front_v15.cpp',
            'build_motion_front_v15.py'}, 'Unexpected source inventory')
    for name, digest in gate['source_sha256'].items():
        require(sha(ROOT/'scripts'/name) == digest, 'Changed generated source '+name)
    build = read(evidence/'build/build.json')
    require(build['returncode'] == 0 and build['source_sha256'] == sha(ROOT/'scripts/motion_front_v15.cpp')
            and build['builder_sha256'] == sha(ROOT/'scripts/build_motion_front_v15.py'), 'Build mismatch')
    require(build['library_sha256'] == gate['library_sha256'] == sha(evidence/'build/libmotion_front_v15.so'), 'Library mismatch')
    require(len(gate['pixels']) == 78 and all(r['exact'] and r['input_unchanged'] and r['different_pixels'] == 0
                                            for r in gate['pixels']), 'Pixel gate incomplete')
    require([(r['depth'], r['mask'], r['phase']) for r in gate['pixels'][:72]] ==
            [(d, m, p) for d in (8, 9, 10, 12, 14, 16) for m in ('all', 'none', 'pattern') for p in range(4)], 'Pixel schedule mismatch')
    require([(r['depth'], r['native_scene']) for r in gate['pixels'][72:]] ==
            [(d, s) for d in (8, 16) for s in ('range', 'dim', 'flat')], 'Native pixel schedule mismatch')
    require(len(gate['independent']) == 48, 'Independent controls missing')
    for row in gate['independent']:
        compare_generated_pair(row)
        require(row['expected_decisions'], 'Expected known-motion decision failed')
    require([(r['shape'], r['depth'], r['kind']) for r in gate['sequences']] ==
            [(list(s), d, k) for s in ((960, 1280), (3190, 4784)) for d in (8, 16)
             for k in ('smooth', 'recovery', 'invalidation', 'reordered')], 'Sequence schedule mismatch')
    for seq in gate['sequences']:
        require(seq['inputs_unchanged'], 'Input mutation')
        require(len(seq['rows']) == (7 if seq['kind'] == 'recovery' else 6 if seq['kind'] == 'reordered' else 5), 'Missing sequence pairs')
        for row in seq['rows']:
            compare_generated_pair(row)
    require([(r['depth'], r['repeat'], r['mode']) for r in gate['timings']] ==
            [(d, i, m) for d in (8, 16) for i in range(4)
             for m in (('reference', 'candidate') if i % 2 == 0 else ('candidate', 'reference'))], 'Timing schedule mismatch')
    generated = {}
    for depth in (8, 16):
        group = [r for r in gate['timings'] if r['depth'] == depth]
        expected = [r['logical'] for r in group[0]['samples']]
        for trial in group:
            require(len(trial['samples']) == 5 and [r['logical'] for r in trial['samples']] == expected, 'Timed generated mismatch')
            for sample in trial['samples']:
                validate_execution_fields(sample['identity'], trial['mode'])
                require(sample['logical'] == logical_identity(sample['identity']), 'Invalid timed normalization')
        generated[depth] = {mode: dict(
            warm_median_ms=statistics.median(s['estimator_ms'] for t in group if t['mode'] == mode for s in t['samples'][1:]),
            all_calls_mean_ms=statistics.mean(s['estimator_ms'] for t in group if t['mode'] == mode for s in t['samples']))
            for mode in ('reference', 'candidate')}
    batch = read(evidence/'real/batch.json')
    require(batch['passed'] and batch['error'] is None and not batch['pipeline_benchmark'], 'Real batch failed')
    require(batch['gate_sha256'] == sha(evidence/'generated_03.json') and
            batch['script_sha256'] == sha(ROOT/'scripts/run_motion_front_real_v15.py'), 'Real gate/source changed')
    schedule = [(i, c, m) for i in range(2) for c in ('0029', '0040')
                for m in (('reference', 'candidate') if i == 0 else ('candidate', 'reference'))]
    require([(r['repeat'], r['clip'], r['mode']) for r in batch['trials']] == schedule, 'Real schedule incomplete')
    all_times, real = {}, {}
    for receipt in batch['trials']:
        name = receipt['name']
        require(name == f'raw_{receipt["clip"]}_repeat{receipt["repeat"]}_{receipt["mode"]}', 'Unexpected trial path')
        path = evidence/'real'/(name+'.json')
        require(sha(path) == receipt['evidence_sha256'], 'Trial hash mismatch')
        trial = read(path)
        require(trial['passed'] and trial['closed'] and trial['error'] is None and trial['reuse_hits'] == 62
                and trial['reuse_misses'] == 1, 'Lifecycle/reuse failure')
        archive = ARCHIVE/f'raw_{trial["clip"]}_repeat0_reuse'
        old_motion = read(archive.with_suffix('.execution.json'))['motion']
        old_fits = read(archive/'global_fit_identities.json')
        require(trial['frames'] == read(archive/'source_frames.json') and len(trial['frames']) == 64, 'Source frames changed')
        require([r['frame'] for r in trial['pairs']] == list(range(1, 64)), 'Missing/reordered real pairs')
        for row, old, fit in zip(trial['pairs'], old_motion, old_fits):
            validate_execution_fields(row['identity'], trial['mode'])
            require(row['archived_exact'] and logical_identity(row['identity']) == logical_identity(old['identity'])
                    and row['fit_identity'] == fit, 'Archived real point/fit mismatch')
        times = [p['estimator_ms'] for p in trial['pairs']]
        require(statistics.mean(times) == receipt['mean_estimator_ms'] and
                statistics.median(times[1:]) == receipt['median_warm_estimator_ms'], 'Timing accounting mismatch')
        all_times[name] = times
        real[name] = {k: receipt[k] for k in ('mode', 'clip', 'repeat', 'mean_estimator_ms', 'median_warm_estimator_ms', 'cold_estimator_ms')}
    performance = {}
    for clip in ('0029', '0040'):
        arms = {mode: [v for name, times in all_times.items() if real[name]['clip'] == clip and real[name]['mode'] == mode
                       for v in times] for mode in ('reference', 'candidate')}
        means = {k: statistics.mean(v) for k, v in arms.items()}
        performance[clip] = dict(mean_estimator_ms=means,
                                 estimator_speedup=means['reference']/means['candidate'],
                                 estimator_time_reduction_percent=100*(1-means['candidate']/means['reference']),
                                 candidate_remaining_budget_ms_at_10fps=100-means['candidate'])
    return dict(schema='seaqr.motion-front-v15-summary.v1', verified=True,
                source_sha256=gate['source_sha256'], summarizer_sha256=sha(__file__),
                evidence_sha256={p: sha(evidence/p) for p in ('generated_01.json', 'generated_02.json', 'generated_03.json', 'real/batch.json')},
                pixel_cases=78, independent_motion_controls=48, sequence_cases=16,
                generated_estimator_timed_calls=80, real_trials=8, real_pair_comparisons=504,
                generated=generated, real_trials_detail=real, real_motion_performance=performance,
                numerical_point_fit_parity=True, native_detection_pixels_unchanged=True,
                new_full_pipeline_measured=False, new_accuracy_measured=False, defaults_changed=False,
                production_approved=False,
                warning='Motion-only native RAW prefixes, not end-to-end FPS. Byte/point/fit parity is regression evidence, '
                        'not airborne recall/FAR or resolution of existing misses. Two execution-metadata fields are excluded explicitly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evidence)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
