"""Sequential, fail-closed real-video gates and reversed-order paired timings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_motion_video_v13 import RUNTIME, VISIBLE, RAW_RESULTS, read, sha, write


def setup_imports():
    if RUNTIME.exists():
        sys.path[:0] = [str(RUNTIME), str(RUNTIME/'scripts')]
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def motion_identity(record):
    return [{k: row[k] for k in ('frame', 'identity', 'error') if k in row}
            for row in record['motion']]


def check_visible(reference, output, count, full):
    setup_imports()
    from compare_phase20_exact_runs import (without_timing, validate_decode,
        validate_gpu_transition, shape_accelerator)
    from repeat_phase20_kernel_speed import check_prefix
    left, right = [read(p/'launch.json') for p in (reference, output)]
    a, b = [read(p/'report.json') for p in (reference, output)]
    for key in ('source_sha256', 'fps', 'motion_config_sha256', 'configuration'):
        if left[key] != right[key]:
            raise AssertionError('Visible provenance changed: '+key)
    if not b['completed'] or b['frames'] != count or b['full_clip'] != full:
        raise AssertionError('Incomplete visible run')
    validate_decode(right, b)
    shape_accelerator(right)
    validate_gpu_transition(left, right, None)
    check_prefix(reference, output, count)
    return dict(exact=True, frames=count, reference_sha256=sha(reference/'frames.jsonl'),
                candidate_sha256=sha(output/'frames.jsonl'))


def check_raw(reference, output):
    setup_imports()
    from run_raw16_background_v7 import normalized_report
    from summarize_raw16_cpu_v6 import compare_source_motion, difference
    a, b = [normalized_report(read(p/'report.json')) for p in (reference, output)]
    diff = difference(a, b)
    source = compare_source_motion(reference, output)
    if diff or not source['source_frames_exact'] or not source['motion_points_exact']:
        raise AssertionError('RAW report/source/motion differs: '+str(diff or source))
    for name in ('candidate_decisions.json', 'global_fit_identities.json'):
        if read(reference/name) != read(output/name):
            raise AssertionError('RAW decisions differ: '+name)
    if not read(output/'comparison.json')['exact_gate_passed']:
        raise AssertionError('Original RAW gate failed')
    return dict(exact=True, frames=64, source=source,
                synthetic_controls_passed=read(output/'checks.json')['checks']['synthetic_controls_passed'])


def check_pair(left, right):
    a, b = [read(p.with_suffix('.execution.json')) for p in (left, right)]
    for record in (a, b):
        if not record['passed'] or not record['closed'] or record['error']:
            raise AssertionError('Incomplete execution lifecycle')
        if len(record['motion']) != record['processed_frames']-1:
            raise AssertionError('Motion record missing')
    for key in ('runtime_sha256', 'wrapper_sha256', 'adapter_sha256', 'method_sha256',
                'branch', 'clip', 'frames', 'injected', 'processed_frames'):
        if a[key] != b[key]:
            raise AssertionError('Pair identity changed: '+key)
    if a['mode'] != 'reference' or b['mode'] != 'reuse':
        raise AssertionError('Unexpected timing arms')
    if motion_identity(a) != motion_identity(b):
        raise AssertionError('Complete non-timing motion outputs differ')
    if b['reuse_hits'] != b['processed_frames']-2 or b['reuse_misses'] != 1:
        raise AssertionError('Incomplete reuse')
    return dict(exact=True, frames=a['processed_frames'],
                reference_fps=a['pipeline_fps'], reuse_fps=b['pipeline_fps'],
                speedup=b['pipeline_fps']/a['pipeline_fps'], reuse_hits=b['reuse_hits'])


def schedule(stage):
    if stage == 'smoke':
        return [('visible', '0126', 96, False, 'smoke', mode) for mode in ('reference', 'reuse')]
    if stage == 'full':
        return [('visible', clip, None, False, 'full', mode)
                for i, clip in enumerate(('0029', '0126', '0055', '0082'))
                for mode in (('reference', 'reuse') if i % 2 == 0 else ('reuse', 'reference'))]
    if stage == 'visible_repeats':
        return [('visible', clip, 128, False, 'repeat'+str(repeat), mode)
                for repeat in range(2) for clip in ('0126', '0082')
                for mode in (('reference', 'reuse') if repeat == 0 else ('reuse', 'reference'))]
    if stage == 'raw_checks':
        return [('raw', clip, 64, injected, 'check', 'reuse')
                for clip, injected in (('0040', False), ('0029', False), ('0040', True))]
    if stage == 'raw_repeats':
        return [('raw', clip, 64, False, 'repeat'+str(repeat), mode)
                for repeat in range(2) for clip in ('0040', '0029')
                for mode in (('reference', 'reuse') if repeat == 0 else ('reuse', 'reference'))]
    raise ValueError('Unknown stage')


def run(args):
    args.output.mkdir(exist_ok=True)
    prerequisite = {'full': 'smoke', 'visible_repeats': 'full', 'raw_repeats': 'raw_checks'}.get(args.stage)
    if prerequisite and not read(args.output/(prerequisite+'.json'))['passed']:
        raise ValueError('Prerequisite failed')
    target = args.output/(args.stage+'.json')
    if target.exists():
        raise FileExistsError(target)
    record = dict(stage=args.stage, passed=False, rows=[], comparisons=[], error=None,
                  script_sha256=sha(__file__), schedule=schedule(args.stage))
    pending = {}
    try:
        for branch, clip, frames, injected, tag, mode in record['schedule']:
            key = f'{branch}_{clip}_{tag}'+('_injected' if injected else '')
            name = key+'_'+mode
            output = args.output/name
            if output.exists() or output.with_suffix('.execution.json').exists():
                raise FileExistsError(output)
            command = [sys.executable, '-u', str(Path(__file__).with_name('run_motion_video_v13.py')),
                       '--branch', branch, '--clip', clip, '--mode', mode, '--output', str(output)]
            if frames is not None:
                command += ['--frames', str(frames)]
            if injected:
                command += ['--injected']
            print(json.dumps(dict(starting=name, unix=time.time())), flush=True)
            with output.with_suffix('.log').open('x') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            row = dict(name=name, command=command, returncode=process.returncode)
            record['rows'].append(row)
            if process.returncode:
                raise RuntimeError('Child failed: '+name)
            if branch == 'visible':
                extent = frames or read(VISIBLE/('pva_'+clip)/'report.json')['frames']
                row['archived'] = check_visible(VISIBLE/('pva_'+clip), output, extent, frames is None)
            else:
                reference = RAW_RESULTS/(('injected_0040_exact' if injected else 'timed_'+clip+'_1_exact'))
                row['archived'] = check_raw(reference, output)
            execution = read(output.with_suffix('.execution.json'))
            row.update(fps=execution['pipeline_fps'], hits=execution['reuse_hits'])
            pending.setdefault(key, {})[mode] = output
            if set(pending[key]) == {'reference', 'reuse'}:
                pair = pending[key]
                record['comparisons'].append(dict(key=key, **check_pair(pair['reference'], pair['reuse'])))
            print(json.dumps(row), flush=True)
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        write(target, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'full', 'visible_repeats', 'raw_checks', 'raw_repeats'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
