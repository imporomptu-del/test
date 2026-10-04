"""Exact 8-bit-only integration and paired timing; no RAW16 decoder calls."""
import argparse
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import sys
import time
from unittest.mock import patch
from profile_visible_v17 import HERE, RUNTIME, V13, VISIBLE, read, sha, write

CLIPS = ('0029', '0126', '0055', '0082')


def scope(clip, mode, frames):
    if clip not in CLIPS or mode not in ('reference', 'candidate') or frames not in (None, 128):
        raise ValueError('Outside frozen 8-bit experiment scope')
    if frames is not None and clip not in ('0126', '0082'):
        raise ValueError('Prefix timing limited to two frozen workloads')


def verify(clip):
    reference = V13/'results'/f'visible_{clip}_full_reuse'
    launch = read(reference/'launch.json')
    for name, digest in launch['package_sha256'].items():
        if sha(RUNTIME/'tiny_target'/name) != digest:
            raise ValueError('Frozen visible package changed: '+name)
    if sha(VISIBLE/'config.json') != launch['config_sha256']:
        raise ValueError('Frozen visible configuration changed')
    cfg = launch['configuration']
    if (sha(cfg['cuda_median_library']) != launch['exact_cuda_stabilization']['library_sha256']
            or sha(cfg['native_shape_library']) != cfg['native_shape_library_sha256']):
        raise ValueError('Original accelerator changed')
    gate = read(HERE/'generated_01.json')
    if (not gate['passed'] or gate['error'] is not None or gate['real_media_read']
            or len(gate['cases']) != 150 or len(gate['invalid']) != 3 or len(gate['timings']) != 8
            or not all(r['exact'] and r['inputs_unchanged'] for r in gate['cases'])
            or not any(r['native'] for r in gate['cases']) or not any(not r['native'] for r in gate['cases'])):
        raise ValueError('Complete generated gate required')
    for name, digest in gate['source_sha256'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Generated candidate changed: '+name)
    if sha(HERE/'build/liblearning_mask_v17.so') != gate['library_sha256']:
        raise ValueError('Generated library changed')
    return reference, launch


def run(args):
    scope(args.clip, args.mode, args.frames)
    if args.output.exists() or args.output.with_suffix('.v17.json').exists():
        raise FileExistsError(args.output)
    reference, launch = verify(args.clip)
    sys.path[:0] = [str(HERE), str(V13), str(RUNTIME), str(RUNTIME/'scripts')]
    from run_motion_video_v13 import run as baseline
    from batch_motion_video_v13 import check_visible
    from tiny_target import visible_resident as resident
    from tiny_target.visible_decode import VisibleFrameReader
    from learning_mask_v17 import LearningMaskV17, reference as reference_mask
    candidate = LearningMaskV17(HERE/'build/liblearning_mask_v17.so')
    mask_function = reference_mask if args.mode == 'reference' else candidate
    mask_times, intervals = [], []
    previous_read = None
    original_read, original_close = VisibleFrameReader.read, VisibleFrameReader.close
    def read_frame(reader):
        nonlocal previous_read
        now = time.perf_counter()
        if previous_read is not None:
            intervals.append(1000*(now-previous_read))
        previous_read = now
        result = original_read(reader)
        if result[0] is None:
            previous_read = None
        return result
    def close_reader(reader):
        nonlocal previous_read
        if previous_read is not None:
            intervals.append(1000*(time.perf_counter()-previous_read))
            previous_read = None
        return original_close(reader)
    def learning_mask(*a, **kw):
        start = time.perf_counter()
        result = mask_function(*a, **kw)
        mask_times.append(1000*(time.perf_counter()-start))
        return result
    record = dict(passed=False, error=None, clip=args.clip, mode=args.mode, frames=args.frames,
                  script_sha256=sha(__file__), candidate_plan_sha256=sha(HERE/'visible_speed_v17_candidate.md'),
                  gate_sha256=sha(HERE/'generated_01.json'), library_sha256=sha(HERE/'build/liblearning_mask_v17.so'),
                  raw16_accessed=False, defaults_changed=False, production_approved=False)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(resident, 'shape_learning_mask', learning_mask))
            stack.enter_context(patch.object(VisibleFrameReader, 'read', read_frame))
            stack.enter_context(patch.object(VisibleFrameReader, 'close', close_reader))
            baseline(argparse.Namespace(branch='visible', clip=args.clip, frames=args.frames,
                                        injected=False, mode='reuse', output=args.output))
        report = read(args.output/'report.json')
        count = report['frames']
        record['comparison'] = check_visible(reference, args.output, count, args.frames is None)
        expected = read(reference.with_suffix('.execution.json'))['motion'][:count-1]
        observed = read(args.output.with_suffix('.execution.json'))
        if [(r['frame'], r['identity']) for r in expected] != [(r['frame'], r['identity']) for r in observed['motion']]:
            raise AssertionError('Complete motion outputs changed')
        if not observed['passed'] or not observed['closed'] or observed['reuse_hits'] != count-2:
            raise AssertionError('Motion lifecycle failed')
        if len(intervals) != count:
            raise AssertionError('Incomplete consumer frame timing')
        if args.mode == 'candidate' and candidate.calls == 0:
            raise AssertionError('Candidate sparse branch was not exercised')
        if args.frames is None:
            old = read(reference/'report.json')
            for key in ('counts', 'qualified_tracks', 'qualified_track_count', 'availability', 'detection_status'):
                if report[key] != old[key]:
                    raise AssertionError('Full aggregate changed: '+key)
        record.update(passed=True, processed_frames=count, fps=report['processed_fps'], wall_s=report['elapsed_seconds'],
                      journal_sha256=sha(args.output/'frames.jsonl'), execution_sha256=sha(args.output.with_suffix('.execution.json')))
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(native_mask_calls=candidate.calls, mask_call_ms=mask_times,
                      consumer_frame_ms=intervals,
                      service_time_boundary='Consumer read-entry to next read-entry, or last pre-close. Includes decode wait, '
                                            'processing and journaling; excludes final EOF wait/join. Not camera-to-alert latency.')
        write(args.output.with_suffix('.v17.json'), record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', required=True)
    p.add_argument('--mode', required=True)
    p.add_argument('--frames', type=int)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
