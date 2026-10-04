"""Frozen v20 algorithms, v23 ownership, v24 shared admission and stage release."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import os
from pathlib import Path
import resource
import subprocess
import sys
from run_visible_overlap_v23 import (V20, V17, V13, RUNTIME, sha, read, write,
                                     ensure_output_fresh, validate_snapshot as validate_ownership)

HERE = Path(__file__).resolve().parent
DEPENDENCIES = {
    'frame_lookahead_v23.py': '8193584fda014cff16945d92fbe19ca89266fc4b713dae05ebf4672a41052ad8',
    'visible_overlap_v23.py': 'b611ea55a68a67a727c2754751b478e8d3d6a4ff0b731af8a83c08612312426f',
    'run_visible_overlap_v23.py': '4bd9e41b8f4eb72b2f6a5c2f71318dc6d15875989bc75e58938365fac8e27959',
    'test_frame_lookahead_v23.py': '1c803531f93abbb231a949d0755c7fc3a1dd33450e8a4b5430b626bdd3704fa7',
    'test_visible_overlap_v23.py': '76627e859593e65742829fe47d9d8045416961fc33bfba376253c96b646675d1',
}
FROZEN = tuple(DEPENDENCIES) + ('stage_control_v24.py', 'visible_stage_v24.py',
    'run_visible_stage_v24.py', 'batch_visible_stage_v24.py', 'test_visible_stage_v24.py',
    'visible_stage_v24_plan.md')
TESTS = ('test_frame_lookahead_v23', 'test_visible_overlap_v23', 'test_visible_stage_v24')


def verify_freeze():
    frozen = read(HERE/'freeze.json')
    if set(frozen['files']) != set(FROZEN):
        raise ValueError('Incomplete v24 freeze')
    for name, digest in frozen['files'].items():
        if sha(HERE/name) != digest or (name in DEPENDENCIES and digest != DEPENDENCIES[name]):
            raise ValueError('Frozen dependency changed: '+name)
    g = read(HERE/'generated_01.json')
    if (not g['passed'] or g['returncode'] != 0 or g['real_media_read'] or g['tests'] != list(TESTS)
            or g['source_sha256'] != frozen['files']
            or g['log_sha256'] != sha(HERE/'generated_01.log')
            or frozen['generated_sha256'] != sha(HERE/'generated_01.json')):
        raise ValueError('Failed or changed generated ownership gate')
    return frozen


def freeze():
    if any((HERE/n).exists() for n in ('freeze.json','generated_01.json','generated_01.log')):
        raise FileExistsError('Fresh generated gate and freeze only')
    files = {name: sha(HERE/name) for name in FROZEN}
    if any(files[name] != digest for name,digest in DEPENDENCIES.items()):
        raise ValueError('Immutable v23 dependency changed')
    command = [sys.executable, '-m', 'unittest', '-v', *TESTS]
    env = dict(os.environ, PYTHONPATH=str(RUNTIME))
    with (HERE/'generated_01.log').open('x') as log:
        done = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(HERE/'generated_01.json', dict(passed=done.returncode==0, returncode=done.returncode,
        tests=list(TESTS), command=command, real_media_read=False, source_sha256=files,
        log_sha256=sha(HERE/'generated_01.log')))
    if done.returncode:
        raise RuntimeError('Generated stage/ownership tests failed')
    write(HERE/'freeze.json', dict(files=files, generated_sha256=sha(HERE/'generated_01.json')))


def validate_snapshot(snap, count):
    validate_ownership(snap, count)
    rows = snap['frames']
    if snap['policy'] not in ('reference','bounded','staged') or snap['mode'] != (
            'reference' if snap['policy']=='reference' else 'overlap'):
        raise AssertionError('Policy/ownership mismatch')
    admission, gate = snap['admission'], snap['gpu_release']
    if not (admission['capacity']==2 and 0 < admission['maximum'] <= 2
            and admission['stopped'] and not admission['held'] and gate['stopped']
            and gate['completed']==count-1 and len(gate['release_times_ns'])==count):
        raise AssertionError('Incomplete shared admission or GPU gate lifecycle')
    events = admission['events']
    if len(events) not in (count, count+1) or [e['frame'] for e in events] != list(range(len(events))):
        raise AssertionError('Admission sequence differs from consumed frames')
    endpoints = []
    for i,e in enumerate(events):
        if not (0 < e['request_ns'] <= e['admitted_ns'] <= e['released_ns']):
            raise AssertionError('Invalid admission interval')
        if e['disposition'] != ('consumed' if i<count else 'eof'):
            raise AssertionError('Unreleased or failed input reservation')
        endpoints += [(e['admitted_ns'], 1), (e['released_ns'], -1)]
    outstanding = maximum = 0
    for _,delta in sorted(endpoints):
        outstanding += delta
        maximum = max(maximum,outstanding)
        if not 0 <= outstanding <= 2:
            raise AssertionError('Global two-frame admission budget exceeded')
    if outstanding or maximum != admission['maximum']:
        raise AssertionError('Recorded admission maximum differs from event history')
    for i,r in enumerate(rows):
        keys = ('request_ns','admitted_ns','ready_ns','prepare_start_ns','cpu_prepare_start_ns',
                'cpu_prepare_end_ns','warp_wait_start_ns','warp_gpu_start_ns','warp_gpu_end_ns',
                'prepare_end_ns','detector_start_ns','detector_end_ns','gpu_release_ns',
                'tracking_start_ns','tracking_end_ns','consumer_complete_ns')
        values = [r[k] for k in keys]
        if any(type(v) is not int or v <= 0 for v in values) or values != sorted(values):
            raise AssertionError('Invalid stage/request timestamps')
        if not (r['request_ns']==events[i]['request_ns'] and r['admitted_ns']==events[i]['admitted_ns']
                and r['consumer_complete_ns']<=events[i]['released_ns']
                and r['gpu_release_ns']==gate['release_times_ns'][i]):
            raise AssertionError('Stage timestamps differ from admission/release receipts')
        if i and (rows[i-1]['consumer_complete_ns'] > r['consumer_received_ns']
                  or rows[i-1]['prepare_end_ns'] > r['prepare_start_ns']):
            raise AssertionError('Out-of-order preparation/consumption')
        if i and snap['policy']=='staged' and r['warp_gpu_start_ns'] < rows[i-1]['gpu_release_ns']:
            raise AssertionError('Future full-resolution warp released before current detector')
    return True


def run(args):
    if (args.clip not in ('0029','0126','0055','0082') or args.mode not in ('reference','bounded','staged')
            or args.frames not in (128,None) or (args.frames and args.clip not in ('0126','0082'))
            or (args.library and (args.frames != 128 or args.mode != 'staged'))):
        raise ValueError('Outside frozen development scope')
    ensure_output_fresh(args.output)
    frozen = verify_freeze()
    sys.path[:0] = [str(V20),str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    import run_visible_v20 as baseline
    baseline.generated_gate()
    from visible_stage_v24 import StageBinding, install
    bridge, counts = None, [0,0]
    if args.library:
        bridge = ctypes.CDLL(str(args.library.resolve(strict=True)))
        bridge.seaqr_trace_push.argtypes = [ctypes.c_char_p]
        bridge.seaqr_trace_push.restype = ctypes.c_int
        bridge.seaqr_trace_pop.argtypes = []
        bridge.seaqr_trace_pop.restype = ctypes.c_int
    @contextmanager
    def marker(frame, stage):
        version = 'seaqr24' if stage in ('cpu_prepare','warp_wait','warp_gpu') else 'seaqr23'
        bridge.seaqr_trace_push(f'{version}|{frame}|{stage}'.encode())
        counts[0] += 1
        try:
            yield
        finally:
            bridge.seaqr_trace_pop()
            counts[1] += 1
    binding = StageBinding(args.mode, marker if bridge else None)
    record = dict(schema='seaqr.visible-stage-v24.v1', passed=False,error=None,
        clip=args.clip,mode=args.mode,frames=args.frames,source_sha256=frozen['files'],
        freeze_sha256=sha(HERE/'freeze.json'),generated_sha256=sha(HERE/'generated_01.json'),
        baseline_freeze_sha256=sha(V20/'freeze.json'),raw16_accessed=False,defaults_changed=False,
        gpu_changed=False,algorithm_changed=False,production_approved=False,traced=bridge is not None,
        bridge_sha256=sha(args.library) if bridge else None)
    try:
        with ExitStack() as stack:
            install(stack, binding)
            baseline.run(argparse.Namespace(clip=args.clip,mode='candidate',frames=args.frames,output=args.output))
        old = read(args.output.with_suffix('.v20.json'))
        if not old['passed'] or old['error'] is not None:
            raise AssertionError('Frozen complete non-timing output parity failed')
        validate_snapshot(binding.snapshot(), old['processed_frames'])
        if counts != ([6*old['processed_frames']]*2 if bridge else [0,0]):
            raise AssertionError('Incomplete stage annotations')
        record.update(passed=True,baseline_receipt_sha256=sha(args.output.with_suffix('.v20.json')),
                      fps=old['fps'],wall_s=old['wall_s'],processed_frames=old['processed_frames'])
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(execution=binding.snapshot(),nvtx_push_pop_counts=counts,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output.with_suffix('.v24.json'), record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--clip')
    parser.add_argument('--mode')
    parser.add_argument('--frames',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    freeze() if args.freeze else run(args)
