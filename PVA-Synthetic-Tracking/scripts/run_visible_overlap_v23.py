"""Frozen v20 algorithms with explicit bounded execution ownership and receipts."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
V20 = Path('/tmp/seaqr_visible_speed_v20_XUf1LR')
V17 = Path('/tmp/seaqr_visible_speed_v17_EER6lm')
V13 = Path('/tmp/seaqr_video_v13_IyK7eQ')
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')
FROZEN = ('frame_lookahead_v23.py', 'visible_overlap_v23.py', 'run_visible_overlap_v23.py',
          'batch_visible_overlap_v23.py', 'visible_overlap_v23_plan.md', 'test_frame_lookahead_v23.py',
          'test_visible_overlap_v23.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def verify_freeze():
    frozen = read(HERE/'freeze.json')
    if set(frozen['files']) != set(FROZEN) or any(sha(HERE/n) != d for n,d in frozen['files'].items()):
        raise ValueError('v23 source/test/plan freeze changed')
    tests = read(HERE/'generated_01.json')
    if (not tests['passed'] or tests['returncode'] != 0 or tests['real_media_read']
            or tests['test_sha256'] != frozen['files']['test_frame_lookahead_v23.py']
            or tests['adapter_test_sha256'] != frozen['files']['test_visible_overlap_v23.py']
            or tests['adapter_sha256'] != frozen['files']['visible_overlap_v23.py']
            or tests['controller_sha256'] != frozen['files']['frame_lookahead_v23.py']
            or tests['log_sha256'] != sha(HERE/'generated_01.log')
            or frozen['generated_sha256'] != sha(HERE/'generated_01.json')):
        raise ValueError('Generated ownership gate changed/failed')
    return frozen


def freeze():
    if any((HERE/n).exists() for n in ('freeze.json','generated_01.json','generated_01.log')):
        raise FileExistsError('Fresh generated gate and freeze only')
    files = {n:sha(HERE/n) for n in FROZEN}
    command = [sys.executable,'-m','unittest','-v','test_frame_lookahead_v23','test_visible_overlap_v23']
    with (HERE/'generated_01.log').open('x') as log:
        result = subprocess.run(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    write(HERE/'generated_01.json',dict(passed=result.returncode==0,returncode=result.returncode,
        command=command,real_media_read=False,log_sha256=sha(HERE/'generated_01.log'),
        test_sha256=files['test_frame_lookahead_v23.py'],adapter_test_sha256=files['test_visible_overlap_v23.py'],
        controller_sha256=files['frame_lookahead_v23.py'],adapter_sha256=files['visible_overlap_v23.py']))
    if result.returncode:
        raise RuntimeError('Generated ownership and adapter tests failed')
    write(HERE/'freeze.json',dict(files=files,generated_sha256=sha(HERE/'generated_01.json')))


def validate_snapshot(snap, count):
    rows = snap['frames']
    if [r['frame'] for r in rows] != list(range(count)):
        raise AssertionError('Missing/extra decoded, prepared or completed frames')
    for r in rows:
        keys = ['ready_ns', 'prepare_start_ns', 'prepare_end_ns', 'detector_start_ns',
                'detector_end_ns', 'tracking_start_ns', 'tracking_end_ns', 'consumer_complete_ns']
        values = [r[k] for k in keys]
        if values != sorted(values) or r['queue_wait_ms'] < 0:
            raise AssertionError('Invalid timing order')
        if not r['ready_ns'] <= r['consumer_received_ns'] <= r['detector_start_ns']:
            raise AssertionError('Invalid consumer timing')
    if snap['engine'] is not None:
        s = snap['engine']; capacity = 1 if snap['mode'] == 'serial' else 2
        if not (s['closed'] and s['joined'] and s['discarded_on_cancel'] == 0
                and s['prepared'] == s['delivered'] == s['released'] == count
                and 0 < s['max_owned'] <= capacity and s['owner_thread_id'] != snap['consumer_thread_id']):
            raise AssertionError('Incomplete or unsafe ownership lifecycle')
        for i,r in enumerate(rows):
            if r['slot'] != i % 2:
                raise AssertionError('Incorrect buffer rotation')
            prior = i-capacity
            if prior >= 0 and r['prepare_start_ns'] < rows[prior]['consumer_complete_ns']:
                raise AssertionError('A leased frame was overwritten')
    return True


def ensure_output_fresh(output):
    # The parent batch exclusively creates the run's log before spawning us.
    # Existing data/receipts remain forbidden; that active log is not a receipt.
    if output.exists() or any(p.suffix != '.log' for p in output.parent.glob(output.name+'.*')):
        raise FileExistsError(output)


def run(args):
    if (args.clip not in ('0029','0126','0055','0082') or args.mode not in ('reference','serial','overlap')
            or args.frames not in (128,None) or (args.frames and args.clip not in ('0126','0082'))
            or (args.library and args.frames != 128)):
        raise ValueError('Outside bounded development experiment')
    ensure_output_fresh(args.output)
    frozen = verify_freeze()
    sys.path[:0] = [str(V20),str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    import run_visible_v20 as baseline
    # v20's implementation and generated native helper gates remain authoritative.
    baseline.generated_gate()
    from visible_overlap_v23 import Binding, install
    counts = [0,0]
    bridge = None
    if args.library:
        bridge = ctypes.CDLL(str(args.library.resolve(strict=True)))
        bridge.seaqr_trace_push.argtypes = [ctypes.c_char_p]
        bridge.seaqr_trace_push.restype = ctypes.c_int
        bridge.seaqr_trace_pop.argtypes = []
        bridge.seaqr_trace_pop.restype = ctypes.c_int
    @contextmanager
    def marker(frame, stage):
        bridge.seaqr_trace_push(f'seaqr23|{frame}|{stage}'.encode())
        counts[0] += 1
        try:
            yield
        finally:
            bridge.seaqr_trace_pop()
            counts[1] += 1
    binding = Binding(args.mode, marker if bridge else None)
    record = dict(schema='seaqr.visible-overlap-v23.v1', passed=False,error=None,
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
        if counts[0] != counts[1] or (bridge and counts[0] != 3*old['processed_frames']):
            raise AssertionError('Incomplete annotations')
        record.update(passed=True,baseline_receipt_sha256=sha(args.output.with_suffix('.v20.json')),
                      fps=old['fps'],wall_s=old['wall_s'],processed_frames=old['processed_frames'])
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(execution=binding.snapshot(),nvtx_push_pop_counts=counts,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output.with_suffix('.v23.json'), record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--clip')
    parser.add_argument('--mode')
    parser.add_argument('--frames',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    freeze() if args.freeze else run(args)
