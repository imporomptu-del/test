"""Matched host/NVTX and read-only telemetry over the immutable v29 harness."""
import argparse
from contextlib import contextmanager, ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

V29 = Path('/tmp/seaqr_visible_combined_v29_s8XhL1')
V22 = Path('/tmp/seaqr_gpu_timeline_v22_UStOG5')
HERE = Path(__file__).resolve().parent
MAJOR = ('motion_reference', 'cpu_prepare', 'warp_wait', 'warp_gpu', 'detector', 'tracking')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def scope(clip, arm):
    if clip not in ('0082', '0126') or arm not in ('v26', 'combined'):
        raise ValueError('Only frozen two-arm development prefixes')


def parse_stat(value):
    # comm may include spaces or parentheses; fields after its last ')' start at 3.
    fields = value[value.rindex(')')+2:].split()
    return dict(state=fields[0], user_ticks=int(fields[11]), system_ticks=int(fields[12]),
                processor=int(fields[36]))


def read_pseudo(path):
    """Bounded raw read: unavailable sysfs clocks may return EAGAIN, not text."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        data = os.read(fd, 65536)
        if len(data) == 65536:
            raise ValueError('Unexpectedly large proc/sys record')
        return data.decode()
    finally:
        os.close(fd)


def runtime_info():
    import cv2
    import numpy as np
    libs = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'openblas' in line.lower() and line.split()[-1].startswith('/')})
    details = []
    for path in libs:
        library = ctypes.CDLL(path)
        row = dict(path=path, sha256=sha(path))
        for label, stems, restype in (
            ('threads', ('openblas_get_num_threads64_', 'openblas_get_num_threads'), ctypes.c_int),
            ('config', ('openblas_get_config64_', 'openblas_get_config'), ctypes.c_char_p),
            ('core', ('openblas_get_corename64_', 'openblas_get_corename'), ctypes.c_char_p)):
            row[label] = None
            for symbol in stems:
                if hasattr(library, symbol):
                    function = getattr(library, symbol)
                    function.argtypes, function.restype = [], restype
                    result = function()
                    row[label] = result.decode() if isinstance(result, bytes) else result
                    break
        details.append(row)
    return dict(numpy=np.__version__, opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(),
        blas=details, affinity=sorted(os.sched_getaffinity(0)), clock_ticks=os.sysconf('SC_CLK_TCK'),
        thread_environment={k: os.environ.get(k) for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GOTO_NUM_THREADS')})


class Trace:
    def __init__(self, bridge=None):
        self.bridge, self.rows, self.local = bridge, [], threading.local()
        self.counts = [0, 0]

    @contextmanager
    def mark(self, frame, name):
        previous = getattr(self.local, 'frame', None)
        self.local.frame = frame
        row = dict(frame=frame, name=name, tid=threading.get_native_id(),
                   start_ns=time.perf_counter_ns(), error=None)
        if self.bridge:
            self.bridge.seaqr_trace_push(f'seaqr30|{frame}|{name}'.encode())
        self.counts[0] += 1
        cpu, process = time.thread_time_ns(), time.process_time_ns()
        try:
            yield
        except BaseException as exc:
            row['error'] = repr(exc)
            raise
        finally:
            row.update(end_ns=time.perf_counter_ns(), thread_cpu_ns=time.thread_time_ns()-cpu,
                       process_cpu_ns=time.process_time_ns()-process)
            self.rows.append(row)
            self.counts[1] += 1
            if self.bridge:
                self.bridge.seaqr_trace_pop()
            self.local.frame = previous

    def wrap(self, function, name):
        def wrapped(*a, **kw):
            with self.mark(getattr(self.local, 'frame', None), name):
                return function(*a, **kw)
        return wrapped

    def validate(self):
        if self.counts[0] != self.counts[1] or any(r['error'] for r in self.rows):
            raise AssertionError('Failed or unbalanced annotations')
        for name in MAJOR:
            rows = [r for r in self.rows if r['name'] == name]
            if [r['frame'] for r in rows] != list(range(128)):
                raise AssertionError('Missing/duplicate stage frames: '+name)
        if len({r['tid'] for r in self.rows if r['name'] in MAJOR}) != 1:
            raise AssertionError('Unexpected consumer migration')


class Telemetry:
    def __init__(self):
        self.stop = threading.Event()
        self.rows, self.errors = [], []
        self.paths = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'))
        self.paths += [Path('/sys/class/devfreq/17000000.gpu/cur_freq')]
        self.paths += sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
        self.types = {str(p.parent): (p.parent/'type').read_text().strip()
                      for p in self.paths if p.name == 'temp'}
        self.worker = threading.Thread(target=self.loop, name='v30-read-only-telemetry', daemon=True)

    def sample(self):
        row = dict(monotonic_ns=time.perf_counter_ns(), realtime_ns=time.time_ns(), threads={}, sensors={})
        for directory in Path('/proc/self/task').iterdir():
            try:
                row['threads'][directory.name] = dict(parse_stat(read_pseudo(directory/'stat')),
                    name=read_pseudo(directory/'comm').strip())
            except FileNotFoundError:
                pass  # A worker can exit between enumeration and reading.
        for path in self.paths:
            try:
                row['sensors'][str(path)] = int(read_pseudo(path).strip())
            except (OSError, ValueError) as exc:
                row['sensors'][str(path)] = dict(error=repr(exc))
        self.rows.append(row)

    def loop(self):
        try:
            while not self.stop.is_set():
                self.sample()
                self.stop.wait(.5)
        except BaseException as exc:
            self.errors.append(repr(exc))

    def close(self):
        self.stop.set()
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            raise RuntimeError('Telemetry worker failed to stop')


def run(clip, arm, output):
    scope(clip, arm)
    if output.exists() or output.with_suffix('.trace30.json').exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(V29))
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    from visible_stage_v24 import StageBinding
    import visible_stage_v24 as stage
    import visible_front_v26 as front
    from tiny_target import motion, visible_shapes_native as shapes
    from motion_reuse_v12 import ReuseMotionV12
    library = V22/'libnvtx_bridge.so'
    if sha(library) != read(V22/'batch_01.json')['bridge_sha256']:
        raise ValueError('Changed existing NVTX bridge')
    bridge = ctypes.CDLL(str(library))
    bridge.seaqr_trace_push.argtypes, bridge.seaqr_trace_push.restype = [ctypes.c_char_p], ctypes.c_int
    bridge.seaqr_trace_pop.argtypes, bridge.seaqr_trace_pop.restype = [], ctypes.c_int
    trace, telemetry = Trace(bridge), Telemetry()
    result = dict(passed=False, error=None, clip=clip, arm=arm, frames=128, traced=True,
        script_sha256=sha(__file__), baseline_freeze_sha256=sha(V29/'freeze.json'),
        bridge_sha256=sha(library), pid=os.getpid(), runtime_before=runtime_info(),
        raw16_accessed=False, defaults_changed=False, numerical_threads_changed=False)
    class MarkedBinding(StageBinding):
        def __init__(self, policy, marker=None):
            if marker is not None:
                raise ValueError('Unexpected existing marker')
            super().__init__(policy, trace.mark)
    try:
        telemetry.worker.start()
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage, 'StageBinding', MarkedBinding))
            def wrap(owner, name, label):
                stack.enter_context(patch.object(owner, name, trace.wrap(getattr(owner, name), label)))
            original = front.ResidentFrontV26.__init__
            def initialize(instance, *a, **kw):
                original(instance, *a, **kw)
                for name in ('seaqr_front_v26_prepare_warp', 'seaqr_resident_patches', 'seaqr_front_v26_finish'):
                    wrap(instance.lib, name, 'native.'+name)
            stack.enter_context(patch.object(front.ResidentFrontV26, '__init__', initialize))
            for owner, name, label in (
                (ReuseMotionV12, 'estimate', 'motion.estimate'),
                (motion, 'fit_global_motion', 'motion.fit'),
                (shapes.NativeShapes, 'consolidate', 'detector.shapes'),
                (front, 'decode_peak_cells', 'detector.decode_peaks'),
                (front, 'pack_learning', 'detector.pack_learning')):
                wrap(owner, name, label)
            baseline.run(argparse.Namespace(clip=clip, arm=arm, frames=128, state_audit=False, output=output))
        trace.validate()
        receipt = read(output.with_suffix('.v29.json'))
        if not receipt['passed'] or receipt['processed_frames'] != 128:
            raise AssertionError('Frozen output gate failed')
        result.update(passed=True, receipt_sha256=sha(output.with_suffix('.v29.json')))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        telemetry.close()
        result.update(events=trace.rows, push_pop_counts=trace.counts, telemetry=telemetry.rows,
            telemetry_errors=telemetry.errors, sensor_types=telemetry.types, runtime_after=runtime_info())
        write(output.with_suffix('.trace30.json'), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.clip, args.arm, args.output)
