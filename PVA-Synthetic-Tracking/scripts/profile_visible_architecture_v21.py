"""Bounded host timeline of frozen v20; no device timeline or clean FPS claim."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

V20 = Path('/tmp/seaqr_visible_speed_v20_XUf1LR')
V17 = Path('/tmp/seaqr_visible_speed_v17_EER6lm')
V13 = Path('/tmp/seaqr_video_v13_IyK7eQ')
RUNTIME = Path('/tmp/seaqr_exact_v9_rS2LFx')


class HostTrace:
    def __init__(self):
        self.rows = []
        self.local = threading.local()
        self.origin = time.perf_counter_ns()
        self.frame = None

    def wrap(self, function, name, frame_entry=False):
        def traced(*args, **kwargs):
            if frame_entry:
                self.frame = args[2] if len(args) > 2 else kwargs['frame_index']
            frames = getattr(self.local, 'stack', None)
            if frames is None:
                frames = self.local.stack = []
            parent = frames[-1] if frames else None
            row = dict(name=name, frame=self.frame, thread=threading.get_ident(),
                       start_ns=time.perf_counter_ns()-self.origin,
                       parent=parent['id'] if parent else None, id=len(self.rows),
                       children_ns=0, error=None)
            self.rows.append(row)
            frames.append(row)
            cpu_start = time.thread_time_ns()
            try:
                return function(*args, **kwargs)
            except BaseException as exc:
                row['error'] = repr(exc)
                raise
            finally:
                row['thread_cpu_ns'] = time.thread_time_ns()-cpu_start
                row['duration_ns'] = time.perf_counter_ns()-self.origin-row['start_ns']
                row['exclusive_host_ns'] = row['duration_ns']-row['children_ns']
                frames.pop()
                if parent:
                    parent['children_ns'] += row['duration_ns']
        return traced


def run(clip, output):
    if clip not in ('0126', '0082'):
        raise ValueError('Only two authorized development AVI prefixes')
    receipt = output.with_suffix('.host_trace.json')
    if output.exists() or receipt.exists():
        raise FileExistsError(output)
    sys.path[:0] = [str(V20), str(V17), str(V13), str(RUNTIME), str(RUNTIME/'scripts')]
    import run_visible_v20
    from tiny_target import visible_baseline as visible, visible_resident as resident
    from tiny_target import visible_warp_exact as warp, visible_shapes_native as shapes
    from tiny_target import motion
    from learning_mask_v17 import LearningMaskV17
    from motion_reuse_v12 import ReuseMotionV12

    trace = HostTrace()
    result = dict(passed=False, error=None, clip=clip, frames=128,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  raw16_accessed=False, defaults_changed=False,
                  warning='Host intervals only. Native calls combine GPU work, copies and waits. '
                          'Exclusive host time excludes measured children, not unmeasured GPU waits. '
                          'Thread CPU time excludes other CPU workers. No device utilization, '
                          'bandwidth, clean throughput or camera-to-alert latency claim.')
    try:
        with ExitStack() as stack:
            def wrap(owner, method, name, frame_entry=False):
                stack.enter_context(patch.object(owner, method,
                    trace.wrap(getattr(owner, method), name, frame_entry)))
            old_init = resident.VisibleCudaResident.__init__
            def init(instance, *a, **kw):
                old_init(instance, *a, **kw)
                for method in ('select', 'patches', 'finish'):
                    wrap(instance.lib, 'seaqr_resident_'+method, 'native.'+method)
            stack.enter_context(patch.object(resident.VisibleCudaResident, '__init__', init))
            wrap(visible.PvaMotion, 'update', 'motion_and_warp', True)
            for owner, method, name in (
                (ReuseMotionV12, 'estimate', 'motion.estimate'),
                (motion, 'fit_global_motion', 'motion.global_fit'),
                (warp.CudaCubicTranslation, '__call__', 'warp.total'),
                (warp.CudaCubicTranslation, 'gaussian', 'warp.gaussian'),
                (warp.CudaWarpFrame, 'prepare', 'native.prepare_warp'),
                (resident.VisibleCudaResident, 'update', 'detector.total'),
                (resident, 'tile_noise_statistics', 'detector.noise'),
                (resident, 'decode_peak_cells', 'detector.decode_peaks'),
                (shapes.NativeShapes, 'consolidate', 'detector.shapes'),
                (LearningMaskV17, '__call__', 'detector.learning_mask'),
                (visible.VisibleTracks, 'update', 'tracks.total'),
                (visible.VisibleTracks, 'learning_centers', 'tracks.learning_centers'),
            ):
                wrap(owner, method, name)
            run_visible_v20.run(argparse.Namespace(clip=clip, mode='candidate',
                                                   frames=128, output=output))
        baseline = json.loads(output.with_suffix('.v20.json').read_text())
        if not baseline['passed'] or baseline['processed_frames'] != 128:
            raise AssertionError('Frozen v20 complete parity gate failed')
        for name in ('motion_and_warp', 'detector.total', 'tracks.total'):
            rows = [r for r in trace.rows if r['name'] == name]
            if len(rows) != 128 or [r['frame'] for r in rows] != list(range(128)):
                raise AssertionError('Missing/duplicate frame events: '+name)
        result['passed'] = True
        result['v20_receipt_sha256'] = hashlib.sha256(output.with_suffix('.v20.json').read_bytes()).hexdigest()
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['events'] = trace.rows
        with receipt.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.clip, args.output)
