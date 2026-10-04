"""Execution-only adapter: original algorithms, worker-owned motion, two warp leases."""
from contextlib import ExitStack
from copy import deepcopy
import threading
import time
from unittest.mock import patch
from frame_lookahead_v23 import PreparedFrames


class Binding:
    def __init__(self, mode, marker=None):
        if mode not in ('reference', 'serial', 'overlap'):
            raise ValueError('Unknown execution mode')
        self.mode, self.marker = mode, marker
        self.consumer_thread_id = threading.get_ident()
        self.reader = self.engine = self.current = self.job = None
        self.consumed_motion = False
        self.rows = {}

    def complete(self):
        if self.current is not None:
            self.rows[self.current.index]['consumer_complete_ns'] = time.perf_counter_ns()
            self.current = self.job = None

    def snapshot(self):
        return dict(mode=self.mode, frames=[self.rows[i] for i in sorted(self.rows)],
                    consumer_thread_id=self.consumer_thread_id,
                    engine=None if self.engine is None else dict(self.engine.stats),
                    timing_semantics='ready_ns is grayscale completion; consumer_complete_ns is next '
                    'consumer read/pre-close after journaling. Includes queue residence, not camera '
                    'acquisition. Existing motion_and_warp is proxy handoff time in prepared modes; '
                    'use prepare_start/end for actual worker motion work. decode_wait remains decoder '
                    'worker wait; queue_wait_ms is separate. Never sum overlapped stages.',
                    ownership='decode retains at most one raw gray frame ahead of its new motion-worker '
                    'consumer; additionally up to two leased/preparing motion slots, including the '
                    'main current frame. Detector, learning protection, association and journals '
                    'remain ordered on the main consumer. One motion estimator; two warp buffers.')

    def mark(self, frame, stage):
        if self.marker is not None:
            return self.marker(frame, stage)
        from contextlib import nullcontext
        return nullcontext()


def install(stack: ExitStack, binding):
    from tiny_target import visible_baseline as visible, visible_decode as decode
    from tiny_target.visible_warp_exact import CudaCubicTranslation
    OriginalReader, OriginalMotion = decode.VisibleFrameReader, visible.PvaMotion

    class Reader:
        def __init__(self, *a, **kw):
            if binding.reader is not None:
                raise RuntimeError('Only one source per execution adapter')
            self.inner = OriginalReader(*a, **kw)
            binding.reader = self
            original_decode = self.inner._decode
            def decoded():
                frame = original_decode()
                if frame is not None:
                    binding.rows[frame.index] = dict(frame=frame.index, ready_ns=time.perf_counter_ns())
                return frame
            self.inner._decode = decoded

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *a):
            self.close()

        def start(self):
            self.inner.start()
            if binding.engine is not None:
                binding.engine.start()

        def read(self):
            binding.complete()
            begin = time.perf_counter_ns()
            if binding.engine is None:
                frame, wait_ms = self.inner.read()
                binding.job = None
            else:
                binding.job = binding.engine.read()
                frame = None if binding.job is None else binding.job.frame
                wait_ms = 0.0 if binding.job is None else binding.job.read_wait_ms
            if frame is not None:
                binding.current = frame
                binding.consumed_motion = False
                row = binding.rows[frame.index]
                row.update(consumer_received_ns=time.perf_counter_ns(),
                           queue_wait_ms=(time.perf_counter_ns()-begin)/1e6)
                if binding.job is not None:
                    row.update(slot=binding.job.slot, prepare_start_ns=binding.job.prepare_start_ns,
                               prepare_end_ns=binding.job.prepare_end_ns)
            return frame, wait_ms

        def close(self):
            binding.complete()
            if binding.engine is not None:
                binding.engine.close()
            else:
                self.inner.close()

    class Motion:
        def __init__(self, *a, **kw):
            if binding.reader is None or binding.engine is not None:
                raise RuntimeError('Invalid reader/motion construction order')
            fps = binding.reader.fps

            class Processor:
                def __init__(self):
                    self.real = None
                    self.warps = []
                    try:
                        # Preserve partial construction so a conformance or VPI
                        # failure cannot orphan resources on their owner thread.
                        self.real = OriginalMotion.__new__(OriginalMotion)
                        OriginalMotion.__init__(self.real, *a, **kw)
                        if self.real.execution != 'cuda_cubic_resident' or self.real.cuda_warp is None:
                            raise ValueError('Only frozen resident visible CUDA execution')
                        self.warps.append(self.real.cuda_warp)
                        second = CudaCubicTranslation.__new__(CudaCubicTranslation)
                        # Frozen close() can safely inspect these even if the
                        # second wrapper's library/table initialization fails.
                        second.handle, second.shape, second.generation = None, None, 0
                        self.warps.append(second)
                        CudaCubicTranslation.__init__(second, self.real.cuda_warp.lib._name)
                        self.metadata = dict(conformance=deepcopy(self.real.conformance))
                    except BaseException:
                        self.close()
                        raise

                def prepare(self, frame, slot):
                    self.real.cuda_warp = self.warps[slot]
                    with binding.mark(frame.index, 'motion_worker'):
                        value = self.real.update(frame.gray, frame.index, round(frame.index/fps*1e9))
                    # Buffer-backed image/valid retain their slot lease. Small
                    # mapping and metadata are independent of future updates.
                    image, valid, matrix, segment, meta = value
                    return image, valid, matrix.copy(), segment, deepcopy(meta)

                def close(self):
                    errors = []
                    warps = list(self.warps)
                    partial = getattr(self.real, 'cuda_warp', None)
                    if partial is not None and all(partial is not w for w in warps):
                        warps.append(partial)
                    for warp in warps:
                        try:
                            warp.close()
                        except BaseException as exc:
                            errors.append(exc)
                    estimator = getattr(self.real, 'estimator', None)
                    if estimator is not None:
                        try:
                            estimator.close()
                        except BaseException as exc:
                            errors.append(exc)
                    if errors:
                        raise errors[0]

            binding.engine = PreparedFrames(binding.reader.inner.read, Processor,
                capacity=1 if binding.mode == 'serial' else 2,
                shutdown_source=binding.reader.inner.close, timeout=10.0)
            self.conformance = binding.engine.metadata['conformance']

        def update(self, gray, frame_index, timestamp_ns):
            frame, job = binding.current, binding.job
            if (frame is None or job is None or binding.consumed_motion or frame.gray is not gray
                    or frame.index != frame_index or timestamp_ns != round(frame_index/binding.reader.fps*1e9)):
                raise RuntimeError('Stale, mismatched or already consumed prepared frame')
            binding.consumed_motion = True
            return job.result

        def close(self):
            binding.reader.close()

    stack.enter_context(patch.object(visible, 'VisibleFrameReader', Reader))
    stack.enter_context(patch.object(decode, 'VisibleFrameReader', Reader))
    if binding.mode != 'reference':
        stack.enter_context(patch.object(visible, 'PvaMotion', Motion))
    else:
        original = OriginalMotion.update
        def reference_motion(self, gray, frame_index, timestamp_ns, _original=original):
            row = binding.rows[frame_index]
            row['prepare_start_ns'] = time.perf_counter_ns()
            try:
                with binding.mark(frame_index, 'motion_reference'):
                    return _original(self, gray, frame_index, timestamp_ns)
            finally:
                row['prepare_end_ns'] = time.perf_counter_ns()
        stack.enter_context(patch.object(OriginalMotion, 'update', reference_motion))
    for cls, stage in ((visible.VisiblePointDetector, 'detector'), (visible.VisibleTracks, 'tracking')):
        original = cls.update
        def timed(self, *a, _original=original, _stage=stage, **kw):
            frame = binding.current.index
            row = binding.rows[frame]
            row[_stage+'_start_ns'] = time.perf_counter_ns()
            try:
                with binding.mark(frame, _stage):
                    return _original(self, *a, **kw)
            finally:
                row[_stage+'_end_ns'] = time.perf_counter_ns()
        stack.enter_context(patch.object(cls, 'update', timed))
