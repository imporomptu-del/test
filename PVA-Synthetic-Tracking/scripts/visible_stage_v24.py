"""Execution-only v24 policy layered over the immutable v23 ownership adapter."""
import threading
import time
from unittest.mock import patch
from visible_overlap_v23 import Binding, install as install_ownership
from stage_control_v24 import Admission, GpuRelease, StageCancelled


class StageBinding(Binding):
    def __init__(self, policy, marker=None):
        if policy not in ('reference','bounded','staged'):
            raise ValueError('Unknown stage policy')
        super().__init__('reference' if policy=='reference' else 'overlap', marker)
        self.policy = policy
        self.admission, self.gpu_release = Admission(), GpuRelease()
        self.active = threading.local()

    def complete(self):
        frame = self.current
        super().complete()
        if frame is not None:
            self.admission.release(frame.index)

    def cancel(self):
        # Wake stage/admission waits BEFORE joining either owning worker.
        self.gpu_release.stop()
        self.admission.stop()

    def snapshot(self):
        result = super().snapshot()
        result.update(policy=self.policy,admission=self.admission.snapshot(),gpu_release=self.gpu_release.snapshot())
        result['ownership'] = ('One shared capacity of two across native decode-in-flight, decoded queue, '
            'preparation and main consumer. One motion owner, two warp buffers, ordered detector/tracker. '
            'Frozen one-slot decoder remains a substage, not an independent extra admission allowance.')
        result['timing_semantics'] += (' request_ns precedes admission wait at the decoder request boundary; '
            'request-to-complete includes admission, decode, preparation and queues, not sensor acquisition '
            'or an upstream camera/codec backlog. cpu_prepare includes existing VPI proxy CUDA work. '
            'warp_gpu denotes a host span including copies, kernels and waits, not device execution alone.')
        return result


def install(stack, binding):
    from tiny_target import visible_baseline as visible, visible_decode as decode, visible_warp_exact as warp
    OriginalWarp, OriginalMotion = warp.CudaCubicTranslation, visible.PvaMotion
    original_motion_update = OriginalMotion.update

    class GatedWarp(OriginalWarp):
        def __call__(self, *a, **kw):
            frame = getattr(binding.active, 'frame', None)
            if frame is None:  # Constructor conformance is unchanged and outside the frame pipeline.
                return super().__call__(*a, **kw)
            if binding.active.warp_called:
                raise RuntimeError('More than one full-resolution warp in a motion update')
            binding.active.warp_called = True
            row = binding.rows[frame]
            row['cpu_prepare_end_ns'] = time.perf_counter_ns()
            binding.active.cpu_context.__exit__(None,None,None)
            binding.active.cpu_context = None
            row['warp_wait_start_ns'] = time.perf_counter_ns()
            with binding.mark(frame,'warp_wait'):
                if binding.policy == 'staged':
                    binding.gpu_release.wait(frame)
            row['warp_gpu_start_ns'] = time.perf_counter_ns()
            try:
                with binding.mark(frame,'warp_gpu'):
                    return super().__call__(*a, **kw)
            finally:
                row['warp_gpu_end_ns'] = time.perf_counter_ns()

    def motion_update(self, gray, frame_index, timestamp_ns):
        if getattr(binding.active,'frame',None) is not None:
            raise RuntimeError('Reentrant motion preparation')
        binding.active.frame, binding.active.warp_called = frame_index, False
        binding.rows[frame_index]['cpu_prepare_start_ns'] = time.perf_counter_ns()
        binding.active.cpu_context = binding.mark(frame_index,'cpu_prepare')
        binding.active.cpu_context.__enter__()
        try:
            result = original_motion_update(self,gray,frame_index,timestamp_ns)
            if not binding.active.warp_called:
                raise RuntimeError('Frozen full-resolution warp boundary was not exercised')
            return result
        finally:
            if binding.active.cpu_context is not None:
                binding.active.cpu_context.__exit__(None,None,None)
                binding.active.cpu_context = None
            binding.active.frame = None

    stack.enter_context(patch.object(warp,'CudaCubicTranslation',GatedWarp))
    stack.enter_context(patch.object(OriginalMotion,'update',motion_update))
    install_ownership(stack,binding)
    WrappedReader = decode.VisibleFrameReader
    original_init, original_close = WrappedReader.__init__, WrappedReader.close

    def reader_init(reader,*a,**kw):
        original_init(reader,*a,**kw)
        original_decode = reader.inner._decode
        def admitted_decode():
            index = reader.inner.decoded
            maximum = getattr(reader.inner,'max_frames',None)
            if maximum is not None and index >= maximum:
                return original_decode()  # Known prefix EOF: no real input request.
            try:
                event = binding.admission.acquire(index)
            except StageCancelled:
                return None  # Shutdown only; completed-run gates require every expected frame.
            try:
                frame = original_decode()
            except BaseException:
                binding.admission.release(index,'error')
                raise
            if frame is None:
                binding.admission.release(index,'eof')
            else:
                if frame.index != index:
                    binding.admission.release(index,'error')
                    raise ValueError('Decoder order differs from admitted input')
                binding.rows[index].update(request_ns=event['request_ns'],admitted_ns=event['admitted_ns'])
            return frame
        reader.inner._decode = admitted_decode

    def reader_close(reader):
        binding.cancel()
        return original_close(reader)

    original_detector = visible.VisiblePointDetector.update
    def detector_update(self,*a,**kw):
        frame = binding.current.index
        result = original_detector(self,*a,**kw)
        binding.rows[frame]['gpu_release_ns'] = binding.gpu_release.detector_complete(frame)
        return result

    stack.enter_context(patch.object(WrappedReader,'__init__',reader_init))
    stack.enter_context(patch.object(WrappedReader,'close',reader_close))
    stack.enter_context(patch.object(visible.VisiblePointDetector,'update',detector_update))
