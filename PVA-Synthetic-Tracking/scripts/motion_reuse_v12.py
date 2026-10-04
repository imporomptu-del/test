"""Research-only single-stream adjacent-frame preparation reuse; no default hook."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import textwrap
import threading

from raw16_speed_v8_common import FeaturePixelsCache
from tiny_target.motion import pva_pyrlk as pva

MOTION_SHA='7b96fef337350500835a64be3f43e32651c52ffb18a637a1b35ed0d4c627949f'
REFERENCE_PIXELS=pva._feature_pixels


def generated_method():
    source=Path(pva.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest()!=MOTION_SHA:
        raise ValueError('Frozen estimator source changed')
    method=textwrap.dedent(inspect.getsource(pva.PvaPyrLkMotionEstimator.estimate))
    method=method.replace('def estimate(', 'def _estimate_v12(',1)
    for name in ('previous','current'):
        old=f'_feature_pixels({name}, config.feature_intensity_mapping)'
        if method.count(old)!=1:raise ValueError('Feature preparation anchor changed')
        method=method.replace(old,f'self._pixel_cache({name}, config.feature_intensity_mapping)')
    first=method.index('        image_format = vpi.Format.U16')
    end=method.index('        completed = time.perf_counter_ns()',first)
    method=method[:first]+'''        previous_image, current_image, previous_motion, current_motion, rescale_backend = self._prepare_images(
            previous, current, previous_pixels, current_pixels, motion_size, full_size, uses_u16)
        submitted = time.perf_counter_ns()
        if motion_size != full_size:
            self._stream.sync()
'''+method[end:]
    first=method.index('        previous_pyramid = previous_motion.gaussian_pyramid(')
    end=method.index('        submitted = time.perf_counter_ns()',first)
    method=method[:first]+'''        previous_pyramid, current_pyramid = self._prepare_pyramids(
            previous_motion, current_motion, pyramid_backend)
'''+method[end:]
    return method


@dataclass
class Prepared:
    frame: object
    pixels: object
    image: object
    motion: object
    pyramid: object
    config: object
    stream: object


class ReuseMotionV12(pva.PvaPyrLkMotionEstimator):
    def __init__(self,config):
        super().__init__(config)
        self.owner=threading.get_ident();self.owner_stream=self._stream
        self.closed=False;self.failed=False;self._cached=None;self._pending=None;self._active_hit=False
        self.hits=0;self.misses=0;self.resets=0
        self._pixel_cache=FeaturePixelsCache(REFERENCE_PIXELS)

    def _live(self):
        if self.closed or self.failed or threading.get_ident()!=self.owner or self._stream is not self.owner_stream:
            raise RuntimeError('Preparation cache closed or used on another thread/stream')

    def _eligible(self,previous,current):
        entry=self._cached
        return bool(entry is not None and entry.frame is previous
            and entry.config is self.config and entry.stream is self._stream
            and previous.source_id==current.source_id
            and current.frame_index==previous.frame_index+1
            and current.timestamp_ns>previous.timestamp_ns
            and previous.shape==current.shape and previous.bit_depth==current.bit_depth
            and not previous.discontinuities and not current.discontinuities
            and ((previous.sequence is None and current.sequence is None)
                 or (previous.sequence is not None and current.sequence==previous.sequence+1)))

    def _prepare_images(self,previous,current,previous_pixels,current_pixels,motion_size,full_size,uses_u16):
        vpi=self._vpi;image_format=vpi.Format.U16 if uses_u16 else vpi.Format.U8
        self._active_hit=self._eligible(previous,current)
        if self._active_hit:
            previous_image=self._cached.image;previous_motion=self._cached.motion;self.hits+=1
        else:
            self._cached=None;self.misses+=1
            previous_image=vpi.asimage(previous_pixels,image_format)
            previous_motion=(previous_image.rescale(motion_size,backend=vpi.Backend.CUDA,stream=self._stream)
                if motion_size!=full_size else previous_image)
        current_image=vpi.asimage(current_pixels,image_format)
        current_motion=(current_image.rescale(motion_size,backend=vpi.Backend.CUDA,stream=self._stream)
            if motion_size!=full_size else current_image)
        self._pending=Prepared(current,current_pixels,current_image,current_motion,None,self.config,self._stream)
        return previous_image,current_image,previous_motion,current_motion,('CUDA' if motion_size!=full_size else 'none')

    def _prepare_pyramids(self,previous_motion,current_motion,backend):
        config=self.config
        previous=(self._cached.pyramid if self._active_hit else previous_motion.gaussian_pyramid(
            config.pyramid_levels,config.pyramid_scale,backend=backend,stream=self._stream))
        current=current_motion.gaussian_pyramid(config.pyramid_levels,config.pyramid_scale,backend=backend,stream=self._stream)
        self._pending.pyramid=current
        return previous,current

    def estimate(self,previous,current):
        self._live();self._pending=None
        try:
            result=_ESTIMATE(self,previous,current)
            if self._pending is None or self._pending.pyramid is None:
                raise RuntimeError('Preparation was not completed')
            self._cached=self._pending;self._pending=None
            return result
        except BaseException as exc:
            # No stale result may be reused after unavailable or failed motion.
            expected_unavailable=isinstance(exc,pva.PvaMotionError) and any(x in str(exc)
                for x in ('zero features','No finite in-bounds'))
            self.failed=not expected_unavailable
            try:self._stream.sync()
            except BaseException:self.failed=True
            self._cached=None;self._pending=None;self._active_hit=False;self._pixel_cache.clear()
            raise

    def reset(self):
        self._live();self._stream.sync()
        self._cached=None;self._pending=None;self._active_hit=False
        self._pixel_cache.clear();self.resets+=1

    def close(self):
        if self.closed:return
        if threading.get_ident()!=self.owner or self._stream is not self.owner_stream:
            raise RuntimeError('Close on owning thread/stream')
        try:self._stream.sync()
        finally:
            self._cached=None;self._pending=None;self._active_hit=False;self._pixel_cache.clear();self.closed=True


_scope=dict(vars(pva))
exec(compile(generated_method(),__file__+':generated-estimate','exec'),_scope)
_ESTIMATE=_scope['_estimate_v12']
