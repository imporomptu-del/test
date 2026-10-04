"""Opt-in resident temporal state/support; preserve the CPU FFT point filter.

All calls are synchronous and errors fail closed. Diagnostic snapshots are
explicit downloads outside timed calls. No automatic CPU fallback or tolerance.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import math
from pathlib import Path

import numpy as np

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1]/'build/cuda/libraw16_background_v7.so'


class RawBackgroundCudaError(RuntimeError):
    pass


def cpu_flush_to_zero_enabled():
    # Match the installed Jetson OpenCV/NumPy environment without changing its
    # floating-point control register. Check both a subnormal result and input.
    probe=np.array([0x00800000,0x007fffff],np.uint32).view(np.float32)
    return bool(np.all(np.multiply(probe,np.float32(.5)).view(np.uint32)==0))


def parameters(config):
    if config.spatial_background_radius_px != 4:
        raise ValueError('The opt-in RAW GPU support kernel currently requires radius 4')
    values = np.asarray([config.noise_sigma_floor_dn**2,
        config.background_update_rate, 1.0-config.background_update_rate,
        config.background_outlier_update_rate, 1.0-config.background_outlier_update_rate,
        config.background_update_exclusion_sigma, config.background_outlier_clip_sigma],np.float32)
    if not np.isfinite(values).all() or not np.all(values[[0,1,3,5,6]] >= np.finfo(np.float32).tiny):
        raise ValueError('GPU parameters must remain finite normal float32 values')
    warmup=config.background_warmup_frames
    if isinstance(warmup,bool) or not isinstance(warmup,int) or not 1 <= warmup <= 2**31-1:
        raise ValueError('GPU warmup exceeds int32 bounds')
    required=int(math.ceil(float(np.float32(81*config.minimum_spatial_support_fraction))))
    if not 1 <= required <= 81:
        raise ValueError('Invalid support requirement')
    return values,warmup,required


def checked_shape(shape):
    if (len(shape)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in shape)
            or math.prod(shape)>32000000):
        raise ValueError('GPU stage requires a nonempty 2D shape of at most 32 million pixels')
    return tuple(shape)


class RawBackgroundCuda:
    def __init__(self,config,shape,library=DEFAULT_LIBRARY):
        self.handle=None;self.failed=False;self.count=0
        self.shape=checked_shape(shape)
        values,self.warmup,self.required=parameters(config)
        self.path=Path(library).resolve(strict=True)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.lib=C.CDLL(str(self.path))
        ptr=C.c_void_p;i=C.c_int
        signatures={
            'abi':([],i),'error':([i],C.c_char_p),
            'create':([i,i,i,i,ptr,C.POINTER(ptr)],i),'destroy':([ptr],None),
            'step':([ptr,ptr,ptr,i,i,ptr,ptr],i),'debug':([ptr,ptr,ptr,ptr,ptr,ptr],i),
            'set_state':([ptr,ptr,ptr,ptr],i),'point_probe':([ptr,ptr,ptr,C.c_float,ptr],i)}
        for name,(args,result) in signatures.items():
            fn=getattr(self.lib,'seaqr_raw_background_'+name);fn.argtypes=args;fn.restype=result
        if self.lib.seaqr_raw_background_abi()!=2:
            raise RawBackgroundCudaError('Unsupported RAW background CUDA ABI')
        if not cpu_flush_to_zero_enabled():
            raise RawBackgroundCudaError('GPU ABI 2 requires the validated Jetson CPU flush-to-zero mode; no fallback')
        handle=ptr()
        self._check(self.lib.seaqr_raw_background_create(*self.shape,self.warmup,self.required,
                                                        values.ctypes.data,C.byref(handle)))
        if not handle.value:
            raise RawBackgroundCudaError('CUDA returned a null workspace')
        self.handle=handle

    def _check(self,code):
        if code:
            self.failed=True
            detail=self.lib.seaqr_raw_background_error(code).decode('utf-8',errors='replace')
            raise RawBackgroundCudaError(f'RAW background CUDA failed ({code}: {detail}); no fallback')

    def _live(self):
        if not self.handle or self.failed:
            raise RawBackgroundCudaError('GPU workspace is closed or failed; cannot continue')

    def close(self):
        if self.handle:
            self.lib.seaqr_raw_background_destroy(self.handle);self.handle=None

    def __del__(self):
        try:self.close()
        except Exception:pass

    def reset(self):
        self._live();self.count=0

    def step(self,image,input_valid):
        self._live()
        if image.dtype!=np.float32 or image.shape!=self.shape or not np.isfinite(image).all():
            raise ValueError('GPU image must be finite float32 with the initialized shape')
        if input_valid.shape!=self.shape or input_valid.dtype!=np.bool_:
            raise ValueError('GPU input validity must be a matching boolean mask')
        image=np.ascontiguousarray(image);valid=np.ascontiguousarray(input_valid,dtype=np.uint8)
        white=np.empty(self.shape,np.float32);mask=np.empty(self.shape,np.bool_)
        reset=self.count==0;ready=self.count>=self.warmup
        self._check(self.lib.seaqr_raw_background_step(self.handle,image.ctypes.data,valid.ctypes.data,
                     int(reset),int(ready),white.ctypes.data,mask.ctypes.data))
        self.count+=1
        return None if reset else (white,mask,ready)

    def debug_state(self):
        self._live()
        if self.count==0:raise RawBackgroundCudaError('No initialized background state')
        arrays={name:np.empty(self.shape,dtype) for name,dtype in
                (('location',np.float32),('variance',np.float32),('history',np.uint16),
                 ('whitened',np.float32),('detection_valid',np.bool_))}
        self._check(self.lib.seaqr_raw_background_debug(self.handle,*[a.ctypes.data for a in arrays.values()]))
        return arrays

    def set_state_for_test(self,location,variance,history):
        self._live()
        if self.count==0:raise ValueError('Initialize before setting test state')
        for array,dtype in ((location,np.float32),(variance,np.float32),(history,np.uint16)):
            if array.shape!=self.shape or array.dtype!=dtype or not array.flags.c_contiguous:
                raise ValueError('Test-state shape/dtype/contiguity mismatch')
        if not np.isfinite(location).all() or not np.isfinite(variance).all() or np.any(variance<0):
            raise ValueError('Invalid test state')
        self._check(self.lib.seaqr_raw_background_set_state(self.handle,location.ctypes.data,
                                                         variance.ctypes.data,history.ctypes.data))

    def point_filter_probe(self,white,kernel,normalizer):
        """Diagnostic only. Cannot corrupt an active temporal model's white buffer."""
        self._live()
        if self.count:raise ValueError('Use a separate fresh workspace for the point-filter probe')
        if white.shape!=self.shape or white.dtype!=np.float32 or not np.isfinite(white).all():
            raise ValueError('Invalid probe input')
        if kernel.shape!=(9,9) or kernel.dtype!=np.float32 or not np.isfinite(kernel).all():
            raise ValueError('Invalid probe kernel')
        if not math.isfinite(normalizer) or normalizer<=0:raise ValueError('Invalid normalizer')
        white=np.ascontiguousarray(white);kernel=np.ascontiguousarray(kernel)
        result=np.empty(self.shape,np.float32)
        self._check(self.lib.seaqr_raw_background_point_probe(self.handle,white.ctypes.data,
                     kernel.ctypes.data,float(np.float32(normalizer)),result.ctypes.data))
        return result
