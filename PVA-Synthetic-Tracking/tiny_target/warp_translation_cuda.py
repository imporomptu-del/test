"""Explicit exact-translation experiment, not OpenCV's CUDA cubic backend."""
from __future__ import annotations
import ctypes as C
import hashlib
from pathlib import Path
import platform
import threading
import numpy as np

DEFAULT_LIBRARY=Path(__file__).resolve().parents[1]/'build/exact_v9/libwarp_translation_v9.so'
OPENCV_BUILD_SHA256='aaaa7ac0485021190968244469b00ede7fb18a7104630a74cbf0641388604f81'
TABLE_SHA256='d44c52b06e81bbad059b3be1fb84426bb2dca7df4d103a80da2f87683ff30cf9'


def translation_inverse(matrix):
    import cv2
    a=np.asarray(matrix)
    if a.shape!=(3,3) or a.dtype!=np.float64 or not np.isfinite(a).all():
        raise ValueError('Finite float64 3x3 transform required')
    probe=a.copy();probe[:2,2]=0
    if not np.array_equal(probe,np.eye(3)) or np.max(np.abs(a[:2,2]))>1000000:
        raise ValueError('Only bounded exact translations are supported; no approximation')
    ok,inv=cv2.invert(a)
    if not ok:raise ValueError('Noninvertible transform')
    return inv


def cubic_table():
    """Extract all exact float32 reference weights through isolated basis probes."""
    import cv2
    phase=np.arange(32,dtype=np.float32)/np.float32(32)+np.float32(1)
    mx=np.tile(phase,(32,1));my=mx.T.copy()
    table=np.empty((32,32,16),np.float32)
    for k in range(16):
        basis=np.zeros((4,4),np.float32);basis.flat[k]=1
        table[:,:,k]=cv2.remap(basis,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return table


class WarpTranslationCuda:
    def __init__(self,shape,library=DEFAULT_LIBRARY):
        import cv2
        from .raw_background_cuda import checked_shape,cpu_flush_to_zero_enabled
        self.handle=None;self.failed=False;self.thread=threading.get_ident()
        self.shape=checked_shape(shape)
        if max(self.shape)>=32767:raise ValueError('OpenCV short-coordinate limits exceeded')
        if platform.machine()!='aarch64' or cv2.__version__!='4.10.0' or not cpu_flush_to_zero_enabled():
            raise RuntimeError('Requires the validated Jetson OpenCV 4.10 NEON/FTZ environment')
        if hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()!=OPENCV_BUILD_SHA256:
            raise RuntimeError('Unvalidated OpenCV build; exact warp disabled')
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)))
        ptr=C.c_void_p;i=C.c_int
        for name,args,res in (
            ('abi',[],i),('error',[i],C.c_char_p),('create',[i,i,ptr,C.POINTER(ptr)],i),
            ('run',[ptr,ptr,ptr,C.c_double,C.c_double,ptr,ptr],i),('destroy',[ptr],None)):
            fn=getattr(self.lib,'seaqr_warp_v9_'+name);fn.argtypes=args;fn.restype=res
        if self.lib.seaqr_warp_v9_abi()!=1:raise RuntimeError('Unknown exact warp ABI')
        table=cubic_table();self.table_sha256=hashlib.sha256(table.tobytes()).hexdigest()
        if self.table_sha256!=TABLE_SHA256:raise RuntimeError('Reference interpolation table changed')
        handle=ptr();self._check(self.lib.seaqr_warp_v9_create(*self.shape,table.ctypes.data,C.byref(handle)))
        if not handle.value:raise RuntimeError('Null exact warp workspace')
        self.handle=handle

    def _check(self,status):
        if status:
            self.failed=True
            raise RuntimeError('Exact warp CUDA failed: '+self.lib.seaqr_warp_v9_error(status).decode()+'; no fallback')

    def __call__(self,image,mask,matrix):
        if not self.handle or self.failed or threading.get_ident()!=self.thread:
            raise RuntimeError('Warp is closed, failed or used from another thread')
        inverse=translation_inverse(matrix)
        if image.shape!=self.shape or image.dtype!=np.float32 or not np.isfinite(image).all():
            raise ValueError('Matching finite float32 image required')
        if mask.shape!=self.shape or mask.dtype!=np.uint8 or np.any(mask>1):
            raise ValueError('Matching uint8 binary source mask required')
        image=np.ascontiguousarray(image);mask=np.ascontiguousarray(mask)
        out=np.empty(self.shape,np.float32);valid=np.empty(self.shape,np.uint8)
        self._check(self.lib.seaqr_warp_v9_run(self.handle,image.ctypes.data,mask.ctypes.data,
            inverse[0,2],inverse[1,2],out.ctypes.data,valid.ctypes.data))
        if not np.isfinite(out).all():
            self.failed=True;raise RuntimeError('Nonfinite exact warp output')
        return out,valid

    def close(self):
        if self.handle:
            if threading.get_ident()!=self.thread:raise RuntimeError('Close warp on owning thread')
            self.lib.seaqr_warp_v9_destroy(self.handle);self.handle=None

    def __del__(self):
        try:self.close()
        except Exception:pass
