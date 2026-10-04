"""Explicit CUDA float32 median; failure never silently selects a CPU backend."""
import ctypes
from pathlib import Path
import numpy as np


class CudaMedian5:
    def __init__(self,path):
        self.path=Path(path).resolve(strict=True)
        self.lib=ctypes.CDLL(str(self.path))
        self.lib.seaqr_create.restype=ctypes.c_void_p
        self.lib.seaqr_destroy.argtypes=[ctypes.c_void_p]
        self.lib.seaqr_median.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
        self.lib.seaqr_median.restype=ctypes.c_int
        self.handle=self.lib.seaqr_create()
        if not self.handle:raise RuntimeError('CUDA workspace allocation failed')

    def __call__(self,image):
        if image.ndim!=2 or image.dtype!=np.float32 or min(image.shape)<1 or not np.isfinite(image).all():
            raise ValueError('CUDA median requires finite nonempty 2D float32 input')
        source=np.ascontiguousarray(image);result=np.empty_like(source)
        code=self.lib.seaqr_median(self.handle,source.ctypes.data,result.ctypes.data,*source.shape)
        if code:raise RuntimeError(f'CUDA median failed ({code}); no CPU fallback')
        return result

    def close(self):
        if self.handle:
            self.lib.seaqr_destroy(self.handle);self.handle=None
