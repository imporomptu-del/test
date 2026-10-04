"""Thread-confined experimental point filter; no default dispatch or fallback."""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import threading
import numpy as np

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1]/'build/point_v8/libpoint_filter_v8.so'


class PointFilterCuda:
    def __init__(self, shape, kernel, normalizer, library=DEFAULT_LIBRARY):
        from .raw_background_cuda import checked_shape, cpu_flush_to_zero_enabled
        self.handle = None
        self.failed = False
        self.thread = threading.get_ident()
        self.shape = checked_shape(shape)
        if kernel.shape != (9, 9) or kernel.dtype != np.float32 or not np.isfinite(kernel).all():
            raise ValueError('Finite float32 9x9 kernel required')
        self.normalizer = np.float32(normalizer)
        if not np.isfinite(self.normalizer) or self.normalizer <= np.finfo(np.float32).tiny:
            raise ValueError('Positive normal float32 normalizer required')
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        ptr, integer = C.c_void_p, C.c_int
        for name, args, result in (
            ('abi', [], integer), ('error', [integer], C.c_char_p),
            ('create', [integer, integer, ptr, C.c_float, C.POINTER(ptr)], integer),
            ('run', [ptr, ptr, ptr], integer), ('destroy', [ptr], None),
        ):
            fn = getattr(self.lib, 'seaqr_point_v8_'+name)
            fn.argtypes, fn.restype = args, result
        if self.lib.seaqr_point_v8_abi() != 1 or not cpu_flush_to_zero_enabled():
            raise RuntimeError('Requires point ABI 1 and validated Jetson FTZ mode')
        kernel = np.ascontiguousarray(kernel)
        handle = ptr()
        self._check(self.lib.seaqr_point_v8_create(*self.shape, kernel.ctypes.data,
                                                 float(self.normalizer), C.byref(handle)))
        if not handle.value:
            raise RuntimeError('Null point-filter workspace')
        self.handle = handle

    def _check(self, status):
        if status:
            self.failed = True
            detail = self.lib.seaqr_point_v8_error(status).decode(errors='replace')
            raise RuntimeError(f'Point-filter CUDA failed: {status}: {detail}; no fallback')

    def __call__(self, image):
        if not self.handle or self.failed or threading.get_ident() != self.thread:
            raise RuntimeError('Point filter is closed, failed or used from another thread')
        if image.shape != self.shape or image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError('Matching finite float32 image required')
        image = np.ascontiguousarray(image)
        result = np.empty(self.shape, np.float32)
        self._check(self.lib.seaqr_point_v8_run(self.handle, image.ctypes.data, result.ctypes.data))
        if not np.isfinite(result).all():
            self.failed = True
            raise RuntimeError('Nonfinite point-filter output; no fallback')
        return result

    def close(self):
        if self.handle:
            if threading.get_ident() != self.thread:
                raise RuntimeError('Close point filter on its owning thread')
            self.lib.seaqr_point_v8_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
