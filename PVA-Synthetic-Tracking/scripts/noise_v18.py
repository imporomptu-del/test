"""Opt-in exact bounded native noise path; original generic semantics retained."""
import ctypes as C
import hashlib
import inspect
from pathlib import Path
import numpy as np
from tiny_target.visible_noise import tile_noise_statistics as reference

REFERENCE_SHA = '59e63d655c4ec64c9dabfb7a1d3f9457af50798c0ca7a2ab47ac51c58e0401a9'


class NoiseV18:
    def __init__(self, library):
        if hashlib.sha256(Path(inspect.getfile(reference)).read_bytes()).hexdigest() != REFERENCE_SHA:
            raise ValueError('Unknown noise reference')
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        self.fn = self.lib.seaqr_noise_v18
        self.fn.argtypes = [C.c_void_p, C.c_int64, C.c_void_p, C.c_int64,
                           C.c_void_p, C.c_void_p, C.c_int64, C.c_double,
                           C.c_void_p, C.c_void_p]
        self.fn.restype = C.c_int
        self.calls = self.fallbacks = self.geometry_builds = 0
        self.key = self.indices = self.boundaries = None

    def geometry(self, samples, support, layout, stride):
        if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32 or samples.ndim != 1
                or not isinstance(support, np.ndarray) or support.dtype != np.bool_ or support.ndim != 2
                or not 0 < support.size <= 32000000 or not 0 < samples.size <= 32000000
                or not isinstance(stride, (int, np.integer)) or stride <= 0 or not layout):
            return False
        parts = []
        for ys, xs, offset, count in layout:
            if not isinstance(ys, slice) or not isinstance(xs, slice):
                return False
            if ys.step not in (None, 1) or xs.step not in (None, 1):
                return False
            fields = (ys.start, ys.stop, xs.start, xs.stop, offset, count)
            if not all(isinstance(v, (int, np.integer)) for v in fields):
                return False
            parts.append(tuple(int(v) for v in fields))
        key = (support.shape, samples.size, int(stride), tuple(parts))
        if key == self.key:
            return True
        ids, bounds, expected = [], [0], 0
        h, w = support.shape
        for y0, y1, x0, x1, offset, count in parts:
            if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w and offset == expected and count > 0):
                return False
            yy, xx = np.arange(y0, y1, stride, dtype=np.int64), np.arange(x0, x1, stride, dtype=np.int64)
            if len(yy)*len(xx) != count:
                return False
            ids.append((yy[:, None]*w + xx).ravel())
            expected += count
            bounds.append(expected)
        if expected != samples.size:
            return False
        self.indices = np.concatenate(ids)
        self.boundaries = np.asarray(bounds, dtype=np.int64)
        self.key = key
        self.geometry_builds += 1
        return True

    def __call__(self, samples, support, layout, stride, noise_floor):
        if (not np.isscalar(noise_floor) or not np.isfinite(noise_floor)
                or noise_floor < 0 or noise_floor > 1e15 or np.signbit(noise_floor)
                or not self.geometry(samples, support, layout, stride)):
            self.fallbacks += 1
            return reference(samples, support, layout, stride, noise_floor)
        source = np.ascontiguousarray(samples)
        mask = np.ascontiguousarray(support)
        stats = np.empty((len(layout), 2), np.float32)
        sigmas = np.empty(len(layout), np.float64)
        status = self.fn(source.ctypes.data, source.size, mask.ctypes.data, mask.size,
                         self.indices.ctypes.data, self.boundaries.ctypes.data, len(layout),
                         float(noise_floor), stats.ctypes.data, sigmas.ctypes.data)
        if status == 1:
            self.fallbacks += 1
            return reference(samples, support, layout, stride, noise_floor)
        if status:
            raise RuntimeError(f'Native noise failed ({status}); no error fallback')
        self.calls += 1
        return stats, float(np.median(sigmas))
