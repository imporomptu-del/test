"""Opt-in cached reference FFT correlation; never substitute direct convolution.

Matches the installed OpenCV crossCorr path for contiguous single-channel
float32 input, float32 9x9 kernel, zero borders and zero delta. Independent tiles
may run on a bounded CPU pool; each tile retains the reference FP64 operations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np


class PointFilterFftExact:
    def __init__(self, shape, kernel, normalizer, *, workers=2):
        import cv2
        from .raw_background_cuda import checked_shape
        self.closed = False
        self.failed = False
        self.pool = None
        self.owner = threading.get_ident()
        self.shape = checked_shape(shape)
        # OpenCV's SSE3 dispatch uses spatial filtering for a 9x9 kernel.
        if 'SSE3' in cv2.getCPUFeaturesLine():
            raise RuntimeError('This reference-FFT execution does not support the SSE3 spatial dispatcher')
        if isinstance(workers, bool) or workers not in (1, 2, 4):
            raise ValueError('Explicit FFT worker count must be 1, 2 or 4')
        if kernel.shape != (9, 9) or kernel.dtype != np.float32 or not np.isfinite(kernel).all():
            raise ValueError('Finite float32 9x9 kernel required')
        self.normalizer = np.float32(normalizer)
        if not np.isfinite(self.normalizer) or self.normalizer <= np.finfo(np.float32).tiny:
            raise ValueError('Positive finite normal float32 normalization required')
        self.cv = cv2
        h, w = self.shape
        self.dft_w = max(cv2.getOptimalDFTSize(min(w, 248) + 8), 2)
        self.dft_h = cv2.getOptimalDFTSize(min(h, 248) + 8)
        self.block_w, self.block_h = min(self.dft_w - 8, w), min(self.dft_h - 8, h)
        templ = np.zeros((self.dft_h, self.dft_w), np.float64)
        templ[:9, :9] = kernel
        self.spectrum = cv2.dft(templ, dst=templ, nonzeroRows=9)
        self.spectrum.setflags(write=False)
        self.tiles = tuple((x, y, min(self.block_w, w-x), min(self.block_h, h-y))
                           for y in range(0, h, self.block_h)
                           for x in range(0, w, self.block_w))
        self.workers = workers
        self.local = threading.local()
        if workers != 1:
            self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='seaqr-exact-fft')

    def _tile(self, job):
        image, out, tile = job
        x, y, bw, bh = tile
        h, w = self.shape
        scratch = getattr(self.local, 'scratch', None)
        if scratch is None:
            scratch = np.empty((self.dft_h, self.dft_w), np.float64)
            self.local.scratch = scratch
        scratch.fill(0)
        x0, y0 = x-4, y-4
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(w, x0+bw+8), min(h, y0+bh+8)
        scratch[y1-y0:y2-y0, x1-x0:x2-x0] = image[y1:y2, x1:x2]
        self.cv.dft(scratch, dst=scratch, nonzeroRows=bh+8)
        self.cv.mulSpectrums(scratch, self.spectrum, 0, c=scratch, conjB=True)
        self.cv.dft(scratch, dst=scratch, flags=self.cv.DFT_INVERSE | self.cv.DFT_SCALE,
                    nonzeroRows=bh)
        out[y:y+bh, x:x+bw] = scratch[:bh, :bw]

    def _compute(self, image, normalize):
        if self.closed or self.failed or threading.get_ident() != self.owner:
            raise RuntimeError('FFT filter is closed, failed or used from another thread')
        if image.shape != self.shape or image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError('Matching finite float32 image required')
        image = np.ascontiguousarray(image)
        out = np.empty(self.shape, np.float32)
        jobs = ((image, out, tile) for tile in self.tiles)
        try:
            if self.pool is None:
                for job in jobs:
                    self._tile(job)
            else:
                # Consume every future before returning; tiles write disjoint output.
                for _ in self.pool.map(self._tile, jobs):
                    pass
            if normalize:
                out /= self.normalizer
            if not np.isfinite(out).all():
                raise RuntimeError('Nonfinite FFT result')
        except BaseException:
            self.failed = True
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
                self.pool = None
            raise
        return out

    def __call__(self, image):
        return self._compute(image, True)

    def correlate(self, image):
        """Unnormalized result for an unchanged filter2D caller's own division."""
        return self._compute(image, False)

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Close FFT filter on its owning thread')
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.closed = True
        self.local = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
