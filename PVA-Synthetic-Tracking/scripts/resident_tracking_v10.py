"""Explicit research-only persistent temporal buffer; no decoder or default hook."""
from collections import deque
import ctypes as C
import math
from pathlib import Path
import threading
import time

import numpy as np


class ResidentTracker:
    def __init__(self, shape, velocity_grid, library):
        self.handle = None
        self.failed = False
        self.owner = threading.get_ident()
        if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
                or math.prod(shape) > 32000000):
            raise ValueError('Bounded nonempty image shape required')
        self.shape = tuple(shape)
        if velocity_grid.shape != (48, 2) or velocity_grid.dtype != np.float32 or not np.isfinite(velocity_grid).all():
            raise ValueError('Frozen 48-velocity float32 grid required')
        self.grid = np.ascontiguousarray(velocity_grid).copy()
        self.grid.setflags(write=False)
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        ptr, integer = C.c_void_p, C.c_int
        signatures = {
            'abi': ([], integer), 'error': ([integer], C.c_char_p),
            'create': ([integer]*7+[ptr,C.POINTER(ptr)], integer),
            'push': ([ptr,ptr,ptr], integer), 'reset': ([ptr], integer),
            'run': ([ptr,ptr,integer,ptr], integer),
            'download': ([ptr]*5, integer), 'destroy': ([ptr], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, 'seaqr_ring_v10_'+name)
            fn.argtypes, fn.restype = args, result
        if self.lib.seaqr_ring_v10_abi() != 1:
            raise RuntimeError('Unknown resident ABI')
        handle = ptr()
        self._check(self.lib.seaqr_ring_v10_create(*self.shape,16,48,12,32,256,
            self.grid.ctypes.data,C.byref(handle)))
        if not handle.value:
            raise RuntimeError('Null resident workspace')
        self.handle = handle
        self.rows = deque(maxlen=16)
        self.segment, self.polarity, self.count = 0, 'bright', 0
        self.output_generation = None

    def _live(self):
        if not self.handle or self.failed or threading.get_ident() != self.owner:
            raise RuntimeError('Resident tracker closed, failed or used from another thread')

    def _check(self, status):
        if status:
            self.failed = True
            raise RuntimeError(self.lib.seaqr_ring_v10_error(status).decode()+'; no fallback')

    def validate_metadata(self, index, timestamp, segment, polarity):
        self._live()
        if (type(index) is not int or index < 0 or type(timestamp) is not int
                or not 0 <= timestamp < 2**63 or segment != self.segment or polarity != self.polarity):
            raise ValueError('Invalid frame metadata; discontinuity requires explicit reset')
        if self.rows and (index <= self.rows[-1][0] or timestamp <= self.rows[-1][1]):
            raise ValueError('Frame indices and timestamps must strictly increase')

    def commit_metadata(self, index, timestamp):
        self.rows.append((index,timestamp))
        self.count += 1
        self.output_generation = None

    def push(self, response, mask, index, timestamp, *, segment=0, polarity='bright'):
        self.validate_metadata(index,timestamp,segment,polarity)
        if response.shape != self.shape or response.dtype != np.float32 or not np.isfinite(response).all():
            raise ValueError('Matching finite float32 response required')
        if mask.shape != self.shape or mask.dtype != np.bool_:
            raise ValueError('Matching boolean mask required')
        response = np.ascontiguousarray(response)
        valid = np.ascontiguousarray(mask,dtype=np.uint8)
        self._check(self.lib.seaqr_ring_v10_push(self.handle,response.ctypes.data,valid.ctypes.data))
        self.commit_metadata(index,timestamp)

    def reset(self, *, segment=0, polarity='bright'):
        self._live()
        if type(segment) is not int or segment < 0 or polarity not in ('bright','dark'):
            raise ValueError('Invalid segment or polarity')
        self._check(self.lib.seaqr_ring_v10_reset(self.handle))
        self.rows.clear()
        self.segment,self.polarity,self.count = segment,polarity,0
        self.output_generation = None

    def run(self):
        self._live()
        if len(self.rows) != 16:
            raise ValueError('A full 16-frame window is required')
        timestamps = np.array([r[1] for r in self.rows],np.int64)
        midpoint = (int(timestamps[0])+int(timestamps[-1]))//2
        offsets = np.ascontiguousarray((timestamps.astype(np.float64)-midpoint)/1e9)
        kernel = C.c_float()
        self._check(self.lib.seaqr_ring_v10_run(self.handle,offsets.ctypes.data,
            1 if self.polarity == 'bright' else -1,C.byref(kernel)))
        self.output_generation = self.count
        return float(kernel.value)

    def download(self):
        self._live()
        if self.output_generation != self.count:
            raise RuntimeError('No current completed window; stale reads forbidden')
        arrays = [np.empty(self.shape,dtype) for dtype in (np.float32,np.uint16,np.uint16,np.uint8)]
        self._check(self.lib.seaqr_ring_v10_download(self.handle,*[a.ctypes.data for a in arrays]))
        if not np.isfinite(arrays[0]).all() or np.any(arrays[1][arrays[3] != 0] >= 48):
            self.failed = True
            raise RuntimeError('Invalid resident result')
        return arrays

    def close(self):
        if self.handle:
            if threading.get_ident() != self.owner:
                raise RuntimeError('Close on owning thread')
            self.lib.seaqr_ring_v10_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
