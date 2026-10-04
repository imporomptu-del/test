"""Generated-data-only RAW16 core probe; deliberately not a production backend."""
import ctypes as C
import hashlib
from pathlib import Path
import platform
import threading

import cv2
import numpy as np

from resident_tracking_v10 import ResidentTracker
from raw16_speed_v8_common import filter_parameters
from tiny_target.raw_background_cuda import parameters,cpu_flush_to_zero_enabled
from tiny_target.warp_translation_cuda import (
    cubic_table,translation_inverse,OPENCV_BUILD_SHA256,TABLE_SHA256,
)


class ResidentRawProbe:
    def __init__(self,config,shape,grid,library):
        self.handle=None;self.failed=False;self.owner=threading.get_ident();self.ring=None
        if (platform.machine()!='aarch64' or not cpu_flush_to_zero_enabled()
            or hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()!=OPENCV_BUILD_SHA256):
            raise RuntimeError('Only the pinned Jetson environment is allowed')
        if config.background_warmup_frames!=4:
            raise ValueError('Frozen four-frame warmup required')
        table=cubic_table()
        if hashlib.sha256(table.tobytes()).hexdigest()!=TABLE_SHA256:
            raise ValueError('CPU warp table changed')
        self.shape=shape;self.count=0;self.last_metadata=None
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)))
        ptr=C.c_void_p;i=C.c_int;f=C.c_float;d=C.c_double
        signatures={
            'abi':([],i),'error':([i],C.c_char_p),
            'create':([i,i,i,i,ptr,ptr,ptr,f,f,f,C.POINTER(ptr)],i),
            'step':([ptr,ptr,ptr,d,d,ptr,ptr,ptr],i),'reset':([ptr],i),
            'debug':([ptr]*6,i),'destroy':([ptr],None),
        }
        for name,(args,result) in signatures.items():
            fn=getattr(self.lib,'seaqr_front_v10_'+name);fn.argtypes=args;fn.restype=result
        if self.lib.seaqr_front_v10_abi()!=1:raise RuntimeError('Unknown RAW probe ABI')
        self.ring=ResidentTracker(shape,grid,library)
        values,warmup,required=parameters(config);kernel,norm=filter_parameters()
        handle=ptr()
        try:
            self._check(self.lib.seaqr_front_v10_create(*shape,warmup,required,values.ctypes.data,
                table.ctypes.data,kernel.ctypes.data,float(np.float32(norm)),float(np.float32(config.dark_floor_dn)),
                float(np.float32(65535*config.saturation_fraction)),C.byref(handle)))
            if not handle.value:raise RuntimeError('Null RAW probe handle')
            self.handle=handle
        except BaseException:
            self.ring.close();raise

    def _live(self):
        if not self.handle or self.failed or threading.get_ident()!=self.owner:
            raise RuntimeError('RAW probe closed, failed or used from another thread')
        self.ring._live()

    def _check(self,status):
        if status:
            self.failed=True
            if self.ring is not None:self.ring.failed=True
            raise RuntimeError(self.lib.seaqr_front_v10_error(status).decode()+'; no fallback')

    def reset(self):
        self._live();self._check(self.lib.seaqr_front_v10_reset(self.handle));self.ring.reset()
        self.count=0;self.last_metadata=None

    def push(self,raw,index,timestamp,matrix,mask=None):
        self._live()
        self.ring.validate_metadata(index,timestamp,0,'bright')
        if self.last_metadata is not None and (index<=self.last_metadata[0] or timestamp<=self.last_metadata[1]):
            raise ValueError('Warmup metadata must also strictly increase')
        if raw.shape!=self.shape or raw.dtype!=np.uint16:
            raise ValueError('Original native uint16 input required; no quantization')
        inverse=translation_inverse(matrix)
        raw=np.ascontiguousarray(raw);valid=None
        if mask is not None:
            if mask.shape!=self.shape or mask.dtype!=np.bool_:
                raise ValueError('Matching boolean source mask required')
            valid=np.ascontiguousarray(mask,dtype=np.uint8)
        emitted=C.c_int();compute=C.c_float()
        self._check(self.lib.seaqr_front_v10_step(self.handle,raw.ctypes.data,
            None if valid is None else valid.ctypes.data,inverse[0,2],inverse[1,2],self.ring.handle,
            C.byref(emitted),C.byref(compute)))
        expected=self.count>=4
        if bool(emitted.value)!=expected:
            self.failed=True;self.ring.failed=True
            raise RuntimeError('Unexpected background warmup state')
        if expected:self.ring.commit_metadata(index,timestamp)
        self.last_metadata=(index,timestamp);self.count+=1
        return float(compute.value)

    def debug(self):
        self._live()
        if self.count<2:raise ValueError('No residual yet')
        arrays=[np.empty(self.shape,dtype) for dtype in (np.float32,np.uint8,np.float32,np.float32,np.uint8)]
        self._check(self.lib.seaqr_front_v10_debug(self.handle,*[a.ctypes.data for a in arrays]))
        return arrays

    def close(self):
        if self.handle:
            if threading.get_ident()!=self.owner:raise RuntimeError('Close on owning thread')
            self.lib.seaqr_front_v10_destroy(self.handle);self.handle=None
        if self.ring:self.ring.close()

    def __del__(self):
        try:self.close()
        except Exception:pass
