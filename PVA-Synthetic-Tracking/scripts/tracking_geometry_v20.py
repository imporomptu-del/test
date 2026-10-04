"""Opt-in exact geometry fusion; unchanged tracker outside one guarded block."""
import ctypes as C
import hashlib
import inspect
from pathlib import Path
import textwrap
import numpy as np

REFERENCE_SHA='54e36d56d98eb4533661d58f921e5350c12805fcec7acc14c11258e5884c29ac'
OLD='''            residuals = candidate_measurements[:, :dimension] - track.mean[:dimension]
            position_residuals = np.sqrt(np.sum(residuals[:, :2] ** 2, axis=1))
            velocity_residuals = np.sqrt(np.sum(residuals[:, 2:] ** 2, axis=1))
            position_pass = (
                position_residuals <= self.config.maximum_position_residual_px
            )
            velocity_pass = (
                velocity_residuals <= self.config.maximum_velocity_residual_px_s
            )
            rejected_by_position += int(np.count_nonzero(~position_pass))
            rejected_by_velocity += int(
                np.count_nonzero(position_pass & ~velocity_pass)
            )
            if not np.any(position_pass & velocity_pass):'''
NEW='''            (residuals, position_residuals, velocity_residuals, position_pass,
             velocity_pass, rejected_position, rejected_velocity, any_pass) = _geometry_v20(
                 candidate_measurements, track.mean, dimension,
                 self.config.maximum_position_residual_px,
                 self.config.maximum_velocity_residual_px_s)
            rejected_by_position += rejected_position
            rejected_by_velocity += rejected_velocity
            if not any_pass:'''


def reference(candidates,mean,dimension,position_limit,velocity_limit):
    r=candidates[:,:dimension]-mean[:dimension]
    p=np.sqrt(np.sum(r[:,:2]**2,axis=1));v=np.sqrt(np.sum(r[:,2:]**2,axis=1))
    pp=p<=position_limit;vp=v<=velocity_limit
    return r,p,v,pp,vp,int(np.count_nonzero(~pp)),int(np.count_nonzero(pp&~vp)),bool(np.any(pp&vp))


class GeometryV20:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)))
        self.fn=self.lib.seaqr_tracking_geometry_v20
        self.fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_double,C.c_double]+[C.c_void_p]*6
        self.fn.restype=C.c_int
        self.calls=0;self.fallbacks=0

    def __call__(self,candidates,mean,dimension,position_limit,velocity_limit):
        if (not isinstance(candidates,np.ndarray) or not isinstance(mean,np.ndarray)
            or candidates.dtype!=np.float64 or mean.dtype!=np.float64
            or candidates.ndim!=2 or candidates.shape[1]!=4 or mean.shape!=(4,)
            or not candidates.flags.c_contiguous or not mean.flags.c_contiguous
            or not candidates.flags.aligned or not mean.flags.aligned
            or dimension not in (2,4) or len(candidates)>100000):
            self.fallbacks+=1
            return reference(candidates,mean,dimension,position_limit,velocity_limit)
        n=len(candidates)
        r=np.empty((n,dimension),np.float64);p=np.empty(n,np.float64);v=np.empty(n,np.float64)
        pp=np.empty(n,np.bool_);vp=np.empty(n,np.bool_);counts=np.empty(3,np.int64)
        status=self.fn(candidates.ctypes.data,mean.ctypes.data,n,dimension,position_limit,velocity_limit,
            r.ctypes.data,p.ctypes.data,v.ctypes.data,pp.ctypes.data,vp.ctypes.data,counts.ctypes.data)
        if status==1:
            self.fallbacks+=1
            return reference(candidates,mean,dimension,position_limit,velocity_limit)
        if status:raise RuntimeError('Native tracking geometry failed: '+str(status))
        self.calls+=1
        return r,p,v,pp,vp,int(counts[0]),int(counts[1]),bool(counts[2])

    def adapter(self,original):
        path=Path(inspect.getfile(original))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=REFERENCE_SHA:
            raise ValueError('Unknown frozen tracker implementation')
        source=inspect.getsource(original)
        if source.count(OLD)!=1:raise ValueError('Exactly one frozen geometry block required')
        modified=textwrap.dedent(source.replace(OLD,NEW))
        self.transformed_sha256=hashlib.sha256(modified.encode()).hexdigest()
        namespace=dict(original.__globals__,_geometry_v20=self)
        exec(compile(modified,'<tracking_geometry_v20:exact>','exec'),namespace)
        return namespace[original.__name__]
