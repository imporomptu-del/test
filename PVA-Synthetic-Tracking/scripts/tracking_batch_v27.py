"""Opt-in update-local batched geometry; original covariance and association."""
import ctypes as C
import hashlib
import inspect
from pathlib import Path
import textwrap
import numpy as np
from tracking_geometry_v20 import OLD,NEW,REFERENCE_SHA

START='''        log_volumes = {}
        assert self.config.maximum_position_residual_px is not None'''
BATCH_START='''        log_volumes = {}
        _batch_geometry_v27 = _batch_v27(candidate_measurements, self._tracks, dimension,
            self.config.maximum_position_residual_px,
            self.config.maximum_velocity_residual_px_s)
        assert self.config.maximum_position_residual_px is not None'''
BATCH_NEW=NEW.replace('= _geometry_v20(','= _batch_geometry_v27.get(').replace(
    'candidate_measurements, track.mean, dimension,','track_id, candidate_measurements, track.mean, dimension,')


class BatchResult:
    def __init__(self,arrays,indices,fallback):
        self.arrays=arrays;self.indices=indices;self.fallback=fallback

    def get(self,track_id,candidates,mean,dimension,pl,vl):
        if self.arrays is None:return self.fallback(candidates,mean,dimension,pl,vl)
        i=self.indices[track_id];r,p,v,pp,vp,counts=self.arrays;c=counts[i]
        return r[i],p[i],v[i],pp[i],vp[i],int(c[0]),int(c[1]),bool(c[2])


class BatchGeometryV27:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)));self.fn=self.lib.seaqr_tracking_batch_v27
        self.fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_double,C.c_double]+[C.c_void_p]*6
        self.fn.restype=C.c_int
        self.calls=self.track_rows=self.fallbacks=0;self.fallback=None

    def __call__(self,candidates,tracks,dimension,pl,vl):
        if self.fallback is None:raise RuntimeError('Adapter not bound')
        ids=sorted(tracks);means=[tracks[i].mean for i in ids]
        if not ids:return BatchResult(None,None,self.fallback)
        supported=(isinstance(candidates,np.ndarray) and candidates.dtype==np.float64 and candidates.ndim==2
            and candidates.shape[1]==4 and candidates.flags.c_contiguous and candidates.flags.aligned
            and dimension in (2,4) and len(ids)<=512 and len(candidates)<=1024 and len(ids)*len(candidates)<=262144
            and isinstance(pl,(int,float,np.integer,np.floating)) and isinstance(vl,(int,float,np.integer,np.floating))
            and all(isinstance(m,np.ndarray) and m.dtype==np.float64 and m.shape==(4,)
                    and m.flags.c_contiguous and m.flags.aligned for m in means))
        if not supported:
            self.fallbacks+=1;return BatchResult(None,None,self.fallback)
        means=np.array(means,dtype=np.float64).reshape((-1,4));t,n=len(ids),len(candidates)
        arrays=(np.empty((t,n,dimension),np.float64),np.empty((t,n),np.float64),np.empty((t,n),np.float64),
                np.empty((t,n),np.bool_),np.empty((t,n),np.bool_),np.empty((t,3),np.int64))
        code=self.fn(candidates.ctypes.data,means.ctypes.data,t,n,dimension,pl,vl,*(a.ctypes.data for a in arrays))
        if code==1:
            self.fallbacks+=1;return BatchResult(None,None,self.fallback)
        if code:raise RuntimeError('Batched geometry failed: '+str(code))
        self.calls+=1;self.track_rows+=t
        return BatchResult(arrays,{tid:i for i,tid in enumerate(ids)},self.fallback)

    def adapt(self,geometry_method):
        from tiny_target.tracking.kalman import KalmanTrackManager
        original=KalmanTrackManager.update
        if hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest()!=REFERENCE_SHA:
            raise ValueError('Unknown reference tracker')
        source=inspect.getsource(original)
        if source.count(OLD)!=1 or source.count(START)!=1:raise ValueError('Changed tracker anchors')
        expected=compile(textwrap.dedent(source.replace(OLD,NEW)),'<tracking_geometry_v20:exact>','exec')
        scope=dict(original.__globals__,_geometry_v20=geometry_method.__globals__.get('_geometry_v20'))
        exec(expected,scope)
        if geometry_method.__code__!=scope['update'].__code__:
            raise ValueError('Expected original v20 geometry adapter')
        self.fallback=geometry_method.__globals__['_geometry_v20']
        source=textwrap.dedent(source.replace(START,BATCH_START).replace(OLD,BATCH_NEW))
        self.transformed_sha256=hashlib.sha256(source.encode()).hexdigest()
        namespace=dict(original.__globals__,_batch_v27=self)
        exec(compile(source,'<tracking_batch_v27:exact>','exec'),namespace)
        return namespace['update']
