"""Source-bound coarse native RANSAC scoring; refinement/gates remain reference."""
import ctypes as C
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import numpy as np
from tiny_target.motion import global_motion as gm

REFERENCE_SHA='ae850a7bbe1b3965c412bd63f78497483b5db62761b6e24624920e96717eb850'
OLD='''    for sample in samples if execution == "reference" else ():
        matrix = fitter(previous[list(sample)], current[list(sample)])
        if matrix is None:
            continue
        residuals = _residuals(matrix, previous, current)
        mask = residuals <= resolved.ransac_reprojection_px
        inlier_count = int(np.count_nonzero(mask))
        median = float(np.median(residuals[mask])) if inlier_count else math.inf
        score = (inlier_count, -median, tuple(-item for item in sample))
        if best_score is None or score > best_score:
            best_score = score
            best_matrix = matrix
            best_mask = mask
'''
NEW='''    best_matrix, best_mask, best_score = _native_score_v25(
        previous, current, samples, resolved.ransac_reprojection_px)
'''


def reference_samples(previous,current,samples,threshold):
    best_matrix=best_mask=best_score=None
    for sample in samples:
        matrix=gm._fit_translation(previous[list(sample)],current[list(sample)])
        if matrix is None:continue
        residuals=gm._residuals(matrix,previous,current);mask=residuals<=threshold
        count=int(np.count_nonzero(mask));median=float(np.median(residuals[mask])) if count else math.inf
        score=(count,-median,tuple(-item for item in sample))
        if best_score is None or score>best_score:best_matrix,best_mask,best_score=matrix,mask,score
    return best_matrix,best_mask,best_score


class NativeMotionV25:
    def __init__(self,library):
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)))
        self.fn=self.lib.seaqr_translation_score_v25
        self.fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_double,
                          C.c_void_p,C.c_void_p,C.c_void_p]
        self.fn.restype=C.c_int
        self.calls=0;self.fallbacks=0;self.passthroughs=0

    def __call__(self,previous,current,samples,threshold):
        supported=(isinstance(previous,np.ndarray) and isinstance(current,np.ndarray)
            and previous.dtype==current.dtype==np.float64 and previous.ndim==current.ndim==2
            and previous.shape==current.shape and previous.shape[1]==2 and 1<=len(previous)<=4096
            and all(a.flags.c_contiguous and a.flags.aligned for a in (previous,current))
            and isinstance(samples,list) and 1<=len(samples)<=1024
            and all(isinstance(s,tuple) and len(s)==1 and isinstance(s[0],(int,np.integer))
                    and not isinstance(s[0],(bool,np.bool_)) and 0<=s[0]<len(previous) for s in samples))
        if not supported:
            self.fallbacks+=1;return reference_samples(previous,current,samples,threshold)
        indices=np.array([s[0] for s in samples],dtype=np.int64)
        mask=np.empty(len(previous),np.bool_);result=np.empty(2,np.int64);median=np.empty(1,np.float64)
        code=self.fn(previous.ctypes.data,current.ctypes.data,len(previous),indices.ctypes.data,
                     len(indices),threshold,mask.ctypes.data,result.ctypes.data,median.ctypes.data)
        if code==1:
            self.fallbacks+=1;return reference_samples(previous,current,samples,threshold)
        if code:raise RuntimeError('Native hypothesis scoring failed: '+str(code))
        self.calls+=1;index,count=(int(x) for x in result)
        matrix=gm._fit_translation(previous[[index]],current[[index]])
        return matrix,mask,(count,-float(median[0]),(-index,))

    def adapter(self,original):
        if hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest()!=REFERENCE_SHA:
            raise ValueError('Unrecognized frozen global motion source')
        source=textwrap.dedent(inspect.getsource(original))
        if source.count(OLD)!=1:raise ValueError('Exactly one frozen hypothesis loop required')
        changed=source.replace(OLD,NEW)
        self.transformed_sha256=hashlib.sha256(changed.encode()).hexdigest()
        namespace=dict(original.__globals__,_native_score_v25=self)
        exec(compile(changed,'<native_motion_v25:scoring-only>','exec'),namespace)
        transformed=namespace[original.__name__]
        def dispatch(correspondences,config=None,*,execution='reference'):
            resolved=config if isinstance(config,gm.GlobalMotionConfig) else gm.GlobalMotionConfig.from_mapping(config)
            if resolved.model!='translation' or execution!='reference':
                self.passthroughs+=1;return original(correspondences,resolved,execution=execution)
            return transformed(correspondences,resolved,execution=execution)
        return dispatch
