"""Exact generated-test adapter; no production default or estimator changes."""
import numpy as np
from tiny_target.motion import pva_pyrlk as pva
REFERENCE_FEATURE_PIXELS=pva._feature_pixels


def feature_pixels_lut(frame,mapping):
    if mapping!='raw_robust_u16_v1' or frame.bit_depth<=8:
        return REFERENCE_FEATURE_PIXELS(frame,mapping)
    image=frame.image
    if image.dtype.kind!='u' or image.dtype.itemsize!=2 or not 9<=frame.bit_depth<=16:
        raise pva.PvaMotionError('RAW U16 motion requires unsigned 9..16-bit source samples')
    if frame.bit_depth<16 and np.any(image>(1<<frame.bit_depth)-1):
        raise pva.PvaMotionError('RAW source samples exceed the declared bit depth')
    scale,offset=pva._raw_affine_parameters(frame)
    work=np.arange(65536,dtype=np.float32)
    np.multiply(work,scale,out=work);np.add(work,offset,out=work)
    np.clip(work,0,65535,out=work);np.rint(work,out=work)
    return np.ascontiguousarray(work.astype(np.uint16)[image])
