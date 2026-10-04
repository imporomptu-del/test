"""Synthetic numerical check of Gaussian5 after the conformant cubic warp."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_warp_exact import CudaCubicTranslation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    cv2.setNumThreads(2);warp=CudaCubicTranslation(a.library,3)
    fn=warp.lib.seaqr_warp_gaussian;fn.argtypes=[C.c_void_p,C.c_float,C.c_float,C.c_float,C.c_int,C.c_void_p];fn.restype=C.c_int
    k=cv2.getGaussianKernel(5,.8,cv2.CV_32F).ravel();results=[]
    try:
        for mode in (6,):
            rng=np.random.default_rng(741);cases=[]
            for shape in [(1,1),(1,19),(19,1),(3,5),(17,31),(67,96),(67,99),(3190,4784)]+[(11,w) for w in range(1,130)]:
                im=rng.normal(40,20,shape).astype(np.float32);mask=np.ones(shape,np.uint8)
                warp(im,mask,np.eye(3));out=np.empty_like(im)
                if fn(warp.handle,k[2],k[1],k[0],mode,out.ctypes.data):raise RuntimeError('CUDA gaussian failed')
                cpu=cv2.GaussianBlur(im,(5,5),.8);diff=cpu!=out
                cases.append(dict(shape=list(shape),different=int(diff.sum()),max_abs=float(np.abs(cpu-out).max()),
                                  bad_columns=np.flatnonzero(np.any(diff,axis=0)).tolist()[:30]))
            results.append(dict(mode=mode,cases=cases))
    finally:warp.close()
    with a.output.open('x') as f:json.dump(results,f,indent=2)
    print(json.dumps(results))

if __name__=='__main__':main()
