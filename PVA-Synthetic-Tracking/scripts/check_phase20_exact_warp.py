"""Select arithmetic by synthetic numerical conformance, never target labels."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_warp_exact import CudaCubicTranslation
from tiny_target.visible_baseline import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extended',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    cv2.setNumThreads(2);rng=np.random.default_rng(942);results=[]
    for mode in ((3,) if a.extended else (0,1,2,3)):
        rng=np.random.default_rng(942)
        warp=CudaCubicTranslation(a.library,mode);cases=[]
        try:
            trials=[(shape,[(0,0),(.25,-.5),(.017,-.036),(1.234,5.678),(-8.3,2.9),(.5,.5),(1/64,-1/64)])
                    for shape in [(1,1),(3,5),(17,31),(67,99)]]
            if a.extended:
                trials += [((21,71),[(x/32,y/32) for y in range(32) for x in range(32)]),
                           ((19,131),[(s*(k+.5)/32+e,-s*(k+.5)/32-e)
                                       for s in (-1,1) for k in range(32) for e in (-1e-10,0,1e-10)]),
                           ((3190,4784),[(0,0),(1.234,5.678),(-8.3,2.9),(.5,-.5)]),
                           ((1,99),[(.5,-.5),(2.123,0)]),((99,1),[(0,2.123),(.5,-.5)]),
                           ((17,31),[(200,-300),(-32700,32700)])]
            for shape,shifts in trials:
                for dx,dy in shifts:
                    m=np.eye(3);m[:2,2]=dx,dy
                    for kind in (np.uint8,np.float32):
                        im=rng.integers(0,256,shape,np.uint8) if kind==np.uint8 else rng.normal(20,40,shape).astype(np.float32)
                        mask=(rng.random(shape)>.05).astype(np.uint8);h,w=shape
                        cpu=cv2.warpPerspective(im.astype(np.float32),m,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                        expected_mask=cv2.warpPerspective(mask,m,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                        gpu,valid=warp(im,mask,m);diff=np.abs(cpu-gpu)
                        cases.append(dict(shape_hw=list(shape),shift=[dx,dy],dtype=str(im.dtype),pixels_differ=int(np.count_nonzero(diff)),
                                          max_abs=float(diff.max()),mask_differ=int(np.count_nonzero(valid!=expected_mask))))
        finally:warp.close()
        results.append(dict(mode=mode,exact_cases=sum(c['pixels_differ']==0 and c['mask_differ']==0 for c in cases),cases=cases))
    with a.output.open('x') as f:json.dump(dict(library_sha256=sha256(a.library),script_sha256=sha256(__file__),results=results),f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in results]))


if __name__=='__main__':main()
