"""Measure the existing CUDA cubic warp against frozen CPU pixel semantics."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256


def compare(image,matrix):
    h,w=image.shape
    start=time.perf_counter()
    reference=cv2.warpPerspective(image,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    cpu_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter();gpu=cv2.cuda_GpuMat();gpu.upload(image)
    actual=cv2.cuda.warpPerspective(gpu,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0).download()
    gpu_ms=1000*(time.perf_counter()-start);delta=np.abs(reference-actual)
    return dict(shape_hw=[h,w],exact=bool(np.array_equal(reference,actual)),different_pixels=int(np.count_nonzero(delta)),
                max_abs_dn=float(delta.max()),mean_abs_dn=float(delta.mean()),cpu_ms=cpu_ms,cuda_with_copies_ms=gpu_ms)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    if a.source.name!='chunk_0126.avi' or sha256(a.source)!='c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344':raise ValueError('Only authorized126')
    cv2.setNumThreads(2);rng=np.random.default_rng(762);cases=[]
    for dx,dy in [(0,0),(.25,-.5),(.017,-.036),(1.234,5.678),(-8.3,2.9)]:
        im=rng.integers(0,256,(67,99),np.uint8).astype(np.float32)
        m=np.eye(3);m[:2,2]=dx,dy
        cases.append(dict(kind='synthetic',translation=[dx,dy],**compare(im,m)))
    selected={80,100,120,140,160,180,200,220}
    matrices={}
    with a.reference.open() as f:
        for line in f:
            r=json.loads(line)
            if r['frame_index'] in selected:matrices[r['frame_index']]=r['source_to_reference']
    if set(matrices)!=selected:raise ValueError('Incomplete frozen matrices')
    cap=cv2.VideoCapture(str(a.source))
    try:
        for index,matrix in matrices.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
            if not ok:raise ValueError('Decode failed')
            im=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            cases.append(dict(kind='native',frame=index,**compare(im,np.asarray(matrix))))
    finally:cap.release()
    record=dict(source_sha256=sha256(a.source),reference_sha256=sha256(a.reference),script_sha256=sha256(__file__),
                cases=cases,all_exact=all(c['exact'] for c in cases),promoted=False,
                note='Pixel comparison only. Existing CUDA image warp includes upload/download; mask remains CPU. No pipeline accuracy or FPS claim.')
    with a.output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
