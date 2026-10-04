"""Exact warp, Gaussian, detector state and closed-loop track conformance."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector,VisibleTracks,sha256
from tiny_target.visible_warp_exact import CudaCubicTranslation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('library','config','source','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    if a.source.name!='chunk_0126.avi' or sha256(a.source)!='c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344':
        raise ValueError('Only authorized development126 is permitted')
    cfg=VisibleConfig(**json.loads(a.config.read_text()));cv2.setNumThreads(cfg.opencv_threads)
    result=dict(passed=False,synthetic_pairs=0,native_pairs=0,library_sha256=sha256(a.library),
                script_sha256=sha256(__file__),source_sha256=sha256(a.source),reference_sha256=sha256(a.reference))
    warp=CudaCubicTranslation(a.library);current=[];cap=None
    try:
        result['runtime_conformance']=warp.verify_reference(include_gaussian=True)
        rng=np.random.default_rng(1381)
        for suite in ('synthetic','native'):
            c=cfg if suite=='native' else replace(cfg,tile_size=32,noise_sample_stride=3,
                 max_candidates_per_tile_polarity=3,max_candidates_per_frame=40,warmup_frames=2)
            cpu=VisiblePointDetector(replace(c,stabilization_execution='reference',state_update_backend='inplace',
                                            spatial_filter_backend='cpu',cuda_median_library=None))
            gpu=VisiblePointDetector(replace(c,stabilization_execution='reference',state_update_backend='cuda_resident',
                                            spatial_filter_backend='cuda_median5',cuda_median_library=str(a.library.resolve())))
            current=[cpu,gpu];tracks=[VisibleTracks(c,10),VisibleTracks(c,10)]
            if suite=='native':
                rows=[json.loads(line) for line in a.reference.open()]
                cap=cv2.VideoCapture(str(a.source));cap.set(cv2.CAP_PROP_POS_FRAMES,70)
            for i in range(24 if suite=='native' else 72):
                if suite=='native':
                    ok,bgr=cap.read()
                    if not ok:raise ValueError('Decode failed')
                    im=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);matrix=np.asarray(rows[70+i]['source_to_reference'])
                    segment=rows[70+i]['segment'];mask=np.ones(im.shape,np.uint8)
                else:
                    shape=(67,99) if i<48 else (65,97)
                    im=rng.normal(40,4,shape).astype(np.float32);im[30,20+i%40]=180;im[44,70-i%40]=-30
                    if i%9==0:im[:]=40;im[25:34,24:36]=120
                    mask=(rng.random(shape)>.001).astype(np.uint8)
                    if i%11==0:mask[:]=0
                    matrix=np.eye(3);matrix[:2,2]=(i%32)/32,-((i*11)%32)/32;segment=i//24
                h,w=im.shape
                expected=cv2.warpPerspective(im.astype(np.float32),matrix,(w,h),flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                valid=cv2.warpPerspective(mask,matrix,(w,h),flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                valid=cv2.erode(valid,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                ticket,gv=warp(im,mask,matrix,device=True)
                actual,blur=ticket.download_for_verification()
                for label,left,right in [('warp',expected,actual),('validity',valid,gv),
                    ('gaussian',cv2.GaussianBlur(expected,(5,5),.8),blur)]:
                    if not np.array_equal(left,right):raise AssertionError(f'{suite} {i}: {label} differs at {np.count_nonzero(left!=right)} pixels')
                timestamp=i*100_000_000;centers=[t.learning_centers(timestamp,segment) for t in tracks]
                if centers[0]!=centers[1]:raise AssertionError('Learning centers differ')
                cp,cc=cpu.update(expected,valid,segment,centers[0]);gp,gc=gpu.update(ticket,gv,segment,centers[1])
                cc.pop('detection_ms');gc.pop('detection_ms')
                if (cp,cc)!=(gp,gc):raise AssertionError(f'{suite} {i}: candidates/coverage differ')
                bg,var=gpu._resident.debug_state()
                if not np.array_equal(bg,cpu.background) or not np.array_equal(var,cpu.variance):
                    raise AssertionError(f'{suite} {i}: detector state differs')
                states=[t.update(ps,i,timestamp,segment,matrix,im.shape)[0] for t,ps in zip(tracks,(cp,gp))]
                if states[0]!=states[1]:raise AssertionError(f'{suite} {i}: tracks differ')
                result[suite+'_pairs']+=1
                if suite=='native' and i%4==3:print(json.dumps({'native_pairs':i+1}),flush=True)
            for d in current:d.close()
            current=[]
        result.update(passed=True,exact_pixels_masks_gaussian_candidates_coverage_state_tracks=True,
                      native_source_frames_inclusive=[70,93],no_performance_claim=True)
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for d in current:d.close()
        if cap is not None:cap.release()
        warp.close()
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='traceback'}),flush=True)

if __name__=='__main__':main()
