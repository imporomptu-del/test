"""Small adversarial lifecycle sequence, optionally run under Compute Sanitizer.

Passing this functional check alone does not establish memory safety.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector,sha256
from tiny_target.visible_warp_exact import CudaCubicTranslation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    cv2.setNumThreads(2);rng=np.random.default_rng(1048)
    cfg=VisibleConfig(spatial_background='median5',spatial_filter_backend='cuda_median5',
        cuda_median_library=str(a.library.resolve()),state_update_backend='cuda_resident',pixel_noise_enabled=True,
        pixel_noise_model='background_residual',tile_size=32,noise_sample_stride=3,warmup_frames=2,
        max_candidates_per_tile_polarity=16,max_candidates_per_frame=512)
    warp=CudaCubicTranslation(a.library);gpu=VisiblePointDetector(cfg)
    cpu=VisiblePointDetector(replace(cfg,state_update_backend='inplace',spatial_filter_backend='cpu',cuda_median_library=None))
    result=dict(passed=False,frames=0,media_read=False,
        check_type='functional_lifecycle_and_state_parity',
        memory_safety_proven=False,
        library_sha256=sha256(a.library),script_sha256=sha256(__file__))
    previous=None
    try:
        for i in range(24):
            shape=(65,99) if i<12 else (67,97)
            im=rng.normal(40,8,shape).astype(np.float32);mask=(rng.random(shape)>.001).astype(np.uint8)
            if i%7==0:mask[:]=0
            m=np.eye(3);m[:2,2]=(i%5)/32,-(i%7)/32
            ticket,valid=warp(im,mask,m,device=True)
            if previous is not None:
                try:previous.validate(previous.valid)
                except ValueError:pass
                else:raise AssertionError('Stale frame accepted')
            expected=cv2.warpPerspective(im,m,(shape[1],shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            a_result=cpu.update(expected,valid,i//12);b_result=gpu.update(ticket,valid,i//12)
            for pair in (a_result,b_result):pair[1].pop('detection_ms')
            if a_result!=b_result:raise AssertionError('Stateful proposals/coverage differ')
            bg,var=gpu._resident.debug_state()
            if not np.array_equal(bg,cpu.background) or not np.array_equal(var,cpu.variance):raise AssertionError('State differs')
            previous=ticket;result['frames']+=1
        gpu.close();gpu.close();warp.close();warp.close()
        try:previous.validate(previous.valid)
        except ValueError:pass
        else:raise AssertionError('Closed frame accepted')
        result['passed']=True
    finally:
        gpu.close();cpu.close();warp.close()
        with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
