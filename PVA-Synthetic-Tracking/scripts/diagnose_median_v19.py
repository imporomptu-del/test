"""Diagnostic-only actual-input guard/median checks; NOT a pipeline FPS trial."""
import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
from profile_visible_v17 import RUNTIME,VISIBLE,read,sha,write
from check_median_v19 import Median
from run_visible_v19 import HERE,run


def guard_counts(a):
    if a.ndim!=2 or a.dtype!=np.float32 or not a.size:raise ValueError('Finite geometry and float32 storage required')
    bits=a.view(np.uint32);mag=bits&0x7fffffff
    negative_zero=bits==0x80000000
    nonfinite=mag>=0x7f800000
    subnormal=(mag!=0)&(mag<0x00800000)
    special=negative_zero|nonfinite|subnormal
    guarded=cv2.dilate(special.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_REPLICATE)
    pixels=int(a.size);count=int(np.count_nonzero(guarded))
    return dict(pixels=pixels,negative_zero_pixels=int(np.count_nonzero(negative_zero)),
        nonfinite_pixels=int(np.count_nonzero(nonfinite)),subnormal_pixels=int(np.count_nonzero(subnormal)),
        guarded_windows=count,guarded_fraction=count/pixels)


def diagnose(clip,output):
    if clip not in ('0126','0082'):raise ValueError('Only the two existing diagnostic prefixes')
    sys.path[:0]=[str(RUNTIME),str(RUNTIME/'scripts')]
    from tiny_target.visible_warp_exact import CudaWarpFrame
    original=CudaWarpFrame.prepare
    rows=[];frame=-1
    record=dict(passed=False,error=None,diagnostic_only=True,not_pipeline_fps=True,
        clip=clip,frames=128,selected_frame_indices=[31,63,95,127],raw16_accessed=False,
        script_sha256=sha(__file__),gate_sha256=sha(HERE/'generated_01.json'),rows=rows)
    libraries={}
    try:
        for mode in ('reference','candidate'):
            libraries[mode]=Median(HERE/'build'/(mode+'_probe.so'),True)
        def prepare(image,library,handle,support,reset,floor2,samples):
            nonlocal frame
            frame+=1
            if frame in record['selected_frame_indices']:
                # Explicit verification download: independent scratch, no detector state writes.
                a,_=image.download_for_verification()
                counts=guard_counts(a);result={}
                # These timings are isolated repeated kernels on actual inputs,
                # after an extra download/upload. Never mix them into pipeline FPS.
                order=('reference','candidate') if len(rows)%2==0 else ('candidate','reference')
                for mode in order:
                    output_pixels=libraries[mode](a)
                    result[mode]=dict(output_sha256=hashlib.sha256(output_pixels.tobytes()).hexdigest(),
                        event_ms=[libraries[mode].event(a.shape) for _ in range(3)])
                if result['reference']['output_sha256']!=result['candidate']['output_sha256']:
                    raise AssertionError('Actual-input median pixels changed')
                rows.append(dict(frame=frame,**counts,input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                    exact=True,timings=result))
            return original(image,library,handle,support,reset,floor2,samples)
        with patch.object(CudaWarpFrame,'prepare',prepare):
            run(argparse.Namespace(clip=clip,mode='candidate',frames=128,output=output))
        if [r['frame'] for r in rows]!=record['selected_frame_indices']:raise AssertionError('Missing diagnostic samples')
        trial=read(output.with_suffix('.v19.json'))
        if not trial['passed']:raise AssertionError('Diagnostic journal mismatch')
        record.update(passed=True,trial_sha256=sha(output.with_suffix('.v19.json')))
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:
        for lib in libraries.values():lib.close()
        write(output.with_suffix('.diagnostic.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnose(a.clip,a.output)
