"""Adversarial and native resident-core parity before any pipeline speed claim."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector,VisibleTracks,sha256


def pair(config,library):
    cpu=replace(config,spatial_filter_backend='cpu',cuda_median_library=None,state_update_backend='inplace')
    gpu=replace(cpu,spatial_filter_backend='cuda_median5',cuda_median_library=str(library.resolve()),state_update_backend='cuda_resident')
    return VisiblePointDetector(cpu),VisiblePointDetector(gpu)


def check(a,b,im,mask,segment,centers=(),reverse=False):
    result={};times={}
    for name,d in ([('cpu',a),('gpu',b)] if not reverse else [('gpu',b),('cpu',a)]):
        start=time.perf_counter();result[name]=d.update(im,mask,segment,centers);times[name]=1000*(time.perf_counter()-start)
        result[name][1].pop('detection_ms')
    if result['cpu']!=result['gpu']:
        raise AssertionError('Proposals/coverage differ: '+repr(result)[:2500])
    bg,var=b._resident.debug_state()
    for name,x,y in [('background',a.background,bg),('variance',a.variance,var)]:
        if not np.array_equal(x,y):
            bad=np.argwhere(x!=y);point=tuple(bad[0]);raise AssertionError(f'{name} differs at {point}: {x[point]} vs {y[point]}; count={len(bad)}')
    return result['cpu'][0],times


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output already exists')
    if a.source.name!='chunk_0126.avi' or sha256(a.source)!='c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344':
        raise ValueError('Only authorized development126')
    cfg=VisibleConfig(**json.loads(a.config.read_text()));cv2.setNumThreads(cfg.opencv_threads)
    record=dict(source_sha256=sha256(a.source),script_sha256=sha256(__file__),library_sha256=sha256(a.library),
        reference_configuration=asdict(cfg),synthetic_pairs=0,native_pairs=0,passed=False)
    current=None
    try:
        rng=np.random.default_rng(742)
        for mode in ['none','circle','shape']:
            params=dict(tile_size=32,noise_sample_stride=3,max_candidates_per_tile_polarity=3,max_candidates_per_frame=40,warmup_frames=2)
            if mode=='none':params.update(learning_protection_geometry='circle',learning_exclusion_radius_px=0)
            if mode=='circle':params.update(learning_protection_geometry='circle',learning_exclusion_radius_px=3)
            if mode=='shape':params.update(learning_protection_geometry='observed_shape',learning_exclusion_radius_px=0)
            c=replace(cfg,**params);x,y=pair(c,a.library);current=(x,y)
            for i in range(24):
                im=rng.normal(20,3,(67,99)).astype(np.float32)
                if i%5==0:im[:]=20;im[25:34,24:36]=120
                im[30,20+i]=160;im[40,70-i]=0
                mask=np.ones(im.shape,bool);mask[15:20,30:35]=i%3!=0
                centers=([dict(support_reference_xy=[[20+i,30]])] if mode=='shape' else [(20+i,30)]) if i%2 else []
                check(x,y,im,mask,i//12,centers);record['synthetic_pairs']+=1
            x.close();y.close();current=None
        # Native comparisons include closed-loop shape learning and tracking.
        x,y=pair(cfg,a.library);current=(x,y);tx=VisibleTracks(x.config,10);ty=VisibleTracks(y.config,10)
        cap=cv2.VideoCapture(str(a.source));cap.set(cv2.CAP_PROP_POS_FRAMES,70);times={'cpu':[],'gpu':[]}
        for i in range(24):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Decode failed')
            im=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32);mask=np.ones(im.shape,bool)
            timestamp=i*100_000_000;cx=tx.learning_centers(timestamp,0);cy=ty.learning_centers(timestamp,0)
            if cx!=cy:raise AssertionError('Closed-loop centers differ')
            ps,elapsed=check(x,y,im,mask,0,cx,reverse=bool(i%2))
            rx,mx=tx.update(ps,i,timestamp,0,np.eye(3),im.shape);ry,my=ty.update(ps,i,timestamp,0,np.eye(3),im.shape)
            # Manager metrics contain elapsed_ms; observations and states are exact.
            if rx!=ry:raise AssertionError('Tracks differ')
            if i>=8:
                for k in times:times[k].append(elapsed[k])
            record['native_pairs']+=1;print(json.dumps({'native_pairs':i+1}),flush=True)
        cap.release();record.update(passed=True,native_source_frames_inclusive=[70,93],
            exact_candidates_coverage_background_variance_tracks=True,
            detector_median_ms={k:float(np.median(v)) for k,v in times.items()},samples_ms=times,
            debug_state_download_excluded_from_detector_timing=True,pipeline_speed_claim=False)
    except Exception as exc:
        record.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if current:
            for d in current:d.close()
        with a.output.open('x') as f:json.dump(record,f,indent=2)
        print(json.dumps({k:v for k,v in record.items() if k not in ('reference_configuration','traceback','samples_ms')}),flush=True)


if __name__=='__main__':main()
