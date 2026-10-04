"""Bounded real PVA profile; instrumentation is not a throughput benchmark."""
import argparse
import cProfile
import ctypes as C
from dataclasses import replace
import json
from pathlib import Path
import pstats
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,VisiblePointDetector,VisibleTracks,PvaMotion,map_point,sha256

STAGES=['image_upload','blur_upload','support_upload','median_kernel','residual_kernel',
        'sample_gather_kernel','sample_download','stats_upload','peak_kernel','peak_download',
        'counts_download','seeds_upload','patch_gather_kernel','patch_download','learning_mask_upload','state_kernel']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','config','motion-config','reference','library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    if a.source.name!='chunk_0126.avi' or sha256(a.source)!='c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344':
        raise ValueError('Only authorized development126')
    cfg=replace(VisibleConfig(**json.loads(a.config.read_text())),cuda_median_library=str(a.library.resolve()))
    if cfg.state_update_backend!='cuda_resident' or cfg.motion_backend!='pva':raise ValueError('Expected resident PVA')
    cv2.setNumThreads(cfg.opencv_threads)
    detector=VisiblePointDetector(cfg);tracks=VisibleTracks(cfg,10);motion=PvaMotion(a.motion_config)
    lib=detector._resident.lib;lib.seaqr_profile_reset.argtypes=[];lib.seaqr_profile_reset.restype=None
    lib.seaqr_profile_read.argtypes=[C.c_void_p]*3;lib.seaqr_profile_read.restype=None
    cap=cv2.VideoCapture(str(a.source));profile=cProfile.Profile();bad=[];count=0;timings=[]
    record=dict(source_sha256=sha256(a.source),library_sha256=sha256(a.library),script_sha256=sha256(__file__),
                reference_sha256=sha256(a.reference),profile_frames_inclusive=[72,95],passed=False,
                warning='Instrumentation adds synchronization; not production FPS or GPU utilization/occupancy measurement.')
    try:
        with a.reference.open() as reference:
            for i in range(96):
                baseline=json.loads(next(reference))
                if i==72:lib.seaqr_profile_reset()
                if i>=72:profile.enable()
                start=time.perf_counter();ok,bgr=cap.read()
                if not ok:raise ValueError('Decode failed')
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);ts=i*100_000_000
                image,valid,matrix,segment,meta=motion.update(gray,i,ts)
                ps,coverage=detector.update(image,valid,segment,tracks.learning_centers(ts,segment))
                records,metrics=tracks.update(ps,i,ts,segment,matrix,image.shape)
                inverse=np.linalg.inv(matrix)
                for proposal in ps:proposal['source_xy']=map_point(inverse,proposal['x'],proposal['y'])
                if i>=72:timings.append(1000*(time.perf_counter()-start));profile.disable()
                for field,value in [('candidates',ps),('tracks',records),('tracking_metrics',metrics),('source_to_reference',matrix.tolist())]:
                    if baseline[field]!=value:bad.append(dict(frame=i,field=field))
                if i and (meta.get('pva_failure') or meta['reset'] or meta['motion_backends']['cpu_fallback']):raise ValueError('PVA error/reset/fallback')
                count+=1
                if count%24==0:print(json.dumps(dict(frames=count)),flush=True)
        host=np.zeros(16,np.float64);events=np.zeros_like(host);calls=np.zeros(16,np.int32)
        lib.seaqr_profile_read(host.ctypes.data,events.ctypes.data,calls.ctypes.data)
        stats=pstats.Stats(profile)
        functions=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
                   for k,v in stats.stats.items()]
        record.update(passed=not bad,exact_frames=count,differences=bad[:20],difference_count=len(bad),
            cuda_stages={name:dict(calls=int(calls[j]),mean_host_ms=float(host[j]/calls[j]) if calls[j] else None,
                                    mean_event_ms=float(events[j]/calls[j]) if calls[j] else None) for j,name in enumerate(STAGES)},
            python_profile=sorted(functions,key=lambda f:-f['self_ms']),instrumented_frame_ms=timings)
    finally:
        profile.disable();cap.release();detector.close()
        with a.output.open('x') as f:json.dump(record,f,indent=2)
    if bad:raise AssertionError(bad[:10])


if __name__=='__main__':main()
