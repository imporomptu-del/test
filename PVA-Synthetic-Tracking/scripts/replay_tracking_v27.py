"""Bounded tracking-only replay; never opens media, compares archived outputs."""
import argparse
from collections import deque
from contextlib import nullcontext
import cProfile
from dataclasses import fields,is_dataclass
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch
import numpy as np
from profile_visible_v17 import read,sha,write
from tiny_target.visible_baseline import VisibleConfig,VisibleTracks
from tiny_target.tracking.kalman import KalmanTrackManager
from tracking_geometry_v20 import GeometryV20,REFERENCE_SHA

HERE=Path(__file__).resolve().parent


def normalized(value):
    if isinstance(value,np.ndarray):return ['array',value.dtype.str,list(value.shape),value.tobytes().hex()]
    if isinstance(value,np.generic):return normalized(value.item())
    if is_dataclass(value):return ['dataclass',type(value).__name__,[(f.name,normalized(getattr(value,f.name))) for f in fields(value)]]
    if isinstance(value,dict):return ['dict',[(normalized(k),normalized(v)) for k,v in sorted(value.items(),key=lambda p:repr(p[0]))]]
    if isinstance(value,(list,tuple)):return [type(value).__name__,[normalized(v) for v in value]]
    if isinstance(value,set):return ['set',sorted(normalized(v) for v in value)]
    if isinstance(value,deque):return ['deque',value.maxlen,[normalized(v) for v in value]]
    if type(value) is float:return ['float',value.hex()]
    if type(value) in (int,str,bool) or value is None:return value
    raise TypeError('Unrecognized replay state '+str(type(value)))


def digest(value):return hashlib.sha256(json.dumps(normalized(value),allow_nan=False,separators=(',',':')).encode()).hexdigest()


def execute(parent,geometry,profiled=False,adapter=None):
    launch=read(parent/'launch.json');report=read(parent/'report.json')
    if not report['completed'] or report['full_clip'] or report['frames']!=128 or launch['max_frames']!=128:
        raise ValueError('Only complete frozen 128-frame prefixes')
    clip=Path(launch['source']).stem
    if clip not in ('chunk_0126','chunk_0082'):raise ValueError('Outside development replay scope')
    import tiny_target
    root=Path(tiny_target.__file__).parent
    for name,d in launch['package_sha256'].items():
        if sha(root/name)!=d:raise ValueError('Changed frozen runtime '+name)
    cfg=VisibleConfig(**launch['configuration']);tracker=VisibleTracks(cfg,launch['fps'])
    helper=GeometryV20(geometry);method=helper.adapter(KalmanTrackManager.update)
    if sha(root/'tracking/kalman.py')!=REFERENCE_SHA:raise ValueError('Tracker reference changed')
    if adapter is not None:method=adapter.adapt(method)
    profile=cProfile.Profile();times=[];digests=[];populations=[];index=0
    with patch.object(KalmanTrackManager,'update',method),(parent/'frames.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['frame_index']!=index:raise ValueError('Replay sequence mismatch')
            centers=tracker.learning_centers(row['timestamp_ns'],row['segment'])
            populations.append({p:dict(tracks=len(m._tracks),candidates=sum(c['polarity']==p for c in row['candidates']))
                                for p,m in tracker.managers.items()})
            matrix=np.array(row['source_to_reference'])
            start=time.perf_counter_ns()
            if profiled:profile.enable()
            try:
                tracks,metrics=tracker.update(row['candidates'],index,row['timestamp_ns'],row['segment'],matrix,row['coverage']['full_shape_hw'])
            finally:
                if profiled:profile.disable()
            times.append((time.perf_counter_ns()-start)/1e6)
            if tracks!=row['tracks'] or metrics!=row['tracking_metrics']:
                raise AssertionError(f'Archived tracking output changed at frame {index}')
            state=dict(managers={p:vars(m) for p,m in tracker.managers.items()},extents=tracker.extents,
                qualified=tracker.qualified,summary=tracker.summary,ever_qualified=tracker.ever_qualified,
                previous_records=tracker.previous_records,previous_timestamp_ns=tracker.previous_timestamp_ns,
                quality={k:vars(v) for k,v in tracker.quality.items()})
            digests.append(dict(frame=index,output=digest([tracks,metrics]),state=digest(state),learning=digest(centers)))
            index+=1
    if index!=128:raise ValueError('Incomplete replay')
    return dict(exact=True,frames=index,clip=clip,profiled=profiled,tracking_ms=times,digests=digests,
        geometry_calls=helper.calls,geometry_fallbacks=helper.fallbacks,populations=populations,
        profile=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
            for k,v in pstats.Stats(profile).stats.items()],key=lambda r:-r['self_ms']) if profiled else [],
        journal_sha256=sha(parent/'frames.jsonl'),launch_sha256=sha(parent/'launch.json'),
        report_sha256=sha(parent/'report.json'),geometry_library_sha256=sha(geometry))


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    result=dict(passed=False,error=None,media_read=False,script_sha256=sha(__file__),
        plan_sha256=sha(HERE/'tracking_v27_plan.md'),replay=None,
        warning='Tracking replay only. Profiling/serialization are not clean pipeline FPS or live latency.')
    try:result['replay']=execute(args.parent,args.geometry,args.profile);result['passed']=True
    except BaseException as exc:result['error']=repr(exc);raise
    finally:write(args.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--profile',action='store_true')
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
