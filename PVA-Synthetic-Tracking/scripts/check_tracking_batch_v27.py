"""Generated batch geometry and tracker/state gates, followed by bounded replays."""
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import numpy as np
from profile_visible_v17 import read,sha,write
from tracking_geometry_v20 import GeometryV20,reference
from tracking_batch_v27 import BatchGeometryV27
from check_tracking_geometry_v20 import primitive_cases,same,replay
from replay_tracking_v27 import execute,digest

HERE=Path(__file__).resolve().parent
FILES=('tracking_batch_v27.cpp','tracking_batch_v27.py','build_tracking_batch_v27.py',
       'check_tracking_batch_v27.py','replay_tracking_v27.py','tracking_v27_plan.md','test_tracking_batch_v27.py')


def generated(library,geometry):
    adapter=BatchGeometryV27(library);adapter.fallback=GeometryV20(geometry)
    cases=[]
    for name,values,mean,d,pl,vl in primitive_cases():
        tracks={8:SimpleNamespace(mean=mean),2:SimpleNamespace(mean=mean.copy())}
        before=digest([values,mean]);calls=adapter.calls
        with np.errstate(all='ignore'):
            batch=adapter(values,tracks,d,pl,vl)
            for tid in tracks:same(reference(values,tracks[tid].mean,d,pl,vl),batch.get(tid,values,tracks[tid].mean,d,pl,vl))
        if digest([values,mean])!=before:raise AssertionError('Batch mutated input')
        cases.append(dict(name=name,exact=True,native=adapter.calls>calls))
    rng=np.random.default_rng(202027)
    for t,n,d in ((0,0,2),(0,5,4),(1,0,2),(2,17,4),(256,512,2),(512,512,4),(513,1,2),(1,1025,2)):
        values=rng.normal(size=(n,4));tracks={i:SimpleNamespace(mean=rng.normal(size=4)) for i in range(t)}
        calls=adapter.calls;batch=adapter(values,tracks,d,5.,5.)
        for tid,m in tracks.items():same(reference(values,m.mean,d,5.,5.),batch.get(tid,values,m.mean,d,5.,5.))
        if batch.arrays is not None and t>1 and n:
            a=batch.get(0,values,tracks[0].mean,d,5.,5.)[0]
            b=batch.get(1,values,tracks[1].mean,d,5.,5.)[0]
            if np.shares_memory(a,b):raise AssertionError('Track rows alias')
        cases.append(dict(name=f'capacity_{t}_{n}_{d}',exact=True,native=adapter.calls>calls))
    # Reuse all 12 generated lifecycle/tie/gate/capacity scenarios from v20.
    geo=GeometryV20(geometry);native=BatchGeometryV27(library)
    class Combined:
        def adapter(self,original):return native.adapt(geo.adapter(original))
    replays=replay(Combined())
    return dict(cases=cases,replays=replays,native_calls=adapter.calls,fallbacks=adapter.fallbacks,
                transformed_sha256=native.transformed_sha256)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    record=dict(passed=False,error=None,media_read=False,source_sha256={n:sha(HERE/n) for n in FILES},
        library_sha256=sha(args.library),geometry_library_sha256=sha(args.geometry),build_sha256=sha(args.library.parent/'build.json'),
        generated=None,replays=[],timing_note='Tracker-only replay times exclude JSON parsing, state serialization and comparison; not pipeline FPS.')
    try:
        record['generated']=generated(args.library,args.geometry)
        for clip in ('0126','0082'):
            parent=args.parents/(clip+'_repeat0_reference');baseline=read(HERE/f'profile_{clip}_01.json')
            if not baseline['passed'] or baseline['error'] is not None:raise ValueError('Baseline profile missing')
            for repeat in range(2):
                for mode in (('reference','candidate') if repeat==0 else ('candidate','reference')):
                    adapter=BatchGeometryV27(args.library) if mode=='candidate' else None
                    r=execute(parent,args.geometry,False,adapter)
                    if r['digests']!=baseline['replay']['digests']:raise AssertionError('Complete state/learning replay differs')
                    r.update(repeat=repeat,mode=mode,batch_calls=adapter.calls if adapter else 0,
                        batch_track_rows=adapter.track_rows if adapter else 0,batch_fallbacks=adapter.fallbacks if adapter else 0)
                    record['replays'].append(r)
                    print(clip,repeat,mode,float(np.mean(r['tracking_ms'])),flush=True)
        record['passed']=True
    except BaseException as exc:record['error']=repr(exc);raise
    finally:write(args.output,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
