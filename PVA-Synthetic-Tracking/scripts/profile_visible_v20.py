"""Bounded current-v17 tracking profile; no clean FPS claim."""
import argparse
from contextlib import ExitStack
import cProfile
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch
import profile_visible_v17 as common
from profile_visible_v17 import read,sha,write,RUNTIME,V13
HERE=Path(__file__).resolve().parent
V17=Path('/tmp/seaqr_visible_speed_v17_EER6lm')


def run(clip,output):
    if clip not in ('0126','0082'):raise ValueError('Only frozen timing prefixes')
    if output.exists() or output.with_suffix('.profile.json').exists():raise FileExistsError(output)
    common.HERE=V17
    sys.path[:0]=[str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    from run_visible_v17 import run as baseline
    from tiny_target import visible_baseline as visible,visible_resident as resident,visible_warp_exact as warp
    from tiny_target.tracking.kalman import KalmanTrackManager
    profile=cProfile.Profile();rows={};counts=[]
    record=dict(passed=False,error=None,clip=clip,frames=128,raw16_accessed=False,defaults_changed=False,
        script_sha256=sha(__file__),plan_sha256=sha(HERE/'visible_speed_v20_plan.md'),
        warning='Tracking-only cProfile adds overhead; host accelerator calls include work and waits, not isolated synchronization. Nested times overlap. Not benchmark FPS.')
    try:
        with ExitStack() as stack:
            def wrap(owner,name,label,profiling=False):
                original=getattr(owner,name)
                def timed(*a,**kw):
                    start=time.perf_counter()
                    if profiling:profile.enable()
                    try:return original(*a,**kw)
                    finally:
                        if profiling:profile.disable()
                        rows.setdefault(label,[]).append(1000*(time.perf_counter()-start))
                stack.enter_context(patch.object(owner,name,timed))
            original_update=KalmanTrackManager.update
            def counted(manager,batch,**kw):
                counts.append(dict(tracks=len(manager._tracks),candidates=len(batch.candidates)))
                return original_update(manager,batch,**kw)
            stack.enter_context(patch.object(KalmanTrackManager,'update',counted))
            original_init=resident.VisibleCudaResident.__init__
            def init(instance,*a,**kw):
                original_init(instance,*a,**kw)
                for name in ('select','patches','finish'):wrap(instance.lib,'seaqr_resident_'+name,'native.'+name)
            stack.enter_context(patch.object(resident.VisibleCudaResident,'__init__',init))
            wrap(visible.VisibleTracks,'update','tracks.total',True)
            wrap(warp.CudaWarpFrame,'prepare','native.prepare_warp')
            wrap(warp.CudaCubicTranslation,'gaussian','warp.gaussian')
            baseline(argparse.Namespace(clip=clip,mode='candidate',frames=128,output=output))
        r=read(output.with_suffix('.v17.json'))
        if not r['passed']:raise AssertionError('Baseline parity failed')
        record.update(passed=True,baseline_receipt_sha256=sha(output.with_suffix('.v17.json')))
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:
        profile.disable()
        record['calls']={k:dict(calls=len(v),mean_ms=sum(v)/len(v),samples_ms=v) for k,v in rows.items()}
        record['tracker_populations']=counts
        record['python_profile']=sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],
            self_ms=1000*v[2],cumulative_ms=1000*v[3]) for k,v in pstats.Stats(profile).stats.items()],key=lambda r:-r['self_ms'])
        write(output.with_suffix('.profile.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.clip,a.output)
