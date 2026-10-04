"""Thread-local preparation attribution over unchanged v24/v20; no tuning."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

V24=Path('/tmp/seaqr_visible_stage_v24_Uo3Tze')
sys.path.insert(0,str(V24))
from run_visible_stage_v24 import V20,V17,V13,RUNTIME,sha,read,write,verify_freeze,validate_snapshot,ensure_output_fresh


class Attribution:
    def __init__(self,binding):self.binding=binding;self.events=[];self.lock=threading.Lock()

    def wrap(self,function,name,worker=True):
        def measured(*a,**kw):
            index=(getattr(self.binding.active,'frame',None) if worker else
                   None if self.binding.current is None else self.binding.current.index)
            if index is None:return function(*a,**kw)  # Initialization/conformance excluded.
            start=time.perf_counter_ns();cpu=time.thread_time_ns();error=None
            try:return function(*a,**kw)
            except BaseException as exc:error=repr(exc);raise
            finally:
                cpu_end=time.thread_time_ns();end=time.perf_counter_ns()
                with self.lock:self.events.append(dict(name=name,frame=index,thread=threading.get_ident(),
                    start_ns=start,end_ns=end,thread_cpu_ns=cpu_end-cpu,error=error))
        return measured


def install_attribution(stack,binding,collect=False):
    from tiny_target import motion,visible_baseline as visible,visible_warp_exact as warp
    from motion_reuse_v12 import ReuseMotionV12
    trace=Attribution(binding);cases=[];arrays={}
    if collect:
        original=motion.fit_global_motion
        def capture(correspondences,config,*a,**kw):
            result=original(correspondences,config,*a,**kw)
            index=correspondences.current_frame_index;key='f'+str(index)
            names=('previous_points','current_points','harris_scores','forward_backward_error_px')
            # Copy after the measured call; collection is outside attributed fit time.
            for name in names:arrays[key+'_'+name]=getattr(correspondences,name).copy()
            from tiny_target.visible_baseline import finite_json
            row={k:v for k,v in vars_for_slots(correspondences).items() if k not in names}
            estimate=finite_json(result.to_dict());estimate.pop('timing_ms')
            arrays[key+'_expected_mask']=result.inlier_mask.copy()
            arrays[key+'_expected_residuals']=result.residuals_px.copy()
            if result.previous_to_current_matrix is not None:
                arrays[key+'_expected_matrix']=result.previous_to_current_matrix.copy()
            cases.append(dict(key=key,correspondence=row,config=asdict(config),expected=estimate))
            return result
        # Timing encloses original only, not the correspondence copy/serialization.
        measured=trace.wrap(original,'global_fit')
        original=measured
        stack.enter_context(patch.object(motion,'fit_global_motion',capture))
    else:
        stack.enter_context(patch.object(motion,'fit_global_motion',trace.wrap(motion.fit_global_motion,'global_fit')))
    for owner,name,label,worker in (
        (ReuseMotionV12,'estimate','correspondence',True),
        (motion.GlobalMotionTracker,'update','composition',True),
        (warp.CudaCubicTranslation,'__call__','warp',True),
        (visible.VisiblePointDetector,'update','detector',False),
        (visible.VisibleTracks,'update','tracking',False)):
        stack.enter_context(patch.object(owner,name,trace.wrap(getattr(owner,name),label,worker)))
    return trace,cases,arrays


def vars_for_slots(value):
    return {name:getattr(value,name) for name in value.__dataclass_fields__}


def run(args):
    if args.clip not in ('0126','0082') or args.mode not in ('reference','staged'):
        raise ValueError('Only frozen development prefixes')
    ensure_output_fresh(args.output);frozen=verify_freeze()
    sys.path[:0]=[str(V20),str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    import numpy as np
    import run_visible_v20 as baseline
    from visible_stage_v24 import StageBinding,install
    binding=StageBinding(args.mode)
    record=dict(passed=False,error=None,clip=args.clip,mode=args.mode,frames=128,
        script_sha256=sha(__file__),plan_sha256=sha(Path(__file__).with_name('visible_native_v25_plan.md')),
        v24_freeze_sha256=sha(V24/'freeze.json'),v24_sources=frozen['files'],
        raw16_accessed=False,defaults_changed=False,algorithm_changed=False,native_enabled=False)
    trace=None
    try:
        with ExitStack() as stack:
            install(stack,binding)
            trace,cases,arrays=install_attribution(stack,binding,collect=args.mode=='reference')
            baseline.run(argparse.Namespace(clip=args.clip,mode='candidate',frames=128,output=args.output))
        old=read(args.output.with_suffix('.v20.json'))
        if not old['passed'] or old['error'] is not None:raise AssertionError('Frozen output parity failed')
        validate_snapshot(binding.snapshot(),128)
        for label in ('warp','detector','tracking'):
            if [r['frame'] for r in trace.events if r['name']==label]!=list(range(128)):
                raise AssertionError('Incomplete attributed frame coverage: '+label)
        if args.mode=='reference':
            file=args.output.with_suffix('.fits.npz')
            with file.open('xb') as out:np.savez_compressed(out,**arrays)
            write(args.output.with_suffix('.fits.json'),dict(cases=cases,npz_sha256=sha(file),
                clip=args.clip,frames=128,source='frozen reference diagnostics only'))
            record['replay_json_sha256']=sha(args.output.with_suffix('.fits.json'))
        record.update(passed=True,baseline_receipt_sha256=sha(args.output.with_suffix('.v20.json')),
                      fps=old['fps'],wall_s=old['wall_s'],processed_frames=128)
    except BaseException as exc:record['error']=repr(exc);raise
    finally:
        record.update(execution=binding.snapshot(),events=[] if trace is None else trace.events,
            warning='Instrumented host wall and calling-thread CPU spans; not clean FPS or GIL causality. '
                    'Reference input collection adds overhead outside fit timing. No media is copied.')
        write(args.output.with_suffix('.attribution.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clip',required=True)
    p.add_argument('--mode',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
