"""Explicit GPU/front transition over serial v20; original full output gates."""
import argparse
import importlib
from contextlib import ExitStack
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
V24=Path('/tmp/seaqr_visible_stage_v24_Uo3Tze')
V20=Path('/tmp/seaqr_visible_speed_v20_XUf1LR')
V17=Path('/tmp/seaqr_visible_speed_v17_EER6lm')
V13=Path('/tmp/seaqr_video_v13_IyK7eQ')
RUNTIME=Path('/tmp/seaqr_exact_v9_rS2LFx')
VISIBLE=Path('/tmp/seaqr_phase20_decode_v10_Iz9RSF')
from profile_visible_v17 import sha,read,write
GATE='generated_02.json'
CHECK_MODULES=('compare_phase20_exact_runs','repeat_phase20_kernel_speed','video_checks_v19')
FROZEN=('visible_front_v26.cu','visible_front_v26.py','build_visible_front_v26.py',
    'check_visible_front_v26.py','run_visible_front_v26.py','batch_visible_front_v26.py',
    'visible_front_v26_plan.md','test_visible_front_v26.py')+tuple(n+'.py' for n in CHECK_MODULES)


def verifier_dependencies():
    """Preflight the packaged journal checks before any freeze or media read."""
    sys.path.insert(0,str(HERE))
    for name in CHECK_MODULES:
        module=importlib.import_module(name)
        if sha(module.__file__)!=sha(HERE/(name+'.py')):
            raise ValueError('Journal verification dependency changed: '+name)
    from video_checks_v19 import check
    return check


def generated_gate():
    from visible_front_v26 import CUDA_SOURCES,REFERENCE_LIBRARY_SHA
    g=read(HERE/GATE);b=read(HERE/g['build_relative'])
    if not (g['passed'] and g['error'] is None and not g['real_media_read'] and len(g['noise'])==50
        and len(g['detector'])==130 and all(c['exact'] for c in g['noise']+g['detector'])
        and g['conformance']['exact'] and g['conformance']['warp_cases']==32 and g['conformance']['gaussian_cases']==33):
        raise ValueError('Complete generated front-end gate required')
    for n,d in g['source_sha256'].items():
        if sha(HERE/n)!=d:raise ValueError('Changed generated source '+n)
    if (g['plan_sha256']!=sha(HERE/'visible_front_v26_plan.md') or not b['passed'] or b['returncode']!=0
        or g['build_sha256']!=sha(HERE/g['build_relative'])
        or g['library_sha256']!=b['library_sha256'] or g['reference_library_sha256']!=REFERENCE_LIBRARY_SHA
        or sha(HERE/g['library_relative'])!=g['library_sha256']
        or b['original_sources_sha256']!=CUDA_SOURCES
        or b['adapter_sha256']!=sha(HERE/'visible_front_v26.py')
        or b['builder_sha256']!=sha(HERE/'build_visible_front_v26.py')
        or not all(flag in b['command'] for flag in ('--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'))):
        raise ValueError('Changed native front build')
    for n,d in b['source_sha256'].items():
        if sha((HERE/g['build_relative']).parent/'source'/n)!=d:raise ValueError('Changed compiled source '+n)
    return g


def dependencies():
    sys.path[:0]=[str(V24),str(V20),str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    import run_visible_stage_v24 as stage
    stage.verify_freeze()
    import run_visible_v20 as old
    old.generated_gate();f=read(V20/'freeze.json')
    if any(sha(V20/n)!=d for n,d in f['files'].items()):raise ValueError('Changed v20 source')
    if f['gate_sha256']!=sha(V20/'generated_01.json') or f['build_sha256']!=sha(V20/'build/build.json'):
        raise ValueError('Changed v20 gate')
    import profile_visible_v17 as common
    common.HERE=V17
    verifier_dependencies()
    return stage


def prepare():
    g=generated_gate();dependencies()
    if any((HERE/n).exists() for n in ('freeze.json','candidate_config.json','transition.json','unit_gate.json','unit_gate.log')):
        raise FileExistsError('Fresh front-end freeze only')
    cfg=read(VISIBLE/'config.json')
    if sha(VISIBLE/'config.json')!=g['config_sha256'] or sha(cfg['cuda_median_library'])!=g['reference_library_sha256']:
        raise ValueError('Changed original config/GPU')
    cfg['cuda_median_library']=str(HERE/g['library_relative']);write(HERE/'candidate_config.json',cfg)
    write(HERE/'transition.json',dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256=g['reference_library_sha256'],
        after_library_sha256=g['library_sha256'],candidate_build_sha256=g['build_sha256']))
    command=[sys.executable,'-m','unittest','-v','test_visible_front_v26']
    with (HERE/'unit_gate.log').open('x') as handle:
        done=subprocess.run(command,cwd=HERE,stdout=handle,stderr=subprocess.STDOUT)
    write(HERE/'unit_gate.json',dict(passed=done.returncode==0,returncode=done.returncode,command=command,
        log_sha256=sha(HERE/'unit_gate.log'),test_sha256=sha(HERE/'test_visible_front_v26.py')))
    if done.returncode:raise RuntimeError('Front ownership/learning/source unit gate failed')
    write(HERE/'freeze.json',dict(files={n:sha(HERE/n) for n in FROZEN},gate_sha256=sha(HERE/GATE),gate_file=GATE,
        config_sha256=sha(HERE/'candidate_config.json'),transition_sha256=sha(HERE/'transition.json'),
        unit_gate_sha256=sha(HERE/'unit_gate.json'),v20_freeze_sha256=sha(V20/'freeze.json'),v24_freeze_sha256=sha(V24/'freeze.json')))


def verify_freeze():
    g=generated_gate();f=read(HERE/'freeze.json');dependencies()
    if set(f['files'])!=set(FROZEN) or any(sha(HERE/n)!=d for n,d in f['files'].items()):raise ValueError('Changed front freeze')
    if f['gate_file']!=GATE:raise ValueError('Changed selected gate')
    for key,name in (('gate',GATE),('config','candidate_config.json'),
                     ('transition','transition.json'),('unit_gate','unit_gate.json')):
        if f[key+'_sha256']!=sha(HERE/name):raise ValueError('Changed frozen '+key)
    u=read(HERE/'unit_gate.json')
    if not u['passed'] or u['returncode'] or u['log_sha256']!=sha(HERE/'unit_gate.log'):
        raise ValueError('Changed unit gate')
    if f['v20_freeze_sha256']!=sha(V20/'freeze.json') or f['v24_freeze_sha256']!=sha(V24/'freeze.json'):
        raise ValueError('Changed dependency freeze')
    return f,g


def run(args):
    if (args.clip not in ('0029','0126','0055','0082') or args.mode not in ('reference','candidate')
        or args.frames not in (None,128) or (args.frames and args.clip not in ('0126','0082'))):
        raise ValueError('Outside frozen development scope')
    if args.output.exists() or args.output.with_suffix('.v26.json').exists():raise FileExistsError(args.output)
    f,g=verify_freeze()
    from run_visible_v17 import verify as verify_baseline
    reference,_=verify_baseline(args.clip)
    from run_motion_video_v13 import run as baseline
    from visible_stage_v24 import StageBinding,install
    from run_visible_stage_v24 import validate_snapshot
    from learning_mask_v17 import LearningMaskV17
    from tracking_geometry_v20 import GeometryV20
    from tiny_target.tracking.kalman import KalmanTrackManager
    from tiny_target import visible_baseline as visible,visible_resident as resident,visible_warp_exact as warp
    from video_checks_v19 import check
    from visible_front_v26 import ResidentFrontV26,attach_warp
    mask=LearningMaskV17(V17/'build/liblearning_mask_v17.so')
    geometry=GeometryV20(V20/'build/libtracking_geometry_v20.so');method=geometry.adapter(KalmanTrackManager.update)
    if geometry.transformed_sha256!=read(V20/'generated_01.json')['transformed_sha256']:raise ValueError('Changed geometry transform')
    config=HERE/'candidate_config.json' if args.mode=='candidate' else VISIBLE/'config.json'
    original_run=visible.run
    def configured(source,old_config,output,motion_config,frames):
        if old_config!=VISIBLE/'config.json':raise ValueError('Unexpected config interception')
        return original_run(source,config,output,motion_config,frames)
    binding=StageBinding('reference');instances=[];intervals=[];last=None
    class CapturedFront(ResidentFrontV26):
        def __init__(self,cfg):super().__init__(cfg);instances.append(self)
    r=dict(schema='seaqr.visible-front-v26.v1',passed=False,error=None,clip=args.clip,mode=args.mode,frames=args.frames,
        source_sha256=f['files'],freeze_sha256=sha(HERE/'freeze.json'),gate_sha256=sha(HERE/GATE),
        config_sha256=sha(config),transition_sha256=sha(HERE/'transition.json'),library_sha256=sha(read(config)['cuda_median_library']),
        v20_freeze_sha256=sha(V20/'freeze.json'),v24_freeze_sha256=sha(V24/'freeze.json'),
        v17_gate_sha256=sha(V17/'generated_01.json'),v17_library_sha256=sha(V17/'build/liblearning_mask_v17.so'),
        geometry_library_sha256=sha(V20/'build/libtracking_geometry_v20.so'),geometry_transformed_sha256=geometry.transformed_sha256,
        execution_policy='serial_reference',gpu_changed=args.mode=='candidate',native_front_enabled=args.mode=='candidate',
        raw16_accessed=False,defaults_changed=False,production_approved=False,new_accuracy_validated=False,
        staged_v24_enabled=False,native_motion_v25_enabled=False)
    try:
        with ExitStack() as stack:
            install(stack,binding)
            from tiny_target.visible_decode import VisibleFrameReader
            old_read,old_close=VisibleFrameReader.read,VisibleFrameReader.close
            def read_frame(reader):
                nonlocal last
                now=time.perf_counter()
                if last is not None:intervals.append(1000*(now-last))
                last=now;value=old_read(reader)
                if value[0] is None:last=None
                return value
            def close_reader(reader):
                nonlocal last
                if last is not None:intervals.append(1000*(time.perf_counter()-last));last=None
                return old_close(reader)
            stack.enter_context(patch.object(VisibleFrameReader,'read',read_frame))
            stack.enter_context(patch.object(VisibleFrameReader,'close',close_reader))
            stack.enter_context(patch.object(visible,'run',configured))
            stack.enter_context(patch.object(resident,'shape_learning_mask',mask))
            stack.enter_context(patch.object(KalmanTrackManager,'update',method))
            if args.mode=='candidate':
                stack.enter_context(patch.object(resident,'VisibleCudaResident',CapturedFront))
                stack.enter_context(patch.object(warp.CudaCubicTranslation,'__call__',attach_warp(warp.CudaCubicTranslation.__call__)))
            baseline(argparse.Namespace(branch='visible',clip=args.clip,frames=args.frames,injected=False,mode='reuse',output=args.output))
        report=read(args.output/'report.json');count=report['frames']
        r['comparison']=check(reference,args.output,count,args.frames is None,args.mode,read(config),sha(config),read(HERE/'transition.json'))
        validate_snapshot(binding.snapshot(),count)
        if len(intervals)!=count or geometry.calls==0:raise AssertionError('Missing frame timing/v20 geometry')
        if args.mode=='candidate':
            if (len(instances)!=1 or instances[0].calls!=count or instances[0].device_calls!=count
                or instances[0].finish_calls!=count or instances[0].host_calls or instances[0].handle
                or instances[0].front or mask.calls):raise AssertionError('Incomplete front ABI/lifecycle or unexpected CPU learning')
        elif instances or mask.calls==0:raise AssertionError('Reference execution changed')
        r.update(passed=True,processed_frames=count,fps=report['processed_fps'],wall_s=report['elapsed_seconds'])
    except BaseException as exc:r['error']=repr(exc);raise
    finally:
        r.update(consumer_frame_ms=intervals,execution=binding.snapshot(),geometry_calls=geometry.calls,
            geometry_fallbacks=geometry.fallbacks,native_mask_calls=mask.calls,
            fronts=[dict(calls=x.calls,device_calls=x.device_calls,host_calls=x.host_calls,finish_calls=x.finish_calls,
                         learning_points=x.learning_points,closed=x.front is None and x.handle is None) for x in instances],
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output.with_suffix('.v26.json'),r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--clip');p.add_argument('--mode');p.add_argument('--frames',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args();prepare() if a.prepare else run(a)
