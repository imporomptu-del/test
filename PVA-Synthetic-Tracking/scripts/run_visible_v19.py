"""Scoped 8-bit integration: v17 baseline; explicit candidate GPU-library transition."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time
from unittest.mock import patch
from profile_visible_v17 import RUNTIME,V13,VISIBLE,read,sha,write
from median_v19 import REFERENCE_LIBRARY_SHA

HERE=Path(__file__).resolve().parent
V17=Path('/tmp/seaqr_visible_speed_v17_EER6lm')


def generated_gate():
    r=read(HERE/'generated_01.json');b=read(HERE/'build/build.json')
    if (not r['passed'] or r['error'] is not None or r['real_media_read'] or len(r['cases'])!=259
            or len(r['timings'])!=24 or not all(v['exact'] and v['inputs_unchanged'] for v in r['cases'])
            or r['build_sha256']!=sha(HERE/'build/build.json') or not b['passed']
            or not b['proof']['passed'] or b['proof']['cases']!=1<<25
            or r['reference_library_sha256']!=REFERENCE_LIBRARY_SHA):
        raise ValueError('Missing complete generated GPU/proof gate')
    for name,digest in b['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Generated source changed: '+name)
    if (sha(HERE/'check_median_v19.py')!=r['script_sha256']
            or sha(HERE/'visible_speed_v19_plan.md')!=r['plan_sha256']
            or sha(HERE/'build/candidate.so')!=r['candidate_library_sha256']
            or b['builds']['candidate']['library_sha256']!=r['candidate_library_sha256']):
        raise ValueError('Generated candidate identity changed')
    return r,b


def prepare():
    gate,build=generated_gate()
    cfg=read(VISIBLE/'config.json')
    if sha(VISIBLE/'config.json')!='b953c3dfd90e095921b177547297da5d26b1bf6da84f8fe59dc3de352a644f37':
        raise ValueError('Unknown base configuration')
    if sha(cfg['cuda_median_library'])!=REFERENCE_LIBRARY_SHA:raise ValueError('Base library changed')
    cfg['cuda_median_library']=str(HERE/'build/candidate.so')
    write(HERE/'candidate_config.json',cfg)
    transition=dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256=REFERENCE_LIBRARY_SHA,
        after_library_sha256=gate['candidate_library_sha256'],candidate_build_sha256=sha(HERE/'build/build.json'))
    write(HERE/'transition.json',transition)
    write(HERE/'integration_freeze.json',dict(config_sha256=sha(HERE/'candidate_config.json'),
        transition_sha256=sha(HERE/'transition.json'),gate_sha256=sha(HERE/'generated_01.json'),
        scripts_sha256={name:sha(HERE/name) for name in ('run_visible_v19.py','video_checks_v19.py','batch_visible_v19.py')}))


def run(args):
    if args.clip not in ('0029','0126','0055','0082') or args.mode not in ('reference','candidate'):
        raise ValueError('Outside authorized visible scope')
    if args.frames not in (None,128) or (args.frames is not None and args.clip not in ('0126','0082')):
        raise ValueError('Outside frozen timing scope')
    if args.output.exists() or args.output.with_suffix('.v19.json').exists():raise FileExistsError(args.output)
    gate,build=generated_gate()
    frozen=read(HERE/'integration_freeze.json')
    for name,digest in frozen['scripts_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Integration harness changed')
    for name,key in (('candidate_config.json','config_sha256'),('transition.json','transition_sha256'),('generated_01.json','gate_sha256')):
        if sha(HERE/name)!=frozen[key]:raise ValueError('Integration freeze changed: '+name)
    import profile_visible_v17 as common
    common.HERE=V17
    sys.path[:0]=[str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    from run_visible_v17 import verify as verify_baseline
    reference,launch=verify_baseline(args.clip)
    from run_motion_video_v13 import run as baseline
    from learning_mask_v17 import LearningMaskV17
    from tiny_target import visible_baseline as visible,visible_resident as resident
    from tiny_target.visible_decode import VisibleFrameReader
    from video_checks_v19 import check
    mask=LearningMaskV17(V17/'build/liblearning_mask_v17.so')
    config=HERE/'candidate_config.json' if args.mode=='candidate' else VISIBLE/'config.json'
    original_run=visible.run
    def configured(source,old_config,output,motion_config,frames):
        if old_config!=VISIBLE/'config.json':raise ValueError('Unexpected config interception')
        return original_run(source,config,output,motion_config,frames)
    intervals=[];previous_read=None
    original_read,original_close=VisibleFrameReader.read,VisibleFrameReader.close
    def read_frame(reader):
        nonlocal previous_read
        now=time.perf_counter()
        if previous_read is not None:intervals.append(1000*(now-previous_read))
        previous_read=now;result=original_read(reader)
        if result[0] is None:previous_read=None
        return result
    def close_reader(reader):
        nonlocal previous_read
        if previous_read is not None:
            intervals.append(1000*(time.perf_counter()-previous_read));previous_read=None
        return original_close(reader)
    record=dict(passed=False,error=None,clip=args.clip,mode=args.mode,frames=args.frames,
        script_sha256=sha(__file__),freeze_sha256=sha(HERE/'integration_freeze.json'),
        gate_sha256=sha(HERE/'generated_01.json'),config_sha256=sha(config),
        v17_gate_sha256=sha(V17/'generated_01.json'),v17_library_sha256=sha(V17/'build/liblearning_mask_v17.so'),
        raw16_accessed=False,defaults_changed=False,noise_v18_enabled=False)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(visible,'run',configured))
            stack.enter_context(patch.object(resident,'shape_learning_mask',mask))
            stack.enter_context(patch.object(VisibleFrameReader,'read',read_frame))
            stack.enter_context(patch.object(VisibleFrameReader,'close',close_reader))
            baseline(argparse.Namespace(branch='visible',clip=args.clip,frames=args.frames,
                                        injected=False,mode='reuse',output=args.output))
        report=read(args.output/'report.json');count=report['frames']
        record['comparison']=check(reference,args.output,count,args.frames is None,args.mode,
                                    read(config),sha(config),read(HERE/'transition.json'))
        if len(intervals)!=count or mask.calls==0:raise AssertionError('Missing timing or v17 mask use')
        record.update(passed=True,processed_frames=count,fps=report['processed_fps'],wall_s=report['elapsed_seconds'])
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:
        record.update(native_mask_calls=mask.calls,consumer_frame_ms=intervals,
            service_time_boundary='Read-entry to next read-entry or last pre-close; includes processing/journaling, not capture-to-alert latency.')
        write(args.output.with_suffix('.v19.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--clip');p.add_argument('--mode')
    p.add_argument('--frames',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.prepare:prepare()
    else:run(a)
