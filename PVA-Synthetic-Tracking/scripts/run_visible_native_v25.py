"""Opt-in native scoring plus frozen staged execution; exact video receipts."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
V24=Path('/tmp/seaqr_visible_stage_v24_Uo3Tze')
sys.path.insert(0,str(V24))
import run_visible_stage_v24 as parent
from run_visible_stage_v24 import sha,read,write,ensure_output_fresh

FROZEN=('native_motion_v25.cpp','native_motion_v25.py','build_native_motion_v25.py',
    'check_native_motion_v25.py','run_visible_native_v25.py','batch_visible_native_v25.py',
    'attribute_visible_native_v25.py','attribute_batch_v25.py','visible_native_v25_plan.md',
    'test_native_motion_v25.py','test_native_motion_v25_attribution.py')
CORE=('native_motion_v25.cpp','native_motion_v25.py','build_native_motion_v25.py','check_native_motion_v25.py')


def generated_gate():
    g=read(HERE/'generated_01.json');b=read(HERE/'build/build.json')
    diagnostics=read(HERE/'attribution_batch.json')
    if not (diagnostics['passed'] and diagnostics['error'] is None and
        [(r['clip'],r['mode']) for r in diagnostics['rows']]==[
            ('0126','reference'),('0126','staged'),('0082','staged'),('0082','reference')]):
        raise AssertionError('Attribution must finish before candidate testing')
    if not (g['passed'] and not g['real_media_read'] and len(g['cases'])==47 and len(g['primitive'])==11
        and all(c['exact'] for c in g['cases']+g['primitive']) and g['native_calls']>0 and g['fallbacks']>0
        and g['passthroughs']==2 and g['independent_outputs'] and g['reentrant_calls']==12
        and g['gil_probe']['passed'] and g['gil_probe']['other_thread_progress_during_call']
        and [r['clip'] for r in g['replays']]==['0126','0082'] and all(r['exact'] and
            [c['frame'] for c in r['cases']]==list(range(1,128)) for r in g['replays'])):
        raise AssertionError('Generated/full correspondence replay gate failed')
    if set(g['source_sha256'])!=set(CORE) or any(sha(HERE/n)!=d for n,d in g['source_sha256'].items()):
        raise AssertionError('Native gate source changed')
    if not (b['returncode']==0 and b['source_sha256']==g['source_sha256']['native_motion_v25.cpp']
        and b['builder_sha256']==g['source_sha256']['build_native_motion_v25.py']
        and b['library_sha256']==g['library_sha256']==sha(HERE/'build/libnative_motion_v25.so')
        and '-fno-fast-math' in b['command'] and '-ffp-contract=off' in b['command']):
        raise AssertionError('Changed/unsafe native build')
    for r in g['replays']:
        p=HERE/f"attribute_{r['clip']}_reference.fits.json"
        if r['json_sha256']!=sha(p) or r['npz_sha256']!=sha(p.with_suffix('.npz')):
            raise AssertionError('Correspondence replay evidence changed')
    parent.verify_freeze()
    return g


def freeze():
    generated_gate()
    if any((HERE/n).exists() for n in ('freeze.json','unit_gate.json','unit_gate.log')):
        raise FileExistsError('Fresh native freeze only')
    command=[sys.executable,'-m','unittest','-v','test_native_motion_v25','test_native_motion_v25_attribution']
    import os
    with (HERE/'unit_gate.log').open('x') as log:
        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=HERE,
                            env=dict(os.environ,PYTHONPATH=str(parent.RUNTIME)))
    write(HERE/'unit_gate.json',dict(passed=done.returncode==0,returncode=done.returncode,command=command,
        log_sha256=sha(HERE/'unit_gate.log'),tests_sha256={n:sha(HERE/n) for n in FROZEN if n.startswith('test_')}))
    if done.returncode:raise RuntimeError('Native unit gate failed')
    write(HERE/'freeze.json',dict(files={n:sha(HERE/n) for n in FROZEN},
        generated_sha256=sha(HERE/'generated_01.json'),unit_gate_sha256=sha(HERE/'unit_gate.json'),
        build_sha256=sha(HERE/'build/build.json'),v24_freeze_sha256=sha(V24/'freeze.json'),
        attribution_batch_sha256=sha(HERE/'attribution_batch.json')))


def verify_freeze():
    g=generated_gate();f=read(HERE/'freeze.json');u=read(HERE/'unit_gate.json')
    if (set(f['files'])!=set(FROZEN) or any(sha(HERE/n)!=d for n,d in f['files'].items())
        or f['generated_sha256']!=sha(HERE/'generated_01.json') or f['build_sha256']!=sha(HERE/'build/build.json')
        or f['unit_gate_sha256']!=sha(HERE/'unit_gate.json') or not u['passed'] or u['returncode']!=0
        or u['log_sha256']!=sha(HERE/'unit_gate.log') or f['v24_freeze_sha256']!=sha(V24/'freeze.json')
        or f['attribution_batch_sha256']!=sha(HERE/'attribution_batch.json')):
        raise AssertionError('Frozen v25 source/gate evidence changed')
    return f,g


def run(args):
    if args.mode not in ('reference','native_staged') or (args.attribute and (args.frames!=128 or args.mode!='native_staged')):
        raise ValueError('Outside frozen native experiment')
    ensure_output_fresh(args.output);f,g=verify_freeze()
    sys.path[:0]=[str(parent.V20),str(parent.V17),str(parent.V13),str(parent.RUNTIME),str(parent.RUNTIME/'scripts')]
    from native_motion_v25 import NativeMotionV25,gm
    from tiny_target import motion
    import visible_stage_v24 as staged
    from attribute_visible_native_v25 import install_attribution
    helper=NativeMotionV25(HERE/'build/libnative_motion_v25.so');adapted=helper.adapter(gm.fit_global_motion)
    if helper.transformed_sha256!=g['transformed_sha256']:raise ValueError('Native transformation changed')
    original_install=staged.install;holder={}
    def install(stack,binding):
        original_install(stack,binding)
        if args.mode=='native_staged':stack.enter_context(patch.object(motion,'fit_global_motion',adapted))
        if args.attribute:holder['trace']=install_attribution(stack,binding)[0]
    record=dict(schema='seaqr.visible-native-v25.v1',passed=False,error=None,clip=args.clip,mode=args.mode,
        frames=args.frames,attributed=args.attribute,source_sha256=f['files'],freeze_sha256=sha(HERE/'freeze.json'),
        generated_sha256=sha(HERE/'generated_01.json'),library_sha256=g['library_sha256'],
        transformed_sha256=helper.transformed_sha256,raw16_accessed=False,defaults_changed=False,
        gpu_changed=False,algorithm_policy_changed=False,production_approved=False)
    try:
        with patch.object(staged,'install',install):
            parent.run(argparse.Namespace(clip=args.clip,mode='staged' if args.mode=='native_staged' else 'reference',
                       frames=args.frames,output=args.output,library=None))
        r=read(args.output.with_suffix('.v24.json'))
        if not r['passed'] or r['error'] is not None:raise AssertionError('Frozen v24/v20 output gate failed')
        if (args.mode=='native_staged')!=(helper.calls>0):raise AssertionError('Native execution missing/unexpected')
        record.update(passed=True,v24_receipt_sha256=sha(args.output.with_suffix('.v24.json')),
                      fps=r['fps'],wall_s=r['wall_s'],processed_frames=r['processed_frames'])
    except BaseException as exc:record['error']=repr(exc);raise
    finally:
        record.update(native_calls=helper.calls,fallbacks=helper.fallbacks,passthroughs=helper.passthroughs,
                      events=holder['trace'].events if 'trace' in holder else [])
        write(args.output.with_suffix('.v25.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--clip');p.add_argument('--mode');p.add_argument('--frames',type=int)
    p.add_argument('--attribute',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();freeze() if a.freeze else run(a)
