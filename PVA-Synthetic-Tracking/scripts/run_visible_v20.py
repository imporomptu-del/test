"""v17 baseline plus opt-in exact CPU geometry; original GPU/config unchanged."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
from unittest.mock import patch
import profile_visible_v17 as common
from profile_visible_v17 import read,sha,write,RUNTIME,V13
from tracking_geometry_v20 import GeometryV20,REFERENCE_SHA
HERE=Path(__file__).resolve().parent
V17=Path('/tmp/seaqr_visible_speed_v17_EER6lm')
FROZEN=('run_visible_v20.py','batch_visible_v20.py','tracking_geometry_v20.py',
        'tracking_geometry_v20.cpp','build_tracking_geometry_v20.py','check_tracking_geometry_v20.py',
        'profile_visible_v17.py','visible_speed_v20_plan.md')


def generated_gate():
    g=read(HERE/'generated_01.json');b=read(HERE/'build/build.json')
    if (not g['passed'] or g['error'] is not None or g['real_media_read'] or len(g['cases'])!=136
        or not all(c['exact'] and c['inputs_unchanged'] for c in g['cases'])
        or not any(c['native'] for c in g['cases']) or not any(not c['native'] for c in g['cases'])
        or len(g['replays'])!=12 or not all(r['exact'] and r['frames']==32 for r in g['replays'])
        or len(g['timings'])!=8 or g['reference_sha256']!=REFERENCE_SHA or b['returncode']!=0):
        raise ValueError('Complete generated geometry/replay gate required')
    for name,digest in g['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Generated source changed: '+name)
    if (g['library_sha256']!=sha(HERE/'build/libtracking_geometry_v20.so') or g['library_sha256']!=b['library_sha256']
        or b['source_sha256']!=sha(HERE/'tracking_geometry_v20.cpp')
        or b['builder_sha256']!=sha(HERE/'build_tracking_geometry_v20.py')):raise ValueError('Build identity changed')
    return g


def freeze():
    generated_gate()
    write(HERE/'freeze.json',dict(files={name:sha(HERE/name) for name in FROZEN},
        gate_sha256=sha(HERE/'generated_01.json'),build_sha256=sha(HERE/'build/build.json')))


def run(args):
    if args.clip not in ('0029','0126','0055','0082') or args.mode not in ('reference','candidate'):
        raise ValueError('Outside authorized scope')
    if args.frames not in (128,None) or (args.frames and args.clip not in ('0126','0082')):raise ValueError('Frozen timing scope')
    if args.output.exists() or args.output.with_suffix('.v20.json').exists():raise FileExistsError(args.output)
    g=generated_gate();f=read(HERE/'freeze.json')
    if set(f['files'])!=set(FROZEN) or any(sha(HERE/n)!=d for n,d in f['files'].items()):raise ValueError('Frozen source changed')
    if f['gate_sha256']!=sha(HERE/'generated_01.json') or f['build_sha256']!=sha(HERE/'build/build.json'):raise ValueError('Frozen gate changed')
    common.HERE=V17
    sys.path[:0]=[str(V17),str(V13),str(RUNTIME),str(RUNTIME/'scripts')]
    from run_visible_v17 import run as baseline
    from tiny_target.tracking.kalman import KalmanTrackManager
    helper=GeometryV20(HERE/'build/libtracking_geometry_v20.so');method=helper.adapter(KalmanTrackManager.update)
    if helper.transformed_sha256!=g['transformed_sha256']:raise ValueError('Tracker transformation changed')
    record=dict(passed=False,error=None,clip=args.clip,mode=args.mode,frames=args.frames,
        script_sha256=sha(__file__),freeze_sha256=sha(HERE/'freeze.json'),gate_sha256=sha(HERE/'generated_01.json'),
        library_sha256=g['library_sha256'],transformed_sha256=helper.transformed_sha256,
        raw16_accessed=False,defaults_changed=False,gpu_changed=False,noise_v18_enabled=False,median_v19_enabled=False)
    try:
        context=patch.object(KalmanTrackManager,'update',method) if args.mode=='candidate' else nullcontext()
        with context:baseline(argparse.Namespace(clip=args.clip,mode='candidate',frames=args.frames,output=args.output))
        old=read(args.output.with_suffix('.v17.json'))
        if not old['passed'] or old['error'] is not None:raise AssertionError('Complete baseline parity failed')
        if (args.mode=='candidate') != (helper.calls>0):raise AssertionError('Unexpected candidate execution')
        record.update(passed=True,baseline_receipt_sha256=sha(args.output.with_suffix('.v17.json')),
            fps=old['fps'],wall_s=old['wall_s'],processed_frames=old['processed_frames'])
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:
        record.update(native_geometry_calls=helper.calls,geometry_fallbacks=helper.fallbacks)
        write(args.output.with_suffix('.v20.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--clip');p.add_argument('--mode');p.add_argument('--frames',type=int);p.add_argument('--output',type=Path)
    a=p.parse_args();freeze() if a.freeze else run(a)
