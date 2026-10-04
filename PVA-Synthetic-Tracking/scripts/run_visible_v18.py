"""8-bit-only integration: v17 mask enabled in both arms; noise is the variable."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
from profile_visible_v17 import HERE, RUNTIME, V13, read, sha, write

V17 = Path('/tmp/seaqr_visible_speed_v17_EER6lm')


def verify_gate():
    gate = read(HERE/'generated_01.json')
    if (not gate['passed'] or gate['error'] or gate['real_media_read']
            or len(gate['cases']) != 354 or len(gate['timings']) != 8
            or not all(r['exact'] and r['inputs_unchanged'] for r in gate['cases'])
            or {r['native'] for r in gate['cases']} != {True, False}):
        raise ValueError('Incomplete generated noise gate')
    for name, digest in gate['source_sha256'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Generated noise source changed: '+name)
    if sha(HERE/'build/libnoise_v18.so') != gate['library_sha256']:
        raise ValueError('Generated noise library changed')
    if sha(HERE/'visible_speed_v18_plan.md') != gate['plan_sha256']:
        raise ValueError('Frozen plan changed')
    return gate


def run(args):
    # Imported v17 helper must keep its own dependency directory, not this one.
    import profile_visible_v17 as common
    common.HERE = V17
    sys.path[:0] = [str(V17), str(V13), str(RUNTIME), str(RUNTIME/'scripts')]
    from run_visible_v17 import run as previous, scope
    from tiny_target import visible_resident as resident
    from noise_v18 import NoiseV18, reference
    scope(args.clip,args.mode,args.frames)
    if args.output.exists() or args.output.with_suffix('.v18.json').exists():
        raise FileExistsError(args.output)
    gate = verify_gate()
    candidate = NoiseV18(HERE/'build/libnoise_v18.so')
    function = candidate if args.mode=='candidate' else reference
    durations, identities = [], []
    def noise(*a, **kw):
        start = time.perf_counter()
        result = function(*a, **kw)
        durations.append(1000*(time.perf_counter()-start))
        digest = hashlib.sha256(result[0].tobytes()+np.float64(result[1]).tobytes()).hexdigest()
        identities.append(digest)
        return result
    record = dict(passed=False,error=None,clip=args.clip,mode=args.mode,frames=args.frames,
        script_sha256=sha(__file__),plan_sha256=gate['plan_sha256'],gate_sha256=sha(HERE/'generated_01.json'),
        library_sha256=gate['library_sha256'],v17_wrapper_sha256=sha(V17/'run_visible_v17.py'),
        raw16_accessed=False,defaults_changed=False)
    try:
        with patch.object(resident,'tile_noise_statistics',noise):
            previous(argparse.Namespace(clip=args.clip,mode='candidate',frames=args.frames,output=args.output))
        old=read(args.output.with_suffix('.v17.json'))
        if not old['passed'] or old['error'] or old['native_mask_calls']<=0:
            raise AssertionError('v17 regression/mask baseline failed')
        if args.mode=='candidate' and candidate.calls==0:
            raise AssertionError('Native noise not exercised')
        if len(identities)!=old['processed_frames']:
            raise AssertionError('Noise not called for every frame')
        record.update(passed=True,processed_frames=old['processed_frames'],fps=old['fps'],wall_s=old['wall_s'],
                      v17_receipt_sha256=sha(args.output.with_suffix('.v17.json')))
    except BaseException as exc:
        record['error']=repr(exc)
        raise
    finally:
        record.update(native_noise_calls=candidate.calls,fallback_calls=candidate.fallbacks,
                      geometry_builds=candidate.geometry_builds,noise_ms=durations,noise_identities=identities)
        write(args.output.with_suffix('.v18.json'),record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip',required=True);p.add_argument('--mode',required=True)
    p.add_argument('--frames',type=int);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
