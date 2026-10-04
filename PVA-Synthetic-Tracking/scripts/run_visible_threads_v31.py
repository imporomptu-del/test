"""New child only: verify loaded OpenBLAS threads then run untouched v29."""
import argparse
import os
from pathlib import Path
import sys

from profile_visible_interaction_v30 import V29, read, sha, write, runtime_info
from visible_threads_v31 import policy, scope, KEYS

HERE = Path(__file__).resolve().parent
V30 = Path('/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ')


def verify_freeze():
    frozen = read(HERE/'freeze.json')
    for name, digest in frozen['sources'].items():
        if sha(HERE/name) != digest:
            raise ValueError('Changed v31 frozen source '+name)
    if frozen['baseline_freeze_sha256'] != sha(V29/'freeze.json') or frozen['diagnostic_batch_sha256'] != sha(V30/'batch.json'):
        raise ValueError('Changed baseline or diagnostic selection evidence')
    return frozen


def validate_runtime(info, mode):
    expected = 1 if policy(mode)[1] == 'one' else 12
    if len(info['blas']) != 1 or info['blas'][0]['threads'] != expected:
        raise ValueError('Unexpected loaded numerical-library worker count')
    setting = {k: os.environ.get(k) for k in KEYS}
    target = dict.fromkeys(KEYS)
    if expected == 1:
        target['OPENBLAS_NUM_THREADS'] = '1'
    if setting != target:
        raise ValueError('Numerical environment differs from declared arm')


def run(args):
    scope(args.clip,args.mode,args.frames,args.audit,args.traced)
    if args.output.exists() or args.output.with_suffix('.v31.json').exists():
        raise FileExistsError(args.output)
    frozen = verify_freeze()
    sys.path.insert(0,str(V29))
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    arm, setting = policy(args.mode)
    before = runtime_info()
    validate_runtime(before,args.mode)
    result = dict(passed=False,error=None,clip=args.clip,mode=args.mode,arm=arm,
        thread_policy=setting,frames=args.frames,audit=args.audit,traced=args.traced,
        freeze_sha256=sha(HERE/'freeze.json'),runtime_before=before,
        raw16_accessed=False,defaults_changed=False,algorithm_changed=False)
    try:
        if args.traced:
            import profile_visible_interaction_v30 as profiler
            profiler.run(args.clip,arm,args.output)
        else:
            baseline.run(argparse.Namespace(clip=args.clip,arm=arm,frames=args.frames,state_audit=args.audit,output=args.output))
        after = runtime_info()
        validate_runtime(after,args.mode)
        r = read(args.output.with_suffix('.v29.json'))
        if not r['passed'] or r['error'] is not None:
            raise AssertionError('Frozen output check failed')
        result.update(passed=True,runtime_after=after,receipt_sha256=sha(args.output.with_suffix('.v29.json')),
                      fps=r['fps'],wall_s=r['wall_s'],count=r['processed_frames'])
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        write(args.output.with_suffix('.v31.json'),result)


def generated():
    verify_freeze()
    sys.path[:0] = [str(V29),'/tmp/seaqr_tracking_v28_Nn629D','/tmp/seaqr_exact_v9_rS2LFx']
    from run_visible_combined_v29 import verify_freeze as old_verify
    old_verify()
    from check_tracking_stage_v28 import generated as check
    validate_runtime(runtime_info(),'combined')
    actual = check(Path('/tmp/seaqr_tracking_v27_pmUXGZ/build_01/libtracking_batch_v27.so'),
                   Path('/tmp/seaqr_visible_speed_v20_XUf1LR/build/libtracking_geometry_v20.so'))
    expected = read('/tmp/seaqr_tracking_v28_Nn629D/replays_01.json')['generated']
    passed = actual == expected
    write(HERE/'generated.json',dict(passed=passed,actual=actual,expected_sha256=sha('/tmp/seaqr_tracking_v28_Nn629D/replays_01.json'),
        runtime=runtime_info(),media_read=False,freeze_sha256=sha(HERE/'freeze.json')))
    if not passed:
        raise AssertionError('Generated state/output differs under new thread policy')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated',action='store_true')
    p.add_argument('--clip'); p.add_argument('--mode'); p.add_argument('--frames',type=int)
    p.add_argument('--audit',action='store_true'); p.add_argument('--traced',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    generated() if a.generated else run(a)
