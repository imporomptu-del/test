"""Unprivileged full-video wrapper over frozen v29 and v30 implementations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import visible_validation_v34 as control

HERE=Path(__file__).resolve().parent
V29=Path('/tmp/seaqr_visible_combined_v29_s8XhL1')
V30=Path('/tmp/seaqr_visible_interaction_v30_retry_UpdrpJ')
V13=Path('/tmp/seaqr_video_v13_IyK7eQ')
RUNTIME_KEYS=('blas','affinity','numpy','opencv','thread_environment','clock_ticks')


def request(trial, output):
    matches=[s for s in control.schedule() if s['name']==trial]
    if len(matches)!=1:
        raise ValueError('Only frozen v34 trials')
    if output.resolve()!=HERE/'run'/trial or output.is_symlink():
        raise ValueError('Output outside isolated trial location')
    if output.exists() or output.with_suffix('.v34.json').exists():
        raise FileExistsError('Existing trial evidence')
    return matches[0]


def validate_runtime(actual, expected, after=False):
    if any(actual[k]!=expected[k] for k in RUNTIME_KEYS):
        raise ValueError('Numerical runtime identity/policy changed')
    if len(actual['blas'])!=1 or actual['blas'][0]['threads']!=12:
        raise ValueError('Expected inherited 12-thread BLAS runtime')
    if any(v is not None for v in actual['thread_environment'].values()):
        raise ValueError('Numerical thread environment must remain unset')
    if after and actual['opencv_threads']!=2:
        raise ValueError('OpenCV worker policy changed')


def dependencies():
    if os.geteuid()==0:
        raise PermissionError('Video/dependency work must never run as root')
    sys.path[:0]=[str(control.V31),str(V29)]
    from run_visible_threads_v31 import verify_freeze
    verify_freeze()
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    from profile_visible_interaction_v30 import runtime_info
    from run_visible_v17 import verify
    refs={}
    for c,n in control.COUNTS.items():
        path,launch=verify(c)
        report=control.read(path/'report.json')
        if report['frames']!=n or not report['completed'] or not report['full_clip']:
            raise ValueError('Changed archived full development reference '+c)
        if launch['fps']!=10:
            raise ValueError('Changed nominal video rate '+c)
        refs[c]=dict(frames=n,nominal_file_fps=launch['fps'],source_sha256=launch['source_sha256'],
            launch_sha256=control.sha(path/'launch.json'),report_sha256=control.sha(path/'report.json'))
    # The existing profiler is copied into v31's frozen source set. It verifies
    # the existing NVTX bridge again when actually invoked. No installation.
    import profile_visible_interaction_v30 as profiler
    if control.sha(profiler.__file__)!=control.read(control.V31/'freeze.json')['sources']['profile_visible_interaction_v30.py']:
        raise ValueError('Changed frozen profiler')
    return baseline,profiler,runtime_info,refs


def dependency_preflight():
    control.check_v33_audit(HERE)
    _,_,runtime_info,refs=dependencies()
    before=runtime_info()
    expected=control.read(control.V33/'run/0126_repeat0_fixed_combined_default.v31.json')['runtime_before']
    validate_runtime(before,expected)
    nsys=shutil.which('nsys')
    if nsys is None:
        raise FileNotFoundError('Existing nsys installation required; do not install automatically')
    version=subprocess.run([nsys,'--version'],capture_output=True,text=True,check=True)
    return dict(passed=True,media_accessed=False,settings_changed=False,references=refs,
        runtime=before,nsys=dict(path=nsys,version=version.stdout.strip()))


def run(trial,output):
    spec=request(trial,output)  # scope checked before loading any video runtime
    frozen=control.verify_freeze(HERE)
    baseline,profiler,runtime_info,refs=dependencies()
    before=runtime_info(); validate_runtime(before,frozen['runtime_reference'])
    r=dict(passed=False,error=None,**spec,count=None,fps=None,wall_s=None,
        freeze_sha256=control.sha(HERE/'freeze.json'),runtime_before=before,
        references=refs,raw16_accessed=False,defaults_changed=False,algorithm_changed=False)
    try:
        if spec['traced']:
            profiler.run(spec['clip'],'combined',output)
        else:
            baseline.run(argparse.Namespace(clip=spec['clip'],arm='combined',frames=spec['frames'],
                state_audit=spec['audit'],output=output))
        after=runtime_info(); validate_runtime(after,frozen['runtime_reference'],after=True)
        old=control.read(output.with_suffix('.v29.json'))
        if not old['passed'] or old['error'] is not None or old['processed_frames']!=control.expected_count(spec):
            raise AssertionError('Frozen exact-output/full-frame gate failed')
        r.update(passed=True,runtime_after=after,receipt_sha256=control.sha(output.with_suffix('.v29.json')),
            fps=old['fps'],wall_s=old['wall_s'],count=old['processed_frames'])
        if spec['traced']:
            r['trace_sha256']=control.sha(output.with_suffix('.trace30.json'))
    except BaseException as exc:
        r['error']=repr(exc)
        raise
    finally:
        control.write(output.with_suffix('.v34.json'),r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true'); p.add_argument('--trial'); p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.preflight:
        if a.trial or a.output:p.error('Preflight does not accept a trial/output')
        print(json.dumps(dependency_preflight(),indent=2))
    else:
        if not a.trial or a.output is None:p.error('Trial/output required')
        run(a.trial,a.output)
