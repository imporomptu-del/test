"""Unprivileged, scoring-only experiment over the frozen serial v29 harness.

Parent computations/checks are unmodified. Its final serialization is explicitly
relabeled as v35base because the old receipt hardcodes native scoring as disabled.
No v29 receipt is written or passed off as an unchanged v29 execution.
"""
import argparse
from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
import sys
from unittest.mock import patch

import visible_native_serial_v35 as control

HERE=Path(__file__).resolve().parent
V29=Path('/tmp/seaqr_visible_combined_v29_s8XhL1')
RUNTIME_KEYS=('blas','affinity','numpy','opencv','thread_environment','clock_ticks')


def request(trial,output):
    matches=[s for s in control.schedule() if s['name']==trial]
    if len(matches)!=1:
        raise ValueError('Only frozen v35 trials')
    if output.resolve()!=HERE/'run'/trial or output.is_symlink() or output.parent.is_symlink():
        raise ValueError('Output outside isolated trial location')
    if output.exists() or any(output.with_suffix(s).exists() for s in ('.v35.json','.v35base.json','.v29.json')):
        raise FileExistsError('Existing trial evidence')
    return matches[0]


def validate_runtime(actual,expected,after=False):
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
    sys.path[:0]=[str(control.V31),str(V29),str(control.V25)]
    from run_visible_threads_v31 import verify_freeze
    verify_freeze()
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    import run_visible_native_v25 as native
    native.verify_freeze()
    from profile_visible_interaction_v30 import runtime_info
    from native_motion_v25 import NativeMotionV25,gm
    from tiny_target import motion
    if motion.fit_global_motion is not gm.fit_global_motion:
        raise ValueError('Unexpected pre-existing motion patch')
    from run_visible_v17 import verify
    refs={}
    for c,n in control.COUNTS.items():
        path,launch=verify(c)
        report=control.read(path/'report.json')
        if report['frames']!=n or not report['completed'] or not report['full_clip'] or launch['fps']!=10:
            raise ValueError('Changed archived full development reference '+c)
        refs[c]=dict(frames=n,nominal_file_fps=launch['fps'],source_sha256=launch['source_sha256'],
            launch_sha256=control.sha(path/'launch.json'),report_sha256=control.sha(path/'report.json'))
    helper=NativeMotionV25(control.V25/'build/libnative_motion_v25.so')
    adapted=helper.adapter(gm.fit_global_motion)
    if helper.transformed_sha256!=control.read(control.V25/'generated_01.json')['transformed_sha256']:
        raise ValueError('Native transformation changed')
    return baseline,runtime_info,refs,motion,helper,adapted


def dependency_preflight():
    control.check_v34_audit(HERE)
    _,runtime_info,refs,_,_,_=dependencies()
    before=runtime_info()
    expected=control.read(control.V34/'run/full_repeat0_0126.v34.json')['runtime_before']
    validate_runtime(before,expected)
    import check_native_motion_v25 as checker
    # Reuse the frozen binary. Re-run exact generated and captured-correspondence
    # checks; no source videos, compilation, detector tuning or clocks involved.
    log=io.StringIO()
    with redirect_stdout(log):
        checker.main(argparse.Namespace(library=control.V25/'build/libnative_motion_v25.so',
            replay=[control.V25/f'attribute_{c}_reference.fits.json' for c in ('0126','0082')],
            output=HERE/'generated_01.json'))
    control.check_generated(HERE)
    after=runtime_info(); validate_runtime(after,expected)
    return dict(passed=True,source_video_decoded=False,settings_changed=False,references=refs,
        captured_correspondences_replayed=254,runtime=before,runtime_after=after,
        generated_sha256=control.sha(HERE/'generated_01.json'),generated_stdout=log.getvalue())


class ReceiptAdapter:
    """One explicit metadata adaptation; preserve every computational check."""
    def __init__(self,output,native,provenance):
        self.output=output; self.native=native; self.provenance=provenance; self.calls=0

    def __call__(self,path,value):
        if (self.calls or Path(path)!=self.output.with_suffix('.v29.json')
                or value['schema']!='seaqr.visible-combined-v29.v1'
                or value['native_motion_v25_enabled'] is not False
                or value['staged_v24_enabled'] is not False
                or value['execution_policy']!='serial_reference'):
            raise ValueError('Unexpected parent receipt interception')
        if self.output.with_suffix('.v29.json').exists():
            raise FileExistsError('Misleading v29 receipt already exists')
        new=copy.deepcopy(value)
        new.update(schema='seaqr.visible-native-serial-base-v35.v1',
            parent_schema=value['schema'],native_motion_v25_enabled=self.native,
            receipt_adapter=dict(self.provenance))
        control.write(self.output.with_suffix('.v35base.json'),new)
        self.calls+=1


def validate_calls(spec,count,fit_calls,helper):
    if fit_calls!=count-1 or helper.fallbacks or helper.passthroughs:
        raise AssertionError('Unexpected motion coverage/fallback/pass-through')
    if spec['arm']=='native':
        # Exact reference quality gates can return before hypothesis scoring.
        if not 0<helper.calls<=fit_calls:
            raise AssertionError('Native scorer unused or called more than once per fit')
    elif helper.calls:
        raise AssertionError('Reference arm executed native scoring')


def run(trial,output):
    spec=request(trial,output)  # before imports that can load a media runtime
    frozen=control.verify_freeze(HERE)
    if spec['kind']=='full':
        gate=control.read(HERE/'run/performance_gate.json')
        if not gate['passed'] or gate!=control.performance_gate(HERE/'run'):
            raise ValueError('Full replay requires an unchanged passing prefix gate')
    baseline,runtime_info,refs,motion,helper,adapted=dependencies()
    before=runtime_info(); validate_runtime(before,frozen['runtime_reference'])
    original=motion.fit_global_motion; old_write=baseline.write; fit_calls=0
    native=spec['arm']=='native'
    selected=adapted if native else original
    def counted(*args,**kwargs):
        nonlocal fit_calls
        fit_calls+=1
        return selected(*args,**kwargs)
    provenance=dict(wrapper_sha256=control.sha(__file__),
        parent_source_sha256=control.sha(baseline.__file__),
        parent_freeze_sha256=control.sha(V29/'freeze.json'),
        v25_freeze_sha256=control.sha(control.V25/'freeze.json'),
        library_sha256=control.sha(control.V25/'build/libnative_motion_v25.so'),
        transformed_sha256=helper.transformed_sha256)
    adapter=ReceiptAdapter(output,native,provenance)
    r=dict(schema='seaqr.visible-native-serial-v35.v1',passed=False,error=None,**spec,
        count=None,fps=None,wall_s=None,freeze_sha256=control.sha(HERE/'freeze.json'),
        generated_sha256=control.sha(HERE/'generated_01.json'),runtime_before=before,references=refs,
        implementation=provenance,raw16_accessed=False,defaults_changed=False,
        algorithm_policy_changed=False,staged_v24_enabled=False,new_accuracy_validated=False,
        production_approved=False)
    try:
        with patch.object(motion,'fit_global_motion',counted),patch.object(baseline,'write',adapter):
            baseline.run(argparse.Namespace(clip=spec['clip'],arm='combined',frames=spec['frames'],
                state_audit=spec['audit'],output=output))
        after=runtime_info(); validate_runtime(after,frozen['runtime_reference'],after=True)
        old=control.read(output.with_suffix('.v35base.json'))
        if (adapter.calls!=1 or output.with_suffix('.v29.json').exists() or not old['passed']
                or old['error'] is not None or old['processed_frames']!=control.expected_count(spec)
                or old['native_motion_v25_enabled']!=native):
            raise AssertionError('Frozen exact-output/full-frame gate failed')
        validate_calls(spec,old['processed_frames'],fit_calls,helper)
        r.update(passed=True,runtime_after=after,receipt_sha256=control.sha(output.with_suffix('.v35base.json')),
            fps=old['fps'],wall_s=old['wall_s'],count=old['processed_frames'])
    except BaseException as exc:
        r['error']=repr(exc)
        raise
    finally:
        restored=motion.fit_global_motion is original and baseline.write is old_write
        r.update(bindings_restored=restored,fit_calls=fit_calls,native_calls=helper.calls,
            fallbacks=helper.fallbacks,passthroughs=helper.passthroughs,receipt_adapter_calls=adapter.calls)
        if not restored:
            r.update(passed=False,error=str(r['error'])+'; wrapper bindings not restored')
        control.write(output.with_suffix('.v35.json'),r)
        if not restored:
            raise AssertionError('Wrapper bindings not restored')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true'); p.add_argument('--trial'); p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.preflight:
        if a.trial or a.output:p.error('Preflight does not accept a trial/output')
        print(control.json.dumps(dependency_preflight(),indent=2))
    else:
        if not a.trial or a.output is None:p.error('Trial/output required')
        run(a.trial,a.output)
