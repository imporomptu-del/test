"""Generated-only exact pixels/estimator checks and bounded conversion timing."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motion_lut_v11 import feature_pixels_lut,REFERENCE_FEATURE_PIXELS
from raw16_speed_v8_common import FeaturePixelsCache
from profile_raw16_efficiency import compact,sha,write_json
from check_raw16_motion_controls import cases,frame,MAX_TRANSLATION_ERROR_PX
from tiny_target.motion import pva_pyrlk as pva
from tiny_target.motion import GlobalMotionConfig,fit_global_motion,PvaMotionError
from tiny_target.types import Frame,TimestampSource

CONFIG=ROOT/'configs/evaluation/raw16_motion_v6.json'


def observe(estimator,fitter,previous,current,truth):
    try:pairs=estimator.estimate(previous,current)
    except PvaMotionError as exc:
        if truth is None and any(x in str(exc) for x in ('zero features','No finite in-bounds')):
            return dict(identity=dict(unavailable=str(exc)),accepted=False,expected_decision=True,error=None)
        raise
    fit=fit_global_motion(pairs,fitter,execution='translation_batched_exact_v1')
    error=float(np.linalg.norm(fit.previous_to_current_matrix[:2,2]-truth)) if fit.accepted and truth is not None else None
    correct=(fit.accepted and error<=MAX_TRANSLATION_ERROR_PX) if truth is not None else not fit.accepted
    return dict(identity=compact(dict(pairs=pairs,fit=fit)),accepted=fit.accepted,expected_decision=correct,error=error)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    config=json.loads(CONFIG.read_text());motion=pva.PvaMotionConfig.from_mapping(config['motion'])
    fitter=GlobalMotionConfig.from_mapping(config['global_motion'])
    record=dict(real_media_read=False,production_approved=False,pipeline_benchmark=False,
        script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'scripts/motion_lut_v11.py'),
        motion_source_sha256=sha(ROOT/'tiny_target/motion/pva_pyrlk.py'),
        config_sha256=sha(CONFIG),pixel_checks=[],motion_checks=[],timings=[],passed=False)
    methods={'reference':REFERENCE_FEATURE_PIXELS,'candidate':feature_pixels_lut}
    try:
        allcodes=np.arange(65536,dtype=np.uint16).reshape(256,256)
        for depth in (9,10,12,14,16):
            for masking in ('all','none','pattern'):
                values=allcodes%2**depth if depth<16 else allcodes
                mask=None if masking=='all' else np.zeros(values.shape,bool) if masking=='none' else values%7!=0
                f=Frame(values,0,0,'v11-all-codes',depth,TimestampSource.MANIFEST,valid_mask=mask)
                out=[fn(f,'raw_robust_u16_v1') for fn in methods.values()]
                exact=out[0].tobytes()==out[1].tobytes()
                record['pixel_checks'].append(dict(depth=depth,mask=masking,exact=exact))
                if not exact:raise AssertionError('Feature mapping changed source codes')
        estimators={mode:pva.PvaPyrLkMotionEstimator(motion) for mode in methods}
        caches={mode:FeaturePixelsCache(fn) for mode,fn in methods.items()}
        for seed in (75316,75317,75318,75319):
            for index,(name,previous,current,truth) in enumerate(cases(seed)):
                a,b=frame(previous,0,name),frame(current,1,name);original=[a.pixel_sha256(),b.pixel_sha256()]
                observed={}
                for mode in (('reference','candidate') if index%2==0 else ('candidate','reference')):
                    with patch.object(pva,'_feature_pixels',caches[mode]):
                        observed[mode]=observe(estimators[mode],fitter,a,b,truth)
                exact=observed['reference']['identity']==observed['candidate']['identity']
                unchanged=original==[a.pixel_sha256(),b.pixel_sha256()]
                record['motion_checks'].append(dict(seed=seed,case=name,exact=exact,inputs_unchanged=unchanged,
                    expected_decisions=all(v['expected_decision'] for v in observed.values()),
                    accepted={k:v['accepted'] for k,v in observed.items()},
                    error={k:v['error'] for k,v in observed.items()}))
                if not exact or not unchanged:raise AssertionError('Motion point/fit identities changed')
                print('motion',seed,name,'exact',exact,flush=True)
        # Only conversion is timed here. VPI/resize/flow/fitting are NOT timed.
        for scene in ('range','dim_texture'):
            rng=np.random.default_rng(11916)
            image=rng.integers(0,65536 if scene=='range' else 2048,(3190,4784),dtype=np.uint16)
            f=Frame(image,0,0,'v11-native-'+scene,16,TimestampSource.MANIFEST)
            expected=REFERENCE_FEATURE_PIXELS(f,'raw_robust_u16_v1').tobytes()
            if feature_pixels_lut(f,'raw_robust_u16_v1').tobytes()!=expected:raise AssertionError('Native feature pixels differ')
            for repeat in range(4):
                for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference')):
                    samples=[]
                    for cycle in range(3):
                        tick=time.perf_counter();out=methods[mode](f,'raw_robust_u16_v1');elapsed=time.perf_counter()-tick
                        exact=out.tobytes()==expected
                        samples.append(dict(cycle=cycle,host_s=elapsed,exact=exact))
                        if not exact:raise AssertionError('Timed conversion differs')
                    record['timings'].append(dict(scene=scene,repeat=repeat,mode=mode,samples=samples))
        record['expected_motion_decisions_passed']=all(r['expected_decisions'] for r in record['motion_checks'])
        record['passed']=(len(record['pixel_checks'])==15 and len(record['motion_checks'])==48
            and len(record['timings'])==16 and record['expected_motion_decisions_passed'])
    finally:write_json(args.output,record)
    print(json.dumps({'passed':record['passed'],'motion_checks':len(record['motion_checks'])}));return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
