"""Generated sequence equality and bounded estimator timings; never opens media."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from motion_reuse_v12 import ReuseMotionV12,REFERENCE_PIXELS,generated_method
from raw16_speed_v8_common import FeaturePixelsCache
from check_raw16_motion_controls import cases,frame,texture
from check_motion_lut_v11 import observe
from profile_raw16_efficiency import sha,write_json,compact
from tiny_target.motion import pva_pyrlk as pva
from tiny_target.motion import GlobalMotionConfig,fit_global_motion,PvaMotionError
from tiny_target.types import Frame,TimestampSource,Discontinuity

CONFIG=ROOT/'configs/evaluation/raw16_motion_v6.json'


class Reference:
    def __init__(self,config):
        self.estimator=pva.PvaPyrLkMotionEstimator(config)
        self.cache=FeaturePixelsCache(REFERENCE_PIXELS)
    def estimate(self,a,b):
        with patch.object(pva,'_feature_pixels',self.cache):return self.estimator.estimate(a,b)
    def reset(self):self.estimator._stream.sync();self.cache.clear()
    def close(self):self.reset()


def sequence(depth,shape,kind,seed=75316,*,legacy_upscale=False):
    h,w=shape;tile=texture(seed,bright=True)
    # Preserve feature scale when extending to native geometry. The initial
    # upscaled U8 fixture was unobservable and is retained below as a negative.
    base=(cv2.resize(tile,(w,h),interpolation=cv2.INTER_LINEAR) if legacy_upscale
        else np.ascontiguousarray(np.tile(tile,((h+959)//960,(w+1279)//1280))[:h,:w]))
    frames=[];rng=np.random.default_rng(seed+300)
    length=8 if kind=='recovery' else 6
    for i in range(length):
        image=cv2.warpAffine(base.astype(np.float32),np.array([[1,0,i*1.5],[0,1,-i*.75]],np.float32),
            (w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
        image=np.clip(np.rint(image),0,65535).astype(np.uint16)
        if kind=='recovery' and i in (2,3):image=rng.integers(8000,26000,shape,dtype=np.uint16)
        if kind=='recovery' and i==4:image[:]=12000
        if depth==8:image=np.clip(np.rint(image.astype(np.float32)/128),0,255).astype(np.uint8)
        mask=np.ones(shape,bool);mask[:8]=False;mask[-8:]=False
        frames.append(Frame(image,i*100000000,i,'v12-generated-'+kind,depth,TimestampSource.MANIFEST,valid_mask=mask))
    if kind=='invalidation':
        frames[2]=replace(frames[2],discontinuities=(Discontinuity.CHUNK_BOUNDARY,))
        for i in range(3,len(frames)):frames[i]=replace(frames[i],frame_index=i+1,source_id='v12-new-source')
    return frames


def outcome(estimator,a,b,fitter):
    tick=time.perf_counter()
    try:pairs=estimator.estimate(a,b)
    except PvaMotionError as exc:
        if any(x in str(exc) for x in ('zero features','No finite in-bounds')):
            return dict(identity={'unavailable':str(exc)},accepted=False,host_s=time.perf_counter()-tick)
        raise
    elapsed=time.perf_counter()-tick
    fit=fit_global_motion(pairs,fitter,execution='translation_batched_exact_v1')
    return dict(identity=compact(dict(pairs=pairs,fit=fit)),accepted=fit.accepted,host_s=elapsed,
        substage_ms=pairs.timings_ms)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    cv2.setNumThreads(2)
    config=json.loads(CONFIG.read_text());motion=pva.PvaMotionConfig.from_mapping(config['motion'])
    fitter=GlobalMotionConfig.from_mapping(config['global_motion'])
    record=dict(real_media_read=False,production_approved=False,pipeline_benchmark=False,
        script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'scripts/motion_reuse_v12.py'),
        config_sha256=sha(CONFIG),motion_source_sha256=sha(ROOT/'tiny_target/motion/pva_pyrlk.py'),
        generated_method_sha256=hashlib.sha256(generated_method().encode()).hexdigest(),
        plan_sha256=sha(ROOT/'docs/motion_reuse_v12_plan.md'),independent=[],sequences=[],timings=[],
        original_upscaled_negative=[],
        quality_passed=False,passed=False,
        timing_boundary='Entire estimator call including pixel cache, preparation, Harris, flow, readback and diagnostics. '
            'Global fitting, input generation and output verification excluded. Not whole-pipeline FPS.')
    try:
        reference=Reference(motion);candidate=ReuseMotionV12(motion)
        try:
            for seed in (75316,75317,75318,75319):
                for name,previous,current,truth in cases(seed):
                    a,b=frame(previous,0,name),frame(current,1,name)
                    left=observe(reference,fitter,a,b,truth);right=observe(candidate,fitter,a,b,truth)
                    exact=left['identity']==right['identity'];expected=left['expected_decision'] and right['expected_decision']
                    record['independent'].append(dict(seed=seed,case=name,exact=exact,expected=expected))
                    if not exact or not expected:raise AssertionError('Independent generated motion control changed')
        finally:reference.close();candidate.close()
        # Preserve the original failing positive fixture as an explicit
        # unobservable control: it must agree and must not produce a false hit.
        original_frames=sequence(8,(3190,4784),'smooth',legacy_upscale=True)
        reference=Reference(motion);candidate=ReuseMotionV12(motion)
        try:
            for i in range(1,4):
                left=outcome(reference,original_frames[i-1],original_frames[i],fitter)
                right=outcome(candidate,original_frames[i-1],original_frames[i],fitter)
                exact=left['identity']==right['identity']
                record['original_upscaled_negative'].append(dict(index=i,exact=exact,
                    reference_accepted=left['accepted'],candidate_accepted=right['accepted'],hits=candidate.hits))
                if not exact or left['accepted'] or right['accepted'] or candidate.hits:
                    raise AssertionError('Original unobservable fixture behavior changed')
        finally:reference.close();candidate.close()
        for shape in ((960,1280),(3190,4784)):
            for depth in (8,16):
                for kind in ('smooth','recovery','invalidation','reordered'):
                    frames=sequence(depth,shape,kind)
                    pairs=([(0,1),(1,2),(0,1),(0,1),(2,3),(3,4)] if kind=='reordered'
                        else [(i-1,i) for i in range(1,len(frames))])
                    reference=Reference(motion);candidate=ReuseMotionV12(motion);rows=[]
                    hashes=[f.pixel_sha256() for f in frames]
                    try:
                        for order,(i,j) in enumerate(pairs):
                            a,b=frames[i],frames[j]
                            if kind=='invalidation' and order==3:
                                reference.reset();candidate.reset();a=replace(a)
                            left=outcome(reference,a,b,fitter);right=outcome(candidate,a,b,fitter)
                            exact=left['identity']==right['identity']
                            rows.append(dict(pair=[i,j],exact=exact,accepted=right['accepted'],hits=candidate.hits,misses=candidate.misses))
                            if not exact:raise AssertionError(f'Sequence changed: {shape}, {depth}, {kind}, {order}')
                        unchanged=hashes==[f.pixel_sha256() for f in frames]
                        require_hit=kind in ('smooth','recovery','reordered')
                        record['sequences'].append(dict(shape=shape,depth=depth,kind=kind,rows=rows,
                            inputs_unchanged=unchanged,hits=candidate.hits,misses=candidate.misses))
                        if not unchanged or (require_hit and candidate.hits<1):raise AssertionError('Input mutation or reuse was not exercised')
                        if kind=='recovery' and not rows[-1]['accepted']:raise AssertionError('Clean motion failed to recover')
                        if kind=='smooth' and not all(r['accepted'] for r in rows):raise AssertionError('Smooth positive sequence not recovered')
                    finally:reference.close();candidate.close()
                    print('sequence',shape,depth,kind,'passed',flush=True)
        record['quality_passed']=len(record['independent'])==48 and len(record['sequences'])==16
        if not record['quality_passed']:raise AssertionError('Incomplete prerequisite coverage')
        for depth in (8,16):
            frames=sequence(depth,(3190,4784),'smooth');expected=[]
            reference=Reference(motion)
            try:
                for i in range(1,len(frames)):expected.append(outcome(reference,frames[i-1],frames[i],fitter)['identity'])
            finally:reference.close()
            for repeat in range(4):
                for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference')):
                    estimator=Reference(motion) if mode=='reference' else ReuseMotionV12(motion);samples=[]
                    try:
                        for i in range(1,len(frames)):
                            observed=outcome(estimator,frames[i-1],frames[i],fitter)
                            exact=observed.pop('identity')==expected[i-1]
                            samples.append(dict(index=i,exact=exact,**observed))
                            if not exact:raise AssertionError('Timed native estimator outputs changed')
                        hits=estimator.hits if mode=='candidate' else None
                        if mode=='candidate' and hits!=4:raise AssertionError('Unexpected reuse count in timing')
                        record['timings'].append(dict(depth=depth,repeat=repeat,mode=mode,hits=hits,samples=samples))
                        print(json.dumps(dict(depth=depth,repeat=repeat,mode=mode,host_s=[s['host_s'] for s in samples])),flush=True)
                    finally:estimator.close()
        record['passed']=len(record['timings'])==16
    finally:write_json(args.output,record)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
