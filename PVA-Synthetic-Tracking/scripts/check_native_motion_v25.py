"""Generated exact-fit gates and compact development-correspondence replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
from native_motion_v25 import NativeMotionV25,reference_samples,gm,REFERENCE_SHA
from build_native_motion_v25 import sha


def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(x) for x in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def equal_array(a,b):
    assert (a is None)==(b is None)
    if a is not None:assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def estimate_payload(value):
    result=value.to_dict();result.pop('timing_ms');return finite(result)


def equal_estimate(a,b):
    for field in ('previous_to_current_matrix','inlier_mask','residuals_px'):
        equal_array(getattr(a,field),getattr(b,field))
    assert estimate_payload(a)==estimate_payload(b)


def correspondence(previous,current,index=1,metrics=None):
    return gm.MotionCorrespondences(previous_points=previous,current_points=current,
        harris_scores=np.ones(len(previous),np.float32),forward_backward_error_px=np.zeros(len(previous),np.float32),
        previous_frame_index=index-1,current_frame_index=index,previous_timestamp_ns=(index-1)*100000000,
        current_timestamp_ns=index*100000000,full_image_size=(4784,3190),motion_image_size=(2392,1595),
        metrics=metrics or {},timings_ms={},backends={})


def cases():
    rng=np.random.default_rng(253716)
    config=gm.GlobalMotionConfig(minimum_correspondences=1,minimum_inliers=1,minimum_inlier_ratio=.2,
        minimum_inlier_grid_coverage=0,ransac_reprojection_px=.5)
    for i,n in enumerate((0,1,2,3,7,31,64,65,127,257,500,1000)):
        for variant in range(3):
            previous=rng.uniform([64,64],[4700,3100],(n,2)).astype(np.float32)
            current=(previous+np.array([3.125,-1.75],np.float32)).astype(np.float32)
            if variant==1:current+=rng.normal(0,.04,current.shape).astype(np.float32)
            if variant==2 and n:
                count=max(1,n//3);current[:count]=rng.uniform([64,64],[4700,3100],(count,2))
            yield f'random_{n}_{variant}',correspondence(previous,current,index=i+1),config,'reference'
    p=np.column_stack((np.arange(64,96),np.arange(64,96))).astype(np.float32)
    for name,delta in (('zero',np.zeros_like(p)),('ties',np.tile([[1,0],[-1,0]],(16,1))),
                       ('odd_even',np.column_stack((np.arange(32)%3,np.zeros(32))).astype(np.float32))):
        yield name,correspondence(p,p+delta),config,'reference'
    for threshold in (.5,np.nextafter(.5,0),np.nextafter(.5,np.inf)):
        c=p.copy();c[::3,0]+=.5;c[1::3,0]=np.nextafter(c[1::3,0]+.5,np.float32(np.inf))
        yield f'boundary_{threshold.hex()}',correspondence(p,c),replace(config,ransac_reprojection_px=threshold),'reference'
    for label,metrics,cfg in (
        ('quality',{'usable_for_transform':False,'quality_rejection_reasons':['low_grid_coverage']},config),
        ('sparse',{'usable_for_transform':False,'quality_rejection_reasons':['low_grid_coverage']},
         replace(config,coverage_policy='translation_consensus',minimum_inlier_grid_coverage=.5)),
        ('limits',{},replace(config,maximum_translation_px=.1)),
        ('similarity',{},replace(config,model='similarity',ransac_iterations=32)),
        ('batched',{},config)):
        yield label,correspondence(p,p+[2,1],metrics=metrics),cfg,('translation_batched_exact_v1' if label=='batched' else 'reference')


def generated(library):
    helper=NativeMotionV25(library);adapted=helper.adapter(gm.fit_global_motion);rows=[]
    for name,corr,config,execution in cases():
        before=(corr.previous_points.tobytes(),corr.current_points.tobytes())
        a=gm.fit_global_motion(corr,config,execution=execution);b=adapted(corr,config,execution=execution)
        equal_estimate(a,b)
        assert before==(corr.previous_points.tobytes(),corr.current_points.tobytes())
        rows.append(dict(name=name,exact=True))
    rng=np.random.default_rng(253717)
    p=rng.integers(64,1000,(80,2)).astype(np.float64);c=p+[2.,1.]
    samples=[(int(i),) for i in rng.permutation(len(p))]
    primitive=[]
    for name,a,b,selected in (
        ('ordinary',p,c,samples),('duplicate_ties',p,c,[(17,),(2,),(17,)]),
        ('reverse_layout',p[::-1],c[::-1],samples),('float32',p.astype(np.float32),c.astype(np.float32),samples),
        ('negative',-p,-c,samples),('non_float32',p+.0000000001,c,samples),
        ('too_large',p+70000,c+70000,samples),('empty_samples',p,c,[])):
        calls=helper.calls;fallbacks=helper.fallbacks
        x=reference_samples(a,b,selected,.5);y=helper(a,b,selected,.5)
        equal_array(x[0],y[0]);equal_array(x[1],y[1]);assert x[2]==y[2]
        primitive.append(dict(name=name,exact=True,native=helper.calls>calls,fallback=helper.fallbacks>fallbacks))
    for value in (-0.,np.inf,np.nan):
        a=p.copy();a[0,0]=value
        with np.errstate(all='ignore'):
            x=reference_samples(a,c,samples,.5);y=helper(a,c,samples,.5)
        equal_array(x[0],y[0]);equal_array(x[1],y[1]);assert finite(x[2])==finite(y[2])
        primitive.append(dict(name=repr(value),exact=True,native=False,fallback=True))
    x=helper(p,c,samples,.5);saved=x[1].copy();helper(p,c+np.array([1.,0.]),samples,.5);equal_array(x[1],saved)
    def parallel(offset):
        a=reference_samples(p,c+[offset,0],samples,.5);b=helper(p,c+[offset,0],samples,.5)
        equal_array(a[0],b[0]);equal_array(a[1],b[1]);assert a[2]==b[2]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(parallel,range(12)))
    return dict(passed=True,real_media_read=False,cases=rows,primitive=primitive,
                native_calls=helper.calls,fallbacks=helper.fallbacks,passthroughs=helper.passthroughs,
                independent_outputs=True,reentrant_calls=12,reference_sha256=REFERENCE_SHA,
                transformed_sha256=helper.transformed_sha256)


def gil_probe(library):
    helper=NativeMotionV25(library);n=4096;m=1024;rng=np.random.default_rng(253718)
    p=rng.uniform(64,4096,(n,2)).astype(np.float32).astype(np.float64)
    c=(p.astype(np.float32)+rng.normal(0,1,p.shape).astype(np.float32)).astype(np.float64)
    indices=np.arange(m,dtype=np.int64);mask=np.empty(n,np.uint8);result=np.empty(2,np.int64);median=np.empty(1)
    args=(p.ctypes.data,c.ctypes.data,n,indices.ctypes.data,m,.5,mask.ctypes.data,result.ctypes.data,median.ctypes.data)
    ready=threading.Event();go=threading.Event();observed=[]
    def other():ready.set();go.wait();observed.append(time.perf_counter_ns())
    worker=threading.Thread(target=other);worker.start();assert ready.wait(2)
    interval=sys.getswitchinterval()
    try:
        sys.setswitchinterval(10.0);start=time.perf_counter_ns();go.set()
        code=helper.fn(*args);end=time.perf_counter_ns()
    finally:sys.setswitchinterval(interval);worker.join(2)
    assert code==0 and not worker.is_alive() and len(observed)==1 and start<observed[0]<end
    return dict(passed=True,host_call_ms=(end-start)/1e6,other_thread_progress_during_call=True,
                interpretation='Python thread progressed inside CDLL host-call interval. Not a speedup or GIL-contention diagnosis.')


def replay(library,path):
    payload=json.loads(path.read_text());npz=path.with_suffix('.npz')
    assert sha(npz)==payload['npz_sha256']
    helper=NativeMotionV25(library);adapted=helper.adapter(gm.fit_global_motion);rows=[]
    with np.load(npz,allow_pickle=False) as arrays:
        for case in payload['cases']:
            key=case['key'];fields=dict(case['correspondence'])
            for name in ('previous_points','current_points','harris_scores','forward_backward_error_px'):
                fields[name]=arrays[key+'_'+name]
            corr=gm.MotionCorrespondences(**fields);config=gm.GlobalMotionConfig(**case['config'])
            expected=gm.fit_global_motion(corr,config);actual=adapted(corr,config)
            equal_estimate(expected,actual)
            assert estimate_payload(expected)==case['expected']
            equal_array(expected.inlier_mask,arrays[key+'_expected_mask'])
            equal_array(expected.residuals_px,arrays[key+'_expected_residuals'])
            if expected.previous_to_current_matrix is not None:equal_array(expected.previous_to_current_matrix,arrays[key+'_expected_matrix'])
            rows.append(dict(frame=corr.current_frame_index,exact=True))
    assert [r['frame'] for r in rows]==list(range(1,128))
    return dict(clip=payload['clip'],cases=rows,native_calls=helper.calls,fallbacks=helper.fallbacks,
                json_sha256=sha(path),npz_sha256=sha(npz),exact=True)


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    result=generated(args.library);result['gil_probe']=gil_probe(args.library)
    result['replays']=[replay(args.library,p) for p in args.replay]
    result.update(library_sha256=sha(args.library),source_sha256={n:sha(Path(__file__).with_name(n)) for n in (
        'native_motion_v25.cpp','native_motion_v25.py','build_native_motion_v25.py','check_native_motion_v25.py')})
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','primitive','replays')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--replay',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
