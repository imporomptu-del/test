"""Media-free exact state/support checks; direct GPU point filter is diagnostic only."""
from __future__ import annotations

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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_raw16_efficiency import compact,sha,write_json
from tiny_target.dense_screen import DensePointScreener,DenseScreenConfig,load_dense_screen_config
from tiny_target.raw_background_cuda import RawBackgroundCuda
from tiny_target.types import Frame,TimestampSource


def exact(name,a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        unequal=np.argwhere(a.view(np.uint8).reshape(a.shape+(-1,))!=b.view(np.uint8).reshape(b.shape+(-1,)))
        point=tuple(unequal[0][:-1])
        raise AssertionError(f'{name} is not bit-exact: {point} {a[point]!r} != {b[point]!r}')


def fixture(config,shape,dtype,frames,seed,library,stationary=False,seed_underflow=False):
    rng=np.random.default_rng(seed)
    cpu=DensePointScreener(config);gpu=RawBackgroundCuda(config,shape,library)
    original_filter=cv2.filter2D
    source_identity=hashlib.sha256();state_identity=hashlib.sha256();rows=[]
    try:
        for index in range(frames):
            image=rng.integers(1000,1200,shape).astype(dtype)
            # Signal ramps, saturation and invalid histories, not selected media.
            image.flat[::97]=0;image.flat[::101]=65535
            image.flat[12:15]=[32768,32769,32770]
            image.flat[(index*19)%image.size]=60000
            if dtype==np.float32:
                image+=np.float32(.03125)
                image.flat[3:7]=np.asarray([1.,np.nextafter(np.float32(1),np.float32(0)),
                    np.nextafter(np.float32(1),np.float32(2)),np.float32(65535*.995)],np.float32)
            validity=rng.random(shape)>.17
            if index%12==5:validity[:]=False
            if index%12==6:validity[:]=True
            if stationary:
                image[:]=1000;validity[:]=True
            frame=Frame(image=image,valid_mask=validity,timestamp_ns=index*173000001,
                        frame_index=index,source_id='generated-raw-gpu-parity',bit_depth=16,
                        timestamp_source=TimestampSource.SIDECAR_UNIX_NS,
                        source_timestamp_ns=1700000000000000000+index*173000001)
            before=frame.pixel_sha256();source_identity.update(before.encode())
            if index and index%12==0 and not stationary:
                cpu._background_location=None;cpu._background_variance=None
                cpu._background_support=None;cpu._background_frame_count=0;gpu.reset()
            captured=[]
            def capture(value,*args,**kwargs):
                captured.append(value.copy())
                return original_filter(value,*args,**kwargs)
            with patch.object(cv2,'filter2D',capture):
                start=time.perf_counter();cpu._events_for_frame(frame);cpu_s=time.perf_counter()-start
            valid=(image>config.dark_floor_dn)&(image<float(65535)*config.saturation_fraction)&validity
            start=time.perf_counter();product=gpu.step(image.astype(np.float32,copy=False),valid)
            gpu_s=time.perf_counter()-start
            state=gpu.debug_state()
            for name,attribute in (('location','_background_location'),('variance','_background_variance'),('history','_background_support')):
                reference=getattr(cpu,attribute);exact(name,reference,state[name]);state_identity.update(reference.tobytes())
            if product is None:
                if cpu._last_synthetic_frame is not None:raise AssertionError('Initial-state mismatch')
            else:
                white,mask,ready=product;reference=cpu._last_synthetic_frame
                exact('whitened',captured[0],white);exact('filter_valid',reference.valid_mask,mask)
                exact('downloaded_white',white,state['whitened'])
                response=original_filter(white,cv2.CV_32F,cpu._point_kernel,borderType=cv2.BORDER_CONSTANT)
                response/=cpu._point_kernel_l2
                exact('point_response_cpu_preserved',reference.response,response)
                if ready!=reference.detection_ready:raise AssertionError('Warmup mismatch')
                state_identity.update(response.tobytes());state_identity.update(mask.tobytes())
            if gpu.count!=cpu._background_frame_count:raise AssertionError('Model-age mismatch')
            if frame.pixel_sha256()!=before:raise AssertionError('Source frame mutated')
            if index%12==7:
                # Exercise saturated counters without 65,535 warmup frames.
                cpu._background_support.flat[::7]=65535
                gpu.set_state_for_test(cpu._background_location,cpu._background_variance,cpu._background_support)
            if seed_underflow and index==0:
                cpu._background_variance.view(np.uint32).flat[:6]=np.array(
                    [0,1,0x007fffff,0x00800000,0x00800001,0x00800010],np.uint32)
                gpu.set_state_for_test(cpu._background_location,cpu._background_variance,cpu._background_support)
            rows.append(dict(cpu_s=cpu_s,gpu_temporal_support_s=gpu_s))
        return dict(shape=list(shape),dtype=np.dtype(dtype).str,frames=frames,seed=seed,
                    exact=True,state_sha256=state_identity.hexdigest(),source_sha256=source_identity.hexdigest(),
                    stationary=stationary,seed_underflow=seed_underflow,
                    timings=rows,parameters=dict(normal_rate=config.background_update_rate,
                    outlier_rate=config.background_outlier_update_rate,floor=config.noise_sigma_floor_dn))
    finally:gpu.close()


def point_probe(config,library):
    # The dimensions also exercise partial thread blocks and image borders.
    shape=(67,100);rng=np.random.default_rng(562781)
    white=rng.normal(0,8,shape).astype(np.float32)
    white[20,44]=300.;white[0,0]=-100.;white[-1,-1]=100.
    cpu=DensePointScreener(config);gpu=RawBackgroundCuda(config,shape,library)
    try:
        a=cv2.filter2D(white,cv2.CV_32F,cpu._point_kernel,borderType=cv2.BORDER_CONSTANT)
        a/=cpu._point_kernel_l2
        b=gpu.point_filter_probe(white,cpu._point_kernel,cpu._point_kernel_l2)
        delta=a.astype(np.float64)-b.astype(np.float64)
        return dict(exact=a.tobytes()==b.tobytes(),different_pixels=int(np.count_nonzero(a!=b)),
                    total_pixels=int(a.size),max_abs_error=float(np.max(np.abs(delta))),
                    rms_error=float(np.sqrt(np.mean(delta*delta))),reference=compact(a),candidate=compact(b),
                    production_enabled=False,
                    conclusion='Direct GPU convolution is diagnostic only; retain the exact OpenCV point filter.')
    finally:gpu.close()


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg,_=load_dense_screen_config(ROOT/'configs/evaluation/raw16_full_frame_v2.json')
    cfg=replace(cfg,synthetic_tracking_enabled=False)
    record=dict(schema_version='seaqr.raw16-gpu-background-controls.v7',passed=False,cases=[],
        script_sha256=sha(__file__),library_sha256=sha(args.library),
        package_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tiny_target').rglob('*.py')) if not p.name.startswith('._')},
        cuda_source_sha256=sha(ROOT/'tiny_target/detection/cuda/raw_background.cu'),
        opencv=cv2.__version__,numpy=np.__version__,opencv_build_information=cv2.getBuildInformation(),
        real_media_read=False,point_filter_backend='unchanged_cpu_opencv')
    try:
        record['direct_gpu_point_filter_probe']=point_probe(cfg,args.library)
        for shape in ((5,7),(13,20),(47,65),(64,96)):
            for dtype in (np.uint16,np.float32):
                for rates in ((.2,.03,16.),(1.,1.,1.),(.7,.13,.03125)):
                    c=replace(cfg,background_update_rate=rates[0],background_outlier_update_rate=rates[1],noise_sigma_floor_dn=rates[2])
                    result=fixture(c,shape,dtype,24,1845,args.library)
                    record['cases'].append(result)
                    print(json.dumps({k:v for k,v in result.items() if k!='timings'}),flush=True)
        if args.native:
            for dtype in (np.uint16,np.float32):
                result=fixture(cfg,(3190,4784),dtype,16,85477,args.library)
                record['cases'].append(result)
                print(json.dumps({k:v for k,v in result.items() if k!='timings'}),flush=True)
        # A genuinely long static sequence reaches underflow naturally; the
        # short seeded case tests adjacent normal/subnormal float bit patterns.
        for frames,seed_underflow in ((512,False),(8,True)):
            result=fixture(cfg,(16,24),np.float32,frames,92416,args.library,
                           stationary=True,seed_underflow=seed_underflow)
            record['cases'].append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='timings'}),flush=True)
        record.update(passed=True,frame_comparisons=sum(r['frames'] for r in record['cases']),native_enabled=args.native)
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:write_json(args.output,record)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native',action='store_true')
    raise SystemExit(run(parser.parse_args()))
