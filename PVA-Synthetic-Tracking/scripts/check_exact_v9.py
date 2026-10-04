"""Generated-only exact FFT/warp gates, including native dimensions. No media."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from raw16_speed_v8_common import cpu_filter,filter_parameters,package_identity
from profile_raw16_efficiency import sha,write_json
from tiny_target.point_filter_fft_exact import PointFilterFftExact
from tiny_target.warp_translation_cuda import WarpTranslationCuda,DEFAULT_LIBRARY,cubic_table

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def compare(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Array contract mismatch')
    ba,bb=a.view(np.uint8),b.view(np.uint8)
    return dict(exact=bool(np.array_equal(ba,bb)),maximum_error=float(np.max(np.abs(a.astype(np.float64)-b))),
        differing_values=int(np.count_nonzero(a!=b)),reference_sha256=digest(a),candidate_sha256=digest(b))

def images(shape,seed):
    rng=np.random.default_rng(seed)
    a=rng.integers(0,65536,shape,dtype=np.uint16).astype(np.float32)
    yield 'sensor_noise',a
    yield 'zero',np.zeros(shape,np.float32)
    yield 'constant',np.full(shape,2048,np.float32)
    a=np.zeros(shape,np.float32);a[0,0]=65535;a[-1,-1]=32768;a[shape[0]//2,shape[1]//2]=.0001
    yield 'impulses',a
    a=rng.normal(0,3,shape).astype(np.float32);a[shape[0]//3:2*shape[0]//3,shape[1]//3:2*shape[1]//3]=0
    yield 'signed_noise_hole',a

def fft_checks(native,quick):
    kernel,norm=filter_parameters();rows=[]
    shapes=[(1,1),(3,7),(67,100),(248,248),(249,249),(257,499),(497,513)]
    if quick:shapes=[(37,65),(257,499)]
    if native:shapes.append((3190,4784))
    for shape in shapes:
        devices={n:PointFilterFftExact(shape,kernel,norm,workers=n) for n in (1,2,4)}
        try:
            for kind,a in images(shape,652179):
                before=digest(a);start=time.perf_counter();reference=cpu_filter(a,kernel,norm);cpu_s=time.perf_counter()-start
                row=dict(shape=shape,kind=kind,reference_s=cpu_s,workers={})
                for n,dev in devices.items():
                    times=[]
                    for _ in range(2 if not quick else 1):
                        start=time.perf_counter();out=dev(a);times.append(time.perf_counter()-start)
                    row['workers'][str(n)]=dict(**compare(reference,out),times_s=times,median_s=statistics.median(times),
                        threshold4_changes=int(np.count_nonzero((reference>=4)!=(out>=4))),
                        raw_exact=compare(cv2.filter2D(a,cv2.CV_32F,kernel,borderType=cv2.BORDER_CONSTANT),dev.correlate(a))['exact'])
                if digest(a)!=before:raise AssertionError('FFT input mutated')
                rows.append(row)
                print('fft',shape,kind,[r['exact'] for r in row['workers'].values()],flush=True)
        finally:
            for d in devices.values():d.close()
    return rows

def warp_checks(native,quick):
    rows=[]
    shifts=[(0.,0.),(2.,-3.),(.21875,-.65625),(.015625,-.015625),(.5,-.5),
        (float(np.nextafter(.015625,0.)),float(np.nextafter(-.015625,0.))),
        (float(np.nextafter(.015625,1.)),float(np.nextafter(-.015625,-1.))),
        (float(np.nextafter(.5,0.)),float(np.nextafter(-.5,0.))),
        (-120.25,120.75),(1000000.,-1000000.)]
    shapes=[(1,1),(3,7),(16,63),(17,65),(37,59),(129,257)]
    if quick:shapes=[(37,65),(129,257)];shifts=shifts[:5]
    if native:shapes.append((3190,4784))
    for shape in shapes:
        dev=WarpTranslationCuda(shape)
        try:
            for kind,a in images(shape,562178):
                mask=(np.random.default_rng(173).random(shape)>.12).astype(np.uint8)
                before=(digest(a),digest(mask))
                for tx,ty in shifts:
                    matrix=np.eye(3);matrix[:2,2]=[tx,ty]
                    start=time.perf_counter()
                    ref=cv2.warpPerspective(a,matrix,(shape[1],shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
                    valid=cv2.warpPerspective(mask,matrix,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                    cpu_s=time.perf_counter()-start
                    start=time.perf_counter();out,vm=dev(a,mask,matrix);gpu_s=time.perf_counter()-start
                    row=dict(shape=shape,kind=kind,translation=[tx,ty],image=compare(ref,out),mask=compare(valid,vm),
                        reference_s=cpu_s,candidate_s=gpu_s)
                    rows.append(row)
                    print('warp',shape,kind,(tx,ty),row['image']['exact'],row['mask']['exact'],row['image']['maximum_error'],flush=True)
                if before!=(digest(a),digest(mask)):raise AssertionError('Warp inputs mutated')
        finally:dev.close()
    # All 1024 independent fractional phase combinations on generated pixels.
    phases=[]
    if not quick:
        shape=(37,65);rng=np.random.default_rng(916762)
        a=rng.integers(0,65536,shape,dtype=np.uint16).astype(np.float32)
        mask=(rng.random(shape)>.1).astype(np.uint8);dev=WarpTranslationCuda(shape)
        try:
            for fy in range(32):
                for fx in range(32):
                    m=np.eye(3);m[:2,2]=[fx/32.,fy/32.]
                    ref=cv2.warpPerspective(a,m,(65,37),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
                    valid=cv2.warpPerspective(mask,m,(65,37),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                    out,vm=dev(a,mask,m)
                    phases.append(dict(phase=[fx,fy],image=compare(ref,out),mask=compare(valid,vm)))
            print('phases',sum(r['image']['exact'] and r['mask']['exact'] for r in phases),len(phases),flush=True)
        finally:dev.close()
    return rows,phases

def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    record=dict(schema_version='seaqr.raw16-exact-v9-generated.v1',real_media_read=False,
        native=args.native,quick=args.quick,package_sha256=package_identity(),
        checker_sha256=sha(__file__),plan_sha256=sha(ROOT/'docs/raw16_exact_v9_plan.md'),
        library_sha256=sha(DEFAULT_LIBRARY),cuda_source_sha256=sha(ROOT/'tiny_target/detection/cuda/warp_translation_v9.cu'),
        table_sha256=digest(cubic_table()),opencv_version=cv2.__version__,machine=platform.machine(),
        opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),passed=False)
    try:
        record['fft']=fft_checks(args.native,args.quick)
        record['warp'],record['phases']=warp_checks(args.native,args.quick)
        record['fft_exact']=all(v['exact'] and v['raw_exact'] for r in record['fft'] for v in r['workers'].values())
        record['warp_exact']=all(r['image']['exact'] and r['mask']['exact'] for r in record['warp']+record['phases'])
        if not args.quick:
            import check_raw16_speed_v8 as previous
            with patch.object(previous,'PointFilterCuda',PointFilterFftExact):
                record['moving']=previous.moving_checks()
            record['moving_exact']=len(record['moving'])==18 and all(
                r['candidate_decisions_exact'] and r['maximum_response_error']==0 for r in record['moving'])
            kernel,norm=filter_parameters();a=np.zeros((67,100),np.float32);a[33,50]=1
            unit=cpu_filter(a,kernel,norm)[33,50];center=np.float32(4/unit);lo=hi=center
            values=[center]
            for _ in range(16):
                lo=np.nextafter(lo,np.float32(-np.inf));hi=np.nextafter(hi,np.float32(np.inf));values.extend([lo,hi])
            device=PointFilterFftExact(a.shape,kernel,norm);record['thresholds']=[]
            try:
                for amplitude in values:
                    a.fill(0);a[33,50]=amplitude
                    record['thresholds'].append(dict(amplitude=float(amplitude),**compare(cpu_filter(a,kernel,norm),device(a))))
            finally:device.close()
        record['passed']=(record['fft_exact'] and record['warp_exact'] and
            (args.quick or (record['moving_exact'] and all(r['exact'] for r in record['thresholds']))))
    finally:write_json(args.output,record)
    print({k:record[k] for k in ('passed','fft_exact','warp_exact')},flush=True)
    return 0 if record['passed'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native',action='store_true');p.add_argument('--quick',action='store_true')
    raise SystemExit(run(p.parse_args()))
