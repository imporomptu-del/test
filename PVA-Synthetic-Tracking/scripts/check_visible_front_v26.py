"""Generated, media-free exact connected-front conformance and feasibility."""
import argparse
import ctypes as C
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import cv2
import numpy as np
from profile_visible_v17 import read,sha,write
from visible_front_v26 import ResidentFrontV26,pack_learning,attach_warp,signatures,REFERENCE_LIBRARY_SHA
from tiny_target.visible_baseline import VisibleConfig,VisibleTracks
from tiny_target.visible_resident import VisibleCudaResident,sample_layout
from tiny_target.visible_noise import tile_noise_statistics
from tiny_target.visible_learning import shape_learning_mask
from tiny_target.visible_warp_exact import CudaCubicTranslation


def equal(a,b,label):
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
        bad=np.argwhere(a!=b)
        raise AssertionError(f'{label}: array mismatch shape={a.shape}; first={bad[:2].tolist()}')


def noise_cases():
    rng=np.random.default_rng(2631)
    for shape,tile,stride in (((1,1),1,1),((9,13),16,1),((67,99),32,3),
                              ((256,256),256,4),((257,319),256,4),((65,129),64,1)):
        for kind in ('random','empty','sparse','ties','zeros','threshold_neighbors'):
            values=rng.normal(0,3,shape).astype(np.float32);mask=np.ones(shape,np.bool_)
            if kind=='empty':mask[:]=False
            if kind=='sparse':mask=rng.random(shape)<.1
            if kind=='ties':values=np.resize(np.array([-4,-1,0,0,0,1,4],np.float32),shape)
            if kind=='zeros':values=np.resize(np.array([-0.,0.],np.float32),shape)
            if kind=='threshold_neighbors':
                values=np.resize(np.array([np.nextafter(np.float32(.5),np.float32(0)),.5,
                                           np.nextafter(np.float32(.5),np.float32(1))],np.float32),shape)
            yield f'{shape}_{tile}_{stride}_{kind}',values,mask,tile,stride
    for n in (1,2,3,31,32,63,64,65,4095,4096):
        values=rng.normal(0,4,(64,64)).astype(np.float32)
        mask=np.zeros(values.shape,np.bool_);mask.flat[:n]=True
        yield f'count_{n}',values,mask,64,1
    values=np.resize(np.array([0,np.nextafter(np.float32(0),np.float32(1)),1e-37,-1e-37],np.float32),(64,64))
    yield 'subnormal',values,np.ones(values.shape,np.bool_),64,1


def check_noise(library,rows):
    lib=C.CDLL(str(library.resolve()));signatures(lib)
    for name,temporal,support,tile,stride in noise_cases():
        front=lib.seaqr_front_v26_create(*temporal.shape,tile,stride)
        if not front:raise AssertionError('Noise probe allocation failed')
        try:
            layout,indices=sample_layout(temporal.shape,tile,stride)
            sample=temporal.ravel()[indices];expected,median=tile_noise_statistics(sample,support,layout,stride,.5)
            # Preserve each tile sigma before the final CPU median, not just that median.
            sigmas=[]
            for ys,xs,offset,count in layout:
                selected=sample[offset:offset+count][support[ys,xs][::stride,::stride].ravel()]
                sigmas.append(max(.5,1.4826*float(np.median(np.abs(selected-np.median(selected))))) if len(selected) else .5)
            actual=np.empty_like(expected);actual_sigmas=np.empty(len(layout),np.float64)
            code=lib.seaqr_front_v26_noise_probe(front,temporal.ctypes.data,support.ctypes.data,.5,
                                                actual.ctypes.data,actual_sigmas.ctypes.data)
            if code:raise RuntimeError(f'Noise probe failed {name}: {code}')
            equal(actual,expected,name+' stats');equal(actual_sigmas,np.array(sigmas,np.float64),name+' sigmas')
            if float(np.median(actual_sigmas))!=median:raise AssertionError('Sigma telemetry differs')
            rows.append(dict(name=name,exact=True))
        finally:lib.seaqr_front_v26_destroy(front)
    for value in (np.inf,-np.inf,np.nan):
        front=lib.seaqr_front_v26_create(9,13,16,1)
        try:
            data=np.zeros((9,13),np.float32);data[0,0]=value;mask=np.ones_like(data,dtype=np.bool_)
            stats=np.empty((1,2),np.float32);sigma=np.empty(1,np.float64)
            code=lib.seaqr_front_v26_noise_probe(front,data.ctypes.data,mask.ctypes.data,.5,stats.ctypes.data,sigma.ctypes.data)
            if code==0:raise AssertionError('Nonfinite supported sample accepted')
            rows.append(dict(name='reject_'+repr(value),exact=True))
        finally:lib.seaqr_front_v26_destroy(front)
    if lib.seaqr_front_v26_create(256,256,256,1):raise AssertionError('Unbounded sample tile accepted')


def compare(left,right,image_l,image_r,valid_l,valid_r,segment,centers,name,rows,reverse=False):
    before=(valid_l.tobytes(),valid_r.tobytes());result={};elapsed={}
    for key,d,image,valid in ((('candidate',right,image_r,valid_r),('reference',left,image_l,valid_l)) if reverse
                            else (('reference',left,image_l,valid_l),('candidate',right,image_r,valid_r))):
        start=time.perf_counter();result[key]=d.update(image,valid,segment,centers)
        elapsed[key]=1000*(time.perf_counter()-start);result[key][1].pop('detection_ms')
    if result['reference']!=result['candidate']:
        raise AssertionError(name+' proposals/coverage mismatch: '+repr(result)[:1500])
    if before!=(valid_l.tobytes(),valid_r.tobytes()):raise AssertionError('Input mask mutated')
    for n,a,b in zip(('background','variance'),left.debug_state(),right.debug_state()):equal(a,b,name+' '+n)
    support,learn,stats,sigmas=right.debug_front()
    expected_support=cv2.erode(valid_l.astype(np.uint8),np.ones((13,13),np.uint8),
                              borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    equal(support,expected_support,name+' support')
    expected_learn=shape_learning_mask(support,centers,right.config.position_sigma_px) if centers else support
    equal(learn,expected_learn,name+' learning')
    equal(stats,left.stats,name+' detector statistics')
    rows.append(dict(name=name,exact=True,shape=list(valid_l.shape),detector_ms=elapsed))
    return result['reference'][0]


def host_sequences(cfg,library,rows):
    rng=np.random.default_rng(2632)
    for margin in (.5,2,4.25):
        for shape in ((7,13),(67,99),(257,319)):
            params=dict(tile_size=32 if max(shape)<256 else 256,noise_sample_stride=3 if max(shape)<256 else 4,
                        warmup_frames=2,max_candidates_per_tile_polarity=3,max_candidates_per_frame=40,
                        position_sigma_px=margin)
            c=replace(cfg,**params);left=VisibleCudaResident(c);right=ResidentFrontV26(replace(c,cuda_median_library=str(library.resolve())))
            try:
                for i in range(12):
                    im=rng.normal(20,3,shape).astype(np.float32)
                    if min(shape)>30:im[30,20+i]=160;im[40,70-i]=0
                    if i in (0,7):im[:]=20
                    valid=rng.random(shape)>.003
                    if i==5:valid[:]=False
                    centers=[] if i%3==0 else [dict(support_reference_xy=[[i+15.5,20.5],[-1,4],[4,999]])]
                    if i==10:centers=[dict(support_reference_xy=rng.integers(0,min(shape),(1024,2)).tolist()) for _ in range(7)]
                    compare(left,right,im,im,valid,valid,i//6,centers,f'host_margin{margin}_{shape}_{i}',rows,bool(i%2))
            finally:left.close();right.close();right.close()


def device_sequences(cfg,library,rows):
    rng=np.random.default_rng(2633)
    for shape,frames in (((67,99),12),((3190,4784),10)):
        c=replace(cfg,warmup_frames=2) if min(shape)<100 else cfg
        left=VisibleCudaResident(c);right=ResidentFrontV26(replace(c,cuda_median_library=str(library.resolve())))
        warps=[CudaCubicTranslation(c.cuda_median_library),CudaCubicTranslation(library)]
        tx=VisibleTracks(c,10);ty=VisibleTracks(c,10)
        try:
            with patch.object(CudaCubicTranslation,'__call__',attach_warp(CudaCubicTranslation.__call__)):
                for i in range(frames):
                    image=rng.integers(0,256,shape,dtype=np.uint8);mask=np.ones(shape,np.uint8)
                    if min(shape)<100:mask[25:27,30:32]=(i%3)*100
                    matrix=np.eye(3);matrix[:2,2]=[(i%5-2)/3,(i%7-3)/5]
                    erosion=(0,1,2,5,16)[i%5] if min(shape)<100 else 2
                    a,va=warps[0](image,mask,matrix,device=True,erosion_px=erosion)
                    b,vb=warps[1](image,mask,matrix,device=True,erosion_px=erosion)
                    equal(va,vb,'warp validity')
                    segment=i//6 if min(shape)<100 else 0;timestamp=i*100000000
                    ca=tx.learning_centers(timestamp,segment);cb=ty.learning_centers(timestamp,segment)
                    if ca!=cb:raise AssertionError('Closed-loop learning centers differ')
                    proposals=compare(left,right,a,b,va,vb,segment,ca,f'device_{shape}_{i}',rows,bool(i%2))
                    ra,_=tx.update(proposals,i,timestamp,segment,matrix,shape)
                    rb,_=ty.update(proposals,i,timestamp,segment,matrix,shape)
                    if ra!=rb:raise AssertionError('Closed-loop tracks differ')
                    try:right.update(b,vb,segment,ca)
                    except ValueError:pass
                    else:raise AssertionError('Consumed device frame accepted')
        finally:
            left.close();right.close()
            for w in warps:w.close()


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    if sha(args.reference)!=REFERENCE_LIBRARY_SHA:raise ValueError('Changed reference GPU')
    cfg=VisibleConfig(**read(args.config));cv2.setNumThreads(cfg.opencv_threads)
    if cfg.cuda_median_library!=str(args.reference):raise ValueError('Unexpected original configuration')
    record=dict(passed=False,error=None,real_media_read=False,noise=[],detector=[],conformance=None,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in (
            'visible_front_v26.cu','visible_front_v26.py','build_visible_front_v26.py','check_visible_front_v26.py')},
        plan_sha256=sha(Path(__file__).with_name('visible_front_v26_plan.md')),
        library_sha256=sha(args.library),reference_library_sha256=sha(args.reference),config_sha256=sha(args.config),
        build_sha256=sha(args.library.parent/'build.json'),
        build_relative=str((args.library.parent/'build.json').resolve().relative_to(Path(__file__).resolve().parent)),
        library_relative=str(args.library.resolve().relative_to(Path(__file__).resolve().parent)))
    try:
        warp=CudaCubicTranslation(args.library)
        try:record['conformance']=warp.verify_reference(include_gaussian=True)
        finally:warp.close()
        check_noise(args.library,record['noise']);print('NOISE EXACT '+str(len(record['noise'])),flush=True)
        host_sequences(cfg,args.library,record['detector']);print('HOST EXACT '+str(len(record['detector'])),flush=True)
        device_sequences(cfg,args.library,record['detector']);print('DEVICE EXACT '+str(len(record['detector'])),flush=True)
        record['passed']=True
    except BaseException as exc:record.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        write(args.output,record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('noise','detector','traceback')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
