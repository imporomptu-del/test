"""Generated-only production/probe CUDA pixel conformance and separate timings."""
import argparse
import ctypes as C
from pathlib import Path
import time
import cv2
import numpy as np
from profile_visible_v17 import sha,read,write
from median_v19 import REFERENCE_LIBRARY_SHA


class Median:
    def __init__(self,path,probe=False):
        self.lib=C.CDLL(str(Path(path).resolve(strict=True)))
        self.lib.seaqr_create.restype=C.c_void_p
        self.lib.seaqr_destroy.argtypes=[C.c_void_p]
        self.lib.seaqr_median.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int]
        self.lib.seaqr_median.restype=C.c_int
        if hasattr(self.lib,'seaqr_median_event') != probe:
            raise ValueError('Production/diagnostic ABI separation failed')
        if probe:
            self.lib.seaqr_median_event.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p]
            self.lib.seaqr_median_event.restype=C.c_int
        self.handle=self.lib.seaqr_create()
        if not self.handle:raise RuntimeError('GPU workspace allocation failed')

    def __call__(self,source):
        source=np.ascontiguousarray(source,dtype=np.float32)
        output=np.empty_like(source)
        code=self.lib.seaqr_median(self.handle,source.ctypes.data,output.ctypes.data,*source.shape)
        if code:raise RuntimeError('GPU median failed '+str(code))
        return output

    def event(self,shape,iterations=8):
        elapsed=C.c_float()
        code=self.lib.seaqr_median_event(self.handle,*shape,iterations,C.byref(elapsed))
        if code:raise RuntimeError('GPU event timing failed '+str(code))
        return elapsed.value/iterations

    def close(self):
        if self.handle:self.lib.seaqr_destroy(self.handle);self.handle=None


def generated():
    rng=np.random.default_rng(190918)
    shapes=((1,1),(1,2),(1,31),(2,1),(31,1),(2,3),(3,7),(4,4),(5,5),
            (7,9),(17,31),(32,64),(33,65),(67,129))
    for h,w in shapes:
        for pattern in range(16):
            a=rng.normal(30,70,(h,w)).astype(np.float32)
            if pattern==1:a[:]=13.5
            if pattern==2:a[:]=0;a[h//2,w//2]=255
            if pattern==3:a=(np.arange(h*w).reshape(h,w)%257).astype(np.float32)
            if pattern==4:a=(np.arange(h*w,0,-1).reshape(h,w)%257).astype(np.float32)
            if pattern==5:a[:]=rng.choice([-5.,0.,12.,255.],(h,w))
            if pattern==6:a[:]=np.nextafter(np.float32(1),np.float32(2));a[::2,::2]=1
            if pattern==7:a[:]=0
            bits=a.view(np.uint32)
            if pattern==8:bits[:]=np.resize(np.array([0,0x80000000],np.uint32),(h,w))
            if pattern==9:bits[:]=np.resize(np.array([1,0x80000001,0x007fffff,0x807fffff,0x00800000],np.uint32),(h,w))
            if pattern==10:bits[:]=np.resize(np.array([0x7fc00001,0xffc00401,0x3f800000,0x40000000],np.uint32),(h,w))
            if pattern==11:bits[:]=np.resize(np.array([0x7f800000,0xff800000,0,0x3f800000],np.uint32),(h,w))
            if pattern==12:bits[:]=0x7fc00317
            if pattern==13:bits[:]=0x80000000
            if pattern==14:bits[:]=np.resize(np.array([0x7f7fffff,0xff7fffff,0x3f800000],np.uint32),(h,w))
            if pattern==15:bits[h//2,w//2]=0x7fa00001
            yield f'{h}x{w}_pattern{pattern}',a,pattern<8
    for i in range(32):
        a=rng.permutation(np.arange(-12,13,dtype=np.float32)).reshape(5,5)
        a.flat[i%25]=np.nextafter(a.flat[i%25],np.float32(100))
        yield f'permuted_rank_{i}',a,True
    for i in range(3):
        a=rng.normal(40,30,(3190,4784)).astype(np.float32)
        if i==1:a[:]=0
        if i==2:a.view(np.uint32)[::127,::131]=0x80000000
        yield f'native_{i}',a,i!=2


def run(build,reference,output):
    metadata=read(build/'build.json')
    if sha(reference)!=REFERENCE_LIBRARY_SHA or not metadata['passed'] or metadata['proof']['cases']!=1<<25:
        raise ValueError('Missing frozen build and exhaustive proof')
    for name,r in metadata['builds'].items():
        if sha(build/(name+'.so'))!=r['library_sha256']:raise ValueError('Changed library')
    here=Path(__file__).resolve().parent
    record=dict(passed=False,error=None,real_media_read=False,build_sha256=sha(build/'build.json'),
        reference_library_sha256=sha(reference),candidate_library_sha256=sha(build/'candidate.so'),
        script_sha256=sha(__file__),plan_sha256=sha(here/'visible_speed_v19_plan.md'),
        cases=[],timings=[],opencv_version=cv2.__version__,numpy_version=np.__version__)
    libraries={}
    try:
        libraries['reference']=Median(reference)
        libraries['candidate']=Median(build/'candidate.so')
        for name in ('reference_probe','candidate_probe'):libraries[name]=Median(build/(name+'.so'),True)
        cv2.setNumThreads(2)
        import hashlib
        for name,a,cpu in generated():
            before=a.tobytes();expected=libraries['reference'](a)
            for mode in ('candidate','reference_probe','candidate_probe'):
                actual=libraries[mode](a)
                if actual.tobytes()!=expected.tobytes():raise AssertionError('GPU pixel mismatch '+name+' '+mode)
            if cpu and cv2.medianBlur(a,5).tobytes()!=expected.tobytes():
                raise AssertionError('Independent CPU median mismatch '+name)
            if before!=a.tobytes():raise AssertionError('Input mutated')
            record['cases'].append(dict(name=name,shape=list(a.shape),exact=True,cpu_oracle=cpu,
                inputs_unchanged=True,output_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
        for scene in ('random','flat','guarded'):
            a=np.random.default_rng(191).normal(40,30,(3190,4784)).astype(np.float32)
            if scene=='flat':a[:]=13
            if scene=='guarded':a.view(np.uint32)[::7,::11]=0x80000000
            for lib in libraries.values():lib(a)
            for repeat in range(4):
                for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference')):
                    start=time.perf_counter();libraries[mode](a);wall_ms=1000*(time.perf_counter()-start)
                    event_ms=libraries[mode+'_probe'].event(a.shape)
                    record['timings'].append(dict(scene=scene,repeat=repeat,mode=mode,
                        production_with_transfers_ms=wall_ms,isolated_kernel_event_ms=event_ms))
        record['passed']=True
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:
        for lib in libraries.values():lib.close()
        write(output,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.build,a.reference,a.output)
