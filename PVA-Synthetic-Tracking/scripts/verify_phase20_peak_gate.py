"""Media-free exact kernel oracle, adversarial thresholds/ties, paired device timing."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import sha256
from tiny_target.visible_resident import PEAK_DTYPE


class Core:
    def __init__(self, path, shape, tile):
        self.lib=C.CDLL(str(path.resolve()))
        ptr=C.c_void_p; i=C.c_int; f=C.c_float
        for name,(args,ret) in {
            'seaqr_resident_create':([i,i,i,i,ptr],ptr),'seaqr_resident_destroy':([ptr],None),
            'seaqr_test_seed':([ptr]*6,i), 'seaqr_resident_select':([ptr,ptr,i,i,f,f,ptr,ptr],i),
            'seaqr_test_kernel_time':([ptr,ptr,i,i,f,f,i,ptr],i)}.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ret
        index=np.array([0],np.int32)
        self.handle=self.lib.seaqr_resident_create(*shape,tile,1,index.ctypes.data)
        if not self.handle:raise RuntimeError('Test allocation failed')
        self.cells=2*((shape[0]+tile-1)//tile)*((shape[1]+tile-1)//tile)

    def check(self, code):
        if code:raise RuntimeError('CUDA conformance failure: '+str(code))

    def seed(self, fields):
        self.check(self.lib.seaqr_test_seed(self.handle,*(a.ctypes.data for a in fields)))

    def select(self, stats, k, ready, threshold, spatial_threshold):
        peaks=np.empty((self.cells,k),PEAK_DTYPE);counts=np.empty(self.cells,np.int32)
        self.check(self.lib.seaqr_resident_select(self.handle,stats.ctypes.data,k,ready,
            threshold,spatial_threshold,peaks.ctypes.data,counts.ctypes.data))
        return peaks,counts

    def timed(self, stats,k,ready,threshold,spatial_threshold,iterations=8):
        milliseconds=C.c_float()
        self.check(self.lib.seaqr_test_kernel_time(self.handle,stats.ctypes.data,k,ready,
            threshold,spatial_threshold,iterations,C.byref(milliseconds)))
        return milliseconds.value/iterations

    def close(self):
        if self.handle:self.lib.seaqr_resident_destroy(self.handle);self.handle=None


def equal_outputs(a,b):
    # Raw bytes retain score/response/noise bits, signed zero and invalid slots.
    return all(x.shape==y.shape and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def scalar_oracle(fields,stats,tile,k,ready,threshold,spatial_threshold):
    temporal,spatial,variance,support,previous=fields
    h,w=temporal.shape;nx=(w+tile-1)//tile
    peaks=np.zeros((2*len(stats),k),PEAK_DTYPE);peaks['x']=-1;peaks['y']=-1
    counts=np.zeros(2*len(stats),np.int32)
    if not ready:return peaks,counts
    for cell in range(len(peaks)):
        number=cell//2; sign=np.float32(1 if cell%2==0 else -1)
        x0=(number%nx)*tile;y0=(number//nx)*tile
        center,sigma=stats[number];found=[]
        for y in range(y0,min(h,y0+tile)):
            for x in range(x0,min(w,x0+tile)):
                if not support[y,x] or not previous[y,x]:continue
                r=temporal[y,x]
                if np.any(np.abs(temporal[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)])>abs(r)):continue
                noise=np.fmax(sigma,np.sqrt(variance[y,x],dtype=np.float32))
                signed=np.float32(sign*np.float32(r-center))
                if signed<np.float32(threshold*noise) or np.float32(sign*spatial[y,x])<np.float32(spatial_threshold*noise):continue
                counts[cell]+=1
                found.append((x,y,np.float32(signed/noise),r,noise))
        found.sort(key=lambda p:(-float(p[2]),p[1],p[0]))
        for j,p in enumerate(found[:k]):peaks[cell,j]=p
    return peaks,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-library','candidate-library','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--small-only',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite conformance evidence')
    builds=[]
    for lib,variant in ((a.reference_library,'reference'),(a.candidate_library,'threshold_before_local_max')):
        build=json.loads(lib.with_suffix('.so.build.json').read_text())
        if not build['diagnostic_only'] or build['variant']!=variant or sha256(lib)!=build['library_sha256']:
            raise ValueError('Test library provenance changed')
        for name, expected in build['sources_sha256'].items():
            if sha256(ROOT/'scripts'/name)!=expected:raise ValueError('Test source changed')
        builds.append(build)
    record=dict(passed=False,media_accessed=False,synthetic_pairs=0,scalar_oracle_pairs=0,
        native_size_synthetic_pairs=0,builds=builds,script_sha256=sha256(__file__),timings=[])
    current=[];rng=np.random.default_rng(915208)
    try:
        shapes=[(1,1),(1,19),(19,1),(3,5),(7,17),(33,65),(67,99),(129,259)]
        for shape in shapes:
            for tile in (1,16,64,256):
                current=[Core(lib,shape,tile) for lib in (a.reference_library,a.candidate_library)]
                nt=current[0].cells//2
                for pattern in ('noise','ties','flat','boundary','sparse','masked','cold'):
                    t=rng.normal(0,3,shape).astype(np.float32)
                    s=rng.normal(0,5,shape).astype(np.float32)
                    v=rng.choice(np.array([.25,1,4,16],np.float32),size=shape)
                    support=(rng.random(shape)>.1).astype(np.uint8)
                    previous=(rng.random(shape)>.1).astype(np.uint8)
                    stats=np.zeros((nt,2),np.float32);stats[:,1]=1
                    threshold=np.float32(2);sth=np.float32(1)
                    if pattern=='ties':t[:]=4;s[:]=4;v[:]=1;support[:]=previous[:]=1
                    if pattern=='flat':t[:]=s[:]=0;v[:]=1;support[:]=previous[:]=1;threshold=sth=np.float32(0)
                    if pattern=='boundary':
                        t=rng.choice(np.array([-2,np.nextafter(np.float32(-2),np.float32(0)),
                            np.nextafter(np.float32(2),np.float32(0)),2,np.nextafter(np.float32(2),np.float32(3))],np.float32),size=shape)
                        s=t.copy();v[:]=1;support[:]=previous[:]=1
                    if pattern=='sparse':t[:]=s[:]=0;t[::7,::7]=s[::7,::7]=20
                    if pattern=='masked':support[:]=0
                    if pattern=='noise':stats[:,0]=rng.uniform(-1,1,nt).astype(np.float32)
                    fields=[t,s,v,support,previous]
                    for c in current:c.seed(fields)
                    for k in (1,3,16):
                        ready=int(pattern!='cold')
                        values=[c.select(stats,k,ready,threshold,sth) for c in current]
                        if not equal_outputs(*values):raise AssertionError(f'Peak output changed: {shape,tile,pattern,k}')
                        record['synthetic_pairs']+=1
                        if shape[0]*shape[1]<=1120:
                            expected=scalar_oracle(fields,stats,tile,k,ready,threshold,sth)
                            if not equal_outputs(values[0],expected):raise AssertionError(f'Scalar oracle mismatch: {shape,tile,pattern,k}')
                            record['scalar_oracle_pairs']+=1
                for c in current:c.close()
                current=[]
        if not a.small_only:
            shape=(3190,4784);tile=256
            current=[Core(lib,shape,tile) for lib in (a.reference_library,a.candidate_library)]
            nt=current[0].cells//2
            for pattern in ('quiet','sparse','dense_plateau'):
                t=rng.normal(0,.6,shape).astype(np.float32);s=t.copy();v=np.ones(shape,np.float32)
                support=np.ones(shape,np.uint8);previous=support.copy()
                stats=np.zeros((nt,2),np.float32);stats[:,1]=1
                if pattern=='sparse':t[::101,::103]=s[::101,::103]=12
                if pattern=='dense_plateau':t[:]=s[:]=4
                fields=[t,s,v,support,previous]
                for c in current:c.seed(fields)
                values=[c.select(stats,12,1,2,1) for c in current]
                if not equal_outputs(*values):raise AssertionError('Native-size synthetic mismatch')
                record['native_size_synthetic_pairs']+=1
                for c in current:c.timed(stats,12,1,2,1,iterations=4)
                samples={'before':[],'after':[]}
                for pair in range(12):
                    order=(0,1) if pair%2==0 else (1,0)
                    for j in order:samples['before' if j==0 else 'after'].append(current[j].timed(stats,12,1,2,1))
                record['timings'].append(dict(pattern=pattern,alternating_pairs=12,kernel_only=True,
                    samples_ms=samples,median_ms={key:float(np.median(v)) for key,v in samples.items()}))
        record.update(passed=True,exact_all_peak_bytes_and_counts=True,full_clip_fps_claim=False)
    except Exception as exc:
        record['error']=repr(exc)
        raise
    finally:
        for c in current:c.close()
        with a.output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='builds'},indent=2))


if __name__=='__main__':main()
