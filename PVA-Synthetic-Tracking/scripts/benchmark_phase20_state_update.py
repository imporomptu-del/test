"""Exact masked-state-update microbenchmark; no detector-policy changes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def original(background, variance, temporal, support, learn, alpha=.05, noise_alpha=.1, clip=4.):
    background[support] += alpha * temporal[support]
    observed=np.minimum(temporal*temporal,variance*clip**2)
    variance[learn] += noise_alpha*(observed[learn]-variance[learn])
    np.maximum(variance,.25,out=variance)


def inplace(background, variance, temporal, support, learn, scratch, observed, alpha=.05, noise_alpha=.1, clip=4.):
    np.multiply(temporal,alpha,out=scratch)
    np.add(background,scratch,out=background,where=support)
    np.multiply(temporal,temporal,out=observed)
    np.multiply(variance,clip**2,out=scratch)
    np.minimum(observed,scratch,out=observed)
    np.subtract(observed,variance,out=observed)
    np.multiply(observed,noise_alpha,out=observed)
    np.add(variance,observed,out=variance,where=learn)
    np.maximum(variance,.25,out=variance)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output exists')
    rng=np.random.default_rng(1234);results=[]
    for shape in [(67,99),(3190,4784)]:
        b=rng.normal(0,10,shape).astype(np.float32);v=np.full(shape,.25,np.float32)
        old_b,new_b=b.copy(),b.copy();old_v,new_v=v.copy(),v.copy()
        scratch=np.empty_like(b);observed=np.empty_like(b);times={'original':[],'inplace':[]}
        for i in range(24):
            t=rng.normal(0,10,shape).astype(np.float32)
            support=rng.random(shape)>.05;learn=support & (rng.random(shape)>.02)
            for mode in (['original','inplace'] if i%2==0 else ['inplace','original']):
                start=time.perf_counter()
                if mode=='original':original(old_b,old_v,t,support,learn)
                else:inplace(new_b,new_v,t,support,learn,scratch,observed)
                if i>=4:times[mode].append(1000*(time.perf_counter()-start))
            if not np.array_equal(old_b,new_b) or not np.array_equal(old_v,new_v):
                raise AssertionError('State differs')
        results.append(dict(shape=list(shape),exact_frame_pairs=24,
            median_ms={k:float(np.median(v)) for k,v in times.items()},samples_ms=times))
    with a.output.open('x') as f:json.dump(dict(results=results,scope='Synthetic masked-state microbenchmark, not pipeline FPS'),f,indent=2)
    print(json.dumps(results))


if __name__=='__main__':main()
