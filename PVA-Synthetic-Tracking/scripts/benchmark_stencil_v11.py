"""Matched native generated v10/v11 resident tracking timings; not pipeline FPS."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from benchmark_resident_v10 import native_inputs,SHAPE,STEP_NS,legacy
from check_resident_v10 import reference_objects,digest_arrays
from resident_tracking_v10 import ResidentTracker
from profile_raw16_efficiency import sha,write_json

BASE_SHA='8a817884e89a8d158f20f694440dc6e17457a412a4c632470acc139c325a4668'


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if sha(args.reference)!=BASE_SHA:raise ValueError('Frozen v10 reference binary changed')
    for gate in (args.gate,args.edges):
        q=json.loads(gate.read_text())
        if q['passed'] is not True or q['library_sha256']!=sha(args.candidate):raise ValueError('Matching quality gates required')
    screen,reference,_=reference_objects()
    schedule=[dict(scene=scene,repeat=r,mode=mode) for scene in ('dense','holes') for r in range(4)
        for mode in (('reference','candidate') if r%2==0 else ('candidate','reference'))]
    record=dict(real_media_read=False,pipeline_benchmark=False,shape=SHAPE,schedule=schedule,trials=[],passed=False,
        library_sha256={'reference':BASE_SHA,'candidate':sha(args.candidate)},
        gate_sha256=sha(args.gate),edges_sha256=sha(args.edges),script_sha256=sha(__file__),
        wrapper_sha256=sha(ROOT/'scripts/resident_tracking_v10.py'),
        includes='Eight response/mask uploads, mirrored ring copies, unchanged search coverage; outputs stay on GPU',
        exclusions=['RAW front end','motion estimation','capture/decode','candidates/association','verification downloads/hashes'])
    try:
        for scene in ('dense','holes'):
            frames,masks=native_inputs(scene);bool_masks=masks.astype(bool);expected={}
            for phase in (0,8):
                ids=(np.arange(16)+phase)%16
                out,_,_=legacy(reference,np.ascontiguousarray(frames[ids]),np.ascontiguousarray(masks[ids]))
                expected[phase]=digest_arrays(out)
            for trial in [t for t in schedule if t['scene']==scene]:
                tick=time.perf_counter();p=ResidentTracker(SHAPE,reference.velocity_grid,getattr(args,trial['mode']))
                samples=[]
                try:
                    for i in range(16):p.push(frames[i],bool_masks[i],i,i*STEP_NS)
                    p.run()
                    if digest_arrays(p.download())!=expected[0]:raise AssertionError('Native prefill changed')
                    init=time.perf_counter()-tick
                    for cycle in range(3):
                        end=23+cycle*8;tick=time.perf_counter()
                        for i in range(end-7,end+1):p.push(frames[i%16],bool_masks[i%16],i,i*STEP_NS)
                        append=time.perf_counter()-tick;kernel=p.run();elapsed=time.perf_counter()-tick
                        exact=digest_arrays(p.download())==expected[(end-15)%16]
                        samples.append(dict(cycle=cycle,host_s=elapsed,append_host_s=append,kernel_ms=kernel,outputs_exact=exact))
                        if not exact:raise AssertionError('Native timing output differs from frozen tracker')
                finally:p.close()
                record['trials'].append(dict(**trial,initialization_s=init,samples=samples))
                print(json.dumps(dict(**trial,kernel_ms=[s['kernel_ms'] for s in samples],host_s=[s['host_s'] for s in samples])),flush=True)
        record['passed']=len(record['trials'])==16
    finally:screen.close();write_json(args.output,record)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','gate','edges','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
