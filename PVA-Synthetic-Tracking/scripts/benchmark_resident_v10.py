"""Generated native-resolution residency benchmark, never a pipeline FPS claim."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from check_resident_v10 import reference_objects,digest_arrays,REFERENCE_LIBRARY_SHA
from resident_tracking_v10 import ResidentTracker
from profile_raw16_efficiency import sha,write_json

SHAPE=(3190,4784)
ROUNDS=4
CYCLES=3
STEP_NS=100000000


def schedule():
    modes=('stateless','resident_host','resident_device')
    return [dict(scene=scene,repeat=repeat,mode=mode)
        for scene in ('dense','holes') for repeat in range(ROUNDS)
        for mode in (modes if repeat%2==0 else modes[::-1])]


def legacy(reference,frames,masks):
    score=np.empty(SHAPE,np.float32);velocity=np.empty(SHAPE,np.uint16)
    support=np.empty(SHAPE,np.uint16);valid=np.empty(SHAPE,np.uint8)
    timestamps=np.arange(16,dtype=np.int64)*STEP_NS
    offsets=(timestamps.astype(np.float64)-int(timestamps[-1]//2))/1e9
    timing=np.zeros(5,np.float32);counters=np.zeros(7,np.uint64);launch=np.zeros(12,np.int32)
    error=C.create_string_buffer(2048)
    status=reference._function(frames.ctypes.data,masks.ctypes.data,offsets.ctypes.data,
        reference.velocity_grid.ctypes.data,16,48,*SHAPE,12,1,32,256,0,
        score.ctypes.data,velocity.ctypes.data,support.ctypes.data,valid.ctypes.data,
        timing.ctypes.data,counters.ctypes.data,launch.ctypes.data,error,len(error))
    if status:raise RuntimeError(error.value.decode())
    return [score,velocity,support,valid],timing.tolist(),counters.tolist()


def native_inputs(scene):
    rng=np.random.default_rng(161001)
    # Pre-generated response-space stimuli; generation is deliberately excluded.
    frames=rng.standard_normal((16,*SHAPE),dtype=np.float32)
    masks=np.ones((16,*SHAPE),np.uint8)
    masks[:,:4,:]=0;masks[:,-4:,:]=0;masks[:,:,:4]=0;masks[:,:,-4:]=0
    if scene=='holes':masks[:,SHAPE[0]//3:2*SHAPE[0]//3,SHAPE[1]//4:3*SHAPE[1]//4]=0
    for i in range(16):
        masks[i,100+i:110+i,100:130]=0
        frames[i,SHAPE[0]//2,SHAPE[1]//2+i//4]+=4
        frames[i][masks[i]==0]=0
    return frames,masks


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    gate=json.loads(args.gate.read_text())
    if (gate['passed'] is not True or gate['library_sha256']!=sha(args.library)
            or gate['wrapper_sha256']!=sha(ROOT/'scripts/resident_tracking_v10.py')):
        raise ValueError('A matching complete quality gate is required')
    screen,reference,_=reference_objects()
    record=dict(schema='seaqr.resident-v10-timing.v1',real_media_read=False,shape=SHAPE,
        target_fps=10,window_frames=16,stride_frames=8,velocity_count=48,
        gate_sha256=sha(args.gate),library_sha256=sha(args.library),
        reference_library_sha256=REFERENCE_LIBRARY_SHA,script_sha256=sha(__file__),
        plan_sha256=sha(ROOT/'docs/raw16_feasibility_v10_plan.md'),
        schedule=schedule(),trials=[],passed=False,pipeline_benchmark=False,
        exclusions=['source generation/capture/decode','motion estimation','warp/background/filter',
            'candidate extraction','association','audit hashes and verification downloads for device-only mode'],
        warning='Resident device mode still uploads new CPU response frames; it omits map downloads. '
            'This is an isolated tracking-stage experiment, not end-to-end RAW16 throughput.')
    try:
        for scene in ('dense','holes'):
            frames,masks=native_inputs(scene)
            bool_masks=masks.astype(bool)
            expected={}
            for phase in (0,8):
                ids=(np.arange(16)+phase)%16
                out,_,_=legacy(reference,np.ascontiguousarray(frames[ids]),np.ascontiguousarray(masks[ids]))
                expected[phase]=digest_arrays(out)
            for trial in [r for r in schedule() if r['scene']==scene]:
                started=time.perf_counter();device=None
                if trial['mode']!='stateless':
                    device=ResidentTracker(SHAPE,reference.velocity_grid,args.library)
                    for i in range(16):device.push(frames[i],bool_masks[i],i,i*STEP_NS)
                    device.run();out=device.download()
                else:out,_,_=legacy(reference,frames,masks)
                if digest_arrays(out)!=expected[0]:raise AssertionError('Warmup output changed')
                initialization=time.perf_counter()-started
                samples=[]
                try:
                    for cycle in range(CYCLES):
                        end=23+cycle*8;phase=(end-15)%16
                        start=time.perf_counter();upload=0.;download=0.;packing=0.
                        native_timings=None;native_counters=None
                        if device is None:
                            tick=time.perf_counter();ids=np.arange(end-15,end+1)%16
                            stack=np.ascontiguousarray(frames[ids]);valid=np.ascontiguousarray(masks[ids])
                            packing=time.perf_counter()-tick
                            out,native_timings,native_counters=legacy(reference,stack,valid)
                            kernel_ms=native_timings[2]
                        else:
                            tick=time.perf_counter()
                            for i in range(end-7,end+1):
                                device.push(frames[i%16],bool_masks[i%16],i,i*STEP_NS)
                            upload=time.perf_counter()-tick
                            kernel_ms=device.run()
                            if trial['mode']=='resident_host':
                                tick=time.perf_counter();out=device.download();download=time.perf_counter()-tick
                        elapsed=time.perf_counter()-start
                        if trial['mode']=='resident_device':out=device.download()
                        exact=digest_arrays(out)==expected[phase]
                        samples.append(dict(cycle=cycle,advance_frames=8,host_s=elapsed,
                            append_host_s=upload,packing_s=packing,download_host_s=download,
                            kernel_ms=kernel_ms,native_timings_ms=native_timings,
                            native_counters=native_counters,outputs_exact=exact))
                        if not exact:raise AssertionError('Timed output changed')
                finally:
                    if device:device.close()
                record['trials'].append(dict(**trial,initialization_and_warmup_s=initialization,samples=samples))
                print(json.dumps(dict(**trial,host_s=[s['host_s'] for s in samples],
                    kernel_ms=[s['kernel_ms'] for s in samples])),flush=True)
            del frames,masks,bool_masks
        record['passed']=len(record['trials'])==24 and all(s['outputs_exact'] for t in record['trials'] for s in t['samples'])
    finally:
        screen.close();write_json(args.output,record)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','gate','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
