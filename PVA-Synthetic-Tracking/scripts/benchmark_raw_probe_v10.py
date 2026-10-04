"""Generated RAW16 GPU core timing; excludes motion estimation and decisions."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from resident_frontend_v10 import ResidentRawProbe
from check_raw_probe_v10 import generated
from check_resident_v10 import reference_objects,digest_arrays
from profile_raw16_efficiency import sha,write_json

SHAPE=(3190,4784)
ENDS=(27,35,43)


def schedule():
    modes=('resident_device','resident_host')
    return [dict(scene=scene,repeat=repeat,mode=mode)
        for scene in ('dense','holes') for repeat in range(4)
        for mode in (modes if repeat%2==0 else modes[::-1])]


def matching_gate(args):
    gate=json.loads(args.gate.read_text())
    if (gate['implementation_integrity_passed'] is not True
        or gate['library_sha256']!=sha(args.library)
        or gate['wrapper_sha256']!=sha(ROOT/'scripts/resident_frontend_v10.py')
        or gate['script_sha256']!=sha(ROOT/'scripts/check_raw_probe_v10.py')
        or len(gate['frames'])!=516 or len(gate['windows'])!=48 or len(gate['native'])!=14):
        raise ValueError('Complete matching RAW implementation gate required')
    # A failed FFT equivalence gate is retained, never reclassified as accepted.
    return gate


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    gate=matching_gate(args)
    screen,tracker,_=reference_objects()
    record=dict(schema='seaqr.raw-probe-v10-timing.v1',shape=SHAPE,real_media_read=False,
        generated_data_only=True,production_approved=False,pipeline_benchmark=False,
        gate_sha256=sha(args.gate),library_sha256=sha(args.library),script_sha256=sha(__file__),
        wrapper_sha256=sha(ROOT/'scripts/resident_frontend_v10.py'),
        generator_sha256=sha(ROOT/'scripts/check_raw_probe_v10.py'),
        plan_sha256=sha(ROOT/'docs/raw16_feasibility_v10_plan.md'),
        strict_cpu_fft_equivalent=gate['strict_cpu_fft_equivalent'],
        generated_sensitivity_gate_passed=gate['generated_sensitivity_gate_passed'],
        window_frames=16,stride_frames=8,velocity_count=48,target_fps=10,target_provisional=True,
        warmup_frames=4,source_masks='full native mask uploaded each frame',
        verification='Each timed result hash matches an untimed replay of this same probe; '
            'independent composition checks are in the prerequisite quality report, not this replay.',
        exclusions=['source generation/capture/decode','motion estimation: known transforms supplied',
            'candidate extraction and association','audit hashes',
            'verification downloads outside device-only timing'],
        warning='Nonexact point filtering is diagnostic only. These are core service times, not accepted pipeline FPS.',
        schedule=schedule(),trials=[],replay_hashes={},passed=False)
    try:
        for scene in ('dense','holes'):
            inputs=[generated(SHAPE,i,128.,scene) for i in range(44)]
            expected={}
            probe=ResidentRawProbe(screen.config,SHAPE,tracker.velocity_grid,args.library)
            try:
                for index,(raw,mask,matrix) in enumerate(inputs):
                    probe.push(raw,index,index*100000000,matrix,mask)
                    if index in (19,*ENDS):
                        probe.ring.run();expected[index]=digest_arrays(probe.ring.download())
            finally:probe.close()
            record['replay_hashes'][scene]=expected
            for trial in [r for r in schedule() if r['scene']==scene]:
                tick=time.perf_counter()
                probe=ResidentRawProbe(screen.config,SHAPE,tracker.velocity_grid,args.library)
                samples=[]
                try:
                    for index in range(20):
                        raw,mask,matrix=inputs[index]
                        probe.push(raw,index,index*100000000,matrix,mask)
                    probe.ring.run()
                    if digest_arrays(probe.ring.download())!=expected[19]:
                        raise AssertionError('Warmup replay changed')
                    initialization=time.perf_counter()-tick
                    for end in ENDS:
                        tick=time.perf_counter();frame_rows=[]
                        for index in range(end-7,end+1):
                            raw,mask,matrix=inputs[index];start=time.perf_counter()
                            front_ms=probe.push(raw,index,index*100000000,matrix,mask)
                            frame_rows.append(dict(index=index,host_s=time.perf_counter()-start,
                                frontend_compute_ms=front_ms))
                        append_s=time.perf_counter()-tick
                        start=time.perf_counter();kernel_ms=probe.ring.run()
                        tracking_s=time.perf_counter()-start
                        download_s=0
                        if trial['mode']=='resident_host':
                            start=time.perf_counter();arrays=probe.ring.download()
                            download_s=time.perf_counter()-start
                        elapsed=time.perf_counter()-tick
                        if trial['mode']=='resident_device':arrays=probe.ring.download()
                        exact=digest_arrays(arrays)==expected[end]
                        samples.append(dict(last_frame=end,advance_frames=8,host_s=elapsed,
                            append_host_s=append_s,tracking_host_s=tracking_s,kernel_ms=kernel_ms,
                            download_host_s=download_s,frames=frame_rows,outputs_exact=exact))
                        if not exact:raise AssertionError('Timed replay changed')
                finally:probe.close()
                record['trials'].append(dict(**trial,initialization_and_warmup_s=initialization,samples=samples))
                print(json.dumps(dict(**trial,host_s=[s['host_s'] for s in samples],
                    append_s=[s['append_host_s'] for s in samples],
                    kernel_ms=[s['kernel_ms'] for s in samples])),flush=True)
            del inputs
        record['passed']=len(record['trials'])==16 and all(s['outputs_exact'] for t in record['trials'] for s in t['samples'])
    finally:screen.close();write_json(args.output,record)
    return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','gate','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
