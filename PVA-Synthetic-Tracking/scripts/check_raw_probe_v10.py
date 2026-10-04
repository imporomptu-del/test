"""Generated RAW16 probe: preserve prefilter state, expose FFT differences honestly."""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from resident_frontend_v10 import ResidentRawProbe
from check_resident_v10 import reference_objects,arrays_equal,digest_arrays
from raw16_speed_v8_common import dense,filter_parameters,cpu_filter,numeric_comparison
from tiny_target.raw_background_cuda import RawBackgroundCuda
from tiny_target.detection import integrated_gaussian_kernel
from profile_raw16_efficiency import sha,write_json,compact


def generated(shape,index,flux,scene):
    rng=np.random.default_rng(191016+index)
    raw=rng.integers(19980,20021,shape,dtype=np.uint16)
    t=index*.1
    matrix=np.eye(3)
    matrix[:2,2]=[.21875*np.sin(index*.3),-.21875*np.cos(index*.2)]
    if index==0:matrix[:2,2]=0
    x=shape[1]/2+.25+2*t-matrix[0,2]
    y=shape[0]/2+.25+t-matrix[1,2]
    ix,iy=round(x),round(y)
    psf=integrated_gaussian_kernel(.8,3,x-ix,y-iy)
    raw[iy-3:iy+4,ix-3:ix+4]=np.rint(raw[iy-3:iy+4,ix-3:ix+4].astype(np.float32)+flux*psf).astype(np.uint16)
    mask=np.ones(shape,bool)
    raw.flat[0:4]=[0,1,32769,65535]
    if scene=='holes':
        raw[shape[0]//3:shape[0]//3+9,shape[1]//3:shape[1]//3+11]=65535
        mask[20:29,30:43]=False
        if 12<=index<=20:mask[shape[0]//2-4:shape[0]//2+2,shape[1]//2:shape[1]//2+10]=False
    return raw,mask,matrix


def reference_step(raw,mask,matrix,config,background):
    size=(raw.shape[1],raw.shape[0]);image=raw.astype(np.float32)
    if not np.array_equal(matrix,np.eye(3)):
        image=cv2.warpPerspective(image,matrix,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        mask=cv2.warpPerspective(mask.astype(np.uint8),matrix,size,flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).astype(bool)
    mask=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    valid=mask & (image>config.dark_floor_dn) & (image<float(65535)*config.saturation_fraction)
    return image,valid,background.step(image,valid)


def check_prepoint(reference,debug,product):
    image,valid=reference
    white,mask,_=product
    pairs=((image,debug[0]),(valid.astype(np.uint8),debug[1]),(white,debug[2]),(mask.astype(np.uint8),debug[4]))
    return all(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes() for a,b in pairs)


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    ring_gate=json.loads(args.ring_gate.read_text())
    if not ring_gate['passed'] or ring_gate['library_sha256']!=sha(args.library):
        raise ValueError('Combined library must pass the complete ring quality gate first')
    screen,tracker,extractor=reference_objects();config=screen.config
    kernel,norm=filter_parameters()
    record=dict(schema='seaqr.raw-probe-v10-checks.v1',real_media_read=False,production_approved=False,
        generated_data_only=True,library_sha256=sha(args.library),ring_gate_sha256=sha(args.ring_gate),
        script_sha256=sha(__file__),wrapper_sha256=sha(ROOT/'scripts/resident_frontend_v10.py'),
        plan_sha256=sha(ROOT/'docs/raw16_feasibility_v10_plan.md'),frames=[],windows=[],native=[],
        implementation_integrity_passed=False,strict_cpu_fft_equivalent=False,
        warning='Known nonexact direct GPU filter is diagnostic only. Sensitivity rows are generated regression evidence, not real airborne recall.')
    try:
        for scene in ('dense','holes'):
            for flux in (0.,32.,64.,128.,256.,512.):
                shape=(192,256);probe=ResidentRawProbe(config,shape,tracker.velocity_grid,args.library)
                background=RawBackgroundCuda(config,shape)
                windows={'reference':deque(maxlen=16),'direct':deque(maxlen=16)}
                try:
                    for index in range(44):
                        raw,mask,matrix=generated(shape,index,flux,scene)
                        probe.push(raw,index,index*100000000,matrix,mask)
                        image,valid,product=reference_step(raw,mask,matrix,config,background)
                        if product is None:continue
                        debug=probe.debug()
                        prepoint=check_prepoint((image,valid),debug,product)
                        if not prepoint:raise AssertionError('Resident prefilter pixels/state changed')
                        cpu=cpu_filter(product[0],kernel,norm)
                        comparison=numeric_comparison(cpu,debug[3],product[0])
                        record['frames'].append(dict(scene=scene,flux=flux,index=index,
                            prepoint_exact=prepoint,**comparison))
                        if not comparison['numerical_screen_passed']:raise AssertionError('Direct-filter diagnostic bound failed')
                        if index<4:continue
                        for mode,response in (('reference',cpu),('direct',debug[3])):
                            windows[mode].append(dense._DenseMatchedFrame(response,product[1],index*100000000,index,0,True))
                        if index not in (19,27,35,43):continue
                        outputs={mode:tracker.integrate(list(frames)) for mode,frames in windows.items()}
                        probe.ring.run();arrays=probe.ring.download()
                        composed_exact=arrays_equal(outputs['direct'],arrays)
                        if not composed_exact:raise AssertionError('Device handoff changed tracking output')
                        batches={mode:extractor.extract(w) for mode,w in outputs.items()}
                        t=outputs['reference'].reference_timestamp_ns/1e9
                        truth=(shape[1]/2+.25+2*t,shape[0]/2+.25+t)
                        near={mode:any(np.hypot(c.x_px-truth[0],c.y_px-truth[1])<=3 for c in batch.candidates)
                            for mode,batch in batches.items()}
                        identities={mode:compact([c.to_dict() for c in batch.candidates]) for mode,batch in batches.items()}
                        record['windows'].append(dict(scene=scene,flux=flux,last_frame=index,
                            handoff_exact=composed_exact,near_truth=near,
                            candidate_counts={mode:len(batch.candidates) for mode,batch in batches.items()},
                            ordered_decisions_exact=identities['reference']==identities['direct'],
                            output_sha256=digest_arrays(arrays)))
                    print('raw control',scene,flux,flush=True)
                finally:probe.close();background.close()
        # Full native geometry, genuine uint16 generation, all prefilter pixels.
        for scene in ('dense','holes'):
            shape=(3190,4784);probe=ResidentRawProbe(config,shape,tracker.velocity_grid,args.library)
            background=RawBackgroundCuda(config,shape)
            try:
                for index in range(8):
                    raw,mask,matrix=generated(shape,index,128.,scene)
                    probe.push(raw,index,index*100000000,matrix,mask)
                    image,valid,product=reference_step(raw,mask,matrix,config,background)
                    if product is None:continue
                    debug=probe.debug();exact=check_prepoint((image,valid),debug,product)
                    if not exact:raise AssertionError('Native prefilter identity failed')
                    comparison=numeric_comparison(cpu_filter(product[0],kernel,norm),debug[3],product[0])
                    record['native'].append(dict(scene=scene,index=index,prepoint_exact=exact,**comparison))
                    if not comparison['numerical_screen_passed']:raise AssertionError('Native diagnostic bound failed')
            finally:probe.close();background.close()
        positive=[r for r in record['windows'] if r['scene']=='dense' and r['flux']==512]
        record['positive_control_observed']=any(r['near_truth']['reference'] for r in positive)
        record['generated_target_losses']=sum(r['near_truth']['reference'] and not r['near_truth']['direct'] for r in record['windows'] if r['flux']>0)
        record['generated_target_gains']=sum(r['near_truth']['direct'] and not r['near_truth']['reference'] for r in record['windows'] if r['flux']>0)
        record['strict_cpu_fft_equivalent']=all(r['bit_exact'] for r in record['frames']+record['native'])
        record['generated_sensitivity_gate_passed']=record['positive_control_observed'] and record['generated_target_losses']==0
        record['implementation_integrity_passed']=(len(record['frames'])==516 and len(record['windows'])==48 and len(record['native'])==14)
    finally:
        screen.close();write_json(args.output,record)
    print(json.dumps({k:record[k] for k in ('implementation_integrity_passed','strict_cpu_fft_equivalent','generated_sensitivity_gate_passed','generated_target_losses','generated_target_gains')}))
    return 0 if record['implementation_integrity_passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','ring-gate','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
