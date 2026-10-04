"""Generated-only exact residency/sensitivity regression checks; no media access."""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from resident_tracking_v10 import ResidentTracker
from raw16_speed_v8_common import dense,filter_parameters,cpu_filter
from tiny_target.detection import integrated_gaussian_kernel
from profile_raw16_efficiency import compact,sha,write_json

REFERENCE_LIBRARY_SHA = 'e29dc8bae949e41497aff82cfa2fe07d52e1c039b5c337187bdb8c2e87b1fc65'


def reference_objects():
    cfg,_ = dense.load_dense_screen_config(ROOT/'configs/evaluation/raw16_background_v7.json')
    screen = dense.DensePointScreener(cfg)
    tracker = screen._synthetic_window.tracker
    if tracker.library_sha256 != REFERENCE_LIBRARY_SHA:
        raise ValueError('Frozen reference CUDA library changed')
    return screen,tracker,screen._synthetic_extractor


def arrays_equal(window,arrays):
    reference = [window.score,window.velocity_index,window.valid_support_count,window.valid_mask.astype(np.uint8)]
    return all(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(reference,arrays))


def digest_arrays(arrays):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]


def trajectory(scene,t):
    if scene == 'turn':
        return 128.25+2*min(t,2)-2*max(0,t-2),96.25+t
    if scene == 'acceleration':
        return 128.25+t+.2*t*t,96.25+t
    return 128.25+2*t,96.25+t


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    screen,reference,extractor = reference_objects()
    record = dict(schema='seaqr.resident-v10-checks.v1',real_media_read=False,
        production_approved=False,library_sha256=sha(args.library),
        reference_library_sha256=REFERENCE_LIBRARY_SHA,checker_sha256=sha(__file__),
        wrapper_sha256=sha(ROOT/'scripts/resident_tracking_v10.py'),
        plan_sha256=sha(ROOT/'docs/raw16_feasibility_v10_plan.md'),rows=[],guards=[],passed=False)
    kernel,norm = filter_parameters()
    device = ResidentTracker((192,256),reference.velocity_grid,args.library)
    try:
        scenes = ('straight','turn','acceleration','short_visibility','holes','clutter')
        for scene in scenes:
            for flux in (0.,2.,4.,8.,16.):
                for polarity in ('bright','dark'):
                    rng = np.random.default_rng(162097)
                    device.reset(segment=3,polarity=polarity)
                    frames = deque(maxlen=16)
                    for index in range(40):
                        # Exposure-independent, matched-response controls. The CPU FFT
                        # defines the unchanged reference input to BOTH trackers.
                        image = rng.normal(0,1,(192,256)).astype(np.float32)
                        mask = np.ones(image.shape,bool)
                        mask[:4]=False; mask[-4:]=False; mask[:,:4]=False; mask[:,-4:]=False
                        if scene == 'holes':
                            mask[80:90,110:150]=False
                            if 18 <= index <= 23: mask[95:103,128:139]=False
                        if scene == 'clutter':
                            image[70:130,70:73] += 8
                            image[90,120] += 12
                        t = index*.1
                        x,y = trajectory(scene,t)
                        active = scene != 'short_visibility' or 12 <= index <= 23
                        if active:
                            ix,iy = round(x),round(y)
                            psf = integrated_gaussian_kernel(.8,3,x-ix,y-iy)
                            image[iy-3:iy+4,ix-3:ix+4] += np.float32(flux*(1 if polarity=='bright' else -1))*psf
                        image[~mask] = 0
                        response = cpu_filter(image,kernel,norm)
                        frame = dense._DenseMatchedFrame(response,mask,index*100000000,index,3,True,polarity)
                        frames.append(frame)
                        device.push(response,mask,index,frame.timestamp_ns,segment=3,polarity=polarity)
                        if index not in (15,23,31,39):
                            continue
                        old = reference.integrate(list(frames))
                        device.run(); arrays = device.download()
                        exact = arrays_equal(old,arrays)
                        new = replace(old,score=arrays[0],velocity_index=arrays[1],
                            valid_support_count=arrays[2],valid_mask=arrays[3].astype(bool))
                        batches = [extractor.extract(w) for w in (old,new)]
                        identities = [compact([c.to_dict() for c in b.candidates]) for b in batches]
                        truth = trajectory(scene,old.reference_timestamp_ns/1e9)
                        near = [[c.candidate_index for c in b.candidates
                            if np.hypot(c.x_px-truth[0],c.y_px-truth[1]) <= 3] for b in batches]
                        row = dict(scene=scene,flux=flux,polarity=polarity,last_frame=index,
                            arrays_exact=exact,ordered_decisions_exact=identities[0]==identities[1],
                            reference_candidate_count=len(identities[0]),near_truth_candidate_ids=near,
                            target_active_frames=sum(scene!='short_visibility' or 12<=f.frame_index<=23 for f in frames),
                            output_sha256=digest_arrays(arrays),candidate_sha256=sha_json(identities[0]))
                        record['rows'].append(row)
                        if not exact or identities[0] != identities[1]:
                            raise AssertionError('Residency changed numerical or ordered decisions')
                    print('checked',scene,flux,polarity,flush=True)
        # Metadata guards must fail BEFORE mutating valid native state.
        a=np.zeros((192,256),np.float32);m=np.ones(a.shape,bool)
        for name,call in (
            ('nonmonotonic',lambda:device.push(a,m,39,3900000000,segment=3,polarity='dark')),
            ('segment',lambda:device.push(a,m,40,4000000000,segment=4,polarity='dark')),
            ('polarity',lambda:device.push(a,m,40,4000000000,segment=3,polarity='bright')),
            ('nonfinite',lambda:device.push(a+np.nan,m,40,4000000000,segment=3,polarity='dark')),
        ):
            before=device.count
            try: call()
            except ValueError: record['guards'].append(dict(name=name,passed=device.count==before))
            else: raise AssertionError('Guard did not reject '+name)
        device.reset(segment=4)
        try: device.run()
        except ValueError: record['guards'].append(dict(name='premature_window',passed=True))
        else: raise AssertionError('Premature window allowed')
        try: device.download()
        except RuntimeError: record['guards'].append(dict(name='stale_download',passed=True))
        else: raise AssertionError('Stale output allowed')
        # Short/odd geometry, full mask and irregular timestamps exercise borders
        # and the ring's chronological alignment without relying on visible targets.
        record['geometry'] = []
        for shape in ((1,1),(3,7),(17,65),(193,257)):
            small=ResidentTracker(shape,reference.velocity_grid,args.library)
            frames=deque(maxlen=16);rng=np.random.default_rng(31916)
            try:
                for index in range(33):
                    a=rng.normal(0,2,shape).astype(np.float32);m=rng.random(shape)>.2
                    timestamp=index*100000000+(300000000 if index>=18 else 0)
                    frame=dense._DenseMatchedFrame(a,m,timestamp,index,0,True)
                    frames.append(frame);small.push(a,m,index,timestamp)
                    if index in (15,16,23,31,32):
                        old=reference.integrate(list(frames));small.run();out=small.download()
                        exact=arrays_equal(old,out)
                        record['geometry'].append(dict(shape=shape,last_frame=index,exact=exact))
                        if not exact:raise AssertionError('Geometry/gap equality failed')
            finally:small.close()
        strong = [r for r in record['rows'] if r['scene']=='straight' and r['flux']==16]
        record['positive_control_observed']=all(any(r['near_truth_candidate_ids'][0] for r in strong if r['polarity']==p) for p in ('bright','dark'))
        record['passed'] = (len(record['rows'])==240 and len(record['geometry'])==20
            and all(r['passed'] for r in record['guards']) and record['positive_control_observed'])
    finally:
        device.close();screen.close();write_json(args.output,record)
    print(json.dumps({k:record[k] for k in ('passed','positive_control_observed')}))
    return 0 if record['passed'] else 2


def sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
