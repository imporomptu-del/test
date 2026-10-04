"""Generated geometry and complete tracker replay gates; no media access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType
import numpy as np
from profile_visible_v17 import sha,write
from tracking_geometry_v20 import GeometryV20,reference,REFERENCE_SHA


def primitive_cases():
    rng=np.random.default_rng(202018)
    for dimension in (2,4):
        for n in (0,1,2,3,15,64,257,1024):
            for pattern in range(7):
                a=rng.normal(10,150,(n,4));mean=rng.normal(0,30,4)
                if pattern==1:a[:]=0;mean[:]=0
                if pattern==2:a[:]=np.resize([3.,4.,0.,5.],a.shape);mean[:]=0
                if pattern==3:a[:]=np.nextafter(a,0)
                if pattern==4:a[::2]*=-1;a[a==0]=-0.0;mean[::2]=-0.0
                if pattern==5:a[:]=rng.normal(0,1,(n,4))*1e120;mean[:]=0
                if pattern==6:a[:]=rng.normal(0,1,(n,4))*1e-120;mean[:]=0
                yield f'd{dimension}_n{n}_p{pattern}',a,mean,dimension,5.,5.
        for offset in (-np.inf,0,np.inf):
            a=np.array([[3.,4.,3.,4.],[5.,0.,5.,0.]])
            limit=5. if offset==0 else np.nextafter(5.,offset)
            yield f'gate_d{dimension}_{offset}',a,np.zeros(4),dimension,limit,limit
        for value in (np.nan,np.inf,-np.inf,1e200,1e-200,np.nextafter(0.,1.)):
            a=np.ones((4,4));a[1,0]=value
            yield f'special_d{dimension}_{value}',a,np.zeros(4),dimension,5.,5.
        a=rng.normal(size=(12,4))
        yield f'strided_d{dimension}',a[::2],np.zeros(4),dimension,5.,5.
        yield f'float32_d{dimension}',a.astype(np.float32),np.zeros(4,dtype=np.float32),dimension,5.,5.
        yield f'mean_strided_d{dimension}',a,np.zeros(8)[::2],dimension,5.,5.


def same(left,right):
    for a,b in zip(left[:5],right[:5]):
        if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise AssertionError('Geometry bytes changed')
    if left[5:]!=right[5:]:raise AssertionError('Gate counts changed')


def replay(helper):
    from tiny_target.detection import CandidateBatch,CandidateRecord
    from tiny_target.tracking.kalman import KalmanTrackManager,KalmanTrackingConfig
    adapted=helper.adapter(KalmanTrackManager.update)
    rows=[]
    for scenario in range(12):
        rng=np.random.default_rng(2030+scenario)
        cfg=KalmanTrackingConfig(position_measurement_sigma_px=1.,velocity_measurement_sigma_px_s=1.,
            acceleration_process_sigma_px_s2=1.,initial_position_sigma_px=2.,initial_velocity_sigma_px_s=2.,
            mahalanobis_gate_squared=16.,maximum_position_residual_px=6.,maximum_velocity_residual_px_s=3.,
            confirmation_independent_hits=2,max_missed_windows=3,maximum_timestamp_gap_s=2.,
            measurement_noise_source='synthetic_characterization',max_active_tracks=24,
            measurement_model='position_only' if scenario%2==0 else 'position_velocity',
            association_cost='gaussian_nll' if scenario%3==0 else 'mahalanobis',
            birth_policy='spatial_fair' if scenario%3==1 else 'input_order',birth_cell_size_px=16.,
            association_assignment='global_min_cost' if scenario>=10 else 'greedy')
        a,b=KalmanTrackManager(cfg),KalmanTrackManager(cfg)
        b.update=MethodType(adapted,b);digest=hashlib.sha256()
        for frame in range(32):
            points=[]
            if frame%11!=8:
                for i in range(32):
                    if (i+frame)%7==0:continue
                    x,y=rng.normal(0,.2,2)+np.array([(i%8)*4+frame*.15,(i//8)*5])
                    if scenario%4==1:x,y=10.,10. # Exact ties and capacity competition.
                    score=9.+i%3
                    points.append(CandidateRecord(candidate_index=len(points),x_px=float(x),y_px=float(y),
                        velocity_index=0,velocity_xy_px_s=(1.5,0.),normalized_score_snr=score,raw_sum_score=2*score,
                        supporting_frame_count=1,support_weight=1.,peak_neighbor_max_score_snr=None,
                        peak_contrast_snr=None,peak_to_neighbor_ratio=None,distance_to_border_px=10,
                        distance_to_invalid_chebyshev_px=None,distance_to_invalid_is_lower_bound=True))
            batch=CandidateBatch(candidates=tuple(points),frame_indices=(frame,),
                reference_timestamp_ns=frame*100000000+(3000000000 if frame>=25 else 0),
                segment_index=int(frame>=17),metrics={},timings_ms={})
            outputs=[m.update(batch,include_quality_evidence=scenario%2==0).to_dict() for m in (a,b)]
            for output in outputs:output.pop('timings_ms')
            encoded=[json.dumps(o,sort_keys=True,allow_nan=False) for o in outputs]
            if encoded[0]!=encoded[1]:raise AssertionError(f'Tracker replay changed scenario {scenario}, frame {frame}')
            digest.update(encoded[0].encode())
        rows.append(dict(scenario=scenario,frames=32,exact=True,output_sha256=digest.hexdigest()))
    return rows


def run(library,output):
    helper=GeometryV20(library);here=Path(__file__).resolve().parent
    record=dict(passed=False,error=None,real_media_read=False,library_sha256=sha(library),
        reference_sha256=REFERENCE_SHA,source_sha256={n:sha(here/n) for n in
        ('tracking_geometry_v20.cpp','tracking_geometry_v20.py','build_tracking_geometry_v20.py','check_tracking_geometry_v20.py')},
        cases=[],replays=[],timings=[])
    try:
        for name,a,m,d,p,v in primitive_cases():
            before=(a.tobytes(),m.tobytes());native=helper.calls
            with np.errstate(all='ignore'):expected=reference(a,m,d,p,v);actual=helper(a,m,d,p,v)
            same(expected,actual)
            if before!=(a.tobytes(),m.tobytes()):raise AssertionError('Inputs changed')
            record['cases'].append(dict(name=name,exact=True,native=helper.calls>native,inputs_unchanged=True))
        record['replays']=replay(helper)
        rng=np.random.default_rng(2021);a=rng.normal(size=(256,4));m=np.zeros(4)
        for repeat in range(4):
            for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference')):
                fn=reference if mode=='reference' else helper
                start=time.perf_counter()
                for _ in range(1000):result=fn(a,m,2,5.,5.)
                elapsed=time.perf_counter()-start
                same(reference(a,m,2,5.,5.),result)
                record['timings'].append(dict(repeat=repeat,mode=mode,iterations=1000,mean_us=elapsed*1000))
        record.update(passed=True,native_calls=helper.calls,fallbacks=helper.fallbacks,
            transformed_sha256=helper.transformed_sha256)
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:write(output,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime',type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.runtime or Path(__file__).resolve().parents[1]))
    run(a.library,a.output)
