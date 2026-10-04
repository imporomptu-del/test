"""Frozen-patch model diagnostics. No detection, tuning or new video decoding."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import run_accuracy_v46_shadow as cache
from accuracy_v43_components import prepare_components
from accuracy_v44_causal_probe import _hash_array
from accuracy_v47_guard_gain import estimate_guard_gain
from accuracy_v49_guard_slack import minimum_guard_response_slack


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/tiny_target/accuracy_v48_20260926/shadow_01'
OUTPUT = ROOT / 'results/tiny_target/accuracy_v49_20260926'
RECEIPT_SHA = 'b21753dfdfc6fb253f800c6a6fc3ad1ee393200058e488af6e423ffd3128a251'
CASES = (('0029',345,0,'bright:2673'), ('0029',352,0,'bright:2641'),
         ('0029',604,0,'bright:4204'), ('0126',144,0,'bright:1001'),
         ('0126',182,0,'bright:1001'))
MAX_GAP_NS = 2_000_000_000
ARM = 'bounded_background_guard_gain'


def timestamp(row):
    return row['geometry']['geometry']['current_timestamp_ns']


def gain_record(row):
    return row['arms'][ARM]['gain_calibration']


def select_scope(rows, cases=CASES):
    """Metadata-only scope; guard availability but never source signs select controls."""
    lookup = {cache.state_key(r):r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate state keys')
    decisions, roles = [], {}
    def add(key, role):
        roles.setdefault(key, []).append(role)
    for case in cases:
        row = lookup[case]
        if not row['archive']:
            raise ValueError('Case has no frozen archive')
        add(case, dict(kind='case', case_key=list(case)))
        for direction, sign in (('before', -1), ('after', 1)):
            candidates = [r for r in rows if r['archive'] is not None
                and r['clip'] == row['clip'] and r['segment'] == row['segment']
                and r['track_id'].split(':')[0] == row['track_id'].split(':')[0]
                and 0 < sign*(timestamp(r)-timestamp(row)) <= MAX_GAP_NS]
            order = lambda r: (abs(timestamp(r)-timestamp(row)), cache.state_key(r))
            available = [r for r in candidates if gain_record(r) is not None
                         and gain_record(r)['available']]
            same = [r for r in available if r['track_id'] == row['track_id']]
            control = min(same or available, key=order) if available else None
            for kind, chosen in (('guard_control', control), ('same_track_neighbor',
                min((r for r in candidates if r['track_id'] == row['track_id']),
                    key=order, default=None))):
                item = dict(case_key=list(case), kind=kind, direction=direction,
                    selected_key=None if chosen is None else list(cache.state_key(chosen)),
                    nominal_timestamp_gap_seconds=None if chosen is None else
                    abs(timestamp(chosen)-timestamp(row))/1e9,
                    same_track=None if chosen is None else chosen['track_id']==row['track_id'])
                decisions.append(item)
                if chosen is not None:
                    add(cache.state_key(chosen), item)
    selected = [dict(state_key=list(k), roles=roles[k], archive=lookup[k]['archive'],
        guard_reached=gain_record(lookup[k]) is not None,
        timestamp_ns=timestamp(lookup[k]),
        current_center_xy=lookup[k]['geometry']['geometry']['current_center_xy'])
        for k in sorted(roles)]
    if len(selected) > 5*len(cases):
        raise ValueError('Selection exceeded bounded scope')
    return dict(cases=[list(k) for k in cases], decisions=decisions, selected=selected,
        source_signs_or_references_used_for_selection=False,
        nominal_timestamps_not_verified_camera_capture_cadence=True)


def contrasts_from_points(response, background, points, stencils):
    """Independent exact regeneration from stored floating samples on fixed support."""
    y, b = np.asarray(response), np.asarray(background)
    if y.shape != b.shape or y.shape != (len(points),) or not np.isfinite([y,b]).all():
        raise ValueError('Finite complete fixed support required')
    coords=np.asarray(points)
    if (coords.shape!=(len(points),2) or not np.isfinite(coords).all()
            or not np.equal(coords,np.floor(coords)).all()
            or len({tuple(p) for p in points})!=len(points)):
        raise ValueError('Unique integer point coordinates required')
    index = {tuple(p):i for i,p in enumerate(points)}
    result = []
    for s in stencils:
        if s['weights'] != [1,-2,1] or len(s['pixels_xy'])!=3:
            raise ValueError('Only unchanged V47 stencils are authorized')
        if any(tuple(p) not in index for p in s['pixels_xy']):
            raise ValueError('Stencil point missing from fixed support')
        ids = [index[tuple(p)] for p in s['pixels_xy']]
        values = [sum((w*Fraction(float(a[i])) for w,i in zip(s['weights'],ids)), Fraction())
                  for a in (y,b)]
        result.append(dict(response=str(values[0]),background=str(values[1]),
                           response_error='2',background_error='2'))
    return result


def unit_gain_violations(constraints):
    excess = [max(Fraction(), abs(Fraction(c['response'])-Fraction(c['background']))
                  -Fraction(c['response_error'])-Fraction(c['background_error']))
              for c in constraints]
    return dict(violating_stencil_indices=[i for i,e in enumerate(excess) if e>0],
        count=sum(e>0 for e in excess), maximum_excess_contrast_dn=float(max(excess,default=Fraction())),
        maximum_excess_exact=str(max(excess,default=Fraction())),
        all_excess_contrast_dn=[float(e) for e in excess])


def describe(values):
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all():
        raise ValueError('Descriptive stats require finite nonempty fixed support')
    return dict(min=float(a.min()), median=float(np.median(a)),
                p90=float(np.quantile(a,.9)), max=float(a.max()))


def fit_guard_plane(current, background, points, gradients=None):
    """Descriptive least squares only; no pixel exclusion or operative score use."""
    y,b=np.asarray(current),np.asarray(background)
    p=np.asarray(points,dtype=float)
    if (y.ndim!=1 or y.shape!=b.shape or len(y)==0 or p.shape!=(len(y),2)
            or not np.isfinite(p).all()):
        raise ValueError('Matching one-dimensional values and finite Nx2 points required')
    extra=np.column_stack((np.ones(len(p)),(p[:,0]-64)/56,(p[:,1]-64)/56))
    if gradients is not None:
        g=np.asarray(gradients)
        if g.shape!=(len(p),2) or not np.isfinite(g).all():
            return dict(available=False,reason='gradient_neighbors_missing_on_fixed_support')
        extra=np.column_stack((extra,g))
    design=np.column_stack((b,extra))
    if not np.isfinite(design).all() or not np.isfinite(y).all():
        raise ValueError('No row dropping allowed')
    beta,_,rank,svals=np.linalg.lstsq(design,y,rcond=None)
    constrained=bool(beta[0]<0)
    if constrained:
        beta=np.r_[0.,np.linalg.lstsq(extra,y,rcond=None)[0]]
    residual=y-design@beta
    return dict(available=True,points=len(p),parameters=design.shape[1],rank=int(rank),
        condition_number=None if not len(svals) or svals[-1]==0 else float(svals[0]/svals[-1]),
        coefficients=beta.tolist(),gain_nonnegative_constraint_active=constrained,
        rmse_dn=float(np.sqrt(np.mean(residual**2))),absolute_residual_dn=describe(abs(residual)),
        residual_dn=residual.tolist(),in_sample_diagnostic_not_validated_registration=True)


def render_case(path, packet, background, guard, summary):
    """Measured-array diagnostic sheet: fixed display scales and nearest enlargement."""
    font=ImageFont.load_default(size=16)
    cellw,cellh=284,294
    canvas=Image.new('RGB',(cellw*4,cellh*3+80),(24,26,30))
    draw=ImageDraw.Draw(canvas)
    title='/'.join(map(str,summary['state_key']))
    draw.text((12,8),title+' | 129x129 samples shown at 2x nearest-neighbor',fill='white',font=font)
    draw.text((12,30),'Gray: 0-255 DN. Residual: fixed +/-8 DN. Magenta: missing.',fill='white',font=font)
    frames=[(f'prior {i-8}',v,False) for i,v in enumerate(packet['history129'])]
    frames.extend([('prior median B',background,False),('current',packet['current129'],False),
                   ('current minus B',packet['current129']-background,True),
                   ('guard: unit-gain violations',packet['current129'],False)])
    for i,(name,arr,residual) in enumerate(frames):
        finite=np.isfinite(arr)
        safe=np.where(finite,arr,0.)
        if residual:
            t=np.clip(safe/8,-1,1)
            rgb=np.stack((255*np.maximum(t,0),255*(1-abs(t)),255*np.maximum(-t,0)),axis=-1)
        else:
            rgb=np.repeat(np.clip(np.rint(safe),0,255)[...,None],3,axis=2)
        rgb[~finite]=[255,0,255]
        im=Image.fromarray(rgb.astype(np.uint8)).resize((258,258),Image.Resampling.NEAREST)
        if i==11:
            overlay=ImageDraw.Draw(im)
            for x,y in guard['used_points_xy']:
                overlay.ellipse((x*2-1,y*2-1,x*2+1,y*2+1),fill=(0,220,220))
            for index in summary['unit_gain']['violating_stencil_indices']:
                coords=[(x*2,y*2) for x,y in guard['stencils'][index]['pixels_xy']]
                overlay.line(coords,fill=(255,70,40),width=2)
        x,y=(i%4)*cellw+12,(i//4)*cellh+80
        draw.text((x,y-23),name,fill='white',font=font)
        canvas.paste(im,(x,y))
    canvas.save(path)


def analyze_packet(row, packet):
    centers=[p.tolist() if np.isfinite(p).all() else None for p in packet['prior_centers_xy']]
    components=prepare_components(packet['history129'],centers,packet['predicted_offset_xy'],
                                  row['track_id'].split(':')[0])
    background=components['background']
    raw=row['arms'][ARM]['raw_adapter_result']
    if _hash_array(background)!=raw['learned_design_sha256']['background']:
        raise ValueError('Prior background differs from V48')
    old=gain_record(row)
    guard=estimate_guard_gain(packet['current129'],packet['history129'],background,
                              np.where(np.isfinite(background),.5,np.nan),centers)
    if guard!=old:
        raise ValueError('Full guard record no longer reproduces V48')
    points=guard['used_points_xy']; p=np.asarray(points); xx,yy=p[:,0],p[:,1]
    current=packet['current129'][yy,xx]; history=packet['history129'][:,yy,xx]
    b=background[yy,xx]
    if not np.array_equal(np.median(history,axis=0),b):
        raise ValueError('Eligible guard must use all eight original prior values')
    constraints=contrasts_from_points(current,b,points,guard['stencils'])
    if constraints!=guard['contrast_constraints']:
        raise ValueError('Independent exact contrast reconstruction differs')
    slack=minimum_guard_response_slack(constraints)
    loo=[]
    for i in range(8):
        other=np.median(np.delete(history,i,axis=0),axis=0)
        c=contrasts_from_points(history[i],other,points,guard['stencils'])
        loo.append(dict(history_index=i,frame_index=row['frame_index']-8+i,
            background_values=other.tolist(),contrast_constraints=c,
            unit_gain=unit_gain_violations(c),slack=minimum_guard_response_slack(c)))
    gradients=np.column_stack(((background[yy,xx+1]-background[yy,xx-1])/2,
                               (background[yy+1,xx]-background[yy-1,xx])/2))
    geometry=row['geometry']['geometry']
    matrices=np.asarray(geometry['current_to_prior_matrices'])
    result=dict(state_key=list(cache.state_key(row)),guard_available=guard['available'],
        original_guard_reasons=guard['reasons'],used_points=len(points),stencils=len(constraints),
        original_background_hash_reproduced=True,complete_guard_record_reproduced=True,
        independent_exact_contrasts_reproduced=True,unit_gain=unit_gain_violations(constraints),
        slack=slack,prior_leave_one_out=loo,
        temporal_guard_range_dn=describe(np.ptp(history,axis=0)),
        temporal_guard_median_absolute_deviation_dn=describe(np.median(abs(history-b),axis=0)),
        absolute_current_minus_B_dn=describe(abs(current-b)),
        outside_prior_range_point_count=int(((current<history.min(axis=0))|(current>history.max(axis=0))).sum()),
        affine_fit=fit_guard_plane(current,b,points),
        gradient_augmented_fit=fit_guard_plane(current,b,points,gradients),
        maximum_saved_relative_translation_px=float(np.abs(matrices[:,:2,2]).max()),
        original_current_center_xy=geometry['current_center_xy'],
        saved_transform_magnitude_not_a_bound_on_registration_error=True,
        support_unchanged=True,no_score_or_detector_updated=True)
    samples=dict(used_points_xy=points,stencils=guard['stencils'],current_values=current.tolist(),
                 background_values=b.tolist(),history_values=history.tolist(),
                 contrast_constraints=constraints,gradient_values=[
                     [float(v) if np.isfinite(v) else None for v in pair] for pair in gradients],
                 original_current_center_xy=geometry['current_center_xy'])
    return result,samples,background,guard


def run(output):
    output=Path(output).absolute()
    if output.parent!=OUTPUT or output.exists() or output.resolve()!=output:
        raise ValueError('Fresh immediate V49 output child required')
    receipt_path=cache._regular_exact(BASE/'completion_receipt.json')
    if cache.sha(receipt_path)!=RECEIPT_SHA:
        raise ValueError('V48 receipt changed')
    receipt=cache.read_json(receipt_path)
    states_path=cache._regular_exact(BASE/'states.jsonl')
    expected=receipt['files_sha256'][str(states_path)]
    if cache.sha(states_path)!=expected:
        raise ValueError('V48 state records changed')
    rows=[json.loads(line) for line in states_path.read_text().splitlines()]
    if len(rows)!=1211 or sum(r['archive'] is not None for r in rows)!=509:
        raise ValueError('Frozen denominator changed')
    scope=select_scope(rows)
    lookup={cache.state_key(r):r for r in rows}
    code={Path(m.__file__).resolve() for m in list(sys.modules.values())
          if getattr(m,'__file__',None) and Path(m.__file__).resolve().parent==ROOT/'scripts'}
    code.add(Path(__file__).resolve())
    code.update((ROOT/'tests/unit').glob('test_accuracy_v49_*.py'))
    code.add(ROOT/'docs/accuracy_v49_plan.md')
    inputs={str(p):cache.sha(p) for p in sorted(code)}
    for name,digest in inputs.items():
        if name in receipt['files_sha256'] and receipt['files_sha256'][name]!=digest:
            raise ValueError('Inherited source changed')
    inputs.update({str(receipt_path):RECEIPT_SHA,str(states_path):expected})
    packets={}
    for item in scope['selected']:
        if not item['guard_reached']:continue
        row=lookup[tuple(item['state_key'])]
        path=cache.packet_path(row)
        digest=row['archive']['sha256']
        if receipt['files_sha256'].get(str(path))!=digest:
            raise ValueError('Packet not bound to V48')
        packets[str(path)]=digest
    output.mkdir(parents=True)
    cache.write_json(output/'freeze.json',dict(created_at_utc=cache.now(),
        completed_before_packet_bytes_read=True,selection=scope,code_and_metadata_sha256=inputs,
        selected_packet_sha256=packets,production_changed=False,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pillow=Image.__version__)))
    freeze_sha=cache.sha(output/'freeze.json')
    records=[]
    for item in scope['selected']:
        row=lookup[tuple(item['state_key'])]
        stem='_'.join(map(str,item['state_key'])).replace(':','_')
        if not item['guard_reached']:
            records.append(dict(state_key=item['state_key'],packet_opened=False,
                skipped='original_guard_not_reached',
                original_earlier_stage_reasons=row['arms'][ARM]['raw_adapter_result']['reasons']))
            continue
        packet=cache.load_packet(cache.packet_path(row),row)
        before=cache.packet_fingerprint(packet)
        result,samples,background,guard=analyze_packet(row,packet)
        result.update(packet_opened=True,roles=item['roles'])
        cache.write_json(output/(stem+'_samples.json'),samples)
        if tuple(item['state_key']) in CASES:
            render_case(output/(stem+'_history.png'),packet,background,guard,result)
            result['history_sheet']=stem+'_history.png'
        if cache.packet_fingerprint(packet)!=before:
            raise ValueError('Packet mutated')
        records.append(result)
        print(stem, 'completed',flush=True)
    summary=dict(completed=True,created_at_utc=cache.now(),selected_states=len(scope['selected']),
        packets_opened=len(packets),records=records,full_v48_denominator=1211,
        production_changed=False,no_new_video_decode_raw16_holdout_ssh_or_journal=True,
        no_source_score_reference_assignment_or_threshold_changed=True,
        not_an_accuracy_or_false_alarm_evaluation=True,
        additional_dn_slack_is_not_a_proposed_threshold_or_calibrated_noise_bound=True,
        prior_leave_one_out_is_retrospective_dependent_and_not_causal_validation=True)
    cache.write_json(output/'summary.json',summary)
    cache.require_hashes(inputs);cache.require_hashes(packets)
    if cache.sha(output/'freeze.json')!=freeze_sha:
        raise ValueError('Freeze changed')
    files={str(p):cache.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    files.update(inputs);files.update(packets)
    cache.write_json(output/'completion_receipt.json',dict(completed=True,
        created_at_utc=cache.now(),files_sha256=files,production_changed=False,
        frozen_v48_results_unchanged=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
