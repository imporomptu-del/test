"""Chronologically split prior-value forecasts; diagnostic only, never a detector."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np

import run_accuracy_v46_shadow as cache
from accuracy_v50_predictive_background import ARMS, forecast, measure_current, forecast_fingerprint
from accuracy_v50_scope import split_scope, partition_for
from accuracy_v50_calibration import empirical_quantile, frame_score, select_calibration_frames


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/tiny_target/accuracy_v48_20260926/shadow_01'
OUTPUT=ROOT/'results/tiny_target/accuracy_v50_20260926'
RECEIPT_SHA='b21753dfdfc6fb253f800c6a6fc3ad1ee393200058e488af6e423ffd3128a251'
POLICIES=('disjoint_anchor_frames','all_calibration_frames')


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def group_key(row):
    return row['clip'],row['segment'],row['frame_index']


def load_prior_arrays(path,row):
    """Verify complete archive bytes, but never decode its current129 member."""
    payload=cache._regular_exact(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=row['archive']['sha256']:
        raise ValueError('Frozen packet changed before prior decode')
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        if set(archive.files)!=set(cache.ARRAY_SHAPES):
            raise ValueError('Unexpected packet members')
        prior={k:archive[k] for k in ('history129','prior_centers_xy','predicted_offset_xy')}
    for name,a in prior.items():
        if a.shape!=cache.ARRAY_SHAPES[name] or a.dtype!=np.float64:
            raise ValueError('Invalid prior array')
        a.setflags(write=False)
    centers_array=prior['prior_centers_xy']
    center_rows_valid=np.isfinite(centers_array).all(axis=1)|np.isnan(centers_array).all(axis=1)
    if not center_rows_valid.all() or not np.isfinite(prior['predicted_offset_xy']).all():
        raise ValueError('Invalid prior geometry coordinates')
    g=row['geometry']['geometry']
    centers=np.asarray([[np.nan,np.nan] if c is None else c for c in g['prior_centers_xy']])
    if (not np.array_equal(centers,prior['prior_centers_xy'],equal_nan=True)
            or not np.array_equal(g['predicted_offset_xy'],prior['predicted_offset_xy'])):
        raise ValueError('Prior geometry changed')
    return prior


def load_current_array(path,row):
    """Decode only current member after forecast freeze; no whole-image value scan."""
    payload=cache._regular_exact(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=row['archive']['sha256']:
        raise ValueError('Frozen packet changed before current decode')
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        if set(archive.files)!=set(cache.ARRAY_SHAPES):
            raise ValueError('Unexpected packet members')
        current=archive['current129']
    if current.shape!=cache.ARRAY_SHAPES['current129'] or current.dtype!=np.float64:
        raise ValueError('Invalid current array schema')
    current.setflags(write=False)
    return current


def write_jsonl(path,rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(jsonable(row),allow_nan=False,separators=(',',':'))+'\n')


def build_forecasts(rows,output,partition):
    predictions={}
    for row in rows:
        prior=load_prior_arrays(cache.packet_path(row),row)
        centers=[c.tolist() if np.isfinite(c).all() else None for c in prior['prior_centers_xy']]
        value=forecast(prior['history129'],centers)
        predictions[cache.state_key(row)]=value
    path=output/(partition+'_forecasts.jsonl')
    write_jsonl(path,[dict(state_key=list(k),forecast=v) for k,v in predictions.items()])
    cache.write_json(output/(partition+'_forecasts_frozen.json'),dict(
        completed=True,created_at_utc=cache.now(),forecasts_sha256=cache.sha(path),
        packet_count=len(rows),current129_members_decoded=False,
        prior_values_only_conditional_on_original_current_registered_geometry=True))
    return predictions,cache.sha(path)


def score_forecasts(rows,predictions,output,partition,expected_hash):
    path=output/(partition+'_forecasts.jsonl')
    if cache.sha(path)!=expected_hash:raise ValueError('Forecast file changed before response access')
    saved=[json.loads(line) for line in path.read_text().splitlines()]
    saved_by_key={tuple(r['state_key']):r['forecast'] for r in saved}
    if (len(saved_by_key)!=len(saved) or set(saved_by_key)!=set(predictions)
            or set(saved_by_key)!={cache.state_key(r) for r in rows}):
        raise ValueError('Saved forecast membership differs')
    records=[]
    for row in rows:
        key=cache.state_key(row);value=predictions[key]
        if jsonable(value)!=saved_by_key[key]:
            raise ValueError('In-memory forecast differs from the frozen file')
        before=forecast_fingerprint(value)
        current=load_current_array(cache.packet_path(row),row)
        result=measure_current(current,value)
        if forecast_fingerprint(value)!=before:raise ValueError('Response mutated prior forecast')
        records.append(dict(state_key=list(key),clip=row['clip'],segment=row['segment'],
            frame_index=row['frame_index'],forecast_available=value['available'],
            measurement=jsonable(result)))
    if cache.sha(path)!=expected_hash:raise ValueError('Forecast file changed during response scoring')
    write_jsonl(output/(partition+'_measurements.jsonl'),records)
    return records


def frame_units(records):
    grouped=defaultdict(list)
    for row in records:grouped[group_key(row)].append(row)
    units=[]
    for (clip,segment,frame),rows in sorted(grouped.items()):
        units.append(dict(clip=clip,segment=segment,frame_index=frame,
            state_keys=[r['state_key'] for r in rows],
            scores={arm:frame_score([r['measurement']['arms'][arm]['max_score']
                if r['measurement']['available'] else None for r in rows]) for arm in ARMS},
            unavailable_packet_count=sum(not r['measurement']['available'] for r in rows)))
    return units


def calibrate(records):
    units=frame_units(records)
    grouped=defaultdict(list)
    for unit in units:grouped[(unit['clip'],unit['segment'])].append(unit)
    policies={}
    for policy in POLICIES:
        entries=[]
        for (clip,segment),frames in sorted(grouped.items()):
            indices=select_calibration_frames([f['frame_index'] for f in frames],policy)
            chosen=[f for f in frames if f['frame_index'] in indices]
            entries.append(dict(clip=clip,segment=segment,selected_frame_indices=indices,
                frames=chosen,arms={arm:empirical_quantile([f['scores'][arm] for f in chosen]) for arm in ARMS}))
        policies[policy]=entries
    return dict(completed=True,created_at_utc=cache.now(),policies=policies,
        empirical_target=.9,no_iid_or_exchangeability_coverage_guarantee=True,
        unavailable_units_not_replaced=True,no_automatic_policy_fallback=True,
        held_later_current_arrays_not_decoded=True)


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return dict(count=0,mean=None,median=None,p90=None,p95=None,max=None)
    if not np.isfinite(a).all():raise ValueError('Nonfinite metric')
    return dict(count=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
        p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def interval_records(records,predictions,calibration):
    output=[]
    for row in records:
        key=tuple(row['state_key']);pred=predictions[key];m=row['measurement']
        item=dict(state_key=row['state_key'],clip=row['clip'],segment=row['segment'],
                  frame_index=row['frame_index'],policies={})
        for policy in POLICIES:
            c=next(e for e in calibration['policies'][policy]
                   if (e['clip'],e['segment'])==(row['clip'],row['segment']))
            arms={}
            for arm in ARMS:
                q=c['arms'][arm]
                reason=('forecast_unavailable' if not pred['available'] else
                        'current_support_unavailable' if not m['available'] else
                        'calibration_unavailable' if not q['available'] else None)
                if reason:
                    arms[arm]=dict(available=False,reason=reason)
                    continue
                # Inclusive comparison on recorded normalized errors; empirical
                # floating summaries, never directed-rounding physical bounds.
                multiplier=q['q'];errs=np.asarray(m['arms'][arm]['normalized_absolute_errors'])
                center=np.asarray(pred['arms'][arm]['prediction'])
                half=multiplier*np.asarray(pred['arms'][arm]['scale'])
                covered=errs<=multiplier
                arms[arm]=dict(available=True,point_count=len(errs),
                    covered_point_count=int(covered.sum()),whole_packet_covered=bool(covered.all()),
                    half_width_dn=distribution(half),half_width_values=half.tolist(),
                    full_8bit_range_included_points=int(((center-half<=0)&(center+half>=255)).sum()),
                    q=multiplier,
                    status='within_empirical_band' if covered.all() else 'outside_empirical_band',
                    not_an_object_or_model_validity_decision=True)
            item['policies'][policy]=arms
        output.append(item)
    return output


def summarize_intervals(items,policy,arm):
    valid=[r['policies'][policy][arm] for r in items if r['policies'][policy][arm]['available']]
    reasons=Counter(r['policies'][policy][arm].get('reason') for r in items
                    if not r['policies'][policy][arm]['available'])
    frames=defaultdict(list)
    for r in items:frames[group_key(r)].append(r['policies'][policy][arm])
    available_frames=[v for v in frames.values() if all(x['available'] for x in v)]
    points=sum(v['point_count'] for v in valid);covered=sum(v['covered_point_count'] for v in valid)
    packet_hits=sum(v['whole_packet_covered'] for v in valid)
    frame_hits=sum(all(v['whole_packet_covered'] for v in f) for f in available_frames)
    return dict(archived_packets=len(items),interval_available_packets=len(valid),
        unavailable_reasons=dict(reasons),covered_packets=packet_hits,
        conditional_packet_coverage=packet_hits/len(valid) if valid else None,
        guard_state_point_pairs=points,covered_guard_state_point_pairs=covered,
        conditional_point_coverage=covered/points if points else None,
        archived_frames=len(frames),interval_available_whole_frames=len(available_frames),
        covered_whole_frames=frame_hits,
        conditional_whole_frame_coverage=frame_hits/len(available_frames) if available_frames else None,
        half_width_dn=distribution([x for v in valid for x in v['half_width_values']]),
        full_8bit_range_included_point_pairs=sum(v['full_8bit_range_included_points'] for v in valid),
        frames_and_pixels_not_independent=True)


def metrics(records,intervals):
    result={}
    groups={'combined':records}
    groups.update({clip:[r for r in records if r['clip']==clip] for clip in sorted({r['clip'] for r in records})})
    for label,rows in groups.items():
        keys={tuple(r['state_key']) for r in rows}
        items=[r for r in intervals if tuple(r['state_key']) in keys]
        good=[r for r in rows if r['measurement']['available']]
        frames=defaultdict(list)
        for r in rows:frames[group_key(r)].append(r)
        whole=[v for v in frames.values() if all(r['measurement']['available'] for r in v)]
        arms={}
        for arm in ARMS:
            arms[arm]=dict(packet_mae_dn=distribution([r['measurement']['arms'][arm]['mae_dn'] for r in good]),
                packet_rmse_dn=distribution([r['measurement']['arms'][arm]['rmse_dn'] for r in good]),
                packet_maximum_absolute_error_dn=distribution([
                    max(abs(x) for x in r['measurement']['arms'][arm]['residuals']) for r in good]),
                packet_maximum_normalized_error=distribution([
                    r['measurement']['arms'][arm]['max_score'] for r in good]),
                complete_frame_macro_mae_dn=distribution([
                    np.mean([r['measurement']['arms'][arm]['mae_dn'] for r in f]) for f in whole]),
                policies={p:summarize_intervals(items,p,arm) for p in POLICIES})
        paired=[r['measurement']['arms']['median3_temporal_scale']['mae_dn']-
                r['measurement']['arms']['median8_temporal_scale']['mae_dn'] for r in good]
        result[label]=dict(archived_packets=len(rows),scorable_packets=len(good),
            packet_unavailability_reasons=dict(Counter(reason for r in rows if not r['measurement']['available']
                for reason in r['measurement']['reasons'])),
            archived_frames=len(frames),complete_scorable_frames=len(whole),arms=arms,
            paired_candidate_minus_median8_mae_dn=distribution(paired),
            paired_candidate_better=sum(x<0 for x in paired),paired_equal=sum(x==0 for x in paired),
            paired_candidate_worse=sum(x>0 for x in paired))
    return result


def nine_frame_bins(state_results,intervals):
    """Keep every selected evaluation state, including bins with no archive."""
    blocks=defaultdict(list);measured=defaultdict(list)
    for r in state_results:
        if r['partition']=='evaluation':
            clip,frame,segment,_=r['state_key']
            blocks[(clip,segment,frame//9)].append(r)
    for r in intervals:
        measured[(r['clip'],r['segment'],r['frame_index']//9)].append(r)
    if not set(measured)<=set(blocks):raise ValueError('Unselected interval bin')
    report=[]
    for k,states in sorted(blocks.items()):
        items=measured[k]
        frames={r['state_key'][1] for r in states}
        archived_frames={r['frame_index'] for r in items}
        report.append(dict(clip=k[0],segment=k[1],nine_frame_bin=k[2],
            selected_states=len(states),selected_response_frames=len(frames),
            state_status_counts=dict(Counter(r['v50_status'] for r in states)),
            response_frames_without_archives=sorted(frames-archived_frames),
            frames_with_history_unknown_states=sorted({r['state_key'][1] for r in states
                if r['v50_status']=='history_unknown'}),
            policies={p:{a:summarize_intervals(items,p,a) for a in ARMS} for p in POLICIES}))
    return dict(bins=report,adjacent_bins_can_share_prior_frames=True,not_independent_trials=True,
        history_unknown_states_are_not_covered=True)


def reference_context(original,scope,state_results):
    result=deepcopy(original);lookup={tuple(r['state_key']):r for r in state_results}
    groups={(g['clip'],g['segment']) for g in scope['cutoffs']}
    for s in result['samples']:
        clip=s['original']['clip_id'];frame=s['original']['frame_index']
        segments=[segment for c,segment in groups if c==clip]
        if len(segments)!=1:raise ValueError('Reference segment ambiguous')
        partition=partition_for(clip,segments[0],frame,scope['cutoffs'])
        assigned=s['original_strict_assigned_identity']
        key=None if assigned is None else (clip,frame,int(assigned.split('/')[0]),assigned.split('/',1)[1])
        s['v50_background_context']=dict(partition=partition,
            original_assigned_state_key=None if key is None else list(key),
            original_assigned_state_status=None if key is None else lookup[key]['v50_status'],
            original_unassigned_stays_unassigned=assigned is None,
            not_detection_recall_or_airborne_identity=True)
    stripped=deepcopy(result)
    for s in stripped['samples']:del s['v50_background_context']
    if stripped!=original:raise ValueError('Original reference evidence changed')
    return result


def source_dependencies():
    files={Path(m.__file__).resolve() for m in list(sys.modules.values())
           if getattr(m,'__file__',None) and Path(m.__file__).resolve().parent==ROOT/'scripts'}
    files.add(Path(__file__).resolve())
    files.update((ROOT/'scripts').glob('*accuracy_v50*.py'))
    files.update((ROOT/'tests/unit').glob('test_accuracy_v50_*.py'))
    files.add(ROOT/'docs/accuracy_v50_plan.md')
    return sorted(files)


def run(output):
    output=Path(output).absolute()
    if output.parent!=OUTPUT or output.exists() or output.resolve()!=output:
        raise ValueError('Fresh immediate V50 output child required')
    receipt_path=cache._regular_exact(BASE/'completion_receipt.json')
    if cache.sha(receipt_path)!=RECEIPT_SHA:raise ValueError('V48 receipt changed')
    old=cache.read_json(receipt_path)
    inputs={str(receipt_path):RECEIPT_SHA}
    for name in ('states.jsonl','selected_ledger.json','reference_evidence.json'):
        p=cache._regular_exact(BASE/name);inputs[str(p)]=old['files_sha256'][str(p)]
    cache.require_hashes(inputs)
    rows=[json.loads(x) for x in (BASE/'states.jsonl').read_text().splitlines()]
    ledger=cache.read_json(BASE/'selected_ledger.json')
    if ledger['states']!=[{k:v for k,v in r.items() if k!='arms'} for r in rows]:
        raise ValueError('Original ledger differs')
    if len(rows)!=1211 or sum(r['archive'] is not None for r in rows)!=509:
        raise ValueError('Original denominator changed')
    refs=cache.read_json(BASE/'reference_evidence.json')
    if len(refs['samples'])!=355:raise ValueError('Reference denominator changed')
    scope=split_scope(rows)
    part={tuple(x['state_key']):x['partition'] for x in scope['assignments']}
    subsets={p:[r for r in rows if part[cache.state_key(r)]==p and r['archive'] is not None]
             for p in ('calibration','evaluation')}
    if (len(subsets['calibration']),len(subsets['evaluation']))!=(190,303):
        raise ValueError('Predeclared partition counts differ')
    packets={}
    for subset in subsets.values():
        for r in subset:
            p=cache.packet_path(r);digest=r['archive']['sha256']
            if old['files_sha256'].get(str(p))!=digest:raise ValueError('Packet not bound by V48')
            packets[str(p)]=digest
    for path in source_dependencies():
        digest=cache.sha(path)
        if str(path) in old['files_sha256'] and digest!=old['files_sha256'][str(path)]:
            raise ValueError('Inherited source modified')
        inputs[str(path)]=digest
    output.mkdir(parents=True)
    cache.write_json(output/'freeze.json',dict(created_at_utc=cache.now(),scope=scope,
        input_files_sha256=inputs,packet_files_sha256=packets,completed_before_packet_access=True,
        predictors=list(ARMS),policies=list(POLICIES),production_changed=False,
        runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    fixed={str(output/'freeze.json'):cache.sha(output/'freeze.json')}
    print('V50: frozen scope; calibration prior-only forecasts',flush=True)
    cp,ch=build_forecasts(subsets['calibration'],output,'calibration')
    cm=score_forecasts(subsets['calibration'],cp,output,'calibration',ch)
    calibration=calibrate(cm)
    cache.write_json(output/'calibration.json',calibration)
    fixed[str(output/'calibration.json')]=cache.sha(output/'calibration.json')
    fixed[str(output/'calibration_forecasts.jsonl')]=ch
    print('V50: calibration frozen; later prior-only forecasts',flush=True)
    ep,eh=build_forecasts(subsets['evaluation'],output,'evaluation')
    if calibration!=cache.read_json(output/'calibration.json'):
        raise ValueError('In-memory calibration differs from frozen calibration')
    em=score_forecasts(subsets['evaluation'],ep,output,'evaluation',eh)
    fixed[str(output/'evaluation_forecasts.jsonl')]=eh
    intervals=interval_records(em,ep,calibration)
    write_jsonl(output/'evaluation_intervals.jsonl',intervals)
    measurements={tuple(r['state_key']):r for r in cm+em}
    state_results=[]
    for r in rows:
        key=cache.state_key(r);partition=part[key]
        m=measurements.get(key)
        status=('history_unknown' if r['archive'] is None else 'embargo_not_scored' if partition=='embargo'
                else 'forecast_unavailable' if not m['forecast_available']
                else 'response_unavailable' if not m['measurement']['available'] else 'background_measured')
        state_results.append(dict(state_key=list(key),partition=partition,v50_status=status,
            original_qualified_moving=r['qualified_moving'],original_reference_samples=r['reference_samples'],
            source_scores_and_original_detections_unchanged=True))
    write_jsonl(output/'state_results.jsonl',state_results)
    reference=reference_context(refs,scope,state_results)
    cache.write_json(output/'reference_context.json',reference)
    cache.write_json(output/'nine_frame_bins.json',nine_frame_bins(state_results,intervals))
    anchor_keys={tuple(k) for unit in scope['anchors']['evaluation'] for k in unit['state_keys']}
    anchors=[r for r in intervals if tuple(r['state_key']) in anchor_keys]
    summary=dict(completed=True,created_at_utc=cache.now(),original_states=1211,original_archives=509,
        opened_packets=len(packets),embargo_archives_unread=16,
        scope_counts=scope['counts'],
        partitions={p:dict(Counter(r['v50_status'] for r in state_results if r['partition']==p))
                    for p in ('calibration','embargo','evaluation')},
        evaluation=metrics(em,intervals),
        evaluation_disjoint_anchor_intervals={p:{a:summarize_intervals(anchors,p,a) for a in ARMS} for p in POLICIES},
        reference_partition_counts=dict(Counter(s['v50_background_context']['partition'] for s in reference['samples'])),
        original_reference_samples=355,original_reference_alternatives=sum(len(s['measured_alternatives']) for s in reference['samples']),
        original_unassigned_references=[dict(sample_index=s['sample_index'],clip=s['original']['clip_id'],frame=s['original']['frame_index'])
            for s in reference['samples'] if s['original_strict_assigned_identity'] is None],
        production_changed=False,source_solver_or_detector_not_called=True,
        no_untouched_test_airborne_accuracy_false_alarm_or_generalization_claim=True,
        current_registered_geometry_is_not_a_fully_causal_camera_pipeline=True,
        forecasts_and_intervals_are_empirical_not_certified_physical_bounds=True,
        no_automatic_arm_selection_or_policy_fallback=True)
    cache.write_json(output/'summary.json',summary)
    cache.require_hashes(inputs);cache.require_hashes(packets);cache.require_hashes(fixed)
    bindings={str(p):cache.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    bindings.update(inputs);bindings.update(packets)
    cache.write_json(output/'completion_receipt.json',dict(completed=True,created_at_utc=cache.now(),
        files_sha256=bindings,production_changed=False,no_new_video_raw16_holdout_ssh_journal_access=True))
    print('V50: complete; no production change',flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
