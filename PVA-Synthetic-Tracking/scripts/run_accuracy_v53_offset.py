"""Frozen shadow current-guard experiment; never changes source decisions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

import accuracy_v52_benchmark as benchmark
import accuracy_v53_offset as model
import accuracy_v52_real_scope as real

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/tiny_target/accuracy_v53_20260926'
SPLITS = ('left_right', 'checkerboard')
LOSSES = ('median_offset',)


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, dict):
        return {k:plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def sha(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve()!=path or not path.is_file():
        raise ValueError('Canonical regular file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_bindings(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:
            raise ValueError('Frozen input changed: '+path)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value),stream,indent=2,allow_nan=False)
        stream.write('\n')


def write_lines(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(plain(row),allow_nan=False,separators=(',',':'))+'\n')


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


SOURCE_NAMES = (
    'docs/accuracy_v53_plan.md', 'scripts/accuracy_v53_offset.py',
    'scripts/run_accuracy_v53_offset.py', 'scripts/audit_accuracy_v53_offset.py',
    'tests/unit/test_accuracy_v53_offset.py', 'tests/unit/test_accuracy_v53_runner.py',
    'tests/unit/test_accuracy_v53_audit.py', 'scripts/accuracy_v52_benchmark.py',
    'scripts/accuracy_v52_real_scope.py', 'scripts/audit_accuracy_v52_crossfit.py',
    'tests/unit/test_accuracy_v52_benchmark.py', 'tests/unit/test_accuracy_v52_real_scope.py',
    'tests/unit/test_accuracy_v52_audit.py',
)
INHERITED_SHA256 = {
    'scripts/accuracy_v52_benchmark.py': '702e04065dd2e67facdaf8cc616e8fe2e7e199d69a1748aea936f98a7177fe4b',
    'scripts/accuracy_v52_real_scope.py': '8433633a62ab76dc48f6e65f72d95406009b3d133469ed5169e8f7594feaeef0',
    'scripts/audit_accuracy_v52_crossfit.py': 'a17fc7e3f52603146c4dbcb976e88247dfb082efa08f076723fe65c53653476f',
    'tests/unit/test_accuracy_v52_benchmark.py': '5ab538619835a82d4fe7bb155c9dc0f5e3f2c08b75befc9d8dc12c632bf88138',
    'tests/unit/test_accuracy_v52_real_scope.py': '46933c316eeba758c26b8122e2d41833c3349ab8a7549502293bc8a87176c368',
    'tests/unit/test_accuracy_v52_audit.py': '96ce1eee10d7d740c896edc777c738e416293b14b760a116f98915b4f63ad078',
}


def source_paths():
    return sorted(ROOT/name for name in SOURCE_NAMES)


def metrics(errors):
    a = np.asarray(errors,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError('Finite one-dimensional errors required')
    with np.errstate(over='ignore',invalid='ignore'):
        absolute = np.abs(a); mass=float(absolute.sum()); squared=float(np.square(a).sum())
    if not np.isfinite([mass,squared]).all():
        raise ValueError('Nonfinite scoring arithmetic')
    return dict(count=len(a),absolute_sum_dn=mass,squared_sum_dn2=squared,
        mae_dn=float(absolute.mean()) if len(a) else None,
        median_absolute_error_dn=float(np.median(absolute)) if len(a) else None,
        p90_absolute_error_dn=float(np.quantile(absolute,.9)) if len(a) else None,
        max_absolute_error_dn=float(absolute.max()) if len(a) else None)


def distribution(values):
    a = np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError('Finite metric distribution required')
    with np.errstate(over='ignore',invalid='ignore'):
        result=dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p90=float(np.quantile(a,.9)) if len(a) else None,
        max=float(a.max()) if len(a) else None)
    if any(v is not None and not np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite aggregate distribution')
    return result


def evaluate(input_row, forecast):
    if forecast['crossfit_sha256']!=model.crossfit_fingerprint(forecast):
        raise ValueError('Forecast fingerprint differs')
    before = content_sha(forecast)
    current=np.asarray(input_row['current'],dtype=float)
    slow=np.asarray(input_row['median8'],dtype=float); fast=np.asarray(input_row['median3'],dtype=float)
    n=len(current)
    if current.shape!=slow.shape or current.shape!=fast.shape or current.ndim!=1 or forecast['total_count']!=n:
        raise ValueError('Score shape mismatch')
    truth=np.asarray(input_row['clean_current_background'],dtype=float) if input_row['kind']=='synthetic' else None
    if truth is not None and truth.shape!=current.shape:
        raise ValueError('Synthetic truth shape mismatch')
    result=dict(input_id=input_row['input_id'],kind=input_row['kind'],split=forecast['split'],
        metadata=input_row['metadata'],total_points=n,current_available_count=int(np.isfinite(current).sum()),
        baseline_all={},arms={})
    for name,pred in (('median8',slow),('median3',fast)):
        keep=np.isfinite(current)&np.isfinite(pred)
        result['baseline_all'][name]=dict(metrics=metrics(current[keep]-pred[keep]),
            complete=bool(n and keep.all()),scored_indices=np.flatnonzero(keep).tolist())
    for loss in LOSSES:
        pred=forecast['predictions'][loss]
        values=np.asarray(pred['values'],dtype=float); available=np.asarray(pred['available'],dtype=bool)
        if values.shape!=current.shape or available.shape!=current.shape or not np.isfinite(values[available]).all():
            raise ValueError('Malformed forecast values/availability')
        keep=available&np.isfinite(current)&np.isfinite(slow)&np.isfinite(fast)
        values_by_arm={'corrected':values,'median8':slow,'median3':fast}
        entry=dict(prediction_available_count=int(available.sum()),scored_count=int(keep.sum()),
            complete=bool(n and keep.all()),scored_indices=np.flatnonzero(keep).tolist(),
            prediction_unavailable_reasons=dict(Counter(v for v in pred['unavailable_reasons'] if v is not None)),
            metrics={k:metrics(current[keep]-v[keep]) for k,v in values_by_arm.items()},
            residuals={k:(current[keep]-v[keep]).tolist() for k,v in values_by_arm.items()},
            fold_fits={k:dict(available=f['available'],reason=f['unavailable_reason'],
                training_count=f['training_count'],training_used_count=f['training_used_count'],
                offset_dn=f['offset_dn'],median_interval_dn=f['median_interval_dn'],
                objective_mae_dn=f['objective_mae_dn'],subgradient_interval=f['subgradient_interval'])
                for k,f in forecast['fits'][loss].items()})
        if truth is not None:
            clean=available&np.isfinite(truth)&np.isfinite(slow)&np.isfinite(fast)
            entry['clean_truth_scored_indices']=np.flatnonzero(clean).tolist()
            entry['clean_truth_metrics']={k:metrics(truth[clean]-v[clean]) for k,v in values_by_arm.items()}
        result['arms'][loss]=entry
    if content_sha(forecast)!=before:
        raise ValueError('Scoring mutated forecast')
    return result


def combine_metrics(records):
    count=sum(r['count'] for r in records)
    absolute=sum(r['absolute_sum_dn'] for r in records)
    squared=sum(r['squared_sum_dn2'] for r in records)
    if not np.isfinite([absolute,squared]).all():
        raise ValueError('Nonfinite aggregate scoring arithmetic')
    nonempty=[r for r in records if r['count']]
    return dict(point_count=count,conditional_point_mae_dn=absolute/count if count else None,
        conditional_point_rmse_dn=float(np.sqrt(squared/count)) if count else None,
        maximum_absolute_error_dn=max(r['max_absolute_error_dn'] for r in nonempty) if nonempty else None,
        packet_mae=distribution([r['mae_dn'] for r in nonempty]),
        packet_p90_absolute_error=distribution([r['p90_absolute_error_dn'] for r in nonempty]),
        packet_maximum_absolute_error=distribution([r['max_absolute_error_dn'] for r in nonempty]))


def aggregate(rows, states=None):
    result=dict(packet_records=len(rows),point_opportunities=sum(r['total_points'] for r in rows),
        current_available_points=sum(r['current_available_count'] for r in rows),baseline_all={},arms={})
    for base in ('median8','median3'):
        result['baseline_all'][base]=dict(complete_packets=sum(r['baseline_all'][base]['complete'] for r in rows),
            metrics=combine_metrics([r['baseline_all'][base]['metrics'] for r in rows]))
    for loss in LOSSES:
        records=[r['arms'][loss] for r in rows]; complete=[r for r in records if r['complete']]
        entry=dict(prediction_available_points=sum(r['prediction_available_count'] for r in records),
            scored_points=sum(r['scored_count'] for r in records),complete_packets=len(complete),
            incomplete_packets=len(records)-len(complete),
            fold_unavailable_reasons=dict(Counter(f['reason'] for r in records for f in r['fold_fits'].values() if not f['available'])),
            matched_metrics={a:combine_metrics([r['metrics'][a] for r in records]) for a in ('corrected','median8','median3')},
            shared_complete_packet_metrics={a:combine_metrics([r['metrics'][a] for r in complete]) for a in ('corrected','median8','median3')})
        if rows and rows[0]['kind']=='synthetic':
            entry['clean_truth_metrics']={a:combine_metrics([r['clean_truth_metrics'][a] for r in records]) for a in ('corrected','median8','median3')}
        result['arms'][loss]=entry
    if states is not None:
        state_map={tuple(s['state_key']):s for s in states}
        expected={k for k,s in state_map.items() if s['v50_status'] in ('background_measured','response_unavailable')}
        keys=[tuple(r['metadata']['state_key']) for r in rows]
        if len(state_map)!=len(states) or len(set(keys))!=len(keys) or set(keys)!=expected:
            raise ValueError('Exact archive state membership differs')
        if any(r['metadata']['partition']!=state_map[tuple(r['metadata']['state_key'])]['partition'] for r in rows):
            raise ValueError('Archive partition differs from state ledger')
        frames=defaultdict(list)
        for state in states:
            key=state['state_key']; frames[(key[0],key[2],key[1])].append(state)
        records_by_frame=defaultdict(list)
        for row in rows:
            key=row['metadata']['state_key']; records_by_frame[(key[0],key[2],key[1])].append(row)
        if not set(records_by_frame)<=set(frames):
            raise ValueError('Archive frame absent from state ledger')
        result.update(states=len(states),original_status_counts=dict(Counter(s['v50_status'] for s in states)),
            selected_response_frames=len(frames),frames_with_scored_archives=len(records_by_frame),
            frames_without_scored_archives=len(frames)-len(records_by_frame))
        for base in ('median8','median3'):
            complete=[rs for rs in records_by_frame.values() if all(r['baseline_all'][base]['complete'] for r in rs)]
            result['baseline_all'][base]['complete_frame_count']=len(complete)
            result['baseline_all'][base]['complete_frame_mean_packet_mae']=distribution([
                float(np.mean([r['baseline_all'][base]['metrics']['mae_dn'] for r in rs])) for rs in complete])
        for loss in LOSSES:
            complete=[rs for rs in records_by_frame.values() if all(r['arms'][loss]['complete'] for r in rs)]
            entry=result['arms'][loss]; entry['complete_frame_count']=len(complete)
            entry['incomplete_archived_frame_count']=len(records_by_frame)-len(complete)
            entry['complete_frame_mean_packet_mae']={a:distribution([
                float(np.mean([r['arms'][loss]['metrics'][a]['mae_dn'] for r in rs])) for rs in complete]) for a in ('corrected','median8','median3')}
            entry['complete_frame_maximum_absolute_error']={a:distribution([
                max(r['arms'][loss]['metrics'][a]['max_absolute_error_dn'] for r in rs) for rs in complete]) for a in ('corrected','median8','median3')}
    return result


def summarize(rows, states):
    result=dict(completed=True,created_at_utc=now(),splits={},no_source_decisions=True,
        current_guard_estimation_not_prior_forecasting=True,no_physical_or_object_accuracy_claim=True)
    for split in SPLITS:
        selected=[r for r in rows if r['split']==split]
        synth=[r for r in selected if r['kind']=='synthetic']; actual=[r for r in selected if r['kind']=='real']
        value=dict(synthetic=aggregate(synth),real=aggregate(actual,states),synthetic_groups={},real_groups={},nine_frame_bins={})
        for field in ('family','background','noise_level'):
            groups=defaultdict(list)
            for row in synth: groups[str(row['metadata'][field])].append(row)
            value['synthetic_groups'][field]={k:aggregate(v) for k,v in sorted(groups.items())}
        for partition in ('calibration','embargo','evaluation'):
            for clip in sorted({s['state_key'][0] for s in states}):
                subset=[s for s in states if s['partition']==partition and s['state_key'][0]==clip]
                packets=[r for r in actual if r['metadata']['partition']==partition and r['metadata']['state_key'][0]==clip]
                value['real_groups'][partition+'_'+clip]=aggregate(packets,subset)
                bins=sorted({s['state_key'][1]//9 for s in subset})
                for index in bins:
                    value['nine_frame_bins'][f'{partition}_{clip}_{index}']=aggregate(
                        [r for r in packets if r['metadata']['state_key'][1]//9==index],
                        [s for s in subset if s['state_key'][1]//9==index])
        result['splits'][split]=value
    return result


def prepare_inputs(assembled):
    rows=[]
    for spec in benchmark.specifications():
        c=benchmark.generate_case(spec)
        rows.append(dict(input_id=spec['case_id'],kind='synthetic',metadata=spec,
            points_xy=c['points_xy'],median8=c['median8'],median3=c['median3'],current=c['current'],
            clean_current_background=c['clean_current_background'],contamination_mask=c['contamination_mask']))
    for p in assembled['packets']:
        rows.append(dict(input_id='real_'+'_'.join(map(str,p['state_key'])),kind='real',
            metadata={k:p[k] for k in ('state_key','partition','available','reasons')},
            points_xy=p['points_xy'],median8=p['median8'],median3=p['median3'],current=p['current']))
    if len({r['input_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate input id')
    return plain(rows)


def prepare_forecasts(inputs, output):
    records=[]
    for i,row in enumerate(inputs,1):
        for split in SPLITS:
            f=model.crossfit(np.asarray(row['points_xy']),np.asarray(row['median8'],dtype=float),
                np.asarray(row['current'],dtype=float),split=split)
            if model.crossfit_fingerprint(f)!=f['crossfit_sha256']:
                raise ValueError('Forecast changed before serialization')
            records.append(dict(input_id=row['input_id'],forecast=plain(f)))
        if i%100==0 or i==len(inputs):
            print(f'V53: fitted {i}/{len(inputs)} inputs; no evaluation scoring yet',flush=True)
    write_lines(output/'forecasts.jsonl',records)
    write_json(output/'forecasts_frozen.json',dict(created_at_utc=now(),completed=True,
        input_count=len(inputs),forecast_count=len(records),evaluation_scoring_started=False,
        current_training_pixels_used=True,files_sha256={str(output/n):sha(output/n) for n in ('inputs.jsonl','forecasts.jsonl')}))


def validate_forecast_membership(inputs, records):
    ids=[r['input_id'] for r in inputs]
    expected={(name,split) for name in ids for split in SPLITS}
    actual=[(r['input_id'],r['forecast']['split']) for r in records]
    if len(set(ids))!=len(ids) or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Exact forecast membership differs')


def run(output):
    output=Path(output).absolute()
    if output.parent!=OUTPUT or output.resolve()!=output or output.exists():
        raise ValueError('Fresh immediate V53 output child required')
    check_bindings({str(ROOT/name):digest for name,digest in INHERITED_SHA256.items()})
    bindings={str(p):sha(p) for p in source_paths()}
    documents,input_bindings=real.read_authorized()
    assembled=real.assemble(documents)
    specs=benchmark.specifications()
    if len(specs)!=120 or len(assembled['packets'])!=493 or len(assembled['states'])!=1211:
        raise ValueError('Frozen experiment denominators differ')
    output.mkdir(parents=True)
    write_json(output/'freeze.json',dict(created_at_utc=now(),source_files_sha256=bindings,
        input_files_sha256=input_bindings,specifications=specs,model_constants=model.model_constants(),
        splits=SPLITS,before_actual_fitting=True,before_evaluation_scoring=True,
        real_counts=assembled['counts'],runtime={'python':platform.python_version(),'numpy':np.__version__}))
    freeze_hash=sha(output/'freeze.json')
    inputs=prepare_inputs(assembled)
    write_lines(output/'inputs.jsonl',inputs)
    write_lines(output/'state_results.jsonl',assembled['states'])
    write_json(output/'reference_context.json',assembled['references'])
    prepare_forecasts(inputs,output)
    print('V53: all cross-fitted predictions frozen; evaluating held-out errors',flush=True)
    manifest=json.loads((output/'forecasts_frozen.json').read_text()); check_bindings(manifest['files_sha256'])
    restored={r['input_id']:r for r in read_lines(output/'inputs.jsonl')}
    records=read_lines(output/'forecasts.jsonl')
    validate_forecast_membership(list(restored.values()),records)
    rows=[evaluate(restored[r['input_id']],r['forecast']) for r in records]
    write_lines(output/'scores.jsonl',rows)
    write_json(output/'summary.json',summarize(rows,assembled['states']))
    check_bindings(bindings);check_bindings(input_bindings);check_bindings(manifest['files_sha256'])
    if sha(output/'freeze.json')!=freeze_hash:
        raise ValueError('Freeze changed')
    write_json(output/'completion_receipt.json',dict(completed=True,created_at_utc=now(),
        files_sha256={**bindings,**input_bindings,**{str(p):sha(p) for p in sorted(output.iterdir()) if p.is_file()}},
        source_decisions_changed=False,production_changed=False,media_accessed=False))
    print('V53: experiment complete; detector and references unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
