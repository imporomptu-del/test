"""Frozen synthetic source-preservation test; never runs or changes a detector."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

import accuracy_v54_adapter as adapter
import accuracy_v54_benchmark as benchmark

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/tiny_target/accuracy_v54_20260926'
METHODS = ('median8','median3','offset_left_right_fold0','offset_left_right_fold1',
           'offset_checkerboard_fold0','offset_checkerboard_fold1')
CORRECTED = METHODS[2:]
SIGN_TOLERANCE_DN = 1e-10
OBSERVED_NAMES = ('guard_xy','core_xy','guard_history','guard_current',
                 'core_history_on','core_history_off','core_current_on','core_current_off')
TRUTH_NAMES = ('clean_current_background','current_source','source_template','guard_contamination_current')
SOURCE_NAMES = (
    'docs/accuracy_v54_plan.md','scripts/accuracy_v54_benchmark.py','scripts/accuracy_v54_adapter.py',
    'scripts/run_accuracy_v54_sources.py','scripts/audit_accuracy_v54_sources.py',
    'tests/unit/test_accuracy_v54_benchmark.py','tests/unit/test_accuracy_v54_adapter.py',
    'tests/unit/test_accuracy_v54_runner.py','tests/unit/test_accuracy_v54_audit.py',
    'scripts/accuracy_v53_offset.py','tests/unit/test_accuracy_v53_offset.py',
)
INHERITED_SHA256 = {
    'scripts/accuracy_v53_offset.py':'aa490ecf8305dd0c5f3facff83a5fa8fef43c67460eb55572f516b298a079d04',
    'tests/unit/test_accuracy_v53_offset.py':'d5c53052fca8f47bf1f380582702e62217a353587b906fe759d564b460e70f05',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def content_sha(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or not path.is_file() or path.is_symlink():
        raise ValueError('Canonical regular file required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bindings(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Frozen binding changed: '+path)


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(plain(value),stream,indent=2,allow_nan=False);stream.write('\n')


def write_lines(path,rows):
    with Path(path).open('x') as stream:
        for row in rows:stream.write(json.dumps(plain(row),separators=(',',':'),allow_nan=False)+'\n')


def read_lines(path):
    with Path(path).open() as stream:
        for line in stream:yield json.loads(line)


def evaluation_constants():
    return dict(sign_tolerance_dn=SIGN_TOLERANCE_DN,sign_tolerance_is_detection_threshold=False,
        amplitude_requires_entire_positive_template=True,absent_retention_is_null=True,
        paired_on_off_noise_shared=True,matched_controls=('median8','median3'),
        source_on_off_common_scoring_support=True,methods=METHODS)


def prepare_input(spec):
    case=benchmark.generate_case(spec)
    return plain(dict(case_id=spec['case_id'],spec=spec,
        observed={k:case[k] for k in OBSERVED_NAMES},truth={k:case[k] for k in TRUTH_NAMES}))


def predict_input(row):
    values=row['observed']
    before=content_sha(row)
    result=adapter.predict(**{k:np.asarray(values[k],dtype=float) for k in
        ('guard_xy','core_xy','guard_history','guard_current','core_history_on','core_history_off')})
    if content_sha(row)!=before:raise ValueError('Prediction mutated source input')
    if result['prediction_sha256']!=adapter.prediction_fingerprint(result):
        raise ValueError('Prediction fingerprint differs')
    return dict(case_id=row['case_id'],forecast=plain(result))


def point_metrics(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Finite error vector required')
    if not len(a):return dict(count=0,mae_dn=None,rmse_dn=None,max_abs_dn=None)
    with np.errstate(over='ignore',invalid='ignore'):
        absolute=np.abs(a);mae=float(absolute.mean());rmse=float(np.sqrt(np.mean(a*a)))
    if not np.isfinite([mae,rmse]).all():raise ValueError('Nonfinite error arithmetic')
    return dict(count=len(a),mae_dn=mae,rmse_dn=rmse,max_abs_dn=float(absolute.max()))


def sign_category(value,amplitude):
    signed=np.sign(amplitude)*value
    if signed < -SIGN_TOLERANCE_DN:return 'reversed'
    if abs(signed)<=SIGN_TOLERANCE_DN:return 'zero'
    return 'same'


def metric_bundle(row,on,off,mask):
    observed=row['observed'];truth=row['truth']
    c_on=np.asarray(observed['core_current_on'],dtype=float)
    c_off=np.asarray(observed['core_current_off'],dtype=float)
    background=np.asarray(truth['clean_current_background'],dtype=float)
    source=np.asarray(truth['current_source'],dtype=float)
    template=np.asarray(truth['source_template'],dtype=float)
    if not (on.shape==off.shape==mask.shape==c_on.shape==c_off.shape==background.shape==source.shape==template.shape):
        raise ValueError('Metric shapes differ')
    if template.ndim!=1 or not np.isfinite(template).all() or np.any(template<0) or not np.any(template>0):
        raise ValueError('Positive finite source template required')
    amplitude=float(row['spec']['amplitude']);present=amplitude!=0
    if not np.isfinite(amplitude) or not np.isfinite(background).all() or not np.isfinite(source).all():
        raise ValueError('Finite analytic truth required')
    if not np.allclose(source,amplitude*template,rtol=0,atol=1e-12):
        raise ValueError('Current source differs from declared template')
    on_error=c_on[mask]-on[mask];off_error=c_off[mask]-off[mask]
    errors=dict(on_background_error=point_metrics(on[mask]-background[mask]),
        off_background_error=point_metrics(off[mask]-background[mask]),
        on_residual=point_metrics(on_error),off_residual=point_metrics(off_error),
        paired_increment_error=point_metrics(on_error-off_error-source[mask]))
    footprint=template>0;available=bool(mask[footprint].all())
    oracle_available=bool(np.isfinite(c_on[footprint]).all() and np.isfinite(background[footprint]).all())
    amplitudes=dict(source_present=present,template_support_count=int(footprint.sum()),available=available,
        oracle_available=oracle_available,
        ideal_source_dn=amplitude,raw_on_dn=None,raw_off_dn=None,paired_increment_dn=None,oracle_on_dn=None,
        raw_retention=None,paired_retention=None,oracle_retention=None,
        raw_amplitude_error_dn=None,paired_amplitude_error_dn=None,raw_sign=None,paired_sign=None)
    weights=template[footprint];denominator=float(weights@weights)
    if oracle_available:
        oracle=float(weights@(c_on[footprint]-background[footprint])/denominator)
        if not np.isfinite(oracle):raise ValueError('Nonfinite oracle amplitude arithmetic')
        amplitudes.update(oracle_on_dn=oracle,oracle_retention=oracle/amplitude if present else None)
    if available:
        on_res=c_on[footprint]-on[footprint];off_res=c_off[footprint]-off[footprint]
        raw_on=float(weights@on_res/denominator);raw_off=float(weights@off_res/denominator)
        paired=float(weights@(on_res-off_res)/denominator)
        if not np.isfinite([raw_on,raw_off,paired]).all():raise ValueError('Nonfinite amplitude arithmetic')
        amplitudes.update(raw_on_dn=raw_on,raw_off_dn=raw_off,paired_increment_dn=paired)
        if present:
            amplitudes.update(raw_retention=raw_on/amplitude,paired_retention=paired/amplitude,
                raw_amplitude_error_dn=raw_on-amplitude,
                paired_amplitude_error_dn=paired-amplitude,raw_sign=sign_category(raw_on,amplitude),
                paired_sign=sign_category(paired,amplitude))
    return dict(point_errors=errors,amplitudes=amplitudes)


def evaluate(row,forecast):
    if adapter.prediction_fingerprint(forecast)!=forecast['prediction_sha256']:
        raise ValueError('Prediction fingerprint differs')
    if set(forecast['predictions'])!=set(METHODS):raise ValueError('Method membership differs')
    before=content_sha(forecast)
    observed=row['observed'];c_on=np.asarray(observed['core_current_on'],dtype=float)
    c_off=np.asarray(observed['core_current_off'],dtype=float)
    n=forecast['total_core_count']
    if c_on.shape!=(n,) or c_off.shape!=(n,):raise ValueError('Current shape differs')
    result=dict(case_id=row['case_id'],spec=row['spec'],total_core_points=n,methods={},matched={})
    values={};masks={}
    for name in METHODS:
        pred=forecast['predictions'][name];on=np.asarray(pred['on']['values'],dtype=float);off=np.asarray(pred['off']['values'],dtype=float)
        on_available=np.asarray(pred['on']['available'],dtype=bool);off_available=np.asarray(pred['off']['available'],dtype=bool)
        if any(v.shape!=(n,) for v in (on,off,on_available,off_available)):
            raise ValueError('Prediction shape differs')
        if not np.isfinite(on[on_available]).all() or not np.isfinite(off[off_available]).all():
            raise ValueError('Available prediction is nonfinite')
        mask=on_available&off_available&np.isfinite(c_on)&np.isfinite(c_off)
        values[name]=(on,off);masks[name]=mask
        result['methods'][name]=dict(prediction_available_on=int(on_available.sum()),
            prediction_available_off=int(off_available.sum()),current_available_on=int(np.isfinite(c_on).sum()),
            current_available_off=int(np.isfinite(c_off).sum()),scored_indices=np.flatnonzero(mask).tolist(),
            complete=bool(n and mask.all()),metrics=metric_bundle(row,on,off,mask))
    for name in CORRECTED:
        mask=masks[name]&masks['median8']&masks['median3']
        result['matched'][name]=dict(scored_indices=np.flatnonzero(mask).tolist(),complete=bool(n and mask.all()),
            methods={key:metric_bundle(row,*values[actual],mask) for key,actual in
                (('corrected',name),('median8','median8'),('median3','median3'))})
    if content_sha(forecast)!=before:raise ValueError('Scoring mutated predictions')
    return result


def distribution(values):
    a=np.asarray(list(values),dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Finite distribution required')
    if not len(a):return dict(count=0,mean=None,median=None,p10=None,p90=None,min=None,max=None)
    with np.errstate(over='ignore',invalid='ignore'):
        result=dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
            p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite distribution arithmetic')
    return result


def aggregate_metrics(bundles,completes):
    errors=('on_background_error','off_background_error','on_residual','off_residual','paired_increment_error')
    numeric=('raw_on_dn','raw_off_dn','paired_increment_dn','oracle_on_dn','raw_retention','paired_retention',
        'oracle_retention','raw_amplitude_error_dn','paired_amplitude_error_dn')
    amps=[b['amplitudes'] for b in bundles]
    return dict(case_count=len(bundles),complete_cases=sum(completes),
        amplitude_available_cases=sum(a['available'] for a in amps),
        oracle_available_cases=sum(a['oracle_available'] for a in amps),
        present_cases=sum(a['source_present'] for a in amps),
        present_amplitude_available_cases=sum(a['source_present'] and a['available'] for a in amps),
        present_amplitude_unknown_cases=sum(a['source_present'] and not a['available'] for a in amps),
        present_oracle_available_cases=sum(a['source_present'] and a['oracle_available'] for a in amps),
        point_errors={key:dict(scored_points=sum(b['point_errors'][key]['count'] for b in bundles),
            case_mae=distribution(b['point_errors'][key]['mae_dn'] for b in bundles if b['point_errors'][key]['count']),
            case_rmse=distribution(b['point_errors'][key]['rmse_dn'] for b in bundles if b['point_errors'][key]['count']),
            case_max_abs=distribution(b['point_errors'][key]['max_abs_dn'] for b in bundles if b['point_errors'][key]['count'])) for key in errors},
        amplitudes={key:distribution(a[key] for a in amps if a[key] is not None) for key in numeric},
        raw_sign_counts=dict(Counter(a['raw_sign'] for a in amps if a['raw_sign'] is not None)),
        paired_sign_counts=dict(Counter(a['paired_sign'] for a in amps if a['paired_sign'] is not None)))


def aggregate(rows):
    return dict(case_count=len(rows),methods={name:aggregate_metrics([r['methods'][name]['metrics'] for r in rows],
        [r['methods'][name]['complete'] for r in rows]) for name in METHODS},
        matched={name:{control:aggregate_metrics([r['matched'][name]['methods'][control] for r in rows],
            [r['matched'][name]['complete'] for r in rows]) for control in ('corrected','median8','median3')} for name in CORRECTED})


def summarize(rows,specs):
    if [r['case_id'] for r in rows]!=[s['case_id'] for s in specs] or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('Exact score case membership differs')
    if any(r['spec']!=s for r,s in zip(rows,specs)):raise ValueError('Score specs differ')
    result=dict(completed=True,created_at_utc=now(),case_count=len(rows),strata={},groups={},
        no_detector_or_object_accuracy_claim=True,no_production_change=True,paired_source_counterfactual=True)
    for stratum in ('factorial','availability'):
        result['strata'][stratum]=aggregate([r for r in rows if r['spec']['stratum']==stratum])
    factorial=[r for r in rows if r['spec']['stratum']=='factorial']
    for field in ('condition','motion','amplitude','background','noise_level'):
        grouped=defaultdict(list)
        for row in factorial:grouped[str(row['spec'][field])].append(row)
        result['groups'][field]={key:aggregate(value) for key,value in sorted(grouped.items())}
    grouped=defaultdict(list)
    for row in factorial:
        s=row['spec'];grouped[f"{s['condition']}/{s['motion']}/{s['amplitude']}"] .append(row)
    result['groups']['condition_source']={key:aggregate(value) for key,value in sorted(grouped.items())}
    result['groups']['missingness']={r['spec']['missingness']:aggregate([r]) for r in rows if r['spec']['stratum']=='availability'}
    return result


def run(output):
    output=Path(output).absolute()
    if output.parent!=OUTPUT or output.resolve()!=output or output.exists():raise ValueError('Fresh immediate V54 child required')
    check_bindings({str(ROOT/k):v for k,v in INHERITED_SHA256.items()})
    bindings={str(ROOT/name):sha(ROOT/name) for name in SOURCE_NAMES}
    specs=benchmark.specifications()
    if len(specs)!=420 or Counter(s['stratum'] for s in specs)!=dict(factorial=416,availability=4):
        raise ValueError('Frozen case denominators differ')
    if len({s['case_id'] for s in specs})!=420:raise ValueError('Duplicate specs')
    output.mkdir(parents=True)
    write_json(output/'freeze.json',dict(created_at_utc=now(),source_files_sha256=bindings,
        specifications=specs,adapter_constants=adapter.model_constants(),evaluation_constants=evaluation_constants(),
        before_actual_fitting=True,before_truth_scoring=True,runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    freeze_hash=sha(output/'freeze.json')
    write_lines(output/'inputs.jsonl',(prepare_input(spec) for spec in specs))
    def forecasts():
        for index,row in enumerate(read_lines(output/'inputs.jsonl'),1):
            yield predict_input(row)
            if index%50==0 or index==len(specs):print(f'V54: predicted {index}/{len(specs)}; truth scoring not started',flush=True)
    write_lines(output/'predictions.jsonl',forecasts())
    prediction_bindings={str(output/n):sha(output/n) for n in ('inputs.jsonl','predictions.jsonl')}
    write_json(output/'predictions_frozen.json',dict(created_at_utc=now(),completed=True,case_count=420,
        truth_scoring_started=False,current_guard_training_used=True,files_sha256=prediction_bindings))
    print('V54: all predictions frozen; scoring source preservation',flush=True)
    rows=[]
    for row,forecast in zip(read_lines(output/'inputs.jsonl'),read_lines(output/'predictions.jsonl'),strict=True):
        if row['case_id']!=forecast['case_id']:raise ValueError('Prediction case membership differs')
        rows.append(evaluate(row,forecast['forecast']))
    write_lines(output/'scores.jsonl',rows)
    write_json(output/'summary.json',summarize(rows,specs))
    check_bindings(bindings);check_bindings(prediction_bindings)
    if sha(output/'freeze.json')!=freeze_hash:raise ValueError('Freeze changed')
    write_json(output/'completion_receipt.json',dict(completed=True,created_at_utc=now(),
        source_decisions_changed=False,production_changed=False,real_data_accessed=False,
        files_sha256={**bindings,**{str(p):sha(p) for p in sorted(output.iterdir()) if p.is_file()}}))
    print('V54: stress test complete; production unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
