"""Fail-closed report-only verification of the generated motion-reuse experiment."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from motion_reuse_v12 import generated_method

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def coverage(rows,keys,expected):
    actual=[tuple(r[k] for k in keys) for r in rows]
    require(len(actual)==len(expected) and set(actual)==set(expected),'Incomplete or duplicate coverage')


def stats(values,n):
    require(len(values)==n and all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in values),'Invalid timing samples')
    return dict(n=n,minimum=min(values),median=statistics.median(values),maximum=max(values),mean=statistics.mean(values))


def summarize(path):
    r=json.loads(path.read_text())
    require(r['passed'] is True and r['quality_passed'] is True,'Incomplete/failed experiment')
    require(r['real_media_read'] is False and r['production_approved'] is False and r['pipeline_benchmark'] is False,'Scope changed')
    sources={'script_sha256':'scripts/check_motion_reuse_v12.py','adapter_sha256':'scripts/motion_reuse_v12.py',
        'config_sha256':'configs/evaluation/raw16_motion_v6.json','motion_source_sha256':'tiny_target/motion/pva_pyrlk.py',
        'plan_sha256':'docs/motion_reuse_v12_plan.md'}
    for key,file in sources.items():require(r[key]==sha(ROOT/file),'Source/configuration changed: '+file)
    require(r['generated_method_sha256']==hashlib.sha256(generated_method().encode()).hexdigest(),'Generated estimator changed')
    names=('dim_stationary','dim_subpixel','dim_translation','dim_larger_shift','high_dynamic_range',
        'bright_translation','gain_and_offset_change','sensor_fixed_pattern','moving_foreground_patch',
        'unobservable_flat','independent_noise','unsupported_rotation')
    coverage(r['independent'],('seed','case'),list(itertools.product((75316,75317,75318,75319),names)))
    require(all(x['exact'] is True and x['expected'] is True for x in r['independent']),'Independent motion failure')
    coverage(r['original_upscaled_negative'],('index',),[(1,),(2,),(3,)])
    require(all(x['exact'] and not x['reference_accepted'] and not x['candidate_accepted'] and x['hits']==0
        for x in r['original_upscaled_negative']),'Original unobservable behavior changed')
    rows=[dict(x,shape=tuple(x['shape'])) for x in r['sequences']]
    coverage(rows,('shape','depth','kind'),list(itertools.product(((960,1280),(3190,4784)),(8,16),
        ('smooth','recovery','invalidation','reordered'))))
    comparisons=0
    for x in rows:
        pairs=([(0,1),(1,2),(0,1),(0,1),(2,3),(3,4)] if x['kind']=='reordered'
            else [(i-1,i) for i in range(1,8 if x['kind']=='recovery' else 6)])
        require([tuple(y['pair']) for y in x['rows']]==pairs,'Sequence schedule changed')
        require(x['inputs_unchanged'] and all(y['exact'] for y in x['rows']),'Sequence output/input changed')
        if x['kind']!='invalidation':require(x['hits']>0,'Reuse not exercised')
        if x['kind']=='smooth':require(all(y['accepted'] for y in x['rows']),'Positive motion rejected')
        if x['kind']=='recovery':require(x['rows'][-1]['accepted'],'Recovery failed')
        comparisons+=len(pairs)
    expected=[dict(depth=d,repeat=i,mode=m) for d in (8,16) for i in range(4)
        for m in (('reference','candidate') if i%2==0 else ('candidate','reference'))]
    require([{k:t[k] for k in ('depth','repeat','mode')} for t in r['timings']]==expected,'Incomplete timing schedule')
    for t in r['timings']:
        require([s['index'] for s in t['samples']]==[1,2,3,4,5] and all(s['exact'] and s['accepted'] for s in t['samples']),
            'Timed motion output mismatch/rejection')
        if t['mode']=='candidate':require(t['hits']==4,'Timed reuse count changed')
    timings={}
    for depth in (8,16):
        arms={}
        for mode in ('reference','candidate'):
            trials=[t for t in r['timings'] if (t['depth'],t['mode'])==(depth,mode)]
            warm=[s for t in trials for s in t['samples'][1:]]
            arms[mode]=dict(cold_host_s=stats([t['samples'][0]['host_s'] for t in trials],4),
                warm_host_s=stats([s['host_s'] for s in warm],16),
                warm_mean_s_by_repeat=[statistics.mean(s['host_s'] for s in t['samples'][1:]) for t in trials],
                warm_substage_median_ms={k:statistics.median(s['substage_ms'][k] for s in warm)
                    for k in ('motion_image_prepare','gaussian_pyramids','flow_state_cache_reset','intensity_conversion_cpu')})
        gains=[1-b/a for a,b in zip(arms['reference']['warm_mean_s_by_repeat'],arms['candidate']['warm_mean_s_by_repeat'])]
        arms['paired_warm_time_reduction_fractions']=gains
        arms['median_warm_time_reduction_fraction']=1-arms['candidate']['warm_host_s']['median']/arms['reference']['warm_host_s']['median']
        timings[str(depth)]=arms
    return dict(schema='seaqr.motion-reuse-v12-summary.v1',verified=True,report_sha256=sha(path),
        verifier_sha256=sha(__file__),source_sha256={file:r[key] for key,file in sources.items()},
        quality=dict(independent_pairs=48,sequence_pairs=comparisons,retained_unobservable_pairs=3,
            timed_pairs=80,actual_timed_reuse_hits=32),timings=timings,
        gates=dict(generated_equality=True,sequence_recovery=True,
            faster_in_every_warm_pair=all(x>0 for a in timings.values() for x in a['paired_warm_time_reduction_fractions']),
            real_video_regression=False,full_pipeline_benchmarked=False,airborne_accuracy_validated=False,
            production_approved=False,defaults_changed=False),
        warning='Generated estimator-stage timing, not detector accuracy or complete-pipeline FPS. '
            'Both arms use the RAW motion configuration; U8 tests exercise its documented U8 fallback.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.report)
    with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['gates'],indent=2))
