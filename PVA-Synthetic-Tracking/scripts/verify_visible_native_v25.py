"""Independent report-only v25 native/provenance/parity/latency audit; no media."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap
from run_visible_native_v25 import FROZEN,CORE
from run_visible_stage_v24 import validate_snapshot
from verify_visible_overlap_v23 import (ROOT,V20,V22,read,sha,write,require,positive,
    distribution,local_source,baseline,verify_dependencies,intersection_ns)
from verify_visible_stage_v24 import verify_sources as verify_v24_sources
from native_motion_v25 import OLD,NEW,REFERENCE_SHA,gm
V24=ROOT/'results/tiny_target/visible_stage_v24_20260920/evidence'
WORKLOADS=('0126','0082')
MODES=('reference','native_staged')
CLIPS=('0029','0126','0055','0082')

def audit_underlying(evidence,spec,row,frozen,dependency):
    path=evidence/spec['name'];r=read(path.with_suffix('.v24.json'))
    require(row['returncode']==0 and r['passed'] and r['error'] is None,'Failed trial')
    require(all(r[k]==spec[k] for k in ('clip','mode','frames','traced')),'Scope mismatch')
    require(r['schema']=='seaqr.visible-stage-v24.v1' and not any(r[k] for k in (
        'raw16_accessed','defaults_changed','gpu_changed','algorithm_changed','production_approved')),
        'Algorithm/default or evidence scope changed')
    require(r['source_sha256']==frozen['files'] and r['freeze_sha256']==sha(V24/'freeze.json')
        and r['generated_sha256']==sha(V24/'generated_01.json')
        and r['baseline_freeze_sha256']==dependency['freeze_sha256']
        and row['v24_sha256']==sha(path.with_suffix('.v24.json'))
        and r['baseline_receipt_sha256']==sha(path.with_suffix('.v20.json')),'v24 receipt chain changed')
    v20=read(path.with_suffix('.v20.json'))
    require(v20['passed'] and v20['error'] is None and v20['mode']=='candidate'
        and all(v20[k]==spec[k] for k in ('clip','frames')) and not any(v20[k] for k in (
            'raw16_accessed','defaults_changed','gpu_changed','noise_v18_enabled','median_v19_enabled')),
        'v20 runtime candidate changed')
    require(v20['script_sha256']==sha(V20/'run_visible_v20.py')
        and v20['freeze_sha256']==dependency['freeze_sha256']
        and v20['gate_sha256']==dependency['generated_sha256']
        and v20['library_sha256']==dependency['library_sha256']
        and v20['transformed_sha256']==dependency['transformed_sha256']
        and v20['baseline_receipt_sha256']==sha(path.with_suffix('.v17.json'))
        and v20['native_geometry_calls']>0 and v20['geometry_fallbacks']>=0,'v20 helper identity changed')
    v17,report,motion=baseline(path,spec['clip'],spec['frames'])
    count=report['frames']
    require(count==r['processed_frames']==v20['processed_frames'],'Frame count changed')
    for k in ('fps','wall_s'):
        positive(r[k],k);require(r[k]==row[k]==v20[k]==v17[k],'Timing receipt differs')
    require(math.isclose(r['fps'],count/r['wall_s'],rel_tol=1e-12),'FPS denominator changed')
    snap=r['execution'];validate_snapshot(snap,count)
    require(snap['policy']==spec['mode'],'Stage policy changed')
    require((snap['engine'] is None)==(spec['mode']=='reference'),'Worker policy changed')
    require(r['nvtx_push_pop_counts']==([6*count]*2 if spec['traced'] else [0,0]),'Incomplete annotations')
    require(r['bridge_sha256']==(sha(V22/'libnvtx_bridge.so') if spec['traced'] else None),'Bridge changed')
    positive(r['process_peak_rss_kib'],'RSS')
    frames=snap['frames']
    def duration(start,end):return [(x[end]-x[start])/1e6 for x in frames]
    samples=dict(consumer_cadence=v17['consumer_frame_ms'],
        queue_aware=duration('ready_ns','consumer_complete_ns'),
        request_to_complete=duration('request_ns','consumer_complete_ns'),
        admission_wait=duration('request_ns','admitted_ns'),decode=duration('admitted_ns','ready_ns'),
        preparation=duration('prepare_start_ns','prepare_end_ns'),
        cpu_prepare=duration('cpu_prepare_start_ns','cpu_prepare_end_ns'),
        warp_gate_wait=duration('warp_wait_start_ns','warp_gpu_start_ns'),
        warp_gpu_host=duration('warp_gpu_start_ns','warp_gpu_end_ns'),
        detector=duration('detector_start_ns','detector_end_ns'),
        tracking=duration('tracking_start_ns','tracking_end_ns'),
        prepared_to_complete=duration('prepare_end_ns','consumer_complete_ns'))
    overlaps={}
    for main_stage in ('detector','tracking'):
        for worker_stage in ('cpu_prepare','warp_gpu'):
            overlaps[worker_stage+'_with_previous_'+main_stage]=distribution([
                intersection_ns([(a[main_stage+'_start_ns'],a[main_stage+'_end_ns'])],
                                [(b[worker_stage+'_start_ns'],b[worker_stage+'_end_ns'])])/1e6
                for a,b in zip(frames,frames[1:])]) if count>1 else None
    return dict(**spec,count=count,fps=r['fps'],wall_s=r['wall_s'],exact=True,
        metrics={k:distribution(v) for k,v in samples.items()},
        host_adjacent_overlap=overlaps,process_peak_rss_kib=r['process_peak_rss_kib'],
        admission_maximum=snap['admission']['maximum'],geometry_calls=v20['native_geometry_calls'],
        geometry_fallbacks=v20['geometry_fallbacks'],v24_sha256=sha(path.with_suffix('.v24.json'))),samples


def performance(trials,samples):
    clips={}
    for c in WORKLOADS:
        names={m:[f'{c}_repeat{i}_{m}' for i in range(3)] for m in MODES}
        if not all(n in trials for v in names.values() for n in v):continue
        selected={m:[trials[n] for n in names[m]] for m in MODES}
        require(all(sum(t['count'] for t in arm)==384 and not any(t['traced'] or t['attributed'] for t in arm)
                    for arm in selected.values()),'Invalid performance samples')
        pooled={m:sum(t['count'] for t in arm)/sum(t['wall_s'] for t in arm) for m,arm in selected.items()}
        ratios=[selected['native_staged'][i]['fps']/selected['reference'][i]['fps'] for i in range(3)]
        latency={}
        for key in ('queue_aware','consumer_cadence','request_to_complete'):
            p95={m:distribution([x for n in names[m] for x in samples[n][key]])['p95_ms'] for m in MODES}
            pair=[selected['native_staged'][i]['metrics'][key]['p95_ms']/
                  selected['reference'][i]['metrics'][key]['p95_ms'] for i in range(3)]
            latency[key]=dict(pooled_p95_ms=p95,paired_p95_ratios=pair,
                consistent_regression=p95['native_staged']>p95['reference'] and sum(x>1 for x in pair)>=2)
        speedup=pooled['native_staged']/pooled['reference']
        clips[c]=dict(pooled_fps=pooled,speedup=speedup,paired_speedups=ratios,latency=latency,
            passed=speedup>=1.2 and all(x>1 for x in ratios)
                   and not any(v['consistent_regression'] for v in latency.values()))
    return dict(passed=set(clips)==set(WORKLOADS) and all(c['passed'] for c in clips.values()),clips=clips)


def schedules():
    def spec(name,c,m,frames=128,attributed=False):
        return dict(name=name,clip=c,mode=m,frames=frames,attributed=attributed)
    prefix=[spec('smoke_native_staged_'+c,c,'native_staged') for c in WORKLOADS]
    prefix += [spec(f'{c}_repeat{i}_{m}',c,m) for c in WORKLOADS for i in range(3)
               for m in (MODES if i%2==0 else tuple(reversed(MODES)))]
    return (prefix,[spec('full_'+c+'_native_staged',c,'native_staged',None) for c in CLIPS],
            [spec('attribute_'+c+'_native_staged',c,'native_staged',attributed=True) for c in WORKLOADS])


def verify_sources(evidence):
    f,g=read(evidence/'freeze.json'),read(evidence/'generated_01.json')
    require(set(f['files'])==set(FROZEN),'Incomplete native freeze')
    for n,d in f['files'].items():
        require(d==sha(evidence/n)==sha(local_source(n)),'Changed native source '+n)
    for key,name in (('generated','generated_01.json'),('build','build/build.json'),
                     ('unit_gate','unit_gate.json'),('attribution_batch','attribution_batch.json')):
        require(f[key+'_sha256']==sha(evidence/name),'Changed frozen '+key)
    require(f['v24_freeze_sha256']==sha(V24/'freeze.json'),'Changed v24 freeze')
    require(g['passed'] and not g['real_media_read'] and len(g['cases'])==47 and len(g['primitive'])==11
        and all(c['exact'] for c in g['cases']+g['primitive']) and g['native_calls']>0 and g['fallbacks']>0
        and g['passthroughs']==2 and g['independent_outputs'] and g['reentrant_calls']==12
        and g['gil_probe']['passed'] and g['gil_probe']['other_thread_progress_during_call'],
        'Generated correctness/concurrency gate failed')
    require(set(g['source_sha256'])==set(CORE) and all(g['source_sha256'][n]==f['files'][n] for n in CORE),
            'Changed generated sources')
    require(g['reference_sha256']==REFERENCE_SHA==sha(ROOT/'tiny_target/motion/global_motion.py'),
            'Changed global motion reference')
    source=textwrap.dedent(inspect.getsource(gm.fit_global_motion))
    require(source.count(OLD)==1,'Missing unique native transformation anchor')
    transformed=hashlib.sha256(source.replace(OLD,NEW).encode()).hexdigest()
    require(g['transformed_sha256']==transformed,'Changed native transformation')
    b=read(evidence/'build/build.json')
    require(b['returncode']==0 and b['source_sha256']==f['files']['native_motion_v25.cpp']
        and b['builder_sha256']==f['files']['build_native_motion_v25.py']
        and b['library_sha256']==g['library_sha256']==sha(evidence/'build/libnative_motion_v25.so')
        and '-fno-fast-math' in b['command'] and '-ffp-contract=off' in b['command'],
        'Changed or unsafe native build')
    u=read(evidence/'unit_gate.json')
    require(u['passed'] and u['returncode']==0 and u['log_sha256']==sha(evidence/'unit_gate.log')
        and u['tests_sha256']=={n:d for n,d in f['files'].items() if n.startswith('test_')},
        'Native reset/lifecycle/error-path unit gate failed')
    require([r['clip'] for r in g['replays']]==list(WORKLOADS),'Changed replay scope')
    for r in g['replays']:
        p=evidence/f"attribute_{r['clip']}_reference.fits.json";inputs=read(p)
        require(r['exact'] and r['native_calls']>0 and r['fallbacks']>=0
            and [c['frame'] for c in r['cases']]==list(range(1,128))
            and all(c['exact'] for c in r['cases']) and r['json_sha256']==sha(p)
            and r['npz_sha256']==inputs['npz_sha256']==sha(p.with_suffix('.npz'))
            and inputs['clip']==r['clip'] and inputs['frames']==128
            and [c['correspondence']['current_frame_index'] for c in inputs['cases']]==list(range(1,128)),
            'Captured correspondence replay incomplete/changed')
    return f,g


def attribution(events,snapshot):
    """Calling-thread CPU is distinct from host wall time; stages overlap."""
    frames=snapshot['frames'];count=len(frames)
    expected={'correspondence':range(1,count),'global_fit':range(1,count),
              'composition':range(1,count),'warp':range(count),
              'detector':range(count),'tracking':range(count)}
    require(set(e['name'] for e in events)==set(expected),'Unexpected attribution stages')
    grouped={k:[e for e in events if e['name']==k] for k in expected}
    for name,rows in grouped.items():
        require([r['frame'] for r in rows]==list(expected[name]),'Incomplete attribution '+name)
        for e in rows:
            require(e['error'] is None and 0<=e['thread_cpu_ns']<=e['end_ns']-e['start_ns'],
                    'Invalid/error attribution interval')
            f=frames[e['frame']]
            start,end=(('prepare_start_ns','prepare_end_ns') if name not in ('detector','tracking')
                       else ('consumer_received_ns','consumer_complete_ns'))
            require(f[start]<=e['start_ns']<=e['end_ns']<=f[end],'Attribution outside frame interval')
    worker={e['thread'] for e in events if e['name'] not in ('detector','tracking')}
    consumer={e['thread'] for e in events if e['name'] in ('detector','tracking')}
    require(len(worker)==len(consumer)==1 and ((worker==consumer)==(snapshot['policy']=='reference')),
            'Incorrect attribution thread ownership')
    result={}
    for scope,first in (('all',0),('steady_32_127',32)):
        result[scope]={name:dict(wall=distribution([(e['end_ns']-e['start_ns'])/1e6 for e in rows if e['frame']>=first]),
             thread_cpu=distribution([e['thread_cpu_ns']/1e6 for e in rows if e['frame']>=first]))
             for name,rows in grouped.items()}
    return dict(stages=result,events=len(events),instrumented_not_clean_fps=True,
        proves_gil_causality=False,proves_device_overlap=False,
        warning='Wall spans include native work, waits and scheduling. Thread CPU excludes other threads. '
                'Warp includes its gate wait. Nested/concurrent spans must not be summed.')


def audit_diagnostics(evidence,frozen,v24,dependency):
    b=read(evidence/'attribution_batch.json')
    require(b['passed'] and b['error'] is None and b['script_sha256']==frozen['files']['attribute_batch_v25.py']
        and b['attribution_sha256']==frozen['files']['attribute_visible_native_v25.py']
        and b['plan_sha256']==frozen['files']['visible_native_v25_plan.md'],'Diagnostic batch changed')
    require([(r['clip'],r['mode']) for r in b['rows']]==[
        ('0126','reference'),('0126','staged'),('0082','staged'),('0082','reference')],
        'Diagnostic schedule changed')
    result={}
    for row in b['rows']:
        c,m=row['clip'],row['mode'];name=f'attribute_{c}_{m}';path=evidence/name
        r=read(path.with_suffix('.attribution.json'))
        require(row['name']==name and row['returncode']==0 and row['receipt_sha256']==sha(path.with_suffix('.attribution.json'))
            and row['log_sha256']==sha(path.with_suffix('.log')) and r['passed'] and r['error'] is None
            and r['clip']==c and r['mode']==m and r['frames']==r['processed_frames']==128
            and not any(r[k] for k in ('raw16_accessed','defaults_changed','algorithm_changed','native_enabled')),
            'Diagnostic scope/receipt failed')
        require(r['script_sha256']==frozen['files']['attribute_visible_native_v25.py']
            and r['plan_sha256']==frozen['files']['visible_native_v25_plan.md']
            and r['v24_freeze_sha256']==sha(V24/'freeze.json') and r['v24_sources']==v24['files']
            and r['baseline_receipt_sha256']==sha(path.with_suffix('.v20.json')),'Diagnostic provenance changed')
        v=read(path.with_suffix('.v20.json'));v17,report,_=baseline(path,c,128)
        require(v['passed'] and v['error'] is None and v['clip']==c and v['frames']==128 and v['mode']=='candidate'
            and v['processed_frames']==128 and not any(v[k] for k in (
                'raw16_accessed','defaults_changed','gpu_changed','noise_v18_enabled','median_v19_enabled'))
            and v['script_sha256']==sha(V20/'run_visible_v20.py') and v['freeze_sha256']==dependency['freeze_sha256']
            and v['gate_sha256']==dependency['generated_sha256'] and v['library_sha256']==dependency['library_sha256']
            and v['transformed_sha256']==dependency['transformed_sha256']
            and v['baseline_receipt_sha256']==sha(path.with_suffix('.v17.json'))
            and v['native_geometry_calls']>0 and v['geometry_fallbacks']>=0
            and all(r[k]==v[k]==v17[k] for k in ('fps','wall_s')),'Diagnostic baseline chain changed')
        validate_snapshot(r['execution'],128)
        require(r['execution']['policy']==m,'Diagnostic scheduling changed')
        if m=='reference':require(r['replay_json_sha256']==sha(path.with_suffix('.fits.json')),'Capture changed')
        result[name]=dict(clip=c,mode=m,count=128,exact=True,attribution=attribution(r['events'],r['execution']))
    return result


def audit_trial(evidence,spec,row,frozen,generated,v24,dependency):
    path=evidence/spec['name'];r=read(path.with_suffix('.v25.json'))
    require(row['returncode']==0 and r['passed'] and r['error'] is None
        and all(r[k]==spec[k] for k in ('clip','mode','frames','attributed'))
        and r['schema']=='seaqr.visible-native-v25.v1' and not any(r[k] for k in (
            'raw16_accessed','defaults_changed','gpu_changed','algorithm_policy_changed','production_approved')),
        'Native trial scope/failure')
    require(row['v25_sha256']==sha(path.with_suffix('.v25.json')) and r['source_sha256']==frozen['files']
        and r['freeze_sha256']==sha(evidence/'freeze.json') and r['generated_sha256']==sha(evidence/'generated_01.json')
        and r['library_sha256']==generated['library_sha256'] and r['transformed_sha256']==generated['transformed_sha256']
        and r['v24_receipt_sha256']==sha(path.with_suffix('.v24.json')),'Native receipt chain changed')
    require((r['native_calls']>0)==(spec['mode']=='native_staged') and r['fallbacks']>=0 and r['passthroughs']>=0,
            'Native execution missing/unexpected')
    if spec['mode']=='reference':require(r['fallbacks']==r['passthroughs']==0,'Native fallback in reference')
    mapped={k:spec[k] for k in ('name','clip','frames')}
    mapped.update(mode='staged' if spec['mode']=='native_staged' else 'reference',traced=False)
    linked=dict(row,v24_sha256=r['v24_receipt_sha256'])
    trial,samples=audit_underlying(evidence,mapped,linked,v24,dependency)
    require(trial['count']==r['processed_frames'] and all(trial[k]==r[k] for k in ('fps','wall_s')),
            'Native timing/denominator changed')
    trial.update(**spec,native_calls=r['native_calls'],fallbacks=r['fallbacks'],passthroughs=r['passthroughs'],
                 v25_sha256=sha(path.with_suffix('.v25.json')))
    if spec['attributed']:
        trial['attribution']=attribution(r['events'],read(path.with_suffix('.v24.json'))['execution'])
    else:require(not r['events'],'Instrumented clean trial')
    return trial,samples


def verify(evidence):
    evidence=evidence.resolve(strict=True)
    frozen,generated=verify_sources(evidence);v24,_=verify_v24_sources(V24);dependency=verify_dependencies()
    diagnostics=audit_diagnostics(evidence,frozen,v24,dependency)
    b=read(evidence/'batch.json');prefix,full,attributed=schedules()
    require(b['source_sha256']==frozen['files'] and not b['raw16_accessed'] and not b['defaults_changed'],
            'Batch source/scope changed')
    specs,rows=b['schedule'],b['rows']
    require(specs in (prefix,prefix+attributed,prefix+full,prefix+full+attributed),'Unbounded native schedule')
    require(len(rows)<=len(specs),'Extra native trials')
    trials,samples,failed={},{},[]
    for spec,row in zip(specs,rows):
        require(all(row[k]==v for k,v in spec.items()),'Changed trial order')
        require(row['log']==spec['name']+'.log' and row['log_sha256']==sha(evidence/row['log']),'Changed trial log')
        if row['returncode']:
            failed.append(dict(name=spec['name'],returncode=row['returncode']))
            require(len(trials)+1==len(rows),'Continued after failure')
            break
        trials[spec['name']],samples[spec['name']]=audit_trial(evidence,spec,row,frozen,generated,v24,dependency)
    gate=performance(trials,samples)
    if b['performance'] is not None:require(b['performance']==gate,'Independent speed/latency gate differs')
    if full[0] in specs:require(gate['passed'],'Full regressions before passing speed/latency gate')
    if attributed[0] in specs:
        require(set(gate['clips'])==set(WORKLOADS) and all(s['name'] in trials for s in prefix),
                'Attribution before clean pairs')
    completed=not failed and len(rows)==len(specs) and b['passed']
    full_complete=all(s['name'] in trials for s in full)
    require(b['full_regression_run']==full_complete,'Full-regression flag changed')
    if b['passed']:
        require(completed and b['error'] is None and b['performance'] is not None
            and specs==prefix+(full if gate['passed'] else [])+attributed,'False batch completion')
        require(b['decision']==('candidate_requires_independent_audit' if gate['passed'] else 'reject_v25_keep_v20'),
                'Decision inconsistent with gates')
    return dict(schema='seaqr.visible-native-v25-summary.v1',verified=True,completed=completed,
        decision=('incomplete_or_failed' if not completed else 'reject_native_staged_keep_v20' if not gate['passed']
                  else 'opt_in_exact_native_staged_candidate'),prefix_performance=gate,
        completed_batch_runs=len(trials),initial_diagnostics=len(diagnostics),failed_runs=failed,
        verified_frame_instances=sum(t['count'] for t in trials.values())+512,
        full_regression_complete=full_complete,trials=list(trials.values()),diagnostics=diagnostics,
        generated_gate=generated,frozen_sources_sha256=frozen['files'],v20_dependency=dependency,
        batch_sha256=sha(evidence/'batch.json'),verifier_sha256=sha(__file__),
        verifier_dependencies_sha256={n:sha(ROOT/'scripts'/n) for n in (
            'verify_visible_overlap_v23.py','verify_visible_stage_v24.py','verify_visible_v20.py',
            'verify_visible_v17.py','run_visible_stage_v24.py','native_motion_v25.py')},
        defaults_changed=False,production_approved=False,raw16_paused=True,new_airborne_accuracy_validated=False,
        warning='Repeated development prefixes, not accuracy/generalization or real-time validation. '
            'Latency starts at file-frame request, not sensor exposure or upstream codec backlog. '
            'Attribution is diagnostic only, not clean throughput or proof of GIL causality/GPU overlap. '
            'All native motion scoring changes are layered in v25 receipts over unchanged v24 scheduling receipts.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.evidence);write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','diagnostics','generated_gate')},indent=2))

