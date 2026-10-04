"""Report-only independent v24 parity, provenance and latency audit; no media."""
import argparse
import json
import math
from pathlib import Path
from run_visible_stage_v24 import FROZEN, DEPENDENCIES, TESTS, validate_snapshot
from verify_visible_overlap_v23 import (ROOT, V20, V22, read, sha, write, require, positive,
    distribution, local_source, baseline, verify_dependencies, intersection_ns)

WORKLOADS=('0126','0082')
MODES=('reference','staged')
CLIPS=('0029','0126','0055','0082')


def schedules():
    def spec(name,c,m,frames=128,traced=False):
        return dict(name=name,clip=c,mode=m,frames=frames,traced=traced)
    prefix=[spec('bounded_'+c,c,'bounded') for c in WORKLOADS]
    prefix += [spec('smoke_staged_'+c,c,'staged') for c in WORKLOADS]
    prefix += [spec(f'{c}_repeat{i}_{m}',c,m) for c in WORKLOADS for i in range(3)
               for m in (MODES if i%2==0 else tuple(reversed(MODES)))]
    return (prefix, [spec('full_'+c+'_staged',c,'staged',None) for c in CLIPS],
            [spec('trace_'+c+'_staged',c,'staged',traced=True) for c in WORKLOADS])


def verify_sources(evidence):
    f,g=read(evidence/'freeze.json'),read(evidence/'generated_01.json')
    require(set(f['files'])==set(FROZEN),'Incomplete v24 source freeze')
    for name,digest in f['files'].items():
        require(digest==sha(evidence/name)==sha(local_source(name)),'Changed source '+name)
        if name in DEPENDENCIES:
            require(digest==DEPENDENCIES[name],'Changed immutable v23 dependency')
    require(g['passed'] and g['returncode']==0 and not g['real_media_read']
            and g['tests']==list(TESTS) and g['source_sha256']==f['files']
            and g['log_sha256']==sha(evidence/'generated_01.log')
            and f['generated_sha256']==sha(evidence/'generated_01.json'),'Generated test chain failed')
    return f,g


def audit_trial(evidence,spec,row,frozen,dependency):
    path=evidence/spec['name'];r=read(path.with_suffix('.v24.json'))
    require(row['returncode']==0 and r['passed'] and r['error'] is None,'Failed trial')
    require(all(r[k]==spec[k] for k in ('clip','mode','frames','traced')),'Scope mismatch')
    require(r['schema']=='seaqr.visible-stage-v24.v1' and not any(r[k] for k in (
        'raw16_accessed','defaults_changed','gpu_changed','algorithm_changed','production_approved')),
        'Algorithm/default or evidence scope changed')
    require(r['source_sha256']==frozen['files'] and r['freeze_sha256']==sha(evidence/'freeze.json')
        and r['generated_sha256']==sha(evidence/'generated_01.json')
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
        require(all(sum(t['count'] for t in arm)==384 and not any(t['traced'] for t in arm)
                    for arm in selected.values()),'Invalid performance samples')
        pooled={m:sum(t['count'] for t in arm)/sum(t['wall_s'] for t in arm) for m,arm in selected.items()}
        ratios=[selected['staged'][i]['fps']/selected['reference'][i]['fps'] for i in range(3)]
        latency={}
        for key in ('queue_aware','consumer_cadence','request_to_complete'):
            p95={m:distribution([x for n in names[m] for x in samples[n][key]])['p95_ms'] for m in MODES}
            pair=[selected['staged'][i]['metrics'][key]['p95_ms']/
                  selected['reference'][i]['metrics'][key]['p95_ms'] for i in range(3)]
            latency[key]=dict(pooled_p95_ms=p95,paired_p95_ratios=pair,
                consistent_regression=p95['staged']>p95['reference'] and sum(x>1 for x in pair)>=2)
        speedup=pooled['staged']/pooled['reference']
        clips[c]=dict(pooled_fps=pooled,speedup=speedup,paired_speedups=ratios,latency=latency,
            passed=speedup>=1.2 and all(x>1 for x in ratios)
                   and not any(v['consistent_regression'] for v in latency.values()))
    return dict(passed=set(clips)==set(WORKLOADS) and all(c['passed'] for c in clips.values()),clips=clips)


def verify(evidence):
    evidence=evidence.resolve(strict=True)
    frozen,generated=verify_sources(evidence);dependency=verify_dependencies()
    batch=read(evidence/'batch.json');prefix,full,traces=schedules()
    require(batch['source_sha256']==frozen['files'] and not batch['raw16_accessed']
            and not batch['defaults_changed'],'Batch source/scope changed')
    specs,rows=batch['schedule'],batch['rows']
    require(specs in (prefix,prefix+traces,prefix+full,prefix+full+traces),'Unbounded schedule')
    require(len(rows)<=len(specs),'Unexpected extra runs')
    trials,samples,failed={},{},[]
    for spec,row in zip(specs,rows):
        require(all(row[k]==v for k,v in spec.items()),'Batch order changed')
        require(row['log']==spec['name']+'.log' and row['log_sha256']==sha(evidence/row['log']),'Log changed')
        if row['returncode']:
            receipt=evidence/(spec['name']+'.v24.json')
            failed.append(dict(name=spec['name'],returncode=row['returncode'],
                error=read(receipt).get('error') if receipt.exists() else None))
            require(len(trials)+len(failed)==len(rows),'Continued after failure')
            break
        trials[spec['name']],samples[spec['name']]=audit_trial(evidence,spec,row,frozen,dependency)
    gate=performance(trials,samples)
    complete_gate=set(gate['clips'])==set(WORKLOADS)
    if batch['performance'] is not None:
        require(complete_gate and batch['performance']==gate,'Independent performance gate differs')
    if full[0] in specs:require(gate['passed'],'Full clips scheduled before passing prefix gate')
    if traces[0] in specs:
        require(complete_gate and all(s['name'] in trials for s in prefix),'Traces before clean prefixes')
    completed=not failed and len(rows)==len(specs) and batch['passed']
    full_complete=all(s['name'] in trials for s in full)
    require(batch['full_regression_run']==full_complete,'Full regression flag mismatch')
    if batch['passed']:
        require(completed and batch['error'] is None and batch['performance'] is not None,'False batch success')
        require(specs==prefix+(full if gate['passed'] else [])+traces,'Required runs omitted')
        require(batch['decision']==('candidate_requires_independent_audit' if gate['passed'] else 'reject_v24_keep_v20'),
                'Decision contradicts evidence')
    return dict(schema='seaqr.visible-stage-v24-summary.v1',verified=True,completed=completed,
        decision=('incomplete_or_failed' if not completed else 'reject_staged_retain_v20' if not gate['passed']
                  else 'opt_in_exact_staged_candidate'),prefix_performance=gate,
        completed_successful_runs=len(trials),scheduled_runs=len(specs),failed_runs=failed,
        verified_frame_instances=sum(t['count'] for t in trials.values()),trials=list(trials.values()),
        full_regression_complete=full_complete,trace_device_timeline_verified=False,
        generated_gate=generated,frozen_sources_sha256=frozen['files'],v20_dependency=dependency,
        batch_sha256=sha(evidence/'batch.json'),verifier_sha256=sha(__file__),
        verifier_dependencies_sha256={n:sha(ROOT/'scripts'/n) for n in (
            'verify_visible_overlap_v23.py','verify_visible_v20.py','run_visible_stage_v24.py')},
        raw16_paused=True,defaults_changed=False,production_approved=False,new_airborne_accuracy_validated=False,
        warning='Repeated development prefixes, not new accuracy/generalization or real-time validation. '
            'Request latency includes admission wait but not camera acquisition or upstream codec backlog. '
            'warp_gpu is a host span, not device execution. Overlapped spans must not be summed. '
            'Traced throughput is excluded from acceptance. Peak RSS is a process high-water mark.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.evidence);write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','generated_gate')},indent=2))
