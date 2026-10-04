"""Independent report-only v26 provenance, full parity and latency audit."""
import argparse
import json
import math
from pathlib import Path
from run_visible_front_v26 import FROZEN,GATE
from visible_front_v26 import CUDA_SOURCES,REFERENCE_LIBRARY_SHA
from check_visible_front_v26 import noise_cases
from verify_visible_overlap_v23 import (ROOT,V20,read,sha,write,require,positive,
    distribution,local_source,verify_dependencies)
from verify_visible_stage_v24 import verify_sources as verify_v24_sources
from run_visible_stage_v24 import validate_snapshot
from verify_visible_v17 import ARCHIVE
from video_checks_v19 import check

V24=ROOT/'results/tiny_target/visible_stage_v24_20260920/evidence'
V17=ROOT/'results/tiny_target/visible_speed_v17_20260917/evidence'
WORKLOADS=('0126','0082');MODES=('reference','candidate');CLIPS=('0029','0126','0055','0082')


def schedules():
    def spec(name,c,m,frames=128):return dict(name=name,clip=c,mode=m,frames=frames)
    prefix=[spec('smoke_candidate_'+c,c,'candidate') for c in WORKLOADS]
    prefix += [spec(f'{c}_repeat{i}_{m}',c,m) for c in WORKLOADS for i in range(3)
               for m in (MODES if i%2==0 else tuple(reversed(MODES)))]
    return prefix,[spec('full_'+c+'_candidate',c,'candidate',None) for c in CLIPS]


def safe_relative(value):
    p=Path(value)
    require(not p.is_absolute() and '..' not in p.parts and bool(p.parts),'Unsafe relative artifact')
    return p


def verify_sources(evidence):
    f=read(evidence/'freeze.json');require(f['gate_file']==GATE,'Changed selected generated gate')
    g=read(evidence/GATE);b=read(evidence/safe_relative(g['build_relative']))
    require(set(f['files'])==set(FROZEN),'Incomplete source freeze')
    for n,d in f['files'].items():require(d==sha(evidence/n)==sha(local_source(n)),'Changed frozen source '+n)
    for key,name in (('gate',GATE),('config','candidate_config.json'),('transition','transition.json'),('unit_gate','unit_gate.json')):
        require(f[key+'_sha256']==sha(evidence/name),'Changed frozen '+key)
    require(f['v20_freeze_sha256']==sha(V20/'freeze.json') and f['v24_freeze_sha256']==sha(V24/'freeze.json'),
            'Changed reference dependency')
    expected_noise=[r[0] for r in noise_cases()]+['reject_inf','reject_-inf','reject_nan']
    expected_detector=[f'host_margin{margin}_{shape}_{i}' for margin in (.5,2,4.25)
        for shape in ((7,13),(67,99),(257,319)) for i in range(12)]
    expected_detector += [f'device_{shape}_{i}' for shape,n in (((67,99),12),((3190,4784),10)) for i in range(n)]
    require(g['passed'] and g['error'] is None and not g['real_media_read']
        and [r['name'] for r in g['noise']]==expected_noise and len(g['noise'])==50
        and [r['name'] for r in g['detector']]==expected_detector and len(g['detector'])==130
        and all(r['exact'] for r in g['noise']+g['detector']),'Incomplete generated exactness gate')
    original_launch=read(ARCHIVE/'visible_0126_full_reuse/launch.json')
    require(g['conformance']==original_launch['exact_cuda_stabilization']['conformance']
        and g['config_sha256']==original_launch['config_sha256']
        and g['reference_library_sha256']==REFERENCE_LIBRARY_SHA==original_launch['exact_cuda_stabilization']['library_sha256'],
        'Changed reference arithmetic/configuration')
    require(g['plan_sha256']==f['files']['visible_front_v26_plan.md']
        and set(g['source_sha256'])=={'visible_front_v26.cu','visible_front_v26.py','build_visible_front_v26.py','check_visible_front_v26.py'}
        and all(d==f['files'][n] for n,d in g['source_sha256'].items()),'Generated source mismatch')
    require(b['passed'] and b['returncode']==0 and b['reference_library_sha256']==REFERENCE_LIBRARY_SHA
        and b['original_sources_sha256']==CUDA_SOURCES and set(b['source_sha256'])==set(CUDA_SOURCES)|{'visible_front_v26.cu'}
        and g['build_sha256']==sha(evidence/g['build_relative']) and g['library_sha256']==b['library_sha256']
        and g['library_sha256']==sha(evidence/safe_relative(g['library_relative']))
        and b['builder_sha256']==f['files']['build_visible_front_v26.py']
        and b['adapter_sha256']==f['files']['visible_front_v26.py'],'Changed additive GPU build')
    require(all(v in b['command'] for v in ('--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','-arch=sm_87'))
        and '--use_fast_math' not in b['command'],'Unsafe CUDA arithmetic flags')
    for n,d in b['source_sha256'].items():
        require(d==sha((evidence/g['build_relative']).parent/'source'/n),'Compiled source changed '+n)
        require(d==(CUDA_SOURCES[n] if n in CUDA_SOURCES else f['files'][n]),'Original kernel unexpectedly edited '+n)
    u=read(evidence/'unit_gate.json')
    require(u['passed'] and u['returncode']==0 and u['test_sha256']==f['files']['test_visible_front_v26.py']
        and u['log_sha256']==sha(evidence/'unit_gate.log'),'Ownership/error-path unit gate failed')
    transition=read(evidence/'transition.json')
    require(transition==dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256=REFERENCE_LIBRARY_SHA,
        after_library_sha256=g['library_sha256'],candidate_build_sha256=g['build_sha256']),'Wrong directional GPU transition')
    cfg=read(evidence/'candidate_config.json');expected=dict(original_launch['configuration'])
    expected['cuda_median_library']=cfg['cuda_median_library']
    require(cfg==expected and cfg['cuda_median_library'].endswith('/'+g['library_relative']),'Configuration policy changed')
    return f,g


def audit_trial(evidence,spec,row,frozen,generated,dependency):
    path=evidence/spec['name'];r=read(path.with_suffix('.v26.json'));candidate=spec['mode']=='candidate'
    require(row['returncode']==0 and r['passed'] and r['error'] is None
        and r['schema']=='seaqr.visible-front-v26.v1' and all(r[k]==spec[k] for k in ('clip','mode','frames')),
        'Trial failed/scope changed')
    require(not any(r[k] for k in ('raw16_accessed','defaults_changed','production_approved','new_accuracy_validated',
        'staged_v24_enabled','native_motion_v25_enabled')) and r['execution_policy']=='serial_reference'
        and r['gpu_changed']==r['native_front_enabled']==candidate,'Algorithm/scheduling policy changed')
    require(row['v26_sha256']==sha(path.with_suffix('.v26.json')) and r['source_sha256']==frozen['files']
        and r['freeze_sha256']==sha(evidence/'freeze.json') and r['gate_sha256']==sha(evidence/GATE)
        and r['transition_sha256']==sha(evidence/'transition.json')
        and r['library_sha256']==(generated['library_sha256'] if candidate else REFERENCE_LIBRARY_SHA)
        and r['v20_freeze_sha256']==sha(V20/'freeze.json') and r['v24_freeze_sha256']==sha(V24/'freeze.json')
        and r['v17_gate_sha256']==sha(V17/'generated_01.json')
        and r['v17_library_sha256']==sha(V17/'build/liblearning_mask_v17.so')
        and r['geometry_library_sha256']==dependency['library_sha256']
        and r['geometry_transformed_sha256']==dependency['transformed_sha256'],'Provenance chain changed')
    ref=ARCHIVE/f"visible_{spec['clip']}_full_reuse";left=read(ref/'launch.json')
    config=read(evidence/'candidate_config.json') if candidate else left['configuration']
    config_sha=sha(evidence/'candidate_config.json') if candidate else left['config_sha256']
    count=spec['frames'] or read(ref/'report.json')['frames']
    result=check(ref,path,count,spec['frames'] is None,spec['mode'],config,config_sha,read(evidence/'transition.json'))
    require(r['comparison']==result and r['config_sha256']==config_sha and r['processed_frames']==count,
            'Output/metadata comparison changed')
    motion=read(path.with_suffix('.execution.json'));report=read(path/'report.json')
    require(motion['clip']==spec['clip'] and motion['frames']==spec['frames'] and not motion['injected'],
            'Changed motion scope')
    require(r['geometry_calls']>0 and r['geometry_fallbacks']>=0,'Missing v20 geometry')
    if candidate:
        require(len(r['fronts'])==1 and r['native_mask_calls']==0,'Unexpected front/CPU learning path')
        front=r['fronts'][0]
        require(front['calls']==front['device_calls']==front['finish_calls']==count and front['host_calls']==0
            and front['closed'] and front['learning_points']>=0,'Incomplete native front/lifecycle')
    else:require(not r['fronts'] and r['native_mask_calls']>0,'Reference helper changed')
    for k,report_key in (('fps','processed_fps'),('wall_s','elapsed_seconds')):
        positive(r[k],k);require(r[k]==row[k]==report[report_key],'Timing receipt mismatch')
    require(math.isclose(r['fps'],count/r['wall_s'],rel_tol=1e-12),'Changed FPS denominator')
    snap=r['execution'];validate_snapshot(snap,count)
    require(snap['policy']=='reference' and snap['mode']=='reference' and snap['engine'] is None,'Staging unexpectedly enabled')
    require(len(r['consumer_frame_ms'])==count,'Missing cadence samples')
    frames=snap['frames']
    def duration(start,end):return [(f[end]-f[start])/1e6 for f in frames]
    samples=dict(consumer_cadence=r['consumer_frame_ms'],queue_aware=duration('ready_ns','consumer_complete_ns'),
        request_to_complete=duration('request_ns','consumer_complete_ns'),admission_wait=duration('request_ns','admitted_ns'),
        decode=duration('admitted_ns','ready_ns'),cpu_prepare=duration('cpu_prepare_start_ns','cpu_prepare_end_ns'),
        warp_gpu_host=duration('warp_gpu_start_ns','warp_gpu_end_ns'),
        detector=duration('detector_start_ns','detector_end_ns'),tracking=duration('tracking_start_ns','tracking_end_ns'))
    positive(r['process_peak_rss_kib'],'RSS')
    return dict(**spec,count=count,fps=r['fps'],wall_s=r['wall_s'],exact=True,
        metrics={k:distribution(v) for k,v in samples.items()},process_peak_rss_kib=r['process_peak_rss_kib'],
        geometry_calls=r['geometry_calls'],geometry_fallbacks=r['geometry_fallbacks'],fronts=r['fronts'],
        native_mask_calls=r['native_mask_calls'],v26_sha256=sha(path.with_suffix('.v26.json'))),samples


def performance(trials,samples):
    result={}
    for c in WORKLOADS:
        names={m:[f'{c}_repeat{i}_{m}' for i in range(3)] for m in MODES}
        if not all(n in trials for arm in names.values() for n in arm):continue
        selected={m:[trials[n] for n in arm] for m,arm in names.items()}
        require(all(t['count']==t['frames']==128 and t['mode']==m and t['clip']==c for m,arm in selected.items() for t in arm),
                'Invalid benchmark samples')
        fps={m:384/sum(t['wall_s'] for t in arm) for m,arm in selected.items()}
        paired=[selected['candidate'][i]['fps']/selected['reference'][i]['fps'] for i in range(3)];latency={}
        for label in ('queue_aware','consumer_cadence','request_to_complete'):
            p95={m:distribution([v for n in names[m] for v in samples[n][label]])['p95_ms'] for m in MODES}
            ratios=[selected['candidate'][i]['metrics'][label]['p95_ms']/selected['reference'][i]['metrics'][label]['p95_ms'] for i in range(3)]
            latency[label]=dict(pooled_p95_ms=p95,paired_p95_ratios=ratios,
                consistent_regression=p95['candidate']>p95['reference'] and sum(v>1 for v in ratios)>=2)
        speedup=fps['candidate']/fps['reference']
        result[c]=dict(pooled_fps=fps,speedup=speedup,paired_speedups=paired,latency=latency,
            passed=speedup>=1.2 and all(v>1 for v in paired) and not any(v['consistent_regression'] for v in latency.values()))
    return dict(passed=set(result)==set(WORKLOADS) and all(v['passed'] for v in result.values()),clips=result)


def verify(evidence):
    evidence=evidence.resolve(strict=True);f,g=verify_sources(evidence)
    verify_v24_sources(V24);dependency=verify_dependencies()
    b=read(evidence/'batch.json');prefix,full=schedules()
    require(b['source_sha256']==f['files'] and not b['raw16_accessed'] and not b['defaults_changed'],'Batch source/scope changed')
    specs,rows=b['schedule'],b['rows'];require(specs in (prefix,prefix+full) and len(rows)<=len(specs),'Unexpected video schedule')
    trials,samples,failed={},{},[]
    remote=Path(read(evidence/'candidate_config.json')['cuda_median_library']).parents[1]
    for spec,row in zip(specs,rows):
        require(all(row[k]==v for k,v in spec.items()),'Changed trial order')
        command=['/usr/bin/python3',str(remote/'run_visible_front_v26.py'),'--clip',spec['clip'],
                 '--mode',spec['mode'],'--output',str(remote/spec['name'])]
        if spec['frames'] is not None:command+=['--frames',str(spec['frames'])]
        require(row['command']==command,'Changed trial launch command')
        require(row['log']==spec['name']+'.log' and row['log_sha256']==sha(evidence/row['log']),'Changed trial log')
        if row['returncode']:
            failed.append(dict(name=spec['name'],returncode=row['returncode']))
            require(len(trials)+1==len(rows),'Continued after failure');break
        trials[spec['name']],samples[spec['name']]=audit_trial(evidence,spec,row,f,g,dependency)
    gate=performance(trials,samples)
    if b['performance'] is not None:require(b['performance']==gate,'Independent performance gate differs')
    if len(specs)>len(prefix):require(gate['passed'],'Full regression scheduled before passing prefix gate')
    full_complete=all(s['name'] in trials for s in full)
    require(b['full_regression_run']==full_complete,'Full regression flag changed')
    completed=not failed and len(rows)==len(specs) and b['passed']
    if b['passed']:
        require(completed and b['error'] is None and b['performance'] is not None
            and specs==prefix+(full if gate['passed'] else []),'False batch completion')
        require(b['decision']==('candidate_requires_independent_audit' if gate['passed'] else 'reject_front_keep_v20'),
                'Decision contradicts evidence')
    native_diag=[t for t in g['detector'] if t['shape']==[3190,4784]][2:]
    pooled_stages={}
    for clip in WORKLOADS:
        names={mode:[f'{clip}_repeat{i}_{mode}' for i in range(3)] for mode in MODES}
        if all(n in trials for arm in names.values() for n in arm):
            pooled_stages[clip]={mode:{stage:distribution([v for n in arm for v in samples[n][stage]])
                for stage in samples[arm[0]]} for mode,arm in names.items()}
    return dict(schema='seaqr.visible-front-v26-summary.v1',verified=True,completed=completed,
        decision=('incomplete_or_failed' if not completed else 'opt_in_exact_resident_front_candidate' if gate['passed'] and full_complete
                  else 'reject_resident_front_keep_v20'),prefix_performance=gate,
        completed_runs=len(trials),scheduled_runs=len(specs),failed_runs=failed,full_regression_complete=full_complete,
        verified_frame_instances=sum(t['count'] for t in trials.values()),
        full_regression_frames=sum(t['count'] for t in trials.values() if t['frames'] is None),
        unique_development_frames=sum(max((t['count'] for t in trials.values() if t['clip']==c),default=0) for c in CLIPS),
        paired_stage_metrics=pooled_stages,
        trials=list(trials.values()),generated_gate=g,frozen_sources_sha256=f['files'],
        generated_native_detector_ms={m:distribution([t['detector_ms'][m] for t in native_diag]) for m in MODES},
        batch_sha256=sha(evidence/'batch.json'),verifier_sha256=sha(__file__),
        verifier_dependencies_sha256={n:sha(ROOT/'scripts'/n) for n in (
            'verify_visible_overlap_v23.py','verify_visible_stage_v24.py','video_checks_v19.py',
            'compare_phase20_exact_runs.py','repeat_phase20_kernel_speed.py','run_visible_stage_v24.py','visible_front_v26.py',
            'run_visible_front_v26.py','verify_visible_v17.py','verify_visible_v20.py','check_visible_front_v26.py')},
        production_approved=False,defaults_changed=False,raw16_paused=True,new_airborne_accuracy_validated=False,
        warning='Development output parity, not airborne accuracy/generalization or live real-time validation. '
            'Only prefixes have fresh paired timing. Latency begins at decoder request, not sensor exposure. '
            'Generated timings exclude warp and are not pipeline FPS. Host stage times are not GPU busy time.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.evidence);write(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trials','generated_gate')},indent=2))
