"""Independent report-only verification of exact GPU median experiment evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_visible_v17 import read,sha,write
from median_v19 import SOURCE_SHA,REFERENCE_LIBRARY_SHA,network,transform,proof_source
from batch_visible_v19 import schedule
from video_checks_v19 import check
from verify_visible_v17 import require,distribution,ARCHIVE

V17=ROOT/'results/tiny_target/visible_speed_v17_20260917/evidence'
CUDA_SOURCE=ROOT/'results/tiny_target/phase20/kernel_efficiency_v9_20260914/jetson/libseaqr_peak_gate_sources'


def verify_diagnostics(evidence,transition):
    """Verify separate actual-input probes; never include their pipeline timing."""
    require(sha(evidence/'diagnose_median_v19.py')==sha(ROOT/'scripts/diagnose_median_v19.py'),'Diagnostic source changed')
    summaries=[]
    for clip in ('0126','0082'):
        path=evidence/f'diagnostic_{clip}'
        d=read(path.with_suffix('.diagnostic.json'));r=read(path.with_suffix('.v19.json'))
        require(d['passed'] and d['error'] is None and d['diagnostic_only'] and d['not_pipeline_fps']
                and not d['raw16_accessed'] and d['clip']==clip and d['frames']==128,'Diagnostic scope mismatch')
        require(d['script_sha256']==sha(evidence/'diagnose_median_v19.py')
                and d['gate_sha256']==sha(evidence/'generated_01.json')
                and d['trial_sha256']==sha(path.with_suffix('.v19.json')),'Diagnostic provenance mismatch')
        require(r['passed'] and r['error'] is None and r['clip']==clip and r['frames']==128
                and r['mode']=='candidate' and not r['raw16_accessed'] and not r['defaults_changed']
                and not r['noise_v18_enabled'] and r['processed_frames']==128,'Diagnostic trial failed')
        require(r['script_sha256']==sha(evidence/'run_visible_v19.py')
                and r['freeze_sha256']==sha(evidence/'integration_freeze.json')
                and r['gate_sha256']==sha(evidence/'generated_01.json')
                and r['v17_gate_sha256']==sha(V17/'generated_01.json')
                and r['v17_library_sha256']==sha(V17/'build/liblearning_mask_v17.so')
                and r['native_mask_calls']>0,'Diagnostic baseline mismatch')
        cfg=read(evidence/'candidate_config.json');digest=sha(evidence/'candidate_config.json')
        result=check(ARCHIVE/f'visible_{clip}_full_reuse',path,128,False,'candidate',cfg,digest,transition)
        require(result==r['comparison'] and r['config_sha256']==digest,'Diagnostic journal changed')
        require(d['selected_frame_indices']==[31,63,95,127]
                and [row['frame'] for row in d['rows']]==d['selected_frame_indices'],'Diagnostic samples changed')
        for row in d['rows']:
            require(row['exact'] and row['pixels']==3190*4784
                    and 0<=row['guarded_windows']<=row['pixels']
                    and row['guarded_fraction']==row['guarded_windows']/row['pixels'],'Invalid guard count')
            special=sum(row[k] for k in ('negative_zero_pixels','nonfinite_pixels','subnormal_pixels'))
            require(all(isinstance(row[k],int) and row[k]>=0 for k in
                    ('negative_zero_pixels','nonfinite_pixels','subnormal_pixels'))
                    and special<=row['guarded_windows']<=min(row['pixels'],25*special),'Inconsistent guard population')
            require(len(row['input_sha256'])==64 and row['timings']['reference']['output_sha256']
                    ==row['timings']['candidate']['output_sha256'],'Actual-input median mismatch')
            for mode in ('reference','candidate'):
                require(len(row['timings'][mode]['event_ms'])==3
                        and all(0<v<float('inf') for v in row['timings'][mode]['event_ms']),'Invalid diagnostic event timing')
        summaries.append(dict(clip=clip,sampled_frames=4,rows=d['rows'],exact_non_timing_outputs=True,
            event_mean_ms={mode:sum(v for row in d['rows'] for v in row['timings'][mode]['event_ms'])/12
                           for mode in ('reference','candidate')}))
    return summaries


def verify(evidence):
    build=read(evidence/'build/build.json');gate=read(evidence/'generated_01.json')
    require(build['passed'] and build['reference_library_sha256']==REFERENCE_LIBRARY_SHA
            and build['original_sources_sha256']==SOURCE_SHA,'Build baseline mismatch')
    require(set(build['source_sha256'])=={'median_v19.py','build_median_v19.py','median_probe_v19.h'},'Incomplete build provenance')
    for name,digest in build['source_sha256'].items():
        require(sha(ROOT/'scripts'/name)==digest==sha(evidence/name),'Builder source changed: '+name)
    proof=read(evidence/'build/proof/proof.json')
    require(proof==build['proof'] and proof['passed'] and proof['cases']==1<<25 and proof['minmax_nodes']==len(network()[0])
            and proof['source_sha256']==sha(evidence/'build/proof/proof.cpp')
            and (evidence/'build/proof/proof.cpp').read_text()==proof_source(),'Exhaustive proof changed')
    require(proof['generator_sha256']==sha(ROOT/'scripts/median_v19.py'),'Proof generator mismatch')
    for variant in ('candidate','reference_probe','candidate_probe'):
        record=read(evidence/'build'/(variant+'.build.json'))
        require(record==build['builds'][variant] and record['returncode']==0
                and record['library_sha256']==sha(evidence/'build'/(variant+'.so')),'Compiled library mismatch')
        require('--fmad=false' in record['command'] and '-arch=sm_87' in record['command'],'Compiler contract changed')
        directory=evidence/'build'/variant
        expected_names=set(SOURCE_SHA) | ({'probe.cu','median_probe_v19.h'} if variant.endswith('_probe') else set())
        require(set(record['generated_sha256'])==expected_names,'Unexpected generated source topology')
        for name,digest in record['generated_sha256'].items():
            require(sha(directory/name)==digest,'Generated source changed: '+variant+'/'+name)
        for name,digest in SOURCE_SHA.items():
            require(sha(CUDA_SOURCE/name)==digest,'Frozen kernel source changed')
            expected=(CUDA_SOURCE/name).read_text()
            if name=='phase20_cuda_median.cu' and variant!='reference_probe':expected=transform(expected)
            require((directory/name).read_text()==expected,'Only median network may change')
        if variant.endswith('_probe'):
            require((directory/'probe.cu').read_text()=='#include "phase20_cuda_integrated.cu"\n#include "median_probe_v19.h"\n'
                    and sha(directory/'median_probe_v19.h')==sha(ROOT/'scripts/median_probe_v19.h'),'Diagnostic source mismatch')
    require(gate['passed'] and gate['error'] is None and not gate['real_media_read'] and len(gate['cases'])==259
            and len(gate['timings'])==24 and all(r['exact'] and r['inputs_unchanged'] for r in gate['cases']), 'GPU gate failed')
    require(gate['build_sha256']==sha(evidence/'build/build.json') and gate['script_sha256']==sha(ROOT/'scripts/check_median_v19.py')
            ==sha(evidence/'check_median_v19.py') and gate['reference_library_sha256']==REFERENCE_LIBRARY_SHA
            and gate['candidate_library_sha256']==sha(evidence/'build/candidate.so'),'GPU gate provenance mismatch')
    require(gate['plan_sha256']==sha(ROOT/'docs/visible_speed_v19_plan.md')==sha(evidence/'visible_speed_v19_plan.md'), 'Protocol changed')
    require(sha(evidence/'profile_visible_v17.py')==sha(ROOT/'scripts/profile_visible_v17.py'),'Shared utilities changed')
    transition=read(evidence/'transition.json')
    require(transition==dict(schema='seaqr.exact-gpu-transition.v1',before_library_sha256=REFERENCE_LIBRARY_SHA,
        after_library_sha256=gate['candidate_library_sha256'],candidate_build_sha256=sha(evidence/'build/build.json')),'Wrong binary transition')
    frozen=read(evidence/'integration_freeze.json')
    require(frozen['config_sha256']==sha(evidence/'candidate_config.json') and frozen['transition_sha256']==sha(evidence/'transition.json')
            and frozen['gate_sha256']==sha(evidence/'generated_01.json'),'Integration freeze mismatch')
    require(set(frozen['scripts_sha256'])=={'run_visible_v19.py','video_checks_v19.py','batch_visible_v19.py'},'Incomplete integration provenance')
    for name,digest in frozen['scripts_sha256'].items():
        require(sha(ROOT/'scripts'/name)==digest==sha(evidence/name),'Harness changed: '+name)
    batch=read(evidence/'trials_01/batch.json')
    require(batch['passed'] and batch['error'] is None and batch['schedule']==schedule() and len(batch['rows'])==16
            and batch['script_sha256']==sha(ROOT/'scripts/batch_visible_v19.py'),'Incomplete batch')
    trials,stored={},{}
    for spec,row in zip(schedule(),batch['rows']):
        require(all(row[k]==v for k,v in spec.items()),'Schedule/order changed')
        name=spec['name'];path=evidence/'trials_01'/name
        r=read(path.with_suffix('.v19.json'))
        require(r['passed'] and r['error'] is None and row['sha256']==sha(path.with_suffix('.v19.json')),'Failed trial')
        require(all(r[k]==spec[k] for k in ('clip','mode','frames')),'Wrong execution scope')
        require(r['script_sha256']==sha(ROOT/'scripts/run_visible_v19.py') and r['freeze_sha256']==sha(evidence/'integration_freeze.json')
                and r['gate_sha256']==sha(evidence/'generated_01.json') and not r['raw16_accessed']
                and not r['defaults_changed'] and not r['noise_v18_enabled'],'Trial provenance mismatch')
        require(r['v17_gate_sha256']==sha(V17/'generated_01.json')
                and r['v17_library_sha256']==sha(V17/'build/liblearning_mask_v17.so') and r['native_mask_calls']>0,'v17 baseline missing')
        reference=ARCHIVE/f'visible_{spec["clip"]}_full_reuse'
        old=read(reference/'launch.json');report=read(path/'report.json')
        cfg=read(evidence/'candidate_config.json') if spec['mode']=='candidate' else old['configuration']
        config_sha=sha(evidence/'candidate_config.json') if spec['mode']=='candidate' else old['config_sha256']
        count=spec['frames'] or read(reference/'report.json')['frames']
        result=check(reference,path,count,spec['frames'] is None,spec['mode'],cfg,config_sha,transition)
        require(result==r['comparison'] and r['config_sha256']==config_sha,'Comparison receipt changed')
        require(r['processed_frames']==count==len(r['consumer_frame_ms'])
                and r['fps']==row['fps']==report['processed_fps'] and r['wall_s']==row['wall_s']==report['elapsed_seconds'],'Timing mismatch')
        motion=read(path.with_suffix('.execution.json'))
        require(motion['clip']==spec['clip'] and motion['frames']==spec['frames'] and not motion['injected'],'Motion request mismatch')
        rss=[int(v['rss']['VmRSS'].split()[0]) for v in motion['motion'] if v.get('rss')]
        require(bool(rss),'Missing RSS samples')
        trials[name]=dict(**spec,count=count,fps=r['fps'],wall_s=r['wall_s'],exact=True,
            consumer_service=distribution(r['consumer_frame_ms']),
            stage_means_ms={k:v['mean'] for k,v in report['timings_ms'].items()},native_mask_calls=r['native_mask_calls'],
            rss_min_kib=min(rss),rss_max_kib=max(rss),rss_first_kib=rss[0],rss_last_kib=rss[-1])
        stored[name]=r
    performance={}
    for clip in ('0126','0082'):
        arms={}
        for mode in ('reference','candidate'):
            names=[n for n,r in trials.items() if r['clip']==clip and r['mode']==mode and r['frames']==128]
            wall=sum(trials[n]['wall_s'] for n in names)
            arms[mode]=dict(frames=384,wall_s=wall,fps=384/wall,
                consumer_service=distribution([v for n in names for v in stored[n]['consumer_frame_ms']]),
                stage_means_ms={k:sum(trials[n]['stage_means_ms'][k] for n in names)/len(names) for k in trials[names[0]]['stage_means_ms']})
        performance[clip]=dict(arms=arms,speedup=arms['candidate']['fps']/arms['reference']['fps'],
            paired_speedups=[trials[f'{clip}_repeat{i}_candidate']['fps']/trials[f'{clip}_repeat{i}_reference']['fps'] for i in range(3)])
    combined={m:768/sum(r['arms'][m]['wall_s'] for r in performance.values()) for m in ('reference','candidate')}
    full=[r for r in trials.values() if r['frames'] is None];count=sum(r['count'] for r in full);wall=sum(r['wall_s'] for r in full)
    kernel={scene:{mode:dict(event_mean_ms=sum(r['isolated_kernel_event_ms'] for r in gate['timings'] if r['scene']==scene and r['mode']==mode)/4,
        wall_with_copies_mean_ms=sum(r['production_with_transfers_ms'] for r in gate['timings'] if r['scene']==scene and r['mode']==mode)/4)
        for mode in ('reference','candidate')} for scene in ('random','flat','guarded')}
    diagnostics=verify_diagnostics(evidence,transition)
    consistent=all(all(v>1 for v in r['paired_speedups']) for r in performance.values())
    return dict(schema='seaqr.visible-speed-v19.v1',verified=True,raw16_paused=True,
        exhaustive_rank_cases=1<<25,generated_gpu_cases=259,cpu_oracle_cases=sum(r['cpu_oracle'] for r in gate['cases']),
        generated_kernel_timings=kernel,actual_input_diagnostics=diagnostics,
        prefix_performance=performance,combined_prefix_fps=combined,
        combined_prefix_speedup=combined['candidate']/combined['reference'],consistent_pipeline_gain=consistent,
        decision='opt_in_exact_median_candidate' if consistent else 'retain_v17_mixed_pipeline_timings',
        trials=list(trials.values()),full_regression_frames=count,full_candidate_fps=count/wall,
        full_candidate_wall_s=wall,full_stage_means_ms={k:sum(r['stage_means_ms'][k]*r['count'] for r in full)/count for k in full[0]['stage_means_ms']},
        full_candidate_service=distribution([v for r in full for v in stored[r['name']]['consumer_frame_ms']]),
        exact_non_timing_outputs=True,defaults_changed=False,new_airborne_accuracy_validated=False,
        summarizer_sha256=sha(__file__),warning='Only prefix timings are fresh paired comparisons. '
        'Full candidates compare archived outputs, not fresh paired full timing. Generated kernel timings exclude pipeline scheduling. '
        'No new airborne labels, live latency or physical sensor cadence validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.evidence);write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
