"""Independent saved-evidence verification; never opens any source media."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_visible_v17 import read,sha,write
from batch_visible_v18 import schedule
from verify_visible_v17 import require,distribution,ARCHIVE
from compare_phase20_exact_runs import validate_decode,shape_accelerator
from repeat_phase20_kernel_speed import check_prefix

V17=ROOT/'results/tiny_target/visible_speed_v17_20260917/evidence'


def verify(evidence):
    gate=read(evidence/'generated_01.json')
    require(gate['passed'] and gate['error'] is None and not gate['real_media_read'],'Generated gate failed')
    require(len(gate['cases'])==354 and len(gate['timings'])==8,'Incomplete generated schedule')
    require(all(r['exact'] and r['inputs_unchanged'] for r in gate['cases']),'Numerical mismatch')
    require({r['native'] for r in gate['cases']}=={True,False},'Missing native/fallback coverage')
    require(gate['reference_sha256']==sha(ROOT/'tiny_target/visible_noise.py'),'Noise reference changed')
    require(gate['plan_sha256']==sha(ROOT/'docs/visible_speed_v18_plan.md')==sha(evidence/'visible_speed_v18_plan.md'),'Plan changed')
    require(sha(evidence/'profile_visible_v17.py')==sha(ROOT/'scripts/profile_visible_v17.py'),'Shared harness utilities changed')
    require(set(gate['source_sha256'])=={'noise_v18.py','noise_v18.cpp','build_noise_v18.py','check_noise_v18.py'},
            'Incomplete generated source provenance')
    for name,digest in gate['source_sha256'].items():
        require(sha(ROOT/'scripts'/name)==digest==sha(evidence/name),'Noise source changed: '+name)
    build=read(evidence/'build/build.json')
    require(build['returncode']==0 and build['source_sha256']==sha(ROOT/'scripts/noise_v18.cpp')
            and build['builder_sha256']==sha(ROOT/'scripts/build_noise_v18.py')
            and build['library_sha256']==gate['library_sha256']==sha(evidence/'build/libnoise_v18.so'),'Build mismatch')
    old_gate=read(V17/'generated_01.json')
    batch=read(evidence/'trials_01/batch.json')
    require(batch['passed'] and batch['error'] is None and batch['schedule']==schedule()
            and len(batch['rows'])==12 and batch['script_sha256']==sha(ROOT/'scripts/batch_visible_v18.py'),'Batch incomplete')
    require(batch['script_sha256']==sha(evidence/'batch_visible_v18.py'),'Batch source receipt mismatch')
    trials,stored={},{}
    for spec,row in zip(schedule(),batch['rows']):
        require(all(row[k]==v for k,v in spec.items()),'Schedule changed')
        name=spec['name'];path=evidence/'trials_01'/name
        r=read(path.with_suffix('.v18.json')); m=read(path.with_suffix('.v17.json'))
        require(r['passed'] and r['error'] is None and sha(path.with_suffix('.v18.json'))==row['sha256'],'Trial failed')
        require(all(r[k]==spec[k] for k in ('clip','mode','frames')),'Trial scope changed')
        require(r['script_sha256']==sha(ROOT/'scripts/run_visible_v18.py')==sha(evidence/'run_visible_v18.py')
                and r['plan_sha256']==gate['plan_sha256'] and r['gate_sha256']==sha(evidence/'generated_01.json')
                and r['library_sha256']==gate['library_sha256'] and not r['raw16_accessed']
                and not r['defaults_changed'],'Trial provenance failed')
        require(m['passed'] and m['error'] is None and m['mode']=='candidate'
                and m['clip']==spec['clip'] and m['frames']==spec['frames']
                and r['v17_receipt_sha256']==sha(path.with_suffix('.v17.json')),'Baseline helper receipt failed')
        require(m['script_sha256']==r['v17_wrapper_sha256']==sha(ROOT/'scripts/run_visible_v17.py')
                and m['gate_sha256']==sha(V17/'generated_01.json')
                and m['library_sha256']==old_gate['library_sha256']
                and m['candidate_plan_sha256']==sha(ROOT/'docs/visible_speed_v17_candidate.md')
                and not m['raw16_accessed'] and not m['defaults_changed']
                and not m['production_approved'] and m['native_mask_calls']>0,'v17 baseline changed')
        reference=ARCHIVE/f'visible_{spec["clip"]}_full_reuse'
        report,launch=read(path/'report.json'),read(path/'launch.json')
        old_report,old_launch=read(reference/'report.json'),read(reference/'launch.json')
        count=spec['frames'] or old_report['frames']
        require(report['completed'] and report['frames']==count==r['processed_frames']==m['processed_frames']
                and report['full_clip']==(spec['frames'] is None),'Incomplete clip')
        require(report['configuration']==old_report['configuration']
                and report['faint_target_synthetic_branch_enabled'] is False,'Branch or detector policy changed')
        for key in ('source_sha256','fps','configuration','package_sha256','motion_config_sha256',
                    'external_accelerators','exact_cuda_stabilization'):
            require(launch[key]==old_launch[key],'Launch identity changed: '+key)
        validate_decode(launch,report);shape_accelerator(launch);check_prefix(reference,path,count)
        require(sha(path/'frames.jsonl')==m['journal_sha256']
                and sha(path.with_suffix('.execution.json'))==m['execution_sha256'],'Journal hash mismatch')
        motion=read(path.with_suffix('.execution.json'));old_motion=read(reference.with_suffix('.execution.json'))
        for key in ('adapter_sha256','method_sha256','wrapper_sha256','runtime_sha256'):
            require(motion[key]==old_motion[key],'Motion provenance changed: '+key)
        require(motion['passed'] and motion['error'] is None and motion['closed']
                and motion['branch']=='visible' and motion['mode']=='reuse' and motion['clip']==spec['clip']
                and motion['processed_frames']==count and motion['reuse_hits']==count-2
                and motion['reuse_misses']==1,'Motion lifecycle failed')
        require([(v['frame'],v['identity']) for v in motion['motion']]==
                [(v['frame'],v['identity']) for v in old_motion['motion'][:count-1]],'Motion changed')
        require(len(r['noise_identities'])==len(r['noise_ms'])==len(m['consumer_frame_ms'])==count,'Timing count mismatch')
        require(r['fps']==m['fps']==row['fps']==report['processed_fps']
                and r['wall_s']==m['wall_s']==row['wall_s']==report['elapsed_seconds'],'Timing receipt mismatch')
        if spec['mode']=='reference':
            require(r['native_noise_calls']==r['fallback_calls']==r['geometry_builds']==0,'Reference used candidate')
        else:
            require(r['native_noise_calls']>0 and r['native_noise_calls']+r['fallback_calls']==count
                    and r['geometry_builds']>0,'Candidate not fully accounted')
        if spec['frames'] is None:
            for key in ('counts','qualified_tracks','qualified_track_count','availability','detection_status'):
                require(report[key]==old_report[key],'Aggregate changed: '+key)
        rss=[int(v['rss']['VmRSS'].split()[0]) for v in motion['motion'] if v.get('rss')]
        require(bool(rss),'Missing process RSS samples')
        trials[name]=dict(**spec,count=count,fps=r['fps'],wall_s=r['wall_s'],exact=True,
            native_noise_calls=r['native_noise_calls'],fallback_calls=r['fallback_calls'],
            geometry_builds=r['geometry_builds'],noise=distribution(r['noise_ms']),
            rss_sample_min_kib=min(rss),rss_sample_max_kib=max(rss),rss_first_kib=rss[0],rss_last_kib=rss[-1],
            consumer_service=distribution(m['consumer_frame_ms']),
            stage_means_ms={k:v['mean'] for k,v in report['timings_ms'].items()})
        stored[name]=(r,m)
    performance={}
    for clip in ('0126','0082'):
        expected=stored[f'{clip}_repeat0_reference'][0]['noise_identities']
        for repeat in range(2):
            for mode in ('reference','candidate'):
                require(stored[f'{clip}_repeat{repeat}_{mode}'][0]['noise_identities']==expected,'Per-tile noise outputs changed')
        require(stored[f'{clip}_full_candidate'][0]['noise_identities'][:128]==expected,'Full-run noise prefix changed')
        arms={}
        for mode in ('reference','candidate'):
            names=[n for n,r in trials.items() if r['clip']==clip and r['mode']==mode and r['frames']==128]
            wall=sum(trials[n]['wall_s'] for n in names)
            arms[mode]=dict(fps=256/wall,wall_s=wall,frames=256,
                noise=distribution([v for n in names for v in stored[n][0]['noise_ms']]),
                consumer_service=distribution([v for n in names for v in stored[n][1]['consumer_frame_ms']]),
                stage_means_ms={k:sum(trials[n]['stage_means_ms'][k] for n in names)/len(names)
                                for k in trials[names[0]]['stage_means_ms']})
        performance[clip]=dict(arms=arms,speedup=arms['candidate']['fps']/arms['reference']['fps'],
            paired_speedups=[trials[f'{clip}_repeat{i}_candidate']['fps']/trials[f'{clip}_repeat{i}_reference']['fps'] for i in range(2)])
    full=[r for r in trials.values() if r['frames'] is None]
    count=sum(r['count'] for r in full);wall=sum(r['wall_s'] for r in full)
    stages={k:sum(r['stage_means_ms'][k]*r['count'] for r in full)/count for k in full[0]['stage_means_ms']}
    paired_fps={mode:512/sum(performance[c]['arms'][mode]['wall_s'] for c in performance)
                for mode in ('reference','candidate')}
    consistent=all(v['speedup']>1 and all(s>1 for s in v['paired_speedups']) for v in performance.values())
    return dict(schema='seaqr.visible-speed-v18.v1',verified=True,raw16_paused=True,
        generated_cases=354,prefix_performance=performance,trials=list(trials.values()),
        combined_prefix_fps=paired_fps,combined_prefix_speedup=paired_fps['candidate']/paired_fps['reference'],
        consistent_pipeline_gain=consistent,
        decision='opt_in_candidate_only' if consistent else 'retain_v17_baseline_mixed_pipeline_timings',
        full_regression_frames=count,full_candidate_fps=count/wall,full_candidate_wall_s=wall,
        full_candidate_service=distribution([v for r in full for v in stored[r['name']][1]['consumer_frame_ms']]),
        full_stage_means_ms=stages,exact_non_timing_outputs=True,exact_prefix_noise_outputs=True,
        defaults_changed=False,new_airborne_accuracy_validated=False,summarizer_sha256=sha(__file__),
        warning='Only prefix timings are fresh paired comparisons. Full candidate compares archived outputs, not paired full timing. '
                'Four existing development clips are not independent accuracy validation. Consumer service is not camera-to-alert latency.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.evidence);write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
