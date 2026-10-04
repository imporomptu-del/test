"""Independent report-only audit of v20; never opens media or holdout manifests."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_visible_v17 import read,sha,write
from verify_visible_v17 import require,distribution,ARCHIVE
from video_checks_v19 import check
from batch_visible_v20 import schedule
from run_visible_v20 import FROZEN
from tracking_geometry_v20 import REFERENCE_SHA,OLD,NEW
from check_tracking_geometry_v20 import primitive_cases
from tiny_target.tracking.kalman import KalmanTrackManager
V17=ROOT/'results/tiny_target/visible_speed_v17_20260917/evidence'


def baseline(path,clip,frames):
    r=read(path.with_suffix('.v17.json'));reference=ARCHIVE/f'visible_{clip}_full_reuse'
    left=read(reference/'launch.json');report=read(path/'report.json')
    count=frames or read(reference/'report.json')['frames']
    require(r['passed'] and r['error'] is None and r['clip']==clip and r['mode']=='candidate'
            and r['frames']==frames and r['processed_frames']==count and not r['raw16_accessed']
            and not r['defaults_changed'] and not r['production_approved'],'Original baseline receipt failed')
    require(r['script_sha256']==sha(ROOT/'scripts/run_visible_v17.py')==sha(V17/'run_visible_v17.py')
            and r['candidate_plan_sha256']==sha(V17/'visible_speed_v17_candidate.md')
            and r['gate_sha256']==sha(V17/'generated_01.json')
            and r['library_sha256']==sha(V17/'build/liblearning_mask_v17.so')
            and r['native_mask_calls']>0,'v17 identity missing')
    # No configuration/GPU transition is permitted in this experiment.
    result=check(reference,path,count,frames is None,'reference',left['configuration'],left['config_sha256'],
        dict(before_library_sha256=left['exact_cuda_stabilization']['library_sha256']))
    require(r['journal_sha256']==result['journal_sha256'] and r['execution_sha256']==result['execution_sha256']
            and r['fps']==report['processed_fps'] and r['wall_s']==report['elapsed_seconds']
            and len(r['consumer_frame_ms'])==count,'Journal or timing receipt changed')
    motion=read(path.with_suffix('.execution.json'))
    require(motion['clip']==clip and motion['frames']==frames and not motion['injected'],'Motion request changed')
    return r,report,motion


def verify(evidence):
    g=read(evidence/'generated_01.json');b=read(evidence/'build/build.json');f=read(evidence/'freeze.json')
    require(g['passed'] and g['error'] is None and not g['real_media_read']
            and g['reference_sha256']==REFERENCE_SHA==sha(ROOT/'tiny_target/tracking/kalman.py'),'Generated baseline mismatch')
    names=[r[0] for r in primitive_cases()]
    require([c['name'] for c in g['cases']]==names and len(names)==136
            and all(c['exact'] and c['inputs_unchanged'] for c in g['cases'])
            and {c['native'] for c in g['cases']}=={True,False},'Generated geometry schedule incomplete')
    require([r['scenario'] for r in g['replays']]==list(range(12))
            and all(r['exact'] and r['frames']==32 for r in g['replays']),'Incomplete tracker replay')
    require(set(g['source_sha256'])=={'tracking_geometry_v20.cpp','tracking_geometry_v20.py',
            'build_tracking_geometry_v20.py','check_tracking_geometry_v20.py'},'Missing generated provenance')
    for n,d in g['source_sha256'].items():require(d==sha(ROOT/'scripts'/n)==sha(evidence/n),'Changed source '+n)
    require(b['returncode']==0 and b['source_sha256']==g['source_sha256']['tracking_geometry_v20.cpp']
            and b['builder_sha256']==g['source_sha256']['build_tracking_geometry_v20.py']
            and b['library_sha256']==g['library_sha256']==sha(evidence/'build/libtracking_geometry_v20.so'),'Build mismatch')
    require('-fno-fast-math' in b['command'] and '-ffp-contract=off' in b['command'],'Strict arithmetic flags missing')
    source=inspect.getsource(KalmanTrackManager.update)
    require(source.count(OLD)==1,'Reference transform no longer exact')
    transformed=hashlib.sha256(textwrap.dedent(source.replace(OLD,NEW)).encode()).hexdigest()
    require(g['transformed_sha256']==transformed,'Unexpected tracker method change')
    require(set(f['files'])==set(FROZEN) and f['gate_sha256']==sha(evidence/'generated_01.json')
            and f['build_sha256']==sha(evidence/'build/build.json'),'Incomplete freeze')
    for n,d in f['files'].items():
        local=ROOT/('docs' if n.endswith('.md') else 'scripts')/n
        require(d==sha(local)==sha(evidence/n),'Frozen input changed '+n)
    require([(r['repeat'],r['mode']) for r in g['timings']]==[(i,m) for i in range(4)
            for m in (('reference','candidate') if i%2==0 else ('candidate','reference'))]
            and all(r['iterations']==1000 and 0<r['mean_us']<float('inf') for r in g['timings']),'Generated timing schedule changed')
    profiles={}
    for clip in ('0126','0082'):
        path=evidence/f'profile_{clip}';p=read(path.with_suffix('.profile.json'))
        baseline(path,clip,128)
        require(p['passed'] and p['error'] is None and p['clip']==clip and p['frames']==128
                and not p['raw16_accessed'] and not p['defaults_changed']
                and p['baseline_receipt_sha256']==sha(path.with_suffix('.v17.json'))
                and p['script_sha256']==sha(evidence/'profile_visible_v20.py')==sha(ROOT/'scripts/profile_visible_v20.py')
                and p['plan_sha256']==sha(evidence/'visible_speed_v20_plan.md'),'Profile mismatch')
        require(len(p['tracker_populations'])==256 and p['calls']['tracks.total']['calls']==128,'Profile schedule incomplete')
        profiles[clip]=dict(calls={k:dict(calls=v['calls'],mean_ms=v['mean_ms']) for k,v in p['calls'].items()},
            total_prior_tracks=sum(r['tracks'] for r in p['tracker_populations']),top_calls=p['python_profile'][:15],
            instrumented_not_pipeline_fps=True)
    batch=read(evidence/'trials_01/batch.json')
    require(batch['passed'] and batch['error'] is None and batch['schedule']==schedule()
            and len(batch['rows'])==16 and batch['script_sha256']==sha(evidence/'batch_visible_v20.py'),'Incomplete timed batch')
    trials,stored={},{}
    for spec,row in zip(schedule(),batch['rows']):
        require(all(row[k]==v for k,v in spec.items()),'Trial order changed')
        name=spec['name'];path=evidence/'trials_01'/name;r=read(path.with_suffix('.v20.json'))
        v17,report,motion=baseline(path,spec['clip'],spec['frames'])
        require(r['passed'] and r['error'] is None and r['mode']==spec['mode']
                and r['clip']==spec['clip'] and r['frames']==spec['frames']
                and not any(r[k] for k in ('raw16_accessed','defaults_changed','gpu_changed','noise_v18_enabled','median_v19_enabled')),'Trial scope changed')
        require(row['sha256']==sha(path.with_suffix('.v20.json'))
                and r['baseline_receipt_sha256']==sha(path.with_suffix('.v17.json'))
                and r['script_sha256']==sha(evidence/'run_visible_v20.py') and r['freeze_sha256']==sha(evidence/'freeze.json')
                and r['gate_sha256']==sha(evidence/'generated_01.json') and r['library_sha256']==g['library_sha256']
                and r['transformed_sha256']==transformed,'Trial provenance changed')
        require((r['native_geometry_calls']>0)==(spec['mode']=='candidate') and r['geometry_fallbacks']>=0,'Candidate not exercised')
        require(all(r[k]==row[k]==v17[k] for k in ('fps','wall_s'))
                and r['processed_frames']==v17['processed_frames'],'Timing receipt changed')
        rss=[int(v['rss']['VmRSS'].split()[0]) for v in motion['motion'] if v.get('rss')]
        require(bool(rss),'Missing memory samples')
        trials[name]=dict(**spec,count=r['processed_frames'],fps=r['fps'],wall_s=r['wall_s'],exact=True,
            geometry_calls=r['native_geometry_calls'],geometry_fallbacks=r['geometry_fallbacks'],
            service=distribution(v17['consumer_frame_ms']),stage_means_ms={k:v['mean'] for k,v in report['timings_ms'].items()},
            rss_min_kib=min(rss),rss_max_kib=max(rss),rss_first_kib=rss[0],rss_last_kib=rss[-1])
        stored[name]=v17
    performance={}
    for clip in ('0126','0082'):
        arms={}
        for mode in ('reference','candidate'):
            selected=[r for r in trials.values() if r['clip']==clip and r['mode']==mode and r['frames']==128]
            wall=sum(r['wall_s'] for r in selected)
            arms[mode]=dict(frames=384,wall_s=wall,fps=384/wall,
                service=distribution([v for r in selected for v in stored[r['name']]['consumer_frame_ms']]),
                stage_means_ms={k:sum(r['stage_means_ms'][k] for r in selected)/3 for k in selected[0]['stage_means_ms']})
        performance[clip]=dict(arms=arms,speedup=arms['candidate']['fps']/arms['reference']['fps'],
            paired_speedups=[trials[f'{clip}_repeat{i}_candidate']['fps']/trials[f'{clip}_repeat{i}_reference']['fps'] for i in range(3)])
    combined={m:768/sum(v['arms'][m]['wall_s'] for v in performance.values()) for m in ('reference','candidate')}
    full=[r for r in trials.values() if r['frames'] is None];n=sum(r['count'] for r in full);wall=sum(r['wall_s'] for r in full)
    consistent=all(v>1 for p in performance.values() for v in p['paired_speedups'])
    return dict(schema='seaqr.visible-speed-v20.v1',verified=True,raw16_paused=True,defaults_changed=False,
        generated_cases=len(g['cases']),generated_tracker_frames=384,profiles=profiles,
        isolated_geometry_mean_us={m:sum(r['mean_us'] for r in g['timings'] if r['mode']==m)/4 for m in ('reference','candidate')},
        prefix_performance=performance,combined_prefix_fps=combined,combined_speedup=combined['candidate']/combined['reference'],
        consistent_pipeline_gain=consistent,decision='opt_in_exact_geometry_candidate' if consistent else 'retain_v17_mixed_pipeline_timings',
        trials=list(trials.values()),full_regression_frames=n,full_candidate_fps=n/wall,
        full_stage_means_ms={k:sum(r['stage_means_ms'][k]*r['count'] for r in full)/n for k in full[0]['stage_means_ms']},
        full_service=distribution([v for r in full for v in stored[r['name']]['consumer_frame_ms']]),
        exact_non_timing_outputs=True,new_airborne_accuracy_validated=False,summarizer_sha256=sha(__file__),
        warning='Only prefixes have fresh paired timings. Profiles add overhead and native call durations include work plus waits. '
        'Full candidate FPS has no fresh full reference arm. No new airborne accuracy or real-time validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.evidence);write(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('trials','profiles')},indent=2))
