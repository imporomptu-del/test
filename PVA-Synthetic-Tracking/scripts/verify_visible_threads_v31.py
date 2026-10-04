"""Report-only independent six-arm/thread-policy, parity and timing audit."""
import argparse
import math
from pathlib import Path
from profile_visible_interaction_v30 import read,sha,write
from analyze_visible_interaction_v30 import require,analyze
from verify_visible_combined_v29 import verify_sources,transformed_hash,performance,ROOT
from verify_visible_interaction_v30 import V29_LOCAL,audit_relocated_trial

V30_LOCAL=ROOT/'results/tiny_target/visible_interaction_v30_20260920/evidence'
REMOTE=Path('/tmp/seaqr_visible_threads_v31_QvwNyq')
MODES=('v20','v26_default','v26','v28','combined_default','combined')


def specs():
    initial=[dict(name='smoke_'+c,clip=c,mode='combined',frames=128,audit=True,traced=False) for c in ('0126','0082')]
    orders=(MODES,tuple(reversed(MODES)),('v28','combined','v20','combined_default','v26','v26_default'))
    initial += [dict(name=f'{c}_repeat{i}_{m}',clip=c,mode=m,frames=128,audit=False,traced=False)
        for c in ('0126','0082') for i,order in enumerate(orders) for m in order]
    initial += [dict(name=f'trace_{c}_{m}',clip=c,mode=m,frames=128,audit=False,traced=True)
        for c,modes in (('0126',('combined_default','combined')),('0082',('combined','combined_default'))) for m in modes]
    full=[dict(name='full_'+c,clip=c,mode='combined',frames=None,audit=False,traced=False) for c in ('0029','0126','0055','0082')]
    return initial,full


def independent_gate(trials,samples):
    expected={s['name']:s for s in specs()[0] if not s['audit'] and not s['traced']}
    require(set(trials)==set(expected),'Complete independent six-arm schedule required')
    for n,r in trials.items():
        s=expected[n]
        require(r['mode']==s['mode'] and r['clip']==s['clip'] and r['arm']==s['mode'].removesuffix('_default')
            and r['count']==r['frames']==128 and not r['state_audit'] and not r.get('traced',False),'Wrong clean sample scope')
        require(type(r['wall_s']) in (int,float) and math.isfinite(r['wall_s']) and r['wall_s']>0
            and math.isclose(r['fps'],128/r['wall_s'],rel_tol=1e-12),'Invalid independent timing')
    main={n:r for n,r in trials.items() if '_repeat' in n and r['mode'] in ('v20','v26','v28','combined')}
    original=performance(main,samples)
    extra={}
    for c in ('0126','0082'):
        rows={m:[trials[f'{c}_repeat{i}_{m}'] for i in range(3)] for m in MODES}
        fps={m:sum(r['count'] for r in v)/sum(r['wall_s'] for r in v) for m,v in rows.items()}
        extra[c]=dict(pooled_fps=fps,candidate_not_worse_than_default_controls=fps['combined']>=max(fps['v26_default'],fps['combined_default']),
            thread_only_combined_speedup=fps['combined']/fps['combined_default'],thread_only_gpu_speedup=fps['v26']/fps['v26_default'])
    return dict(passed=original['passed'] and all(v['candidate_not_worse_than_default_controls'] for v in extra.values()),
                original_gate=original,additional_controls=extra)


def verify(evidence):
    manifest=read(evidence/'export_manifest.json')
    require(manifest['post_run'] and not manifest['media_included'],'Invalid export manifest')
    for n,h in manifest['files'].items():
        require(not Path(n).is_absolute() and '..' not in Path(n).parts,'Unsafe export pathname')
        require(sha(evidence/n)==h,'Changed exported artifact '+n)
    frozen,generated,geometry=verify_sources(V29_LOCAL)
    tracking_hash=transformed_hash()
    f=read(evidence/'freeze.json'); batch=read(evidence/'batch.json')
    require(f['pre_run'] and batch['freeze_sha256']==sha(evidence/'freeze.json')
        and f['baseline_freeze_sha256']==sha(V29_LOCAL/'freeze.json')
        and f['diagnostic_batch_sha256']==sha(V30_LOCAL/'batch.json'),'Changed frozen selection/dependency evidence')
    source_names={'visible_threads_v31.py','run_visible_threads_v31.py','batch_visible_threads_v31.py',
        'test_visible_threads_v31.py','visible_threads_v31_plan.md','profile_visible_interaction_v30.py'}
    require(set(f['sources'])==source_names,'Incomplete source freeze')
    for n,h in f['sources'].items():
        sub='tests/unit' if n.startswith('test_') else 'docs' if n.endswith('.md') else 'scripts'
        require(sha(evidence/n)==h==sha(ROOT/sub/n),'Frozen source changed '+n)
    require(f['unit_log_sha256']==sha(evidence/'unit.log') and '\nOK\n' in (evidence/'unit.log').read_text(),'Harness tests failed')
    new_generated=read(evidence/'generated.json')
    archived=ROOT/'results/tiny_target/tracking_v28_20260920/evidence/replays_01.json'
    require(new_generated['passed'] and not new_generated['media_read']
        and new_generated['actual']==read(archived)['generated']
        and new_generated['expected_sha256']==sha(archived)
        and new_generated['freeze_sha256']==sha(evidence/'freeze.json')
        and batch['generated_sha256']==sha(evidence/'generated.json'),'Generated private-state gate failed')
    reference_runtime=read(V30_LOCAL/'trace_0_0082_v26.trace30.json')['runtime_before']
    expected_blas=reference_runtime['blas'][0]
    def runtime(info,one):
        require(len(info['blas'])==1,'Changed numerical library count')
        require(info['blas'][0]==dict(expected_blas,threads=1 if one else 12),'Numerical library/policy changed')
        expected_env={k:None for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','GOTO_NUM_THREADS')}
        if one:expected_env['OPENBLAS_NUM_THREADS']='1'
        require(info['thread_environment']==expected_env and info['affinity']==reference_runtime['affinity']
            and info['numpy']==reference_runtime['numpy'] and info['opencv']==reference_runtime['opencv'], 'Unexpected runtime environment')
    runtime(new_generated['runtime'],True)
    initial,full=specs(); schedule=batch['schedule']
    require(batch['passed'] and batch['error'] is None and not read(evidence/'status.json')['running']
        and schedule in (initial,initial+full) and len(batch['rows'])==len(schedule),'Incomplete experiment/schedule')
    trials,samples,traces={},{},{}
    for s,row in zip(schedule,batch['rows']):
        require(all(row[k]==v for k,v in s.items()),'Changed trial ordering')
        n=s['name'];r=read(evidence/(n+'.v31.json'))
        arm=s['mode'].removesuffix('_default'); one=s['mode']!='v20' and not s['mode'].endswith('_default')
        require(r['passed'] and r['error'] is None and all(r[k]==s[k] for k in ('clip','mode','frames','audit','traced'))
            and r['arm']==arm and r['thread_policy']==row['thread_policy']==('one' if one else 'inherited')
            and not r['raw16_accessed'] and not r['defaults_changed'] and not r['algorithm_changed'],'Invalid candidate receipt/scope')
        require(r['freeze_sha256']==sha(evidence/'freeze.json') and row['v31_sha256']==sha(evidence/(n+'.v31.json'))
            and row['receipt_sha256']==r['receipt_sha256'] and row['log_sha256']==sha(evidence/(n+'.log')),'Changed candidate provenance')
        runtime(r['runtime_before'],one);runtime(r['runtime_after'],one)
        require(r['runtime_after']['opencv_threads']==2,'Changed OpenCV threading policy')
        command=['/usr/bin/python3',str(REMOTE/'run_visible_threads_v31.py'),'--clip',s['clip'],'--mode',s['mode'],'--output',str(REMOTE/n)]
        if s['frames'] is not None:command+=['--frames',str(s['frames'])]
        if s['audit']:command+=['--audit']
        if s['traced']:
            command+=['--traced']
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=process-tree',
                '--osrt-threshold=100000','--cuda-flush-interval=0','--kill=none','--force-overwrite=false',
                '--export=sqlite','--output='+str(REMOTE/n)]+command
        require(row['command']==command,'Wrong experiment command')
        trial_spec=dict(name=n,clip=s['clip'],arm=arm,frames=s['frames'],state_audit=s['audit'])
        trial,values=audit_relocated_trial(evidence,trial_spec,row,frozen,generated,geometry,tracking_hash)
        require(trial['count']==r['count'] and trial['fps']==r['fps'] and trial['wall_s']==r['wall_s'],'Changed timing denominator')
        trial.update(mode=s['mode'],traced=s['traced'],thread_policy=r['thread_policy'])
        trials[n],samples[n]=trial,values
        if s['traced']:
            t=read(evidence/(n+'.trace30.json'))
            require(row['trace_sha256']==sha(evidence/(n+'.trace30.json')) and row['sqlite_sha256']==sha(evidence/(n+'.sqlite'))
                and t['script_sha256']==f['sources']['profile_visible_interaction_v30.py']
                and t['receipt_sha256']==r['receipt_sha256'] and t['arm']==arm and t['clip']==s['clip'],'Trace provenance changed')
            runtime(t['runtime_before'],one);runtime(t['runtime_after'],one)
            traces[n]=analyze(evidence/(n+'.sqlite'),evidence/(n+'.trace30.json'))
    timing_names={s['name'] for s in initial if not s['audit'] and not s['traced']}
    g=independent_gate({n:trials[n] for n in timing_names},{n:samples[n] for n in timing_names})
    require(g==batch['performance'],'On-device and independently recomputed gates differ')
    require(batch['full_regression_run']==g['passed']==(schedule==initial+full),'Full-regression gate violated')
    return dict(verified=True,completed=True,performance=g,trials=trials,traces=traces,
        full_regression_complete=batch['full_regression_run'],generated_scenarios=len(new_generated['actual']['scenarios']),
        frame_instances=sum(t['count'] for t in trials.values()),unique_development_frames=2741 if batch['full_regression_run'] else 256,
        raw16_accessed=False,defaults_changed=False,production_approved=False,
        verifier_sha256=sha(__file__),analyzer_sha256=sha(ROOT/'scripts/analyze_visible_interaction_v30.py'),
        manifest_sha256=sha(evidence/'export_manifest.json'),batch_sha256=sha(evidence/'batch.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();write(a.output,verify(a.evidence))
