"""One-worker frozen serial reference/resident-front experiment."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from run_visible_front_v26 import FROZEN,sha,read,write,verify_freeze
HERE=Path(__file__).resolve().parent


def initial_schedule():
    def spec(n,c,m):return dict(name=n,clip=c,mode=m,frames=128)
    return [spec('smoke_candidate_'+c,c,'candidate') for c in ('0126','0082')]+[
        spec(f'{c}_repeat{i}_{m}',c,m) for c in ('0126','0082') for i in range(3)
        for m in (('reference','candidate') if i%2==0 else ('candidate','reference'))]


def speed_gate(directory):
    result={}
    for c in ('0126','0082'):
        rows={m:[read(directory/f'{c}_repeat{i}_{m}.v26.json') for i in range(3)] for m in ('reference','candidate')}
        if not all(r['passed'] and r['error'] is None and r['processed_frames']==128 for arm in rows.values() for r in arm):
            raise ValueError('Incomplete timing sample')
        fps={m:384/sum(r['wall_s'] for r in arm) for m,arm in rows.items()}
        paired=[rows['candidate'][i]['fps']/rows['reference'][i]['fps'] for i in range(3)]
        latency={}
        for label,start in (('queue_aware','ready_ns'),('consumer_cadence',None),('request_to_complete','request_ns')):
            samples={m:[r['consumer_frame_ms'] if start is None else
                        [(f['consumer_complete_ns']-f[start])/1e6 for f in r['execution']['frames']] for r in arm]
                     for m,arm in rows.items()}
            p95={m:float(np.percentile([v for row in arm for v in row],95)) for m,arm in samples.items()}
            ratios=[float(np.percentile(samples['candidate'][i],95)/np.percentile(samples['reference'][i],95)) for i in range(3)]
            latency[label]=dict(pooled_p95_ms=p95,paired_p95_ratios=ratios,
                consistent_regression=p95['candidate']>p95['reference'] and sum(v>1 for v in ratios)>=2)
        speedup=fps['candidate']/fps['reference']
        result[c]=dict(pooled_fps=fps,speedup=speedup,paired_speedups=paired,latency=latency,
            passed=speedup>=1.2 and all(v>1 for v in paired) and not any(v['consistent_regression'] for v in latency.values()))
    return dict(passed=all(v['passed'] for v in result.values()),clips=result)


def main():
    verify_freeze()
    if (HERE/'batch.json').exists() or (HERE/'smoke_candidate_0126.log').exists():raise FileExistsError('Fresh frozen batch only')
    record=dict(passed=False,error=None,rows=[],schedule=initial_schedule(),performance=None,
        full_regression_run=False,source_sha256={n:sha(HERE/n) for n in FROZEN},
        raw16_accessed=False,defaults_changed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def execute(spec):
        command=[sys.executable,str(HERE/'run_visible_front_v26.py'),'--clip',spec['clip'],'--mode',spec['mode'],
                 '--output',str(HERE/spec['name'])]
        if spec['frames'] is not None:command+=['--frames',str(spec['frames'])]
        print(json.dumps(dict(event='start',**spec)),flush=True);start=time.monotonic();log=spec['name']+'.log'
        with (HERE/log).open('x') as handle:done=subprocess.run(command,cwd=HERE,stdout=handle,stderr=subprocess.STDOUT)
        row=dict(**spec,command=command,returncode=done.returncode,process_wall_s=time.monotonic()-start,
                 log=log,log_sha256=sha(HERE/log));record['rows'].append(row)
        if done.returncode:raise RuntimeError('Experiment failed: '+spec['name'])
        r=read(HERE/(spec['name']+'.v26.json'))
        if not r['passed'] or r['error'] is not None:raise AssertionError('Output gate failed')
        row.update(v26_sha256=sha(HERE/(spec['name']+'.v26.json')),fps=r['fps'],wall_s=r['wall_s'])
        print(json.dumps(row),flush=True)
    try:
        for s in initial_schedule():execute(s)
        record['performance']=speed_gate(HERE);print(json.dumps(dict(event='speed_gate',**record['performance'])),flush=True)
        if record['performance']['passed']:
            full=[dict(name='full_'+c+'_candidate',clip=c,mode='candidate',frames=None) for c in ('0029','0126','0055','0082')]
            record['schedule']+=full
            for s in full:execute(s)
            record['full_regression_run']=True
        record['passed']=True
        record['decision']='candidate_requires_independent_audit' if record['performance']['passed'] else 'reject_front_keep_v20'
    except BaseException as exc:record['error']=repr(exc);raise
    finally:
        record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(HERE/'batch.json',record)


if __name__=='__main__':main()
