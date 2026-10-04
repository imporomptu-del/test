"""Single-worker native hypothesis-scoring, parity and speed experiment."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from run_visible_native_v25 import FROZEN, sha, read, write, verify_freeze

HERE = Path(__file__).resolve().parent
V24 = Path('/tmp/seaqr_visible_stage_v24_Uo3Tze')


def spec(name, clip, mode, frames=128, attributed=False):
    return dict(name=name,clip=clip,mode=mode,frames=frames,attributed=attributed)


def initial_schedule():
    return [spec(f'smoke_native_staged_{c}',c,'native_staged') for c in ('0126','0082')] + [
        spec(f'{c}_repeat{i}_{m}',c,m) for c in ('0126','0082') for i in range(3)
        for m in (('reference','native_staged') if i%2 == 0 else ('native_staged','reference'))]


def speed_gate(directory):
    clips = {}
    for clip in ('0126','0082'):
        trials = {m:[dict(read(directory/f'{clip}_repeat{i}_{m}.v25.json'),name=f'{clip}_repeat{i}_{m}') for i in range(3)]
                  for m in ('reference','native_staged')}
        latency = {m:[[float(r['consumer_complete_ns']-r['ready_ns'])/1e6 for r in read(directory/(t['name']+'.v24.json'))['execution']['frames']]
                      for t in trials[m]] for m in trials}
        request_latency = {m:[[(r['consumer_complete_ns']-r['request_ns'])/1e6 for r in read(directory/(t['name']+'.v24.json'))['execution']['frames']]
                              for t in trials[m]] for m in trials}
        cadence = {m:[read(directory/f'{clip}_repeat{i}_{m}.v17.json')['consumer_frame_ms']
                      for i in range(3)] for m in trials}
        pooled_fps = {m:384/sum(t['wall_s'] for t in trials[m]) for m in trials}
        paired = [trials['native_staged'][i]['fps']/trials['reference'][i]['fps'] for i in range(3)]
        regressions = {}
        for label,measure in [('queue_aware',latency),('consumer_cadence',cadence),('request_to_complete',request_latency)]:
            pooled = {m:float(np.percentile([x for row in measure[m] for x in row],95)) for m in trials}
            ratios = [float(np.percentile(measure['native_staged'][i],95)/np.percentile(measure['reference'][i],95))
                      for i in range(3)]
            regressions[label] = dict(pooled_p95_ms=pooled,paired_p95_ratios=ratios,
                consistent_regression=pooled['native_staged']>pooled['reference'] and sum(x>1 for x in ratios)>=2)
        speedup = pooled_fps['native_staged']/pooled_fps['reference']
        clips[clip] = dict(pooled_fps=pooled_fps,speedup=speedup,paired_speedups=paired,
            latency=regressions,passed=bool(speedup>=1.2 and all(x>1 for x in paired)
                and not any(v['consistent_regression'] for v in regressions.values())))
    return dict(passed=all(v['passed'] for v in clips.values()),clips=clips)


def main():
    verify_freeze()
    if (HERE/'batch.json').exists() or (HERE/'smoke_native_staged_0126.log').exists():
        raise FileExistsError('Fresh one-worker batch only; preserve partial receipts')
    record = dict(passed=False,error=None,rows=[],schedule=initial_schedule(),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256={n:sha(HERE/n) for n in FROZEN},raw16_accessed=False,
                  defaults_changed=False,performance=None,full_regression_run=False)
    def execute(s):
        prefix = HERE/s['name']
        command = [sys.executable,str(HERE/'run_visible_native_v25.py'),'--clip',s['clip'],
                   '--mode',s['mode'],'--output',str(prefix)]
        if s['frames'] is not None:
            command += ['--frames',str(s['frames'])]
        if s['attributed']:
            command += ['--attribute']
        print(json.dumps(dict(event='start',**s)),flush=True)
        start=time.monotonic();log=s['name']+'.log'
        with (HERE/log).open('x') as handle:
            done=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,cwd=HERE)
        row=dict(**s,command=command,returncode=done.returncode,log=log,log_sha256=sha(HERE/log),
                 process_wall_s=time.monotonic()-start)
        record['rows'].append(row)
        if done.returncode:
            print(json.dumps(row),flush=True)
            raise RuntimeError('Experiment command failed: '+s['name'])
        result=read(prefix.with_suffix('.v25.json'))
        if not result['passed'] or result['error'] is not None:
            raise AssertionError('Execution/parity receipt failed')
        row.update(v25_sha256=sha(prefix.with_suffix('.v25.json')),fps=result['fps'],wall_s=result['wall_s'])
        print(json.dumps(row),flush=True)
    try:
        for s in initial_schedule():
            execute(s)
        record['performance']=speed_gate(HERE)
        print(json.dumps(dict(event='speed_gate',**record['performance'])),flush=True)
        if record['performance']['passed']:
            full=[spec('full_'+c+'_native_staged',c,'native_staged',None) for c in ('0029','0126','0055','0082')]
            record['schedule'] += full
            for s in full:
                execute(s)
            record['full_regression_run']=True
        traces=[spec('attribute_'+c+'_native_staged',c,'native_staged',attributed=True) for c in ('0126','0082')]
        record['schedule'] += traces
        for s in traces:
            execute(s)
        record['passed']=True
        record['decision']='candidate_requires_independent_audit' if record['performance']['passed'] else 'reject_v25_keep_v20'
    except BaseException as exc:
        record['error']=repr(exc)
        raise
    finally:
        record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(HERE/'batch.json',record)


if __name__=='__main__':
    main()

