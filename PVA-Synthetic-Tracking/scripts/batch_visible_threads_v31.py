"""Frozen generated gate, six clean arms, causal traces, optional full regression."""
import os
from pathlib import Path
import subprocess
import sys
import time

from profile_visible_interaction_v30 import read, write, sha, V29
from visible_threads_v31 import schedule, traces, full, policy, environment, gate
from run_visible_threads_v31 import V30

HERE = Path(__file__).resolve().parent
SOURCES = ('visible_threads_v31.py','run_visible_threads_v31.py','batch_visible_threads_v31.py',
    'test_visible_threads_v31.py','visible_threads_v31_plan.md','profile_visible_interaction_v30.py')


def main():
    if any((HERE/n).exists() for n in ('freeze.json','status.json','batch.json','unit.log','generated.json')):
        raise FileExistsError('Fresh thread experiment directory required')
    environment(os.environ,'v20')
    sys.path.insert(0,str(V29))
    from run_visible_combined_v29 import verify_freeze
    verify_freeze()
    diagnostic=read(V30/'batch.json')
    if not diagnostic['passed'] or diagnostic['error'] is not None or len(diagnostic['rows'])!=12:
        raise ValueError('Complete successful v30 diagnosis required')
    with (HERE/'unit.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','-v','test_visible_threads_v31'],cwd=HERE,
            env=dict(os.environ,PYTHONPATH=str(V29)),stdout=log,stderr=subprocess.STDOUT,check=True)
    write(HERE/'freeze.json',dict(pre_run=True,created_ns=time.time_ns(),sources={n:sha(HERE/n) for n in SOURCES},
        unit_log_sha256=sha(HERE/'unit.log'),baseline_freeze_sha256=sha(V29/'freeze.json'),diagnostic_batch_sha256=sha(V30/'batch.json')))
    record=dict(passed=False,error=None,rows=[],schedule=schedule()+traces(),performance=None,
        freeze_sha256=sha(HERE/'freeze.json'),started_ns=time.time_ns(),full_regression_run=False,
        raw16_accessed=False,defaults_changed=False)
    def status(current=None,running=True):
        temp=HERE/'status.json.tmp'
        write(temp,dict(running=running,current=current,completed=len(record['rows']),scheduled=len(record['schedule']),
                       error=record['error'],performance=record['performance'],updated_ns=time.time_ns()))
        os.replace(temp,HERE/'status.json')
    def execute(spec):
        status(spec)
        n=spec['name']; output=HERE/n
        for suffix in ('','.v29.json','.v31.json','.log','.sqlite','.nsys-rep'):
            if (HERE/(n+suffix)).exists():raise FileExistsError(n+suffix)
        command=[sys.executable,str(HERE/'run_visible_threads_v31.py'),'--clip',spec['clip'],'--mode',spec['mode'],'--output',str(output)]
        if spec['frames'] is not None:command+=['--frames',str(spec['frames'])]
        if spec['audit']:command+=['--audit']
        if spec['traced']:
            command+=['--traced']
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=process-tree',
                '--osrt-threshold=100000','--cuda-flush-interval=0','--kill=none','--force-overwrite=false',
                '--export=sqlite','--output='+str(output)]+command
        env=environment(os.environ,spec['mode'])
        print('Starting '+n,flush=True)
        begin=time.monotonic()
        with (HERE/(n+'.log')).open('x') as log:
            done=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=dict(**spec,command=command,returncode=done.returncode,process_wall_s=time.monotonic()-begin,
                 log_sha256=sha(HERE/(n+'.log')),thread_policy=policy(spec['mode'])[1])
        record['rows'].append(row)
        if done.returncode:raise RuntimeError('Failed trial '+n)
        r=read(output.with_suffix('.v31.json'))
        if not r['passed'] or r['error'] is not None:raise AssertionError('Trial gate failed '+n)
        row.update(v31_sha256=sha(output.with_suffix('.v31.json')),receipt_sha256=r['receipt_sha256'],fps=r['fps'],wall_s=r['wall_s'])
        if spec['traced']:
            row.update(sqlite_sha256=sha(output.with_suffix('.sqlite')),trace_sha256=sha(output.with_suffix('.trace30.json')))
        print(row,flush=True)
    try:
        status(dict(name='generated_state_gate'))
        with (HERE/'generated.log').open('x') as log:
            subprocess.run([sys.executable,str(HERE/'run_visible_threads_v31.py'),'--generated'],cwd=HERE,
                env=environment(os.environ,'combined'),stdout=log,stderr=subprocess.STDOUT,check=True)
        record['generated_sha256']=sha(HERE/'generated.json')
        for spec in schedule():execute(spec)
        receipts={s['name']:read((HERE/s['name']).with_suffix('.v29.json')) for s in schedule() if not s['audit']}
        record['performance']=gate(receipts)
        print('Performance gate '+str(record['performance']),flush=True)
        for spec in traces():execute(spec)
        if record['performance']['passed']:
            record['schedule']+=full()
            for spec in full():execute(spec)
            record['full_regression_run']=True
        record['passed']=True
    except BaseException as exc:
        record['error']=repr(exc)
        raise
    finally:
        record['finished_ns']=time.time_ns()
        write(HERE/'batch.json',record)
        status(running=False)


if __name__=='__main__':main()
