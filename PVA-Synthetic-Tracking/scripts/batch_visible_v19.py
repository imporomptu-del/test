"""One sequential worker, three alternating pairs per workload, then four full clips."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from profile_visible_v17 import read,sha,write


def schedule():
    return [dict(name=f'{clip}_repeat{repeat}_{mode}',clip=clip,mode=mode,frames=128)
            for repeat in range(3) for clip in ('0126','0082')
            for mode in (('reference','candidate') if repeat%2==0 else ('candidate','reference'))]+[
                dict(name=f'{clip}_full_candidate',clip=clip,mode='candidate',frames=None)
                for clip in ('0029','0126','0055','0082')]


def run(output):
    output.mkdir()
    record=dict(passed=False,error=None,schedule=schedule(),rows=[],script_sha256=sha(__file__))
    try:
        for spec in record['schedule']:
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_visible_v19.py')),
                 '--clip',spec['clip'],'--mode',spec['mode'],'--output',str(output/spec['name'])]
            if spec['frames']:cmd+=['--frames',str(spec['frames'])]
            print('START',spec['name'],time.time(),flush=True)
            with (output/(spec['name']+'.log')).open('x') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            if result.returncode:raise RuntimeError('Failed '+spec['name']+' exit '+str(result.returncode))
            path=(output/spec['name']).with_suffix('.v19.json');trial=read(path)
            if not trial['passed'] or trial['error']:raise AssertionError('Failed receipt')
            record['rows'].append(dict(**spec,sha256=sha(path),fps=trial['fps'],wall_s=trial['wall_s']))
            print('DONE',spec['name'],trial['fps'],flush=True)
        record['passed']=True
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:write(output/'batch.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
