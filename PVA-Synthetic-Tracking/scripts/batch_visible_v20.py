"""Frozen sequential alternating prefix pairs and complete four-clip regression."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from profile_visible_v17 import read,sha,write


def schedule():
    return [dict(name=f'{c}_repeat{i}_{m}',clip=c,mode=m,frames=128)
        for i in range(3) for c in ('0126','0082')
        for m in (('reference','candidate') if i%2==0 else ('candidate','reference'))]+[
        dict(name=f'{c}_full_candidate',clip=c,mode='candidate',frames=None) for c in ('0029','0126','0055','0082')]


def run(output):
    output.mkdir();record=dict(passed=False,error=None,script_sha256=sha(__file__),schedule=schedule(),rows=[])
    try:
        for spec in record['schedule']:
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_visible_v20.py')),
                '--clip',spec['clip'],'--mode',spec['mode'],'--output',str(output/spec['name'])]
            if spec['frames']:cmd+=['--frames',str(spec['frames'])]
            print('START',spec['name'],time.time(),flush=True)
            with (output/(spec['name']+'.log')).open('x') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            if result.returncode:raise RuntimeError('Failed '+spec['name']+' exit '+str(result.returncode))
            path=(output/spec['name']).with_suffix('.v20.json');r=read(path)
            if not r['passed'] or r['error'] is not None:raise AssertionError('Failed trial receipt')
            record['rows'].append(dict(**spec,sha256=sha(path),fps=r['fps'],wall_s=r['wall_s']))
            print('DONE',spec['name'],r['fps'],flush=True)
        record['passed']=True
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally:write(output/'batch.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
