"""Sequential frozen checks and reversed two-arm timing; exclusive artifacts."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_raw16_efficiency import sha,write_json
from run_exact_v9 import read

def schedule(stage):
    if stage=='checks':
        return [dict(name=f'audit_{clip}_exact',clip=clip,mode='exact',audit=True) for clip in ('0040','0029')]+[
            dict(name='injected_0040_exact',clip='0040',mode='exact',audit=True,injected=True),
            dict(name='profile_0040_reference',clip='0040',mode='reference',profile=True),
            dict(name='profile_0040_exact',clip='0040',mode='exact',profile=True)]
    if stage!='timing':raise ValueError('Unknown batch stage')
    return [dict(name=f'timed_{clip}_{repeat}_{mode}',clip=clip,mode=mode,repeat=repeat)
        for repeat in (1,2) for clip in ('0040','0029')
        for mode in (('reference','exact') if repeat==1 else ('exact','reference'))]

def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='timing':
        for row in schedule('checks'):
            if read(args.output/row['name']/'comparison.json')['exact_gate_passed'] is not True:
                raise ValueError('All native checks must pass before timing')
    rows=schedule(args.stage)
    for row in rows:
        if (args.output/row['name']).exists() or (args.output/(row['name']+'.log')).exists():
            raise FileExistsError('No overwriting or selective timing retries')
    write_json(args.output/(args.stage+'_plan.json'),dict(schedule=rows,script_sha256=sha(__file__),
        wrapper_sha256=sha(ROOT/'scripts/run_exact_v9.py'),adapter_sha256=sha(ROOT/'scripts/exact_v9_common.py')))
    with (args.output/(args.stage+'_journal.jsonl')).open('x') as journal:
        for row in rows:
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_exact_v9.py'),'--clip',row['clip'],
                 '--mode',row['mode'],'--output',str(args.output/row['name'])]
            for name in ('archive','v7_archive','v8_archive','component','motion_controls','evidence','v8_results'):
                cmd+=['--'+name.replace('_','-'),str(getattr(args,name))]
            for flag in ('audit','profile','injected'):
                if row.get(flag):cmd.append('--'+flag)
            start=time.perf_counter();entered=datetime.now(timezone.utc).isoformat()
            print(json.dumps(dict(starting=row,entered_utc=entered)),flush=True)
            with (args.output/(row['name']+'.log')).open('x') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(**row,command=cmd,returncode=result.returncode,entered_utc=entered,
                exited_utc=datetime.now(timezone.utc).isoformat(),process_wall_s=time.perf_counter()-start)
            journal.write(json.dumps(event,sort_keys=True)+'\n');journal.flush();print(json.dumps(event),flush=True)
            if result.returncode:return result.returncode
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('checks','timing'),required=True)
    for name in ('archive','v7-archive','v8-archive','component','motion-controls','evidence','v8-results','output'):
        p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
