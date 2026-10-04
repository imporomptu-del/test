"""Predeclared sequential fresh-process diagnostics/timings; never overwrite."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from profile_raw16_efficiency import write_json, sha
from run_raw16_speed_v8 import read
from compare_raw16_v8_audits import compare_audits


def schedule(stage):
    if stage=='checks':
        return [dict(name=f'audit_{clip}_cpu',clip=clip,mode='cpu',audit=True) for clip in ('0040','0029')] + [
            dict(name=f'shadow_{clip}_combined',clip=clip,mode='combined',shadow=True) for clip in ('0040','0029')] + [
            dict(name='trace_0040_reference',clip='0040',mode='reference',trace=True,injected=True),
            dict(name='trace_0040_combined',clip='0040',mode='combined',trace=True,injected=True),
            dict(name='profile_0040_cpu',clip='0040',mode='cpu',profile=True),
            dict(name='profile_0040_combined',clip='0040',mode='combined',profile=True)]
    return [dict(name=f'timed_{clip}_{repeat}_{mode}',clip=clip,mode=mode,repeat=repeat)
            for repeat in (1,2) for clip in ('0040','0029')
            for mode in (('reference','cpu','combined') if repeat==1 else ('combined','cpu','reference'))]


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='timing':
        for clip in ('0040','0029'):
            audit=args.output/f'audit_{clip}_cpu'
            if not read(audit/'comparison.json')['exact_gate_passed']:
                raise ValueError('CPU exact report gate failed')
            parity=compare_audits(args.v7_results/f'audit_{clip}_gpu.audit.jsonl',audit.with_suffix('.audit.jsonl'))
            if not parity['passed']: raise ValueError('CPU array audit gate failed')
            shadow=args.output/f'shadow_{clip}_combined'
            report=read(shadow/'experiment.json')
            if (not read(shadow/'comparison.json')['diagnostic_run_passed'] or len(report['filter_comparisons'])!=63
                    or not all(r['numerical_screen_passed'] for r in report['filter_comparisons'])):
                raise ValueError('Shadow filter numerical screen failed')
    rows=schedule(args.stage)
    for row in rows:
        if (args.output/row['name']).exists() or (args.output/(row['name']+'.log')).exists():
            raise FileExistsError('Existing trials are never overwritten or selectively repeated')
    plan=args.output/(args.stage+'_plan.json')
    write_json(plan,dict(schedule=rows,stage=args.stage,script_sha256=sha(__file__),
        wrapper_sha256=sha(ROOT/'scripts/run_raw16_speed_v8.py'),
        warning='Combined arm is an explicitly NON-BIT-EXACT diagnostic GPU-filter experiment, not approved production. '
                'Two alternating three-arm timing rounds per clip; no clock or detector changes.'))
    with (args.output/(args.stage+'_journal.jsonl')).open('x') as journal:
        for row in rows:
            command=[sys.executable,'-u',str(ROOT/'scripts/run_raw16_speed_v8.py'),
                '--clip',row['clip'],'--mode',row['mode'],'--output',str(args.output/row['name'])]
            for name in ('archive','v7_archive','component','evidence','motion_controls'):
                command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            for name in ('audit','shadow','trace','profile','injected'):
                if row.get(name):command.append('--'+name)
            entered=datetime.now(timezone.utc).isoformat();start=time.perf_counter()
            print(json.dumps(dict(starting=row,entered_utc=entered)),flush=True)
            with (args.output/(row['name']+'.log')).open('x') as log:
                completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(**row,command=command,entered_utc=entered,
                exited_utc=datetime.now(timezone.utc).isoformat(),returncode=completed.returncode,
                process_wall_s=time.perf_counter()-start)
            journal.write(json.dumps(event,sort_keys=True)+'\n');journal.flush()
            print(json.dumps(event),flush=True)
            if completed.returncode:return completed.returncode
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('checks','timing'),required=True)
    for name in ('archive','v7-archive','component','evidence','motion-controls','v7-results','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
