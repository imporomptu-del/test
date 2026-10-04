"""Sequential, predeclared CPU/GPU timing pairs after exact array audits.

Invoke under the isolated workspace's flock. Each child is a fresh process;
stop on any failure, preserve all output, never replace or resume a trial.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_raw16_background_v7 import compare_audits, sha, write_json


def schedule():
    rows = []
    for pair in (1, 2):
        for clip in ('0040', '0029'):
            for arm in (('cpu','gpu') if pair == 1 else ('gpu','cpu')):
                rows.append(dict(name=f'timed_{clip}_{pair}_{arm}', clip=clip,
                                 gpu=arm=='gpu', injected=False, pair=pair))
    rows.append(dict(name='injected_0040_gpu', clip='0040', gpu=True, injected=True, pair=None))
    return rows


def run(args):
    for clip in ('0040', '0029'):
        result = compare_audits(args.output/f'audit_{clip}_cpu.audit.jsonl',
                                args.output/f'audit_{clip}_gpu.audit.jsonl')
        if not result['passed']:
            raise ValueError('Complete exact array audits required before timing')
    rows = schedule()
    for row in rows:
        if (args.output/row['name']).exists() or (args.output/(row['name']+'.log')).exists():
            raise FileExistsError('Never overwrite or selectively repeat valid trials')
    write_json(args.output/'timing_plan.json', dict(schedule=rows, pairs_per_clip=2,
        script_sha256=sha(__file__), wrapper_sha256=sha(ROOT/'scripts/run_raw16_background_v7.py'),
        warning='Predeclared two alternating pairs per clip, fresh sequential workers; no clock/power changes.'))
    with (args.output/'timing_journal.jsonl').open('x') as journal:
        for row in rows:
            command = [sys.executable, '-u', str(ROOT/'scripts/run_raw16_background_v7.py'),
                '--clip', row['clip'], '--evidence', str(args.evidence), '--runtime-archive', str(args.runtime_archive),
                '--component', str(args.component), '--output', str(args.output/row['name'])]
            if row['gpu']: command.append('--gpu')
            if row['injected']: command.append('--injected')
            start = time.perf_counter()
            entered = datetime.now(timezone.utc).isoformat()
            print(json.dumps(dict(starting=row, entered_utc=entered)), flush=True)
            with (args.output/(row['name']+'.log')).open('x') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            event = dict(**row, command=command, returncode=result.returncode,
                entered_utc=entered, exited_utc=datetime.now(timezone.utc).isoformat(),
                process_wall_s=time.perf_counter()-start)
            journal.write(json.dumps(event, sort_keys=True)+'\n'); journal.flush()
            print(json.dumps(event), flush=True)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path, required=True)
    parser.add_argument('--component', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
