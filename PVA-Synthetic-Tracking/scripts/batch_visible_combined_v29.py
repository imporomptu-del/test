"""One-worker four-arm prefix experiment, then gate-controlled full regression."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_visible_combined_v29 import verify_freeze, read, sha, write
from combined_v29_protocol import schedule, full_schedule, speed_gate

HERE = Path(__file__).resolve().parent


def main():
    _, frozen, _ = verify_freeze()
    if any((HERE/n).exists() for n in ('batch.json', 'status.json', 'smoke_0126.log')):
        raise FileExistsError('Fresh v29 batch only')
    record = dict(passed=False, error=None, source_sha256=frozen['files'],
        freeze_sha256=sha(HERE/'freeze.json'), schedule=schedule(), rows=[], performance=None,
        full_regression_run=False, raw16_accessed=False, defaults_changed=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def status(current=None, running=True):
        temporary = HERE/'status.json.tmp'
        with temporary.open('x') as stream:
            json.dump(dict(running=running, completed=len(record['rows']), scheduled=len(record['schedule']),
                current=current, error=record['error'], performance=record['performance'],
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), stream, indent=2)
        os.replace(temporary, HERE/'status.json')
    def execute(spec):
        status(spec)
        command = [sys.executable, str(HERE/'run_visible_combined_v29.py'), '--clip', spec['clip'],
                   '--arm', spec['arm'], '--output', str(HERE/spec['name'])]
        if spec['frames'] is not None:
            command += ['--frames', str(spec['frames'])]
        if spec['state_audit']:
            command += ['--state-audit']
        print(json.dumps(dict(event='start', **spec)), flush=True)
        start = time.monotonic()
        log = spec['name']+'.log'
        with (HERE/log).open('x') as handle:
            done = subprocess.run(command, cwd=HERE, stdout=handle, stderr=subprocess.STDOUT)
        row = dict(**spec, command=command, returncode=done.returncode, process_wall_s=time.monotonic()-start,
                   log=log, log_sha256=sha(HERE/log))
        record['rows'].append(row)
        if done.returncode:
            raise RuntimeError('Experiment failed: '+spec['name'])
        path = HERE/(spec['name']+'.v29.json')
        receipt = read(path)
        if not receipt['passed'] or receipt['error'] is not None:
            raise AssertionError('Output gate failed: '+spec['name'])
        row.update(receipt_sha256=sha(path), fps=receipt['fps'], wall_s=receipt['wall_s'])
        print(json.dumps(row), flush=True)
    try:
        for spec in schedule():
            execute(spec)
        record['performance'] = speed_gate({s['name']: read(HERE/(s['name']+'.v29.json'))
            for s in schedule() if not s['state_audit']})
        print(json.dumps(dict(event='speed_gate', **record['performance'])), flush=True)
        if record['performance']['passed']:
            record['schedule'] += full_schedule()
            for spec in full_schedule():
                execute(spec)
            record['full_regression_run'] = True
        record['passed'] = True
        record['decision'] = ('combined_requires_independent_audit' if record['performance']['passed']
                              else 'reject_combined_keep_v20')
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(HERE/'batch.json', record)
        status(running=False)


if __name__ == '__main__':
    main()
