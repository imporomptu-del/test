"""One sequential worker: frozen prefix pairs followed by complete regression."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from profile_visible_v17 import read, sha, write


def schedule():
    rows = [dict(name=f'{clip}_repeat{repeat}_{mode}', clip=clip, mode=mode, frames=128)
            for repeat in range(2) for clip in ('0126', '0082')
            for mode in (('reference', 'candidate') if repeat == 0 else ('candidate', 'reference'))]
    return rows + [dict(name=f'{clip}_full_candidate', clip=clip, mode='candidate', frames=None)
                   for clip in ('0029', '0126', '0055', '0082')]


def run(output):
    output.mkdir()
    record = dict(passed=False, error=None, schedule=schedule(), rows=[], script_sha256=sha(__file__))
    try:
        for spec in record['schedule']:
            name = spec['name']
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('run_visible_v17.py')),
                   '--clip', spec['clip'], '--mode', spec['mode'], '--output', str(output/name)]
            if spec['frames']:
                cmd += ['--frames', str(spec['frames'])]
            print('START', name, time.time(), flush=True)
            with (output/(name+'.log')).open('x') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            if result.returncode:
                raise RuntimeError('Failed '+name+' exit '+str(result.returncode))
            path = output/(name+'.v17.json')
            trial = read(path)
            if not trial['passed'] or trial['error']:
                raise AssertionError('Failed receipt '+name)
            record['rows'].append(dict(**spec, sha256=sha(path), fps=trial['fps'], wall_s=trial['wall_s']))
            print('DONE', name, trial['fps'], flush=True)
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        write(output/'batch.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
