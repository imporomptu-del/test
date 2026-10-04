"""One-worker fail-closed frozen integration schedule."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from run_motion_pipeline_v16 import read, write, sha


def schedule():
    rows = [dict(name=f'raw_{clip}_repeat{repeat}_{mode}', clip=clip, mode=mode,
                 injected=False, profile=False)
            for repeat in range(2) for clip in ('0029', '0040')
            for mode in (('reference', 'candidate') if repeat == 0 else ('candidate', 'reference'))]
    rows.append(dict(name='raw_0040_injected_candidate', clip='0040', mode='candidate', injected=True, profile=False))
    rows.extend(dict(name='raw_0040_profile_'+mode, clip='0040', mode=mode, injected=False, profile=True)
                for mode in ('reference', 'candidate'))
    return rows


def run(output):
    output.mkdir()
    record = dict(passed=False, error=None, schedule=schedule(), rows=[], script_sha256=sha(__file__))
    try:
        for row in record['schedule']:
            command = [sys.executable, '-u', str(Path(__file__).with_name('run_motion_pipeline_v16.py')),
                       '--clip', row['clip'], '--mode', row['mode'], '--output', str(output/row['name'])]
            command += ['--'+key for key in ('injected', 'profile') if row[key]]
            print('START', row['name'], time.time(), flush=True)
            with (output/(row['name']+'.log')).open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            if result.returncode:
                raise RuntimeError(f'Failed {row["name"]}: {result.returncode}')
            path = output/(row['name']+'.execution.json')
            execution = read(path)
            if not execution['passed'] or execution['error'] or not execution['closed']:
                raise AssertionError('Incomplete execution')
            record['rows'].append(dict(**row, fps=execution['fps'], wall_s=execution['wall_s'], sha256=sha(path)))
            print('DONE', row['name'], execution['fps'], flush=True)
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        write(output/'batch.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
