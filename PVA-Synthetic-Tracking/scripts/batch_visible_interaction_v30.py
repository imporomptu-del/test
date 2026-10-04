"""Fresh one-worker matched traces/controls; no tuning or adoption gate changes."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from profile_visible_interaction_v30 import V29, V22, read, write, sha, Telemetry

HERE = Path(__file__).resolve().parent
SOURCES = ('profile_visible_interaction_v30.py', 'batch_visible_interaction_v30.py',
           'test_visible_interaction_v30.py', 'visible_interaction_v30_plan.md')


def schedule():
    pairs = [('0082', 'v26'), ('0082', 'combined'), ('0082', 'combined'), ('0082', 'v26'),
             ('0126', 'v26'), ('0126', 'combined')]
    return [dict(name=f'{mode}_{i}_{clip}_{arm}', mode=mode, clip=clip, arm=arm)
            for mode in ('trace', 'clean') for i, (clip, arm) in enumerate(pairs)]


def main():
    if any((HERE/n).exists() for n in ('freeze.json', 'batch.json', 'status.json', 'unit.log')):
        raise FileExistsError('Fresh experiment directory required')
    sys.path.insert(0, str(V29))
    import run_visible_combined_v29 as baseline
    baseline.verify_freeze()
    probe = Telemetry()
    probe.sample()
    write(HERE/'telemetry_probe.json', probe.rows)
    with (HERE/'unit.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_visible_interaction_v30'],
                       cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(HERE/'freeze.json', dict(pre_run=True, created_ns=time.time_ns(),
        sources={name: sha(HERE/name) for name in SOURCES}, unit_log_sha256=sha(HERE/'unit.log'),
        telemetry_probe_sha256=sha(HERE/'telemetry_probe.json'),
        baseline_freeze_sha256=sha(V29/'freeze.json'), bridge_sha256=sha(V22/'libnvtx_bridge.so')))
    record = dict(passed=False, error=None, schedule=schedule(), rows=[],
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_sha256=sha(HERE/'freeze.json'), raw16_accessed=False, defaults_changed=False)
    def status(current=None, running=True):
        temp = HERE/'status.json.tmp'
        write(temp, dict(current=current, running=running, completed=len(record['rows']),
            scheduled=len(record['schedule']), error=record['error'], updated_ns=time.time_ns()))
        os.replace(temp, HERE/'status.json')
    try:
        for spec in schedule():
            status(spec)
            name, traced = spec['name'], spec['mode'] == 'trace'
            output = HERE/name
            for suffix in ('', '.v29.json', '.trace30.json', '.sqlite', '.nsys-rep', '.log', '.tegrastats.log'):
                if (HERE/(name+suffix)).exists():
                    raise FileExistsError(name+suffix)
            script = HERE/'profile_visible_interaction_v30.py' if traced else V29/'run_visible_combined_v29.py'
            command = [sys.executable, str(script), '--clip', spec['clip'], '--arm', spec['arm'], '--output', str(output)]
            if traced:
                command = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                    '--cpuctxsw=process-tree', '--osrt-threshold=100000', '--cuda-flush-interval=0',
                    '--kill=none', '--force-overwrite=false', '--export=sqlite', '--output='+str(output)] + command
            else:
                command += ['--frames', '128']
            print('Starting '+name, flush=True)
            start = time.monotonic()
            sampler = telemetry_log = None
            try:
                if traced:
                    telemetry_log = (HERE/(name+'.tegrastats.log')).open('x')
                    sampler = subprocess.Popen(['tegrastats', '--interval', '500'],
                        stdout=telemetry_log, stderr=subprocess.STDOUT)
                with (HERE/(name+'.log')).open('x') as log:
                    done = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            finally:
                if sampler is not None:
                    sampler.terminate()
                    try:
                        sampler.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sampler.kill()
                        sampler.wait(timeout=5)
                if telemetry_log is not None:
                    telemetry_log.close()
            row = dict(**spec, command=command, returncode=done.returncode,
                       process_wall_s=time.monotonic()-start, log_sha256=sha(HERE/(name+'.log')))
            record['rows'].append(row)
            if done.returncode:
                raise RuntimeError('Failed command '+name)
            r = read(output.with_suffix('.v29.json'))
            if not r['passed'] or r['error'] is not None or r['processed_frames'] != 128:
                raise AssertionError('Failed output gate '+name)
            row.update(receipt_sha256=sha(output.with_suffix('.v29.json')), fps=r['fps'], wall_s=r['wall_s'])
            if traced:
                t = read(output.with_suffix('.trace30.json'))
                if not t['passed'] or t['error'] is not None or t['telemetry_errors'] or len(t['telemetry']) < 2:
                    raise AssertionError('Failed trace '+name)
                row['trace_sha256'] = sha(output.with_suffix('.trace30.json'))
                row['sqlite_sha256'] = sha(output.with_suffix('.sqlite'))
            print(row, flush=True)
        record['passed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(HERE/'batch.json', record)
        status(running=False)


if __name__ == '__main__':
    main()
