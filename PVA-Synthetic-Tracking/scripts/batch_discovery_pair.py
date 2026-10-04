"""One-worker supervisor for two exact discovery clips; no hardware writes."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temperatures():
    readings = {}
    for directory in Path('/sys/devices/virtual/thermal').glob('thermal_zone*'):
        name = (directory/'type').read_text().strip()
        if any(s in name.lower() for s in ('cpu', 'gpu', 'tj', 'junction')):
            try:
                readings[name] = int((directory/'temp').read_text().strip()) / 1000
            except (OSError, TypeError):
                # Powered-down GPU domains can report EAGAIN while idle.
                # Python3.10's TextIO wrapper can instead fail decoding None
                # from that nonblocking sysfs read. Treat either as missing.
                # CPU and junction are mandatory and checked below.
                continue
    require(any('cpu' in s.lower() for s in readings), 'mandatory CPU temperature unavailable')
    require(any('tj' in s.lower() or 'junction' in s.lower() for s in readings),
            'mandatory junction temperature unavailable')
    require(all(-20 < v < 130 for v in readings.values()), 'invalid temperature')
    return readings


def stop_owned(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main(workspace):
    workspace = Path(workspace)
    require(re.fullmatch(r'/tmp/seaqr_discovery_pair_20260928_[A-Za-z0-9]{6}', str(workspace)), 'wrong workspace')
    require(workspace.is_dir() and not workspace.is_symlink() and os.geteuid() != 0, 'unprivileged real workspace required')
    freeze = json.loads((workspace/'freeze.json').read_text())
    require(freeze['schema'] == 'discovery_pair.v1', 'wrong freeze schema')
    freeze_sha = sha(workspace/'freeze.json')
    require(set(freeze['sources']) == {'0170','0240'}, 'exact pair required')
    required_files = {'batch_discovery_pair.py', 'run_discovery_pair_baseline.py',
                      'render_discovery_pair_review.py', 'visible_output.py',
                      'discovery_pair_20260928_plan.md'}
    require(required_files <= set(freeze['files']), 'incomplete frozen bundle')
    for name, digest in freeze['files'].items():
        require(Path(name).name == name and not (workspace/name).is_symlink(), 'unsafe bundle filename')
        require(sha(workspace/name) == digest, 'changed frozen bundle: '+name)
    require(max(temperatures().values()) < 65, 'too warm to start')
    for name in ('batch_status.json', 'batch_status.tmp', 'batch.lock'):
        require(not (workspace/name).exists() and not (workspace/name).is_symlink(), 'existing batch evidence')
    lock = (workspace/'batch.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    began = time.monotonic()
    status = dict(schema='seaqr.discovery-pair.batch.v1', complete=False, passed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), freeze_sha256=freeze_sha,
                  supervisor_pid=os.getpid(), phases=[], current=None, error=None,
                  clock_writes_performed=False, detector_scope='exact 0170/0240, full native8bit, frozen v34 stack')

    def save():
        status['elapsed_s'] = time.monotonic()-began
        temporary = workspace/'batch_status.tmp'
        with temporary.open('x') as stream:
            stream.write(json.dumps(status, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, workspace/'batch_status.json')

    commands = []
    for mode in ('preflight','run'):
        for clip in ('0170','0240'):
            commands.append((mode+'_'+clip,[sys.executable,'-I',str(workspace/'run_discovery_pair_baseline.py'),
                '--workspace',str(workspace),'--clip',clip,'--'+mode]))
    for clip in ('0170','0240'):
        commands.append(('render_'+clip,[sys.executable,'-I',str(workspace/'render_discovery_pair_review.py'),
            '--source',freeze['sources'][clip]['path'],'--journal',str(workspace/clip/'run/frames.jsonl'),
            '--report',str(workspace/clip/'run/report.json'),'--output',str(workspace/clip/'review'),'--clip',clip]))
    child = None
    def interrupted(signum, frame):
        raise RuntimeError('supervisor interrupted by signal '+str(signum))

    original_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        save()
        with (workspace/'telemetry.jsonl').open('x') as telemetry:
            for name, command in commands:
                require(sha(workspace/'freeze.json') == freeze_sha, 'freeze changed')
                for filename, digest in freeze['files'].items():
                    require(not (workspace/filename).is_symlink()
                            and sha(workspace/filename) == digest, 'bundle changed before phase')
                require(max(temperatures().values()) < 65, 'too warm to start next phase')
                start = time.monotonic()
                phase = dict(name=name, command=command, elapsed_s=None, returncode=None)
                status['phases'].append(phase); status['current']=name; save()
                print(json.dumps({'phase':name,'status':'starting'}),flush=True)
                with (workspace/(name+'.log')).open('x') as log:
                    child = subprocess.Popen(command,cwd=workspace,stdout=log,stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL,start_new_session=True)
                    phase['pid']=child.pid; save()
                    while child.poll() is None:
                        temps=temperatures()
                        telemetry.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                            'phase':name,'temperatures_c':temps})+'\n');telemetry.flush()
                        require(max(temps.values()) < 75, 'temperature stop threshold reached')
                        require(time.monotonic()-start < 900, 'phase deadline exceeded')
                        require(time.monotonic()-began < 3600, 'batch deadline exceeded')
                        time.sleep(2)
                    phase.update(elapsed_s=time.monotonic()-start,returncode=child.returncode)
                    require(child.returncode == 0, 'failed phase '+name)
                child=None; save()
                print(json.dumps({'phase':name,'status':'completed','elapsed_s':phase['elapsed_s']}),flush=True)
        for name,digest in freeze['files'].items():
            require(sha(workspace/name)==digest,'bundle changed during execution')
        status.update(complete=True,passed=True,current=None)
    except BaseException as error:
        status['error']=repr(error)
        if child is not None:
            stop_owned(child)
            status['owned_child_stopped']=True
        raise
    finally:
        save()
        lock.close()
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    main(parser.parse_args().workspace)
